# Tests for LCC coexistence with the discrete-control continuation.
#
# ── 1. Checkpoint contract ────────────────────────────────────────────────────────────────
# Every `Matrix`-typed field of the LCC state goes in exactly one bucket below, so a new one
# added without extending the checkpoint fails a test instead of breaking rollback silently.
# `i_dc` is checkpointed because it is derived from `p_set` at construction and no solve path
# rewrites it. `branch_admittances` is a `Vector`, so it falls outside this partition.

const LCC_CHECKPOINTED_FIELDS = Dict(
    PF.LCCConverterParameters => [:tap, :thyristor_angle],
    PF.LCCParameters => [:i_dc],
)
const LCC_DERIVED_FIELDS = Dict(
    PF.LCCConverterParameters => [:phi],
    PF.LCCParameters => Symbol[],
)
const LCC_OUTPUT_FIELDS = Dict(
    PF.LCCConverterParameters => Symbol[],
    PF.LCCParameters => [
        :arc_active_power_flow_from_to, :arc_active_power_flow_to_from,
        :arc_reactive_power_flow_from_to, :arc_reactive_power_flow_to_from,
    ],
)
const LCC_STATIC_MATRIX_FIELDS = Dict(
    PF.LCCConverterParameters => Symbol[],
    PF.LCCParameters => [:p_set],  # setpoint schedule: input, never solver-mutated
)

@testset "LCC per-time-step state classification is exhaustive" begin
    for T in (PF.LCCParameters, PF.LCCConverterParameters)
        matrix_fields = [f for f in fieldnames(T) if fieldtype(T, f) <: Matrix]
        classified = vcat(
            LCC_CHECKPOINTED_FIELDS[T], LCC_DERIVED_FIELDS[T],
            LCC_OUTPUT_FIELDS[T], LCC_STATIC_MATRIX_FIELDS[T],
        )
        @test sort(matrix_fields) == sort(classified)
        @test allunique(classified)
    end
end

@testset "continuation checkpoint restores LCC solver state" begin
    raw = joinpath(TEST_DATA_DIR, "case5_2_lcc.raw")
    sys = make_system(PFP.PowerModelsData(raw); runchecks = false)
    pf = ACPolarPowerFlow(; check_reactive_power_limits = false)
    data = PowerFlowData(pf, sys)
    ts = 1
    @test PowerFlows.get_lcc_count(data) > 0

    # Driven off the classification rather than naming columns literally, so a newly
    # classified field is exercised here automatically.
    checkpointed_cols = vcat(
        [
            (conv, f)
            for conv in (data.lcc.rectifier, data.lcc.inverter)
            for f in LCC_CHECKPOINTED_FIELDS[PF.LCCConverterParameters]
        ],
        [(data.lcc, f) for f in LCC_CHECKPOINTED_FIELDS[PF.LCCParameters]],
    )

    # `_lcc_state_cols` must contribute exactly one snapshot field per checkpointed column;
    # a field dropped from either side would silently stop being rolled back.
    lcc_snap_fields =
        filter(f -> startswith(String(f), "lcc_"), fieldnames(PF.ControlStateSnapshot))
    @test length(lcc_snap_fields) == length(checkpointed_cols)

    # Spread the columns apart first: the fixture parses rectifier and inverter tap to the
    # same value, which would hide a restore that crossed those two columns.
    for (i, (obj, f)) in enumerate(checkpointed_cols)
        getfield(obj, f)[:, ts] .+= 0.01 * i
    end
    PowerFlows._update_ybus_lcc!(data, ts)
    ref_cols = [copy(getfield(obj, f)[:, ts]) for (obj, f) in checkpointed_cols]
    @test allunique(ref_cols)   # guards the swap detection above
    ref_admittances = copy(data.lcc.branch_admittances)
    snap = PowerFlows._snapshot_state(data, ts)

    # Simulate a diverged trial. Garbage is distinct per column, as the references are.
    for (i, (obj, f)) in enumerate(checkpointed_cols)
        getfield(obj, f)[:, ts] .= 100.0 + i
    end
    data.bus_magnitude[:, ts] .= 0.5
    PowerFlows._update_ybus_lcc!(data, ts)  # caches now reflect the garbage

    PowerFlows._restore_state!(data, ts, snap)

    for ((obj, f), ref) in zip(checkpointed_cols, ref_cols)
        @test getfield(obj, f)[:, ts] == ref
    end
    # Caches recomputed at the restored state, not left holding the garbage-derived values.
    @test data.lcc.branch_admittances == ref_admittances
end

@testset "LCC + control_discrete_devices constructs at any time_steps" begin
    raw = joinpath(TEST_DATA_DIR, "case5_2_lcc.raw")
    sys = make_system(PFP.PowerModelsData(raw); runchecks = false)
    for nts in (1, 3)
        pf = ACPolarPowerFlow(; control_discrete_devices = true, time_steps = nts)
        data = PowerFlowData(pf, sys)   # must not throw
        @test PowerFlows.get_lcc_count(data) > 0
    end
end

# ── 2. Single-period control ──────────────────────────────────────────────────────────────

# Twin-parity tolerance. A locked twin represents its FACTS device on the Y-bus rather than in
# the withdrawal the continuation uses, so some difference is expected; this sits above it and
# well below the multiperiod per-step spread.
const LCC_PARITY_ATOL = 1e-7

"""Parse the bundled two-LCC fixture and add enrollable controlled devices: a stepping
switched shunt and a shunt FACTS device at bus 101 (PQ, 230 kV, largest load). The fixture
has no transformers, so no tap device is enrolled. `p_set_mw` overrides both LCC transfer
setpoints (0.0 exercises the i_dc = 0 tap-pinning branch).

Every branch carries x = 1e-4 pu against a much larger r, so bus 101 is electrically bolted to
the REF bus and the network is resistance-dominated — a reactive move there shifts angle far
more than magnitude. Device ratings and setpoints are therefore sized past anything realistic,
so the devices clear `CONTROL_GAIN_FLOOR` and enroll instead of being frozen as insensitive,
and their setpoints sit above the reachable voltage so the continuation keeps driving them."""
function build_lcc_control_system(; p_set_mw::Union{Nothing, Float64} = nothing)
    raw = joinpath(TEST_DATA_DIR, "case5_2_lcc.raw")
    sys = make_system(PFP.PowerModelsData(raw); runchecks = false)
    bus101 = get_bus(sys, 101)
    add_component!(
        sys,
        SwitchedAdmittance(; name = "ctrl_shunt_101", available = true,
            bus = bus101, Y = 0.0 + 0.0im, initial_status = [0], number_of_steps = [8],
            Y_increase = [0.0 + 0.5im], admittance_limits = (min = 1.05, max = 1.08),
            control_mode = PSY.SwitchedAdmittanceControlMode.DISCRETE_VOLTAGE,
        ),
    )
    add_component!(
        sys,
        FACTSControlDevice(;
            name = "ctrl_facts_101",
            available = true,
            bus = bus101,
            control_mode = PSY.FACTSOperationModes.NML,
            voltage_setpoint = 1.06,
            max_shunt_current = 1000.0,
            max_reactive_power = 9999.0,
            shunt_control_type = PSY.FACTSShuntControlType.STATCOM,
            regulated_bus_number = 0,
        ),
    )
    if p_set_mw !== nothing
        # `initialize_LCCParameters!` seeds `p_set` from this setter's value in MW.
        for l in get_components(TwoTerminalLCCLine, sys)
            set_transfer_setpoint!(l, p_set_mw)
        end
    end
    return sys
end

"""Rebuild the fixture with both controlled devices hard-locked at the settings `results`
reports for time step `ts`, so it can be solved with control off. The shunt locks through its
own `Y`; the FACTS device needs a `FixedAdmittance` because its Q is reporting-only."""
function build_locked_twin(results, ts::Int)
    step_results = results[results.time_step .== ts, :]
    shunt_final = only(step_results.final[step_results.family .== "SwitchedAdmittance"])
    facts_b = only(step_results.final[step_results.family .== "FACTSControlDevice"])
    facts_q =
        only(step_results.delivered_q_mvar[step_results.family .== "FACTSControlDevice"])
    sys = build_lcc_control_system()
    sa = get_component(SwitchedAdmittance, sys, "ctrl_shunt_101")
    set_Y!(sa, Complex(real(get_Y(sa)), shunt_final))
    set_reactive_power_required!(
        get_component(FACTSControlDevice, sys, "ctrl_facts_101"), facts_q)
    add_component!(
        sys,
        FixedAdmittance(;
            name = "ctrl_facts_101_locked_shunt", available = true,
            bus = get_bus(sys, 101), Y = Complex(0.0, facts_b),
        ),
    )
    return sys
end

@testset "LCC + control: equivalence with devices locked at converged settings" begin
    sys = build_lcc_control_system()
    pf = ACPolarPowerFlow(; control_discrete_devices = true)
    data = PowerFlowData(pf, sys)
    solve_power_flow!(data)
    @test all(data.converged)

    results = PowerFlows.get_controlled_device_results(data)
    # One base solve plus at least one probe solve per enrolled device: the probe phase is
    # what guarantees capture->solve->restore ran with LCC state live.
    @test PowerFlows.get_control_inner_solve_count(data) >= 1 + DataFrames.nrow(results)

    vm_ctrl = copy(data.bus_magnitude[:, 1])
    va_ctrl = copy(data.bus_angles[:, 1])
    lcc_taps = copy(data.lcc.rectifier.tap[:, 1])

    data2 = PowerFlowData(
        ACPolarPowerFlow(; control_discrete_devices = false),
        build_locked_twin(results, 1))
    solve_power_flow!(data2)
    @test all(data2.converged)
    @test isapprox(data2.bus_magnitude[:, 1], vm_ctrl; atol = LCC_PARITY_ATOL)
    @test isapprox(data2.bus_angles[:, 1], va_ctrl; atol = LCC_PARITY_ATOL)
    @test isapprox(data2.lcc.rectifier.tap[:, 1], lcc_taps; atol = LCC_PARITY_ATOL)
end

@testset "LCC + control with 0 MW transfer (i_dc = 0 tap pinning) survives restores" begin
    sys = build_lcc_control_system(; p_set_mw = 0.0)
    pf = ACPolarPowerFlow(; control_discrete_devices = true)
    data = PowerFlowData(pf, sys)
    solve_power_flow!(data)
    @test all(data.converged)
    @test all(
        isapprox.(
            data.lcc.rectifier.tap[:, 1], data.lcc.rectifier.tap_setpoint; atol = 1e-8),
    )
end

# ── 3. Multiperiod control ────────────────────────────────────────────────────────────────

"""Per-step reactive-withdrawal scale for bus 101, mirroring `_shunt_step_q_scale`
(test/test_utils/common.jl). Scaling is narrowed to bus 101, the bus hosting the enrolled
devices, so the fixture's other loads stay fixed."""
_lcc_bus101_q_scale(t::Int) = 0.8 + 0.2 * t

"""Scale bus 101's reactive withdrawal across time-step columns `1:n` of a multiperiod
`data`, taking column 1 as the base."""
function _set_lcc_bus101_loads!(data, n::Int)
    bix = PF.get_bus_lookup(data)[101]
    base_q = data.bus_reactive_power_withdrawals[bix, 1]
    for t in 1:n
        data.bus_reactive_power_withdrawals[bix, t] = base_q * _lcc_bus101_q_scale(t)
    end
    return
end

"""Set bus 101's reactive withdrawal in a single-period `data` to the value
`_set_lcc_bus101_loads!` assigns to multiperiod column `t`, for building one step's twin."""
function _set_lcc_bus101_load_at_step!(data, t::Int)
    bix = PF.get_bus_lookup(data)[101]
    data.bus_reactive_power_withdrawals[bix, 1] *= _lcc_bus101_q_scale(t)
    return
end

"""Assert each pair of time-step columns in `pairs` is separated by more than the parity
tolerance, so comparing against the wrong column would be caught. LCC converter state is not
guarded: its terminal buses are voltage-pinned, so those columns are identical across steps
and the LCC comparisons below are control-on/off parity checks, not per-step ones."""
function _assert_steps_separated(data, pairs)
    for (a, b) in pairs
        @test maximum(abs.(data.bus_angles[:, a] .- data.bus_angles[:, b])) >
              LCC_PARITY_ATOL
        @test maximum(abs.(data.bus_magnitude[:, a] .- data.bus_magnitude[:, b])) >
              LCC_PARITY_ATOL
    end
    return
end

@testset "LCC + control multiperiod: per-step equivalence" begin
    time_steps = 3
    sys = build_lcc_control_system()
    pf = ACPolarPowerFlow(; control_discrete_devices = true, time_steps = time_steps)
    data = PowerFlowData(pf, sys)
    _set_lcc_bus101_loads!(data, time_steps)
    solve_power_flow!(data)
    @test all(data.converged)
    # The loop below compares every step, so every pair must be distinguishable.
    _assert_steps_separated(data, ((1, 2), (2, 3), (1, 3)))

    results = PowerFlows.get_controlled_device_results(data)
    for ts in 1:time_steps
        data2 = PowerFlowData(
            ACPolarPowerFlow(; control_discrete_devices = false),
            build_locked_twin(results, ts),
        )
        _set_lcc_bus101_load_at_step!(data2, ts)
        solve_power_flow!(data2)
        @test all(data2.converged)
        @test isapprox(
            data2.bus_angles[:, 1], data.bus_angles[:, ts]; atol = LCC_PARITY_ATOL)
        @test isapprox(
            data2.bus_magnitude[:, 1], data.bus_magnitude[:, ts]; atol = LCC_PARITY_ATOL)
        @test isapprox(
            data2.lcc.rectifier.tap[:, 1], data.lcc.rectifier.tap[:, ts];
            atol = LCC_PARITY_ATOL)
    end
end

@testset "LCC + control multiperiod: steps solve independently" begin
    time_steps = 3
    sys = build_lcc_control_system()
    pf = ACPolarPowerFlow(; control_discrete_devices = true, time_steps = time_steps)
    data = PowerFlowData(pf, sys)
    _set_lcc_bus101_loads!(data, time_steps)
    solve_power_flow!(data)
    @test all(data.converged)
    _assert_steps_separated(data, ((1, 3),))
    # Steps 1 and 3 must match independent single-period solves, so nothing from step 2's
    # continuation (probe excursions, rolled-back passes) reached its neighbours.
    for ts in (1, 3)
        data_s = PowerFlowData(
            ACPolarPowerFlow(; control_discrete_devices = true),
            build_lcc_control_system(),
        )
        _set_lcc_bus101_load_at_step!(data_s, ts)
        solve_power_flow!(data_s)
        @test all(data_s.converged)
        @test isapprox(
            data_s.bus_angles[:, 1], data.bus_angles[:, ts]; atol = LCC_PARITY_ATOL)
        @test isapprox(
            data_s.bus_magnitude[:, 1], data.bus_magnitude[:, ts]; atol = LCC_PARITY_ATOL)
        @test isapprox(
            data_s.lcc.rectifier.tap[:, 1], data.lcc.rectifier.tap[:, ts];
            atol = LCC_PARITY_ATOL)
    end
end

@testset "LCC + control multiperiod: single-step i_dc = 0 tap pinning" begin
    time_steps = 3
    sys_ref = build_lcc_control_system()
    pf = ACPolarPowerFlow(; control_discrete_devices = true, time_steps = time_steps)
    data_ref = PowerFlowData(pf, sys_ref)
    _set_lcc_bus101_loads!(data_ref, time_steps)
    solve_power_flow!(data_ref)
    @test all(data_ref.converged)
    _assert_steps_separated(data_ref, ((1, 3),))

    # Same construction, but force LCC transfer to 0 MW at step 2 only.
    sys = build_lcc_control_system()
    data = PowerFlowData(pf, sys)
    _set_lcc_bus101_loads!(data, time_steps)
    # `i_dc` is derived from `p_set` at construction and never recomputed during a solve, so
    # overriding `p_set` afterwards requires zeroing `i_dc` too.
    data.lcc.p_set[:, 2] .= 0.0
    data.lcc.i_dc[:, 2] .= 0.0
    solve_power_flow!(data)
    @test all(data.converged)

    @test all(
        isapprox.(
            data.lcc.rectifier.tap[:, 2], data.lcc.rectifier.tap_setpoint; atol = 1e-8),
    )
    # Steps 1 and 3 are unaffected by step 2's pinned converters.
    for ts in (1, 3)
        @test isapprox(
            data.bus_angles[:, ts], data_ref.bus_angles[:, ts]; atol = LCC_PARITY_ATOL)
        @test isapprox(
            data.bus_magnitude[:, ts], data_ref.bus_magnitude[:, ts];
            atol = LCC_PARITY_ATOL)
        @test isapprox(
            data.lcc.rectifier.tap[:, ts], data_ref.lcc.rectifier.tap[:, ts];
            atol = LCC_PARITY_ATOL)
    end
end

# ── 4. Non-polar formulations ─────────────────────────────────────────────────────────────

@testset "LCC + control: non-polar formulation (ACRectangularPowerFlow)" begin
    # The linear sensitivity path is polar-Jacobian-only, so a non-polar formulation drives
    # the continuation through the FD `_plant_sign` probe fallback instead.
    data_polar = PowerFlowData(
        ACPolarPowerFlow(; control_discrete_devices = true), build_lcc_control_system(),
    )
    solve_power_flow!(data_polar)
    @test all(data_polar.converged)

    data_rect = PowerFlowData(
        ACRectangularPowerFlow(; control_discrete_devices = true),
        build_lcc_control_system(),
    )
    solve_power_flow!(data_rect)
    @test all(data_rect.converged)

    # Rectangular-CI carries an e/f state internally but still populates `bus_magnitude`, so
    # the physical quantities compare directly.
    @test isapprox(
        data_rect.bus_magnitude[:, 1], data_polar.bus_magnitude[:, 1]; atol = 1e-6)
    @test isapprox(
        data_rect.lcc.rectifier.tap[:, 1], data_polar.lcc.rectifier.tap[:, 1]; atol = 1e-6)
    @test isapprox(
        data_rect.lcc.inverter.tap[:, 1], data_polar.lcc.inverter.tap[:, 1]; atol = 1e-6)
    @test isapprox(
        data_rect.lcc.rectifier.thyristor_angle[:, 1],
        data_polar.lcc.rectifier.thyristor_angle[:, 1];
        atol = 1e-6,
    )
    @test isapprox(data_rect.lcc.i_dc[:, 1], data_polar.lcc.i_dc[:, 1]; atol = 1e-6)
end
