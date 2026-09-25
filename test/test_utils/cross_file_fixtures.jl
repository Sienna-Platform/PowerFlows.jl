# Fixtures and helpers called from more than one test_*.jl file. Under ParallelTestRunner
# each test file runs in its own worker, so anything shared across files must live here
# (included by includes.jl) rather than in the file that happens to use it first.

# --- area interchange (test_area_interchange_enrollment/_solve, test_jacobian,
# test_residual_condition_diagnostics) ---

function _find_tie(ties::Vector{PF.AreaTie}, fix::Int, tix::Int)
    return only(
        filter(
            tie ->
                (tie.from_bus_ix == fix && tie.to_bus_ix == tix) ||
                    (tie.from_bus_ix == tix && tie.to_bus_ix == fix),
            ties,
        ),
    )
end

_set_slack!(sys, bus_name) =
    PSY.set_bustype!(PSY.get_component(PSY.ACBus, sys, bus_name), PSY.ACBusTypes.SLACK)

function _add_area_interchange!(
    sys,
    from_name::String,
    to_name::String,
    flow::Float64;
    name::String = "$(from_name)_$(to_name)",
)
    PSY.add_component!(
        sys,
        PSY.AreaInterchange(;
            name = name,
            available = true,
            active_power_flow = flow,
            from_area = PSY.get_component(PSY.Area, sys, from_name),
            to_area = PSY.get_component(PSY.Area, sys, to_name),
            flow_limits = (from_to = 0.0, to_from = 0.0), input_basis = PSY.CU,
        ),
    )
    return
end

# Shared by the rule-9 (unenforceable-schedule) and happy-path tests. Area1 owns REF,
# never SLACK; Area2/Area3 can each
# optionally hold SLACK (Area3's Bus 9 has a small gen so it's PV-eligible). AreaInterchange:
# Area2->Area1 0.3, Area3->Area1 0.2 => pdes(Area1)=-0.5, pdes(Area2)=0.3, pdes(Area3)=0.2.
function _three_area_transfer_fixture(; slack_area3::Bool = true)
    sys = _make_three_area_system()
    bus9 = PSY.get_component(PSY.ACBus, sys, "Bus 9")
    gen9 = PSY.ThermalStandard(;
        name = "Bus9Gen",
        available = true,
        status = PSY.OperationalStates.ONLINE,
        bus = bus9,
        active_power = 0.1,
        reactive_power = 0.0,
        rating = 1.0,
        active_power_limits = (min = 0.0, max = 1.0),
        reactive_power_limits = (min = -1.0, max = 1.0),
        ramp_limits = nothing,
        operation_cost = PSY.ThermalGenerationCost(nothing),
        base_power = 100.0, input_basis = PSY.CU,
    )
    PSY.add_component!(sys, gen9)
    _set_slack!(sys, "Bus 6")
    slack_area3 && _set_slack!(sys, "Bus 9")
    _add_area_interchange!(sys, "Area2", "Area1", 0.3; name = "A2_A1")
    _add_area_interchange!(sys, "Area3", "Area1", 0.2; name = "A3_A1")
    return sys
end

# A boundary-crossing 3W transformer winding whose star bus's Y-bus diagonal is polluted by
# BOTH a sibling winding of the same transformer and an unrelated extra line -- neither is a
# member of the boundary-crossing winding's own corridor. Tertiary winding disabled: not
# needed here.
function _make_3w_boundary_fixture()
    sys = System(100.0)
    area_a = PSY.Area(; name = "AreaA", input_basis = PSY.CU)
    area_b = PSY.Area(; name = "AreaB", input_basis = PSY.CU)
    PSY.add_component!(sys, area_a)
    PSY.add_component!(sys, area_b)

    bus1 = _add_simple_bus!(sys, 1, ACBusTypes.REF, 230)
    bus2 = _add_simple_bus!(sys, 2, ACBusTypes.PV, 230)
    bus3 = _add_simple_bus!(sys, 3, ACBusTypes.PQ, 230)
    bus4 = _add_simple_bus!(sys, 4, ACBusTypes.PQ, 230)
    bus5 = _add_simple_bus!(sys, 5, ACBusTypes.PQ, 230)
    PSY.set_area!(bus1, area_a)
    PSY.set_area!(bus2, area_b)
    PSY.set_area!(bus3, area_a)
    PSY.set_area!(bus4, area_b)
    PSY.set_area!(bus5, area_a)

    _add_simple_source!(sys, bus1, 0.0, 0.0)
    _add_simple_thermal_standard!(sys, bus2, 0.1, 0.0)
    _add_simple_load!(sys, bus3, 5.0, 2.0)
    _add_simple_load!(sys, bus4, 5.0, 2.0)
    _add_simple_load!(sys, bus5, 2.0, 1.0)

    _add_simple_line!(sys, bus1, bus3)
    _add_simple_line!(sys, bus2, bus4)

    xfmr = _add_simple_transformer_3w!(sys, bus3, bus4, bus3, 99)
    star_bus = PSY.get_star_bus(xfmr)
    PSY.set_area!(star_bus, area_a)
    _add_simple_line!(sys, star_bus, bus5)

    PSY.set_bustype!(bus2, ACBusTypes.SLACK)
    return sys
end

# --- jacobian (test_jacobian, test_rectangular_ci_jacobian, test_mixed_cpb_power_flow,
# test_rectangular_ci_power_flow) ---

function verify_jacobian(
    sys::PSY.System;
    pf::PF.ACPowerFlow = PF.ACPowerFlow{NewtonRaphsonACPowerFlow}(;
        correct_bustypes = true,
    ),
    label::String = "",
    perturbation::Float64 = 0.02,
    seed::Int = 42,
)
    data = PF.PowerFlowData(pf, sys)
    time_step = 1
    residual = PF.ACPowerFlowResidual(data, time_step)
    J = PF.ACPowerFlowJacobian(residual, time_step)
    x0 = PF.calculate_x0(data, time_step)
    # Verify away from the flat-start state. At flat start θ=0 for every bus,
    # which silently zeroes all `sin(Δθ)` cross-terms — a sign flip in the
    # symbolic Jacobian for those entries would not be detected. A small
    # deterministic perturbation breaks the symmetry.
    if perturbation > 0
        Random.seed!(seed)
        x0 .+= perturbation .* randn(length(x0))
    end
    residual(x0, time_step)
    J(time_step)
    verify_jacobian_asymptotic(
        residual, deepcopy(J.Jv), x0, time_step; label = label,
    )
end

# Two-swing island: buses 1 and 2 are both REF in one island, bus 3 is a PQ load. The
# second swing has a nonzero fixed angle so the check exercises real off-diagonal ∂P/∂θ terms.
function _two_swing_system()
    sys = System(100.0)
    b1 = _add_simple_bus!(sys, 1, ACBusTypes.REF, 230, 1.06, 0.0)
    b2 = _add_simple_bus!(sys, 2, ACBusTypes.REF, 230, 1.05, 0.05)
    b3 = _add_simple_bus!(sys, 3, ACBusTypes.PQ, 230, 1.0, 0.0)
    _add_simple_source!(sys, b1, 0.0, 0.0)
    _add_simple_source!(sys, b2, 0.0, 0.0)
    _add_simple_load!(sys, b3, 40, 15)
    _add_simple_line!(sys, b1, b3, 5e-3, 5e-3, 1e-3)
    _add_simple_line!(sys, b2, b3, 5e-3, 5e-3, 1e-3)
    return sys
end

# --- LCC discrete control (test_lcc_discrete_control, test_discrete_control) ---

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
            bus = bus101, number_engaged = [0], number_of_steps = [8],
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
            regulated_bus_number = 0, input_basis = PSY.CU,
        ),
    )
    if p_set_mw !== nothing
        # `transfer_setpoint` is stored per-unit on the system base.
        base = get_base_power(sys, PSY.NU)
        for l in get_components(TwoTerminalLCCLine, sys)
            set_transfer_setpoint!(l, p_set_mw / base)
        end
    end
    return sys
end

# --- Mixed CPB polar parity (test_mixed_cpb_polar_parity, test_mixed_cpb_power_flow) ---

const MIXED_PARITY_ATOL = 1e-7
# ACTIVSg2000: zero-injection buses with G_ii ≈ 0 make the MCPB Jacobian more
# ill-conditioned than the small synthetic systems. The imag-first column
# ordering + KLU partial pivoting keep it solvable, but the converged-state
# round-off floor is looser than 1e-7; 1e-5 still pins formulation parity.
const MIXED_PARITY_ATOL_2K = 1e-5

_mixed_pf_settings() = SolutionParameters(; validate_voltage_magnitudes = false)

# Assert MCPB matches polar (and, when requested, rectangular CI) on the four
# reported bus quantities, for an arbitrary AC solver. Mirrors
# `_rect_polar_parity`, parametrized over `solver` so the same fixture matrix
# validates Newton-Raphson and Trust-Region against the MCPB Jacobian.
function _mixed_polar_parity(
    sys_p::PSY.System,
    sys_h::PSY.System;
    sys_r::Union{Nothing, PSY.System} = nothing,
    pf_kwargs::NamedTuple = NamedTuple(),
    atol::Float64 = MIXED_PARITY_ATOL,
    solver = NewtonRaphsonACPowerFlow,
)
    pf_p = ACPowerFlow{solver}(; pf_kwargs...)
    pf_h = ACMixedPowerFlow{solver}(;
        pf_kwargs...,
        solution_parameters = _mixed_pf_settings(),
    )
    res_p = solve_power_flow(pf_p, sys_p)
    res_h = solve_power_flow(pf_h, sys_h)
    @test res_p !== missing
    @test res_h !== missing
    bus_p = res_p["bus_results"]
    bus_h = res_h["bus_results"]
    @test maximum(abs.(bus_p.Vm .- bus_h.Vm)) < atol
    @test maximum(abs.(bus_p.θ .- bus_h.θ)) < atol
    # P_gen / Q_gen parity catches slack-recovery and Q-writeback bugs that
    # Vm/θ parity alone cannot — the internal residual math can converge to the
    # correct voltages while the reported generator outputs disagree.
    @test maximum(abs.(bus_p.P_gen .- bus_h.P_gen)) < atol
    @test maximum(abs.(bus_p.Q_gen .- bus_h.Q_gen)) < atol
    if sys_r !== nothing
        pf_r = ACRectangularPowerFlow{solver}(;
            pf_kwargs...,
            solution_parameters = _mixed_pf_settings(),
        )
        res_r = solve_power_flow(pf_r, sys_r)
        @test res_r !== missing
        bus_r = res_r["bus_results"]
        @test maximum(abs.(bus_r.Vm .- bus_h.Vm)) < atol
        @test maximum(abs.(bus_r.θ .- bus_h.θ)) < atol
        @test maximum(abs.(bus_r.P_gen .- bus_h.P_gen)) < atol
        @test maximum(abs.(bus_r.Q_gen .- bus_h.Q_gen)) < atol
    end
    return
end

# Multi-period analogue of `_mixed_polar_parity`: solve both formulations in
# place and assert full per-time-step state-array parity. Exercises the minimal
# per-step `improve_x0` / per-ts offsets & caches (`time_step` threaded
# correctly). This checks per-step correctness, not warm-start efficiency.
function _mixed_polar_parity_data(
    pf_p::ACPowerFlow,
    pf_h::PF.ACMixedPowerFlow,
    sys_p::PSY.System,
    sys_h::PSY.System;
    atol::Float64 = MIXED_PARITY_ATOL,
)
    data_p = PowerFlowData(pf_p, sys_p)
    data_h = PowerFlowData(pf_h, sys_h)
    @test PowerFlows.solve_power_flow!(data_p)
    @test PowerFlows.solve_power_flow!(data_h)
    n_ts = size(data_p.bus_magnitude, 2)
    for ts in 1:n_ts
        @test maximum(
            abs.(data_p.bus_magnitude[:, ts] - data_h.bus_magnitude[:, ts]),
        ) < atol
        @test maximum(
            abs.(data_p.bus_angles[:, ts] - data_h.bus_angles[:, ts]),
        ) < atol
        @test maximum(
            abs.(
                data_p.bus_active_power_injections[:, ts] -
                data_h.bus_active_power_injections[:, ts]
            ),
        ) < atol
        @test maximum(
            abs.(
                data_p.bus_reactive_power_injections[:, ts] -
                data_h.bus_reactive_power_injections[:, ts]
            ),
        ) < atol
    end
    return
end

function _two_swing_mixed_system()
    sys = System(100.0)
    b1 = _add_simple_bus!(sys, 1, ACBusTypes.REF, 230, 1.06, 0.0)
    b2 = _add_simple_bus!(sys, 2, ACBusTypes.REF, 230, 1.05, 0.05)
    b3 = _add_simple_bus!(sys, 3, ACBusTypes.PQ, 230, 1.0, 0.0)
    _add_simple_source!(sys, b1, 0.0, 0.0)
    _add_simple_source!(sys, b2, 0.0, 0.0)
    _add_simple_load!(sys, b3, 40, 15)
    _add_simple_line!(sys, b1, b3, 5e-3, 5e-3, 1e-3)
    _add_simple_line!(sys, b2, b3, 5e-3, 5e-3, 1e-3)
    return sys
end

# --- Rectangular CI polar parity (test_rectangular_ci_polar_parity, test_mixed_cpb_polar_parity,
# test_mixed_cpb_jacobian, test_mixed_cpb_residual) ---

const RECT_PARITY_ATOL = 1e-7

_rect_parity_settings() = SolutionParameters(; validate_voltage_magnitudes = false)

function _rect_polar_parity(
    sys_p::PSY.System,
    sys_r::PSY.System;
    pf_kwargs::NamedTuple = NamedTuple(),
    pf_r_extra_settings::AbstractDict = Dict{Symbol, Any}(),
    atol::Float64 = RECT_PARITY_ATOL,
)
    pf_p = ACPowerFlow{NewtonRaphsonACPowerFlow}(; pf_kwargs...)
    pf_r = ACRectangularPowerFlow{NewtonRaphsonACPowerFlow}(;
        pf_kwargs...,
        solution_parameters = PF._override(_rect_parity_settings(), pf_r_extra_settings),
    )
    res_p = solve_power_flow(pf_p, sys_p)
    res_r = solve_power_flow(pf_r, sys_r)
    @test res_p !== missing
    @test res_r !== missing
    bus_p = res_p["bus_results"]
    bus_r = res_r["bus_results"]
    @test maximum(abs.(bus_p.Vm - bus_r.Vm)) < atol
    @test maximum(abs.(bus_p.θ - bus_r.θ)) < atol
    # P_gen / Q_gen parity catches slack-recovery and Q-writeback bugs that
    # Vm/θ parity alone cannot — the internal residual math can converge to the
    # correct voltages while the reported generator outputs disagree (e.g., if
    # the subnetwork slack is over-attributed to REF instead of distributed
    # across participating buses).
    @test maximum(abs.(bus_p.P_gen - bus_r.P_gen)) < atol
    @test maximum(abs.(bus_p.Q_gen - bus_r.Q_gen)) < atol
    return
end

# Multi-period analogue of `_rect_polar_parity`: solve both formulations in
# place and assert full state-array parity.
function _rect_polar_parity_data(
    pf_p::ACPowerFlow,
    pf_r::ACRectangularPowerFlow,
    sys_p::PSY.System,
    sys_r::PSY.System,
)
    data_p = PowerFlowData(pf_p, sys_p)
    data_r = PowerFlowData(pf_r, sys_r)
    @test PowerFlows.solve_power_flow!(data_p)
    @test PowerFlows.solve_power_flow!(data_r)
    @test maximum(abs.(data_p.bus_magnitude - data_r.bus_magnitude)) < RECT_PARITY_ATOL
    @test maximum(abs.(data_p.bus_angles - data_r.bus_angles)) < RECT_PARITY_ATOL
    @test maximum(
        abs.(data_p.bus_active_power_injections -
             data_r.bus_active_power_injections),
    ) < RECT_PARITY_ATOL
    @test maximum(
        abs.(data_p.bus_reactive_power_injections -
             data_r.bus_reactive_power_injections),
    ) < RECT_PARITY_ATOL
    return
end

function _build_zip_2bus_system(;
    power_pq::Tuple{Float64, Float64} = (0.0, 0.0),
    current_pq::Tuple{Float64, Float64} = (0.0, 0.0),
    impedance_pq::Tuple{Float64, Float64} = (0.0, 0.0),
    zip_on_ref::Bool = false,
)
    sys = System(100.0)
    b1 = _add_simple_bus!(sys, 1, ACBusTypes.REF, 230, 1.1, 0.0)
    b2 = _add_simple_bus!(sys, 2, ACBusTypes.PQ, 230, 1.1, 0.0)
    _add_simple_line!(sys, b1, b2, 5e-3, 5e-3, 1e-3)
    _add_simple_source!(sys, b1, 0.0, 0.0)
    zip_bus = zip_on_ref ? b1 : b2
    _add_simple_zip_load!(
        sys,
        zip_bus;
        constant_power_active_power = power_pq[1],
        constant_power_reactive_power = power_pq[2],
        constant_current_active_power = current_pq[1],
        constant_current_reactive_power = current_pq[2],
        constant_impedance_active_power = impedance_pq[1],
        constant_impedance_reactive_power = impedance_pq[2],
    )
    return sys
end

# --- Rectangular CI power flow (test_rectangular_ci_power_flow, test_rectangular_ci_jacobian) ---

function _rect_pf_settings()
    return SolutionParameters(; validate_voltage_magnitudes = false)
end

function _rect_two_swing_system()
    sys = System(100.0)
    b1 = _add_simple_bus!(sys, 1, PSY.ACBusTypes.REF, 230, 1.06, 0.0)
    b2 = _add_simple_bus!(sys, 2, PSY.ACBusTypes.REF, 230, 1.05, 0.05)
    b3 = _add_simple_bus!(sys, 3, PSY.ACBusTypes.PQ, 230, 1.0, 0.0)
    _add_simple_source!(sys, b1, 0.0, 0.0)
    _add_simple_source!(sys, b2, 0.0, 0.0)
    _add_simple_load!(sys, b3, 40, 15)
    _add_simple_line!(sys, b1, b3, 5e-3, 5e-3, 1e-3)
    _add_simple_line!(sys, b2, b3, 5e-3, 5e-3, 1e-3)
    return sys
end

# --- VSC (test_vsc_power_flow, test_area_interchange_solve, test_nr_cache_reuse) ---

# Build c_sys5 and add one point-to-point VSC line: the `from` converter controls DC voltage
# (DC slack), the `to` converter controls (P, Q). This is the physically well-posed config: one
# terminal fixes V_dc, the other sets power.
function _build_vsc_system(; g = 50.0)
    sys = deepcopy(PSB.build_system(PSB.PSITestSystems, "c_sys5"; add_forecasts = false))
    buses = sort!(collect(PSY.get_components(PSY.ACBus, sys)); by = PSY.get_number)
    from_bus = buses[1]
    to_bus = buses[4]
    arc = _get_or_make_arc(sys, from_bus, to_bus)
    vsc = PSY.TwoTerminalVSCLine(;
        name = "vsc_test",
        available = true,
        arc = arc,
        active_power_flow = 0.5,
        rating = 1.0,
        active_power_limits_from = (min = -1.0, max = 1.0),
        active_power_limits_to = (min = -1.0, max = 1.0),
        g = g,
        # from converter: DC-voltage control (DC slack), no AC-voltage control
        dc_control_from = PSY.VSCDCControlModes.DC_VOLTAGE,
        ac_control_from = PSY.VSCACControlModes.AC_REACTIVE_POWER,
        dc_setpoint_from = 1.0,
        ac_setpoint_from = 1.0,
        # to converter: power control (P, Q)
        dc_control_to = PSY.VSCDCControlModes.DC_POWER,
        ac_control_to = PSY.VSCACControlModes.AC_REACTIVE_POWER,
        dc_setpoint_to = 0.5,
        ac_setpoint_to = 1.0, input_basis = PSY.CU,
    )
    PSY.add_component!(sys, vsc)
    return sys
end

# Regression: the polar VSC Jacobian must be bus-type aware. Column `2ix-1` is the |V_ac| state
# only for PQ buses; for PV it is Q_gen and for REF it is P_gen (see state_indexing_helpers.jl).
# A lossy converter whose AC terminal is a PV (or REF) bus has a nonzero ∂KCL/∂|V_ac| loss term —
# writing it into column `2ix-1` (which is not |V_ac| there) corrupts the Jacobian. |V_ac| is fixed
# at PV/REF buses, so that derivative must not enter the Jacobian at all.
function _vsc_system_pv_terminal(; g = 45.0)
    sys = deepcopy(PSB.build_system(PSB.PSITestSystems, "c_sys14"; add_forecasts = false))
    pick(t) = first(
        sort!(
            collect(PSY.get_components(b -> PSY.get_bustype(b) == t, PSY.ACBus, sys));
            by = PSY.get_number,
        ),
    )
    from_bus = pick(PSY.ACBusTypes.PQ)        # DC-voltage slack converter on a PQ bus
    to_bus = pick(PSY.ACBusTypes.PV)          # lossy power-control converter on a PV bus
    arc = _get_or_make_arc(sys, from_bus, to_bus)
    vsc = PSY.TwoTerminalVSCLine(;
        name = "vsc_pv",
        available = true,
        arc = arc,
        active_power_flow = 0.3,
        rating = 2.0,
        active_power_limits_from = (min = -2.0, max = 2.0),
        active_power_limits_to = (min = -2.0, max = 2.0),
        g = g,
        dc_control_from = PSY.VSCDCControlModes.DC_VOLTAGE,
        ac_control_from = PSY.VSCACControlModes.AC_REACTIVE_POWER,
        dc_setpoint_from = 1.03,
        reactive_power_from = 0.0,
        dc_control_to = PSY.VSCDCControlModes.DC_POWER,
        ac_control_to = PSY.VSCACControlModes.AC_REACTIVE_POWER,
        dc_setpoint_to = 0.35,
        reactive_power_to = 0.05,
        converter_loss_to = PSY.LossCurve(
            PSY.QuadraticCurve(0.01, 0.02, 0.005),
            PSY.NaturalUnit(),
        ), input_basis = PSY.CU,
    )
    PSY.add_component!(sys, vsc)
    return sys
end
