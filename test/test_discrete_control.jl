const IEEE14_FACTS_RAW = "/Users/jdlara/cache/vg_pfs_validation/14_bus.raw"

"""Build the IEEE 14-bus system (PSS/E `14_bus.raw`) with a `FACTSControlDevice` at bus 14
(`control_mode=NML`). Reused by discrete-control / reactive-power-control tests.

`stress` scales every `StandardLoad`'s ZIP P/Q components (the PSS/E parser's load type);
`shunt9_off` disables the bus-9 fixed shunt so the FACTS device carries more of the
reactive burden. `svc`/`shunt_control_type` selects the SVC vs STATCOM reactive-limit law;
`regulated_bus_number` sets FCREG (0 ⇒ local/sending-bus regulation)."""
function build_ieee14_facts_system(;
    regulated_bus_number::Int = 0,
    shmx_mva::Float64 = 25.0,
    mva_cap::Float64 = 9999.0,
    svc::Bool = false,
    vset::Float64 = 1.0,
    stress::Float64 = 1.0,
    shunt9_off::Bool = false,
)
    sys = System(IEEE14_FACTS_RAW; runchecks = false)
    if !isone(stress)
        for load in get_components(StandardLoad, sys)
            set_constant_active_power!(load, get_constant_active_power(load) * stress)
            set_constant_reactive_power!(
                load, get_constant_reactive_power(load) * stress)
            set_impedance_active_power!(load, get_impedance_active_power(load) * stress)
            set_impedance_reactive_power!(
                load, get_impedance_reactive_power(load) * stress)
            set_current_active_power!(load, get_current_active_power(load) * stress)
            set_current_reactive_power!(
                load, get_current_reactive_power(load) * stress)
        end
    end
    if shunt9_off
        for fa in get_components(PSY.FixedAdmittance, sys)
            get_number(get_bus(fa)) == 9 && set_available!(fa, false)
        end
    end
    shunt_control_type = PSY.FACTSShuntControlType.STATCOM
    if svc
        shunt_control_type = PSY.FACTSShuntControlType.SVC
    end
    facts = FACTSControlDevice(;
        name = "facts_14",
        available = true,
        bus = get_bus(sys, 14),
        control_mode = PSY.FACTSOperationModes.NML,
        voltage_setpoint = vset,
        max_shunt_current = shmx_mva,
        max_reactive_power = mva_cap,
        shunt_control_type = shunt_control_type,
        regulated_bus_number = regulated_bus_number,
    )
    add_component!(sys, facts)
    return sys
end

@testset "discrete control: FACTS enrolled by default (no experimental flag)" begin
    sys = build_ieee14_facts_system()
    pf = ACPowerFlow(; control_discrete_devices = true)
    data = PowerFlows.PowerFlowData(pf, sys)
    set = PowerFlows.get_controlled_devices(data)
    @test !isnothing(set)
    @test length(set.facts) == 1
end

@testset "FACTS remote FCREG regulation targets the remote bus" begin
    sys = build_ieee14_facts_system(; regulated_bus_number = 13)   # device at 14, regulates 13
    pf = ACPowerFlow(; control_discrete_devices = true)
    data = PowerFlows.PowerFlowData(pf, sys)
    set = PowerFlows.get_controlled_devices(data)
    f = only(set.facts)
    bl = PowerFlows.get_bus_lookup(data)
    @test PowerFlows.controlled_bus_ix(f) == bl[13]   # remote, not the device's own bus 14
    @test PowerFlows.controlled_bus_ix(f) != f.bus_ix
end

@testset "discrete control: pf field defaults" begin
    @test ACPolarPowerFlow().control_discrete_devices == false
    @test ACRectangularPowerFlow().control_discrete_devices == false
    @test ACMixedPowerFlow().control_discrete_devices == false
    @test ACPolarPowerFlow(; control_discrete_devices = true).control_discrete_devices ==
          true
    @test ACRectangularPowerFlow(;
        control_discrete_devices = true,
    ).control_discrete_devices ==
          true
    @test ACMixedPowerFlow(; control_discrete_devices = true).control_discrete_devices ==
          true
    @test PowerFlows.get_control_discrete_devices(ACPolarPowerFlow()) == false
    @test PowerFlows.get_control_discrete_devices(
        ACPolarPowerFlow(; control_discrete_devices = true),
    ) ==
          true
    @test PowerFlows.get_control_discrete_devices(DCPowerFlow()) == false
end

@testset "discrete control: build set" begin
    sys = _make_tap_shunt_system()
    data = PowerFlowData(ACPolarPowerFlow(), sys)
    bus_lookup = PF.get_bus_lookup(data)
    ybus = data.power_network_matrix
    set = PowerFlows.build_controlled_device_set(sys, bus_lookup, ybus)
    @test set isa PowerFlows.ControlledDeviceSet
    @test length(set.taps) == 1
    @test length(set.shunts) == 1
    t = set.taps[1]
    @test 0.9 <= t.p_min < t.p_max <= 1.1
    @test t.controlled_ix == t.to_ix
    @test t.vset > 0.0
    @test t.vset == 1.0
    s = set.shunts[1]
    @test s.b_min <= s.b0 <= s.b_max
    @test length(s.block_n) == length(s.block_dB)
    @test s.b0 == 0.0
    @test s.b_min == 0.0
    @test s.b_max == 0.2
    @test s.current == 0.0
    @test s.vset == (0.9 + 1.1) / 2
end

@testset "discrete control: MODSW gating" begin
    function _build(mode)
        sys = _make_tap_shunt_system()
        sa = first(PSY.get_components(PSY.SwitchedAdmittance, sys))
        PSY.set_control_mode!(sa, mode)
        data = PowerFlowData(ACPolarPowerFlow(), sys)
        return PowerFlows.build_controlled_device_set(
            sys, PF.get_bus_lookup(data), data.power_network_matrix)
    end

    # FIXED (locked): shunt not enrolled; tap unaffected.
    set0 = _build(PSY.SwitchedAdmittanceControlMode.FIXED)
    @test length(set0.shunts) == 0
    @test length(set0.taps) == 1

    # DISCRETE_VOLTAGE: enrolled, continuous == false.
    set1 = _build(PSY.SwitchedAdmittanceControlMode.DISCRETE_VOLTAGE)
    @test length(set1.shunts) == 1
    @test set1.shunts[1].continuous == false

    # CONTINUOUS_VOLTAGE: enrolled, continuous == true.
    set2 = _build(PSY.SwitchedAdmittanceControlMode.CONTINUOUS_VOLTAGE)
    @test length(set2.shunts) == 1
    @test set2.shunts[1].continuous == true

    # Remote reactive-power / remote-device control modes: unsupported ⇒ the shunt is
    # de-enrolled with a warning and stays locked (PSS/E posture); never an error.
    unsupported = (
        PSY.SwitchedAdmittanceControlMode.DISCRETE_REACTIVE_PLANT,
        PSY.SwitchedAdmittanceControlMode.DISCRETE_REACTIVE_VSC,
        PSY.SwitchedAdmittanceControlMode.DISCRETE_ADMITTANCE_REMOTE,
    )
    for mode in unsupported
        setm = @test_logs (:warn, r"not supported") match_mode = :any _build(mode)
        @test length(setm.shunts) == 0
        @test length(setm.taps) == 1   # the tap is unaffected by the bad shunt record
    end
end

@testset "discrete control: warn-and-lock on unresolvable controlled bus" begin
    # A remote controlled-bus number that does not exist in the network must
    # de-enroll the device with a warning, not abort PowerFlowData construction.
    sys = _make_tap_shunt_system()
    tx = first(PSY.get_components(PSY.TapTransformer, sys))
    PSY.set_regulated_bus_number!(tx, 99)   # no bus 99
    data = PowerFlowData(ACPolarPowerFlow(), sys)
    set = @test_logs (:warn, r"controlled bus 99") match_mode = :any (
        PowerFlows.build_controlled_device_set(
        sys, PF.get_bus_lookup(data), data.power_network_matrix)
    )
    @test length(set.taps) == 0
    @test length(set.shunts) == 1   # the shunt still enrolls
end

@testset "discrete control: tap reads first-class control fields" begin
    # The builder reads regulated_bus_number (controlled bus), tap_limits (ratio band),
    # number_of_tap_positions, and voltage_setpoint directly off the PSY component.
    sys = _make_tap_shunt_system()
    tx = first(PSY.get_components(PSY.TapTransformer, sys))
    PSY.set_regulated_bus_number!(tx, 3)
    PSY.set_tap_limits!(tx, (min = 0.88, max = 1.12))
    PSY.set_number_of_tap_positions!(tx, 25)
    PSY.set_voltage_setpoint!(tx, 1.03)
    data = PowerFlowData(ACPolarPowerFlow(), sys)
    bl = PF.get_bus_lookup(data)
    set = PowerFlows.build_controlled_device_set(
        sys, bl, data.power_network_matrix)
    @test length(set.taps) == 1
    t = set.taps[1]
    @test t.controlled_ix == bl[3]
    @test t.p_min ≈ 0.88
    @test t.p_max ≈ 1.12
    @test length(t.levels) == 25
    @test t.vset ≈ 1.03
end

@testset "discrete control: implausible vset locks the device" begin
    # An API-built shunt whose admittance_limits hold actual susceptance bounds
    # (per the PSY docstring) would yield a garbage voltage setpoint; the builder
    # must de-enroll it with a warning instead of regulating |V| toward ~0.
    sys = _make_tap_shunt_system()
    sa = first(PSY.get_components(PSY.SwitchedAdmittance, sys))
    PSY.set_admittance_limits!(sa, (min = -0.3, max = 0.3))
    data = PowerFlowData(ACPolarPowerFlow(), sys)
    set = @test_logs (:warn, r"voltage setpoint") match_mode = :any (
        PowerFlows.build_controlled_device_set(
        sys, PF.get_bus_lookup(data), data.power_network_matrix)
    )
    @test length(set.shunts) == 0
end

@testset "discrete control: PSS/E parser shunt convention (Y = BINIT)" begin
    # A full-length zeroed initial_status marks the parser (BINIT) convention: Y holds
    # the TOTAL in-service admittance, so the reachable range is spanned by the blocks
    # alone (base 0) and the control baseline sits at BINIT.
    sys = _make_tap_shunt_system()
    sa = first(PSY.get_components(PSY.SwitchedAdmittance, sys))
    PSY.set_control_mode!(sa, PSY.SwitchedAdmittanceControlMode.DISCRETE_VOLTAGE)
    PSY.set_Y!(sa, 0.0 + 0.1im)   # BINIT = 0.1 p.u. (two of the four 0.05 blocks in)
    data = PowerFlowData(ACPolarPowerFlow(), sys)
    set = PowerFlows.build_controlled_device_set(
        sys, PF.get_bus_lookup(data), data.power_network_matrix)
    s = set.shunts[1]
    @test s.b0 == 0.0            # block-counting base
    @test s.b_min == 0.0
    @test s.b_max ≈ 0.2
    @test s.current ≈ 0.1        # baseline at BINIT, inside the reachable range
end

@testset "discrete control: shunt controlled bus (regulated_bus_number)" begin
    function _shunt(regulated_bus_number = nothing)
        sys = _make_tap_shunt_system()
        sa = first(PSY.get_components(PSY.SwitchedAdmittance, sys))
        isnothing(regulated_bus_number) ||
            PSY.set_regulated_bus_number!(sa, regulated_bus_number)
        data = PowerFlowData(ACPolarPowerFlow(), sys)
        bl = PF.get_bus_lookup(data)
        set = PowerFlows.build_controlled_device_set(
            sys, bl, data.power_network_matrix)
        return set.shunts[1], bl
    end

    # regulated_bus_number set ⇒ remote controlled bus.
    s_remote, bl = _shunt(2)
    @test s_remote.controlled_ix == bl[2]

    # regulated_bus_number 0 (default) ⇒ falls back to the shunt's local bus (3).
    s_local, bl2 = _shunt()
    @test s_local.controlled_ix == bl2[3]
end

@testset "discrete control: shunt invariant validation (warn-and-lock)" begin
    # b0 outside [b_min, b_max]: warn, device de-enrolled (returns false).
    r1 = @test_logs (:warn, r"outside") match_mode = :any PowerFlows._validate_shunt(
        "bad_shunt",
        0.0, # b_min
        0.5, # b0 — above b_max, outside [b_min, b_max]
        0.2, # b_max
        [4],
        [0.05],
    )
    @test r1 == false
    # Zero-step block with nonzero dB: malformed metadata → warn + false.
    r2 = @test_logs (:warn, r"zero steps") match_mode = :any PowerFlows._validate_shunt(
        "bad_shunt2",
        0.0,  # b_min
        0.0,  # b0
        0.2,  # b_max
        [0],  # zero steps
        [0.05], # nonzero dB — malformed
    )
    @test r2 == false
    # b_min == b_max (no controllable range): warn + false.
    r3 = @test_logs (:warn, r"no controllable") match_mode = :any (
        PowerFlows._validate_shunt("no_range", 0.5, 0.5, 0.5, [0], [0.0])
    )
    @test r3 == false
    # Valid case — true, no logs.
    @test PowerFlows._validate_shunt("ok_shunt", 0.0, 0.0, 0.2, [4], [0.05]) == true
end

@testset "discrete control: tap invariant validation (warn-and-lock)" begin
    # p_min > p_max (malformed): warn + false.
    r1 = @test_logs (:warn, r"exceeds") match_mode = :any (
        PowerFlows._validate_tap("bad_tap", 1.1, 0.9, 33)
    )
    @test r1 == false
    # p_min == p_max (no controllable range): warn + false.
    r2 = @test_logs (:warn, r"no controllable") match_mode = :any (
        PowerFlows._validate_tap("no_range", 1.0, 1.0, 33)
    )
    @test r2 == false
    # ntp < 2: a degenerate/locked changer must NOT become an active 33-level
    # controller; it is de-enrolled with a warning.
    r3 = @test_logs (:warn, r"fewer than 2 tap positions") match_mode = :any (
        PowerFlows._validate_tap("one_pos", 0.9, 1.1, 1)
    )
    @test r3 == false
    # Valid case — true.
    @test PowerFlows._validate_tap("ok_tap", 0.9, 1.1, 33) == true
end

@testset "tap ratio band comes from tap_limits" begin
    sys = _make_tap_shunt_system()
    tx = first(PSY.get_components(PSY.TapTransformer, sys))
    PSY.set_tap_limits!(tx, (min = 0.9, max = 1.1))
    md = PowerFlows._tap_metadata(tx, 2)
    @test md.pmin ≈ 0.9
    @test md.pmax ≈ 1.1
end

@testset "out-of-band initial tap ratio de-enrolls with a warning" begin
    sys = _make_tap_shunt_system()
    tx = first(PSY.get_components(PSY.TapTransformer, sys))
    PSY.set_tap!(tx, 1.5)   # far outside any [0.9, 1.1]-class band
    pf = ACPolarPowerFlow{NewtonRaphsonACPowerFlow}(; control_discrete_devices = true)
    data =
        @test_logs (:warn, r"outside the tap-ratio band") match_mode = :any PowerFlowData(
            pf,
            sys,
        )
    set = PowerFlows.get_controlled_devices(data)
    @test isnothing(set) || isempty(set.taps)
end

@testset "discrete control: sigmoid law" begin
    # The raw sigmoid: midpoint at x = xset, saturating to hi (x ≪ xset) and
    # lo (x ≫ xset) when called as _sigmoid(lo, hi, ...) with hi > lo.
    S = 100.0
    @test PowerFlows._sigmoid(0.9, 1.1, S, 1.0, 1.0) ≈ 1.0 atol = 1e-9
    @test PowerFlows._sigmoid(0.9, 1.1, S, 1.5, 1.0) ≈ 0.9 atol = 1e-6
    @test PowerFlows._sigmoid(0.9, 1.1, S, 0.5, 1.0) ≈ 1.1 atol = 1e-6
    # Swapped limits give the increasing orientation.
    @test PowerFlows._sigmoid(1.1, 0.9, S, 1.5, 1.0) ≈ 1.1 atol = 1e-6
end

@testset "discrete control: negative-feedback orientation" begin
    # The effective control law `_control_target` must produce negative feedback:
    # sign(d target / dV) opposite to sign(dV/dp), so the closed-loop gain
    # g' = σ'(V)·dV/dp ≤ 0 for ANY device wiring — the orientation comes solely
    # from the measured plant sensitivity, for both signs of dV/dp.
    S = 100.0
    δ = 0.02
    tap = PowerFlows.ControlledTap("tp", 1, 2, 2, 1.0,
        1.0 / (0.01 + 0.1im), 0.0, 0.9, 1.1,
        collect(range(0.9, 1.1; length = 33)), (1, 2, 3, 4), 1.0, 1.0, 1.0)
    remote = PowerFlows.ControlledTap("ts", 1, 2, 1, 1.0,
        1.0 / (0.01 + 0.1im), 0.0, 0.9, 1.1,
        collect(range(0.9, 1.1; length = 33)), (1, 2, 3, 4), 1.0, 1.0, 1.0)
    shunt = PowerFlows.ControlledSwitchedShunt("sh", 3, 3, 1.0, 0.95, 1.05, 0.0, 0.0,
        [4], [0.05], 0.0, 0.2, zeros(Int, 1), false, 0.0, 0.0, false)
    for d in (tap, remote, shunt)
        vset = PowerFlows.voltage_setpoint(d)
        for dVdp in (1.0, -1.0)
            up = PowerFlows._control_target(d, vset + δ, S, dVdp)
            dn = PowerFlows._control_target(d, vset - δ, S, dVdp)
            # negative feedback: the slope in V and dV/dp have opposite signs.
            @test (up - dn) * dVdp <= 0.0
        end
    end
end

@testset "discrete control: relaxation yields monotone slope" begin
    # `_relaxation` must keep the damped fixed-point map slope NON-NEGATIVE
    # (monotone, no oscillation) for the worst-case gain. The slope is
    # m = 1 + ω·(g'−1) with g' ≤ 0 and |g'| ≤ gbound = 0.25|hi-lo|·S·|dVdp|, so
    # m ≥ 0 iff ω·(1+gbound) ≤ 1. (A negative slope is what previously made the
    # iterate alternate every step and tripped the oscillation-freeze detector.)
    d = PowerFlows.ControlledTap("t", 1, 2, 2, 1.0, 1.0 / (0.01 + 0.1im),
        0.0, 0.9, 1.1, collect(range(0.9, 1.1; length = 33)),
        (1, 2, 3, 4), 1.0, 1.0, 1.0)
    lo, hi = PowerFlows.parameter_limits(d)
    for S in (1.0e2, 1.0e3, 5.0e3), dVdp in (-5.0, -1.0, -0.1, 0.1, 1.0, 5.0)
        ω = PowerFlows._relaxation(d, S, dVdp)
        gbound = 0.25 * abs(hi - lo) * S * abs(dVdp)
        # ω = (1−θ)/(1+gbound) ≤ 1−θ = 0.5 by construction (no separate cap).
        @test 0.0 < ω <= 1.0 - PowerFlows.CONTROL_CONTRACTION
        @test ω * (1.0 + gbound) <= 1.0 + 1e-12   # ⇒ map slope m ≥ 0 (monotone)
    end
end

@testset "discrete control: ramp completes, tight regulation" begin
    # After the monotone-convergence fix the steepness ramp runs to completion
    # (no oscillation freeze at the initial steepness), so the controlled bus is
    # regulated to within one discrete tap step — previously it stalled ~5e-3 off
    # and only passed under a much looser +1e-2 tolerance.
    sys = _make_solvable_tap_shunt_system()
    pf = ACPolarPowerFlow(; control_discrete_devices = true)
    data = PowerFlowData(pf, sys)
    solve_power_flow!(data)
    @test all(data.converged)
    t = data.controlled_devices.taps[1]
    @test t.current in t.levels
    spacing = (t.p_max - t.p_min) / (length(t.levels) - 1)
    @test abs(data.bus_magnitude[t.controlled_ix, 1] - t.vset) < spacing
end

@testset "discrete control: continuous shunt not grid-pinned" begin
    sys = _make_solvable_tap_shunt_system()
    sa = first(PSY.get_components(PSY.SwitchedAdmittance, sys))
    PSY.set_control_mode!(sa, PSY.SwitchedAdmittanceControlMode.CONTINUOUS_VOLTAGE)
    pf = ACPolarPowerFlow(; control_discrete_devices = true)
    data = PowerFlowData(pf, sys)
    solve_power_flow!(data)
    @test all(data.converged)
    sh = data.controlled_devices.shunts[1]
    @test sh.continuous == true
    # final susceptance stays within the controllable band (continuous, not snapped).
    @test sh.b_min - 1e-9 <= sh.current <= sh.b_max + 1e-9
end

@testset "discrete control: snap" begin
    d = PowerFlows.ControlledTap("t", 1, 2, 2, 1.0, 1.0 + 0im,
        0.0, 0.9, 1.1, collect(range(0.9, 1.1; length = 5)),
        (1, 2, 3, 4), 1.0, 1.0, 1.0)  # levels: 0.9,0.95,1.0,1.05,1.1
    @test PowerFlows.snap_to_discrete(d, 1.03) == 1.05
    @test PowerFlows.snap_to_discrete(d, 1.20) == 1.1   # clamp
    block_dB_sh_snap = [0.05]
    sh = PowerFlows.ControlledSwitchedShunt("s", 3, 3, 1.0, 0.95, 1.05, 0.0, 0.0,
        [4], block_dB_sh_snap, 0.0, 0.2,
        zeros(Int, length(block_dB_sh_snap)),
        false, 0.0, 0.0, false)  # reachable: 0,0.05,0.10,0.15,0.20
    @test PowerFlows.snap_to_discrete(sh, 0.12) == 0.10
    @test sh.block_n == [2]
    block_dB_sh2 = [0.1, 0.02]
    sh2 = PowerFlows.ControlledSwitchedShunt("s2", 3, 3, 1.0, 0.95, 1.05, 0.0, 0.0,
        [2, 3], block_dB_sh2, 0.0, 0.26,
        zeros(Int, length(block_dB_sh2)),
        false, 0.0, 0.0, false)  # PSS/E cumulative chain: blocks activate in listed order
    # Chain totals: 0.1, 0.2 (block 1), then 0.22, 0.24, 0.26 (block 2).
    # 0.16 (= 1×0.1 + 3×0.02 with block 1 partially on) is NOT physically reachable;
    # nearest chain point to 0.17 is 0.2.
    @test PowerFlows.snap_to_discrete(sh2, 0.17) == 0.2
    @test sh2.block_n == [2, 0]
end

@testset "mixed-sign shunt snap reaches both chain sides" begin
    d = PowerFlows.ControlledSwitchedShunt(
        "mixed", 1, 1, 1.0, 0.95, 1.05, 0.0,
        0.0,                    # b0: all blocks off = neutral
        [1, 1],                 # one reactor step, one capacitor step
        [-0.5, 0.5],            # reactor listed first (RAW convention)
        -0.5, 0.5,              # envelope
        [0, 0], false, 0.0, 0.0, false)
    @test PowerFlows.snap_to_discrete(d, 0.45) == 0.5    # capacitive side reachable
    @test PowerFlows.snap_to_discrete(d, -0.45) == -0.5  # reactive side reachable
    @test iszero(PowerFlows.snap_to_discrete(d, 0.1))    # neutral still nearest
    @test (@allocated PowerFlows.snap_to_discrete(d, 0.45)) == 0
end

@testset "discrete control: continuous shunt no snap" begin
    # continuous == true ⇒ snap_to_discrete returns the clamped continuous value,
    # NOT the nearest reachable block grid point.
    block_dB = [0.05]
    cont = PowerFlows.ControlledSwitchedShunt("c", 3, 3, 1.0, 0.95, 1.05, 0.0, 0.0,
        [4], block_dB, 0.0, 0.2, zeros(Int, length(block_dB)), true, 0.0, 0.0, false)
    # 0.12 is between grid points 0.10 and 0.15; continuous must return it unchanged.
    @test PowerFlows.snap_to_discrete(cont, 0.12) == 0.12
    # clamped at the rails.
    @test PowerFlows.snap_to_discrete(cont, 0.30) == 0.2
    @test PowerFlows.snap_to_discrete(cont, -0.10) == 0.0
    # sanity: the discrete twin DOES snap 0.12 → 0.10.
    disc = PowerFlows.ControlledSwitchedShunt("d", 3, 3, 1.0, 0.95, 1.05, 0.0, 0.0,
        [4], block_dB, 0.0, 0.2, zeros(Int, length(block_dB)), false, 0.0, 0.0, false)
    @test PowerFlows.snap_to_discrete(disc, 0.12) == 0.10
end

@testset "discrete control: in-place Ybus == rebuild" begin
    sys = _make_tap_shunt_system()
    data = PowerFlowData(ACPolarPowerFlow(), sys)
    bus_lookup = PF.get_bus_lookup(data)
    ybus = data.power_network_matrix
    set = PowerFlows.build_controlled_device_set(sys, bus_lookup, ybus)
    @test length(set.taps) == 1
    d = set.taps[1]

    # Verify nz_offsets are now real (not all-ones stub).
    A = ybus.data
    fix = d.from_ix
    tix = d.to_ix
    @test A[fix, fix] ≈ A.nzval[d.nz_offsets[1]]
    @test A[fix, tix] ≈ A.nzval[d.nz_offsets[2]]
    @test A[tix, fix] ≈ A.nzval[d.nz_offsets[3]]
    @test A[tix, tix] ≈ A.nzval[d.nz_offsets[4]]

    # Apply a new tap ratio in-place and compare against a fresh rebuild.
    newtap = 1.05
    PowerFlows.apply_parameter!(d, data, newtap, 1)
    after = copy(A)

    # Rebuild reference: mutate sys, build fresh data2.
    txs = collect(PSY.get_components(PSY.TapTransformer, sys))
    PSY.set_tap!(txs[1], newtap)
    data2 = PowerFlowData(ACPolarPowerFlow(), sys)
    A2 = data2.power_network_matrix.data
    @test after ≈ A2

    # Parallel-branch variant: a Line paralleling the transformer aggregates
    # contributions in the same (fix,tix) off-diagonal slots.  Delta update is
    # required to preserve the parallel branch's contribution.
    @testset "discrete control: in-place Ybus == rebuild (parallel branch)" begin
        sys2 = _make_tap_shunt_system()
        txs2 = collect(PSY.get_components(PSY.TapTransformer, sys2))
        arc2 = PSY.get_arc(txs2[1])
        from_bus = PSY.get_from(arc2)
        to_bus = PSY.get_to(arc2)
        # Add a Line with distinct r,x so the off-diagonal slot accumulates two
        # different values (not a zero-sum cancellation of contributions).
        _add_simple_line!(sys2, from_bus, to_bus, 0.05, 0.15, 0.0)
        data2 = PowerFlowData(ACPolarPowerFlow(), sys2)
        bl = PF.get_bus_lookup(data2)
        ybus2 = data2.power_network_matrix
        set2 = PowerFlows.build_controlled_device_set(sys2, bl, ybus2)
        @test length(set2.taps) == 1
        d2 = set2.taps[1]
        newtap2 = 1.05
        PowerFlows.apply_parameter!(d2, data2, newtap2, 1)
        after2 = copy(ybus2.data)
        # Rebuild reference.
        PSY.set_tap!(txs2[1], newtap2)
        data2r = PowerFlowData(ACPolarPowerFlow(), sys2)
        @test after2 ≈ data2r.power_network_matrix.data
    end

    # Zero-allocation check: nzval writes only, no heap traffic.
    let
        f() = PowerFlows.apply_parameter!(d, data, 1.05, 1)
        f()                       # warm-up: force specialization
        @test (@allocated f()) == 0
    end
end

@testset "discrete control: PowerFlowData wiring" begin
    sys = _make_tap_shunt_system()
    # Default (control_discrete_devices = false): controlled_devices is nothing.
    data_off = PowerFlowData(ACPolarPowerFlow(), sys)
    @test isnothing(data_off.controlled_devices)

    # Enabled: controlled_devices is a ControlledDeviceSet with expected contents.
    data_on = PowerFlowData(ACPolarPowerFlow(; control_discrete_devices = true), sys)
    @test data_on.controlled_devices isa PowerFlows.ControlledDeviceSet
    @test length(data_on.controlled_devices.taps) == 1
    @test length(data_on.controlled_devices.shunts) == 1
end

@testset "discrete control: disabled == baseline" begin
    sys = _make_tap_shunt_system()
    a = PowerFlowData(ACPolarPowerFlow(), sys)
    b = PowerFlowData(ACPolarPowerFlow(; control_discrete_devices = false), sys)
    @test isnothing(a.controlled_devices)
    @test isnothing(b.controlled_devices)
end

@testset "discrete control: tap+shunt converges (NR)" begin
    # _make_solvable_tap_shunt_system has V_2 ~0.968 at tap=1.0; the tap has
    # full authority (dV/dp ≈ -1.04) and V_2 = vset = 1.0 is reachable at
    # tap ≈ 0.973, well inside [0.9, 1.1].  The damped steepness-homotopy
    # controller must drive the controlled bus into the vset deadband.
    sys = _make_solvable_tap_shunt_system()
    pf = ACPolarPowerFlow(; control_discrete_devices = true)
    data = PowerFlowData(pf, sys)
    solve_power_flow!(data)
    @test all(data.converged)
    t = data.controlled_devices.taps[1]
    @test t.current in t.levels
    @test abs(data.bus_magnitude[t.controlled_ix, 1] - t.vset) <=
          (t.p_max - t.p_min) / length(t.levels) + 1e-2
end

@testset "discrete control: primary-controlled tap orientation (NR)" begin
    # Exercises from-side (primary) control: the controlled bus is the tap's FROM
    # bus, so the measured plant sensitivity dV/dp has the opposite sign to the
    # usual to-side wiring, and `_control_target` must pick the orientation that
    # keeps the closed-loop gain negative. The to-side tests do NOT cover this.
    sys = _make_primary_controlled_tap_system()
    pf = ACPolarPowerFlow(; control_discrete_devices = true)
    data = PowerFlowData(pf, sys)
    solve_power_flow!(data)
    @test all(data.converged)
    t = data.controlled_devices.taps[1]
    # The controlled bus is the tap's FROM bus (set via regulated_bus_number = 2).
    @test t.controlled_ix == t.from_ix
    @test t.current in t.levels
    @test abs(data.bus_magnitude[t.controlled_ix, 1] - t.vset) <=
          (t.p_max - t.p_min) / length(t.levels) + 1e-2
end

@testset "discrete control: snap + restore" begin
    sys = _make_solvable_tap_shunt_system()
    pf = ACPolarPowerFlow(; control_discrete_devices = true)
    data = PowerFlowData(pf, sys)
    solve_power_flow!(data)
    t = data.controlled_devices.taps[1]
    @test t.current in t.levels
    @test all(data.converged)
end

@testset "discrete control: refactor is no-op (disabled path)" begin
    # Verify that _ac_power_flow with no controlled devices routes through
    # _solve_with_q_limits! unchanged (pure code-motion regression check).
    sys = PSB.build_system(PSB.PSITestSystems, "c_sys14"; add_forecasts = false)
    set_units_base_system!(sys, UnitSystem.SYSTEM_BASE)
    pf = ACPolarPowerFlow()
    data1 = PowerFlowData(pf, sys)
    data2 = PowerFlowData(pf, sys)
    @test isnothing(data1.controlled_devices)
    converged1 = solve_power_flow!(data1)
    converged2 = solve_power_flow!(data2)
    @test converged1
    @test converged2
    @test all(isapprox.(data1.bus_magnitude, data2.bus_magnitude; atol = 1e-12))
    @test all(isapprox.(data1.bus_angles, data2.bus_angles; atol = 1e-12))
end

@testset "discrete control: NR vs TR" begin
    # Same solvable fixture solved with NR and TR; the continuation engine must
    # drive the controlled bus into the strict vset deadband for both solvers.
    sys_nr = _make_solvable_tap_shunt_system()
    pf_nr = ACPolarPowerFlow{NewtonRaphsonACPowerFlow}(; control_discrete_devices = true)
    data_nr = PowerFlowData(pf_nr, sys_nr)
    solve_power_flow!(data_nr)
    @test all(data_nr.converged)
    t_nr = data_nr.controlled_devices.taps[1]
    @test t_nr.current in t_nr.levels
    @test abs(data_nr.bus_magnitude[t_nr.controlled_ix, 1] - t_nr.vset) <=
          (t_nr.p_max - t_nr.p_min) / length(t_nr.levels) + 1e-2

    sys_tr = _make_solvable_tap_shunt_system()
    pf_tr = ACPolarPowerFlow{TrustRegionACPowerFlow}(; control_discrete_devices = true)
    data_tr = PowerFlowData(pf_tr, sys_tr)
    solve_power_flow!(data_tr)
    @test all(data_tr.converged)
    t_tr = data_tr.controlled_devices.taps[1]
    @test t_tr.current in t_tr.levels
    @test abs(data_tr.bus_magnitude[t_tr.controlled_ix, 1] - t_tr.vset) <=
          (t_tr.p_max - t_tr.p_min) / length(t_tr.levels) + 1e-2

    # Both solvers must agree on the controlled-bus voltage within a tight tolerance.
    @test isapprox(
        data_nr.bus_magnitude[t_nr.controlled_ix, 1],
        data_tr.bus_magnitude[t_tr.controlled_ix, 1];
        atol = 1e-4,
    )
end

@testset "discrete control: formulation-agnostic" begin
    # The outer continuation loop reads data.bus_magnitude[controlled_ix, ts]
    # after each inner solve.  For the solvable fixture the controlled bus is PQ,
    # so bus_magnitude is updated on every residual call for all three
    # formulations (polar: _update_residual_values!, rectangular CI:
    # rect_update_data!, mixed CPB: mixed_update_data!).  All three must
    # converge and satisfy the strict vset deadband.
    formulations = (
        ACPolarPowerFlow{NewtonRaphsonACPowerFlow},
        ACRectangularPowerFlow{NewtonRaphsonACPowerFlow},
        ACMixedPowerFlow{NewtonRaphsonACPowerFlow},
    )
    vmags = Float64[]
    for F in formulations
        sys = _make_solvable_tap_shunt_system()
        pf = F(; control_discrete_devices = true)
        data = PowerFlowData(pf, sys)
        solve_power_flow!(data)
        @test all(data.converged)
        t = data.controlled_devices.taps[1]
        @test t.current in t.levels
        @test abs(data.bus_magnitude[t.controlled_ix, 1] - t.vset) <=
              (t.p_max - t.p_min) / length(t.levels) + 1e-2
        push!(vmags, data.bus_magnitude[t.controlled_ix, 1])
    end
    # All three formulations must agree on the regulated voltage.
    @test maximum(vmags) - minimum(vmags) < 1e-4
end

@testset "discrete control: SVC holds weak bus to setpoint (NR)" begin
    # Ample reactive capability (100 MVA ⇒ b_lim = 1.0 p.u.): the SVC must inject
    # reactive power to drive the weak PQ bus up to its voltage_setpoint (1.0 p.u.).
    sys = _make_svc_system(; max_shunt_current = 100.0)
    pf = ACPolarPowerFlow(; control_discrete_devices = true)
    data = PowerFlowData(pf, sys)
    solve_power_flow!(data)
    @test all(data.converged)
    f = data.controlled_devices.facts[1]
    @test abs(data.bus_magnitude[f.controlled_ix, 1] - f.vset) <= 5e-3
    # Regulating, not saturated at a susceptance limit.
    @test -f.b_lim < f.current < f.b_lim
end

@testset "discrete control: SVC clamps at reactive limit (NR)" begin
    # Tight reactive capability (5 MVA ⇒ b_lim = 0.05 p.u.) cannot supply the
    # bus's reactive deficit: the SVC saturates at b_lim and the bus stays below
    # setpoint — the homotopy equivalent of the PV→PQ Q-limit release.
    sys = _make_svc_system(; max_shunt_current = 5.0)
    pf = ACPolarPowerFlow(; control_discrete_devices = true)
    data = PowerFlowData(pf, sys)
    solve_power_flow!(data)
    @test all(data.converged)
    f = data.controlled_devices.facts[1]
    @test data.bus_magnitude[f.controlled_ix, 1] < f.vset - 1e-3
    @test isapprox(f.current, f.b_lim; atol = 1e-3)
end

@testset "FACTS classification: regulating device is not saturated; results carry Q" begin
    # Ample capability: the device reaches setpoint with headroom, so it is not saturated
    # and never freezes short of both its limit and its setpoint.
    sys = _make_svc_system(; max_shunt_current = 100.0)
    pf = ACPolarPowerFlow(; control_discrete_devices = true)
    data = PowerFlowData(pf, sys)
    solve_power_flow!(data)
    f = data.controlled_devices.facts[1]
    @test !f.saturated
    at_setpoint = abs(data.bus_magnitude[f.controlled_ix, 1] - f.vset) <= 5e-3
    at_limit = abs(f.current) >= (1.0 - 1e-2) * f.b_lim
    @test at_setpoint || at_limit          # converged to setpoint OR genuinely at the bound
    @test !(f.saturated && at_setpoint)    # never both
    res = PowerFlows.get_controlled_device_results(data)
    frow = only(eachrow(res[res.family .== "FACTSControlDevice", :]))
    @test frow.saturated == false
    v_local = data.bus_magnitude[f.bus_ix, 1]
    @test frow.delivered_q_mvar ≈ f.current * v_local^2 * f.base_mva
    # Neutral columns for non-FACTS devices stay rectangular (no FACTS-only leakage).
    @test all(ismissing, res[res.family .!= "FACTSControlDevice", :delivered_q_mvar])
end

@testset "FACTS classification: clamped device is saturated and warns" begin
    # Tight capability cannot hold the setpoint: the device pins at b_lim off setpoint and is
    # reported saturated with a warning.
    sys = _make_svc_system(; max_shunt_current = 5.0)
    pf = ACPolarPowerFlow(; control_discrete_devices = true)
    data = PowerFlowData(pf, sys)
    @test_logs (:warn, r"saturated") match_mode = :any solve_power_flow!(data)
    f = data.controlled_devices.facts[1]
    @test f.saturated
    res = PowerFlows.get_controlled_device_results(data)
    frow = only(eachrow(res[res.family .== "FACTSControlDevice", :]))
    @test frow.saturated == true
end

@testset "FACTS write-back sets reactive_power_required" begin
    sys = _make_svc_system(; max_shunt_current = 100.0)
    pf = ACPolarPowerFlow(; control_discrete_devices = true)
    data = PowerFlowData(pf, sys)
    solve_power_flow!(data)
    PowerFlows.write_device_settings!(sys, data)
    fd = only(get_components(PSY.FACTSControlDevice, sys))
    # Solver-populated delivered Q = b·|V|²·base_mva at the device bus (capacitive ⇒ > 0).
    @test PSY.get_reactive_power_required(fd) > 0.0
end

@testset "discrete control: tap reads first-class PSY fields (#1684)" begin
    # Controllability set via the first-class PSY 5.12 fields (no ext); the builder
    # reads them directly.
    sys = _make_field_controlled_tap_system()
    pf = ACPolarPowerFlow(; control_discrete_devices = true)
    data = PowerFlowData(pf, sys)
    t = data.controlled_devices.taps[1]
    @test t.p_min ≈ 0.85
    @test t.p_max ≈ 1.15
    @test length(t.levels) == 17
    @test t.vset ≈ 1.02
    # regulated_bus_number = 3 ⇒ remote controlled bus, not the to-bus.
    @test t.controlled_ix == PNM.get_bus_lookup(data.power_network_matrix)[3]
end

@testset "discrete control: construction guards" begin
    # Only NR/TR inner solvers are validated for the continuation.
    @test_throws ArgumentError ACPolarPowerFlow{LevenbergMarquardtACPowerFlow}(;
        control_discrete_devices = true)
    @test_throws ArgumentError ACPolarPowerFlow{FastDecoupledACPowerFlow}(;
        control_discrete_devices = true)
    @test_throws ArgumentError ACRectangularPowerFlow{LevenbergMarquardtACPowerFlow}(;
        control_discrete_devices = true)
    @test_throws ArgumentError ACMixedPowerFlow{LevenbergMarquardtACPowerFlow}(;
        control_discrete_devices = true)
    # Device state does not track per-time-step baselines yet.
    @test_throws ArgumentError ACPolarPowerFlow(;
        control_discrete_devices = true, time_steps = 2)
    # NR/TR with a single time step construct fine.
    @test ACPolarPowerFlow{TrustRegionACPowerFlow}(;
        control_discrete_devices = true) isa ACPolarPowerFlow
    # LCC systems are rejected at PowerFlowData construction (rollback does not
    # cover the per-time-step LCC state).
    lcc_sys, _ = simple_lcc_system()
    @test_throws ArgumentError PowerFlowData(
        ACPolarPowerFlow(; control_discrete_devices = true), lcc_sys)
end

@testset "discrete control: FACTS enrolls without a flag; taps/shunts unaffected" begin
    sys = _make_svc_system(; max_shunt_current = 100.0)
    pf = ACPolarPowerFlow(; control_discrete_devices = true)
    data = PowerFlowData(pf, sys)
    @test length(data.controlled_devices.facts) == 1
end

@testset "discrete control: arc-admittance rows synced to final parameters" begin
    # Reported branch flows are computed from arc_admittance_from_to/to_from AFTER the
    # time-step loop; the post-continuation sync must bring the moved devices' rows in
    # line with their final tap, exactly matching a fresh rebuild at that tap.
    sys = _make_tap_shunt_system()
    data = PowerFlowData(ACPolarPowerFlow(), sys)
    set = PowerFlows.build_controlled_device_set(
        sys, PF.get_bus_lookup(data), data.power_network_matrix)
    d = set.taps[1]
    newtap = 1.05
    PowerFlows.apply_parameter!(d, data, newtap, 1)
    PowerFlows._sync_arc_admittances!(data, set)
    # Rebuild reference at the new tap.
    txs = collect(PSY.get_components(PSY.TapTransformer, sys))
    PSY.set_tap!(txs[1], newtap)
    data2 = PowerFlowData(ACPolarPowerFlow(), sys)
    @test data.power_network_matrix.arc_admittance_from_to.data ≈
          data2.power_network_matrix.arc_admittance_from_to.data
    @test data.power_network_matrix.arc_admittance_to_from.data ≈
          data2.power_network_matrix.arc_admittance_to_from.data
    # Idempotency: a second sync with no parameter change is a no-op.
    PowerFlows._sync_arc_admittances!(data, set)
    @test data.power_network_matrix.arc_admittance_from_to.data ≈
          data2.power_network_matrix.arc_admittance_from_to.data
end

@testset "discrete control: reported flows match a rebuild at the snapped tap" begin
    # End-to-end parity: solve with control, then rebuild the system at the solver's
    # final tap and re-solve WITHOUT control — voltages AND reported branch flows must
    # agree, proving the flow computation saw the same network as the inner solver.
    sys = _make_solvable_tap_shunt_system()
    pf = ACPolarPowerFlow(; control_discrete_devices = true)
    data = PowerFlowData(pf, sys)
    solve_power_flow!(data)
    @test all(data.converged)
    t = data.controlled_devices.taps[1]
    sh = data.controlled_devices.shunts[1]
    # Rebuild: same system with the tap fixed at the solved position and the shunt's
    # solved susceptance as a fixed admittance baseline.
    txs = collect(PSY.get_components(PSY.TapTransformer, sys))
    PSY.set_tap!(txs[1], t.current)
    # data_ref is built WITHOUT control_discrete_devices, so the VOLTAGE objective is
    # inert: this is a plain solve of the snapped network.
    sas = collect(PSY.get_components(PSY.SwitchedAdmittance, sys))
    PSY.set_Y!(sas[1], PSY.get_Y(sas[1]) + im * (sh.current - sh.initial))
    data_ref = PowerFlowData(ACPolarPowerFlow(), sys)
    solve_power_flow!(data_ref)
    @test all(data_ref.converged)
    @test all(isapprox.(data.bus_magnitude, data_ref.bus_magnitude; atol = 1e-6))
    @test all(
        isapprox.(
            data.arc_active_power_flow_from_to,
            data_ref.arc_active_power_flow_from_to;
            atol = 1e-5,
        ),
    )
    @test all(
        isapprox.(
            data.arc_reactive_power_flow_from_to,
            data_ref.arc_reactive_power_flow_from_to;
            atol = 1e-5,
        ),
    )
end

@testset "discrete control: solved device settings surface" begin
    sys = _make_solvable_tap_shunt_system()
    pf = ACPolarPowerFlow(; control_discrete_devices = true)
    data = PowerFlowData(pf, sys)
    solve_power_flow!(data)
    df = PowerFlows.get_controlled_device_results(data)
    @test size(df, 1) == 2
    tap_row = df[df.family .== "TapTransformer", :]
    @test tap_row.name == ["tap_1_2"]
    @test tap_row.initial == [1.0]
    t = data.controlled_devices.taps[1]
    @test tap_row.final == [t.current]
    @test tap_row.lower_limit == [t.p_min]
    @test tap_row.upper_limit == [t.p_max]
    # Disabled path: empty frame with the same schema.
    data_off = PowerFlowData(ACPolarPowerFlow(), _make_tap_shunt_system())
    df_off = PowerFlows.get_controlled_device_results(data_off)
    @test size(df_off, 1) == 0
    @test names(df_off) ==
          ["family", "name", "lower_limit", "upper_limit", "initial", "final",
        "delivered_q_mvar", "saturated"]
end

@testset "discrete control: shunt deadband semantics" begin
    # In-band voltages hold the device (PSS/E VSWLO/VSWHI semantics); out-of-band do not.
    sh = PowerFlows.ControlledSwitchedShunt("s", 3, 3, 1.0, 0.95, 1.05, 0.0, 0.0,
        [4], [0.05], 0.0, 0.2, zeros(Int, 1), false, 0.0, 0.0, false)
    @test PowerFlows._in_deadband(sh, 1.0)
    @test PowerFlows._in_deadband(sh, 0.96)
    @test !PowerFlows._in_deadband(sh, 0.94)
    @test !PowerFlows._in_deadband(sh, 1.06)
    # Taps carry a point setpoint: never in a deadband.
    tap = PowerFlows.ControlledTap("t", 1, 2, 2, 1.0, 1.0 + 0im,
        0.0, 0.9, 1.1, collect(range(0.9, 1.1; length = 5)),
        (1, 2, 3, 4), 1.0, 1.0, 1.0)
    @test !PowerFlows._in_deadband(tap, 1.0)
    # Scale-aware settle tolerance: wide-range devices get a relative floor.
    @test PowerFlows._param_tol(tap) ≈
          max(PowerFlows.CONTROL_PARAM_TOL, PowerFlows.CONTROL_PARAM_RTOL * 0.2)
    wide = PowerFlows.ControlledSwitchedShunt("w", 3, 3, 1.0, 0.95, 1.05, 0.0, 0.0,
        [10], [1.0], 0.0, 10.0, zeros(Int, 1), false, 0.0, 0.0, false)
    @test PowerFlows._param_tol(wide) ≈ PowerFlows.CONTROL_PARAM_RTOL * 10.0
end

@testset "discrete control: inner-solve budget regression" begin
    # Iteration counts (not wall-clock) are the robust performance metric. With
    # full-step-first continuation, single-solve probes, and per-stage budgets, the
    # 2-device solvable fixture regulates in well under 150 inner solves; the pre-fix
    # engine needed >500 (16-sub-step walks per move, 2 solves per probe). Budget has
    # ~2x headroom over the measured count to absorb solver-version noise.
    sys = _make_solvable_tap_shunt_system()
    pf = ACPolarPowerFlow(; control_discrete_devices = true)
    data = PowerFlowData(pf, sys)
    solve_power_flow!(data)
    @test all(data.converged)
    n = PowerFlows.get_control_inner_solve_count(data)
    @test 0 < n < 300
    # Disabled path reports zero.
    data_off = PowerFlowData(ACPolarPowerFlow(), sys)
    solve_power_flow!(data_off)
    @test PowerFlows.get_control_inner_solve_count(data_off) == 0
end

@testset "discrete control: device-settings write-back" begin
    # A controlled solve moves devices; solve_and_store writes the solved settings back
    # into the system so the stored branch flows stay self-consistent with the device
    # state (write_power_flow_solution! recomputes flows from the system components and
    # asserts they match the stored moved-device flows). Write-back is therefore automatic
    # when controls are active.
    sys = _make_solvable_tap_shunt_system()
    tx0 = first(PSY.get_components(PSY.TapTransformer, sys))
    tap_before = PSY.get_tap(tx0)
    levels =
        PowerFlowData(
            ACPolarPowerFlow(; control_discrete_devices = true), sys,
        ).controlled_devices.taps[1].levels
    pf = ACPolarPowerFlow(; control_discrete_devices = true)
    @test solve_and_store_power_flow!(pf, sys)
    @test PSY.get_tap(tx0) != tap_before   # the fixture regulates away from tap = 1.0
    @test PSY.get_tap(tx0) in levels       # the written tap is a valid discrete level
end

@testset "write-back round-trips the API shunt convention" begin
    sys = _make_tap_shunt_system()
    sa = first(PSY.get_components(PSY.SwitchedAdmittance, sys))
    # A nonzero initial_status marks the API convention (Y is the fixed base,
    # initial_status meaningful), as opposed to the parser's zeroed-status BINIT
    # convention.
    # One block already switched on at enrollment; the fixture's wide deadband holds the
    # shunt here (it never re-snaps), so this also exercises the never-snapped realizability
    # guard: block_n stays [0] while d.current reflects the pre-activated block.
    PSY.set_initial_status!(sa, [1])
    pf = ACPolarPowerFlow{NewtonRaphsonACPowerFlow}(; control_discrete_devices = true)
    data = PowerFlowData(pf, sys)
    ts = 1
    @test PowerFlows.solve_power_flow!(data)
    PowerFlows.write_device_settings!(sys, data)
    solved =
        PowerFlows.current_parameter(first(PowerFlows.get_controlled_devices(data).shunts))
    # Re-enrolling from the written-back system must reproduce the solved susceptance,
    # not double-count it.
    data2 = PowerFlowData(pf, sys)
    d2 = first(PowerFlows.get_controlled_devices(data2).shunts)
    @test PowerFlows.current_parameter(d2) ≈ solved atol = 1e-9
end

@testset "write-back hits the primary API shunt branch on a genuine snap" begin
    # _make_shunt_snap_system's weak bus needs 3 of 4 discrete steps to reach the
    # deadband, so the controlled solve snaps block_n to a nonzero, non-saturated
    # count. Unlike the never-snapped fixture above, this exercises the PRIMARY
    # (realizable) branch of write_device_settings!, not the fallback.
    sys = _make_shunt_snap_system()
    sa0 = first(PSY.get_components(PSY.SwitchedAdmittance, sys))
    @test isempty(PSY.get_initial_status(sa0))   # API convention, not the BINIT marker
    pf = ACPolarPowerFlow{NewtonRaphsonACPowerFlow}(; control_discrete_devices = true)
    data = PowerFlowData(pf, sys)
    @test PowerFlows.solve_power_flow!(data)
    d = first(PowerFlows.get_controlled_devices(data).shunts)
    @test !iszero(sum(d.block_n))
    PowerFlows.write_device_settings!(sys, data)
    sa = PSY.get_component(PSY.SwitchedAdmittance, sys, d.name)
    @test PSY.get_Y(sa) ≈ Complex(d.g0, d.b0)
    @test PSY.get_initial_status(sa) == d.block_n
    data2 = PowerFlowData(pf, sys)
    d2 = first(PowerFlows.get_controlled_devices(data2).shunts)
    @test PowerFlows.current_parameter(d2) ≈ d.current atol = 1e-9
end

@testset "sub-noise secant samples do not freeze devices" begin
    sys = _make_tap_shunt_system()
    pf = ACPolarPowerFlow{NewtonRaphsonACPowerFlow}(; control_discrete_devices = true)
    data = PowerFlowData(pf, sys)
    ts = 1
    @test PowerFlows._solve_with_q_limits!(pf, data, ts)
    set = PowerFlows.get_controlled_devices(data)
    d = first(set.taps)
    frozen = [false]
    dVdp = [0.05]
    # A sign-flipped but sub-floor sample must be ignored: gain unchanged, not frozen.
    PowerFlows._maybe_refresh_gain!(
        d,
        1,
        data,
        ts,
        frozen,
        dVdp,
        0.05,
        -5.0e-6 / 2.0e-5,
        5.0e-6,
    )
    @test !frozen[1]
    @test dVdp[1] == 0.05
end

@testset "discrete control: linearized plant sensitivity matches FD probe (P2)" begin
    # The linearized sensitivity dy/dp = (−J⁻¹ ∂F/∂p)[Vm(controlled)] must agree with the
    # finite-difference probe in SIGN and magnitude (the FD probe carries O(δ) truncation, so
    # the linear form is if anything more accurate). A sign error here would silently invert a
    # control's feedback direction — the highest-consequence P2 failure mode.
    sys = _make_solvable_tap_shunt_system()
    pf = ACPolarPowerFlow(; control_discrete_devices = true)
    data = PowerFlowData(pf, sys)
    PowerFlows._solve_with_q_limits!(pf, data, 1)   # converge base case only
    set = data.controlled_devices
    ctx = PowerFlows._sensitivity_context(pf, data, 1)
    @test !isnothing(ctx)
    scratch_snap = PowerFlows._snapshot_state(data, 1)
    for d in (set.taps[1], set.shunts[1])
        lin, ok_lin = PowerFlows._linear_plant_sign(d, data, 1, ctx)
        fd, ok_fd = PowerFlows._plant_sign(d, data, 1, pf, scratch_snap)
        @test ok_lin && ok_fd
        @test sign(lin) == sign(fd)
        @test isapprox(lin, fd; rtol = 1e-2)
    end
end

@testset "discrete control: P2 keeps symbolic factorization at 2 per continuation" begin
    # P1 (symbolic reuse) + P2 (the sensitivity context reuses that symbolic factor, refactors
    # numerically): the whole continuation performs exactly two counted builds — one for the
    # first `_ctrl_solve!`'s persisted `PolarNRCache` (the real KLU symbolic factorization) and
    # one for the single full `_sensitivity_context` build in the probe phase. Every subsequent
    # batched-pass gain refresh goes through `_refresh_sensitivity_context!` (values-only,
    # reusing both the KLU symbolic factor AND the persisted residual/Jacobian objects), so
    # neither count grows with the number of passes.
    sys = _make_solvable_tap_shunt_system()
    pf = ACPolarPowerFlow(; control_discrete_devices = true)
    data = PowerFlowData(pf, sys)
    solve_power_flow!(data)
    @test all(data.converged)
    @test PowerFlows.get_control_symbolic_factor_count(data) == 2
    @test PowerFlows.get_control_inner_solve_count(data) > 1
end

@testset "discrete control: batched passes keep inner solves ~flat in device count (P3)" begin
    # P3 does one inner solve per PASS (not per device). On a set of decoupled controlled
    # feeders the inner-solve count must stay ~flat as the device count grows — the sequential
    # path would scale it ~linearly. Build K feeders (REF ─tap─ PQ-load, REF ─line─ PQ-shunt).
    function _feeders(K)
        sys = System(100.0)
        ref = _add_simple_bus!(sys, 1, ACBusTypes.REF, 230, 1.0, 0.0)
        _add_simple_source!(sys, ref, 0.0, 0.0)
        for k in 1:K
            bl = _add_simple_bus!(sys, 2k, ACBusTypes.PQ, 230, 1.0, 0.0)
            bs = _add_simple_bus!(sys, 2k + 1, ACBusTypes.PQ, 230, 1.0, 0.0)
            add_component!(
                sys,
                PowerLoad(; name = "l$k", available = true, bus = bl,
                    active_power = 0.5, reactive_power = 0.25, base_power = 100.0,
                    max_active_power = 100.0, max_reactive_power = 100.0),
            )
            add_component!(
                sys,
                PowerLoad(; name = "s$k", available = true, bus = bs,
                    active_power = 0.05, reactive_power = 0.025, base_power = 100.0,
                    max_active_power = 100.0, max_reactive_power = 100.0),
            )
            _add_simple_line!(sys, ref, bs, 1e-2, 1e-2, 0.0)
            add_component!(
                sys,
                TapTransformer(; name = "t$k", available = true,
                    active_power_flow = 0.0, reactive_power_flow = 0.0,
                    arc = Arc(; from = ref, to = bl), r = 0.01, x = 0.10,
                    primary_shunt = 0.0 + 0.0im, tap = 1.0, rating = 1.0,
                    base_power = 100.0,
                    control_objective = PSY.TransformerControlObjective.VOLTAGE),
            )
            add_component!(
                sys,
                SwitchedAdmittance(; name = "sh$k", available = true,
                    bus = bs, Y = 0.0 + 0.0im, initial_status = [0], number_of_steps = [4],
                    Y_increase = [0.0 + 0.05im], admittance_limits = (min = 0.9, max = 1.1),
                ),
            )
        end
        return sys
    end
    function _inner(K)
        pf = ACPolarPowerFlow(; control_discrete_devices = true)
        data = PowerFlowData(pf, _feeders(K))
        solve_power_flow!(data)
        @test all(data.converged)
        return PowerFlows.get_control_inner_solve_count(data)
    end
    n1, n8 = _inner(1), _inner(8)   # 2 vs 16 devices
    # Batched: n8 ≈ n1 (one solve/pass, flat). Sequential would give n8 ≳ 8·n1. Allow generous
    # headroom for the extra per-pass work while still failing loudly if batching regresses.
    @test n8 < 2 * n1 + 20
end

@testset "linear plant sign matches FD probe at PV controlled bus" begin
    sys = _make_tap_shunt_system()
    pf = ACPolarPowerFlow{NewtonRaphsonACPowerFlow}(; control_discrete_devices = true)
    data = PowerFlowData(pf, sys)
    ts = 1
    # Converge the base state, then build the sensitivity context the probes use.
    @test PowerFlows._solve_with_q_limits!(pf, data, ts)
    ctx = PowerFlows._sensitivity_context(pf, data, ts)
    @test !isnothing(ctx)
    set = PowerFlows.get_controlled_devices(data)
    for d in set.shunts
        cbus = PowerFlows.controlled_bus_ix(d)
        # Force the controlled bus PV (the must_be_PV production case).
        original_bt = data.bus_type[cbus, ts]
        data.bus_type[cbus, ts] = PSY.ACBusTypes.PV
        s_lin, ok = PowerFlows._linear_plant_sign(d, data, ts, ctx)
        data.bus_type[cbus, ts] = original_bt
        @test ok
        # Voltage at a PV bus is pinned: the true sensitivity is exactly zero.
        @test iszero(s_lin)
    end
end

@testset "_restore_one! early-return restores state on a failed solve" begin
    sys = _make_tap_shunt_system()
    pf = ACPolarPowerFlow{NewtonRaphsonACPowerFlow}(; control_discrete_devices = true)
    data = PowerFlowData(pf, sys)
    ts = 1
    @test PowerFlows._solve_with_q_limits!(pf, data, ts)
    set = PowerFlows.get_controlled_devices(data)
    d = first(set.taps)
    pre_v = copy(data.bus_magnitude[:, ts])
    pre_q = copy(data.bus_reactive_power_injections[:, ts])
    # `_run_power_flow_method`'s loop guard is `i < maxIterations` with i starting at 1,
    # so maxIterations=1 (or 0) takes ZERO real steps — maxIterations=2 is the smallest
    # budget that runs exactly one real NR step (mutating bus_magnitude away from the
    # poisoned start) while still failing to converge from such a bad warm start.
    data.bus_magnitude[:, ts] .= 0.05
    snapshot_v = copy(data.bus_magnitude[:, ts])
    # The forced non-convergence emits an @error at finalization; capture it with @test_logs so
    # it does not trip run_tests()'s zero-Logging.Error-events assertion (full suite).
    scratch_snap = PowerFlows._snapshot_state(data, ts)
    ok =
        @test_logs (:error, r"failed to converge") match_mode = :any PowerFlows._restore_one!(
            d, data, ts, PowerFlows.current_parameter(d), pf, scratch_snap;
            maxIterations = 2)
    @test !ok
    # On failure the pre-call state must be untouched (no diverged iterate left).
    @test data.bus_magnitude[:, ts] == snapshot_v
    # Restore sanity for later testsets.
    data.bus_magnitude[:, ts] .= pre_v
    data.bus_reactive_power_injections[:, ts] .= pre_q
end

@testset "snap holds never-moved off-grid devices" begin
    sys = _make_tap_shunt_system()
    pf = ACPolarPowerFlow{NewtonRaphsonACPowerFlow}(; control_discrete_devices = true)
    data = PowerFlowData(pf, sys)
    ts = 1
    @test PowerFlows._solve_with_q_limits!(pf, data, ts)
    set = PowerFlows.get_controlled_devices(data)
    d = first(set.shunts)
    # Simulate a deadband-held device whose baseline is OFF the prefix chain: current
    # equals initial (never moved) and sits between two chain points.
    off_chain = d.initial + 0.4 * minimum(abs, d.block_dB)
    d.initial = off_chain
    PowerFlows.apply_parameter!(d, data, off_chain, ts)
    PowerFlows._snap_device_group!([d], data, ts)
    @test PowerFlows.current_parameter(d) == off_chain   # held, not snapped
end

@testset "probe restores VSC DC-network state" begin
    # A VSC system passes the construction guard (only LCC is rejected), so it can enter the
    # continuation with a controlled tap. `_read_vsc_state!` writes the solver iterate into
    # dcn.p_c/q_c/node_vdc on every residual evaluation; a probe that fails to restore them
    # would leak a diverged DC state into the next warm start.
    #
    # The "to" converter must NOT pin P via DC_POWER for this to be observable: with a fixed P
    # setpoint, p_c/node_vdc are pure DC-side algebra (P_from + P_to = losses(Vdc), Vdc[slack]
    # fixed) and are exactly invariant to any AC-only perturbation regardless of restore
    # correctness. Giving the "to" converter AC-voltage control (ControlPVac: P pinned, Q free)
    # makes q_c a genuine AC-network-coupled unknown — moving under a nearby tap probe and
    # needing a real restore — while p_c/node_vdc stay checked too (they should never move).
    sys = _build_vsc_pq_system(;
        ac_control_to = PSY.VSCACControlModes.AC_VOLTAGE,
        ac_setpoint_to = 1.0,
    )
    pq = sort!(
        collect(
            PSY.get_components(
                b -> PSY.get_bustype(b) == PSY.ACBusTypes.PQ,
                PSY.ACBus,
                sys,
            ),
        );
        by = PSY.get_number,
    )
    _add_control_tap!(sys, pq[2], pq[3])   # incident to the AC-voltage-controlled VSC bus
    pf = ACPolarPowerFlow{NewtonRaphsonACPowerFlow}(;
        control_discrete_devices = true,
        solver_settings = VSC_SETTINGS,
    )
    data = PowerFlowData(pf, sys)
    ts = 1
    @test PowerFlows._solve_with_q_limits!(pf, data, ts)
    dcn = PowerFlows.get_dc_network(data)
    @test PowerFlows.has_dc_network(dcn)
    pre_p = copy(dcn.p_c[:, ts])
    pre_q = copy(dcn.q_c[:, ts])
    pre_v = copy(dcn.node_vdc[:, ts])
    set = PowerFlows.get_controlled_devices(data)
    @test !isempty(set.taps)
    d = first(set.taps)
    scratch_snap = PowerFlows._snapshot_state(data, ts)
    PowerFlows._plant_sign(d, data, ts, pf, scratch_snap)   # probe solves, then must restore
    @test dcn.p_c[:, ts] == pre_p
    @test dcn.node_vdc[:, ts] == pre_v
    @test dcn.q_c[:, ts] == pre_q
end

@testset "continuation reuses symbolic factorization and bounded inner solves" begin
    # Task 13(b): the sensitivity context (residual + Jacobian + numeric refactor) must be
    # built ONCE per continuation and REFRESHED (values only) thereafter, not rebuilt from
    # scratch on every converged batched pass. `get_control_symbolic_factor_count` is the
    # regression metric: a full `_sensitivity_context` build counts, a values-only
    # `_refresh_sensitivity_context!` does not.
    sys = _make_tap_shunt_system()
    pf = ACPolarPowerFlow{NewtonRaphsonACPowerFlow}(; control_discrete_devices = true)
    data = PowerFlowData(pf, sys)
    @test PowerFlows.solve_power_flow!(data)
    @test PowerFlows.get_control_symbolic_factor_count(data) <= 2
    n_inner = PowerFlows.get_control_inner_solve_count(data)
    @test 0 < n_inner < 200
end

@testset "STATCOM |V|-dependent limit: Q_solved ≈ V·Imax at saturation" begin
    # Deep-sag case (+60% load, bus-9 shunt OOS — the calibrated 14-bus stress anchor) with an
    # undersized current limit: this system's PV buses give bus 14 enough native support that
    # a ~6.5 Mvar-capable device (shmx_mva=15) already reaches vset unsaturated (verified by a
    # direct sweep of shmx_mva at this stress point) — shmx_mva=5 is comfortably below that
    # natural need, so the STATCOM saturates at its |V|-dependent current bound before reaching
    # vset. `_facts_b_limit`'s STATCOM branch is current-limited: |Q| <= V·Imax ⇒ b <= Imax/V,
    # so at saturation Q = b_lim·V² = (Imax/V)·V² = V·Imax exactly (mva_cap=9999, non-binding).
    sys = build_ieee14_facts_system(;
        shmx_mva = 5.0, mva_cap = 9999.0, vset = 1.0, stress = 1.6, shunt9_off = true)
    pf = ACPowerFlow(; control_discrete_devices = true)
    data = PowerFlows.PowerFlowData(pf, sys)
    @test PowerFlows.solve_power_flow!(data)
    @test all(data.converged)
    res = PowerFlows.get_controlled_device_results(data)
    row = only(eachrow(res[res.family .== "FACTSControlDevice", :]))
    bl = PowerFlows.get_bus_lookup(data)
    v14 = data.bus_magnitude[bl[14], 1]
    base_mva = PSY.get_base_power(sys)
    q_mvar = row.final * v14^2 * base_mva
    f = only(data.controlled_devices.facts)
    @test isapprox(row.final, f.b_lim; atol = 1e-3)   # riding the |V|-dependent limit
    @test data.bus_magnitude[f.controlled_ix, 1] < f.vset - 1e-3   # setpoint unreached
    @test isapprox(q_mvar, v14 * 5.0; atol = 0.5)   # Q ≈ V·Imax at the current limit
end

@testset "STATCOM riding its limit converges without oscillation chatter" begin
    # Regression for the chatter-freeze guard: refreshing `b_lim` from the measured voltage
    # every pass must not itself induce direction reversals in the damped target (a "b_lim
    # chases V, V chases b_lim" feedback loop) — the device should settle cleanly against its
    # limit, never trip `CONTROL_OSCILLATION_LIMIT`, and stay well under the per-stage pass
    # budget (`MAX_CONTROL_PASSES_PER_STAGE` per steepness stage). Same saturating fixture as
    # the test above.
    sys = build_ieee14_facts_system(;
        shmx_mva = 5.0, mva_cap = 9999.0, vset = 1.0, stress = 1.6, shunt9_off = true)
    pf = ACPowerFlow(; control_discrete_devices = true)
    data = PowerFlows.PowerFlowData(pf, sys)
    tl = Test.TestLogger(; min_level = Logging.Warn)
    converged = Logging.with_logger(tl) do
        PowerFlows.solve_power_flow!(data)
    end
    @test converged
    @test all(data.converged)
    @test !any(occursin("oscillat", r.message) for r in tl.logs)
    n_inner = PowerFlows.get_control_inner_solve_count(data)
    @test 0 < n_inner < 300
end
