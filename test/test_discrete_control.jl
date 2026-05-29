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
    @test length(s.block_order) == length(s.block_dB)
    @test s.b0 == 0.0
    @test s.b_min == 0.0
    @test s.b_max == 0.2
    @test s.current == 0.0
    @test s.vset == (0.9 + 1.1) / 2
end

@testset "discrete control: shunt invariant validation" begin
    # Test that b0 outside [b_min, b_max] triggers error.
    @test_throws ErrorException PowerFlows._validate_shunt!(
        "bad_shunt",
        0.0, # b_min
        0.5, # b0 — above b_max, outside [b_min, b_max]
        0.2, # b_max
        [4],
        [0.05],
    )
    # Test that a zero-step block with nonzero dB triggers error.
    @test_throws ErrorException PowerFlows._validate_shunt!(
        "bad_shunt2",
        0.0,  # b_min
        0.0,  # b0
        0.2,  # b_max
        [0],  # zero steps
        [0.05], # nonzero dB — malformed
    )
    # Test that b_min == b_max (no controllable range) triggers error.
    @test_throws ErrorException PowerFlows._validate_shunt!(
        "no_range",
        0.5,
        0.5,
        0.5,
        [0],
        [0.0],
    )
    # Valid case — no error.
    @test PowerFlows._validate_shunt!(
        "ok_shunt",
        0.0,
        0.0,
        0.2,
        [4],
        [0.05],
    ) === nothing
end

@testset "discrete control: tap invariant validation" begin
    # p_min > p_max (malformed) triggers error.
    @test_throws ErrorException PowerFlows._validate_tap!("bad_tap", 1.1, 0.9, 33)
    # p_min == p_max (no controllable range) triggers error.
    @test_throws ErrorException PowerFlows._validate_tap!("no_range", 1.0, 1.0, 33)
    # ntp < 2 (no controllable positions) triggers error.
    @test_throws ErrorException PowerFlows._validate_tap!("one_pos", 0.9, 1.1, 1)
    # Valid case — no error.
    @test PowerFlows._validate_tap!("ok_tap", 0.9, 1.1, 33) === nothing
end

@testset "discrete control: sigmoid target" begin
    # ControlledTap, controlled-on-primary (eq.46):
    # tr = (tr_max-tr_min)/(1+exp(S*(|V|-Vset))) + tr_min
    d = PowerFlows.ControlledTap("t", 1, 2, 2, true, 1.0, 1.0 / (0.01 + 0.1im),
        0.0 + 0.0im, 0.0, 0.9, 1.1, collect(range(0.9, 1.1; length = 33)),
        (1, 2, 3, 4), 1.0)
    S = 100.0
    # At |V| = Vset, sigmoid midpoint:
    @test PowerFlows.target_from_voltage(d, 1.0, S) ≈ (1.1 - 0.9) / 2 + 0.9 atol = 1e-9
    # |V| ≫ Vset → saturates near tr_min (controlled-on-primary, eq.46)
    @test PowerFlows.target_from_voltage(d, 1.5, S) ≈ 0.9 atol = 1e-6
    # |V| ≪ Vset → saturates near tr_max
    @test PowerFlows.target_from_voltage(d, 0.5, S) ≈ 1.1 atol = 1e-6
    # secondary-controlled flips sign (eq.47)
    d2 = PowerFlows.ControlledTap("t2", 1, 2, 1, false, 1.0,
        1.0 / (0.01 + 0.1im), 0.0 + 0.0im, 0.0, 0.9, 1.1,
        collect(range(0.9, 1.1; length = 33)), (1, 2, 3, 4), 1.0)
    @test PowerFlows.target_from_voltage(d2, 1.5, S) ≈ 1.1 atol = 1e-6

    block_dB_sh = [0.05]
    sh = PowerFlows.ControlledSwitchedShunt("s", 3, 3, 1.0, 0.0, 0.0,
        [4], block_dB_sh, 0.0, 0.2,
        [1], zeros(Int, length(block_dB_sh)), 0.0)
    @test PowerFlows.target_from_voltage(sh, 1.0, S) ≈ 0.1 atol = 1e-9
    @test PowerFlows.target_from_voltage(sh, 0.5, S) ≈ 0.2 atol = 1e-6  # low V → max B
end

@testset "discrete control: negative-feedback orientation" begin
    # The effective control law `_control_target` must produce negative feedback:
    # sign(d target / dV) opposite to sign(dV/dp), so the closed-loop gain
    # g' = σ'(V)·dV/dp ≤ 0 for ANY device wiring (primary tap, secondary tap,
    # shunt) and for both signs of the measured plant sensitivity dV/dp.
    S = 100.0
    δ = 0.02
    primary = PowerFlows.ControlledTap("tp", 1, 2, 2, true, 1.0,
        1.0 / (0.01 + 0.1im), 0.0 + 0.0im, 0.0, 0.9, 1.1,
        collect(range(0.9, 1.1; length = 33)), (1, 2, 3, 4), 1.0)
    secondary = PowerFlows.ControlledTap("ts", 1, 2, 1, false, 1.0,
        1.0 / (0.01 + 0.1im), 0.0 + 0.0im, 0.0, 0.9, 1.1,
        collect(range(0.9, 1.1; length = 33)), (1, 2, 3, 4), 1.0)
    shunt = PowerFlows.ControlledSwitchedShunt("sh", 3, 3, 1.0, 0.0, 0.0,
        [4], [0.05], 0.0, 0.2, [1], zeros(Int, 1), 0.0)
    for d in (primary, secondary, shunt)
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
    d = PowerFlows.ControlledTap("t", 1, 2, 2, true, 1.0, 1.0 / (0.01 + 0.1im),
        0.0 + 0.0im, 0.0, 0.9, 1.1, collect(range(0.9, 1.1; length = 33)),
        (1, 2, 3, 4), 1.0)
    lo, hi = PowerFlows.parameter_limits(d)
    for S in (1.0e2, 1.0e3, 5.0e3), dVdp in (-5.0, -1.0, -0.1, 0.1, 1.0, 5.0)
        ω = PowerFlows._relaxation(d, S, dVdp)
        gbound = 0.25 * abs(hi - lo) * S * abs(dVdp)
        @test 0.0 < ω <= PowerFlows.CONTROL_RELAXATION_MAX
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

@testset "discrete control: snap" begin
    d = PowerFlows.ControlledTap("t", 1, 2, 2, true, 1.0, 1.0 + 0im,
        0im, 0.0, 0.9, 1.1, collect(range(0.9, 1.1; length = 5)),
        (1, 2, 3, 4), 1.0)  # levels: 0.9,0.95,1.0,1.05,1.1
    @test PowerFlows.snap_to_discrete(d, 1.03) == 1.05
    @test PowerFlows.snap_to_discrete(d, 1.20) == 1.1   # clamp
    block_dB_sh_snap = [0.05]
    sh = PowerFlows.ControlledSwitchedShunt("s", 3, 3, 1.0, 0.0, 0.0,
        [4], block_dB_sh_snap, 0.0, 0.2,
        [1], zeros(Int, length(block_dB_sh_snap)),
        0.0)  # reachable: 0,0.05,0.10,0.15,0.20
    @test PowerFlows.snap_to_discrete(sh, 0.12) == 0.10
    block_dB_sh2 = [0.1, 0.02]
    sh2 = PowerFlows.ControlledSwitchedShunt("s2", 3, 3, 1.0, 0.0, 0.0,
        [2, 3], block_dB_sh2, 0.0, 0.26,
        [1, 2], zeros(Int, length(block_dB_sh2)),
        0.0)  # block-greedy with ±1 refinement
    # Floor-greedy: block 0.1 first → n=floor(0.17/0.1)=1 → 0.10;
    # residual 0.07 → n=floor(0.07/0.02)=3 → 0.06; total=0.16.
    # ±1 refinement: no block improves on abs(0.16-0.17)=0.01 → stays 0.16.
    @test PowerFlows.snap_to_discrete(sh2, 0.17) == 0.16
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
    @test data_off.controlled_devices === nothing

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
    @test a.controlled_devices === nothing
    @test b.controlled_devices === nothing
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
    # Exercises the controlled_on_primary=true path (eq.46), which uses
    # (lo=p_min, hi=p_max) in the sigmoid — the DECREASING orientation.
    # The negative-feedback condition then requires dVdp > 0 (so flip=true
    # reverses the law to increasing, making the closed-loop gain negative).
    # This is the orientation the existing secondary-controlled tests do NOT cover.
    sys = _make_primary_controlled_tap_system()
    pf = ACPolarPowerFlow(; control_discrete_devices = true)
    data = PowerFlowData(pf, sys)
    solve_power_flow!(data)
    @test all(data.converged)
    t = data.controlled_devices.taps[1]
    @test t.controlled_on_primary
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
    @test data1.controlled_devices === nothing
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
