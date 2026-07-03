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

@testset "discrete control: MODSW gating" begin
    function _build(modsw)
        sys = _make_tap_shunt_system()
        sa = first(PSY.get_components(PSY.SwitchedAdmittance, sys))
        PSY.get_ext(sa)["MODSW"] = modsw
        data = PowerFlowData(ACPolarPowerFlow(), sys)
        return PowerFlows.build_controlled_device_set(
            sys, PF.get_bus_lookup(data), data.power_network_matrix)
    end

    # MODSW=0 (locked): shunt not enrolled; tap unaffected.
    set0 = _build(0)
    @test length(set0.shunts) == 0
    @test length(set0.taps) == 1

    # MODSW=1 (discrete voltage): enrolled, continuous == false.
    set1 = _build(1)
    @test length(set1.shunts) == 1
    @test set1.shunts[1].continuous == false

    # MODSW=2 (continuous voltage): enrolled, continuous == true.
    set2 = _build(2)
    @test length(set2.shunts) == 1
    @test set2.shunts[1].continuous == true

    # MODSW 3–6 (remote reactive-power / device control): unsupported ⇒ the shunt is
    # de-enrolled with a warning and stays locked (PSS/E posture); never an error.
    for m in (3, 4, 5, 6)
        setm = @test_logs (:warn, Regex("MODSW=$m")) match_mode = :any _build(m)
        @test length(setm.shunts) == 0
        @test length(setm.taps) == 1   # the tap is unaffected by the bad shunt record
    end
end

@testset "discrete control: warn-and-lock on unresolvable controlled bus" begin
    # A remote controlled-bus number that does not exist in the network must
    # de-enroll the device with a warning, not abort PowerFlowData construction.
    sys = _make_tap_shunt_system()
    tx = first(PSY.get_components(PSY.TapTransformer, sys))
    PSY.get_ext(tx)["CONT1"] = 99   # no bus 99
    data = PowerFlowData(ACPolarPowerFlow(), sys)
    set = @test_logs (:warn, r"controlled bus 99") match_mode = :any (
        PowerFlows.build_controlled_device_set(
            sys, PF.get_bus_lookup(data), data.power_network_matrix)
    )
    @test length(set.taps) == 0
    @test length(set.shunts) == 1   # the shunt still enrolls
end

@testset "discrete control: tap ext keys (parser spellings)" begin
    # The PSS/E parser writes winding-suffixed keys: CONT1 (controlled bus),
    # RMI1/RMA1 (ratio band), NTP1 (positions). VSET is a user-facing override.
    sys = _make_tap_shunt_system()
    tx = first(PSY.get_components(PSY.TapTransformer, sys))
    ext = PSY.get_ext(tx)
    ext["CONT1"] = 3
    ext["RMI1"] = 0.88
    ext["RMA1"] = 1.12
    ext["NTP1"] = 25
    ext["VSET"] = 1.03
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
    # With the parser's MODSW key present, Y holds the TOTAL in-service admittance
    # (BINIT) and initial_status is zeroed: the reachable range is spanned by the
    # blocks alone (base 0) and the control baseline sits at BINIT.
    sys = _make_tap_shunt_system()
    sa = first(PSY.get_components(PSY.SwitchedAdmittance, sys))
    PSY.get_ext(sa)["MODSW"] = 1
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

@testset "discrete control: shunt controlled bus SWREM/NREG" begin
    function _shunt(ext_pairs...)
        sys = _make_tap_shunt_system()
        sa = first(PSY.get_components(PSY.SwitchedAdmittance, sys))
        for (k, v) in ext_pairs
            PSY.get_ext(sa)[k] = v
        end
        data = PowerFlowData(ACPolarPowerFlow(), sys)
        bl = PF.get_bus_lookup(data)
        set = PowerFlows.build_controlled_device_set(
            sys, bl, data.power_network_matrix)
        return set.shunts[1], bl
    end

    # v32/33: regulated bus comes from SWREM.
    s_swrem, bl = _shunt("SWREM" => 2)
    @test s_swrem.controlled_ix == bl[2]

    # v35: NREG takes precedence over SWREM.
    s_nreg, bl2 = _shunt("NREG" => 2, "SWREM" => 3)
    @test s_nreg.controlled_ix == bl2[2]

    # No remote-bus key ⇒ falls back to the shunt's local bus (3).
    s_local, bl3 = _shunt()
    @test s_local.controlled_ix == bl3[3]
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
        1.0 / (0.01 + 0.1im), 0.0 + 0.0im, 0.0, 0.9, 1.1,
        collect(range(0.9, 1.1; length = 33)), (1, 2, 3, 4), 1.0)
    remote = PowerFlows.ControlledTap("ts", 1, 2, 1, 1.0,
        1.0 / (0.01 + 0.1im), 0.0 + 0.0im, 0.0, 0.9, 1.1,
        collect(range(0.9, 1.1; length = 33)), (1, 2, 3, 4), 1.0)
    shunt = PowerFlows.ControlledSwitchedShunt("sh", 3, 3, 1.0, 0.0, 0.0,
        [4], [0.05], 0.0, 0.2, [1], zeros(Int, 1), false, 0.0)
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
        0.0 + 0.0im, 0.0, 0.9, 1.1, collect(range(0.9, 1.1; length = 33)),
        (1, 2, 3, 4), 1.0)
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
    PSY.get_ext(sa)["MODSW"] = 2
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
        0im, 0.0, 0.9, 1.1, collect(range(0.9, 1.1; length = 5)),
        (1, 2, 3, 4), 1.0)  # levels: 0.9,0.95,1.0,1.05,1.1
    @test PowerFlows.snap_to_discrete(d, 1.03) == 1.05
    @test PowerFlows.snap_to_discrete(d, 1.20) == 1.1   # clamp
    block_dB_sh_snap = [0.05]
    sh = PowerFlows.ControlledSwitchedShunt("s", 3, 3, 1.0, 0.0, 0.0,
        [4], block_dB_sh_snap, 0.0, 0.2,
        [1], zeros(Int, length(block_dB_sh_snap)),
        false, 0.0)  # reachable: 0,0.05,0.10,0.15,0.20
    @test PowerFlows.snap_to_discrete(sh, 0.12) == 0.10
    block_dB_sh2 = [0.1, 0.02]
    sh2 = PowerFlows.ControlledSwitchedShunt("s2", 3, 3, 1.0, 0.0, 0.0,
        [2, 3], block_dB_sh2, 0.0, 0.26,
        [1, 2], zeros(Int, length(block_dB_sh2)),
        false, 0.0)  # block-greedy with ±1 refinement
    # Floor-greedy: block 0.1 first → n=floor(0.17/0.1)=1 → 0.10;
    # residual 0.07 → n=floor(0.07/0.02)=3 → 0.06; total=0.16.
    # ±1 refinement: no block improves on abs(0.16-0.17)=0.01 → stays 0.16.
    @test PowerFlows.snap_to_discrete(sh2, 0.17) == 0.16
end

@testset "discrete control: continuous shunt no snap" begin
    # continuous == true ⇒ snap_to_discrete returns the clamped continuous value,
    # NOT the nearest reachable block grid point.
    block_dB = [0.05]
    cont = PowerFlows.ControlledSwitchedShunt("c", 3, 3, 1.0, 0.0, 0.0,
        [4], block_dB, 0.0, 0.2, [1], zeros(Int, length(block_dB)), true, 0.0)
    # 0.12 is between grid points 0.10 and 0.15; continuous must return it unchanged.
    @test PowerFlows.snap_to_discrete(cont, 0.12) == 0.12
    # clamped at the rails.
    @test PowerFlows.snap_to_discrete(cont, 0.30) == 0.2
    @test PowerFlows.snap_to_discrete(cont, -0.10) == 0.0
    # sanity: the discrete twin DOES snap 0.12 → 0.10.
    disc = PowerFlows.ControlledSwitchedShunt("d", 3, 3, 1.0, 0.0, 0.0,
        [4], block_dB, 0.0, 0.2, [1], zeros(Int, length(block_dB)), false, 0.0)
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
    # The controlled bus is the tap's FROM bus (set via ext["NREG"] = 2).
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

@testset "discrete control: SVC holds weak bus to setpoint (NR)" begin
    # Ample reactive capability (100 MVA ⇒ b_max = 1.0 p.u.): the SVC must inject
    # reactive power to drive the weak PQ bus up to its voltage_setpoint (1.0 p.u.).
    sys = _make_svc_system(; max_shunt_current = 100.0)
    pf = ACPolarPowerFlow(;
        control_discrete_devices = true,
        solver_settings = Dict{Symbol, Any}(:experimental_controls => true),
    )
    data = PowerFlowData(pf, sys)
    solve_power_flow!(data)
    @test all(data.converged)
    f = data.controlled_devices.facts[1]
    @test abs(data.bus_magnitude[f.controlled_ix, 1] - f.vset) <= 5e-3
    # Regulating, not saturated at a susceptance limit.
    @test f.b_min < f.current < f.b_max
end

@testset "discrete control: SVC clamps at reactive limit (NR)" begin
    # Tight reactive capability (5 MVA ⇒ b_max = 0.05 p.u.) cannot supply the
    # bus's reactive deficit: the SVC saturates at b_max and the bus stays below
    # setpoint — the homotopy equivalent of the PV→PQ Q-limit release.
    sys = _make_svc_system(; max_shunt_current = 5.0)
    pf = ACPolarPowerFlow(;
        control_discrete_devices = true,
        solver_settings = Dict{Symbol, Any}(:experimental_controls => true),
    )
    data = PowerFlowData(pf, sys)
    solve_power_flow!(data)
    @test all(data.converged)
    f = data.controlled_devices.facts[1]
    @test data.bus_magnitude[f.controlled_ix, 1] < f.vset - 1e-3
    @test isapprox(f.current, f.b_max; atol = 1e-3)
end

@testset "discrete control: PAR regulates branch flow to target (NR)" begin
    # Two parallel paths (line ∥ PAR) feed a load; the PAR steers its own active-power
    # flow to the setpoint by adjusting its phase angle within the (ample) band.
    sys = _make_par_system(; p_target = 0.3)
    pf = ACPolarPowerFlow(;
        control_discrete_devices = true,
        solver_settings = Dict{Symbol, Any}(:experimental_controls => true),
    )
    data = PowerFlowData(pf, sys)
    solve_power_flow!(data)
    @test all(data.converged)
    p = data.controlled_devices.phase_shifters[1]
    @test abs(PowerFlows.measured_value(p, data, 1) - p.p_target) <= 5e-3
    # Regulating, not pinned at a phase-angle limit.
    @test p.angle_min < p.current < p.angle_max
end

@testset "discrete control: PAR clamps at angle limit (NR)" begin
    # A 0.45 p.u. flow target needs ≈0.04 rad of phase boost, but the band is only ±0.01:
    # the angle saturates at a limit and the regulated flow falls short of the setpoint.
    sys = _make_par_system(; p_target = 0.45, angle_min = -0.01, angle_max = 0.01)
    pf = ACPolarPowerFlow(;
        control_discrete_devices = true,
        solver_settings = Dict{Symbol, Any}(:experimental_controls => true),
    )
    data = PowerFlowData(pf, sys)
    solve_power_flow!(data)
    @test all(data.converged)
    p = data.controlled_devices.phase_shifters[1]
    @test isapprox(p.current, p.angle_max; atol = 1e-3) ||
          isapprox(p.current, p.angle_min; atol = 1e-3)
    @test PowerFlows.measured_value(p, data, 1) < p.p_target - 1e-2
end

@testset "discrete control: tap reads first-class PSY fields (#1684)" begin
    # Controllability set via the new PSY fields (no ext); the builder must read them.
    # Requires the psy6 PSY branch (PSY #1705); on released PSY 5.x the fields do not
    # exist and the builder uses the DEFAULT_TAP_* fallbacks instead.
    if !PowerFlows.PSY_HAS_TAP_CONTROL_FIELDS
        @test_skip "requires a PowerSystems.jl with the PSY #1705 tap-control fields (psy6 branch)"
    else
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
    lcc_sys = simple_lcc_system()
    @test_throws ArgumentError PowerFlowData(
        ACPolarPowerFlow(; control_discrete_devices = true), lcc_sys)
end

@testset "discrete control: FACTS/PAR are experimental-gated" begin
    # Without the flag, FACTS devices are not enrolled (an @info points at the flag);
    # taps/shunts are unaffected.
    sys = _make_svc_system(; max_shunt_current = 100.0)
    pf_off = ACPolarPowerFlow(; control_discrete_devices = true)
    data_off = PowerFlowData(pf_off, sys)
    @test data_off.controlled_devices === nothing   # SVC was the only device
    # With the flag, the SVC enrolls.
    pf_on = ACPolarPowerFlow(;
        control_discrete_devices = true,
        solver_settings = Dict{Symbol, Any}(:experimental_controls => true),
    )
    data_on = PowerFlowData(pf_on, sys)
    @test length(data_on.controlled_devices.facts) == 1
end

@testset "discrete control: PAR with parser-default setpoint is not enrolled" begin
    # active_power_flow == 0.0 is the parser default, not a real flow setpoint:
    # enrolling it would command the PAR to erase its own flow.
    sys = _make_par_system(; p_target = 0.0)
    data = PowerFlowData(ACPolarPowerFlow(), sys)
    set = @test_logs (:warn, r"active_power_flow is 0.0") match_mode = :any (
        PowerFlows.build_controlled_device_set(
            sys, PF.get_bus_lookup(data), data.power_network_matrix;
            include_experimental = true)
    )
    @test length(set.phase_shifters) == 0
end
