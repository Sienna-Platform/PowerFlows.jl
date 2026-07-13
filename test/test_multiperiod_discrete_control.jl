@testset "Multiperiod switched-shunt voltage regulation" begin
    sys = _make_multiperiod_shunt_system()
    time_steps = 3
    pf = ACPowerFlow{NewtonRaphsonACPowerFlow}(;
        control_discrete_devices = true, time_steps = time_steps)
    data = PowerFlowData(pf, sys)
    _set_multiperiod_shunt_loads!(data, time_steps)
    @test solve_power_flow!(data)
    @test all(data.converged)
    reg_ix = _regulated_bus_index(data)
    for t in 1:time_steps
        @test isapprox(data.bus_magnitude[reg_ix, t], 1.0; atol = 1e-3)
    end
    results = PowerFlows.get_controlled_device_results(data)
    @test nrow(results) >= time_steps
end

@testset "Multiperiod shunt with nonzero constant-Z baseline (b0) — per-step parity" begin
    # Regression: `initialize_power_flow_data!` seeded the shunt's constant-Z baseline only into
    # column 1, so `ts≥2` solved without it. Oracle: each column must match an independent single-ts
    # solve at that step's load (device setting is the sharpest observable).
    time_steps = 3
    sys = _make_multiperiod_shunt_system_with_baseline()
    pf = ACPowerFlow{NewtonRaphsonACPowerFlow}(;
        control_discrete_devices = true, time_steps = time_steps)
    data = PowerFlowData(pf, sys)
    _set_multiperiod_shunt_loads!(data, time_steps)
    @test solve_power_flow!(data)
    @test all(data.converged)
    results = PowerFlows.get_controlled_device_results(data)
    for t in 1:time_steps
        ref = _solve_shunt_single_ts_at_step(t)
        @test all(ref.converged)
        @test isapprox(data.bus_magnitude[:, t], ref.bus_magnitude[:, 1]; atol = 1e-6)
        @test isapprox(data.bus_angles[:, t], ref.bus_angles[:, 1]; atol = 1e-6)
        ref_final = PowerFlows.get_controlled_device_results(ref).final
        multi_final = results[results.time_step .== t, :].final
        @test isapprox(multi_final, ref_final; atol = 1e-6)
    end
end

@testset "Multiperiod reactive-power limits enforced at every step (not just ts=1)" begin
    # Regression: `bus_reactive_power_bounds` was seeded only into column 1 (otherwise `(-Inf,Inf)`),
    # so `_check_q_limit_bounds!` never fired the PV→PQ switch for `ts≥2`. c_sys14 + correct_bustypes
    # makes Bus8's PV gen violate its Q limit; every identical step must reproduce the switch.
    time_steps = 3
    sys = PSB.build_system(PSB.PSITestSystems, "c_sys14"; add_forecasts = false)
    set_units_base_system!(sys, UnitSystem.SYSTEM_BASE)
    pf = ACPowerFlow{NewtonRaphsonACPowerFlow}(;
        check_reactive_power_limits = true, correct_bustypes = true,
        time_steps = time_steps)
    data = PowerFlowData(pf, sys)
    _replicate_col1_to_all_steps!(data, time_steps)   # identical snapshot per step; bounds untouched
    @test solve_power_flow!(data)
    @test all(data.converged)

    # Independent single-ts reference (its only column has correct bounds).
    sys_ref = PSB.build_system(PSB.PSITestSystems, "c_sys14"; add_forecasts = false)
    set_units_base_system!(sys_ref, UnitSystem.SYSTEM_BASE)
    pf_ref = ACPowerFlow{NewtonRaphsonACPowerFlow}(;
        check_reactive_power_limits = true, correct_bustypes = true)
    data_ref = PowerFlowData(pf_ref, sys_ref)
    ix8 = PF.get_bus_lookup(data_ref)[8]
    @test data_ref.bus_type[ix8, 1] == PSY.ACBusTypes.PV   # flat start: Bus8 is PV
    @test solve_power_flow!(data_ref)
    @test data_ref.bus_type[ix8, 1] == PSY.ACBusTypes.PQ   # limit binds → switched to PQ

    for t in 1:time_steps
        @test data.bus_type[:, t] == data_ref.bus_type[:, 1]
        @test isapprox(data.bus_magnitude[:, t], data_ref.bus_magnitude[:, 1]; atol = 1e-6)
        @test isapprox(data.bus_reactive_power_injections[:, t],
            data_ref.bus_reactive_power_injections[:, 1]; atol = 1e-6)
    end
end

@testset "Multiperiod FACTS (SVC) voltage regulation" begin
    # Guard test: FACTS state lives in the per-ts store, so multiperiod SVC control round-trips
    # through load/save_device_state!. Pins that behavior against regression.
    sys = _make_multiperiod_facts_system()
    time_steps = 3
    pf = ACPowerFlow{NewtonRaphsonACPowerFlow}(;
        control_discrete_devices = true, time_steps = time_steps)
    data = PowerFlowData(pf, sys)
    _set_multiperiod_facts_loads!(data, time_steps)
    @test solve_power_flow!(data)
    @test all(data.converged)
    reg_ix = _regulated_bus_index(data)
    for t in 1:time_steps
        @test data.bus_magnitude[reg_ix, t] <= 1.05
    end
    results = PowerFlows.get_controlled_device_results(data)
    @test nrow(results) >= time_steps
    # SVC susceptance genuinely differs across steps (control moved):
    facts_rows = results[results.family .== "FACTSControlDevice", :]
    @test length(unique(round.(facts_rows.final; digits = 6))) >= 2
end

@testset "write-back is skipped under time_steps>1 (no silent last-ts write)" begin
    sys = _make_multiperiod_shunt_system()
    # capture the shunt's PSY Y before solving
    shunt = only(collect(PSY.get_components(PSY.SwitchedAdmittance, sys)))
    y_before = PSY.get_Y(shunt)
    time_steps = 3
    pf = ACPowerFlow{NewtonRaphsonACPowerFlow}(;
        control_discrete_devices = true, time_steps = time_steps)
    data = PowerFlowData(pf, sys)
    _set_multiperiod_shunt_loads!(data, time_steps)
    @test solve_power_flow!(data)
    # multi-ts: PSY component is NOT mutated (per-ts results are in get_controlled_device_results)
    PowerFlows.write_device_settings!(sys, data)
    @test PSY.get_Y(shunt) == y_before
    @test nrow(PowerFlows.get_controlled_device_results(data)) >= time_steps
end

@testset "write-back still happens for time_steps==1 (no regression)" begin
    sys = _make_multiperiod_shunt_system()
    shunt = only(collect(PSY.get_components(PSY.SwitchedAdmittance, sys)))
    y_before = PSY.get_Y(shunt)
    pf = ACPowerFlow{NewtonRaphsonACPowerFlow}(; control_discrete_devices = true)
    data = PowerFlowData(pf, sys)
    @test solve_power_flow!(data)
    PowerFlows.write_device_settings!(sys, data)
    @test PSY.get_Y(shunt) != y_before
end

@testset "Branch-flow-inside-loop parity (no taps, multi-step)" begin
    sys = PSB.build_system(PSB.PSITestSystems, "c_sys14"; add_forecasts = false)
    time_steps = 3
    pf = ACPowerFlow{NewtonRaphsonACPowerFlow}(; time_steps = time_steps)
    data = PowerFlowData(pf, sys)
    prepare_ts_data!(data, time_steps)
    @test solve_power_flow!(data)
    @test all(data.converged)
    for t in 1:time_steps
        @test all(isfinite, data.arc_active_power_flow_from_to[:, t])
        @test all(isfinite, data.arc_reactive_power_flow_from_to[:, t])
        @test all(isfinite, data.arc_active_power_flow_to_from[:, t])
        @test all(isfinite, data.arc_reactive_power_flow_to_from[:, t])
        @test all(isfinite, data.arc_angle_differences[:, t])
    end

    # HARD parity: recompute each step's arc flows from its converged voltages and the SAME shared
    # arc-admittance matrices, and assert the solver's stored per-step flows match bit-for-bit —
    # proving per-step (in-loop) computation equals the old post-loop batched computation.
    Yft = data.power_network_matrix.arc_admittance_from_to
    Ytf = data.power_network_matrix.arc_admittance_to_from
    arcs = PowerFlows.PNM.get_arc_axis(Yft)
    bus_lookup = PF.get_bus_lookup(data)
    fb_ix = [bus_lookup[bus_no] for bus_no in first.(arcs)]
    tb_ix = [bus_lookup[bus_no] for bus_no in last.(arcs)]
    for t in 1:time_steps
        V = data.bus_magnitude[:, t] .* exp.(1im .* data.bus_angles[:, t])
        Sft = V[fb_ix] .* conj.(Yft.data * V)
        Stf = V[tb_ix] .* conj.(Ytf.data * V)
        @test data.arc_active_power_flow_from_to[:, t] == real.(Sft)
        @test data.arc_reactive_power_flow_from_to[:, t] == imag.(Sft)
        @test data.arc_active_power_flow_to_from[:, t] == real.(Stf)
        @test data.arc_reactive_power_flow_to_from[:, t] == imag.(Stf)
        @test data.arc_angle_differences[:, t] ==
              data.bus_angles[fb_ix, t] .- data.bus_angles[tb_ix, t]
    end
end

@testset "Multiperiod controlled-tap voltage regulation" begin
    time_steps = 3
    sys = _make_multiperiod_tap_system()
    pf = ACPowerFlow{NewtonRaphsonACPowerFlow}(;
        control_discrete_devices = true, time_steps = time_steps)
    data = PowerFlowData(pf, sys)
    _set_multiperiod_tap_loads!(data, time_steps)
    @test solve_power_flow!(data)
    @test all(data.converged)

    # Gold standard: each multi-ts step equals an INDEPENDENT single-ts solve at that load.
    for t in 1:time_steps
        sys_t = _make_multiperiod_tap_system()
        pf1 = ACPowerFlow{NewtonRaphsonACPowerFlow}(;
            control_discrete_devices = true, time_steps = 1)
        data1 = PowerFlowData(pf1, sys_t)
        _set_single_tap_load!(data1, t)
        @test solve_power_flow!(data1)
        @test isapprox(data.bus_magnitude[:, t], data1.bus_magnitude[:, 1]; atol = 1e-6)
        @test isapprox(data.bus_angles[:, t], data1.bus_angles[:, 1]; atol = 1e-6)
    end

    # Tap moved differently across steps (control actually acted per-step):
    res = PowerFlows.get_controlled_device_results(data)
    tap_rows = res[res.family .== "TapTransformer", :]
    @test length(unique(round.(tap_rows.final; digits = 6))) >= 2
end

@testset "multiperiod construction seeds every column from the snapshot" begin
    # `initialize_power_flow_data!` must seed ALL columns from the snapshot; columns left at their
    # zero/(-Inf,Inf) allocation defaults silently produced wrong `ts≥2` solutions.
    sys = _make_multiperiod_shunt_system_with_baseline()
    pf = ACPowerFlow{NewtonRaphsonACPowerFlow}(; time_steps = 3)
    data = PowerFlowData(pf, sys)
    for t in 2:3
        @test data.bus_active_power_injections[:, t] ==
              data.bus_active_power_injections[:, 1]
        @test data.bus_reactive_power_injections[:, t] ==
              data.bus_reactive_power_injections[:, 1]
        @test data.bus_active_power_withdrawals[:, t] ==
              data.bus_active_power_withdrawals[:, 1]
        @test data.bus_reactive_power_withdrawals[:, t] ==
              data.bus_reactive_power_withdrawals[:, 1]
        @test data.bus_reactive_power_bounds[:, t] == data.bus_reactive_power_bounds[:, 1]
        @test data.bus_reactive_power_constant_impedance_withdrawals[:, t] ==
              data.bus_reactive_power_constant_impedance_withdrawals[:, 1]
    end
    # Behavioral: with untouched columns, every step's solution equals the single-ts solve.
    @test solve_power_flow!(data)
    @test all(data.converged)
    data1 = PowerFlowData(ACPowerFlow{NewtonRaphsonACPowerFlow}(), sys)
    @test solve_power_flow!(data1)
    for t in 1:3
        @test isapprox(data.bus_magnitude[:, t], data1.bus_magnitude[:, 1]; atol = 1e-6)
        @test isapprox(data.bus_angles[:, t], data1.bus_angles[:, 1]; atol = 1e-6)
    end
end

@testset "combined discrete control + Q-limit switching, multiperiod parity" begin
    # Both mechanisms active in one multiperiod solve: the shunt regulates the PQ bus while
    # the tight-limited PV generator hits its Q bound and switches PV->PQ at heavier steps.
    time_steps = 3
    sys = _make_multiperiod_qlimit_shunt_system()
    pf = ACPowerFlow{NewtonRaphsonACPowerFlow}(;
        control_discrete_devices = true,
        check_reactive_power_limits = true,
        time_steps = time_steps,
    )
    data = PowerFlowData(pf, sys)
    _set_multiperiod_shunt_loads!(data, time_steps)
    @test solve_power_flow!(data)
    @test all(data.converged)
    # The Q-limit mechanism genuinely engaged: the PV bus switched to PQ at some step.
    pv_ix = PF.get_bus_lookup(data)[2]
    @test any(t -> data.bus_type[pv_ix, t] == PSY.ACBusTypes.PQ, 1:time_steps)
    # Gold-standard parity: each step matches an independent single-ts combined-mode solve.
    results = PowerFlows.get_controlled_device_results(data)
    for t in 1:time_steps
        ref = _solve_qlimit_shunt_single_ts_at_step(t)
        @test all(ref.converged)
        @test data.bus_type[:, t] == ref.bus_type[:, 1]
        @test isapprox(data.bus_magnitude[:, t], ref.bus_magnitude[:, 1]; atol = 1e-6)
        @test isapprox(data.bus_angles[:, t], ref.bus_angles[:, 1]; atol = 1e-6)
        ref_final = PowerFlows.get_controlled_device_results(ref).final
        multi_final = results[results.time_step .== t, :].final
        @test isapprox(multi_final, ref_final; atol = 1e-6)
    end
end
