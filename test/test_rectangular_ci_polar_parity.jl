# Solve representative scenarios with both polar NR and rectangular CI; assert
# Vm/θ parity within RECT_PARITY_ATOL.
#
# `RECT_PARITY_ATOL`, `_rect_parity_settings`, `_rect_polar_parity`,
# `_rect_polar_parity_data` and `_build_zip_2bus_system` live in
# `test_utils/cross_file_fixtures.jl` (also used by test_mixed_cpb_*.jl).

@testset "Rectangular CI polar parity: ZIP loads (constant current)" begin
    sys_p = _build_zip_2bus_system(; current_pq = (2.0, 1.0))
    sys_r = _build_zip_2bus_system(; current_pq = (2.0, 1.0))
    _rect_polar_parity(
        sys_p,
        sys_r;
        pf_kwargs = (; correct_bustypes = true),
    )
end

@testset "Rectangular CI polar parity: ZIP-I load at REF bus" begin
    sys_p = _build_zip_2bus_system(; current_pq = (2.0, 1.0), zip_on_ref = true)
    sys_r = _build_zip_2bus_system(; current_pq = (2.0, 1.0), zip_on_ref = true)
    pf_p = ACPowerFlow{NewtonRaphsonACPowerFlow}(; correct_bustypes = true)
    pf_r = ACRectangularPowerFlow{NewtonRaphsonACPowerFlow}(;
        correct_bustypes = true,
        solution_parameters = _rect_parity_settings(),
    )
    res_p = solve_power_flow(pf_p, sys_p)
    res_r = solve_power_flow(pf_r, sys_r)
    @test res_p !== missing
    @test res_r !== missing
    bus_p = res_p["bus_results"]
    bus_r = res_r["bus_results"]
    @test maximum(abs.(bus_p.Vm - bus_r.Vm)) < RECT_PARITY_ATOL
    @test maximum(abs.(bus_p.θ - bus_r.θ)) < RECT_PARITY_ATOL
    # ZIP-I at REF: reported generator P/Q must include the constant-current
    # draw, otherwise the slack accounting is off by `const_I * |V_set|`.
    @test maximum(abs.(bus_p.P_gen - bus_r.P_gen)) < RECT_PARITY_ATOL
    @test maximum(abs.(bus_p.Q_gen - bus_r.Q_gen)) < RECT_PARITY_ATOL
end

@testset "Rectangular CI polar parity: ZIP loads (constant impedance)" begin
    sys_p = _build_zip_2bus_system(; impedance_pq = (2.0, 1.0))
    sys_r = _build_zip_2bus_system(; impedance_pq = (2.0, 1.0))
    _rect_polar_parity(
        sys_p,
        sys_r;
        pf_kwargs = (; correct_bustypes = true),
    )
end

@testset "Rectangular CI polar parity: ZIP loads (full P+I+Z combination)" begin
    sys_p = _build_zip_2bus_system(;
        power_pq = (0.5, 0.2),
        current_pq = (2.0, 1.0),
        impedance_pq = (1.5, 0.8),
    )
    sys_r = _build_zip_2bus_system(;
        power_pq = (0.5, 0.2),
        current_pq = (2.0, 1.0),
        impedance_pq = (1.5, 0.8),
    )
    _rect_polar_parity(
        sys_p,
        sys_r;
        pf_kwargs = (; correct_bustypes = true),
    )
end

@testset "Rectangular CI polar parity: headroom-proportional distributed slack" begin
    sys_p = PSB.build_system(PSB.PSITestSystems, "c_sys14"; add_forecasts = false)
    sys_r = deepcopy(sys_p)
    # With distribute_slack_proportional_to_headroom, PV/REF generators share
    # the slack according to (Pmax - Pset). Routed through _rect_polar_parity
    # so P_gen / Q_gen parity is asserted alongside Vm/θ — catches slack-recovery
    # bugs where the right voltages can mask a wrong attribution of slack.
    _rect_polar_parity(
        sys_p,
        sys_r;
        pf_kwargs = (; distribute_slack_proportional_to_headroom = true),
    )
end

@testset "Rectangular CI polar parity: explicit generator participation factors" begin
    sys_p = PSB.build_system(PSB.PSITestSystems, "c_sys14"; add_forecasts = false)
    sys_r = deepcopy(sys_p)
    spf = Dict{Tuple{DataType, String}, Float64}(
        (ThermalStandard, get_name(g)) => 1.0
        for g in get_components(ThermalStandard, sys_p)
    )
    _rect_polar_parity(
        sys_p,
        sys_r;
        pf_kwargs = (; generator_slack_participation_factors = spf),
    )
end

@testset "Rectangular CI polar parity: ACTIVSg2000 (large-scale)" begin
    sys_p = build_system(MatpowerTestSystems, "matpower_ACTIVSg2000_sys")
    sys_r = build_system(MatpowerTestSystems, "matpower_ACTIVSg2000_sys")
    _rect_polar_parity(
        sys_p,
        sys_r;
        pf_kwargs = (; correct_bustypes = true),
    )
end

@testset "Rectangular CI polar parity: Q-limit enforcement (PV → PQ switching)" begin
    sys_p = PSB.build_system(PSB.PSITestSystems, "c_sys14"; add_forecasts = false)
    sys_r = deepcopy(sys_p)
    _rect_polar_parity(
        sys_p,
        sys_r;
        pf_kwargs = (; check_reactive_power_limits = true),
    )
end

@testset "Rectangular CI polar parity: Q-limit enforcement (c_sys5)" begin
    sys_p = PSB.build_system(PSB.PSITestSystems, "c_sys5")
    sys_r = deepcopy(sys_p)
    _rect_polar_parity(
        sys_p,
        sys_r;
        pf_kwargs = (; check_reactive_power_limits = true),
    )
end

@testset "Rectangular CI polar parity: radial network reduction" begin
    sys_p = PSB.build_system(
        PSB.PSSEParsingTestSystems, "psse_14_network_reduction_test_system")
    sys_r = deepcopy(sys_p)
    _rect_polar_parity(
        sys_p,
        sys_r;
        pf_kwargs = (;
            network_reductions = PNM.NetworkReduction[PNM.RadialReduction()]),
    )
end

@testset "Rectangular CI polar parity: generator reactive redistribution" begin
    sys_p = PSB.build_system(PSB.PSITestSystems, "c_sys5")
    sys_r = deepcopy(sys_p)
    @test PF.solve_and_store_power_flow!(
        ACPowerFlow{NewtonRaphsonACPowerFlow}(), sys_p)
    @test PF.solve_and_store_power_flow!(
        ACRectangularPowerFlow{NewtonRaphsonACPowerFlow}(;
            solution_parameters = _rect_parity_settings()), sys_r)
    for g_p in get_components(Generator, sys_p)
        g_r = get_component(typeof(g_p), sys_r, get_name(g_p))
        @test isapprox(
            get_reactive_power(g_p, PSY.SU), get_reactive_power(g_r, PSY.SU);
            atol = RECT_PARITY_ATOL)
    end
end

@testset "Rectangular CI polar parity: multi-period (same network, no time-varying loads)" begin
    sys_p = PSB.build_system(PSB.PSITestSystems, "c_sys14"; add_forecasts = false)
    sys_r = deepcopy(sys_p)
    pf_p = ACPowerFlow{NewtonRaphsonACPowerFlow}(; time_steps = 3)
    pf_r = ACRectangularPowerFlow{NewtonRaphsonACPowerFlow}(;
        time_steps = 3,
        solution_parameters = _rect_parity_settings())
    _rect_polar_parity_data(pf_p, pf_r, sys_p, sys_r)
end

@testset "Rectangular CI polar parity: multi-period time-varying distributed slack" begin
    sys_p = PSB.build_system(PSB.PSITestSystems, "c_sys14"; add_forecasts = false)
    sys_r = deepcopy(sys_p)
    gens = collect(get_components(ThermalStandard, sys_p))
    # One participation dict per time step makes the slack split time-varying.
    spf = [
        Dict{Tuple{DataType, String}, Float64}(
            (ThermalStandard, get_name(g)) => (g === gens[k] ? 2.0 : 1.0)
            for g in gens)
        for k in (1, length(gens))
    ]
    pf_p = ACPowerFlow{NewtonRaphsonACPowerFlow}(;
        time_steps = 2, generator_slack_participation_factors = spf)
    pf_r = ACRectangularPowerFlow{NewtonRaphsonACPowerFlow}(;
        time_steps = 2, generator_slack_participation_factors = spf,
        solution_parameters = _rect_parity_settings())
    _rect_polar_parity_data(pf_p, pf_r, sys_p, sys_r)
end

# `rect_finalize_bus_injections!` used to write P_gen/Q_gen back with the constant-power
# withdrawal ONLY, dropping the ZIP constant-current term. Since `ACRectangularCIResidual`'s
# constructor rebuilds its `P_net_set` FROM the previously-written `bus_active_power_injections`
# (mirroring polar), the dropped term compounded every re-solve on the same `data` — a PV bus
# carrying a 0.1 pu constant-current load drifted 0.398 → 0.296 → 0.194 over three solves.
function _pv_zip_drift_system()
    sys = System(100.0)
    b1 = _add_simple_bus!(sys, 1, ACBusTypes.REF, 230, 1.0, 0.0)
    b2 = _add_simple_bus!(sys, 2, ACBusTypes.PV, 230, 1.0, 0.0)
    _add_simple_source!(sys, b1, 0.0, 0.0)
    _add_simple_thermal_standard!(sys, b2, 0.5, 0.0)
    # 1.0 pu on the load's own 10 MVA device base = 0.1 pu at the 100 MVA system base.
    _add_simple_zip_load!(sys, b2; constant_current_active_power = 1.0)
    _add_simple_line!(sys, b1, b2, 5e-3, 5e-2, 1e-3)
    return sys
end

@testset "Rectangular CI: repeated solves on the same data do not drift ZIP-PV injections" begin
    sys_r = _pv_zip_drift_system()
    sys_p = deepcopy(sys_r)
    pf_r = ACRectangularPowerFlow{NewtonRaphsonACPowerFlow}(;
        correct_bustypes = true, solution_parameters = _rect_parity_settings())
    pf_p = ACPowerFlow{NewtonRaphsonACPowerFlow}(; correct_bustypes = true)

    data_p = PowerFlowData(pf_p, sys_p)
    @test PowerFlows.solve_power_flow!(data_p)
    bus_ix_p = PF.get_bus_lookup(data_p)[2]
    p_gen_polar = data_p.bus_active_power_injections[bus_ix_p, 1]
    q_gen_polar = data_p.bus_reactive_power_injections[bus_ix_p, 1]

    data_r = PowerFlowData(pf_r, sys_r)
    bus_ix_r = PF.get_bus_lookup(data_r)[2]
    p_gen_solves = Float64[]
    q_gen_solves = Float64[]
    for _ in 1:3
        @test PowerFlows.solve_power_flow!(data_r)
        push!(p_gen_solves, data_r.bus_active_power_injections[bus_ix_r, 1])
        push!(q_gen_solves, data_r.bus_reactive_power_injections[bus_ix_r, 1])
    end
    @test all(isapprox.(p_gen_solves, p_gen_solves[1]; atol = 1e-9))
    @test all(isapprox.(q_gen_solves, q_gen_solves[1]; atol = 1e-9))
    @test p_gen_solves[end] ≈ p_gen_polar atol = RECT_PARITY_ATOL
    @test q_gen_solves[end] ≈ q_gen_polar atol = RECT_PARITY_ATOL
end
