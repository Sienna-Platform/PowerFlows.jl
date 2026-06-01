# Solve representative scenarios with both polar NR and rectangular CI; assert
# Vm/θ parity within RECT_PARITY_ATOL.

const RECT_PARITY_ATOL = 1e-7

_rect_parity_settings() = Dict{Symbol, Any}(:validate_voltage_magnitudes => false)

function _rect_polar_parity(
    sys_p::PSY.System,
    sys_r::PSY.System;
    pf_kwargs::NamedTuple = NamedTuple(),
    pf_r_extra_settings::Dict{Symbol, Any} = Dict{Symbol, Any}(),
    atol::Float64 = RECT_PARITY_ATOL,
)
    pf_p = ACPowerFlow{NewtonRaphsonACPowerFlow}(; pf_kwargs...)
    pf_r = ACRectangularPowerFlow{NewtonRaphsonACPowerFlow}(;
        pf_kwargs...,
        solver_settings = merge(_rect_parity_settings(), pf_r_extra_settings),
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
        solver_settings = _rect_parity_settings(),
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
            solver_settings = _rect_parity_settings()), sys_r)
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
        solver_settings = _rect_parity_settings())
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
        solver_settings = _rect_parity_settings())
    _rect_polar_parity_data(pf_p, pf_r, sys_p, sys_r)
end
