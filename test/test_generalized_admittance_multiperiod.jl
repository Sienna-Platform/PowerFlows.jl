# GA islands, multi-period cache reuse, and radial network reductions (spec §3.1, §5):
# one REF per island, the factorization cache reused across time steps, and parity
# under PNM radial reduction.

@testset "GA: two islands with two REF buses" begin
    sys = PSB.build_system(PSB.PSITestSystems, "c_sys5"; add_forecasts = false)
    b101 = _add_simple_bus!(sys, 101, PSY.ACBusTypes.REF, 230, 1.0)
    b102 = _add_simple_bus!(sys, 102, PSY.ACBusTypes.PQ, 230, 1.0)
    b103 = _add_simple_bus!(sys, 103, PSY.ACBusTypes.PQ, 230, 1.0)
    _add_simple_source!(sys, b101)
    _add_simple_line!(sys, b101, b102, 0.01, 0.1, 0.02)
    _add_simple_line!(sys, b102, b103, 0.01, 0.1, 0.02)
    _add_simple_load!(sys, b102, 30.0, 10.0)
    _add_simple_load!(sys, b103, 20.0, 5.0)
    _, data_ga = ga_parity(sys)
    @test length(PF.GAPartition(data_ga, 1, Dict{Int, Float64}()).s_ix) == 2
end

@testset "GA: multi-period solve reuses the factorization cache" begin
    sys = PSB.build_system(PSB.PSITestSystems, "c_sys14"; add_forecasts = false)
    data_ga =
        PowerFlowData(ACPowerFlow{GeneralizedAdmittanceACPowerFlow}(; time_steps = 3), sys)
    data_nr = PowerFlowData(ACPowerFlow{NewtonRaphsonACPowerFlow}(; time_steps = 3), sys)
    for (t, scale) in enumerate((1.0, 1.1, 0.9)), d in (data_ga, data_nr)
        d.bus_active_power_withdrawals[:, t] .*= scale
    end
    @test solve_power_flow!(data_nr)
    @test solve_power_flow!(data_ga)
    @test maximum(abs.(data_nr.bus_magnitude .- data_ga.bus_magnitude)) < 1e-6
    @test maximum(abs.(data_nr.bus_angles .- data_ga.bus_angles)) < 1e-6
    cache = data_ga.solver_cache[]
    @test typeof(cache) === PF.GeneralizedAdmittanceCache
    @test PF._get_or_build_ga_cache!(
        data_ga,
        PF.GAPartition(data_ga, 1, Dict{Int, Float64}()),
    ) === cache
end

@testset "GA: radial network reduction parity" begin
    sys = PSB.build_system(PSB.PSITestSystems, "c_sys14"; add_forecasts = false)
    ga_parity(
        sys;
        pf_kwargs = (; network_reductions = PNM.NetworkReduction[PNM.RadialReduction()]),
    )
end
