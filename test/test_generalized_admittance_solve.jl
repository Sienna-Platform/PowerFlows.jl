@testset "GA: parity with NR on $name" for name in ("c_sys5", "c_sys14")
    ga_parity(PSB.build_system(PSB.PSITestSystems, name; add_forecasts = false))
end

@testset "GA: solve report on c_sys14" begin
    sys = PSB.build_system(PSB.PSITestSystems, "c_sys14"; add_forecasts = false)
    pf = ACPowerFlow{GeneralizedAdmittanceACPowerFlow}()
    report = PF._ga_solve(pf, PowerFlowData(pf, sys), 1)
    @test report.converged
    @test report.stage_exit === PF.GAConverged()
    @test iszero(report.handoff_iterations)
    @info "GA c_sys14" report.stage_iterations
end

@testset "GA: non-zero REF angle" begin
    sys = PSB.build_system(PSB.PSITestSystems, "c_sys14"; add_forecasts = false)
    ref = only(
        collect(
            PSY.get_components(
                x -> PSY.get_bustype(x) == PSY.ACBusTypes.REF, PSY.ACBus, sys),
        ),
    )
    PSY.set_angle!(ref, 0.12345)
    ga_parity(sys)
end

@testset "GA: loss factors match NR" begin
    sys = PSB.build_system(PSB.PSITestSystems, "c_sys14"; add_forecasts = false)
    data_nr, data_ga = ga_parity(sys; pf_kwargs = (; calculate_loss_factors = true))
    @test maximum(abs.(data_nr.loss_factors .- data_ga.loss_factors)) < 1e-6
end

@testset "GA: large ACTIVSg2000 with NR polish" begin
    sys = PSB.build_system(PSB.MatpowerTestSystems, "matpower_ACTIVSg2000_sys")
    data_nr = PowerFlowData(
        ACPowerFlow{NewtonRaphsonACPowerFlow}(; correct_bustypes = true), sys)
    @test solve_power_flow!(data_nr)
    pf = ACPowerFlow{GeneralizedAdmittanceACPowerFlow}(; correct_bustypes = true)
    data = PowerFlowData(pf, sys)
    report = PF._ga_solve(
        pf, data, 1; handoff_solver = NewtonRaphsonACPowerFlow, handoff_tol = 1e-3)
    @info "GA ACTIVSg2000 NR polish" report.stage_exit report.stage_iterations report.handoff_iterations
    @test report.converged
    @test maximum(abs.(data_nr.bus_magnitude .- data.bus_magnitude)) < 1e-6
    @test maximum(abs.(data_nr.bus_angles .- data.bus_angles)) < 1e-6
end
