# GA handoff (spec §4.4): the NR polish after a `GAConverged` stage, the NR rescue
# after a `GAMaxIter` stage, and the no-handoff non-convergence that must not
# NaN-poison the state.

@testset "GA: handoff paths on c_sys14" begin
    sys = PSB.build_system(PSB.PSITestSystems, "c_sys14"; add_forecasts = false)
    pf = ACPowerFlow{GeneralizedAdmittanceACPowerFlow}()
    data_nr = PowerFlowData(ACPowerFlow{NewtonRaphsonACPowerFlow}(), sys)
    @test solve_power_flow!(data_nr)

    data = PowerFlowData(pf, sys)
    polish = PF._ga_solve(
        pf, data, 1; handoff_solver = NewtonRaphsonACPowerFlow, handoff_tol = 1e-3)
    @test polish.converged
    @test polish.stage_exit === PF.GAConverged()
    @test polish.handoff_iterations <= 4
    @test maximum(abs.(data_nr.bus_magnitude .- data.bus_magnitude)) < 1e-6

    rescue = PF._ga_solve(
        pf, PowerFlowData(pf, sys), 1;
        maxIterations = 2, handoff_solver = NewtonRaphsonACPowerFlow,
        handoff_tol = 1e-12)
    @test rescue.stage_exit === PF.GAMaxIter()
    @test rescue.handoff_iterations > 0
    @test rescue.converged

    data_fail = PowerFlowData(pf, sys)
    fail = PF._ga_solve(pf, data_fail, 1; maxIterations = 2)
    @test !fail.converged
    @test all(isfinite, data_fail.bus_magnitude[:, 1])

    @test_throws ArgumentError PF._ga_solve(
        pf, PowerFlowData(pf, sys), 1; handoff_solver = GradientDescentACPowerFlow)
end
