@testset "test robust homotopy power flow" begin
    sys = PSB.build_system(PSB.PSITestSystems, "c_sys14")
    sys2 = deepcopy(sys)
    pf_hom = ACPowerFlow{PF.RobustHomotopyPowerFlow}()
    data_hom = PowerFlowData(pf_hom, sys)
    # infologger = ConsoleLogger(stderr, Logging.Info)
    # with_logger(infologger) do; solve_power_flow!(data_hom; pf = pf_hom); end;
    solve_power_flow!(data_hom; pf = pf_hom)

    pf_nr = ACPowerFlow()
    data_nr = PowerFlowData(pf_nr, sys2)
    solve_power_flow!(data_nr; pf = pf_nr)
    @test isapprox(data_nr.bus_angles, data_hom.bus_angles; atol = 1e-4)
    @test isapprox(data_nr.bus_magnitude, data_hom.bus_magnitude; atol = 1e-6)
end

@testset "RobustHomotopy rejects systems with LCCs" begin
    sys, _ = simple_lcc_system()
    pf_hom = ACPowerFlow{PF.RobustHomotopyPowerFlow}()
    @test_throws ArgumentError solve_power_flow(pf_hom, sys)
end

@testset "test robust homotopy power flow with headroom-proportional slack" begin
    sys = PSB.build_system(PSB.PSITestSystems, "c_sys14")
    sys2 = deepcopy(sys)
    pf_hom = ACPowerFlow{PF.RobustHomotopyPowerFlow}(;
        distribute_slack_proportional_to_headroom = true,
    )
    data_hom = PowerFlowData(pf_hom, sys)
    solve_power_flow!(data_hom; pf = pf_hom)

    pf_nr = ACPowerFlow(; distribute_slack_proportional_to_headroom = true)
    data_nr = PowerFlowData(pf_nr, sys2)
    solve_power_flow!(data_nr; pf = pf_nr)
    @test isapprox(data_nr.bus_angles, data_hom.bus_angles; atol = 1e-4)
    @test isapprox(data_nr.bus_magnitude, data_hom.bus_magnitude; atol = 1e-6)
end
