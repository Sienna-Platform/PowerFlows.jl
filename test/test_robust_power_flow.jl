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

@testset "RobustHomotopy on LCC HVDC system: matches NR ($(label))" for (label, raw_file) in
                                                                        (
    ("case5_lcc", "case5_lcc.raw"),
    ("case5_2_lcc", "case5_2_lcc.raw"),
)
    raw_path = joinpath(TEST_DATA_DIR, raw_file)
    sys = System(raw_path)
    sys2 = deepcopy(sys)
    pf_hom = ACPowerFlow{PF.RobustHomotopyPowerFlow}()
    data_hom = PowerFlowData(pf_hom, sys)
    solve_power_flow!(data_hom; pf = pf_hom)
    @test all(data_hom.converged)

    pf_nr = ACPowerFlow()
    data_nr = PowerFlowData(pf_nr, sys2)
    solve_power_flow!(data_nr; pf = pf_nr)
    @test all(data_nr.converged)

    @test isapprox(data_nr.bus_angles, data_hom.bus_angles; atol = 1e-4)
    @test isapprox(data_nr.bus_magnitude, data_hom.bus_magnitude; atol = 1e-6)
    @test isapprox(data_nr.lcc.rectifier.tap, data_hom.lcc.rectifier.tap; atol = 1e-4)
    @test isapprox(data_nr.lcc.inverter.tap, data_hom.lcc.inverter.tap; atol = 1e-4)
    @test isapprox(
        data_nr.lcc.rectifier.thyristor_angle,
        data_hom.lcc.rectifier.thyristor_angle;
        atol = 1e-4,
    )
    @test isapprox(
        data_nr.lcc.inverter.thyristor_angle,
        data_hom.lcc.inverter.thyristor_angle;
        atol = 1e-4,
    )
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
