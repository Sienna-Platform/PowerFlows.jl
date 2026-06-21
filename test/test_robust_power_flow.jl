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

# Build the case5_2_lcc HVDC system, optionally flipping every LCC's transfer
# setpoint negative so the P-setpoint is metered at the inverter
# (`setpoint_at_rectifier = false`). The inverter case exercises the
# side-aware branch of both the LCC Jacobian and the homotopy Hessian.
function _case5_lcc_system(; setpoint_at_inverter::Bool = false)
    sys = System(joinpath(TEST_DATA_DIR, "case5_2_lcc.raw"))
    if setpoint_at_inverter
        for lcc in get_components(PSY.TwoTerminalLCCLine, sys)
            set_transfer_setpoint!(lcc, -abs(get_transfer_setpoint(lcc)))
        end
    end
    return sys
end

@testset "RobustHomotopy on LCC HVDC system: matches NR ($(label))" for (
    label,
    setpoint_at_inverter,
) in (
    ("setpoint at rectifier", false),
    ("setpoint at inverter", true),
)
    sys = _case5_lcc_system(; setpoint_at_inverter)
    sys2 = deepcopy(sys)
    pf_hom = ACPowerFlow{PF.RobustHomotopyPowerFlow}()
    data_hom = PowerFlowData(pf_hom, sys)
    solve_power_flow!(data_hom; pf = pf_hom)
    @test all(data_hom.converged)
    # Confirm the parametrization actually toggles the setpoint side.
    @test all(data_hom.lcc.setpoint_at_rectifier .== !setpoint_at_inverter)

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
