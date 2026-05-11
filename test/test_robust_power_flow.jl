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

if PF.OVERRIDE_x0
    @testset "robust homotopy respects override_x0" begin
        sys = PSB.build_system(PSB.PSITestSystems, "c_sys14")
        pf_hom = ACPowerFlow{PF.RobustHomotopyPowerFlow}()
        data = PowerFlowData(pf_hom, sys)

        x0 = PF.calculate_x0(data, 1)
        # perturb angles so the override is detectable
        for (i, bt) in enumerate(PF.get_bus_type(data)[:, 1])
            if bt == PSY.ACBusTypes.PQ
                x0[2 * i] *= 1.05  # small angle perturbation
            end
        end

        @test_logs (:warn, r"Overriding initial guess x0.*") match_mode = :any solve_power_flow!(
            data;
            pf = pf_hom,
            x0 = x0,
        )
    end
end
