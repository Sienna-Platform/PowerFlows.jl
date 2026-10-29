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
    raw_path = joinpath(TEST_DATA_DIR, "case5_2_lcc.raw")
    sys = system_from_openapi(PFP.PowerModelsData(raw_path); runchecks = false)
    if setpoint_at_inverter
        for lcc in get_components(PSY.TwoTerminalLCCLine, sys)
            set_power_transfer_setpoint!(
                lcc,
                (-abs(get_power_transfer_setpoint(lcc, u"CU"))) * u"CU",
            )
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

@testset "flat-start stages: DC warm start on a cold start, GA before the DC angles" begin
    sys = PSB.build_system(PSB.PSITestSystems, "c_sys14")
    count_logs(logs, pattern) = count(l -> occursin(pattern, string(l.message)), logs)
    function solve_logged!(data, pf)
        return Test.collect_test_logs(; min_level = Logging.Debug) do
            return solve_power_flow!(data; pf = pf)
        end
    end
    function flat_data(pf)
        data = PowerFlowData(pf, sys)
        data.bus_angles .= 0.0
        return data
    end
    dc_taken = "DC power flow angles yields smaller residual"
    ga_ran = "Generalized-admittance flat start:"

    # The flat start's residual is below the LARGE_RESIDUAL gate, so only flatness triggers DC.
    pf_default = ACPowerFlow()
    data = flat_data(pf_default)
    residual = PF.ACPowerFlowResidual(data, 1)
    residual(data, PF.calculate_x0(data, 1), 1)
    @test PF._is_flat_start(residual, data, 1)
    @test !PF._large_residual(residual)
    logs, ok = solve_logged!(data, pf_default)
    @test ok
    @test count_logs(logs, dc_taken) == 1
    residual(data, PF.calculate_x0(data, 1), 1)
    @test !PF._is_flat_start(residual, data, 1)
    # Zeroed angles with the REF kept at a case-file angle are flat too.
    data_ref = flat_data(pf_default)
    ref = findfirst(==(PSY.ACBusTypes.REF), data_ref.bus_type[:, 1])
    data_ref.bus_angles[ref, 1] = 0.3
    residual_ref = PF.ACPowerFlowResidual(data_ref, 1)
    residual_ref(data_ref, PF.calculate_x0(data_ref, 1), 1)
    @test PF._is_flat_start(residual_ref, data_ref, 1)

    pf_off = ACPowerFlow(; dc_warm_start = false)
    data_off = flat_data(pf_off)
    @test isnothing(PF.get_aux_network_matrix(data_off))
    logs, ok = solve_logged!(data_off, pf_off)
    @test ok
    @test count_logs(logs, dc_taken) == 0
    @test isapprox(data.bus_magnitude, data_off.bus_magnitude; atol = 1e-8)
    @test isapprox(data.bus_angles, data_off.bus_angles; atol = 1e-8)

    # `ga_flat_start` implies the DC warm start. GA runs first; a GA start at `handoff_tol`
    # skips the DC angles.
    pf_ga_only = ACPowerFlow(; dc_warm_start = false, ga_flat_start = true)
    @test PF.get_dc_warm_start(pf_ga_only)
    logs, ok = solve_logged!(flat_data(pf_ga_only), pf_ga_only)
    @test ok
    @test count_logs(logs, ga_ran) == 1
    @test count_logs(logs, "DC power flow angles") == 0

    # `converged` is read as of entry, so a first multi-period solve starts every step flat.
    pf_ga = ACPowerFlow(; ga_flat_start = true, time_steps = 3)
    data_ts = flat_data(pf_ga)
    data_ts.bus_active_power_withdrawals[:, 2:3] .*= 1.05
    logs, ok = solve_logged!(data_ts, pf_ga)
    @test ok
    @test count_logs(logs, ga_ran) == 3

    # A re-solve warm-starts every step from a converged earlier one: no GA.
    data_ts.bus_active_power_withdrawals .*= 1.02
    logs, ok = solve_logged!(data_ts, pf_ga)
    @test ok
    @test count_logs(logs, ga_ran) == 0

    # A solved, non-flat start with no converged earlier step (a re-solve) skips GA too.
    fill!(data_ts.converged, false)
    data_ts.bus_active_power_withdrawals .*= 1.05
    residual = PF.ACPowerFlowResidual(data_ts, 1)
    residual(data_ts, PF.calculate_x0(data_ts, 1), 1)
    @test norm(residual.Rv, Inf) > PF.get_solution_parameters(pf_ga).handoff_tol
    logs, ok = solve_logged!(data_ts, pf_ga)
    @test ok
    @test count_logs(logs, ga_ran) == 0
end
