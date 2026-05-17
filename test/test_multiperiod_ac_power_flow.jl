# work in progress
@testset "MULTI-PERIOD power flows evaluation" begin
    for ACSolver in AC_SOLVERS_TO_TEST
        @testset "AC Solver: $(ACSolver)" begin
            # get system
            sys = PSB.build_system(PSB.PSITestSystems, "c_sys14"; add_forecasts = false)

            # create structure for multi-period case
            time_steps = 24
            pf = ACPowerFlow{ACSolver}(; time_steps = time_steps)
            data = PowerFlowData(pf, sys)

            # allocate timeseries data from csv
            prepare_ts_data!(data, time_steps)

            # get power flows with NR KLU method and write results
            solve_power_flow!(data)

            # check results
            # for t in 1:length(get_time_step_map(data))
            #     res_t = solve_power_flow(pf, sys, t)  # does not work - ts data not set in sys
            #     flow_ft = res_t["flow_results"].P_from_to
            #     flow_tf = res_t["flow_results"].P_to_from
            #     ts_flow_ft = results[get_time_step_map(data)[t]]["flow_results"].P_from_to
            #     ts_flow_tf = results[get_time_step_map(data)[t]]["flow_results"].P_to_from
            #     @test isapprox(ts_flow_ft, flow_ft, atol = 1e-9)
            #     @test isapprox(ts_flow_tf, flow_tf, atol = 1e-9)
            # end
        end
    end
end

@testset "MULTI-PERIOD power flows evaluation: compare results for different solvers" begin
    # get system
    sys = PSB.build_system(PSB.PSITestSystems, "c_sys14"; add_forecasts = false)

    # create structure for multi-period case
    time_steps = 24
    pf_tr = ACPowerFlow{TrustRegionACPowerFlow}(; time_steps = time_steps)
    pf_nr = ACPowerFlow{NewtonRaphsonACPowerFlow}(; time_steps = time_steps)

    data_tr = PowerFlowData(pf_tr, sys)
    data_nr = PowerFlowData(pf_nr, sys)

    # allocate timeseries data from csv
    prepare_ts_data!(data_tr, time_steps)
    prepare_ts_data!(data_nr, time_steps)

    # solve with both methods
    solve_power_flow!(data_tr)
    solve_power_flow!(data_nr)

    # check results
    @test isapprox(data_tr.bus_magnitude, data_nr.bus_magnitude, atol = 1e-9)
    @test isapprox(data_tr.bus_angles, data_nr.bus_angles, atol = 1e-9)
    @test isapprox(
        data_tr.arc_active_power_flow_from_to,
        data_nr.arc_active_power_flow_from_to,
        atol = 1e-9,
    )
    @test isapprox(
        data_tr.arc_active_power_flow_to_from,
        data_nr.arc_active_power_flow_to_from,
        atol = 1e-9,
    )
    @test isapprox(
        data_tr.arc_reactive_power_flow_from_to,
        data_nr.arc_reactive_power_flow_from_to,
        atol = 1e-9,
    )
    @test isapprox(
        data_tr.arc_reactive_power_flow_to_from,
        data_nr.arc_reactive_power_flow_to_from,
        atol = 1e-9,
    )
end
