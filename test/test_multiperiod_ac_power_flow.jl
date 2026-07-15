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

# `solve_power_flow!`'s loop containers are sized to `length(sorted_time_steps)` but were
# indexed by the raw time-step VALUE, not its position — isolated/non-contiguous subsets
# like `[2]` or `[1, 3]` threw `BoundsError`.
@testset "solve_power_flow! with an isolated/non-contiguous time_steps subset" begin
    @testset "isolated non-leading subset [2] on a 2-step fixture" begin
        sys = PSB.build_system(PSB.PSITestSystems, "c_sys14"; add_forecasts = false)

        pf_full = ACPowerFlow{NewtonRaphsonACPowerFlow}(; time_steps = 2)
        data_full = PowerFlowData(pf_full, sys)
        prepare_ts_data!(data_full, 2)
        @test solve_power_flow!(data_full)

        pf_sub = ACPowerFlow{NewtonRaphsonACPowerFlow}(; time_steps = 2)
        data_sub = PowerFlowData(pf_sub, sys)
        prepare_ts_data!(data_sub, 2)
        # Snapshot ts=1's pristine (never-solved) state before touching ts=2.
        magnitude_ts1_before = copy(data_sub.bus_magnitude[:, 1])
        angles_ts1_before = copy(data_sub.bus_angles[:, 1])
        flow_ts1_before = copy(data_sub.arc_active_power_flow_from_to[:, 1])
        @test solve_power_flow!(data_sub; time_steps = [2])

        @test data_sub.converged[2]
        @test isapprox(data_sub.bus_magnitude[:, 2], data_full.bus_magnitude[:, 2];
            atol = 1e-9)
        @test isapprox(data_sub.bus_angles[:, 2], data_full.bus_angles[:, 2]; atol = 1e-9)
        @test isapprox(
            data_sub.arc_active_power_flow_from_to[:, 2],
            data_full.arc_active_power_flow_from_to[:, 2];
            atol = 1e-9,
        )

        @test !data_sub.converged[1]
        @test data_sub.bus_magnitude[:, 1] == magnitude_ts1_before
        @test data_sub.bus_angles[:, 1] == angles_ts1_before
        @test data_sub.arc_active_power_flow_from_to[:, 1] == flow_ts1_before
    end

    @testset "non-contiguous subset [1, 3] on a 3-step fixture" begin
        sys = PSB.build_system(PSB.PSITestSystems, "c_sys14"; add_forecasts = false)

        pf_full = ACPowerFlow{NewtonRaphsonACPowerFlow}(; time_steps = 3)
        data_full = PowerFlowData(pf_full, sys)
        prepare_ts_data!(data_full, 3)
        @test solve_power_flow!(data_full)

        pf_sub = ACPowerFlow{NewtonRaphsonACPowerFlow}(; time_steps = 3)
        data_sub = PowerFlowData(pf_sub, sys)
        prepare_ts_data!(data_sub, 3)
        # Snapshot ts=2's pristine (never-solved) state before touching ts=1/ts=3.
        magnitude_ts2_before = copy(data_sub.bus_magnitude[:, 2])
        angles_ts2_before = copy(data_sub.bus_angles[:, 2])
        @test solve_power_flow!(data_sub; time_steps = [1, 3])

        for t in (1, 3)
            @test data_sub.converged[t]
            @test isapprox(data_sub.bus_magnitude[:, t], data_full.bus_magnitude[:, t];
                atol = 1e-9)
            @test isapprox(data_sub.bus_angles[:, t], data_full.bus_angles[:, t];
                atol = 1e-9)
        end

        @test !data_sub.converged[2]
        @test data_sub.bus_magnitude[:, 2] == magnitude_ts2_before
        @test data_sub.bus_angles[:, 2] == angles_ts2_before
    end
end
