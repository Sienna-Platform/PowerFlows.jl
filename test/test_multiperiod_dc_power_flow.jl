@testset "MULTI-PERIOD power flows evaluation: ABA, PTDF, VirtualPTDF" begin
    # get system
    sys = PSB.build_system(PSB.PSITestSystems, "c_sys14"; add_forecasts = false)
    injections = CSV.read(
        joinpath(TEST_DATA_DIR, "c_sys14_injections.csv"),
        DataFrame;
        header = 0,
    )
    withdrawals = CSV.read(
        joinpath(TEST_DATA_DIR, "c_sys14_withdrawals.csv"),
        DataFrame;
        header = 0,
    )
    flows = CSV.read(
        joinpath(TEST_DATA_DIR, "c_sys14_flows.csv"),
        DataFrame;
        header = 0,
    )
    angles = CSV.read(
        joinpath(TEST_DATA_DIR, "c_sys14_angles.csv"),
        DataFrame;
        header = 0,
    )

    ##############################################################################

    # create structure for multi-period case
    time_steps = 24
    data_1 =
        PowerFlowData(DCPowerFlow(; time_steps = time_steps, correct_bustypes = true), sys)
    data_2 =
        PowerFlowData(
            PTDFDCPowerFlow(; time_steps = time_steps, correct_bustypes = true),
            sys,
        )
    data_3 =
        PowerFlowData(
            vPTDFDCPowerFlow(; time_steps = time_steps, correct_bustypes = true),
            sys,
        )

    # allocate data from csv
    injs = Matrix(injections)
    withs = Matrix(withdrawals)

    data_1.bus_active_power_injections .= deepcopy(injs)
    data_1.bus_active_power_withdrawals .= deepcopy(withs)

    data_2.bus_active_power_injections .= deepcopy(injs)
    data_2.bus_active_power_withdrawals .= deepcopy(withs)

    data_3.bus_active_power_injections .= deepcopy(injs)
    data_3.bus_active_power_withdrawals .= deepcopy(withs)

    # case 1: get power flows with ABA method and write results
    results_1 = solve_power_flow(data_1, sys, PF.FlowReporting.ARC_FLOWS)

    # case 2: get power flows PTDF method and write results
    results_2 = solve_power_flow(data_2, sys, PF.FlowReporting.ARC_FLOWS)

    # case 3: get power flows Virtual PTDF method and write results
    results_3 = solve_power_flow(data_3, sys, PF.FlowReporting.ARC_FLOWS)

    # check results
    # CSVs are in p.u., line flows are in natural units: convert back to p.u.
    basepower = PSY.get_base_power(sys, PSY.NU)

    # case 1
    for i in 1:length(data_1.time_step_map)
        net_flow = results_1[data_1.time_step_map[i]]["flow_results"].P_from_to
        net_flow_tf = results_1[data_1.time_step_map[i]]["flow_results"].P_to_from
        @test isapprox(1 / basepower .* net_flow, flows[:, i], atol = 1e-5)
        @test isapprox(1 / basepower .* net_flow_tf, -flows[:, i], atol = 1e-5)
    end

    # case 2
    for i in 1:length(data_1.time_step_map)
        net_flow = results_1[data_2.time_step_map[i]]["flow_results"].P_from_to
        net_flow_tf = results_1[data_2.time_step_map[i]]["flow_results"].P_to_from
        @test isapprox(1 / basepower .* net_flow, flows[:, i], atol = 1e-5)
        @test isapprox(1 / basepower .* net_flow_tf, -flows[:, i], atol = 1e-5)
    end

    # case 3
    for i in 1:length(data_1.time_step_map)
        net_flow = results_1[data_3.time_step_map[i]]["flow_results"].P_from_to
        net_flow_tf = results_1[data_3.time_step_map[i]]["flow_results"].P_to_from
        @test isapprox(1 / basepower .* net_flow, flows[:, i], atol = 1e-5)
        @test isapprox(1 / basepower .* net_flow_tf, -flows[:, i], atol = 1e-5)
    end
end

@testset "MULTI-PERIOD arc_angle_differences validation" begin
    sys = PSB.build_system(PSB.PSITestSystems, "c_sys14"; add_forecasts = false)
    time_steps = 24
    data = PowerFlowData(
        DCPowerFlow(; time_steps = time_steps, correct_bustypes = true),
        sys,
    )
    prepare_ts_data!(data, time_steps)
    solve_power_flow!(data)

    n_arcs = length(PF.get_arc_axis(data))
    @test size(data.arc_angle_differences) == (n_arcs, time_steps)
    validate_arc_angle_differences(data, [1, 12, 24])
end

@testset "MULTI-PERIOD DC branch losses estimation" begin
    sys = PSB.build_system(PSB.PSITestSystems, "c_sys14"; add_forecasts = false)
    base_power = PSY.get_base_power(sys, PSY.NU)
    time_steps = 24

    data = PowerFlowData(
        DCPowerFlow(; time_steps = time_steps, correct_bustypes = true),
        sys,
    )
    prepare_ts_data!(data, time_steps)
    results = solve_power_flow(data, sys, PF.FlowReporting.ARC_FLOWS)

    # Losses should be non-negative at every time step.
    for t in 1:time_steps
        flow_df = results[data.time_step_map[t]]["flow_results"]
        @test all(flow_df[!, :P_losses] .>= 0.0)
        @test all(flow_df[!, :Q_losses] .== 0.0)
    end

    # Validate P_losses = R * flow^2 at sampled time steps.
    validate_dc_branch_losses(data, results, base_power, [1, 12, 24])
end

@testset "MULTI-PERIOD DC branch losses estimation" begin
    sys = PSB.build_system(PSB.PSITestSystems, "c_sys14"; add_forecasts = false)
    base_power = PSY.get_base_power(sys, PSY.NU)
    time_steps = 24

    data = PowerFlowData(
        DCPowerFlow(; time_steps = time_steps, correct_bustypes = true),
        sys,
    )
    prepare_ts_data!(data, time_steps)
    results = solve_power_flow(data, sys, PF.FlowReporting.ARC_FLOWS)

    # Losses should be non-negative at every time step.
    for t in 1:time_steps
        flow_df = results[data.time_step_map[t]]["flow_results"]
        @test all(flow_df[!, :P_losses] .>= 0.0)
        @test all(flow_df[!, :Q_losses] .== 0.0)
    end

    # Validate P_losses = R * flow^2 at sampled time steps.
    validate_dc_branch_losses(data, results, base_power, [1, 12, 24])
end

@testset "MULTI-PERIOD DC branch losses estimation" begin
    sys = PSB.build_system(PSB.PSITestSystems, "c_sys14"; add_forecasts = false)
    base_power = PSY.get_base_power(sys, PSY.NU)
    time_steps = 24

    data = PowerFlowData(
        DCPowerFlow(; time_steps = time_steps, correct_bustypes = true),
        sys,
    )
    prepare_ts_data!(data, time_steps)
    results = solve_power_flow(data, sys, PF.FlowReporting.ARC_FLOWS)

    # Losses should be non-negative at every time step.
    for t in 1:time_steps
        flow_df = results[data.time_step_map[t]]["flow_results"]
        @test all(flow_df[!, :P_losses] .>= 0.0)
        @test all(flow_df[!, :Q_losses] .== 0.0)
    end

    # Validate P_losses = R * flow^2 at sampled time steps.
    validate_dc_branch_losses(data, results, base_power, [1, 12, 24])
end
