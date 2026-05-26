function flows_from_dataframe(flow_results_df::DataFrame,
    arc_lookup::Dict{Tuple{Int, Int}, Int},
    direction::Symbol = :P_from_to,
)
    flows = fill(NaN, length(arc_lookup))
    for row in eachrow(flow_results_df)
        flows[arc_lookup[(row.bus_from, row.bus_to)]] = row[direction]
    end
    @assert !any(isnan.(flows))
    return flows
end

@testset "SINGLE PERIOD power flows evaluation: ABA, PTDF, VirtualPTDF" begin
    # get system
    sys = PSB.build_system(PSB.PSITestSystems, "c_sys14"; add_forecasts = false)

    # get indices
    buses = collect(PSY.get_components(PSY.ACBus, sys))

    # get sorted indices for branches
    branches = collect(PSY.get_available_components(PSY.ACBranch, sys))
    from_bus = [PSY.get_number(PSY.get_arc(x).from) for x in branches]

    # get reference values: flows and angles.
    # See issue 210: would be better to compare against external program.
    data = PowerFlowData(DCPowerFlow(; correct_bustypes = true), sys)
    power_injections =
        deepcopy(data.bus_active_power_injections - data.bus_active_power_withdrawals)
    matrix_data = deepcopy(data.power_network_matrix.K)       # LU factorization of ABA
    aux_network_matrix = deepcopy(data.aux_network_matrix)    # BA matrix

    valid_ix = setdiff(
        1:length(power_injections),
        PNM.get_ref_bus_position(data.aux_network_matrix),
    )
    ref_bus_angles = deepcopy(data.bus_angles)
    ref_flow_values = deepcopy(data.arc_active_power_flow_from_to)

    ref_bus_angles[valid_ix] = matrix_data \ power_injections[valid_ix]
    ref_flow_values = transpose(aux_network_matrix.data) * ref_bus_angles

    basepower = PSY.get_base_power(sys)
    arc_lookup = PF.get_arc_lookup(data)
    # CASE 1: ABA and BA matrices
    solved_data_ABA = solve_power_flow(
        DCPowerFlow(; correct_bustypes = true),
        sys,
        PF.FlowReporting.ARC_FLOWS,
    )
    ABA_branch_flows = solved_data_ABA["1"]["flow_results"]
    @test isapprox(
        1 / basepower .* flows_from_dataframe(ABA_branch_flows, arc_lookup, :P_from_to),
        ref_flow_values,
        atol = 1e-6,
    )
    @test isapprox(
        1 / basepower .* flows_from_dataframe(ABA_branch_flows, arc_lookup, :P_to_from),
        -ref_flow_values,
        atol = 1e-6,
    )
    @test isapprox(solved_data_ABA["1"]["bus_results"].θ, ref_bus_angles, atol = 1e-6)

    # CASE 2: PTDF and ABA MATRICES
    solved_data_PTDF = solve_power_flow(
        PTDFDCPowerFlow(; correct_bustypes = true),
        sys,
        PF.FlowReporting.ARC_FLOWS,
    )
    PTDF_branch_flows = solved_data_PTDF["1"]["flow_results"]
    @test isapprox(
        1 / basepower .* flows_from_dataframe(PTDF_branch_flows, arc_lookup, :P_from_to),
        ref_flow_values,
        atol = 1e-6,
    )
    @test isapprox(
        1 / basepower .* flows_from_dataframe(PTDF_branch_flows, arc_lookup, :P_to_from),
        -ref_flow_values,
        atol = 1e-6,
    )
    @test isapprox(solved_data_PTDF["1"]["bus_results"].θ, ref_bus_angles, atol = 1e-6)

    # CASE 3: VirtualPTDF and ABA MATRICES
    solved_data_vPTDF = solve_power_flow(
        vPTDFDCPowerFlow(; correct_bustypes = true),
        sys,
        PF.FlowReporting.ARC_FLOWS,
    )
    vPTDF_branch_flows = solved_data_vPTDF["1"]["flow_results"]
    @test isapprox(
        1 / basepower .* flows_from_dataframe(vPTDF_branch_flows, arc_lookup, :P_from_to),
        ref_flow_values,
        atol = 1e-6,
    )
    @test isapprox(
        1 / basepower .* flows_from_dataframe(vPTDF_branch_flows, arc_lookup, :P_to_from),
        -ref_flow_values,
        atol = 1e-6,
    )
    @test isapprox(solved_data_vPTDF["1"]["bus_results"].θ, ref_bus_angles, atol = 1e-6)
end

@testset "DC power flow with an LCC" begin
    sys, lcc = simple_lcc_system()
    @assert get_base_power(sys) == 100.0 "Test system base power changed."
    @assert get_units_base(sys) == "SYSTEM_BASE" "Test system unit setting changed."
    set_active_power_flow!(lcc, 0.3)
    for T in (DCPowerFlow, PTDFDCPowerFlow, vPTDFDCPowerFlow)
        results =
            solve_power_flow(T(; correct_bustypes = true), sys, PF.FlowReporting.ARC_FLOWS)
        lcc_flow = results["1"]["lcc_results"][1, :P_from_to]
        # 1st arg must be lcc, not sys, else test fails. See issue #1590 in PowerSystems.jl
        with_units_base(lcc, PSY.UnitSystem.NATURAL_UNITS) do
            @test lcc_flow == get_active_power_flow(lcc)
        end
    end
end

@testset "DC power flow with an LCC: i_dc initialization edge cases" begin
    sys, lcc = simple_lcc_system()
    PSY.set_r!(lcc, 0.0)

    # With zero resistance, R * I_dc^2 + I_dc - P_set = 0 reduces to I_dc = P_set.
    PSY.set_transfer_setpoint!(lcc, 25.0)
    for T in (DCPowerFlow, PTDFDCPowerFlow, vPTDFDCPowerFlow)
        data = PowerFlowData(T(; correct_bustypes = true), sys)
        @test !isnan(data.lcc.i_dc[1, 1])
        @test isapprox(data.lcc.i_dc[1, 1], data.lcc.p_set[1, 1]; atol = 1e-12)
    end

    # Zero setpoint should also initialize safely (no NaN).
    PSY.set_transfer_setpoint!(lcc, 0.0)
    data = PowerFlowData(DCPowerFlow(; correct_bustypes = true), sys)
    @test !isnan(data.lcc.i_dc[1, 1])
    @test iszero(data.lcc.i_dc[1, 1])
end

# TODO LCC DC test case with nonzero loss.

@testset "DC power flow: results independent of units" begin
    sys = PSB.build_system(PSB.PSITestSystems, "c_sys14"; add_forecasts = false)
    for T in (DCPowerFlow, PTDFDCPowerFlow, vPTDFDCPowerFlow)
        line_name, flow_natural =
            power_flow_with_units(sys, T, PSY.UnitSystem.NATURAL_UNITS)
        line_name2, flow_system = power_flow_with_units(sys, T, PSY.UnitSystem.SYSTEM_BASE)
        @test line_name == line_name2
        @test flow_natural == flow_system
    end
end

function set_zip_load_in_mva!(sys::PSY.System, tp::Tuple{Float64, Float64, Float64})
    set_units_base_system!(sys, PSY.UnitSystem.NATURAL_UNITS)
    load = only(get_components(StandardLoad, sys))
    set_zip_loads_active_power!(load, tp)
    set_units_base_system!(sys, PSY.UnitSystem.SYSTEM_BASE)
end

function set_zip_loads_active_power!(
    load::StandardLoad,
    tp::Tuple{Float64, Float64, Float64},
)
    set_constant_active_power!(load, tp[1])
    set_impedance_active_power!(load, tp[2])
    set_current_active_power!(load, tp[3])
end

@testset "DC power flow: StandardLoad" begin
    sys = PSB.build_system(PSB.PSITestSystems, "c_sys5")
    # change all loads to StandardLoad
    dc_baseline = solve_power_flow(
        DCPowerFlow(; correct_bustypes = true),
        sys,
        PF.FlowReporting.ARC_FLOWS,
    )
    set_units_base_system!(sys, PSY.UnitSystem.NATURAL_UNITS)
    load = first(get_components(PowerLoad, sys))
    P = PSY.get_active_power(load)
    println("original load draws: ", P, " MVA")
    remove_component!(sys, load)
    new_load = PSY.StandardLoad(;
        name = get_name(load),
        available = true,
        bus = PSY.get_bus(load),
        base_power = PSY.get_base_power(load),
        constant_active_power = 0.0,
        constant_reactive_power = 0.0,
        impedance_active_power = 0.0,
        impedance_reactive_power = 0.0,
        current_active_power = 0.0,
        current_reactive_power = 0.0,
        max_constant_active_power = PSY.get_max_active_power(load),
        max_constant_reactive_power = PSY.get_max_reactive_power(load),
        max_impedance_active_power = PSY.get_max_active_power(load),
        max_impedance_reactive_power = PSY.get_max_reactive_power(load),
        max_current_active_power = PSY.get_max_active_power(load),
        max_current_reactive_power = PSY.get_max_reactive_power(load),
    )
    add_component!(sys, new_load)
    set_zip_load_in_mva!(sys, (0.0, P, 0.0))
    impedance_solved = solve_power_flow(DCPowerFlow(), sys, PF.FlowReporting.ARC_FLOWS)
    set_zip_load_in_mva!(sys, (0.0, 0.0, P))
    current_solved = solve_power_flow(DCPowerFlow(), sys, PF.FlowReporting.ARC_FLOWS)

    @test isapprox(
        dc_baseline["1"]["bus_results"],
        impedance_solved["1"]["bus_results"],
        atol = 1e-6,
    )
    @test isapprox(
        dc_baseline["1"]["bus_results"],
        current_solved["1"]["bus_results"],
        atol = 1e-6,
    )

    set_zip_load_in_mva!(sys, (P * 0.2, P * 0.3, P * 0.5))
    combined_solved = solve_power_flow(DCPowerFlow(), sys, PF.FlowReporting.ARC_FLOWS)
    @test isapprox(
        dc_baseline["1"]["bus_results"],
        combined_solved["1"]["bus_results"],
        atol = 1e-6,
    )
end

@testset "DC arc_angle_differences validation" begin
    sys = PSB.build_system(PSB.PSITestSystems, "c_sys14"; add_forecasts = false)
    for T in (DCPowerFlow, PTDFDCPowerFlow, vPTDFDCPowerFlow)
        data = PowerFlowData(T(; correct_bustypes = true), sys)
        solve_power_flow!(data)
        validate_arc_angle_differences(data, [1])
    end

    # DataFrame column check
    results = solve_power_flow(
        DCPowerFlow(; correct_bustypes = true),
        sys,
        PF.FlowReporting.ARC_FLOWS,
    )
    @test :angle_difference in propertynames(results["1"]["flow_results"])
end

@testset "DC branch losses estimation" begin
    sys = PSB.build_system(PSB.PSITestSystems, "c_sys14"; add_forecasts = false)
    base_power = PSY.get_base_power(sys)

    for T in (DCPowerFlow, PTDFDCPowerFlow, vPTDFDCPowerFlow)
        data = PowerFlowData(T(; correct_bustypes = true), sys)
        results = solve_power_flow(data, sys, PF.FlowReporting.ARC_FLOWS)

        # P_losses should be non-negative and non-trivially zero.
        flow_df = results["1"]["flow_results"]
        @test all(flow_df[!, :P_losses] .>= 0.0)
        @test any(flow_df[!, :P_losses] .> 0.0)

        # Q_losses must remain zero for DC.
        @test all(flow_df[!, :Q_losses] .== 0.0)

        # Validate P_losses = R * flow^2 (scaled by base_power).
        validate_dc_branch_losses(data, results, base_power, [1])
    end
end

@testset "DC arc active power losses: loss = r * flow^2" begin
    sys = PSB.build_system(PSB.PSITestSystems, "c_sys14"; add_forecasts = false)

    for T in (DCPowerFlow, PTDFDCPowerFlow, vPTDFDCPowerFlow)
        data = PowerFlowData(T(; correct_bustypes = true), sys)
        solve_power_flow!(data)

        # The field should be populated after solve.
        @test data.arc_active_power_losses !== nothing
        losses = data.arc_active_power_losses

        # Recompute expected losses from resistances and flows.
        Rs = PF._get_arc_resistances(data)
        expected = Rs .* data.arc_active_power_flow_from_to .^ 2
        @test isapprox(losses, expected; atol = 1e-12)

        # Losses must be non-negative.
        @test all(losses .>= 0.0)
    end
end

@testset "DC branch-level losses with BRANCH_FLOWS reporting" begin
    sys = PSB.build_system(PSB.PSITestSystems, "c_sys14"; add_forecasts = false)
    base_power = PSY.get_base_power(sys)

    for T in (DCPowerFlow, PTDFDCPowerFlow, vPTDFDCPowerFlow)
        results = solve_power_flow(
            T(; correct_bustypes = true),
            sys,
            PF.FlowReporting.BRANCH_FLOWS,
        )
        flow_df = results["1"]["flow_results"]

        # P_losses column must exist and be non-negative.
        @test :P_losses in propertynames(flow_df)
        @test all(flow_df[!, :P_losses] .>= 0.0)
        @test any(flow_df[!, :P_losses] .> 0.0)

        # Q_losses must be zero for DC.
        @test all(flow_df[!, :Q_losses] .== 0.0)
    end
end
