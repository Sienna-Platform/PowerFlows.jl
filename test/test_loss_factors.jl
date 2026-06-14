
@testset "test_loss_factors_case_14" begin
    for ACSolver in AC_SOLVERS_TO_TEST
        # FIXME failing for LevenbergMarquardtACPowerFlow. investigate.
        # FastDecoupled is skipped here too: its linear-rate perturbed solves make the
        # finite-difference brute-force reference noisier than the atol=1e-4 bound (same symptom
        # as LM). Its Jacobian-based loss/voltage-stability factors ARE validated against the NR
        # reference in test_fast_decoupled.jl ("FastDecoupled WP5: loss/vstab factor parity (T9)").
        if ACSolver in (LevenbergMarquardtACPowerFlow, FastDecoupledACPowerFlow)
            continue
        end
        @testset "AC Solver: $(ACSolver)" begin
            sys = build_system(PSITestSystems, "c_sys14"; add_forecasts = false)

            time_steps = 24
            pf = ACPowerFlow{ACSolver}(; time_steps = time_steps)
            pf_lf = ACPowerFlow{ACSolver}(;
                calculate_loss_factors = true,
                time_steps = time_steps,
            )
            data_loss_factors = PowerFlowData(pf_lf, sys)
            data_brute_force = PowerFlowData(pf, sys)

            # allocate timeseries data from csv
            prepare_ts_data!(data_loss_factors, time_steps)
            prepare_ts_data!(data_brute_force, time_steps)

            # get power flows with NR KLU method and write results
            solve_power_flow!(data_loss_factors)

            # get loss factors using brute force approach (sequential power flow evaluations for each bus)
            bf_loss_factors = penalty_factors_brute_force(data_brute_force, pf)

            # confirm that loss factors match for the Jacobian-based and brute force approaches
            @test all(
                isapprox.(
                    bf_loss_factors,
                    data_loss_factors.loss_factors,
                    atol = 1e-4,
                    rtol = 0,
                ),
            )

            # get power flow results without loss factors
            solve_power_flow!(data_brute_force)
            @test isnothing(data_brute_force.loss_factors)
        end
    end
end

@testset "test_loss_factors_multiple_ref_buses" begin
    for ACSolver in AC_SOLVERS_TO_TEST
        # FIXME failing for LevenbergMarquardtACPowerFlow. investigate.
        # FastDecoupled is skipped here too: its linear-rate perturbed solves make the
        # finite-difference brute-force reference noisier than the atol=1e-4 bound (same symptom
        # as LM). Its Jacobian-based loss/voltage-stability factors ARE validated against the NR
        # reference in test_fast_decoupled.jl ("FastDecoupled WP5: loss/vstab factor parity (T9)").
        if ACSolver in (LevenbergMarquardtACPowerFlow, FastDecoupledACPowerFlow)
            continue
        end
        @testset "AC Solver: $(ACSolver)" begin
            # Create a system with two disconnected islands, each with its own REF bus
            sys = System(100.0)

            # Island 1: buses 1-3
            b1 = _add_simple_bus!(sys, 1, ACBusTypes.REF, 230, 1.05, 0.0)
            b2 = _add_simple_bus!(sys, 2, ACBusTypes.PQ, 230, 1.0, 0.0)
            b3 = _add_simple_bus!(sys, 3, ACBusTypes.PQ, 230, 1.0, 0.0)

            # Island 2: buses 4-6
            b4 = _add_simple_bus!(sys, 4, ACBusTypes.REF, 230, 1.02, 0.0)
            b5 = _add_simple_bus!(sys, 5, ACBusTypes.PQ, 230, 1.0, 0.0)
            b6 = _add_simple_bus!(sys, 6, ACBusTypes.PQ, 230, 1.0, 0.0)

            # Add sources at REF buses
            _add_simple_source!(sys, b1, 0.5, 0.1)
            _add_simple_source!(sys, b4, 0.4, 0.08)

            # Add loads at PQ buses
            _add_simple_load!(sys, b2, 0.25, 0.05)
            _add_simple_load!(sys, b3, 0.2, 0.04)
            _add_simple_load!(sys, b5, 0.2, 0.04)
            _add_simple_load!(sys, b6, 0.15, 0.03)

            # Connect buses within island 1 (no connection between islands)
            _add_simple_line!(sys, b1, b2, 0.01, 0.05, 0.02)
            _add_simple_line!(sys, b2, b3, 0.015, 0.08, 0.01)
            _add_simple_line!(sys, b1, b3, 0.012, 0.06, 0.015)

            # Connect buses within island 2 (no connection between islands)
            _add_simple_line!(sys, b4, b5, 0.01, 0.05, 0.02)
            _add_simple_line!(sys, b5, b6, 0.015, 0.08, 0.01)
            _add_simple_line!(sys, b4, b6, 0.012, 0.06, 0.015)

            pf_lf = ACPowerFlow{ACSolver}(; calculate_loss_factors = true)
            data_loss_factors = PowerFlowData(pf_lf, sys)

            # Verify we have multiple REF buses before solving
            ref_buses = findall(==(PSY.ACBusTypes.REF), data_loss_factors.bus_type[:, 1])
            @test length(ref_buses) == 2

            # Solving should succeed (with a warning about multiple REF buses)
            @test_throws ErrorException solve_power_flow!(data_loss_factors; pf = pf_lf)
        end
    end
end

"""
Compute DC loss factors via the element-wise summation formula:
    ∂Loss/∂Pᵢ = Σₖ 2·Rₖ·PTDFₖᵢ·Σⱼ PTDFₖⱼ·Pⱼ
Used as a reference implementation to validate the matrix-based `dc_loss_factors`.
"""
function _summation_dc_loss_factors(sys, data)
    Rs = Dict{Tuple{Int, Int}, Float64}()
    for line in get_components(PSY.Line, sys)
        Rs[PNM.get_arc_tuple(line)] = get_r(line)
    end
    for comp_type in (PSY.TapTransformer, PSY.Transformer2W)
        for line in get_components(comp_type, sys)
            Rs[PNM.get_arc_tuple(line)] = PSY.get_r(line)
        end
    end
    ptdf = data.power_network_matrix
    n_buses = length(get_components(PSY.ACBus, sys))
    injections = data.bus_active_power_injections .- data.bus_active_power_withdrawals
    injections .+= data.bus_hvdc_net_power
    loss_p = zeros(n_buses)
    for i in 1:n_buses
        loss_p[i] = sum(
            2 * Rs[arc_tuple] * ptdf[arc_tuple, i] *
            sum(ptdf[arc_tuple, j] * injections[j, 1] for j in 1:n_buses)
            for arc_tuple in keys(Rs)
        )
    end
    return loss_p, injections
end

@testset "compare summation DC loss factors to matrix" begin
    sys = build_system(PSITestSystems, "c_sys14"; add_forecasts = false)
    pf = PTDFDCPowerFlow(; time_steps = 1)
    data = PF.PowerFlowData(pf, sys)
    PF.solve_power_flow!(data)
    loss_p, injections = _summation_dc_loss_factors(sys, data)
    calculated_loss = PF.dc_loss_factors(data, injections)
    @test isapprox(calculated_loss[:, 1], loss_p; atol = 1e-10)
end

@testset "compare summation DC loss factors to vPTDF matrix" begin
    sys = build_system(PSITestSystems, "c_sys14"; add_forecasts = false)
    pf = vPTDFDCPowerFlow(; time_steps = 1)
    data = PF.PowerFlowData(pf, sys)
    PF.solve_power_flow!(data)
    loss_p, injections = _summation_dc_loss_factors(sys, data)
    calculated_loss = PF.dc_loss_factors(data, injections)
    @test isapprox(calculated_loss[:, 1], loss_p; atol = 1e-10)
end

@testset "DC loss factors via PTDFDCPowerFlow pipeline" begin
    sys = build_system(PSITestSystems, "c_sys14"; add_forecasts = false)

    # With calculate_loss_factors = true, loss_factors should be populated
    pf_lf = PTDFDCPowerFlow(; calculate_loss_factors = true)
    data_lf = PowerFlowData(pf_lf, sys)
    solve_power_flow!(data_lf)
    @test data_lf.loss_factors !== nothing
    n_buses = length(get_components(PSY.ACBus, sys))
    @test size(data_lf.loss_factors) == (n_buses, 1)

    # Compare against manual call to dc_loss_factors
    injections = data_lf.bus_active_power_injections .- data_lf.bus_active_power_withdrawals
    injections .+= data_lf.bus_hvdc_net_power
    manual_lf = PF.dc_loss_factors(data_lf, injections)
    @test isapprox(data_lf.loss_factors, manual_lf; atol = 1e-10)

    # Default (calculate_loss_factors = false) should leave loss_factors as nothing
    pf_no_lf = PTDFDCPowerFlow()
    data_no_lf = PowerFlowData(pf_no_lf, sys)
    solve_power_flow!(data_no_lf)
    @test isnothing(data_no_lf.loss_factors)
end

@testset "DC loss factors via vPTDFDCPowerFlow pipeline" begin
    sys = build_system(PSITestSystems, "c_sys14"; add_forecasts = false)

    # With calculate_loss_factors = true, loss_factors should be populated
    pf_lf = vPTDFDCPowerFlow(; calculate_loss_factors = true)
    data_lf = PowerFlowData(pf_lf, sys)
    solve_power_flow!(data_lf)
    @test data_lf.loss_factors !== nothing
    n_buses = length(get_components(PSY.ACBus, sys))
    @test size(data_lf.loss_factors) == (n_buses, 1)

    # Compare against PTDFDCPowerFlow result (should match)
    pf_ptdf = PTDFDCPowerFlow(; calculate_loss_factors = true)
    data_ptdf = PowerFlowData(pf_ptdf, sys)
    solve_power_flow!(data_ptdf)
    @test isapprox(data_lf.loss_factors, data_ptdf.loss_factors; atol = 1e-10)

    # Default (calculate_loss_factors = false) should leave loss_factors as nothing
    pf_no_lf = vPTDFDCPowerFlow()
    data_no_lf = PowerFlowData(pf_no_lf, sys)
    solve_power_flow!(data_no_lf)
    @test isnothing(data_no_lf.loss_factors)
end
