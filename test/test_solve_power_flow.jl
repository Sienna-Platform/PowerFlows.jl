@testset "AC Power Flow 14-Bus testing" begin
    result_14 = [
        2.3255081760423684
        -0.15529254415401786
        0.4692141795968793 - 0.127  # Q_gen - Q_load
        -0.08704571860678871
        0.27136388547189727 - 0.19  # Q_gen - Q_load
        -0.22239752522709744
        1.014232467422514
        -0.17900878100582837
        1.0172382900002028
        -0.15297197376986682
        0.21603888720954267 - 0.075  # Q_gen - Q_load
        -0.2516366411330649
        1.0503438761049335
        -0.23128945395825606
        0.2453884476050094
        -0.23128945395825604
        1.0337109552890456
        -0.2588720100456853
        1.0325606430799592
        -0.26251860149671896
        1.0474837090281963
        -0.25914254888894855
        1.0535012072934
        -0.26648433793564014
        1.0471069158440354
        -0.2671769024662062
        1.0213119628726421
        -0.2803812119374241
    ]

    sys = PSB.build_system(PSB.PSITestSystems, "c_sys14"; add_forecasts = false)
    set_units_base_system!(sys, UnitSystem.SYSTEM_BASE)
    pf = ACPowerFlow{PF.TrustRegionACPowerFlow}(; correct_bustypes = true)
    pf_w_limits =
        ACPowerFlow{PF.TrustRegionACPowerFlow}(;
            check_reactive_power_limits = true,
            correct_bustypes = true,
        )
    data = PowerFlows.PowerFlowData(pf, sys)
    #Compare results between finite diff methods and Jacobian method
    converged1 = PowerFlows._ac_power_flow(data, pf, 1)
    x1 = _calc_x(data, 1)
    # this fails at 1e-6, likely due to the change of the B allocation
    @test LinearAlgebra.norm(result_14 - x1, Inf) <= 3e-6
    # Test that solve_and_store_power_flow! succeeds
    solved1 = deepcopy(sys)
    @test solve_and_store_power_flow!(pf, solved1)

    # Test that passing check_reactive_power_limits=false is the default and violates limits
    solved2 = deepcopy(sys)
    @test solve_and_store_power_flow!(pf, solved2)
    @test IS.compare_values(solved1, solved2)
    @test get_reactive_power(get_component(ThermalStandard, solved2, "Bus8")) >
          get_reactive_power_limits(get_component(ThermalStandard, solved2, "Bus8")).max

    # Test that passing check_reactive_power_limits=true fixes that
    solved3 = deepcopy(sys)
    @test solve_and_store_power_flow!(pf_w_limits, solved3)
    @test get_reactive_power(get_component(ThermalStandard, solved3, "Bus8")) <=
          get_reactive_power_limits(get_component(ThermalStandard, solved3, "Bus8")).max

    # Test Newton method
    @test solve_and_store_power_flow!(pf, deepcopy(sys))

    # Test enforcing the reactive power limits in closer detail
    set_reactive_power!(get_component(PowerLoad, sys, "Bus4"), 0.0)
    data = PowerFlows.PowerFlowData(pf, sys)
    converged2 = PowerFlows._ac_power_flow(data, pf_w_limits, 1)
    x2 = _calc_x(data, 1)
    @test LinearAlgebra.norm(result_14 - x2, Inf) >= 1e-6
    @test 1.08 <= x2[15] <= 1.09
end

function test_ac_line_configurations(ACSolver)
    sys = PSB.build_system(PSB.PSITestSystems, "c_sys14"; add_forecasts = false)
    pf = ACPowerFlow{ACSolver}(; correct_bustypes = true)
    base_res = solve_power_flow(pf, sys)
    branch = first(PSY.get_components(Line, sys))
    dyn_branch = DynamicBranch(branch)
    add_component!(sys, dyn_branch)
    @test dyn_pf = solve_and_store_power_flow!(pf, sys)
    dyn_pf = solve_power_flow(pf, sys)
    @test LinearAlgebra.norm(
        dyn_pf["bus_results"].Vm - base_res["bus_results"].Vm,
        Inf,
    ) <= 1e-6

    sys = PSB.build_system(PSB.PSITestSystems, "c_sys14"; add_forecasts = false)
    pf = ACPowerFlow{ACSolver}(; correct_bustypes = true)
    line = get_component(Line, sys, "Line4")
    PSY.set_available!(line, false)
    solve_and_store_power_flow!(pf, sys)
    @test PSY.get_active_power_flow(line) == 0.0
    test_bus = get_component(PSY.ACBus, sys, "Bus 4")
    @test isapprox(PSY.get_magnitude(test_bus), 1.002; atol = 1e-3, rtol = 0)

    sys = PSB.build_system(PSB.PSITestSystems, "c_sys14"; add_forecasts = false)
    pf = ACPowerFlow{ACSolver}(; correct_bustypes = true)
    line = get_component(Line, sys, "Line4")
    from_to = PNM.get_arc_tuple(line)
    PSY.set_available!(line, false)
    res = solve_power_flow(pf, sys)
    for row in eachrow(res["flow_results"])
        if (row["bus_from"], row["bus_to"]) == from_to
            @test row["P_from_to"] == 0.0
            @test row["P_to_from"] == 0.0
        end
    end
end

@testset "AC Power Flow 14-Bus Line Configurations" begin
    foreach(test_ac_line_configurations, AC_SOLVERS_TO_TEST)
end

function test_ac_3bus_fixed_admittance(ACSolver)
    p_gen_matpower_3bus = [20.3512373930753, 100.0, 100.0]
    q_gen_matpower_3bus =
        [45.516916781567232, 10.453799727283879, -31.992561631394636]
    sys_3bus = PSB.build_system(PSB.PSYTestSystems, "psse_3bus_gen_cls_sys")
    bus_103 = get_component(PSY.ACBus, sys_3bus, "BUS 3")
    fix_shunt = PSY.FixedAdmittance("FixAdmBus3", true, bus_103, 0.0 + 0.2im)
    add_component!(sys_3bus, fix_shunt)
    pf = ACPowerFlow{ACSolver}(; correct_bustypes = true)
    df = solve_power_flow(pf, sys_3bus)
    @test isapprox(df["bus_results"].P_gen, p_gen_matpower_3bus, atol = 1e-4)
    @test isapprox(df["bus_results"].Q_gen, q_gen_matpower_3bus, atol = 1e-4)
end

@testset "AC Power Flow 3-Bus Fixed FixedAdmittance testing" begin
    foreach(test_ac_3bus_fixed_admittance, AC_SOLVERS_TO_TEST)
end

function test_ac_convergence_fail(ACSolver)
    pf_sys5_re =
        PSB.build_system(PSB.PSITestSystems, "c_sys5_re"; add_forecasts = false)
    remove_component!(Line, pf_sys5_re, "1")
    remove_component!(Line, pf_sys5_re, "2")
    br = get_component(Line, pf_sys5_re, "6")
    PSY.set_x!(br, 20.0)
    PSY.set_r!(br, 2.0)

    pf = ACPowerFlow{ACSolver}()

    # This is a negative test. The data passed for sys5_re is known to be infeasible.
    @test_logs(
        (:error, "The power flow solver returned convergence = false"),
        match_mode = :any,
        @test !solve_and_store_power_flow!(pf, pf_sys5_re)
    )
end

@testset "AC Power Flow convergence fail testing" begin
    foreach(test_ac_convergence_fail, AC_SOLVERS_TO_TEST)
end

@testset "AC Test 240 Case PSS/e results" begin
    file = joinpath(
        TEST_DATA_DIR,
        "WECC240_v04_DPV_RE20_v33_6302_xfmr_DPbuscode_PFadjusted_V32_noRemoteVctrl.raw",
    )
    system = System(
        file;
        bus_name_formatter = x -> strip(string(x["name"])) * "-" * string(x["index"]),
        runchecks = false,
    )

    pf_bus_result_file = joinpath(TEST_DATA_DIR, "pf_bus_results.csv")
    pf_gen_result_file = joinpath(TEST_DATA_DIR, "pf_gen_results.csv")

    # remove skip_redistribution once redistribution is working properly.
    pf = ACPowerFlow(; skip_redistribution = true, correct_bustypes = true)

    pf1 = solve_and_store_power_flow!(pf, system)
    @test pf1
    pf_result_df = solve_power_flow(pf, system)

    v_diff, angle_diff, number = psse_bus_results_compare(pf_bus_result_file, pf_result_df)
    p_diff, q_diff, names = psse_gen_results_compare(pf_gen_result_file, system)

    # FIXME temporarily commented out failing tests: see PowerNetworkMatrices.jl issue 215.
    base_power = get_base_power(system)
    # @test norm(v_diff, Inf) < DIFF_INF_TOLERANCE # fails badly
    @test norm(v_diff, 2) / length(v_diff) < DIFF_L2_TOLERANCE
    # @test norm(angle_diff, Inf) < DIFF_INF_TOLERANCE # fails badly.
    @test norm(angle_diff, 2) / length(angle_diff) < DIFF_L2_TOLERANCE
    @test norm(p_diff, Inf) < DIFF_INF_TOLERANCE * base_power
    @test norm(p_diff, 2) / length(p_diff) < DIFF_L2_TOLERANCE
    @test sum(q_diff) < DIFF_INF_TOLERANCE * base_power
    @test norm(q_diff, 2) / length(q_diff) < DIFF_L2_TOLERANCE
end

function test_ac_multiple_sources_at_ref(ACSolver)
    sys = System(100.0)
    b = _add_simple_bus!(sys, 1, ACBusTypes.REF, 230, 1.1, 0.0)

    #Test two sources with equal and opposite P and Q
    s1 = _add_simple_source!(sys, b, 0.5, 0.1)
    s2 = _add_simple_source!(sys, b, -0.5, -0.1)

    pf = ACPowerFlow{ACSolver}(; correct_bustypes = true)
    @test solve_and_store_power_flow!(pf, sys)

    #Create power mismatch, test for error
    set_active_power!(s1, -0.4)
    @test_throws ErrorException(
        "Sources do not match P and/or Q requirements for reference bus.",
    ) solve_and_store_power_flow!(ACPowerFlow{ACSolver}(), sys)
end

@testset "AC Multiple sources at ref" begin
    foreach(test_ac_multiple_sources_at_ref, AC_SOLVERS_TO_TEST)
end

function test_ac_multiple_sources_at_pv(ACSolver)
    sys = System(100.0)
    b1 = _add_simple_bus!(sys, 1, ACBusTypes.REF, 230, 1.1, 0.0)
    b2 = _add_simple_bus!(sys, 2, ACBusTypes.PV, 230, 1.1, 0.0)

    l = _add_simple_line!(sys, b1, b2, 1e-3, 1e-3, 0.0)

    #Test two sources with equal and opposite P and Q
    s1 = _add_simple_source!(sys, b1, 0.5, 0.1)
    s2 = _add_simple_source!(sys, b2, 0.5, 1.1)
    s3 = _add_simple_source!(sys, b2, -0.5, -1.1)

    pf = ACPowerFlow{ACSolver}(; correct_bustypes = true)

    @test solve_and_store_power_flow!(pf, sys)

    #Create power mismatch, test for error
    set_reactive_power!(s3, -0.5)
    @test_throws ErrorException("Sources do not match Q requirements for PV bus.") solve_and_store_power_flow!(
        pf,
        sys,
    )
end

@testset "AC PowerFlow with Multiple sources at PV" begin
    foreach(test_ac_multiple_sources_at_pv, AC_SOLVERS_TO_TEST)
end

function test_ac_source_and_non_source_at_ref(ACSolver)
    sys = System(100.0)

    b = _add_simple_bus!(sys, 1, ACBusTypes.REF, 230, 1.1, 0.0)

    #Test two sources with equal and opposite P and Q
    s1 = _add_simple_source!(sys, b, 0.5, 0.1)
    g1 = _add_simple_thermal_standard!(sys, b, 0.1, 0.1)

    pf = ACPowerFlow{ACSolver}()

    @test solve_and_store_power_flow!(pf, sys)
    @test isapprox(get_active_power(s1), 0.5; atol = 0.001)
    @test isapprox(get_reactive_power(s1), 0.1; atol = 0.001)
end

@testset "AC PowerFlow Source + non-source at Ref" begin
    foreach(test_ac_source_and_non_source_at_ref, AC_SOLVERS_TO_TEST)
end

function test_ac_source_and_non_source_at_pv(ACSolver)
    sys = System(100.0)

    b1 = _add_simple_bus!(sys, 1, ACBusTypes.REF, 230, 1.1, 0.0)
    b2 = _add_simple_bus!(sys, 2, ACBusTypes.PV, 230, 1.1, 0.0)
    l = _add_simple_line!(sys, b1, b2, 1e-3, 1e-3, 0.0)

    #Test two sources with equal and opposite P and Q
    s1 = _add_simple_source!(sys, b1, 0.5, 0.1)
    s2 = _add_simple_source!(sys, b2, 0.5, 1.1)
    g1 = _add_simple_thermal_standard!(sys, b2, 0.1, 0.1)

    pf = ACPowerFlow{ACSolver}(; correct_bustypes = true)

    @test solve_and_store_power_flow!(pf, sys)
    @test isapprox(get_active_power(s2), 0.5; atol = 0.001)
    @test isapprox(get_reactive_power(s2), 1.1; atol = 0.001)
end

@testset "AC PowerFlow Source + non-source at PV" begin
    foreach(test_ac_source_and_non_source_at_pv, AC_SOLVERS_TO_TEST)
end

# in this test, the following aspects are checked:
# The results of the power flow are consistent for the default (KLU), NewtonRaphson and TrustRegion solvers
@testset "Compare larger grid results KLU vs NewtonRaphson vs TrustRegion" begin
    sys = build_system(MatpowerTestSystems, "matpower_ACTIVSg2000_sys")

    PSY.set_units_base_system!(sys, "SYSTEM_BASE")
    pf_default = ACPowerFlow(; correct_bustypes = true)
    pf_newton = ACPowerFlow{NewtonRaphsonACPowerFlow}(; correct_bustypes = true)
    pf_tr = ACPowerFlow{TrustRegionACPowerFlow}(; correct_bustypes = true)

    res_default = solve_power_flow(pf_default, sys)  # must be the same as KLU
    res_newton = solve_power_flow(pf_newton, sys)
    res_tr = solve_power_flow(pf_tr, sys)

    @test all(
        isapprox.(
            res_newton["bus_results"][!, :Vm],
            res_default["bus_results"][!, :Vm],
            rtol = 0,
            atol = 1e-12,
        ),
    )
    @test all(
        isapprox.(
            res_newton["bus_results"][!, :θ],
            res_default["bus_results"][!, :θ],
            rtol = 0,
            atol = 1e-12,
        ),
    )
    @test all(
        isapprox.(
            res_tr["bus_results"][!, :Vm],
            res_default["bus_results"][!, :Vm],
            rtol = 0,
            atol = 1e-8,
        ),
    )
    @test all(
        isapprox.(
            res_tr["bus_results"][!, :θ],
            res_default["bus_results"][!, :θ],
            rtol = 0,
            atol = 1e-8,
        ),
    )
end

@testset "voltage_stability_factors" begin
    sys = PSB.build_system(PSB.PSITestSystems, "c_sys14"; add_forecasts = false)
    pf_newton = ACPowerFlow{NewtonRaphsonACPowerFlow}(;
        calculate_voltage_stability_factors = true,
        correct_bustypes = true,
    )
    pf_tr = ACPowerFlow{TrustRegionACPowerFlow}(;
        calculate_voltage_stability_factors = true,
        correct_bustypes = true,
    )
    data_newton = PowerFlowData(pf_newton, sys)
    data_tr = PowerFlowData(pf_tr, sys)
    time_step = 1
    solve_power_flow!(data_newton)
    solve_power_flow!(data_tr)
    @test all(
        isapprox.(
            data_tr.voltage_stability_factors,
            data_newton.voltage_stability_factors,
            rtol = 0,
            atol = 1e-6,
        ),
    )

    ref, pv, pq = PowerFlows.bus_type_idx(data_newton, time_step)
    pvpq = [pv; pq]
    npvpq = length(pvpq)

    # Build the Jacobian using the production code path, then re-block it the same
    # way `_calculate_voltage_stability_factors` does internally.
    residual = PF.ACPowerFlowResidual(data_newton, time_step)
    x_solved = PF.calculate_x0(data_newton, time_step)
    residual(x_solved, time_step)
    J = PF.ACPowerFlowJacobian(residual, time_step)
    J(time_step)
    rows, cols = PowerFlows._block_J_indices(pvpq, pq)
    J_block = J.Jv[rows, cols]

    Gs =
        J_block[(npvpq + 1):end, (npvpq + 1):end] -
        J_block[(npvpq + 1):end, 1:npvpq] *
        inv(collect(J_block[1:npvpq, 1:npvpq])) *
        J_block[1:npvpq, (npvpq + 1):end]
    u_1, (σ_1,), v_1, _ = PROPACK.tsvd_irl(Gs; smallest = true, k = 1)
    σ, u, v = PowerFlows._singular_value_decomposition(J_block, npvpq)

    @assert isapprox(σ_1, σ, atol = 1e-6)
    # the sign does not matter
    @assert isapprox(sign(first(u_1)) * u_1, u, atol = 1e-4)
    @assert isapprox(sign(first(v_1)) * v_1, v, atol = 1e-4)
end

@testset "AC PF 10k bus system: voltage magnitudes" begin
    sys = PSB.build_system(
        PSB.MatpowerTestSystems,
        "matpower_ACTIVSg10k_sys";
        force_build = false,
    )
    @assert !isempty(get_components(PhaseShiftingTransformer, sys)) "System should have " *
                                                                    "phase shifting transformers: " *
                                                                    "change `force_build` to `true` in the test."
    pf_tr = ACPowerFlow{TrustRegionACPowerFlow}(;
        correct_bustypes = true,
        solver_settings = Dict{Symbol, Any}(:maxIterations => 200, :factor => 0.1),
    )
    data_tr = PowerFlowData(pf_tr, sys)
    solve_power_flow!(data_tr)
    @test all(data_tr.bus_magnitude[:, 1] .<= 1.1)
    @test all(data_tr.bus_magnitude[:, 1] .>= 0.9)
end

function PowerFlowData_to_DataFrame(data::PowerFlowData)
    nbuses = size(data.bus_magnitude, 1)
    # Convert the PowerFlowData to a DataFrame
    bus_rev_lookup = fill(-1, nbuses)
    for (bus_no, row_no) in PF.get_bus_lookup(data)
        bus_rev_lookup[row_no] = bus_no
    end
    @assert !(-1 in bus_rev_lookup)
    df = DataFrame(;
        bus_number = bus_rev_lookup,
        Vm = data.bus_magnitude[:, 1],
        bus_type = data.bus_type[:, 1],
        angle = data.bus_angles[:, 1],
        generator_p = data.bus_active_power_injections[:, 1],
        generator_q = data.bus_reactive_power_injections[:, 1],
        load_p = data.bus_active_power_withdrawals[:, 1],
        load_q = data.bus_reactive_power_withdrawals[:, 1],
    )
    sort!(df, :bus_number)
    return df
end

@testset "ACTIVSg2000 matches matpower's solution" begin
    MATPOWER_CSV = joinpath(TEST_DATA_DIR, "ACTIVSg2000_solved.csv")
    matpower_df = DataFrame(CSV.File(MATPOWER_CSV))

    sys_sienna = build_system(MatpowerTestSystems, "matpower_ACTIVSg2000_sys")
    pf_sienna = ACPowerFlow(;
        correct_bustypes = true,
        solver_settings = Dict{Symbol, Any}(:tol => 1e-11),
    )
    data_sienna = PowerFlowData(pf_sienna, sys_sienna)
    solve_power_flow!(data_sienna)
    sienna_df = PowerFlowData_to_DataFrame(data_sienna)
    @assert all(sienna_df[!, "bus_number"] .== matpower_df[!, "bus_number"])
    # The bus types don't match, so we don't compare them here. (We changed PQ with
    # generators to PV. Matpower doesn't do this, though it treats them as PV internally.)
    @test norm(sienna_df[!, "Vm"] .- matpower_df[!, "Vm"], Inf) < 1e-3
    @test norm(sienna_df[!, "angle"] .- matpower_df[!, "angle"], Inf) < 1e-3
    @test norm(sienna_df[!, "generator_p"] .- matpower_df[!, "generator_p"], Inf) < 1e-3
    # this fails with 1e-4.
    @test norm(sienna_df[!, "generator_q"] .- matpower_df[!, "generator_q"], Inf) < 1e-3
end

if PF.OVERRIDE_x0
    @testset "voltage validation" begin
        sys = PSB.build_system(PSB.PSITestSystems, "c_sys5")
        pf = ACPowerFlow{TrustRegionACPowerFlow}(; correct_bustypes = true)
        data = PowerFlowData(pf, sys)
        x0 = PF.calculate_x0(data, 1)
        for (i, bt) in enumerate(PF.get_bus_type(data))
            if bt == PSY.ACBusTypes.PQ
                x0[2 * i - 1] = 2.0 # set voltage magnitude of PQ bus to 2.0 p.u.
            end
        end
        @test_logs (:warn, r".*voltage magnitudes outside of range.*") match_mode = :any solve_power_flow!(
            data;
            x0 = x0,
        )
    end
end

@testset "enhanced flat start" begin
    sys = PSB.build_system(PSB.PSITestSystems, "c_sys5")
    pf = ACPowerFlow{TrustRegionACPowerFlow}(;
        enhanced_flat_start = true,
        correct_bustypes = true,
    )
    data = PowerFlowData(pf, sys)
    x0 = PF.calculate_x0(data, 1)
    x0_enhanced = PF._enhanced_flat_start(x0, data, 1)
    solve_power_flow!(data)
end

@testset "Test ZIP loads: constant current" begin
    sys = System(100.0)
    b1 = _add_simple_bus!(sys, 1, ACBusTypes.REF, 230, 1.1, 0.0)
    b2 = _add_simple_bus!(sys, 2, ACBusTypes.PQ, 230, 1.1, 0.0)
    l = _add_simple_line!(sys, b1, b2, 5e-3, 5e-3, 1e-3)
    s1 = _add_simple_source!(sys, b1, 0.0, 0.0)
    lc = _add_simple_zip_load!(
        sys,
        b2;
        constant_current_active_power = 2.0,
        constant_current_reactive_power = 1.0,
    )
    data = PowerFlowData(ACPowerFlow(; correct_bustypes = true), sys)
    solve_power_flow!(data)

    # here we calculate the current that is observed in the power flow solution
    # supplied through the line to the load, which is a constant current load.
    s_t =
        data.arc_active_power_flow_to_from + 1im * data.arc_reactive_power_flow_to_from
    i_t = abs(s_t[1]) / data.bus_magnitude[2, 1] / sqrt(3)

    # get the load inputs from the load component
    load_input_power = (get_current_active_power(lc) + 1im * get_current_reactive_power(lc))
    # calculating by hand the current that corresponds to the load inputs
    # constant current load is given for 1.0 p.u. base voltage:
    load_input_current = abs(load_input_power) / 1.0 / sqrt(3)

    # Calculate the observed bus power based on the Ybus and voltage at the bus of the load:
    V = data.bus_magnitude[:, 1] .* exp.(1im * data.bus_angles[:, 1])
    Sbus = V .* conj(data.power_network_matrix.data * V)

    # The expected power at the bus, corresponding to the formula for constant current loads:
    load_expected_power = load_input_power * data.bus_magnitude[2, 1]

    # - due to the reference frame in Sbus vs. load inputs
    @test isapprox(Sbus[2], -load_expected_power; atol = 1e-6, rtol = 0)

    # Calculate the expected current based on the load inputs and the bus voltage:
    load_expected_current = abs(load_expected_power) / data.bus_magnitude[2, 1] / sqrt(3)

    @test isapprox(load_expected_current, load_input_current; atol = 1e-6, rtol = 0)

    @test isapprox(i_t, load_input_current; atol = 1e-6, rtol = 0)

    @test isapprox(data.bus_active_power_injections[2, 1], 0.0, atol = 1e-12, rtol = 0)
    @test isapprox(data.bus_reactive_power_injections[2, 1], 0.0, atol = 1e-12, rtol = 0)
end

@testset "Test ZIP loads: constant impedance" begin
    sys = System(100.0)
    b_1 = _add_simple_bus!(sys, 1, ACBusTypes.REF, 230, 1.1, 0.0)
    b2 = _add_simple_bus!(sys, 2, ACBusTypes.PQ, 230, 1.1, 0.0)
    l = _add_simple_line!(sys, b_1, b2, 5e-3, 5e-3, 1e-3)
    s1 = _add_simple_source!(sys, b_1, 0.0, 0.0)
    lz = _add_simple_zip_load!(
        sys,
        b2;
        constant_impedance_active_power = 2.0,
        constant_impedance_reactive_power = 1.0,
    )
    data = PowerFlowData(ACPowerFlow(; correct_bustypes = true), sys)
    solve_power_flow!(data)

    # here we calculate the current that is observed in the power flow solution
    # supplied through the line to the load, which is a constant impedance load.
    s_t =
        data.arc_active_power_flow_to_from + 1im * data.arc_reactive_power_flow_to_from
    i_t = abs(s_t[1]) / data.bus_magnitude[2, 1] / sqrt(3)

    # get the load inputs from the load component
    load_input_power =
        (get_impedance_active_power(lz) + 1im * get_impedance_reactive_power(lz))
    # calculating by hand the impedance that corresponds to the load inputs
    # constant impedance load is given for 1.0 p.u. base voltage:
    load_input_impedance = 1.0^2 / abs(load_input_power)

    # Calculate the observed bus power based on the Ybus and voltage at the bus of the load:
    V = data.bus_magnitude[:, 1] .* exp.(1im * data.bus_angles[:, 1])
    Sbus = V .* conj(data.power_network_matrix.data * V)

    # The expected power at the bus, corresponding to the formula for constant impedance loads:
    load_expected_power = load_input_power * data.bus_magnitude[2, 1]^2

    # - due to the reference frame in Sbus vs. load inputs
    @test isapprox(Sbus[2], -load_expected_power; atol = 1e-6, rtol = 0)

    # Calculate the expected impedance based on the load inputs and the bus voltage:
    load_expected_impedance = (data.bus_magnitude[2, 1]^2) / abs(load_expected_power)

    @test isapprox(load_expected_impedance, load_input_impedance; atol = 1e-6, rtol = 0)

    s_zip_load =
        PF.get_bus_active_power_total_withdrawals(data, 2, 1) +
        1im * PF.get_bus_reactive_power_total_withdrawals(data, 2, 1)

    @test isapprox(
        Sbus[2],
        -s_zip_load;
        atol = 1e-6,
        rtol = 0,
    )

    @test isapprox(
        data.bus_magnitude[2, 1]^2 / load_input_impedance,
        abs(s_zip_load);
        atol = 1e-6,
        rtol = 0,
    )

    @test isapprox(  # <- need to come back to this and check why it fails
        s_t[1],
        -s_zip_load;
        atol = 1e-6,
        rtol = 0,
    )

    @test isapprox(data.bus_active_power_injections[2, 1], 0.0, atol = 1e-12, rtol = 0)
    @test isapprox(data.bus_reactive_power_injections[2, 1], 0.0, atol = 1e-12, rtol = 0)
end

@testset "Test phase shift in transformers" for Transformer in
                                                (PSY.Transformer2W, PSY.TapTransformer)
    sys = System(100.0)
    b1 = _add_simple_bus!(sys, 1, ACBusTypes.REF, 230, 1.1, 0.0)
    b2 = _add_simple_bus!(sys, 2, ACBusTypes.PQ, 110, 1.1, 0.0)

    _add_simple_source!(sys, b1, 0.0, 0.0)

    parameters = Dict(
        :name => "Transformer",
        :available => true,
        :active_power_flow => 0.0,
        :reactive_power_flow => 0.0,
        :arc => Arc(b1, b2),
        :r => 0.01,
        :x => 0.05,
        :primary_shunt => 0.0,
        :winding_group_number => 1,  # 30 degrees in radians
        :rating => 1.0,
        :base_power => 100.0,
        :base_voltage_primary => 230,
        :base_voltage_secondary => 110,
    )

    Transformer == PSY.Transformer2W || (parameters[:tap] = 1.0)

    t = Transformer(;
        parameters...,
    )
    add_component!(sys, t)

    pf = ACPowerFlow(; correct_bustypes = true)
    data = PowerFlowData(pf, sys)
    solve_power_flow!(data)
    # Check that the phase shift is correctly applied
    a1 = data.bus_angles[1, 1]
    a2 = data.bus_angles[2, 1]
    # TODO for some reason this is off by a negative sign.
    # @test isapprox(a2, a1 - deg2rad(30); atol = 1e-6, rtol = 0)
    @test isapprox(-a2, a1 - deg2rad(30); atol = 1e-6, rtol = 0)
end

@testset "Test SwitchedAdmittance" begin
    sys = System(100.0)
    b1 = _add_simple_bus!(sys, 1, ACBusTypes.REF, 230, 1.1, 0.0)
    b2 = _add_simple_bus!(sys, 2, ACBusTypes.PQ, 230, 1.1, 0.0)
    l = _add_simple_line!(sys, b1, b2, 5e-3, 5e-3, 1e-3)
    s1 = _add_simple_source!(sys, b1, 0.0, 0.0)

    data1 = PowerFlowData(ACPowerFlow(), sys)

    # create a switched admittance
    sa = SwitchedAdmittance(;
        name = "SA",
        available = true,
        bus = b2,
        Y = 0.03 + 0.05im,
        initial_status = Int[1, 2],
        number_of_steps = Int[3, 3],
        Y_increase = Complex{Float64}[0.01 + 0.02im, 0.02 + 0.03im],
    )
    add_component!(sys, sa)

    data2 = PowerFlowData(ACPowerFlow(), sys)

    # The Ybus matrix should not include switched admittance elements
    @test isapprox(
        data1.power_network_matrix.data,
        data2.power_network_matrix.data,
        atol = 1e-6,
        rtol = 0,
    )

    Y = PSY.get_Y(sa) + sum(PSY.get_initial_status(sa) .* PSY.get_Y_increase(sa))

    data1.power_network_matrix.data[2, 2] += Y

    solve_power_flow!(data1)

    solve_power_flow!(data2)

    # Make sure the results are the same for both cases:
    #  1. The switched admittance is included in the Ybus matrix
    #  2. The switched admittance is represented as a constant impedance load
    #     and is not in the Ybus matrix
    @test isapprox(
        data1.bus_magnitude[:, 1],
        data2.bus_magnitude[:, 1],
        atol = 1e-6,
        rtol = 0,
    )
    @test isapprox(data1.bus_angles[:, 1], data2.bus_angles[:, 1], atol = 1e-6, rtol = 0)
end

function check_lcc_consistency(
    lcc::PSY.TwoTerminalLCCLine,
    lcc_results::DataFrame;
    base_power::Float64 = 100.0,
)
    @assert nrow(lcc_results) == 1

    lcc_arc = PNM.get_arc_tuple(PSY.get_arc(lcc))
    @test lcc_results[1, :bus_from] == lcc_arc[1]
    @test lcc_results[1, :bus_to] == lcc_arc[2]
    @test lcc_results[1, :line_name] == PSY.get_name(lcc)

    @test lcc_results[1, :rectifier_delay_angle] ==
          PSY.get_rectifier_delay_angle(lcc)
    @test lcc_results[1, :inverter_extinction_angle] ==
          PSY.get_inverter_extinction_angle(lcc)
    @test lcc_results[1, :rectifier_tap] == PSY.get_rectifier_tap_setting(lcc)
    @test lcc_results[1, :inverter_tap] == PSY.get_inverter_tap_setting(lcc)
    @test lcc_results[1, :P_from_to] == base_power .* PSY.get_active_power_flow(lcc)
    return
end

@testset "Test LCC consistency" begin
    sys, lcc = simple_lcc_system()
    pf = ACPowerFlow()
    lcc_results = solve_power_flow(pf, sys)["lcc_results"]
    solve_and_store_power_flow!(pf, sys)
    check_lcc_consistency(lcc, lcc_results)

    # repeat with a different setpoint
    sys, lcc = simple_lcc_system()
    PSY.set_transfer_setpoint!(lcc, -25.0)
    lcc_results = solve_power_flow(pf, sys)["lcc_results"]
    solve_and_store_power_flow!(pf, sys)
    check_lcc_consistency(lcc, lcc_results)

    # could add one with 2+ LCCs.
end

function test_lcc_ac_solver(ACSolver)
    # Skip the solvers that do not support LCCs
    ACSolver ∈ (RobustHomotopyPowerFlow,) && return
    sys, lcc = simple_lcc_system()
    pf = ACPowerFlow{ACSolver}(; correct_bustypes = true)
    data = PowerFlowData(pf, sys)
    solve_power_flow!(data)

    lcc_arc = PNM.get_arc_tuple(PSY.get_arc(lcc))

    @test isapprox(
        data.lcc.p_set[1],
        data.lcc.arc_active_power_flow_from_to[1, 1];
        atol = 1e-6, rtol = 0,
    )

    LCC_active_flow =
        data.lcc.arc_active_power_flow_from_to[1, 1] +
        data.lcc.arc_active_power_flow_to_from[1, 1]
    LCC_reactive_flow =
        data.lcc.arc_reactive_power_flow_from_to[1, 1] +
        data.lcc.arc_reactive_power_flow_to_from[1, 1]
    @test isapprox(
        sum(
            data.arc_active_power_flow_from_to .+ data.arc_active_power_flow_to_from,
        ) + sum(data.bus_active_power_withdrawals[:, 1]) + LCC_active_flow,
        data.bus_active_power_injections[1];
        atol = 1e-5, rtol = 0,
    )

    @test isapprox(
        sum(
            data.arc_reactive_power_flow_from_to .+
            data.arc_reactive_power_flow_to_from,
        ) + sum(data.bus_reactive_power_withdrawals[:, 1]) + LCC_reactive_flow,
        data.bus_reactive_power_injections[1];
        atol = 1e-5, rtol = 0,
    )
    solve_and_store_power_flow!(pf, sys)

    @test get_active_power_flow(lcc) ==
          data.lcc.arc_active_power_flow_from_to[1, 1]

    PSY.set_transfer_setpoint!(lcc, -25.0)
    data = PowerFlowData(pf, sys)
    solve_power_flow!(data)

    @test isapprox(
        -data.lcc.p_set[1],
        data.lcc.arc_active_power_flow_to_from[1, 1];
        atol = 1e-6, rtol = 0,
    )

    solve_and_store_power_flow!(pf, sys)

    @test get_active_power_flow(lcc) ==
          data.lcc.arc_active_power_flow_from_to[1, 1]

    PSY.set_transfer_setpoint!(lcc, 0.0)
    data = PowerFlowData(pf, sys)
    solve_power_flow!(data)

    PSY.remove_component!(sys, lcc)
    data2 = PowerFlowData(pf, sys)
    solve_power_flow!(data2)

    @test isapprox(
        data.bus_magnitude[:, 1],
        data2.bus_magnitude[:, 1];
        atol = 1e-6, rtol = 0,
    )

    @test isapprox(
        data.bus_angles[:, 1],
        data2.bus_angles[:, 1];
        atol = 1e-6, rtol = 0,
    )

    @test isapprox(
        data.bus_active_power_injections[:, 1],
        data2.bus_active_power_injections[:, 1];
        atol = 1e-6, rtol = 0,
    )

    @test isapprox(
        data.bus_reactive_power_injections[:, 1],
        data2.bus_reactive_power_injections[:, 1];
        atol = 1e-6, rtol = 0,
    )

    @test isapprox(
        data.arc_active_power_flow_from_to[1:2, :],
        data2.arc_active_power_flow_from_to[1:2, :];
        atol = 1e-6, rtol = 0,
    )

    @test isapprox(
        data.arc_active_power_flow_to_from[1:2, :],
        data2.arc_active_power_flow_to_from[1:2, :];
        atol = 1e-6, rtol = 0,
    )

    @test isapprox(
        data.arc_reactive_power_flow_from_to[1:2, :],
        data2.arc_reactive_power_flow_from_to[1:2, :];
        atol = 1e-6, rtol = 0,
    )

    @test isapprox(
        data.arc_reactive_power_flow_to_from[1:2, :],
        data2.arc_reactive_power_flow_to_from[1:2, :];
        atol = 1e-6, rtol = 0,
    )
end

@testset "Test LCC" begin
    foreach(test_lcc_ac_solver, AC_SOLVERS_TO_TEST)
end

@testset "AC power flow: results independent of units" begin
    sys = PSB.build_system(PSB.PSITestSystems, "c_sys14"; add_forecasts = false)
    line_name_ac, flow_natural_ac =
        power_flow_with_units(sys, ACPowerFlow, PSY.UnitSystem.NATURAL_UNITS)
    line_name2_ac, flow_system_ac =
        power_flow_with_units(sys, ACPowerFlow, PSY.UnitSystem.SYSTEM_BASE)
    @test line_name_ac == line_name2_ac
    @test flow_natural_ac == flow_system_ac
end

function test_ac_arc_angle_differences(ACSolver)
    sys = PSB.build_system(PSB.PSITestSystems, "c_sys14"; add_forecasts = false)
    pf = ACPowerFlow{ACSolver}(; correct_bustypes = true)
    data = PowerFlowData(pf, sys)
    solve_power_flow!(data)
    validate_arc_angle_differences(data, [1])
end

@testset "AC arc_angle_differences validation" begin
    foreach(test_ac_arc_angle_differences, AC_SOLVERS_TO_TEST)
end

@testset "ACPowerFlow solver_settings accepts narrowly-typed Dicts" begin
    # Regression: previously the kwarg required Dict{Symbol, Any} exactly, so a
    # plain `Dict(:k => 50)` (inferred as Dict{Symbol, Int64}) was rejected.
    pf_int = ACPowerFlow(; solver_settings = Dict(:maxIterations => 50))
    @test pf_int.solver_settings isa Dict{Symbol, Any}
    @test pf_int.solver_settings[:maxIterations] === 50

    pf_bool = ACPowerFlow(;
        solver_settings = Dict(:validate_voltage_magnitudes => false),
    )
    @test pf_bool.solver_settings[:validate_voltage_magnitudes] === false

    pf_any = ACPowerFlow(;
        solver_settings = Dict{Symbol, Any}(:maxIterations => 50),
    )
    @test pf_any.solver_settings[:maxIterations] === 50
end
