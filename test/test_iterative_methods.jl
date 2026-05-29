@testset "Iwamoto multiplier: exact quadratic minimizer" begin
    # The optimal multiplier must minimize the exact quadratic mismatch model
    # g(μ) = ‖f₀ + μ·b + μ²·a‖² over μ ∈ [0, 1] (b = J·Δx, a = true quadratic term).
    g_model(f0, b, a, μ) = sum(abs2, f0 .+ μ .* b .+ (μ * μ) .* a)
    function brute(f0, b, a)
        best_μ, best_g = 0.0, g_model(f0, b, a, 0.0)
        for k in 0:200_000
            μ = k / 200_000
            gv = g_model(f0, b, a, μ)
            gv < best_g && ((best_μ, best_g) = (μ, gv))
        end
        return best_μ
    end
    cases = (
        ([1.0, -2.0, 0.5], [-0.3, 1.2, -0.7], [0.4, -0.1, 0.9]),
        ([2.0, 1.0], [-2.0, -1.0], [0.5, -0.3]),
        ([0.1, 0.2, -0.4, 0.3], [1.0, -1.0, 0.2, 0.0], [-0.5, 0.5, -0.5, 0.5]),
        ([3.0], [-3.0], [1.0]),
    )
    for (f0, b, a) in cases
        c_fb, c_bb = dot(f0, b), dot(b, b)
        c_fa, c_ba, c_aa = dot(f0, a), dot(b, a), dot(a, a)
        μ = PF._iwamoto_multiplier(2c_fb, c_bb + 2c_fa, 2c_ba, c_aa)
        @test 0.0 <= μ <= 1.0
        # The analytic minimizer must be no worse than a fine brute-force grid.
        @test g_model(f0, b, a, μ) <= g_model(f0, b, a, brute(f0, b, a)) + 1e-6
    end
    # Newton-step special case (b = −f₀): 3-arg convenience == 4-arg general form.
    f0 = [1.0, -0.5, 2.0, 0.3]
    a = [0.2, 0.7, -0.4, 1.1]
    b = -f0
    g0, g1, g2 = dot(f0, f0), dot(f0, a), dot(a, a)
    @test PF._iwamoto_multiplier(g0, g1, g2) == PF._iwamoto_multiplier(
        2 * dot(f0, b), dot(b, b) + 2 * dot(f0, a), 2 * dot(b, a), dot(a, a))
end

@testset "NewtonRaphsonACPowerFlow kwargs" begin
    # test NR kwargs.
    sys = PSB.build_system(PSB.PSITestSystems, "c_sys5")
    nr_pf = ACPowerFlow{NewtonRaphsonACPowerFlow}(;
        solver_settings = Dict{Symbol, Any}(
            :maxIterations => 50,
            :tol => 1e-10,
            :refinement_threshold => 0.01,
            :refinement_eps => 1e-7,
        ))
    @test_logs (:info, r".*NewtonRaphsonACPowerFlow solver converged"
    ) match_mode = :any PF.solve_power_flow(nr_pf, sys)
end

@testset "TrustRegionACPowerFlow kwargs" begin
    # test trust region kwargs.
    sys = PSB.build_system(PSB.PSITestSystems, "c_sys5")
    tr_pf = ACPowerFlow{TrustRegionACPowerFlow}(;
        solver_settings = Dict{Symbol, Any}(
            :eta => 1e-5,
            :tol => 1e-10,
            :factor => 1.1,
            :maxIterations => 50,
        ))
    @test_logs (:info, r".*TrustRegionACPowerFlow solver converged"
    ) match_mode = :any PF.solve_power_flow(tr_pf, sys)
end

function bad_x0!(sys::PSY.System)
    for comp in get_components(PSY.PowerLoad, sys)
        set_angle!(PSY.get_bus(comp), 1.0)
    end
end

@testset "TrustRegionACPowerFlow behavior" begin
    sys = PSB.build_system(PSB.PSITestSystems, "c_sys5")

    # Small trust region size => Cauchy or dogleg step
    tr_pf_small = ACPowerFlow{TrustRegionACPowerFlow}(;
        solver_settings = Dict{Symbol, Any}(:factor => 0.01, :maxIterations => 1))
    @test_logs (:debug, r"(Dogleg step selected|Cauchy step selected)") match_mode = :any min_level =
        Logging.Debug PF.solve_power_flow(tr_pf_small, sys)

    # Large trust region size => Newton-Raphson step
    tr_pf_large = ACPowerFlow{TrustRegionACPowerFlow}(;
        solver_settings = Dict{Symbol, Any}(:factor => 10.0, :maxIterations => 1))
    @test_logs (:debug, r"Newton-Raphson step selected.*") match_mode = :any min_level =
        Logging.Debug PF.solve_power_flow(tr_pf_large, sys)

    # Large eta => step rejected, Iwamoto fallback attempted (default on)
    tr_pf_large_eta = ACPowerFlow{TrustRegionACPowerFlow}(;
        solver_settings = Dict{Symbol, Any}(:eta => 2.0, :maxIterations => 1))
    @test_logs (:debug, r"Iwamoto fallback.*") match_mode = :any min_level = Logging.Debug PF.solve_power_flow(
        tr_pf_large_eta,
        sys,
    )

    # Large eta with iwamoto_fallback disabled => plain rejection
    tr_pf_no_iwamoto = ACPowerFlow{TrustRegionACPowerFlow}(;
        solver_settings = Dict{Symbol, Any}(
            :eta => 2.0, :maxIterations => 1, :iwamoto_fallback => false))
    @test_logs (:debug, r"Step rejected.*") match_mode = :any min_level = Logging.Debug PF.solve_power_flow(
        tr_pf_no_iwamoto,
        sys,
    )

    # Small eta => step accepted
    tr_pf_small_eta = ACPowerFlow{TrustRegionACPowerFlow}(;
        solver_settings = Dict{Symbol, Any}(:eta => 1e-6, :maxIterations => 1))
    @test_logs (:debug, r"Step accepted.*") match_mode = :any min_level = Logging.Debug PF.solve_power_flow(
        tr_pf_small_eta,
        sys,
    )
end

@testset "Iwamoto step control convergence" begin
    sys = PSB.build_system(PSB.PSITestSystems, "c_sys5")
    iwamoto_pf = ACPowerFlow{NewtonRaphsonACPowerFlow}(;
        solver_settings = Dict{Symbol, Any}(:iwamoto => true))
    @test_logs (:info, r".*NewtonRaphsonACPowerFlow solver converged"
    ) match_mode = :any PF.solve_power_flow(iwamoto_pf, sys)
end

@testset "Iwamoto step control kwargs" begin
    sys = PSB.build_system(PSB.PSITestSystems, "c_sys5")
    iwamoto_pf = ACPowerFlow{NewtonRaphsonACPowerFlow}(;
        solver_settings = Dict{Symbol, Any}(
            :iwamoto => true,
            :maxIterations => 50,
            :tol => 1e-10,
        ))
    @test_logs (:info, r".*NewtonRaphsonACPowerFlow solver converged"
    ) match_mode = :any PF.solve_power_flow(iwamoto_pf, sys)
end

@testset "Iwamoto result equivalence with plain NR" begin
    sys = PSB.build_system(PSB.PSITestSystems, "c_sys5")
    nr_pf = ACPowerFlow{NewtonRaphsonACPowerFlow}()
    iwamoto_pf = ACPowerFlow{NewtonRaphsonACPowerFlow}(;
        solver_settings = Dict{Symbol, Any}(:iwamoto => true))
    nr_result = PF.solve_power_flow(nr_pf, sys)
    sys2 = PSB.build_system(PSB.PSITestSystems, "c_sys5")
    iwamoto_result = PF.solve_power_flow(iwamoto_pf, sys2)
    # Both should converge to the same solution
    for (key, nr_df) in nr_result
        iw_df = iwamoto_result[key]
        for col in names(nr_df)
            if eltype(nr_df[!, col]) <: Number
                @test isapprox(nr_df[!, col], iw_df[!, col]; atol = 1e-6)
            end
        end
    end
end

@testset "Iwamoto convergence with bad initial guess" begin
    # Bad initial guess — Iwamoto should still converge
    sys = PSB.build_system(PSB.PSITestSystems, "c_sys5")
    bad_x0!(sys)
    iwamoto_pf = ACPowerFlow{NewtonRaphsonACPowerFlow}(;
        enhanced_flat_start = false,
        solver_settings = Dict{Symbol, Any}(:iwamoto => true))
    @test_logs (:info, r".*NewtonRaphsonACPowerFlow solver converged"
    ) match_mode = :any PF.solve_power_flow(iwamoto_pf, sys)
end

@testset "Iwamoto on larger system (RTS_GMLC)" begin
    sys = PSB.build_system(PSB.PSISystems, "RTS_GMLC_DA_sys")
    PSY.set_units_base_system!(sys, PSY.UnitSystem.SYSTEM_BASE)
    nr_pf = ACPowerFlow{NewtonRaphsonACPowerFlow}(; correct_bustypes = true)
    iwamoto_pf = ACPowerFlow{NewtonRaphsonACPowerFlow}(;
        correct_bustypes = true,
        solver_settings = Dict{Symbol, Any}(:iwamoto => true))
    nr_result = PF.solve_power_flow(nr_pf, sys)
    sys2 = PSB.build_system(PSB.PSISystems, "RTS_GMLC_DA_sys")
    PSY.set_units_base_system!(sys2, PSY.UnitSystem.SYSTEM_BASE)
    iwamoto_result = PF.solve_power_flow(iwamoto_pf, sys2)
    for (key, nr_df) in nr_result
        iw_df = iwamoto_result[key]
        for col in names(nr_df)
            if eltype(nr_df[!, col]) <: Number
                @test isapprox(nr_df[!, col], iw_df[!, col]; atol = 1e-6)
            end
        end
    end
end

@testset "Singular Jacobian triggers backend-agnostic fallback" begin
    # Zeroing voltage magnitudes makes the Newton Jacobian singular. KLU signals this
    # by throwing SingularException; AppleAccelerate and MKLPardiso instead return a
    # finite garbage solution. The backend-agnostic residual guard in `_set_Δx_nr!`
    # must route every backend through the regularized fallback, which emits the
    # "Jacobian is singular" warning. Pre-fix, the AppleAccelerate default skipped the
    # fallback silently, so this test asserts the warning is actually produced.
    sys = PSB.build_system(PSB.PSITestSystems, "c_sys5")
    pf = ACPowerFlow{NewtonRaphsonACPowerFlow}(;
        enhanced_flat_start = false,
        solver_settings = Dict{Symbol, Any}(:maxIterations => 3),
    )
    data = PowerFlowData(pf, sys)
    data.bus_magnitude .= 0.0
    @test_logs(
        (:warn, r"Jacobian is singular"),
        match_mode = :any,
        solve_power_flow!(data),
    )
end

@testset "Iwamoto early termination on stagnation" begin
    # Sabotage voltage magnitudes so that every Newton step worsens the residual,
    # triggering consecutive reverts and the early-termination break.
    sys = PSB.build_system(PSB.PSITestSystems, "c_sys5")
    iwamoto_pf = ACPowerFlow{NewtonRaphsonACPowerFlow}(;
        enhanced_flat_start = false,
        solver_settings = Dict{Symbol, Any}(
            :iwamoto => true,
            :maxIterations => 20,
        ))
    data = PowerFlowData(iwamoto_pf, sys)
    # Set all voltage magnitudes to zero so the Jacobian is singular
    # and every proposed step increases the residual.
    data.bus_magnitude .= 0.0
    # Solver should fail to converge, and terminate early (not exhaust maxIterations).
    @test_logs(
        (:error, r".*solver failed to converge"),
        match_mode = :any,
        @test !solve_power_flow!(data)
    )
end

@testset "Iwamoto multiplier root-finding" begin
    # Verify that _iwamoto_multiplier recovers the global minimizer of the
    # classical (exact-Newton-step) Iwamoto objective
    # g(μ) = (1-μ)²g₀ + 2μ²(1-μ)g₁ + μ⁴g₂ on [0, 1] by comparing against
    # a brute-force grid search.
    g_classic(μ, g0, g1, g2) = (1 - μ)^2 * g0 + 2 * μ^2 * (1 - μ) * g1 + μ^4 * g2
    grid = range(0.0, 1.0; length = 10001)

    test_cases = [
        # (g0, g1, g2, description)
        # Three distinct real roots (trigonometric branch, Δ > 0).
        (1.0, 0.5, 2.0, "typical damping case"),
        (4.0, 1.0, 3.0, "three real roots, moderate values"),
        # One real root (Cardano branch, Δ < 0).
        (1.0, -0.5, 0.5, "negative cross-term, one real root"),
        (1.0, 0.0, 1.0, "orthogonal residuals"),
        # Repeated roots (Δ ≈ 0).
        (1.0, 1.0, 1.0, "all gram scalars equal"),
        # Scaled values (should give same μ as unscaled).
        (100.0, 50.0, 200.0, "scaled version of typical case"),
        # Near-full-step optimality.
        (1.0, 0.99, 1.01, "near-unit multiplier"),
        # Large disparity.
        (1.0, 0.1, 100.0, "large g2, strong damping expected"),
    ]

    for (g0, g1, g2, desc) in test_cases
        μ = PF._iwamoto_multiplier(g0, g1, g2)
        @test 0.0 <= μ <= 1.0
        g_opt = g_classic(μ, g0, g1, g2)
        # Brute-force minimum over the grid.
        g_grid_min = minimum(g_classic(m, g0, g1, g2) for m in grid)
        @test g_opt <= g_grid_min + 1e-10
    end
end

@testset "Iwamoto multiplier returns μ=0 when no step improves" begin
    # When g₁ and g₂ are large, any positive μ worsens the objective.
    # g(0) = g₀ should be the global minimum on [0, 1].
    g0 = 1.0
    g1 = 10.0
    g2 = 100.0
    μ = PF._iwamoto_multiplier(g0, g1, g2)
    g_at_mu = (1 - μ)^2 * g0 + 2 * μ^2 * (1 - μ) * g1 + μ^4 * g2
    # The optimizer must be able to return μ=0 or at least match g(0)=g₀.
    @test g_at_mu <= g0 + 1e-12
end

@testset "Iwamoto multiplier degenerate cases" begin
    # Degenerate cubic (g₂ ≈ 0): leading coefficient of derivative cubic is ~0.
    g_classic(μ, g0, g1, g2) = (1 - μ)^2 * g0 + 2 * μ^2 * (1 - μ) * g1 + μ^4 * g2
    g0 = 1.0
    g1 = 0.5
    g2 = 1e-35
    μ = PF._iwamoto_multiplier(g0, g1, g2)
    @test 0.0 <= μ <= 1.0
    g_opt = g_classic(μ, g0, g1, g2)
    grid = range(0.0, 1.0; length = 10001)
    g_grid_min = minimum(g_classic(m, g0, g1, g2) for m in grid)
    @test g_opt <= g_grid_min + 1e-10

    # Degenerate quadratic (g₂ ≈ 0 and g₁ ≈ 0): both leading coefficients ~0.
    g0_b = 1.0
    g1_b = 1e-35
    g2_b = 1e-35
    μ_b = PF._iwamoto_multiplier(g0_b, g1_b, g2_b)
    @test 0.0 <= μ_b <= 1.0
end

@testset "dc fallback" begin
    dc_pf = ACPowerFlow{NewtonRaphsonACPowerFlow}(;
        robust_power_flow = true,
        enhanced_flat_start = false,
    )
    no_dc_pf = ACPowerFlow{NewtonRaphsonACPowerFlow}(;
        robust_power_flow = false,
        enhanced_flat_start = false,
    )
    # test that _dc_power_flow_fallback! solves correctly.
    sys = PSB.build_system(PSB.PSITestSystems, "c_sys5")
    sys2 = deepcopy(sys)
    data = PowerFlowData(dc_pf, sys2)
    PF._dc_power_flow_fallback!(data, 1)
    valid_ix = PF.get_valid_ix(data)
    ABA_angles = data.bus_angles[valid_ix, 1]
    p_inj =
        data.bus_active_power_injections[valid_ix, 1] -
        data.bus_active_power_withdrawals[valid_ix, 1]
    @test data.aux_network_matrix.data * ABA_angles ≈ p_inj

    # check behavior of improved_x0 via creating bogus awful starting point.
    sys3 = deepcopy(sys)
    bad_x0!(sys3)
    data3 = PowerFlowData(no_dc_pf, sys3)
    x0 = PF.calculate_x0(data3, 1)
    residual = PF.ACPowerFlowResidual(data3, 1)
    residual(x0, 1)
    residualSize = norm(residual.Rv, 1)
    newx0 = deepcopy(x0)
    PF.dc_power_flow_start!(newx0, data, 1, residual)
    residual(newx0, 1)
    newResidualSize = norm(residual.Rv, 1)
    @test x0 !== newx0
    @test newResidualSize < residualSize
    # TODO: case with bad residual where DC power flow doesn't yield improvement?

    # check that it does the DC fallback.
    # _initialize_bus_data! corrects the voltages to be "reasonable," between 0.8 and 1.2
    sys4 = deepcopy(sys)
    bad_x0!(sys4)
    improvement_regex = r".*DC power flow fallback yields smaller residual.*"
    @test_logs (:info, improvement_regex) match_mode = :any PF.solve_power_flow(
        dc_pf,
        sys4,
    )
    sys5 = deepcopy(sys)
    bad_x0!(sys5)
    @test_logs (:debug, "skipping running DC power flow fallback") match_mode = :any min_level =
        Logging.Debug PF.solve_power_flow(no_dc_pf, sys5)
end

@testset "large residual warning" begin
    # a system where bus numbers aren't 1, 2,...is there a smaller one with this property?
    sys = PSB.build_system(PSB.PSISystems, "RTS_GMLC_DA_sys")
    PSY.set_units_base_system!(sys, PSY.UnitSystem.SYSTEM_BASE)
    for i in [1, 35, 52, 57, 43, 66, 49, 68, 71, 25, 69, 58, 3, 73]
        pf = ACPowerFlow(; enhanced_flat_start = false, correct_bustypes = true)
        data = PowerFlowData(pf, sys)
        # First, write solution to data. Then set magnitude of a random-ish bus to a huge number
        # and try to solve again: the "large residual warning" should be about that bus.
        solve_power_flow!(data)
        bus = collect(PSY.get_components(PSY.ACBus, sys))[i]
        bus_no = bus.number
        bus_ix = PowerFlows.get_bus_lookup(data)[bus_no]
        data.bus_magnitude[bus_ix] = 100.0
        @test_logs (:warn, Regex(".*Largest residual at bus $(bus_no).*")
        ) match_mode = :any solve_power_flow!(data)
    end
end
