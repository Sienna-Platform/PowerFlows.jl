@testset "test acpf_hvvp via O(ε²) finite-difference convergence" begin
    # Verify acpf_hvvp via the convergence-order test:
    #   w_fd(ε) := [F(x+εv+εu) - F(x+εv-εu) - F(x-εv+εu) + F(x-εv-εu)] / (4ε²)
    #            = vᵀ H u + O(ε²)
    # Halving ε should shrink the truncation error by ~4×. This is a principled
    # correctness check that doesn't depend on a hand-picked atol.
    sys = PSB.build_system(PSB.PSITestSystems, "c_sys14")
    pf = ACPowerFlow{NewtonRaphsonACPowerFlow}(; correct_bustypes = true)
    data = PowerFlowData(pf, sys)
    time_step = 1

    residual = PF.ACPowerFlowResidual(data, time_step)
    x0 = PF.calculate_x0(data, time_step)
    n = length(x0)

    function residual_at(x)
        residual(x, time_step)
        return copy(residual.Rv)
    end

    Random.seed!(42)
    # normalize directions so ε is comparable across n
    v = randn(n) ./ sqrt(n)
    u = randn(n) ./ sqrt(n)

    function w_fd(ε)
        return (
            residual_at(x0 .+ ε .* v .+ ε .* u) .-
            residual_at(x0 .+ ε .* v .- ε .* u) .-
            residual_at(x0 .- ε .* v .+ ε .* u) .+
            residual_at(x0 .- ε .* v .- ε .* u)
        ) ./ (4 * ε^2)
    end

    w_analytic = PF.acpf_hvvp(data, time_step, v, u)
    residual_at(x0)  # restore data state

    # Sweep ε in the regime where O(ε²) truncation dominates, well above the
    # roundoff floor (~eps^(1/4) ≈ 1.2e-4 for a 4-point central scheme).
    εs = [1e-2, 5e-3, 2.5e-3, 1.25e-3]
    errs = [norm(w_analytic .- w_fd(ε), Inf) for ε in εs]
    residual_at(x0)

    # Each halving of ε should shrink the error by ~4. Allow [3, 5] to absorb
    # finite-precision noise.
    ratios = errs[1:(end - 1)] ./ errs[2:end]
    for r in ratios
        @test 3.0 <= r <= 5.0
    end
    # Sanity: errors are monotonically shrinking.
    @test issorted(errs; rev = true)
end

@testset "test fixed-point Jacobian-vector product via O(ε²) FD convergence" begin
    # G(x) = ∂g/∂x where g(x) = x - J(x)⁻¹ F(x). The matrix-free product is
    #   G v = J⁻¹ · acpf_hvvp(v, u),  u = J⁻¹ F(x)
    # Verify it via central-difference convergence:
    #   (g(x+εv) - g(x-εv)) / (2ε) = G v + O(ε²)
    # so the error halves by 4× when ε halves.
    sys = PSB.build_system(PSB.PSITestSystems, "c_sys14")
    pf = ACPowerFlow{NewtonRaphsonACPowerFlow}(; correct_bustypes = true)
    data = PowerFlowData(pf, sys)
    time_step = 1

    residual = PF.ACPowerFlowResidual(data, time_step)
    jac = PF.ACPowerFlowJacobian(
        data,
        residual.bus_slack_participation_factors,
        residual.subnetworks,
        time_step,
    )
    x0 = PF.calculate_x0(data, time_step)
    n = length(x0)

    function g(x)
        residual(x, time_step)
        jac(time_step)
        return x .- (jac.Jv \ copy(residual.Rv))
    end

    Random.seed!(7)
    v = randn(n) ./ sqrt(n)

    Gv_fd(ε) = (g(x0 .+ ε .* v) .- g(x0 .- ε .* v)) ./ (2 * ε)

    # Matrix-free G·v via the same machinery the diagnostic uses.
    residual(x0, time_step)
    jac(time_step)
    u = jac.Jv \ copy(residual.Rv)
    Gv_analytic = jac.Jv \ PF.acpf_hvvp(data, time_step, v, u)

    εs = [1e-3, 5e-4, 2.5e-4, 1.25e-4]
    errs = [norm(Gv_analytic .- Gv_fd(ε), Inf) for ε in εs]
    g(x0)  # restore

    ratios = errs[1:(end - 1)] ./ errs[2:end]
    for r in ratios
        @test 3.0 <= r <= 5.0
    end
    @test issorted(errs; rev = true)
end

@testset "test compute_fixed_point_spectral_radius smoke test" begin
    sys = PSB.build_system(PSB.PSITestSystems, "c_sys14")
    pf = ACPowerFlow{NewtonRaphsonACPowerFlow}(; correct_bustypes = true)
    data = PowerFlowData(pf, sys)
    ρ, info, condest = PF.compute_fixed_point_spectral_radius(data, 1)
    # c_sys14 is a well-conditioned 14-bus system; NR converges fast from flat
    # start, so ρ should be safely below 1.
    @test isfinite(ρ)
    @test 0 < ρ < 1
    # condest is a 1-norm condition estimate; for a healthy 14-bus system it
    # should be finite and well below the borderline-singular regime we saw
    # on CATS (~1e9).
    @test isfinite(condest)
    @test condest > 0
    @test condest < 1e6
end

@testset "test compute_fixed_point_spectral_radius option flag" begin
    sys = PSB.build_system(PSB.PSITestSystems, "c_sys14")
    pf = ACPowerFlow{NewtonRaphsonACPowerFlow}(;
        correct_bustypes = true,
        compute_fixed_point_spectral_radius = true,
    )
    data = PowerFlowData(pf, sys)
    # smoke test: solve runs end-to-end with the flag enabled
    @test solve_power_flow!(data)
end

@testset "test per-iteration spectral radius monitor emits log lines" begin
    sys = PSB.build_system(PSB.PSITestSystems, "c_sys14")
    pf = ACPowerFlow{NewtonRaphsonACPowerFlow}(;
        correct_bustypes = true,
        compute_fixed_point_spectral_radius = true,
    )
    data = PowerFlowData(pf, sys)

    # Capture log output through a TestLogger and verify the diagnostic lines
    # appear in the expected order: a single x0 line, then NR iter lines.
    test_logger = Test.TestLogger(; min_level = Logging.Info)
    Logging.with_logger(test_logger) do
        solve_power_flow!(data)
    end
    msgs = [r.message for r in test_logger.logs]

    x0_lines = filter(m -> occursin("x0 (time_step 1)", m), msgs)
    iter_lines = filter(m -> occursin(r"NR iter \d+", m), msgs)
    @test length(x0_lines) == 1
    @test length(iter_lines) >= 1
    # Each diagnostic line carries ρ, ‖F‖_∞, the binding bus/equation, and κ̂(J).
    for line in vcat(x0_lines, iter_lines)
        @test occursin("ρ = ", line)
        @test occursin("‖F‖_∞ = ", line)
        @test occursin(r"at bus \d+ \(P\)|at bus \d+ \(Q\)", line)
        @test occursin("κ̂(J) = ", line)
    end
end

@testset "test per-iteration monitor under TrustRegionACPowerFlow" begin
    sys = PSB.build_system(PSB.PSITestSystems, "c_sys14")
    pf = ACPowerFlow{TrustRegionACPowerFlow}(;
        correct_bustypes = true,
        compute_fixed_point_spectral_radius = true,
    )
    data = PowerFlowData(pf, sys)

    test_logger = Test.TestLogger(; min_level = Logging.Info)
    Logging.with_logger(test_logger) do
        solve_power_flow!(data)
    end
    msgs = [r.message for r in test_logger.logs]
    iter_lines = filter(m -> occursin(r"TR iter \d+", m), msgs)
    @test length(iter_lines) >= 1
end
