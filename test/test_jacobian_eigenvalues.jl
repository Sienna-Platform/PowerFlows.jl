@testset "test compute_min_jacobian_eigenvalue matches dense eigvals" begin
    # Ground-truth check: the inverse-iteration estimate of the smallest-magnitude
    # eigenvalue of J should match the dense `eigvals` of the same Jacobian to
    # solver tolerance. J is non-symmetric so the eigenvalue is generally complex;
    # we compare the full complex value, not just its magnitude.
    sys = PSB.build_system(PSB.PSITestSystems, "c_sys14")
    pf = ACPowerFlow{NewtonRaphsonACPowerFlow}(; correct_bustypes = true)
    data = PowerFlowData(pf, sys)
    time_step = 1

    # Build the Jacobian at the flat start, exactly as the routine does internally.
    residual = PF.ACPowerFlowResidual(data, time_step)
    jac = PF.ACPowerFlowJacobian(residual, time_step)
    x0 = PF.calculate_x0(data, time_step)
    residual(x0, time_step)
    jac(time_step)

    ev = eigvals(Matrix(jac.Jv))
    λ_true = ev[argmin(abs.(ev))]

    λ, info, condest = PF.compute_min_jacobian_eigenvalue(data, time_step)
    @test info.converged >= 1
    # Match the complex eigenvalue closest to the origin to tight relative error.
    @test abs(λ - λ_true) / abs(λ_true) < 1e-8
    # 1/λ is the largest-magnitude eigenvalue of J⁻¹; sanity-check the inverse map.
    @test abs(abs(λ) - minimum(abs.(ev))) < 1e-8

    # condest is a 1-norm condition estimate from the same KLU factor; healthy
    # 14-bus system should be well-conditioned.
    @test isfinite(condest)
    @test 0 < condest < 1e6
end

@testset "test compute_min_jacobian_eigenvalue accepts a custom x0" begin
    # The eigenvalue is state-dependent; evaluating at a perturbed point should
    # still agree with the dense ground truth built at that same point.
    sys = PSB.build_system(PSB.PSITestSystems, "c_sys14")
    pf = ACPowerFlow{NewtonRaphsonACPowerFlow}(; correct_bustypes = true)
    data = PowerFlowData(pf, sys)
    time_step = 1

    residual = PF.ACPowerFlowResidual(data, time_step)
    jac = PF.ACPowerFlowJacobian(residual, time_step)
    x0 = PF.calculate_x0(data, time_step)
    Random.seed!(11)
    x_pert = x0 .+ 0.05 .* randn(length(x0))

    residual(x_pert, time_step)
    jac(time_step)
    ev = eigvals(Matrix(jac.Jv))
    λ_true = ev[argmin(abs.(ev))]

    λ, info, _ = PF.compute_min_jacobian_eigenvalue(data, time_step; x0 = x_pert)
    @test info.converged >= 1
    @test abs(λ - λ_true) / abs(λ_true) < 1e-8
end

# Helper: solve under `pf` and return the per-iteration/x0 diagnostic log lines.
function _diagnostic_lines(pf, sys)
    data = PowerFlowData(pf, sys)
    tl = Test.TestLogger(; min_level = Logging.Info)
    Logging.with_logger(tl) do
        solve_power_flow!(data)
    end
    return [
        r.message for r in tl.logs
        if occursin("x0 (time_step", r.message) || occursin(r"iter \d+", r.message)
    ]
end

@testset "compute_min_jacobian_eigenvalue flag is independent of spectral radius" begin
    # With only the min-eigenvalue flag set, the per-iteration diagnostic lines
    # carry λ_min(J) and κ̂(J) but NOT ρ — the (polar-only) spectral-radius path
    # must not run.
    sys = PSB.build_system(PSB.PSITestSystems, "c_sys14")
    pf = ACPowerFlow{NewtonRaphsonACPowerFlow}(;
        correct_bustypes = true, compute_min_jacobian_eigenvalue = true)
    lines = _diagnostic_lines(pf, sys)
    @test length(lines) >= 2
    for line in lines
        @test occursin("λ_min(J) = ", line)
        @test occursin("|λ_min| = ", line)
        @test occursin("κ̂(J) = ", line)
        @test !occursin("ρ = ", line)   # spectral radius must be absent
    end
end

@testset "compute_min_jacobian_eigenvalue works for the rectangular-CI formulation" begin
    # The spectral-radius flag is polar-only (getter returns false for rect/mixed),
    # so rect-CI never logged before. The min-eigenvalue flag is available on every
    # AC formulation.
    sys = PSB.build_system(PSB.PSITestSystems, "c_sys14")
    pf = PF.ACRectangularPowerFlow{NewtonRaphsonACPowerFlow}(;
        correct_bustypes = true, compute_min_jacobian_eigenvalue = true)
    lines = _diagnostic_lines(pf, sys)
    @test length(lines) >= 2
    for line in lines
        @test occursin("λ_min(J) = ", line)
        @test !occursin("ρ = ", line)
    end
end

@testset "compute_min_jacobian_eigenvalue works on LCC systems (spectral radius does not)" begin
    # On an LCC system the spectral-radius monitor errors (its Hessian-vector
    # product covers only bus states, mismatching the LCC-augmented factor),
    # whereas the min-eigenvalue diagnostic only factorizes the Jacobian and works.
    sys = System(joinpath(TEST_DATA_DIR, "case5_2_lcc.raw"))
    @test length(collect(get_components(TwoTerminalLCCLine, sys))) == 2

    # spectral-radius flag: the monitor blows up on the LCC-augmented system.
    data_ρ = PowerFlowData(
        ACPowerFlow{NewtonRaphsonACPowerFlow}(;
            compute_fixed_point_spectral_radius = true),
        sys)
    @test_throws Exception solve_power_flow!(data_ρ)

    # min-eigenvalue flag: solves cleanly and logs λ_min on every line.
    pf = ACPowerFlow{NewtonRaphsonACPowerFlow}(; compute_min_jacobian_eigenvalue = true)
    data = PowerFlowData(pf, sys)
    tl = Test.TestLogger(; min_level = Logging.Info)
    converged = Logging.with_logger(tl) do
        solve_power_flow!(data)
    end
    @test converged
    lines = [
        r.message for r in tl.logs
        if occursin("x0 (time_step", r.message) || occursin(r"iter \d+", r.message)
    ]
    @test length(lines) >= 2
    for line in lines
        @test occursin("λ_min(J) = ", line)
    end
end
