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
