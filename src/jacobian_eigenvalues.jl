"""
    compute_min_jacobian_eigenvalue(data, time_step; x0, tol, maxiter, krylovdim)
        -> (λ, info, condest::Float64)

Estimate the eigenvalue of the AC power flow Jacobian ``J(x)`` of smallest
magnitude (the one closest to the origin) at the state `x0` (default: the result
of `calculate_x0(data, time_step)`). The eigenvalue closest to zero measures how
near ``J`` is to singular: at a saddle-node bifurcation (voltage collapse) a real
eigenvalue of ``J`` crosses zero, so a small ``|λ_min|`` flags an ill-conditioned
or near-singular operating point where Newton-type methods struggle.

# Method

We employ **inverse iteration**, reusing a sparse `KLU` factorization of ``J``.
matvec ``v \\mapsto J^{-1} v`` is applied as a back-solve against that factor —
the inverse is *never* formed explicitly, as it would be dense.

Returns `(λ_min, info, condest)`, where `λ_min` may be complex, `info` is the
KrylovKit convergence info, and `condest` is a Hager 1-norm estimate of the
condition number of ``J`` computed from the same `KLU` factor.

# Notes

- Unlike [`compute_fixed_point_spectral_radius`](@ref), this needs only the
  Jacobian itself (not the residual Hessian), so it supports every AC
  formulation, including LCC HVDC systems.
"""
function compute_min_jacobian_eigenvalue(
    data::ACPowerFlowData,
    time_step::Int;
    x0::Union{Vector{Float64}, Nothing} = nothing,
    tol::Float64 = 1e-6,
    maxiter::Int = 200,
    krylovdim::Int = 30,
)
    # PERF: have caller pass in the already constructed J and residual.
    residual = ACPowerFlowResidual(data, time_step)
    jac = ACPowerFlowJacobian(residual, time_step)
    x = isnothing(x0) ? calculate_x0(data, time_step) : copy(x0)
    residual(x, time_step)
    jac(time_step)
    return _min_jacobian_eigenvalue!(
        jac;
        tol = tol, maxiter = maxiter, krylovdim = krylovdim,
    )
end

"""In-place smallest-magnitude Jacobian eigenvalue computation that reuses an
already-evaluated `jac` (its `jac.Jv` must hold the Jacobian at the current
state). Factorizes `jac.Jv` once with `KLU` and runs inverse iteration; see
[`compute_min_jacobian_eigenvalue`](@ref). Returns `(λ_min, info, condest)`.

Accepts any AC formulation's Jacobian (polar, rectangular-CI, mixed-CPB) — it
only touches `jac.Jv`, which every formulation provides."""
function _min_jacobian_eigenvalue!(
    jac::Union{ACPowerFlowJacobian, ACRectangularCIJacobian, ACMixedCPBJacobian};
    tol::Float64 = 1e-6,
    maxiter::Int = 200,
    krylovdim::Int = 30,
)
    n = size(jac.Jv, 1)
    # PERF: use existing factorization.
    F = KLU.klu(jac.Jv)
    matvec(v::AbstractVector) = F \ v
    v_init = ones(Float64, n) ./ sqrt(n)
    # Largest-magnitude eigenvalue μ of J⁻¹ ⇒ smallest-magnitude eigenvalue 1/μ of J.
    vals, _, info = KrylovKit.eigsolve(
        matvec, v_init, 1, :LM;
        tol = tol, maxiter = maxiter, krylovdim = krylovdim,
    )
    λ_min = isempty(vals) ? complex(NaN, NaN) : inv(vals[1])
    condest = KLU.condest(F)
    return λ_min, info, condest
end
