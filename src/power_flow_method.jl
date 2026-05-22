"""Format a per-iteration diagnostic line: ρ, ‖F‖_∞ and the bus/equation where
that infinity-norm is attained, and a κ̂(J) condition estimate. All numerics
rounded to 4 significant figures."""
function _log_diagnostics(
    label::String,
    ρ::Float64,
    Rv::AbstractVector{Float64},
    condest::Float64,
)
    abs_max, ix = findmax(abs, Rv)
    pow_type = ix % 2 == 1 ? "P" : "Q"
    bus_ix = div(ix + 1, 2)
    sf(x) = round(x; sigdigits = 4)
    @info "$label: ρ = $(sf(ρ)), ‖F‖_∞ = $(sf(abs_max)) at bus $bus_ix ($pow_type), " *
          "κ̂(J) = $(sf(condest))"
    return
end

"""Cache for non-linear methods.

# Fields
- `x::Vector{Float64}`: the current state vector.
- `r::Vector{Float64}`: the current residual.
- `Δx_nr::Vector{Float64}`: the step under the Newton-Raphson method.
The remainder of the fields are only used in the `TrustRegionACPowerFlow`:
- `r_predict::Vector{Float64}`: the predicted residual at `x+Δx_proposed`,
    under a linear approximation: i.e `J_x⋅(x+Δx_proposed)`.
- `Δx_proposed::Vector{Float64}`: the suggested step `Δx`, selected among `Δx_nr`,
    `Δx_cauchy`, and the dogleg interpolation between the two. The first is chosen when
    `x+Δx_nr` is inside the trust region, the second when both `x+Δx_cauchy`
    and `x+Δx_nr` are outside the trust region, and the third when `x+Δx_cauchy`
    is inside and `x+Δx_nr` outside. The dogleg step selects the point where the line
    from `x+Δx_cauchy` to `x+Δx_nr` crosses the boundary of the trust region.
- `Δx_cauchy::Vector{Float64}`: the step to the Cauchy point if the Cauchy point
    lies within the trust region, otherwise a step in that direction."""
struct StateVectorCache
    x::Vector{Float64}
    r::Vector{Float64} # residual
    r_predict::Vector{Float64} # predicted residual
    Δx_proposed::Vector{Float64} # proposed Δx: Cauchy, NR, or dogleg step.
    Δx_cauchy::Vector{Float64} # Cauchy step
    Δx_nr::Vector{Float64} # Newton-Raphson step
    d::Vector{Float64}
end

function StateVectorCache(x0::Vector{Float64}, f0::Vector{Float64})
    x = copy(x0)
    r = copy(f0)
    r_predict = copy(x0)
    Δx_proposed = copy(x0)
    Δx_cauchy = copy(x0)
    Δx_nr = copy(x0)
    return StateVectorCache(x, r, r_predict, Δx_proposed, Δx_cauchy, Δx_nr, ones(size(x0)))
end

"""Solve for the Newton-Raphson step, given the factorization object for `J.Jv`
(if non-singular) or its stand-in (if singular)."""
function _solve_Δx_nr!(stateVector::StateVectorCache, cache::KLULinSolveCache{J_INDEX_TYPE})
    copyto!(stateVector.Δx_nr, stateVector.r)
    solve!(cache, stateVector.Δx_nr)
    return
end

"""Check error and do refinement."""
function _do_refinement!(stateVector::StateVectorCache,
    A::SparseMatrixCSC{Float64, J_INDEX_TYPE},
    cache::KLULinSolveCache{J_INDEX_TYPE},
    refinement_threshold::Float64,
    refinement_eps::Float64,
)
    # use stateVector.r_predict as temporary buffer.
    δ_temp = stateVector.r_predict
    mul!(δ_temp, A, stateVector.Δx_nr)
    δ_temp .-= stateVector.r
    delta = norm(δ_temp, 1) / norm(stateVector.r, 1)
    if delta > refinement_threshold
        stateVector.Δx_nr .= solve_w_refinement(cache,
            A,
            stateVector.r,
            refinement_eps)
    end
    return
end

"""Sets the Newton-Raphson step. Usually, this is just `J.Jv \\ stateVector.r`, but
`J.Jv` might be singular."""
function _set_Δx_nr!(stateVector::StateVectorCache,
    J::Union{ACPowerFlowJacobian, ACRectangularCIJacobian, ACMixedCPBJacobian},
    linSolveCache::KLULinSolveCache{J_INDEX_TYPE},
    solver::ACPowerFlowSolverType,
    refinement_threshold::Float64,
    refinement_eps::Float64)
    try
        numeric_refactor!(linSolveCache, J.Jv)
    catch e
        if e isa LinearAlgebra.SingularException
            @warn("$solver hit a point where the Jacobian is singular.")
            M = _singular_J_fallback(J.Jv, stateVector.x)
            tempCache = KLULinSolveCache(M)
            # creating a new solver cache to solve Mx = stateVector.r once. Default ldiv!
            # might be faster, but singular J's should be rare.
            full_factor!(tempCache, M)
            _solve_Δx_nr!(stateVector, tempCache)
            _do_refinement!(stateVector, M, tempCache, refinement_threshold, refinement_eps)
            LinearAlgebra.rmul!(stateVector.Δx_nr, -1.0)
        else
            @error("KLU factorization failed: $e")
        end
    else
        _solve_Δx_nr!(stateVector, linSolveCache)
        _do_refinement!(
            stateVector,
            J.Jv,
            linSolveCache,
            refinement_threshold,
            refinement_eps,
        )
        LinearAlgebra.rmul!(stateVector.Δx_nr, -1.0)
    end
    return
end

"""Returns a stand-in matrix for singular J's."""
function _singular_J_fallback(Jv::SparseMatrixCSC{Float64, J_INDEX_TYPE},
    x::Vector{Float64})
    fjac2 = Jv' * Jv
    lambda = NR_SINGULAR_SCALING * sqrt(length(x) * eps()) * norm(fjac2, 1)
    return -(fjac2 + lambda * LinearAlgebra.I)
end

"""Sets `Δx_proposed` equal to the `Δx` by which we should update `x`. Decides
between the Cauchy step `Δx_cauchy`, Newton-Raphson step `Δx_nr`, and the dogleg
interpolation between the two, based on which fall within the trust region."""
function _dogleg!(Δx_proposed::Vector{Float64},
    Δx_cauchy::Vector{Float64},
    Δx_nr::Vector{Float64},
    r::Vector{Float64},
    Jv::SparseMatrixCSC{Float64, J_INDEX_TYPE},
    d::Vector{Float64},
    delta::Float64,
)
    nr_norm = wnorm(d, Δx_nr)
    @debug "Trust region: ||Δx_nr|| = $(siground(nr_norm)), δ = $(siground(delta))"

    if nr_norm <= delta
        copyto!(Δx_proposed, Δx_nr) # update Δx_proposed: newton-raphson case.
        @debug "Newton-Raphson step selected (inside trust region)"
    else
        # using Δx_proposed as a temporary buffer: alias to g for readability
        g = Δx_proposed
        LinearAlgebra.mul!(g, Jv', r)
        g .= g ./ d .^ 2
        Δx_cauchy .= -wnorm(d, g)^2 / sum(abs2, Jv * g) .* g # Cauchy point

        cauchy_norm = wnorm(d, Δx_cauchy)
        @debug "Cauchy point: ||Δx_cauchy|| = $(siground(cauchy_norm))"

        if cauchy_norm >= delta
            # Δx_cauchy outside region => take step of length delta in direction of -g.
            LinearAlgebra.rmul!(g, -delta / wnorm(d, g))
            @debug "Cauchy step selected (truncated to trust region boundary)"
            # not needed because g is already an alias for Δx_proposed.
            # copyto!(Δx_proposed, g) # update Δx_proposed: cauchy point case
        else
            # Δx_cauchy inside region => next point is the spot where the line from
            # Δx_cauchy to Δx_nr crosses the boundary of the trust region.
            # this is the "dogleg" part.

            # using Δx_nr as temporary buffer: alias to Δx_diff for readability.
            Δx_nr .-= Δx_cauchy
            Δx_diff = Δx_nr

            b = wdot(d, Δx_cauchy, d, Δx_diff)
            a = wnorm(d, Δx_diff)^2
            tau = (-b + sqrt(b^2 - 4a * (wnorm(d, Δx_cauchy)^2 - delta^2))) / (2a)
            Δx_cauchy .+= tau .* Δx_diff
            copyto!(Δx_proposed, Δx_cauchy) # update Δx_proposed: dogleg case.
            @debug "Dogleg step selected (τ = $(siground(tau)))"
        end
    end
    return
end

"""Accept a trust region step: update cached residual and autoscale vector `d`.
The caller is responsible for recomputing the Jacobian via `J(time_step)` before
calling this, so that `Jv` reflects the new state."""
function _accept_trust_region_step!(
    stateVector::StateVectorCache,
    residual::Union{ACPowerFlowResidual, ACRectangularCIResidual, ACMixedCPBResidual},
    Jv::SparseMatrixCSC{Float64, J_INDEX_TYPE},
    autoscale::Bool,
)
    stateVector.r .= residual.Rv
    if autoscale
        for i in 1:length(stateVector.x)
            stateVector.d[i] = max(0.1 * stateVector.d[i], norm(view(Jv, :, i)))
        end
    end
    return nothing
end

"""Attempt Iwamoto damping on a rejected trust region step.

Uses the already-evaluated trial-point residual to compute an optimal damped step.
Returns `true` if the damped step was accepted, `false` if reverted."""
function _iwamoto_fallback!(
    time_step::Int,
    stateVector::StateVectorCache,
    residual::Union{ACPowerFlowResidual, ACRectangularCIResidual, ACMixedCPBResidual},
    J::Union{ACPowerFlowJacobian, ACRectangularCIJacobian, ACMixedCPBJacobian},
    old_residual::Vector{Float64},
    old_residual_norm::Float64,
    new_residual_norm::Float64,
    autoscale::Bool,
)::Bool
    g0 = old_residual_norm
    g1 = dot(old_residual, residual.Rv)
    g2 = new_residual_norm
    μ = _iwamoto_multiplier(g0, g1, g2)
    # Revert full step, apply damped step in a single fused pass.
    @. stateVector.x += (μ - 1.0) * stateVector.Δx_proposed
    residual(stateVector.x, time_step)
    g_damped = dot(residual.Rv, residual.Rv)
    if g_damped < g0
        @debug "Iwamoto fallback accepted: μ = $(siground(μ)), " *
               "g_damped/g₀ = $(siground(g_damped / g0))"
        J(time_step)
        _accept_trust_region_step!(stateVector, residual, J.Jv, autoscale)
        return true
    else
        # Damped step also failed — full revert.
        @. stateVector.x -= μ * stateVector.Δx_proposed
        copyto!(residual.Rv, old_residual)
        @debug "Iwamoto fallback rejected: μ = $(siground(μ)), " *
               "g_damped/g₀ = $(siground(g_damped / g0)); reverting"
        return false
    end
end

"""Does a single iteration of the `TrustRegionNRMethod`:
updates the `x` and `r` fields of the `stateVector` and computes
the value of the Jacobian at the new `x`, if needed. Unlike
`_simple_step`, this has a return value, the updated value of `delta``."""
function _trust_region_step(time_step::Int,
    stateVector::StateVectorCache,
    linSolveCache::KLULinSolveCache{J_INDEX_TYPE},
    residual::Union{ACPowerFlowResidual, ACRectangularCIResidual, ACMixedCPBResidual},
    J::Union{ACPowerFlowJacobian, ACRectangularCIJacobian, ACMixedCPBJacobian},
    delta::Float64,
    delta_max::Float64,
    eta::Float64,
    autoscale::Bool,
    iwamoto_fallback::Bool,
)
    old_delta = delta
    _set_Δx_nr!(
        stateVector,
        J,
        linSolveCache,
        TrustRegionACPowerFlow(),
        DEFAULT_REFINEMENT_THRESHOLD,
        DEFAULT_REFINEMENT_EPS,
    )
    _dogleg!(
        stateVector.Δx_proposed,
        stateVector.Δx_cauchy,
        stateVector.Δx_nr,
        stateVector.r,
        J.Jv,
        stateVector.d,
        delta,
    )
    # find proposed next point.
    stateVector.x .+= stateVector.Δx_proposed

    # use cache.Δx_nr as temporary buffer to store old residual
    # to avoid recomputing if we don't change x.
    oldResidual = stateVector.Δx_nr
    copyto!(oldResidual, residual.Rv)
    old_residual_norm = sum(abs2, stateVector.r)
    residual(stateVector.x, time_step)
    new_residual_norm = sum(abs2, residual.Rv)

    # Ratio of actual to predicted reduction
    LinearAlgebra.mul!(stateVector.r_predict, J.Jv, stateVector.Δx_proposed)
    stateVector.r_predict .+= stateVector.r
    rho =
        (sum(abs2, stateVector.r) - sum(abs2, residual.Rv)) /
        (sum(abs2, stateVector.r) - sum(abs2, stateVector.r_predict))

    @debug "Trust region step: ρ = $(siground(rho)), η = $(siground(eta)), ||Δx|| = $(siground(norm(stateVector.Δx_proposed)))"

    step_accepted = false
    iwamoto_accepted = false
    if rho > eta
        # Successful iteration
        @debug "Step accepted: sum of squares $(siground(dot(residual.Rv, residual.Rv))), L ∞ norm $(siground(norm(residual.Rv, Inf))), Δ = $(siground(delta)), ||Δx|| = $(siground(norm(stateVector.Δx_proposed)))"
        J(time_step)
        _accept_trust_region_step!(stateVector, residual, J.Jv, autoscale)
        step_accepted = true
    else
        # Unsuccessful step — try Iwamoto damping before reverting.
        if iwamoto_fallback
            iwamoto_accepted = _iwamoto_fallback!(
                time_step, stateVector, residual, J,
                oldResidual, old_residual_norm, new_residual_norm, autoscale)
            if iwamoto_accepted
                # Iwamoto accepted a damped step — shrink trust region since the
                # full proposed step was rejected by rho. Do not use rho-based
                # expansion logic because rho corresponds to the rejected full step.
                delta = min(delta / 2, delta_max)
                @debug "Trust region decreased (Iwamoto fallback accepted): δ $(siground(old_delta)) → $(siground(delta))"
                return delta, true
            end
        else
            stateVector.x .-= stateVector.Δx_proposed
            copyto!(residual.Rv, oldResidual)
            @debug "Step rejected: ρ = $(siground(rho)) ≤ η = $(siground(eta))"
        end
    end

    # Update size of trust region based on rho (only reached when the full step
    # was accepted via rho, or Iwamoto is disabled, or Iwamoto didn't help).
    if rho < HALVE_TRUST_REGION # rho < 0.1: insufficient improvement
        delta = delta / 2
        @debug "Trust region decreased: δ $(siground(old_delta)) → $(siground(delta)) (ρ < $(HALVE_TRUST_REGION))"
    elseif step_accepted && rho >= DOUBLE_TRUST_REGION # rho >= 0.9: good improvement
        delta = 2 * wnorm(stateVector.d, stateVector.Δx_proposed)
        @debug "Trust region increased (good): δ $(siground(old_delta)) → $(siground(delta)) (ρ ≥ $(DOUBLE_TRUST_REGION))"
    elseif step_accepted && rho >= MAX_DOUBLE_TRUST_REGION # rho >= 0.5: so-so improvement
        delta = max(delta, 2 * wnorm(stateVector.d, stateVector.Δx_proposed))
        @debug "Trust region increased (moderate): δ $(siground(old_delta)) → $(siground(delta)) (ρ ≥ $(MAX_DOUBLE_TRUST_REGION))"
    else
        @debug "Trust region unchanged: δ = $(siground(delta))"
    end
    delta = min(delta, delta_max)
    return delta, (step_accepted || iwamoto_accepted)
end

"""Evaluate the Iwamoto objective g(μ) = ‖(1-μ)f₀ + μ²f₁‖² expanded as
g(μ) = (1-μ)²g₀ + 2μ²(1-μ)g₁ + μ⁴g₂ where g₀=‖f₀‖², g₁=f₀ᵀf₁, g₂=‖f₁‖²."""
@inline function _iwamoto_objective(
    μ::Float64, g0::Float64, g1::Float64, g2::Float64,
)::Float64
    om = 1.0 - μ
    μ2 = μ * μ
    return om * om * g0 + 2.0 * μ2 * om * g1 + μ2 * μ2 * g2
end

"""If μ ∈ [IWAMOTO_MU_MIN, IWAMOTO_MU_MAX] and g(μ) < best_g, return the
improved (μ, g(μ)); otherwise return (best_μ, best_g) unchanged."""
@inline function _try_iwamoto_candidate(
    μ::Float64,
    best_μ::Float64,
    best_g::Float64,
    g0::Float64,
    g1::Float64,
    g2::Float64,
)::Tuple{Float64, Float64}
    if IWAMOTO_MU_MIN <= μ <= IWAMOTO_MU_MAX
        gval = _iwamoto_objective(μ, g0, g1, g2)
        if gval < best_g
            return μ, gval
        end
    end
    return best_μ, best_g
end

"""Compute the optimal Iwamoto step multiplier μ ∈ [IWAMOTO_MU_MIN, IWAMOTO_MU_MAX]
by minimizing g(μ) = (1-μ)²g₀ + 2μ²(1-μ)g₁ + μ⁴g₂.

The stationary points satisfy the cubic g'(μ)/2 = 2g₂μ³ - 3g₁μ² + (g₀+2g₁)μ - g₀ = 0.
All real roots are found analytically via the depressed-cubic trigonometric/Cardano form,
and the global minimizer of g over the domain is returned. O(1), zero-allocation."""
function _iwamoto_multiplier(g0::Float64, g1::Float64, g2::Float64)::Float64
    # Initialize best candidate from domain boundaries.
    best_μ = IWAMOTO_MU_MIN
    best_g = _iwamoto_objective(IWAMOTO_MU_MIN, g0, g1, g2)
    best_μ, best_g = _try_iwamoto_candidate(IWAMOTO_MU_MAX, best_μ, best_g, g0, g1, g2)

    # Cubic coefficients: c₃μ³ + c₂μ² + c₁μ + c₀ = 0
    c3 = 2.0 * g2
    c2 = -3.0 * g1
    c1 = g0 + 2.0 * g1
    c0 = -g0

    if abs(c3) < IWAMOTO_DEGENERACY_TOL
        # Degenerate: solve quadratic c₂μ² + c₁μ + c₀ = 0
        if abs(c2) > IWAMOTO_DEGENERACY_TOL
            disc = c1 * c1 - 4.0 * c2 * c0
            if disc >= 0.0
                sq = sqrt(disc)
                for μ in ((-c1 + sq) / (2.0 * c2), (-c1 - sq) / (2.0 * c2))
                    best_μ, best_g = _try_iwamoto_candidate(μ, best_μ, best_g, g0, g1, g2)
                end
            end
        elseif abs(c1) > IWAMOTO_DEGENERACY_TOL
            best_μ, best_g = _try_iwamoto_candidate(-c0 / c1, best_μ, best_g, g0, g1, g2)
        end
        return best_μ
    end

    # Full cubic — depress to t³ + At + B = 0 via μ = t - p/3
    p = c2 / c3
    q = c1 / c3
    c0_n = c0 / c3
    p3 = p / 3.0
    A = q - p * p3
    B = c0_n - q * p3 + 2.0 * p3^3
    Δ = -4.0 * A^3 - 27.0 * B^2

    if Δ > 0.0
        # Three distinct real roots — trigonometric form (A < 0 guaranteed when Δ > 0).
        s = sqrt(-A / 3.0)
        m = 2.0 * s
        arg = clamp(-B / (2.0 * s * s * s), -1.0, 1.0)
        φ3 = acos(arg) / 3.0
        for k in 0:2
            best_μ, best_g = _try_iwamoto_candidate(
                m * cos(φ3 - 2.0 * π * k / 3.0) - p3,
                best_μ, best_g, g0, g1, g2)
        end
    elseif Δ < 0.0
        # One real root — Cardano's formula.
        sqD = sqrt(max(-Δ / 108.0, 0.0))
        best_μ, best_g = _try_iwamoto_candidate(
            cbrt(-B / 2.0 + sqD) + cbrt(-B / 2.0 - sqD) - p3,
            best_μ, best_g, g0, g1, g2)
    else
        # Δ ≈ 0 — repeated roots.
        if abs(A) < IWAMOTO_DEGENERACY_TOL
            # Triple root at t = 0.
            best_μ, best_g = _try_iwamoto_candidate(-p3, best_μ, best_g, g0, g1, g2)
        else
            # Simple root t₁ = 3B/A and double root t₂ = -3B/(2A).
            for t in (3.0 * B / A, -3.0 * B / (2.0 * A))
                best_μ, best_g = _try_iwamoto_candidate(
                    t - p3, best_μ, best_g, g0, g1, g2)
            end
        end
    end

    return best_μ
end

"""Does a single iteration of `NewtonRaphsonACPowerFlow`. Updates the `r` and `x`
 fields of the `stateVector`, and computes the Jacobian at the new `x`."""
function _simple_step(time_step::Int,
    stateVector::StateVectorCache,
    linSolveCache::KLULinSolveCache{J_INDEX_TYPE},
    residual::Union{ACPowerFlowResidual, ACRectangularCIResidual, ACMixedCPBResidual},
    J::Union{ACPowerFlowJacobian, ACRectangularCIJacobian, ACMixedCPBJacobian},
    refinement_threshold::Float64 = DEFAULT_REFINEMENT_THRESHOLD,
    refinement_eps::Float64 = DEFAULT_REFINEMENT_EPS,
)
    copyto!(stateVector.r, residual.Rv)
    _set_Δx_nr!(
        stateVector,
        J,
        linSolveCache,
        NewtonRaphsonACPowerFlow(),
        refinement_threshold,
        refinement_eps,
    )
    # update x
    stateVector.x .+= stateVector.Δx_nr
    # update data's fields (the bus angles/voltages) to match x, and update the residual.
    # do this BEFORE updating the Jacobian. The Jacobian computation uses data's fields, not x.
    residual(stateVector.x, time_step)
    # update jacobian.
    J(time_step)
    return
end

"""Does a single iteration of Newton-Raphson with Iwamoto step control.
Computes the Newton step, takes a full trial step, and checks whether the
residual norm decreased. If not, computes an optimal damping multiplier `μ`
and applies a damped step instead. When the damped step also fails to reduce
the residual, the step is reverted to avoid divergence.

Returns `true` if the step made progress (residual decreased), `false` if
the step was reverted. Consecutive reverts signal stagnation and the caller
should terminate early."""
function _iwamoto_step(time_step::Int,
    stateVector::StateVectorCache,
    linSolveCache::KLULinSolveCache{J_INDEX_TYPE},
    residual::Union{ACPowerFlowResidual, ACRectangularCIResidual, ACMixedCPBResidual},
    J::Union{ACPowerFlowJacobian, ACRectangularCIJacobian, ACMixedCPBJacobian},
    refinement_threshold::Float64 = DEFAULT_REFINEMENT_THRESHOLD,
    refinement_eps::Float64 = DEFAULT_REFINEMENT_EPS,
)::Bool
    # Save pre-step residual f into stateVector.r
    copyto!(stateVector.r, residual.Rv)
    # Compute Newton step Δx_nr = -J⁻¹f
    _set_Δx_nr!(
        stateVector,
        J,
        linSolveCache,
        NewtonRaphsonACPowerFlow(),
        refinement_threshold,
        refinement_eps,
    )
    # Take full trial step: x += Δx_nr
    stateVector.x .+= stateVector.Δx_nr
    # Evaluate trial residual b = F(x + Δx)
    residual(stateVector.x, time_step)

    # Compute gram scalars for Iwamoto criterion
    g0 = dot(stateVector.r, stateVector.r)
    g1 = dot(stateVector.r, residual.Rv)
    g2 = dot(residual.Rv, residual.Rv)

    if g2 < g0
        # Full step reduced residual — accept it (μ = 1).
        @debug "Iwamoto: full step accepted (g₂/g₀ = $(g2/g0))"
        J(time_step)
        return true
    end

    # Full step did not reduce residual — compute optimal μ.
    μ = _iwamoto_multiplier(g0, g1, g2)
    @debug "Iwamoto: damped step μ = $μ (g₂/g₀ = $(g2/g0))"
    # Undo full step and apply damped step.
    stateVector.x .-= stateVector.Δx_nr
    stateVector.x .+= μ .* stateVector.Δx_nr
    # Re-evaluate residual at damped point.
    residual(stateVector.x, time_step)
    # Check whether the damped step actually improved the residual.
    g_damped = dot(residual.Rv, residual.Rv)
    if g_damped >= g0
        # Damped step did not improve — revert to pre-step state.
        @debug "Iwamoto: damped step did not reduce residual " *
               "(g_damped/g₀ = $(g_damped/g0), μ = $μ); reverting"
        stateVector.x .-= μ .* stateVector.Δx_nr
        residual(stateVector.x, time_step)
        return false
    end
    # Damped step improved — accept it.
    J(time_step)
    return true
end

# Formulation-dispatched voltage-magnitude validation, driven entirely by the
# per-formulation index list precomputed once on the residual. Polar indexes
# the state as `[|V|, θ, …]` (`x[2i-1]` = |V|, PQ only); rectangular CI and
# mixed CPB states are `(e, f, …)` per-bus blocks validating `e²+f² ∈
# [min², max²]` over PQ/PV.
function _validate_state_magnitudes(
    r::ACPowerFlowResidual,
    x::Vector{Float64},
    range::MinMax,
    i::Int64,
)
    validate_voltage_magnitudes(x, r.validate_indices, range, i)
    return
end

function _validate_state_magnitudes(
    r::ACRectangularCIResidual,
    x::Vector{Float64},
    range::MinMax,
    i::Int64,
)
    _validate_squared_voltage_magnitudes(x, r.validate_offsets, range, i)
    return
end

function _validate_state_magnitudes(
    r::ACMixedCPBResidual,
    x::Vector{Float64},
    range::MinMax,
    i::Int64,
)
    _validate_squared_voltage_magnitudes(x, r.validate_offsets, range, i)
    return
end

"""Runs the full `NewtonRaphsonACPowerFlow`.
# Keyword arguments:
- `maxIterations::Int`: maximum iterations. Default: $DEFAULT_NR_MAX_ITER.
- `tol::Float64`: tolerance. The iterative search ends when `norm(abs.(residual)) < tol`.
    Default: $DEFAULT_NR_TOL.
- `refinement_threshold::Float64`: If the solution to `J_x Δx = r` satisfies
    `norm(J_x Δx - r, 1)/norm(r, 1) > refinement_threshold`, do iterative refinement to
    improve the accuracy. Default: $DEFAULT_REFINEMENT_THRESHOLD.
- `refinement_eps::Float64`: run iterative refinement on `J_x Δx = r` until
    `norm(Δx_{i}-Δx_{i+1}, 1)/norm(r,1) < refinement_eps`. Default:
    $DEFAULT_REFINEMENT_EPS """
function _run_power_flow_method(time_step::Int,
    stateVector::StateVectorCache,
    linSolveCache::KLULinSolveCache{J_INDEX_TYPE},
    residual::Union{ACPowerFlowResidual, ACRectangularCIResidual, ACMixedCPBResidual},
    J::Union{ACPowerFlowJacobian, ACRectangularCIJacobian, ACMixedCPBJacobian},
    ::Type{NewtonRaphsonACPowerFlow};
    maxIterations::Int = DEFAULT_NR_MAX_ITER,
    tol::Float64 = DEFAULT_NR_TOL,
    refinement_threshold::Float64 = DEFAULT_REFINEMENT_THRESHOLD,
    refinement_eps::Float64 = DEFAULT_REFINEMENT_EPS,
    validate_voltage_magnitudes::Bool = DEFAULT_VALIDATE_VOLTAGES,
    vm_validation_range::MinMax = DEFAULT_VALIDATION_RANGE,
    iwamoto::Bool = false,
    bail_on_stagnation::Bool = false,
    detect_stagnation::Bool = true,
    _ignored...,  # absorb unknown keys from caller without error
)
    validate_vms = validate_voltage_magnitudes
    i, converged = 1, false
    consecutive_reverts = 0
    monitor_ρ = get_compute_fixed_point_spectral_radius(J.data)
    status::ACPowerFlowSolveStatus = ACPowerFlowSolveStatus.RUNNING
    F_window = Float64[]
    ρ_window = Float64[]
    κ_window = Float64[]
    while i < maxIterations && !converged
        if iwamoto
            made_progress = _iwamoto_step(
                time_step,
                stateVector,
                linSolveCache,
                residual,
                J,
                refinement_threshold,
                refinement_eps,
            )
            if made_progress
                consecutive_reverts = 0
            else
                consecutive_reverts += 1
                if consecutive_reverts >= IWAMOTO_MAX_REVERTS
                    @debug "Iwamoto: $consecutive_reverts consecutive reverted steps; terminating early"
                    break
                end
            end
        else
            _simple_step(
                time_step,
                stateVector,
                linSolveCache,
                residual,
                J,
                refinement_threshold,
                refinement_eps,
            )
        end
        validate_vms && _validate_state_magnitudes(
            residual,
            stateVector.x,
            vm_validation_range,
            i,
        )
        ρ_now::Union{Float64, Nothing} = nothing
        condest_now::Union{Float64, Nothing} = nothing
        if monitor_ρ
            ρ_now, _, condest_now =
                _fixed_point_spectral_radius!(J.data, residual, J, time_step)
            _log_diagnostics("NR iter $i", ρ_now, residual.Rv, condest_now)
        end
        F_inf_now = norm(residual.Rv, Inf)
        if detect_stagnation && status === ACPowerFlowSolveStatus.RUNNING
            kind = _check_stagnation!(
                F_window, ρ_window, κ_window, F_inf_now, ρ_now, condest_now)
            if kind === :fixed_point
                msg, status = _stagnation_diagnostic(
                    J.data, time_step, residual.Rv, J.Jv; condest = condest_now)
                @info "NR stagnated at fixed point: ‖F‖_∞ = " *
                      "$(siground(F_inf_now)), stable across " *
                      "$(STAGNATION_WINDOW) iterations" * msg
            elseif kind === :limit_cycle
                msg, status = _limit_cycle_diagnostic(
                    residual.Rv, ρ_window, κ_window)
                @info "NR in limit cycle: ‖F‖_∞ = " *
                      "$(siground(F_inf_now)) across " *
                      "$(STAGNATION_WINDOW) iterations" * msg
            end
        end
        if status !== ACPowerFlowSolveStatus.RUNNING && bail_on_stagnation
            break
        end
        converged = F_inf_now < tol
        if !converged
            i += 1
        end
    end
    if converged
        status = ACPowerFlowSolveStatus.CONVERGED
    elseif status === ACPowerFlowSolveStatus.RUNNING
        status = ACPowerFlowSolveStatus.FAILED
    end
    return converged, i, status
end

"""Runs the full `TrustRegionNRMethod`.
# Keyword arguments:
- `maxIterations::Int`: maximum iterations. Default: $DEFAULT_NR_MAX_ITER.
- `tol::Float64`: tolerance. The iterative search ends when `maximum(abs.(residual)) < tol`.
    Default: $DEFAULT_NR_TOL.
- `factor::Float64`: the trust region starts out with radius `factor*norm(x_0, 1)`,
    where `x_0` is our initial guess, taken from `data`. Default: $DEFAULT_TRUST_REGION_FACTOR.
- `eta::Float64`: improvement threshold. If the observed improvement in our residual
    exceeds `eta` times the predicted improvement, we accept the new `x_i`.
    Default: $DEFAULT_TRUST_REGION_ETA.
- `iwamoto_fallback::Bool`: when a trust region step is rejected, attempt Iwamoto
    damping to salvage the step before reverting. Default: $DEFAULT_IWAMOTO_FALLBACK."""
function _run_power_flow_method(time_step::Int,
    stateVector::StateVectorCache,
    linSolveCache::KLULinSolveCache{J_INDEX_TYPE},
    residual::Union{ACPowerFlowResidual, ACRectangularCIResidual, ACMixedCPBResidual},
    J::Union{ACPowerFlowJacobian, ACRectangularCIJacobian, ACMixedCPBJacobian},
    ::Type{TrustRegionACPowerFlow};
    maxIterations::Int = DEFAULT_NR_MAX_ITER,
    tol::Float64 = DEFAULT_NR_TOL,
    factor::Float64 = DEFAULT_TRUST_REGION_FACTOR,
    eta::Float64 = DEFAULT_TRUST_REGION_ETA,
    autoscale::Bool = DEFAULT_AUTOSCALE,
    iwamoto_fallback::Bool = DEFAULT_IWAMOTO_FALLBACK,
    validate_voltage_magnitudes::Bool = DEFAULT_VALIDATE_VOLTAGES,
    vm_validation_range::MinMax = DEFAULT_VALIDATION_RANGE,
    bail_on_stagnation::Bool = false,
    detect_stagnation::Bool = true,
    _ignored...,  # absorb unknown keys from caller without error
)
    validate_vms = validate_voltage_magnitudes

    if eta > 1.0 || eta < 0.0
        @warn("η = $eta is outside [0, 1]") # eta is set to 2.0 in one test.
    end

    if autoscale
        for i in 1:length(stateVector.x)
            stateVector.d[i] = norm(view(J.Jv, :, i))
            if stateVector.d[i] == 0.0
                stateVector.d[i] = 1.0
            end
        end
    end

    delta = norm(stateVector.x) > 0 ? factor * norm(stateVector.x) : factor
    delta_max = DEFAULT_TRUST_REGION_DELTA_MAX_FACTOR * delta
    i, converged = 0, false
    residualSize = dot(residual.Rv, residual.Rv)
    linf = norm(residual.Rv, Inf)
    @debug "initially: sum of squares $(siground(residualSize)), L ∞ norm $(siground(linf)), Δ $(siground(delta))"

    monitor_ρ = get_compute_fixed_point_spectral_radius(J.data)
    status::ACPowerFlowSolveStatus = ACPowerFlowSolveStatus.RUNNING
    F_window = Float64[]
    ρ_window = Float64[]
    κ_window = Float64[]
    # Cache ρ/κ̂ across iterations so we don't recompute when TR rejects a step
    # (x and J unchanged on rejection ⇒ ρ and κ̂ unchanged). The cached values
    # still feed the stagnation windows so a stretch of rejections registers.
    ρ_cached::Union{Float64, Nothing} = nothing
    condest_cached::Union{Float64, Nothing} = nothing
    while i < maxIterations && !converged
        delta, accepted = _trust_region_step(
            time_step,
            stateVector,
            linSolveCache,
            residual,
            J,
            delta,
            delta_max,
            eta,
            autoscale,
            iwamoto_fallback,
        )
        validate_vms && _validate_state_magnitudes(
            residual,
            stateVector.x,
            vm_validation_range,
            i,
        )
        if monitor_ρ
            if accepted
                ρ_cached, _, condest_cached =
                    _fixed_point_spectral_radius!(J.data, residual, J, time_step)
                _log_diagnostics("TR iter $i", ρ_cached, residual.Rv, condest_cached)
            else
                @info "TR iter $i: step rejected (state unchanged; ρ/κ̂ skipped)"
            end
        end
        F_inf_now = norm(residual.Rv, Inf)
        if detect_stagnation && status === ACPowerFlowSolveStatus.RUNNING
            kind = _check_stagnation!(
                F_window, ρ_window, κ_window, F_inf_now, ρ_cached, condest_cached)
            if kind === :fixed_point
                msg, status = _stagnation_diagnostic(
                    J.data, time_step, residual.Rv, J.Jv; condest = condest_cached)
                @info "TR stagnated at fixed point: ‖F‖_∞ = " *
                      "$(siground(F_inf_now)), stable across " *
                      "$(STAGNATION_WINDOW) iterations" * msg
            elseif kind === :limit_cycle
                msg, status = _limit_cycle_diagnostic(
                    residual.Rv, ρ_window, κ_window)
                @info "TR in limit cycle: ‖F‖_∞ = " *
                      "$(siground(F_inf_now)) across " *
                      "$(STAGNATION_WINDOW) iterations" * msg
            end
        end
        if status !== ACPowerFlowSolveStatus.RUNNING && bail_on_stagnation
            break
        end
        converged = F_inf_now < tol
        if !converged
            i += 1
        end
    end
    if converged
        status = ACPowerFlowSolveStatus.CONVERGED
    elseif status === ACPowerFlowSolveStatus.RUNNING
        status = ACPowerFlowSolveStatus.FAILED
    end
    return converged, i, status
end

# Hardcoded escalation thresholds for AdaptiveACPowerFlow. Calibrated against
# CATS multi-period traces: TR's diverging time steps all had two consecutive
# iterates with ρ ≥ 1 by iter 3–4, while converging time steps never crossed
# ρ ≥ 1 after iter 0.
const ADAPTIVE_ρ_THRESHOLD = 1.0
const ADAPTIVE_ρ_CONSECUTIVE = 2

"""Driver for the AdaptiveACPowerFlow method: runs Trust Region with a
spectral-radius watchdog, and escalates to Levenberg-Marquardt (reusing the
current iterate as the starting point) if `ρ ≥ 1` for
`$ADAPTIVE_ρ_CONSECUTIVE` consecutive iterations."""
function _newton_power_flow(
    pf::AbstractACPowerFlow{AdaptiveACPowerFlow},
    data::ACPowerFlowData,
    time_step::Int64;
    tol::Float64 = DEFAULT_NR_TOL,
    maxIterations::Int = DEFAULT_NR_MAX_ITER,
    validate_voltage_magnitudes::Bool = DEFAULT_VALIDATE_VOLTAGES,
    vm_validation_range::MinMax = DEFAULT_VALIDATION_RANGE,
    factor::Float64 = DEFAULT_TRUST_REGION_FACTOR,
    eta::Float64 = DEFAULT_TRUST_REGION_ETA,
    autoscale::Bool = DEFAULT_AUTOSCALE,
    iwamoto_fallback::Bool = DEFAULT_IWAMOTO_FALLBACK,
    λ_0::Float64 = DEFAULT_λ_0,
    marquardt_scaling::Union{Bool, Nothing} = nothing,
    x0::Union{Vector{Float64}, Nothing} = nothing,
    detect_stagnation::Bool = true,
    extra_kwargs...,
)
    init_kwargs = if isnothing(x0)
        (; validate_voltage_magnitudes, vm_validation_range)
    else
        (; validate_voltage_magnitudes, vm_validation_range, x0)
    end
    residual, J, x0_init = initialize_power_flow_variables(
        pf, data, time_step; init_kwargs...)
    converged = norm(residual.Rv, Inf) < tol

    ρ_x0, _, condest_x0 = _fixed_point_spectral_radius!(data, residual, J, time_step)
    _log_diagnostics("x0 (time_step $time_step)", ρ_x0, residual.Rv, condest_x0)

    i = 0
    x_final = x0_init
    escalated = false
    status::ACPowerFlowSolveStatus = ACPowerFlowSolveStatus.RUNNING
    if !converged
        linSolveCache = KLULinSolveCache(J.Jv)
        symbolic_factor!(linSolveCache, J.Jv)
        stateVector = StateVectorCache(x0_init, residual.Rv)

        if autoscale
            for k in 1:length(stateVector.x)
                stateVector.d[k] = norm(view(J.Jv, :, k))
                if stateVector.d[k] == 0.0
                    stateVector.d[k] = 1.0
                end
            end
        end

        delta = norm(stateVector.x) > 0 ? factor * norm(stateVector.x) : factor
        delta_max = DEFAULT_TRUST_REGION_DELTA_MAX_FACTOR * delta
        consec_bad_ρ = 0
        F_window_tr = Float64[]
        ρ_window_tr = Float64[]
        κ_window_tr = Float64[]
        ρ = NaN
        condest = NaN

        # ---- Phase 1: Trust Region with spectral-radius watchdog ----
        while i < maxIterations && !converged && !escalated
            delta, accepted = _trust_region_step(
                time_step,
                stateVector,
                linSolveCache,
                residual,
                J,
                delta,
                delta_max,
                eta,
                autoscale,
                iwamoto_fallback,
            )
            validate_voltage_magnitudes && _validate_state_magnitudes(
                residual,
                stateVector.x,
                vm_validation_range,
                i,
            )
            if accepted
                ρ, _, condest = _fixed_point_spectral_radius!(data, residual, J, time_step)
                _log_diagnostics("Adaptive TR iter $i", ρ, residual.Rv, condest)
            else
                @info "Adaptive TR iter $i: step rejected (state unchanged; ρ/κ̂ skipped)"
            end

            F_inf_now = norm(residual.Rv, Inf)
            ρ_for_window = isfinite(ρ) ? ρ : nothing
            κ_for_window = isfinite(condest) ? condest : nothing
            if detect_stagnation
                kind = _check_stagnation!(
                    F_window_tr, ρ_window_tr, κ_window_tr,
                    F_inf_now, ρ_for_window, κ_for_window)
                if kind === :fixed_point
                    msg, status = _stagnation_diagnostic(
                        J.data, time_step, residual.Rv, J.Jv; condest = condest)
                    @info "Adaptive TR stagnated at fixed point: ‖F‖_∞ = " *
                          "$(siground(F_inf_now)), stable across " *
                          "$(STAGNATION_WINDOW) iterations; stopping before " *
                          "escalation" * msg
                    break
                elseif kind === :limit_cycle
                    msg, status = _limit_cycle_diagnostic(
                        residual.Rv, ρ_window_tr, κ_window_tr)
                    @info "Adaptive TR in limit cycle: ‖F‖_∞ = " *
                          "$(siground(F_inf_now)) across " *
                          "$(STAGNATION_WINDOW) iterations; stopping before " *
                          "escalation" * msg
                    break
                end
            end

            consec_bad_ρ = ρ >= ADAPTIVE_ρ_THRESHOLD ? consec_bad_ρ + 1 : 0
            if consec_bad_ρ >= ADAPTIVE_ρ_CONSECUTIVE
                @info "Adaptive solver: ρ ≥ $(ADAPTIVE_ρ_THRESHOLD) for " *
                      "$(consec_bad_ρ) consecutive iters; escalating to " *
                      "LevenbergMarquardt at iter $i"
                escalated = true
                break
            end
            converged = norm(residual.Rv, Inf) < tol
            if !converged
                i += 1
            end
        end
        x_final = stateVector.x

        # ---- Phase 2: Levenberg-Marquardt handoff (reuses current iterate) ----
        if escalated && !converged
            use_scaling = something(marquardt_scaling, _default_marquardt_scaling(pf))
            ws = LMWorkspace(J.Jv; marquardt_scaling = use_scaling)
            lm_max = max(0, maxIterations - i)
            lm_converged, lm_iters, lm_status = _run_power_flow_method(
                time_step,
                x_final,
                residual,
                J,
                ws;
                tol,
                maxIterations = lm_max,
                λ_0,
                monitor_ρ = true,
                bail_on_stagnation = true,
                detect_stagnation,
                iter_offset = i + 1,
                extra_kwargs...,
            )
            converged = lm_converged
            status = lm_status
            i += lm_iters
        end
    end
    if converged
        status = ACPowerFlowSolveStatus.CONVERGED
    elseif status === ACPowerFlowSolveStatus.RUNNING
        status = ACPowerFlowSolveStatus.FAILED
    end
    _finalize_formulation!(pf, data, x_final, residual, time_step)
    return _finalize_power_flow(
        converged, i, "AdaptiveACPowerFlow", residual, data, J.Jv, time_step;
        status)
end

"""Log final residual, report convergence, compute optional post-processing factors,
and return `true`/`false`. Shared by all AC power flow drivers. `status` is a
fine-grained classification of the concluding state — one of `ACPowerFlowSolveStatus.CONVERGED`,
`ACPowerFlowSolveStatus.NON_ROOT_STATIONARY` (infeasibility), `ACPowerFlowSolveStatus.SINGULAR_SUBSPACE` (trapped on
near-singularity), `ACPowerFlowSolveStatus.FOLD` (curvature dominates), `ACPowerFlowSolveStatus.STAGNATED_OTHER`, or
`ACPowerFlowSolveStatus.FAILED` (iteration limit, no stagnation detected). The function returns
`true` iff `status == ACPowerFlowSolveStatus.CONVERGED`."""
function _finalize_power_flow(
    converged::Bool,
    i::Int,
    solver_name::String,
    residual::Union{ACPowerFlowResidual, ACRectangularCIResidual, ACMixedCPBResidual},
    data::ACPowerFlowData,
    Jv::SparseMatrixCSC{Float64, J_INDEX_TYPE},
    time_step::Int64;
    status::ACPowerFlowSolveStatus = ACPowerFlowSolveStatus.FAILED,
)
    @info("Final residual size: $(norm(residual.Rv, 2)) L2, $(norm(residual.Rv, Inf)) L∞.")
    if converged
        @info("The $solver_name solver converged after $i iterations [status: $status].")
        if get_calculate_loss_factors(data)
            _calculate_loss_factors(data, Jv, time_step)
        end
        if get_calculate_voltage_stability_factors(data)
            _calculate_voltage_stability_factors(data, Jv, time_step)
        end
        return true
    end
    msg = if status === ACPowerFlowSolveStatus.NON_ROOT_LOCAL_MIN
        "stopped at a non-root local minimum of ½‖F‖² (positive directional " *
        "curvature). The power flow equations are infeasible at this " *
        "operating point — consider PV→PQ reactive-limit switching, switched-" *
        "shunt regulation, dispatch changes, or topology checks"
    elseif status === ACPowerFlowSolveStatus.NON_ROOT_SADDLE
        "stopped at a non-root saddle of ½‖F‖² (negative directional " *
        "curvature). A different initial guess may find a root; consider " *
        "homotopy, a DC-power-flow seed, or a perturbed warm start"
    elseif status === ACPowerFlowSolveStatus.NON_ROOT_STATIONARY
        "stopped at a non-root stationary point of ½‖F‖² (curvature " *
        "inconclusive). The power flow equations are likely infeasible at " *
        "this operating point — consider PV→PQ reactive-limit switching, " *
        "switched-shunt regulation, dispatch changes, or topology checks"
    elseif status === ACPowerFlowSolveStatus.SINGULAR_SUBSPACE
        "stopped trapped on a near-singular subspace of J. The system is at " *
        "or past a fold/loadability boundary — homotopy/continuation may be " *
        "needed to track the solution branch"
    elseif status === ACPowerFlowSolveStatus.FOLD
        "stopped at a fold of the residual manifold: quadratic curvature " *
        "dominates the Newton step, so further linear iteration cannot " *
        "reduce ‖F‖. Try homotopy or a more robust formulation"
    elseif status === ACPowerFlowSolveStatus.LIMIT_CYCLE
        "settled into a period-≥2 limit cycle: the iteration map has no " *
        "fixed point in this neighborhood (typically a fold/saddle-node " *
        "bifurcation). Point-wise classifiers (local-min/saddle/fold) don't " *
        "apply. Try homotopy/continuation or relax the operating point"
    elseif status === ACPowerFlowSolveStatus.STAGNATED_OTHER
        "stagnated for unclear reasons (‖F‖_∞ flat across 3+ iterates but no " *
        "specific diagnostic threshold tripped); inspect the solver log"
    else
        "failed to converge"
    end
    # Stagnation-classified statuses get @warn (we know what happened, just
    # couldn't reach tol); plain FAILED / DIVERGED keep @error to match
    # historical behavior — divergence already logged its own @error inside LM,
    # but the duplicate at finalize is harmless.
    full_msg =
        "The $solver_name solver $msg after $i iterations " *
        "[status: $status]. Reporting non-convergence."
    if status === ACPowerFlowSolveStatus.FAILED ||
       status === ACPowerFlowSolveStatus.DIVERGED
        @error(full_msg)
    else
        @warn(full_msg)
    end
    return false
end

"""Formulation-specific post-Newton step. Polar needs nothing; the rectangular
CI formulation distributes the converged subnetwork slack into the bus
injection arrays."""
_finalize_formulation!(::ACPolarPowerFlow, data, x, residual, time_step) = nothing

function _finalize_formulation!(
    ::ACRectangularPowerFlow,
    data::ACPowerFlowData,
    x::Vector{Float64},
    residual::ACRectangularCIResidual,
    time_step::Int64,
)
    rect_finalize_bus_injections!(
        data, x, residual.bus_state_offset, residual.P_net_set,
        residual.bus_slack_participation_factors, residual.subnetworks,
        time_step,
    )
    return
end

function _finalize_formulation!(
    ::ACMixedPowerFlow,
    data::ACPowerFlowData,
    x::Vector{Float64},
    residual::ACMixedCPBResidual,
    time_step::Int64,
)
    mixed_finalize_bus_injections!(
        data, x, residual.bus_state_offset,
        residual.bus_slack_participation_factors, residual.subnetworks,
        residual.e_state, residual.f_state,
        time_step,
    )
    return
end

function _newton_power_flow(
    pf::AbstractACPowerFlow{T},
    data::ACPowerFlowData,
    time_step::Int64;
    # shared kwargs
    tol::Float64 = DEFAULT_NR_TOL,
    maxIterations::Int = DEFAULT_NR_MAX_ITER,
    validate_voltage_magnitudes::Bool = DEFAULT_VALIDATE_VOLTAGES,
    vm_validation_range::MinMax = DEFAULT_VALIDATION_RANGE,
    # NR-specific
    refinement_threshold::Float64 = DEFAULT_REFINEMENT_THRESHOLD,
    refinement_eps::Float64 = DEFAULT_REFINEMENT_EPS,
    iwamoto::Bool = false,
    # TR-specific
    factor::Float64 = DEFAULT_TRUST_REGION_FACTOR,
    eta::Float64 = DEFAULT_TRUST_REGION_ETA,
    autoscale::Bool = DEFAULT_AUTOSCALE,
    iwamoto_fallback::Bool = DEFAULT_IWAMOTO_FALLBACK,
    # initialize_power_flow_variables
    x0::Union{Vector{Float64}, Nothing} = nothing,
    _ignored...,
) where {T <: Union{TrustRegionACPowerFlow, NewtonRaphsonACPowerFlow}}

    # setup: common code
    init_kwargs = if isnothing(x0)
        (; validate_voltage_magnitudes, vm_validation_range)
    else
        (; validate_voltage_magnitudes, vm_validation_range, x0)
    end
    residual, J, x0_init = initialize_power_flow_variables(
        pf, data, time_step; init_kwargs...)
    converged = norm(residual.Rv, Inf) < tol

    if get_compute_fixed_point_spectral_radius(data)
        ρ, _, condest = _fixed_point_spectral_radius!(data, residual, J, time_step)
        _log_diagnostics("x0 (time_step $time_step)", ρ, residual.Rv, condest)
        if ρ >= 1.0
            @warn "fixed-point spectral radius ρ = $(round(ρ; sigdigits = 4)) ≥ 1 " *
                  "at x0 (time_step $time_step); Newton-Raphson may not converge " *
                  "from this starting point"
        end
    end

    i = 0
    x_final = x0_init
    status::ACPowerFlowSolveStatus =
        converged ? ACPowerFlowSolveStatus.CONVERGED : ACPowerFlowSolveStatus.RUNNING
    if !converged
        linSolveCache = KLULinSolveCache(J.Jv)
        symbolic_factor!(linSolveCache, J.Jv)
        stateVector = StateVectorCache(x0_init, residual.Rv)
        converged, i, status = _run_power_flow_method(
            time_step,
            stateVector,
            linSolveCache,
            residual,
            J,
            T;
            tol,
            maxIterations,
            validate_voltage_magnitudes,
            vm_validation_range,
            refinement_threshold,
            refinement_eps,
            iwamoto,
            factor,
            eta,
            autoscale,
            iwamoto_fallback,
        )
        x_final = stateVector.x
    end
    _finalize_formulation!(pf, data, x_final, residual, time_step)
    return _finalize_power_flow(
        converged, i, string(T), residual, data, J.Jv, time_step;
        status)
end
