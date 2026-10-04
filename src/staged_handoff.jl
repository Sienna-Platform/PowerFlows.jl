"""
    _validate_handoff_solver(handoff_solver, solver_label)

Throw an `ArgumentError` that names `solver_label` when `handoff_solver` is not `NoHandoff`,
`NewtonRaphsonACPowerFlow`, `TrustRegionACPowerFlow`, or `LevenbergMarquardtACPowerFlow`.
"""
function _validate_handoff_solver(handoff_solver, solver_label::String)
    if !(
        handoff_solver === NoHandoff ||
        handoff_solver === NewtonRaphsonACPowerFlow ||
        handoff_solver === TrustRegionACPowerFlow ||
        handoff_solver === LevenbergMarquardtACPowerFlow
    )
        throw(
            ArgumentError(
                "$(solver_label): unsupported handoff_solver $(handoff_solver). Must be " *
                "NoHandoff (stage only), NewtonRaphsonACPowerFlow, TrustRegionACPowerFlow, or " *
                "LevenbergMarquardtACPowerFlow.",
            ),
        )
    end
    return nothing
end

"""The stage exit tolerance: `handoff_tol` when a handoff solver refines the result to `tol`,
else `tol`."""
_stage_tol(::Type{NoHandoff}, tol, handoff_tol) = tol
_stage_tol(::Type{<:ACPowerFlowSolverType}, tol, handoff_tol) = handoff_tol

# The `Jv` argument for `_finalize_power_flow`: `J.Jv`, or `nothing` when the stage driver did not
# build `J`.
_finalize_jv(::Nothing) = nothing
_finalize_jv(J) = J.Jv

"""A handoff's linear-solver cache, kept by a stage cache across solves of the same data. Valid
while the Jacobian comes from the `structure` memo it was analyzed for, so repeated solves skip
the symbolic analysis."""
struct HandoffLinearCache{C <: PNM.LinearSolverCache}
    structure::ACJacobianStructureCache
    cache::C
end

# The handoff runs on the stage's own `J` with the polar NR solve's linear-solver setup: a KLU cache
# seeded with the structure memo's lean-LU plan (planned at flat on first use), so it pivots like
# the plain NR solve and skips the symbolic analysis.
function _new_handoff_cache(
    pf::AbstractACPowerFlow,
    linear_solver::Union{Nothing, AbstractString},
    J::Union{ACPowerFlowJacobian, ACRectangularCIJacobian, ACMixedCPBJacobian},
    data::ACPowerFlowData,
    time_step::Int64,
)
    backend = resolve_linear_solver_backend(linear_solver)
    hcache = _polar_jacobian_cache(backend, J.Jv)
    _seed_handoff_cache!(data.ac_jacobian_structure_cache[], pf, hcache, J, data, time_step)
    return hcache
end

# An area-interchange relax cleared the memo mid-solve: no lean slot to seed from.
_seed_handoff_cache!(
    ::Nothing,
    ::AbstractACPowerFlow,
    hcache,
    J,
    ::ACPowerFlowData,
    ::Int64,
) =
    symbolic_factor!(hcache, J.Jv)
_seed_handoff_cache!(
    ::ACJacobianStructureCache,
    pf::AbstractACPowerFlow,
    hcache,
    J,
    data::ACPowerFlowData,
    time_step::Int64,
) = _symbolic_step!(pf, hcache, J, data, time_step)

# A stage cache without a handoff slot (GA, fixed-Jacobian FD) builds the cache per solve.
_handoff_linear_cache!(
    ::Any,
    pf::AbstractACPowerFlow,
    data::ACPowerFlowData,
    J,
    time_step::Int64,
    linear_solver,
) = _new_handoff_cache(pf, linear_solver, J, data, time_step)

# Reuse `slot`'s cache when its structure memo is the one `J` was built from. The kept Numeric
# is dropped (the last solve's pivot order can hit a zero pivot after a bus-type change) and the
# lean path re-armed for this solve's bus types, as the polar NR cache's reuse does.
function _reuse_handoff_cache!(
    slot::Base.RefValue{<:Union{Nothing, HandoffLinearCache}},
    kept::HandoffLinearCache,
    structure::ACJacobianStructureCache,
    pf::AbstractACPowerFlow,
    data::ACPowerFlowData,
    J,
    time_step::Int64,
    linear_solver::Union{Nothing, AbstractString},
)
    if kept.structure === structure
        hcache = kept.cache
        _drop_numeric!(hcache)
        _resume_lean!(hcache)
        _align_lean_plan!(hcache, structure.lean, view(data.bus_type, :, time_step))
        return hcache
    end
    return _reuse_handoff_cache!(
        slot, nothing, structure, pf, data, J, time_step, linear_solver)
end

function _reuse_handoff_cache!(
    slot::Base.RefValue{<:Union{Nothing, HandoffLinearCache}},
    ::Nothing,
    structure::ACJacobianStructureCache,
    pf::AbstractACPowerFlow,
    data::ACPowerFlowData,
    J,
    time_step::Int64,
    linear_solver::Union{Nothing, AbstractString},
)
    hcache = _new_handoff_cache(pf, linear_solver, J, data, time_step)
    slot[] = HandoffLinearCache(structure, hcache)
    return hcache
end

# An area-interchange relax cleared the memo mid-solve: nothing to key the kept cache on.
_reuse_handoff_cache!(
    ::Base.RefValue{<:Union{Nothing, HandoffLinearCache}},
    ::Any,
    ::Nothing,
    pf::AbstractACPowerFlow,
    data::ACPowerFlowData,
    J,
    time_step::Int64,
    linear_solver::Union{Nothing, AbstractString},
) = _new_handoff_cache(pf, linear_solver, J, data, time_step)

# As `_newton_power_flow`: a handoff that fails on a reused pivot order (a lean plan or a kept
# KLU numeric) is rerun once from the stage state on a fresh factorization, so the reuse never
# changes a solve's status.
function _run_handoff_newton!(
    handoff_solver::Type{<:ACPowerFlowSolverType},
    hcache::PNM.LinearSolverCache,
    sv::StateVectorCache,
    residual::Union{ACPowerFlowResidual, ACRectangularCIResidual, ACMixedCPBResidual},
    J::Union{ACPowerFlowJacobian, ACRectangularCIJacobian, ACMixedCPBJacobian},
    data::ACPowerFlowData,
    time_step::Int64,
    tol::Float64,
)
    run() = _run_power_flow_method(
        time_step, sv, hcache, residual, J, data, handoff_solver;
        tol, maxIterations = DEFAULT_NR_MAX_ITER,
    )
    _reuses_pivot_order(hcache) || return run()
    x_start = copy(sv.x)
    _save_solve_start!(residual)
    counts = _retry_start(hcache)
    converged, i = run()
    if converged || _pivoted_fresh_at_start(hcache, counts)
        return converged, i
    end
    @debug "handoff failed on a reused pivot order; retrying on a fresh factorization" time_step
    PNM.KLUWrapper.cold_restart!(hcache)
    _restart_from!(sv, residual, J, data, x_start, time_step)
    converged, i_cold = run()
    return converged, i + i_cold
end

"""
    _maybe_handoff!(handoff_solver, pf, sv, residual, J, data, time_step, tol, linear_solver,
                    solver_name, stage_iters) -> (converged::Bool, handoff_iters::Int)

Refine the stage state `sv.x` to `tol` with `handoff_solver` (NR, TR, or LM). The method does
nothing for `NoHandoff` or when the stage state already meets `tol`. It updates `sv.x`,
`residual`, and `J` in place, so the caller finalizes the refined state.
"""
function _maybe_handoff!(
    ::Type{NoHandoff},
    pf::AbstractACPowerFlow,
    sv::StateVectorCache,
    residual::Union{ACPowerFlowResidual, ACRectangularCIResidual, ACMixedCPBResidual},
    J::Union{Nothing, ACPowerFlowJacobian, ACRectangularCIJacobian, ACMixedCPBJacobian},
    ::ACPowerFlowData,
    time_step::Int64,
    tol::Float64,
    linear_solver::Union{Nothing, AbstractString},
    solver_name::String,
    stage_iters::Int,
)
    return (norm(residual.Rv, Inf) < tol, 0)
end

function _maybe_handoff!(
    handoff_solver::Type{<:ACPowerFlowSolverType},
    pf::AbstractACPowerFlow,
    sv::StateVectorCache,
    residual::Union{ACPowerFlowResidual, ACRectangularCIResidual, ACMixedCPBResidual},
    J::Union{Nothing, ACPowerFlowJacobian, ACRectangularCIJacobian, ACMixedCPBJacobian},
    data::ACPowerFlowData,
    time_step::Int64,
    tol::Float64,
    linear_solver::Union{Nothing, AbstractString},
    solver_name::String,
    stage_iters::Int,
)
    if norm(residual.Rv, Inf) < tol
        return (true, 0)
    end
    J(data, time_step)
    if handoff_solver === LevenbergMarquardtACPowerFlow
        # LM's inner method takes the raw state vector + an LMWorkspace (a different signature
        # from NR/TR) and mutates x0 in place; see src/levenberg-marquardt.jl.
        ws = LMWorkspace(J.Jv; marquardt_scaling = _default_marquardt_scaling(pf))
        converged, i2 = _run_power_flow_method(
            time_step, sv.x, residual, J, data, ws;
            tol, maxIterations = DEFAULT_NR_MAX_ITER, λ_0 = DEFAULT_λ_0,
        )
    else
        hcache = _handoff_linear_cache!(
            data.solver_cache[], pf, data, J, time_step, linear_solver)
        converged, i2 = _run_handoff_newton!(
            handoff_solver, hcache, sv, residual, J, data, time_step, tol)
    end
    status = if converged
        "converged"
    else
        "did NOT converge"
    end
    @info "$solver_name: stage $stage_iters iters → handoff $(handoff_solver) " *
          "$status in $i2 iters."
    return (converged, i2)
end
