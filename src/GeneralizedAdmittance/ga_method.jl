# Driver for the generalized-admittance (PFPD) AC power flow (spec §4): the fixed-point
# stage with its exit singletons, best-iterate write-back, the polar residual check
# through the shared explicit-state sync, and the opt-in NR/TR/LM handoff.

abstract type GAStageExit end
struct GAConverged <: GAStageExit end
struct GAMaxIter <: GAStageExit end
struct GANonFinite <: GAStageExit end
struct GADiverged <: GAStageExit end
struct GAStagnated <: GAStageExit end

struct GASolveReport
    converged::Bool
    stage_exit::GAStageExit
    stage_iterations::Int
    handoff_iterations::Int
    best_gap::Float64
end

# Stagnation only matters when a handoff can rescue; without one, let slow runs finish.
_ga_stagnation_enabled(::Type{NoHandoff}) = false
_ga_stagnation_enabled(::Type{<:ACPowerFlowSolverType}) = true

# The polar residual's single-swing REF P row after the explicit sync is the negated sum
# of the island's other P rows (§4.3), so it can reach n_l × the gap while the gap (a
# per-bus max) sits at tol. With no handoff to polish, the stage must therefore hold the
# gap to tol / n_l for the residual ∞-norm check at tol to pass reliably.
_ga_stage_tol(::Type{NoHandoff}, tol::Float64, ::Float64, n_l::Int) = tol / n_l
_ga_stage_tol(handoff_solver::Type{<:ACPowerFlowSolverType}, tol::Float64,
    handoff_tol::Float64, ::Int) = _fd_stage_tol(handoff_solver, tol, handoff_tol)

function _ga_run_stage!(
    ws::GAWorkspace,
    cache::GeneralizedAdmittanceCache,
    np::GANodalPower,
    y::Vector{ComplexF64},
    part::GAPartition,
    dc,
    conv::GAConverterTerms,
    time_step::Int,
    max_iter::Int,
    stage_tol::Float64,
    stagnation::Bool,
)
    best_gap = Inf
    window_best = Inf
    for k in 1:max_iter
        dc_change = 0.0
        if k > 1
            dc_change = _ga_dc_substep!(dc, ws, np, part, conv, time_step)
        end
        gap = max(_ga_iterate!(ws, cache, np, y, part.Vset, n_v(part)), dc_change)
        if !isfinite(gap)
            return (GANonFinite(), k, best_gap)
        end
        if gap < best_gap
            best_gap = gap
            copyto!(ws.u_best, ws.u)
        end
        if gap <= stage_tol
            return (GAConverged(), k, best_gap)
        end
        if gap > GA_DIVERGENCE_FACTOR * best_gap
            return (GADiverged(), k, best_gap)
        end
        if stagnation && iszero(k % GA_STAGNATION_WINDOW)
            if best_gap > (1.0 - GA_STAGNATION_RATIO) * window_best
                return (GAStagnated(), k, best_gap)
            end
            window_best = best_gap
        end
    end
    return (GAMaxIter(), max_iter, best_gap)
end

function _ga_write_back!(
    data::ACPowerFlowData,
    part::GAPartition,
    u_l::Vector{ComplexF64},
    dc,
    time_step::Int,
)
    for (j, ix) in enumerate(part.l_ix)
        data.bus_magnitude[ix, time_step] = abs(u_l[j])
        data.bus_angles[ix, time_step] = angle(u_l[j])
    end
    _fd_converter_substep!(data, time_step)
    _ga_dc_finalize!(dc, data, time_step)
    return
end

# REF Q and PV Q close their own residual rows exactly (unit coefficients); the
# single-swing REF P row closes to the negated sum of the island's other P rows, exact once
# those converge. The sync plus one re-evaluation makes the polar residual the true
# residual of the best state.
function _ga_polar_state(data::ACPowerFlowData, time_step::Int64)
    x = calculate_x0(data, time_step)
    residual = ACPowerFlowResidual(data, time_step)
    residual(data, x, time_step)
    sv = StateVectorCache(x, residual.Rv)
    _sync_explicit_state!(sv, residual, data, time_step)
    residual(data, sv.x, time_step)
    return residual, sv
end

function _ga_check_consistency(
    ::GAConverged, ::Type{NoHandoff}, ::GANoDC, residual::ACPowerFlowResidual,
    tol::Float64,
)
    r = norm(residual.Rv, Inf)
    if r > GA_CONSISTENCY_FACTOR * tol
        row = argmax(abs.(residual.Rv))
        error(
            "GeneralizedAdmittanceACPowerFlow: gap met tol=$tol but the polar residual " *
            "is $r at row $row (bus index $(cld(row, 2))): formulation bug.",
        )
    end
    return
end

_ga_check_consistency(
    ::GAStageExit, ::Any, ::Any, ::ACPowerFlowResidual, ::Float64,
) = nothing

function _ga_solve(
    pf::ACPolarPowerFlow{GeneralizedAdmittanceACPowerFlow},
    data::ACPowerFlowData,
    time_step::Int64;
    tol::Float64 = DEFAULT_NR_TOL,
    maxIterations::Int = DEFAULT_GA_MAX_ITER,
    handoff_solver = NoHandoff,
    handoff_tol::Float64 = DEFAULT_FD_HANDOFF_TOL,
    linear_solver::Union{Nothing, AbstractString} = nothing,
    _ignored...,
)
    name = "GeneralizedAdmittanceACPowerFlow"
    _validate_handoff_solver(handoff_solver, name)
    part = GAPartition(data, time_step, _ga_vsc_ac_voltage_targets(data, time_step))
    conv = GAConverterTerms(size(data.bus_type, 1))
    _ga_add_lcc_terms!(conv, data, time_step)
    dc = _ga_dc_context(data, part, conv, time_step)
    np = GANodalPower(data, part, conv, time_step)
    cache = _get_or_build_ga_cache!(data, part)
    y = _ga_initial_shunts(cache.blocks, np, part, data, time_step)
    _ga_factor!(cache, y, part)
    ws = cache.ws
    _ga_u0!(ws, cache, _ga_slack_voltages(data, part, time_step))
    fill!(ws.i, zero(ComplexF64))
    exit, iters, best_gap = _ga_run_stage!(
        ws, cache, np, y, part, dc, conv, time_step,
        maxIterations, _ga_stage_tol(handoff_solver, tol, handoff_tol, n_l(part)),
        _ga_stagnation_enabled(handoff_solver),
    )
    @debug "GeneralizedAdmittance stage" exit iters best_gap
    if isfinite(best_gap)
        _ga_write_back!(data, part, ws.u_best, dc, time_step)
    end
    residual, sv = _ga_polar_state(data, time_step)
    need_factors =
        get_calculate_loss_factors(data) || get_calculate_voltage_stability_factors(data)
    J = nothing
    if handoff_solver !== NoHandoff || need_factors
        J = ACPowerFlowJacobian(data, residual, time_step)
    end
    converged, handoff_iters = _maybe_handoff!(
        handoff_solver, pf, sv, residual, J, data, time_step, tol, linear_solver, name,
        iters,
    )
    _ga_check_consistency(exit, handoff_solver, dc, residual, tol)
    if converged && need_factors
        J(data, time_step)
    end
    converged = _finalize_power_flow(
        converged, iters + handoff_iters, name, residual, data, _fd_finalize_jv(J),
        time_step,
    )
    return GASolveReport(converged, exit, iters, handoff_iters, best_gap)
end

function _newton_power_flow(
    pf::ACPolarPowerFlow{GeneralizedAdmittanceACPowerFlow},
    data::ACPowerFlowData,
    time_step::Int64;
    kwargs...,
)
    return _ga_solve(pf, data, time_step; kwargs...).converged
end
