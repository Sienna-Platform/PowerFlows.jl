@enum GAStageExit::Int8 GAConverged GAMaxIter GANonFinite GADiverged GAStagnated

struct GASolveReport
    converged::Bool
    stage_exit::GAStageExit
    stage_iterations::Int
    handoff_iterations::Int
    best_gap::Float64
    refreshes::Int
end

# Stagnation only matters when a handoff can rescue; without one, let slow runs finish.
_ga_stagnation_enabled(::Type{NoHandoff}) = false
_ga_stagnation_enabled(::Type{<:ACPowerFlowSolverType}) = true

# Without a handoff the exit gap also bounds each island's signed P sum, which is the REF P
# row of the polar residual after the explicit sync; a handoff solver closes that row itself.
function _ga_exit_gap(::Type{NoHandoff}, bus_gap::Float64, ws::GAWorkspace)
    return max(bus_gap, maximum(abs, ws.psum; init = 0.0))
end
_ga_exit_gap(::Type{<:ACPowerFlowSolverType}, bus_gap::Float64, ::GAWorkspace) = bus_gap

# Shunts from the current iterate; moving the currents by Δy ⊙ u offsets the change of Yℓℓ
# at that iterate, so the next voltages stay close.
function _ga_refresh_shunts!(
    cache::GeneralizedAdmittanceCache,
    np::GANodalPower,
    y::Vector{ComplexF64},
    part::GAPartition,
    κ::Float64,
    u_s::Vector{ComplexF64},
    bus_lookup::Dict{Int, Int},
)
    ws = cache.ws
    _ga_ideal_shunts!(ws.y_new, ws, np, part)
    _ga_stiffen_pv!(ws.y_new, cache.blocks, n_v(part), κ)
    @inbounds for k in eachindex(y)
        ws.i[k] += (ws.y_new[k] - y[k]) * ws.u[k]
        y[k] = ws.y_new[k]
    end
    _ga_factor!(cache, y, part, bus_lookup)
    _ga_u0!(ws, cache, u_s)
    _ga_anderson_reset!(cache.aa, ws.i)
    return
end

function _ga_run_stage!(
    cache::GeneralizedAdmittanceCache,
    np::GANodalPower,
    y::Vector{ComplexF64},
    part::GAPartition,
    dc::GADCContext,
    conv::GAConverterTerms,
    u_s::Vector{ComplexF64},
    bus_lookup::Dict{Int, Int},
    time_step::Int,
    max_iter::Int,
    stage_tol::Float64,
    handoff_solver::Type{H},
) where {H}
    stagnation = _ga_stagnation_enabled(handoff_solver)
    ws = cache.ws
    aa = cache.aa
    _ga_anderson_reset!(aa, ws.i)
    best_gap = Inf
    window_best = Inf
    refresh_gap = 0.0
    segment_best = Inf
    stall_best = Inf
    stall = 0
    κ = GA_PV_STIFFNESS_FRACTION
    refreshes = 0
    for k in 1:max_iter
        dc_change = 0.0
        if k > 1
            dc_change = _ga_dc_substep!(dc, ws, np, part, conv, time_step)
        end
        copyto!(ws.i, aa.x)
        gap = max(_ga_iterate!(cache, np, y, part), dc_change)
        exit_gap = _ga_exit_gap(handoff_solver, gap, ws)
        if !isfinite(exit_gap)
            return (GANonFinite, k, best_gap, refreshes)
        end
        if exit_gap < best_gap
            best_gap = exit_gap
            copyto!(ws.u_best, ws.u)
        end
        if exit_gap <= stage_tol
            return (GAConverged, k, best_gap, refreshes)
        end
        segment_best = min(segment_best, gap)
        if gap > GA_DIVERGENCE_FACTOR * segment_best
            return (GADiverged, k, best_gap, refreshes)
        end
        # With a handoff the exit gap is the per-bus gap, so best_gap tracks it.
        if stagnation && iszero(k % GA_STAGNATION_WINDOW)
            if best_gap > (1.0 - GA_STAGNATION_RATIO) * window_best
                return (GAStagnated, k, best_gap, refreshes)
            end
            window_best = best_gap
        end
        if k == 1
            refresh_gap = gap
        end
        if gap < (1.0 - GA_STALL_GAIN) * stall_best
            stall_best = gap
            stall = 0
        else
            stall += 1
        end
        dropped = GA_REFRESH_DROP * gap <= refresh_gap
        if dropped || stall >= GA_STALL_ITERATIONS
            if dropped
                κ = 0.0
            else
                κ = max(GA_RESTIFFEN_GROWTH * κ,
                    GA_RESTIFFEN_FLOOR * GA_PV_STIFFNESS_FRACTION)
            end
            _ga_refresh_shunts!(cache, np, y, part, κ, u_s, bus_lookup)
            refreshes += 1
            refresh_gap = gap
            segment_best = Inf
            stall_best = Inf
            stall = 0
        else
            _ga_anderson_step!(aa, ws.i)
        end
    end
    return (GAMaxIter, max_iter, best_gap, refreshes)
end

function _ga_write_back!(
    data::ACPowerFlowData,
    part::GAPartition,
    u_l::Vector{ComplexF64},
    dc::GADCContext,
    time_step::Int,
)
    for (j, ix) in enumerate(part.l_ix)
        get_bus_magnitude(data)[ix, time_step] = abs(u_l[j])
        get_bus_angles(data)[ix, time_step] = angle(u_l[j])
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
    residual, x = _ga_eval_polar_residual(data, time_step)
    sv = StateVectorCache(x, residual.Rv)
    _sync_explicit_state!(sv, residual, data, time_step)
    residual(data, sv.x, time_step)
    return residual, sv
end

function _ga_check_consistency(
    exit::GAStageExit, ::Type{NoHandoff}, ::GANoDC, residual::ACPowerFlowResidual,
    tol::Float64,
)
    if exit != GAConverged
        return
    end
    r, row = findmax(abs, residual.Rv)
    if r > GA_CONSISTENCY_FACTOR * tol
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

_ga_partition(data::ACPowerFlowData, time_step::Int) =
    GAPartition(data, time_step, _ga_vsc_ac_voltage_targets(data, time_step))

# The first iterate is the voltages in `data`, as for every other solver. From the
# zero-current u0 (|u| ≈ 0.3, rotated by the PV stiffening on resistive ties), Anderson can
# move a near-unit-gain PV angle to a different power flow root.
function _ga_stage!(
    data::ACPowerFlowData,
    part::GAPartition,
    cache::GeneralizedAdmittanceCache,
    time_step::Int,
    max_iter::Int,
    stage_tol::Float64,
    handoff_solver::Type{H},
) where {H}
    conv = GAConverterTerms(size(get_bus_type(data), 1))
    _ga_add_lcc_terms!(conv, data, time_step)
    dc = _ga_dc_context(data, part, conv, time_step)
    np = GANodalPower(data, part, conv, time_step)
    y = _ga_initial_shunts(cache.blocks, np, part, data, time_step)
    _ga_stiffen_pv!(y, cache.blocks, n_v(part), GA_PV_STIFFNESS_FRACTION)
    bus_lookup = get_bus_lookup(data)
    _ga_factor!(cache, y, part, bus_lookup)
    ws = cache.ws
    u_s = _ga_slack_voltages(data, part, time_step)
    _ga_u0!(ws, cache, u_s)
    for (j, ix) in enumerate(part.l_ix)
        ws.u[j] =
            get_bus_magnitude(data)[ix, time_step] *
            cis(get_bus_angles(data)[ix, time_step])
    end
    mul!(ws.i, cache.blocks.Yll, ws.u)
    mul!(ws.i, cache.blocks.Yls, u_s, 1.0, 1.0)
    exit, iters, best_gap, refreshes = _ga_run_stage!(
        cache, np, y, part, dc, conv, u_s, bus_lookup, time_step,
        max_iter, stage_tol, handoff_solver,
    )
    @debug "GeneralizedAdmittance stage" exit iters best_gap refreshes
    return (; dc, ws, exit, iters, best_gap, refreshes)
end

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
    part = _ga_partition(data, time_step)
    (; dc, ws, exit, iters, best_gap, refreshes) = _ga_stage!(
        data, part, _get_or_build_ga_cache!(data, part), time_step, maxIterations,
        _stage_tol(handoff_solver, tol, handoff_tol), handoff_solver,
    )
    if isfinite(best_gap)
        _ga_write_back!(data, part, ws.u_best, dc, time_step)
    end
    residual, sv = _ga_polar_state(data, time_step)
    need_factors =
        get_calculate_loss_factors(data) || get_calculate_voltage_stability_factors(data)
    J = nothing
    if _fd_needs_handoff_jacobian(handoff_solver) || need_factors
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
        converged, iters + handoff_iters, name, residual, data, _finalize_jv(J),
        time_step,
    )
    return GASolveReport(converged, exit, iters, handoff_iters, best_gap, refreshes)
end

function _newton_power_flow(
    pf::ACPolarPowerFlow{GeneralizedAdmittanceACPowerFlow},
    data::ACPowerFlowData,
    time_step::Int64;
    kwargs...,
)
    return _ga_solve(pf, data, time_step; kwargs...).converged
end

# NR start from a GA stage run to `handoff_tol`: bus states from the best iterate, REF/PV
# slots closed by the explicit sync. The stage runs on a private cache and the VSC state is
# restored, so a rejected candidate leaves only the residual's own injection writes behind.
function _ga_flat_start(
    x0::Vector{Float64},
    data::ACPowerFlowData,
    residual::ACPowerFlowResidual,
    time_step::Int64,
    handoff_tol::Float64,
)
    dcn = get_dc_network(data)
    p_c, q_c, node_vdc = copy(dcn.p_c), copy(dcn.q_c), copy(dcn.node_vdc)
    part = _ga_partition(data, time_step)
    (; ws, exit, iters, best_gap) = _ga_stage!(
        data, part, _build_ga_cache(data, part), time_step, DEFAULT_GA_MAX_ITER,
        handoff_tol, NewtonRaphsonACPowerFlow)
    copyto!(dcn.p_c, p_c)
    copyto!(dcn.q_c, q_c)
    copyto!(dcn.node_vdc, node_vdc)
    @info "Generalized-admittance flat start: $exit after $iters " *
          "iterations, gap $best_gap."
    newx0 = copy(x0)
    if !isfinite(best_gap)
        return newx0
    end
    bus_types = view(get_bus_type(data), :, time_step)
    for (j, ix) in enumerate(part.l_ix)
        if bus_types[ix] == PSY.ACBusTypes.PQ
            newx0[2 * ix - 1] = abs(ws.u_best[j])
        end
        newx0[2 * ix] = angle(ws.u_best[j])
    end
    residual(data, newx0, time_step)
    sv = StateVectorCache(newx0, residual.Rv)
    _sync_explicit_state!(sv, residual, data, time_step)
    return sv.x
end
