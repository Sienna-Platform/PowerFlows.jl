_log_initial_residual(residual) =
    @debug "Initial residual size: " *
           "$(norm(residual.Rv, 2)) L2, " *
           "$(norm(residual.Rv, Inf)) L∞"

improve_x0(
    pf::ACPolarPowerFlow,
    data::ACPowerFlowData,
    residual::ACPowerFlowResidual,
    time_step::Int64,
) = improve_x0!(calculate_x0(data, time_step), pf, data, residual, time_step)

# `x0` holds `calculate_x0(data, time_step)` on entry and the chosen start on return.
function improve_x0!(x0::Vector{Float64},
    pf::ACPolarPowerFlow,
    data::ACPowerFlowData,
    residual::ACPowerFlowResidual,
    time_step::Int64,
)
    residual(data, x0, time_step)
    prev = findlast(@view(data.converged[1:(time_step - 1)]))
    if !isnothing(prev)
        newx0 = _previous_solution_start(x0, data, prev)
        _pick_better_x0(x0, newx0, time_step, residual, data, "previous converged solution")
    end
    if sum(abs, residual.Rv) > LARGE_RESIDUAL * length(residual.Rv) &&
       get_enhanced_flat_start(pf)
        newx0 = _enhanced_flat_start(x0, data, time_step)
        _pick_better_x0(x0, newx0, time_step, residual, data, "enhanced flat start")
    else
        @debug "skipping enhanced flat start"
    end
    # Warm starts (after a converged step, a contingency off a solved base) skip both stages;
    # the DC stage still runs on a large residual.
    large = _large_residual(residual)
    cold = isnothing(prev) && (large || _is_flat_start(residual, data, time_step))
    dc_taken = false
    if get_robust_power_flow(pf) && (large || cold)
        dc_taken = dc_power_flow_start!(x0, data, time_step, residual)
    else
        @debug "skipping running DC power flow fallback"
    end
    # GA from DC angles stagnates where NR from them converges (ACTIVSg10k flat: 40 GA
    # iterations, gap 1.6), so GA is the rescue for a start the DC stage did not improve.
    handoff_tol = get_solution_parameters(pf).handoff_tol
    if get_ga_flat_start(pf) && cold && !dc_taken && norm(residual.Rv, Inf) > handoff_tol
        newx0 = _ga_flat_start(x0, data, residual, time_step, handoff_tol)
        # The GA stage leaves `data` and `residual` at its own iterate, not at `x0`.
        residual(data, x0, time_step)
        _pick_better_x0(x0, newx0, time_step, residual, data,
            "generalized-admittance flat start")
    end

    _large_residual(residual) && _warn_large_initial_residual(residual, data, time_step)
    return x0
end

_large_residual(residual) = sum(abs, residual.Rv) > LARGE_RESIDUAL * length(residual.Rv)

# Every island's non-REF buses at one angle: the start of a case with no solved point, whose REF
# may keep a case-file angle. Reads the angles of the last iterate `residual` evaluated.
function _is_flat_start(
    residual::ACPowerFlowResidual,
    data::ACPowerFlowData,
    time_step::Int64,
)
    θ = residual.bus_state.θ
    bus_types = view(data.bus_type, :, time_step)
    for buses in values(residual.subnetworks)
        seen = false
        θ_flat = 0.0
        for ix in buses
            bus_types[ix] == PSY.ACBusTypes.REF && continue
            if !seen
                seen = true
                θ_flat = θ[ix]
            elseif θ[ix] != θ_flat
                return false
            end
        end
    end
    return true
end

# `improve_x0!` compares candidate starts only after a converged earlier step; on a cold start
# `_fused_x0!` decides whether a start stage runs.
function _x0_has_no_candidates(
    ::ACPolarPowerFlow,
    data::ACPowerFlowData,
    time_step::Int64,
)
    return !any(@view(data.converged[1:(time_step - 1)]))
end

function _warn_large_initial_residual(residual, data::ACPowerFlowData, time_step::Int64)
    # Tail rows (LCC/VSC/area) are not bus quantities: let the resolver label the index.
    lg_res, ix = findmax(abs, residual.Rv)
    lg_res_rounded = round(lg_res; sigdigits = 3)
    @warn "Initial guess provided results in a large initial residual of $lg_res_rounded. " *
          "Largest residual at $(_describe_residual_entry(residual, data, time_step, ix))"
    return
end

"""Rectangular analog of the polar [`improve_x0`](@ref): base flat start →
previous-converged-timestep warm start → enhanced flat start (gated on
`get_enhanced_flat_start(pf)`) → large-residual warning. No DC robust
fallback: `ACRectangularPowerFlow` has no `robust_power_flow` field and a
CI-aware DC fallback is out of scope (see the formulation/solver-split spec)."""
function improve_x0(pf::ACRectangularPowerFlow,
    data::ACPowerFlowData,
    residual::ACRectangularCIResidual,
    time_step::Int64,
)
    x0 = Vector{Float64}(undef, length(residual.Rv))
    rect_initial_state!(
        x0, data, residual.bus_state_offset, residual.bus_block_size, time_step,
    )
    residual(data, x0, time_step)
    prev = findlast(@view(data.converged[1:(time_step - 1)]))
    if !isnothing(prev)
        newx0 = copy(x0)
        _rect_fill_state!(newx0, data, residual.bus_state_offset, time_step, prev)
        _pick_better_x0(x0, newx0, time_step, residual, data, "previous converged solution")
    end
    if sum(abs, residual.Rv) > LARGE_RESIDUAL * length(residual.Rv) &&
       get_enhanced_flat_start(pf)
        newx0 = _enhanced_flat_start(x0, data, residual, time_step)
        _pick_better_x0(x0, newx0, time_step, residual, data, "enhanced flat start")
    else
        @debug "skipping enhanced flat start"
    end
    if sum(abs, residual.Rv) > LARGE_RESIDUAL * length(residual.Rv)
        lg_res, ix = findmax(abs.(residual.Rv))
        lg_res_rounded = round(lg_res; sigdigits = 3)
        @warn "Initial guess provided results in a large initial residual of " *
              "$lg_res_rounded (rectangular current-injection residual index $ix)."
    end
    return x0
end

"""MCPB analog of the rectangular [`improve_x0`](@ref): base flat start (via
[`mixed_initial_state!`](@ref)) → previous-converged-timestep warm start (via
[`_mixed_fill_state!`](@ref)) → enhanced flat start (gated on
`get_enhanced_flat_start(pf)`) → large-residual warning. Mirrors the
rectangular path verbatim, swapping `rect_initial_state!`→`mixed_initial_state!`,
`_rect_fill_state!`→`_mixed_fill_state!`, and the rectangular
`_enhanced_flat_start`→the MCPB `ACMixedCPBResidual` overload. No DC robust
fallback: `ACMixedPowerFlow` has no `robust_power_flow` field
(`get_robust_power_flow(::AbstractACPowerFlow) == false`)."""
function improve_x0(pf::ACMixedPowerFlow,
    data::ACPowerFlowData,
    residual::ACMixedCPBResidual,
    time_step::Int64,
)
    x0 = Vector{Float64}(undef, length(residual.Rv))
    mixed_initial_state!(
        x0, data, residual.bus_state_offset, residual.bus_block_size, time_step,
    )
    residual(data, x0, time_step)
    prev = findlast(@view(data.converged[1:(time_step - 1)]))
    if !isnothing(prev)
        newx0 = copy(x0)
        _mixed_fill_state!(newx0, data, residual.bus_state_offset, time_step, prev)
        _pick_better_x0(x0, newx0, time_step, residual, data, "previous converged solution")
    end
    if sum(abs, residual.Rv) > LARGE_RESIDUAL * length(residual.Rv) &&
       get_enhanced_flat_start(pf)
        newx0 = _enhanced_flat_start(x0, data, residual, time_step)
        _pick_better_x0(x0, newx0, time_step, residual, data, "enhanced flat start")
    else
        @debug "skipping enhanced flat start"
    end
    if sum(abs, residual.Rv) > LARGE_RESIDUAL * length(residual.Rv)
        lg_res, ix = findmax(abs.(residual.Rv))
        lg_res_rounded = round(lg_res; sigdigits = 3)
        @warn "Initial guess provided results in a large initial residual of " *
              "$lg_res_rounded (mixed current/power-balance residual index $ix)."
    end
    return x0
end

"""Replace `x0` by `newx0` when `newx0` has the smaller 1-norm residual; returns whether it did.
Requires `data` and `residual.Rv` to hold the evaluation at `x0` on entry, and leaves them
holding the evaluation at the returned `x0`, so neither point is evaluated twice."""
function _pick_better_x0(x0::Vector{Float64},
    newx0::Vector{Float64},
    time_step::Int64,
    residual::Union{ACPowerFlowResidual, ACRectangularCIResidual, ACMixedCPBResidual},
    data::ACPowerFlowData,
    improvement_method::String,
    success_level::Logging.LogLevel = Logging.Debug,
)
    residualSize = sum(abs, residual.Rv)
    residual(data, newx0, time_step)
    if sum(abs, residual.Rv) < residualSize
        Logging.@logmsg success_level "success: $improvement_method yields smaller residual"
        copyto!(x0, newx0)
        return true
    end
    @debug "no improvement from $improvement_method"
    residual(data, x0, time_step)
    return false
end

"""Run a DC power flow and see if that gives a better starting point for angles. If so, then
overwrite `x0` with the result of the DC power flow. If not, keep the original `x0`. Returns
whether `x0` changed."""
function dc_power_flow_start!(x0::Vector{Float64},
    data::ACPowerFlowData,
    time_step::Int64,
    residual::ACPowerFlowResidual,
)
    _dc_power_flow_fallback!(data, time_step)
    newx0 = calculate_x0(data, time_step)
    # The fallback overwrote `data`'s angles, so re-establish `_pick_better_x0`'s precondition.
    residual(data, x0, time_step)
    return _pick_better_x0(
        x0, newx0, time_step, residual, data, "DC power flow fallback", Logging.Info)
end

"""Calculate x0 from data."""
function calculate_x0(data::ACPowerFlowData,
    time_step::Int64)
    n_buses = size(data.bus_type, 1)
    dcn = get_dc_network(data)
    x0 = Vector{Float64}(undef,
        2 * n_buses + state_tail_length(data, dcn))
    # update_state! fills the area tail from `data.area_interchange.delta_p` (0.0 for a
    # freshly enrolled area -> genuine flat start; the last-solved ΔP_a for a warm re-solve).
    update_state!(x0, data, time_step)
    return x0
end

"""Use state variables from a previous converged time step (`prev`) as a
candidate starting point."""
function _previous_solution_start(
    x0::Vector{Float64},
    data::ACPowerFlowData,
    prev::Int64,
)
    newx0 = copy(x0)
    update_state!(newx0, data, prev)
    return newx0
end

function _enhanced_flat_start(
    x0::Vector{Float64},
    data::ACPowerFlowData,
    time_step::Int64,
)
    newx0 = copy(x0)
    bus_lookup = get_bus_lookup(data)
    bus_types = view(data.bus_type, :, time_step)
    for subnetwork_bus_axes in values(data.power_network_matrix.subnetwork_axes)
        members = [bus_lookup[ix] for ix in subnetwork_bus_axes[1]]
        ref = [i for i in members if bus_types[i] == PSY.ACBusTypes.REF]
        pv = [i for i in members if bus_types[i] == PSY.ACBusTypes.PV]
        pq = [i for i in members if bus_types[i] == PSY.ACBusTypes.PQ]
        if !isempty(ref)
            ref_bus_angle = sum(data.bus_angles[ref, time_step]) / length(ref)
            if !iszero(ref_bus_angle)
                newx0[2 .* vcat(pv, pq)] .= ref_bus_angle
            end
        end
        sources = vcat(pv, ref)
        (isempty(pq) || isempty(sources)) && continue
        newx0[2 .* pq .- 1] .= sum(data.bus_magnitude[sources, time_step]) / length(sources)
    end
    return newx0
end

"""Rectangular/MCPB analog of [`_enhanced_flat_start`](@ref): per subnetwork,
set PV/PQ bus angles to the mean REF-bus angle and PQ magnitudes to the mean PV and REF
setpoint magnitude, written back as `(e, f) = (Vm·cosθ, Vm·sinθ)`. PV buses
keep their setpoint magnitude (only the angle changes); REF blocks and the
PV `Q` / REF `(P,Q)` slots are left as in `x0`. Uses `residual.subnetworks`
(ref-bus index → member bus indices) for the partition. Identical for the
rectangular and MCPB layouts (both use 2-slot `(e, f)` PV/PQ blocks and never
touch a PV `Q` slot)."""
function _enhanced_flat_start(
    x0::Vector{Float64},
    data::ACPowerFlowData,
    residual::Union{ACRectangularCIResidual, ACMixedCPBResidual},
    time_step::Int64,
)
    newx0 = copy(x0)
    bus_types = view(data.bus_type, :, time_step)
    for (_, members) in residual.subnetworks
        ref = [i for i in members if bus_types[i] == PSY.ACBusTypes.REF]
        pv = [i for i in members if bus_types[i] == PSY.ACBusTypes.PV]
        pq = [i for i in members if bus_types[i] == PSY.ACBusTypes.PQ]
        (isempty(pv) && isempty(pq)) && continue
        ref_angle =
            if isempty(ref)
                0.0
            else
                sum(data.bus_angles[r, time_step] for r in ref) / length(ref)
            end
        sources = vcat(pv, ref)
        pq_vm = 0.0
        if !isempty(sources)
            pq_vm = sum(data.bus_magnitude[s, time_step] for s in sources) / length(sources)
        end
        for i in pv
            off = Int(residual.bus_state_offset[i])
            θ = ref_angle != 0.0 ? ref_angle : data.bus_angles[i, time_step]
            Vm = data.bus_magnitude[i, time_step]
            newx0[off] = Vm * cos(θ)
            newx0[off + 1] = Vm * sin(θ)
        end
        for i in pq
            off = Int(residual.bus_state_offset[i])
            θ = ref_angle != 0.0 ? ref_angle : data.bus_angles[i, time_step]
            Vm = data.bus_magnitude[i, time_step]
            if !isempty(sources)
                Vm = pq_vm
            end
            newx0[off] = Vm * cos(θ)
            newx0[off + 1] = Vm * sin(θ)
        end
    end
    return newx0
end

const _DC_FALLBACK_LOCK = ReentrantLock()

"""When solving AC power flows, if the initial guess has large residual, we run a DC power
flow as a fallback. This runs a DC power flow on `data::ACPowerFlowData` for the given
`time_step`, and writes the solution to `data.bus_angles`."""
function _dc_power_flow_fallback!(data::ACPowerFlowData, time_step::Int)
    # dev note: for DC, we can efficiently solve for all time_steps at once, and we want branch
    # flows. For AC fallback, we're only interested in the current time_step, and no branch flows
    solver_cache = get_aux_network_matrix(data).K
    # factored in constructor; no need to factor again (as long as network is same)
    valid_ix = get_valid_ix(data)
    p_inj =
        data.bus_active_power_injections[valid_ix, time_step] -
        data.bus_active_power_withdrawals[valid_ix, time_step] +
        data.bus_hvdc_net_power[valid_ix, time_step] +
        data.bus_phase_shift_injections[valid_ix]
    # PNM's KLUWrapper.KLULinSolveCache exposes solve! (in-place) instead of ldiv!.
    # The factored ABA is shared by every threaded time-step worker and KLU solves through
    # its numeric workspace. ponytail: one global lock; the fallback only runs on a large residual.
    @lock _DC_FALLBACK_LOCK PNM.solve!(solver_cache, p_inj)
    data.bus_angles[valid_ix, time_step] .= p_inj
    # The reduced solve is referenced to 0 at each ref bus, but the AC solve holds each
    # ref bus fixed at its stored angle: shift the warm start onto the AC reference so
    # arcs incident to a ref bus with nonzero stored angle don't start with a spurious
    # angle difference. Only this column — others may hold solved AC states.
    _shift_angles_to_stored_reference!(data, time_step)
end

"""
    _initialize_residual_x0(pf::ACPolarPowerFlow, data, time_step; kwargs...)
        -> (residual, x0_computed)

Build the polar residual and the (warm-started, validated) initial state vector WITHOUT
constructing the formulation Jacobian. Shared by `initialize_power_flow_variables` (which
adds the Jacobian) and by the fast-decoupled `:decoupled` driver, whose B′/B″ half-steps never use
the formulation Jacobian — that driver only materializes `J` when a handoff solver or loss/voltage-
stability factors are requested, so building it eagerly here would waste a full sparse-Jacobian
allocation + evaluation per solve (and per time step in multi-period runs).
"""
function _initialize_residual_x0(pf::ACPolarPowerFlow,
    data::ACPowerFlowData,
    time_step::Int64;
    x0::Union{Vector{Float64}, Nothing} = nothing,
    validate_voltage_magnitudes::Bool = DEFAULT_VALIDATE_VOLTAGES,
    vm_validation_range::MinMax = DEFAULT_VALIDATION_RANGE,
    _ignored...,
)
    residual = ACPowerFlowResidual(data, time_step)
    if isnothing(x0)
        x0_computed = improve_x0(pf, data, residual, time_step)
    else
        x0_computed = copy(x0)
        @warn "Using caller-provided x0; skipping improve_x0."
        residual(data, x0_computed, time_step)
    end
    _log_initial_residual(residual)

    validate_voltage_magnitudes && PowerFlows.validate_voltage_magnitudes(
        x0_computed,
        residual.validate_indices,
        vm_validation_range,
        0,
    )
    return residual, x0_computed
end

function initialize_power_flow_variables(pf::ACPolarPowerFlow{T},
    data::ACPowerFlowData,
    time_step::Int64;
    x0::Union{Vector{Float64}, Nothing} = nothing,
    validate_voltage_magnitudes::Bool = DEFAULT_VALIDATE_VOLTAGES,
    vm_validation_range::MinMax = DEFAULT_VALIDATION_RANGE,
    _ignored...,
) where {T <: ACPowerFlowSolverType}
    residual, x0_computed = _initialize_residual_x0(
        pf, data, time_step; x0, validate_voltage_magnitudes, vm_validation_range,
    )

    J = ACPowerFlowJacobian(data, residual, time_step)
    J(data, time_step)

    return residual, J, x0_computed
end

function initialize_power_flow_variables(pf::ACRectangularPowerFlow{T},
    data::ACPowerFlowData,
    time_step::Int64;
    x0::Union{Vector{Float64}, Nothing} = nothing,
    validate_voltage_magnitudes::Bool = DEFAULT_VALIDATE_VOLTAGES,
    vm_validation_range::MinMax = DEFAULT_VALIDATION_RANGE,
    _ignored...,
) where {T <: ACPowerFlowSolverType}
    residual = ACRectangularCIResidual(data, time_step)
    if isnothing(x0)
        x0_computed = improve_x0(pf, data, residual, time_step)
    else
        x0_computed = copy(x0)
        @warn "Using caller-provided x0; skipping improve_x0."
        residual(data, x0_computed, time_step)
    end
    _log_initial_residual(residual)
    J = ACRectangularCIJacobian(data, residual, time_step)
    return residual, J, x0_computed
end

function initialize_power_flow_variables(pf::ACMixedPowerFlow{T},
    data::ACPowerFlowData,
    time_step::Int64;
    x0::Union{Vector{Float64}, Nothing} = nothing,
    validate_voltage_magnitudes::Bool = DEFAULT_VALIDATE_VOLTAGES,
    vm_validation_range::MinMax = DEFAULT_VALIDATION_RANGE,
    _ignored...,
) where {T <: ACPowerFlowSolverType}
    residual = ACMixedCPBResidual(data, time_step)
    if isnothing(x0)
        x0_computed = improve_x0(pf, data, residual, time_step)
    else
        x0_computed = copy(x0)
        @warn "Using caller-provided x0; skipping improve_x0."
        residual(data, x0_computed, time_step)
    end
    _log_initial_residual(residual)
    J = ACMixedCPBJacobian(data, residual, time_step)
    return residual, J, x0_computed
end
