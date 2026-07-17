# Fast/Fixed Decoupled Newton-Raphson (FDNR) — driver, settings, shared safeguards, the two FD
# iteration loops (polar `:decoupled` B′/B″ half-steps and the frozen-Jacobian `:fixed_jacobian`
# variant), the opt-in handoff stage, and the factor-once `FastDecoupledCache`.
#
# B′/B″ assembly lives in fast_decoupled_matrices.jl.

"""
    _default_fd_variant(pf::AbstractACPowerFlow) -> FDVariant

The default [`FDVariant`](@ref) for a bare (unparametrized) [`FastDecoupledACPowerFlow`](@ref) on a
given formulation: [`FDDecoupled`](@ref) (classic B′/B″) for the polar formulation,
[`FDFixedJacobian`](@ref) (frozen Jacobian) for the rectangular current-injection and mixed
current-power-balance formulations.
"""
_default_fd_variant(::ACPolarPowerFlow) = FDDecoupled()
_default_fd_variant(::ACRectangularPowerFlow) = FDFixedJacobian()
_default_fd_variant(::ACMixedPowerFlow) = FDFixedJacobian()

# The variant/scheme carried as the FastDecoupledACPowerFlow type parameters, extracted by
# dispatching on the solver TYPE (the formulation's first type parameter). A bare, unparametrized
# `FastDecoupledACPowerFlow` falls back to the per-formulation variant default and the XB scheme.
_fd_variant(pf::AbstractACPowerFlow{S}) where {S} = _fd_variant_from_solver(S, pf)
_fd_variant_from_solver(::Type{<:FastDecoupledACPowerFlow{V}}, pf) where {V <: FDVariant} =
    V()
_fd_variant_from_solver(::Type{FastDecoupledACPowerFlow}, pf) = _default_fd_variant(pf)

_fd_scheme(pf::AbstractACPowerFlow{S}) where {S} = _fd_scheme_from_solver(S)
_fd_scheme_from_solver(
    ::Type{<:FastDecoupledACPowerFlow{V, S}},
) where {V <: FDVariant, S <: FDScheme} =
    S()
_fd_scheme_from_solver(::Type{FastDecoupledACPowerFlow}) = FDSchemeXB()

"""
    _validate_fd_handoff_solver(handoff_solver)

Validate the [`FastDecoupledACPowerFlow`](@ref) `handoff_solver` setting. Throws a descriptive
`ArgumentError` on an unsupported value. The `fd_variant`/`fd_scheme` choices are now carried as
[`FastDecoupledACPowerFlow`](@ref) type parameters, so invalid values are unrepresentable (the type
system rejects them) and the `FDDecoupled`-is-polar-only constraint is enforced at construction
(see `_reject_fd_decoupled_on_nonpolar`). Returns `nothing` when valid.
"""
function _validate_fd_handoff_solver(handoff_solver)
    if !(
        handoff_solver === nothing ||
        handoff_solver === NewtonRaphsonACPowerFlow ||
        handoff_solver === TrustRegionACPowerFlow ||
        handoff_solver === LevenbergMarquardtACPowerFlow
    )
        throw(
            ArgumentError(
                "FastDecoupled: unsupported handoff_solver $(handoff_solver). Must be " *
                "nothing (pure FD), NewtonRaphsonACPowerFlow, TrustRegionACPowerFlow, or " *
                "LevenbergMarquardtACPowerFlow.",
            ),
        )
    end
    return nothing
end

"""
    _newton_power_flow(pf::AbstractACPowerFlow{<:FastDecoupledACPowerFlow}, data, time_step; ...)

Driver for the [`FastDecoupledACPowerFlow`](@ref) solver. Reads the variant/scheme from the
solver's type parameters (`_fd_variant`/`_fd_scheme`), validates the `handoff_solver`, then
dispatches on the variant via [`_fd_run`](@ref) to `_fd_decoupled_power_flow` (polar B′/B″
half-steps) or `_fd_fixed_jacobian_power_flow` (frozen Jacobian). Returns `converged::Bool`.
"""
function _newton_power_flow(
    pf::AbstractACPowerFlow{<:FastDecoupledACPowerFlow},
    data::ACPowerFlowData,
    time_step::Int64;
    tol = DEFAULT_NR_TOL,
    maxIterations = DEFAULT_FD_MAX_ITER,
    handoff_solver = nothing,
    handoff_tol = DEFAULT_FD_HANDOFF_TOL,
    refreeze_on_stall = DEFAULT_FD_REFREEZE_ON_STALL,
    fd_non_divergent = DEFAULT_FD_NON_DIVERGENT,
    validate_voltage_magnitudes = DEFAULT_VALIDATE_VOLTAGES,
    vm_validation_range = DEFAULT_VALIDATION_RANGE,
    x0 = nothing,
    linear_solver = nothing,
    _ignored...,
)
    _validate_fd_handoff_solver(handoff_solver)
    return _fd_run(
        _fd_variant(pf), _fd_scheme(pf), pf, data, time_step;
        tol,
        maxIterations,
        handoff_solver,
        handoff_tol,
        refreeze_on_stall,
        fd_non_divergent,
        validate_voltage_magnitudes,
        vm_validation_range,
        x0,
        linear_solver,
        _ignored...,
    )
end

"""
    _fd_run(variant::FDVariant, scheme::FDScheme, pf, data, time_step; kwargs...) -> Bool

Variant dispatch for the FD driver. [`FDDecoupled`](@ref) runs the polar B′/B″ loop with `scheme`;
when LCC HVDC is present it uses the sequential AC–DC method (the B′/B″ half-steps solve the AC
network while a per-LCC converter sub-solve refreshes the DC boundary conditions each cycle — see
[`_fd_converter_substep!`](@ref)). [`FDFixedJacobian`](@ref) runs the frozen-Jacobian loop (scheme
unused). Each inner loop absorbs any extra safeguard kwargs via its own `_ignored...`.
"""
function _fd_run(
    ::FDDecoupled,
    scheme::FDScheme,
    pf::AbstractACPowerFlow{<:FastDecoupledACPowerFlow},
    data::ACPowerFlowData,
    time_step::Int64;
    kwargs...,
)
    return _fd_decoupled_power_flow(pf, data, time_step, scheme; kwargs...)
end

function _fd_run(
    ::FDFixedJacobian,
    ::FDScheme,
    pf::AbstractACPowerFlow{<:FastDecoupledACPowerFlow},
    data::ACPowerFlowData,
    time_step::Int64;
    kwargs...,
)
    return _fd_fixed_jacobian_power_flow(pf, data, time_step; kwargs...)
end

# =====================================================================================
# Shared FD safeguards (WP2; reused by WP3's decoupled half-steps and WP5).
#
# These operate on a generic Newton-style cycle: a candidate step `Δx` (already
# solve-then-negated, ready for `x .+= Δx`) is conditioned in place before being applied,
# and divergence/restore bookkeeping is tracked across cycles by an `FDSafeguardState`.
# =====================================================================================

"""
    FDSafeguardState

Mutable bookkeeping for the shared FD safeguards across iterations of a single driver
invocation. Tracks the previous cycle's sum-of-squares mismatch (for the non-divergent
improvement test), the best (smallest) sum-of-squares mismatch seen and a snapshot of the
state vector that achieved it (for best-state restore on non-divergent termination), and
the cycle-start state snapshot (for re-applying a halved step from a clean base).
"""
mutable struct FDSafeguardState
    prev_ss::Float64        # Σ(Rv²) at the start of the current cycle
    best_ss::Float64        # smallest Σ(Rv²) seen so far
    best_x::Vector{Float64} # state vector achieving best_ss
    cycle_x::Vector{Float64} # state snapshot at the start of the current cycle
end

function FDSafeguardState(x0::Vector{Float64}, ss0::Float64)
    return FDSafeguardState(ss0, ss0, copy(x0), copy(x0))
end

"""Record `x` as the best-seen state if its sum-of-squares mismatch `ss` improved on the record."""
function _fd_update_best!(sg::FDSafeguardState, x::Vector{Float64}, ss::Float64)
    if ss < sg.best_ss
        sg.best_ss = ss
        copyto!(sg.best_x, x)
    end
    return
end

"""Reset the non-divergence bookkeeping (used after a one-shot refreeze): the previous
sum-of-squares becomes the current `ss`; the best-state record is preserved."""
function _fd_reset_safeguard!(sg::FDSafeguardState, x::Vector{Float64}, ss::Float64)
    sg.prev_ss = ss
    copyto!(sg.cycle_x, x)
    _fd_update_best!(sg, x, ss)
    return
end

"""Record the start-of-cycle snapshot and update the best-state record if the current
state improved on it."""
function _fd_begin_cycle!(sg::FDSafeguardState, x::Vector{Float64}, ss::Float64)
    copyto!(sg.cycle_x, x)
    sg.prev_ss = ss
    _fd_update_best!(sg, x, ss)
    return
end

"""
    _fd_dvlim_clamp!(Δx, v_state_idx, v_vals, dvlim) -> Bool

DVLIM safeguard. `v_state_idx` are the `x`-indices whose entries are
voltage magnitudes (the "ΔV portion" of the step) and `v_vals` the matching current |V|
values. Uniformly scales the ENTIRE step `Δx` so the largest applied |ΔV| ≤ `dvlim`, and
additionally guards the positivity constraint ΔV/V > −1 (a step that would drive a bus
voltage to ≤ 0). Returns `true` if any clamping was applied. No-op (returns `false`) when
`v_state_idx` is empty (e.g. all-PV systems, or formulations without a scalar |V| state)."""
function _fd_dvlim_clamp!(
    Δx::Vector{Float64},
    v_state_idx::AbstractVector{<:Integer},
    v_vals::AbstractVector{Float64},
    dvlim::Float64,
)
    isempty(v_state_idx) && return false
    scale = 1.0
    @inbounds for (k, ix) in enumerate(v_state_idx)
        dv = Δx[ix]
        absdv = abs(dv)
        if absdv > dvlim
            scale = min(scale, dvlim / absdv)
        end
        # Positivity guard: ΔV/V > −1 ⇒ V + ΔV > 0. If a (scaled) step would still drive
        # V non-positive, scale further so the applied ΔV leaves a small positive margin.
        v = v_vals[k]
        if v > 0.0 && dv < 0.0
            # Largest admissible |ΔV| keeping V positive (with a 1% margin).
            max_drop = 0.99 * v
            if absdv > max_drop
                scale = min(scale, max_drop / absdv)
            end
        end
    end
    if scale < 1.0
        LinearAlgebra.rmul!(Δx, scale)
        return true
    end
    return false
end

"""
    _fd_blowup(Δx, blowup) -> Bool

BLOWUP safeguard (used when non-divergent backtracking is disabled): returns
`true` if the largest-magnitude component of the proposed step exceeds `blowup`, signalling
the FD stage should abort."""
function _fd_blowup(Δx::Vector{Float64}, blowup::Float64)
    return norm(Δx, Inf) > blowup
end

"""
    _fd_vm_abort(vm, vm_abort) -> Bool

V≈0 abort: returns `true` if any bus voltage magnitude has been driven below
`vm_abort`."""
function _fd_vm_abort(vm::AbstractVector{Float64}, vm_abort::Float64)
    @inbounds for v in vm
        (isfinite(v) && v >= vm_abort) || return true
    end
    return false
end

"""Frozen-Jacobian Newton step: reuse `_solve_Δx_nr!` (`Δx ← cache \\ r`) then negate, so
the update is `x .+= Δx`. Does NOT refactor the cache — that is the whole point of the
fixed-Jacobian variant (cf. `_set_Δx_nr!`, which refactors every call)."""
function _solve_Δx_nr_frozen!(
    stateVector::StateVectorCache,
    cache::PFLinearSolverCache,
)
    _solve_Δx_nr!(stateVector, cache)
    LinearAlgebra.rmul!(stateVector.Δx_nr, -1.0)
    return
end

"""The `x`-indices that hold a scalar voltage magnitude (the DVLIM "ΔV portion"). For the
polar formulation these are the PQ-bus |V| entries (precomputed on the residual as
`validate_indices`). The rectangular-CI / mixed-CPB formulations carry `(e, f)` voltage
state with no scalar |V| entry, so DVLIM voltage clamping does not apply there (returns an
empty vector); their blowup / non-divergent / V≈0 safeguards still operate on the full
step."""
_fd_v_state_indices(residual::ACPowerFlowResidual) = residual.validate_indices
_fd_v_state_indices(::ACRectangularCIResidual) = Int[]
_fd_v_state_indices(::ACMixedCPBResidual) = Int[]

# =====================================================================================
# Opt-in handoff (WP4). The FD stage iterates to a loose `handoff_tol` (`stage_tol`), then
# this helper hands the FD state off to the existing NR/TR inner method for final
# refinement to the real `tol`. No-op when handoff is disabled or FD already met `tol`.
# =====================================================================================

"""The FD-stage exit tolerance: the loose `handoff_tol` when a handoff will polish the result to
the real `tol`, else `tol` itself (pure FD)."""
_fd_stage_tol(handoff_solver, tol, handoff_tol) =
    handoff_solver === nothing ? tol : handoff_tol

# The `Jv` argument for `_finalize_power_flow`: the assembled Jacobian values when the :decoupled
# driver built `J`, or `nothing` when it skipped it (no handoff, no loss/vstab factors).
_fd_finalize_jv(::Nothing) = nothing
_fd_finalize_jv(J) = J.Jv

"""
    _fd_maybe_handoff!(pf, sv, residual, J, time_step, handoff_solver, tol, linear_solver,
                       solver_name, fd_iters) -> (converged::Bool, handoff_iters::Int)

Run the opt-in handoff solver (`NewtonRaphsonACPowerFlow` / `TrustRegionACPowerFlow` /
`LevenbergMarquardtACPowerFlow`) from the current FD state `sv.x` for final refinement to the
real `tol`. No-op (returns the current convergence status and `0` handoff iterations) when
`handoff_solver === nothing` or the FD state already meets `tol`. Otherwise refreshes the
formulation Jacobian VALUES at the current FD state and calls the matching inner method:
NR/TR via the shared `_run_power_flow_method(::StateVectorCache, ::PFLinearSolverCache, ...)`;
LM via its workspace-based `_run_power_flow_method(x0::Vector, ::LMWorkspace, ...)` adapter.
All paths mutate `sv.x` / `residual` / `J` in place (the SAME objects the FD loop used), so the
caller's subsequent `J(time_step)` / `_finalize_*` see the refined solution. `fd_iters` and
`solver_name` are used only for the `@info` handoff log line.
"""
function _fd_maybe_handoff!(
    pf::AbstractACPowerFlow{<:FastDecoupledACPowerFlow},
    sv::StateVectorCache,
    residual::Union{ACPowerFlowResidual, ACRectangularCIResidual, ACMixedCPBResidual},
    J::Union{Nothing, ACPowerFlowJacobian, ACRectangularCIJacobian, ACMixedCPBJacobian},
    time_step::Int64,
    handoff_solver,
    tol::Float64,
    linear_solver::Union{Nothing, AbstractString},
    solver_name::String,
    fd_iters::Int,
)
    # `J === nothing` only when `handoff_solver === nothing` (the :decoupled driver builds `J`
    # whenever a handoff is configured), so the early return below fires before any `J` deref.
    fd_met_tol = norm(residual.Rv, Inf) < tol
    if handoff_solver === nothing || fd_met_tol
        return (fd_met_tol, 0)
    end
    J(time_step)                                  # refresh Jacobian VALUES at current FD state
    if handoff_solver === LevenbergMarquardtACPowerFlow
        # LM's inner method takes the raw state vector + an LMWorkspace (a different signature
        # from NR/TR) and mutates x0 in place; see src/levenberg-marquardt.jl.
        ws = LMWorkspace(J.Jv; marquardt_scaling = _default_marquardt_scaling(pf))
        converged, i2 = _run_power_flow_method(
            time_step, sv.x, residual, J, ws;
            tol, maxIterations = DEFAULT_NR_MAX_ITER, λ_0 = DEFAULT_λ_0,
        )
    else
        backend = resolve_linear_solver_backend(linear_solver)
        hcache = make_linear_solver_cache(backend, J.Jv)
        symbolic_factor!(hcache, J.Jv)
        converged, i2 = _run_power_flow_method(
            time_step, sv, hcache, residual, J, handoff_solver;
            tol, maxIterations = DEFAULT_NR_MAX_ITER,
        )
    end
    @info "$solver_name: FD stage $fd_iters iters → handoff $(handoff_solver) " *
          "$(converged ? "converged" : "did NOT converge") in $i2 iters."
    return (converged, i2)
end

"""
    _fd_fixed_jacobian_power_flow(pf, data, time_step; ...) -> Bool

The `:fixed_jacobian` (frozen-Jacobian / "dishonest Newton") FD loop for ALL three
formulations. Factors the formulation Jacobian ONCE at `x0` and reuses the factorization
across every iteration. Exact residual every iteration ⇒ converges to the same solution as
NR (linear rate). Shared safeguards (non-divergent backtracking with best-state restore,
BLOWUP, DVLIM, V≈0 abort) protect against the documented FD failure modes."""
function _fd_fixed_jacobian_power_flow(
    pf::AbstractACPowerFlow{<:FastDecoupledACPowerFlow},
    data::ACPowerFlowData,
    time_step::Int64;
    tol::Float64 = DEFAULT_NR_TOL,
    maxIterations::Int = DEFAULT_FD_MAX_ITER,
    refreeze_on_stall::Bool = DEFAULT_FD_REFREEZE_ON_STALL,
    fd_non_divergent::Bool = DEFAULT_FD_NON_DIVERGENT,
    fd_blowup::Float64 = DEFAULT_FD_BLOWUP,
    fd_dvlim::Float64 = DEFAULT_FD_DVLIM,
    fd_vm_abort::Float64 = DEFAULT_FD_VM_ABORT,
    fd_ndvfct::Float64 = DEFAULT_FD_NDVFCT,
    fd_max_step_halvings::Int = DEFAULT_FD_MAX_STEP_HALVINGS,
    handoff_solver = nothing,
    handoff_tol::Float64 = DEFAULT_FD_HANDOFF_TOL,
    validate_voltage_magnitudes::Bool = DEFAULT_VALIDATE_VOLTAGES,
    vm_validation_range::MinMax = DEFAULT_VALIDATION_RANGE,
    x0 = nothing,
    linear_solver::Union{Nothing, AbstractString} = nothing,
    _return_stage_iters::Bool = false,
    _ignored...,
)
    # FD stage exits on `stage_tol`: the loose `handoff_tol` when a handoff is configured
    # (the handoff polishes to the real `tol`), else the real `tol` (pure FD).
    stage_tol = _fd_stage_tol(handoff_solver, tol, handoff_tol)
    init_kwargs = if isnothing(x0)
        (; validate_voltage_magnitudes, vm_validation_range)
    else
        (; validate_voltage_magnitudes, vm_validation_range, x0)
    end
    residual, J, x0_init =
        initialize_power_flow_variables(pf, data, time_step; init_kwargs...)

    # Early-exit uses the REAL `tol` (not `stage_tol`): if x0 already meets `tol` we are truly
    # done and skip everything (including handoff). If x0 meets only the loose `stage_tol` but
    # not `tol`, we must still enter the block so the in-loop `stage_tol` check exits the FD
    # stage at 0 iters and the handoff refines to `tol` (mirrors the decoupled loop, whose
    # handoff runs unconditionally after its while-loop).
    converged = norm(residual.Rv, Inf) < tol
    i = 0
    handoff_iters = 0
    x_final = x0_init
    solver_name = "FastDecoupled(:fixed_jacobian)"

    if !converged
        # Ensure J holds VALUES at x0. Polar's setup already calls `J(time_step)`; rect/mixed
        # constructors do not, so call it here unconditionally (cheap, once).
        J(time_step)
        backend = resolve_linear_solver_backend(linear_solver)
        cache = make_linear_solver_cache(backend, J.Jv)
        full_factor!(cache, J.Jv)                       # factor the frozen J ONCE

        sv = StateVectorCache(x0_init, residual.Rv)
        v_state_idx = _fd_v_state_indices(residual)
        vm_view = view(data.bus_magnitude, :, time_step)

        residual(sv.x, time_step)
        ss = dot(residual.Rv, residual.Rv)
        sg = FDSafeguardState(sv.x, ss)
        converged = norm(residual.Rv, Inf) < stage_tol
        refrozen = false

        while i < maxIterations && !converged
            _fd_begin_cycle!(sg, sv.x, ss)

            # --- frozen Newton step on the exact residual ---
            copyto!(sv.r, residual.Rv)
            _solve_Δx_nr_frozen!(sv, cache)

            if !fd_non_divergent && _fd_blowup(sv.Δx_nr, fd_blowup)
                @warn(
                    "$solver_name: BLOWUP — proposed step $(norm(sv.Δx_nr, Inf)) " *
                    "exceeds $(fd_blowup); aborting FD stage."
                )
                break
            end

            # DVLIM voltage clamp (+ positivity guard) on the applied step.
            if !isempty(v_state_idx)
                v_vals = @view sv.x[v_state_idx]
                _fd_dvlim_clamp!(sv.Δx_nr, v_state_idx, v_vals, fd_dvlim)
            end

            # apply step, evaluate exact residual (syncs data: V/θ/P/Q)
            sv.x .+= sv.Δx_nr
            residual(sv.x, time_step)
            ss = dot(residual.Rv, residual.Rv)

            # V≈0 abort.
            if _fd_vm_abort(vm_view, fd_vm_abort)
                @warn(
                    "$solver_name: a bus voltage magnitude was driven below " *
                    "$(fd_vm_abort); aborting FD stage."
                )
                # restore best state before bailing out
                _fd_restore_best!(sv, residual, sg, time_step)
                ss = sg.best_ss
                break
            end

            # --- non-divergent backtracking ---
            if fd_non_divergent && ss >= fd_ndvfct * sg.prev_ss
                # The full step failed to improve enough: re-apply a halved step from the
                # cycle-start state, up to `fd_max_step_halvings` times.
                accepted = false
                factor = 1.0
                for _ in 1:fd_max_step_halvings
                    factor *= 0.5
                    copyto!(sv.x, sg.cycle_x)
                    @inbounds @. sv.x += factor * sv.Δx_nr
                    residual(sv.x, time_step)
                    ss = dot(residual.Rv, residual.Rv)
                    _fd_update_best!(sg, sv.x, ss)
                    if ss < fd_ndvfct * sg.prev_ss &&
                       !_fd_vm_abort(vm_view, fd_vm_abort)
                        accepted = true
                        break
                    end
                end
                if !accepted
                    # Exhausted halvings: optionally refreeze ONCE, else terminate and
                    # restore the best-Σ(Rv²) state seen.
                    if refreeze_on_stall && !refrozen
                        refrozen = true
                        # refresh J at the best state, refactor in place, continue
                        _fd_restore_best!(sv, residual, sg, time_step)
                        J(time_step)
                        numeric_refactor!(cache, J.Jv)
                        ss = sg.best_ss
                        _fd_reset_safeguard!(sg, sv.x, ss)
                        converged = norm(residual.Rv, Inf) < stage_tol
                        i += 1
                        continue
                    else
                        _fd_restore_best!(sv, residual, sg, time_step)
                        ss = sg.best_ss
                        @warn(
                            "$solver_name: non-divergent backtracking exhausted; " *
                            "restoring best state (Σmismatch² = $(ss))."
                        )
                        break
                    end
                end
            else
                # step accepted; update best-state record
                _fd_update_best!(sg, sv.x, ss)
            end

            validate_voltage_magnitudes && _validate_state_magnitudes(
                residual, sv.x, vm_validation_range, i,
            )

            converged = norm(residual.Rv, Inf) < stage_tol
            if !converged
                i += 1
                # On a plain stall (no non-divergent control) allow a one-shot refreeze.
                if !fd_non_divergent && refreeze_on_stall && !refrozen &&
                   ss >= fd_ndvfct * sg.prev_ss
                    refrozen = true
                    J(time_step)
                    numeric_refactor!(cache, J.Jv)
                    _fd_reset_safeguard!(sg, sv.x, ss)
                end
            end
        end
        # Opt-in handoff: refine the FD state to the real `tol` with NR/TR. No-op when
        # handoff is disabled or the FD state already met `tol`. Threads handoff iters into
        # the reported count so finalize happens ONCE, on the refined state.
        converged, i2 = _fd_maybe_handoff!(
            pf, sv, residual, J, time_step, handoff_solver, tol, linear_solver,
            solver_name,
            i,
        )
        i += i2
        handoff_iters = i2
        x_final = sv.x
    end

    # Refresh J at the SOLUTION only when loss/voltage-stability factors are requested (they read
    # J.Jv in _finalize_power_flow); the FD loop never otherwise touches J, so skip the eval.
    if get_calculate_loss_factors(data) || get_calculate_voltage_stability_factors(data)
        J(time_step)
    end
    _finalize_formulation!(pf, data, x_final, residual, time_step)
    result = _finalize_power_flow(
        converged, i, solver_name, residual, data, J.Jv, time_step,
    )
    return _return_stage_iters ? (result, i - handoff_iters, handoff_iters) : result
end

"""Restore the best-Σ(Rv²) state recorded in `sg` into `sv.x` and re-evaluate the residual
there (syncing `data`). Used on non-divergent termination, V≈0 abort, and before a
one-shot refreeze."""
function _fd_restore_best!(
    sv::StateVectorCache,
    residual::Union{ACPowerFlowResidual, ACRectangularCIResidual, ACMixedCPBResidual},
    sg::FDSafeguardState,
    time_step::Int64,
)
    copyto!(sv.x, sg.best_x)
    residual(sv.x, time_step)
    return
end

# =====================================================================================
# Polar :decoupled (B′/B″ half-iteration) loop — WP3.
#
# Classic fast decoupled power flow: solve the active-power
# mismatch against the constant B′ (P-θ half-step), re-evaluate the residual, then the
# reactive-power mismatch against the constant B″ (Q-V half-step), re-evaluate again. The
# REF/PV "explicit" residual rows (slack P, REF Q, PV Q) are closed-form given (V, θ) and are
# synced after every half-step so ‖Rv‖∞ stays a meaningful global convergence criterion and
# data.bus_reactive_power_injections stays current for the Q-limit outer loop.
#
# Sign convention (the load-bearing piece): both half-steps mirror NR's solve-then-negate
# (`x .-= B⁻¹·r`), pinned by T2 NR-parity. The explicit-row updates use the SAME contract;
# `FD_EXPLICIT_SYNC_SIGN = -1` corresponds to the plan's `x[...] -= sign·Rv[...]` so each
# explicit update reduces to `x[...] += Rv[...]` (single Newton step on a ±1-coefficient row /
# the rank-1 distributed-slack column where Σγ = 1).
# =====================================================================================

# Plan §4.1 "Explicit-row sync": the slack scalar column has ∂Rv[2i-1]/∂s = −γᵢ (Σγ = 1 over a
# subnetwork ⇒ a single exact rank-1 update), and the PV/REF Q rows have a −1 self-coefficient.
# The solve-then-negate contract therefore gives `x[...] += Rv[...]`, i.e. `−= sign·Rv` with:
const FD_EXPLICIT_SYNC_SIGN = -1.0

# =====================================================================================
# Factor-once caching for the polar :decoupled loop (WP5b).
#
# The fast-decoupled performance contract: the active-power/angle matrix B′ remains
# fixed throughout the solution, and the reactive-power/voltage matrix B″ changes only as
# voltage controlled buses switch between PV and PQ. The :decoupled B′/B″ are
# voltage-INSENSITIVE constants of (network, scheme, backend):
#   * B′ is restricted to the non-REF set `fd.pvpq`. PV→PQ Q-limit switching keeps both PV and
#     PQ buses non-REF, so `pvpq` — hence B′ and its factorization — is INVARIANT across Q-limit
#     retries and across time steps. Factor B′ exactly ONCE per (data, scheme, backend) lifetime.
#   * Only the B″ `[pq, pq]` submatrix changes when the PQ set changes. Factor it once per
#     distinct PQ set (bus-type signature); reuse on repeat signatures.
# The cache is a [`FastDecoupledCache`](@ref) `<: SolverCache` stored in `data.solver_cache[]`; it
# is type-disjoint from the DC path's [`DCSolverCache`](@ref), so `_get_or_build_fd_cache!`
# dispatches on the slot's type and a cross-use fails loudly (a `MethodError`) — no sentinel needed.
#
# The :fixed_jacobian loop deliberately does NOT use this cache: its frozen Jacobian is
# evaluated at x0 and x0 changes per time step (loads change), so it cannot be reused.
# =====================================================================================

"""
    FDCacheKey{S <: FDScheme}

Invalidation key for a [`FastDecoupledCache`](@ref): the cached B′/B″ are valid only while the
network identity, the B′/B″ scheme, and the linear-solver backend are unchanged. Bundled into one
value so the cache-hit test is a single `==` (`==` falls back to field-wise `===` for this
immutable struct). Parametrized on the scheme type `S` so `scheme` is a concretely-typed field;
keys with different schemes are different concrete types, so a scheme change never compares equal.

# Fields
- `ybus_id::UInt`: `objectid(data.power_network_matrix)` — network identity.
- `scheme::S`: the scheme instance, [`FDSchemeXB`](@ref)/[`FDSchemeBX`](@ref).
- `backend_id::DataType`: `typeof(resolve_linear_solver_backend(linear_solver))`.
"""
struct FDCacheKey{S <: FDScheme}
    ybus_id::UInt
    scheme::S
    backend_id::DataType
end

"""
    FDPQData

Per-PQ-set (per bus-type signature) data for the polar :decoupled loop: the factored B″
submatrix over that PQ set, plus the preallocated reactive half-step buffers/index vectors. Built
once per distinct PQ set by [`_get_pq_data!`](@ref) and reused on repeat signatures (Q-limit
retries / multi-period steps that return to the same PQ set).

# Fields
- `bpp::FDBppCache`: factored `[pq, pq]` B″ submatrix.
- `pq::Vector{Int}`: PQ bus indices (sorted; `== bpp.pq`).
- `v_x_idx::Vector{Int}`: `x`-indices of the |V| state at `pq` (`2i-1`).
- `q_row_idx::Vector{Int}`: `Rv`-indices of the Q-mismatch rows at `pq` (`2i`).
- `rq::Vector{Float64}`: preallocated reactive half-step buffer (length `length(pq)`).
- `dvlim_pos::Vector{Int}`: `1:length(pq)` (DVLIM operates on `rq` positionally).
"""
mutable struct FDPQData
    bpp::FDBppCache
    pq::Vector{Int}
    v_x_idx::Vector{Int}
    q_row_idx::Vector{Int}
    rq::Vector{Float64}
    dvlim_pos::Vector{Int}
end

"""
    FastDecoupledCache{S <: FDScheme}

Factor-once cache for the polar :decoupled FD loop, stored in `data.solver_cache[]` (a
[`SolverCache`](@ref) subtype, type-disjoint from the DC path's [`DCSolverCache`](@ref)). Holds the
[`FDCacheKey`](@ref) invalidation key, the constant [`FDMatrices`](@ref) (recovered params +
factored B′ + assembled B″_full), the `pvpq`-invariant half-step buffers/index vectors (factored
ONCE per `(data, scheme, backend)` lifetime), and a `Dict` of per-PQ-set [`FDPQData`](@ref) keyed on
a bus-type signature. `bp_factor_count`/`bpp_factor_count` count B′ and B″ factorizations for
testability (factor-once verification). Parametrized on the scheme type `S` (shared with `key`/`fd`)
so all fields are concretely typed; every field the hot half-step loop reads is `S`-independent, so
retrieval through the abstract `solver_cache` slot stays type-stable.

# Fields
- `key::FDCacheKey{S}`: invalidation key (network identity, scheme, backend).
- `fd::FDMatrices{S}`: recovered params + factored B′ + B″_full.
- `pvpq::Vector{Int}`: non-REF bus indices (`== fd.pvpq`).
- `theta_x_idx::Vector{Int}`: `x`-indices of the θ state at `pvpq` (`2i`).
- `p_row_idx::Vector{Int}`: `Rv`-indices of the P-mismatch rows at `pvpq` (`2i-1`).
- `rp::Vector{Float64}`: preallocated active half-step buffer (length `length(pvpq)`).
- `pq_data::Dict{Vector{PSY.ACBusTypes}, FDPQData}`: bus-type column → per-PQ-set data (the
  materialized column is the key, so distinct PQ sets can never collide).
- `bp_factor_count::Int`: number of B′ factorizations (must be 1 over the cache lifetime).
- `bpp_factor_count::Int`: number of B″ factorizations (one per distinct PQ signature).
- `pvpq_pos::Vector{Int}`: bus index → position in `pvpq` (`1:length(pvpq)`), `0` for a REF bus
  or any bus not in `pvpq`. Length `nbus`; built once alongside the other `pvpq`-invariant
  buffers. Used by the area-interchange bordered substep ([`_fd_area_substep!`](@ref)) to place
  each tie endpoint's θ-partial into the right row of the border system without a per-call
  `searchsortedfirst` over `pvpq`.
- `area_W::Matrix{Float64}`: bordered-Schur scratch, `B′⁻¹·C` (size `length(pvpq) ×
  n_controlled_areas`). Rebuilt (not just resized) if `n_controlled_areas(data)` changes mid-
  lifetime (greedy relax de-enrollment) — see [`_fd_area_substep!`](@ref).
- `area_S::Matrix{Float64}`: bordered-Schur scratch, the dense `Dᵀ·W` Schur complement
  (`n_controlled_areas × n_controlled_areas`).
- `area_g::Vector{Float64}`: bordered-Schur scratch, `Dᵀ·u − r_a` (length `n_controlled_areas`).
- `area_u::Vector{Float64}`: bordered-Schur scratch, `B′⁻¹·rp` at the substep's current state
  (length `length(pvpq)`).
- `area_dtheta::Vector{Float64}`: bordered-Schur scratch, the final `Δθ = u − W·ΔP_a`
  (length `length(pvpq)`).
"""
mutable struct FastDecoupledCache{S <: FDScheme} <: SolverCache
    key::FDCacheKey{S}
    fd::FDMatrices{S}
    pvpq::Vector{Int}
    theta_x_idx::Vector{Int}
    p_row_idx::Vector{Int}
    rp::Vector{Float64}
    pq_data::Dict{Vector{PSY.ACBusTypes}, FDPQData}
    bp_factor_count::Int
    bpp_factor_count::Int
    pvpq_pos::Vector{Int}
    area_W::Matrix{Float64}
    area_S::Matrix{Float64}
    area_g::Vector{Float64}
    area_u::Vector{Float64}
    area_dtheta::Vector{Float64}
end

"""
    _fd_backend_id(linear_solver) -> DataType

The backend identity used as a [`FastDecoupledCache`](@ref) invalidation key: the concrete type
of the resolved linear-solver backend (e.g. `PNM.KLUSolver`). Matches the DC path's
`typeof(cached_backend) === typeof(backend)` reuse test."""
_fd_backend_id(linear_solver::Union{Nothing, AbstractString}) =
    typeof(resolve_linear_solver_backend(linear_solver))

# Reuse on a matching key, else `nothing` to signal a rebuild. Dispatch on the slot's type: an
# empty slot returns `nothing`; a stray non-FD `SolverCache` (cross-use with the DC path, impossible
# today since the data types are disjoint) is a loud `MethodError` rather than a silent mis-read.
_reuse_fd_cache(::Nothing, key::FDCacheKey) = nothing
_reuse_fd_cache(cache::FastDecoupledCache, key::FDCacheKey) =
    cache.key == key ? cache : nothing

"""
    _get_or_build_fd_cache!(data, time_step, scheme, backend_id, linear_solver)
        -> FastDecoupledCache

Fetch the cached [`FastDecoupledCache`](@ref) from `data.solver_cache[]`, or build it. Reuse
requires the slot to hold a `FastDecoupledCache` whose [`FDCacheKey`](@ref) (Ybus objectid, scheme,
backend identity) matches — then B′ and any per-PQ-set B″ are reused with NO refactorization. The
reuse test (`_reuse_fd_cache`) dispatches on the slot's type, so an empty slot rebuilds and a stray
non-FD `SolverCache` (cross-use, impossible today) is a loud `MethodError`. Otherwise builds via
`build_fd_matrices` (B′ factored once ⇒ `bp_factor_count = 1`), precomputes the `pvpq`-invariant
buffers, and stores the cache."""
function _get_or_build_fd_cache!(
    data::ACPowerFlowData,
    time_step::Int64,
    scheme::FDScheme,
    backend_id::DataType,
    linear_solver::Union{Nothing, AbstractString},
)
    key = FDCacheKey(objectid(data.power_network_matrix), scheme, backend_id)
    reused = _reuse_fd_cache(data.solver_cache[], key)
    isnothing(reused) || return reused

    fd = build_fd_matrices(data, time_step, scheme; linear_solver)   # B′ assembled + factored ONCE
    theta_x_idx = [2 * i for i in fd.pvpq]
    p_row_idx = [2 * i - 1 for i in fd.pvpq]
    rp = Vector{Float64}(undef, length(fd.pvpq))
    nbus = size(data.bus_type, 1)
    pvpq_pos = zeros(Int, nbus)
    @inbounds for (k, bus_ix) in enumerate(fd.pvpq)
        pvpq_pos[bus_ix] = k
    end
    n_areas = n_controlled_areas(data)
    cache = FastDecoupledCache(
        key,
        fd,
        copy(fd.pvpq),
        theta_x_idx,
        p_row_idx,
        rp,
        Dict{Vector{PSY.ACBusTypes}, FDPQData}(),
        1,   # bp_factor_count: build_fd_matrices factored B′ exactly once
        0,   # bpp_factor_count: bumped per distinct PQ signature in _get_pq_data!
        pvpq_pos,
        Matrix{Float64}(undef, length(fd.pvpq), n_areas),
        Matrix{Float64}(undef, n_areas, n_areas),
        Vector{Float64}(undef, n_areas),
        Vector{Float64}(undef, length(fd.pvpq)),
        Vector{Float64}(undef, length(fd.pvpq)),
    )
    data.solver_cache[] = cache
    return cache
end

"""
    _get_pq_data!(cache, data, time_step, linear_solver) -> FDPQData

Fetch the [`FDPQData`](@ref) for `time_step`'s PQ set from `cache.pq_data`, or build it. The
materialized bus-type column `collect(view(data.bus_type, :, time_step))` keys the dict: identical
bus-type columns (across time steps or Q-limit retries returning to a previously-seen PQ set) hit
the cache with NO refactorization. Keying on the column itself (rather than a hash of it) makes the
lookup collision-free — distinct PQ sets can never alias to the same entry. On a miss, `extract_bpp`
factors the `[pq, pq]` B″ submatrix once (bumping `cache.bpp_factor_count`), the half-step
buffers/index vectors are preallocated, and the result is stored."""
function _get_pq_data!(
    cache::FastDecoupledCache,
    data::ACPowerFlowData,
    time_step::Int64,
    linear_solver::Union{Nothing, AbstractString},
)
    sig = collect(view(data.bus_type, :, time_step))
    existing = get(cache.pq_data, sig, nothing)
    existing === nothing || return existing

    ref, pv, pq = bus_type_idx(data, time_step)
    # B′ and the cached θ-index maps are keyed on the non-REF set, which is invariant under the
    # only supported within-data change (PV↔PQ Q-limit flips). If the REF/isolated set itself
    # drifted, the cached B′/pvpq would be applied to the wrong buses — fail loudly rather than
    # silently mis-update. (A key hit means the bus-type column matched exactly.)
    sort(vcat(pv, pq)) == cache.pvpq || error(
        "FastDecoupled: the non-REF bus set changed for an existing cache (REF/isolated buses " *
        "drifted); the cached B′ factorization is no longer valid. This is unsupported within a " *
        "single PowerFlowData.",
    )
    pq_sorted = sort(pq)
    bpp = extract_bpp(cache.fd, pq_sorted; linear_solver)
    cache.bpp_factor_count += 1
    v_x_idx = [2 * i - 1 for i in bpp.pq]
    q_row_idx = [2 * i for i in bpp.pq]
    rq = Vector{Float64}(undef, length(bpp.pq))
    dvlim_pos = collect(1:length(rq))
    pqdata = FDPQData(bpp, copy(bpp.pq), v_x_idx, q_row_idx, rq, dvlim_pos)
    cache.pq_data[sig] = pqdata
    return pqdata
end

"""
    _sync_explicit_state!(sv, residual, time_step)

Set the REF/PV "explicit" state entries (subnetwork slack P, REF Q, PV Q) to the values that
zero their own residual rows given the current (V, θ). Per subnetwork (`residual.subnetworks`
maps a REF bus to its member buses) the slack scalar `s = x[2·ref−1]` gets the rank-1
distributed-slack update `s −= sign·Σ_{i∈subnet} Rv[2i−1]` (Σγ = 1 ⇒ exact; reduces to the
classic REF-row update when participation is REF-only). REF Q: `x[2·ref] −= sign·Rv[2·ref]`.
PV Q: `x[2i−1] −= sign·Rv[2i]`. `sign = FD_EXPLICIT_SYNC_SIGN` (same solve-then-negate contract
as the half-steps; T2 NR-parity is the arbiter). The caller must re-evaluate the residual after
this to refresh `data`/`Rv`.
"""
function _sync_explicit_state!(
    sv::StateVectorCache,
    residual::ACPowerFlowResidual,
    time_step::Int64,
)
    x = sv.x
    Rv = residual.Rv
    bus_types = view(residual.data.bus_type, :, time_step)
    sign = FD_EXPLICIT_SYNC_SIGN
    independent_ref =
        _multi_swing_ref_indices(residual.data.bus_type, residual.subnetworks, time_step)
    for (ref_bus, subnet) in residual.subnetworks
        if ref_bus in independent_ref
            # Multi-swing island: each swing carries its own slack, so it closes its OWN P and Q
            # rows (no rank-1 island sum, which would tie the swings' P together and mis-solve).
            @inbounds for i in subnet
                if bus_types[i] == PSY.ACBusTypes.REF
                    x[2 * i - 1] -= sign * Rv[2 * i - 1]
                    x[2 * i] -= sign * Rv[2 * i]
                elseif bus_types[i] == PSY.ACBusTypes.PV
                    x[2 * i - 1] -= sign * Rv[2 * i]
                end
            end
            continue
        end
        # Subnetwork slack scalar (the REF P slot): rank-1 update over all member P rows.
        slack_sum = 0.0
        @inbounds for i in subnet
            slack_sum += Rv[2 * i - 1]
        end
        x[2 * ref_bus - 1] -= sign * slack_sum
        # REF reactive power (explicit, ±1 coefficient on its own Q row).
        x[2 * ref_bus] -= sign * Rv[2 * ref_bus]
        # PV reactive power entries (x[2i-1]) close their own Q rows (Rv[2i]).
        @inbounds for i in subnet
            if bus_types[i] == PSY.ACBusTypes.PV
                x[2 * i - 1] -= sign * Rv[2 * i]
            end
        end
    end
    return
end

"""
    _fd_lcc_substep!(sv, residual, data, time_step)

Sequential AC–DC step for the polar `:decoupled` loop when LCC HVDC is present: solve each LCC's
converter control equations for the current AC voltages ([`_fd_converter_substep!`](@ref)), write
the converged converter states back into the trailing `4·n_lcc` slots of `sv.x`
([`_write_lcc_state_to_x!`](@ref)), then re-sync explicit rows and re-evaluate the residual so `Rv`
(AC rows + LCC tail rows) reflects the refreshed DC boundary conditions.
"""
function _fd_lcc_substep!(
    sv::StateVectorCache,
    residual::ACPowerFlowResidual,
    data::ACPowerFlowData,
    time_step::Int64,
)
    _fd_converter_substep!(data, time_step)
    _write_lcc_state_to_x!(sv.x, data, time_step)
    _sync_explicit_state!(sv, residual, time_step)
    residual(sv.x, time_step)
    return
end

"""
    _fd_vsc_substep!(sv, residual, data, time_step)

Sequential AC–DC step for the polar `:decoupled` loop when a VSC/DC network is present: re-solve
the DC tail (converter P_c, Q_c and node V_dc) for the current AC voltages
([`_vsc_warm_start!`](@ref)), write it back into the trailing tail slots of `sv.x`, then re-sync
explicit rows and re-evaluate the residual. The B′/B″ half-steps never touch the tail, so without
this sub-step the tail states would stay frozen at their initial values and any AC↔DC coupling
(converter losses in particular) would keep the tail rows from converging.
"""
function _fd_vsc_substep!(
    sv::StateVectorCache,
    residual::ACPowerFlowResidual,
    data::ACPowerFlowData,
    time_step::Int64,
)
    dcn = get_dc_network(data)
    _vsc_warm_start!(dcn, view(data.bus_magnitude, :, time_step), time_step)
    # VSC tail offset in the [buses | LCC | VSC | area] layout (same value the residual's
    # `_read_vsc_state!` reads from) — front-anchored so a trailing area tail can't shift it.
    vsc_off = 2 * size(data.bus_type, 1) + 4 * size(data.lcc.p_set, 1)
    _write_vsc_state_to_x!(sv.x, dcn, vsc_off, time_step)
    _sync_explicit_state!(sv, residual, time_step)
    residual(sv.x, time_step)
    return
end

# =====================================================================================
# Area-interchange bordered Schur substep for the polar :decoupled loop.
#
# The B′/B″ half-steps never touch the ΔP_a tail (their RHS/Jacobian are restricted to the
# ordinary bus P-θ/Q-V rows — see `_fd_lcc_substep!`/`_fd_vsc_substep!`'s docstrings for the
# same structural reason with the LCC/VSC tail), so without this substep ΔP_a stays frozen at
# its flat-start (or warm-started) value and the area residual `r_a = NI_a − PDES_a` never
# converges. This substep drives `r_a → 0` by solving the bordered P-θ system
#
#   [ B′   C ] [Δθ  ]   [rp ]
#   [ Dᵀ   0 ] [ΔP_a] = [ra ]
#
# (`fd.pvpq`/Δθ ordering throughout) via the Schur complement on the tiny, structurally-zero
# (2,2) block, reusing the FIXED `fd.bp_cache` factorization (no `full_factor!`/refactor of
# B′ — only extra back-solves through the already-factored matrix):
#
#   u = B′⁻¹·rp                    (1 back-solve)
#   w_a = B′⁻¹·C[:,a]  ∀ area a    (n_areas back-solves)
#   S = Dᵀ·W  (n_areas×n_areas dense);  g = Dᵀ·u − ra
#   ΔP_a_step = S \ g              (dense n_areas solve, negligible — n_areas is small)
#   Δθ = u − W·ΔP_a_step
#
# `C` column a is the constant `−1.0` at area a's slack-bus pvpq row (the same `−1.0` the
# Jacobian border stamps at `F[2·slack−1]`, `_set_entries_for_area`). `Dᵀ` row a is `∂r_a/∂θ`
# — the θ-part only of `_tie_metered_active_power_partials` at every tie endpoint incident to
# area a, same `±σ` sign convention as `_accumulate_area_row!` (metered side `+1`, other side
# `−1`); the |V| part is dropped (the FD approximation — the RESIDUAL `r_a` used to build `g`
# is still the exact kernel, so the interchange target is met exactly at convergence; only the
# STEP direction is FD-approximate). Both `[Δθ; ΔP_a_step]` are Newton STEPS under this
# codebase's solve-then-negate convention (`_fd_decoupled_power_flow`'s file-header comment),
# so BOTH are subtracted from the current state, mirroring the P half-step's `θ -= Δθ`.
# =====================================================================================

"""
    _fd_area_substep!(sv, cache, residual, data, time_step)

Bordered-Schur area-interchange correction for the polar `:decoupled` loop (see the file-header
comment above for the full math): recompute the P-θ RHS `rp` at the CURRENT state (post half-
steps / LCC-VSC substep — NOT the stale `rp` from the earlier P half-step, since Vm/θ have moved
since then), solve the bordered system by Schur complement against the fixed `cache.fd.bp_cache`,
update `θ` and the `ΔP_a` tail, then re-sync explicit rows and re-evaluate the residual (mirrors
`_fd_lcc_substep!`/`_fd_vsc_substep!`'s contract exactly). Zero work when no area is controlled.
"""
function _fd_area_substep!(
    sv::StateVectorCache,
    cache::FastDecoupledCache,
    residual::ACPowerFlowResidual,
    data::ACPowerFlowData,
    time_step::Int64,
)
    n_areas = n_controlled_areas(data)
    iszero(n_areas) && return
    fd = cache.fd
    isempty(fd.pvpq) && return

    # A greedy-relax de-enrollment can shrink `n_controlled_areas(data)` without changing the
    # `FastDecoupledCache` reuse key (Ybus identity/scheme/backend — the area count is not part
    # of it), so the border scratch is (re)allocated here on a size mismatch rather than only at
    # cache-build time.
    if size(cache.area_S, 1) != n_areas
        cache.area_W = Matrix{Float64}(undef, length(fd.pvpq), n_areas)
        cache.area_S = Matrix{Float64}(undef, n_areas, n_areas)
        cache.area_g = Vector{Float64}(undef, n_areas)
    end

    aid = data.area_interchange
    dcn = get_dc_network(data)
    area_off = area_tail_offset(data, dcn)
    Vm = view(data.bus_magnitude, :, time_step)
    θ = view(data.bus_angles, :, time_step)
    ybus_nzval = SparseArrays.nonzeros(data.power_network_matrix.data)
    pos = cache.pvpq_pos
    theta_x_idx = cache.theta_x_idx
    p_row_idx = cache.p_row_idx

    # u = B′⁻¹·rp, rp recomputed fresh (the current residual/Vm, not the stale P half-step rp).
    u = cache.area_u
    @inbounds for k in eachindex(fd.pvpq)
        u[k] = residual.Rv[p_row_idx[k]] / Vm[fd.pvpq[k]]
    end
    solve!(fd.bp_cache, u)

    # W columns: w_a = B′⁻¹·C[:,a]. The exact-Jacobian border coupling ∂P_slack/∂ΔP_a = −1
    # (`_set_entries_for_area`) lives in the slack bus's P-row; entering the row-scaled B′ space
    # (rp = Rv_P/Vm) divides that row — and its coupling — by V_slack, so C[:,a] = −1/V_slack at
    # area a's slack-bus pvpq row (not −1). W is rebuilt every substep, so the V-dependence is free.
    W = cache.area_W
    @inbounds for area in aid.areas
        wcol = @view W[:, area.tail_ix]
        fill!(wcol, 0.0)
        wcol[pos[area.slack_bus_ix]] = -1.0 / Vm[area.slack_bus_ix]
        solve!(fd.bp_cache, wcol)
    end

    # g = Dᵀ·u − ra ; S = Dᵀ·W, accumulated tie-by-tie (only 2 nonzero Dᵀ columns per tie side)
    # rather than materializing Dᵀ. `_area_residual_gaps` is the same exact kernel the Newton
    # residual tail uses — the FD approximation is confined to the STEP, not `ra` itself.
    S = cache.area_S
    g = cache.area_g
    fill!(S, 0.0)
    ra = _area_residual_gaps(data, time_step)
    @inbounds for area in aid.areas
        g[area.tail_ix] = -ra[area.tail_ix]
    end
    @inbounds for tie in aid.ties
        f = tie.from_bus_ix
        t = tie.to_bus_ix
        (_, dPm_dθf, _, dPm_dθt) =
            _tie_metered_active_power_partials(tie, Vm[f], θ[f], Vm[t], θ[t], ybus_nzval)
        if tie.metered_from
            metered_tail = tie.from_area_tail
            other_tail = tie.to_area_tail
        else
            metered_tail = tie.to_area_tail
            other_tail = tie.from_area_tail
        end
        kf = pos[f]
        kt = pos[t]
        if !iszero(metered_tail)
            _fd_area_row_accumulate!(g, S, W, u, metered_tail, kf, dPm_dθf)
            _fd_area_row_accumulate!(g, S, W, u, metered_tail, kt, dPm_dθt)
        end
        if !iszero(other_tail)
            _fd_area_row_accumulate!(g, S, W, u, other_tail, kf, -dPm_dθf)
            _fd_area_row_accumulate!(g, S, W, u, other_tail, kt, -dPm_dθt)
        end
    end

    dp = S \ g                              # ΔP_a Newton step, n_areas×n_areas dense (tiny)
    dtheta = cache.area_dtheta
    mul!(dtheta, W, dp)
    @inbounds @. dtheta = u - dtheta

    @inbounds for k in eachindex(theta_x_idx)
        sv.x[theta_x_idx[k]] -= dtheta[k]
    end
    @inbounds for area in aid.areas
        sv.x[area_off + area.tail_ix] -= dp[area.tail_ix]
    end

    _sync_explicit_state!(sv, residual, time_step)
    residual(sv.x, time_step)
    return
end

# Accumulate a tie endpoint's contribution into area row `row` of `g`/`S`: `k` is the endpoint
# bus's `pvpq` position (`0` for a REF endpoint — `_push_area_row_bus_cols!`'s exclusion), `coeff`
# is `σ·∂Pm/∂θ` for that endpoint. `S[row, :] += coeff · W[k, :]` since `S = Dᵀ·W` and row `k` of
# `W` is the only nonzero contribution `Dᵀ[row, k]` makes to that dense product.
function _fd_area_row_accumulate!(
    g::Vector{Float64},
    S::Matrix{Float64},
    W::Matrix{Float64},
    u::Vector{Float64},
    row::Int,
    k::Int,
    coeff::Float64,
)
    iszero(k) && return
    g[row] += coeff * u[k]
    @inbounds for b in axes(S, 2)
        S[row, b] += coeff * W[k, b]
    end
    return
end

"""
    _fd_decoupled_power_flow(pf, data, time_step; ...) -> (converged::Bool, iters::Int)

Polar classic fast-decoupled (B′/B″ half-iteration) FD loop. Builds the constant B′/B″ matrices
once (factor-once via `build_fd_matrices`/`extract_bpp`), then iterates strict P-θ → Q-V
half-steps with an exact residual re-evaluation after EACH half-step (the mid-cycle refresh
prevents convergence cycling — do NOT skip it). Shared WP2 safeguards
(non-divergent backtracking with best-state restore, BLOWUP, DVLIM, V≈0 abort) protect the
documented FD failure modes. The FD stage converges on `‖Rv‖∞ < stage_tol`, where `stage_tol`
is the real `tol` for pure FD (`handoff_solver === nothing`) or the loose `handoff_tol` when an
opt-in handoff (WP4) is configured — the handoff then refines to the real `tol`
(`_fd_maybe_handoff!`).

Distributed slack is supported via the per-iteration rank-1 slack sync in
[`_sync_explicit_state!`] (decision recorded in WP3 / T6). Returns `(converged, iters)`; the
public driver returns only `converged`, so this is wrapped by `_newton_power_flow`. When
`_return_iters = true` the tuple is returned directly (used by T2 to assert the FD iteration
count); when `_return_stage_iters = true` it returns `(converged, fd_iters, handoff_iters)`
(used by T4 to assert the FD stage ran and the handoff was small/skipped).
"""
function _fd_decoupled_power_flow(
    pf::AbstractACPowerFlow{<:FastDecoupledACPowerFlow},
    data::ACPowerFlowData,
    time_step::Int64,
    scheme::FDScheme;
    tol::Float64 = DEFAULT_NR_TOL,
    maxIterations::Int = DEFAULT_FD_MAX_ITER,
    fd_non_divergent::Bool = DEFAULT_FD_NON_DIVERGENT,
    fd_blowup::Float64 = DEFAULT_FD_BLOWUP,
    fd_dvlim::Float64 = DEFAULT_FD_DVLIM,
    fd_vm_abort::Float64 = DEFAULT_FD_VM_ABORT,
    fd_ndvfct::Float64 = DEFAULT_FD_NDVFCT,
    fd_max_step_halvings::Int = DEFAULT_FD_MAX_STEP_HALVINGS,
    handoff_solver = nothing,
    handoff_tol::Float64 = DEFAULT_FD_HANDOFF_TOL,
    validate_voltage_magnitudes::Bool = DEFAULT_VALIDATE_VOLTAGES,
    vm_validation_range::MinMax = DEFAULT_VALIDATION_RANGE,
    x0 = nothing,
    linear_solver::Union{Nothing, AbstractString} = nothing,
    _return_iters::Bool = false,
    _return_stage_iters::Bool = false,
    _ignored...,
)
    # FD stage exits on `stage_tol`: the loose `handoff_tol` when a handoff is configured
    # (the handoff polishes to the real `tol`), else the real `tol` (pure FD).
    stage_tol = _fd_stage_tol(handoff_solver, tol, handoff_tol)
    init_kwargs = if isnothing(x0)
        (; validate_voltage_magnitudes, vm_validation_range)
    else
        (; validate_voltage_magnitudes, vm_validation_range, x0)
    end
    # The :decoupled half-steps run on B′/B″ from the cache and never touch the formulation
    # Jacobian. Build it ONLY when its sole consumers are active — a handoff solver, or loss/
    # voltage-stability factors — otherwise skip the full sparse-Jacobian allocation + evaluation
    # entirely (a per-solve, per-time-step saving; see `_initialize_residual_x0`).
    need_jacobian =
        handoff_solver !== nothing ||
        get_calculate_loss_factors(data) ||
        get_calculate_voltage_stability_factors(data)
    if need_jacobian
        residual, J, x0_init =
            initialize_power_flow_variables(pf, data, time_step; init_kwargs...)
    else
        residual, x0_init = _initialize_residual_x0(pf, data, time_step; init_kwargs...)
        J = nothing
    end

    solver_name = "FastDecoupled(FDDecoupled,$(nameof(typeof(scheme))))"

    # The sequential VSC sub-solve pins each converter's Q at its setpoint (`_vsc_warm_start!`),
    # which cannot honor an AC-voltage control row — those converters need the full coupled
    # Jacobian. Reject before any B′/B″ factorization work.
    dcn = get_dc_network(data)
    has_vsc = has_dc_network(dcn)
    if has_vsc && any(controls_ac_voltage, dcn.converter_mode)
        throw(
            ArgumentError(
                "FastDecoupled (FDDecoupled) does not support AC-voltage-controlling VSC " *
                "converters. Use FastDecoupledACPowerFlow{FDFixedJacobian, <scheme>} or a " *
                "full-Newton solver (NR/TR/LM).",
            ),
        )
    end

    # Factor-once cache (WP5b): B′ over fd.pvpq factored exactly once per (data, scheme, backend)
    # lifetime; the [pq, pq] B″ submatrix + half-step buffers factored once per distinct PQ set
    # (bus-type signature) and reused across Q-limit retries / multi-period steps. The hot loop
    # fetches its matrices, index vectors, and rp/rq buffers from the cache (no per-invocation
    # build_fd_matrices/extract_bpp call, no per-iteration allocation). Behavior is identical to
    # building them inline — only WHERE they come from changes (T2/T6 arbitrate equivalence).
    backend_id = _fd_backend_id(linear_solver)
    cache = _get_or_build_fd_cache!(data, time_step, scheme, backend_id, linear_solver)
    pqdata = _get_pq_data!(cache, data, time_step, linear_solver)

    fd = cache.fd
    bpp = pqdata.bpp
    # Position maps (cached). θ entry x[2i] for i∈pvpq; V entry x[2i-1] for i∈pq;
    # P row Rv[2i-1] for i∈pvpq; Q row Rv[2i] for i∈pq.
    theta_x_idx = cache.theta_x_idx
    v_x_idx = pqdata.v_x_idx
    p_row_idx = cache.p_row_idx
    q_row_idx = pqdata.q_row_idx
    Vm = view(data.bus_magnitude, :, time_step)

    rp = cache.rp
    rq = pqdata.rq
    # DVLIM operates on the rq vector directly (positions 1:length(rq) = the ΔV entries).
    dvlim_pos = pqdata.dvlim_pos
    # Live view of |V| at the PQ buses (reflects residual updates); built once, not per iteration.
    vm_pq = view(Vm, bpp.pq)

    sv = StateVectorCache(x0_init, residual.Rv)
    # Sequential AC–DC method: with LCC or VSC present the B′/B″ half-steps solve the AC network
    # while a per-converter sub-solve refreshes the DC boundary conditions each cycle. Neither
    # present ⇒ the sub-solves are never invoked (pure-AC path unchanged).
    n_lcc = get_lcc_count(data)
    # Same structural reason as LCC/VSC: the B′/B″ half-steps never touch the ΔP_a tail, so the
    # bordered Schur substep (`_fd_area_substep!`) runs every cycle a controlled area exists.
    has_area = !iszero(n_controlled_areas(data))

    # Sync explicit rows, then evaluate the residual so Rv / data reflect (V, θ, explicit P/Q).
    _sync_explicit_state!(sv, residual, time_step)
    residual(sv.x, time_step)
    # Make the converter state consistent with the start voltages so the LCC/VSC tail residuals
    # enter the loop already small (they are refreshed each cycle after the Q half-step).
    n_lcc > 0 && _fd_lcc_substep!(sv, residual, data, time_step)
    has_vsc && _fd_vsc_substep!(sv, residual, data, time_step)
    has_area && _fd_area_substep!(sv, cache, residual, data, time_step)
    ss = dot(residual.Rv, residual.Rv)
    sg = FDSafeguardState(sv.x, ss)
    converged = norm(residual.Rv, Inf) < stage_tol
    i = 0

    # One FD half-step + explicit sync + residual refresh, returning whether it diverged on a
    # BLOWUP/V≈0 guard (only consulted when fd_non_divergent is off).
    while i < maxIterations && !converged
        _fd_begin_cycle!(sg, sv.x, ss)
        diverged = false

        # --- P half-step (i.0): rp = Rv_P / Vm over pvpq; solve B′·Δθ = rp; θ -= Δθ. Skipped for
        # a lone-REF-bus island (no non-REF buses ⇒ 0×0 B′, left unfactored); the explicit REF/PV
        # sync below still runs and drives ‖Rv‖∞ to tol. ---
        if !isempty(fd.pvpq)
            @inbounds for k in eachindex(fd.pvpq)
                rp[k] = residual.Rv[p_row_idx[k]] / Vm[fd.pvpq[k]]
            end
            solve!(fd.bp_cache, rp)
            if !fd_non_divergent && _fd_blowup(rp, fd_blowup)
                @warn(
                    "$solver_name: BLOWUP — proposed Δθ $(norm(rp, Inf)) exceeds $(fd_blowup); " *
                    "aborting FD stage."
                )
                diverged = true
            else
                @inbounds for k in eachindex(theta_x_idx)
                    sv.x[theta_x_idx[k]] -= rp[k]
                end
            end
        end
        if !diverged
            _sync_explicit_state!(sv, residual, time_step)
            residual(sv.x, time_step)
        end

        # --- Q half-step (i.5): rq = Rv_Q / Vm over pq; solve B″·ΔV = rq; DVLIM; V -= ΔV.
        # Skipped entirely for an all-PV/REF network (no PQ buses ⇒ 0×0 B″, left unfactored); the
        # P half-step plus the explicit REF/PV Q sync still drive ‖Rv‖∞ to tol. ---
        if !diverged && !isempty(bpp.pq)
            @inbounds for k in eachindex(bpp.pq)
                rq[k] = residual.Rv[q_row_idx[k]] / Vm[bpp.pq[k]]
            end
            solve!(bpp.bpp_cache, rq)
            # Negate so `rq` is the ΔV actually applied (`x .+= rq`): this is the sign convention
            # `_fd_dvlim_clamp!`'s positivity guard assumes, and the half-step's solve-then-negate.
            @inbounds @. rq = -rq
            # BLOWUP on the UNSCALED step, before the DVLIM clamp (threshold on the raw step).
            if !fd_non_divergent && _fd_blowup(rq, fd_blowup)
                @warn(
                    "$solver_name: BLOWUP — proposed ΔV $(norm(rq, Inf)) exceeds " *
                    "$(fd_blowup); aborting FD stage."
                )
                diverged = true
            else
                # DVLIM clamp (+ positivity guard) on the applied ΔV. v_vals = current |V| at pq.
                if !isempty(dvlim_pos)
                    _fd_dvlim_clamp!(rq, dvlim_pos, vm_pq, fd_dvlim)
                end
                @inbounds for k in eachindex(v_x_idx)
                    sv.x[v_x_idx[k]] += rq[k]
                end
                _sync_explicit_state!(sv, residual, time_step)
                residual(sv.x, time_step)
            end
        end

        # Sequential AC–DC converter sub-solve: refresh the DC boundary conditions for the just
        # updated voltages and re-evaluate the residual before the convergence / backtracking checks
        # (so `ss` and `‖Rv‖∞` see the converged converter tail rows).
        if !diverged && n_lcc > 0
            _fd_lcc_substep!(sv, residual, data, time_step)
        end
        if !diverged && has_vsc
            _fd_vsc_substep!(sv, residual, data, time_step)
        end
        if !diverged && has_area
            _fd_area_substep!(sv, cache, residual, data, time_step)
        end

        ss = dot(residual.Rv, residual.Rv)

        # V≈0 abort.
        if _fd_vm_abort(Vm, fd_vm_abort)
            @warn(
                "$solver_name: a bus voltage magnitude was driven below $(fd_vm_abort); " *
                "aborting FD stage."
            )
            _fd_restore_best!(sv, residual, sg, time_step)
            ss = sg.best_ss
            break
        end

        if diverged   # only reachable with fd_non_divergent = false
            _fd_restore_best!(sv, residual, sg, time_step)
            ss = sg.best_ss
            break
        end

        # --- non-divergent backtracking (shared with WP2): if the full cycle failed to
        # improve Σ(Rv²) enough, re-apply a halved cycle step from the cycle-start state. ---
        if fd_non_divergent && ss >= fd_ndvfct * sg.prev_ss
            accepted = false
            factor = 1.0
            # Snapshot the full-cycle step ONCE. Backtracking re-applies a scaled copy from the
            # cycle-start state; re-reading the (already-mutated) `sv.x` each pass would compound
            # the halving (0.5^(k(k+1)/2) instead of 0.5^k) and fold in the explicit-sync deltas.
            # The explicit rows are re-derived by `_sync_explicit_state!` after each rescale.
            Δcycle = sv.x .- sg.cycle_x
            for _ in 1:fd_max_step_halvings
                factor *= 0.5
                @inbounds @. sv.x = sg.cycle_x + factor * Δcycle
                _sync_explicit_state!(sv, residual, time_step)
                residual(sv.x, time_step)
                # Re-solve the LCC/VSC converters at the rescaled voltages so the tail rows stay
                # zeroed during backtracking — otherwise the halved step un-solves the converter
                # state and the tail mismatch dominates `ss`, defeating the line search.
                if n_lcc > 0
                    _fd_lcc_substep!(sv, residual, data, time_step)
                end
                if has_vsc
                    _fd_vsc_substep!(sv, residual, data, time_step)
                end
                if has_area
                    _fd_area_substep!(sv, cache, residual, data, time_step)
                end
                ss = dot(residual.Rv, residual.Rv)
                _fd_update_best!(sg, sv.x, ss)
                if ss < fd_ndvfct * sg.prev_ss && !_fd_vm_abort(Vm, fd_vm_abort)
                    accepted = true
                    break
                end
            end
            if !accepted
                _fd_restore_best!(sv, residual, sg, time_step)
                ss = sg.best_ss
                @warn(
                    "$solver_name: non-divergent backtracking exhausted; restoring best " *
                    "state (Σmismatch² = $(ss))."
                )
                break
            end
        else
            _fd_update_best!(sg, sv.x, ss)
        end

        validate_voltage_magnitudes && _validate_state_magnitudes(
            residual, sv.x, vm_validation_range, i,
        )

        converged = norm(residual.Rv, Inf) < stage_tol
        converged || (i += 1)
    end

    # Opt-in handoff: refine the FD state to the real `tol` with NR/TR. No-op when handoff
    # is disabled or the FD state already met `tol`. Threads handoff iters into the reported
    # count so finalize happens ONCE, on the refined state.
    converged, handoff_iters = _fd_maybe_handoff!(
        pf, sv, residual, J, time_step, handoff_solver, tol, linear_solver, solver_name,
        i,
    )
    i += handoff_iters

    # Refresh J at the SOLUTION only when loss/voltage-stability factors are requested (they read
    # J.Jv in _finalize_power_flow); the FD loop never otherwise touches J, so skip the eval.
    if get_calculate_loss_factors(data) || get_calculate_voltage_stability_factors(data)
        J(time_step)
    end
    _finalize_formulation!(pf, data, sv.x, residual, time_step)
    result = _finalize_power_flow(
        converged, i, solver_name, residual, data, _fd_finalize_jv(J), time_step,
    )
    if _return_stage_iters
        return (result, i - handoff_iters, handoff_iters)
    end
    return _return_iters ? (result, i) : result
end
