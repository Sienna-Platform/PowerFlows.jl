# Fast/Fixed Decoupled Newton-Raphson (FDNR) — driver, settings, shared safeguards, the two FD
# iteration loops (polar `:decoupled` B′/B″ half-steps and the frozen-Jacobian `:fixed_jacobian`
# variant), the opt-in handoff stage, and the factor-once `FastDecoupledCache`.
#
# B′/B″ assembly lives in fast_decoupled_matrices.jl.

"""
    _default_fd_variant(pf::AbstractACPowerFlow) -> Symbol

The default `fd_variant` for a [`FastDecoupledACPowerFlow`](@ref) on a given formulation:
`:decoupled` (classic B′/B″) for the polar formulation, `:fixed_jacobian` (frozen Jacobian) for
the rectangular current-injection and mixed current-power-balance formulations.
"""
_default_fd_variant(::ACPolarPowerFlow) = :decoupled
_default_fd_variant(::ACRectangularPowerFlow) = :fixed_jacobian
_default_fd_variant(::ACMixedPowerFlow) = :fixed_jacobian

"""
    _validate_fd_settings(pf, fd_variant, fd_scheme, handoff_solver)

Data-free validation of the [`FastDecoupledACPowerFlow`](@ref) settings. Throws a descriptive
`ArgumentError` on an invalid `fd_variant`/`fd_scheme`, on a `:decoupled` request against a
non-polar formulation (decoupled is polar-only in v1; rectangular/mixed decoupled is the gated
WP7), or on an unsupported `handoff_solver`. Returns `nothing` when the configuration is valid.
"""
function _validate_fd_settings(
    pf::AbstractACPowerFlow{FastDecoupledACPowerFlow},
    fd_variant::Symbol,
    fd_scheme::Symbol,
    handoff_solver,
)
    if !(fd_variant in (:decoupled, :fixed_jacobian))
        throw(
            ArgumentError(
                "FastDecoupled: invalid fd_variant $(fd_variant). " *
                "Must be :decoupled (classic B′/B″, polar only) or :fixed_jacobian " *
                "(frozen Jacobian, all formulations).",
            ),
        )
    end
    if !(fd_scheme in (:XB, :BX))
        throw(
            ArgumentError(
                "FastDecoupled: invalid fd_scheme $(fd_scheme). " *
                "Must be :XB (Stott–Alsac) or :BX (van Amerongen).",
            ),
        )
    end
    if fd_variant == :decoupled && !(pf isa ACPolarPowerFlow)
        throw(
            ArgumentError(
                "FastDecoupled fd_variant=:decoupled is only supported on the polar " *
                "formulation (ACPolarPowerFlow/ACPowerFlow) in v1. For $(typeof(pf)), use " *
                "fd_variant=:fixed_jacobian (rectangular/mixed decoupled is the gated WP7).",
            ),
        )
    end
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
    _newton_power_flow(pf::AbstractACPowerFlow{FastDecoupledACPowerFlow}, data, time_step; ...)

Driver for the [`FastDecoupledACPowerFlow`](@ref) solver. Validates settings, applies the
data-dependent LCC guard for the polar `:decoupled` variant, then dispatches to
`_fd_decoupled_power_flow` (polar B′/B″ half-steps) or `_fd_fixed_jacobian_power_flow` (frozen
Jacobian) per `fd_variant`. Returns `converged::Bool`.
"""
function _newton_power_flow(
    pf::AbstractACPowerFlow{FastDecoupledACPowerFlow},
    data::ACPowerFlowData,
    time_step::Int64;
    tol = DEFAULT_NR_TOL,
    maxIterations = DEFAULT_FD_MAX_ITER,
    fd_variant = _default_fd_variant(pf),
    fd_scheme = DEFAULT_FD_SCHEME,
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
    _validate_fd_settings(pf, fd_variant, fd_scheme, handoff_solver)

    # Data-dependent guard: polar :decoupled half-iterations don't span the 4 trailing LCC state
    # entries per converter, so ‖Rv‖∞ can't converge with LCC present. Point users at the
    # LCC-capable paths.
    if fd_variant == :decoupled && get_lcc_count(data) > 0
        throw(
            ArgumentError(
                "FastDecoupled fd_variant=:decoupled does not support LCC HVDC " *
                "(get_lcc_count(data) = $(get_lcc_count(data))): the B′/B″ half-iterations " *
                "do not span the LCC state variables. Use fd_variant=:fixed_jacobian, a " *
                "rectangular/mixed FD formulation, or NewtonRaphsonACPowerFlow / " *
                "TrustRegionACPowerFlow.",
            ),
        )
    end

    if fd_variant == :decoupled
        return _fd_decoupled_power_flow(
            pf, data, time_step;
            tol,
            maxIterations,
            fd_scheme,
            fd_non_divergent,
            handoff_solver,
            handoff_tol,
            validate_voltage_magnitudes,
            vm_validation_range,
            x0,
            linear_solver,
            _ignored...,
        )
    else
        return _fd_fixed_jacobian_power_flow(
            pf, data, time_step;
            tol,
            maxIterations,
            refreeze_on_stall,
            fd_non_divergent,
            handoff_solver,
            handoff_tol,
            validate_voltage_magnitudes,
            vm_validation_range,
            x0,
            linear_solver,
            _ignored...,
        )
    end
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
    pf::AbstractACPowerFlow{FastDecoupledACPowerFlow},
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
    pf::AbstractACPowerFlow{FastDecoupledACPowerFlow},
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
        ss = sum(abs2, residual.Rv)
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
            ss = sum(abs2, residual.Rv)

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
                    ss = sum(abs2, residual.Rv)
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
# Classic Stott–Alsac / van Amerongen fast decoupled power flow: solve the active-power
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
# The cache is stored in `data.solver_cache[]` as a TAGGED tuple `(FD_CACHE_TAG, cache)` so any
# future cross-use of the DC-path slot fails loudly (the DC path stores a 4-tuple whose first
# element is a SparseMatrixCSC, never `FD_CACHE_TAG`).
#
# The :fixed_jacobian loop deliberately does NOT use this cache: its frozen Jacobian is
# evaluated at x0 and x0 changes per time step (loads change), so it cannot be reused.
# =====================================================================================

"""Tag marking `data.solver_cache[]` as holding a [`FastDecoupledCache`](@ref) (polar
:decoupled). The DC path stores a `(matrix, backend, cache, scratch)` tuple whose first element
is a `SparseMatrixCSC`, never this symbol — so the tag makes the two uses type-disjoint and any
collision fails loudly in [`_get_or_build_fd_cache!`](@ref)."""
const FD_CACHE_TAG = :fdnr_cache

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
    FastDecoupledCache

Factor-once cache for the polar :decoupled FD loop, stored in `data.solver_cache[]` as
`(FD_CACHE_TAG, cache)`. Holds the invalidation key, the constant [`FDMatrices`](@ref) (recovered
params + factored B′ + assembled B″_full), the `pvpq`-invariant half-step buffers/index vectors
(factored ONCE per `(data, scheme, backend)` lifetime), and a `Dict` of per-PQ-set
[`FDPQData`](@ref) keyed on a bus-type signature. `bp_factor_count`/`bpp_factor_count` count B′ and
B″ factorizations for testability (factor-once verification).

# Fields
- `ybus_id::UInt`: `objectid(data.power_network_matrix)` — invalidation key (network identity).
- `scheme::Symbol`: `:XB`/`:BX` — invalidation key.
- `backend_id::DataType`: `typeof(resolve_linear_solver_backend(linear_solver))` — invalidation key.
- `fd::FDMatrices`: recovered params + factored B′ + B″_full.
- `pvpq::Vector{Int}`: non-REF bus indices (`== fd.pvpq`).
- `theta_x_idx::Vector{Int}`: `x`-indices of the θ state at `pvpq` (`2i`).
- `p_row_idx::Vector{Int}`: `Rv`-indices of the P-mismatch rows at `pvpq` (`2i-1`).
- `rp::Vector{Float64}`: preallocated active half-step buffer (length `length(pvpq)`).
- `pq_data::Dict{Vector{PSY.ACBusTypes}, FDPQData}`: bus-type column → per-PQ-set data (the
  materialized column is the key, so distinct PQ sets can never collide).
- `bp_factor_count::Int`: number of B′ factorizations (must be 1 over the cache lifetime).
- `bpp_factor_count::Int`: number of B″ factorizations (one per distinct PQ signature).
"""
mutable struct FastDecoupledCache
    ybus_id::UInt
    scheme::Symbol
    backend_id::DataType
    fd::FDMatrices
    pvpq::Vector{Int}
    theta_x_idx::Vector{Int}
    p_row_idx::Vector{Int}
    rp::Vector{Float64}
    pq_data::Dict{Vector{PSY.ACBusTypes}, FDPQData}
    bp_factor_count::Int
    bpp_factor_count::Int
end

"""
    _fd_backend_id(linear_solver) -> DataType

The backend identity used as a [`FastDecoupledCache`](@ref) invalidation key: the concrete type
of the resolved linear-solver backend (e.g. `PNM.KLUSolver`). Matches the DC path's
`typeof(cached_backend) === typeof(backend)` reuse test."""
_fd_backend_id(linear_solver::Union{Nothing, AbstractString}) =
    typeof(resolve_linear_solver_backend(linear_solver))

"""
    _get_or_build_fd_cache!(data, time_step, scheme, backend_id, linear_solver)
        -> FastDecoupledCache

Fetch the cached [`FastDecoupledCache`](@ref) from `data.solver_cache[]`, or build it. Reuse
requires the slot to hold `(FD_CACHE_TAG, cache)` whose invalidation key (Ybus objectid, scheme,
backend identity) matches — then B′ and any per-PQ-set B″ are reused with NO refactorization.
If the slot holds a non-`nothing`, non-FD value (e.g. the DC path's tuple) this `error`s loudly
(cache collision — the slot must stay type-disjoint). Otherwise builds via `build_fd_matrices`
(B′ factored once ⇒ `bp_factor_count = 1`), precomputes the `pvpq`-invariant buffers, and stores
the tagged cache."""
function _get_or_build_fd_cache!(
    data::ACPowerFlowData,
    time_step::Int64,
    scheme::Symbol,
    backend_id::DataType,
    linear_solver::Union{Nothing, AbstractString},
)
    ybus_id = objectid(data.power_network_matrix)
    entry = data.solver_cache[]
    if entry !== nothing
        if entry isa Tuple{Symbol, FastDecoupledCache} && entry[1] === FD_CACHE_TAG
            cache = entry[2]
            if cache.ybus_id == ybus_id &&
               cache.scheme === scheme &&
               cache.backend_id === backend_id
                return cache
            end
            # Same slot, different (network, scheme, backend): rebuild below.
        else
            error(
                "FastDecoupled: data.solver_cache[] holds an unexpected value " *
                "$(typeof(entry)); the AC/FD path expected either `nothing` or a " *
                "`(FD_CACHE_TAG, FastDecoupledCache)` tuple. This indicates a cross-use " *
                "collision with the DC-path solver cache — the slot must stay type-disjoint.",
            )
        end
    end

    fd = build_fd_matrices(data, time_step, scheme; linear_solver)   # B′ assembled + factored ONCE
    theta_x_idx = [2 * i for i in fd.pvpq]
    p_row_idx = [2 * i - 1 for i in fd.pvpq]
    rp = Vector{Float64}(undef, length(fd.pvpq))
    cache = FastDecoupledCache(
        ybus_id,
        scheme,
        backend_id,
        fd,
        copy(fd.pvpq),
        theta_x_idx,
        p_row_idx,
        rp,
        Dict{Vector{PSY.ACBusTypes}, FDPQData}(),
        1,   # bp_factor_count: build_fd_matrices factored B′ exactly once
        0,   # bpp_factor_count: bumped per distinct PQ signature in _get_pq_data!
    )
    data.solver_cache[] = (FD_CACHE_TAG, cache)
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
    for (ref_bus, subnet) in residual.subnetworks
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
    _fd_decoupled_power_flow(pf, data, time_step; ...) -> (converged::Bool, iters::Int)

Polar classic fast-decoupled (B′/B″ half-iteration) FD loop. Builds the constant B′/B″ matrices
once (factor-once via `build_fd_matrices`/`extract_bpp`), then iterates strict P-θ → Q-V
half-steps with an exact residual re-evaluation after EACH half-step (van Amerongen: the
mid-cycle refresh prevents convergence cycling — do NOT skip it). Shared WP2 safeguards
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
    pf::AbstractACPowerFlow{FastDecoupledACPowerFlow},
    data::ACPowerFlowData,
    time_step::Int64;
    tol::Float64 = DEFAULT_NR_TOL,
    maxIterations::Int = DEFAULT_FD_MAX_ITER,
    fd_scheme::Symbol = DEFAULT_FD_SCHEME,
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

    solver_name = "FastDecoupled(:decoupled,$(fd_scheme))"

    # Factor-once cache (WP5b): B′ over fd.pvpq factored exactly once per (data, scheme, backend)
    # lifetime; the [pq, pq] B″ submatrix + half-step buffers factored once per distinct PQ set
    # (bus-type signature) and reused across Q-limit retries / multi-period steps. The hot loop
    # fetches its matrices, index vectors, and rp/rq buffers from the cache (no per-invocation
    # build_fd_matrices/extract_bpp call, no per-iteration allocation). Behavior is identical to
    # building them inline — only WHERE they come from changes (T2/T6 arbitrate equivalence).
    backend_id = _fd_backend_id(linear_solver)
    cache = _get_or_build_fd_cache!(data, time_step, fd_scheme, backend_id, linear_solver)
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

    # Sync explicit rows, then evaluate the residual so Rv / data reflect (V, θ, explicit P/Q).
    _sync_explicit_state!(sv, residual, time_step)
    residual(sv.x, time_step)
    ss = sum(abs2, residual.Rv)
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

        ss = sum(abs2, residual.Rv)

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
                ss = sum(abs2, residual.Rv)
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
