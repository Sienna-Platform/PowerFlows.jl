# Analytic sensitivity for the discrete-control continuation: dy/dp from the factored power
# flow Jacobian instead of a finite-difference probe.
#
# This lives in its own file, included AFTER every formulation's residual and Jacobian, because
# the methods here dispatch on those concrete types in their SIGNATURES (which resolve at
# definition time). `control_discrete_devices/control_continuation.jl` is included before them,
# so the dispatch cannot live there.
#
# Adding a formulation means adding four methods here — `_sensitivity_residual_jacobian`,
# `_dF_dp!`, `_dVm_from_sol`, and (to earn batched passes) `_refresh_sensitivity_context!` plus
# `_refreshable` — and touching no device code and no continuation logic.

# The residual/Jacobian pair for a formulation, built and evaluated at the state currently in
# `data`. One method per formulation; `_sensitivity_context` below is generic over them.
#
# Deliberately NOT `initialize_power_flow_variables`: all of its methods route through
# `improve_x0` (previous-time-step warm-start comparison, enhanced flat start, DC fallback), so
# it would build the context at a warm-start CANDIDATE rather than at the converged base — and
# every gain read off it would be wrong by an unbounded amount.
function _sensitivity_residual_jacobian(::ACPolarPowerFlow, data, ts::Int)
    residual = ACPowerFlowResidual(data, ts)
    x = calculate_x0(data, ts)
    residual(x, ts)                       # evaluate at current state; fills P_net/Q_net
    J = ACPowerFlowJacobian(residual, ts)
    J(ts)                                 # Jacobian values at current state
    return residual, J
end

# No analytic form for this formulation ⇒ the caller uses FD probes. Kept as the escape for any
# formulation added later without a `_sensitivity_residual_jacobian` method.
_sensitivity_context(::AbstractACPowerFlow, data, ts::Int; kwargs...) = nothing
function _sensitivity_context(pf::ACPolarPowerFlow, data, ts::Int; kwargs...)
    backend = resolve_linear_solver_backend(get(kwargs, :linear_solver, nothing))
    residual, J = _sensitivity_residual_jacobian(pf, data, ts)
    lin_cache =
        _nr_linear_solver_cache!(data, J, backend, residual.bus_slack_participation_factors)
    try
        numeric_refactor!(lin_cache, J.Jv)
    catch e
        e isa LinearAlgebra.SingularException || rethrow()
        return                    # singular base Jacobian ⇒ fall back to FD probes
    end
    # One full sensitivity-context build (fresh residual + Jacobian structure + refactor) per
    # continuation. `_refresh_sensitivity_context!` re-evaluates VALUES into these same objects
    # on every subsequent batched pass and does NOT count here — the Jacobian structure (and thus
    # the persisted symbolic factorization this counter tracks) is topology-invariant across the
    # continuation, so only this one build should ever register.
    _count_symbolic_factor!(data)
    n = length(residual.Rv)
    return _SensitivityContext(
        lin_cache, residual, J, zeros(n), zeros(n), copy(view(data.bus_type, :, ts)))
end

# Whether a live context can be kept current across batched passes, i.e. whether its
# formulation has a `_refresh_sensitivity_context!` that re-syncs every p-dependent cache.
# Gates `use_batched`. A formulation that supplies analytic probe sensitivities but no refresh
# stays on the sequential path: batching it would read a Jacobian that went stale the moment a
# device moved, and be wrong on every pass after the first.
_supports_batched_refresh(::Nothing) = false
_supports_batched_refresh(ctx::_SensitivityContext) = _refreshable(ctx.residual)
_refreshable(::ACPowerFlowResidual) = true

function _refresh_sensitivity_context!(ctx::_SensitivityContext, data, ts::Int)::Bool
    view(data.bus_type, :, ts) == ctx.bus_type || return false
    residual = ctx.residual
    copyto!(
        residual.bus_active_constant_I,
        view(data.bus_active_power_constant_current_withdrawals, :, ts),
    )
    copyto!(
        residual.bus_reactive_constant_I,
        view(data.bus_reactive_power_constant_current_withdrawals, :, ts),
    )
    copyto!(
        residual.bus_active_constant_Z,
        view(data.bus_active_power_constant_impedance_withdrawals, :, ts),
    )
    copyto!(
        residual.bus_reactive_constant_Z,
        view(data.bus_reactive_power_constant_impedance_withdrawals, :, ts),
    )
    @inbounds for ix in eachindex(residual.P_net)
        residual.P_net[ix] =
            data.bus_active_power_injections[ix, ts] -
            get_bus_active_power_total_withdrawals(data, ix, ts) +
            data.bus_hvdc_net_power[ix, ts]
        residual.Q_net[ix] =
            data.bus_reactive_power_injections[ix, ts] -
            get_bus_reactive_power_total_withdrawals(data, ix, ts)
        residual.P_net_set[ix] = residual.P_net[ix]
    end
    x = calculate_x0(data, ts)
    ctx.residual(x, ts)
    ctx.J(ts)
    try
        numeric_refactor!(ctx.lin_cache, ctx.J.Jv)
    catch e
        e isa LinearAlgebra.SingularException || rethrow()
        return false
    end
    return true
end

# ∂F/∂p into `rhs` (zeroed first). Returns `true` iff the family has an analytic polar form here;
# the `false` path is reserved for a future family without one — the caller then uses the FD probe
# (see `_linear_plant_sign`). Row convention: F[2b−1] active, F[2b] reactive balance at bus b
# (see `_update_residual_values!`).
function _dF_dp!(
    rhs::Vector{Float64},
    d::ControlledTap,
    ::ACPowerFlowResidual,
    data,
    ts::Int,
)
    fill!(rhs, 0.0)
    f, t = d.from_ix, d.to_ix
    Vf = data.bus_magnitude[f, ts] * cis(data.bus_angles[f, ts])
    Vt = data.bus_magnitude[t, ts] * cis(data.bus_angles[t, ts])
    p, a = d.current, d.alpha
    # ∂Y/∂p of the from-side terms (t_c = p·cis(a)); Y_tt = yt is p-independent (see `_branch_terms`).
    dYff = -2.0 * d.yt / p^3
    dYft = d.yt * cis(a) / p^2
    dYtf = d.yt * cis(-a) / p^2
    # ∂S_i/∂p = V_i·conj(Σ_k ∂Y_ik/∂p·V_k); only Y_ff,Y_ft (row f) and Y_tf (row t) change.
    dSf = Vf * conj(dYff * Vf + dYft * Vt)
    dSt = Vt * conj(dYtf * Vf)
    @inbounds begin
        rhs[2 * f - 1] = real(dSf)
        rhs[2 * f] = imag(dSf)
        rhs[2 * t - 1] = real(dSt)
        rhs[2 * t] = imag(dSt)
    end
    return true
end

function _dF_dp!(
    rhs::Vector{Float64},
    d::Union{ControlledSwitchedShunt, ControlledFACTS},
    ::ACPowerFlowResidual,
    data,
    ts::Int,
)
    fill!(rhs, 0.0)
    b = d.bus_ix
    Vm = data.bus_magnitude[b, ts]
    # Constant-Z reactive withdrawal w enters Q_net as −w·Vm²; apply_parameter! sets ∂w/∂susc = −1,
    # so ∂Q_net/∂susc = +Vm² and ∂F[2b]/∂susc = −∂Q_net/∂susc = −Vm² (reactive row only).
    @inbounds rhs[2 * b] = -Vm^2
    return true
end

# Linearized dy/dp for a voltage device via the factored Jacobian. `y = Vm(controlled_bus)`
# = x[2·cbus−1], and dx/dp = −J⁻¹·(∂F/∂p). Returns `(dy/dp, true)`, or `(0.0, false)` when the
# family has no analytic form (caller then uses the FD probe).
# dVm(cbus)/dp from `sol = J⁻¹·(∂F/∂p)`, given `dx/dp = −sol`. One method per formulation:
# each knows where its own state keeps the controlled bus's voltage.
# Polar: x[2b−1] is Vm directly.
_dVm_from_sol(::ACPowerFlowResidual, sol::Vector{Float64}, cbus::Int, data, ts::Int) =
    -sol[2 * cbus - 1]

function _linear_plant_sign(d, data, ts::Int, ctx::_SensitivityContext)
    _dF_dp!(ctx.rhs, d, ctx.residual, data, ts) || return 0.0, false
    cbus = controlled_bus_ix(d)
    # The controlled bus's voltage is a free state variable ONLY at a PQ bus (polar keeps Q_gen
    # there at PV and P at REF; the rectangular/mixed formulations pin |V|² with their own
    # constraint row). At a PV/REF controlled bus the voltage is pinned by the bus model, so
    # dVm/dp = 0 exactly: return a reliable zero so the caller's gain floor freezes the device,
    # matching the FD probe's behavior.
    if data.bus_type[cbus, ts] != PSY.ACBusTypes.PQ
        return 0.0, true
    end
    copyto!(ctx.sol, ctx.rhs)
    solve!(ctx.lin_cache, ctx.sol)        # sol = J⁻¹·(∂F/∂p)
    return _dVm_from_sol(ctx.residual, ctx.sol, cbus, data, ts), true
end

# Shared bisection sub-step walk for the robust continuation applicators
# (`_continuation_to!` and `_restore_one!`). The full move having failed, step from `start`
# toward `target` (each interpolated point clamped to `[lo, hi]`), growing the step on NR
# success and halving it on failure; give up below `MIN_LAMBDA_STEP`. On entry `snap` holds
# the converged state at `start` and the device/data sit at `start`. Leaves the system at the
# last converged iterate and returns `(reached, completed)`: the parameter reached (== `start`
# if nothing budged) and whether the full `[start, target]` interval was traversed.
function _bisection_walk!(d, data, ts::Int, start::Float64, target::Float64,
    lo::Float64, hi::Float64, pf, snap::ControlStateSnapshot; kwargs...)
    done = 0.0                       # fraction of [start, target] applied so far
    step = 0.5
    reached = start
    while done < 1.0
        trial = min(1.0, done + step)
        p = clamp(start + trial * (target - start), lo, hi)
        apply_parameter!(d, data, p, ts)
        if _ctrl_solve!(pf, data, ts; kwargs...)
            done = trial
            reached = p
            _capture_state!(snap, data, ts)
            step = min(step * CONTROL_STEP_GROWTH, MAX_LAMBDA_STEP)
        else
            # Revert to the last converged parameter and state, not the failed trial.
            apply_parameter!(d, data, reached, ts)
            _restore_state!(data, ts, snap)
            step /= 2.0
            step < MIN_LAMBDA_STEP && return reached, false
        end
    end
    return reached, true
end

# Incremental robust applicator: walk the parameter from `start = d.current`
# toward `target` so NR stays converged.  The full move is tried FIRST (one
# inner solve in the common case); only if it fails does the walk fall back to
# bisection sub-stepping (via `_bisection_walk!`).  Returns `(reached, moved)`:
# the parameter actually reached (solver left converged there) and whether ANY
# sub-step was applied — a requested move that could not budge at all must not
# masquerade as a settled device.
function _continuation_to!(
    d, data, ts::Int, target::Float64, pf, snap::ControlStateSnapshot; kwargs...,
)
    start = current_parameter(d)
    abs(target - start) < _param_tol(d) && return start, true
    lo, hi = parameter_limits(d)
    _capture_state!(snap, data, ts)  # last converged state, restored on a failed trial
    clamped = clamp(target, lo, hi)  # no-op here (damped target is pre-clamped)
    # Full move first: the damped target is usually within the warm-started solver's
    # reach, so the common case costs ONE inner solve instead of a multi-sub-step walk.
    apply_parameter!(d, data, clamped, ts)
    _ctrl_solve!(pf, data, ts; kwargs...) && return clamped, true
    apply_parameter!(d, data, start, ts)
    _restore_state!(data, ts, snap)
    # Bisection fallback: the full step failed, so retry from half the interval.
    reached, completed =
        _bisection_walk!(d, data, ts, start, target, lo, hi, pf, snap; kwargs...)
    completed && return reached, true
    if reached != start
        # Re-solve from the restored converged state (partial move applied).
        _ctrl_solve!(pf, data, ts; kwargs...)
    end
    return reached, reached != start
end

# Adaptive under-relaxation. The damped iteration p ← p + ω·(σ(V(p)) − p) has local
# slope m = 1 + ω·(g′−1), g′ = σ′(V)·dV/dp ≤ 0 after sign correction. ω is chosen to
# keep m NON-negative (monotone, 0≤m<1, not merely |m|<1): m ≥ θ ⟹ ω ≤ (1−θ)/(1+|g′|).
# |σ′| ≤ |hi−lo|·S/4 bounds g′ at the CURRENT gain estimate (refreshed each step by a
# secant update, so the bound tracks the operating point).
@inline function _relaxation(d, S::Float64, dVdp::Float64)
    lo, hi = parameter_limits(d)
    gbound = 0.25 * abs(hi - lo) * S * abs(dVdp)
    # ω ≤ 1−θ = 0.5 for any gbound ≥ 0, so no additional cap is needed.
    return (1.0 - CONTROL_CONTRACTION) / (1.0 + gbound)
end

# Freeze a device (PSS/E lock-and-continue): it holds its current parameter, counts as
# settled so it cannot stall the steepness ramp for healthy devices, and is reported.
function _freeze_device!(frozen::Vector{Bool}, idx::Int, d, ts::Int, reason::String)
    frozen[idx] = true
    @warn "discrete control: freezing device $(d.name) at parameter \
        $(current_parameter(d)) (time step $ts): $reason"
    return
end

# Refresh one ControlledFACTS's voltage-dependent susceptance bound (`_facts_b_limit`) from its
# measured controlled-bus voltage, once per continuation pass, BEFORE this pass's damped
# target/clamp reads `parameter_limits`. A `frozen` device holds `b_lim` at its last refreshed
# value: it is never stepped again, so re-tracking voltage would only chatter the bound with no
# corresponding parameter move.
function _refresh_facts_limit!(d::ControlledFACTS, data, ts::Int, frozen::Bool)
    frozen && return d.b_lim
    d.b_lim = _facts_b_limit(d, measured_value(d, data, ts))
    return d.b_lim
end

# Per-pass refresh of every FACTS device's bound (function-barrier group wrapper, offset-
# indexed into the shared `frozen` bookkeeping — mirrors `_probe_device_signs!`).
function _refresh_facts_limits!(
    devices::Vector{ControlledFACTS}, offset::Int, data, ts::Int, frozen::Vector{Bool},
)
    for (i, d) in enumerate(devices)
        _refresh_facts_limit!(d, data, ts, frozen[offset + i])
    end
    return
end

# Compute the damped, sign-corrected target parameter for one device and advance its
# oscillation state. Returns `(p_new, yc, ok)`: `yc` is the pre-move regulated quantity (for a
# secant refresh); `ok=false` means the device was frozen (oscillation) or held in its deadband —
# do not move it. Shared by the sequential (`_step_device!`) and batched (`_batched_pass!`) paths.
function _damped_target!(
    d, idx::Int, data, ts::Int, S::Float64, frozen::Vector{Bool},
    dVdp::Vector{Float64}, osc::Vector{Int}, prev_sign::Vector{Int}, n_shared::Vector{Int},
)
    yc = measured_value(d, data, ts)
    p_now = current_parameter(d)
    if _in_deadband(d, yc)
        # PSS/E deadband semantics: a device whose regulated quantity is inside its
        # band is held, not driven to the band midpoint.
        prev_sign[idx] = 0
        return p_now, yc, false
    end
    lo, hi = parameter_limits(d)
    tol_d = _param_tol(d)
    dv = dVdp[idx]
    # Devices sharing a controlled bus split the correction: the per-device contraction
    # bound does not see the cross-coupling, and N co-located controllers stepping the
    # full error together have an in-phase gain ≈ N× the measured self-gain.
    ω = _relaxation(d, S, dv) / n_shared[idx]
    p_tgt = _control_target(d, yc, S, dv)
    # Track sign reversals to detect within-stage oscillation. Sub-tolerance target
    # moves carry no direction information (grid/tolerance dither, not instability).
    s = 0
    if abs(p_tgt - p_now) >= tol_d
        s = Int(sign(p_tgt - p_now))
    end
    ps = prev_sign[idx]
    if !iszero(ps) && !iszero(s) && s != ps
        osc[idx] += 1
        if osc[idx] > CONTROL_OSCILLATION_LIMIT
            _freeze_device!(frozen, idx, d, ts,
                "oscillating ($(osc[idx]) direction reversals within a steepness stage)")
            return p_now, yc, false
        end
    end
    prev_sign[idx] = s
    return clamp(p_now + ω * (p_tgt - p_now), lo, hi), yc, true
end

# Freeze on a detected plant-gain sign reversal (OLTC reverse action) or a collapse below the
# effectiveness floor; otherwise accept the refreshed gain `g`. `dv` is the pre-refresh gain.
function _apply_gain_refresh!(d, idx::Int, data, ts::Int, frozen::Vector{Bool},
    dVdp::Vector{Float64}, dv::Float64, g::Float64)
    lo, hi = parameter_limits(d)
    if !iszero(dv) && !iszero(g) && sign(g) != sign(dv)
        _freeze_device!(frozen, idx, d, ts,
            "plant sensitivity changed sign along the trajectory (reverse action); \
            continuing would be positive feedback")
    elseif abs(g) * (hi - lo) < CONTROL_GAIN_FLOOR
        _freeze_device!(frozen, idx, d, ts,
            "plant sensitivity collapsed below the effectiveness floor \
            (|dy/dp|·range = $(abs(g) * (hi - lo)))")
    else
        dVdp[idx] = g
    end
    return
end

# Secant refresh gate: a measured Δy at or below solver noise carries no sign
# information — skip the refresh entirely (both the reverse-action freeze AND the
# effectiveness-floor freeze would be spurious on a noise sample).
function _maybe_refresh_gain!(d, idx::Int, data, ts::Int, frozen::Vector{Bool},
    dVdp::Vector{Float64}, dv::Float64, g::Float64, Δy::Float64)
    abs(Δy) < CONTROL_MEASUREMENT_FLOOR && return
    _apply_gain_refresh!(d, idx, data, ts, frozen, dVdp, dv, g)
    return
end

# One damped, sign-corrected proportional update of a single device (SEQUENTIAL path: apply +
# solve per device). Returns the magnitude of the parameter change actually applied (for the
# settling test); frozen and in-deadband devices return 0.0. The measured plant gain is refreshed
# by a secant update from the numbers the step just produced (zero extra solves).
function _step_device!(
    d,
    idx::Int,
    data,
    ts::Int,
    S::Float64,
    pf,
    scratch_snap::ControlStateSnapshot,
    frozen::Vector{Bool},
    dVdp::Vector{Float64},
    osc::Vector{Int},
    prev_sign::Vector{Int},
    n_shared::Vector{Int};
    kwargs...,
)::Float64
    frozen[idx] && return 0.0
    p_new, yc, ok =
        _damped_target!(d, idx, data, ts, S, frozen, dVdp, osc, prev_sign, n_shared)
    ok || return 0.0
    p_now = current_parameter(d)
    tol_d = _param_tol(d)
    dv = dVdp[idx]
    reached, moved = _continuation_to!(d, data, ts, p_new, pf, scratch_snap; kwargs...)
    if !moved && abs(p_new - p_now) >= tol_d
        # Inner solver rejects any movement — freeze rather than let a zero change look settled.
        _freeze_device!(frozen, idx, d, ts,
            "the inner solver rejects any parameter movement (requested \
            $(p_new - p_now))")
        return 0.0
    end
    Δp = reached - p_now
    if abs(Δp) >= tol_d
        Δy = measured_value(d, data, ts) - yc
        _maybe_refresh_gain!(d, idx, data, ts, frozen, dVdp, dv, Δy / Δp, Δy)
    end
    return abs(Δp)
end
