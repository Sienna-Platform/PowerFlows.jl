# Target slope of the local iteration map near the fixed point (0<m<1 ⇒ monotone,
# non-oscillatory). 0.5 trades settling speed for a 2× margin on the worst-case gain bound.
# It also bounds the relaxation factor itself: ω = (1−θ)/(1+gbound) ≤ 1−θ = 0.5.
const CONTROL_CONTRACTION = 0.5

# State a rolled-back trial must not permanently change: V/θ, `bus_type` (one-way PV→PQ flips), the
# injection columns (Q clamps, distributed-slack setpoint updates), and the VSC DC-network state
# (`_read_vsc_state!` writes the iterate into dcn.p_c/q_c/node_vdc every residual eval, so a
# diverged trial corrupts it exactly like bus state).
@inline function _dc_state_cols(data, ts::Int)
    dcn = get_dc_network(data)
    if has_dc_network(dcn)
        return dcn.p_c[:, ts], dcn.q_c[:, ts], dcn.node_vdc[:, ts]
    end
    return Float64[], Float64[], Float64[]
end
@inline function _snapshot_state(data, ts::Int)
    dc_p, dc_q, dc_v = _dc_state_cols(data, ts)
    return (
        data.bus_magnitude[:, ts],
        data.bus_angles[:, ts],
        data.bus_type[:, ts],
        data.bus_active_power_injections[:, ts],
        data.bus_reactive_power_injections[:, ts],
        dc_p, dc_q, dc_v,
    )
end
@inline function _capture_state!(
    (vmag, vang, btype, pinj, qinj, dc_p, dc_q, dc_v),
    data,
    ts::Int,
)
    vmag .= view(data.bus_magnitude, :, ts)
    vang .= view(data.bus_angles, :, ts)
    btype .= view(data.bus_type, :, ts)
    pinj .= view(data.bus_active_power_injections, :, ts)
    qinj .= view(data.bus_reactive_power_injections, :, ts)
    if !isempty(dc_v)
        dcn = get_dc_network(data)
        dc_p .= view(dcn.p_c, :, ts)
        dc_q .= view(dcn.q_c, :, ts)
        dc_v .= view(dcn.node_vdc, :, ts)
    end
    return
end
@inline function _restore_state!(
    data,
    ts::Int,
    (vmag, vang, btype, pinj, qinj, dc_p, dc_q, dc_v),
)
    data.bus_magnitude[:, ts] .= vmag
    data.bus_angles[:, ts] .= vang
    data.bus_type[:, ts] .= btype
    data.bus_active_power_injections[:, ts] .= pinj
    data.bus_reactive_power_injections[:, ts] .= qinj
    if !isempty(dc_v)
        dcn = get_dc_network(data)
        dcn.p_c[:, ts] .= dc_p
        dcn.q_c[:, ts] .= dc_q
        dcn.node_vdc[:, ts] .= dc_v
    end
    return
end

# Counting wrapper around the inner solve: iteration counts are the repo's robust
# performance metric, and the continuation's cost IS its inner-solve count.
@inline function _ctrl_solve!(pf, data, ts::Int; kwargs...)
    cd = data.controlled_devices
    isnothing(cd) || (cd.inner_solves[] += 1)
    return _solve_with_q_limits!(pf, data, ts; kwargs...)
end

# Sign-corrected sigmoid law: orientation comes from the MEASURED dy/dp (not the
# device's primary/secondary wiring) so the closed-loop gain σ'(y)·dy/dp ≤ 0
# (negative feedback) regardless of wiring. `_sigmoid(lo,hi,…)` decreases in y when hi>lo.
# `y` is the regulated quantity — bus voltage for every currently supported device family.
@inline function _control_target(d, y::Float64, S::Float64, dydp::Float64)
    lo, hi = parameter_limits(d)
    return if dydp > 0.0
        clamp(_sigmoid(lo, hi, S, y, control_setpoint(d)), lo, hi)
    else
        clamp(_sigmoid(hi, lo, S, y, control_setpoint(d)), lo, hi)
    end
end

# Scale-aware settle tolerance: 1e-5 absolute is 5e-5 relative for a 0.2-wide tap band
# but only 1e-6 relative for a 10 p.u. shunt; a relative floor lets wide-range devices
# settle in comparable pass counts. Sub-grid precision is wasted anyway — final values
# snap to discrete grids.
@inline function _param_tol(d)
    lo, hi = parameter_limits(d)
    return max(CONTROL_PARAM_TOL, CONTROL_PARAM_RTOL * (hi - lo))
end

# Measure dy/dp (sensitivity of the regulated quantity to the parameter) by a small
# parameter perturbation. The probe is definitionally a temporary excursion: the full
# pre-probe state (incl. any Q-limit flips the probe solve caused) is restored afterward,
# which also makes the second re-converging solve unnecessary. `reliable=false` (the
# probe solve failed) ⇒ orientation unknown, so the caller freezes the device rather
# than stepping it with an unknown sign.
function _plant_sign(d, data, ts::Int, pf, scratch_snap; kwargs...)::Tuple{Float64, Bool}
    p0 = current_parameter(d)
    y0 = measured_value(d, data, ts)
    _capture_state!(scratch_snap, data, ts)
    snap = scratch_snap
    lo, hi = parameter_limits(d)
    δ = 1e-3 * (hi - lo)
    if δ <= 0.0
        δ = 1e-6
    end
    p1 = clamp(p0 + δ, lo, hi)
    p1 == p0 && (p1 = clamp(p0 - δ, lo, hi))
    h = p1 - p0
    apply_parameter!(d, data, p1, ts)
    ok = _ctrl_solve!(pf, data, ts; kwargs...)
    y1 = measured_value(d, data, ts)
    apply_parameter!(d, data, p0, ts)
    _restore_state!(data, ts, snap)
    reliable = ok && !iszero(h)
    dVdp = 0.0
    if reliable
        dVdp = (y1 - y0) / h
    end
    return dVdp, reliable
end

# ── Linearized plant sensitivities ───────────────────────────────────────────────────────
# Differentiating F(x,p)=0 at the converged base: dx/dp = −J⁻¹·(∂F/∂p), and dy/dp is the
# controlled-bus voltage component — no perturbation, one triangular solve per device on the base
# NR solve's reused factorization instead of a full nonlinear solve. Polar + voltage-device (tap/shunt/FACTS)
# only (state layout x[2b−1]=Vm, x[2b]=Va and ∂F/∂p are polar-specific); rect/mixed formulations
# fall back to the FD `_plant_sign`. Signs are validated against the FD probe in the tests.

# Sensitivity context: residual+Jacobian built at the CURRENT converged base state and numerically
# factored (reusing the base NR solve's persisted symbolic factorization). Built ONCE per
# continuation (probe phase) and thereafter kept current by `_refresh_sensitivity_context!`
# (values-only, no rebuild) after each batched-pass joint solve. `nothing` ⇒ the linear path is
# unavailable (non-polar formulation, or the base Jacobian is singular) and the caller uses FD probes.
struct _SensitivityContext{C, R, JT}
    lin_cache::C
    residual::R             # persisted; re-evaluated in place by _refresh_sensitivity_context!
    J::JT                   # persisted; re-evaluated in place by _refresh_sensitivity_context!
    rhs::Vector{Float64}    # ∂F/∂p scratch, refilled per device
    sol::Vector{Float64}    # J⁻¹·∂F/∂p scratch
    # Snapshot of bus_type at build time. `ACPowerFlowResidual`'s subnetworks/slack-participation
    # factors/validate_indices are computed ONCE at construction from bus_type and are NOT
    # recomputed by the residual functor — a Q-limit PV→PQ flip elsewhere in the network after
    # this ctx was built silently stales that structure. `_refresh_sensitivity_context!` checks
    # this snapshot and refuses to reuse a ctx whose bus-type pattern has since changed.
    bus_type::Vector{PSY.ACBusTypes}
end

_sensitivity_context(::AbstractACPowerFlow, data, ts::Int; kwargs...) = nothing
function _sensitivity_context(pf::ACPolarPowerFlow, data, ts::Int; kwargs...)
    backend = resolve_linear_solver_backend(get(kwargs, :linear_solver, nothing))
    residual = ACPowerFlowResidual(data, ts)
    x = calculate_x0(data, ts)
    residual(x, ts)                       # evaluate at current state; fills P_net/Q_net
    J = ACPowerFlowJacobian(residual, ts)
    J(ts)                                 # Jacobian values at current state
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

# Values-only refresh at a new converged base state: the Jacobian sparsity depends only on
# topology/REF layout (invariant across the continuation), so re-evaluating values + numeric
# refactor into the persisted objects replaces a full rebuild — UNLESS a PV↔PQ flip has changed
# the slack-participation/subnetwork layout the persisted `residual` baked in at construction
# (see `_SensitivityContext.bus_type`); that case, and a singular refactor, both return `false`
# so the caller falls back to the sequential path (which rebuilds fresh).
#
# `P_net`/`Q_net`/`P_net_set` and the four constant-I/Z withdrawal vectors are all captured from
# `data` ONCE at `ACPowerFlowResidual` construction and are NOT independently re-read by the
# residual functor: `_update_residual_values!`'s PQ-bus case does `P_net[ix] += ...` (a
# TELESCOPING correction from the bus's voltage at the residual's LAST evaluation to its new
# value), which is only correct when the SAME residual object is the one driving every
# evaluation of a single NR solve. Here `ctx.residual` is evaluated once per PASS while `data`'s
# voltage is moved BETWEEN passes by the joint solve's OWN, unrelated residual object — so
# `P_net`/`Q_net` must be rebuilt from `data` (exactly like the constructor) before every
# evaluation, or they silently accumulate a wrong, ever-growing correction. Likewise
# `apply_parameter!` for switched shunts/FACTS writes device moves directly into
# `data.bus_reactive_power_constant_impedance_withdrawals`, so the four constant-I/Z vectors need
# the same re-sync.
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
function _dF_dp!(rhs::Vector{Float64}, d::ControlledTap, data, ts::Int)
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
function _linear_plant_sign(d, data, ts::Int, ctx::_SensitivityContext)
    _dF_dp!(ctx.rhs, d, data, ts) || return 0.0, false
    cbus = controlled_bus_ix(d)
    # x[2b−1] is Vm ONLY at PQ buses (Q_gen at PV, P at REF — see update_state!). At a
    # PV/REF controlled bus the voltage is pinned by the bus model, so dVm/dp = 0 exactly:
    # return a reliable zero so the caller's gain floor freezes the device, matching the
    # FD probe's behavior.
    if data.bus_type[cbus, ts] != PSY.ACBusTypes.PQ
        return 0.0, true
    end
    copyto!(ctx.sol, ctx.rhs)
    solve!(ctx.lin_cache, ctx.sol)        # sol = J⁻¹·(∂F/∂p)
    return -ctx.sol[2 * cbus - 1], true       # dy/dp = (dx/dp)[Vm(cbus)] = −sol[2·cbus−1]
end

# Shared bisection sub-step walk for the robust continuation applicators
# (`_continuation_to!` and `_restore_one!`). The full move having failed, step from `start`
# toward `target` (each interpolated point clamped to `[lo, hi]`), growing the step on NR
# success and halving it on failure; give up below `MIN_LAMBDA_STEP`. On entry `snap` holds
# the converged state at `start` and the device/data sit at `start`. Leaves the system at the
# last converged iterate and returns `(reached, completed)`: the parameter reached (== `start`
# if nothing budged) and whether the full `[start, target]` interval was traversed.
function _bisection_walk!(d, data, ts::Int, start::Float64, target::Float64,
    lo::Float64, hi::Float64, pf, snap; kwargs...)
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
function _continuation_to!(d, data, ts::Int, target::Float64, pf, scratch_snap; kwargs...)
    start = current_parameter(d)
    abs(target - start) < _param_tol(d) && return start, true
    lo, hi = parameter_limits(d)
    _capture_state!(scratch_snap, data, ts)  # last converged state, restored on a failed trial
    snap = scratch_snap
    clamped = clamp(target, lo, hi)          # no-op here (damped target is pre-clamped)
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
    scratch_snap,
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

# One device's plant sign: the linear sensitivity when a polar `ctx` is live and the family has an
# analytic form (tap/shunt/FACTS), else the FD probe (non-polar formulations, singular base).
_probe_one_sign(d, data, ts::Int, pf, scratch_snap, ::Nothing; kwargs...) =
    _plant_sign(d, data, ts, pf, scratch_snap; kwargs...)
function _probe_one_sign(
    d, data, ts::Int, pf, scratch_snap, ctx::_SensitivityContext; kwargs...,
)
    dydp, ok = _linear_plant_sign(d, data, ts, ctx)
    ok && return dydp, true
    return _plant_sign(d, data, ts, pf, scratch_snap; kwargs...)
end

# Probe every device in a concrete vector (function barrier — no dynamic dispatch over the
# heterogeneous set), writing dy/dp at `offset + i`. Freeze probes that fail or whose full-range
# effect is below the gain floor (e.g. a PV-pinned bus, sensitivity 0) — stepping them would rail.
function _probe_device_signs!(
    devices, offset::Int, dVdp::Vector{Float64}, frozen::Vector{Bool},
    ctx::Union{Nothing, _SensitivityContext}, data, ts::Int, pf, scratch_snap; kwargs...,
)
    for (i, d) in enumerate(devices)
        s, reliable = _probe_one_sign(d, data, ts, pf, scratch_snap, ctx; kwargs...)
        lo, hi = parameter_limits(d)
        if reliable && abs(s) * (hi - lo) >= CONTROL_GAIN_FLOOR
            dVdp[offset + i] = s
        else
            frozen[offset + i] = true
        end
    end
    return
end

# One proportional update over a concrete device vector; returns `true` iff every device
# in it settled (parameter change below its scale-aware tolerance). Same function-barrier
# + `offset` indexing.
function _step_device_group!(
    devices, offset::Int, data, ts::Int, S::Float64, pf, scratch_snap,
    frozen::Vector{Bool}, dVdp::Vector{Float64}, osc::Vector{Int},
    prev_sign::Vector{Int}, n_shared::Vector{Int}; kwargs...,
)::Bool
    settled = true
    for (i, d) in enumerate(devices)
        g = _step_device!(
            d, offset + i, data, ts, S, pf, scratch_snap, frozen, dVdp, osc, prev_sign,
            n_shared; kwargs...)
        g >= _param_tol(d) && (settled = false)
    end
    return settled
end

# ── Batched (Jacobi) pass ─────────────────────────────────────────────────────────────────
# Apply every device's damped target, then ONE joint inner solve (vs ~N sequential — the
# relaxation theory already models a pass as one damped joint update). Gains are refreshed
# analytically (a joint move corrupts the secant gain at co-located buses); a non-converged joint
# solve rolls the whole pass back and the caller falls to the sequential path.

# Apply every non-frozen device's damped target WITHOUT solving; record pre-move parameters in
# `p_prev`/`did_move` (indexed globally) for rollback and the post-solve gain refresh. Returns
# whether any device moved. Function-barrier per concrete device vector (no dynamic dispatch).
function _apply_targets_group!(
    devices, offset::Int, data, ts::Int, S::Float64, frozen::Vector{Bool},
    dVdp::Vector{Float64}, osc::Vector{Int}, prev_sign::Vector{Int}, n_shared::Vector{Int},
    p_prev::Vector{Float64}, did_move::Vector{Bool},
)
    any_moved = false
    for (i, d) in enumerate(devices)
        idx = offset + i
        frozen[idx] && continue
        p_now = current_parameter(d)
        p_new, _, ok =
            _damped_target!(d, idx, data, ts, S, frozen, dVdp, osc, prev_sign, n_shared)
        (ok && abs(p_new - p_now) >= _param_tol(d)) || continue
        p_prev[idx] = p_now
        apply_parameter!(d, data, p_new, ts)
        did_move[idx] = true
        any_moved = true
    end
    return any_moved
end

# Undo an applied batched pass: restore each moved device's parameter (delta-based apply reverses
# the Y-bus/withdrawal edit). The caller also `_restore_state!`s V/θ/bustype/injections.
function _rollback_targets_group!(
    devices, offset::Int, data, ts::Int, p_prev::Vector{Float64}, did_move::Vector{Bool},
)
    for (i, d) in enumerate(devices)
        idx = offset + i
        did_move[idx] || continue
        apply_parameter!(d, data, p_prev[idx], ts)
    end
    return
end

# Analytic (coupling-free) gain refresh for the moved devices of one group, using a `ctx` built at
# the post-solve state. Same sign-reversal / gain-floor freeze policy as the secant refresh.
function _refresh_gains_group!(
    devices, offset::Int, data, ts::Int, frozen::Vector{Bool}, dVdp::Vector{Float64},
    did_move::Vector{Bool}, ctx::_SensitivityContext,
)
    for (i, d) in enumerate(devices)
        idx = offset + i
        did_move[idx] || continue
        g, ok = _linear_plant_sign(d, data, ts, ctx)
        ok || continue
        _apply_gain_refresh!(d, idx, data, ts, frozen, dVdp, dVdp[idx], g)
    end
    return
end

# One batched pass over the voltage-device groups (taps, shunts, FACTS). Returns
# `(settled, converged)`: `converged=false` means the joint solve failed (or, rarely, the
# post-solve sensitivity refresh hit a singular Jacobian) and the pass was fully rolled back —
# the caller must run the sequential path for this pass. `ctx` is the ONE
# `_SensitivityContext` persisted for the whole continuation (built once in
# `_control_continuation!`); a converged joint solve refreshes it in place instead of rebuilding.
function _batched_pass!(
    set::ControlledDeviceSet, n_taps::Int, n_shunts::Int, data, ts::Int, S::Float64, pf,
    scratch_snap, frozen::Vector{Bool}, dVdp::Vector{Float64}, osc::Vector{Int},
    prev_sign::Vector{Int}, n_shared::Vector{Int}, p_prev::Vector{Float64},
    did_move::Vector{Bool}, ctx::_SensitivityContext; kwargs...,
)
    fill!(did_move, false)
    _capture_state!(scratch_snap, data, ts)
    snap = scratch_snap
    # A failed+rolled-back batched attempt must leave NO trace, so the sequential fallback is the
    # authoritative pass: snapshot the per-device bookkeeping `_damped_target!` mutates
    # (oscillation counter, direction, freezes) and restore it on rollback.
    osc0, prev0, frozen0 = copy(osc), copy(prev_sign), copy(frozen)
    off_facts = n_taps + n_shunts
    moved = _apply_targets_group!(set.taps, 0, data, ts, S, frozen, dVdp, osc, prev_sign,
        n_shared, p_prev, did_move)
    moved |= _apply_targets_group!(set.shunts, n_taps, data, ts, S, frozen, dVdp, osc,
        prev_sign, n_shared, p_prev, did_move)
    # FACTS branch: refresh each device's |V|-dependent susceptance bound from the voltage at
    # the start of this pass (no solve has run yet within this batched pass) before it moves.
    _refresh_facts_limits!(set.facts, off_facts, data, ts, frozen)
    moved |= _apply_targets_group!(set.facts, off_facts, data, ts, S, frozen, dVdp, osc,
        prev_sign, n_shared, p_prev, did_move)
    moved || return true, true            # nothing wanted to move ⇒ settled, no solve needed
    if _ctrl_solve!(pf, data, ts; kwargs...) && _refresh_sensitivity_context!(ctx, data, ts)
        _refresh_gains_group!(set.taps, 0, data, ts, frozen, dVdp, did_move, ctx)
        _refresh_gains_group!(set.shunts, n_taps, data, ts, frozen, dVdp, did_move, ctx)
        _refresh_gains_group!(set.facts, off_facts, data, ts, frozen, dVdp, did_move, ctx)
        return false, true                # moved + converged ⇒ not settled
    end
    _rollback_targets_group!(set.taps, 0, data, ts, p_prev, did_move)
    _rollback_targets_group!(set.shunts, n_taps, data, ts, p_prev, did_move)
    _rollback_targets_group!(set.facts, off_facts, data, ts, p_prev, did_move)
    _restore_state!(data, ts, snap)
    copyto!(osc, osc0)
    copyto!(prev_sign, prev0)
    copyto!(frozen, frozen0)
    return false, false          # joint solve or sensitivity refresh failed ⇒ run sequential path
end

function _count_controlled_buses!(counts::Dict{Int, Int}, devices)
    for d in devices
        cix = controlled_bus_ix(d)
        counts[cix] = get(counts, cix, 0) + 1
    end
    return
end

function _fill_shared!(
    n_shared::Vector{Int}, offset::Int, counts::Dict{Int, Int}, devices,
)
    for (i, d) in enumerate(devices)
        n_shared[offset + i] = counts[controlled_bus_ix(d)]
    end
    return
end

# Sequential update of the three voltage-device groups; returns whether all settled.
function _step_voltage_groups!(
    set::ControlledDeviceSet, n_taps::Int, n_shunts::Int, data, ts::Int, S::Float64, pf,
    scratch_snap, frozen::Vector{Bool}, dVdp::Vector{Float64}, osc::Vector{Int},
    prev_sign::Vector{Int}, n_shared::Vector{Int}; kwargs...,
)
    settled = _step_device_group!(set.taps, 0, data, ts, S, pf, scratch_snap, frozen, dVdp,
        osc, prev_sign, n_shared; kwargs...)
    settled &= _step_device_group!(set.shunts, n_taps, data, ts, S, pf, scratch_snap,
        frozen, dVdp, osc, prev_sign, n_shared; kwargs...)
    # FACTS branch: refresh each device's |V|-dependent susceptance bound from the CURRENT
    # measured voltage (post taps/shunts moves earlier in this same pass) before it is stepped.
    _refresh_facts_limits!(set.facts, n_taps + n_shunts, data, ts, frozen)
    settled &= _step_device_group!(set.facts, n_taps + n_shunts, data, ts, S, pf,
        scratch_snap, frozen, dVdp, osc, prev_sign, n_shared; kwargs...)
    return settled
end

# One continuation pass. Voltage devices go through the batched (one-solve) path when it is
# enabled and its joint solve converges, else the sequential path (which fully preserves the
# backtracking/freeze behavior). Returns whether the whole pass settled. `ctx` is the
# persisted sensitivity context (non-`nothing` iff `use_batched`); passed through unchanged
# so `_batched_pass!` can refresh it in place.
function _control_pass!(
    set::ControlledDeviceSet, n_taps::Int, n_shunts::Int, use_batched::Bool,
    data, ts::Int, S::Float64, pf, scratch_snap, frozen::Vector{Bool},
    dVdp::Vector{Float64},
    osc::Vector{Int}, prev_sign::Vector{Int}, n_shared::Vector{Int},
    p_prev::Vector{Float64}, did_move::Vector{Bool},
    ctx::Union{Nothing, _SensitivityContext}; kwargs...,
)
    if use_batched
        s, converged =
            _batched_pass!(set, n_taps, n_shunts, data, ts, S, pf, scratch_snap, frozen,
                dVdp, osc, prev_sign, n_shared, p_prev, did_move, ctx; kwargs...)
        converged && return s
    end
    return _step_voltage_groups!(set, n_taps, n_shunts, data, ts, S, pf, scratch_snap,
        frozen, dVdp, osc, prev_sign, n_shared; kwargs...)
end

function _control_continuation!(
    pf,
    data,
    ts::Int;
    kwargs...,
)::Bool
    set = data.controlled_devices::ControlledDeviceSet
    set.inner_solves[] = 0
    set.symbolic_factors[] = 0
    set.numeric_refactors[] = 0
    converged = _ctrl_solve!(pf, data, ts; kwargs...)
    converged || return false

    n_taps = length(set.taps)
    n_shunts = length(set.shunts)
    n_facts = length(set.facts)
    n_dev = n_taps + n_shunts + n_facts
    # Per-device state, indexed in parallel with [taps; shunts; facts]. dVdp: sign
    # sets the negative-feedback orientation, magnitude drives ω; frozen devices are held, not stepped.
    dVdp = zeros(n_dev)
    frozen = fill(false, n_dev)
    osc = zeros(Int, n_dev)
    prev_sign = zeros(Int, n_dev)
    # Voltage devices sharing a controlled bus split the correction (ω / n_shared).
    n_shared = ones(Int, n_dev)
    counts = Dict{Int, Int}()
    _count_controlled_buses!(counts, set.taps)
    _count_controlled_buses!(counts, set.shunts)
    _count_controlled_buses!(counts, set.facts)
    _fill_shared!(n_shared, 0, counts, set.taps)
    _fill_shared!(n_shared, n_taps, counts, set.shunts)
    _fill_shared!(n_shared, n_taps + n_shunts, counts, set.facts)

    # Single reused snapshot buffer for the probe phase, the pass loop, and the post-loop
    # per-device restore (`_plant_sign`, `_continuation_to!`, `_batched_pass!`, `_restore_one!`):
    # those phases run one after another and are never simultaneously live, so one buffer
    # suffices. `snap_and_restore!`'s own `pre` snapshot is the exception — it stays live WHILE
    # it calls `_restore_one!`, so it gets its own separate buffer, never this one.
    scratch_snap = _snapshot_state(data, ts)

    # Build the linearized-sensitivity context ONCE (one numeric factorization reusing the base NR
    # solve's symbolic factor); all device probes below are then triangular solves against it. `nothing`
    # for non-polar formulations or a singular base ⇒ each probe falls back to the FD solve.
    ctx = _sensitivity_context(pf, data, ts; kwargs...)
    _probe_device_signs!(
        set.taps,
        0,
        dVdp,
        frozen,
        ctx,
        data,
        ts,
        pf,
        scratch_snap;
        kwargs...,
    )
    _probe_device_signs!(
        set.shunts, n_taps, dVdp, frozen, ctx, data, ts, pf, scratch_snap; kwargs...)
    _probe_device_signs!(
        set.facts, n_taps + n_shunts, dVdp, frozen, ctx, data, ts, pf, scratch_snap;
        kwargs...)
    if any(frozen)
        frozen_names = join(
            vcat(
                [set.taps[i].name for i in 1:n_taps if frozen[i]],
                [set.shunts[j].name
                    for j in eachindex(set.shunts) if frozen[n_taps + j]],
                [
                    set.facts[k].name
                    for k in eachindex(set.facts) if frozen[n_taps + n_shunts + k]
                ],
            ),
            ", ",
        )
        @warn "discrete control: $(count(frozen)) device(s) had an unreliable or \
            insensitive plant probe and were frozen at their current parameter \
            (time step $ts): $frozen_names"
    end

    # Per-stage pass budget + per-stage oscillation reset (a ramp legitimately reverses direction
    # once). Intermediate stages solve at CONTROL_STAGE_TOL; full tol only at the final stage and
    # snap/restore, and never looser than a user-supplied tol.
    user_tol = Float64(get(kwargs, :tol, DEFAULT_NR_TOL))
    # Batch the voltage-device passes when the polar linear path is live (a non-`nothing` probe
    # `ctx`): one joint solve per pass with analytic per-pass gain refresh, falling back to the
    # sequential path on a failed joint solve. Non-polar formulations step sequentially.
    use_batched = !isnothing(ctx)
    p_prev = zeros(n_dev)
    did_move = fill(false, n_dev)
    S = INITIAL_CONTROL_STEEPNESS
    regulation_complete = false
    while true
        stage_tol = user_tol
        if S < MAX_CONTROL_STEEPNESS
            stage_tol = max(user_tol, CONTROL_STAGE_TOL)
        end
        settled = false
        for _ in 1:MAX_CONTROL_PASSES_PER_STAGE
            settled = _control_pass!(
                set, n_taps, n_shunts, use_batched, data, ts, S, pf,
                scratch_snap,
                frozen, dVdp, osc, prev_sign, n_shared, p_prev, did_move, ctx;
                kwargs..., tol = stage_tol)
            settled && break
        end
        if S >= MAX_CONTROL_STEEPNESS
            regulation_complete = settled
            break
        end
        S = min(S * CONTROL_STEEPNESS_GROWTH, MAX_CONTROL_STEEPNESS)
        fill!(osc, 0)
        fill!(prev_sign, 0)
    end
    if !regulation_complete
        @warn "discrete control: the final steepness stage did not settle within \
            MAX_CONTROL_PASSES_PER_STAGE=$(MAX_CONTROL_PASSES_PER_STAGE) passes \
            (S=$S); regulation may be loose at time step $ts"
    end
    ok = snap_and_restore!(pf, data, set, ts, scratch_snap; kwargs...)
    # Classify each FACTS endpoint against its final voltage-dependent bound and setpoint
    # (`saturated` reported by `get_controlled_device_results` and written back to PSY). Uses
    # the final converged voltages from `snap_and_restore!`.
    for d in set.facts
        classify_facts_saturation!(d, measured_value(d, data, ts))
    end
    # Reported branch flows are computed from the arc-admittance matrices right after
    # this time step in `solve_power_flow!`; bring the moved branch devices' rows in
    # line with their final parameters so flows match the solved network.
    _sync_arc_admittances!(data, set)
    return ok
end

# Incremental λ-restore of one device from its snapped value toward the
# continuous value (used only if snapping made the system infeasible).  First
# probe is a SMALL step, matching `_continuation_to!`.
function _restore_one!(
    d, data, ts::Int, continuous::Float64, pf, scratch_snap; kwargs...,
)::Bool
    snapped = current_parameter(d)
    if abs(continuous - snapped) < _param_tol(d)
        # No parameter to walk, but the solve can still fail (another device's snap
        # made the network infeasible): protect it like every other solve path so a
        # diverged iterate never leaks into the next device's "last converged" snapshot.
        _capture_state!(scratch_snap, data, ts)
        snap = scratch_snap
        ok = _ctrl_solve!(pf, data, ts; kwargs...)
        ok || _restore_state!(data, ts, snap)
        return ok
    end
    lo, hi = parameter_limits(d)
    _capture_state!(scratch_snap, data, ts)  # last converged state, restored on a failed trial
    snap = scratch_snap
    # Full move first, matching `_continuation_to!`.
    apply_parameter!(d, data, clamp(continuous, lo, hi), ts)
    _ctrl_solve!(pf, data, ts; kwargs...) && return true
    apply_parameter!(d, data, snapped, ts)
    _restore_state!(data, ts, snap)
    _, completed =
        _bisection_walk!(d, data, ts, snapped, continuous, lo, hi, pf, snap; kwargs...)
    completed || return false
    return _ctrl_solve!(pf, data, ts; kwargs...)
end

# Snap a concrete device vector onto its discrete grid (continuous devices clamp),
# returning the pre-snap continuous values index-aligned with `devices` for the restore
# path. (Index alignment, not name keying: device names are only unique per concrete
# type, and a cross-family collision must not cross the stashed values.)
function _snap_device_group!(devices, data, ts::Int)
    cont = Vector{Float64}(undef, length(devices))
    for (i, d) in enumerate(devices)
        cont[i] = d.current
        if abs(d.current - d.initial) < _param_tol(d)
            # Never moved (deadband-held or already regulated at enrollment): hold the
            # original setting. PSS/E keeps an in-band shunt at BINIT and an untouched
            # tap at its parsed ratio even when that value is off the discrete grid.
            continue
        end
        apply_parameter!(d, data, snap_to_discrete(d, d.current), ts)
    end
    return cont
end

# λ-restore a concrete device vector toward its stashed continuous value; returns `true`
# iff all restored to a converged state.
function _restore_device_group!(
    devices, data, ts::Int, cont::Vector{Float64}, pf, scratch_snap; kwargs...,
)::Bool
    ok = true
    for (i, d) in enumerate(devices)
        ok &= _restore_one!(d, data, ts, cont[i], pf, scratch_snap; kwargs...)
    end
    return ok
end

function snap_and_restore!(
    pf,
    data,
    set::ControlledDeviceSet,
    ts::Int,
    scratch_snap;
    kwargs...,
)::Bool
    # Last continuous converged state: the restore path must start from here, not from
    # whatever diverged iterate a failed post-snap solve leaves in `data`. This snapshot
    # stays LIVE across the `_restore_one!` calls below, so it needs its OWN buffer —
    # it must NOT alias `scratch_snap`, which those nested `_restore_one!` calls reuse.
    pre = _snapshot_state(data, ts)
    cont_taps = _snap_device_group!(set.taps, data, ts)
    cont_shunts = _snap_device_group!(set.shunts, data, ts)
    cont_facts = _snap_device_group!(set.facts, data, ts)
    _ctrl_solve!(pf, data, ts; kwargs...) && return true
    _restore_state!(data, ts, pre)
    ok = _restore_device_group!(set.taps, data, ts, cont_taps, pf, scratch_snap; kwargs...)
    ok &= _restore_device_group!(
        set.shunts, data, ts, cont_shunts, pf, scratch_snap; kwargs...)
    ok &= _restore_device_group!(
        set.facts, data, ts, cont_facts, pf, scratch_snap; kwargs...)
    if !ok
        names = join(
            vcat(
                [d.name for d in set.taps],
                [d.name for d in set.shunts],
                [d.name for d in set.facts],
            ),
            ", ",
        )
        @error "discrete control could not restore feasibility after \
            snapping; devices: $names (time step $ts)"
        data.converged[ts] = false
        return false
    end
    return true
end

"""
    get_controlled_device_results(data) -> DataFrames.DataFrame

Solved discrete-control device settings: one row per enrolled device per time step, with
its family, name, time step, control band, enrollment-time (`initial`) and solved
(`final`) parameter for that step. Every family (including taps) reports its own
per-time-step state. For a single-time-step solve (`time_steps == 1`), the solved
settings are also written back to
the `PSY.System` by [`solve_and_store_power_flow!`](@ref) under active controls, and
applied to PSS/E exports by [`update_exporter!`](@ref) — see
[`write_device_settings!`](@ref). For `time_steps > 1`, a PSY component cannot hold a
per-time-step schedule, so this DataFrame is the only place the full per-step results
are available. Returns an empty frame when the data was built without discrete control.
"""
function get_controlled_device_results(data)
    family = String[]
    name = String[]
    time_step = Int[]
    lower = Float64[]
    upper = Float64[]
    initial = Float64[]
    final = Float64[]
    delivered_q_mvar = Union{Float64, Missing}[]
    saturated = Bool[]
    set = get_controlled_devices(data)
    if !isnothing(set)
        for ts in 1:get_time_steps(data)
            for (fam, devices) in (
                ("TapTransformer", set.taps),
                ("SwitchedAdmittance", set.shunts),
                ("FACTSControlDevice", set.facts),
            )
                for (i, d) in enumerate(devices)
                    lo, hi = stored_parameter_limits(set, d, i, ts)
                    push!(family, fam)
                    push!(name, d.name)
                    push!(time_step, ts)
                    push!(lower, lo)
                    push!(upper, hi)
                    push!(initial, d.initial)
                    push!(final, stored_current(set, d, i, ts))
                    push!(delivered_q_mvar, stored_delivered_q_mvar(set, d, i, data, ts))
                    push!(saturated, stored_saturated(set, d, i, ts))
                end
            end
        end
    end
    return DataFrames.DataFrame(
        "family" => family,
        "name" => name,
        "time_step" => time_step,
        "lower_limit" => lower,
        "upper_limit" => upper,
        "initial" => initial,
        "final" => final,
        "delivered_q_mvar" => delivered_q_mvar,
        "saturated" => saturated,
    )
end
