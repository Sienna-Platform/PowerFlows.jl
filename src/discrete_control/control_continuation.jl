# Target slope of the local iteration map near the fixed point (0<m<1 ⇒ monotone,
# non-oscillatory). 0.5 trades settling speed for a 2× margin on the worst-case gain bound.
# It also bounds the relaxation factor itself: ω = (1−θ)/(1+gbound) ≤ 1−θ = 0.5.
const CONTROL_CONTRACTION = 0.5

# State a rolled-back trial must restore. `bus_type` is here because the inner solve's Q-limit
# logic makes one-way PV→PQ flips. An absent VSC or LCC subsystem is carried as empty vectors,
# which is what the capture/restore guards test.
struct ControlStateSnapshot
    vmag::Vector{Float64}
    vang::Vector{Float64}
    btype::Vector{PSY.ACBusTypes}
    pinj::Vector{Float64}
    qinj::Vector{Float64}
    dc_p::Vector{Float64}
    dc_q::Vector{Float64}
    dc_v::Vector{Float64}
    lcc_rectifier_tap::Vector{Float64}
    lcc_inverter_tap::Vector{Float64}
    lcc_rectifier_angle::Vector{Float64}
    lcc_inverter_angle::Vector{Float64}
    lcc_i_dc::Vector{Float64}
end

@inline function _dc_state_cols(data, ts::Int)
    dcn = get_dc_network(data)
    if has_dc_network(dcn)
        return dcn.p_c[:, ts], dcn.q_c[:, ts], dcn.node_vdc[:, ts]
    end
    return Float64[], Float64[], Float64[]
end
@inline function _lcc_state_cols(data, ts::Int)
    if get_lcc_count(data) > 0
        lcc = data.lcc
        return lcc.rectifier.tap[:, ts], lcc.inverter.tap[:, ts],
        lcc.rectifier.thyristor_angle[:, ts], lcc.inverter.thyristor_angle[:, ts],
        lcc.i_dc[:, ts]
    end
    return Float64[], Float64[], Float64[], Float64[], Float64[]
end
@inline function _snapshot_state(data, ts::Int)
    dc_p, dc_q, dc_v = _dc_state_cols(data, ts)
    lcc_rt, lcc_it, lcc_ra, lcc_ia, lcc_idc = _lcc_state_cols(data, ts)
    return ControlStateSnapshot(
        data.bus_magnitude[:, ts],
        data.bus_angles[:, ts],
        data.bus_type[:, ts],
        data.bus_active_power_injections[:, ts],
        data.bus_reactive_power_injections[:, ts],
        dc_p, dc_q, dc_v,
        lcc_rt, lcc_it, lcc_ra, lcc_ia, lcc_idc,
    )
end
@inline function _capture_state!(snap::ControlStateSnapshot, data, ts::Int)
    snap.vmag .= view(data.bus_magnitude, :, ts)
    snap.vang .= view(data.bus_angles, :, ts)
    snap.btype .= view(data.bus_type, :, ts)
    snap.pinj .= view(data.bus_active_power_injections, :, ts)
    snap.qinj .= view(data.bus_reactive_power_injections, :, ts)
    if !isempty(snap.dc_v)
        dcn = get_dc_network(data)
        snap.dc_p .= view(dcn.p_c, :, ts)
        snap.dc_q .= view(dcn.q_c, :, ts)
        snap.dc_v .= view(dcn.node_vdc, :, ts)
    end
    if !isempty(snap.lcc_i_dc)
        lcc = data.lcc
        snap.lcc_rectifier_tap .= view(lcc.rectifier.tap, :, ts)
        snap.lcc_inverter_tap .= view(lcc.inverter.tap, :, ts)
        snap.lcc_rectifier_angle .= view(lcc.rectifier.thyristor_angle, :, ts)
        snap.lcc_inverter_angle .= view(lcc.inverter.thyristor_angle, :, ts)
        snap.lcc_i_dc .= view(lcc.i_dc, :, ts)
    end
    return
end
@inline function _restore_state!(data, ts::Int, snap::ControlStateSnapshot)
    data.bus_magnitude[:, ts] .= snap.vmag
    data.bus_angles[:, ts] .= snap.vang
    data.bus_type[:, ts] .= snap.btype
    data.bus_active_power_injections[:, ts] .= snap.pinj
    data.bus_reactive_power_injections[:, ts] .= snap.qinj
    if !isempty(snap.dc_v)
        dcn = get_dc_network(data)
        dcn.p_c[:, ts] .= snap.dc_p
        dcn.q_c[:, ts] .= snap.dc_q
        dcn.node_vdc[:, ts] .= snap.dc_v
    end
    if !isempty(snap.lcc_i_dc)
        lcc = data.lcc
        lcc.rectifier.tap[:, ts] .= snap.lcc_rectifier_tap
        lcc.inverter.tap[:, ts] .= snap.lcc_inverter_tap
        lcc.rectifier.thyristor_angle[:, ts] .= snap.lcc_rectifier_angle
        lcc.inverter.thyristor_angle[:, ts] .= snap.lcc_inverter_angle
        lcc.i_dc[:, ts] .= snap.lcc_i_dc
        # Re-derive phi/branch_admittances at the restored state.
        _update_ybus_lcc!(data, ts)
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
function _plant_sign(
    d, data, ts::Int, pf, snap::ControlStateSnapshot; kwargs...,
)::Tuple{Float64, Bool}
    p0 = current_parameter(d)
    y0 = measured_value(d, data, ts)
    _capture_state!(snap, data, ts)
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
# One device's plant sign: the linear sensitivity when a polar `ctx` is live and the family has an
# analytic form (tap/shunt/FACTS), else the FD probe (non-polar formulations, singular base).
_probe_one_sign(
    d, data, ts::Int, pf, scratch_snap::ControlStateSnapshot, ::Nothing; kwargs...,
) = _plant_sign(d, data, ts, pf, scratch_snap; kwargs...)
function _probe_one_sign(
    d, data, ts::Int, pf, scratch_snap::ControlStateSnapshot,
    ctx::_SensitivityContext; kwargs...,
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
    ctx::Union{Nothing, _SensitivityContext}, data, ts::Int, pf,
    scratch_snap::ControlStateSnapshot; kwargs...,
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
    devices, offset::Int, data, ts::Int, S::Float64, pf,
    scratch_snap::ControlStateSnapshot,
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
    snap::ControlStateSnapshot, frozen::Vector{Bool}, dVdp::Vector{Float64},
    osc::Vector{Int}, prev_sign::Vector{Int}, n_shared::Vector{Int},
    p_prev::Vector{Float64}, did_move::Vector{Bool},
    ctx::_SensitivityContext; kwargs...,
)
    fill!(did_move, false)
    _capture_state!(snap, data, ts)
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
    scratch_snap::ControlStateSnapshot, frozen::Vector{Bool}, dVdp::Vector{Float64},
    osc::Vector{Int}, prev_sign::Vector{Int}, n_shared::Vector{Int}; kwargs...,
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
    data, ts::Int, S::Float64, pf, scratch_snap::ControlStateSnapshot,
    frozen::Vector{Bool}, dVdp::Vector{Float64},
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

    # One buffer shared by the probe phase, the pass loop, and the post-loop per-device
    # restore: those phases run in sequence and are never simultaneously live. The exception
    # is `snap_and_restore!`'s `pre`, which allocates its own.
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
    # Batch the voltage-device passes when the analytic path is live AND its context can be kept
    # current across passes: one joint solve per pass with analytic per-pass gain refresh,
    # falling back to the sequential path on a failed joint solve. Batching is gated on
    # `_refreshable` rather than on the presence of a `ctx`, so a formulation can supply
    # analytic PROBE sensitivities before it supplies a per-pass refresh — batching without a
    # working refresh would read a stale Jacobian every pass after the first move.
    use_batched = _supports_batched_refresh(ctx)
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
    d, data, ts::Int, continuous::Float64, pf, snap::ControlStateSnapshot; kwargs...,
)::Bool
    snapped = current_parameter(d)
    if abs(continuous - snapped) < _param_tol(d)
        # No parameter to walk, but the solve can still fail (another device's snap
        # made the network infeasible): protect it like every other solve path so a
        # diverged iterate never leaks into the next device's "last converged" snapshot.
        _capture_state!(snap, data, ts)
        ok = _ctrl_solve!(pf, data, ts; kwargs...)
        ok || _restore_state!(data, ts, snap)
        return ok
    end
    lo, hi = parameter_limits(d)
    _capture_state!(snap, data, ts)  # last converged state, restored on a failed trial
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
    devices, data, ts::Int, cont::Vector{Float64}, pf,
    scratch_snap::ControlStateSnapshot; kwargs...,
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
    scratch_snap::ControlStateSnapshot;
    kwargs...,
)::Bool
    # Last continuous converged state: the restore path must start from here, not from
    # whatever diverged iterate a failed post-snap solve leaves in `data`. Needs its OWN
    # buffer because it stays live across the `_restore_one!` calls below, which reuse
    # `scratch_snap`.
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

"""Reporting family of a controlled device. Taps report the PSY object that actually carries
the control — the `TransformerCircuit` — not the owning transformer, so a 2W and a 3W winding
land in the same family."""
control_family(::ControlledTap) = "TransformerCircuit"
control_family(::ControlledSwitchedShunt) = "SwitchedAdmittance"
control_family(::ControlledFACTS) = "FACTSControlDevice"

# Owning-component address, reported so a tap row resolves to its circuit for either arity
# (`PSY.get_circuits(device_name)[circuit_index]`). Families with no sub-component address
# report `missing` rather than a fabricated index.
_report_device_name(d::ControlledTap) = d.device_name
_report_device_name(::ControlledSwitchedShunt) = missing
_report_device_name(::ControlledFACTS) = missing

_report_circuit_index(d::ControlledTap) = d.circuit_index
_report_circuit_index(::ControlledSwitchedShunt) = missing
_report_circuit_index(::ControlledFACTS) = missing

"""
    get_controlled_device_results(data) -> DataFrames.DataFrame

Solved discrete-control device settings: one row per enrolled device per time step, with
its family, name, time step, control band, enrollment-time (`initial`) and solved
(`final`) parameter for that step. Every family reports its own per-time-step state.

Tap rows are reported at the `PSY.TransformerCircuit` level and are identical in shape for
both transformer arities: `family` is `"TransformerCircuit"`, and `device_name` +
`circuit_index` address the owning circuit as `PSY.get_circuits(device_name)[circuit_index]`
(`circuit_index == 1` for a two-winding transformer). `name` is the control's own name,
which for a three-winding winding is suffixed per circuit. Non-transformer families leave
`device_name`/`circuit_index` `missing`.

For a single-time-step solve (`time_steps == 1`), the solved settings are also written back
to the `PSY.System` by [`solve_and_store_power_flow!`](@ref) under active controls, and
applied to PSS/E exports by [`update_exporter!`](@ref) — see
[`write_device_settings!`](@ref). For `time_steps > 1`, a PSY component cannot hold a
per-time-step schedule, so this DataFrame is the only place the full per-step results
are available. Returns an empty frame when the data was built without discrete control.
"""
function get_controlled_device_results(data)
    family = String[]
    name = String[]
    device_name = Union{String, Missing}[]
    circuit_index = Union{Int, Missing}[]
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
            for devices in (set.taps, set.shunts, set.facts)
                for (i, d) in enumerate(devices)
                    lo, hi = stored_parameter_limits(set, d, i, ts)
                    push!(family, control_family(d))
                    push!(name, d.name)
                    push!(device_name, _report_device_name(d))
                    push!(circuit_index, _report_circuit_index(d))
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
        "device_name" => device_name,
        "circuit_index" => circuit_index,
        "time_step" => time_step,
        "lower_limit" => lower,
        "upper_limit" => upper,
        "initial" => initial,
        "final" => final,
        "delivered_q_mvar" => delivered_q_mvar,
        "saturated" => saturated,
    )
end
