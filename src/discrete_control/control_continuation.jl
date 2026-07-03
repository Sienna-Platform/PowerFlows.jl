# Target slope of the local iteration map near the fixed point (0<m<1 ⇒ monotone,
# non-oscillatory). 0.5 trades settling speed for a 2× margin on the worst-case gain bound.
# It also bounds the relaxation factor itself: ω = (1−θ)/(1+gbound) ≤ 1−θ = 0.5.
const CONTROL_CONTRACTION = 0.5

# Snapshot / capture / restore the per-time-step bus voltage state. A failed solve leaves
# the diverged iterate in `data`, so the backtracking applicators roll back to the last
# converged columns before retrying with a smaller step.
@inline _snapshot_voltage(data, ts::Int) =
    (data.bus_magnitude[:, ts], data.bus_angles[:, ts])
@inline function _capture_voltage!((vmag, vang), data, ts::Int)
    vmag .= view(data.bus_magnitude, :, ts)
    vang .= view(data.bus_angles, :, ts)
    return nothing
end
@inline function _restore_voltage!(data, ts::Int, (vmag, vang))
    data.bus_magnitude[:, ts] .= vmag
    data.bus_angles[:, ts] .= vang
    return nothing
end

# Sign-corrected sigmoid law: orientation comes from the MEASURED dy/dp (not the
# device's primary/secondary wiring) so the closed-loop gain σ'(y)·dy/dp ≤ 0
# (negative feedback) regardless of wiring. `_sigmoid(lo,hi,…)` decreases in y when hi>lo.
# `y` is the regulated quantity — bus voltage for voltage devices, branch flow for the PAR.
@inline function _control_target(d, y::Float64, S::Float64, dydp::Float64)
    lo, hi = parameter_limits(d)
    return if dydp > 0.0
        clamp(_sigmoid(lo, hi, S, y, control_setpoint(d)), lo, hi)
    else
        clamp(_sigmoid(hi, lo, S, y, control_setpoint(d)), lo, hi)
    end
end

# Measure dy/dp (sensitivity of the regulated quantity to the parameter) by a small parameter
# perturbation; restore afterward. `reliable=false` (a probe solve failed) ⇒ orientation unknown,
# so the caller freezes the device rather than stepping it with an unknown sign.
function _plant_sign(d, data, ts::Int, pf; kwargs...)::Tuple{Float64, Bool}
    p0 = current_parameter(d)
    y0 = measured_value(d, data, ts)
    snap = _snapshot_voltage(data, ts)   # rolled back below if the probe can't re-converge
    lo, hi = parameter_limits(d)
    δ = 1e-3 * (hi - lo)
    δ = δ > 0.0 ? δ : 1e-6
    p1 = clamp(p0 + δ, lo, hi)
    p1 == p0 && (p1 = clamp(p0 - δ, lo, hi))
    h = p1 - p0
    apply_parameter!(d, data, p1, ts)
    ok1 = _solve_with_q_limits!(pf, data, ts; kwargs...)
    y1 = measured_value(d, data, ts)
    apply_parameter!(d, data, p0, ts)
    ok2 = _solve_with_q_limits!(pf, data, ts; kwargs...)
    reliable = ok1 && ok2 && h != 0.0
    dVdp = reliable ? (y1 - y0) / h : 0.0
    # Parameter is already back at p0; restore the converged state when the probe failed.
    reliable || _restore_voltage!(data, ts, snap)
    return dVdp, reliable
end

# Incremental robust applicator: walk the parameter from `start = d.current`
# toward `target` so NR stays converged.  The FIRST probe is a SMALL step
# (fraction `MIN_LAMBDA_STEP` of the interval), growing on NR success and
# halving on failure (bisection backtracking).  Returns the parameter actually
# reached and leaves the solver converged there.
function _continuation_to!(d, data, ts::Int, target::Float64, pf; kwargs...)
    start = current_parameter(d)
    abs(target - start) < CONTROL_PARAM_TOL && return start
    done = 0.0                       # fraction of [start,target] applied so far
    step = MIN_LAMBDA_STEP
    reached = start
    snap = _snapshot_voltage(data, ts)   # last converged state, restored on a failed trial
    while done < 1.0
        trial = min(1.0, done + step)
        p = start + trial * (target - start)
        apply_parameter!(d, data, p, ts)
        if _solve_with_q_limits!(pf, data, ts; kwargs...)
            done = trial
            reached = p
            _capture_voltage!(snap, data, ts)
            step = min(step * CONTROL_STEP_GROWTH, MAX_LAMBDA_STEP)
        else
            apply_parameter!(d, data, reached, ts)
            _restore_voltage!(data, ts, snap)
            step /= 2.0
            if step < MIN_LAMBDA_STEP
                # Re-solve from the restored converged state to reset `converged[ts]`.
                _solve_with_q_limits!(pf, data, ts; kwargs...)
                return reached
            end
        end
    end
    return reached
end

# Adaptive under-relaxation. The damped iteration p ← p + ω·(σ(V(p)) − p) has local
# slope m = 1 + ω·(g′−1), g′ = σ′(V)·dV/dp ≤ 0 after sign correction. ω is chosen to
# keep m NON-negative (monotone, 0≤m<1, not merely |m|<1): m ≥ θ ⟹ ω ≤ (1−θ)/(1+|g′|).
# |σ′| ≤ |hi−lo|·S/4 bounds g′, so the bound holds at every S without re-measuring.
@inline function _relaxation(d, S::Float64, dVdp::Float64)
    lo, hi = parameter_limits(d)
    gbound = 0.25 * abs(hi - lo) * S * abs(dVdp)
    # ω ≤ 1−θ = 0.5 for any gbound ≥ 0, so no additional cap is needed.
    return (1.0 - CONTROL_CONTRACTION) / (1.0 + gbound)
end

# One damped, sign-corrected proportional update of a single device.  Returns
# the magnitude of the parameter change actually applied (for the settling
# test).  Returns Inf when the device is oscillation-frozen so the outer loop
# does not count it as settled.  Adaptive under-relaxation keeps the iteration
# a contraction at S.
function _step_device!(
    d,
    idx::Int,
    data,
    ts::Int,
    S::Float64,
    pf,
    frozen::Vector{Bool},
    dVdp::Vector{Float64},
    osc::Vector{Int},
    prev_sign::Vector{Int};
    kwargs...,
)::Float64
    # Frozen (unreliable probe): hold parameter; a 0 change counts as settled.
    frozen[idx] && return 0.0
    n_osc = osc[idx]
    if n_osc > CONTROL_OSCILLATION_LIMIT
        # Frozen: return Inf so the outer loop never counts this device as settled.
        return Inf
    end
    yc = measured_value(d, data, ts)
    p_now = current_parameter(d)
    lo, hi = parameter_limits(d)
    dv = dVdp[idx]
    ω = _relaxation(d, S, dv)
    p_tgt = _control_target(d, yc, S, dv)
    # Track sign reversals to detect oscillation.
    s = Int(sign(p_tgt - p_now))
    ps = prev_sign[idx]
    if ps != 0 && s != 0 && s != ps
        new_n = n_osc + 1
        osc[idx] = new_n
        # Fires once: the n_osc > LIMIT early-return prevents osc[idx] passing LIMIT+1.
        new_n == CONTROL_OSCILLATION_LIMIT + 1 &&
            @warn "discrete control: device $(d.name) is oscillating \
                ($(new_n) sign reversals); freezing at current parameter $(p_now) \
                (time step $ts)"
    end
    prev_sign[idx] = s
    osc[idx] > CONTROL_OSCILLATION_LIMIT && return Inf
    p_new = clamp(p_now + ω * (p_tgt - p_now), lo, hi)
    reached = _continuation_to!(d, data, ts, p_new, pf; kwargs...)
    set_current_parameter!(d, reached)
    return abs(reached - p_now)
end

# Probe each device in one concrete vector for its plant sign (dV/dp) and write the result into
# the shared per-device state at `offset + i`. A function barrier: the body specializes on the
# concrete element type, so there is no dynamic dispatch over a heterogeneous device collection.
# `offset` reproduces the global [taps; shunts; facts] indexing of `dVdp`/`frozen`.
function _probe_device_signs!(
    devices, offset::Int, dVdp::Vector{Float64}, frozen::Vector{Bool},
    data, ts::Int, pf; kwargs...,
)
    for (i, d) in enumerate(devices)
        s, reliable = _plant_sign(d, data, ts, pf; kwargs...)
        reliable ? (dVdp[offset + i] = s) : (frozen[offset + i] = true)
    end
    return nothing
end

# One proportional update over a concrete device vector; returns `true` iff every device in it
# settled (parameter change < CONTROL_PARAM_TOL). Same function-barrier + `offset` indexing.
function _step_device_group!(
    devices, offset::Int, data, ts::Int, S::Float64, pf,
    frozen::Vector{Bool}, dVdp::Vector{Float64}, osc::Vector{Int},
    prev_sign::Vector{Int}; kwargs...,
)::Bool
    settled = true
    for (i, d) in enumerate(devices)
        g = _step_device!(
            d, offset + i, data, ts, S, pf, frozen, dVdp, osc, prev_sign; kwargs...)
        g >= CONTROL_PARAM_TOL && (settled = false)
    end
    return settled
end

function _control_continuation!(
    pf,
    data,
    ts::Int;
    kwargs...,
)::Bool
    set = data.controlled_devices::ControlledDeviceSet
    converged = _solve_with_q_limits!(pf, data, ts; kwargs...)
    converged || return false

    n_taps = length(set.taps)
    n_shunts = length(set.shunts)
    n_facts = length(set.facts)
    n_volt = n_taps + n_shunts + n_facts   # offset of the phase-shifter block
    n_dev = n_volt + length(set.phase_shifters)
    # Per-device state indexed in parallel with [taps; shunts; facts; phase_shifters] (plain
    # vectors, no per-iteration string hashing). dVdp is measured once from the converged base
    # point: its sign sets the negative-feedback orientation, its magnitude drives
    # under-relaxation. Unreliable-probe devices are frozen, not stepped.
    dVdp = zeros(n_dev)
    frozen = fill(false, n_dev)
    osc = zeros(Int, n_dev)
    prev_sign = zeros(Int, n_dev)
    _probe_device_signs!(set.taps, 0, dVdp, frozen, data, ts, pf; kwargs...)
    _probe_device_signs!(set.shunts, n_taps, dVdp, frozen, data, ts, pf; kwargs...)
    _probe_device_signs!(
        set.facts, n_taps + n_shunts, dVdp, frozen, data, ts, pf; kwargs...)
    _probe_device_signs!(
        set.phase_shifters, n_volt, dVdp, frozen, data, ts, pf; kwargs...)
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
                [
                    set.phase_shifters[m].name
                    for m in eachindex(set.phase_shifters) if frozen[n_volt + m]
                ],
            ),
            ", ",
        )
        @warn "discrete control: $(count(frozen)) device(s) had an unreliable \
            plant-sensitivity probe and were frozen at their current parameter \
            (time step $ts): $frozen_names"
    end

    S = INITIAL_CONTROL_STEEPNESS
    regulation_complete = false
    for _ in 1:MAX_CONTROL_OUTER_ITERATIONS
        settled = _step_device_group!(
            set.taps, 0, data, ts, S, pf, frozen, dVdp, osc, prev_sign; kwargs...)
        settled &= _step_device_group!(
            set.shunts, n_taps, data, ts, S, pf, frozen, dVdp, osc, prev_sign; kwargs...,
        )
        settled &= _step_device_group!(
            set.facts, n_taps + n_shunts, data, ts, S, pf, frozen, dVdp, osc, prev_sign;
            kwargs...)
        settled &= _step_device_group!(
            set.phase_shifters, n_volt, data, ts, S, pf, frozen, dVdp, osc, prev_sign;
            kwargs...)
        if settled
            if S >= MAX_CONTROL_STEEPNESS
                regulation_complete = true
                break
            end
            S = min(S * CONTROL_STEEPNESS_GROWTH, MAX_CONTROL_STEEPNESS)
        end
    end
    if !regulation_complete
        @warn "discrete control: reached MAX_CONTROL_OUTER_ITERATIONS without \
            full-steepness convergence (S=$S, target=$MAX_CONTROL_STEEPNESS); \
            regulation may be loose at time step $ts"
    end
    return snap_and_restore!(pf, data, set, ts; kwargs...)
end

# Incremental λ-restore of one device from its snapped value toward the
# continuous value (used only if snapping made the system infeasible).  First
# probe is a SMALL step, matching `_continuation_to!`.
function _restore_one!(d, data, ts::Int, continuous::Float64, pf; kwargs...)::Bool
    snapped = current_parameter(d)
    abs(continuous - snapped) < CONTROL_PARAM_TOL &&
        return _solve_with_q_limits!(pf, data, ts; kwargs...)
    done = 0.0
    step = MIN_LAMBDA_STEP
    last_good = snapped
    snap = _snapshot_voltage(data, ts)   # last converged state, restored on a failed trial
    while done < 1.0
        trial = min(1.0, done + step)
        p = snapped + trial * (continuous - snapped)
        apply_parameter!(d, data, p, ts)
        if _solve_with_q_limits!(pf, data, ts; kwargs...)
            done = trial
            last_good = p
            _capture_voltage!(snap, data, ts)
            step = min(step * CONTROL_STEP_GROWTH, MAX_LAMBDA_STEP)
        else
            # Revert to the last converged parameter and state, not the failed trial.
            apply_parameter!(d, data, last_good, ts)
            _restore_voltage!(data, ts, snap)
            step /= 2.0
            if step < MIN_LAMBDA_STEP
                return false
            end
        end
    end
    return _solve_with_q_limits!(pf, data, ts; kwargs...)
end

# Snap a concrete device vector onto its discrete grid (continuous devices clamp), stashing the
# pre-snap continuous value in `cont` for the restore path. Function barrier over the eltype.
function _snap_device_group!(devices, data, ts::Int, cont::Dict{String, Float64})
    for d in devices
        cont[d.name] = d.current
        apply_parameter!(d, data, snap_to_discrete(d, d.current), ts)
    end
    return nothing
end

# λ-restore a concrete device vector toward its stashed continuous value; returns `true` iff all
# restored to a converged state.
function _restore_device_group!(
    devices, data, ts::Int, cont::Dict{String, Float64}, pf; kwargs...,
)::Bool
    ok = true
    for d in devices
        ok &= _restore_one!(d, data, ts, cont[d.name], pf; kwargs...)
    end
    return ok
end

function snap_and_restore!(
    pf,
    data,
    set::ControlledDeviceSet,
    ts::Int;
    kwargs...,
)::Bool
    cont = Dict{String, Float64}()
    _snap_device_group!(set.taps, data, ts, cont)
    _snap_device_group!(set.shunts, data, ts, cont)
    _snap_device_group!(set.facts, data, ts, cont)
    _snap_device_group!(set.phase_shifters, data, ts, cont)
    _solve_with_q_limits!(pf, data, ts; kwargs...) && return true
    ok = _restore_device_group!(set.taps, data, ts, cont, pf; kwargs...)
    ok &= _restore_device_group!(set.shunts, data, ts, cont, pf; kwargs...)
    ok &= _restore_device_group!(set.facts, data, ts, cont, pf; kwargs...)
    ok &= _restore_device_group!(set.phase_shifters, data, ts, cont, pf; kwargs...)
    if !ok
        names = join(
            vcat(
                [d.name for d in set.taps],
                [d.name for d in set.shunts],
                [d.name for d in set.facts],
                [d.name for d in set.phase_shifters],
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
