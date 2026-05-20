# Upper bound on the under-relaxation factor ω∈(0,1].  The effective ω is
# adapted down from this cap so the damped fixed-point iteration stays a
# contraction at the current steepness S (see `_relaxation`); the fixed point
# itself (where p == sigmoid(V(p))) is independent of ω.
const CONTROL_RELAXATION_MAX = 0.8
# Target one-step contraction ratio for the damped iteration.  ω is chosen so
# the local iteration-map slope magnitude is ≤ this; < 1 ⇒ monotone, stable.
const CONTROL_CONTRACTION = 0.7

@inline _read_vmag(data, ix::Int, ts::Int) = data.bus_magnitude[ix, ts]

# Sign-corrected sigmoid control law.  `target_from_voltage` encodes a fixed
# orientation via `controlled_on_primary`.  The engine measures the plant
# sensitivity dV/dp once per device (see `_plant_sign`) and flips the sigmoid
# orientation when the device's nominal orientation would give *positive*
# feedback, so the closed loop is always negative feedback regardless of the
# device's primary/secondary wiring.
@inline function _control_target(d, vmag::Float64, S::Float64, flip::Bool)
    flip || return target_from_voltage(d, vmag, S)
    lo, hi = parameter_limits(d)
    return clamp(_sigmoid(hi, lo, S, vmag, voltage_setpoint(d)), lo, hi)
end

# Measure the plant sensitivity dV/dp at the controlled bus by a small
# perturbation about the current parameter; restore afterwards.  Returns
# (dVdp, flip) where `flip=true` means the nominal sigmoid orientation is
# positive feedback for this plant and must be reversed.
function _plant_sign(d, data, ts::Int, pf; kwargs...)::Tuple{Float64, Bool}
    p0 = current_parameter(d)
    cix = controlled_bus_ix(d)
    V0 = _read_vmag(data, cix, ts)
    lo, hi = parameter_limits(d)
    δ = 1e-3 * (hi - lo)
    δ = δ > 0.0 ? δ : 1e-6
    p1 = clamp(p0 + δ, lo, hi)
    p1 == p0 && (p1 = clamp(p0 - δ, lo, hi))
    h = p1 - p0
    apply_parameter!(d, data, p1, ts)
    ok1 = _solve_with_q_limits!(pf, data, ts; kwargs...)
    V1 = _read_vmag(data, cix, ts)
    apply_parameter!(d, data, p0, ts)
    ok2 = _solve_with_q_limits!(pf, data, ts; kwargs...)
    # If either solve failed: V1 or the restored base point is unreliable.
    # dVdp=0 makes _relaxation return CONTROL_RELAXATION_MAX (safe conservative cap).
    dVdp = (ok1 && ok2 && h != 0.0) ? (V1 - V0) / h : 0.0
    # Sign analysis for negative-feedback orientation:
    #   _sigmoid(lo, hi, S, x, xset) is DECREASING in x when hi > lo, INCREASING when hi < lo.
    #   controlled_on_primary=true  (eq.46): target_from_voltage uses (lo=p_min, hi=p_max),
    #     so hi > lo → nominal d(p_target)/dV < 0.  Negative feedback needs dVdp > 0
    #     (product < 0). flip=true swaps lo/hi → increasing law → product still < 0.
    #   controlled_on_primary=false (eq.47): target_from_voltage uses (lo=p_max, hi=p_min),
    #     so hi < lo → nominal d(p_target)/dV > 0.  Negative feedback needs dVdp < 0.
    #   In both cases: `flip = (dVdp > 0)` flips exactly when the nominal orientation
    #   would give positive feedback, ensuring the closed-loop gain is always negative.
    flip = dVdp > 0.0
    return dVdp, flip
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
    while done < 1.0
        trial = min(1.0, done + step)
        p = start + trial * (target - start)
        apply_parameter!(d, data, p, ts)
        if _solve_with_q_limits!(pf, data, ts; kwargs...)
            done = trial
            reached = p
            step = min(step * CONTROL_STEP_GROWTH, MAX_LAMBDA_STEP)
        else
            apply_parameter!(d, data, reached, ts)
            step /= 2.0
            if step < MIN_LAMBDA_STEP
                _solve_with_q_limits!(pf, data, ts; kwargs...)
                return reached
            end
        end
    end
    return reached
end

# Adaptive under-relaxation.  The damped fixed-point iteration
# p ← p + ω·(σ(V(p)) − p) has local map slope m = 1 + ω·(g′ − 1), where
# g′ = σ′(V)·dV/dp is the closed-loop gain.  After sign correction g′ ≤ 0, so
# m decreases in ω and the binding bound is m ≥ −θ ⟹ ω ≤ (1+θ)/(1+|g′|).
# |σ′(V)| ≤ |hi−lo|·S/4 (sigmoid slope, max at V=Vset), giving a guaranteed
# contraction at every steepness without per-iteration plant re-measurement.
@inline function _relaxation(d, S::Float64, dVdp::Float64)
    lo, hi = parameter_limits(d)
    gbound = 0.25 * abs(hi - lo) * S * abs(dVdp)
    ω = (1.0 + CONTROL_CONTRACTION) / (1.0 + gbound)
    return min(CONTROL_RELAXATION_MAX, ω)
end

# One damped, sign-corrected proportional update of a single device.  Returns
# the magnitude of the parameter change actually applied (for the settling
# test).  Returns Inf when the device is oscillation-frozen so the outer loop
# does not count it as settled.  Adaptive under-relaxation keeps the iteration
# a contraction at S.
function _step_device!(
    d,
    data,
    ts::Int,
    S::Float64,
    pf,
    flip::Dict{String, Bool},
    dVdp::Dict{String, Float64},
    osc::Dict{String, Int},
    prev_sign::Dict{String, Int};
    kwargs...,
)::Float64
    n_osc = get(osc, d.name, 0)
    if n_osc > CONTROL_OSCILLATION_LIMIT
        # Frozen: return Inf so the outer loop never counts this device as settled.
        return Inf
    end
    Vc = _read_vmag(data, controlled_bus_ix(d), ts)
    p_now = current_parameter(d)
    lo, hi = parameter_limits(d)
    ω = _relaxation(d, S, get(dVdp, d.name, 0.0))
    p_tgt = _control_target(d, Vc, S, get(flip, d.name, false))
    # Track sign reversals to detect oscillation.
    s = Int(sign(p_tgt - p_now))
    ps = get(prev_sign, d.name, 0)
    if ps != 0 && s != 0 && s != ps
        new_n = n_osc + 1
        osc[d.name] = new_n
        # Warn exactly once when the limit is first exceeded.
        # Fires exactly once: the n_osc > LIMIT early-return above prevents
        # osc[d.name] from ever advancing past LIMIT+1.
        new_n == CONTROL_OSCILLATION_LIMIT + 1 &&
            @warn "discrete control: device $(d.name) is oscillating \
                ($(new_n) sign reversals); freezing at current parameter $(p_now) \
                (time step $ts)"
    end
    prev_sign[d.name] = s
    # Re-check after potential increment.
    get(osc, d.name, 0) > CONTROL_OSCILLATION_LIMIT && return Inf
    p_new = clamp(p_now + ω * (p_tgt - p_now), lo, hi)
    reached = _continuation_to!(d, data, ts, p_new, pf; kwargs...)
    set_current_parameter!(d, reached)
    return abs(reached - p_now)
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

    # Measure each device's plant sensitivity dV/dp once, from the converged
    # base point.  `flip` makes the sigmoid law negative feedback regardless of
    # primary/secondary wiring; `dVdp` drives the adaptive under-relaxation.
    flip = Dict{String, Bool}()
    dVdp = Dict{String, Float64}()
    for d in set.taps
        s, fl = _plant_sign(d, data, ts, pf; kwargs...)
        flip[d.name] = fl
        dVdp[d.name] = s
    end
    for d in set.shunts
        s, fl = _plant_sign(d, data, ts, pf; kwargs...)
        flip[d.name] = fl
        dVdp[d.name] = s
    end

    S = INITIAL_CONTROL_STEEPNESS
    osc = Dict{String, Int}()
    prev_sign = Dict{String, Int}()
    regulation_complete = false
    for _ in 1:MAX_CONTROL_OUTER_ITERATIONS
        settled = true
        for d in set.taps
            g = _step_device!(d, data, ts, S, pf, flip, dVdp, osc, prev_sign; kwargs...)
            g >= CONTROL_PARAM_TOL && (settled = false)
        end
        for d in set.shunts
            g = _step_device!(d, data, ts, S, pf, flip, dVdp, osc, prev_sign; kwargs...)
            g >= CONTROL_PARAM_TOL && (settled = false)
        end
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
    while done < 1.0
        trial = min(1.0, done + step)
        p = snapped + trial * (continuous - snapped)
        apply_parameter!(d, data, p, ts)
        if _solve_with_q_limits!(pf, data, ts; kwargs...)
            done = trial
            last_good = p
            step = min(step * CONTROL_STEP_GROWTH, MAX_LAMBDA_STEP)
        else
            step /= 2.0
            if step < MIN_LAMBDA_STEP
                # Leave data at the last converged parameter, not the failed trial.
                apply_parameter!(d, data, last_good, ts)
                return false
            end
        end
    end
    return _solve_with_q_limits!(pf, data, ts; kwargs...)
end

function snap_and_restore!(
    pf,
    data,
    set::ControlledDeviceSet,
    ts::Int;
    kwargs...,
)::Bool
    cont = Dict{String, Float64}()
    for d in set.taps
        cont[d.name] = d.current
        apply_parameter!(d, data, snap_to_discrete(d, d.current), ts)
    end
    for d in set.shunts
        cont[d.name] = d.current
        apply_parameter!(d, data, snap_to_discrete(d, d.current), ts)
    end
    _solve_with_q_limits!(pf, data, ts; kwargs...) && return true
    ok = true
    for d in set.taps
        ok &= _restore_one!(d, data, ts, cont[d.name], pf; kwargs...)
    end
    for d in set.shunts
        ok &= _restore_one!(d, data, ts, cont[d.name], pf; kwargs...)
    end
    if !ok
        names = join(
            vcat([d.name for d in set.taps], [d.name for d in set.shunts]),
            ", ",
        )
        @error "discrete control could not restore feasibility after \
            snapping; devices: $names (time step $ts)"
        data.converged[ts] = false
        return false
    end
    return true
end
