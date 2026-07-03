# Target slope of the local iteration map near the fixed point (0<m<1 ⇒ monotone,
# non-oscillatory). 0.5 trades settling speed for a 2× margin on the worst-case gain bound.
# It also bounds the relaxation factor itself: ω = (1−θ)/(1+gbound) ≤ 1−θ = 0.5.
const CONTROL_CONTRACTION = 0.5

# Snapshot / capture / restore the per-time-step solved state. A failed solve leaves the
# diverged iterate in `data`, so the backtracking applicators roll back to the last
# converged columns before retrying with a smaller step. Beyond voltages, the snapshot
# covers the state the Q-limit loop mutates: `bus_type` (one-way PV→PQ flips) and the
# bus injection columns (Q clamps at flipped buses, distributed-slack updates baked into
# the next residual's setpoints) — a rolled-back TRIAL must not permanently change the
# problem definition.
@inline _snapshot_state(data, ts::Int) = (
    data.bus_magnitude[:, ts],
    data.bus_angles[:, ts],
    data.bus_type[:, ts],
    data.bus_active_power_injections[:, ts],
    data.bus_reactive_power_injections[:, ts],
)
@inline function _capture_state!((vmag, vang, btype, pinj, qinj), data, ts::Int)
    vmag .= view(data.bus_magnitude, :, ts)
    vang .= view(data.bus_angles, :, ts)
    btype .= view(data.bus_type, :, ts)
    pinj .= view(data.bus_active_power_injections, :, ts)
    qinj .= view(data.bus_reactive_power_injections, :, ts)
    return nothing
end
@inline function _restore_state!(data, ts::Int, (vmag, vang, btype, pinj, qinj))
    data.bus_magnitude[:, ts] .= vmag
    data.bus_angles[:, ts] .= vang
    data.bus_type[:, ts] .= btype
    data.bus_active_power_injections[:, ts] .= pinj
    data.bus_reactive_power_injections[:, ts] .= qinj
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
function _plant_sign(d, data, ts::Int, pf; kwargs...)::Tuple{Float64, Bool}
    p0 = current_parameter(d)
    y0 = measured_value(d, data, ts)
    snap = _snapshot_state(data, ts)
    lo, hi = parameter_limits(d)
    δ = 1e-3 * (hi - lo)
    δ = δ > 0.0 ? δ : 1e-6
    p1 = clamp(p0 + δ, lo, hi)
    p1 == p0 && (p1 = clamp(p0 - δ, lo, hi))
    h = p1 - p0
    apply_parameter!(d, data, p1, ts)
    ok = _solve_with_q_limits!(pf, data, ts; kwargs...)
    y1 = measured_value(d, data, ts)
    apply_parameter!(d, data, p0, ts)
    _restore_state!(data, ts, snap)
    reliable = ok && h != 0.0
    dVdp = reliable ? (y1 - y0) / h : 0.0
    return dVdp, reliable
end

# Incremental robust applicator: walk the parameter from `start = d.current`
# toward `target` so NR stays converged.  The FIRST probe is a SMALL step
# (fraction `MIN_LAMBDA_STEP` of the interval), growing on NR success and
# halving on failure (bisection backtracking).  Returns `(reached, moved)`:
# the parameter actually reached (solver left converged there) and whether ANY
# sub-step was applied — a requested move that could not budge at all must not
# masquerade as a settled device.
function _continuation_to!(d, data, ts::Int, target::Float64, pf; kwargs...)
    start = current_parameter(d)
    abs(target - start) < _param_tol(d) && return start, true
    done = 0.0                       # fraction of [start,target] applied so far
    step = MIN_LAMBDA_STEP
    reached = start
    snap = _snapshot_state(data, ts)   # last converged state, restored on a failed trial
    while done < 1.0
        trial = min(1.0, done + step)
        p = start + trial * (target - start)
        apply_parameter!(d, data, p, ts)
        if _solve_with_q_limits!(pf, data, ts; kwargs...)
            done = trial
            reached = p
            _capture_state!(snap, data, ts)
            step = min(step * CONTROL_STEP_GROWTH, MAX_LAMBDA_STEP)
        else
            apply_parameter!(d, data, reached, ts)
            _restore_state!(data, ts, snap)
            step /= 2.0
            if step < MIN_LAMBDA_STEP
                if reached != start
                    # Re-solve from the restored converged state (partial move applied).
                    _solve_with_q_limits!(pf, data, ts; kwargs...)
                end
                return reached, reached != start
            end
        end
    end
    return reached, true
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
    return nothing
end

# One damped, sign-corrected proportional update of a single device. Returns the
# magnitude of the parameter change actually applied (for the settling test); frozen
# and in-deadband devices return 0.0 (settled). The measured plant gain is refreshed
# by a secant update from the numbers the step just produced (zero extra solves); a
# detected sign reversal (OLTC reverse action) or a collapse below the gain floor
# freezes the device instead of silently running with positive feedback.
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
    prev_sign::Vector{Int},
    n_shared::Vector{Int};
    kwargs...,
)::Float64
    frozen[idx] && return 0.0
    yc = measured_value(d, data, ts)
    if _in_deadband(d, yc)
        # PSS/E deadband semantics: a device whose regulated quantity is inside its
        # band is held, not driven to the band midpoint.
        prev_sign[idx] = 0
        return 0.0
    end
    p_now = current_parameter(d)
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
    s = abs(p_tgt - p_now) < tol_d ? 0 : Int(sign(p_tgt - p_now))
    ps = prev_sign[idx]
    if ps != 0 && s != 0 && s != ps
        osc[idx] += 1
        if osc[idx] > CONTROL_OSCILLATION_LIMIT
            _freeze_device!(frozen, idx, d, ts,
                "oscillating ($(osc[idx]) direction reversals within a steepness stage)")
            return 0.0
        end
    end
    prev_sign[idx] = s
    p_new = clamp(p_now + ω * (p_tgt - p_now), lo, hi)
    reached, moved = _continuation_to!(d, data, ts, p_new, pf; kwargs...)
    if !moved && abs(p_new - p_now) >= tol_d
        # The inner solver rejected even the smallest sub-step: the device cannot move
        # from here. Freeze it (with its warning) instead of letting a zero change
        # masquerade as a settled, regulated device.
        _freeze_device!(frozen, idx, d, ts,
            "the inner solver rejects any parameter movement (requested \
            $(p_new - p_now))")
        return 0.0
    end
    set_current_parameter!(d, reached)
    Δp = reached - p_now
    if abs(Δp) >= tol_d
        # Secant refresh of the plant gain from numbers this step already produced.
        g = (measured_value(d, data, ts) - yc) / Δp
        if dv != 0.0 && g != 0.0 && sign(g) != sign(dv)
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
    end
    return abs(Δp)
end

# Probe each device in one concrete vector for its plant sign (dy/dp) and write the
# result into the shared per-device state at `offset + i`. A function barrier: the body
# specializes on the concrete element type, so there is no dynamic dispatch over a
# heterogeneous device collection. `offset` reproduces the global
# [taps; shunts; facts; phase_shifters] indexing of `dVdp`/`frozen`. Devices whose probe
# fails OR whose full-range effect on the regulated quantity is below the gain floor
# (e.g. a PV-pinned controlled bus, where the measured sensitivity is exactly 0) are
# frozen — stepping them would slam the parameter to a rail with no feedback.
function _probe_device_signs!(
    devices, offset::Int, dVdp::Vector{Float64}, frozen::Vector{Bool},
    data, ts::Int, pf; kwargs...,
)
    for (i, d) in enumerate(devices)
        s, reliable = _plant_sign(d, data, ts, pf; kwargs...)
        lo, hi = parameter_limits(d)
        if reliable && abs(s) * (hi - lo) >= CONTROL_GAIN_FLOOR
            dVdp[offset + i] = s
        else
            frozen[offset + i] = true
        end
    end
    return nothing
end

# One proportional update over a concrete device vector; returns `true` iff every device
# in it settled (parameter change below its scale-aware tolerance). Same function-barrier
# + `offset` indexing.
function _step_device_group!(
    devices, offset::Int, data, ts::Int, S::Float64, pf,
    frozen::Vector{Bool}, dVdp::Vector{Float64}, osc::Vector{Int},
    prev_sign::Vector{Int}, n_shared::Vector{Int}; kwargs...,
)::Bool
    settled = true
    for (i, d) in enumerate(devices)
        g = _step_device!(
            d, offset + i, data, ts, S, pf, frozen, dVdp, osc, prev_sign, n_shared;
            kwargs...)
        g >= _param_tol(d) && (settled = false)
    end
    return settled
end

function _count_controlled_buses!(counts::Dict{Int, Int}, devices)
    for d in devices
        cix = controlled_bus_ix(d)
        counts[cix] = get(counts, cix, 0) + 1
    end
    return nothing
end

function _fill_shared!(
    n_shared::Vector{Int}, offset::Int, counts::Dict{Int, Int}, devices,
)
    for (i, d) in enumerate(devices)
        n_shared[offset + i] = counts[controlled_bus_ix(d)]
    end
    return nothing
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
    # Per-device state indexed in parallel with [taps; shunts; facts; phase_shifters]
    # (plain vectors, no per-iteration string hashing). dVdp is measured from the
    # converged base point and refreshed each step by a secant update: its sign sets the
    # negative-feedback orientation, its magnitude drives under-relaxation.
    # Unreliable-probe and insensitive devices are frozen, not stepped.
    dVdp = zeros(n_dev)
    frozen = fill(false, n_dev)
    osc = zeros(Int, n_dev)
    prev_sign = zeros(Int, n_dev)
    # Voltage devices sharing a controlled bus split the correction (ω / n_shared);
    # phase shifters regulate their own branch flow and never share.
    n_shared = ones(Int, n_dev)
    counts = Dict{Int, Int}()
    _count_controlled_buses!(counts, set.taps)
    _count_controlled_buses!(counts, set.shunts)
    _count_controlled_buses!(counts, set.facts)
    _fill_shared!(n_shared, 0, counts, set.taps)
    _fill_shared!(n_shared, n_taps, counts, set.shunts)
    _fill_shared!(n_shared, n_taps + n_shunts, counts, set.facts)

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
        @warn "discrete control: $(count(frozen)) device(s) had an unreliable or \
            insensitive plant probe and were frozen at their current parameter \
            (time step $ts): $frozen_names"
    end

    # Steepness ladder: each stage gets its own pass budget so a slow-settling stage
    # cannot starve the later (stiffer) stages, and the oscillation bookkeeping is
    # reset per stage — a ramp legitimately reverses update directions once.
    S = INITIAL_CONTROL_STEEPNESS
    regulation_complete = false
    while true
        settled = false
        for _ in 1:MAX_CONTROL_PASSES_PER_STAGE
            settled = _step_device_group!(
                set.taps, 0, data, ts, S, pf, frozen, dVdp, osc, prev_sign, n_shared;
                kwargs...)
            settled &= _step_device_group!(
                set.shunts, n_taps, data, ts, S, pf, frozen, dVdp, osc, prev_sign,
                n_shared; kwargs...)
            settled &= _step_device_group!(
                set.facts, n_taps + n_shunts, data, ts, S, pf, frozen, dVdp, osc,
                prev_sign, n_shared; kwargs...)
            settled &= _step_device_group!(
                set.phase_shifters, n_volt, data, ts, S, pf, frozen, dVdp, osc,
                prev_sign, n_shared; kwargs...)
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
    ok = snap_and_restore!(pf, data, set, ts; kwargs...)
    # Reported branch flows are computed from the arc-admittance matrices AFTER the
    # time-step loop (`solve_power_flow!`); bring the moved branch devices' rows in
    # line with their final parameters so flows match the solved network.
    _sync_arc_admittances!(data, set)
    return ok
end

# Incremental λ-restore of one device from its snapped value toward the
# continuous value (used only if snapping made the system infeasible).  First
# probe is a SMALL step, matching `_continuation_to!`.
function _restore_one!(d, data, ts::Int, continuous::Float64, pf; kwargs...)::Bool
    snapped = current_parameter(d)
    abs(continuous - snapped) < _param_tol(d) &&
        return _solve_with_q_limits!(pf, data, ts; kwargs...)
    lo, hi = parameter_limits(d)
    done = 0.0
    step = MIN_LAMBDA_STEP
    last_good = snapped
    snap = _snapshot_state(data, ts)   # last converged state, restored on a failed trial
    while done < 1.0
        trial = min(1.0, done + step)
        p = clamp(snapped + trial * (continuous - snapped), lo, hi)
        apply_parameter!(d, data, p, ts)
        if _solve_with_q_limits!(pf, data, ts; kwargs...)
            done = trial
            last_good = p
            _capture_state!(snap, data, ts)
            step = min(step * CONTROL_STEP_GROWTH, MAX_LAMBDA_STEP)
        else
            # Revert to the last converged parameter and state, not the failed trial.
            apply_parameter!(d, data, last_good, ts)
            _restore_state!(data, ts, snap)
            step /= 2.0
            if step < MIN_LAMBDA_STEP
                return false
            end
        end
    end
    return _solve_with_q_limits!(pf, data, ts; kwargs...)
end

# Snap a concrete device vector onto its discrete grid (continuous devices clamp),
# returning the pre-snap continuous values index-aligned with `devices` for the restore
# path. (Index alignment, not name keying: device names are only unique per concrete
# type, and a cross-family collision must not cross the stashed values.)
function _snap_device_group!(devices, data, ts::Int)
    cont = Vector{Float64}(undef, length(devices))
    for (i, d) in enumerate(devices)
        cont[i] = d.current
        apply_parameter!(d, data, snap_to_discrete(d, d.current), ts)
    end
    return cont
end

# λ-restore a concrete device vector toward its stashed continuous value; returns `true`
# iff all restored to a converged state.
function _restore_device_group!(
    devices, data, ts::Int, cont::Vector{Float64}, pf; kwargs...,
)::Bool
    ok = true
    for (i, d) in enumerate(devices)
        ok &= _restore_one!(d, data, ts, cont[i], pf; kwargs...)
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
    # Last continuous converged state: the restore path must start from here, not from
    # whatever diverged iterate a failed post-snap solve leaves in `data`.
    pre = _snapshot_state(data, ts)
    cont_taps = _snap_device_group!(set.taps, data, ts)
    cont_shunts = _snap_device_group!(set.shunts, data, ts)
    cont_facts = _snap_device_group!(set.facts, data, ts)
    cont_pst = _snap_device_group!(set.phase_shifters, data, ts)
    _solve_with_q_limits!(pf, data, ts; kwargs...) && return true
    _restore_state!(data, ts, pre)
    ok = _restore_device_group!(set.taps, data, ts, cont_taps, pf; kwargs...)
    ok &= _restore_device_group!(set.shunts, data, ts, cont_shunts, pf; kwargs...)
    ok &= _restore_device_group!(set.facts, data, ts, cont_facts, pf; kwargs...)
    ok &= _restore_device_group!(set.phase_shifters, data, ts, cont_pst, pf; kwargs...)
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

"""
    get_controlled_device_results(data) -> DataFrames.DataFrame

Solved discrete-control device settings: one row per enrolled device with its family,
name, control band, enrollment-time (`initial`) and solved (`final`) parameter. The
solved settings live only here and in the mutated network arrays — they are NOT written
back to the `PSY.System`, and a PSS/E export of the system reflects the ORIGINAL device
settings (see [`update_exporter!`](@ref)). Returns an empty frame when the data was built
without discrete control.
"""
function get_controlled_device_results(data)
    family = String[]
    name = String[]
    lower = Float64[]
    upper = Float64[]
    initial = Float64[]
    final = Float64[]
    set = get_controlled_devices(data)
    if set !== nothing
        for (fam, devices) in (
            ("TapTransformer", set.taps),
            ("SwitchedAdmittance", set.shunts),
            ("FACTSControlDevice", set.facts),
            ("PhaseShiftingTransformer", set.phase_shifters),
        )
            for d in devices
                lo, hi = parameter_limits(d)
                push!(family, fam)
                push!(name, d.name)
                push!(lower, lo)
                push!(upper, hi)
                push!(initial, d.initial)
                push!(final, d.current)
            end
        end
    end
    return DataFrames.DataFrame(
        "family" => family,
        "name" => name,
        "lower_limit" => lower,
        "upper_limit" => upper,
        "initial" => initial,
        "final" => final,
    )
end
