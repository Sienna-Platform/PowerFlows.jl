# PSY ≥ 6 (the psy6 branch, via PSY #1705) adds first-class tap-control fields to
# `TapTransformer` (`regulated_bus_number`, `tap_limits`, `number_of_tap_positions`,
# `voltage_setpoint`); released PSY 5.x does not have them. Resolved once at load time so
# the branch in `_tap_metadata` is a compile-time constant. When this package is built
# against the psy6 stack (see the psy6 branch's `[sources]` pins in Project.toml) the
# first-class path activates automatically; once PSY 6 is the compat floor, delete this
# constant and the 5.x fallback branch.
const PSY_HAS_TAP_CONTROL_FIELDS = isdefined(PSY, :get_regulated_bus_number)

"""Shunt susceptance invariants; `false` (with a `@warn`) de-enrolls the device, leaving
it locked at its current setting — the PSS/E posture for bad control data."""
function _validate_shunt(
    name::String,
    b_min::Float64,
    b0::Float64,
    b_max::Float64,
    steps::Vector{Int},
    dB::Vector{Float64},
)::Bool
    if !(b_min <= b0 <= b_max)
        @warn "ControlledSwitchedShunt \"$name\": b0=$b0 is outside \
            [b_min=$b_min, b_max=$b_max]; leaving the shunt locked at its current setting."
        return false
    end
    if b_min == b_max
        @warn "ControlledSwitchedShunt \"$name\": no controllable susceptance range \
            (b_min == b_max == $b_min); leaving the shunt locked at its current setting."
        return false
    end
    for k in eachindex(steps, dB)
        if steps[k] == 0 && dB[k] != 0.0
            @warn "ControlledSwitchedShunt \"$name\": block $k has zero steps but \
                nonzero dB=$(dB[k]) — malformed metadata; leaving the shunt locked."
            return false
        end
    end
    return true
end

"""Tap invariants; `false` (with a `@warn`) de-enrolls the device, leaving the tap locked
at its current ratio — the PSS/E posture for bad control data."""
function _validate_tap(
    name::String,
    p_min::Float64,
    p_max::Float64,
    ntp::Int,
)::Bool
    if p_min > p_max
        @warn "ControlledTap \"$name\": p_min=$p_min exceeds p_max=$p_max — malformed \
            tap-ratio limits; leaving the tap locked at its current ratio."
        return false
    end
    if p_min == p_max
        @warn "ControlledTap \"$name\": no controllable tap-ratio range \
            (p_min == p_max == $p_min); leaving the tap locked at its current ratio."
        return false
    end
    if ntp < 2
        # PSS/E treats missing/degenerate position data as a locked changer; fabricating
        # a default grid here would silently turn a locked device into an active one.
        @warn "ControlledTap \"$name\": fewer than 2 tap positions (ntp=$ntp); \
            leaving the tap locked at its current ratio."
        return false
    end
    return true
end

"""Voltage-setpoint plausibility gate shared by all voltage-controlling devices."""
function _validate_vset(kind::String, name::String, vset::Float64)::Bool
    if !(CONTROL_VSET_MIN <= vset <= CONTROL_VSET_MAX)
        @warn "$kind \"$name\": voltage setpoint $vset p.u. is outside \
            [$CONTROL_VSET_MIN, $CONTROL_VSET_MAX] — implausible control data (for parsed \
            systems PSY's admittance_limits holds the PSS/E VSWLO/VSWHI voltage band; \
            other sources may not). Leaving the device locked at its current setting."
        return false
    end
    return true
end

# First ext key (of `keys`, in order) holding a nonzero integer-parsable value; PSS/E uses
# 0 as the "local control" sentinel. Negative controlled-bus numbers (PSS/E CONT1 sign
# convention selects the metering/compensation side) refer to the same bus: take abs.
function _controlled_bus_number(
    ext::Dict,
    keys::Tuple{Vararg{String}},
    fallback::Int,
)
    for k in keys
        haskey(ext, k) || continue
        v = ext[k]
        n = v isa Integer ? Int(v) : tryparse(Int, string(v))
        (n !== nothing && n != 0) && return abs(n)
    end
    return fallback
end

function _ext_float(ext::Dict, keys::Tuple{Vararg{String}}, default::Float64)
    for k in keys
        haskey(ext, k) || continue
        v = ext[k]
        x = v isa Real ? Float64(v) : tryparse(Float64, string(v))
        x === nothing || return x
    end
    return default
end
_ext_float(ext::Dict, key::String, default::Float64) = _ext_float(ext, (key,), default)

function _ext_int(ext::Dict, keys::Tuple{Vararg{String}}, default::Int)
    for k in keys
        haskey(ext, k) || continue
        v = ext[k]
        x = v isa Integer ? Int(v) : tryparse(Int, string(v))
        x === nothing || return x
    end
    return default
end
_ext_int(ext::Dict, key::String, default::Int) = _ext_int(ext, (key,), default)

# Resolve a raw PSY bus number to a network index through the reduction's parent map
# (merged bus → surviving parent); `nothing` when the bus is not in the reduced network.
_resolve_bus_ix(
    bus_lookup::Dict{Int, Int},
    reverse_bus_search_map::Dict{Int, Int},
    n::Int,
) = get(bus_lookup, get(reverse_bus_search_map, n, n), nothing)

"""Tap-control metadata for one `TapTransformer`, merged from (in override order) the
PSS/E parser's ext keys and — on PSY ≥ 6 — the first-class fields.

PSS/E parser reality (PSY `pm_io/psse.jl`): two-winding records carry winding-suffixed
keys `CONT1` (controlled bus), `RMI1`/`RMA1` (ratio band), `NTP1` (positions); a voltage
setpoint is never persisted (`VMI1`/`VMA1` are dropped by the parser). The unsuffixed
`RMI`/`RMA`/`NTP`/`VSET` and `NREG` spellings are honored as user-facing overrides."""
function _tap_metadata(tx, to_bus::Int)
    ext = PSY.get_ext(tx)
    if PSY_HAS_TAP_CONTROL_FIELDS
        reg = PSY.get_regulated_bus_number(tx)   # 0 ⇒ local (to-bus) control
        lims = PSY.get_tap_limits(tx)
        ntp0 = PSY.get_number_of_tap_positions(tx)
        vset0 = PSY.get_voltage_setpoint(tx)
    else
        reg = 0
        lims = (min = DEFAULT_TAP_RATIO_MIN, max = DEFAULT_TAP_RATIO_MAX)
        ntp0 = DEFAULT_TAP_POSITIONS
        vset0 = DEFAULT_TAP_VSET
    end
    cbus = reg != 0 ? reg : _controlled_bus_number(ext, ("CONT1", "NREG"), to_bus)
    return (
        cbus = cbus,
        pmin = _ext_float(ext, ("RMI1", "RMI"), lims.min),
        pmax = _ext_float(ext, ("RMA1", "RMA"), lims.max),
        ntp = _ext_int(ext, ("NTP1", "NTP"), ntp0),
        vset = _ext_float(ext, ("VSET",), vset0),
    )
end

# Susceptance model of a switched shunt. The PSS/E parser (MODSW 0/1/2) stores
# `Y = BINIT` (the TOTAL in-service admittance) and ZEROES `initial_status` to avoid
# double counting, so the reachable set is spanned by the blocks alone (base 0) with the
# current point at BINIT. API-built components follow the PSY docstring instead: `Y` is
# the fixed N=0 base and `initial_status` is meaningful. The presence of the parser's
# MODSW key distinguishes the two conventions.
function _shunt_susceptance_model(
    name::String,
    Y0::Complex{Float64},
    steps::Vector{Int},
    dB::Vector{Float64},
    init_status::Vector{Int},
    ext::Dict,
)
    if haskey(ext, "MODSW")   # PSS/E parser convention
        b_fixed = 0.0
        current = imag(Y0)
    else                      # PSY API convention
        b_fixed = imag(Y0)
        current = imag(Y0) + sum(init_status .* dB; init = 0.0)
    end
    b_min = b_fixed + sum(min.(steps .* dB, 0.0); init = 0.0)
    b_max = b_fixed + sum(max.(steps .* dB, 0.0); init = 0.0)
    if !(b_min - BOUNDS_TOLERANCE <= current <= b_max + BOUNDS_TOLERANCE)
        @warn "ControlledSwitchedShunt \"$name\": initial susceptance $current p.u. lies \
            outside the block-reachable range [$b_min, $b_max]; clamping the control \
            baseline into the range."
        current = clamp(current, b_min, b_max)
    end
    return b_fixed, current, b_min, b_max
end

"""Build the type-stable device set from a `PSY.System`.

`bus_lookup` maps PSY bus number → network index in the (possibly reduced) network;
`reverse_bus_search_map` maps reduction-merged bus numbers to their surviving parent;
`ybus` is the assembled `AC_Ybus_Matrix` from `data.power_network_matrix`.

Per-device data problems (unresolvable buses, degenerate ranges, unsupported control
modes) de-enroll the device with a `@warn` — the device stays at its current setting,
matching PSS/E's warn-and-lock posture — and never abort construction.

`include_experimental` gates the `ControlledFACTS` and `ControlledPhaseShifter` families,
whose data sourcing is not yet production-validated (enable via
`solver_settings[:experimental_controls] = true`)."""
function build_controlled_device_set(
    sys,
    bus_lookup::Dict{Int, Int},
    ybus;
    reverse_bus_search_map::Dict{Int, Int} = Dict{Int, Int}(),
    include_experimental::Bool = false,
)
    taps = ControlledTap[]
    for tx in PSY.get_available_components(PSY.TapTransformer, sys)
        PSY.get_control_objective(tx) == PSY.TransformerControlObjective.VOLTAGE ||
            continue
        name = PSY.get_name(tx)
        arc = PSY.get_arc(tx)
        fb = PSY.get_number(PSY.get_from(arc))
        tb = PSY.get_number(PSY.get_to(arc))
        md = _tap_metadata(tx, tb)
        fix = _resolve_bus_ix(bus_lookup, reverse_bus_search_map, fb)
        tix = _resolve_bus_ix(bus_lookup, reverse_bus_search_map, tb)
        cix = _resolve_bus_ix(bus_lookup, reverse_bus_search_map, md.cbus)
        if fix === nothing || tix === nothing
            @warn "ControlledTap \"$name\": terminal bus $(fix === nothing ? fb : tb) \
                is not in the (reduced) network; leaving the tap locked."
            continue
        end
        if cix === nothing
            @warn "ControlledTap \"$name\": controlled bus $(md.cbus) is not in the \
                (reduced) network; leaving the tap locked."
            continue
        end
        if fix == tix
            @warn "ControlledTap \"$name\": arc collapsed by a network reduction \
                (from == to after bus merging); leaving the tap locked."
            continue
        end
        _validate_tap(name, md.pmin, md.pmax, md.ntp) || continue
        _validate_vset("ControlledTap", name, md.vset) || continue
        yt = 1.0 / (PSY.get_r(tx) + PSY.get_x(tx) * im)
        ysh = PSY.get_primary_shunt(tx)
        tap0 = PSY.get_tap(tx)
        push!(
            taps,
            ControlledTap(
                name,
                fix,
                tix,
                cix,
                md.vset,
                yt,
                ysh,
                PSY.get_α(tx),   # −(π/6)·winding_group_number: PNM stamps t = p·e^{iα}
                md.pmin,
                md.pmax,
                collect(range(md.pmin, md.pmax; length = md.ntp)),
                _ybus_block_offsets(ybus, fix, tix),
                tap0,   # initial (reporting)
                tap0,   # synced (arc-admittance rows reflect this tap)
                tap0,   # current
            ),
        )
    end

    shunts = ControlledSwitchedShunt[]
    for sa in PSY.get_available_components(PSY.SwitchedAdmittance, sys)
        name = PSY.get_name(sa)
        bus = PSY.get_number(PSY.get_bus(sa))
        ext = PSY.get_ext(sa)
        modsw = _ext_int(ext, "MODSW", DEFAULT_SHUNT_MODSW)
        if modsw == 0
            @debug "ControlledSwitchedShunt $name: MODSW=0 (locked); \
                treated as fixed admittance, not enrolled."
            continue
        elseif modsw == 1
            continuous = false
        elseif modsw == 2
            continuous = true
        else
            @warn "ControlledSwitchedShunt \"$name\": MODSW=$modsw (remote \
                reactive-power / remote-device control) is not supported — only MODSW=1 \
                (discrete voltage) and MODSW=2 (continuous voltage) are implemented. \
                Leaving the shunt locked at its current setting."
            continue
        end
        bix = _resolve_bus_ix(bus_lookup, reverse_bus_search_map, bus)
        if bix === nothing
            @warn "ControlledSwitchedShunt \"$name\": bus $bus is not in the (reduced) \
                network; leaving the shunt locked."
            continue
        end
        # v35 stores the regulated bus under NREG, v32/33 under SWREM. (RMIDNT is a
        # character remote-device identifier, not a bus number — deliberately not read.)
        cbus = _controlled_bus_number(ext, ("NREG", "SWREM"), bus)
        cix = _resolve_bus_ix(bus_lookup, reverse_bus_search_map, cbus)
        if cix === nothing
            @warn "ControlledSwitchedShunt \"$name\": controlled bus $cbus is not in \
                the (reduced) network; leaving the shunt locked."
            continue
        end
        lims = PSY.get_admittance_limits(sa)
        vset = (lims.min + lims.max) / 2.0
        _validate_vset("ControlledSwitchedShunt", name, vset) || continue
        Y0 = PSY.get_Y(sa)
        steps = PSY.get_number_of_steps(sa)
        dB = imag.(PSY.get_Y_increase(sa))
        b_fixed, current_b, bmin, bmax = _shunt_susceptance_model(
            name, Y0, steps, dB, PSY.get_initial_status(sa), ext)
        _validate_shunt(name, bmin, b_fixed, bmax, steps, dB) || continue
        push!(
            shunts,
            ControlledSwitchedShunt(
                name,
                bix,
                cix,
                vset,
                real(Y0),
                b_fixed,
                steps,
                dB,
                bmin,
                bmax,
                zeros(Int, length(dB)),
                continuous,
                current_b,   # initial (reporting)
                current_b,   # current
            ),
        )
    end

    facts = ControlledFACTS[]
    phase_shifters = ControlledPhaseShifter[]
    if include_experimental
        _enroll_facts!(facts, sys, bus_lookup, reverse_bus_search_map)
        _enroll_phase_shifters!(
            phase_shifters, sys, bus_lookup, reverse_bus_search_map, ybus)
    else
        n_facts = length(PSY.get_available_components(PSY.FACTSControlDevice, sys))
        n_pst = count(
            ps ->
                PSY.get_control_objective(ps) ==
                PSY.TransformerControlObjective.ACTIVE_POWER_FLOW,
            PSY.get_available_components(PSY.PhaseShiftingTransformer, sys),
        )
        if n_facts + n_pst > 0
            @info "discrete control: $n_facts FACTS device(s) and $n_pst \
                phase-shifter(s) present but not enrolled — FACTS/PAR control is \
                experimental. Enable with solver_settings[:experimental_controls] = true."
        end
    end

    return ControlledDeviceSet(taps, shunts, facts, phase_shifters)
end

# EXPERIMENTAL: continuous shunt FACTS (SVC/STATCOM) voltage control. Data sourcing is
# not yet production-validated (remote FCREG regulation and the |V|-dependent current
# limit are not modeled); gated behind solver_settings[:experimental_controls].
function _enroll_facts!(
    facts::Vector{ControlledFACTS},
    sys,
    bus_lookup::Dict{Int, Int},
    reverse_bus_search_map::Dict{Int, Int},
)
    base_mva = PSY.get_base_power(sys)
    for fd in PSY.get_available_components(PSY.FACTSControlDevice, sys)
        name = PSY.get_name(fd)
        mode = PSY.get_control_mode(fd)
        # OOS or no control mode ⇒ not a voltage-controlling shunt; not enrolled.
        if mode === nothing || mode == PSY.FACTSOperationModes.OOS
            @debug "ControlledFACTS $name: control_mode=$(mode) is not \
                voltage-controlling; not enrolled."
            continue
        end
        bus = PSY.get_number(PSY.get_bus(fd))
        bix = _resolve_bus_ix(bus_lookup, reverse_bus_search_map, bus)
        if bix === nothing
            @warn "ControlledFACTS \"$name\": bus $bus is not in the (reduced) network; \
                device not enrolled."
            continue
        end
        # `max_shunt_current` is MVA at unity voltage ⇒ a symmetric susceptance band in
        # p.u. on system base (Q = b·|V|², so at |V|=1 the MVA rating bounds |b|).
        b_max = PSY.get_max_shunt_current(fd) / base_mva
        if b_max <= 0.0
            @warn "ControlledFACTS \"$name\": max_shunt_current must be positive \
                (series-only FACTS records parse with 0.0); device not enrolled."
            continue
        end
        vset = PSY.get_voltage_setpoint(fd)
        _validate_vset("ControlledFACTS", name, vset) || continue
        push!(
            facts,
            ControlledFACTS(
                name,
                bix,
                bix,               # shunt regulates its own (sending) bus
                vset,
                -b_max,
                b_max,
                0.0,               # initial (reporting)
                0.0,               # start neutral; the controller drives b from 0
            ),
        )
    end
    return nothing
end

# EXPERIMENTAL: phase-angle regulator (PAR) active-power-flow control. The flow setpoint
# and angle band are not persisted by the PSS/E parser (`pf` defaults to 0.0 and
# RMI1/RMA1 are not mapped to phase_angle_limits), so enrollment requires explicitly
# populated PSY fields; gated behind solver_settings[:experimental_controls].
function _enroll_phase_shifters!(
    phase_shifters::Vector{ControlledPhaseShifter},
    sys,
    bus_lookup::Dict{Int, Int},
    reverse_bus_search_map::Dict{Int, Int},
    ybus,
)
    for ps in PSY.get_available_components(PSY.PhaseShiftingTransformer, sys)
        PSY.get_control_objective(ps) ==
        PSY.TransformerControlObjective.ACTIVE_POWER_FLOW || continue
        name = PSY.get_name(ps)
        p_target = PSY.get_active_power_flow(ps)
        if p_target == 0.0
            # A raw-parsed PST always lands here (the parser never sets `pf`): a zero
            # setpoint would command the PAR to erase its own flow — actively harmful.
            @warn "ControlledPhaseShifter \"$name\": active_power_flow is 0.0, which is \
                the parser default rather than a real flow setpoint; device not \
                enrolled. Set a nonzero setpoint on the component to enable control."
            continue
        end
        arc = PSY.get_arc(ps)
        fb = PSY.get_number(PSY.get_from(arc))
        tb = PSY.get_number(PSY.get_to(arc))
        fix = _resolve_bus_ix(bus_lookup, reverse_bus_search_map, fb)
        tix = _resolve_bus_ix(bus_lookup, reverse_bus_search_map, tb)
        if fix === nothing || tix === nothing || fix == tix
            @warn "ControlledPhaseShifter \"$name\": arc bus missing from the (reduced) \
                network or collapsed by a reduction; device not enrolled."
            continue
        end
        lims = PSY.get_phase_angle_limits(ps)
        yt = 1.0 / (PSY.get_r(ps) + PSY.get_x(ps) * im)
        alpha0 = PSY.get_α(ps)
        push!(
            phase_shifters,
            ControlledPhaseShifter(
                name,
                fix,
                tix,
                p_target,
                yt,
                PSY.get_tap(ps),
                lims.min,
                lims.max,
                _ybus_block_offsets(ybus, fix, tix),
                alpha0,   # initial (reporting)
                alpha0,   # synced (arc-admittance rows reflect this angle)
                alpha0,   # current
            ),
        )
    end
    return nothing
end
