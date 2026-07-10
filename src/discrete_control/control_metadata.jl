"""Shunt susceptance invariants; `false` (with a `@warn`) de-enrolls the device, leaving
it locked at its current setting (the safe posture for bad control data)."""
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
        if iszero(steps[k]) && !iszero(dB[k])
            @warn "ControlledSwitchedShunt \"$name\": block $k has zero steps but \
                nonzero dB=$(dB[k]) — malformed metadata; leaving the shunt locked."
            return false
        end
    end
    return true
end

"""Tap invariants; `false` (with a `@warn`) de-enrolls the device, leaving the tap locked
at its current ratio (the safe posture for bad control data)."""
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
        # Missing/degenerate position data means a locked changer; fabricating a default
        # grid here would silently turn a locked device into an active one.
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

# Resolve a raw PSY bus number to a network index through the reduction's parent map
# (merged bus → surviving parent); `nothing` when the bus is not in the reduced network.
_resolve_bus_ix(
    bus_lookup::Dict{Int, Int},
    reverse_bus_search_map::Dict{Int, Int},
    n::Int,
) = get(bus_lookup, get(reverse_bus_search_map, n, n), nothing)

"""Tap-control metadata for one `TapTransformer`, read from its first-class PSY fields.
`get_tap_limits` is already in tap-ratio units (the PSS/E parser scales RMI1/RMA1 by WINDV2);
`get_regulated_bus_number` is 0 for local (to-bus) control."""
function _tap_metadata(tx, to_bus::Int)
    lims = PSY.get_tap_limits(tx)
    reg = PSY.get_regulated_bus_number(tx)
    cbus = to_bus
    if !iszero(reg)
        cbus = reg
    end
    return (
        cbus = cbus,
        pmin = lims.min,
        pmax = lims.max,
        ntp = PSY.get_number_of_tap_positions(tx),
        vset = PSY.get_voltage_setpoint(tx),
    )
end

# Susceptance model of a switched shunt. The PSS/E parser (MODSW 0/1/2) stores
# `Y = BINIT` (the TOTAL in-service admittance) and ZEROES `initial_status` to avoid
# double counting, so the reachable set is spanned by the blocks alone (base 0) with the
# current point at BINIT. API-built components follow the PSY docstring instead: `Y` is
# the fixed N=0 base and `initial_status` is meaningful. The presence of the parser's
# MODSW key distinguishes the two conventions.
# BINIT convention ⇔ the PSS/E parser produced this shunt: it stores the TOTAL in-service
# admittance in `Y` and zeroes a full-length `initial_status` (see pm_io/psse.jl). API-built
# shunts carry the switched part in a nonzero (or empty) `initial_status`. This is the write-back
# convention too (see `ControlledSwitchedShunt.psse_convention`).
function _is_binit_shunt(init_status::Vector{Int})
    return !isempty(init_status) && all(iszero, init_status)
end

function _shunt_susceptance_model(
    name::String,
    Y0::Complex{Float64},
    steps::Vector{Int},
    dB::Vector{Float64},
    init_status::Vector{Int},
)
    if _is_binit_shunt(init_status)   # PSS/E parser (BINIT) convention
        b_fixed = 0.0
        current = imag(Y0)
    else                              # PSY API convention
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
modes) de-enroll the device with a `@warn` — the device stays at its current setting
(a warn-and-lock posture) — and never abort construction."""
function build_controlled_device_set(
    sys,
    bus_lookup::Dict{Int, Int},
    ybus;
    reverse_bus_search_map::Dict{Int, Int} = Dict{Int, Int}(),
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
        if isnothing(fix) || isnothing(tix)
            missing_bus = tb
            if isnothing(fix)
                missing_bus = fb
            end
            @warn "ControlledTap \"$name\": terminal bus $missing_bus \
                is not in the (reduced) network; leaving the tap locked."
            continue
        end
        if isnothing(cix)
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
        tap0 = PSY.get_tap(tx)
        if !(md.pmin - BOUNDS_TOLERANCE <= tap0 <= md.pmax + BOUNDS_TOLERANCE)
            @warn "ControlledTap \"$name\": initial tap ratio $tap0 lies \
                outside the tap-ratio band [$(md.pmin), $(md.pmax)]; leaving the tap \
                locked at its current ratio."
            continue
        end
        push!(
            taps,
            ControlledTap(
                name,
                fix,
                tix,
                cix,
                md.vset,
                yt,
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
        mode = PSY.get_control_mode(sa)
        if mode == PSY.SwitchedAdmittanceControlMode.DISCRETE_VOLTAGE
            continuous = false
        elseif mode == PSY.SwitchedAdmittanceControlMode.CONTINUOUS_VOLTAGE
            continuous = true
        elseif mode == PSY.SwitchedAdmittanceControlMode.FIXED
            @debug "ControlledSwitchedShunt $name: control_mode FIXED (locked); \
                treated as fixed admittance, not enrolled."
            continue
        else
            @warn "ControlledSwitchedShunt \"$name\": control_mode $mode (remote \
                reactive-power / remote-device control) is not supported — only \
                DISCRETE_VOLTAGE and CONTINUOUS_VOLTAGE are implemented. Leaving the \
                shunt locked at its current setting."
            continue
        end
        bix = _resolve_bus_ix(bus_lookup, reverse_bus_search_map, bus)
        if isnothing(bix)
            @warn "ControlledSwitchedShunt \"$name\": bus $bus is not in the (reduced) \
                network; leaving the shunt locked."
            continue
        end
        # `regulated_bus_number` is 0 for local control (PSS/E SWREM/NREG map to it in the parser).
        reg = PSY.get_regulated_bus_number(sa)
        cbus = bus
        if !iszero(reg)
            cbus = reg
        end
        cix = _resolve_bus_ix(bus_lookup, reverse_bus_search_map, cbus)
        if isnothing(cix)
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
        init_status = PSY.get_initial_status(sa)
        b_fixed, current_b, bmin, bmax = _shunt_susceptance_model(
            name, Y0, steps, dB, init_status)
        _validate_shunt(name, bmin, b_fixed, bmax, steps, dB) || continue
        push!(
            shunts,
            ControlledSwitchedShunt(
                name,
                bix,
                cix,
                vset,
                lims.min,   # VSWLO: deadband lower edge
                lims.max,   # VSWHI: deadband upper edge
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
                _is_binit_shunt(init_status),   # psse_convention: true ⇒ parser/BINIT
            ),
        )
    end

    facts = ControlledFACTS[]
    _enroll_facts!(facts, sys, bus_lookup, reverse_bus_search_map)

    return ControlledDeviceSet(taps, shunts, facts)
end

# Continuous shunt FACTS (SVC/STATCOM) voltage control. `rating` (SHMX) bounds the SVC
# susceptance-at-unity or the STATCOM current; `q_cap` is an independent MVA ceiling. Both
# combine into the |V|-dependent limit `_facts_b_limit`. FCREG (`regulated_bus_number`)
# selects local vs. remote-bus regulation.
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
        if isnothing(mode) || mode == PSY.FACTSOperationModes.OOS
            @debug "ControlledFACTS $name: control_mode=$(mode) is not \
                voltage-controlling; not enrolled."
            continue
        end
        bus = PSY.get_number(PSY.get_bus(fd))
        bix = _resolve_bus_ix(bus_lookup, reverse_bus_search_map, bus)
        if isnothing(bix)
            @warn "ControlledFACTS \"$name\": bus $bus is not in the (reduced) network; \
                device not enrolled."
            continue
        end
        reg = PSY.get_regulated_bus_number(fd)
        cix = bix
        if !iszero(reg)
            cix = _resolve_bus_ix(bus_lookup, reverse_bus_search_map, reg)
        end
        if isnothing(cix)
            @warn "ControlledFACTS \"$name\": regulated bus $reg is not in the \
                (reduced) network; device not enrolled."
            continue
        end
        # `rating` (SHMX) is MVA at unity voltage ⇒ the SVC susceptance-at-unity bound or
        # the STATCOM current limit, on system base. `q_cap` is an independent MVA ceiling.
        rating = PSY.get_max_shunt_current(fd) / base_mva
        q_cap = PSY.get_max_reactive_power(fd) / base_mva
        svc = PSY.get_shunt_control_type(fd) == PSY.FACTSShuntControlType.SVC
        if rating <= 0.0
            @warn "ControlledFACTS \"$name\": max_shunt_current must be positive \
                (series-only FACTS records parse with 0.0); device not enrolled."
            continue
        end
        vset = PSY.get_voltage_setpoint(fd)
        _validate_vset("ControlledFACTS", name, vset) || continue
        b0 = rating   # enrollment-time bound at unity voltage
        push!(
            facts,
            ControlledFACTS(
                name,
                bix,
                cix,
                vset,
                svc,
                rating,
                q_cap,
                b0,
                base_mva,
                0.0,               # initial (reporting)
                0.0,               # start neutral; the controller drives b from 0
                false,             # saturated (set post-solve)
            ),
        )
    end
    return
end
