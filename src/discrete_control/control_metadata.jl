"""Shunt susceptance invariants; `false` (with a `@warn`) de-enrolls the device, leaving
it locked at its current setting (the safe posture for bad control data)."""
function _validate_shunt(
    name::String,
    b_min::Float64,
    b_max::Float64,
    steps::Vector{Int},
    dB::Vector{Float64},
)::Bool
    if !(b_min <= 0.0 <= b_max)
        @warn "ControlledSwitchedShunt \"$name\": the all-off susceptance 0.0 is outside \
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

_regulates_voltage(circuit::PSY.TransformerCircuit) =
    PSY.get_control_objective(circuit) == PSY.TransformerControlObjective.VOLTAGE

"""
Voltage-regulating tap changers in the system, one candidate per controlling circuit.

The control objective lives on `PSY.TransformerCircuit`, so a regulating tap can sit on
either arity — a three-winding transformer may regulate on one circuit while the other two
are fixed. Phase-shifting objectives (the active-power ones) are excluded by construction:
they are a different control law, handled as an angle rather than a ratio.
"""
function _voltage_controlled_tap_candidates(sys)
    candidates =
        Tuple{String, PSY.ACTransmission, PSY.TransformerCircuit, String, Int}[]
    for tx in PSY.get_available_components(PSY.TwoWindingTransformer, sys)
        circuit = PSY.get_circuit(tx)
        _regulates_voltage(circuit) || continue
        name = PSY.get_name(tx)
        push!(candidates, (name, tx, circuit, name, 1))
    end
    for tx in PSY.get_available_components(PSY.ThreeWindingTransformer, sys)
        for (i, circuit) in enumerate(PSY.get_circuits(tx))
            PSY.get_available(circuit) || continue
            _regulates_voltage(circuit) || continue
            winding = PNM.ThreeWindingTransformerCircuit(tx, i)
            push!(
                candidates,
                (PNM.get_name(winding), winding, circuit, PSY.get_name(tx), i),
            )
        end
    end
    return candidates
end

"""Tap-control metadata for one regulating `PSY.TransformerCircuit`, of either arity.
`control_limits` is used directly as the tap-ratio band `[pmin, pmax]`, but PSS/E's RMI1/RMA1
bound WINDV1 while `PSY.get_tap` stores the ratio WINDV1/WINDV2; `TransformerCircuit` has no
WINDV2-equivalent field, so this band is wrong by a factor of WINDV2 whenever WINDV2 != 1 for
the parsed transformer (the correct band would be `control_limits ./ WINDV2`). Fixing this
needs a data-model change upstream (PFFP/PSY), not here.
`get_regulated_bus_number` is 0 for local (to-bus) control."""
function _tap_metadata(circuit::PSY.TransformerCircuit, to_bus::Int)
    lims = PSY.get_control_limits(circuit)
    reg = PSY.get_regulated_bus_number(circuit)
    cbus = to_bus
    if !iszero(reg)
        # The sign marks the regulation side (PSS/E CONT<0); the bus number itself is |reg|.
        cbus = abs(reg)
    end
    # The tap is held anywhere inside the VMA/VMI band and regulates toward its midpoint on
    # an excursion — the same posture as a switched shunt's VSWLO/VSWHI.
    vlims = PSY.get_controlled_quantity_limits(circuit)
    return (
        cbus = cbus,
        pmin = lims.min,
        pmax = lims.max,
        ntp = PSY.get_number_of_tap_positions(circuit),
        vset = (vlims.min + vlims.max) / 2,
        vlo = vlims.min,
        vhi = vlims.max,
    )
end

# Susceptance model of a switched shunt. `SwitchedAdmittance` has no fixed base admittance:
# total is `number_engaged .* Y_increase`, unless `solved_admittance` is set, in which case
# that value is the effective admittance directly (PSS/E BINIT, a case read in as solved).
_solved_flag(::Nothing) = false
_solved_flag(::Float64) = true

function _shunt_susceptance_model(
    name::String,
    solved::Union{Nothing, Float64},
    y_increase::Vector{Complex{Float64}},
    steps::Vector{Int},
    dB::Vector{Float64},
    engaged::Vector{Int},
)
    # `_switched_admittance` (common.jl) is the one baseline calculation; a shunt's
    # susceptance is a pure imaginary admittance, so `imag` recovers it.
    current = imag(_switched_admittance(solved, engaged, y_increase))
    b_min = sum(min.(steps .* dB, 0.0); init = 0.0)
    b_max = sum(max.(steps .* dB, 0.0); init = 0.0)
    if !(b_min - BOUNDS_TOLERANCE <= current <= b_max + BOUNDS_TOLERANCE)
        @warn "ControlledSwitchedShunt \"$name\": initial susceptance $current p.u. lies \
            outside the block-reachable range [$b_min, $b_max]; clamping the control \
            baseline into the range."
        current = clamp(current, b_min, b_max)
    end
    return current, b_min, b_max
end

"""Build the type-stable device set from a `PSY.System`.

`bus_lookup` maps PSY bus number → network index in the (possibly reduced) network;
`reverse_bus_search_map` maps reduction-merged bus numbers to their surviving parent;
`ybus` is the assembled `AC_Ybus_Matrix` from `data.power_network_matrix`. `n_time_steps`
sizes the returned set's per-ts shunt/FACTS state store (see `ControlledDeviceSet`).

Per-device data problems (unresolvable buses, degenerate ranges, unsupported control
modes) de-enroll the device with a `@warn` — the device stays at its current setting
(a warn-and-lock posture) — and never abort construction."""
function build_controlled_device_set(
    sys,
    bus_lookup::Dict{Int, Int},
    ybus;
    reverse_bus_search_map::Dict{Int, Int} = Dict{Int, Int}(),
    n_time_steps::Int = 1,
)
    nrd = PNM.get_network_reduction_data(ybus)
    taps = ControlledTap[]
    for (name, branch, circuit, device_name, circuit_index) in
        _voltage_controlled_tap_candidates(sys)
        arc = PSY.get_arc(circuit)
        fb = PSY.get_number(PSY.get_from(arc))
        tb = PSY.get_number(PSY.get_to(arc))
        md = _tap_metadata(circuit, tb)
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
        # PNM owns the π-model, including the r == x == 0 floor that a hand-built
        # `1/(r + jx)` would miss (a jumper under tap control would yield `Inf`), and the
        # `nrd`-aware method applies the same impedance-correction factor the assembled
        # Ybus was stamped with — the uncorrected component form would drift from the
        # actual matrix the moment a correction table is in play.
        adm = PNM.branch_admittance(branch, nrd)
        yt = complex(adm.g, adm.b)
        tap0 = adm.tap
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
                md.vlo,
                md.vhi,
                yt,
                adm.shift,  # PNM stamps t = p·e^{iα}
                md.pmin,
                md.pmax,
                collect(range(md.pmin, md.pmax; length = md.ntp)),
                _ybus_block_offsets(ybus, fix, tix),
                tap0,   # initial (reporting)
                tap0,   # synced (arc-admittance rows reflect this tap)
                tap0,   # current
                device_name,
                circuit_index,
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
        solved = PSY.get_solved_admittance(sa)
        steps = PSY.get_number_of_steps(sa)
        y_increase = PSY.get_Y_increase(sa)
        dB = imag.(y_increase)
        engaged = PSY.get_number_engaged(sa)
        current_b, bmin, bmax = _shunt_susceptance_model(
            name, solved, y_increase, steps, dB, engaged)
        _validate_shunt(name, bmin, bmax, steps, dB) || continue
        push!(
            shunts,
            ControlledSwitchedShunt(
                name,
                bix,
                cix,
                vset,
                lims.min,   # VSWLO: deadband lower edge
                lims.max,   # VSWHI: deadband upper edge
                steps,
                dB,
                bmin,
                bmax,
                zeros(Int, length(dB)),
                continuous,
                current_b,   # initial (reporting)
                current_b,   # current
                _solved_flag(solved),   # psse_convention: true ⇒ case read in as solved
            ),
        )
    end

    facts = ControlledFACTS[]
    _enroll_facts!(facts, sys, bus_lookup, reverse_bus_search_map)

    return ControlledDeviceSet(taps, shunts, facts, n_time_steps)
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
        rating = PSY.get_max_shunt_current(fd, PSY.SU)
        q_cap = PSY.get_max_reactive_power(fd, PSY.SU)
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
