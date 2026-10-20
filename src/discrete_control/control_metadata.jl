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

"""Actuator invariants of a regulating transformer circuit; `false` (with a `@warn`)
de-enrolls the device, leaving it locked at its current setting (the safe posture for bad
control data). `actuator` selects the wording: tap ratio or phase angle."""
function _validate_tap(
    name::String,
    p_min::Float64,
    p_max::Float64,
    ntp::Int,
    actuator::BranchActuator = TapRatio(),
)::Bool
    kind = _actuator_kind(actuator)
    if p_min > p_max
        @warn "Controlled $kind circuit \"$name\": p_min=$p_min exceeds p_max=$p_max — \
            malformed $kind limits; leaving the circuit locked at its current setting."
        return false
    end
    if p_min == p_max
        @warn "Controlled $kind circuit \"$name\": no controllable $kind range \
            (p_min == p_max == $p_min); leaving the circuit locked at its current setting."
        return false
    end
    if ntp < 2
        # Missing/degenerate position data means a locked changer; fabricating a default
        # grid here would silently turn a locked device into an active one.
        @warn "Controlled $kind circuit \"$name\": fewer than 2 tap positions (ntp=$ntp); \
            leaving the circuit locked at its current setting."
        return false
    end
    return true
end

"""Flow-band invariants; `false` (with a `@warn`) de-enrolls the device. The band must be
finite and ordered."""
function _validate_band(label::String, name::String, lo::Float64, hi::Float64)::Bool
    if !(isfinite(lo) && isfinite(hi) && lo <= hi)
        @warn "$label \"$name\": the flow band [$lo, $hi] p.u. is not finite and ordered — \
            malformed control data. Leaving the device locked at its current setting."
        return false
    end
    return true
end

"""Voltage-setpoint plausibility gate shared by all voltage-controlling devices."""
function _validate_vset(kind::String, name::String, vset::Float64)::Bool
    if !(CONTROL_VSET_MIN <= vset <= CONTROL_VSET_MAX)
        @warn "$kind \"$name\": voltage setpoint $vset p.u. is outside \
            [$CONTROL_VSET_MIN, $CONTROL_VSET_MAX] — implausible control data (for parsed \
            systems PSY's voltage_limits holds the PSS/E VSWLO/VSWHI voltage band; \
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

"""A circuit whose control objective this package does not enroll."""
struct NoCircuitControl end

"""The one place that maps a PSY control objective to the device type that regulates it.
An objective that is not listed here is not enrolled."""
function _circuit_control(objective::PSY.TransformerControlObjective.Value)
    if objective == PSY.TransformerControlObjective.VOLTAGE
        return ControlledBranch{TapRatio, BusVoltage}
    elseif objective == PSY.TransformerControlObjective.REACTIVE_POWER_FLOW
        return ControlledBranch{TapRatio, ReactivePowerFlow}
    elseif objective == PSY.TransformerControlObjective.ACTIVE_POWER_FLOW
        return ControlledBranch{PhaseAngle, ActivePowerFlow}
    end
    return NoCircuitControl()
end

_enrolls(::NoCircuitControl) = false
_enrolls(::Type{<:ControlledBranch}) = true
_enrolls(circuit::PSY.TransformerCircuit) =
    _enrolls(_circuit_control(PSY.get_control_objective(circuit)))

_control_label(::Type{ControlledBranch{TapRatio, BusVoltage}}) = "ControlledTap"
_control_label(::Type{ControlledBranch{TapRatio, ReactivePowerFlow}}) = "ControlledFlowTap"
_control_label(::Type{ControlledBranch{PhaseAngle, ActivePowerFlow}}) =
    "ControlledPhaseShifter"

_actuator_kind(::TapRatio) = "tap-ratio"
_actuator_kind(::PhaseAngle) = "phase-angle"

_device_group(::Type{ControlledBranch{TapRatio, BusVoltage}}, groups) = groups.taps
_device_group(::Type{ControlledBranch{TapRatio, ReactivePowerFlow}}, groups) =
    groups.flow_taps
_device_group(::Type{ControlledBranch{PhaseAngle, ActivePowerFlow}}, groups) =
    groups.phase_shifters

"""
Regulating transformer circuits in the system, one candidate per controlling circuit.

The control objective lives on `PSY.TransformerCircuit`, so a regulating circuit can sit on
either arity — a three-winding transformer may regulate on one circuit while the other two
are fixed. A circuit is a candidate when `_circuit_control` enrolls its objective: voltage,
reactive-power flow (both move the tap ratio) or active-power flow (moves the phase angle).
"""
function _controlled_circuit_candidates(sys)
    candidates =
        Tuple{String, PSY.ACTransmission, PSY.TransformerCircuit, String, Int}[]
    for tx in PSY.get_available_components(PSY.TwoWindingTransformer, sys)
        circuit = PSY.get_circuit(tx)
        _enrolls(circuit) || continue
        name = PSY.get_name(tx)
        push!(candidates, (name, tx, circuit, name, 1))
    end
    for tx in PSY.get_available_components(PSY.ThreeWindingTransformer, sys)
        for (i, circuit) in enumerate(PSY.get_circuits(tx))
            PSY.get_available(circuit) || continue
            _enrolls(circuit) || continue
            winding = PNM.ThreeWindingTransformerCircuit(tx, i)
            push!(
                candidates,
                (PNM.get_name(winding), winding, circuit, PSY.get_name(tx), i),
            )
        end
    end
    return candidates
end

# A regulating device must carry the bands its control law selects.
_required_band(band, ::String, ::Symbol) = band
function _required_band(::Nothing, label::String, field::Symbol)
    error("$label is enrolled for control but has no $(field).")
end

_actuator_limits(::TapRatio, circuit::PSY.TransformerCircuit, label::String) =
    _required_band(PSY.get_tap_ratio_limits(circuit), label, :tap_ratio_limits)
_actuator_limits(::PhaseAngle, circuit::PSY.TransformerCircuit, label::String) =
    _required_band(PSY.get_phase_angle_limits(circuit), label, :phase_angle_limits)

_target_band(::BusVoltage, circuit::PSY.TransformerCircuit, label::String) = _required_band(
    PSY.get_controlled_voltage_limits(circuit), label, :controlled_voltage_limits,
)
_target_band(::ReactivePowerFlow, circuit::PSY.TransformerCircuit, label::String) =
    _required_band(
        PSY.get_controlled_reactive_power_flow_limits(circuit, u"SU"),
        label, :controlled_reactive_power_flow_limits,
    )
_target_band(::ActivePowerFlow, circuit::PSY.TransformerCircuit, label::String) =
    _required_band(
        PSY.get_controlled_active_power_flow_limits(circuit, u"SU"),
        label, :controlled_active_power_flow_limits,
    )

# Voltage control regulates the to bus, or the remote bus that `regulated_bus_number` names. The
# sign of that number marks the regulation side (PSS/E CONT<0); the bus number is its magnitude.
# PSS/E measures a flow at the winding-1 (from) bus, and `regulated_bus_number` does not apply.
function _controlled_bus(
    ::BusVoltage, circuit::PSY.TransformerCircuit, from_bus::Int, to_bus::Int,
)
    reg = PSY.get_regulated_bus_number(circuit)
    if iszero(reg)
        return to_bus
    end
    return abs(reg)
end
_controlled_bus(
    ::Union{ReactivePowerFlow, ActivePowerFlow},
    ::PSY.TransformerCircuit, from_bus::Int, to_bus::Int,
) = from_bus

"""Control metadata for one regulating `PSY.TransformerCircuit`, of either arity.
The actuator band is `tap_ratio_limits` (on the same basis as `PSY.get_tap`) or
`phase_angle_limits` (radians). The target band is the VMI/VMA band of the regulated quantity.
The device is held anywhere inside the target band and regulates toward its midpoint on an
excursion — the same posture as a switched shunt's VSWLO/VSWHI."""
function _circuit_metadata(
    ::Type{ControlledBranch{A, Q}},
    circuit::PSY.TransformerCircuit,
    label::String,
    from_bus::Int,
    to_bus::Int,
) where {A, Q}
    lims = _actuator_limits(A(), circuit, label)
    band = _target_band(Q(), circuit, label)
    return (
        cbus = _controlled_bus(Q(), circuit, from_bus, to_bus),
        pmin = lims.min,
        pmax = lims.max,
        ntp = PSY.get_number_of_tap_positions(circuit),
        setpoint = (band.min + band.max) / 2,
        band_lo = band.min,
        band_hi = band.max,
    )
end

function _validate_circuit(
    T::Type{ControlledBranch{A, Q}}, name::String, md,
) where {A, Q}
    return _validate_tap(name, md.pmin, md.pmax, md.ntp, A()) &&
           _validate_target(Q(), _control_label(T), name, md)
end

_validate_target(::BusVoltage, label::String, name::String, md) =
    _validate_vset(label, name, md.setpoint)
_validate_target(
    ::Union{ReactivePowerFlow, ActivePowerFlow}, label::String, name::String, md,
) = _validate_band(label, name, md.band_lo, md.band_hi)

# The actuator value at enrollment and the part of the complex tap that it leaves fixed.
_actuator_state(::TapRatio, adm) = (value = adm.tap, fixed = adm.shift)
_actuator_state(::PhaseAngle, adm) = (value = adm.shift, fixed = adm.tap)

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
    groups = (
        taps = ControlledTap[],
        flow_taps = ControlledBranch{TapRatio, ReactivePowerFlow}[],
        phase_shifters = ControlledBranch{PhaseAngle, ActivePowerFlow}[],
    )
    for candidate in _controlled_circuit_candidates(sys)
        T = _circuit_control(PSY.get_control_objective(candidate[3]))
        _enroll_circuit!(
            _device_group(T, groups), T, candidate, bus_lookup,
            reverse_bus_search_map, ybus, nrd,
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
        lims = _required_band(
            PSY.get_voltage_limits(sa), "SwitchedAdmittance $name", :voltage_limits)
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

    return ControlledDeviceSet(
        groups.taps, groups.flow_taps, groups.phase_shifters, shunts, facts, n_time_steps,
    )
end

# Enroll one regulating circuit into its device group. Any data problem de-enrolls the circuit
# with a `@warn`; it stays at its current setting.
function _enroll_circuit!(
    devices::Vector{ControlledBranch{A, Q}},
    T::Type{ControlledBranch{A, Q}},
    candidate::Tuple{String, PSY.ACTransmission, PSY.TransformerCircuit, String, Int},
    bus_lookup::Dict{Int, Int},
    reverse_bus_search_map::Dict{Int, Int},
    ybus,
    nrd,
) where {A, Q}
    name, branch, circuit, device_name, circuit_index = candidate
    label = _control_label(T)
    kind = _actuator_kind(A())
    arc = PSY.get_arc(circuit)
    fb = PSY.get_number(PSY.get_from(arc))
    tb = PSY.get_number(PSY.get_to(arc))
    md = _circuit_metadata(T, circuit, "$label $name", fb, tb)
    fix = _resolve_bus_ix(bus_lookup, reverse_bus_search_map, fb)
    tix = _resolve_bus_ix(bus_lookup, reverse_bus_search_map, tb)
    cix = _resolve_bus_ix(bus_lookup, reverse_bus_search_map, md.cbus)
    if isnothing(fix) || isnothing(tix)
        missing_bus = tb
        if isnothing(fix)
            missing_bus = fb
        end
        @warn "$label \"$name\": terminal bus $missing_bus \
            is not in the (reduced) network; leaving the circuit locked."
        return
    end
    if isnothing(cix)
        @warn "$label \"$name\": controlled bus $(md.cbus) is not in the \
            (reduced) network; leaving the circuit locked."
        return
    end
    if fix == tix
        @warn "$label \"$name\": arc collapsed by a network reduction \
            (from == to after bus merging); leaving the circuit locked."
        return
    end
    _validate_circuit(T, name, md) || return
    # PNM owns the π-model, including the r == x == 0 floor a hand-built `1/(r + jx)` would
    # miss, and applies the same impedance-correction factor the assembled Ybus was stamped
    # with.
    adm = PNM.branch_admittance(branch, nrd)
    start = _actuator_state(A(), adm)
    if !(md.pmin - BOUNDS_TOLERANCE <= start.value <= md.pmax + BOUNDS_TOLERANCE)
        @warn "$label \"$name\": initial $kind value $(start.value) lies \
            outside the $kind band [$(md.pmin), $(md.pmax)]; leaving the circuit \
            locked at its current setting."
        return
    end
    push!(
        devices,
        ControlledBranch{A, Q}(
            name,
            fix,
            tix,
            cix,
            md.setpoint,
            md.band_lo,
            md.band_hi,
            complex(adm.g, adm.b),
            complex(adm.g_fr, adm.b_fr),
            start.fixed,  # PNM stamps t = ratio·e^{iα}
            md.pmin,
            md.pmax,
            collect(range(md.pmin, md.pmax; length = md.ntp)),
            _ybus_block_offsets(ybus, fix, tix),
            start.value,   # initial (reporting)
            start.value,   # synced (arc-admittance rows reflect this value)
            start.value,   # current
            device_name,
            circuit_index,
        ),
    )
    return
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
        rating = PSY.get_max_shunt_current(fd, u"SU")
        q_cap = PSY.get_max_reactive_power(fd, u"SU")
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
