"""Validate shunt invariants at construction time. Separated to allow direct unit testing."""
function _validate_shunt!(
    name::String,
    b_min::Float64,
    b0::Float64,
    b_max::Float64,
    steps::Vector{Int},
    dB::Vector{Float64},
)
    if !(b_min <= b0 <= b_max)
        error(
            "ControlledSwitchedShunt \"$name\": b0=$b0 is outside [b_min=$b_min, b_max=$b_max]",
        )
    end
    if b_min == b_max
        error(
            "ControlledSwitchedShunt \"$name\": no controllable susceptance range (b_min == b_max == $b_min); not a valid voltage-controlled device",
        )
    end
    for k in eachindex(steps, dB)
        if steps[k] == 0 && dB[k] != 0.0
            error(
                "ControlledSwitchedShunt \"$name\": block $k has zero steps but nonzero dB=$(dB[k]) — malformed metadata",
            )
        end
    end
    return nothing
end

"""Validate tap invariants at construction time. Separated to allow direct unit testing."""
function _validate_tap!(
    name::String,
    p_min::Float64,
    p_max::Float64,
    ntp::Int,
)
    if p_min > p_max
        error(
            "ControlledTap \"$name\": p_min=$p_min exceeds p_max=$p_max — malformed tap-ratio limits",
        )
    end
    if p_min == p_max
        error(
            "ControlledTap \"$name\": no controllable tap-ratio range (p_min == p_max == $p_min); not a valid voltage-controlled device",
        )
    end
    if ntp < 2
        error(
            "ControlledTap \"$name\": needs at least 2 tap positions (got ntp=$ntp)",
        )
    end
    return nothing
end

function _controlled_bus_number(ext::Dict, fallback::Int)
    for k in ("NREG", "SWREM", "RMIDNT")
        if haskey(ext, k)
            v = ext[k]
            n = v isa Integer ? Int(v) : tryparse(Int, string(v))
            (n !== nothing && n != 0) && return n
        end
    end
    return fallback
end

function _ext_float(ext::Dict, key::String, default::Float64)
    haskey(ext, key) || return default
    v = ext[key]
    return v isa Real ? Float64(v) : something(tryparse(Float64, string(v)), default)
end

function _ext_int(ext::Dict, key::String, default::Int)
    haskey(ext, key) || return default
    v = ext[key]
    return v isa Integer ? Int(v) : something(tryparse(Int, string(v)), default)
end

"""Build the type-stable device set from a `PSY.System`.

`bus_lookup` maps PSY bus number → network index; `ybus` is the assembled
`AC_Ybus_Matrix` from `data.power_network_matrix`."""
function build_controlled_device_set(
    sys,
    bus_lookup::Dict{Int, Int},
    ybus,
)
    taps = ControlledTap[]
    for tx in PSY.get_available_components(PSY.TapTransformer, sys)
        PSY.get_control_objective(tx) == PSY.TransformerControlObjective.VOLTAGE ||
            continue
        arc = PSY.get_arc(tx)
        fb = PSY.get_number(PSY.get_from(arc))
        tb = PSY.get_number(PSY.get_to(arc))
        ext = PSY.get_ext(tx)
        # Regulated bus: the first-class field wins; `0` (the documented "local" sentinel) defers
        # to the legacy PSS/e ext keys (NREG/SWREM/RMIDNT), which fall back to the to-bus.
        reg = PSY.get_regulated_bus_number(tx)
        cbus = reg != 0 ? reg : _controlled_bus_number(ext, tb)
        haskey(bus_lookup, cbus) ||
            error("ControlledTap $(PSY.get_name(tx)): controlled bus $cbus not in network")
        fix = bus_lookup[fb]
        tix = bus_lookup[tb]
        cix = bus_lookup[cbus]
        # First-class controllability fields (PSY #1684). Legacy `ext` scrapes still override when
        # present, so externally parsed systems keep working until their parsers populate the fields.
        lims = PSY.get_tap_limits(tx)
        pmin = haskey(ext, "RMI") ? _ext_float(ext, "RMI", lims.min) : lims.min
        pmax = haskey(ext, "RMA") ? _ext_float(ext, "RMA", lims.max) : lims.max
        ntp = if haskey(ext, "NTP")
            Int(_ext_float(ext, "NTP", Float64(DEFAULT_TAP_POSITIONS)))
        else
            PSY.get_number_of_tap_positions(tx)
        end
        ntp < 2 && (ntp = DEFAULT_TAP_POSITIONS)
        _validate_tap!(PSY.get_name(tx), pmin, pmax, ntp)
        vset = if haskey(ext, "VSET")
            _ext_float(ext, "VSET", DEFAULT_TAP_VSET)
        else
            PSY.get_voltage_setpoint(tx)
        end
        yt = 1.0 / (PSY.get_r(tx) + PSY.get_x(tx) * im)
        ysh = PSY.get_primary_shunt(tx)
        push!(
            taps,
            ControlledTap(
                PSY.get_name(tx),
                fix,
                tix,
                cix,
                cix == fix,
                vset,
                yt,
                ysh,
                0.0,
                pmin,
                pmax,
                collect(range(pmin, pmax; length = ntp)),
                _ybus_block_offsets(ybus, fix, tix),
                PSY.get_tap(tx),
            ),
        )
    end

    shunts = ControlledSwitchedShunt[]
    for sa in PSY.get_available_components(PSY.SwitchedAdmittance, sys)
        bus = PSY.get_number(PSY.get_bus(sa))
        haskey(bus_lookup, bus) || continue
        ext = PSY.get_ext(sa)
        modsw = _ext_int(ext, "MODSW", DEFAULT_SHUNT_MODSW)
        if modsw == 0
            @debug "ControlledSwitchedShunt $(PSY.get_name(sa)): MODSW=0 (locked); \
                treated as fixed admittance, not enrolled."
            continue
        elseif modsw == 1
            continuous = false
        elseif modsw == 2
            continuous = true
        else
            error(
                "ControlledSwitchedShunt $(PSY.get_name(sa)): MODSW=$modsw \
                (remote reactive-power / remote-device control) is not supported. \
                Only voltage-control modes are implemented: MODSW=1 (discrete) \
                and MODSW=2 (continuous).",
            )
        end
        cbus = _controlled_bus_number(ext, bus)
        haskey(bus_lookup, cbus) ||
            error(
                "ControlledSwitchedShunt $(PSY.get_name(sa)): bus $cbus not in network",
            )
        lims = PSY.get_admittance_limits(sa)
        vset = (lims.min + lims.max) / 2.0
        Y0 = PSY.get_Y(sa)
        steps = PSY.get_number_of_steps(sa)
        dY = PSY.get_Y_increase(sa)
        dB = imag.(dY)
        b0 = imag(Y0)
        g0 = real(Y0)
        bmax = b0 + sum(max.(steps .* dB, 0.0); init = 0.0)
        bmin = b0 + sum(min.(steps .* dB, 0.0); init = 0.0)
        _validate_shunt!(PSY.get_name(sa), bmin, b0, bmax, steps, dB)
        init_status = PSY.get_initial_status(sa)
        current_b = imag(Y0 + sum(init_status .* dY; init = 0.0 + 0.0im))
        bo = sortperm(dB; rev = true)
        bn = zeros(Int, length(dB))
        push!(
            shunts,
            ControlledSwitchedShunt(
                PSY.get_name(sa),
                bus_lookup[bus],
                bus_lookup[cbus],
                vset,
                lims.min,
                lims.max,
                g0,
                b0,
                steps,
                dB,
                bmin,
                bmax,
                bo,
                bn,
                continuous,
                current_b,
            ),
        )
    end

    facts = ControlledFACTS[]
    base_mva = PSY.get_base_power(sys)
    for fd in PSY.get_available_components(PSY.FACTSControlDevice, sys)
        mode = PSY.get_control_mode(fd)
        # OOS or no control mode ⇒ not a voltage-controlling shunt; not enrolled
        # (matches PSY's own `_facts_is_active` semantics and the MODSW=0 shunt skip).
        if mode === nothing || mode == PSY.FACTSOperationModes.OOS
            @debug "ControlledFACTS $(PSY.get_name(fd)): control_mode=$(mode) is not \
                voltage-controlling; not enrolled."
            continue
        end
        bus = PSY.get_number(PSY.get_bus(fd))
        haskey(bus_lookup, bus) ||
            error("ControlledFACTS $(PSY.get_name(fd)): bus $bus not in network")
        # `max_shunt_current` is MVA at unity voltage ⇒ a symmetric susceptance band in p.u.
        # on system base (Q = b·|V|², so at |V|=1 the MVA rating is the p.u. susceptance bound).
        b_max = PSY.get_max_shunt_current(fd) / base_mva
        b_max > 0.0 ||
            error("ControlledFACTS $(PSY.get_name(fd)): max_shunt_current must be positive")
        push!(
            facts,
            ControlledFACTS(
                PSY.get_name(fd),
                bus_lookup[bus],
                bus_lookup[bus],   # shunt regulates its own (sending) bus
                PSY.get_voltage_setpoint(fd),
                -b_max,
                b_max,
                0.0,               # start neutral; the controller drives b from 0
            ),
        )
    end

    phase_shifters = ControlledPhaseShifter[]
    for ps in PSY.get_available_components(PSY.PhaseShiftingTransformer, sys)
        PSY.get_control_objective(ps) ==
        PSY.TransformerControlObjective.ACTIVE_POWER_FLOW || continue
        arc = PSY.get_arc(ps)
        fb = PSY.get_number(PSY.get_from(arc))
        tb = PSY.get_number(PSY.get_to(arc))
        (haskey(bus_lookup, fb) && haskey(bus_lookup, tb)) ||
            error("ControlledPhaseShifter $(PSY.get_name(ps)): arc bus not in network")
        fix = bus_lookup[fb]
        tix = bus_lookup[tb]
        lims = PSY.get_phase_angle_limits(ps)
        yt = 1.0 / (PSY.get_r(ps) + PSY.get_x(ps) * im)
        push!(
            phase_shifters,
            ControlledPhaseShifter(
                PSY.get_name(ps),
                fix,
                tix,
                PSY.get_active_power_flow(ps),   # flow setpoint (from→to)
                yt,
                PSY.get_tap(ps),
                lims.min,
                lims.max,
                _ybus_block_offsets(ybus, fix, tix),
                PSY.get_α(ps),                   # current phase angle
            ),
        )
    end

    return ControlledDeviceSet(taps, shunts, facts, phase_shifters)
end
