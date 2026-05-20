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

function _controlled_bus_number(ext::Dict, fallback::Int)
    for k in ("NREG", "RMIDNT")
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
        cbus = _controlled_bus_number(ext, tb)
        haskey(bus_lookup, cbus) ||
            error("ControlledTap $(PSY.get_name(tx)): controlled bus $cbus not in network")
        fix = bus_lookup[fb]
        tix = bus_lookup[tb]
        cix = bus_lookup[cbus]
        pmin = _ext_float(ext, "RMI", DEFAULT_TAP_RATIO_MIN)
        pmax = _ext_float(ext, "RMA", DEFAULT_TAP_RATIO_MAX)
        ntp = Int(_ext_float(ext, "NTP", Float64(DEFAULT_TAP_POSITIONS)))
        ntp < 2 && (ntp = DEFAULT_TAP_POSITIONS)
        vset = if haskey(ext, "VSET")
            _ext_float(ext, "VSET", DEFAULT_TAP_VSET)
        else
            controlled_bus = if cbus == fb
                PSY.get_from(arc)
            elseif cbus == tb
                PSY.get_to(arc)
            else
                PSY.get_bus(sys, cbus)
            end
            m = PSY.get_magnitude(controlled_bus)
            m > 0.0 ? m : DEFAULT_TAP_VSET
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
                g0,
                b0,
                steps,
                dB,
                bmin,
                bmax,
                bo,
                bn,
                current_b,
            ),
        )
    end

    return ControlledDeviceSet(taps, shunts)
end
