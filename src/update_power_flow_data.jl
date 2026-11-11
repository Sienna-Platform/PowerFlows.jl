function _check_net_injection_length(
    data::PowerFlowData,
    net_injection::AbstractVector{Float64},
    label::String = "net_injection",
)
    n_time_steps = get_time_steps(data)
    if length(net_injection) != n_time_steps
        throw(
            DimensionMismatch(
                "$(label) has $(length(net_injection)) entries, " *
                "but the data has $(n_time_steps) time steps.",
            ),
        )
    end
    return
end

function _check_load_parts_length(
    data::PowerFlowData,
    parts::NTuple{3, AbstractVector{Float64}},
)
    _check_net_injection_length(data, parts[1], "constant_power")
    _check_net_injection_length(data, parts[2], "constant_current")
    _check_net_injection_length(data, parts[3], "constant_impedance")
    return
end

# One bus rule for every writer, the same rule as the System build. A bus that the network
# reduction removed is skipped. A bus that is not in the power flow and not removed is an error.
function _with_bus_ix(f, data::PowerFlowData, component::PSY.StaticInjection)
    nrd = get_network_reduction_data(data)
    bus_number = PSY.get_number(PSY.get_bus(component))
    if bus_number in PNM.get_removed_buses(nrd)
        return
    end
    key = get(PNM.get_reverse_bus_search_map(nrd), bus_number, bus_number)
    bus_lookup = get_bus_lookup(data)
    if !haskey(bus_lookup, key)
        error(
            "$(typeof(component)) $(PSY.get_name(component)) is at bus $(bus_number), " *
            "which is not in the power flow.",
        )
    end
    f(bus_lookup[key])
    return
end

function _add_load_parts!(
    ::PowerFlowData{<:AbstractACPowerFlow},
    withdrawals::NTuple{3, Matrix{Float64}},
    bus_ix::Int,
    parts::NTuple{3, AbstractVector{Float64}},
)
    for k in 1:3
        @views withdrawals[k][bus_ix, :] .-= parts[k]
    end
    return
end

function _add_load_parts!(
    ::PowerFlowData{<:AbstractDCPowerFlow},
    withdrawals::NTuple{3, Matrix{Float64}},
    bus_ix::Int,
    parts::NTuple{3, AbstractVector{Float64}},
)
    @views withdrawals[1][bus_ix, :] .-= parts[1] .+ parts[2] .+ parts[3]
    return
end

_active_withdrawal_arrays(data::PowerFlowData) = (
    get_bus_active_power_withdrawals(data),
    get_bus_active_power_constant_current_withdrawals(data),
    get_bus_active_power_constant_impedance_withdrawals(data),
)

_reactive_withdrawal_arrays(data::PowerFlowData) = (
    get_bus_reactive_power_withdrawals(data),
    get_bus_reactive_power_constant_current_withdrawals(data),
    get_bus_reactive_power_constant_impedance_withdrawals(data),
)

function _set_off_area_load_power!(
    ::PowerFlowData,
    ::PSY.ElectricLoad,
    ::NTuple{3, AbstractVector{Float64}},
)
    return
end

function _set_off_area_load_power!(
    data::ACPowerFlowData,
    load::PSY.ElectricLoad,
    parts::NTuple{3, AbstractVector{Float64}},
)
    name = PSY.get_name(load)
    if has_off_area_load(data, name)
        for t in eachindex(parts[1])
            set_off_area_load_power!(
                data,
                name,
                t,
                -parts[1][t],
                -parts[2][t],
                -parts[3][t],
            )
        end
    end
    return
end

function _set_converter_p_set!(
    dcn::DCNetwork,
    converter_ix::Int,
    p::AbstractVector{Float64},
)
    if !uses_vdc_setpoint(dcn.converter_mode[converter_ix])
        @views dcn.p_set[converter_ix, :] .= p
    end
    return
end

"""
    add_active_power!(data::PowerFlowData, component::PSY.StaticInjection,
        net_injection::AbstractVector{Float64})

Add the active power of one component to `data` for every time step.

`net_injection[t]` is the power that `component` injects at time step `t`. The unit is p.u. on
the system base. A withdrawal is negative. The function throws `DimensionMismatch` when the
length of `net_injection` is not the number of time steps of `data`. The function adds nothing
when the network reduction removed the bus of the component. It throws an error when the bus is
neither in the power flow nor removed. Otherwise it adds `net_injection[t]` to
`bus_active_power_injections`. It adds to the array. Call `clear_injection_data!` before the
first call.

A `PSY.ElectricLoad` needs the three-array method. The one-array method throws an error for it.
A `PSY.InterconnectingConverter` has its own method. A `PSY.SynchronousCondenser` loses power.
Use [`add_active_power_losses!`](@ref) for the losses.
"""
function add_active_power!(
    data::PowerFlowData,
    component::PSY.StaticInjection,
    net_injection::AbstractVector{Float64},
)
    _check_net_injection_length(data, net_injection)
    _with_bus_ix(data, component) do bus_ix
        @views get_bus_active_power_injections(data)[bus_ix, :] .+= net_injection
    end
    return
end

function add_active_power!(
    ::PowerFlowData,
    load::PSY.ElectricLoad,
    ::AbstractVector{Float64},
)
    error(
        "$(typeof(load)) $(PSY.get_name(load)) is a load. Call add_active_power! with " *
        "the constant power, constant current and constant impedance arrays.",
    )
end

"""
    add_active_power!(data::PowerFlowData, load::PSY.ElectricLoad,
        constant_power::AbstractVector{Float64},
        constant_current::AbstractVector{Float64},
        constant_impedance::AbstractVector{Float64})

Add the active power of one load to `data` for every time step.

Each array holds the signed injection of one ZIP term at time step `t`. The unit is p.u. on the
system base, at 1 p.u. voltage. A withdrawal is negative. The function throws `DimensionMismatch`
when an array does not have one entry for each time step of `data`. The function adds nothing
when the network reduction removed the bus of the load. It throws an error when the bus is
neither in the power flow nor removed.

For an AC power flow, the function subtracts each array from the matching array:
`bus_active_power_withdrawals`, `bus_active_power_constant_current_withdrawals` and
`bus_active_power_constant_impedance_withdrawals`. For a DC power flow, the function subtracts the
sum of the three arrays from `bus_active_power_withdrawals`. For an AC power flow, the function
also sets the off-area record of a load that has one. It does not add to the record.

The function adds to the arrays. Call `clear_injection_data!` before the first call. Call the
function one time for each load between two calls of `clear_injection_data!`.
"""
function add_active_power!(
    data::PowerFlowData,
    load::PSY.ElectricLoad,
    constant_power::AbstractVector{Float64},
    constant_current::AbstractVector{Float64},
    constant_impedance::AbstractVector{Float64},
)
    parts = (constant_power, constant_current, constant_impedance)
    _check_load_parts_length(data, parts)
    _with_bus_ix(data, load) do bus_ix
        _add_load_parts!(data, _active_withdrawal_arrays(data), bus_ix, parts)
        _set_off_area_load_power!(data, load, parts)
    end
    return
end

"""
    add_active_power!(data::PowerFlowData, ic::PSY.InterconnectingConverter,
        p::AbstractVector{Float64})

Add the active power of one converter to `data` for every time step.

`p[t]` is the power that `ic` injects at its AC bus at time step `t`. The unit is p.u. on the
system base. The function throws `DimensionMismatch` when the length of `p` is not the number of
time steps of `data`. The function adds nothing when the network reduction removed the bus of
`ic`. It throws an error when the bus is neither in the power flow nor removed.

When `data` has a DC network that holds `ic`, the DC network sets the injection of `ic`. For a
converter in a power control mode (`ControlPQ` or `ControlPVac`), the function assigns `p[t]` to
the power setpoint `p_set` of the converter. A converter in a DC voltage mode or a droop mode
takes its power from the DC network, so the function writes nothing for it.

Without a DC network, or when the DC network does not hold `ic`, the function adds `p[t]` to
`bus_active_power_injections`. It does this only when the System build also counts the converter
as an injection.
"""
function add_active_power!(
    data::PowerFlowData,
    ic::PSY.InterconnectingConverter,
    p::AbstractVector{Float64},
)
    _check_net_injection_length(data, p)
    _with_bus_ix(data, ic) do bus_ix
        dcn = get_dc_network(data)
        name = PSY.get_name(ic)
        if name in dcn.converter_name
            _set_converter_p_set!(dcn, only(findall(==(name), dcn.converter_name)), p)
        elseif _contributes_active_power(get_pf(data), ic)
            @views get_bus_active_power_injections(data)[bus_ix, :] .+= p
        end
    end
    return
end

"""
    add_reactive_power!(data::PowerFlowData, component::PSY.StaticInjection,
        net_injection::AbstractVector{Float64})

Add the reactive power of one component to `data` for every time step.

`net_injection[t]` is the power that `component` injects at time step `t`. The unit is p.u. on
the system base. A withdrawal is negative. The function throws `DimensionMismatch` when the
length of `net_injection` is not the number of time steps of `data`. The function does nothing
when the power flow does not consider reactive power, or when the network reduction removed the
bus of the component. It throws an error when the bus is neither in the power flow nor removed.
Otherwise it adds `net_injection[t]` to `bus_reactive_power_injections`.

The function adds to the array. Call `clear_injection_data!` before the first call.

A `PSY.ElectricLoad` needs the three-array method. The one-array method throws an error for it.
A `PSY.InterconnectingConverter` has its own method.
"""
function add_reactive_power!(
    data::PowerFlowData,
    component::PSY.StaticInjection,
    net_injection::AbstractVector{Float64},
)
    _check_net_injection_length(data, net_injection)
    if considers_reactive_power(get_pf(data))
        _with_bus_ix(data, component) do bus_ix
            @views get_bus_reactive_power_injections(data)[bus_ix, :] .+= net_injection
        end
    end
    return
end

function add_reactive_power!(
    ::PowerFlowData,
    load::PSY.ElectricLoad,
    ::AbstractVector{Float64},
)
    error(
        "$(typeof(load)) $(PSY.get_name(load)) is a load. Call add_reactive_power! with " *
        "the constant power, constant current and constant impedance arrays.",
    )
end

"""
    add_reactive_power!(data::PowerFlowData, load::PSY.ElectricLoad,
        constant_power::AbstractVector{Float64},
        constant_current::AbstractVector{Float64},
        constant_impedance::AbstractVector{Float64})

Add the reactive power of one load to `data` for every time step.

The arrays have the meaning that [`add_active_power!`](@ref) gives them for a load. The function
does nothing when the power flow does not consider reactive power. The bus rule of
[`add_active_power!`](@ref) applies. Otherwise it subtracts each
array from `bus_reactive_power_withdrawals`, `bus_reactive_power_constant_current_withdrawals`
and `bus_reactive_power_constant_impedance_withdrawals`.
"""
function add_reactive_power!(
    data::PowerFlowData,
    load::PSY.ElectricLoad,
    constant_power::AbstractVector{Float64},
    constant_current::AbstractVector{Float64},
    constant_impedance::AbstractVector{Float64},
)
    parts = (constant_power, constant_current, constant_impedance)
    _check_load_parts_length(data, parts)
    if considers_reactive_power(get_pf(data))
        _with_bus_ix(data, load) do bus_ix
            _add_load_parts!(data, _reactive_withdrawal_arrays(data), bus_ix, parts)
        end
    end
    return
end

"""
    add_reactive_power!(data::PowerFlowData, ic::PSY.InterconnectingConverter,
        q::AbstractVector{Float64})

Check the length of `q` and add nothing. The System build does not count the reactive power of a
converter as an injection. A lowered DC network sets it.
"""
function add_reactive_power!(
    data::PowerFlowData,
    ::PSY.InterconnectingConverter,
    q::AbstractVector{Float64},
)
    _check_net_injection_length(data, q)
    return
end

"""
    add_reactive_power!(data::PowerFlowData, fd::PSY.FACTSControlDevice,
        q::AbstractVector{Float64})

Check the length of `q` and add nothing. The System build does not count the reactive power of a
FACTS device as an injection. The FACTS device holds its voltage setpoint, and the power flow
solves its reactive power.
"""
function add_reactive_power!(
    data::PowerFlowData,
    ::PSY.FACTSControlDevice,
    q::AbstractVector{Float64},
)
    _check_net_injection_length(data, q)
    return
end

_switched_admittance_active_withdrawals(data::PowerFlowData{<:AbstractACPowerFlow}) =
    get_bus_active_power_constant_impedance_withdrawals(data)

# `handle_zip_loads!` moves the constant impedance part of DC data into constant power.
_switched_admittance_active_withdrawals(data::PowerFlowData{<:AbstractDCPowerFlow}) =
    get_bus_active_power_withdrawals(data)

"""
    add_switched_admittance!(data::PowerFlowData, sa::PSY.SwitchedAdmittance)

Add the admittance of one switched admittance to `data` for every time step.

The admittance is the one that the `System` holds for `sa`. The function treats it as a constant
impedance load. It adds the real part of the admittance to
`bus_active_power_constant_impedance_withdrawals`. It subtracts the imaginary part from
`bus_reactive_power_constant_impedance_withdrawals`. The function adds to the arrays. For DC data,
the function adds the real part to `bus_active_power_withdrawals` instead.

The bus rule of [`add_active_power!`](@ref) applies. When the data holds a shunt controller for
`sa`, the function also sets the controller to the susceptance of the `System` at every time
step, so the controller agrees with the arrays.

`clear_injection_data!` sets these arrays to zero. After `clear_injection_data!`, call this function
again for every switched admittance.
"""
function add_switched_admittance!(data::PowerFlowData, sa::PSY.SwitchedAdmittance)
    _with_bus_ix(data, sa) do bus_ix
        (p, q) = _switched_admittance_withdrawal(sa)
        @views _switched_admittance_active_withdrawals(data)[bus_ix, :] .+= p
        @views get_bus_reactive_power_constant_impedance_withdrawals(data)[bus_ix, :] .+= q
        _seed_shunt_controller!(
            get_controlled_devices(data),
            sa,
            fill(-q, get_time_steps(data)),
        )
    end
    return
end

_seed_shunt_controller!(::Nothing, ::PSY.SwitchedAdmittance, ::AbstractVector{Float64}) =
    nothing

# The controller holds the seeded value: `initial` equals `current`, so the deadband hold keeps it.
function _seed_shunt_controller!(
    set::ControlledDeviceSet,
    sa::PSY.SwitchedAdmittance,
    b::AbstractVector{Float64},
)
    name = PSY.get_name(sa)
    for (i, d) in enumerate(set.shunts)
        d.name == name || continue
        set.shunt_current[i, :] .= b
        set.shunt_initial[i, :] .= b
        for t in eachindex(b)
            fill!(d.block_n, 0)
            snap_to_discrete(d, b[t])
            set.shunt_block_n[i, t] .= d.block_n
        end
    end
    return
end

"""
    add_switched_admittance!(data::PowerFlowData, sa::PSY.SwitchedAdmittance,
        b::AbstractVector{Float64})

Add one switched admittance to `data` with the susceptance `b[t]` at time step `t`.

`b[t]` is the susceptance in p.u. on the system base. The conductance comes from the `System`.
The function treats the admittance as a constant impedance load. It adds the conductance to
`bus_active_power_constant_impedance_withdrawals` and `-b[t]` to
`bus_reactive_power_constant_impedance_withdrawals`. For DC data, the function adds the conductance
to `bus_active_power_withdrawals` instead.

When the data holds a shunt controller for `sa`, the controller starts from `b[t]` at time step
`t`. The controller holds `b[t]` inside its deadband. Its block counts are the counts of the
nearest realizable susceptance. The controller does not start from the susceptance of the
`System`.

The function throws `DimensionMismatch` when the length of `b` is not the number of time steps of
`data`. The bus rule of [`add_active_power!`](@ref) applies. The function adds to the arrays.
Call `clear_injection_data!` before the first call.
"""
function add_switched_admittance!(
    data::PowerFlowData,
    sa::PSY.SwitchedAdmittance,
    b::AbstractVector{Float64},
)
    _check_net_injection_length(data, b, "b")
    _with_bus_ix(data, sa) do bus_ix
        (p, _) = _switched_admittance_withdrawal(sa)
        @views _switched_admittance_active_withdrawals(data)[bus_ix, :] .+= p
        @views get_bus_reactive_power_constant_impedance_withdrawals(data)[bus_ix, :] .-= b
        _seed_shunt_controller!(get_controlled_devices(data), sa, b)
    end
    return
end

"""
    add_active_power_losses!(data::PowerFlowData, sc::PSY.SynchronousCondenser)

Add the active power losses of one synchronous condenser to `data` for every time step.

The function adds `PSY.get_active_power_losses(sc, u"SU")` to `bus_active_power_withdrawals`.
The bus rule of [`add_active_power!`](@ref) applies. The function adds to the array.
`clear_injection_data!` sets the array to zero. After `clear_injection_data!`, call this function
again for every condenser.
"""
function add_active_power_losses!(data::PowerFlowData, sc::PSY.SynchronousCondenser)
    _with_bus_ix(data, sc) do bus_ix
        @views get_bus_active_power_withdrawals(data)[bus_ix, :] .+=
            PSY.get_active_power_losses(sc, u"SU")
    end
    return
end

_exponential_slot(
    ::PowerFlowData{<:AbstractACPowerFlow},
    load::PSY.ExponentialLoad,
    e::Float64,
) =
    _exponential_zip_slot(load, e)
# DC data reads only constant power, so every exponent is valid.
_exponential_slot(
    ::PowerFlowData{<:AbstractDCPowerFlow},
    ::PSY.ExponentialLoad,
    ::Float64,
) = 1

function _exponential_load_parts(
    data::PowerFlowData,
    load::PSY.ExponentialLoad,
    exponent::Float64,
    values::AbstractVector{Float64},
)
    _check_net_injection_length(data, values, "values")
    slot = _exponential_slot(data, load, exponent)
    parts = ntuple(_ -> zeros(length(values)), 3)
    parts[slot] .= values
    return (
        constant_power = parts[1],
        constant_current = parts[2],
        constant_impedance = parts[3],
    )
end

"""
    exponential_load_active_parts(data::PowerFlowData, load::PSY.ExponentialLoad,
        values::AbstractVector{Float64})

Split the active power of an `ExponentialLoad` into the three ZIP parts for every time step.

`values[t]` is the active power at time step `t`. The function does not change its sign or unit.
It returns the NamedTuple `(constant_power, constant_current, constant_impedance)`. Each field is
a `Vector{Float64}`. Pass the fields to the three-array method of [`add_active_power!`](@ref).

For AC data, the exponent α of `load` selects the part that holds `values`: 0 gives the constant
power part, 1 the constant current part and 2 the constant impedance part. The other parts are
zero. The function throws an error for any other α. For DC data, the function puts `values` in
the constant power part for every α. The function throws `DimensionMismatch` when the length of
`values` is not the number of time steps of `data`.
"""
function exponential_load_active_parts(
    data::PowerFlowData,
    load::PSY.ExponentialLoad,
    values::AbstractVector{Float64},
)
    return _exponential_load_parts(data, load, PSY.get_α(load), values)
end

"""
    exponential_load_reactive_parts(data::PowerFlowData, load::PSY.ExponentialLoad,
        values::AbstractVector{Float64})

Split the reactive power of an `ExponentialLoad` into the three ZIP parts for every time step.

The function works as [`exponential_load_active_parts`](@ref). For AC data, the exponent β of
`load` selects the part. Pass the fields to the three-array method of
[`add_reactive_power!`](@ref).
"""
function exponential_load_reactive_parts(
    data::PowerFlowData,
    load::PSY.ExponentialLoad,
    values::AbstractVector{Float64},
)
    return _exponential_load_parts(data, load, PSY.get_β(load), values)
end
