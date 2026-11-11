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

function _bus_in_power_flow(data::PowerFlowData, component::PSY.StaticInjection)
    reverse_bus_search_map =
        PNM.get_reverse_bus_search_map(get_network_reduction_data(data))
    bus_number = PSY.get_number(PSY.get_bus(component))
    return haskey(get_bus_lookup(data), get(reverse_bus_search_map, bus_number, bus_number))
end

function _component_bus_ix(data::PowerFlowData, component::PSY.StaticInjection)
    bus_number = PSY.get_number(PSY.get_bus(component))
    if !_bus_in_power_flow(data, component)
        error(
            "$(typeof(component)) $(PSY.get_name(component)) is at bus $(bus_number), " *
            "which is not in the power flow.",
        )
    end
    reverse_bus_search_map =
        PNM.get_reverse_bus_search_map(get_network_reduction_data(data))
    return _get_bus_ix(get_bus_lookup(data), reverse_bus_search_map, bus_number)
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
length of `net_injection` is not the number of time steps of `data`. The function throws an error
when the bus of the component is not in the power flow.
The function adds `net_injection[t]` to `bus_active_power_injections`.
It adds to the array. Call `clear_injection_data!` before the first call.

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
    bus_ix = _component_bus_ix(data, component)
    @views get_bus_active_power_injections(data)[bus_ix, :] .+= net_injection
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
when an array does not have one entry for each time step of `data`. The function throws an error
when the bus of the load is not in the power flow.

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
    bus_ix = _component_bus_ix(data, load)
    _add_load_parts!(data, _active_withdrawal_arrays(data), bus_ix, parts)
    _set_off_area_load_power!(data, load, parts)
    return
end

"""
    add_active_power!(data::PowerFlowData, ic::PSY.InterconnectingConverter,
        p::AbstractVector{Float64})

Add the active power of one converter to `data` for every time step.

`p[t]` is the power that `ic` injects at its AC bus at time step `t`. The unit is p.u. on the
system base. The function throws `DimensionMismatch` when the length of `p` is not the number of
time steps of `data`. The function throws an error when the bus of `ic` is not in the power flow.

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
    bus_ix = _component_bus_ix(data, ic)
    dcn = get_dc_network(data)
    name = PSY.get_name(ic)
    if name in dcn.converter_name
        _set_converter_p_set!(dcn, only(findall(==(name), dcn.converter_name)), p)
    elseif _contributes_active_power(get_pf(data), ic)
        @views get_bus_active_power_injections(data)[bus_ix, :] .+= p
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
when the power flow does not consider reactive power. Otherwise it throws an error when the bus
of the component is not in the power flow, and it adds `net_injection[t]` to
`bus_reactive_power_injections`.

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
        bus_ix = _component_bus_ix(data, component)
        @views get_bus_reactive_power_injections(data)[bus_ix, :] .+= net_injection
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
does nothing when the power flow does not consider reactive power. Otherwise it subtracts each
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
        bus_ix = _component_bus_ix(data, load)
        _add_load_parts!(data, _reactive_withdrawal_arrays(data), bus_ix, parts)
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

When the bus of `sa` is not in the power flow, the function adds nothing. The initialization of
`PowerFlowData` skips such a bus in the same way.

`clear_injection_data!` sets these arrays to zero. After `clear_injection_data!`, call this function
again for every switched admittance.
"""
function add_switched_admittance!(data::PowerFlowData, sa::PSY.SwitchedAdmittance)
    if _bus_in_power_flow(data, sa)
        bus_ix = _component_bus_ix(data, sa)
        (p, q) = _switched_admittance_withdrawal(sa)
        @views _switched_admittance_active_withdrawals(data)[bus_ix, :] .+= p
        @views get_bus_reactive_power_constant_impedance_withdrawals(data)[bus_ix, :] .+= q
    end
    return
end

_seed_shunt_controller!(::Nothing, ::PSY.SwitchedAdmittance, ::AbstractVector{Float64}) =
    nothing

function _seed_shunt_controller!(
    set::ControlledDeviceSet,
    sa::PSY.SwitchedAdmittance,
    b::AbstractVector{Float64},
)
    name = PSY.get_name(sa)
    enrolled = findall(d -> d.name == name, set.shunts)
    for i in enrolled
        set.shunt_current[i, :] .= b
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
`t`. The controller does not start from the susceptance of the `System`.

The function throws `DimensionMismatch` when the length of `b` is not the number of time steps of
`data`. It throws an error when the bus of `sa` is not in the power flow. The function adds to the
arrays. Call `clear_injection_data!` before the first call.
"""
function add_switched_admittance!(
    data::PowerFlowData,
    sa::PSY.SwitchedAdmittance,
    b::AbstractVector{Float64},
)
    _check_net_injection_length(data, b, "b")
    bus_ix = _component_bus_ix(data, sa)
    (p, _) = _switched_admittance_withdrawal(sa)
    @views _switched_admittance_active_withdrawals(data)[bus_ix, :] .+= p
    @views get_bus_reactive_power_constant_impedance_withdrawals(data)[bus_ix, :] .-= b
    _seed_shunt_controller!(get_controlled_devices(data), sa, b)
    return
end

"""
    add_active_power_losses!(data::PowerFlowData, sc::PSY.SynchronousCondenser)

Add the active power losses of one synchronous condenser to `data` for every time step.

The function adds `PSY.get_active_power_losses(sc, u"SU")` to `bus_active_power_withdrawals`.
It throws an error when the bus of `sc` is not in the power flow. The function adds to the array.
`clear_injection_data!` sets the array to zero. After `clear_injection_data!`, call this function
again for every condenser.
"""
function add_active_power_losses!(data::PowerFlowData, sc::PSY.SynchronousCondenser)
    bus_ix = _component_bus_ix(data, sc)
    @views get_bus_active_power_withdrawals(data)[bus_ix, :] .+=
        PSY.get_active_power_losses(sc, u"SU")
    return
end
