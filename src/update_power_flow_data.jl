function _zip_shares(parts::NTuple{3, Float64})
    total = sum(parts)
    if iszero(total)
        return (1.0, 0.0, 0.0)
    end
    return parts ./ total
end

function _exponent_shares(load::PSY.ExponentialLoad, exponent::Float64)
    slot = _exponential_zip_slot(load, exponent)
    return ntuple(k -> ifelse(k == slot, 1.0, 0.0), 3)
end

_active_power_shares(::PowerFlowEvaluationModel, ::PSY.ElectricLoad) = (1.0, 0.0, 0.0)
_active_power_shares(
    pf::AbstractACPowerFlow,
    l::Union{PSY.StandardLoad, PSY.InterruptibleStandardLoad},
) = _zip_shares(_load_active_power_parts(pf, l))
_active_power_shares(::AbstractACPowerFlow, l::PSY.ExponentialLoad) =
    _exponent_shares(l, PSY.get_α(l))

_reactive_power_shares(::PowerFlowEvaluationModel, ::PSY.ElectricLoad) = (1.0, 0.0, 0.0)
_reactive_power_shares(
    pf::AbstractACPowerFlow,
    l::Union{PSY.StandardLoad, PSY.InterruptibleStandardLoad},
) = _zip_shares(_load_reactive_power_parts(pf, l))
_reactive_power_shares(::AbstractACPowerFlow, l::PSY.ExponentialLoad) =
    _exponent_shares(l, PSY.get_β(l))

function _check_net_injection_length(
    data::PowerFlowData,
    net_injection::AbstractVector{Float64},
)
    n_time_steps = get_time_steps(data)
    if length(net_injection) != n_time_steps
        throw(
            DimensionMismatch(
                "net_injection has $(length(net_injection)) entries, " *
                "but the data has $(n_time_steps) time steps.",
            ),
        )
    end
    return
end

function _component_bus_ix(data::PowerFlowData, component::PSY.StaticInjection)
    reverse_bus_search_map =
        PNM.get_reverse_bus_search_map(get_network_reduction_data(data))
    return _get_bus_ix(
        get_bus_lookup(data),
        reverse_bus_search_map,
        PSY.get_number(PSY.get_bus(component)),
    )
end

function _add_injection!(
    injections::Matrix{Float64},
    bus_ix::Int,
    net_injection::AbstractVector{Float64},
)
    for (t, value) in enumerate(net_injection)
        injections[bus_ix, t] += value
    end
    return
end

function _add_withdrawal_parts!(
    withdrawals::NTuple{3, Matrix{Float64}},
    bus_ix::Int,
    net_injection::AbstractVector{Float64},
    shares::NTuple{3, Float64},
)
    for k in 1:3, (t, value) in enumerate(net_injection)
        withdrawals[k][bus_ix, t] -= value * shares[k]
    end
    return
end

function _set_off_area_load_power!(
    ::PowerFlowData,
    ::PSY.ElectricLoad,
    ::AbstractVector{Float64},
    ::NTuple{3, Float64},
)
    return
end

function _set_off_area_load_power!(
    data::ACPowerFlowData,
    load::PSY.ElectricLoad,
    net_injection::AbstractVector{Float64},
    shares::NTuple{3, Float64},
)
    name = PSY.get_name(load)
    if has_off_area_load(data, name)
        for (t, value) in enumerate(net_injection)
            set_off_area_load_power!(
                data,
                name,
                t,
                -value * shares[1],
                -value * shares[2],
                -value * shares[3],
            )
        end
    end
    return
end

function _add_active_power!(
    data::PowerFlowData,
    ::PSY.StaticInjection,
    bus_ix::Int,
    net_injection::AbstractVector{Float64},
)
    _add_injection!(get_bus_active_power_injections(data), bus_ix, net_injection)
    return
end

function _add_active_power!(
    data::PowerFlowData,
    load::PSY.ElectricLoad,
    bus_ix::Int,
    net_injection::AbstractVector{Float64},
)
    shares = _active_power_shares(get_pf(data), load)
    withdrawals = (
        get_bus_active_power_withdrawals(data),
        get_bus_active_power_constant_current_withdrawals(data),
        get_bus_active_power_constant_impedance_withdrawals(data),
    )
    _add_withdrawal_parts!(withdrawals, bus_ix, net_injection, shares)
    _set_off_area_load_power!(data, load, net_injection, shares)
    return
end

function _add_reactive_power!(
    data::PowerFlowData,
    ::PSY.StaticInjection,
    bus_ix::Int,
    net_injection::AbstractVector{Float64},
)
    _add_injection!(get_bus_reactive_power_injections(data), bus_ix, net_injection)
    return
end

function _add_reactive_power!(
    data::PowerFlowData,
    load::PSY.ElectricLoad,
    bus_ix::Int,
    net_injection::AbstractVector{Float64},
)
    shares = _reactive_power_shares(get_pf(data), load)
    withdrawals = (
        get_bus_reactive_power_withdrawals(data),
        get_bus_reactive_power_constant_current_withdrawals(data),
        get_bus_reactive_power_constant_impedance_withdrawals(data),
    )
    _add_withdrawal_parts!(withdrawals, bus_ix, net_injection, shares)
    return
end

"""
    add_active_power!(data::PowerFlowData, component::PSY.StaticInjection,
        net_injection::AbstractVector{Float64})

Add the active power of one component to `data` for every time step.

`net_injection[t]` is the power that `component` injects at time step `t`. The unit is p.u. on
the system base, at 1 p.u. voltage. A withdrawal is negative. The function throws
`DimensionMismatch` when the length of `net_injection` is not the number of time steps of `data`.
The function adds to the arrays. Call `clear_injection_data!` before the first call.

A component that is not a `PSY.ElectricLoad` adds `net_injection[t]` to
`bus_active_power_injections`. A `PSY.ElectricLoad` adds `-net_injection[t]` times a share to
`bus_active_power_withdrawals`, `bus_active_power_constant_current_withdrawals` and
`bus_active_power_constant_impedance_withdrawals`. The three shares are the constant power, constant
current and constant impedance parts of the load.

For an AC power flow, the shares of a `PSY.StandardLoad` or `PSY.InterruptibleStandardLoad` are
its parts in the `System`, divided by their sum. A load with parts that sum to zero is constant
power. The share of a `PSY.ExponentialLoad` is 1 in the slot of its exponent `α` and 0 in the other
two slots. The exponent 0 is constant power, 1 is constant current and 2 is constant impedance. The
value of the load in the `System` does not change the shares. Every other load is constant power.
A DC power flow treats every load as constant power. For an AC power flow, the function also updates
the off-area record of a load that has one.

These functions cover the power that an operations model decides. Condenser losses, FACTS devices
and the DC network of PowerFlows use other paths.
"""
function add_active_power!(
    data::PowerFlowData,
    component::PSY.StaticInjection,
    net_injection::AbstractVector{Float64},
)
    _check_net_injection_length(data, net_injection)
    bus_ix = _component_bus_ix(data, component)
    _add_active_power!(data, component, bus_ix, net_injection)
    return
end

"""
    add_reactive_power!(data::PowerFlowData, component::PSY.StaticInjection,
        net_injection::AbstractVector{Float64})

Add the reactive power of one component to `data` for every time step.

`net_injection[t]` is the power that `component` injects at time step `t`. The unit is p.u. on
the system base, at 1 p.u. voltage. A withdrawal is negative. The function throws
`DimensionMismatch` when the length of `net_injection` is not the number of time steps of `data`.
The function adds to the arrays. Call `clear_injection_data!` before the first call.

A component that is not a `PSY.ElectricLoad` adds `net_injection[t]` to
`bus_reactive_power_injections`. A `PSY.ElectricLoad` adds `-net_injection[t]` times a share to
`bus_reactive_power_withdrawals`, `bus_reactive_power_constant_current_withdrawals` and
`bus_reactive_power_constant_impedance_withdrawals`. The three shares are the constant power,
constant current and constant impedance parts of the load.

For an AC power flow, the shares of a `PSY.StandardLoad` or `PSY.InterruptibleStandardLoad` are
its reactive parts in the `System`, divided by their sum. A load with parts that sum to zero is
constant power. The share of a `PSY.ExponentialLoad` is 1 in the slot of its exponent `β` and 0 in
the other two slots. The exponent 0 is constant power, 1 is constant current and 2 is constant
impedance. The value of the load in the `System` does not change the shares. Every other load is
constant power. A DC power flow treats every load as constant power.

These functions cover the power that an operations model decides. Condenser losses, FACTS devices
and the DC network of PowerFlows use other paths.
"""
function add_reactive_power!(
    data::PowerFlowData,
    component::PSY.StaticInjection,
    net_injection::AbstractVector{Float64},
)
    _check_net_injection_length(data, net_injection)
    bus_ix = _component_bus_ix(data, component)
    _add_reactive_power!(data, component, bus_ix, net_injection)
    return
end
