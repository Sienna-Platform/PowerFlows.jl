"""Partitions the state vector's variables based on what physical quantity each represents. 
Returns a `NamedTuple`, with the 4 keys `Va`, `Vm`, `P`, and `Q`. The 4 values are vectors 
of length equal to the number of buses, with `NaN`s in the positions where that physical 
quantity is not part of the state vector for that bus. (Currently not intended for use in 
spots where performance is critical.)"""
function partition_state(x::Vector{Float64},
    bus_types::AbstractVector{PSY.ACBusTypes},
)
    # usually, bus_types will be data.bus_type[:, time_step]
    nbuses = div(size(x, 1), 2)
    (Vms, Vas, Ps, Qs) = (fill(NaN, nbuses) for _ in 1:4)
    for i in 1:nbuses
        if bus_types[i] == PSY.ACBusTypes.REF
            Ps[i] = x[2 * i - 1]
            Qs[i] = x[2 * i]
        elseif bus_types[i] == PSY.ACBusTypes.PQ
            Vms[i] = x[2 * i - 1]
            Vas[i] = x[2 * i]
        elseif bus_types[i] == PSY.ACBusTypes.PV
            Qs[i] = x[2 * i - 1]
            Vas[i] = x[2 * i]
        end
    end
    return (Va = Vas, Vm = Vms, P = Ps, Q = Qs)
end

"""
Length of the shared non-bus state tail (LCC + VSC + area interchange), identical across
polar/rectangular/mixed formulations. The ONLY place the tail composition is defined — every
state/residual sizing site calls this.
"""
function state_tail_length(data::ACPowerFlowData, dcn::DCNetwork)
    return 4 * size(data.lcc.p_set, 1) + vsc_tail_length(dcn) +
           area_tail_length(data.area_interchange)
end

"""
1-based `x`-index offset for the area-interchange tail in the polar state/residual layout:
`ΔP_a` for tail slot `tail_ix` lives at `x[area_tail_offset(data, dcn) + tail_ix]`; residual
row `r_a` at the same index.
"""
function area_tail_offset(data::ACPowerFlowData, dcn::DCNetwork)
    n_buses = size(data.bus_type, 1)
    return 2 * n_buses + 4 * size(data.lcc.p_set, 1) + vsc_tail_length(dcn)
end

"""Update state vector based on values of fields of data."""
function update_state!(x::Vector{Float64},
    data::ACPowerFlowData,
    time_step::Int64,
)
    # I really only need access to 3 classes of fields of data:
    # bus types, bus power contributions [inj/widthdrawal], and bus voltages
    dcn = get_dc_network(data)
    @assert length(x) ==
            2 * size(data.bus_type, 1) + state_tail_length(data, dcn)
    state_variable_count = 1
    for (ix, b) in enumerate(data.bus_type[:, time_step])
        if b == PSY.ACBusTypes.REF
            x[state_variable_count] =
                data.bus_active_power_injections[ix, time_step] -
                data.bus_active_power_withdrawals[ix, time_step]
            x[state_variable_count + 1] =
                data.bus_reactive_power_injections[ix, time_step] -
                data.bus_reactive_power_withdrawals[ix, time_step]
            state_variable_count += 2
        elseif b == PSY.ACBusTypes.PV
            x[state_variable_count] =
                data.bus_reactive_power_injections[ix, time_step] -
                data.bus_reactive_power_withdrawals[ix, time_step]
            x[state_variable_count + 1] = data.bus_angles[ix, time_step]
            state_variable_count += 2
        elseif b == PSY.ACBusTypes.PQ
            x[state_variable_count] = data.bus_magnitude[ix, time_step]
            x[state_variable_count + 1] = data.bus_angles[ix, time_step]
            state_variable_count += 2
        else
            throw(ArgumentError("$b not recognized as a bustype"))
        end
    end
    @assert state_variable_count - 1 == size(data.bus_type, 1) * 2
    for i in eachindex(data.lcc.p_set[:, time_step])
        x[state_variable_count] = data.lcc.rectifier.tap[i, time_step]
        x[state_variable_count + 1] = data.lcc.inverter.tap[i, time_step]
        x[state_variable_count + 2] = data.lcc.rectifier.thyristor_angle[i, time_step]
        x[state_variable_count + 3] = data.lcc.inverter.thyristor_angle[i, time_step]
        state_variable_count += 4
    end
    # VSC tail: per converter (P_c, Q_c), then per DC node V_dc. Mirrors `_read_vsc_state!`.
    nconv = n_vsc_converters(dcn)
    for c in 1:nconv
        x[state_variable_count] = dcn.p_c[c, time_step]
        x[state_variable_count + 1] = dcn.q_c[c, time_step]
        state_variable_count += 2
    end
    for k in 1:n_dc_nodes(dcn)
        x[state_variable_count] = dcn.node_vdc[k, time_step]
        state_variable_count += 1
    end
    # Area-interchange tail (ΔP_a): mirrored on `data.area_interchange.delta_p`, written back
    # every residual evaluation (`_update_residual_values!`) — read back here, column-indexed
    # by `time_step`, exactly like the LCC tap / VSC tails above, so a warm re-solve on the
    # SAME `data` and time step starts from the converged ΔP_a instead of resetting to 0 (and
    # a different time step's converged ΔP_a cannot leak in). A freshly enrolled area's
    # mirror starts at 0.0 (see `AreaInterchangeData` construction), so a genuine flat start
    # is unaffected.
    for tail_ix in 1:n_controlled_areas(data.area_interchange)
        x[state_variable_count] = data.area_interchange.delta_p[tail_ix, time_step]
        state_variable_count += 1
    end
    @assert state_variable_count - 1 == length(x)
end

"""Update the fields of data based on the values of the state vector."""
function update_data!(data::ACPowerFlowData,
    x::Vector{Float64},
    time_step::Int64,
)
    # same as above. I really only need access to 3 classes of fields of data:
    # bus types, bus power contributions [inj/widthdrawal], and bus voltage [angle/mag]
    bus_types = view(data.bus_type, :, time_step)
    for (ix, bt) in enumerate(bus_types)
        # PERF: try indexing outside of the call and passing references, instead of indexing
        #       inside the call and passing index + vectors.
        # 3 case if-else with Val(constant) is faster than 1 case with Val(bt)
        if bt == PSY.ACBusTypes.REF
            _set_state_variables_at_bus(ix, x, data, time_step, Val(PSY.ACBusTypes.REF))
        elseif bt == PSY.ACBusTypes.PV
            _set_state_variables_at_bus(ix, x, data, time_step, Val(PSY.ACBusTypes.PV))
        elseif bt == PSY.ACBusTypes.PQ
            _set_state_variables_at_bus(ix, x, data, time_step, Val(PSY.ACBusTypes.PQ))
        end
    end
end

# dispatching on Val for performance reasons.
function _set_state_variables_at_bus(
    ix::Int,
    StateVector::Vector{Float64},
    data::ACPowerFlowData,
    time_step::Int64,
    ::Val{PSY.ACBusTypes.REF})
    # When bustype == REFERENCE PSY.Bus, state variables are Active and Reactive Power Generated
    data.bus_active_power_injections[ix, time_step] =
        StateVector[2 * ix - 1] + data.bus_active_power_withdrawals[ix, time_step]
    data.bus_reactive_power_injections[ix, time_step] =
        StateVector[2 * ix] + data.bus_reactive_power_withdrawals[ix, time_step]
end

function _set_state_variables_at_bus(
    ix::Int,
    StateVector::Vector{Float64},
    data::ACPowerFlowData,
    time_step::Int64,
    ::Val{PSY.ACBusTypes.PV})
    # When bustype == PV PSY.Bus, state variables are Reactive Power Generated and Voltage Angle
    data.bus_reactive_power_injections[ix, time_step] =
        StateVector[2 * ix - 1] + data.bus_reactive_power_withdrawals[ix, time_step]
    data.bus_angles[ix, time_step] = StateVector[2 * ix]
end

function _set_state_variables_at_bus(
    ix::Int,
    x::Vector{Float64},
    data::ACPowerFlowData,
    time_step::Int64,
    ::Val{PSY.ACBusTypes.PQ})
    # When bustype == PQ PSY.Bus, state variables are Voltage Magnitude and Voltage Angle
    data.bus_magnitude[ix, time_step] = x[2 * ix - 1]
    data.bus_angles[ix, time_step] = x[2 * ix]
end

# could dispatch on val, and/or combine with update_data!.
"""Update `P_net` and `Q_net` based on the values of the state vector."""
function update_net_power!(P_net::Vector{Float64},
    Q_net::Vector{Float64},
    x::Vector{Float64},
    bus_types::AbstractVector{PSY.ACBusTypes},
)
    # usually, bus_types will be data.bus_type[:, time_step]
    for (ix, bt) in enumerate(bus_types)
        if bt == PSY.ACBusTypes.REF
            P_net[ix] = x[2 * ix - 1]
            Q_net[ix] = x[2 * ix]
        elseif bt == PSY.ACBusTypes.PV
            Q_net[ix] = x[2 * ix - 1]
        end
    end
    return
end
