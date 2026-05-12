"""
    compute_bus_state_offsets(bus_type)

Compute per-bus state-vector offsets and block sizes for the augmented
current-injection (rectangular) formulation. PQ and REF buses occupy 2 entries
each `(e, f)` or `(P_gen, Q_gen)`; PV buses occupy 3 entries `(e, f, Q)`.

Returns `(offsets, block_sizes, total_bus_state)` where
- `offsets[i]` is the 1-based start index of bus `i`'s block in the state vector
- `offsets[end]` is the start of the LCC tail (`== total_bus_state + 1`)
- `block_sizes[i] ∈ {2, 3}`
- `total_bus_state` is the total count of bus-state slots (excluding LCC tail)
"""
function compute_bus_state_offsets(
    bus_type::AbstractVector{PSY.ACBusTypes},
)
    n_buses = length(bus_type)
    offsets = Vector{REC_INDEX_TYPE}(undef, n_buses + 1)
    block_sizes = Vector{Int8}(undef, n_buses)
    pos = REC_INDEX_TYPE(1)
    for i in 1:n_buses
        offsets[i] = pos
        bs = bus_type[i] == PSY.ACBusTypes.PV ? Int8(3) : Int8(2)
        block_sizes[i] = bs
        pos += bs
    end
    offsets[n_buses + 1] = pos
    return offsets, block_sizes, Int(pos - 1)
end

"""
    fold_zip_constant_z!(Y_bus_eff, data, time_step)

Add the constant-impedance ZIP load components into the `Y_bus_eff` diagonal
as fixed shunt admittances. Sienna's ZIP load model parameterizes the load at
`|V| = 1.0 pu` directly: a load with `constant_impedance_active_power = β_P`
and `constant_impedance_reactive_power = β_Q` draws `S = (β_P + jβ_Q)·|V|²`.
As a shunt admittance this is `Y_sh = (β_P − jβ_Q)` because
`|V|²·conj(Y_sh) = (β_P + jβ_Q)·|V|²`.
"""
function fold_zip_constant_z!(
    Y_bus_eff::SparseMatrixCSC{ComplexF64, Int},
    data::ACPowerFlowData,
    time_step::Int64,
)
    n_buses = first(size(data.bus_type))
    for i in 1:n_buses
        β_P = data.bus_active_power_constant_impedance_withdrawals[i, time_step]
        β_Q = data.bus_reactive_power_constant_impedance_withdrawals[i, time_step]
        (β_P == 0.0 && β_Q == 0.0) && continue
        Y_bus_eff[i, i] += complex(β_P, -β_Q)
    end
    return
end

"""
    rect_initial_state!(x, data, bus_state_offset, bus_block_size, time_step)

Initialize the state vector `x` from `data.bus_magnitude`, `data.bus_angles`,
and the bus power-injection fields, plus the LCC tap/angle fields. Counterpart
of [`rect_update_data!`](@ref). At REF buses, the first two slots hold
`(P_gen, Q_gen)` (including any distributed-slack increment); elsewhere
the first two slots hold `(e, f)`, and PV buses' third slot holds `Q_gen`.
"""
function rect_initial_state!(
    x::Vector{Float64},
    data::ACPowerFlowData,
    bus_state_offset::Vector{REC_INDEX_TYPE},
    bus_block_size::Vector{Int8},
    time_step::Int64,
)
    bus_types = view(data.bus_type, :, time_step)
    n_buses = length(bus_types)
    for i in 1:n_buses
        off = Int(bus_state_offset[i])
        bt = bus_types[i]
        if bt == PSY.ACBusTypes.REF
            x[off] =
                data.bus_active_power_injections[i, time_step] -
                data.bus_active_power_withdrawals[i, time_step]
            x[off + 1] =
                data.bus_reactive_power_injections[i, time_step] -
                data.bus_reactive_power_withdrawals[i, time_step]
        else
            Vm = data.bus_magnitude[i, time_step]
            θ = data.bus_angles[i, time_step]
            x[off] = Vm * cos(θ)
            x[off + 1] = Vm * sin(θ)
            if bt == PSY.ACBusTypes.PV
                x[off + 2] =
                    data.bus_reactive_power_injections[i, time_step] -
                    data.bus_reactive_power_withdrawals[i, time_step]
            end
        end
    end
    n_lccs = size(data.lcc.p_set, 1)
    tail_start = Int(bus_state_offset[n_buses + 1])
    for i in 1:n_lccs
        base = tail_start + 4 * (i - 1) - 1
        x[base + 1] = data.lcc.rectifier.tap[i, time_step]
        x[base + 2] = data.lcc.inverter.tap[i, time_step]
        x[base + 3] = data.lcc.rectifier.thyristor_angle[i, time_step]
        x[base + 4] = data.lcc.inverter.thyristor_angle[i, time_step]
    end
    return
end

"""
    rect_update_data!(data, x, bus_state_offset, bus_block_size, time_step)

Write the state vector `x` back into the data fields (`bus_magnitude`,
`bus_angles`, `bus_*_power_injections`, lcc taps/angles). Counterpart
of [`rect_initial_state!`](@ref).
"""
function rect_update_data!(
    data::ACPowerFlowData,
    x::Vector{Float64},
    bus_state_offset::Vector{REC_INDEX_TYPE},
    bus_block_size::Vector{Int8},
    time_step::Int64,
)
    bus_types = view(data.bus_type, :, time_step)
    n_buses = length(bus_types)
    for i in 1:n_buses
        off = Int(bus_state_offset[i])
        bt = bus_types[i]
        if bt == PSY.ACBusTypes.REF
            P_gen = x[off]
            Q_gen = x[off + 1]
            data.bus_active_power_injections[i, time_step] =
                P_gen + data.bus_active_power_withdrawals[i, time_step]
            data.bus_reactive_power_injections[i, time_step] =
                Q_gen + data.bus_reactive_power_withdrawals[i, time_step]
        elseif bt == PSY.ACBusTypes.PQ
            e = x[off]
            f = x[off + 1]
            data.bus_magnitude[i, time_step] = sqrt(e^2 + f^2)
            data.bus_angles[i, time_step] = atan(f, e)
        else  # PV: do NOT overwrite bus_magnitude (V_set must be preserved).
            e = x[off]
            f = x[off + 1]
            data.bus_angles[i, time_step] = atan(f, e)
            Q_gen = x[off + 2]
            data.bus_reactive_power_injections[i, time_step] =
                Q_gen + data.bus_reactive_power_withdrawals[i, time_step]
        end
    end
    n_lccs = size(data.lcc.p_set, 1)
    tail_start = Int(bus_state_offset[n_buses + 1])
    for i in 1:n_lccs
        base = tail_start + 4 * (i - 1) - 1
        data.lcc.rectifier.tap[i, time_step] = x[base + 1]
        data.lcc.inverter.tap[i, time_step] = x[base + 2]
        data.lcc.rectifier.thyristor_angle[i, time_step] = x[base + 3]
        data.lcc.inverter.thyristor_angle[i, time_step] = x[base + 4]
    end
    return
end
