"""
    compute_bus_state_offsets(bus_type)

Per-bus state-vector offsets and block sizes for the rectangular power-mismatch
formulation. Every bus occupies 2 slots: `(e, f)` at PQ and PV, `(P_net, Q_net)` at REF.

Returns `(offsets, block_sizes, total_bus_state)` where
- `offsets[i]` is the 1-based start index of bus `i`'s block in the state vector
- `offsets[end]` is the start of the LCC tail (`== total_bus_state + 1`)
- `block_sizes[i] == 2`
- `total_bus_state` is the total count of bus-state slots (excluding the tail)
"""
function compute_bus_state_offsets(
    bus_type::AbstractVector{PSY.ACBusTypes.Value},
)
    n_buses = length(bus_type)
    offsets = Vector{REC_INDEX_TYPE}(undef, n_buses + 1)
    block_sizes = fill(Int8(2), n_buses)
    for i in 1:(n_buses + 1)
        offsets[i] = REC_INDEX_TYPE(2 * i - 1)
    end
    return offsets, block_sizes, 2 * n_buses
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
    _rect_fill_state!(x, data, bus_state_offset, type_time_step, value_time_step)

Fill the rectangular state vector `x`. The block *meaning* (REF `(P_net, Q_net)`,
otherwise `(e, f)`) is taken from `data.bus_type[:, type_time_step]`; the *values*
(voltages, injections, LCC taps/angles) are read from `*[:, value_time_step]`.

With `type_time_step == value_time_step` this is the plain flat start. With
`value_time_step` pointing at a previously converged step it produces the
previous-solution warm-start candidate while keeping the offsets valid for the
current step's bus types (the rectangular analog of polar `_previous_solution_start` /
`update_state!`).
"""
function _rect_fill_state!(
    x::Vector{Float64},
    data::ACPowerFlowData,
    bus_state_offset::Vector{REC_INDEX_TYPE},
    type_time_step::Int64,
    value_time_step::Int64,
)
    bus_types = view(data.bus_type, :, type_time_step)
    n_buses = length(bus_types)
    for i in 1:n_buses
        off = Int(bus_state_offset[i])
        bt = bus_types[i]
        if bt == PSY.ACBusTypes.REF
            x[off] =
                data.bus_active_power_injections[i, value_time_step] -
                data.bus_active_power_withdrawals[i, value_time_step]
            x[off + 1] =
                data.bus_reactive_power_injections[i, value_time_step] -
                data.bus_reactive_power_withdrawals[i, value_time_step]
        else
            Vm = data.bus_magnitude[i, value_time_step]
            θ = data.bus_angles[i, value_time_step]
            x[off] = Vm * cos(θ)
            x[off + 1] = Vm * sin(θ)
        end
    end
    n_lccs = size(data.lcc.p_set, 1)
    # State-vector layout. The full state is `[bus_block_1 ; … ; bus_block_N ; LCC_tail]`.
    # The bus blocks occupy slots `1 .. total_bus_state`; the LCC tail starts at
    # `total_bus_state + 1`. Each line-commutated converter (LCC) — a two-terminal HVDC
    # link with a rectifier (AC→DC) on one end and an inverter (DC→AC) on the other —
    # contributes 4 state variables: rectifier transformer tap ratio, inverter
    # transformer tap ratio, rectifier thyristor (firing) angle α_r, and inverter
    # thyristor angle α_i. The i-th LCC therefore occupies slots
    # `offset_lcc + 1 .. offset_lcc + 4` with `offset_lcc = total_bus_state + (i-1)*4`.
    # The same layout is used in `rect_update_data!`, the residual's LCC tail,
    # and the Jacobian's LCC structure/value updaters.
    total_bus_state = Int(bus_state_offset[n_buses + 1]) - 1
    for i in 1:n_lccs
        offset_lcc = total_bus_state + (i - 1) * 4
        x[offset_lcc + 1] = data.lcc.rectifier.tap[i, value_time_step]
        x[offset_lcc + 2] = data.lcc.inverter.tap[i, value_time_step]
        x[offset_lcc + 3] = data.lcc.rectifier.thyristor_angle[i, value_time_step]
        x[offset_lcc + 4] = data.lcc.inverter.thyristor_angle[i, value_time_step]
    end
    # VSC / DC-network tail: per converter (P_c, Q_c), then per DC node V_dc.
    dcn = get_dc_network(data)
    vsc_off = total_bus_state + 4 * n_lccs
    nconv = n_vsc_converters(dcn)
    for c in 1:nconv
        x[vsc_off + 2 * c - 1] = dcn.p_c[c, value_time_step]
        x[vsc_off + 2 * c] = dcn.q_c[c, value_time_step]
    end
    for k in 1:n_dc_nodes(dcn)
        x[vsc_off + 2 * nconv + k] = dcn.node_vdc[k, value_time_step]
    end
    return
end

"""
    rect_initial_state!(x, data, bus_state_offset, bus_block_size, time_step)

Initialize the state vector `x` from `data.bus_magnitude`, `data.bus_angles`,
and the bus power-injection fields, plus the LCC tap/angle fields. Counterpart
of [`rect_update_data!`](@ref). At REF buses the two slots hold the net
`(P, Q)` injection (the P slot carries the whole island slack); elsewhere `(e, f)`.
"""
function rect_initial_state!(
    x::Vector{Float64},
    data::ACPowerFlowData,
    bus_state_offset::Vector{REC_INDEX_TYPE},
    bus_block_size::Vector{Int8},
    time_step::Int64,
)
    _rect_fill_state!(x, data, bus_state_offset, time_step, time_step)
    return
end

"""
    rect_update_data!(data, x, bus_state_offset, bus_block_size, time_step)

Write the state-derived voltage fields (`bus_magnitude`, `bus_angles`) and
LCC taps/angles from `x` back into `data`. Per-iteration helper invoked by
the rectangular residual.

Does NOT write `bus_active_power_injections` / `bus_reactive_power_injections`:
those are finalized once after convergence with the correct distributed-slack
share by [`rect_finalize_bus_injections!`](@ref). At REF buses `x[off]` carries
the entire subnetwork slack and would over-attribute it; a PV bus's reactive
power is not a state and is only recovered at the converged voltages.

Counterpart of [`rect_initial_state!`](@ref).
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
            # REF voltage is fixed (V_set, θ_set); bus_magnitude/bus_angles
            # were initialized at PowerFlowData construction and stay constant.
        elseif bt == PSY.ACBusTypes.PQ
            e = x[off]
            f = x[off + 1]
            data.bus_magnitude[i, time_step] = sqrt(e^2 + f^2)
            data.bus_angles[i, time_step] = atan(f, e)
        else  # PV: do NOT overwrite bus_magnitude (V_set must be preserved).
            e = x[off]
            f = x[off + 1]
            data.bus_angles[i, time_step] = atan(f, e)
        end
    end
    n_lccs = size(data.lcc.p_set, 1)
    total_bus_state = Int(bus_state_offset[n_buses + 1]) - 1
    for i in 1:n_lccs
        offset_lcc = total_bus_state + (i - 1) * 4
        data.lcc.rectifier.tap[i, time_step] = x[offset_lcc + 1]
        data.lcc.inverter.tap[i, time_step] = x[offset_lcc + 2]
        data.lcc.rectifier.thyristor_angle[i, time_step] = x[offset_lcc + 3]
        data.lcc.inverter.thyristor_angle[i, time_step] = x[offset_lcc + 4]
    end
    return
end

"""
    rect_finalize_bus_injections!(data, x, residual, time_step)

Distribute the converged subnetwork slack across participating buses and write
`bus_active_power_injections` and `bus_reactive_power_injections` accordingly.

Mirrors polar `_set_state_variables_at_bus!` semantics: at every participating
REF and PV bus the solved net active power is `P_net_set[i] + c_i · P_slack_total`,
where `P_slack_total = x[ref_off] - P_net_set[ref_bus]`. Net reactive power is
`x[off + 1]` at REF; at PV it is the calculated injection `Im(V·conj(I))` of the last
residual evaluation (`residual.Ir_acc`/`Ii_acc`, Y-bus with constant-Z folded in plus the
LCC terminals) less the VSC converters' `Q_c`. Both are net, not generation — the
withdrawal is added back below, differently per bus type (see the branch comments). At PQ
buses no slack attribution is needed: `bus_active_power_injections` /
`bus_reactive_power_injections` already hold the load setpoint from `PowerFlowData`
construction.

A multi-swing island (REF bus in `independent_ref`) holds each swing at its own
fixed voltage and self-balances its own P-slot instead of sharing the island's
distributed scalar: such a REF's net active power is `x[off]` directly.

Called once per time step after the NR loop converges, with the residual last
evaluated at `x`.
"""
function rect_finalize_bus_injections!(
    data::ACPowerFlowData,
    x::Vector{Float64},
    residual::ACRectangularCIResidual,
    time_step::Int64,
)
    bus_types = view(data.bus_type, :, time_step)
    P_net_set = residual.P_net_set
    spf = residual.bus_slack_participation_factors
    e_state = residual.e_state
    f_state = residual.f_state
    for (ref_bus, subnetwork_buses) in residual.subnetworks
        ref_off = Int(residual.bus_state_offset[ref_bus])
        P_slack_total = x[ref_off] - P_net_set[ref_bus]
        for bus_k in subnetwork_buses
            c_k = spf[bus_k]
            bt = bus_types[bus_k]
            off = Int(residual.bus_state_offset[bus_k])
            if bt == PSY.ACBusTypes.REF
                if bus_k in residual.independent_ref
                    P_net_cp = x[off]
                else
                    P_net_cp = P_net_set[bus_k] + c_k * P_slack_total
                end
                Q_net_cp = x[off + 1]
                # REF slots are net of constant power only (cp): the residual
                # subtracts `const_I * V_set` from them itself.
                data.bus_active_power_injections[bus_k, time_step] =
                    P_net_cp + data.bus_active_power_withdrawals[bus_k, time_step] -
                    data.bus_hvdc_net_power[bus_k, time_step]
                data.bus_reactive_power_injections[bus_k, time_step] =
                    Q_net_cp + data.bus_reactive_power_withdrawals[bus_k, time_step]
            elseif bt == PSY.ACBusTypes.PV
                # Net of constant power and constant current (constant-Z is in the Y-bus).
                P_eff = P_net_set[bus_k] + c_k * P_slack_total
                data.bus_active_power_injections[bus_k, time_step] =
                    P_eff +
                    get_bus_active_power_non_impedance_withdrawals(data, bus_k, time_step) -
                    data.bus_hvdc_net_power[bus_k, time_step]
                Q_net =
                    f_state[bus_k] * residual.Ir_acc[bus_k] -
                    e_state[bus_k] * residual.Ii_acc[bus_k]
                data.bus_reactive_power_injections[bus_k, time_step] =
                    Q_net +
                    get_bus_reactive_power_non_impedance_withdrawals(data, bus_k, time_step)
            end
        end
    end
    dcn = get_dc_network(data)
    @inbounds for c in 1:n_vsc_converters(dcn)
        ix = dcn.converter_ac_bus_ix[c]
        if bus_types[ix] == PSY.ACBusTypes.PV
            data.bus_reactive_power_injections[ix, time_step] -= dcn.q_c[c, time_step]
        end
    end
    return
end
