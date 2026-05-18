"""
    compute_mixed_bus_state_offsets(bus_type)

Compute per-bus state-vector offsets and block sizes for the mixed
current/power-balance (MCPB) formulation. Every bus — PQ, PV, and REF —
occupies exactly 2 state entries (no PV→3 expansion as in the rectangular CI
formulation).

Returns `(offsets, block_sizes, total_bus_state)` where
- `offsets[i]` is the 1-based start index of bus `i`'s block in the state vector
- `offsets[end]` is the start of the LCC tail (`== total_bus_state + 1`)
- `block_sizes[i] == 2`
- `total_bus_state` is the total count of bus-state slots (excluding LCC tail)
"""
function compute_mixed_bus_state_offsets(
    bus_type::AbstractVector{PSY.ACBusTypes},
)
    n_buses = length(bus_type)
    offsets = Vector{REC_INDEX_TYPE}(undef, n_buses + 1)
    block_sizes = Vector{Int8}(undef, n_buses)
    # uniformly 2; vector kept for signature parity with the rectangular formulation
    fill!(block_sizes, Int8(2))
    pos = REC_INDEX_TYPE(1)
    for i in 1:n_buses
        offsets[i] = pos
        pos += REC_INDEX_TYPE(2)
    end
    offsets[n_buses + 1] = pos
    return offsets, block_sizes, Int(pos - 1)
end

"""
    _mixed_fill_state!(x, data, bus_state_offset, type_time_step, value_time_step)

Fill the MCPB state vector `x`. The state-block *layout* (REF/PV/PQ → 2-slot
blocks) is taken from `data.bus_type[:, type_time_step]`; the *values*
(voltages, injections, LCC taps/angles) are read from `*[:, value_time_step]`.

With `type_time_step == value_time_step` this is the plain flat start. With
`value_time_step` pointing at a previously converged step it produces the
previous-solution warm-start candidate while keeping the offsets valid for the
current step.
"""
function _mixed_fill_state!(
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
            # MCPB: both PV and PQ use 2 state slots (e, f); no Q slot (unlike rectangular CI).
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
    # The same layout is used in `mixed_update_data!`, the residual's LCC tail,
    # and the Jacobian's LCC structure/value updaters.
    total_bus_state = Int(bus_state_offset[n_buses + 1]) - 1
    for i in 1:n_lccs
        offset_lcc = total_bus_state + (i - 1) * 4
        x[offset_lcc + 1] = data.lcc.rectifier.tap[i, value_time_step]
        x[offset_lcc + 2] = data.lcc.inverter.tap[i, value_time_step]
        x[offset_lcc + 3] = data.lcc.rectifier.thyristor_angle[i, value_time_step]
        x[offset_lcc + 4] = data.lcc.inverter.thyristor_angle[i, value_time_step]
    end
    return
end

"""
    mixed_initial_state!(x, data, bus_state_offset, bus_block_size, time_step)

Initialize the MCPB state vector `x` from `data.bus_magnitude`,
`data.bus_angles`, and the bus power-injection fields, plus the LCC tap/angle
fields. Counterpart of [`mixed_update_data!`](@ref). At REF buses, the two
slots hold `(P_gen, Q_gen)` (including any distributed-slack increment);
elsewhere the two slots hold `(e, f)`.
"""
function mixed_initial_state!(
    x::Vector{Float64},
    data::ACPowerFlowData,
    bus_state_offset::Vector{REC_INDEX_TYPE},
    bus_block_size::Vector{Int8},
    time_step::Int64,
)
    _mixed_fill_state!(x, data, bus_state_offset, time_step, time_step)
    return
end

"""
    mixed_update_data!(data, x, bus_state_offset, bus_block_size, time_step)

Write the state-derived voltage fields (`bus_magnitude`, `bus_angles`) and
LCC taps/angles from `x` back into `data`. Per-iteration helper invoked by
the MCPB residual.

Does NOT write `bus_active_power_injections` / `bus_reactive_power_injections`:
those are finalized once after convergence with the correct distributed-slack
share by [`mixed_finalize_bus_injections!`](@ref). At REF buses `x[off]`
carries the entire subnetwork slack and would over-attribute it; at PV buses
the gen Q is recovered from the network only after convergence.

Counterpart of [`mixed_initial_state!`](@ref).
"""
function mixed_update_data!(
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
Distribute the converged subnetwork slack across participating buses and write
`bus_active_power_injections`/`bus_reactive_power_injections`. Mirrors
[`rect_finalize_bus_injections!`](@ref) and polar `_setpq`: the reported
injection is `net_injection + get_bus_*_power_total_withdrawals` (`- hvdc` for
P) at every participating REF/PV bus; PQ buses need no slack attribution.

The one real difference from rect: MCPB has no Q state slot, so the polar net
injection `S_net,i = V_i·conj((Y_raw·V)_i)` is recovered from a RAW-Y matvec
(raw Y-bus excludes all load components — const-P/I/Z — so this avoids the
mixed-vs-polar const-Z/const-I double-count). `Q_net = imag(S_net)` directly;
`P_net` redistributes the subnetwork slack via a two-pass loop: pass 1 sums the
total physical slack (Σ converged net-P − Σ initial polar net-P set-point),
pass 2 writes each bus `P_net = P_net_set_polar[i] + c_i·P_slack_total`.

Called once per time step after the NR loop converges (the slack distribution
is only meaningful at the converged x).
"""
function mixed_finalize_bus_injections!(
    data::ACPowerFlowData,
    x::Vector{Float64},
    bus_state_offset::Vector{REC_INDEX_TYPE},
    bus_slack_participation_factors::SparseVector{Float64, Int},
    subnetworks::Dict{Int64, Vector{Int64}},
    e_state::Vector{Float64},
    f_state::Vector{Float64},
    time_step::Int64,
)
    bus_types = view(data.bus_type, :, time_step)
    n_buses = length(e_state)
    # Raw Y-bus (NO const-Z fold): excludes ALL load components (const-P,
    # const-I, const-Z), so S_net,i = V_i·conj((Y_raw·V)_i) is the polar net
    # injection (P_gen − P_load_total, Q_gen − Q_load_total). Walk its nonzeros
    # directly to form Y_raw·V (matvec, no mutation): O(nnz), avoids
    # materializing a full complex copy at 10k-bus scale.
    Y_raw = data.power_network_matrix.data
    YVr = zeros(Float64, n_buses)   # Re(Y_raw·V)
    YVi = zeros(Float64, n_buses)   # Im(Y_raw·V)
    Yvals = SparseArrays.nonzeros(Y_raw)
    Yrows = SparseArrays.rowvals(Y_raw)
    @inbounds for col in 1:n_buses
        e_col = e_state[col]
        f_col = f_state[col]
        for j in Y_raw.colptr[col]:(Y_raw.colptr[col + 1] - 1)
            row = Yrows[j]
            y = Yvals[j]
            g = real(y)
            b = imag(y)
            YVr[row] += g * e_col - b * f_col   # Re(Y·V)
            YVi[row] += g * f_col + b * e_col   # Im(Y·V)
        end
    end
    # LCC AC-terminal self-admittance current. The residual accumulates this
    # into the network current (residual step 4), so at an LCC terminal bus the
    # converged network current is `Y_raw·V + Y_lcc·V`, not `Y_raw·V` alone. The
    # LCC current is NOT a ZIP load and NOT in `get_bus_*_power_total_withdrawals`,
    # so it must be added here (same loop/sign as residual step 4) — omitting it
    # drops the full DC-line power at the terminal buses. Additive on top of the
    # raw-Y walk, so the C1 ZIP-double-count fix is preserved. O(n_lcc).
    n_lccs = size(data.lcc.p_set, 1)
    if n_lccs > 0
        for (bus_indices, self_admittances) in
            zip(data.lcc.bus_indices, data.lcc.branch_admittances)
            for (bus_ix, y_val) in zip(bus_indices, self_admittances)
                e_i = e_state[bus_ix]
                f_i = f_state[bus_ix]
                g = real(y_val)
                b = imag(y_val)
                YVr[bus_ix] += g * e_i - b * f_i   # Re(Y_lcc·V)
                YVi[bus_ix] += g * f_i + b * e_i   # Im(Y_lcc·V)
            end
        end
    end
    for (ref_bus, subnetwork_buses) in subnetworks
        # pass 1: total slack. Buffer covers ALL REF/PV buses (pass 2 writes
        # even non-participating c_k==0 ones); slack accumulates only for c_k!=0.
        P_net_set_polar = zeros(Float64, length(subnetwork_buses))
        P_slack_total = 0.0
        for (idx, bus_k) in enumerate(subnetwork_buses)
            bt = bus_types[bus_k]
            (bt == PSY.ACBusTypes.REF || bt == PSY.ACBusTypes.PV) || continue
            # Polar-convention net active set-point (initial) for this bus.
            P_net_set_polar_k =
                data.bus_active_power_injections[bus_k, time_step] -
                get_bus_active_power_total_withdrawals(data, bus_k, time_step) +
                data.bus_hvdc_net_power[bus_k, time_step]
            P_net_set_polar[idx] = P_net_set_polar_k
            bus_slack_participation_factors[bus_k] == 0.0 && continue
            # S_net,k = V_k · conj((Y_raw·V)_k); real part used here.
            P_net_k = e_state[bus_k] * YVr[bus_k] + f_state[bus_k] * YVi[bus_k]
            P_slack_total += P_net_k - P_net_set_polar_k
        end
        # pass 2: distribute + write (P_slack_total must be complete first)
        for (idx, bus_k) in enumerate(subnetwork_buses)
            c_k = bus_slack_participation_factors[bus_k]
            bt = bus_types[bus_k]
            (bt == PSY.ACBusTypes.REF || bt == PSY.ACBusTypes.PV) || continue
            P_net = P_net_set_polar[idx] + c_k * P_slack_total
            # Q_net,k = imag(V_k · conj((Y_raw·V)_k)).
            Q_net = f_state[bus_k] * YVr[bus_k] - e_state[bus_k] * YVi[bus_k]
            # Identical to polar `_setpq`.
            data.bus_active_power_injections[bus_k, time_step] =
                P_net +
                get_bus_active_power_total_withdrawals(data, bus_k, time_step) -
                data.bus_hvdc_net_power[bus_k, time_step]
            data.bus_reactive_power_injections[bus_k, time_step] =
                Q_net +
                get_bus_reactive_power_total_withdrawals(data, bus_k, time_step)
        end
    end
    return
end
