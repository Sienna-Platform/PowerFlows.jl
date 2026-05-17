"""
    struct ACMixedCPBResidual

Residual functor for the mixed current/power-balance (MCPB) AC power flow.
Mirrors [`ACRectangularCIResidual`](@ref) 1:1 (mirror-for-validation
convention) but uses the MCPB per-bus state layout where every bus — PQ, PV,
and REF — occupies exactly 2 state entries (no PV→3 expansion).

# Fields
- `data::ACPowerFlowData`
- `Rf!::Function` — inplace residual update
- `Rv::Vector{Float64}` — current residual values, length `total_bus_state + 4·n_LCC`
- `Y_bus_eff::SparseMatrixCSC{ComplexF64, Int}` — Y_bus with ZIP constant-Z folded in
- `P_net_const::Vector{Float64}` — constant-power net injection (no |V| dependence)
- `Q_net_const::Vector{Float64}` — constant-power net reactive injection
- `const_I_P::Vector{Float64}` — constant-current P-withdrawal coefficient per bus
- `const_I_Q::Vector{Float64}` — constant-current Q-withdrawal coefficient per bus
- `P_net_set::Vector{Float64}` — initial P_net for distributed-slack delta computation
- `bus_slack_participation_factors::SparseVector{Float64, Int}`
- `subnetworks::Dict{Int64, Vector{Int64}}`
- `bus_state_offset::Vector{REC_INDEX_TYPE}`
- `bus_block_size::Vector{Int8}`
- `total_bus_state::Int`
"""
struct ACMixedCPBResidual
    data::ACPowerFlowData
    Rf!::Function
    Rv::Vector{Float64}
    Y_bus_eff::SparseMatrixCSC{ComplexF64, Int}
    P_net_const::Vector{Float64}
    Q_net_const::Vector{Float64}
    const_I_P::Vector{Float64}
    const_I_Q::Vector{Float64}
    P_net_set::Vector{Float64}
    bus_slack_participation_factors::SparseVector{Float64, Int}
    subnetworks::Dict{Int64, Vector{Int64}}
    bus_state_offset::Vector{REC_INDEX_TYPE}
    bus_block_size::Vector{Int8}
    total_bus_state::Int
    # State caches mirrored from x at each residual evaluation. The Jacobian
    # reads these so it sees the SAME (e, f) values that the residual used.
    # This is necessary because data.bus_magnitude holds V_set for PV buses
    # (not |V_state|), and we need both available.
    e_state::Vector{Float64}            # length n_buses; current real(V)
    f_state::Vector{Float64}            # length n_buses; current imag(V)
    P_eff_cache::Vector{Float64}        # length n_buses; effective P at current iter
    Q_eff_cache::Vector{Float64}        # length n_buses; effective Q at current iter
    # Accumulated network current Re/Im per bus, written by the residual and
    # aliased by the MCPB Jacobian (the PV power row needs e·Ir + f·Ii).
    Ir_acc::Vector{Float64}             # length n_buses; accumulated Re(I)
    Ii_acc::Vector{Float64}             # length n_buses; accumulated Im(I)
end

function ACMixedCPBResidual(data::ACPowerFlowData, time_step::Int64)
    n_buses = first(size(data.bus_type))
    n_lccs = size(data.lcc.p_set, 1)
    bus_type = view(data.bus_type, :, time_step)

    offsets, block_sizes, total_bus_state = compute_mixed_bus_state_offsets(bus_type)
    total_state = total_bus_state + 4 * n_lccs

    P_net_const = Vector{Float64}(undef, n_buses)
    Q_net_const = Vector{Float64}(undef, n_buses)
    const_I_P = Vector{Float64}(undef, n_buses)
    const_I_Q = Vector{Float64}(undef, n_buses)
    P_net_set = Vector{Float64}(undef, n_buses)

    subnetworks =
        _find_subnetworks_for_reference_buses(data.power_network_matrix.data, bus_type)

    for ix in 1:n_buses
        # Constant-power net injection (no |V| dependence)
        P_net_const[ix] =
            data.bus_active_power_injections[ix, time_step] -
            data.bus_active_power_withdrawals[ix, time_step] +
            data.bus_hvdc_net_power[ix, time_step]
        Q_net_const[ix] =
            data.bus_reactive_power_injections[ix, time_step] -
            data.bus_reactive_power_withdrawals[ix, time_step]
        # ZIP constant-current coefficients (carried as withdrawals)
        const_I_P[ix] =
            data.bus_active_power_constant_current_withdrawals[ix, time_step]
        const_I_Q[ix] =
            data.bus_reactive_power_constant_current_withdrawals[ix, time_step]
        # P_net_set tracks the initial P injection at setup (for slack delta)
        P_net_set[ix] = P_net_const[ix] -
                        const_I_P[ix] * data.bus_magnitude[ix, time_step]
    end

    bus_slack_participation_factors =
        _build_bus_slack_participation_factors(data, bus_type, subnetworks, time_step)

    # Build Y_bus_eff: copy Y_bus + fold constant-Z ZIP loads
    Y = data.power_network_matrix.data
    Y_bus_eff = SparseArrays.sparse(ComplexF64.(Y))
    fold_zip_constant_z!(Y_bus_eff, data, time_step)

    return ACMixedCPBResidual(
        data,
        _update_mixed_cpb_residual_values!,
        Vector{Float64}(undef, total_state),
        Y_bus_eff,
        P_net_const,
        Q_net_const,
        const_I_P,
        const_I_Q,
        P_net_set,
        bus_slack_participation_factors,
        subnetworks,
        offsets,
        block_sizes,
        total_bus_state,
        Vector{Float64}(undef, n_buses),
        Vector{Float64}(undef, n_buses),
        Vector{Float64}(undef, n_buses),
        Vector{Float64}(undef, n_buses),
        Vector{Float64}(undef, n_buses),
        Vector{Float64}(undef, n_buses),
    )
end

function (R::ACMixedCPBResidual)(
    Rv::Vector{Float64},
    x::Vector{Float64},
    time_step::Int64,
)
    R.Rf!(R.Rv, x, R.Y_bus_eff, R.P_net_const, R.Q_net_const,
        R.const_I_P, R.const_I_Q, R.P_net_set,
        R.bus_slack_participation_factors, R.subnetworks,
        R.bus_state_offset, R.bus_block_size, R.total_bus_state,
        R.e_state, R.f_state, R.P_eff_cache, R.Q_eff_cache,
        R.data, time_step, R.Ir_acc, R.Ii_acc)
    copyto!(Rv, R.Rv)
    return
end

function (R::ACMixedCPBResidual)(x::Vector{Float64}, time_step::Int64)
    R.Rf!(R.Rv, x, R.Y_bus_eff, R.P_net_const, R.Q_net_const,
        R.const_I_P, R.const_I_Q, R.P_net_set,
        R.bus_slack_participation_factors, R.subnetworks,
        R.bus_state_offset, R.bus_block_size, R.total_bus_state,
        R.e_state, R.f_state, R.P_eff_cache, R.Q_eff_cache,
        R.data, time_step, R.Ir_acc, R.Ii_acc)
    return
end

"""
Update residual values `F` for the mixed current/power-balance (MCPB)
formulation (Mixed Current-Power Balance, rectangular — paper §IV).

Mirrors [`_update_rect_ci_residual_values!`](@ref) step-for-step so the MCPB
Jacobian can reuse rect's machinery mechanically. Differences from rect:

- The network current `Y_bus_eff·V` (+ LCC) is accumulated into the per-bus
  `Ir_acc`/`Ii_acc` (positive current), NOT subtracted into `F`. The sign
  convention `residual = I_spec_term − I_network_accumulated` matches rect's
  sign so the Jacobian mirrors rect.
- Per-bus blocks are uniformly 2 slots (no PV→3 expansion):
  - **PQ**: divided-current balance with the two slots SWAPPED to IMAG-FIRST
    ordering (paper §IV) so nonzero `B_ii` lands on the block diagonal.
  - **PV**: real-power balance (eq.7) in slot 0, `|V|² − V_set²` (eq.8) in slot 1.
  - **REF**: copied verbatim from rect's REF branch (NOT imag-first) so rect's
    `_update_ref_diag_block!` is reusable.
"""
function _update_mixed_cpb_residual_values!(
    F::Vector{Float64},
    x::Vector{Float64},
    Y_bus_eff::SparseMatrixCSC{ComplexF64, Int},
    P_net_const::Vector{Float64},
    Q_net_const::Vector{Float64},
    const_I_P::Vector{Float64},
    const_I_Q::Vector{Float64},
    P_net_set::Vector{Float64},
    bus_slack_participation_factors::SparseVector{Float64, Int},
    subnetworks::Dict{Int64, Vector{Int64}},
    bus_state_offset::Vector{REC_INDEX_TYPE},
    bus_block_size::Vector{Int8},
    total_bus_state::Int,
    e_state::Vector{Float64},
    f_state::Vector{Float64},
    P_eff_cache::Vector{Float64},
    Q_eff_cache::Vector{Float64},
    data::ACPowerFlowData,
    time_step::Int64,
    Ir_acc::Vector{Float64},
    Ii_acc::Vector{Float64},
)
    n_buses = first(size(data.bus_type))
    n_lccs = size(data.lcc.p_set, 1)
    bus_types = view(data.bus_type, :, time_step)

    # 1) Push state into data (MCPB: PQ updates bus_magnitude; PV/REF preserve
    #    V_set). Populate state caches before LCC admittance refresh.
    mixed_update_data!(data, x, bus_state_offset, bus_block_size, time_step)
    @inbounds for i in 1:n_buses
        off = Int(bus_state_offset[i])
        bt = bus_types[i]
        if bt == PSY.ACBusTypes.REF
            Vm = data.bus_magnitude[i, time_step]
            θ = data.bus_angles[i, time_step]
            e_state[i] = Vm * cos(θ)
            f_state[i] = Vm * sin(θ)
        else
            e_state[i] = x[off]
            f_state[i] = x[off + 1]
        end
    end
    if n_lccs > 0
        _update_ybus_lcc!(data, time_step, e_state, f_state)
    end

    # 2) Compute P_eff / Q_eff (slack distribution + ZIP constant-current
    #    correction). ZIP constant-Z is folded into `Y_bus_eff` at setup.
    @inbounds for i in 1:n_buses
        Vm_sq = max(e_state[i]^2 + f_state[i]^2, V_FLOOR2)
        Vm = sqrt(Vm_sq)
        P_eff_cache[i] = P_net_const[i] - const_I_P[i] * Vm
        Q_eff_cache[i] = Q_net_const[i] - const_I_Q[i] * Vm
    end
    for (ref_bus, subnetwork_buses) in subnetworks
        ref_off = Int(bus_state_offset[ref_bus])
        P_slack_total = x[ref_off] - P_net_set[ref_bus]
        for bus_k in subnetwork_buses
            c_k = bus_slack_participation_factors[bus_k]
            c_k == 0.0 && continue
            bus_k == ref_bus && continue
            P_eff_cache[bus_k] += c_k * P_slack_total
        end
    end

    # 3) Accumulate +I_network = Y_bus_eff·V into Ir_acc/Ii_acc (NOT into F).
    #    rect subtracts -Re/Im(Y·V) into F; MCPB accumulates the POSITIVE
    #    current here, and the residual is I_spec − I_network below.
    fill!(Ir_acc, 0.0)
    fill!(Ii_acc, 0.0)
    Yvals = SparseArrays.nonzeros(Y_bus_eff)
    Yrows = SparseArrays.rowvals(Y_bus_eff)
    @inbounds for col in 1:n_buses
        e_col = e_state[col]
        f_col = f_state[col]
        for j in Y_bus_eff.colptr[col]:(Y_bus_eff.colptr[col + 1] - 1)
            row = Yrows[j]
            y = Yvals[j]
            g = real(y)
            b = imag(y)
            Ir_acc[row] += g * e_col - b * f_col   # Re(Y·V)
            Ii_acc[row] += g * f_col + b * e_col   # Im(Y·V)
        end
    end

    # 4) LCC current at AC-side buses, accumulated into Ir_acc/Ii_acc so the
    #    PQ/REF current balance AND the PV power row (e·Ir + f·Ii) all see it.
    if n_lccs > 0
        for (bus_indices, self_admittances) in
            zip(data.lcc.bus_indices, data.lcc.branch_admittances)
            for (bus_ix, y_val) in zip(bus_indices, self_admittances)
                e_i = e_state[bus_ix]
                f_i = f_state[bus_ix]
                g = real(y_val)
                b = imag(y_val)
                Ir_acc[bus_ix] += g * e_i - b * f_i   # Re(Y_lcc·V)
                Ii_acc[bus_ix] += g * f_i + b * e_i   # Im(Y_lcc·V)
            end
        end
    end

    # 5) Per-bus residual blocks. residual = I_spec_term − I_network_accumulated.
    fill!(F, 0.0)  # zeroed here (not at step 3) — accumulation is in Ir_acc/Ii_acc
    @inbounds for i in 1:n_buses
        off = Int(bus_state_offset[i])
        bt = bus_types[i]
        e_i = e_state[i]
        f_i = f_state[i]
        D = max(e_i^2 + f_i^2, V_FLOOR2)
        if bt == PSY.ACBusTypes.REF
            # REF: copied VERBATIM from rect's REF branch (slots NOT swapped)
            # so rect's `_update_ref_diag_block!` is reusable.
            c_ref = bus_slack_participation_factors[i]
            P_slack_total = x[off] - P_net_set[i]
            P_gen = P_net_set[i] + c_ref * P_slack_total
            Q_gen = x[off + 1]
            Vm = sqrt(D)
            P_eff = P_gen - const_I_P[i] * Vm
            Q_eff = Q_gen - const_I_Q[i] * Vm
            F[off] = (P_eff * e_i + Q_eff * f_i) / D - Ir_acc[i]
            F[off + 1] = (P_eff * f_i - Q_eff * e_i) / D - Ii_acc[i]
        elseif bt == PSY.ACBusTypes.PV
            # PV: real-power balance (eq.7) + |V|² constraint (eq.8).
            P_i = P_eff_cache[i]
            F[off] = e_i * Ir_acc[i] + f_i * Ii_acc[i] - P_i
            V_set_sq = data.bus_magnitude[i, time_step]^2
            F[off + 1] = (e_i^2 + f_i^2) - V_set_sq
        else  # PQ
            # PQ: divided-current balance, IMAG-FIRST ordering (paper §IV) —
            # rect's two PQ rows with the two slots SWAPPED so nonzero B_ii
            # lands on the block diagonal.
            P_i = P_eff_cache[i]
            Q_i = Q_eff_cache[i]
            F[off] = (P_i * f_i - Q_i * e_i) / D - Ii_acc[i]   # imag → slot 0
            F[off + 1] = (P_i * e_i + Q_i * f_i) / D - Ir_acc[i]  # real → slot 1
        end
    end

    # 6) LCC tail residuals — shared rect helper, unchanged.
    if n_lccs > 0
        _set_lcc_tail_residuals!(
            F, data, total_bus_state, time_step, e_state, f_state,
        )
    end
    return
end
