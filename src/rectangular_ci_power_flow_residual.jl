"""
    struct ACRectangularCIResidual

Residual functor for the augmented current-injection (rectangular) AC power flow.
Mirrors [`ACPowerFlowResidual`](@ref) but operates on the per-bus variable-block
state representation: PQ/REF blocks are 2 entries `(e,f)` or `(P_gen, Q_gen)`;
PV blocks are 3 entries `(e, f, Q)`.

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
struct ACRectangularCIResidual
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
    # reads these so it sees the SAME (e, f, Q, P_gen, Q_gen, V_set²) values
    # that the residual used. This is necessary because data.bus_magnitude
    # holds V_set for PV buses (not |V_state|), and we need both available.
    e_state::Vector{Float64}            # length n_buses; current real(V)
    f_state::Vector{Float64}            # length n_buses; current imag(V)
    Q_state::Vector{Float64}            # length n_buses; PV's Q state (else NaN)
    P_eff_cache::Vector{Float64}        # length n_buses; effective P at current iter
    Q_eff_cache::Vector{Float64}        # length n_buses; effective Q at current iter
end

function ACRectangularCIResidual(data::ACPowerFlowData, time_step::Int64)
    n_buses = first(size(data.bus_type))
    n_lccs = size(data.lcc.p_set, 1)
    bus_type = view(data.bus_type, :, time_step)

    offsets, block_sizes, total_bus_state = compute_bus_state_offsets(bus_type)
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

    return ACRectangularCIResidual(
        data,
        _update_rect_ci_residual_values!,
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
    )
end

function (R::ACRectangularCIResidual)(
    Rv::Vector{Float64},
    x::Vector{Float64},
    time_step::Int64,
)
    R.Rf!(R.Rv, x, R.Y_bus_eff, R.P_net_const, R.Q_net_const,
        R.const_I_P, R.const_I_Q, R.P_net_set,
        R.bus_slack_participation_factors, R.subnetworks,
        R.bus_state_offset, R.bus_block_size, R.total_bus_state,
        R.e_state, R.f_state, R.Q_state, R.P_eff_cache, R.Q_eff_cache,
        R.data, time_step)
    copyto!(Rv, R.Rv)
    return
end

function (R::ACRectangularCIResidual)(x::Vector{Float64}, time_step::Int64)
    R.Rf!(R.Rv, x, R.Y_bus_eff, R.P_net_const, R.Q_net_const,
        R.const_I_P, R.const_I_Q, R.P_net_set,
        R.bus_slack_participation_factors, R.subnetworks,
        R.bus_state_offset, R.bus_block_size, R.total_bus_state,
        R.e_state, R.f_state, R.Q_state, R.P_eff_cache, R.Q_eff_cache,
        R.data, time_step)
    return
end

"""
Update residual values F for the augmented current-injection formulation.

Strategy: walk Y_bus_eff once to accumulate `Y·V` into the F slots, then add the
specified-current contribution per bus type. PV buses add the ΔV² row. The 4
LCC tail residuals are appended. ZIP constant-Z is already folded into Y_bus_eff
at setup, so it contributes via Y·V. ZIP constant-current is subtracted from the
effective P/Q before computing I_spec.
"""
function _update_rect_ci_residual_values!(
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
    Q_state::Vector{Float64},
    P_eff_cache::Vector{Float64},
    Q_eff_cache::Vector{Float64},
    data::ACPowerFlowData,
    time_step::Int64,
)
    n_buses = first(size(data.bus_type))
    n_lccs = size(data.lcc.p_set, 1)
    bus_types = view(data.bus_type, :, time_step)

    # 1) Push state into data (only PQ updates bus_magnitude; PV preserves V_set).
    rect_update_data!(data, x, bus_state_offset, bus_block_size, time_step)
    # Populate state caches before LCC admittance refresh (LCC needs |V_state|).
    @inbounds for i in 1:n_buses
        off = Int(bus_state_offset[i])
        bt = bus_types[i]
        if bt == PSY.ACBusTypes.REF
            Vm = data.bus_magnitude[i, time_step]
            θ = data.bus_angles[i, time_step]
            e_state[i] = Vm * cos(θ)
            f_state[i] = Vm * sin(θ)
            Q_state[i] = NaN
        else
            e_state[i] = x[off]
            f_state[i] = x[off + 1]
            Q_state[i] = bt == PSY.ACBusTypes.PV ? x[off + 2] : NaN
        end
    end
    if n_lccs > 0
        # At PV buses, data.bus_magnitude holds V_set (not the state magnitude).
        # Override the magnitude provider so LCC math sees |V_state|, keeping the
        # residual consistent with the Jacobian which operates on (e_state, f_state).
        _update_ybus_lcc!(data, time_step;
            vm_fn = (i, _) -> sqrt(e_state[i]^2 + f_state[i]^2))
    end

    # 2) Compute P_eff / Q_eff (slack distribution + ZIP constant-current correction).
    # ZIP constant-Z is folded into `Y_bus_eff` at setup (see `fold_zip_constant_z!`
    # in `rectangular_ci_setup.jl`), so only constant-P and constant-I appear here.
    @inbounds for i in 1:n_buses
        # ZIP const-I uses |V_state| (state magnitude), not V_set.
        Vm = sqrt(e_state[i]^2 + f_state[i]^2)
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

    # 3) Initialize F and accumulate -I_inj from Y_bus_eff*V.
    fill!(F, 0.0)
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
            row_off = Int(bus_state_offset[row])
            F[row_off] -= (g * e_col - b * f_col)   # -Re(Y·V)
            F[row_off + 1] -= (g * f_col + b * e_col)   # -Im(Y·V)
        end
    end

    # 4) Add per-bus I_spec contributions and PV's ΔV² row.
    # NOTE on REF distributed slack: x[off] holds `P_net_set[ref] + total_slack`
    # (polar convention — the state variable carries the WHOLE subnetwork slack,
    # not just REF's share). REF's actual P_gen is `P_net_set + c_ref · total_slack`.
    # For the default case c_ref = 1, this collapses to x[off].
    @inbounds for i in 1:n_buses
        off = Int(bus_state_offset[i])
        bt = bus_types[i]
        e_i = e_state[i]
        f_i = f_state[i]
        V_sq = e_i^2 + f_i^2
        if bt == PSY.ACBusTypes.REF
            c_ref = bus_slack_participation_factors[i]
            P_slack_total = x[off] - P_net_set[i]
            P_gen = P_net_set[i] + c_ref * P_slack_total
            Q_gen = x[off + 1]
            # |V| at REF is fixed at V_set; subtract the ZIP constant-current draw
            # so the recovered injection matches polar's `bus_active_power_injections`
            # (which includes `const_I * V_set` via `get_bus_active_power_total_withdrawals`).
            Vm = sqrt(V_sq)
            P_eff = P_gen - const_I_P[i] * Vm
            Q_eff = Q_gen - const_I_Q[i] * Vm
            F[off] += (P_eff * e_i + Q_eff * f_i) / V_sq
            F[off + 1] += (P_eff * f_i - Q_eff * e_i) / V_sq
        else
            P_i = P_eff_cache[i]
            Q_i = bt == PSY.ACBusTypes.PV ? Q_state[i] : Q_eff_cache[i]
            F[off] += (P_i * e_i + Q_i * f_i) / V_sq
            F[off + 1] += (P_i * f_i - Q_i * e_i) / V_sq
            if bt == PSY.ACBusTypes.PV
                # V_set² stored in data.bus_magnitude (preserved by rect_update_data!).
                V_set_sq = data.bus_magnitude[i, time_step]^2
                F[off + 2] = V_set_sq - V_sq
            end
        end
    end

    # 5) LCC current contribution at AC-side buses (mirrors polar's
    #    F[ΔP] += |V|²·G_lcc, F[ΔQ] += -|V|²·B_lcc translated to current mismatch:
    #    F[ΔI_r] -= Re(Y_lcc·V_fb), F[ΔI_i] -= Im(Y_lcc·V_fb)).
    if n_lccs > 0
        for (bus_indices, self_admittances) in
            zip(data.lcc.bus_indices, data.lcc.branch_admittances)
            for (bus_ix, y_val) in zip(bus_indices, self_admittances)
                e_i = e_state[bus_ix]
                f_i = f_state[bus_ix]
                g = real(y_val)
                b = imag(y_val)
                off_i = Int(bus_state_offset[bus_ix])
                F[off_i] -= g * e_i - b * f_i        # -Re(Y_lcc · V)
                F[off_i + 1] -= g * f_i + b * e_i    # -Im(Y_lcc · V)
            end
        end
    end

    # 6) LCC tail residuals (same formulas as polar code, but using |V_state|).
    if n_lccs > 0
        for i in 1:n_lccs
            offset_lcc = total_bus_state + (i - 1) * 4
            (fb, tb) = data.lcc.bus_indices[i]
            tap_r = data.lcc.rectifier.tap[i, time_step]
            tap_i = data.lcc.inverter.tap[i, time_step]
            phi_r = data.lcc.rectifier.phi[i, time_step]
            phi_i = data.lcc.inverter.phi[i, time_step]
            i_dc = data.lcc.i_dc[i, time_step]
            Vm_fb_state = sqrt(e_state[fb]^2 + f_state[fb]^2)
            Vm_tb_state = sqrt(e_state[tb]^2 + f_state[tb]^2)
            P_lcc_from = Vm_fb_state * tap_r * SQRT6_DIV_PI * i_dc * cos(phi_r)
            P_lcc_to = Vm_tb_state * tap_i * SQRT6_DIV_PI * i_dc * cos(phi_i)
            F[offset_lcc + 1] = if data.lcc.setpoint_at_rectifier[i]
                P_lcc_from - data.lcc.p_set[i, time_step]
            else
                -P_lcc_to - data.lcc.p_set[i, time_step]
            end
            F[offset_lcc + 2] =
                P_lcc_from + P_lcc_to -
                data.lcc.dc_line_resistance[i] * i_dc^2
            F[offset_lcc + 3] =
                data.lcc.rectifier.thyristor_angle[i, time_step] -
                data.lcc.rectifier.min_thyristor_angle[i]
            F[offset_lcc + 4] =
                data.lcc.inverter.thyristor_angle[i, time_step] -
                data.lcc.inverter.min_thyristor_angle[i]
        end
    end
    return
end
