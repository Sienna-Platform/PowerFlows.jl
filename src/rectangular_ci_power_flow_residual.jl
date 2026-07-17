"""
    struct ACRectangularCIResidual

Residual functor for the augmented current-injection (rectangular) AC power flow.
Mirrors [`ACPowerFlowResidual`](@ref) but operates on the per-bus variable-block
state representation: PQ/REF blocks are 2 entries `(e,f)` or `(P_gen, Q_gen)`;
PV blocks are 3 entries `(e, f, Q)`.

# Fields
- `data::ACPowerFlowData`
- `Rv::Vector{Float64}` — current residual values, length `total_bus_state + 4·n_LCC`
- `Y_bus_eff::SparseMatrixCSC{ComplexF64, Int}` — Y_bus with ZIP constant-Z folded in
- `P_net_const::Vector{Float64}` — constant-power net injection (no |V| dependence)
- `Q_net_const::Vector{Float64}` — constant-power net reactive injection
- `const_I_P::Vector{Float64}` — constant-current P-withdrawal coefficient per bus
- `const_I_Q::Vector{Float64}` — constant-current Q-withdrawal coefficient per bus
- `P_net_set::Vector{Float64}` — initial P_net for distributed-slack delta computation
- `bus_slack_participation_factors::SparseVector{Float64, Int}`
- `subnetworks::Dict{Int64, Vector{Int64}}`
- `independent_ref::Set{Int}` — REF buses that share an island with another REF
  (multi-swing); precomputed once here (bus REF-status is fixed across a solve)
  so the hot per-iteration path never allocates a `Set`.
- `bus_state_offset::Vector{REC_INDEX_TYPE}`
- `bus_block_size::Vector{Int8}`
- `total_bus_state::Int`
- `validate_offsets::Vector{Int}` — precomputed `x`-offsets of PQ/PV buses for
  the per-iteration voltage-magnitude diagnostic
"""
struct ACRectangularCIResidual
    data::ACPowerFlowData
    Rv::Vector{Float64}
    Y_bus_eff::SparseMatrixCSC{ComplexF64, Int}
    P_net_const::Vector{Float64}
    Q_net_const::Vector{Float64}
    const_I_P::Vector{Float64}
    const_I_Q::Vector{Float64}
    P_net_set::Vector{Float64}
    bus_slack_participation_factors::SparseVector{Float64, Int}
    subnetworks::Dict{Int64, Vector{Int64}}
    independent_ref::Set{Int}
    bus_state_offset::Vector{REC_INDEX_TYPE}
    bus_block_size::Vector{Int8}
    total_bus_state::Int
    validate_offsets::Vector{Int}
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
    validate_offsets = _pqpv_validate_offsets(bus_type, offsets)
    total_state = total_bus_state + state_tail_length(data, get_dc_network(data))

    P_net_const = Vector{Float64}(undef, n_buses)
    Q_net_const = Vector{Float64}(undef, n_buses)
    const_I_P = Vector{Float64}(undef, n_buses)
    const_I_Q = Vector{Float64}(undef, n_buses)
    P_net_set = Vector{Float64}(undef, n_buses)

    subnetworks =
        _find_subnetworks_for_reference_buses(data.power_network_matrix.data, bus_type)
    # REF status is fixed for the life of a solve, so this is computed once here
    # rather than per-iteration (see the `independent_ref` field docstring).
    independent_ref = _multi_swing_ref_indices(data.bus_type, subnetworks, time_step)

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
        Vector{Float64}(undef, total_state),
        Y_bus_eff,
        P_net_const,
        Q_net_const,
        const_I_P,
        const_I_Q,
        P_net_set,
        bus_slack_participation_factors,
        subnetworks,
        independent_ref,
        offsets,
        block_sizes,
        total_bus_state,
        validate_offsets,
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
    _update_rect_ci_residual_values!(R.Rv, x, R.Y_bus_eff, R.P_net_const, R.Q_net_const,
        R.const_I_P, R.const_I_Q, R.P_net_set,
        R.bus_slack_participation_factors, R.subnetworks, R.independent_ref,
        R.bus_state_offset, R.bus_block_size, R.total_bus_state,
        R.e_state, R.f_state, R.Q_state, R.P_eff_cache, R.Q_eff_cache,
        R.data, time_step)
    copyto!(Rv, R.Rv)
    return
end

function (R::ACRectangularCIResidual)(x::Vector{Float64}, time_step::Int64)
    _update_rect_ci_residual_values!(R.Rv, x, R.Y_bus_eff, R.P_net_const, R.Q_net_const,
        R.const_I_P, R.const_I_Q, R.P_net_set,
        R.bus_slack_participation_factors, R.subnetworks, R.independent_ref,
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
    independent_ref::Set{Int},
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
        # PV buses store V_set in data.bus_magnitude; use the rect form so LCC math
        # sees |V_state| = sqrt(e² + f²), matching the rectangular Jacobian.
        _update_ybus_lcc!(data, time_step, e_state, f_state)
    end

    # 2) Compute P_eff / Q_eff (slack distribution + ZIP constant-current correction).
    # ZIP constant-Z is folded into `Y_bus_eff` at setup (see `fold_zip_constant_z!`
    # in `rectangular_ci_setup.jl`), so only constant-P and constant-I appear here.
    @inbounds for i in 1:n_buses
        # ZIP const-I uses |V_state|; V_FLOOR2 (1e-16) guards 1/|V|². The floor only
        # trips at degenerate |V| < 1e-8 pu (never near a solution), where the Jacobian
        # keeps the unfloored derivative: inexact but finite and |V|-restoring, and
        # harmless since the iteration never converges there.
        Vm = sqrt(max(e_state[i]^2 + f_state[i]^2, V_FLOOR2))
        P_eff_cache[i] = P_net_const[i] - const_I_P[i] * Vm
        Q_eff_cache[i] = Q_net_const[i] - const_I_Q[i] * Vm
    end
    for (ref_bus, subnetwork_buses) in subnetworks
        # An island with more than one swing (REF) bus holds each swing at its own
        # fixed complex voltage, so each swing carries its OWN slack (handled in the
        # REF branch below, using x[off] directly); no slack is distributed to any
        # other bus in that island. Single-swing islands keep the distributed path.
        ref_bus in independent_ref && continue
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
        for j in SparseArrays.nzrange(Y_bus_eff, col)
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
        # V_FLOOR2 floor: a degenerate (e,f) (warm/flat start) must not blow up
        # 1/|V|². PV's |V|² row below keeps raw V_sq (−2e/−2f Jacobian is exact).
        D = max(V_sq, V_FLOOR2)
        if bt == PSY.ACBusTypes.REF
            if i in independent_ref
                # Multi-swing island: this swing self-balances at its own P-slot
                # (∂P_gen/∂x[off] = 1), not the distributed c_ref share.
                P_gen = x[off]
            else
                c_ref = bus_slack_participation_factors[i]
                P_slack_total = x[off] - P_net_set[i]
                P_gen = P_net_set[i] + c_ref * P_slack_total
            end
            Q_gen = x[off + 1]
            # |V| at REF is fixed at V_set; subtract the ZIP constant-current draw
            # so the recovered injection matches polar's `bus_active_power_injections`
            # (which includes `const_I * V_set` via `get_bus_active_power_total_withdrawals`).
            Vm = sqrt(D)
            P_eff = P_gen - const_I_P[i] * Vm
            Q_eff = Q_gen - const_I_Q[i] * Vm
            F[off] += (P_eff * e_i + Q_eff * f_i) / D
            F[off + 1] += (P_eff * f_i - Q_eff * e_i) / D
        else
            P_i = P_eff_cache[i]
            # PV: Q_state is the net injection unknown — at convergence it equals
            # Q_gen − Q_load_total(|V_set|), so the ZIP-I term is implicit and a
            # `−const_I_Q·|V|` correction here would double-count. For PQ, Q is a
            # known input, so Q_eff_cache pre-subtracts the constant-current draw.
            Q_i = bt == PSY.ACBusTypes.PV ? Q_state[i] : Q_eff_cache[i]
            F[off] += (P_i * e_i + Q_i * f_i) / D
            F[off + 1] += (P_i * f_i - Q_i * e_i) / D
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

    # 6) LCC tail residuals — shared helper, with |V_state| instead of polar's
    #    bus_magnitude (V_set at PV).
    if n_lccs > 0
        _set_lcc_tail_residuals!(
            F, data, total_bus_state, time_step, e_state, f_state,
        )
    end

    # 7) VSC / DC-network tail: current injection at AC buses + control/DC-KCL rows.
    dcn = get_dc_network(data)
    if has_dc_network(dcn)
        vsc_off = total_bus_state + 4 * n_lccs
        _read_vsc_state!(dcn, x, vsc_off, time_step)
        _apply_vsc_bus_injections_rect!(
            F,
            dcn,
            e_state,
            f_state,
            bus_state_offset,
            time_step,
        )
        _set_vsc_tail_residuals_rect!(F, dcn, e_state, f_state, vsc_off, time_step)
    end
    return
end
