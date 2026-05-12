"""
    struct ACRectangularCIJacobian

Jacobian functor for the augmented current-injection (rectangular) AC power flow.
Mirrors [`ACPowerFlowJacobian`](@ref) but operates on the per-bus variable-block
state representation.

# Fields
- `data::ACPowerFlowData`
- `Jf!::Function` — inplace Jacobian update
- `Jv::SparseMatrixCSC{Float64, J_INDEX_TYPE}` — Jacobian values
- `Y_bus_eff::SparseMatrixCSC{ComplexF64, Int}` — Y_bus with ZIP-Z folded in
- `P_net_const::Vector{Float64}`
- `Q_net_const::Vector{Float64}`
- `const_I_P::Vector{Float64}`
- `const_I_Q::Vector{Float64}`
- `P_net_set::Vector{Float64}`
- `bus_slack_participation_factors::SparseVector{Float64, Int}`
- `subnetworks::Dict{Int64, Vector{Int64}}`
- `bus_state_offset::Vector{REC_INDEX_TYPE}`
- `bus_block_size::Vector{Int8}`
- `total_bus_state::Int`
"""
struct ACRectangularCIJacobian
    data::ACPowerFlowData
    Jf!::Function
    Jv::SparseMatrixCSC{Float64, J_INDEX_TYPE}
    Y_bus_eff::SparseMatrixCSC{ComplexF64, Int}
    Y_diag::Vector{ComplexF64}     # cached Y_bus_eff diagonal; avoids O(log nnz) sparse access per iteration
    e_state::Vector{Float64}       # shared view into residual's e_state
    f_state::Vector{Float64}       # shared view into residual's f_state
    Q_state::Vector{Float64}       # shared view into residual's Q_state
    P_eff_cache::Vector{Float64}   # shared view into residual's P_eff_cache
    Q_eff_cache::Vector{Float64}   # shared view into residual's Q_eff_cache (PQ-bus ZIP-corrected Q)
    bus_slack_participation_factors::SparseVector{Float64, Int}
    subnetworks::Dict{Int64, Vector{Int64}}
    bus_state_offset::Vector{REC_INDEX_TYPE}
    bus_block_size::Vector{Int8}
    total_bus_state::Int
end

function ACRectangularCIJacobian(
    residual::ACRectangularCIResidual,
    time_step::Int64,
)
    Jv0 = _create_rect_ci_jacobian_structure(
        residual.data,
        residual.Y_bus_eff,
        residual.bus_slack_participation_factors,
        residual.subnetworks,
        residual.bus_state_offset,
        residual.bus_block_size,
        residual.total_bus_state,
        time_step,
    )
    # Populate the constant entries (Y_bus off-diagonals and REF row blocks).
    _populate_constant_yb_blocks!(
        Jv0,
        residual.Y_bus_eff,
        residual.bus_state_offset,
        view(residual.data.bus_type, :, time_step),
    )
    n_buses = first(size(residual.data.bus_type))
    Y_diag = Vector{ComplexF64}(undef, n_buses)
    @inbounds for i in 1:n_buses
        Y_diag[i] = residual.Y_bus_eff[i, i]
    end
    J = ACRectangularCIJacobian(
        residual.data,
        _update_rect_ci_jacobian_values!,
        Jv0,
        residual.Y_bus_eff,
        Y_diag,
        residual.e_state,
        residual.f_state,
        residual.Q_state,
        residual.P_eff_cache,
        residual.Q_eff_cache,
        residual.bus_slack_participation_factors,
        residual.subnetworks,
        residual.bus_state_offset,
        residual.bus_block_size,
        residual.total_bus_state,
    )
    J(time_step)  # populate state-dependent entries (diagonals, slack, LCC tail)
    return J
end

function (J::ACRectangularCIJacobian)(time_step::Int64)
    J.Jf!(J.Jv, J.data, J.Y_bus_eff, J.Y_diag,
        J.e_state, J.f_state, J.Q_state, J.P_eff_cache, J.Q_eff_cache,
        J.bus_slack_participation_factors, J.subnetworks,
        J.bus_state_offset, J.bus_block_size, J.total_bus_state, time_step)
    return
end

function (J::ACRectangularCIJacobian)(
    Jv::SparseMatrixCSC{Float64, J_INDEX_TYPE},
    time_step::Int64,
)
    J.Jf!(J.Jv, J.data, J.Y_bus_eff, J.Y_diag,
        J.e_state, J.f_state, J.Q_state, J.P_eff_cache, J.Q_eff_cache,
        J.bus_slack_participation_factors, J.subnetworks,
        J.bus_state_offset, J.bus_block_size, J.total_bus_state, time_step)
    copyto!(Jv, J.Jv)
    return
end

"""
Build the sparsity pattern for the rectangular CI Jacobian. Per-bus blocks
have variable size (2 or 3). Off-diagonal blocks have entries only for
the `(e, f)` columns of the neighbor (current injection from neighbor is
independent of neighbor's Q variable). Slack cross-terms and LCC tail
entries are added with structural zeros.
"""
function _create_rect_ci_jacobian_structure(
    data::ACPowerFlowData,
    Y_bus_eff::SparseMatrixCSC{ComplexF64, Int},
    bus_slack_participation_factors::SparseVector{Float64, Int},
    subnetworks::Dict{Int64, Vector{Int64}},
    bus_state_offset::Vector{REC_INDEX_TYPE},
    bus_block_size::Vector{Int8},
    total_bus_state::Int,
    time_step::Int64,
)
    rows = J_INDEX_TYPE[]
    cols = J_INDEX_TYPE[]
    vals = Float64[]
    n_buses = first(size(data.bus_type))
    n_lccs = size(data.lcc.p_set, 1)
    total_state = total_bus_state + 4 * n_lccs

    sizehint!(rows, 4 * SparseArrays.nnz(Y_bus_eff) + 17 * n_lccs + 2 * n_buses)
    sizehint!(cols, 4 * SparseArrays.nnz(Y_bus_eff) + 17 * n_lccs + 2 * n_buses)
    sizehint!(vals, 4 * SparseArrays.nnz(Y_bus_eff) + 17 * n_lccs + 2 * n_buses)

    Yrows = SparseArrays.rowvals(Y_bus_eff)
    @inbounds for col in 1:n_buses
        col_off = Int(bus_state_offset[col])
        col_bs = bus_block_size[col]
        for j in Y_bus_eff.colptr[col]:(Y_bus_eff.colptr[col + 1] - 1)
            row = Yrows[j]
            row_off = Int(bus_state_offset[row])
            row_bs = bus_block_size[row]
            # Off-diagonal blocks involve only (e, f) columns of the neighbor —
            # the Q column (for PV neighbors) has structural zeros in off-diagonals.
            # On diagonal block: full row_bs × col_bs.
            n_cols_to_write = (row == col) ? Int(col_bs) : 2  # off-diag: only e,f cols
            n_rows_to_write = Int(row_bs)
            for r in 0:(n_rows_to_write - 1)
                for c in 0:(n_cols_to_write - 1)
                    push!(rows, J_INDEX_TYPE(row_off + r))
                    push!(cols, J_INDEX_TYPE(col_off + c))
                    push!(vals, 0.0)
                end
            end
            # For PV columns (when row != col), we still need the Q column entries
            # for the diagonal block (∂I_spec/∂Q at the PV bus itself). Those are
            # captured by row == col case above. No additional entries needed off-diag.
        end
    end

    # Distributed-slack cross-terms: ∂F_k_{r,i}/∂x[bus_state_offset[ref]]
    for (ref_bus, subnetwork_buses) in subnetworks
        ref_off = Int(bus_state_offset[ref_bus])
        for bus_k in subnetwork_buses
            c_k = bus_slack_participation_factors[bus_k]
            c_k == 0.0 && continue
            bus_k == ref_bus && continue
            k_off = Int(bus_state_offset[bus_k])
            # Only add if not already in the Y_bus pattern between k and ref
            push!(rows, J_INDEX_TYPE(k_off))
            push!(cols, J_INDEX_TYPE(ref_off))
            push!(vals, 0.0)
            push!(rows, J_INDEX_TYPE(k_off + 1))
            push!(cols, J_INDEX_TYPE(ref_off))
            push!(vals, 0.0)
        end
    end

    # LCC tail entries: 17 per LCC, mirror polar structure.
    if n_lccs > 0
        _create_rect_ci_lcc_structure!(
            rows, cols, vals, data, bus_state_offset, total_bus_state,
        )
    end

    return SparseArrays.sparse(rows, cols, vals, total_state, total_state)
end

function _create_rect_ci_lcc_structure!(
    rows::Vector{J_INDEX_TYPE},
    cols::Vector{J_INDEX_TYPE},
    vals::Vector{Float64},
    data::ACPowerFlowData,
    bus_state_offset::Vector{REC_INDEX_TYPE},
    total_bus_state::Int,
)
    for (i, (fb, tb)) in enumerate(data.lcc.bus_indices)
        col_e_fb = Int(bus_state_offset[fb])
        col_f_fb = col_e_fb + 1
        col_e_tb = Int(bus_state_offset[tb])
        col_f_tb = col_e_tb + 1
        offset_lcc = total_bus_state + (i - 1) * 4
        idx_tap_r = offset_lcc + 1
        idx_tap_i = offset_lcc + 2
        idx_alpha_r = offset_lcc + 3
        idx_alpha_i = offset_lcc + 4
        # Cross-terms (LCC residual rows depend on AC bus (e, f) and LCC vars).
        # See spec §4 for the LCC tail; mirrors polar `_create_jacobian_matrix_structure_lcc`.
        rcv = [
            # FB-side bus rows × LCC tail cols (∂F_block_fb / ∂t_r, ∂α_r).
            (col_e_fb, idx_tap_r, 0.0),
            (col_e_fb, idx_alpha_r, 0.0),
            (col_f_fb, idx_tap_r, 0.0),
            (col_f_fb, idx_alpha_r, 0.0),
            # TB-side bus rows × LCC tail cols (∂F_block_tb / ∂t_i, ∂α_i).
            (col_e_tb, idx_tap_i, 0.0),
            (col_e_tb, idx_alpha_i, 0.0),
            (col_f_tb, idx_tap_i, 0.0),
            (col_f_tb, idx_alpha_i, 0.0),
            # LCC tail rows (F_t_r, F_t_i) × bus cols.
            (idx_tap_r, col_e_fb, 0.0),
            (idx_tap_r, col_f_fb, 0.0),
            (idx_tap_i, col_e_fb, 0.0),
            (idx_tap_i, col_f_fb, 0.0),
            (idx_tap_i, col_e_tb, 0.0),
            (idx_tap_i, col_f_tb, 0.0),
            # LCC tail rows × LCC tail cols.
            (idx_tap_r, idx_tap_r, 0.0),
            (idx_tap_r, idx_alpha_r, 0.0),
            (idx_tap_i, idx_tap_r, 0.0),
            (idx_tap_i, idx_tap_i, 0.0),
            (idx_tap_i, idx_alpha_r, 0.0),
            (idx_tap_i, idx_alpha_i, 0.0),
            (idx_alpha_r, idx_alpha_r, 1.0),
            (idx_alpha_i, idx_alpha_i, 1.0),
        ]
        for (r, c, v) in rcv
            push!(rows, J_INDEX_TYPE(r))
            push!(cols, J_INDEX_TYPE(c))
            push!(vals, v)
        end
    end
    return
end

"""
Populate the Y_bus off-diagonal blocks (constant across NR iterations) and the
REF row off-diagonal Y_bus blocks. These entries are filled once and not
touched during per-iteration updates.

Off-diagonal Y_bus block: F = I_spec − I_inj, so the contribution to the
Jacobian from `−I_inj = −Y·V` is the 2×2 real representation of `−Y_ij`:

    ∂(−I_inj_r)/∂e_j = −G_ij,  ∂(−I_inj_r)/∂f_j =  B_ij
    ∂(−I_inj_i)/∂e_j = −B_ij,  ∂(−I_inj_i)/∂f_j = −G_ij

For PV neighbor columns, only the (e, f) sub-columns are populated; the Q
column has structural zeros in off-diagonals (captured by pattern construction).
"""
function _populate_constant_yb_blocks!(
    Jv::SparseMatrixCSC{Float64, J_INDEX_TYPE},
    Y_bus_eff::SparseMatrixCSC{ComplexF64, Int},
    bus_state_offset::Vector{REC_INDEX_TYPE},
    bus_types::AbstractVector{PSY.ACBusTypes},
)
    n_buses = length(bus_types)
    Yvals = SparseArrays.nonzeros(Y_bus_eff)
    Yrows = SparseArrays.rowvals(Y_bus_eff)
    @inbounds for col in 1:n_buses
        # Skip REF columns: REF state vars are (P_gen, Q_gen), not (e, f). The Y_bus
        # contribution to F at neighbors uses REF's FIXED (e, f) values and does not
        # add any state-dependent entries to the Jacobian's REF column.
        bus_types[col] == PSY.ACBusTypes.REF && continue
        col_off = Int(bus_state_offset[col])
        for j in Y_bus_eff.colptr[col]:(Y_bus_eff.colptr[col + 1] - 1)
            row = Yrows[j]
            row == col && continue  # diagonal block handled per iteration
            row_off = Int(bus_state_offset[row])
            y = Yvals[j]
            g = real(y)
            b = imag(y)
            Jv[row_off, col_off] = -g
            Jv[row_off, col_off + 1] = b
            Jv[row_off + 1, col_off] = -b
            Jv[row_off + 1, col_off + 1] = -g
        end
    end
    return
end

"""Update state-dependent Jacobian entries: per-bus diagonal blocks, slack
cross-terms, LCC tail entries. Reads state from the residual's state caches
(`e_state`, `f_state`, `Q_state`, `P_eff_cache`, `Q_eff_cache`) — these must
be up to date (call the residual on `x` first)."""
function _update_rect_ci_jacobian_values!(
    Jv::SparseMatrixCSC{Float64, J_INDEX_TYPE},
    data::ACPowerFlowData,
    Y_bus_eff::SparseMatrixCSC{ComplexF64, Int},
    Y_diag::Vector{ComplexF64},
    e_state::Vector{Float64},
    f_state::Vector{Float64},
    Q_state::Vector{Float64},
    P_eff_cache::Vector{Float64},
    Q_eff_cache::Vector{Float64},
    bus_slack_participation_factors::SparseVector{Float64, Int},
    subnetworks::Dict{Int64, Vector{Int64}},
    bus_state_offset::Vector{REC_INDEX_TYPE},
    bus_block_size::Vector{Int8},
    total_bus_state::Int,
    time_step::Int64,
)
    n_buses = first(size(data.bus_type))
    n_lccs = size(data.lcc.p_set, 1)
    bus_types = view(data.bus_type, :, time_step)

    @inbounds for i in 1:n_buses
        off = Int(bus_state_offset[i])
        bt = bus_types[i]
        e_i = e_state[i]
        f_i = f_state[i]
        if bt == PSY.ACBusTypes.PQ
            # Use Q_eff_cache (carries ZIP const-I correction) — matches the
            # Q value the residual uses at this same bus for I_spec.
            _update_pq_diag_block!(Jv, off, e_i, f_i, Y_diag[i],
                P_eff_cache[i], Q_eff_cache[i])
        elseif bt == PSY.ACBusTypes.PV
            _update_pv_diag_block!(Jv, off, e_i, f_i, Q_state[i], Y_diag[i],
                P_eff_cache[i])
        elseif bt == PSY.ACBusTypes.REF
            _update_ref_diag_block!(Jv, off, e_i, f_i,
                bus_slack_participation_factors[i])
        end
    end

    # Distributed-slack cross-terms.
    for (ref_bus, subnetwork_buses) in subnetworks
        ref_off = Int(bus_state_offset[ref_bus])
        for bus_k in subnetwork_buses
            c_k = bus_slack_participation_factors[bus_k]
            c_k == 0.0 && continue
            bus_k == ref_bus && continue
            k_off = Int(bus_state_offset[bus_k])
            bt_k = bus_types[bus_k]
            bt_k == PSY.ACBusTypes.REF && continue
            e_k = e_state[bus_k]
            f_k = f_state[bus_k]
            V_sq = e_k^2 + f_k^2
            Jv[k_off, ref_off] = c_k * e_k / V_sq
            Jv[k_off + 1, ref_off] = c_k * f_k / V_sq
        end
    end

    if n_lccs > 0
        _set_entries_for_lcc_rect!(
            data,
            Jv,
            e_state,
            f_state,
            bus_state_offset,
            total_bus_state,
            time_step,
        )
    end
    return
end

function _update_pq_diag_block!(
    Jv::SparseMatrixCSC{Float64, J_INDEX_TYPE},
    off::Int,
    e::Float64,
    f::Float64,
    y_ii::ComplexF64,
    P_eff::Float64,
    Q_eff::Float64,
)
    V_sq = e^2 + f^2
    g_ii = real(y_ii)
    b_ii = imag(y_ii)
    Is_r = (P_eff * e + Q_eff * f) / V_sq
    Is_i = (P_eff * f - Q_eff * e) / V_sq
    inv_V_sq = 1.0 / V_sq
    Jv[off, off] = (P_eff - 2 * e * Is_r) * inv_V_sq + (-g_ii)
    Jv[off, off + 1] = (Q_eff - 2 * f * Is_r) * inv_V_sq + b_ii
    Jv[off + 1, off] = (-Q_eff - 2 * e * Is_i) * inv_V_sq + (-b_ii)
    Jv[off + 1, off + 1] = (P_eff - 2 * f * Is_i) * inv_V_sq + (-g_ii)
    return
end

function _update_pv_diag_block!(
    Jv::SparseMatrixCSC{Float64, J_INDEX_TYPE},
    off::Int,
    e::Float64,
    f::Float64,
    Q::Float64,
    y_ii::ComplexF64,
    P_eff::Float64,
)
    V_sq = e^2 + f^2
    g_ii = real(y_ii)
    b_ii = imag(y_ii)
    Is_r = (P_eff * e + Q * f) / V_sq
    Is_i = (P_eff * f - Q * e) / V_sq
    inv_V_sq = 1.0 / V_sq
    Jv[off, off] = (P_eff - 2 * e * Is_r) * inv_V_sq + (-g_ii)
    Jv[off, off + 1] = (Q - 2 * f * Is_r) * inv_V_sq + b_ii
    Jv[off + 1, off] = (-Q - 2 * e * Is_i) * inv_V_sq + (-b_ii)
    Jv[off + 1, off + 1] = (P_eff - 2 * f * Is_i) * inv_V_sq + (-g_ii)
    # Q-column: ∂I_spec_r/∂Q = f/V², ∂I_spec_i/∂Q = −e/V²
    Jv[off, off + 2] = f * inv_V_sq
    Jv[off + 1, off + 2] = -e * inv_V_sq
    # ΔV² row: ∂ΔV²/∂e = −2e, ∂ΔV²/∂f = −2f, ∂ΔV²/∂Q = 0 (structural zero)
    Jv[off + 2, off] = -2 * e
    Jv[off + 2, off + 1] = -2 * f
    return
end

function _update_ref_diag_block!(
    Jv::SparseMatrixCSC{Float64, J_INDEX_TYPE},
    off::Int,
    e_r::Float64,
    f_r::Float64,
    c_ref::Float64,
)
    # Residual at REF uses P_gen = P_net_set[ref] + c_ref · (x[off] - P_net_set[ref]).
    # ∂P_gen/∂x[off] = c_ref. So ∂I_spec_r/∂x[off] = c_ref · e_r/V², etc.
    # For default (c_ref = 1.0), this collapses to the original e_r/V² etc.
    V_sq = e_r^2 + f_r^2
    inv_V_sq = 1.0 / V_sq
    Jv[off, off] = c_ref * e_r * inv_V_sq
    Jv[off, off + 1] = f_r * inv_V_sq
    Jv[off + 1, off] = c_ref * f_r * inv_V_sq
    Jv[off + 1, off + 1] = -e_r * inv_V_sq
    return
end

"""
Write the LCC Jacobian entries (17 per LCC). Mirrors polar `_set_entries_for_lcc`
with these changes:

  * Polar's `idx_p_fb` (Vm slot, single column) becomes two rectangular columns
    `(col_e_fb, col_f_fb)`. Polar partials `∂(·)/∂Vm_fb` translate via chain rule:
    `∂(·)/∂e = ∂(·)/∂Vm · e/|V|`, similarly for `f`.
  * The bus residual rows for fb are `ΔI_r_fb` and `ΔI_i_fb` (current mismatch),
    not polar's `ΔP_fb` / `ΔQ_fb`. The LCC contribution to `−Re(Y_lcc·V_fb)` and
    `−Im(Y_lcc·V_fb)` under the α-approximation (φ ≈ α — same approximation polar
    uses for ∂P/∂Vm) gives the bus-diagonal additions and tail cross-terms below.
  * As in polar, the inverter-side bus diagonal block is NOT augmented for LCC
    (this matches polar's missing `∂P_tb/∂Vm_tb` LCC partial; convergence is
    polar-like).

LCC tail row entries (∂F_t_*, ∂F_α_*) use the same polar α-approximation as the
polar code, chain-ruled into `(e, f)` columns.
"""
function _set_entries_for_lcc_rect!(
    data::ACPowerFlowData,
    Jv::SparseMatrixCSC{Float64, J_INDEX_TYPE},
    e_state::Vector{Float64},
    f_state::Vector{Float64},
    bus_state_offset::Vector{REC_INDEX_TYPE},
    total_bus_state::Int,
    time_step::Int,
)
    for (i, (fb, tb)) in enumerate(data.lcc.bus_indices)
        col_e_fb = Int(bus_state_offset[fb])
        col_f_fb = col_e_fb + 1
        col_e_tb = Int(bus_state_offset[tb])
        col_f_tb = col_e_tb + 1
        offset_lcc = total_bus_state + (i - 1) * 4
        idx_tap_r = offset_lcc + 1
        idx_tap_i = offset_lcc + 2
        idx_alpha_r = offset_lcc + 3
        idx_alpha_i = offset_lcc + 4

        i_dc = max(data.lcc.i_dc[i, time_step], 1e-9)
        tap_r = data.lcc.rectifier.tap[i, time_step]
        tap_i = data.lcc.inverter.tap[i, time_step]
        alpha_r = data.lcc.rectifier.thyristor_angle[i, time_step]
        alpha_i = data.lcc.inverter.thyristor_angle[i, time_step]
        bus_type_fb = data.bus_type[fb, time_step]
        bus_type_tb = data.bus_type[tb, time_step]

        cos_alpha_r = cos(alpha_r)
        sin_alpha_r = sin(alpha_r)
        cos_alpha_i = cos(alpha_i)
        sin_alpha_i = sin(alpha_i)

        e_fb = e_state[fb]
        f_fb = f_state[fb]
        Vm_fb_sq = e_fb^2 + f_fb^2
        Vm_fb = sqrt(Vm_fb_sq)
        e_tb = e_state[tb]
        f_tb = f_state[tb]
        Vm_tb_sq = e_tb^2 + f_tb^2
        Vm_tb = sqrt(Vm_tb_sq)

        # FB-side: LCC contribution under α-approximation (φ_r ≈ α_r).
        # F_lcc_fb_r = -A_fb·u_fb,  F_lcc_fb_i = -A_fb·w_fb
        #   A_fb = tap_r·κ·I_dc / |V_fb|
        #   u_fb = cos(α_r)·e_fb + sin(α_r)·f_fb
        #   w_fb = cos(α_r)·f_fb − sin(α_r)·e_fb
        A_fb = tap_r * SQRT6_DIV_PI * i_dc / Vm_fb
        u_fb = cos_alpha_r * e_fb + sin_alpha_r * f_fb
        w_fb = cos_alpha_r * f_fb - sin_alpha_r * e_fb
        # Bus diagonal additions for FB (only PQ/PV — for REF, (e_fb, f_fb)
        # are fixed constants, not state variables in those columns).
        if bus_type_fb == PSY.ACBusTypes.PQ || bus_type_fb == PSY.ACBusTypes.PV
            inv_Vsq_fb = 1.0 / Vm_fb_sq
            Jv[col_e_fb, col_e_fb] += -A_fb * f_fb * w_fb * inv_Vsq_fb
            Jv[col_e_fb, col_f_fb] += A_fb * e_fb * w_fb * inv_Vsq_fb
            Jv[col_f_fb, col_e_fb] += A_fb * f_fb * u_fb * inv_Vsq_fb
            Jv[col_f_fb, col_f_fb] += -A_fb * e_fb * u_fb * inv_Vsq_fb
        end
        # FB-side cross-terms ∂F_lcc_fb / ∂(t_r, α_r) — applicable for all
        # bus types because the row is ΔI residual at fb (always exists).
        Jv[col_e_fb, idx_tap_r] = -A_fb * u_fb / tap_r
        Jv[col_e_fb, idx_alpha_r] = -A_fb * w_fb
        Jv[col_f_fb, idx_tap_r] = -A_fb * w_fb / tap_r
        Jv[col_f_fb, idx_alpha_r] = A_fb * u_fb

        # TB-side: Y_lcc_tb ≈ -(tap_i/|V_tb|)·κ·I_dc·exp(+jα_i) under polar's
        # α-approximation (φ_i ≈ π−α_i, so cos(φ_i) ≈ -cos(α_i)). Thus
        # F_lcc_tb_r = +A_tb·u_tb,  F_lcc_tb_i = +A_tb·w_tb
        #   A_tb = tap_i·κ·I_dc / |V_tb|
        #   u_tb = cos(α_i)·e_tb − sin(α_i)·f_tb
        #   w_tb = cos(α_i)·f_tb + sin(α_i)·e_tb
        A_tb = tap_i * SQRT6_DIV_PI * i_dc / Vm_tb
        u_tb = cos_alpha_i * e_tb - sin_alpha_i * f_tb
        w_tb = cos_alpha_i * f_tb + sin_alpha_i * e_tb
        if bus_type_tb == PSY.ACBusTypes.PQ || bus_type_tb == PSY.ACBusTypes.PV
            inv_Vsq_tb = 1.0 / Vm_tb_sq
            Jv[col_e_tb, col_e_tb] += A_tb * f_tb * w_tb * inv_Vsq_tb
            Jv[col_e_tb, col_f_tb] += -A_tb * e_tb * w_tb * inv_Vsq_tb
            Jv[col_f_tb, col_e_tb] += -A_tb * f_tb * u_tb * inv_Vsq_tb
            Jv[col_f_tb, col_f_tb] += A_tb * e_tb * u_tb * inv_Vsq_tb
        end
        Jv[col_e_tb, idx_tap_i] = A_tb * u_tb / tap_i
        Jv[col_e_tb, idx_alpha_i] = -A_tb * w_tb
        Jv[col_f_tb, idx_tap_i] = A_tb * w_tb / tap_i
        Jv[col_f_tb, idx_alpha_i] = A_tb * u_tb

        # LCC tail row entries (F_t_r, F_t_i) - mirror polar α-approx then chain-rule.
        common_term_tap_r = tap_r * SQRT6_DIV_PI * i_dc * cos_alpha_r
        common_term_fb_polar = Vm_fb * SQRT6_DIV_PI * i_dc
        common_term_alpha_r = -common_term_fb_polar * tap_r * sin_alpha_r
        common_term_tb_polar = Vm_tb * SQRT6_DIV_PI * (-i_dc)
        common_term_tap_i = tap_i * SQRT6_DIV_PI * (-i_dc) * cos_alpha_i

        # LCC tail row entries ∂F_t_*/∂(e, f): chain rule from ∂/∂Vm.
        # At REF buses, the columns are (P_gen, Q_gen) — those don't affect Vm,
        # so the LCC tail partials are zero there.
        if bus_type_fb != PSY.ACBusTypes.REF
            de_dV_fb = e_fb / Vm_fb
            df_dV_fb = f_fb / Vm_fb
            Jv[idx_tap_r, col_e_fb] = common_term_tap_r * de_dV_fb
            Jv[idx_tap_r, col_f_fb] = common_term_tap_r * df_dV_fb
            Jv[idx_tap_i, col_e_fb] = common_term_tap_r * de_dV_fb
            Jv[idx_tap_i, col_f_fb] = common_term_tap_r * df_dV_fb
        else
            Jv[idx_tap_r, col_e_fb] = 0.0
            Jv[idx_tap_r, col_f_fb] = 0.0
            Jv[idx_tap_i, col_e_fb] = 0.0
            Jv[idx_tap_i, col_f_fb] = 0.0
        end
        if bus_type_tb != PSY.ACBusTypes.REF
            de_dV_tb = e_tb / Vm_tb
            df_dV_tb = f_tb / Vm_tb
            Jv[idx_tap_i, col_e_tb] = common_term_tap_i * de_dV_tb
            Jv[idx_tap_i, col_f_tb] = common_term_tap_i * df_dV_tb
        else
            Jv[idx_tap_i, col_e_tb] = 0.0
            Jv[idx_tap_i, col_f_tb] = 0.0
        end

        # LCC tail to LCC tail (identical to polar).
        Jv[idx_tap_r, idx_tap_r] = common_term_fb_polar * cos_alpha_r
        Jv[idx_tap_r, idx_alpha_r] = common_term_alpha_r
        Jv[idx_tap_i, idx_tap_r] = common_term_fb_polar * cos_alpha_r
        Jv[idx_tap_i, idx_tap_i] = common_term_tb_polar * cos_alpha_i
        Jv[idx_tap_i, idx_alpha_r] = common_term_alpha_r
        Jv[idx_tap_i, idx_alpha_i] = -common_term_tb_polar * tap_i * sin_alpha_i
        # idx_alpha_r and idx_alpha_i identity diagonals already 1.0 (set at pattern build).
    end
    return
end
