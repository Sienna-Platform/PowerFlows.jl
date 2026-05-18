"""
    struct ACRectangularCIJacobian

Jacobian functor for the augmented current-injection (rectangular) AC power flow.
Mirrors [`ACPowerFlowJacobian`](@ref) but operates on the per-bus variable-block
state representation.

Per-iteration updates write directly into `nonzeros(Jv)` via nzval-index caches
built once at construction time, so the hot-path cost is `O(N + n_LCC)` rather
than `O((N + n_LCC) · log(nnz_per_col))` of `Jv[r, c] = v` setindex.

# Fields
- `data::ACPowerFlowData`
- `Jf!::Function` — inplace Jacobian update
- `Jv::SparseMatrixCSC{Float64, J_INDEX_TYPE}` — Jacobian values
- `Y_bus_eff::SparseMatrixCSC{ComplexF64, Int}` — Y_bus with ZIP-Z folded in
- `Y_diag::Vector{ComplexF64}` — cached Y_bus_eff diagonal
- Bus-diagonal nzval caches `diag_base_nz` (4×n_buses), `pv_extra_nz` (4×n_buses
  with sentinel 0 for non-PV buses)
- Slack cross-term nzval caches `slack_nz_idx_e`, `slack_nz_idx_f`, plus
  `slack_bus_k` / `slack_c_k` for the corresponding per-iteration data
- LCC tail nzval cache `lcc_nz` (20×n_lccs; the last 2 identity diagonals stay 1.0)
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
    const_I_P::Vector{Float64}     # shared view into residual's const_I_P; needed for ∂P_eff/∂(e,f) chain rule
    const_I_Q::Vector{Float64}     # shared view into residual's const_I_Q
    bus_slack_participation_factors::SparseVector{Float64, Int}
    subnetworks::Dict{Int64, Vector{Int64}}
    bus_state_offset::Vector{REC_INDEX_TYPE}
    bus_block_size::Vector{Int8}
    total_bus_state::Int
    # nzval-index caches (populated once at construction)
    diag_base_nz::Matrix{Int}        # 4 × n_buses; rows: (off,off), (off,off+1), (off+1,off), (off+1,off+1)
    pv_extra_nz::Matrix{Int}         # 4 × n_buses; rows: (off,off+2), (off+1,off+2), (off+2,off), (off+2,off+1); 0 for non-PV
    slack_nz_idx_e::Vector{Int}      # nzval index for Jv[k_off, ref_off]
    slack_nz_idx_f::Vector{Int}      # nzval index for Jv[k_off+1, ref_off]
    slack_bus_k::Vector{Int}         # bus_k per slack cross-term
    slack_c_k::Vector{Float64}       # c_k = bus_slack_participation_factors[bus_k]
    lcc_nz::Matrix{Int}              # 20 × n_lccs; nzval indices for the LCC entries (order documented in _build_lcc_nz_cache!)
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
    diag_base_nz, pv_extra_nz = _build_diag_nz_cache(
        Jv0, residual.bus_state_offset,
        view(residual.data.bus_type, :, time_step),
    )
    slack_nz_idx_e, slack_nz_idx_f, slack_bus_k, slack_c_k =
        _build_slack_nz_cache(
            Jv0, residual.bus_state_offset, residual.subnetworks,
            residual.bus_slack_participation_factors,
        )
    n_lccs = size(residual.data.lcc.p_set, 1)
    lcc_nz = _build_lcc_nz_cache(
        Jv0, residual.data, residual.bus_state_offset,
        residual.total_bus_state, n_lccs,
    )
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
        residual.const_I_P,
        residual.const_I_Q,
        residual.bus_slack_participation_factors,
        residual.subnetworks,
        residual.bus_state_offset,
        residual.bus_block_size,
        residual.total_bus_state,
        diag_base_nz,
        pv_extra_nz,
        slack_nz_idx_e,
        slack_nz_idx_f,
        slack_bus_k,
        slack_c_k,
        lcc_nz,
    )
    J(time_step)  # populate state-dependent entries (diagonals, slack, LCC tail)
    return J
end

function (J::ACRectangularCIJacobian)(time_step::Int64)
    J.Jf!(J.Jv, J.data, J.Y_diag,
        J.e_state, J.f_state, J.Q_state, J.P_eff_cache, J.Q_eff_cache,
        J.const_I_P, J.const_I_Q,
        J.bus_slack_participation_factors,
        J.bus_state_offset, J.total_bus_state,
        J.diag_base_nz, J.pv_extra_nz,
        J.slack_nz_idx_e, J.slack_nz_idx_f, J.slack_bus_k, J.slack_c_k,
        J.lcc_nz, time_step)
    return
end

function (J::ACRectangularCIJacobian)(
    Jv::SparseMatrixCSC{Float64, J_INDEX_TYPE},
    time_step::Int64,
)
    J.Jf!(J.Jv, J.data, J.Y_diag,
        J.e_state, J.f_state, J.Q_state, J.P_eff_cache, J.Q_eff_cache,
        J.const_I_P, J.const_I_Q,
        J.bus_slack_participation_factors,
        J.bus_state_offset, J.total_bus_state,
        J.diag_base_nz, J.pv_extra_nz,
        J.slack_nz_idx_e, J.slack_nz_idx_f, J.slack_bus_k, J.slack_c_k,
        J.lcc_nz, time_step)
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
    bus_types_at_t = view(data.bus_type, :, time_step)
    @inbounds for col in 1:n_buses
        col_off = Int(bus_state_offset[col])
        col_bs = bus_block_size[col]
        is_ref_col = bus_types_at_t[col] == PSY.ACBusTypes.REF
        for j in SparseArrays.nzrange(Y_bus_eff, col)
            row = Yrows[j]
            # REF columns hold (P_gen, Q_gen); neighbors' rows don't depend on them.
            if is_ref_col && row != col
                continue
            end
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

    # Distributed-slack cross-terms: ∂F_k_{r,i}/∂x[bus_state_offset[ref]].
    # Unlike polar, the Y_bus loop above SKIPS off-diagonal entries for REF
    # columns (REF state vars are (P_gen, Q_gen), not (e, f), so the neighbors'
    # rows have no partial w.r.t. REF's state vars). That means `(k_off, ref_off)`
    # is not in the structural pattern yet for any non-self bus_k — we must push
    # it here unconditionally (gated only on `bus_k != ref_bus`, since the REF
    # diagonal block already covers `bus_k == ref_bus`).
    for (ref_bus, subnetwork_buses) in subnetworks
        ref_off = Int(bus_state_offset[ref_bus])
        for bus_k in subnetwork_buses
            bus_slack_participation_factors[bus_k] == 0.0 && continue
            bus_k == ref_bus && continue
            k_off = Int(bus_state_offset[bus_k])
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
        rcv = [
            (col_e_fb, idx_tap_r, 0.0),
            (col_e_fb, idx_alpha_r, 0.0),
            (col_f_fb, idx_tap_r, 0.0),
            (col_f_fb, idx_alpha_r, 0.0),
            (col_e_tb, idx_tap_i, 0.0),
            (col_e_tb, idx_alpha_i, 0.0),
            (col_f_tb, idx_tap_i, 0.0),
            (col_f_tb, idx_alpha_i, 0.0),
            (idx_tap_r, col_e_fb, 0.0),
            (idx_tap_r, col_f_fb, 0.0),
            (idx_tap_i, col_e_fb, 0.0),
            (idx_tap_i, col_f_fb, 0.0),
            (idx_tap_i, col_e_tb, 0.0),
            (idx_tap_i, col_f_tb, 0.0),
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
    _jv_nz_index(Jv, row, col)

Return the nzval index for `Jv[row, col]`. Assumes the entry is structurally
present (errors otherwise). Used at construction time to pre-compute indices
for the hot-path update functions.
"""
@inline function _jv_nz_index(
    Jv::SparseMatrixCSC{Float64, J_INDEX_TYPE},
    row::Int,
    col::Int,
)
    rowvals = SparseArrays.rowvals(Jv)
    rng = SparseArrays.nzrange(Jv, col)
    for k in rng
        rowvals[k] == row && return Int(k)
    end
    error("Jacobian sparsity pattern missing entry at ($row, $col)")
end

"""
    _build_diag_nz_cache(Jv, bus_state_offset, bus_types)

Return `(diag_base_nz::Matrix{Int}, pv_extra_nz::Matrix{Int})`, each `4 × n_buses`,
containing the nzval indices for the per-bus diagonal block entries.

Row layout of `diag_base_nz` (always populated, all bus types):
  1: Jv[off, off],     2: Jv[off, off+1],
  3: Jv[off+1, off],   4: Jv[off+1, off+1]

Row layout of `pv_extra_nz` (only populated for PV; 0 for non-PV):
  1: Jv[off, off+2],   2: Jv[off+1, off+2],
  3: Jv[off+2, off],   4: Jv[off+2, off+1]
"""
function _build_diag_nz_cache(
    Jv::SparseMatrixCSC{Float64, J_INDEX_TYPE},
    bus_state_offset::Vector{REC_INDEX_TYPE},
    bus_types::AbstractVector{PSY.ACBusTypes},
)
    n_buses = length(bus_types)
    diag_base_nz = Matrix{Int}(undef, 4, n_buses)
    pv_extra_nz = zeros(Int, 4, n_buses)
    @inbounds for i in 1:n_buses
        off = Int(bus_state_offset[i])
        diag_base_nz[1, i] = _jv_nz_index(Jv, off, off)
        diag_base_nz[2, i] = _jv_nz_index(Jv, off, off + 1)
        diag_base_nz[3, i] = _jv_nz_index(Jv, off + 1, off)
        diag_base_nz[4, i] = _jv_nz_index(Jv, off + 1, off + 1)
        if bus_types[i] == PSY.ACBusTypes.PV
            pv_extra_nz[1, i] = _jv_nz_index(Jv, off, off + 2)
            pv_extra_nz[2, i] = _jv_nz_index(Jv, off + 1, off + 2)
            pv_extra_nz[3, i] = _jv_nz_index(Jv, off + 2, off)
            pv_extra_nz[4, i] = _jv_nz_index(Jv, off + 2, off + 1)
        end
    end
    return diag_base_nz, pv_extra_nz
end

"""
    _build_slack_nz_cache(Jv, bus_state_offset, subnetworks, bus_slack_participation_factors)

Return `(slack_nz_idx_e, slack_nz_idx_f, slack_bus_k, slack_c_k)`. Each entry
corresponds to one (bus_k != ref_bus, c_k != 0) slack cross-term. The nzval
indices point at `Jv[k_off, ref_off]` and `Jv[k_off+1, ref_off]`.
"""
function _build_slack_nz_cache(
    Jv::SparseMatrixCSC{Float64, J_INDEX_TYPE},
    bus_state_offset::Vector{REC_INDEX_TYPE},
    subnetworks::Dict{Int64, Vector{Int64}},
    bus_slack_participation_factors::SparseVector{Float64, Int},
)
    slack_nz_idx_e = Int[]
    slack_nz_idx_f = Int[]
    slack_bus_k = Int[]
    slack_c_k = Float64[]
    for (ref_bus, subnetwork_buses) in subnetworks
        ref_off = Int(bus_state_offset[ref_bus])
        for bus_k in subnetwork_buses
            c_k = bus_slack_participation_factors[bus_k]
            c_k == 0.0 && continue
            bus_k == ref_bus && continue
            k_off = Int(bus_state_offset[bus_k])
            push!(slack_nz_idx_e, _jv_nz_index(Jv, k_off, ref_off))
            push!(slack_nz_idx_f, _jv_nz_index(Jv, k_off + 1, ref_off))
            push!(slack_bus_k, bus_k)
            push!(slack_c_k, c_k)
        end
    end
    return slack_nz_idx_e, slack_nz_idx_f, slack_bus_k, slack_c_k
end

"""
    _build_lcc_nz_cache(Jv, data, bus_state_offset, total_bus_state, n_lccs)

Return a `20 × n_lccs` matrix of nzval indices for the per-LCC tail entries
that get updated each iteration. The two identity diagonals
(`Jv[idx_alpha_r, idx_alpha_r]` and `Jv[idx_alpha_i, idx_alpha_i]`) are not
included — they are set to 1.0 at structure-build time and never updated.
The 8 FB/TB-side diagonal-block overlay entries are NOT included either —
they share nzval slots with `diag_base_nz` for buses `fb` and `tb` and are
addressed through that cache.

Row layout (matches order pushed by [`_create_rect_ci_lcc_structure!`]):
  1: Jv[col_e_fb, idx_tap_r],   2: Jv[col_e_fb, idx_alpha_r],
  3: Jv[col_f_fb, idx_tap_r],   4: Jv[col_f_fb, idx_alpha_r],
  5: Jv[col_e_tb, idx_tap_i],   6: Jv[col_e_tb, idx_alpha_i],
  7: Jv[col_f_tb, idx_tap_i],   8: Jv[col_f_tb, idx_alpha_i],
  9: Jv[idx_tap_r, col_e_fb],  10: Jv[idx_tap_r, col_f_fb],
 11: Jv[idx_tap_i, col_e_fb],  12: Jv[idx_tap_i, col_f_fb],
 13: Jv[idx_tap_i, col_e_tb],  14: Jv[idx_tap_i, col_f_tb],
 15: Jv[idx_tap_r, idx_tap_r], 16: Jv[idx_tap_r, idx_alpha_r],
 17: Jv[idx_tap_i, idx_tap_r], 18: Jv[idx_tap_i, idx_tap_i],
 19: Jv[idx_tap_i, idx_alpha_r], 20: Jv[idx_tap_i, idx_alpha_i],
"""
function _build_lcc_nz_cache(
    Jv::SparseMatrixCSC{Float64, J_INDEX_TYPE},
    data::ACPowerFlowData,
    bus_state_offset::Vector{REC_INDEX_TYPE},
    total_bus_state::Int,
    n_lccs::Int,
)
    lcc_nz = Matrix{Int}(undef, 20, n_lccs)
    n_lccs == 0 && return lcc_nz
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
        lcc_nz[1, i] = _jv_nz_index(Jv, col_e_fb, idx_tap_r)
        lcc_nz[2, i] = _jv_nz_index(Jv, col_e_fb, idx_alpha_r)
        lcc_nz[3, i] = _jv_nz_index(Jv, col_f_fb, idx_tap_r)
        lcc_nz[4, i] = _jv_nz_index(Jv, col_f_fb, idx_alpha_r)
        lcc_nz[5, i] = _jv_nz_index(Jv, col_e_tb, idx_tap_i)
        lcc_nz[6, i] = _jv_nz_index(Jv, col_e_tb, idx_alpha_i)
        lcc_nz[7, i] = _jv_nz_index(Jv, col_f_tb, idx_tap_i)
        lcc_nz[8, i] = _jv_nz_index(Jv, col_f_tb, idx_alpha_i)
        lcc_nz[9, i] = _jv_nz_index(Jv, idx_tap_r, col_e_fb)
        lcc_nz[10, i] = _jv_nz_index(Jv, idx_tap_r, col_f_fb)
        lcc_nz[11, i] = _jv_nz_index(Jv, idx_tap_i, col_e_fb)
        lcc_nz[12, i] = _jv_nz_index(Jv, idx_tap_i, col_f_fb)
        lcc_nz[13, i] = _jv_nz_index(Jv, idx_tap_i, col_e_tb)
        lcc_nz[14, i] = _jv_nz_index(Jv, idx_tap_i, col_f_tb)
        lcc_nz[15, i] = _jv_nz_index(Jv, idx_tap_r, idx_tap_r)
        lcc_nz[16, i] = _jv_nz_index(Jv, idx_tap_r, idx_alpha_r)
        lcc_nz[17, i] = _jv_nz_index(Jv, idx_tap_i, idx_tap_r)
        lcc_nz[18, i] = _jv_nz_index(Jv, idx_tap_i, idx_tap_i)
        lcc_nz[19, i] = _jv_nz_index(Jv, idx_tap_i, idx_alpha_r)
        lcc_nz[20, i] = _jv_nz_index(Jv, idx_tap_i, idx_alpha_i)
    end
    return lcc_nz
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
        for j in SparseArrays.nzrange(Y_bus_eff, col)
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
be up to date (call the residual on `x` first).

All writes go through the pre-computed nzval-index caches into
`nonzeros(Jv)`, so the per-iteration cost scales as `O(N + n_LCC)` rather
than incurring `O(log(nnz_per_col))` per assignment."""
function _update_rect_ci_jacobian_values!(
    Jv::SparseMatrixCSC{Float64, J_INDEX_TYPE},
    data::ACPowerFlowData,
    Y_diag::Vector{ComplexF64},
    e_state::Vector{Float64},
    f_state::Vector{Float64},
    Q_state::Vector{Float64},
    P_eff_cache::Vector{Float64},
    Q_eff_cache::Vector{Float64},
    const_I_P::Vector{Float64},
    const_I_Q::Vector{Float64},
    bus_slack_participation_factors::SparseVector{Float64, Int},
    bus_state_offset::Vector{REC_INDEX_TYPE},
    total_bus_state::Int,
    diag_base_nz::Matrix{Int},
    pv_extra_nz::Matrix{Int},
    slack_nz_idx_e::Vector{Int},
    slack_nz_idx_f::Vector{Int},
    slack_bus_k::Vector{Int},
    slack_c_k::Vector{Float64},
    lcc_nz::Matrix{Int},
    time_step::Int64,
)
    n_buses = first(size(data.bus_type))
    n_lccs = size(data.lcc.p_set, 1)
    bus_types = view(data.bus_type, :, time_step)
    Jvnz = SparseArrays.nonzeros(Jv)

    @inbounds for i in 1:n_buses
        bt = bus_types[i]
        e_i = e_state[i]
        f_i = f_state[i]
        if bt == PSY.ACBusTypes.PQ
            # Use Q_eff_cache (carries ZIP const-I correction) — matches the
            # Q value the residual uses at this same bus for I_spec.
            _update_pq_diag_block!(Jvnz, diag_base_nz, i, e_i, f_i, Y_diag[i],
                P_eff_cache[i], Q_eff_cache[i],
                const_I_P[i], const_I_Q[i])
        elseif bt == PSY.ACBusTypes.PV
            _update_pv_diag_block!(Jvnz, diag_base_nz, pv_extra_nz, i,
                e_i, f_i, Q_state[i], Y_diag[i],
                P_eff_cache[i], const_I_P[i])
        elseif bt == PSY.ACBusTypes.REF
            c_ref = bus_slack_participation_factors[i]
            _update_ref_diag_block!(Jvnz, diag_base_nz, i, e_i, f_i, c_ref)
        end
    end

    # Distributed-slack cross-terms: write directly into nzval via cached indices.
    @inbounds for k in eachindex(slack_nz_idx_e)
        bus_k = slack_bus_k[k]
        c_k = slack_c_k[k]
        e_k = e_state[bus_k]
        f_k = f_state[bus_k]
        # V_FLOOR2 floor (see rectangular_ci_power_flow_residual.jl).
        inv_V_sq = 1.0 / max(e_k^2 + f_k^2, V_FLOOR2)
        Jvnz[slack_nz_idx_e[k]] = c_k * e_k * inv_V_sq
        Jvnz[slack_nz_idx_f[k]] = c_k * f_k * inv_V_sq
    end

    if n_lccs > 0
        _set_entries_for_lcc_rect!(
            data,
            Jvnz,
            diag_base_nz,
            lcc_nz,
            e_state,
            f_state,
            bus_state_offset,
            time_step,
        )
    end
    return
end

@inline function _update_pq_diag_block!(
    Jvnz::Vector{Float64},
    diag_base_nz::Matrix{Int},
    i::Int,
    e::Float64,
    f::Float64,
    y_ii::ComplexF64,
    P_eff::Float64,
    Q_eff::Float64,
    const_I_P::Float64,
    const_I_Q::Float64,
)
    # V_FLOOR2 floor (see rectangular_ci_power_flow_residual.jl).
    V_sq = max(e^2 + f^2, V_FLOOR2)
    inv_V_sq = 1.0 / V_sq
    inv_Vm = 1.0 / sqrt(V_sq)
    g_ii = real(y_ii)
    b_ii = imag(y_ii)
    Is_r = (P_eff * e + Q_eff * f) * inv_V_sq
    Is_i = (P_eff * f - Q_eff * e) * inv_V_sq
    term_r = (const_I_P * e + const_I_Q * f) * inv_Vm * inv_V_sq
    term_i = (const_I_P * f - const_I_Q * e) * inv_Vm * inv_V_sq
    @inbounds Jvnz[diag_base_nz[1, i]] =
        (P_eff - 2 * e * Is_r) * inv_V_sq - g_ii - e * term_r
    @inbounds Jvnz[diag_base_nz[2, i]] =
        (Q_eff - 2 * f * Is_r) * inv_V_sq + b_ii - f * term_r
    @inbounds Jvnz[diag_base_nz[3, i]] =
        (-Q_eff - 2 * e * Is_i) * inv_V_sq - b_ii - e * term_i
    @inbounds Jvnz[diag_base_nz[4, i]] =
        (P_eff - 2 * f * Is_i) * inv_V_sq - g_ii - f * term_i
    return
end

@inline function _update_pv_diag_block!(
    Jvnz::Vector{Float64},
    diag_base_nz::Matrix{Int},
    pv_extra_nz::Matrix{Int},
    i::Int,
    e::Float64,
    f::Float64,
    Q::Float64,
    y_ii::ComplexF64,
    P_eff::Float64,
    const_I_P::Float64,
)
    # V_FLOOR2 floor (see rectangular_ci_power_flow_residual.jl); the |V|² row's
    # (−2e, −2f) entries below stay raw (exact constraint derivative).
    V_sq = max(e^2 + f^2, V_FLOOR2)
    inv_V_sq = 1.0 / V_sq
    inv_Vm = 1.0 / sqrt(V_sq)
    g_ii = real(y_ii)
    b_ii = imag(y_ii)
    Is_r = (P_eff * e + Q * f) * inv_V_sq
    Is_i = (P_eff * f - Q * e) * inv_V_sq
    # Q is the state variable at PV (not Q_eff), so const_I_Q does not enter.
    term_r = const_I_P * e * inv_Vm * inv_V_sq
    term_i = const_I_P * f * inv_Vm * inv_V_sq
    @inbounds Jvnz[diag_base_nz[1, i]] =
        (P_eff - 2 * e * Is_r) * inv_V_sq - g_ii - e * term_r
    @inbounds Jvnz[diag_base_nz[2, i]] = (Q - 2 * f * Is_r) * inv_V_sq + b_ii - f * term_r
    @inbounds Jvnz[diag_base_nz[3, i]] = (-Q - 2 * e * Is_i) * inv_V_sq - b_ii - e * term_i
    @inbounds Jvnz[diag_base_nz[4, i]] =
        (P_eff - 2 * f * Is_i) * inv_V_sq - g_ii - f * term_i
    @inbounds Jvnz[pv_extra_nz[1, i]] = f * inv_V_sq
    @inbounds Jvnz[pv_extra_nz[2, i]] = -e * inv_V_sq
    @inbounds Jvnz[pv_extra_nz[3, i]] = -2 * e
    @inbounds Jvnz[pv_extra_nz[4, i]] = -2 * f
    return
end

@inline function _update_ref_diag_block!(
    Jvnz::Vector{Float64},
    diag_base_nz::Matrix{Int},
    i::Int,
    e_r::Float64,
    f_r::Float64,
    c_ref::Float64,
)
    # Residual at REF uses P_gen = P_net_set[ref] + c_ref · (x[off] - P_net_set[ref]).
    # ∂P_gen/∂x[off] = c_ref. So ∂I_spec_r/∂x[off] = c_ref · e_r/V², etc.
    # For default (c_ref = 1.0), this collapses to the original e_r/V² etc.
    # V_FLOOR2 floor (see rectangular_ci_power_flow_residual.jl); REF |V| is
    # fixed near V_set so this never triggers in practice.
    V_sq = max(e_r^2 + f_r^2, V_FLOOR2)
    inv_V_sq = 1.0 / V_sq
    @inbounds Jvnz[diag_base_nz[1, i]] = c_ref * e_r * inv_V_sq
    @inbounds Jvnz[diag_base_nz[2, i]] = f_r * inv_V_sq
    @inbounds Jvnz[diag_base_nz[3, i]] = c_ref * f_r * inv_V_sq
    @inbounds Jvnz[diag_base_nz[4, i]] = -e_r * inv_V_sq
    return
end

"""
Write the LCC Jacobian entries (17 per LCC). Mirrors polar `_set_entries_for_lcc`
with these changes:

  * Polar's `idx_p_fb` (Vm slot, single column) becomes two rectangular columns
    `(col_e_fb, col_f_fb)`. Polar partials `∂(·)/∂Vm_fb` translate via chain rule:
    `∂(·)/∂e = ∂(·)/∂Vm · e/|V|`, similarly for `f`.
  * The bus residual rows for fb are `ΔI_r_fb` and `ΔI_i_fb` (current mismatch),
    not polar's `ΔP_fb` / `ΔQ_fb`. The LCC contribution `−Re(Y_lcc·V) = −A·u(ϕ)`
    and `−Im(Y_lcc·V) = −A·w(ϕ)` (with the *true* ϕ from
    [`_calculate_ϕ_lcc`](@ref), not the α-approximation) gives the bus-diagonal
    additions and tail cross-terms below. The inverter sign convention is
    absorbed into `cos(ϕ_i)`, `sin(ϕ_i)` — no separate handling needed.

Tail row entries (∂F_t_*) use the true-ϕ ∂P/∂V helper from
[`_lcc_jacobian_scalars`](@ref) chain-ruled into `(e, f)` columns via
`∂V/∂e = e/V`, `∂V/∂f = f/V`. Every chain term that picks up a
`1/sin(ϕ)` factor (i.e. `cos(ϕ)·∂ϕ/∂x` chain) goes through
[`_dphi_dV_lcc`](@ref) / [`_dphi_dt_lcc`](@ref) /
[`_dphi_dα_lcc`](@ref), which return 0 at the `sin(ϕ) → 0` clamp.
"""
function _set_entries_for_lcc_rect!(
    data::ACPowerFlowData,
    Jvnz::Vector{Float64},
    diag_base_nz::Matrix{Int},
    lcc_nz::Matrix{Int},
    e_state::Vector{Float64},
    f_state::Vector{Float64},
    bus_state_offset::Vector{REC_INDEX_TYPE},
    time_step::Int,
)
    @inbounds for (i, (fb, tb)) in enumerate(data.lcc.bus_indices)
        bus_type_fb = data.bus_type[fb, time_step]
        bus_type_tb = data.bus_type[tb, time_step]

        e_fb = e_state[fb]
        f_fb = f_state[fb]
        Vm_fb_sq = e_fb^2 + f_fb^2
        Vm_fb = sqrt(Vm_fb_sq)
        inv_Vsq_fb = 1.0 / Vm_fb_sq
        e_tb = e_state[tb]
        f_tb = f_state[tb]
        Vm_tb_sq = e_tb^2 + f_tb^2
        Vm_tb = sqrt(Vm_tb_sq)
        inv_Vsq_tb = 1.0 / Vm_tb_sq

        s = _lcc_jacobian_scalars(data, i, time_step, Vm_fb, Vm_tb)

        # True-ϕ formulation throughout. Both LCC bus residual rows use the
        # admittance form `F_real = -Re(Y·V) = -A·u(ϕ)`,
        # `F_imag = -Im(Y·V) = -A·w(ϕ)` with `A = tap·√6/π·I_dc/V` and
        # universal `u(ϕ) = cos(ϕ)·e + sin(ϕ)·f`, `w(ϕ) = cos(ϕ)·f − sin(ϕ)·e`.
        # The inverter convention's sign flip (Y picks up an exp(-iπ) ≈ −1
        # from ϕ_i ≈ π − α_i at x_t = 0) is absorbed into `cos(ϕ_i)`,
        # `sin(ϕ_i)` — no separate sign handling needed in the Jacobian.
        phi_r = data.lcc.rectifier.phi[i, time_step]
        phi_i = data.lcc.inverter.phi[i, time_step]
        xtr_r = data.lcc.rectifier.transformer_reactance[i]
        xtr_i = data.lcc.inverter.transformer_reactance[i]
        alpha_r = data.lcc.rectifier.thyristor_angle[i, time_step]
        alpha_i = data.lcc.inverter.thyristor_angle[i, time_step]
        cos_phi_r = cos(phi_r)
        sin_phi_r = sin(phi_r)
        cos_phi_i = cos(phi_i)
        sin_phi_i = sin(phi_i)
        # ∂ϕ derivatives with sin(ϕ)→0 clamp guard (return 0 at clamp).
        dphi_dV_fb = _dphi_dV_lcc(xtr_r, s.i_dc, Vm_fb, s.tap_r, phi_r)
        dphi_dV_tb = _dphi_dV_lcc(xtr_i, s.i_dc, Vm_tb, s.tap_i, phi_i)
        dphi_dtap_r = _dphi_dt_lcc(xtr_r, s.i_dc, Vm_fb, s.tap_r, phi_r)
        dphi_dtap_i = _dphi_dt_lcc(xtr_i, s.i_dc, Vm_tb, s.tap_i, phi_i)
        dphi_dα_r = _dphi_dα_lcc(alpha_r, phi_r)
        # Inverter ϕ convention flips the sign of ∂ϕ_i/∂α_i.
        dphi_dα_i = -_dphi_dα_lcc(alpha_i, phi_i)

        # FB-side bus contribution.
        A_fb = s.tap_r * SQRT6_DIV_PI * s.i_dc / Vm_fb
        u_fb = cos_phi_r * e_fb + sin_phi_r * f_fb
        w_fb = cos_phi_r * f_fb - sin_phi_r * e_fb
        # Chain-rule factors: ∂ϕ_r/∂(e,f) = ∂ϕ_r/∂V · ∂V/∂(e,f), with
        # ∂V/∂e = e/V, ∂V/∂f = f/V.
        dphi_de_fb = dphi_dV_fb * e_fb / Vm_fb
        dphi_df_fb = dphi_dV_fb * f_fb / Vm_fb
        if bus_type_fb == PSY.ACBusTypes.PQ || bus_type_fb == PSY.ACBusTypes.PV
            Jvnz[diag_base_nz[1, fb]] +=
                -A_fb * f_fb * w_fb * inv_Vsq_fb - A_fb * w_fb * dphi_de_fb
            Jvnz[diag_base_nz[2, fb]] +=
                A_fb * e_fb * w_fb * inv_Vsq_fb - A_fb * w_fb * dphi_df_fb
            Jvnz[diag_base_nz[3, fb]] +=
                A_fb * f_fb * u_fb * inv_Vsq_fb + A_fb * u_fb * dphi_de_fb
            Jvnz[diag_base_nz[4, fb]] +=
                -A_fb * e_fb * u_fb * inv_Vsq_fb + A_fb * u_fb * dphi_df_fb
        end
        # FB-side cross-terms ∂F_lcc_fb / ∂(t_r, α_r), all bus types.
        Jvnz[lcc_nz[1, i]] = -A_fb * u_fb / s.tap_r - A_fb * w_fb * dphi_dtap_r
        Jvnz[lcc_nz[2, i]] = -A_fb * w_fb * dphi_dα_r
        Jvnz[lcc_nz[3, i]] = -A_fb * w_fb / s.tap_r + A_fb * u_fb * dphi_dtap_r
        Jvnz[lcc_nz[4, i]] = A_fb * u_fb * dphi_dα_r

        # TB-side bus contribution (same universal formulas; uses ϕ_i).
        A_tb = s.tap_i * SQRT6_DIV_PI * s.i_dc / Vm_tb
        u_tb = cos_phi_i * e_tb + sin_phi_i * f_tb
        w_tb = cos_phi_i * f_tb - sin_phi_i * e_tb
        dphi_de_tb = dphi_dV_tb * e_tb / Vm_tb
        dphi_df_tb = dphi_dV_tb * f_tb / Vm_tb
        if bus_type_tb == PSY.ACBusTypes.PQ || bus_type_tb == PSY.ACBusTypes.PV
            Jvnz[diag_base_nz[1, tb]] +=
                -A_tb * f_tb * w_tb * inv_Vsq_tb - A_tb * w_tb * dphi_de_tb
            Jvnz[diag_base_nz[2, tb]] +=
                A_tb * e_tb * w_tb * inv_Vsq_tb - A_tb * w_tb * dphi_df_tb
            Jvnz[diag_base_nz[3, tb]] +=
                A_tb * f_tb * u_tb * inv_Vsq_tb + A_tb * u_tb * dphi_de_tb
            Jvnz[diag_base_nz[4, tb]] +=
                -A_tb * e_tb * u_tb * inv_Vsq_tb + A_tb * u_tb * dphi_df_tb
        end
        Jvnz[lcc_nz[5, i]] = -A_tb * u_tb / s.tap_i - A_tb * w_tb * dphi_dtap_i
        Jvnz[lcc_nz[6, i]] = -A_tb * w_tb * dphi_dα_i
        Jvnz[lcc_nz[7, i]] = -A_tb * w_tb / s.tap_i + A_tb * u_tb * dphi_dtap_i
        Jvnz[lcc_nz[8, i]] = A_tb * u_tb * dphi_dα_i

        # LCC tail row entries — chain rule ∂F_t/∂V into (e, f). Use the
        # true-ϕ ∂P/∂V from the scalars helper (already boundary-guarded).
        # At REF buses (e, f) are not state, so the column slots there
        # hold (P_gen, Q_gen); write zero in that case.
        if bus_type_fb == PSY.ACBusTypes.PQ || bus_type_fb == PSY.ACBusTypes.PV
            de_dV_fb = e_fb / Vm_fb
            df_dV_fb = f_fb / Vm_fb
            Jvnz[lcc_nz[9, i]] = s.dP_dV_fb * de_dV_fb
            Jvnz[lcc_nz[10, i]] = s.dP_dV_fb * df_dV_fb
            Jvnz[lcc_nz[11, i]] = s.dP_dV_fb * de_dV_fb
            Jvnz[lcc_nz[12, i]] = s.dP_dV_fb * df_dV_fb
        else
            Jvnz[lcc_nz[9, i]] = 0.0
            Jvnz[lcc_nz[10, i]] = 0.0
            Jvnz[lcc_nz[11, i]] = 0.0
            Jvnz[lcc_nz[12, i]] = 0.0
        end
        if bus_type_tb == PSY.ACBusTypes.PQ || bus_type_tb == PSY.ACBusTypes.PV
            de_dV_tb = e_tb / Vm_tb
            df_dV_tb = f_tb / Vm_tb
            Jvnz[lcc_nz[13, i]] = s.dP_dV_tb * de_dV_tb
            Jvnz[lcc_nz[14, i]] = s.dP_dV_tb * df_dV_tb
        else
            Jvnz[lcc_nz[13, i]] = 0.0
            Jvnz[lcc_nz[14, i]] = 0.0
        end

        # Tail × tail block (shared with polar via _lcc_jacobian_scalars).
        Jvnz[lcc_nz[15, i]] = s.d_Ft_fb_d_tap_r
        Jvnz[lcc_nz[16, i]] = s.d_Ft_fb_d_alpha_r
        Jvnz[lcc_nz[17, i]] = s.d_Ft_tb_d_tap_r
        Jvnz[lcc_nz[18, i]] = s.d_Ft_tb_d_tap_i
        Jvnz[lcc_nz[19, i]] = s.d_Ft_tb_d_alpha_r
        Jvnz[lcc_nz[20, i]] = s.d_Ft_tb_d_alpha_i
        # idx_alpha_r and idx_alpha_i identity diagonals stay at 1.0 (set at pattern build).
    end
    return
end
