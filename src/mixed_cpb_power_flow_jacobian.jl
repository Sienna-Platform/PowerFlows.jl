"""
    struct ACMixedCPBJacobian

Jacobian functor for the mixed current/power-balance (MCPB) AC power flow.
Mirrors [`ACRectangularCIJacobian`](@ref) 1:1, but every bus uses a 2-slot
block (no PV→3 expansion): rect's `pv_extra_nz` is dropped, and `offdiag_pv_nz`
is added for the PV power-balance row's off-diagonals, which are nonlinear in
MCPB and rewritten each iteration. PQ off-diagonals are constant `±Y`
(`_populate_mixed_constant_yb_blocks!`). Per-iteration updates write into
`nonzeros(Jv)` through nzval-index caches built once at construction, so the
hot path is `O(N + n_LCC)`. Field roles are in the inline comments below.
"""
struct ACMixedCPBJacobian
    data::ACPowerFlowData
    Jv::SparseMatrixCSC{Float64, J_INDEX_TYPE}
    Y_bus_eff::SparseMatrixCSC{ComplexF64, Int}
    Y_diag::Vector{ComplexF64}     # cached Y_bus_eff diagonal; avoids O(log nnz) sparse access per iteration
    e_state::Vector{Float64}       # shared view into residual's e_state
    f_state::Vector{Float64}       # shared view into residual's f_state
    P_eff_cache::Vector{Float64}   # shared view into residual's P_eff_cache
    Q_eff_cache::Vector{Float64}   # shared view into residual's Q_eff_cache (PQ-bus ZIP-corrected Q)
    const_I_P::Vector{Float64}     # shared view into residual's const_I_P; needed for ∂P_eff/∂(e,f) chain rule
    const_I_Q::Vector{Float64}     # shared view into residual's const_I_Q
    Ir_acc::Vector{Float64}        # shared view into residual's Ir_acc (accumulated Re(I) per bus)
    Ii_acc::Vector{Float64}        # shared view into residual's Ii_acc (accumulated Im(I) per bus)
    bus_slack_participation_factors::SparseVector{Float64, Int}
    subnetworks::Dict{Int64, Vector{Int64}}
    independent_ref::Set{Int}      # shared view into residual's independent_ref
    bus_state_offset::Vector{REC_INDEX_TYPE}
    bus_block_size::Vector{Int8}
    total_bus_state::Int
    # nzval-index caches (populated once at construction)
    diag_base_nz::Matrix{Int}        # 4 × n_buses; rows: (off,off), (off,off+1), (off+1,off), (off+1,off+1)
    offdiag_pv_nz::Matrix{Int}       # 2 × n_pv_pairs; PV power-balance row only (voltage-constraint row has no off-diagonals)
    offdiag_pv_i::Vector{Int}        # PV bus for each offdiag_pv_nz column
    offdiag_pv_k::Vector{Int}        # neighbor bus for each offdiag_pv_nz column
    offdiag_pv_y::Vector{ComplexF64} # Y_bus_eff[i, k] for each pair (constant; G_ik+jB_ik)
    slack_nz_idx_e::Vector{Int}      # nzval index for Jv[k_off, ref_off]
    slack_nz_idx_f::Vector{Int}      # nzval index for Jv[k_off+1, ref_off]
    slack_c_k::Vector{Float64}       # c_k = bus_slack_participation_factors[bus_k]
    lcc_nz::Matrix{Int}              # 24 × n_lccs; nzval indices for the LCC entries
    vsc_nz::VSCJacobianNZCache       # nzval indices for the VSC tail entries
end

function ACMixedCPBJacobian(
    residual::ACMixedCPBResidual,
    time_step::Int64,
)
    Jv0 = _create_mixed_cpb_jacobian_structure(
        residual.data,
        residual.Y_bus_eff,
        residual.bus_slack_participation_factors,
        residual.subnetworks,
        residual.bus_state_offset,
        residual.bus_block_size,
        residual.total_bus_state,
        time_step,
    )
    # Populate the constant entries: Y_bus off-diagonal blocks for PQ rows ONLY
    # (PV off-diagonals are nonlinear and are written per-iteration by
    # `_update_mixed_cpb_jacobian_values!`).
    _populate_mixed_constant_yb_blocks!(
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
    diag_base_nz = _build_mixed_diag_nz_cache(
        Jv0, residual.bus_state_offset,
    )
    offdiag_pv_nz, offdiag_pv_i, offdiag_pv_k, offdiag_pv_y =
        _build_offdiag_pv_nz_cache(
            Jv0, residual.Y_bus_eff, residual.bus_state_offset,
            view(residual.data.bus_type, :, time_step),
        )
    # REF status is fixed for the life of a solve; reuse the residual's
    # already-computed set instead of reallocating it here.
    slack_nz_idx_e, slack_nz_idx_f, _, slack_c_k =
        _build_slack_nz_cache(
            Jv0, residual.bus_state_offset, residual.subnetworks,
            residual.bus_slack_participation_factors, residual.independent_ref,
        )
    n_lccs = size(residual.data.lcc.p_set, 1)
    lcc_nz = _build_lcc_nz_cache(
        Jv0, residual.data, residual.bus_state_offset,
        residual.total_bus_state, n_lccs,
    )
    vsc_nz = _build_vsc_nz_cache(
        Jv0, get_dc_network(residual.data), residual.bus_state_offset,
        residual.total_bus_state, n_lccs,
    )
    J = ACMixedCPBJacobian(
        residual.data,
        Jv0,
        residual.Y_bus_eff,
        Y_diag,
        residual.e_state,
        residual.f_state,
        residual.P_eff_cache,
        residual.Q_eff_cache,
        residual.const_I_P,
        residual.const_I_Q,
        residual.Ir_acc,
        residual.Ii_acc,
        residual.bus_slack_participation_factors,
        residual.subnetworks,
        residual.independent_ref,
        residual.bus_state_offset,
        residual.bus_block_size,
        residual.total_bus_state,
        diag_base_nz,
        offdiag_pv_nz,
        offdiag_pv_i,
        offdiag_pv_k,
        offdiag_pv_y,
        slack_nz_idx_e,
        slack_nz_idx_f,
        slack_c_k,
        lcc_nz,
        vsc_nz,
    )
    J(time_step)  # populate state-dependent entries (diagonals, PV off-diag, slack, LCC tail)
    return J
end

function (J::ACMixedCPBJacobian)(time_step::Int64)
    _update_mixed_cpb_jacobian_values!(J.Jv, J.data, J.Y_diag,
        J.e_state, J.f_state, J.P_eff_cache, J.Q_eff_cache,
        J.const_I_P, J.const_I_Q, J.Ir_acc, J.Ii_acc,
        J.bus_slack_participation_factors, J.independent_ref,
        J.bus_state_offset, J.total_bus_state,
        J.diag_base_nz, J.offdiag_pv_nz, J.offdiag_pv_i, J.offdiag_pv_k, J.offdiag_pv_y,
        J.slack_nz_idx_e, J.slack_nz_idx_f, J.slack_c_k,
        J.lcc_nz, J.vsc_nz, time_step)
    return
end

function (J::ACMixedCPBJacobian)(
    Jv::SparseMatrixCSC{Float64, J_INDEX_TYPE},
    time_step::Int64,
)
    _update_mixed_cpb_jacobian_values!(J.Jv, J.data, J.Y_diag,
        J.e_state, J.f_state, J.P_eff_cache, J.Q_eff_cache,
        J.const_I_P, J.const_I_Q, J.Ir_acc, J.Ii_acc,
        J.bus_slack_participation_factors, J.independent_ref,
        J.bus_state_offset, J.total_bus_state,
        J.diag_base_nz, J.offdiag_pv_nz, J.offdiag_pv_i, J.offdiag_pv_k, J.offdiag_pv_y,
        J.slack_nz_idx_e, J.slack_nz_idx_f, J.slack_c_k,
        J.lcc_nz, J.vsc_nz, time_step)
    copyto!(Jv, J.Jv)
    return
end

"""
Build the MCPB Jacobian sparsity pattern (all blocks 2×2). Off-diagonal blocks
reserve both `(e, f)` neighbor columns for PQ and PV rows; PQ entries are later
filled constant by `_populate_mixed_constant_yb_blocks!`, PV entries are
structural zeros written per-iteration. Slack cross-terms, LCC tail, and REF
handling mirror rect verbatim.
"""
function _create_mixed_cpb_jacobian_structure(
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
    dcn = get_dc_network(data)
    total_state = total_bus_state + state_tail_length(data, dcn)

    sizehint!(rows, 4 * SparseArrays.nnz(Y_bus_eff) + 17 * n_lccs + 2 * n_buses)
    sizehint!(cols, 4 * SparseArrays.nnz(Y_bus_eff) + 17 * n_lccs + 2 * n_buses)
    sizehint!(vals, 4 * SparseArrays.nnz(Y_bus_eff) + 17 * n_lccs + 2 * n_buses)

    Yrows = SparseArrays.rowvals(Y_bus_eff)
    bus_types_at_t = view(data.bus_type, :, time_step)
    independent_ref = _multi_swing_ref_indices(data.bus_type, subnetworks, time_step)
    @inbounds for col in 1:n_buses
        col_off = Int(bus_state_offset[col])
        col_bs = bus_block_size[col]
        is_ref_col = bus_types_at_t[col] == PSY.ACBusTypes.REF
        # Diagonal block unconditionally: a bus whose Ybus column has NO stored
        # diagonal (an AC-isolated swing, e.g. a DC-tie voltage holder with zero
        # AC branches) still needs its own block — its rows/columns otherwise
        # never enter the pattern and the value writers hit "missing entry".
        for r in 0:(Int(col_bs) - 1)
            for c in 0:(Int(col_bs) - 1)
                push!(rows, J_INDEX_TYPE(col_off + r))
                push!(cols, J_INDEX_TYPE(col_off + c))
                push!(vals, 0.0)
            end
        end
        for j in SparseArrays.nzrange(Y_bus_eff, col)
            row = Yrows[j]
            # Diagonal block already pushed above.
            if row == col
                continue
            end
            # REF columns hold (P_gen, Q_gen); neighbors' rows don't depend on them.
            if is_ref_col
                continue
            end
            row_off = Int(bus_state_offset[row])
            row_bs = bus_block_size[row]
            # Every block is 2×2 in MCPB; off-diagonal blocks reserve both
            # (e, f) columns for ALL non-REF rows (PQ constant, PV nonlinear).
            n_cols_to_write = Int(col_bs)  # 2 for all bus types
            n_rows_to_write = Int(row_bs)  # 2 for all bus types
            for r in 0:(n_rows_to_write - 1)
                for c in 0:(n_cols_to_write - 1)
                    push!(rows, J_INDEX_TYPE(row_off + r))
                    push!(cols, J_INDEX_TYPE(col_off + c))
                    push!(vals, 0.0)
                end
            end
        end
    end

    # Distributed-slack cross-terms: ∂F_k_{r,i}/∂x[bus_state_offset[ref]].
    # The Y_bus loop above SKIPS off-diagonal entries for REF columns (REF
    # state vars are (P_gen, Q_gen), not (e, f)), so `(k_off, ref_off)` is not
    # yet in the pattern for any non-self bus_k — push it here (gated only on
    # `bus_k != ref_bus`, since the REF diagonal block covers `bus_k == ref_bus`).
    for (ref_bus, subnetwork_buses) in subnetworks
        # Multi-swing island: each swing self-balances at its own P-slot; there is
        # no single distributed scalar to couple, so no cross-terms are structural
        # here (mirrors polar's / rect's independent-REF handling).
        ref_bus in independent_ref && continue
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

    # LCC tail entries: mirror rect structure (REF residual rows identical to
    # rect's; the rect helper is layout-generic over bus_state_offset so it is
    # reused directly).
    if n_lccs > 0
        _create_rect_ci_lcc_structure!(
            rows, cols, vals, data, bus_state_offset, total_bus_state,
        )
    end

    # VSC structural slots are identical to the rectangular layout (the imag-first swap affects
    # values, not the (row, col) pattern), so the rect builder is reused directly.
    if has_dc_network(dcn)
        _create_rect_ci_vsc_structure!(
            rows, cols, vals, dcn, bus_state_offset, total_bus_state, n_lccs,
        )
    end

    return SparseArrays.sparse(rows, cols, vals, total_state, total_state)
end

"""
    _build_mixed_diag_nz_cache(Jv, bus_state_offset)

`diag_base_nz::Matrix{Int}` (`4 × n_buses`): nzval indices of each per-bus 2×2
diagonal block, rows ordered `(off,off), (off,off+1), (off+1,off),
(off+1,off+1)`. No `pv_extra_nz` (MCPB PV blocks are 2×2, no Q column).
"""
function _build_mixed_diag_nz_cache(
    Jv::SparseMatrixCSC{Float64, J_INDEX_TYPE},
    bus_state_offset::Vector{REC_INDEX_TYPE},
)
    n_buses = length(bus_state_offset) - 1
    diag_base_nz = Matrix{Int}(undef, 4, n_buses)
    @inbounds for i in 1:n_buses
        off = Int(bus_state_offset[i])
        diag_base_nz[1, i] = _jv_nz_index(Jv, off, off)
        diag_base_nz[2, i] = _jv_nz_index(Jv, off, off + 1)
        diag_base_nz[3, i] = _jv_nz_index(Jv, off + 1, off)
        diag_base_nz[4, i] = _jv_nz_index(Jv, off + 1, off + 1)
    end
    return diag_base_nz
end

"""
    _build_offdiag_pv_nz_cache(Jv, Y_bus_eff, bus_state_offset, bus_types)

Cache nzval indices for the PV power-balance row's off-diagonal `(e_k, f_k)`
columns. Each `2 × n_pv_pairs` column is one ordered `(PV bus i, neighbor
k≠i)` pair (REF neighbors excluded), rows `Jv[i_off, k_off]` and
`Jv[i_off, k_off+1]`. The PV voltage-constraint row has no off-diagonals.
"""
function _build_offdiag_pv_nz_cache(
    Jv::SparseMatrixCSC{Float64, J_INDEX_TYPE},
    Y_bus_eff::SparseMatrixCSC{ComplexF64, Int},
    bus_state_offset::Vector{REC_INDEX_TYPE},
    bus_types::AbstractVector{PSY.ACBusTypes},
)
    n_buses = length(bus_state_offset) - 1
    Yrows = SparseArrays.rowvals(Y_bus_eff)
    Yvals = SparseArrays.nonzeros(Y_bus_eff)
    offdiag_pv_i = Int[]
    offdiag_pv_k = Int[]
    offdiag_pv_y = ComplexF64[]
    cols_nz = Vector{Int}[]
    @inbounds for col in 1:n_buses
        # REF columns hold (P_gen, Q_gen): no (e, f) off-diagonal entries exist.
        bus_types[col] == PSY.ACBusTypes.REF && continue
        col_off = Int(bus_state_offset[col])
        for j in SparseArrays.nzrange(Y_bus_eff, col)
            row = Yrows[j]
            row == col && continue                       # diagonal block
            bus_types[row] != PSY.ACBusTypes.PV && continue  # only PV rows
            row_off = Int(bus_state_offset[row])
            push!(offdiag_pv_i, row)
            push!(offdiag_pv_k, col)
            # Y_bus_eff is stored CSC: column `col`, structural row `row` here
            # is the matrix entry Y_bus_eff[row, col] == Y_bus_eff[i, k].
            push!(offdiag_pv_y, Yvals[j])
            # PV voltage-constraint row (slot row_off+1) has no off-diagonals;
            # only the power-balance row at slot row_off is cached.
            push!(
                cols_nz,
                Int[
                    _jv_nz_index(Jv, row_off, col_off),
                    _jv_nz_index(Jv, row_off, col_off + 1),
                ],
            )
        end
    end
    n_pairs = length(cols_nz)
    offdiag_pv_nz = Matrix{Int}(undef, 2, n_pairs)
    @inbounds for p in 1:n_pairs
        offdiag_pv_nz[1, p] = cols_nz[p][1]
        offdiag_pv_nz[2, p] = cols_nz[p][2]
    end
    return offdiag_pv_nz, offdiag_pv_i, offdiag_pv_k, offdiag_pv_y
end

# NOTE: _build_slack_nz_cache and _build_lcc_nz_cache are layout-generic over
# bus_state_offset; the rect definitions in rectangular_ci_power_flow_jacobian.jl
# are reused verbatim (mirror-for-validation). Likewise _jv_nz_index and
# _create_rect_ci_lcc_structure! are shared.

"""
Fill the constant Y_bus off-diagonal blocks for PQ rows only. PV off-diagonals
are nonlinear (left 0.0 here, written per-iteration via `offdiag_pv_nz`); REF
rows have no `(e, f)` off-diagonals.

Only `−I_acc` carries the off-diagonal dependence (the I_spec terms depend on
bus `i` alone), and `∂Ir_acc/∂(e_k,f_k) = (G_ik, −B_ik)`,
`∂Ii_acc/∂(e_k,f_k) = (B_ik, G_ik)`. With the MCPB imag-first slot order this
gives the 2×2 block

    Jv[off,   k_off] = −B_ik   Jv[off,   k_off+1] = −G_ik
    Jv[off+1, k_off] = −G_ik   Jv[off+1, k_off+1] = +B_ik

i.e. rect's off-diagonal block with its two rows swapped (rect is real-first;
MCPB PQ is imag-first), matching the residual's PQ slot swap.
"""
function _populate_mixed_constant_yb_blocks!(
    Jv::SparseMatrixCSC{Float64, J_INDEX_TYPE},
    Y_bus_eff::SparseMatrixCSC{ComplexF64, Int},
    bus_state_offset::Vector{REC_INDEX_TYPE},
    bus_types::AbstractVector{PSY.ACBusTypes},
)
    n_buses = length(bus_types)
    Yvals = SparseArrays.nonzeros(Y_bus_eff)
    Yrows = SparseArrays.rowvals(Y_bus_eff)
    @inbounds for col in 1:n_buses
        # Skip REF columns: REF state vars are (P_gen, Q_gen), not (e, f).
        bus_types[col] == PSY.ACBusTypes.REF && continue
        col_off = Int(bus_state_offset[col])
        for j in SparseArrays.nzrange(Y_bus_eff, col)
            row = Yrows[j]
            row == col && continue  # diagonal block handled per iteration
            row_off = Int(bus_state_offset[row])
            y = Yvals[j]
            g = real(y)
            b = imag(y)
            row_bt = bus_types[row]
            if row_bt == PSY.ACBusTypes.PQ
                # imag-first PQ block:
                Jv[row_off, col_off] = -b          # ∂F[off]/∂e_k   (imag)
                Jv[row_off, col_off + 1] = -g      # ∂F[off]/∂f_k   (imag)
                Jv[row_off + 1, col_off] = -g      # ∂F[off+1]/∂e_k (real)
                Jv[row_off + 1, col_off + 1] = b   # ∂F[off+1]/∂f_k (real)
            elseif row_bt == PSY.ACBusTypes.REF
                # REF residual is rect-verbatim (NOT swapped): the
                # current-mismatch rows depend on neighbor (e_k, f_k) through
                # −Ir_acc/−Ii_acc, giving rect's constant real-first block
                # [[−G, B], [−B, −G]]. (REF rows ARE present in the structure;
                # only REF columns are skipped.)
                Jv[row_off, col_off] = -g          # ∂F[off]/∂e_k   (real)
                Jv[row_off, col_off + 1] = b       # ∂F[off]/∂f_k   (real)
                Jv[row_off + 1, col_off] = -b      # ∂F[off+1]/∂e_k (imag)
                Jv[row_off + 1, col_off + 1] = -g  # ∂F[off+1]/∂f_k (imag)
            end
            # PV off-diagonals are nonlinear — filled per-iteration via
            # offdiag_pv_nz by _update_mixed_cpb_jacobian_values!.
        end
    end
    return
end

"""Update state-dependent MCPB Jacobian entries (per-bus diagonal blocks, PV
off-diagonals, slack cross-terms, LCC tail) through the nzval-index caches.
Reads the residual's shared state caches, so the residual must have been
evaluated on the current `x` first."""
function _update_mixed_cpb_jacobian_values!(
    Jv::SparseMatrixCSC{Float64, J_INDEX_TYPE},
    data::ACPowerFlowData,
    Y_diag::Vector{ComplexF64},
    e_state::Vector{Float64},
    f_state::Vector{Float64},
    P_eff_cache::Vector{Float64},
    Q_eff_cache::Vector{Float64},
    const_I_P::Vector{Float64},
    const_I_Q::Vector{Float64},
    Ir_acc::Vector{Float64},
    Ii_acc::Vector{Float64},
    bus_slack_participation_factors::SparseVector{Float64, Int},
    independent_ref::Set{Int},
    bus_state_offset::Vector{REC_INDEX_TYPE},
    total_bus_state::Int,
    diag_base_nz::Matrix{Int},
    offdiag_pv_nz::Matrix{Int},
    offdiag_pv_i::Vector{Int},
    offdiag_pv_k::Vector{Int},
    offdiag_pv_y::Vector{ComplexF64},
    slack_nz_idx_e::Vector{Int},
    slack_nz_idx_f::Vector{Int},
    slack_c_k::Vector{Float64},
    lcc_nz::Matrix{Int},
    vsc_nz::VSCJacobianNZCache,
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
            _update_mixed_pq_diag_block!(Jvnz, diag_base_nz, i, e_i, f_i,
                Y_diag[i], P_eff_cache[i], Q_eff_cache[i],
                const_I_P[i], const_I_Q[i])
        elseif bt == PSY.ACBusTypes.PV
            _update_mixed_pv_diag_block!(Jvnz, diag_base_nz, i, e_i, f_i,
                Y_diag[i], Ir_acc[i], Ii_acc[i], const_I_P[i])
        elseif bt == PSY.ACBusTypes.REF
            if i in independent_ref
                # Multi-swing island: this swing self-balances at its own P-slot
                # (P_gen = x[off]), i.e. identity distribution (c_ref = 1.0).
                _update_ref_diag_block!(Jvnz, diag_base_nz, i, e_i, f_i, 1.0)
            else
                c_ref = bus_slack_participation_factors[i]
                _update_ref_diag_block!(Jvnz, diag_base_nz, i, e_i, f_i, c_ref)
            end
        end
    end

    # PV power-balance-row off-diagonals: ∂(e_i·Ir_i + f_i·Ii_i − P_i)/∂(e_k, f_k).
    # With Ir_acc[i] += G_ik·e_k − B_ik·f_k and Ii_acc[i] += G_ik·f_k + B_ik·e_k:
    #   ∂/∂e_k = e_i·G_ik + f_i·B_ik ;  ∂/∂f_k = −e_i·B_ik + f_i·G_ik
    @inbounds for p in eachindex(offdiag_pv_i)
        i = offdiag_pv_i[p]
        e_i = e_state[i]
        f_i = f_state[i]
        y_ik = offdiag_pv_y[p]
        g_ik = real(y_ik)
        b_ik = imag(y_ik)
        Jvnz[offdiag_pv_nz[1, p]] = e_i * g_ik + f_i * b_ik
        Jvnz[offdiag_pv_nz[2, p]] = -e_i * b_ik + f_i * g_ik
    end

    # Distributed-slack cross-terms ∂F_{bus_k}/∂x[ref_off]. The slack increment
    # `c_k·(x[ref_off] − P_net_set[ref])` enters `P_eff_cache[bus_k]`.
    # Only PV buses appear in the slack cache (PQ has c_k=0; REF is excluded by
    # _build_slack_nz_cache) — so only the PV case is reachable here:
    #   PV power-balance row  F[off]   = e·Ir + f·Ii − P_i ⇒ ∂/∂x_ref = −c_k
    #      |V|² row            F[off+1] = (e²+f²) − V_set²   independent of slack ⇒ 0
    # (slack_nz_idx_e indexes row k_off, slack_nz_idx_f indexes row k_off+1.)
    @inbounds for k in eachindex(slack_nz_idx_e)
        c_k = slack_c_k[k]
        Jvnz[slack_nz_idx_e[k]] = -c_k
        Jvnz[slack_nz_idx_f[k]] = 0.0
    end

    if n_lccs > 0
        _set_entries_for_lcc_mixed!(
            data, Jvnz, diag_base_nz, lcc_nz,
            e_state, f_state, bus_state_offset, time_step,
        )
    end
    dcn = get_dc_network(data)
    if has_dc_network(dcn)
        _set_entries_for_vsc_rect_mcpb!(
            Jvnz, diag_base_nz, vsc_nz, dcn, e_state, f_state,
            view(data.bus_type, :, time_step), time_step, true,
        )
    end
    return
end

"""
MCPB PQ diagonal 2×2 (imag-first: slot `off` = imag balance, `off+1` = real
balance). Same divided-current + const-I expressions as rect's
`_update_pq_diag_block!` with the two rows swapped — diag rows `1,2,3,4` are
rect's rows `3,4,1,2` — matching the residual's PQ slot swap.
"""
@inline function _update_mixed_pq_diag_block!(
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
    # Guard 1/|V|² and 1/|V| against a degenerate (e,f) (warm start /
    # enhanced flat start can produce a near-zero voltage), mirroring the
    # residual's V_FLOOR2 floor so the Jacobian stays finite.
    V_sq = max(e^2 + f^2, V_FLOOR2)
    inv_V_sq = 1.0 / V_sq
    inv_Vm = 1.0 / sqrt(V_sq)
    g_ii = real(y_ii)
    b_ii = imag(y_ii)
    Is_r = (P_eff * e + Q_eff * f) * inv_V_sq
    Is_i = (P_eff * f - Q_eff * e) * inv_V_sq
    term_r = (const_I_P * e + const_I_Q * f) * inv_Vm * inv_V_sq
    term_i = (const_I_P * f - const_I_Q * e) * inv_Vm * inv_V_sq
    # rect real-first row 1 (∂real/∂e) and row 2 (∂real/∂f):
    real_de = (P_eff - 2 * e * Is_r) * inv_V_sq - g_ii - e * term_r
    real_df = (Q_eff - 2 * f * Is_r) * inv_V_sq + b_ii - f * term_r
    # rect real-first row 3 (∂imag/∂e) and row 4 (∂imag/∂f):
    imag_de = (-Q_eff - 2 * e * Is_i) * inv_V_sq - b_ii - e * term_i
    imag_df = (P_eff - 2 * f * Is_i) * inv_V_sq - g_ii - f * term_i
    # MCPB imag-first slot assignment (rows swapped):
    @inbounds Jvnz[diag_base_nz[1, i]] = imag_de  # ∂(imag row)/∂e_i
    @inbounds Jvnz[diag_base_nz[2, i]] = imag_df  # ∂(imag row)/∂f_i
    @inbounds Jvnz[diag_base_nz[3, i]] = real_de  # ∂(real row)/∂e_i
    @inbounds Jvnz[diag_base_nz[4, i]] = real_df  # ∂(real row)/∂f_i
    return
end

"""
MCPB PV diagonal 2×2 (slot `off` = eq.7 power balance `e·Ir + f·Ii − P`,
`off+1` = eq.8 `|V|² − V_set²`). Ir/Ii include the `Y_ii` diagonal term; a
const-I `P_eff = P_net_const − cIP·Vm` adds `∂P/∂(e,f) = −cIP·(e,f)/Vm`.
"""
@inline function _update_mixed_pv_diag_block!(
    Jvnz::Vector{Float64},
    diag_base_nz::Matrix{Int},
    i::Int,
    e::Float64,
    f::Float64,
    y_ii::ComplexF64,
    Ir::Float64,
    Ii::Float64,
    const_I_P::Float64,
)
    # Guard 1/|V| against a degenerate (e,f), mirroring the residual's
    # V_FLOOR2 floor (see _update_mixed_pq_diag_block!).
    inv_Vm = 1.0 / sqrt(max(e^2 + f^2, V_FLOOR2))
    g_ii = real(y_ii)
    b_ii = imag(y_ii)
    # eq.7 row: d/de = Ir + e·G_ii + f·B_ii − ∂P/∂e, with ∂P/∂e = −cIP·e/Vm.
    @inbounds Jvnz[diag_base_nz[1, i]] =
        Ir + e * g_ii + f * b_ii + const_I_P * e * inv_Vm
    @inbounds Jvnz[diag_base_nz[2, i]] =
        Ii - e * b_ii + f * g_ii + const_I_P * f * inv_Vm
    # eq.8 row: ∂(e²+f²−V_set²)/∂(e,f) = (2e, 2f).
    @inbounds Jvnz[diag_base_nz[3, i]] = 2 * e
    @inbounds Jvnz[diag_base_nz[4, i]] = 2 * f
    return
end

"""
MCPB LCC tail. PQ/REF terminals match rect's `_set_entries_for_lcc_rect!`,
with PQ using the imag-first slot order. At a PV terminal the LCC current
enters via Ir_acc/Ii_acc, so its contribution lands in the eq.7 power-balance
row (`F[off] += e_i·∂ΔIr_lcc + f_i·∂ΔIi_lcc`); the eq.8 |V|² row gets none.
Tail-row and tail×tail entries are identical to rect.
"""
function _set_entries_for_lcc_mixed!(
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

        if iszero(data.lcc.i_dc[i, time_step])
            # 0-current converter: P_lcc ≡ 0, so it contributes nothing to the bus
            # rows and its P-setpoint / DC-line-balance rows are vacuous. Zero the
            # bus-coupling entries and pin the two tap states with identity rows
            # (matching _write_lcc_tail!) — into existing structure positions only.
            # All 24 LCC entries are ∝ i_dc except the two tap diagonals, so zero
            # the whole block and pin: F_t_fb (P-setpoint) → tap_r at row 15,
            # F_t_tb (DC-line balance) → tap_i at row 18. The α-limit identity
            # diagonals are not in lcc_nz (set at pattern build), so they survive.
            Jvnz[lcc_nz[1:24, i]] .= 0.0
            Jvnz[lcc_nz[15, i]] = 1.0
            Jvnz[lcc_nz[18, i]] = 1.0
            continue
        end

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
        # Inverter uses −xtr_i: the corrected ϕ_i subtracts the commutation drop, flipping
        # the sign of ∂ϕ_i/∂{V,t} (linear in x_t) vs the rectifier form (see _lcc_utils).
        dphi_dV_fb = _dphi_dV_lcc(xtr_r, s.i_dc, Vm_fb, s.tap_r, phi_r)
        dphi_dV_tb = _dphi_dV_lcc(-xtr_i, s.i_dc, Vm_tb, s.tap_i, phi_i)
        dphi_dtap_r = _dphi_dt_lcc(xtr_r, s.i_dc, Vm_fb, s.tap_r, phi_r)
        dphi_dtap_i = _dphi_dt_lcc(-xtr_i, s.i_dc, Vm_tb, s.tap_i, phi_i)
        dphi_dα_r = _dphi_dα_lcc(alpha_r, phi_r)
        dphi_dα_i = -_dphi_dα_lcc(alpha_i, phi_i)

        # FB-side bus contribution. The LCC current enters Ir_acc/Ii_acc as
        #   ΔIr = A·u(ϕ),  ΔIi = A·w(ϕ)
        # so the residual's current-mismatch rows pick up −ΔIr/−ΔIi.
        #   rect_dIr_de = −[−A·f·w·invV² − A·w·∂ϕ/∂e]  (rect diag_base_nz[1])
        # We compute the four rect-real-first overlays then route per bus type.
        A_fb = s.tap_r * SQRT6_DIV_PI * s.i_dc / Vm_fb
        u_fb = cos_phi_r * e_fb + sin_phi_r * f_fb
        w_fb = cos_phi_r * f_fb - sin_phi_r * e_fb
        dphi_de_fb = dphi_dV_fb * e_fb / Vm_fb
        dphi_df_fb = dphi_dV_fb * f_fb / Vm_fb
        # rect real-first overlays (∂F_real/∂e, ∂F_real/∂f, ∂F_imag/∂e,
        # ∂F_imag/∂f) where F = I_spec − I_network and ΔI_lcc adds to network:
        ovl_fb_re_de = -A_fb * f_fb * w_fb * inv_Vsq_fb - A_fb * w_fb * dphi_de_fb
        ovl_fb_re_df = A_fb * e_fb * w_fb * inv_Vsq_fb - A_fb * w_fb * dphi_df_fb
        ovl_fb_im_de = A_fb * f_fb * u_fb * inv_Vsq_fb + A_fb * u_fb * dphi_de_fb
        ovl_fb_im_df = -A_fb * e_fb * u_fb * inv_Vsq_fb + A_fb * u_fb * dphi_df_fb
        if bus_type_fb == PSY.ACBusTypes.PQ
            # MCPB PQ is IMAG-FIRST: slot 1,2 = imag row; slot 3,4 = real row.
            Jvnz[diag_base_nz[1, fb]] += ovl_fb_im_de
            Jvnz[diag_base_nz[2, fb]] += ovl_fb_im_df
            Jvnz[diag_base_nz[3, fb]] += ovl_fb_re_de
            Jvnz[diag_base_nz[4, fb]] += ovl_fb_re_df
        elseif bus_type_fb == PSY.ACBusTypes.PV
            # PV eq.7 row F[off] = e_i·Ir + f_i·Ii − P. The LCC adds ΔIr/ΔIi
            # to Ir_acc/Ii_acc, so ∂F[off]/∂v = e_i·∂ΔIr/∂v + f_i·∂ΔIi/∂v.
            # ∂ΔIr/∂e = −ovl_fb_re_de (ovl is for −ΔIr), ∂ΔIi/∂e = −ovl_fb_im_de.
            Jvnz[diag_base_nz[1, fb]] +=
                -e_fb * ovl_fb_re_de - f_fb * ovl_fb_im_de
            Jvnz[diag_base_nz[2, fb]] +=
                -e_fb * ovl_fb_re_df - f_fb * ovl_fb_im_df
            # eq.8 |V|² row: no LCC contribution.
        end
        # FB-side cross-terms ∂F_lcc_fb / ∂(t_r, α_r).
        cross_fb_re_tap = -A_fb * u_fb / s.tap_r - A_fb * w_fb * dphi_dtap_r
        cross_fb_re_α = -A_fb * w_fb * dphi_dα_r
        cross_fb_im_tap = -A_fb * w_fb / s.tap_r + A_fb * u_fb * dphi_dtap_r
        cross_fb_im_α = A_fb * u_fb * dphi_dα_r
        if bus_type_fb == PSY.ACBusTypes.PV
            # Route into eq.7 power-balance row: e·∂ΔIr + f·∂ΔIi (sign as above).
            Jvnz[lcc_nz[1, i]] = -e_fb * cross_fb_re_tap - f_fb * cross_fb_im_tap
            Jvnz[lcc_nz[2, i]] = -e_fb * cross_fb_re_α - f_fb * cross_fb_im_α
            Jvnz[lcc_nz[3, i]] = 0.0
            Jvnz[lcc_nz[4, i]] = 0.0
        elseif bus_type_fb == PSY.ACBusTypes.PQ
            # imag-first: lcc_nz rows 1,2 feed the imag bus row; 3,4 the real.
            Jvnz[lcc_nz[1, i]] = cross_fb_im_tap
            Jvnz[lcc_nz[2, i]] = cross_fb_im_α
            Jvnz[lcc_nz[3, i]] = cross_fb_re_tap
            Jvnz[lcc_nz[4, i]] = cross_fb_re_α
        else  # REF: rect-verbatim (real-first)
            Jvnz[lcc_nz[1, i]] = cross_fb_re_tap
            Jvnz[lcc_nz[2, i]] = cross_fb_re_α
            Jvnz[lcc_nz[3, i]] = cross_fb_im_tap
            Jvnz[lcc_nz[4, i]] = cross_fb_im_α
        end

        # TB-side bus contribution.
        A_tb = s.tap_i * SQRT6_DIV_PI * s.i_dc / Vm_tb
        u_tb = cos_phi_i * e_tb + sin_phi_i * f_tb
        w_tb = cos_phi_i * f_tb - sin_phi_i * e_tb
        dphi_de_tb = dphi_dV_tb * e_tb / Vm_tb
        dphi_df_tb = dphi_dV_tb * f_tb / Vm_tb
        ovl_tb_re_de = -A_tb * f_tb * w_tb * inv_Vsq_tb - A_tb * w_tb * dphi_de_tb
        ovl_tb_re_df = A_tb * e_tb * w_tb * inv_Vsq_tb - A_tb * w_tb * dphi_df_tb
        ovl_tb_im_de = A_tb * f_tb * u_tb * inv_Vsq_tb + A_tb * u_tb * dphi_de_tb
        ovl_tb_im_df = -A_tb * e_tb * u_tb * inv_Vsq_tb + A_tb * u_tb * dphi_df_tb
        if bus_type_tb == PSY.ACBusTypes.PQ
            Jvnz[diag_base_nz[1, tb]] += ovl_tb_im_de
            Jvnz[diag_base_nz[2, tb]] += ovl_tb_im_df
            Jvnz[diag_base_nz[3, tb]] += ovl_tb_re_de
            Jvnz[diag_base_nz[4, tb]] += ovl_tb_re_df
        elseif bus_type_tb == PSY.ACBusTypes.PV
            Jvnz[diag_base_nz[1, tb]] +=
                -e_tb * ovl_tb_re_de - f_tb * ovl_tb_im_de
            Jvnz[diag_base_nz[2, tb]] +=
                -e_tb * ovl_tb_re_df - f_tb * ovl_tb_im_df
        end
        cross_tb_re_tap = -A_tb * u_tb / s.tap_i - A_tb * w_tb * dphi_dtap_i
        cross_tb_re_α = -A_tb * w_tb * dphi_dα_i
        cross_tb_im_tap = -A_tb * w_tb / s.tap_i + A_tb * u_tb * dphi_dtap_i
        cross_tb_im_α = A_tb * u_tb * dphi_dα_i
        if bus_type_tb == PSY.ACBusTypes.PV
            Jvnz[lcc_nz[5, i]] = -e_tb * cross_tb_re_tap - f_tb * cross_tb_im_tap
            Jvnz[lcc_nz[6, i]] = -e_tb * cross_tb_re_α - f_tb * cross_tb_im_α
            Jvnz[lcc_nz[7, i]] = 0.0
            Jvnz[lcc_nz[8, i]] = 0.0
        elseif bus_type_tb == PSY.ACBusTypes.PQ
            Jvnz[lcc_nz[5, i]] = cross_tb_im_tap
            Jvnz[lcc_nz[6, i]] = cross_tb_im_α
            Jvnz[lcc_nz[7, i]] = cross_tb_re_tap
            Jvnz[lcc_nz[8, i]] = cross_tb_re_α
        else  # REF: rect-verbatim
            Jvnz[lcc_nz[5, i]] = cross_tb_re_tap
            Jvnz[lcc_nz[6, i]] = cross_tb_re_α
            Jvnz[lcc_nz[7, i]] = cross_tb_im_tap
            Jvnz[lcc_nz[8, i]] = cross_tb_im_α
        end

        # LCC tail row entries (∂F_t/∂V chain-ruled into (e, f)). These are the
        # LCC tail residual rows (idx_tap_r/idx_tap_i, independent of bus type
        # routing) — identical to rect. Rows 9,10 are ∂F_t_fb/∂(e_fb,f_fb);
        # 11,12 are ∂F_t_tb/∂(e_fb,f_fb). F_t_fb's fb-side dependence is nonzero
        # only with a rectifier-side set point (s.d_Ft_fb_d_V_fb pre-zeroed
        # otherwise); F_t_tb always sees P_lcc_from, so 11,12 use dP_dV_fb.
        if bus_type_fb == PSY.ACBusTypes.PQ || bus_type_fb == PSY.ACBusTypes.PV
            de_dV_fb = e_fb / Vm_fb
            df_dV_fb = f_fb / Vm_fb
            Jvnz[lcc_nz[9, i]] = s.d_Ft_fb_d_V_fb * de_dV_fb
            Jvnz[lcc_nz[10, i]] = s.d_Ft_fb_d_V_fb * df_dV_fb
            Jvnz[lcc_nz[11, i]] = s.dP_dV_fb * de_dV_fb
            Jvnz[lcc_nz[12, i]] = s.dP_dV_fb * df_dV_fb
        else
            Jvnz[lcc_nz[9, i]] = 0.0
            Jvnz[lcc_nz[10, i]] = 0.0
            Jvnz[lcc_nz[11, i]] = 0.0
            Jvnz[lcc_nz[12, i]] = 0.0
        end
        # Rows 13,14 are ∂F_t_tb/∂(e_tb,f_tb) (always P_lcc_to); 21,22 are
        # ∂F_t_fb/∂(e_tb,f_tb), nonzero only with an inverter-side set point.
        if bus_type_tb == PSY.ACBusTypes.PQ || bus_type_tb == PSY.ACBusTypes.PV
            de_dV_tb = e_tb / Vm_tb
            df_dV_tb = f_tb / Vm_tb
            Jvnz[lcc_nz[13, i]] = s.dP_dV_tb * de_dV_tb
            Jvnz[lcc_nz[14, i]] = s.dP_dV_tb * df_dV_tb
            Jvnz[lcc_nz[21, i]] = s.d_Ft_fb_d_V_tb * de_dV_tb
            Jvnz[lcc_nz[22, i]] = s.d_Ft_fb_d_V_tb * df_dV_tb
        else
            Jvnz[lcc_nz[13, i]] = 0.0
            Jvnz[lcc_nz[14, i]] = 0.0
            Jvnz[lcc_nz[21, i]] = 0.0
            Jvnz[lcc_nz[22, i]] = 0.0
        end

        # Tail × tail block (shared with polar/rect via _lcc_jacobian_scalars).
        # F_t_fb's tap/α dependence switches sides with the set point; the
        # scalars helper zeroes the inactive side, so all four slots
        # (15,16 rectifier; 23,24 inverter) are written unconditionally.
        Jvnz[lcc_nz[15, i]] = s.d_Ft_fb_d_tap_r
        Jvnz[lcc_nz[16, i]] = s.d_Ft_fb_d_alpha_r
        Jvnz[lcc_nz[17, i]] = s.d_Ft_tb_d_tap_r
        Jvnz[lcc_nz[18, i]] = s.d_Ft_tb_d_tap_i
        Jvnz[lcc_nz[19, i]] = s.d_Ft_tb_d_alpha_r
        Jvnz[lcc_nz[20, i]] = s.d_Ft_tb_d_alpha_i
        Jvnz[lcc_nz[23, i]] = s.d_Ft_fb_d_tap_i
        Jvnz[lcc_nz[24, i]] = s.d_Ft_fb_d_alpha_i
    end
    return
end
