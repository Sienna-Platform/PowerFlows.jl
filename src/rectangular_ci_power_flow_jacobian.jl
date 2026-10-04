"""
    struct ACRectangularCIJacobian

Jacobian functor for the rectangular power-mismatch AC power flow
([`ACRectangularCIResidual`](@ref)). Every Y-bus nonzero `(i, j)` owns a 2×2 block, REF columns
included, so the pattern does not depend on the PQ/PV/REF split (a Q-limit flip keeps it); only
the distributed-slack cross-terms follow the participation factors. Per-iteration updates write
`nonzeros(Jv)` through index caches built once at construction.

# Fields
- `Jv::SparseMatrixCSC{Float64, J_INDEX_TYPE}` — Jacobian values
- `Y_bus_eff`, `e_state`, `f_state`, `Ir_acc`, `Ii_acc`, `const_I_P`, `const_I_Q`,
  `independent_ref` — shared with the residual
- `Y_diag::Vector{ComplexF64}` — cached `Y_bus_eff` diagonal
- `yb_nz::Matrix{Int}` — `4 × nnz(Y_bus_eff)`: the `Jv` entries `(P,e) (P,f) (Q,e) (Q,f)` of
  each Y-bus nonzero's block (row bus, column bus)
- `diag_nz::Matrix{Int}` — `4 × n_buses`, same order, for each bus's own block
- `slack_nz_idx_e`, `slack_nz_idx_f`, `slack_c_k` — distributed-slack
  cross-terms (see [`_build_slack_nz_cache`](@ref))
- `lcc_nz::Matrix{Int}`, `vsc_nz::VSCJacobianNZCache` — tail entries
"""
struct ACRectangularCIJacobian
    Jv::SparseMatrixCSC{Float64, J_INDEX_TYPE}
    Y_bus_eff::SparseMatrixCSC{ComplexF64, Int}
    Y_diag::Vector{ComplexF64}
    e_state::Vector{Float64}
    f_state::Vector{Float64}
    Ir_acc::Vector{Float64}
    Ii_acc::Vector{Float64}
    const_I_P::Vector{Float64}
    const_I_Q::Vector{Float64}
    bus_slack_participation_factors::SparseVector{Float64, Int}
    independent_ref::Set{Int}
    bus_state_offset::Vector{REC_INDEX_TYPE}
    total_bus_state::Int
    yb_nz::Matrix{Int}
    diag_nz::Matrix{Int}
    slack_nz_idx_e::Vector{Int}
    slack_nz_idx_f::Vector{Int}
    slack_c_k::Vector{Float64}
    lcc_nz::Matrix{Int}
    vsc_nz::VSCJacobianNZCache
end

function ACRectangularCIJacobian(
    data::ACPowerFlowData,
    residual::ACRectangularCIResidual,
    time_step::Int64,
)
    Jv0 = _create_rect_ci_jacobian_structure(data, residual)
    Y = residual.Y_bus_eff
    n_buses = first(size(data.bus_type))
    Y_diag = Vector{ComplexF64}(undef, n_buses)
    @inbounds for i in 1:n_buses
        Y_diag[i] = Y[i, i]
    end
    yb_nz = _build_rect_yb_nz_cache(Jv0, Y, residual.bus_state_offset)
    diag_nz = _build_mixed_diag_nz_cache(Jv0, residual.bus_state_offset)
    slack_nz_idx_e, slack_nz_idx_f, _, slack_c_k =
        _build_slack_nz_cache(
            Jv0, residual.bus_state_offset, residual.subnetworks,
            residual.bus_slack_participation_factors, residual.independent_ref,
        )
    n_lccs = size(data.lcc.p_set, 1)
    lcc_nz = _build_lcc_nz_cache(
        Jv0, data, residual.bus_state_offset, residual.total_bus_state, n_lccs,
    )
    vsc_nz = _build_vsc_nz_cache(
        Jv0, get_dc_network(data), residual.bus_state_offset,
        residual.total_bus_state, n_lccs,
    )
    J = ACRectangularCIJacobian(
        Jv0,
        Y,
        Y_diag,
        residual.e_state,
        residual.f_state,
        residual.Ir_acc,
        residual.Ii_acc,
        residual.const_I_P,
        residual.const_I_Q,
        residual.bus_slack_participation_factors,
        residual.independent_ref,
        residual.bus_state_offset,
        residual.total_bus_state,
        yb_nz,
        diag_nz,
        slack_nz_idx_e,
        slack_nz_idx_f,
        slack_c_k,
        lcc_nz,
        vsc_nz,
    )
    J(data, time_step)
    return J
end

function (J::ACRectangularCIJacobian)(data::ACPowerFlowData, time_step::Int64)
    _update_rect_ci_jacobian_values!(J, data, time_step)
    return
end

function (J::ACRectangularCIJacobian)(
    data::ACPowerFlowData,
    Jv::SparseMatrixCSC{Float64, J_INDEX_TYPE},
    time_step::Int64,
)
    _update_rect_ci_jacobian_values!(J, data, time_step)
    copyto!(Jv, J.Jv)
    return
end

"""
Build the sparsity pattern of the rectangular power-mismatch Jacobian: a 2×2 block per bus and
per Y-bus nonzero (REF columns included, holding structural zeros), the distributed-slack
cross-terms `(k, ref)`, and the LCC and VSC tail entries.
"""
function _create_rect_ci_jacobian_structure(
    data::ACPowerFlowData,
    residual::ACRectangularCIResidual,
)
    Y_bus_eff = residual.Y_bus_eff
    bus_state_offset = residual.bus_state_offset
    total_bus_state = residual.total_bus_state
    rows = J_INDEX_TYPE[]
    cols = J_INDEX_TYPE[]
    vals = Float64[]
    n_buses = first(size(data.bus_type))
    n_lccs = size(data.lcc.p_set, 1)
    dcn = get_dc_network(data)
    total_state = total_bus_state + state_tail_length(data, dcn)
    n_hint = 4 * (SparseArrays.nnz(Y_bus_eff) + n_buses) + 26 * n_lccs
    sizehint!(rows, n_hint)
    sizehint!(cols, n_hint)
    sizehint!(vals, n_hint)
    function push_block!(r_off::Int, c_off::Int)
        for r in 0:1, c in 0:1
            push!(rows, J_INDEX_TYPE(r_off + r))
            push!(cols, J_INDEX_TYPE(c_off + c))
            push!(vals, 0.0)
        end
        return
    end
    Yrows = SparseArrays.rowvals(Y_bus_eff)
    @inbounds for col in 1:n_buses
        col_off = Int(bus_state_offset[col])
        # Unconditional: a bus with no stored Y-bus diagonal (an AC-isolated swing, e.g. a
        # DC-tie voltage holder) still needs its own block.
        push_block!(col_off, col_off)
        for j in SparseArrays.nzrange(Y_bus_eff, col)
            row = Yrows[j]
            row == col && continue
            push_block!(Int(bus_state_offset[row]), col_off)
        end
    end
    # Distributed-slack cross-terms ∂F_k/∂x[ref_off]; a multi-swing island has none.
    spf = residual.bus_slack_participation_factors
    for (ref_bus, subnetwork_buses) in residual.subnetworks
        ref_bus in residual.independent_ref && continue
        ref_off = Int(bus_state_offset[ref_bus])
        for bus_k in subnetwork_buses
            (iszero(spf[bus_k]) || bus_k == ref_bus) && continue
            k_off = Int(bus_state_offset[bus_k])
            push!(rows, J_INDEX_TYPE(k_off), J_INDEX_TYPE(k_off + 1))
            push!(cols, J_INDEX_TYPE(ref_off), J_INDEX_TYPE(ref_off))
            push!(vals, 0.0, 0.0)
        end
    end
    if n_lccs > 0
        _create_rect_ci_lcc_structure!(
            rows, cols, vals, data, bus_state_offset, total_bus_state,
        )
    end
    if has_dc_network(dcn)
        _create_rect_ci_vsc_structure!(
            rows, cols, vals, dcn, bus_state_offset, total_bus_state, n_lccs,
        )
    end
    return SparseArrays.sparse(rows, cols, vals, total_state, total_state)
end

"""
    _build_rect_yb_nz_cache(Jv, Y_bus_eff, bus_state_offset) -> yb_nz

`nonzeros(Jv)` indices of the 2×2 block of every Y-bus nonzero, by the nonzero's index in
`Y_bus_eff`, in the order `(P,e) (P,f) (Q,e) (Q,f)`.
"""
function _build_rect_yb_nz_cache(
    Jv::SparseMatrixCSC{Float64, J_INDEX_TYPE},
    Y_bus_eff::SparseMatrixCSC{ComplexF64, Int},
    bus_state_offset::Vector{REC_INDEX_TYPE},
)
    yb_nz = Matrix{Int}(undef, 4, SparseArrays.nnz(Y_bus_eff))
    Yrows = SparseArrays.rowvals(Y_bus_eff)
    for col in 1:size(Y_bus_eff, 2)
        c = Int(bus_state_offset[col])
        for k in SparseArrays.nzrange(Y_bus_eff, col)
            r = Int(bus_state_offset[Yrows[k]])
            yb_nz[1, k] = _jv_nz_index(Jv, r, c)
            yb_nz[2, k] = _jv_nz_index(Jv, r, c + 1)
            yb_nz[3, k] = _jv_nz_index(Jv, r + 1, c)
            yb_nz[4, k] = _jv_nz_index(Jv, r + 1, c + 1)
        end
    end
    return yb_nz
end

"""Update every Jacobian entry from the residual's state caches (`e_state`, `f_state`, `Ir_acc`,
`Ii_acc`), which must be current: call the residual on `x` first.

With `I = Y·V`, `P_i = e_i·Ir_i + f_i·Ii_i` and `Q_i = f_i·Ir_i − e_i·Ii_i`, a Y-bus nonzero
`y = g + jb` at `(i, j ≠ i)` gives `∂P_i/∂(e_j, f_j) = (a, c)` and `∂Q_i/∂(e_j, f_j) = (c, −a)`
with `a = e_i·g + f_i·b`, `c = f_i·g − e_i·b`. The bus's own block adds the `I_i` terms and the
constant-current load. A PV bus's second row is `|V|²`; a REF bus's columns are its `(P, Q)`."""
function _update_rect_ci_jacobian_values!(
    J::ACRectangularCIJacobian,
    data::ACPowerFlowData,
    time_step::Int64,
)
    Jvnz = SparseArrays.nonzeros(J.Jv)
    bus_types = view(data.bus_type, :, time_step)
    e_state = J.e_state
    f_state = J.f_state
    Ir = J.Ir_acc
    Ii = J.Ii_acc
    yb_nz = J.yb_nz
    diag_nz = J.diag_nz
    Y = J.Y_bus_eff
    Yvals = SparseArrays.nonzeros(Y)
    Yrows = SparseArrays.rowvals(Y)
    n_buses = size(Y, 2)
    @inbounds for col in 1:n_buses
        col_is_ref = bus_types[col] == PSY.ACBusTypes.REF
        for k in SparseArrays.nzrange(Y, col)
            row = Yrows[k]
            row == col && continue
            if col_is_ref
                Jvnz[yb_nz[1, k]] = 0.0
                Jvnz[yb_nz[2, k]] = 0.0
                Jvnz[yb_nz[3, k]] = 0.0
                Jvnz[yb_nz[4, k]] = 0.0
                continue
            end
            g = real(Yvals[k])
            b = imag(Yvals[k])
            e_i = e_state[row]
            f_i = f_state[row]
            a = e_i * g + f_i * b
            c = f_i * g - e_i * b
            Jvnz[yb_nz[1, k]] = a
            Jvnz[yb_nz[2, k]] = c
            if bus_types[row] == PSY.ACBusTypes.PV
                Jvnz[yb_nz[3, k]] = 0.0
                Jvnz[yb_nz[4, k]] = 0.0
            else
                Jvnz[yb_nz[3, k]] = c
                Jvnz[yb_nz[4, k]] = -a
            end
        end
    end
    @inbounds for i in 1:n_buses
        bt = bus_types[i]
        if bt == PSY.ACBusTypes.REF
            # F_P = P_calc − P_net_set − c_ref·(x[off] − P_net_set); a multi-swing REF
            # self-balances its own P-slot.
            c_ref = J.bus_slack_participation_factors[i]
            if i in J.independent_ref
                c_ref = 1.0
            end
            Jvnz[diag_nz[1, i]] = -c_ref
            Jvnz[diag_nz[2, i]] = 0.0
            Jvnz[diag_nz[3, i]] = 0.0
            Jvnz[diag_nz[4, i]] = -1.0
            continue
        end
        e = e_state[i]
        f = f_state[i]
        g = real(J.Y_diag[i])
        b = imag(J.Y_diag[i])
        # V_FLOOR2: see the residual; the constant-current load is −const_I·|V| in S_spec.
        inv_Vm = 1.0 / sqrt(max(e^2 + f^2, V_FLOOR2))
        Jvnz[diag_nz[1, i]] = Ir[i] + e * g + f * b + J.const_I_P[i] * e * inv_Vm
        Jvnz[diag_nz[2, i]] = Ii[i] + f * g - e * b + J.const_I_P[i] * f * inv_Vm
        if bt == PSY.ACBusTypes.PV
            Jvnz[diag_nz[3, i]] = 2 * e
            Jvnz[diag_nz[4, i]] = 2 * f
        else
            Jvnz[diag_nz[3, i]] = -Ii[i] + f * g - e * b + J.const_I_Q[i] * e * inv_Vm
            Jvnz[diag_nz[4, i]] = Ir[i] - f * b - e * g + J.const_I_Q[i] * f * inv_Vm
        end
    end
    # Distributed-slack cross-terms: the P_spec of bus k carries c_k·(x[ref_off] − P_net_set).
    @inbounds for k in eachindex(J.slack_nz_idx_e)
        Jvnz[J.slack_nz_idx_e[k]] = -J.slack_c_k[k]
        Jvnz[J.slack_nz_idx_f[k]] = 0.0
    end
    if size(data.lcc.p_set, 1) > 0
        _set_entries_for_lcc_rect!(
            data,
            Jvnz,
            diag_nz,
            J.lcc_nz,
            e_state,
            f_state,
            time_step,
        )
    end
    dcn = get_dc_network(data)
    if has_dc_network(dcn)
        _set_entries_for_vsc_rect!(
            Jvnz,
            J.vsc_nz,
            dcn,
            e_state,
            f_state,
            bus_types,
            time_step,
        )
    end
    return
end

# Each converter injects (P_c, Q_c) into its bus's power rows, Q_c only where the second row is
# reactive balance (not a PV bus's |V|² pin).
function _set_entries_for_vsc_rect!(
    Jvnz::Vector{Float64},
    vsc_nz::VSCJacobianNZCache,
    dcn::DCNetwork,
    e_state::Vector{Float64},
    f_state::Vector{Float64},
    bus_types::AbstractVector{PSY.ACBusTypes.Value},
    time_step::Int,
)
    conv = vsc_nz.conv
    @inbounds for c in 1:n_vsc_converters(dcn)
        Jvnz[conv[1, c]] = -1.0
        Jvnz[conv[2, c]] = 0.0
        Jvnz[conv[3, c]] = 0.0
        if bus_types[dcn.converter_ac_bus_ix[c]] == PSY.ACBusTypes.PV
            Jvnz[conv[4, c]] = 0.0
        else
            Jvnz[conv[4, c]] = -1.0
        end
    end
    _set_vsc_tail_entries_rect!(Jvnz, vsc_nz, dcn, e_state, f_state, bus_types, time_step)
    return
end

"""
Write the LCC Jacobian entries. The converter's terminal power `P = |V|²·g`, `Q = −|V|²·b`
(admittance from [`_update_ybus_lcc!`](@ref)) depends on the terminal only through `|V|`, so
the bus rows take polar's `∂/∂|V|` scalars chain-ruled by `∂|V|/∂(e, f) = (e, f)/|V|`, and
polar's tap/α columns unchanged. The tail rows (P-setpoint, DC-line balance, α limits) are
chain-ruled the same way.
"""
function _set_entries_for_lcc_rect!(
    data::ACPowerFlowData,
    Jvnz::Vector{Float64},
    diag_nz::Matrix{Int},
    lcc_nz::Matrix{Int},
    e_state::Vector{Float64},
    f_state::Vector{Float64},
    time_step::Int,
)
    @inbounds for (i, (fb, tb)) in enumerate(data.lcc.bus_indices)
        bus_type_fb = data.bus_type[fb, time_step]
        bus_type_tb = data.bus_type[tb, time_step]

        if iszero(data.lcc.i_dc[i, time_step])
            # 0-current converter: P_lcc ≡ 0, so it contributes nothing to the bus rows and
            # its P-setpoint / DC-line-balance rows are vacuous. Every LCC entry is ∝ i_dc
            # except the two tap diagonals, so zero the block and pin F_t_fb → tap_r (row
            # 15), F_t_tb → tap_i (row 18), matching _write_lcc_tail!. The α-limit identity
            # diagonals are not in lcc_nz (set at pattern build), so they survive.
            Jvnz[lcc_nz[1:24, i]] .= 0.0
            Jvnz[lcc_nz[15, i]] = 1.0
            Jvnz[lcc_nz[18, i]] = 1.0
            continue
        end

        e_fb = e_state[fb]
        f_fb = f_state[fb]
        Vm_fb = sqrt(e_fb^2 + f_fb^2)
        e_tb = e_state[tb]
        f_tb = f_state[tb]
        Vm_tb = sqrt(e_tb^2 + f_tb^2)
        s = _lcc_jacobian_scalars(data, i, time_step, Vm_fb, Vm_tb)
        phi_r = data.lcc.rectifier.phi[i, time_step]
        phi_i = data.lcc.inverter.phi[i, time_step]
        xtr_r = data.lcc.rectifier.transformer_reactance[i]
        xtr_i = data.lcc.inverter.transformer_reactance[i]
        alpha_r = data.lcc.rectifier.thyristor_angle[i, time_step]
        alpha_i = data.lcc.inverter.thyristor_angle[i, time_step]
        y_fb, y_tb = data.lcc.branch_admittances[i]
        # Inverter: −xtr_i flips the commutation-chain terms (see _lcc_jacobian_scalars), and
        # its ϕ_i convention flips ∂Q/∂α_i, exactly as in the polar `_set_entries_for_lcc`.
        _lcc_terminal_entries_rect!(
            Jvnz, diag_nz, lcc_nz, i, 0, fb, bus_type_fb, e_fb, f_fb, Vm_fb, y_fb,
            s.dP_dV_fb, _calculate_dQ_dV_lcc(s.tap_r, s.i_dc, xtr_r, Vm_fb, phi_r),
            s.dP_dt_fb, s.dP_dα_fb,
            _calculate_dQ_dt_lcc(s.tap_r, s.i_dc, xtr_r, Vm_fb, phi_r),
            _calculate_dQ_dα_lcc(s.tap_r, s.i_dc, xtr_r, Vm_fb, phi_r, alpha_r),
        )
        _lcc_terminal_entries_rect!(
            Jvnz, diag_nz, lcc_nz, i, 4, tb, bus_type_tb, e_tb, f_tb, Vm_tb, y_tb,
            s.dP_dV_tb, _calculate_dQ_dV_lcc(s.tap_i, s.i_dc, -xtr_i, Vm_tb, phi_i),
            s.dP_dt_tb, s.dP_dα_tb,
            _calculate_dQ_dt_lcc(s.tap_i, s.i_dc, -xtr_i, Vm_tb, phi_i),
            -_calculate_dQ_dα_lcc(s.tap_i, s.i_dc, xtr_i, Vm_tb, phi_i, alpha_i),
        )

        # Tail rows × bus (e, f), chain-ruled from ∂/∂|V|; zero at a REF terminal, whose
        # columns hold (P, Q). Rows 9,10: ∂F_t_fb/∂(e_fb, f_fb), nonzero only with a
        # rectifier-side set point; 11,12: ∂F_t_tb/∂(e_fb, f_fb).
        if bus_type_fb == PSY.ACBusTypes.REF
            Jvnz[lcc_nz[9, i]] = 0.0
            Jvnz[lcc_nz[10, i]] = 0.0
            Jvnz[lcc_nz[11, i]] = 0.0
            Jvnz[lcc_nz[12, i]] = 0.0
        else
            de_dV_fb = e_fb / Vm_fb
            df_dV_fb = f_fb / Vm_fb
            Jvnz[lcc_nz[9, i]] = s.d_Ft_fb_d_V_fb * de_dV_fb
            Jvnz[lcc_nz[10, i]] = s.d_Ft_fb_d_V_fb * df_dV_fb
            Jvnz[lcc_nz[11, i]] = s.dP_dV_fb * de_dV_fb
            Jvnz[lcc_nz[12, i]] = s.dP_dV_fb * df_dV_fb
        end
        # Rows 13,14: ∂F_t_tb/∂(e_tb, f_tb); 21,22: ∂F_t_fb/∂(e_tb, f_tb), nonzero only with
        # an inverter-side set point.
        if bus_type_tb == PSY.ACBusTypes.REF
            Jvnz[lcc_nz[13, i]] = 0.0
            Jvnz[lcc_nz[14, i]] = 0.0
            Jvnz[lcc_nz[21, i]] = 0.0
            Jvnz[lcc_nz[22, i]] = 0.0
        else
            de_dV_tb = e_tb / Vm_tb
            df_dV_tb = f_tb / Vm_tb
            Jvnz[lcc_nz[13, i]] = s.dP_dV_tb * de_dV_tb
            Jvnz[lcc_nz[14, i]] = s.dP_dV_tb * df_dV_tb
            Jvnz[lcc_nz[21, i]] = s.d_Ft_fb_d_V_tb * de_dV_tb
            Jvnz[lcc_nz[22, i]] = s.d_Ft_fb_d_V_tb * df_dV_tb
        end
        # Tail × tail; the scalars helper zeroes the side the set point is not on.
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

# One LCC terminal's bus-row entries: rows `lcc_nz[r0 + 1 : r0 + 4]` are (P, tap), (P, α),
# (Q, tap), (Q, α). The bus's own block already holds the LCC current through `Ir_acc`/`Ii_acc`
# (the `I_i` terms of ∂S_i/∂V_i); this swaps that part for the exact `|V|`-derivative of
# `P = |V|²·g`, `Q = −|V|²·b`. A PV bus's second row is the |V|² pin and gets nothing.
@inline function _lcc_terminal_entries_rect!(
    Jvnz::Vector{Float64},
    diag_nz::Matrix{Int},
    lcc_nz::Matrix{Int},
    i::Int,
    r0::Int,
    bus::Int,
    bt::PSY.ACBusTypes.Value,
    e::Float64,
    f::Float64,
    Vm::Float64,
    y::ComplexF64,
    dP_dV::Float64,
    dQ_dV::Float64,
    dP_dt::Float64,
    dP_dα::Float64,
    dQ_dt::Float64,
    dQ_dα::Float64,
)
    if bt != PSY.ACBusTypes.REF
        Ir_lcc = real(y) * e - imag(y) * f
        Ii_lcc = real(y) * f + imag(y) * e
        @inbounds Jvnz[diag_nz[1, bus]] += dP_dV * e / Vm - Ir_lcc
        @inbounds Jvnz[diag_nz[2, bus]] += dP_dV * f / Vm - Ii_lcc
        if bt == PSY.ACBusTypes.PQ
            @inbounds Jvnz[diag_nz[3, bus]] += dQ_dV * e / Vm + Ii_lcc
            @inbounds Jvnz[diag_nz[4, bus]] += dQ_dV * f / Vm - Ir_lcc
        end
    end
    @inbounds Jvnz[lcc_nz[r0 + 1, i]] = dP_dt
    @inbounds Jvnz[lcc_nz[r0 + 2, i]] = dP_dα
    if bt == PSY.ACBusTypes.PV
        @inbounds Jvnz[lcc_nz[r0 + 3, i]] = 0.0
        @inbounds Jvnz[lcc_nz[r0 + 4, i]] = 0.0
    else
        @inbounds Jvnz[lcc_nz[r0 + 3, i]] = dQ_dt
        @inbounds Jvnz[lcc_nz[r0 + 4, i]] = dQ_dα
    end
    return
end

# Structural slots for the VSC tail (rectangular / MCPB). Bus×converter injection entries, the two
# control rows per converter (with e,f columns for AC-voltage control), and the DC-KCL rows (G_dc
# pattern + converter coupling, with e,f columns for converter losses). The bus diagonal (e,f)
# block already exists from the Y_bus structure.
function _create_rect_ci_vsc_structure!(
    rows::Vector{J_INDEX_TYPE},
    cols::Vector{J_INDEX_TYPE},
    vals::Vector{Float64},
    dcn::DCNetwork,
    bus_state_offset::AbstractVector,
    total_bus_state::Int,
    n_lccs::Int,
)
    nconv = n_vsc_converters(dcn)
    nnode = n_dc_nodes(dcn)
    vsc_off = total_bus_state + 4 * n_lccs
    base = vsc_off + 2 * nconv
    function push3(r, c)
        push!(rows, J_INDEX_TYPE(r))
        push!(cols, J_INDEX_TYPE(c))
        push!(vals, 0.0)
        return
    end
    for c in 1:nconv
        off = Int(bus_state_offset[dcn.converter_ac_bus_ix[c]])
        k = dcn.converter_dc_node_ix[c]
        pc = vsc_off + 2 * c - 1
        qc = vsc_off + 2 * c
        vk = base + k
        push3(off, pc)       # ∂Ir/∂P_c
        push3(off, qc)       # ∂Ir/∂Q_c
        push3(off + 1, pc)   # ∂Ii/∂P_c
        push3(off + 1, qc)   # ∂Ii/∂Q_c
        push3(pc, pc)        # ∂r1/∂P_c
        push3(pc, vk)        # ∂r1/∂V_dc
        push3(qc, qc)        # ∂r2/∂Q_c
        push3(qc, off)       # ∂r2/∂e (Vac)
        push3(qc, off + 1)   # ∂r2/∂f (Vac)
        push3(vk, pc)        # ∂KCL/∂P_c
        push3(vk, qc)        # ∂KCL/∂Q_c (loss)
        push3(vk, off)       # ∂KCL/∂e (loss)
        push3(vk, off + 1)   # ∂KCL/∂f (loss)
    end
    for k in 1:nnode
        push3(base + k, base + k)
    end
    for b in 1:n_dc_branches(dcn)
        f = dcn.branch_from[b]
        t = dcn.branch_to[b]
        push3(base + f, base + t)
        push3(base + t, base + f)
    end
    return
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
            # Inverter-side slots for the P-setpoint row F_t_fb (idx_tap_r),
            # used when the set point is at the inverter (F_t_fb = −P_lcc_to).
            (idx_tap_r, col_e_tb, 0.0),
            (idx_tap_r, col_f_tb, 0.0),
            (idx_tap_r, idx_tap_i, 0.0),
            (idx_tap_r, idx_alpha_i, 0.0),
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
    _build_slack_nz_cache(Jv, bus_state_offset, subnetworks, bus_slack_participation_factors, independent_ref)

Return `(slack_nz_idx_e, slack_nz_idx_f, slack_bus_k, slack_c_k)`. Each entry
corresponds to one (bus_k != ref_bus, c_k != 0) slack cross-term. The nzval
indices point at `Jv[k_off, ref_off]` and `Jv[k_off+1, ref_off]`. Islands keyed
by a REF bus in `independent_ref` (multi-swing) are skipped entirely — those
islands have no distributed-slack cross-terms in the structural pattern.
"""
function _build_slack_nz_cache(
    Jv::SparseMatrixCSC{Float64, J_INDEX_TYPE},
    bus_state_offset::Vector{REC_INDEX_TYPE},
    subnetworks::Dict{Int64, Vector{Int64}},
    bus_slack_participation_factors::SparseVector{Float64, Int},
    independent_ref::Set{Int},
)
    slack_nz_idx_e = Int[]
    slack_nz_idx_f = Int[]
    slack_bus_k = Int[]
    slack_c_k = Float64[]
    for (ref_bus, subnetwork_buses) in subnetworks
        ref_bus in independent_ref && continue
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

Return a `24 × n_lccs` matrix of nzval indices for the per-LCC tail entries
that get updated each iteration. The two identity diagonals
(`Jv[idx_alpha_r, idx_alpha_r]` and `Jv[idx_alpha_i, idx_alpha_i]`) are not
included — they are set to 1.0 at structure-build time and never updated.
The 8 FB/TB-side diagonal-block overlay entries are NOT included either —
they share nzval slots with `diag_base_nz` for buses `fb` and `tb` and are
addressed through that cache.

Rows 9, 10, 15, 16, 21–24 belong to the P-setpoint row `F_t_fb`
(`idx_tap_r`): rows 9, 10, 15, 16 hold its rectifier-side dependence and
rows 21–24 its inverter-side dependence; `_lcc_jacobian_scalars` zeroes
whichever side the set point is not on.

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
 21: Jv[idx_tap_r, col_e_tb],  22: Jv[idx_tap_r, col_f_tb],
 23: Jv[idx_tap_r, idx_tap_i], 24: Jv[idx_tap_r, idx_alpha_i],
"""
function _build_lcc_nz_cache(
    Jv::SparseMatrixCSC{Float64, J_INDEX_TYPE},
    data::ACPowerFlowData,
    bus_state_offset::Vector{REC_INDEX_TYPE},
    total_bus_state::Int,
    n_lccs::Int,
)
    lcc_nz = Matrix{Int}(undef, 24, n_lccs)
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
        lcc_nz[21, i] = _jv_nz_index(Jv, idx_tap_r, col_e_tb)
        lcc_nz[22, i] = _jv_nz_index(Jv, idx_tap_r, col_f_tb)
        lcc_nz[23, i] = _jv_nz_index(Jv, idx_tap_r, idx_tap_i)
        lcc_nz[24, i] = _jv_nz_index(Jv, idx_tap_r, idx_alpha_i)
    end
    return lcc_nz
end

"""
Pre-compute the `nonzeros(Jv)` indices for the VSC tail (layout-generic over rectangular and
MCPB — both share `_create_rect_ci_vsc_structure!` and `_set_vsc_tail_entries_rect!`). The `conv`
row order matches the slot push order in `_create_rect_ci_vsc_structure!`:

    1-4   bus injection coupling:         (off,pc) (off,qc) (off+1,pc) (off+1,qc)
    5-6   control row r1:                 (pc,pc) (pc,vk)
    7-9   control row r2:                 (qc,qc) (qc,off) (qc,off+1)
    10-13 DC-KCL converter coupling:      (vk,pc) (vk,qc) (vk,off) (vk,off+1)

The (vk,vk) node diagonal is shared by every converter on a node, so it lives in `node` (set to
`G_dc[k,k]` then accumulated) rather than per-converter.
"""
function _build_vsc_nz_cache(
    Jv::SparseMatrixCSC{Float64, J_INDEX_TYPE},
    dcn::DCNetwork,
    bus_state_offset::AbstractVector,
    total_bus_state::Int,
    n_lccs::Int,
)
    nconv = n_vsc_converters(dcn)
    nnode = n_dc_nodes(dcn)
    nbranch = n_dc_branches(dcn)
    conv = Matrix{Int}(undef, 13, nconv)
    node = Vector{Int}(undef, nnode)
    branch = Vector{Int}(undef, 2 * nbranch)
    vsc_off = total_bus_state + 4 * n_lccs
    base = vsc_off + 2 * nconv
    for c in 1:nconv
        off = Int(bus_state_offset[dcn.converter_ac_bus_ix[c]])
        k = dcn.converter_dc_node_ix[c]
        pc = vsc_off + 2 * c - 1
        qc = vsc_off + 2 * c
        vk = base + k
        conv[1, c] = _jv_nz_index(Jv, off, pc)
        conv[2, c] = _jv_nz_index(Jv, off, qc)
        conv[3, c] = _jv_nz_index(Jv, off + 1, pc)
        conv[4, c] = _jv_nz_index(Jv, off + 1, qc)
        conv[5, c] = _jv_nz_index(Jv, pc, pc)
        conv[6, c] = _jv_nz_index(Jv, pc, vk)
        conv[7, c] = _jv_nz_index(Jv, qc, qc)
        conv[8, c] = _jv_nz_index(Jv, qc, off)
        conv[9, c] = _jv_nz_index(Jv, qc, off + 1)
        conv[10, c] = _jv_nz_index(Jv, vk, pc)
        conv[11, c] = _jv_nz_index(Jv, vk, qc)
        conv[12, c] = _jv_nz_index(Jv, vk, off)
        conv[13, c] = _jv_nz_index(Jv, vk, off + 1)
    end
    for k in 1:nnode
        node[k] = _jv_nz_index(Jv, base + k, base + k)
    end
    for b in 1:nbranch
        f = dcn.branch_from[b]
        t = dcn.branch_to[b]
        branch[2 * b - 1] = _jv_nz_index(Jv, base + f, base + t)
        branch[2 * b] = _jv_nz_index(Jv, base + t, base + f)
    end
    return VSCJacobianNZCache(conv, node, branch)
end

# MCPB REF block: current-balance rows with `(P_net, Q_net)` columns.
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
    # V_FLOOR2 floor; REF |V| is fixed near V_set so this never triggers in practice.
    V_sq = max(e_r^2 + f_r^2, V_FLOOR2)
    inv_V_sq = 1.0 / V_sq
    @inbounds Jvnz[diag_base_nz[1, i]] = c_ref * e_r * inv_V_sq
    @inbounds Jvnz[diag_base_nz[2, i]] = f_r * inv_V_sq
    @inbounds Jvnz[diag_base_nz[3, i]] = c_ref * f_r * inv_V_sq
    @inbounds Jvnz[diag_base_nz[4, i]] = -e_r * inv_V_sq
    return
end
