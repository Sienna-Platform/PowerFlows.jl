# Bordered Jacobian tail for embedded PSS/E-style area net-interchange control.
# Column ΔP_a couples into the controlled area's slack-bus P-mismatch row (constant
# −1.0, same seam `_update_residual_values!` reads it at). Row r_a (the area's NI residual)
# couples into the Vm/θ state columns of every tie endpoint bus incident to area a. Follows the
# LCC/VSC structure+fill registration pattern (`_create_jacobian_matrix_structure_lcc/vsc`,
# `_set_entries_for_lcc/vsc` in `ac_power_flow_jacobian.jl`).

"""
    _tie_metered_active_power_partials(tie, Vm_f, θ_f, Vm_t, θ_t, ybus_nzval)
        -> (dPm_dVf, dPm_dθf, dPm_dVt, dPm_dθt)

`∂P_m/∂(Vf,θf,Vt,θt)` for `tie`'s metered-end active power, differentiated term-by-term from
`_tie_metered_active_power`'s exact expression so both always agree — see `_tie_admittances`
(`area_residual.jl`) for the shared, diagonal-pollution-corrected admittance read.

Sign convention: `Δθ = θf − θt` throughout (a single shared `sincos`), including the
metered-at-`to` case — its natural angle argument `θt − θf = −Δθ` is folded into the
`g21`/`b21` terms below instead of needing a second `sincos` call.
"""
function _tie_metered_active_power_partials(
    tie::AreaTie,
    Vm_f::Float64,
    θ_f::Float64,
    Vm_t::Float64,
    θ_t::Float64,
    ybus_nzval::Vector{YBUS_ELTYPE},
)
    (g11, b11, g12, b12, g21, b21, g22, b22) = _tie_admittances(tie, ybus_nzval)
    s, c = sincos(θ_f - θ_t)
    if tie.metered_from
        dPm_dθf = Vm_f * Vm_t * (-g12 * s + b12 * c)
        dPm_dVf = 2 * Vm_f * g11 + Vm_t * (g12 * c + b12 * s)
        dPm_dVt = Vm_f * (g12 * c + b12 * s)
        return (dPm_dVf, dPm_dθf, dPm_dVt, -dPm_dθf)
    end
    dPm_dθt = Vm_t * Vm_f * (g21 * s + b21 * c)
    dPm_dVt = 2 * Vm_t * g22 + Vm_f * (g21 * c - b21 * s)
    dPm_dVf = Vm_t * (g21 * c - b21 * s)
    return (dPm_dVf, -dPm_dθt, dPm_dVt, dPm_dθt)
end

# Stamp rule: a structural slot for every tie endpoint, for ALL bus types. The structure never
# reads `bus_type`, so it holds for every time step. The numeric fill writes zero where the slot
# is not a state (a REF bus's `(2b-1, 2b)` hold (P_gen, Q_gen), a PV bus's Vm is fixed).
function _push_area_row_bus_cols!(push3, row::Int, bus_ix::Int)
    push3(row, 2 * bus_ix - 1)
    push3(row, 2 * bus_ix)
    return
end

function _push_area_row_endpoint_cols!(push3, row::Int, tie::AreaTie)
    _push_area_row_bus_cols!(push3, row, tie.from_bus_ix)
    _push_area_row_bus_cols!(push3, row, tie.to_bus_ix)
    return
end

# `∂/∂|V|` of `_off_area_load_active_power`: the load term depends on its bus |V| only.
function _off_area_load_dV(load::OffAreaLoad, Vm::AbstractVector{Float64}, time_step::Int)
    return load.sign * (
        load.p_constant_current[time_step] +
        2 * load.p_constant_impedance[time_step] * Vm[load.bus_ix]
    )
end

_off_area_load_vm_col(load::OffAreaLoad) = 2 * load.bus_ix - 1

"""
Create the Jacobian matrix structure for the area-interchange tail (polar) — the bordered
block. Column ΔP_a (`area_off + area.tail_ix`): one structural entry at the
area's slack-bus P-mismatch row (`2*slack_bus_ix - 1`). Row r_a
(`area_off + area.tail_ix`): both state columns at every tie endpoint bus of every tie incident to area a, via `_push_area_row_endpoint_cols!`, and the Vm column of every
off-area load bus of area a (same stamp rule: every bus type; the fill writes zero unless
PQ). A tie with
BOTH endpoints controlled (different areas) contributes to TWO area rows. `∂r_a/∂ΔP_a` is
structurally ABSENT (zero diagonal border — KLU's full pivoting handles it; do not stamp).
"""
function _create_jacobian_matrix_structure_area(
    data::ACPowerFlowData,
    rows::Vector{J_INDEX_TYPE},
    columns::Vector{J_INDEX_TYPE},
    values::Vector{Float64},
)
    iszero(n_controlled_areas(data)) && return
    dcn = get_dc_network(data)
    area_off = area_tail_offset(data, dcn)
    function push3(r, c)
        push!(rows, J_INDEX_TYPE(r))
        push!(columns, J_INDEX_TYPE(c))
        push!(values, 0.0)
        return
    end
    for area in data.area_interchange.areas
        push3(2 * area.slack_bus_ix - 1, area_off + area.tail_ix)
    end
    for tie in data.area_interchange.ties
        iszero(tie.from_area_tail) ||
            _push_area_row_endpoint_cols!(
                push3, area_off + tie.from_area_tail, tie)
        iszero(tie.to_area_tail) ||
            _push_area_row_endpoint_cols!(push3, area_off + tie.to_area_tail, tie)
    end
    num_buses = first(size(data.bus_type))
    num_lcc = size(data.lcc.p_set, 1)
    for tie in data.area_interchange.dc_ties
        iszero(tie.from_area_tail) ||
            _push_area_row_dc_cols!(
                push3, area_off + tie.from_area_tail, tie, num_buses, num_lcc)
        iszero(tie.to_area_tail) ||
            _push_area_row_dc_cols!(
                push3, area_off + tie.to_area_tail, tie, num_buses, num_lcc)
    end
    for load in data.area_interchange.off_area_loads
        push3(area_off + load.area_tail, _off_area_load_vm_col(load))
    end
    return
end

# Fill one tie endpoint's contribution into area row `row`: PQ writes both columns; PV writes
# only the θ column (its Vm column was pre-zeroed by `_zero_area_row_bus_cols!` and stays at
# that structural zero — a PV bus's own |V| is fixed, so `P_m` has no dependence on it through
# the STATE vector even though it varies physically); REF contributes nothing (its slots stay
# at the zero the pre-pass wrote). `+=` (not `=`) because a bus can be the shared endpoint of
# multiple ties feeding the same area row (e.g. a degree>1 boundary bus).
function _accumulate_area_endpoint!(
    Jv::SparseArrays.SparseMatrixCSC{Float64, J_INDEX_TYPE},
    row::Int,
    bus_ix::Int,
    bus_type,
    dPm_dVm::Float64,
    dPm_dθ::Float64,
)
    bt = bus_type[bus_ix]
    bt == PSY.ACBusTypes.REF && return
    Jv[row, 2 * bus_ix] += dPm_dθ
    bt == PSY.ACBusTypes.PV && return
    Jv[row, 2 * bus_ix - 1] += dPm_dVm
    return
end

# Zero the structural slots one tie touches at area row `row` before the accumulation pass —
# both columns of both endpoints, unconditionally (mirrors the structure stamp). Without this reset, `_accumulate_area_endpoint!`'s
# `+=` would pile onto a stale value from a previous Jacobian evaluation instead of the fresh one.
function _zero_area_row_bus_cols!(
    Jv::SparseArrays.SparseMatrixCSC{Float64, J_INDEX_TYPE},
    row::Int,
    bus_ix::Int,
)
    Jv[row, 2 * bus_ix - 1] = 0.0
    Jv[row, 2 * bus_ix] = 0.0
    return
end

function _zero_area_row_endpoint_cols!(
    Jv::SparseArrays.SparseMatrixCSC{Float64, J_INDEX_TYPE},
    tie::AreaTie,
    area_off::Int,
)
    iszero(tie.from_area_tail) ||
        _zero_area_row_bus_cols!(Jv, area_off + tie.from_area_tail, tie.from_bus_ix)
    iszero(tie.from_area_tail) ||
        _zero_area_row_bus_cols!(Jv, area_off + tie.from_area_tail, tie.to_bus_ix)
    iszero(tie.to_area_tail) ||
        _zero_area_row_bus_cols!(Jv, area_off + tie.to_area_tail, tie.from_bus_ix)
    iszero(tie.to_area_tail) ||
        _zero_area_row_bus_cols!(Jv, area_off + tie.to_area_tail, tie.to_bus_ix)
    return
end

# Accumulate tie's ∂P_m/∂(Vf,θf,Vt,θt) into area row `row`, scaled by `σ` — the same ± sign the
# residual accumulator (`_set_area_tail_residuals!`) applies to that side's `ni[tail]` (+1 for
# the metered side, −1 for the other side: both endpoints' state columns still enter through
# `P_m`, only the accumulation sign into that particular area's row flips).
function _accumulate_area_row!(
    Jv::SparseArrays.SparseMatrixCSC{Float64, J_INDEX_TYPE},
    row::Int,
    tie::AreaTie,
    bus_type,
    dPm_dVf::Float64,
    dPm_dθf::Float64,
    dPm_dVt::Float64,
    dPm_dθt::Float64,
    σ::Float64,
)
    _accumulate_area_endpoint!(
        Jv, row, tie.from_bus_ix, bus_type, σ * dPm_dVf, σ * dPm_dθf)
    _accumulate_area_endpoint!(Jv, row, tie.to_bus_ix, bus_type, σ * dPm_dVt, σ * dPm_dθt)
    return
end

# DC-tie (LCC/VSC) cross-derivatives into the area-interchange rows, same ± tail routing as
# an AC tie. LCC: `∂P/∂(Vm, tap, α)` at the metered terminal; VSC: `∂P_conv/∂P_c = −1`.

function _dc_tie_metered_bus_ix(tie::DCTie)
    if tie.metered_from
        return tie.from_bus_ix
    end
    return tie.to_bus_ix
end

# (Vm, tap, α) state columns for an LCC tie's metered terminal. Rectifier (metered_from) uses
# the +1/+3 LCC tail slots; inverter uses +2/+4 (`_create_jacobian_matrix_structure_lcc`).
function _lcc_dc_tie_cols(tie::DCTie, num_buses::Int)
    vm_col = 2 * _dc_tie_metered_bus_ix(tie) - 1
    offset_lcc = 2 * num_buses + (tie.lcc_ix - 1) * 4
    if tie.metered_from
        return (vm_col, offset_lcc + 1, offset_lcc + 3)
    end
    return (vm_col, offset_lcc + 2, offset_lcc + 4)
end

function _vsc_dc_tie_col(tie::DCTie, num_buses::Int, num_lcc::Int)
    if tie.metered_from
        conv = tie.from_conv_ix
    else
        conv = tie.to_conv_ix
    end
    return 2 * num_buses + 4 * num_lcc + 2 * conv - 1
end

# `∂P_conv/∂(Vm, tap, α)` at an LCC tie's metered terminal, from the exact `sin(ϕ)→0`-guarded
# kernels the LCC tail Jacobian uses. Inverter terminal: the commutation-chain terms carry the
# opposite sign (−x_t) and ∂P/∂α negates — the ϕ_i ≈ π − α_i convention documented in
# `_lcc_jacobian_scalars`. Raw `i_dc` (matches the residual's `_lcc_ac_active_powers`).
function _lcc_dc_tie_partials(
    data::ACPowerFlowData,
    tie::DCTie,
    Vm::AbstractVector{Float64},
    time_step::Int,
)
    i = tie.lcc_ix
    i_dc = data.lcc.i_dc[i, time_step]
    if tie.metered_from
        t = data.lcc.rectifier.tap[i, time_step]
        α = data.lcc.rectifier.thyristor_angle[i, time_step]
        ϕ = data.lcc.rectifier.phi[i, time_step]
        x_t = data.lcc.rectifier.transformer_reactance[i]
        Vm_m = Vm[tie.from_bus_ix]
        dP_dVm = _calculate_dP_dV_lcc(t, i_dc, x_t, Vm_m, ϕ)
        dP_dtap = _calculate_dP_dt_lcc(t, i_dc, x_t, Vm_m, ϕ)
        dP_dα = _calculate_dP_dα_lcc(t, i_dc, Vm_m, α, ϕ)
        return (dP_dVm, dP_dtap, dP_dα)
    end
    t = data.lcc.inverter.tap[i, time_step]
    α = data.lcc.inverter.thyristor_angle[i, time_step]
    ϕ = data.lcc.inverter.phi[i, time_step]
    x_t = data.lcc.inverter.transformer_reactance[i]
    Vm_m = Vm[tie.to_bus_ix]
    dP_dVm = _calculate_dP_dV_lcc(t, i_dc, -x_t, Vm_m, ϕ)
    dP_dtap = _calculate_dP_dt_lcc(t, i_dc, -x_t, Vm_m, ϕ)
    dP_dα = -_calculate_dP_dα_lcc(t, i_dc, Vm_m, α, ϕ)
    return (dP_dVm, dP_dtap, dP_dα)
end

# Stamp the structural slots one DC tie touches at area row `row`: LCC → metered {tap, α, Vm}
# cols; VSC → metered P_c col. Same stamp rule as the AC path: the Vm slot is stamped for every
# bus type; the numeric fill writes it only at PQ.
function _push_area_row_dc_cols!(
    push3,
    row::Int,
    tie::DCTie,
    num_buses::Int,
    num_lcc::Int,
)
    if tie.kind == DC_TIE_LCC
        (vm_col, tap_col, alpha_col) = _lcc_dc_tie_cols(tie, num_buses)
        push3(row, tap_col)
        push3(row, alpha_col)
        push3(row, vm_col)
        return
    end
    push3(row, _vsc_dc_tie_col(tie, num_buses, num_lcc))
    return
end

# Zero the structural DC-tie slots one tie touches at area row `row` before the accumulation
# pass (same `+=`-reset reason as `_zero_area_row_bus_cols!`).
function _zero_area_row_dc_cols!(
    Jv::SparseArrays.SparseMatrixCSC{Float64, J_INDEX_TYPE},
    row::Int,
    tie::DCTie,
    num_buses::Int,
    num_lcc::Int,
)
    if tie.kind == DC_TIE_LCC
        (vm_col, tap_col, alpha_col) = _lcc_dc_tie_cols(tie, num_buses)
        Jv[row, tap_col] = 0.0
        Jv[row, alpha_col] = 0.0
        Jv[row, vm_col] = 0.0
        return
    end
    Jv[row, _vsc_dc_tie_col(tie, num_buses, num_lcc)] = 0.0
    return
end

function _zero_area_row_dc_tie!(
    Jv::SparseArrays.SparseMatrixCSC{Float64, J_INDEX_TYPE},
    tie::DCTie,
    area_off::Int,
    num_buses::Int,
    num_lcc::Int,
)
    iszero(tie.from_area_tail) ||
        _zero_area_row_dc_cols!(
            Jv, area_off + tie.from_area_tail, tie, num_buses, num_lcc)
    iszero(tie.to_area_tail) ||
        _zero_area_row_dc_cols!(
            Jv, area_off + tie.to_area_tail, tie, num_buses, num_lcc)
    return
end

# Accumulate a DC tie's `∂P_conv/∂(DC state)` into area row `row`, scaled by `σ` (the same ±
# sign the residual applies to that side's `ni[tail]`). LCC: the Vm entry is written only at a
# PQ metered bus (at PV/REF the terminal |V| is fixed, not a state — mirrors
# `_accumulate_area_endpoint!`); tap/α are always state. VSC: `∂P_conv/∂P_c = −1`.
function _accumulate_area_dc_row!(
    Jv::SparseArrays.SparseMatrixCSC{Float64, J_INDEX_TYPE},
    row::Int,
    tie::DCTie,
    bus_type,
    data::ACPowerFlowData,
    Vm::AbstractVector{Float64},
    num_buses::Int,
    num_lcc::Int,
    time_step::Int,
    σ::Float64,
)
    if tie.kind == DC_TIE_LCC
        (dP_dVm, dP_dtap, dP_dα) = _lcc_dc_tie_partials(data, tie, Vm, time_step)
        (vm_col, tap_col, alpha_col) = _lcc_dc_tie_cols(tie, num_buses)
        Jv[row, tap_col] += σ * dP_dtap
        Jv[row, alpha_col] += σ * dP_dα
        if bus_type[_dc_tie_metered_bus_ix(tie)] == PSY.ACBusTypes.PQ
            Jv[row, vm_col] += σ * dP_dVm
        end
        return
    end
    Jv[row, _vsc_dc_tie_col(tie, num_buses, num_lcc)] += σ * (-1.0)
    return
end

"""
Fill the area-interchange tail Jacobian entries (polar). Column ΔP_a is the constant `-1.0`
at each area's slack-bus P-mismatch row. Row r_a is filled by a zero-then-accumulate two-pass
sweep over every tie, since a boundary bus of degree > 1 feeds the same area row from
multiple ties.
"""
function _set_entries_for_area(
    data::ACPowerFlowData,
    Jv::SparseArrays.SparseMatrixCSC{Float64, J_INDEX_TYPE},
    time_step::Int,
)
    iszero(n_controlled_areas(data)) && return
    aid = data.area_interchange
    dcn = get_dc_network(data)
    area_off = area_tail_offset(data, dcn)
    Vm = view(data.bus_magnitude, :, time_step)
    θ = view(data.bus_angles, :, time_step)
    bus_type = view(data.bus_type, :, time_step)
    ybus_nzval = SparseArrays.nonzeros(data.power_network_matrix.data)
    num_buses = first(size(data.bus_type))
    num_lcc = size(data.lcc.p_set, 1)
    dc_present = !isempty(aid.dc_ties)

    @inbounds for area in aid.areas
        Jv[2 * area.slack_bus_ix - 1, area_off + area.tail_ix] = -1.0
    end

    # Zero every AC- and DC-tie slot before any accumulation: a boundary bus's Vm column can be
    # shared between an AC tie and an LCC metered terminal at the SAME area row, so all zeroing
    # must precede all `+=` accumulation or one side's contribution would be clobbered.
    @inbounds for tie in aid.ties
        _zero_area_row_endpoint_cols!(Jv, tie, area_off)
    end
    if dc_present
        @inbounds for tie in aid.dc_ties
            _zero_area_row_dc_tie!(Jv, tie, area_off, num_buses, num_lcc)
        end
    end
    @inbounds for load in aid.off_area_loads
        Jv[area_off + load.area_tail, _off_area_load_vm_col(load)] = 0.0
    end

    @inbounds for tie in aid.ties
        f = tie.from_bus_ix
        t = tie.to_bus_ix
        (dPm_dVf, dPm_dθf, dPm_dVt, dPm_dθt) =
            _tie_metered_active_power_partials(tie, Vm[f], θ[f], Vm[t], θ[t], ybus_nzval)
        metered_tail = tie.from_area_tail
        other_tail = tie.to_area_tail
        if !tie.metered_from
            metered_tail = tie.to_area_tail
            other_tail = tie.from_area_tail
        end
        iszero(metered_tail) || _accumulate_area_row!(
            Jv, area_off + metered_tail, tie, bus_type,
            dPm_dVf, dPm_dθf, dPm_dVt, dPm_dθt, 1.0,
        )
        iszero(other_tail) || _accumulate_area_row!(
            Jv, area_off + other_tail, tie, bus_type,
            dPm_dVf, dPm_dθf, dPm_dVt, dPm_dθt, -1.0,
        )
    end

    if dc_present
        @inbounds for tie in aid.dc_ties
            metered_tail = tie.from_area_tail
            other_tail = tie.to_area_tail
            if !tie.metered_from
                metered_tail = tie.to_area_tail
                other_tail = tie.from_area_tail
            end
            iszero(metered_tail) || _accumulate_area_dc_row!(
                Jv, area_off + metered_tail, tie, bus_type, data, Vm,
                num_buses, num_lcc, time_step, 1.0,
            )
            iszero(other_tail) || _accumulate_area_dc_row!(
                Jv, area_off + other_tail, tie, bus_type, data, Vm,
                num_buses, num_lcc, time_step, -1.0,
            )
        end
    end

    @inbounds for load in aid.off_area_loads
        bus_type[load.bus_ix] == PSY.ACBusTypes.PQ || continue
        Jv[area_off + load.area_tail, _off_area_load_vm_col(load)] +=
            _off_area_load_dV(load, Vm, time_step)
    end
    return
end
