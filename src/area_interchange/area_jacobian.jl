# Task 7: bordered Jacobian tail for embedded PSS/E-style area net-interchange control (design
# spec §2). Column ΔP_a couples into the controlled area's slack-bus P-mismatch row (constant
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
metered-at-`to` case. Its natural angle argument is `θt − θf = −Δθ`; substituting
`cos(−Δθ) = cos(Δθ)` and `sin(−Δθ) = −sin(Δθ)` into the mirrored kernel expression folds the
sign into the `g21`/`b21` terms below instead of needing a second `sincos` call.

Metered at `f`: `∂P_m/∂θf = Vf·Vt·(−g12·s + b12·c)`, `∂P_m/∂θt = −∂P_m/∂θf`,
`∂P_m/∂Vf = 2·Vf·g11 + Vt·(g12·c + b12·s)`, `∂P_m/∂Vt = Vf·(g12·c + b12·s)`.
Metered at `t`: `∂P_m/∂θt = Vt·Vf·(g21·s + b21·c)`, `∂P_m/∂θf = −∂P_m/∂θt`,
`∂P_m/∂Vt = 2·Vt·g22 + Vf·(g21·c − b21·s)`, `∂P_m/∂Vf = Vt·(g21·c − b21·s)`.
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

# Structural slot for one non-REF tie endpoint bus at area row `row`: BOTH state columns
# (`2b-1`, `2b`), stamped unconditionally so the pattern is Q-limit-flip-invariant (mirrors the
# bus-block convention, `ac_power_flow_jacobian.jl` ~:228-261) — numeric fill only writes the
# column(s) valid for the endpoint's CURRENT bus type. A REF endpoint's `(2b-1, 2b)` slots hold
# (P_gen, Q_gen), not (Vm, θ); `P_m` doesn't depend on those (Vm/θ at REF are fixed parameters,
# not state), so no structural entry is stamped there at all — this is bus-type-invariant since
# a bus enrolled as REF never becomes PV/PQ mid-solve.
function _push_area_row_bus_cols!(push3, row::Int, bus_ix::Int, bus_type)
    bus_type[bus_ix] == PSY.ACBusTypes.REF && return
    push3(row, 2 * bus_ix - 1)
    push3(row, 2 * bus_ix)
    return
end

function _push_area_row_endpoint_cols!(push3, row::Int, tie::AreaTie, bus_type)
    _push_area_row_bus_cols!(push3, row, tie.from_bus_ix, bus_type)
    _push_area_row_bus_cols!(push3, row, tie.to_bus_ix, bus_type)
    return
end

"""
Create the Jacobian matrix structure for the area-interchange tail (polar) — the bordered
block of design spec §2. Column ΔP_a (`area_off + area.tail_ix`): one structural entry at the
area's slack-bus P-mismatch row (`2*slack_bus_ix - 1`), bus-type-invariant (survives a
PV<->PQ Q-limit flip of that bus — the row position never depends on bus type). Row r_a
(`area_off + area.tail_ix`): the union pattern of both state columns at every non-REF tie
endpoint bus of every tie incident to area a, via `_push_area_row_endpoint_cols!`. A tie with
BOTH endpoints controlled (different areas) contributes to TWO area rows. `∂r_a/∂ΔP_a` is
structurally ABSENT (zero diagonal border — KLU's full pivoting handles it; do not stamp).
Zero work when no area is controlled.
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
    bus_type = view(data.bus_type, :, 1)
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
                push3, area_off + tie.from_area_tail, tie, bus_type)
        iszero(tie.to_area_tail) ||
            _push_area_row_endpoint_cols!(push3, area_off + tie.to_area_tail, tie, bus_type)
    end
    return
end

# Fill one tie endpoint's contribution into area row `row`: PQ writes both columns; PV writes
# only the θ column (its Vm column was pre-zeroed by `_zero_area_row_bus_cols!` and stays at
# that structural zero — a PV bus's own |V| is fixed, so `P_m` has no dependence on it through
# the STATE vector even though it varies physically); REF contributes nothing (see
# `_push_area_row_bus_cols!`). `+=` (not `=`) because a bus can be the shared endpoint of
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
# both columns of both endpoints, unconditionally (mirrors the union pattern the structure
# stamped; REF endpoints have no slots to zero). Without this reset, `_accumulate_area_endpoint!`'s
# `+=` would pile onto a stale value from a previous Jacobian evaluation instead of the fresh one.
function _zero_area_row_bus_cols!(
    Jv::SparseArrays.SparseMatrixCSC{Float64, J_INDEX_TYPE},
    row::Int,
    bus_ix::Int,
    bus_type,
)
    bus_type[bus_ix] == PSY.ACBusTypes.REF && return
    Jv[row, 2 * bus_ix - 1] = 0.0
    Jv[row, 2 * bus_ix] = 0.0
    return
end

function _zero_area_row_endpoint_cols!(
    Jv::SparseArrays.SparseMatrixCSC{Float64, J_INDEX_TYPE},
    tie::AreaTie,
    area_off::Int,
    bus_type,
)
    iszero(tie.from_area_tail) ||
        _zero_area_row_bus_cols!(
            Jv,
            area_off + tie.from_area_tail,
            tie.from_bus_ix,
            bus_type,
        )
    iszero(tie.from_area_tail) ||
        _zero_area_row_bus_cols!(Jv, area_off + tie.from_area_tail, tie.to_bus_ix, bus_type)
    iszero(tie.to_area_tail) ||
        _zero_area_row_bus_cols!(Jv, area_off + tie.to_area_tail, tie.from_bus_ix, bus_type)
    iszero(tie.to_area_tail) ||
        _zero_area_row_bus_cols!(Jv, area_off + tie.to_area_tail, tie.to_bus_ix, bus_type)
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

"""
Fill the area-interchange tail Jacobian entries (polar) — spec §2's bordered block. Called
each iteration after the bus, LCC, and VSC entries. Column ΔP_a is the constant `-1.0` at each
area's slack-bus P-mismatch row (rewritten every call, matching the LCC angle-clamp-row
convention — `ac_power_flow_jacobian.jl`'s `_set_entries_for_lcc`). Row r_a is filled by a
zero-then-accumulate two-pass sweep over every tie (`_zero_area_row_endpoint_cols!` then
`_accumulate_area_row!`) since a boundary bus of degree > 1 feeds the same area row from
multiple ties. Zero work when no area is controlled.
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

    @inbounds for area in aid.areas
        Jv[2 * area.slack_bus_ix - 1, area_off + area.tail_ix] = -1.0
    end

    @inbounds for tie in aid.ties
        _zero_area_row_endpoint_cols!(Jv, tie, area_off, bus_type)
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
    return
end
