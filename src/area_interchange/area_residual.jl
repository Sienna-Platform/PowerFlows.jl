# Polar residual for embedded PSS/E-style area net-interchange control:
# the tie-flow kernel and the tail writer for the `r_a = NI_a - PDES_a` residual rows. The
# ΔP_a <-> slack-bus P-balance coupling lives in `ac_power_flow_residual.jl` (same seam as
# the distributed-slack `P_slack` term); this file only covers the tie-flow/NI side.

"""
    _tie_admittances(tie, ybus_nzval) -> (g11, b11, g12, b12, g21, b21, g22, b22)

Read `tie`'s corridor admittances from the aggregate Y-bus 2×2 block through
`tie.nz_offsets` (never cached — a controlled tap mutates `ybus_nzval` in place, so this must
be re-read every evaluation). Off-diagonal reads (`o[2]`/`o[3]`) are the corridor's own
mutual admittance already (only corridor members contribute there) and are used as-is. The
DIAGONAL reads (`o[1]`/`o[4]`) are NOT the corridor's own self-admittance: a nodal Y-bus
diagonal sums every branch/shunt incident at that bus. `tie.diag_pollution` is the
enrollment-time-cached `(ybus_diag − Σ_corridor_members y11/y22)` correction subtracted here
to recover the corridor's own self-term — see `AreaTie`'s docstring for the derivation.

Shared, concrete (`NTuple{8, Float64}`), non-allocating so both the residual kernel
(`_tie_metered_active_power`) and the Jacobian partials
(`_tie_metered_active_power_partials`, `area_jacobian.jl`) always differentiate the exact
same values — the corrected DIAGONAL read must never be duplicated between the two.
"""
function _tie_admittances(
    tie::AreaTie,
    ybus_nzval::Vector{YBUS_ELTYPE},
)
    o = tie.nz_offsets
    y11 = ybus_nzval[o[1]] - tie.diag_pollution[1]
    y12 = ybus_nzval[o[2]]
    y21 = ybus_nzval[o[3]]
    y22 = ybus_nzval[o[4]] - tie.diag_pollution[2]
    return (
        real(y11), imag(y11), real(y12), imag(y12),
        real(y21), imag(y21), real(y22), imag(y22),
    )
end

"""
    _tie_metered_active_power(tie, Vm_f, θ_f, Vm_t, θ_t, ybus_nzval) -> Float64

Active power flowing out of `tie`'s METERED end. Admittances are read via `_tie_admittances`
(diagonal-pollution-corrected — see that function's docstring).

Metered at `from`: `P_m = Vf² g11 + Vf·Vt·(g12·cos(θf−θt) + b12·sin(θf−θt))`.
Metered at `to` swaps roles: `P_m = Vt² g22 + Vt·Vf·(g21·cos(θt−θf) + b21·sin(θt−θf))`.
"""
function _tie_metered_active_power(
    tie::AreaTie,
    Vm_f::Float64,
    θ_f::Float64,
    Vm_t::Float64,
    θ_t::Float64,
    ybus_nzval::Vector{YBUS_ELTYPE},
)
    (g11, b11, g12, b12, g21, b21, g22, b22) = _tie_admittances(tie, ybus_nzval)
    if tie.metered_from
        sinΔθ, cosΔθ = sincos(θ_f - θ_t)
        return Vm_f^2 * g11 + Vm_f * Vm_t * (g12 * cosΔθ + b12 * sinΔθ)
    end
    sinΔθ, cosΔθ = sincos(θ_t - θ_f)
    return Vm_t^2 * g22 + Vm_t * Vm_f * (g21 * cosΔθ + b21 * sinΔθ)
end

"""
    _dc_tie_metered_active_power(data, dcn, tie, Vm, time_step) -> Float64

Active power flowing out of `tie`'s METERED end over the DC link — the DC-line analogue of
`_tie_metered_active_power`. For an LCC the metered terminal's `_lcc_ac_active_powers` value
is already "power into the DC link" signed (rectifier `from` `> 0`, inverter `to` `< 0`), so
it is returned as-is. For a VSC the converter's `P_c` state is bus-injection signed (into the
AC bus), so power leaving the area is its negation. Dispatch is on `tie.kind` by equality (see
`DCTie`), never `isa`.
"""
function _dc_tie_metered_active_power(
    data::ACPowerFlowData,
    dcn::DCNetwork,
    tie::DCTie,
    Vm::AbstractVector{Float64},
    time_step::Int,
)
    if tie.kind == DC_TIE_LCC
        (P_from, P_to) =
            _lcc_ac_active_powers(data, tie.lcc_ix, time_step, Vm[tie.from_bus_ix],
                Vm[tie.to_bus_ix])
        if tie.metered_from
            return P_from
        end
        return P_to
    end
    if tie.metered_from
        return -dcn.p_c[tie.from_conv_ix, time_step]
    end
    return -dcn.p_c[tie.to_conv_ix, time_step]
end

"""
    _set_area_tail_residuals!(F, x, data, area_off, time_step)

Write the area-interchange residual rows `F[area_off + a] = NI_a - PDES_a` for every
enrolled controlled area. Accumulates each tie's metered active power into
`data.area_interchange.ni_scratch` (pre-allocated, zeroed here — no per-iteration
allocation): the metered side's controlled area (if any) gets `+P_m`, the other side's
(if controlled) gets `-P_m` (spec §2 sign convention); an uncontrolled side
(`iszero(tail)`) is skipped. `x` is accepted for interface parity with the other tail
writers (`_set_lcc_tail_residuals!`, `_set_vsc_tail_residuals!`); the polar formulation
reads the current iterate off `data.bus_magnitude`/`data.bus_angles`, already updated
in place earlier in `_update_residual_values!`.
"""
function _set_area_tail_residuals!(
    F::Vector{Float64},
    x::Vector{Float64},
    data::ACPowerFlowData,
    area_off::Int,
    time_step::Int,
)
    aid = data.area_interchange
    ni = aid.ni_scratch
    fill!(ni, 0.0)
    Vm = view(data.bus_magnitude, :, time_step)
    θ = view(data.bus_angles, :, time_step)
    ybus_nzval = SparseArrays.nonzeros(data.power_network_matrix.data)
    @inbounds for tie in aid.ties
        f = tie.from_bus_ix
        t = tie.to_bus_ix
        P_m = _tie_metered_active_power(tie, Vm[f], θ[f], Vm[t], θ[t], ybus_nzval)
        if tie.metered_from
            metered_tail = tie.from_area_tail
            other_tail = tie.to_area_tail
        else
            metered_tail = tie.to_area_tail
            other_tail = tie.from_area_tail
        end
        iszero(metered_tail) || (ni[metered_tail] += P_m)
        iszero(other_tail) || (ni[other_tail] -= P_m)
    end
    if !isempty(aid.dc_ties)
        dcn = get_dc_network(data)
        @inbounds for tie in aid.dc_ties
            P_conv = _dc_tie_metered_active_power(data, dcn, tie, Vm, time_step)
            if tie.metered_from
                metered_tail = tie.from_area_tail
                other_tail = tie.to_area_tail
            else
                metered_tail = tie.to_area_tail
                other_tail = tie.from_area_tail
            end
            iszero(metered_tail) || (ni[metered_tail] += P_conv)
            iszero(other_tail) || (ni[other_tail] -= P_conv)
        end
    end
    @inbounds for area in aid.areas
        F[area_off + area.tail_ix] = ni[area.tail_ix] - area.pdes
    end
    return
end

# ---------------------------------------------------------------------------
# Greedy relax loop mechanism. The loop control
# flow itself lives at the driver seam (`solve_ac_power_flow.jl`'s
# `_ac_power_flow_with_area_relax!`); this section supplies the pure field-surgery and
# NI-recomputation primitives it calls.
# ---------------------------------------------------------------------------

"""
    _area_residual_gaps(data, time_step) -> Vector{Float64}

`r_a = NI_a - PDES_a` for every CURRENTLY enrolled (working) controlled area, evaluated at
`data`'s current bus state (`data.bus_magnitude`/`data.bus_angles` for `time_step`) — the
SAME kernel `_set_area_tail_residuals!` uses inside the Newton residual, run here directly
against a scratch vector sized to just the area tail (offset `0`) so it can be called
standalone after a solve returns, converged or not. `x` is unused by the polar kernel (see
`_set_area_tail_residuals!`'s docstring), so an empty placeholder is passed.
"""
function _area_residual_gaps(data::ACPowerFlowData, time_step::Int)
    n = n_controlled_areas(data)
    gaps = zeros(n)
    iszero(n) && return gaps
    _set_area_tail_residuals!(gaps, Float64[], data, 0, time_step)
    return gaps
end

"""
    _area_net_interchange(ties, dc_ties, tail_ix, data, time_step) -> Float64

Net interchange for the area at `tail_ix`, computed directly from the tie-flow kernels at
`data`'s current bus state — independent of whether that area is still in
`data.area_interchange.areas`. Used by the results table to report an achieved NI for a
relaxed area (pass `aid.pristine_ties`/`pristine_dc_ties` with its PRISTINE `tail_ix`).
`ties` and `dc_ties` must come from the SAME snapshot (both pristine or both working).
"""
function _area_net_interchange(
    ties::Vector{AreaTie},
    dc_ties::Vector{DCTie},
    tail_ix::Int,
    data::ACPowerFlowData,
    time_step::Int,
)
    Vm = view(data.bus_magnitude, :, time_step)
    θ = view(data.bus_angles, :, time_step)
    ybus_nzval = SparseArrays.nonzeros(data.power_network_matrix.data)
    ni = 0.0
    @inbounds for tie in ties
        (tie.from_area_tail == tail_ix || tie.to_area_tail == tail_ix) || continue
        f = tie.from_bus_ix
        t = tie.to_bus_ix
        P_m = _tie_metered_active_power(tie, Vm[f], θ[f], Vm[t], θ[t], ybus_nzval)
        metered_tail = tie.from_area_tail
        other_tail = tie.to_area_tail
        if !tie.metered_from
            metered_tail = tie.to_area_tail
            other_tail = tie.from_area_tail
        end
        metered_tail == tail_ix && (ni += P_m)
        other_tail == tail_ix && (ni -= P_m)
    end
    if !isempty(dc_ties)
        dcn = get_dc_network(data)
        @inbounds for tie in dc_ties
            (tie.from_area_tail == tail_ix || tie.to_area_tail == tail_ix) || continue
            P_conv = _dc_tie_metered_active_power(data, dcn, tie, Vm, time_step)
            metered_tail = tie.from_area_tail
            other_tail = tie.to_area_tail
            if !tie.metered_from
                metered_tail = tie.to_area_tail
                other_tail = tie.from_area_tail
            end
            metered_tail == tail_ix && (ni += P_conv)
            other_tail == tail_ix && (ni -= P_conv)
        end
    end
    return ni
end

"""
    _ensure_pristine_area_set!(data, time_step)

Reset the WORKING `areas`/`ties`/`dc_ties`/`ni_scratch`/`delta_p` to the full PRISTINE
enrollment before a time step's own (potential) greedy relax loop runs, undoing any
de-enrollment a PREVIOUS time step made on this same `data` object (relax decisions are
per time step, never permanent). A no-op — no cache churn — when the working
set already matches the pristine count (the common case: no relax has ever happened on this
`data`).

Explicitly invalidates the Jacobian-structure and NR symbolic-factorization caches
(`data.ac_jacobian_structure_cache`/`data.polar_nr_cache`): both are keyed on
`data.area_interchange`/`data.power_network_matrix` IDENTITY, never on its CONTENTS, so a
tail-size change made by mutating the SAME `AreaInterchangeData` object in place — the only
option, since `PowerFlowData` is immutable and `data.area_interchange` itself can't be
reassigned (see `initialize_power_flow_data.jl`) — would otherwise be invisible to the
identity check and silently reuse a wrong-sized cached structure/factorization.
"""
function _ensure_pristine_area_set!(data::ACPowerFlowData, time_step::Int)
    aid = data.area_interchange
    length(aid.areas) == length(aid.pristine_areas) && return
    empty!(aid.areas)
    append!(aid.areas, aid.pristine_areas)
    empty!(aid.ties)
    append!(aid.ties, aid.pristine_ties)
    empty!(aid.dc_ties)
    append!(aid.dc_ties, aid.pristine_dc_ties)
    resize!(aid.ni_scratch, length(aid.areas))
    aid.delta_p = copy(aid.pristine_delta_p)
    data.ac_jacobian_structure_cache[] = nothing
    data.polar_nr_cache[] = nothing
    return
end

"""
    _sync_pristine_delta_p!(data, time_step)

Write the WORKING `delta_p[:, time_step]` (one row per CURRENTLY enrolled area, keyed by its
renumbered working `tail_ix`) back into `pristine_delta_p[:, time_step]` (keyed by the area's
never-changing PRISTINE `tail_ix`, looked up by name), once a time step's solve converges.
The persistent mirror is what the NEXT time step's `_ensure_pristine_area_set!` reseeds from,
so a later time step's warm start recovers THIS time step's converged `ΔP_a` for every area
that survived. A relaxed area's row is left untouched — it has no converged `ΔP_a` to record;
the results table reports `delta_p = 0.0` for it directly, not from this mirror.
"""
function _sync_pristine_delta_p!(data::ACPowerFlowData, time_step::Int)
    aid = data.area_interchange
    pristine_tail_of = Dict(area.name => area.tail_ix for area in aid.pristine_areas)
    @inbounds for area in aid.areas
        aid.pristine_delta_p[pristine_tail_of[area.name], time_step] =
            aid.delta_p[area.tail_ix, time_step]
    end
    return
end

"""
    _deenroll_area!(data, drop_tail_ix) -> ControlledArea

Remove the WORKING area at `drop_tail_ix` from `data.area_interchange` (greedy relax):
the surviving areas' `tail_ix` are renumbered contiguously
(the state vector has no room for gaps), `ties`/`dc_ties` referencing the dropped tail are
translated to `0` (uncontrolled — the kernel already skips a zero tail), and `delta_p` is
rebuilt at the new (smaller) size with each survivor's row carried over from its OLD
`tail_ix`, for every time step column (so OTHER time steps' already-converged mirrors are
not corrupted by a de-enrollment that only THIS time step's relax loop decided on).

Mutates `data.area_interchange`'s FIELDS in place (`PowerFlowData` is immutable, so the
field itself can't be reassigned — see `initialize_power_flow_data.jl`) and explicitly
invalidates the Jacobian-structure/NR-factorization caches for the identity-vs-contents
reason documented on `_ensure_pristine_area_set!`. Returns the dropped `ControlledArea`
(its ORIGINAL `pdes` survives on it, unaffected by the renumbering) for the caller to
report/record.
"""
function _deenroll_area!(data::ACPowerFlowData, drop_tail_ix::Int)
    aid = data.area_interchange
    dropped = aid.areas[drop_tail_ix]
    old_to_new = zeros(Int, length(aid.areas))
    new_areas = ControlledArea[]
    for area in aid.areas
        area.tail_ix == drop_tail_ix && continue
        new_tail = length(new_areas) + 1
        old_to_new[area.tail_ix] = new_tail
        push!(new_areas, ControlledArea(area.name, area.slack_bus_ix, area.pdes, new_tail))
    end
    function _translate(old_tail::Int)
        iszero(old_tail) && return 0
        return old_to_new[old_tail]
    end
    new_ties = [
        AreaTie(
            tie.from_bus_ix, tie.to_bus_ix, tie.nz_offsets, tie.metered_from,
            _translate(tie.from_area_tail), _translate(tie.to_area_tail),
            tie.diag_pollution,
        ) for tie in aid.ties
    ]
    new_dc_ties = [
        DCTie(
            tie.kind, tie.lcc_ix, tie.from_conv_ix, tie.to_conv_ix,
            tie.from_bus_ix, tie.to_bus_ix, tie.metered_from,
            _translate(tie.from_area_tail), _translate(tie.to_area_tail),
        ) for tie in aid.dc_ties
    ]
    n_ts = size(aid.delta_p, 2)
    new_delta_p = zeros(length(new_areas), n_ts)
    for area in aid.areas
        area.tail_ix == drop_tail_ix && continue
        new_delta_p[old_to_new[area.tail_ix], :] .= aid.delta_p[area.tail_ix, :]
    end
    empty!(aid.areas)
    append!(aid.areas, new_areas)
    empty!(aid.ties)
    append!(aid.ties, new_ties)
    empty!(aid.dc_ties)
    append!(aid.dc_ties, new_dc_ties)
    resize!(aid.ni_scratch, length(new_areas))
    aid.delta_p = new_delta_p
    data.ac_jacobian_structure_cache[] = nothing
    data.polar_nr_cache[] = nothing
    return dropped
end

"""
    _warn_area_violations(data, time_step)

Solve-end diagnostic: `@warn` for any CURRENTLY enrolled (working) area
whose achieved NI still misses its target by more than `data.area_interchange.tolerance`
after a converged solve. Defensive — the area residual row is driven to ~0 by the same
Newton tolerance as every other row, so this should not normally fire — as opposed to the
PRIMARY infeasibility signal, which is the greedy relax loop's own per-de-enrollment
`@error` (`_ac_power_flow_with_area_relax!` in `solve_ac_power_flow.jl`).
"""
function _warn_area_violations(data::ACPowerFlowData, time_step::Int)
    aid = data.area_interchange
    gaps = _area_residual_gaps(data, time_step)
    for area in aid.areas
        gap = gaps[area.tail_ix]
        abs(gap) > aid.tolerance || continue
        @warn "Area interchange: area \"$(area.name)\" achieved NI = " *
              "$(area.pdes + gap) misses its target PDES = $(area.pdes) by $gap " *
              "(tolerance $(aid.tolerance))."
    end
    return
end
