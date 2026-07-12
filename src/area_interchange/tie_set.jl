# Pure tie-enumeration for PSS/E-style area interchange control (see `area_types.jl`).
# Phase 1: AC branches only — `TwoTerminalHVDC` (LCC/VSC) is a Y-bus-less `ACBranch`
# subtype (modeled as fixed injections elsewhere) and is excluded via `PSY.ACTransmission`.
# `ThreeWindingTransformer` (abstract; concrete `Transformer3W`/`PhaseShiftingTransformer3W`)
# has no `get_arc` and is handled by decomposing it into its three star-node windings —
# see `_tie_arcs`.

function _area_name(bus)
    area = PSY.get_area(bus)
    isnothing(area) && return ""
    return PSY.get_name(area)
end

"""Whether `ext["metered_end"]` marks the `from` end as metered. Absent key silently
defaults to from-metered (the legitimate default); a present-but-unrecognized value
(typo, wrong case, stray whitespace) warns and also defaults to from-metered rather than
silently coercing it — a "to"-only special case would mask bad data."""
function _metered_from(branch)
    ext = PSY.get_ext(branch)
    haskey(ext, "metered_end") || return true
    value = ext["metered_end"]
    value == "from" && return true
    value == "to" && return false
    @warn "$(PSY.summary(branch)): ext[\"metered_end\"] = $(repr(value)) is neither \
        \"from\" nor \"to\"; defaulting to from-metered."
    return true
end

# Whether `br` should be considered for tie enumeration at all, beyond the `available`
# filter `PSY.get_available_components` already applies. Mirrors the additional gate PNM
# applies when stamping the Y-bus (`YbusACBranches.jl`): a `DiscreteControlledACBranch`
# only gets Y-bus entries when it is also `CLOSED`. An available-but-OPEN switch has no
# Y-bus block, so treating it like a normal in-service branch would either hit a
# structural-zero error in `_nz_index` or silently reuse a parallel branch's offsets.
_tie_in_service(::PSY.ACTransmission) = true
_tie_in_service(br::PSY.DiscreteControlledACBranch) =
    PSY.get_branch_status(br) == PSY.DiscreteControlledBranchStatus.CLOSED

# The two-terminal `Arc`s to treat as tie candidates for one branch, paired with the
# object `PNM.ybus_branch_entries` accepts to compute that arc's OWN primitive Y11/Y22 (the
# corridor's true self-admittance, as opposed to the aggregate Y-bus diagonal — see
# `AreaTie.diag_pollution`). Plain `ACTransmission` contributes its own arc/branch;
# `ThreeWindingTransformer` has none (see the module note above) and is decomposed into its
# three star-node windings instead.
_tie_arcs(branch::PSY.ACTransmission) = ((PSY.get_arc(branch), branch),)

"""Decompose a three-winding transformer into its three star-node windings — the same
`Arc`s PNM stamps into the Y-bus per-winding (`YbusACBranches.jl`'s `_ybus!` for
`ThreeWindingTransformer`, keyed through the same `bus_lookup`/`reverse_bus_search_map`
this file already resolves against), gated by the SAME per-winding availability flags PNM
gates on. The star bus is a real, resolvable network node — PSS/E parsing assigns it the
PRIMARY terminal's area (`_create_starbus_from_transformer` in PowerSystems'
`parsers/pm_io/psse.jl`), not an arealess singleton — so each winding is just a normal
two-terminal tie candidate: a winding wholly inside one area (terminal and star share an
area, e.g. the primary winding) contributes no tie, and a winding crossing an area
boundary becomes a correctly metered tie. KCL at the star bus (zero net injection there)
makes the sum over the boundary-crossing windings equal the transformer's true net export
from each area, with no interior/boundary special-casing needed at the transformer level.
Each winding is paired with the `PNM.ThreeWindingTransformerWinding` PNM itself uses to
stamp that winding's Y11/Y22 into the aggregate Y-bus (`Ybus.jl`'s `_ybus!`), so
`PNM.ybus_branch_entries` on it returns the SAME primitive PNM summed into the diagonal."""
function _tie_arcs(branch::PSY.ThreeWindingTransformer)
    arcs = Tuple{PSY.Arc, PNM.ThreeWindingTransformerWinding}[]
    PSY.get_available_primary(branch) && push!(
        arcs,
        (PSY.get_primary_star_arc(branch), PNM.ThreeWindingTransformerWinding(branch, 1)),
    )
    PSY.get_available_secondary(branch) && push!(
        arcs,
        (
            PSY.get_secondary_star_arc(branch),
            PNM.ThreeWindingTransformerWinding(branch, 2),
        ),
    )
    PSY.get_available_tertiary(branch) && push!(
        arcs,
        (PSY.get_tertiary_star_arc(branch), PNM.ThreeWindingTransformerWinding(branch, 3)),
    )
    return arcs
end

# `(y11, y22)` of `entry`'s own primitive 2×2 block, aligned to `entry`'s own arc
# from/to — i.e. the SAME orientation `build_area_ties` resolves `fix`/`tix` from (both read
# `PSY.get_from(arc)`/`PSY.get_to(arc)`), so `y11` lands at `fix` and `y22` at `tix` with no
# separate bookkeeping. This is the corridor member's contribution to `AreaTie.diag_pollution`.
function _primitive_diag(entry)
    (y11, _, _, y22) = PNM.ybus_branch_entries(entry)
    return (ComplexF64(y11), ComplexF64(y22))
end

# One resolved tie candidate before parallel-branch deduplication.
struct _TieCandidate
    name::String
    fix::Int
    tix::Int
    metered_from::Bool
    tail_from::Int
    tail_to::Int
    y11::ComplexF64   # own primitive diag at `fix`, from `_primitive_diag`
    y22::ComplexF64   # own primitive diag at `tix`
end

# Running accumulator for one deduplicated corridor while candidates are folded in;
# `y11`/`y22` accumulate the SUM of every corridor member's own primitive diag (aligned to
# `fix`/`tix`), which becomes `AreaTie.diag_pollution` once the final aggregate Y-bus
# diagonal is read (see `_dedup_ties`).
mutable struct _TieAccum
    fix::Int
    tix::Int
    metered_from::Bool
    tail_from::Int
    tail_to::Int
    y11::ComplexF64
    y22::ComplexF64
end

"""Deduplicate tie candidates to one `AreaTie` per unordered reduced bus pair.
`_ybus_block_offsets` already returns the AGGREGATE Y-bus block for a bus pair (parallel
admittances are summed into one sparse entry — that IS the corridor flow), so pushing one
`AreaTie` per parallel branch would double-count the corridor when downstream code sums
per-tie flows. The first-seen candidate for a pair fixes the tie's direction and metered
end; later parallel members are folded in (their own primitive diag accumulated into the
corridor's `diag_pollution` sum), and a disagreement on the (direction-adjusted) metered
end is a data inconsistency worth a `@warn`, not a silent pick.

`diag_pollution` is finalized only after every candidate for a key has been folded in: it
is `aggregate_ybus_diag − Σ_corridor_members primitive_diag`, so the kernel can later
recover the corridor's own self-admittance from the aggregate Y-bus diagonal (which sums
EVERY branch/shunt incident at that bus, not just this corridor's members)."""
function _dedup_ties(candidates::Vector{_TieCandidate}, ybus)
    accums = _TieAccum[]
    seen = Dict{Tuple{Int, Int}, Int}()
    for cand in candidates
        key = (min(cand.fix, cand.tix), max(cand.fix, cand.tix))
        ix = get(seen, key, 0)
        if iszero(ix)
            push!(
                accums,
                _TieAccum(
                    cand.fix,
                    cand.tix,
                    cand.metered_from,
                    cand.tail_from,
                    cand.tail_to,
                    cand.y11,
                    cand.y22,
                ),
            )
            seen[key] = length(accums)
            continue
        end
        acc = accums[ix]
        effective_metered_from = cand.metered_from
        if cand.fix != acc.fix
            effective_metered_from = !cand.metered_from
        end
        if effective_metered_from != acc.metered_from
            @warn "$(cand.name): parallel branch shares a reduced bus pair with an \
                already-enumerated tie and disagrees on the (direction-adjusted) metered \
                end; keeping the first-seen branch's metered end."
        end
        if cand.fix == acc.fix
            acc.y11 += cand.y11
            acc.y22 += cand.y22
        else
            acc.y11 += cand.y22
            acc.y22 += cand.y11
        end
    end
    ties = AreaTie[]
    nzval = SparseArrays.nonzeros(ybus.data)
    for acc in accums
        o = _ybus_block_offsets(ybus, acc.fix, acc.tix)
        diag_pollution = (
            ComplexF64(nzval[o[1]]) - acc.y11,
            ComplexF64(nzval[o[4]]) - acc.y22,
        )
        push!(
            ties,
            AreaTie(
                acc.fix,
                acc.tix,
                o,
                acc.metered_from,
                acc.tail_from,
                acc.tail_to,
                diag_pollution,
            ),
        )
    end
    return ties
end

"""
    build_area_ties(sys, bus_lookup, ybus, nrd, bus_area_map) -> Vector{AreaTie}

Enumerate AC-branch ties whose post-reduction endpoints straddle an area boundary.

`bus_lookup` maps PSY bus number to reduced-network index; `ybus` is the assembled
`AC_Ybus_Matrix`; `nrd` is the `PNM.NetworkReductionData` used to resolve merged buses.
`bus_area_map` maps a reduced-network bus index to its enrolled area `tail_ix` (`0` for
an uncontrolled or area-less bus); ties where both endpoints are `0` are not stored, since
no residual row references that boundary.

`nz_offsets` are cached via `_ybus_block_offsets` — offsets only, never admittance
copies — so a controlled tap that is also a tie keeps feeding correct flows after
`apply_parameter!` mutates the Y-bus in place. Out-of-service branches are excluded
(`PSY.get_available_components` already filters on `available`; `_tie_in_service` filters
the additional discrete-control CLOSED requirement). Parallel branches between the same
reduced bus pair are deduplicated to one `AreaTie` (see `_dedup_ties`).
"""
function build_area_ties(
    sys::PSY.System,
    bus_lookup::Dict{Int, Int},
    ybus,
    nrd,
    bus_area_map::Dict{Int, Int},
)
    reverse_bus_search_map = PNM.get_reverse_bus_search_map(nrd)
    candidates = _TieCandidate[]
    for branch in PSY.get_available_components(PSY.ACTransmission, sys)
        _tie_in_service(branch) || continue
        metered_from = _metered_from(branch)
        for (arc, primitive_entry) in _tie_arcs(branch)
            from_bus = PSY.get_from(arc)
            to_bus = PSY.get_to(arc)
            fb = PSY.get_number(from_bus)
            tb = PSY.get_number(to_bus)
            fix = _resolve_bus_ix(bus_lookup, reverse_bus_search_map, fb)
            tix = _resolve_bus_ix(bus_lookup, reverse_bus_search_map, tb)
            if isnothing(fix) || isnothing(tix)
                missing_bus = tb
                if isnothing(fix)
                    missing_bus = fb
                end
                @warn "$(PSY.summary(branch)): bus $missing_bus is not in the (reduced) \
                    network; branch excluded from area-tie enumeration."
                continue
            end
            if fix == tix
                if _area_name(from_bus) != _area_name(to_bus)
                    @warn "$(PSY.summary(branch)): a zero-impedance reduction merged its \
                        endpoints, which belong to different areas \
                        ($(_area_name(from_bus)) / $(_area_name(to_bus))); the boundary is \
                        evaluated on the reduced network and this branch contributes no tie."
                end
                continue
            end
            tail_from = get(bus_area_map, fix, 0)
            tail_to = get(bus_area_map, tix, 0)
            tail_from == tail_to && continue
            (y11, y22) = _primitive_diag(primitive_entry)
            push!(
                candidates,
                _TieCandidate(
                    PSY.summary(branch),
                    fix,
                    tix,
                    metered_from,
                    tail_from,
                    tail_to,
                    y11,
                    y22,
                ),
            )
        end
    end
    return _dedup_ties(candidates, ybus)
end
