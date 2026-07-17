# Area interchange control (PSS/E-style embedded formulation) runtime data types.
# One new state `ΔP_a` and one residual row `r_a = NI_a − PDES_a` per controlled area,
# appended as a tail after the VSC tail — mirrors the LCC/VSC tail pattern.

"""
A PSY `Area` enrolled for embedded net-interchange control.

# Fields
- `name::String`: PSY area name.
- `slack_bus_ix::Int`: reduced-network bus index of the area slack (PSS/E ISW).
- `pdes::Float64`: net interchange target, pu system base.
- `tail_ix::Int`: 1-based slot in the area tail.
"""
struct ControlledArea
    name::String
    slack_bus_ix::Int
    pdes::Float64
    tail_ix::Int
end

"""
An AC branch whose endpoints straddle a controlled-area boundary (or touch one).

# Fields
- `from_bus_ix::Int`, `to_bus_ix::Int`: reduced-network bus indices.
- `nz_offsets::NTuple{4, Int}`: Ybus `nzval` offsets `Y11,Y12,Y21,Y22` (`ControlledTap` device).
- `metered_from::Bool`: metered end is the `from` bus.
- `from_area_tail::Int`: `tail_ix` of the controlled area owning the `from` bus; `0` = uncontrolled.
- `to_area_tail::Int`: `tail_ix` of the controlled area owning the `to` bus; `0` = uncontrolled.
- `diag_pollution::NTuple{2, ComplexF64}`: `(from, to)` correction so the kernel can recover
  the corridor's OWN self-admittance from the aggregate Y-bus diagonal, which sums EVERY
  branch/shunt incident at that bus, not just this tie's members. Cached once at tie-build
  time as `ybus_diag − Σ_{corridor members} y11/y22_primitive`; live evaluation then does
  `ybus_nzval[o[1]] − diag_pollution[1]` (and the `to`-side analog). This keeps flows exact
  under a controlled tap ON the corridor (the tap's own delta cancels against the live
  diagonal read, so live−constant stays exact) while still being exact for any static
  topology; a non-member device mutating an endpoint diagonal is out of scope and fenced by
  an enrollment-time `@warn` (see `enrollment.jl`), not corrected here.
"""
struct AreaTie
    from_bus_ix::Int
    to_bus_ix::Int
    nz_offsets::NTuple{4, Int}
    metered_from::Bool
    from_area_tail::Int
    to_area_tail::Int
    diag_pollution::NTuple{2, ComplexF64}
end

"""Which converter model backs a [`DCTie`](@ref). `Int8`-backed, mirroring `VSCControlMode`
(`vsc_parameters.jl`): use-site code dispatches on it by equality (`tie.kind == DC_TIE_LCC`),
never `isa`/`<:`, keeping `DCTie` a single concrete struct (`Vector{DCTie}` stays
homogeneously typed, unlike an abstract small-type-hierarchy alternative)."""
@enum DCTieKind::Int8 DC_TIE_LCC = 1 DC_TIE_VSC = 2

"""
A converter whose AC bus sits in one controlled area while its DC counterpart terminal sits
in another — the DC-line analogue of `AreaTie`. Structurally mirrors `AreaTie`
(`from_bus_ix`/`to_bus_ix`/`metered_from`/`from_area_tail`/`to_area_tail`) but carries
converter identity instead of a Y-bus block: a DC converter's AC-side active-power injection
(from `lcc_utils.jl`'s `P_lcc_from`/`P_lcc_to`, or a VSC converter's `P_c` tail state) is what
enters `NI_a`, not a branch flow read off Ybus. Kept as its
OWN type rather than an `AreaTie` overload because `nz_offsets`/`diag_pollution` are
Ybus-block concepts with no DC-converter meaning.

Scope: `PSY.TwoTerminalLCCLine` and point-to-point `PSY.TwoTerminalVSCLine` only.
`PSY.InterconnectingConverter` (multi-terminal DC) is NOT enumerated — pairing a converter
with "the" other-area counterpart terminal is ambiguous on an N-terminal DC subnet (unlike a
point-to-point line's unambiguous from/to) and needs its own union-find-style traversal; a
documented gap, mirroring how series-GNE is deferred (see `docs/src/explanation/area_interchange.md`).

# Fields
- `kind::DCTieKind`: `DC_TIE_LCC` or `DC_TIE_VSC`.
- `lcc_ix::Int`: 1-based index into `data.lcc.*` (`data.lcc.bus_indices[lcc_ix] ==
  (from_bus_ix, to_bus_ix)`, rectifier="from"/inverter="to" always, independent of
  `setpoint_at_rectifier`); `0` when `kind == DC_TIE_VSC`.
- `from_conv_ix::Int`, `to_conv_ix::Int`: `DCNetwork` converter indices (`dcn.p_c`/`q_c` row)
  for the VSC line's from/to sides; both `0` when `kind == DC_TIE_LCC`.
- `from_bus_ix::Int`, `to_bus_ix::Int`: reduced-network AC bus indices of the two terminals.
- `metered_from::Bool`: metered end is the `from` terminal (mirrors `AreaTie`/`_metered_from`
  — same `ext["metered_end"]` convention, defaulting to `from`).
- `from_area_tail::Int`, `to_area_tail::Int`: owning-area tail indices; `0` = uncontrolled,
  exactly like `AreaTie`. A DC tie whose in-area terminal is that area's own SLACK bus is
  still emitted — `NI_a` and the bus's own power-balance row are independent residual rows,
  so there is no double-count to guard against.
"""
struct DCTie
    kind::DCTieKind
    lcc_ix::Int
    from_conv_ix::Int
    to_conv_ix::Int
    from_bus_ix::Int
    to_bus_ix::Int
    metered_from::Bool
    from_area_tail::Int
    to_area_tail::Int
end

"""
A controlled area that the greedy relax loop de-enrolled
mid-solve because its interchange schedule proved unenforceable given the network/tie
capacity at a failed Newton iterate. Recorded per time step on
`AreaInterchangeData.relaxed` so the results table (`post_processing.jl`) can still report
it after it drops out of the WORKING `areas` (`pdes` is otherwise unavailable once an area
has been de-enrolled).

# Fields
- `name::String`: PSY area name — matches a `pristine_areas` entry by name.
- `pdes::Float64`: the area's original net-interchange target, pu system base — unaffected
  by relaxation or by any tail renumbering a de-enrollment does to OTHER areas.
"""
struct RelaxedAreaRecord
    name::String
    pdes::Float64
end

"""
    AreaInterchangeData

Holds the enrolled controlled areas and their ties for embedded net-interchange control.
Always present on [`PowerFlowData`](@ref) — empty vectors when control is off, mirroring
`data.lcc` with zero LCCs (no `Union{Nothing}` sentinel).

# Fields
- `areas::Vector{ControlledArea}`, `ties::Vector{AreaTie}`: the WORKING (currently
  enrolled) set — what the residual/Jacobian tail actually sees. The greedy relax loop
  shrinks these mid-solve on a non-converged time step; see
  `pristine_areas`/`pristine_ties` below for the set this is derived from.
- `dc_ties::Vector{DCTie}`: DC-line (LCC/VSC) ties, enumerated alongside `ties` by
  `build_area_ties` and consumed by the residual/Jacobian/relax machinery exactly like
  `ties`. Empty on any system with no cross-area DC converter (zero behavior change for
  AC-only systems).
- `tolerance::Float64`: interchange convergence tolerance.
- `ni_scratch::Vector{Float64}`: pre-allocated per-area net-interchange accumulator, length
  `length(areas)`, reused every residual evaluation by `_set_area_tail_residuals!` (see
  `area_residual.jl`) to avoid a per-iteration allocation on the hot path.
- `delta_p::Matrix{Float64}`: per-area, per-time-step `ΔP_a` mirror, sized
  `(length(areas), n_time_steps)` — mirrors `DCNetwork.p_c`/`q_c`/`node_vdc`. Unlike
  `ni_scratch` this is NOT reset every call — the ΔP<->P-balance coupling pass in
  `_update_residual_values!` writes `delta_p[tail_ix, time_step] = x[area_off + tail_ix]`
  on every residual evaluation (mirrors the LCC tap / `_read_vsc_state!` tail write-back),
  so it always holds the last-evaluated ΔP_a for that time step. `update_state!` reads it
  back, column-indexed by `time_step`, to seed a warm re-solve's `x0` — without this mirror
  there is nowhere to recover a converged ΔP_a from between two top-level
  `solve_power_flow!` calls on the same `data` (see
  `state_indexing_helpers.jl`/`calculate_x0`). The per-time-step column keeps a multi-period
  `data`'s time steps from contaminating each other's warm start.
- `pristine_areas::Vector{ControlledArea}`, `pristine_ties::Vector{AreaTie}`,
  `pristine_dc_ties::Vector{DCTie}`: the FULL enrolled set exactly as originally built by
  `build_area_interchange_data`, kept forever alongside the (possibly shrunk) working fields
  above. Greedy relax de-enrolls areas from the WORKING set
  only, for the REST of the CURRENT time step's attempts; these pristine copies are never
  mutated, so `_ensure_pristine_area_set!` (`area_residual.jl`) can reset the working set back
  to full enrollment before the NEXT time step's own attempt — relax decisions are per time
  step, never permanent for `data`'s lifetime. Also the only source left for computing a
  relaxed area's achieved (floating) net interchange for the results table, since its tail is
  translated to `0` (uncontrolled) out of the WORKING `ties` the moment it is de-enrolled.
- `pristine_delta_p::Matrix{Float64}`: persistent, PRISTINE-`tail_ix`-indexed mirror of
  `delta_p`, sized `(length(pristine_areas), n_time_steps)`. Unlike the working `delta_p`
  (renumbered/shrunk by a de-enrollment), this one's row layout never changes, so it
  survives across time steps regardless of which areas a PREVIOUS time step relaxed.
  `_ensure_pristine_area_set!` reseeds a freshly-reset working `delta_p` from it;
  `_sync_pristine_delta_p!` writes the working mirror back into it once a time step's
  solve converges.
- `relaxed::Dict{Int, Vector{RelaxedAreaRecord}}`: per-time-step record of areas relaxed
  away that time step (absent/empty = none), keyed by `time_step` so a multi-period
  `data`'s relax decisions stay independent.
"""
mutable struct AreaInterchangeData
    areas::Vector{ControlledArea}
    ties::Vector{AreaTie}
    dc_ties::Vector{DCTie}
    tolerance::Float64
    ni_scratch::Vector{Float64}
    delta_p::Matrix{Float64}
    pristine_areas::Vector{ControlledArea}
    pristine_ties::Vector{AreaTie}
    pristine_dc_ties::Vector{DCTie}
    pristine_delta_p::Matrix{Float64}
    relaxed::Dict{Int, Vector{RelaxedAreaRecord}}
end

"""Number of controlled areas enrolled in `aid`."""
n_controlled_areas(aid::AreaInterchangeData) = length(aid.areas)

"""Length of the area-interchange state/residual tail: one `ΔP_a` per enrolled area."""
area_tail_length(aid::AreaInterchangeData) = length(aid.areas)
