# Enrollment: derive the surviving `ControlledArea`/`AreaTie` set for embedded PSS/E-style
# net-interchange control from a `PSY.System` + `PowerFlowData`, applying the enrollment
# guards below. Runs once at `PowerFlowData` construction, gated on
# `get_area_interchange_control(pf)` (see `initialize_power_flow_data!`).

# Group system buses by PSY area name; a bus with `area === nothing` is excluded (never
# SLACK-eligible or REF/island-membership relevant to any area).
function _buses_by_area(sys::PSY.System)
    grouped = Dict{String, Vector{PSY.ACBus}}()
    for bus in PSY.get_components(PSY.ACBus, sys)
        area = PSY.get_area(bus)
        isnothing(area) && continue
        push!(get!(grouped, PSY.get_name(area), PSY.ACBus[]), bus)
    end
    return grouped
end

_area_slack_buses(buses::Vector{PSY.ACBus}) =
    filter(b -> PSY.get_bustype(b) == PSY.ACBusTypes.SLACK, buses)

# Resolved (post-reduction), deduplicated bus indices for a set of PSY buses. A bus that
# was removed by network reduction with no surviving parent is silently skipped here —
# callers that must react to an individually UNRESOLVABLE bus (the area's own SLACK bus,
# guard 2) resolve that one bus directly instead.
function _resolved_bus_ixs(
    buses::Vector{PSY.ACBus},
    bus_lookup::Dict{Int, Int},
    reverse_bus_search_map::Dict{Int, Int},
)
    ixs = Int[]
    for bus in buses
        ix = _resolve_bus_ix(bus_lookup, reverse_bus_search_map, PSY.get_number(bus))
        isnothing(ix) && continue
        push!(ixs, ix)
    end
    return unique(ixs)
end

# Bus index -> electrical island id (its subnetwork's REF bus index), for every bus in the
# reduced network. Mirrors `_find_subnetworks_for_reference_buses` (used at residual
# construction time) so area/island membership (guards 3-4) matches exactly what the solve
# itself will see.
function _bus_island_map(data::ACPowerFlowData)
    bus_type = view(data.bus_type, :, 1)
    subnetworks =
        _find_subnetworks_for_reference_buses(data.power_network_matrix.data, bus_type)
    island = Dict{Int, Int}()
    for (ref_ix, members) in subnetworks
        for m in members
            island[m] = ref_ix
        end
    end
    return island
end

# Net-interchange target and incidence count per PSY area name, aggregated over ALL
# `PSY.AreaInterchange` records (regardless of enrollment): PDES_a = Σ_{from=a} flow −
# Σ_{to=a} flow. `check_unit_setting(sys)` (already asserted at every PowerFlows entry
# point) guarantees SYSTEM_BASE, so `get_active_power_flow` already reads pu — no division
# by base power.
function _area_pdes(sys::PSY.System)
    pdes = Dict{String, Float64}()
    incident = Dict{String, Int}()
    for ai in PSY.get_available_components(PSY.AreaInterchange, sys)
        flow = PSY.get_active_power_flow(ai)
        from_name = PSY.get_name(PSY.get_from_area(ai))
        to_name = PSY.get_name(PSY.get_to_area(ai))
        pdes[from_name] = get(pdes, from_name, 0.0) + flow
        pdes[to_name] = get(pdes, to_name, 0.0) - flow
        incident[from_name] = get(incident, from_name, 0) + 1
        incident[to_name] = get(incident, to_name, 0) + 1
    end
    return pdes, incident
end

# Guards 1, 1b, 2, 3, 4, 6: resolve area `name`'s SLACK bus and validate it in isolation
# (guards that don't need cross-area tie information). Returns the resolved SLACK bus
# index for a surviving candidate, or `0` for a de-enrolled/uncontrolled area (every branch
# below either emits the guard's `@warn` or is the silent rule-1 zero-SLACK case).
function _area_slack_candidate(
    name::String,
    buses::Vector{PSY.ACBus},
    data::ACPowerFlowData,
    bus_lookup::Dict{Int, Int},
    reverse_bus_search_map::Dict{Int, Int},
    island_of::Dict{Int, Int},
)
    slacks = _area_slack_buses(buses)
    isempty(slacks) && return 0
    if length(slacks) > 1
        @warn "Area \"$name\": $(length(slacks)) SLACK buses \
            ($(join(sort(PSY.get_name.(slacks)), ", "))); an area must have exactly one \
            area slack (PSS/E ISW). De-enrolling."
        return 0
    end
    slack_bus = only(slacks)
    slack_ix =
        _resolve_bus_ix(bus_lookup, reverse_bus_search_map, PSY.get_number(slack_bus))
    if isnothing(slack_ix)
        @warn "Area \"$name\": SLACK bus $(PSY.get_name(slack_bus)) was removed by \
            network reduction with no surviving parent; an area slack must resolve to a \
            real network bus. De-enrolling."
        return 0
    end
    if data.bus_type[slack_ix, 1] == PSY.ACBusTypes.PQ
        @warn "Area \"$name\": SLACK bus $(PSY.get_name(slack_bus)) has no in-service \
            voltage-regulating component (normalized to PQ); an area slack must be able \
            to regulate voltage. De-enrolling."
        return 0
    end
    resolved = _resolved_bus_ixs(buses, bus_lookup, reverse_bus_search_map)
    ref_pos = findfirst(ix -> data.bus_type[ix, 1] == PSY.ACBusTypes.REF, resolved)
    if !isnothing(ref_pos)
        ref_ix = resolved[ref_pos]
        ref_bus = first(
            b for b in buses if
            _resolve_bus_ix(bus_lookup, reverse_bus_search_map, PSY.get_number(b)) ==
            ref_ix
        )
        @warn "Area \"$name\": contains the network reference bus \
            $(PSY.get_name(ref_bus)); an area holding REF cannot be embedded \
            net-interchange-controlled (its angle is fixed, not a redistribution \
            target). De-enrolling."
        return 0
    end
    islands = unique(island_of[ix] for ix in resolved)
    if length(islands) > 1
        @warn "Area \"$name\": buses span $(length(islands)) electrical \
            islands/subnetworks; an embedded-controlled area must lie within a single \
            island. De-enrolling."
        return 0
    end
    # Guard 6 runs here (before guard 5's zero-tie pass in `build_area_interchange_data`)
    # because tie enumeration needs the surviving candidate set that guard 6 helps produce;
    # an area failing both is reported via this guard's slack-absorption message, not
    # guard 5's, since it never reaches the tie pass.
    w_a = sum(data.bus_slack_participation_factors[ix, 1] for ix in resolved)
    if w_a > AREA_SLACK_ABSORPTION_LIMIT
        @warn "Area \"$name\": area buses hold a slack-participation weight of \
            $(round(w_a; digits = 4)) (limit $AREA_SLACK_ABSORPTION_LIMIT); embedding \
            net-interchange control here would starve system-wide distributed slack. \
            De-enrolling."
        return 0
    end
    return slack_ix
end

# Guard 5: `tied[i]` is whether candidate `candidate_names[i]` was referenced by >=1
# provisional tie. Factored out of `build_area_interchange_data` so it is directly
# unit-testable against a fabricated `tied` — a real "zero in-service ties" area is, by
# construction, topologically disconnected from the rest of the network, which the
# REF-per-island invariant (`_bus_island_map`) rejects before this guard would ever run.
function _warn_zero_tie_areas(candidate_names::Vector{String}, tied::BitVector)
    for (i, name) in enumerate(candidate_names)
        tied[i] && continue
        @warn "Area \"$name\": zero in-service ties to any other area; an embedded \
            net-interchange-controlled area must exchange power with the rest of the \
            network. De-enrolling."
    end
    return
end

# `AreaTie.diag_pollution` is a CONSTANT cached at tie-build time (aggregate Y-bus diagonal
# minus the sum of the corridor's OWN member primitives). It stays exact under a controlled
# tap ON the corridor itself (the tap's own delta cancels against the live diagonal read),
# but a device that is NOT a corridor member and mutates a tie-endpoint diagonal after
# enrollment silently makes the cached constant stale. Fenced limitation: warn-only, not
# de-enrollment.
_tie_touches(bus_ix::Int, tie::AreaTie) =
    bus_ix == tie.from_bus_ix || bus_ix == tie.to_bus_ix
_tie_corridor_member(fix::Int, tix::Int, tie::AreaTie) =
    (fix == tie.from_bus_ix && tix == tie.to_bus_ix) ||
    (fix == tie.to_bus_ix && tix == tie.from_bus_ix)

# Guard 7a: a voltage-controlling tap incident to a tie endpoint bus, but between a
# DIFFERENT bus pair than that tie's own corridor.
function _warn_tap_pollution_hazard(tx, fix::Int, tix::Int, tie::AreaTie)
    _tie_corridor_member(fix, tix, tie) && return
    (_tie_touches(fix, tie) || _tie_touches(tix, tie)) || return
    @warn "ControlledTap \"$(PSY.get_name(tx))\": incident to the tie between reduced-\
        network buses $(tie.from_bus_ix) and $(tie.to_bus_ix) without being a member of \
        its corridor; its tap movement mutates that tie endpoint's Y-bus diagonal, which \
        the cached `diag_pollution` correction cannot track live. Net-interchange \
        tracking for this tie assumes this tap holds its enrollment-time value."
    return
end

# Guard 7b: a switchable `PSY.SwitchedAdmittance` sitting at a tie endpoint bus (shunts are
# never corridor members — they attach at one bus, not a bus pair — so any incidence is a
# hazard). The current discrete-control path moves shunts through the constant-Z
# reactive-withdrawal slot, not the Y-bus, but the shunt's enrollment-time admittance IS
# stamped into the endpoint diagonal and folded into `diag_pollution` — so any path that
# restamps the Y-bus for it (future implicit embedding, external mutation) goes stale.
function _warn_shunt_pollution_hazard(sa, bix::Int, tie::AreaTie)
    _tie_touches(bix, tie) || return
    @warn "SwitchedAdmittance \"$(PSY.get_name(sa))\": sits at a tie endpoint (reduced-\
        network buses $(tie.from_bus_ix) and $(tie.to_bus_ix)); its stamped admittance is \
        folded into that tie's cached `diag_pollution` constant, which cannot track a \
        Y-bus restamp. Net-interchange tracking for this tie assumes this shunt holds \
        its enrollment-time value."
    return
end

# Component-level PSY scan (availability-filtered), independent of whether
# `build_controlled_device_set` has run yet or in what order — so this guard can't miss a
# hazard because discrete-control metadata doesn't exist yet at this point in construction.
function _warn_diag_pollution_hazards(
    sys::PSY.System,
    ties::Vector{AreaTie},
    bus_lookup::Dict{Int, Int},
    reverse_bus_search_map::Dict{Int, Int},
)
    isempty(ties) && return
    for tx in PSY.get_available_components(PSY.TapTransformer, sys)
        PSY.get_control_objective(tx) == PSY.TransformerControlObjective.VOLTAGE ||
            continue
        arc = PSY.get_arc(tx)
        fix = _resolve_bus_ix(
            bus_lookup, reverse_bus_search_map, PSY.get_number(PSY.get_from(arc)))
        tix = _resolve_bus_ix(
            bus_lookup, reverse_bus_search_map, PSY.get_number(PSY.get_to(arc)))
        (isnothing(fix) || isnothing(tix)) && continue
        for tie in ties
            _warn_tap_pollution_hazard(tx, fix, tix, tie)
        end
    end
    for sa in PSY.get_available_components(PSY.SwitchedAdmittance, sys)
        PSY.get_control_mode(sa) == PSY.SwitchedAdmittanceControlMode.FIXED && continue
        bix = _resolve_bus_ix(
            bus_lookup, reverse_bus_search_map, PSY.get_number(PSY.get_bus(sa)))
        isnothing(bix) && continue
        for tie in ties
            _warn_shunt_pollution_hazard(sa, bix, tie)
        end
    end
    return
end

# Rule 9 (runs last): per island, uncontrolled areas (no SLACK, or de-enrolled by any
# guard) whose derived PDES is nonzero. Exactly one names it via `@info` — met implicitly
# by tie-cancellation once the island's other areas enroll; two or more names all via
# `@warn` — only the combined net is pinned, individual schedules are unenforceable.
function _report_unenforceable_schedules(
    area_names::Vector{String},
    enrolled_set::Set{String},
    buses_by_area::Dict{String, Vector{PSY.ACBus}},
    pdes::Dict{String, Float64},
    bus_lookup::Dict{Int, Int},
    reverse_bus_search_map::Dict{Int, Int},
    island_of::Dict{Int, Int},
)
    island_groups = Dict{Int, Vector{String}}()
    for name in area_names
        name in enrolled_set && continue
        area_pdes = get(pdes, name, 0.0)
        iszero(area_pdes) && continue
        resolved =
            _resolved_bus_ixs(buses_by_area[name], bus_lookup, reverse_bus_search_map)
        for ix in resolved
            isle = get(island_of, ix, 0)
            push!(get!(island_groups, isle, String[]), name)
        end
    end
    for isle in sort!(collect(keys(island_groups)))
        names = sort!(unique(island_groups[isle]))
        if length(names) == 1
            @info "Area \"$(only(names))\" is not embedded net-interchange-controlled \
                but has a nonzero derived interchange target; its schedule is met \
                implicitly via the tie-cancellation identity (Σ NI_a = 0) once the \
                island's other areas enroll."
        else
            @warn "Areas $(join(names, ", ")) are not embedded net-interchange-controlled \
                but have nonzero derived interchange targets; only their combined net is \
                pinned by the tie-cancellation identity — the individual schedules are \
                unenforceable and deviations land on the island's REF/distributed slack. \
                Possibly a missing SLACK designation."
        end
    end
    return
end

"""
    build_area_interchange_data(pf, sys, data) -> AreaInterchangeData

Derive the enrolled `ControlledArea`/`AreaTie` set for embedded PSS/E-style
net-interchange control from `sys`, applying the enrollment guards in
order. Called once, at `PowerFlowData` construction, gated on
`get_area_interchange_control(pf)`; see `initialize_power_flow_data!` for how the result
populates `data.area_interchange`.
"""
function build_area_interchange_data(
    pf::AbstractACPowerFlow,
    sys::PSY.System,
    data::ACPowerFlowData,
)
    bus_lookup = get_bus_lookup(data)
    nrd = get_network_reduction_data(data)
    reverse_bus_search_map = PNM.get_reverse_bus_search_map(nrd)
    buses_by_area = _buses_by_area(sys)
    island_of = _bus_island_map(data)
    pdes, incident = _area_pdes(sys)

    area_names = sort!(collect(keys(buses_by_area)))
    slack_ix_of = Dict{String, Int}()
    for name in area_names
        slack_ix = _area_slack_candidate(
            name,
            buses_by_area[name],
            data,
            bus_lookup,
            reverse_bus_search_map,
            island_of,
        )
        iszero(slack_ix) && continue
        slack_ix_of[name] = slack_ix
    end

    # Guard 5 (zero in-service ties): needs the FULL cross-area tie set, so candidates are
    # first mapped to arbitrary distinct provisional ids (sorted-by-name order) purely to
    # detect tie incidence. The final `tail_ix` numbering (assigned below, after this guard
    # drops zero-tie areas) must stay contiguous 1..n over the SURVIVING set; rather than
    # re-scanning every branch a second time (and double-emitting any unresolvable-bus
    # warning `build_area_ties` logs along the way), the provisional ties' tail fields are
    # translated in place — a dropped candidate's tie can't exist (having >=1 tie is
    # exactly what keeps it from being dropped), so every translated tail stays consistent.
    ybus = get_power_network_matrix(data)
    candidate_names = sort!(collect(keys(slack_ix_of)))
    provisional_map = Dict{Int, Int}()
    for (i, name) in enumerate(candidate_names)
        for ix in _resolved_bus_ixs(buses_by_area[name], bus_lookup, reverse_bus_search_map)
            provisional_map[ix] = i
        end
    end
    (provisional_ties, provisional_dc_ties) = build_area_ties(
        sys, bus_lookup, ybus, nrd, provisional_map, data.lcc, get_dc_network(data))
    tied = falses(length(candidate_names))
    for tie in provisional_ties
        iszero(tie.from_area_tail) || (tied[tie.from_area_tail] = true)
        iszero(tie.to_area_tail) || (tied[tie.to_area_tail] = true)
    end
    for tie in provisional_dc_ties
        iszero(tie.from_area_tail) || (tied[tie.from_area_tail] = true)
        iszero(tie.to_area_tail) || (tied[tie.to_area_tail] = true)
    end
    _warn_zero_tie_areas(candidate_names, tied)
    enrolled_names = String[]
    final_tail = zeros(Int, length(candidate_names))
    for (i, name) in enumerate(candidate_names)
        tied[i] || continue
        push!(enrolled_names, name)
        final_tail[i] = length(enrolled_names)
    end
    function _translate_tail(old_tail::Int)
        iszero(old_tail) && return 0
        return final_tail[old_tail]
    end
    ties = [
        AreaTie(
            tie.from_bus_ix,
            tie.to_bus_ix,
            tie.nz_offsets,
            tie.metered_from,
            _translate_tail(tie.from_area_tail),
            _translate_tail(tie.to_area_tail),
            tie.diag_pollution,
        ) for tie in provisional_ties
    ]
    dc_ties = [
        DCTie(
            tie.kind,
            tie.lcc_ix,
            tie.from_conv_ix,
            tie.to_conv_ix,
            tie.from_bus_ix,
            tie.to_bus_ix,
            tie.metered_from,
            _translate_tail(tie.from_area_tail),
            _translate_tail(tie.to_area_tail),
        ) for tie in provisional_dc_ties
    ]
    _warn_diag_pollution_hazards(sys, ties, bus_lookup, reverse_bus_search_map)

    areas = ControlledArea[]
    for (tail_ix, name) in enumerate(enrolled_names)
        area_pdes = get(pdes, name, 0.0)
        if iszero(get(incident, name, 0))
            @info "Area \"$name\": no incident PSY.AreaInterchange records; \
                net-interchange target defaults to 0.0 pu."
            area_pdes = 0.0
        end
        push!(areas, ControlledArea(name, slack_ix_of[name], area_pdes, tail_ix))
    end

    # Guard 8: sanity assertion, not a user-facing warning — every AreaInterchange record
    # contributes +flow to one area and -flow to another, so the raw pairwise-derived sum
    # over ALL areas is zero by construction regardless of which areas end up enrolled.
    @assert isapprox(sum(values(pdes); init = 0.0), 0.0; atol = 1e-9) "Raw pairwise-\
        derived per-area PDES must sum to ~0 across ALL areas by construction."

    enrolled_set = Set(enrolled_names)
    _report_unenforceable_schedules(
        area_names,
        enrolled_set,
        buses_by_area,
        pdes,
        bus_lookup,
        reverse_bus_search_map,
        island_of,
    )

    n_time_steps = size(data.bus_type, 2)
    return AreaInterchangeData(
        areas,
        ties,
        dc_ties,
        get_interchange_tolerance(pf),
        zeros(length(areas)),
        zeros(length(areas), n_time_steps),
        copy(areas),
        copy(ties),
        copy(dc_ties),
        zeros(length(areas), n_time_steps),
        Dict{Int, Vector{RelaxedAreaRecord}}(),
    )
end
