# Shared context for the tie-enumeration tests: PowerFlowData artifacts plus the
# bus-index -> area-tail map that Task 5's enrollment will eventually build.
function _tie_test_context(sys::PSY.System, area_tail::Dict{String, Int})
    data = PowerFlowData(ACPowerFlow(), sys)
    bus_lookup = PF.get_bus_lookup(data)
    ybus = PF.get_power_network_matrix(data)
    nrd = PF.get_network_reduction_data(data)
    bus_area_map = Dict{Int, Int}()
    for bus in PSY.get_components(PSY.ACBus, sys)
        area = PSY.get_area(bus)
        isnothing(area) && continue
        tail = get(area_tail, PSY.get_name(area), 0)
        if !iszero(tail)
            bus_area_map[bus_lookup[PSY.get_number(bus)]] = tail
        end
    end
    return (
        bus_lookup = bus_lookup,
        ybus = ybus,
        nrd = nrd,
        bus_area_map = bus_area_map,
    )
end

function _find_tie(ties::Vector{PF.AreaTie}, fix::Int, tix::Int)
    return only(
        filter(
            tie ->
                (tie.from_bus_ix == fix && tie.to_bus_ix == tix) ||
                    (tie.from_bus_ix == tix && tie.to_bus_ix == fix),
            ties,
        ),
    )
end

@testset "area interchange tie enumeration" begin
    sys = _make_two_area_system()
    ctx = _tie_test_context(sys, Dict("Area1" => 1, "Area2" => 2))

    ties = PF.build_area_ties(sys, ctx.bus_lookup, ctx.ybus, ctx.nrd, ctx.bus_area_map)

    # Boundary crosses exactly 3 branches: Trans1 (4-9), Trans2 (5-6), Trans3 (4-7).
    @test length(ties) == 3

    for tie in ties
        @test ctx.bus_area_map[tie.from_bus_ix] != ctx.bus_area_map[tie.to_bus_ix]
    end

    # Metered end defaults to "from" for all three (no ext set).
    for tie in ties
        @test tie.metered_from == true
    end

    # nz_offsets point at the correct Y-bus entries.
    A = ctx.ybus.data
    for tie in ties
        f, t = tie.from_bus_ix, tie.to_bus_ix
        o = tie.nz_offsets
        @test A[f, f] == A.nzval[o[1]]
        @test A[f, t] == A.nzval[o[2]]
        @test A[t, f] == A.nzval[o[3]]
        @test A[t, t] == A.nzval[o[4]]
    end
end

@testset "area interchange tie metered end ext flip" begin
    sys = _make_two_area_system()
    trans1 = PSY.get_component(PSY.TapTransformer, sys, "Trans1")
    PSY.get_ext(trans1)["metered_end"] = "to"

    ctx = _tie_test_context(sys, Dict("Area1" => 1, "Area2" => 2))
    ties = PF.build_area_ties(sys, ctx.bus_lookup, ctx.ybus, ctx.nrd, ctx.bus_area_map)
    @test length(ties) == 3

    trans1_tie = _find_tie(ties, ctx.bus_lookup[4], ctx.bus_lookup[9])
    @test trans1_tie.metered_from == false

    other_ties = filter(tie -> tie !== trans1_tie, ties)
    @test length(other_ties) == 2
    for tie in other_ties
        @test tie.metered_from == true
    end
end

@testset "area interchange tie out-of-service exclusion" begin
    sys = _make_two_area_system()
    trans2 = PSY.get_component(PSY.TapTransformer, sys, "Trans2")
    PSY.set_available!(trans2, false)

    ctx = _tie_test_context(sys, Dict("Area1" => 1, "Area2" => 2))
    ties = PF.build_area_ties(sys, ctx.bus_lookup, ctx.ybus, ctx.nrd, ctx.bus_area_map)
    # Trans2 (5-6) excluded: only Trans1 (4-9) and Trans3 (4-7) remain.
    @test length(ties) == 2

    trans2_fix = ctx.bus_lookup[5]
    trans2_tix = ctx.bus_lookup[6]
    for tie in ties
        excluded_pair =
            (tie.from_bus_ix == trans2_fix && tie.to_bus_ix == trans2_tix) ||
            (tie.from_bus_ix == trans2_tix && tie.to_bus_ix == trans2_fix)
        @test !excluded_pair
    end
end

@testset "area interchange tie three-area tails" begin
    sys = _make_three_area_system()
    ctx = _tie_test_context(sys, Dict("Area1" => 1, "Area2" => 2, "Area3" => 3))
    ties = PF.build_area_ties(sys, ctx.bus_lookup, ctx.ybus, ctx.nrd, ctx.bus_area_map)

    # Boundary branches: Trans1 (4-9), Trans2 (5-6), Trans3 (4-7), Line11 (9-10),
    # Line12 (9-14), Line16 (7-9).
    @test length(ties) == 6

    # Tail fields carry the owning areas' tail slots, matched to arc orientation.
    expected = Dict(
        (4, 9) => (1, 3),   # Trans1: Area1 -> Area3
        (5, 6) => (1, 2),   # Trans2: Area1 -> Area2
        (4, 7) => (1, 2),   # Trans3: Area1 -> Area2
        (9, 10) => (3, 2),  # Line11: Area3 -> Area2
        (9, 14) => (3, 2),  # Line12: Area3 -> Area2
        (7, 9) => (2, 3),   # Line16: Area2 -> Area3
    )
    for ((fb, tb), (tail_f, tail_t)) in expected
        tie = _find_tie(ties, ctx.bus_lookup[fb], ctx.bus_lookup[tb])
        if tie.from_bus_ix == ctx.bus_lookup[fb]
            @test tie.from_area_tail == tail_f
            @test tie.to_area_tail == tail_t
        else
            @test tie.from_area_tail == tail_t
            @test tie.to_area_tail == tail_f
        end
    end
end

@testset "area interchange tie partial enrollment" begin
    sys = _make_three_area_system()
    # Only Area2 enrolled: the Area1/Area3 boundary (Trans1, 4-9) is between two
    # uncontrolled areas and must not be stored; every tie touching Area2 is kept
    # with tail 0 on the uncontrolled side.
    ctx = _tie_test_context(sys, Dict("Area2" => 1))
    ties = PF.build_area_ties(sys, ctx.bus_lookup, ctx.ybus, ctx.nrd, ctx.bus_area_map)

    @test length(ties) == 5

    trans1_pair = (ctx.bus_lookup[4], ctx.bus_lookup[9])
    for tie in ties
        not_trans1 =
            (tie.from_bus_ix, tie.to_bus_ix) != trans1_pair &&
            (tie.to_bus_ix, tie.from_bus_ix) != trans1_pair
        @test not_trans1
        @test iszero(tie.from_area_tail) || iszero(tie.to_area_tail)
        @test tie.from_area_tail == 1 || tie.to_area_tail == 1
    end
end

@testset "area interchange tie uncontrolled-uncontrolled not stored" begin
    sys = _make_two_area_system()
    ctx = _tie_test_context(sys, Dict{String, Int}())

    # No enrolled areas -> every bus resolves to tail 0 (uncontrolled) via the
    # `get(..., 0)` default, so no ties are stored despite a real area boundary.
    ties = PF.build_area_ties(sys, ctx.bus_lookup, ctx.ybus, ctx.nrd, ctx.bus_area_map)
    @test isempty(ties)
end

@testset "area interchange tie three-winding transformer boundary" begin
    # `case10_radial_series_reductions` (PSB) already carries a real Transformer3W
    # (HV=101 -> star=1001, LV=103 -> star, MV=102 -> star). Re-area its terminals so
    # the primary winding is interior (terminal + star share AreaA, mirroring PSS/E's
    # own star-bus-inherits-primary's-area convention) and the secondary/tertiary
    # windings straddle the AreaA/AreaB boundary through the star node.
    sys = PSB.build_system(PSB.PSITestSystems, "case10_radial_series_reductions")
    trf = PSY.get_component(PSY.ThreeWindingTransformer, sys, "HV-LV-MV-i_1")
    star_bus = PSY.get_star_bus(trf)
    primary_bus = PSY.get_from(PSY.get_primary_star_arc(trf))
    secondary_bus = PSY.get_from(PSY.get_secondary_star_arc(trf))
    tertiary_bus = PSY.get_from(PSY.get_tertiary_star_arc(trf))

    areaA = PSY.Area(; name = "AreaA")
    areaB = PSY.Area(; name = "AreaB")
    PSY.add_component!(sys, areaA)
    PSY.add_component!(sys, areaB)
    PSY.set_area!(primary_bus, areaA)
    PSY.set_area!(star_bus, areaA)
    PSY.set_area!(secondary_bus, areaB)
    PSY.set_area!(tertiary_bus, areaB)

    ctx = _tie_test_context(sys, Dict("AreaA" => 1, "AreaB" => 2))
    ties = PF.build_area_ties(sys, ctx.bus_lookup, ctx.ybus, ctx.nrd, ctx.bus_area_map)

    star_ix = ctx.bus_lookup[PSY.get_number(star_bus)]
    primary_ix = ctx.bus_lookup[PSY.get_number(primary_bus)]
    secondary_ix = ctx.bus_lookup[PSY.get_number(secondary_bus)]
    tertiary_ix = ctx.bus_lookup[PSY.get_number(tertiary_bus)]

    # The primary winding (terminal + star both AreaA) is interior: no tie.
    @test_throws ArgumentError _find_tie(ties, primary_ix, star_ix)

    # The secondary and tertiary windings straddle AreaA/AreaB through the star bus.
    sec_tie = _find_tie(ties, secondary_ix, star_ix)
    tert_tie = _find_tie(ties, tertiary_ix, star_ix)
    A = ctx.ybus.data
    for tie in (sec_tie, tert_tie)
        @test ctx.bus_area_map[tie.from_bus_ix] != ctx.bus_area_map[tie.to_bus_ix]
        f, t = tie.from_bus_ix, tie.to_bus_ix
        o = tie.nz_offsets
        @test A[f, f] == A.nzval[o[1]]
        @test A[f, t] == A.nzval[o[2]]
        @test A[t, f] == A.nzval[o[3]]
        @test A[t, t] == A.nzval[o[4]]
    end
end

@testset "area interchange tie discrete-controlled branch status" begin
    # Mirror PNM's exact gate (`YbusACBranches.jl`): CLOSED contributes a tie candidate,
    # OPEN does not — checked directly on `PF._tie_in_service` since constructing a full
    # system with an open switch at a boundary (below) only exercises the CLOSED path
    # implicitly (an open switch, correctly, produces no observable tie either way).
    sys = _make_two_area_system()
    bus1 = PSY.get_component(PSY.ACBus, sys, "Bus 1")
    bus6 = PSY.get_component(PSY.ACBus, sys, "Bus 6")
    arc = PSY.Arc(; from = bus1, to = bus6)
    closed_sw = PSY.DiscreteControlledACBranch(;
        name = "sw_closed",
        available = true,
        active_power_flow = 0.0,
        reactive_power_flow = 0.0,
        arc = arc,
        r = 0.001,
        x = 0.01,
        rating = 1.0,
        discrete_branch_type = PSY.DiscreteControlledBranchType.BREAKER,
        branch_status = PSY.DiscreteControlledBranchStatus.CLOSED,
    )
    open_sw = PSY.DiscreteControlledACBranch(;
        name = "sw_open",
        available = true,
        active_power_flow = 0.0,
        reactive_power_flow = 0.0,
        arc = arc,
        r = 0.001,
        x = 0.01,
        rating = 1.0,
        discrete_branch_type = PSY.DiscreteControlledBranchType.BREAKER,
        branch_status = PSY.DiscreteControlledBranchStatus.OPEN,
    )
    @test PF._tie_in_service(closed_sw) == true
    @test PF._tie_in_service(open_sw) == false
end

@testset "area interchange tie discrete-controlled open switch at boundary" begin
    sys = _make_two_area_system()
    bus1 = PSY.get_component(PSY.ACBus, sys, "Bus 1")   # Area1
    bus6 = PSY.get_component(PSY.ACBus, sys, "Bus 6")   # Area2
    arc = PSY.Arc(; from = bus1, to = bus6)
    PSY.add_component!(sys, arc)
    sw = PSY.DiscreteControlledACBranch(;
        name = "sw_open_boundary",
        available = true,
        active_power_flow = 0.0,
        reactive_power_flow = 0.0,
        arc = arc,
        r = 0.001,
        x = 0.01,
        rating = 1.0,
        discrete_branch_type = PSY.DiscreteControlledBranchType.BREAKER,
        branch_status = PSY.DiscreteControlledBranchStatus.OPEN,
    )
    PSY.add_component!(sys, sw)

    ctx = _tie_test_context(sys, Dict("Area1" => 1, "Area2" => 2))
    ties = PF.build_area_ties(sys, ctx.bus_lookup, ctx.ybus, ctx.nrd, ctx.bus_area_map)

    # The 3 original boundary branches (Trans1, Trans2, Trans3) are untouched; the open
    # switch contributes no tie (and, critically, does not crash `_ybus_block_offsets`).
    @test length(ties) == 3
    bus1_ix = ctx.bus_lookup[1]
    bus6_ix = ctx.bus_lookup[6]
    @test_throws ArgumentError _find_tie(ties, bus1_ix, bus6_ix)
end

@testset "area interchange tie unrecognized metered_end value warns and defaults from" begin
    sys = _make_two_area_system()
    trans1 = PSY.get_component(PSY.TapTransformer, sys, "Trans1")
    PSY.get_ext(trans1)["metered_end"] = "From"   # wrong case: unrecognized, not silently coerced

    ctx = _tie_test_context(sys, Dict("Area1" => 1, "Area2" => 2))
    ties =
        @test_logs (:warn, r"metered_end.*neither.*from.*nor.*to") match_mode = :any PF.build_area_ties(
            sys,
            ctx.bus_lookup,
            ctx.ybus,
            ctx.nrd,
            ctx.bus_area_map,
        )
    @test length(ties) == 3
    trans1_tie = _find_tie(ties, ctx.bus_lookup[4], ctx.bus_lookup[9])
    @test trans1_tie.metered_from == true
end

@testset "area interchange tie parallel boundary branches deduplicated" begin
    sys = _make_two_area_system()
    trans2 = PSY.get_component(PSY.TapTransformer, sys, "Trans2")   # 5 -> 6, Area1/Area2
    arc = PSY.get_arc(trans2)
    # A second transformer (not a `Line`: Trans2's terminals differ enough in nominal
    # voltage that PSY rejects a `Line` on that arc) in parallel on the SAME `Arc`.
    parallel_tx = PSY.TapTransformer(;
        name = "Trans2_parallel",
        available = true,
        active_power_flow = 0.0,
        reactive_power_flow = 0.0,
        arc = arc,
        r = 0.01,
        x = 0.10,
        primary_shunt = 0.0 + 0.0im,
        tap = 1.0,
        rating = 1.0,
        base_power = 100.0,
        control_objective = PSY.TransformerControlObjective.UNDEFINED,
    )
    PSY.add_component!(sys, parallel_tx)

    ctx = _tie_test_context(sys, Dict("Area1" => 1, "Area2" => 2))
    ties = PF.build_area_ties(sys, ctx.bus_lookup, ctx.ybus, ctx.nrd, ctx.bus_area_map)

    # Still exactly 3 boundary corridors: the parallel line does not add a 4th tie for
    # the (5,6) pair already covered by Trans2.
    @test length(ties) == 3
    tie = _find_tie(ties, ctx.bus_lookup[5], ctx.bus_lookup[6])
    A = ctx.ybus.data
    f, t = tie.from_bus_ix, tie.to_bus_ix
    o = tie.nz_offsets
    # The Y-bus block is the AGGREGATE of Trans2 + the parallel line -- one tie whose
    # offsets point at the combined admittance, not two ties double-counting the flow.
    @test A[f, f] == A.nzval[o[1]]
    @test A[f, t] == A.nzval[o[2]]
    @test A[t, f] == A.nzval[o[3]]
    @test A[t, t] == A.nzval[o[4]]
end

# ── Task 5: enrollment guards ───────────────────────────────────────────────────────────

_set_slack!(sys, bus_name) =
    PSY.set_bustype!(PSY.get_component(PSY.ACBus, sys, bus_name), PSY.ACBusTypes.SLACK)

function _add_area_interchange!(
    sys,
    from_name::String,
    to_name::String,
    flow::Float64;
    name::String = "$(from_name)_$(to_name)",
)
    PSY.add_component!(
        sys,
        PSY.AreaInterchange(;
            name = name,
            available = true,
            active_power_flow = flow,
            from_area = PSY.get_component(PSY.Area, sys, from_name),
            to_area = PSY.get_component(PSY.Area, sys, to_name),
            flow_limits = (from_to = 0.0, to_from = 0.0),
        ),
    )
    return
end

# Three-area fixture shared by the rule-9 (unenforceable-schedule) and happy-path tests.
# Area1 (Bus 1-5) owns REF (Bus 1) and never gets a SLACK bus. Area2 (Bus 6) and Area3
# (Bus 9, given a small generator so it is P-capable and PV-eligible) can each optionally
# hold SLACK. Two AreaInterchange records make every area transfer-bearing: Area2->Area1
# 0.3, Area3->Area1 0.2, so pdes(Area1) = -0.5, pdes(Area2) = 0.3, pdes(Area3) = 0.2 (pu).
function _three_area_transfer_fixture(; slack_area3::Bool = true)
    sys = _make_three_area_system()
    bus9 = PSY.get_component(PSY.ACBus, sys, "Bus 9")
    gen9 = PSY.ThermalStandard(;
        name = "Bus9Gen",
        available = true,
        status = true,
        bus = bus9,
        active_power = 0.1,
        reactive_power = 0.0,
        rating = 1.0,
        active_power_limits = (min = 0.0, max = 1.0),
        reactive_power_limits = (min = -1.0, max = 1.0),
        ramp_limits = nothing,
        operation_cost = PSY.ThermalGenerationCost(nothing),
        base_power = 100.0,
    )
    PSY.add_component!(sys, gen9)
    _set_slack!(sys, "Bus 6")
    slack_area3 && _set_slack!(sys, "Bus 9")
    _add_area_interchange!(sys, "Area2", "Area1", 0.3; name = "A2_A1")
    _add_area_interchange!(sys, "Area3", "Area1", 0.2; name = "A3_A1")
    return sys
end

@testset "area interchange enrollment rule 1 multiple SLACK buses" begin
    sys = _make_two_area_system()
    _set_slack!(sys, "Bus 2")
    _set_slack!(sys, "Bus 3")
    pf = ACPolarPowerFlow(; area_interchange_control = true)
    data = PowerFlowData(pf, sys)
    aid = @test_logs(
        (:warn, r"Area \"Area1\": 2 SLACK buses"),
        min_level = Logging.Warn,
        PF.build_area_interchange_data(pf, sys, data)
    )
    @test PF.n_controlled_areas(aid) == 0
end

@testset "area interchange enrollment rule 1b SLACK demoted to PQ" begin
    sys = _make_two_area_system()
    _set_slack!(sys, "Bus 2")
    pf = ACPolarPowerFlow(; area_interchange_control = true)
    data = PowerFlowData(pf, sys)
    ix = PF.get_bus_lookup(data)[2]
    @test data.bus_type[ix, 1] == PSY.ACBusTypes.PV
    # A real end-to-end PQ-demotion fixture needs an InterconnectingConverter/HybridSystem
    # bus (see test_area_interchange_types.jl's "(c)" sub-case); poke the post-construction
    # state directly instead -- exactly the precondition the guard checks ("at construction,
    # pre-solve, before any Q-limit flip can occur").
    data.bus_type[ix, 1] = PSY.ACBusTypes.PQ
    aid = @test_logs(
        (:warn, r"Area \"Area1\": SLACK bus Bus 2 has no in-service voltage-regulating"),
        min_level = Logging.Warn,
        PF.build_area_interchange_data(pf, sys, data)
    )
    @test PF.n_controlled_areas(aid) == 0
end

@testset "area interchange enrollment rule 2 SLACK bus unresolvable" begin
    sys = _make_two_area_system()
    _set_slack!(sys, "Bus 8")  # leaf bus: only branch is Trans4 (7-8)
    pf = ACPolarPowerFlow(; area_interchange_control = true)
    data = PowerFlowData(pf, sys)
    # No reduction in this fixture actually drops Bus 8; simulate "unresolvable" directly
    # by removing its bus_lookup entry (mirrors rule 1b's direct-state-poke technique).
    delete!(PF.get_bus_lookup(data), 8)
    aid = @test_logs(
        (:warn, r"Area \"Area2\": SLACK bus Bus 8 was removed by network reduction"),
        (:warn, r"Trans4.*bus 8 is not in the \(reduced\) network"),
        min_level = Logging.Warn,
        PF.build_area_interchange_data(pf, sys, data)
    )
    @test PF.n_controlled_areas(aid) == 0
end

@testset "area interchange enrollment rule 3 area contains REF" begin
    sys = _make_two_area_system()
    _set_slack!(sys, "Bus 2")  # Area1, which also owns REF (Bus 1)
    pf = ACPolarPowerFlow(; area_interchange_control = true)
    data = PowerFlowData(pf, sys)
    aid = @test_logs(
        (:warn, r"Area \"Area1\": contains the network reference bus Bus 1"),
        min_level = Logging.Warn,
        PF.build_area_interchange_data(pf, sys, data)
    )
    @test PF.n_controlled_areas(aid) == 0
end

@testset "area interchange enrollment rule 4 area spans multiple islands" begin
    # A real "spans 2 islands" area can't be built end-to-end: PowerFlows requires exactly
    # one REF bus per electrical island (`_find_subnetworks_for_reference_buses`), so a
    # genuinely disconnected second island needs its own REF -- which, owned by the SAME
    # area, would trip guard 3 (REF-containment) first, and owned by a DIFFERENT area,
    # any available branch connecting the two islands is by definition a tie (not a
    # disconnection). `_area_slack_candidate` takes `island_of` as an explicit argument for
    # exactly this reason: unit-test it directly against a fabricated two-island map.
    sys = _make_two_area_system()
    _set_slack!(sys, "Bus 6")  # Area2, no REF
    pf = ACPolarPowerFlow(; area_interchange_control = true)
    data = PowerFlowData(pf, sys)
    bus_lookup = PF.get_bus_lookup(data)
    nrd = PF.get_network_reduction_data(data)
    rbsm = PNM.get_reverse_bus_search_map(nrd)
    area2_buses = collect(
        PSY.get_components(
            b -> PSY.get_name(PSY.get_area(b)) == "Area2",
            PSY.ACBus,
            sys,
        ),
    )
    fake_island_of = Dict{Int, Int}(bus_lookup[6] => 1)
    for bus in area2_buses
        PSY.get_number(bus) == 6 && continue
        fake_island_of[bus_lookup[PSY.get_number(bus)]] = 2
    end
    result = @test_logs(
        (:warn, r"Area \"Area2\": buses span 2 electrical islands"),
        min_level = Logging.Warn,
        PF._area_slack_candidate(
            "Area2",
            area2_buses,
            data,
            bus_lookup,
            rbsm,
            fake_island_of,
        )
    )
    @test iszero(result)
end

@testset "area interchange enrollment rule 4 area spans multiple islands (end-to-end)" begin
    # Real fixture exercised through the PRODUCTION path (`_bus_island_map` ->
    # `_find_subnetworks_for_reference_buses`), not a fabricated `island_of` dict: two
    # genuinely disconnected islands, each with its own REF bus and its own genuine
    # intra-island tie to area "Span" -- see `_make_two_island_spanning_area_system`.
    sys = _make_two_island_spanning_area_system()
    pf = ACPolarPowerFlow(; area_interchange_control = true)
    # PNM's own `Ybus`/connectivity-check machinery also warns (more than once, with its
    # own wording) that the raw system is genuinely disconnected -- expected, that's the
    # whole point of this fixture -- so match_mode = :any (existing precedent: rule 9's
    # tests) only asserts enrollment's own guard-4 warning is present, not the full,
    # upstream-log-order-dependent sequence.
    data = @test_logs(
        (:warn, r"span.*islands?"),
        min_level = Logging.Warn,
        match_mode = :any,
        PowerFlowData(pf, sys)
    )
    @test PF.n_controlled_areas(data) == 0
end

@testset "area interchange enrollment rule 5 zero in-service ties" begin
    # See rule 4's comment: a real zero-tie area is topologically disconnected, which the
    # REF-per-island invariant rejects before this guard would ever run. Unit-test the
    # factored-out warn-and-drop helper directly against a fabricated `tied` vector.
    @test_logs(
        (:warn, r"Area \"Area1\": zero in-service ties"),
        min_level = Logging.Warn,
        PF._warn_zero_tie_areas(["Area1", "Area2"], BitVector([false, true]))
    )
end

@testset "area interchange enrollment rule 6 slack absorption limit" begin
    sys = _make_two_area_system()
    _set_slack!(sys, "Bus 6")  # Area2, no REF
    pf = ACPolarPowerFlow(;
        area_interchange_control = true,
        generator_slack_participation_factors = Dict(
            (PSY.ThermalStandard, "Bus6") => 1.0,
        ),
    )
    data = PowerFlowData(pf, sys)
    aid = @test_logs(
        (:warn, r"Area \"Area2\": area buses hold a slack-participation weight of 1\.0"),
        min_level = Logging.Warn,
        PF.build_area_interchange_data(pf, sys, data)
    )
    @test PF.n_controlled_areas(aid) == 0
end

@testset "area interchange enrollment PDES aggregation" begin
    sys = _make_three_area_system()
    _add_area_interchange!(sys, "Area1", "Area2", 0.5)
    _add_area_interchange!(sys, "Area3", "Area1", 0.2; name = "A3_A1")
    pdes, incident = PF._area_pdes(sys)
    @test pdes["Area1"] ≈ 0.3
    @test pdes["Area2"] ≈ -0.5
    @test pdes["Area3"] ≈ 0.2
    @test incident["Area1"] == 2
    @test incident["Area2"] == 1
    @test incident["Area3"] == 1
end

@testset "area interchange enrollment rule 9 single uncontrolled area (info)" begin
    sys = _three_area_transfer_fixture(; slack_area3 = true)
    pf = ACPolarPowerFlow(; area_interchange_control = true)
    data = PowerFlowData(pf, sys)
    aid = @test_logs(
        (:info, r"Area \"Area1\" is not embedded net-interchange-controlled"),
        match_mode = :any,
        PF.build_area_interchange_data(pf, sys, data)
    )
    @test PF.n_controlled_areas(aid) == 2
    @test sort([a.name for a in aid.areas]) == ["Area2", "Area3"]
end

@testset "area interchange enrollment rule 9 two uncontrolled areas (warn)" begin
    sys = _three_area_transfer_fixture(; slack_area3 = false)
    pf = ACPolarPowerFlow(; area_interchange_control = true)
    data = PowerFlowData(pf, sys)
    aid = @test_logs(
        (:warn, r"Areas Area1, Area3 are not embedded net-interchange-controlled"),
        match_mode = :any,
        PF.build_area_interchange_data(pf, sys, data)
    )
    @test PF.n_controlled_areas(aid) == 1
    @test only(aid.areas).name == "Area2"
end

@testset "area interchange enrollment rule 9 passive area stays silent" begin
    sys = _make_two_area_system()
    _set_slack!(sys, "Bus 6")
    _add_area_interchange!(sys, "Area1", "Area2", 0.0)  # keeps guard 7 + rule 9 silent
    pf = ACPolarPowerFlow(; area_interchange_control = true)
    data = PowerFlowData(pf, sys)
    aid = @test_logs min_level = Logging.Warn PF.build_area_interchange_data(pf, sys, data)
    @test PF.n_controlled_areas(aid) == 1
end

@testset "area interchange enrollment happy path: REF area de-enrolled, two enrolled" begin
    # Area1 holds REF and never gets SLACK -> uncontrolled (rule 1, silent) and reported by
    # rule 9's @info (see above). Area2 and Area3 both enroll: tail_ix 1,2 in sorted-name
    # order, pdes(Area2)=0.3, pdes(Area3)=0.2 (hand-derived from the fixture's
    # AreaInterchange records: Area2->Area1 0.3, Area3->Area1 0.2).
    sys = _three_area_transfer_fixture(; slack_area3 = true)
    pf = ACPolarPowerFlow(; area_interchange_control = true)
    data = PowerFlowData(pf, sys)
    aid =
        @test_logs (:info, r"Area \"Area1\"") match_mode = :any PF.build_area_interchange_data(
            pf,
            sys,
            data,
        )
    @test PF.n_controlled_areas(aid) == 2
    area2 = only(filter(a -> a.name == "Area2", aid.areas))
    area3 = only(filter(a -> a.name == "Area3", aid.areas))
    @test area2.tail_ix == 1
    @test area3.tail_ix == 2
    @test area2.pdes ≈ 0.3
    @test area3.pdes ≈ 0.2
    @test !isempty(aid.ties)
    for tie in aid.ties
        @test tie.from_area_tail in (0, 1, 2)
        @test tie.to_area_tail in (0, 1, 2)
    end
end

@testset "area interchange enrollment wiring populates data.area_interchange" begin
    sys = _make_two_area_system()
    _set_slack!(sys, "Bus 6")
    pf_off = ACPolarPowerFlow(; area_interchange_control = false)
    data_off = PowerFlowData(pf_off, sys)
    @test PF.n_controlled_areas(data_off) == 0

    pf_on = ACPolarPowerFlow(; area_interchange_control = true)
    data_on = PowerFlowData(pf_on, sys)
    @test PF.n_controlled_areas(data_on) == 1
    @test only(PF.get_area_interchange_data(data_on).areas).name == "Area2"
end
