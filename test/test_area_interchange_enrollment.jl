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

    (ties, _dc_ties) = PF.build_area_ties(
        sys, ctx.bus_lookup, ctx.ybus, ctx.nrd, ctx.bus_area_map)

    @test length(ties) == 3

    for tie in ties
        @test ctx.bus_area_map[tie.from_bus_ix] != ctx.bus_area_map[tie.to_bus_ix]
    end

    for tie in ties
        @test tie.metered_from == true
    end

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
    (ties, _dc_ties) = PF.build_area_ties(
        sys, ctx.bus_lookup, ctx.ybus, ctx.nrd, ctx.bus_area_map)
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
    (ties, _dc_ties) = PF.build_area_ties(
        sys, ctx.bus_lookup, ctx.ybus, ctx.nrd, ctx.bus_area_map)
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
    (ties, _dc_ties) = PF.build_area_ties(
        sys, ctx.bus_lookup, ctx.ybus, ctx.nrd, ctx.bus_area_map)

    @test length(ties) == 6

    # Tail fields carry the owning areas' tail slots, matched to arc orientation.
    expected = Dict(
        (4, 9) => (1, 3),
        (5, 6) => (1, 2),
        (4, 7) => (1, 2),
        (9, 10) => (3, 2),
        (9, 14) => (3, 2),
        (7, 9) => (2, 3),
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
    (ties, _dc_ties) = PF.build_area_ties(
        sys, ctx.bus_lookup, ctx.ybus, ctx.nrd, ctx.bus_area_map)

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
    (ties, _dc_ties) = PF.build_area_ties(
        sys, ctx.bus_lookup, ctx.ybus, ctx.nrd, ctx.bus_area_map)
    @test isempty(ties)
end

@testset "area interchange tie three-winding transformer boundary" begin
    # `case10_radial_series_reductions` (PSB) has a real Transformer3W; re-area so the
    # primary winding stays interior (terminal + star share AreaA, per PSS/E's star-bus
    # convention) while secondary/tertiary straddle the AreaA/AreaB boundary.
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
    (ties, _dc_ties) = PF.build_area_ties(
        sys, ctx.bus_lookup, ctx.ybus, ctx.nrd, ctx.bus_area_map)

    star_ix = ctx.bus_lookup[PSY.get_number(star_bus)]
    primary_ix = ctx.bus_lookup[PSY.get_number(primary_bus)]
    secondary_ix = ctx.bus_lookup[PSY.get_number(secondary_bus)]
    tertiary_ix = ctx.bus_lookup[PSY.get_number(tertiary_bus)]

    @test_throws ArgumentError _find_tie(ties, primary_ix, star_ix)

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
    # Mirrors PNM's gate (`YbusACBranches.jl`): CLOSED contributes a tie, OPEN doesn't.
    # Checked directly on `_tie_in_service` since a full system with an open boundary
    # switch can't distinguish "no tie" from "path not exercised".
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
    bus1 = PSY.get_component(PSY.ACBus, sys, "Bus 1")
    bus6 = PSY.get_component(PSY.ACBus, sys, "Bus 6")
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
    (ties, _dc_ties) = PF.build_area_ties(
        sys, ctx.bus_lookup, ctx.ybus, ctx.nrd, ctx.bus_area_map)

    # Critically, the open switch contributes no tie and does not crash
    # `_ybus_block_offsets`.
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
    (ties, _dc_ties) =
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
    trans2 = PSY.get_component(PSY.TapTransformer, sys, "Trans2")
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
    (ties, _dc_ties) = PF.build_area_ties(
        sys, ctx.bus_lookup, ctx.ybus, ctx.nrd, ctx.bus_area_map)

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
# Shared by rule-9 and happy-path tests. Area1 owns REF, never SLACK; Area2/Area3 can each
# optionally hold SLACK (Area3's Bus 9 has a small gen so it's PV-eligible). AreaInterchange:
# Area2->Area1 0.3, Area3->Area1 0.2 => pdes(Area1)=-0.5, pdes(Area2)=0.3, pdes(Area3)=0.2.
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
    # A real PQ-demotion fixture needs an InterconnectingConverter/HybridSystem bus (see
    # test_area_interchange_types.jl's "(c)"); poke post-construction state directly instead
    # -- exactly the precondition the guard checks (pre-solve, before any Q-limit flip).
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
    # A real "spans 2 islands" area can't be built end-to-end: a same-area second REF trips
    # guard 3 first, and a different-area REF makes the connecting branch a tie, not a
    # disconnection. `_area_slack_candidate` takes `island_of` as an explicit arg for this
    # reason.
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
    sys = _make_two_island_spanning_area_system()
    pf = ACPolarPowerFlow(; area_interchange_control = true)
    # PNM's own Ybus/connectivity checks also warn (different wording) that the raw system
    # is disconnected -- expected, that's the point. match_mode = :any only asserts
    # enrollment's own guard-4 warning is present.
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

"""An LCC whose inverter bus ("Area3") has NO AC branch at all -- only the LCC touches it.
Proves DC-tie-only enumeration is real (not fabricated): `build_area_ties` on this genuinely
AC-disconnected system returns an empty `ac_ties` and a `dc_ties` that touches Area3, driven
entirely by a real `PowerFlowData`/`LCCParameters`/reduced-network build."""
function _dc_tie_only_fixture()
    sys = System(100.0)
    area1 = PSY.Area(; name = "Area1")
    area3 = PSY.Area(; name = "Area3")
    PSY.add_component!(sys, area1)
    PSY.add_component!(sys, area3)

    b1 = _add_simple_bus!(sys, 1, ACBusTypes.REF, 230)
    b2 = _add_simple_bus!(sys, 2, ACBusTypes.PQ, 230)
    b3 = _add_simple_bus!(sys, 3, ACBusTypes.PQ, 230)
    PSY.set_area!(b1, area1)
    PSY.set_area!(b2, area1)
    PSY.set_area!(b3, area3)

    _add_simple_source!(sys, b1, 0.0, 0.0)
    _add_simple_load!(sys, b2, 10.0, 5.0)
    _add_simple_load!(sys, b3, 60.0, 20.0)
    _add_simple_line!(sys, b1, b2, 5e-3, 5e-3, 1e-3)

    _add_simple_lcc!(sys, b2, b3, 0.05, 0.05, 0.08)
    return sys
end

# DC ties count toward the `tied` bitvector in `build_area_interchange_data`, so a DC-tie-only
# area can enroll where the old AC-only guard 5 would drop it. No real system exercises this
# end-to-end: any path to REF makes guard 3 reject it first. This testset verifies the
# tied-bitvector aggregation loop directly on `_dc_tie_only_fixture()`.
@testset "area interchange enrollment: DC-tie-only area is tied; zero-tie (AC+DC) area is not" begin
    sys = _dc_tie_only_fixture()
    pf = ACPolarPowerFlow{NewtonRaphsonACPowerFlow}()
    data = PowerFlowData(pf, sys)
    bus_lookup = PF.get_bus_lookup(data)
    ybus = PF.get_power_network_matrix(data)
    nrd = PF.get_network_reduction_data(data)
    bus_area_map = Dict(bus_lookup[3] => 1)   # only Area3's own bus is a candidate (tail 1)

    (ac_ties, dc_ties) = PF.build_area_ties(
        sys, bus_lookup, ybus, nrd, bus_area_map, data.lcc, PF.get_dc_network(data))
    @test isempty(ac_ties)
    @test length(dc_ties) == 1
    @test only(dc_ties).to_area_tail == 1

    # Mirror of enrollment.jl's tied-bitvector loop (~:337-345), run on this REAL tie data.
    function _tied_from_ties(n::Int, ac::Vector{PF.AreaTie}, dc::Vector{PF.DCTie})
        tied = falses(n)
        for tie in ac
            iszero(tie.from_area_tail) || (tied[tie.from_area_tail] = true)
            iszero(tie.to_area_tail) || (tied[tie.to_area_tail] = true)
        end
        for tie in dc
            iszero(tie.from_area_tail) || (tied[tie.from_area_tail] = true)
            iszero(tie.to_area_tail) || (tied[tie.to_area_tail] = true)
        end
        return tied
    end
    @test _tied_from_ties(1, ac_ties, dc_ties) == BitVector([true])
    @test _tied_from_ties(1, PF.AreaTie[], PF.DCTie[]) == BitVector([false])

    @test_logs(
        (:warn, r"Area \"Area3\": zero in-service ties"),
        min_level = Logging.Warn,
        PF._warn_zero_tie_areas(["Area3"], _tied_from_ties(1, PF.AreaTie[], PF.DCTie[]))
    )

    # Area3's own bus has no electrical path to any REF, so PowerFlows auto-promotes it to
    # REF; guard 3 (not guard 5) is what keeps this fixture from enrolling end-to-end.
    pf_control =
        ACPolarPowerFlow{NewtonRaphsonACPowerFlow}(; area_interchange_control = true)
    data_control = PowerFlowData(pf_control, sys)
    bus_lookup_control = PF.get_bus_lookup(data_control)
    @test data_control.bus_type[bus_lookup_control[3], 1] == PSY.ACBusTypes.REF
    aid = PF.build_area_interchange_data(pf_control, sys, data_control)
    @test PF.n_controlled_areas(aid) == 0
end

@testset "area interchange enrollment rule 6 slack absorption limit" begin
    sys = _make_two_area_system()
    _set_slack!(sys, "Bus 6")
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
    # Area1 (REF, no SLACK) stays uncontrolled; Area2/Area3 enroll with tail_ix 1,2 in
    # sorted-name order. pdes values come from `_three_area_transfer_fixture`'s
    # AreaInterchange records.
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
