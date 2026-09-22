#=
Flow reporting for reduction aggregates that contain other reduction aggregates.

`DegreeTwoReduction` builds three nested shapes (PNM `degree_two_reduction.jl`, and PNM's own
`test_nested_reduction_aggregates.jl`):

  - a parallel group as a chain segment -- `BranchesSeries` holding a `BranchesParallel`
  - sibling chains in parallel          -- `BranchesParallel{BranchesSeries}`
  - both at once                        -- a `BranchesParallel{BranchesSeries}` whose chains
    themselves carry a parallel segment

AC, DC and PTDF branch-flow reporting, and AC write-back, expand every nested aggregate to
its physical branches for all three shapes.
=#

# Minimal systems that produce one nested aggregate each. Buses 1 and 2 carry injections, so
# they are irreducible endpoints; buses 3 and 4 are bare and become chain interiors.
function _nested_reduction_system(shape::Symbol)
    sys = PSY.System(100.0)
    buses = Dict{Int, PSY.ACBus}()
    function add_bus!(number, bustype)
        b = PSY.ACBus(;
            number = number,
            name = "b$number",
            available = true,
            bustype = bustype,
            angle = 0.0,
            magnitude = 1.0,
            voltage_limits = (min = 0.9, max = 1.1),
            base_voltage = 230.0,
        )
        PSY.add_component!(sys, b)
        buses[number] = b
        return b
    end
    function add_line!(name, from, to, r, x)
        arc_name = "$(PSY.get_name(buses[from])) -> $(PSY.get_name(buses[to]))"
        arc = PSY.get_component(PSY.Arc, sys, arc_name)
        if isnothing(arc)
            arc = PSY.Arc(; from = buses[from], to = buses[to])
            PSY.add_component!(sys, arc)
        end
        PSY.add_component!(
            sys,
            PSY.Line(;
                name = name,
                available = true,
                active_power_flow = 0.0,
                reactive_power_flow = 0.0,
                arc = arc,
                r = r,
                x = x,
                b = (from = 0.0, to = 0.0),
                rating = 4.0,
                angle_limits = (min = -pi, max = pi),
            ),
        )
    end

    add_bus!(1, PSY.ACBusTypes.REF)
    add_bus!(2, PSY.ACBusTypes.PQ)
    add_bus!(3, PSY.ACBusTypes.PQ)
    shape != :parallel_in_series && add_bus!(4, PSY.ACBusTypes.PQ)

    # Chain 1-3-2, whose 1-3 segment is a parallel pair.
    if shape in (:parallel_in_series, :nested)
        add_line!("L13a", 1, 3, 0.01, 0.10)
        add_line!("L13b", 1, 3, 0.02, 0.20)
    else
        add_line!("L13", 1, 3, 0.01, 0.10)
    end
    add_line!("L32", 3, 2, 0.01, 0.12)
    # Chain 1-4-2, a sibling of chain 1-3-2 on the same endpoint pair.
    if shape != :parallel_in_series
        add_line!("L14", 1, 4, 0.015, 0.15)
        add_line!("L42", 4, 2, 0.015, 0.14)
    end

    PSY.add_component!(
        sys,
        PSY.PowerLoad(;
            name = "load2",
            available = true,
            bus = buses[2],
            active_power = 1.0,
            reactive_power = 0.2,
            base_power = 100.0,
            max_active_power = 1.0,
            max_reactive_power = 0.2,
        ),
    )
    PSY.add_component!(
        sys,
        PSY.ThermalStandard(;
            name = "gen1",
            available = true,
            status = PSY.OperationalStates.ONLINE,
            bus = buses[1],
            active_power = 1.05,
            reactive_power = 0.25,
            rating = 5.0,
            active_power_limits = (min = 0.0, max = 5.0),
            reactive_power_limits = (min = -5.0, max = 5.0),
            ramp_limits = nothing,
            operation_cost = PSY.ThermalGenerationCost(nothing),
            base_power = 100.0,
            time_limits = nothing,
            prime_mover_type = PSY.PrimeMovers.OT,
            fuel = PSY.ThermalFuels.OTHER,
        ),
    )
    return sys
end

_degree_two() = PNM.NetworkReduction[PNM.DegreeTwoReduction(;
    reduce_reactive_power_injectors = false,
)]

function _nested_reduction_data(sys)
    return PNM.get_network_reduction_data(
        PNM.Ybus(sys; network_reductions = _degree_two()),
    )
end

_ac_pf(; kwargs...) = PF.ACPowerFlow(; network_reductions = _degree_two(), kwargs...)

@testset "nested aggregates: parallel group inside a chain" begin
    sys = _nested_reduction_system(:parallel_in_series)
    nrd = _nested_reduction_data(sys)

    # Pin the fixture: one chain on arc (1, 2), whose first segment is a parallel group.
    @test isempty(PNM.get_parallel_branch_map(nrd))
    chain = only(values(PNM.get_series_branch_map(nrd)))
    @test chain isa PNM.BranchesSeries
    @test count(s -> s isa PNM.AbstractBranchesParallel, chain) == 1

    # Every physical branch is reported, and reported flows match an unreduced solve.
    branch_flows(pf) = solve_power_flow(pf, sys, PF.FlowReporting.BRANCH_FLOWS)
    reduced = branch_flows(_ac_pf())["flow_results"]
    full = branch_flows(PF.ACPowerFlow())["flow_results"]
    @test Set(reduced.flow_name) == Set(["L13a", "L13b", "L32"])
    joined = DataFrames.innerjoin(reduced, full; on = :flow_name, makeunique = true)
    @test size(joined, 1) == 3
    for row in eachrow(joined)
        @test isapprox(row.P_from_to, row.P_from_to_1; atol = 1e-3)
        @test isapprox(row.Q_from_to, row.Q_from_to_1; atol = 1e-3)
    end

    # The DC expanders descend into this shape too.
    for pf in (
        PF.DCPowerFlow(; network_reductions = _degree_two()),
        PF.PTDFDCPowerFlow(; network_reductions = _degree_two()),
    )
        df = solve_power_flow(pf, sys, PF.FlowReporting.BRANCH_FLOWS)["1"]["flow_results"]
        @test Set(df.flow_name) == Set(["L13a", "L13b", "L32"])
    end

    # The chain's interior bus voltage is recovered on write-back.
    sys2 = _nested_reduction_system(:parallel_in_series)
    solve_and_store_power_flow!(_ac_pf(), sys2)
    b3 = PSY.get_component(PSY.ACBus, sys2, "b3")
    full_b3 = filter(
        row -> row.bus_number == 3,
        solve_power_flow(PF.ACPowerFlow(), sys2)["bus_results"],
    )
    @test isapprox(PSY.get_magnitude(b3), full_b3[1, :Vm]; atol = 1e-6)
    @test isapprox(PSY.get_angle(b3), full_b3[1, :θ]; atol = 1e-6)
end

# `:series_in_parallel` nests a chain inside a parallel group (two levels); `:nested` adds a
# parallel segment inside one of those chains (three levels).
_NESTED_PARALLEL_SHAPES = (:series_in_parallel, :nested)

@testset "nested aggregates: chains inside a parallel group ($shape)" for shape in
                                                                          _NESTED_PARALLEL_SHAPES
    sys = _nested_reduction_system(shape)
    nrd = _nested_reduction_data(sys)

    # Pin the fixture: one parallel group on arc (1, 2) whose members are chains.
    @test isempty(PNM.get_series_branch_map(nrd))
    group = only(values(PNM.get_parallel_branch_map(nrd)))
    @test group isa PNM.BranchesParallel{PNM.BranchesSeries}
    @test all(m isa PNM.BranchesSeries for m in group)
    # In the three-level case one of those chains also carries a parallel segment.
    @test any(any(s -> s isa PNM.AbstractBranchesParallel, m) for m in group) ==
          (shape == :nested)

    expected_names = if shape == :nested
        Set(["L13a", "L13b", "L32", "L14", "L42"])
    else
        Set(["L13", "L32", "L14", "L42"])
    end

    # Every physical branch is reported, and reported flows match an unreduced solve.
    reduced = solve_power_flow(_ac_pf(), sys, PF.FlowReporting.BRANCH_FLOWS)["flow_results"]
    full =
        solve_power_flow(PF.ACPowerFlow(), sys, PF.FlowReporting.BRANCH_FLOWS)["flow_results"]
    @test Set(reduced.flow_name) == expected_names
    joined = DataFrames.innerjoin(reduced, full; on = :flow_name, makeunique = true)
    @test size(joined, 1) == length(expected_names)
    for row in eachrow(joined)
        @test isapprox(row.P_from_to, row.P_from_to_1; atol = 1e-3)
        @test isapprox(row.Q_from_to, row.Q_from_to_1; atol = 1e-3)
    end

    for PFType in (PF.DCPowerFlow, PF.PTDFDCPowerFlow)
        df = solve_power_flow(
            PFType(; network_reductions = _degree_two()),
            sys,
            PF.FlowReporting.BRANCH_FLOWS,
        )["1"]["flow_results"]
        full_df =
            solve_power_flow(PFType(), sys, PF.FlowReporting.BRANCH_FLOWS)["1"]["flow_results"]
        @test Set(df.flow_name) == expected_names
        joined = DataFrames.innerjoin(df, full_df; on = :flow_name, makeunique = true)
        @test size(joined, 1) == length(expected_names)
        for row in eachrow(joined)
            @test isapprox(row.P_from_to, row.P_from_to_1; atol = 1e-3)
        end
    end

    # The chain interiors' bus voltages are recovered on write-back.
    sys2 = _nested_reduction_system(shape)
    solve_and_store_power_flow!(_ac_pf(), sys2)
    full_bus = solve_power_flow(PF.ACPowerFlow(), sys2)["bus_results"]
    for name in ("b3", "b4")
        bus = PSY.get_component(PSY.ACBus, sys2, name)
        full_row = filter(row -> row.bus_number == PSY.get_number(bus), full_bus)
        @test isapprox(PSY.get_magnitude(bus), full_row[1, :Vm]; atol = 1e-6)
        @test isapprox(PSY.get_angle(bus), full_row[1, :θ]; atol = 1e-6)
    end

    # Arc-level reporting reports the equivalent arc by design: the whole nest collapses to
    # a single 1-2 row rather than the physical branches.
    arc_df = solve_power_flow(_ac_pf(), sys)["flow_results"]
    @test size(arc_df, 1) == 1
    @test arc_df[1, :bus_from] == 1 && arc_df[1, :bus_to] == 2
end
