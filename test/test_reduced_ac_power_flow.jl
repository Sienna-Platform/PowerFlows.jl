# Select three-winding windings out of the merged direct branch map by dispatch; the house
# rule forbids an `isa` gate even in tests.
_is_3wt_winding(::PNM.ThreeWindingTransformerCircuit) = true
_is_3wt_winding(::PSY.ACTransmission) = false

const UNSUPPORTED =
    Set(
        [
        (PNM.WardReduction, PF.ACPowerFlow{PF.TrustRegionACPowerFlow}),
    ],
    )
# PNM's radial reduction is exact only under the DC approximation; AC drops the leaf
# branch's losses/Q and the radial bus's voltage regulation, so retained-bus results
# aren't expected to match the unreduced solve (convergence-only check).
const NOT_EQUIVALENT =
    Set(
        [
        (PNM.RadialReduction, PF.ACPowerFlow{PF.TrustRegionACPowerFlow}),
    ],
    )

ac_reduction_types = Dict{String, Vector{PNM.NetworkReduction}}(
    "default" => PNM.NetworkReduction[],
    "radial" => PNM.NetworkReduction[PNM.RadialReduction()],
    "degree 2" => PNM.NetworkReduction[PNM.DegreeTwoReduction(;
        reduce_reactive_power_injectors = false,
    )],
    "radial + degree 2" =>
        PNM.NetworkReduction[
            PNM.RadialReduction(),
            PNM.DegreeTwoReduction(; reduce_reactive_power_injectors = false),
        ],
)
@testset "AC power flow on 2k bus system: validate reduce-then-solve" begin
    sys = build_system(MatpowerTestSystems, "matpower_ACTIVSg2000_sys")
    pf_unreduced = PF.ACPowerFlow{PF.TrustRegionACPowerFlow}(; correct_bustypes = true)
    unreduced = PF.PowerFlowData(pf_unreduced, sys)
    PF.solve_power_flow!(unreduced)
    @assert all(unreduced.converged)
    for (k, v) in ac_reduction_types
        isempty(v) && continue # no reduction at all.
        pf = ACPowerFlow{PF.TrustRegionACPowerFlow}(;
            correct_bustypes = true,
            network_reductions = deepcopy(v),
        )
        if any([(typeof(nr), typeof(pf)) in UNSUPPORTED for nr in v])
            @warn "Skipping unsupported combination"
            continue
        end
        @testset "$k reduction" begin
            if any([(typeof(nr), typeof(pf)) in NOT_EQUIVALENT for nr in v])
                result = test_reduced_power_flow(pf, sys, v)
                @test all(result.converged)
            else
                validate_reduced_power_flow(pf, sys, v, unreduced)
            end
        end
    end
end

@testset "all reductions on psse_14_network_reduction_test_system" begin
    sys = PSB.build_system(PSSEParsingTestSystems, "psse_14_network_reduction_test_system")

    for (k, v) in ac_reduction_types
        pf = ACPowerFlow{PF.TrustRegionACPowerFlow}(;
            correct_bustypes = true,
            network_reductions = deepcopy(v),
        )
        if any([(typeof(nr), typeof(pf)) in UNSUPPORTED for nr in v])
            @warn "Skipping unsupported combination"
            continue
        end
        @testset "$k reduction" begin
            result = test_reduced_power_flow(pf, sys, v)
            @test all(result.converged)
        end
    end
end

@testset "system + power flow solver calls" begin
    for (k, v) in ac_reduction_types
        @testset "$k reduction" begin
            sys = build_system(
                PSSEParsingTestSystems,
                "psse_14_network_reduction_test_system",
            )
            pf = ACPowerFlow{PF.TrustRegionACPowerFlow}(;
                correct_bustypes = true,
                network_reductions = deepcopy(v),
            )
            supported = !any([(typeof(nr), typeof(pf)) in UNSUPPORTED for nr in v])
            if !supported
                results = @test_logs((:error, r"did not converge in 1 of 1"),
                    match_mode = :any,
                    solve_power_flow(pf, sys)
                )
            else
                results = solve_power_flow(pf, sys)
            end
            @assert !isempty(PSY.get_components(PSY.ThreeWindingTransformer, sys))
            if supported
                arc_flows = results["flow_results"]
                # Look up 3WT winding flows by bus_from/bus_to (arc endpoints).
                # The branch index exists because the matrix exists; no populate step.
                temp_catalog = PNM.get_branch_catalog(
                    PNM.Ybus(sys; network_reductions = deepcopy(v)))
                test_trf =
                    first(collect(PSY.get_components(PSY.ThreeWindingTransformer, sys)))
                trf_arc_flows = zeros(ComplexF32, 3)
                # 3WT circuits share the direct branch map with ordinary branches, so gate on
                # a dispatched predicate rather than a type check.
                for (arc, winding) in
                    PNM.get_direct_branch_map(PNM.get_network_reduction_data(temp_catalog))
                    _is_3wt_winding(winding) || continue
                    PNM.get_transformer(winding) !== test_trf && continue
                    wnum = PNM.get_winding_number(winding)
                    ix =
                        (arc_flows[!, "bus_from"] .== arc[1]) .&
                        (arc_flows[!, "bus_to"] .== arc[2])
                    @assert sum(ix) > 0 "could not find arc ($arc) in results dataframe"
                    trf_arc_flows[wnum] =
                        sum(arc_flows[ix, "P_from_to"]) +
                        im * sum(arc_flows[ix, "Q_from_to"])
                end
                @test solve_and_store_power_flow!(pf, sys)
                base_power = PSY.get_base_power(sys, PSY.NU)
                # check that transformer bus-to-star entries are there.
                @test isapprox(
                    PSY.get_active_power_flow(PSY.get_circuits(test_trf)[1], PSY.SU),
                    real(trf_arc_flows[1]) / base_power;
                    atol = 1e-5,
                )
                @test isapprox(
                    PSY.get_reactive_power_flow(PSY.get_circuits(test_trf)[1], PSY.SU),
                    imag(trf_arc_flows[1]) / base_power;
                    atol = 1e-5,
                )
                @test isapprox(
                    PSY.get_active_power_flow(PSY.get_circuits(test_trf)[2], PSY.SU),
                    real(trf_arc_flows[2]) / base_power;
                    atol = 1e-5,
                )
                @test isapprox(
                    PSY.get_reactive_power_flow(PSY.get_circuits(test_trf)[2], PSY.SU),
                    imag(trf_arc_flows[2]) / base_power;
                    atol = 1e-5,
                )
                @test isapprox(
                    PSY.get_active_power_flow(PSY.get_circuits(test_trf)[3], PSY.SU),
                    real(trf_arc_flows[3]) / base_power;
                    atol = 1e-5,
                )
                @test isapprox(
                    PSY.get_reactive_power_flow(PSY.get_circuits(test_trf)[3], PSY.SU),
                    imag(trf_arc_flows[3]) / base_power;
                    atol = 1e-5,
                )
            else
                @warn "Skipping testing AC post-processing with unsupported reduction $k"
            end
        end
    end
end

function compare_voltages(
    unreduced::PF.PowerFlowData,
    sys::PSY.System,
    temp_bus_map::Dict{Int, String},
    bus_no::Int,
)
    bus_lookup = PF.get_bus_lookup(unreduced)
    unreduced_Vm = unreduced.bus_magnitude[bus_lookup[bus_no], 1]
    unreduced_Va = unreduced.bus_angles[bus_lookup[bus_no], 1]
    bus_name = temp_bus_map[bus_no]
    bus = PSY.get_component(PSY.ACBus, sys, bus_name)
    reduced_Vm = PSY.get_magnitude(bus)
    reduced_Va = PSY.get_angle(bus)
    @test isapprox(unreduced_Vm, reduced_Vm; atol = 1e-6)
    @test isapprox(unreduced_Va, reduced_Va; atol = 1e-6)
end

function compare_power_flows(
    unreduced::PF.PowerFlowData,
    sys::PSY.System,
    branch::PSY.Branch,
)
    name = PSY.get_name(branch)
    arc_lookup = PF.get_arc_lookup(unreduced)
    arc_ix = arc_lookup[PNM.get_arc_tuple(branch)]
    unreduced_active_flow = unreduced.arc_active_power_flow_from_to[arc_ix, 1]
    unreduced_reactive_flow = unreduced.arc_reactive_power_flow_from_to[arc_ix, 1]
    reduced_active_flow =
        PSY.get_active_power_flow(
            flow_holder(PSY.get_component(PSY.Branch, sys, name)),
            PSY.SU,
        )
    reduced_reactive_flow =
        PSY.get_reactive_power_flow(
            flow_holder(PSY.get_component(PSY.Branch, sys, name)),
            PSY.SU,
        )
    @test isapprox(unreduced_active_flow, reduced_active_flow; atol = 1e-3)
    @test isapprox(unreduced_reactive_flow, reduced_reactive_flow; atol = 1e-3)
end

@testset "parallel branches: recovering flows" begin
    # we can't run an unreduced power flow and compare: parallel branches are always reduced.
    # instead, do a sanity check: flows over the parallel branches should add up to
    # the net flow across the equivalent branch.

    sys = build_system(MatpowerTestSystems, "matpower_ACTIVSg2000_sys")
    pf_unreduced = PF.ACPowerFlow{PF.TrustRegionACPowerFlow}(; correct_bustypes = true)
    unreduced = PF.PowerFlowData(pf_unreduced, sys)
    pf = PF.ACPowerFlow{PF.TrustRegionACPowerFlow}(;
        skip_redistribution = true,
        correct_bustypes = true,
        # Keep reactive-injector hosts so the reduced network is physics-equivalent
        # to the unreduced one; the default drops their shunts (intentional PNM
        # approximation), which breaks reduced-vs-unreduced parity at the
        # MW/MVAr-scale tolerances below. Same rationale as test_post_processing.jl.
        network_reductions = PNM.NetworkReduction[
            PNM.DegreeTwoReduction(; reduce_reactive_power_injectors = false),
        ],
    )
    PF.solve_power_flow!(unreduced)
    PF.solve_and_store_power_flow!(pf, sys)
    temp_ybus = PNM.Ybus(
        sys;
        network_reductions = PNM.NetworkReduction[
            PNM.DegreeTwoReduction(; reduce_reactive_power_injectors = false),
        ],
    )
    nrd = PNM.get_network_reduction_data(temp_ybus)
    temp_bus_map = Dict{Int, String}(
        PSY.get_number(b) => PSY.get_name(b) for b in PSY.get_components(PSY.ACBus, sys)
    )
    parallel_br_map = PNM.get_parallel_branch_map(nrd)
    arc_lookup = PF.get_arc_lookup(unreduced)
    for (equiv_arc, branches) in parallel_br_map
        equiv_arc_ix = arc_lookup[equiv_arc]
        net_flow_from_to =
            unreduced.arc_active_power_flow_from_to[equiv_arc_ix, 1] +
            im * unreduced.arc_reactive_power_flow_from_to[equiv_arc_ix, 1]
        net_flow_to_from =
            unreduced.arc_active_power_flow_to_from[equiv_arc_ix, 1] +
            im * unreduced.arc_reactive_power_flow_to_from[equiv_arc_ix, 1]
        total_flow = zero(ComplexF32)
        expected_from_bus = equiv_arc[1]
        (from_bus_no, to_bus_no) = PNM.get_arc_tuple(first(branches))
        @assert equiv_arc == (from_bus_no, to_bus_no) ||
                equiv_arc == (to_bus_no, from_bus_no)
        reversed = from_bus_no != expected_from_bus
        for br in branches
            @assert PNM.get_arc_tuple(br) == (from_bus_no, to_bus_no)
            total_flow +=
                PSY.get_active_power_flow(flow_holder(br), PSY.SU) +
                im * PSY.get_reactive_power_flow(flow_holder(br), PSY.SU)
        end
        if reversed
            @test isapprox(net_from_to_from, total_flow; atol = 1e-3)
        else
            @test isapprox(net_flow_from_to, total_flow; atol = 1e-3)
        end
    end
end

@testset "degree 2 reduction: recovering flows/voltages" begin
    sys = build_system(MatpowerTestSystems, "matpower_ACTIVSg2000_sys")
    pf_unreduced = PF.ACPowerFlow{PF.TrustRegionACPowerFlow}(; correct_bustypes = true)
    unreduced = PF.PowerFlowData(pf_unreduced, sys)
    pf = PF.ACPowerFlow{PF.TrustRegionACPowerFlow}(;
        skip_redistribution = true,
        correct_bustypes = true,
        # Keep reactive-injector hosts so the reduced network is physics-equivalent
        # to the unreduced one; the default drops their shunts (intentional PNM
        # approximation), which breaks reduced-vs-unreduced parity at the
        # MW/MVAr-scale tolerances below. Same rationale as test_post_processing.jl.
        network_reductions = PNM.NetworkReduction[
            PNM.DegreeTwoReduction(; reduce_reactive_power_injectors = false),
        ],
    )
    PF.solve_power_flow!(unreduced)
    PF.solve_and_store_power_flow!(pf, sys)
    temp_ybus = PNM.Ybus(
        sys;
        network_reductions = PNM.NetworkReduction[
            PNM.DegreeTwoReduction(; reduce_reactive_power_injectors = false),
        ],
    )
    nrd = PNM.get_network_reduction_data(temp_ybus)
    temp_bus_map = Dict{Int, String}(
        PSY.get_number(b) => PSY.get_name(b) for b in PSY.get_components(PSY.ACBus, sys)
    )
    bus_lookup = PF.get_bus_lookup(unreduced)
    for (equivalent_arc, segments) in PNM.get_series_branch_map(nrd)
        for segment in segments
            (from_bus_no, to_bus_no) = PNM.get_arc_tuple(segment)
            compare_voltages(unreduced, sys, temp_bus_map, from_bus_no)
            compare_voltages(unreduced, sys, temp_bus_map, to_bus_no)
            if !(from_bus_no in equivalent_arc)
                @assert unreduced.bus_type[bus_lookup[from_bus_no]] == PSY.ACBusTypes.PQ
            end
            if !(to_bus_no in equivalent_arc)
                @assert unreduced.bus_type[bus_lookup[to_bus_no]] == PSY.ACBusTypes.PQ
            end
        end
    end
    for (equivalent_arc, segments) in PNM.get_series_branch_map(nrd)
        for segment in segments
            # skip parallel branches
            if !(segment isa PNM.BranchesParallel)
                compare_power_flows(unreduced, sys, segment)
            end
        end
    end
end

@testset "Load on a topologically isolated bus does not break PowerFlowData build" begin
    # An available load can sit on a bus that is out of the power flow: topologically isolated
    # (no in-service branch → excluded from the reduced Ybus, no merge representative). Its
    # withdrawal must be skipped, not KeyError in the device-bus lookup.
    sys = System(100.0)
    b1 = _add_simple_bus!(sys, 1, ACBusTypes.REF, 230, 1.0, 0.0)
    b2 = _add_simple_bus!(sys, 2, ACBusTypes.PQ, 230, 1.0, 0.0)
    _add_simple_source!(sys, b1, 0.0, 0.0)
    _add_simple_load!(sys, b2, 30, 10)
    _add_simple_line!(sys, b1, b2, 5e-3, 5e-3, 1e-3)
    biso = _add_simple_bus!(sys, 99, ACBusTypes.ISOLATED, 230, 1.0, 0.0)
    _add_simple_load!(sys, biso, 5, 2)
    pf = PF.ACPowerFlow{NewtonRaphsonACPowerFlow}(; correct_bustypes = false)
    data = PF.PowerFlowData(pf, sys)   # must not throw (isolated-bus load skipped)
    @test !haskey(PF.get_bus_lookup(data), 99)
    @test PF.solve_power_flow!(data)
end

# Two degree-two chains that share an endpoint pair are electrically parallel, so the
# reduction groups them onto one arc: `parallel_branch_map` holds a group whose members are
# `BranchesSeries` rather than physical branches. Per-branch reporting therefore has to
# recurse through both aggregate kinds, and the interior buses of a grouped chain still need
# their solved voltages written back.
function _sibling_degree_two_chains_sys()
    sys = System(100.0)
    b1 = _add_simple_bus!(sys, 1, ACBusTypes.REF, 230, 1.0, 0.0)
    b2 = _add_simple_bus!(sys, 2, ACBusTypes.PQ, 230, 1.0, 0.0)
    b3 = _add_simple_bus!(sys, 3, ACBusTypes.PQ, 230, 1.0, 0.0)
    b4 = _add_simple_bus!(sys, 4, ACBusTypes.PQ, 230, 1.0, 0.0)
    b5 = _add_simple_bus!(sys, 5, ACBusTypes.PQ, 230, 1.0, 0.0)
    b6 = _add_simple_bus!(sys, 6, ACBusTypes.PQ, 230, 1.0, 0.0)
    _add_simple_source!(sys, b1, 0.0, 0.0)
    _add_simple_line!(sys, b1, b2, 0.01, 0.05, 0.002)
    # Chain 2-3-5 and chain 2-4-5: buses 3 and 4 are the only degree-two buses.
    _add_simple_line!(sys, b2, b3, 0.01, 0.07, 0.002)
    _add_simple_line!(sys, b3, b5, 0.01, 0.09, 0.002)
    _add_simple_line!(sys, b2, b4, 0.01, 0.11, 0.002)
    _add_simple_line!(sys, b4, b5, 0.01, 0.13, 0.002)
    _add_simple_line!(sys, b5, b6, 0.01, 0.06, 0.002)
    _add_simple_load!(sys, b2, 30, 10)
    _add_simple_load!(sys, b6, 50, 15)
    return sys
end

@testset "Degree-2 reduction: parallel group of series chains" begin
    sys = _sibling_degree_two_chains_sys()
    reductions = PNM.NetworkReduction[PNM.DegreeTwoReduction(;
        reduce_reactive_power_injectors = false,
    )]
    nrd = PNM.get_network_reduction_data(PNM.Ybus(sys; network_reductions = reductions))

    # Pin the structure under test: a single composite arc holding both chains. Without this
    # the flow assertions below could pass on a topology that never builds the group.
    grouped = [
        (arc, group) for (arc, group) in PNM.get_parallel_branch_map(nrd) if
        any(m -> m isa PNM.BranchesSeries, group)
    ]
    @test length(grouped) == 1
    (group_arc, group) = only(grouped)
    @test Set(group_arc) == Set((2, 5))
    @test all(m -> m isa PNM.BranchesSeries, group)
    @test length(collect(group)) == 2

    n_lines = length(collect(get_components(Line, sys)))
    # AC returns the frames flat for a single period; DC nests them under the time step.
    _flows(results) =
        if haskey(results, "flow_results")
            results["flow_results"]
        else
            results["1"]["flow_results"]
        end
    ac_models = (
        PF.ACPowerFlow{NewtonRaphsonACPowerFlow}(),
        PF.ACPowerFlow{NewtonRaphsonACPowerFlow}(; network_reductions = reductions),
    )
    dc_models = (DCPowerFlow(), DCPowerFlow(; network_reductions = reductions))
    for (unreduced_model, reduced_model) in (ac_models, dc_models)
        unreduced =
            _flows(solve_power_flow(unreduced_model, sys, PF.FlowReporting.BRANCH_FLOWS))
        reduced =
            _flows(solve_power_flow(reduced_model, sys, PF.FlowReporting.BRANCH_FLOWS))
        # Every physical branch is reported exactly once, including those inside the group.
        @test nrow(reduced) == n_lines
        @test Set(reduced.flow_name) == Set(unreduced.flow_name)
        for row in eachrow(unreduced)
            reduced_row = only(eachrow(filter(:flow_name => ==(row.flow_name), reduced)))
            @test isapprox(reduced_row.P_from_to, row.P_from_to; atol = 1e-3)
        end
    end

    # Buses 3 and 4 live only inside the grouped chains; their voltages must still be solved
    # and written back, not left at the flat start.
    sys_unreduced = _sibling_degree_two_chains_sys()
    solve_and_store_power_flow!(PF.ACPowerFlow{NewtonRaphsonACPowerFlow}(), sys_unreduced)
    sys_reduced = _sibling_degree_two_chains_sys()
    solve_and_store_power_flow!(
        PF.ACPowerFlow{NewtonRaphsonACPowerFlow}(; network_reductions = reductions),
        sys_reduced,
    )
    for bus_number in 1:6
        expected = get_component(ACBus, sys_unreduced, "bus_$bus_number")
        actual = get_component(ACBus, sys_reduced, "bus_$bus_number")
        @test isapprox(get_magnitude(actual), get_magnitude(expected); atol = 1e-5)
        @test isapprox(get_angle(actual), get_angle(expected); atol = 1e-5)
    end
end
