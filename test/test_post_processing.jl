function _count_branches(sys)
    two_arc_branches = length(
        get_available_components(
            x -> !isa(x, ThreeWindingTransformer),
            ACTransmission,
            sys,
        ),
    )
    three_arc_branches =
        length(get_components(get_available_primary, ThreeWindingTransformer, sys)) +
        length(get_components(get_available_secondary, ThreeWindingTransformer, sys)) +
        length(get_components(get_available_tertiary, ThreeWindingTransformer, sys))
    return two_arc_branches + three_arc_branches
end

"""Compare two flow DataFrames by joining on flow_name, so row ordering doesn't matter."""
function _compare_flow_dataframes(
    full_branch_df,
    reduced_branch_df;
    atol = 1e-6,
    skip_cols = String["angle_difference", "P_losses", "Q_losses"],
)
    @test size(full_branch_df, 1) == size(reduced_branch_df, 1)
    # Join on flow_name so we compare the same branch regardless of sort order.
    joined = DataFrames.innerjoin(full_branch_df, reduced_branch_df;
        on = :flow_name, makeunique = true)
    @test size(joined, 1) == size(full_branch_df, 1)
    for col in names(full_branch_df)
        col == "flow_name" && continue
        # Angle differences and losses across series-reduced branches are
        # inherently different because the reduction changes the network topology.
        col in skip_cols && continue
        col_full = col
        col_reduced = "$(col)_1"
        for row in eachrow(joined)
            @test isapprox(row[col_full], row[col_reduced]; atol = atol)
        end
    end
end

@testset "DC power flow arc/branch flow reporting" begin
    sys = build_system(PSITestSystems, "case10_radial_series_reductions")
    pf_data_full = PF.PowerFlowData(
        DCPowerFlow(),
        sys,
    )
    pf_data_reduced = PF.PowerFlowData(
        DCPowerFlow(; network_reductions = NetworkReduction[DegreeTwoReduction()]),
        sys,
    )
    res_full_branch =
        solve_power_flow(pf_data_full, sys, PF.FlowReporting.BRANCH_FLOWS)["1"]["flow_results"]
    res_full_arc =
        solve_power_flow(pf_data_full, sys, PF.FlowReporting.ARC_FLOWS)["1"]["flow_results"]
    res_reduced_branch =
        solve_power_flow(pf_data_reduced, sys, PF.FlowReporting.BRANCH_FLOWS)["1"]["flow_results"]
    res_reduced_arc =
        solve_power_flow(pf_data_reduced, sys, PF.FlowReporting.ARC_FLOWS)["1"]["flow_results"]
    n_arcs = length(get_components(Arc, sys))
    @test size(res_full_arc)[1] == n_arcs

    n_branches = _count_branches(sys)
    @test size(res_full_branch)[1] == n_branches

    @test size(res_reduced_arc)[1] < n_arcs
    @test size(res_reduced_branch)[1] == n_branches

    _compare_flow_dataframes(res_full_branch, res_reduced_branch)
end

@testset "AC power flow arc reporting with/without degree-2 reduction" begin
    sys = build_system(MatpowerTestSystems, "matpower_ACTIVSg2000_sys")
    pf_unreduced = PF.ACPowerFlow{PF.TrustRegionACPowerFlow}(; correct_bustypes = true)
    pf = PF.ACPowerFlow{PF.TrustRegionACPowerFlow}(;
        skip_redistribution = true,
        correct_bustypes = true,
        # Protect reactive-injector hosts: with the default (true), PNM folds
        # purely-susceptive FixedAdmittance buses into series equivalents and
        # drops their shunts — an intentional approximation that breaks the
        # flow-parity premise of this testset (visible at MW/MVAr scale).
        network_reductions = PNM.NetworkReduction[
            PNM.DegreeTwoReduction(; reduce_reactive_power_injectors = false),
        ],
    )
    results_unreduced = solve_power_flow(pf_unreduced, sys, PF.FlowReporting.ARC_FLOWS)
    results_reduced = solve_power_flow(pf, sys, PF.FlowReporting.ARC_FLOWS)
    data = PF.PowerFlowData(pf, sys)
    nrd = PNM.get_network_reduction_data(PF.get_power_network_matrix(data))
    for arc in keys(PNM.get_direct_branch_map(nrd))
        arc_name = "$(arc[1])-$(arc[2])"
        df_unreduced =
            filter(row -> row.flow_name == arc_name, results_unreduced["flow_results"])
        df_reduced =
            filter(row -> row.flow_name == arc_name, results_reduced["flow_results"])
        @test size(df_unreduced, 1) == 1
        @test size(df_reduced, 1) == 1
        @test isapprox(
            df_unreduced[1, :P_from_to],
            df_reduced[1, :P_from_to];
            atol = 1e-2,
            rtol = 1e-5,
        )
        @test isapprox(
            df_unreduced[1, :Q_from_to],
            df_reduced[1, :Q_from_to];
            atol = 1e-2,
            rtol = 1e-5,
        )
    end
end

@testset "AC power flow branch reporting with/without degree-2 reduction" begin
    sys = build_system(MatpowerTestSystems, "matpower_ACTIVSg2000_sys")
    pf_unreduced = PF.ACPowerFlow{PF.TrustRegionACPowerFlow}(; correct_bustypes = true)
    pf = PF.ACPowerFlow{PF.TrustRegionACPowerFlow}(;
        skip_redistribution = true,
        correct_bustypes = true,
        # Same rationale as the arc-reporting testset below: protect
        # reactive-injector hosts so the reduced network is physics-equivalent
        # and flow parity at MW/MVAr-scale tolerances is meaningful.
        network_reductions = PNM.NetworkReduction[
            PNM.DegreeTwoReduction(; reduce_reactive_power_injectors = false),
        ],
    )
    results_unreduced = solve_power_flow(pf_unreduced, sys, PF.FlowReporting.BRANCH_FLOWS)
    results_reduced = solve_power_flow(pf, sys, PF.FlowReporting.BRANCH_FLOWS)
    _compare_flow_dataframes(
        results_unreduced["flow_results"],
        results_reduced["flow_results"];
        atol = 1e-2,
    )
end

@testset "AC power flow arc/branch flow reporting with degree-2 reduction" begin
    # Both use DegreeTwoReduction so they have the same branch set.
    # The "full" version solves with reduction (same as reduced) but we compare
    # the BRANCH_FLOWS output from _compute_segment_flows against the system-update
    # path (solve_and_store_power_flow!) to verify consistency.
    sys = build_system(MatpowerTestSystems, "matpower_ACTIVSg2000_sys")
    pf = PF.ACPowerFlow{PF.TrustRegionACPowerFlow}(;
        correct_bustypes = true,
        skip_redistribution = true,
        network_reductions = PNM.NetworkReduction[PNM.DegreeTwoReduction()],
    )

    # Get DataFrame results with BRANCH_FLOWS reporting
    res = solve_power_flow(pf, sys, PF.FlowReporting.BRANCH_FLOWS)
    flow_df = res["flow_results"]

    # Also solve and store to system
    sys2 = build_system(MatpowerTestSystems, "matpower_ACTIVSg2000_sys")
    pf2 = PF.ACPowerFlow{PF.TrustRegionACPowerFlow}(;
        correct_bustypes = true,
        skip_redistribution = true,
        network_reductions = PNM.NetworkReduction[PNM.DegreeTwoReduction()],
    )
    solve_and_store_power_flow!(pf2, sys2)
    base_power = PSY.get_base_power(sys2, PSY.NU)

    # For every series branch segment, verify that DataFrame flow matches system object.
    nrd = PNM.get_network_reduction_data(
        PNM.Ybus(sys2; network_reductions = PNM.NetworkReduction[PNM.DegreeTwoReduction()]),
    )
    n_series_segments = 0
    n_parallel_in_series = 0
    for (equivalent_arc, segments) in PNM.get_series_branch_map(nrd)
        for segment in segments
            if segment isa PNM.BranchesParallel
                # Test individual branches within parallel sub-segments of series chains.
                for branch in segment
                    n_parallel_in_series += 1
                    name = PNM.get_name(branch)
                    df_row = filter(row -> row.flow_name == name, flow_df)
                    @test size(df_row, 1) == 1
                    sys_branch = PSY.get_component(PSY.Branch, sys2, name)
                    sys_P = PSY.get_active_power_flow(sys_branch, PSY.SU)
                    sys_Q = PSY.get_reactive_power_flow(sys_branch, PSY.SU)
                    @test isapprox(df_row[1, :P_from_to], sys_P * base_power; atol = 1e-3)
                    @test isapprox(df_row[1, :Q_from_to], sys_Q * base_power; atol = 1e-3)
                end
            else
                n_series_segments += 1
                name = PSY.get_name(segment)
                df_row = filter(row -> row.flow_name == name, flow_df)
                @test size(df_row, 1) == 1
                sys_branch = PSY.get_component(PSY.Branch, sys2, name)
                sys_P = PSY.get_active_power_flow(sys_branch, PSY.SU)
                sys_Q = PSY.get_reactive_power_flow(sys_branch, PSY.SU)
                # DataFrame is in MW/MVAr, system is in p.u.
                @test isapprox(df_row[1, :P_from_to], sys_P * base_power; atol = 1e-3)
                @test isapprox(df_row[1, :Q_from_to], sys_Q * base_power; atol = 1e-3)
            end
        end
    end
    @test n_series_segments > 0
    @test n_parallel_in_series > 0
end

@testset "_solve_series_interior_voltages: KCL at interior nodes" begin
    # Use the 2k-bus system which has series branches after degree-2 reduction.
    # Verify that at each interior node, the solved voltages satisfy KCL (zero
    # net current injection).
    sys = build_system(MatpowerTestSystems, "matpower_ACTIVSg2000_sys")
    pf = PF.ACPowerFlow{PF.TrustRegionACPowerFlow}(;
        correct_bustypes = true,
        network_reductions = PNM.NetworkReduction[PNM.DegreeTwoReduction()],
    )
    data = PF.PowerFlowData(pf, sys)
    PF.solve_power_flow!(data)

    nrd = PNM.get_network_reduction_data(PF.get_power_network_matrix(data))
    bus_lookup = PF.get_bus_lookup(data)
    n_tested = 0

    for (equivalent_arc, segments) in PNM.get_series_branch_map(nrd)
        (ix_from, ix_to) = (bus_lookup[equivalent_arc[1]], bus_lookup[equivalent_arc[2]])
        V0 = data.bus_magnitude[ix_from, 1] * exp(im * data.bus_angles[ix_from, 1])
        Vn = data.bus_magnitude[ix_to, 1] * exp(im * data.bus_angles[ix_to, 1])
        x = PF._solve_series_interior_voltages(segments, equivalent_arc, (V0, Vn), nrd)
        all_V = vcat([V0], x, [Vn])

        # At each interior node k (1 <= k <= n-1), the total current must be zero:
        # I_in_from_left + I_in_from_right = 0
        chain = collect(segments)
        expected_from = equivalent_arc[1]
        for k in 1:(length(chain) - 1)
            seg_left = chain[k]
            seg_right = chain[k + 1]
            # Get y-parameters oriented along the chain direction
            (sf_l, _) = PNM.get_arc_tuple(seg_left)
            rev_l = sf_l != expected_from
            (_, _, y21_l, y22_l) = if rev_l
                reverse(PNM.ybus_branch_entries(seg_left, nrd))
            else
                PNM.ybus_branch_entries(seg_left, nrd)
            end
            # Current into node k from the left segment (to-side of segment k)
            I_from_left = y21_l * all_V[k] + y22_l * all_V[k + 1]

            next_from =
                rev_l ? PNM.get_arc_tuple(seg_left)[1] :
                PNM.get_arc_tuple(seg_left)[2]
            (sf_r, _) = PNM.get_arc_tuple(seg_right)
            rev_r = sf_r != next_from
            (y11_r, y12_r, _, _) = if rev_r
                reverse(PNM.ybus_branch_entries(seg_right, nrd))
            else
                PNM.ybus_branch_entries(seg_right, nrd)
            end
            # Current into node k from the right segment (from-side of segment k+1)
            I_from_right = y11_r * all_V[k + 1] + y12_r * all_V[k + 2]

            # KCL: net current at interior node = 0
            @test abs(I_from_left + I_from_right) < 1e-10
            n_tested += 1

            expected_from = next_from
        end
    end
    @test n_tested > 0
end

@testset "_compute_segment_flows consistency" begin
    # Verify that _compute_segment_flows returns entries consistent with arc-level flows
    # for direct and parallel branches. Series branches (including parallel-within-series)
    # are covered by the integration test above.
    sys = build_system(MatpowerTestSystems, "matpower_ACTIVSg2000_sys")
    pf = PF.ACPowerFlow{PF.TrustRegionACPowerFlow}(;
        correct_bustypes = true,
        network_reductions = PNM.NetworkReduction[PNM.DegreeTwoReduction()],
    )
    data = PF.PowerFlowData(pf, sys)
    PF.solve_power_flow!(data)

    nrd = PNM.get_network_reduction_data(PF.get_power_network_matrix(data))
    arc_lookup = PF.get_arc_lookup(data)

    # Test direct branches: single entry matching arc flow.
    # Tolerance is 1e-3 because Float32 matrix entries with large admittances (~10k)
    # introduce rounding errors around 1e-4.
    for (arc, branch) in PNM.get_direct_branch_map(nrd)
        entries = PF._compute_segment_flows(branch, data, arc, 1)
        @test length(entries) == 1
        arc_ix = arc_lookup[arc]
        @test isapprox(
            entries[1].P_from_to,
            data.arc_active_power_flow_from_to[arc_ix, 1];
            atol = 1e-3,
        )
        @test isapprox(
            entries[1].Q_from_to,
            data.arc_reactive_power_flow_from_to[arc_ix, 1];
            atol = 1e-3,
        )
    end

    # Test parallel branches: individual segment flows should sum to arc flow
    for (arc, parallel) in PNM.get_parallel_branch_map(nrd)
        entries = PF._compute_segment_flows(parallel, data, arc, 1)
        @test length(entries) == PNM.length(parallel)
        arc_ix = arc_lookup[arc]
        total_P_ft = sum(e.P_from_to for e in entries)
        total_Q_ft = sum(e.Q_from_to for e in entries)
        @test isapprox(
            total_P_ft,
            data.arc_active_power_flow_from_to[arc_ix, 1];
            atol = 1e-3,
        )
        @test isapprox(
            total_Q_ft,
            data.arc_reactive_power_flow_from_to[arc_ix, 1];
            atol = 1e-3,
        )
    end
end
