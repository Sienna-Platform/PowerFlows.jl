"""
Run one DC and one AC power flow on a system with a given set of network reductions.
"""
function test_all_power_flow_types(
    sys::System,
    network_reductions::Vector{PNM.NetworkReduction},
)
    # AC power flow has different syntax
    pf = ACPowerFlow{PF.TrustRegionACPowerFlow}(;
        network_reductions = deepcopy(network_reductions),
        correct_bustypes = true,
    )
    data = PF.PowerFlowData(pf, sys)
    solve_power_flow!(data) # should run without errors.
    @test !isempty(data.arc_active_power_flow_from_to)
    pf = ACPowerFlow{PF.TrustRegionACPowerFlow}(;
        network_reductions = deepcopy(network_reductions),
        correct_bustypes = true,
    )
    solve_and_store_power_flow!(pf, sys)
    pf = PF.DCPowerFlow(; network_reductions = deepcopy(network_reductions))
    data = PF.PowerFlowData(pf, sys)
    solve_power_flow!(data) # should run without errors.
    @test !isempty(data.arc_active_power_flow_from_to)
end

@testset "radial reduction with 3WT/parallel lines" begin
    sys = build_system(PSB.PSITestSystems, "case10_radial_series_reductions")
    network_reductions = NetworkReduction[RadialReduction()]
    test_all_power_flow_types(sys, network_reductions)
end

@testset "degreee 2 reduction with 3WT/parallel lines" begin
    sys = build_system(PSB.PSITestSystems, "case10_radial_series_reductions")
    network_reductions =
        NetworkReduction[DegreeTwoReduction(; reduce_reactive_power_injectors = false)]
    test_all_power_flow_types(sys, network_reductions)
end

@testset "AC power flow rejects degree-2 reduction of reactive injectors" begin
    sys = build_system(PSB.PSITestSystems, "case10_radial_series_reductions")
    # `reduce_reactive_power_injectors = true` discards reactive injections the AC
    # solution depends on, so PowerFlowData construction must reject the combination.
    pf_ac = ACPowerFlow{PF.TrustRegionACPowerFlow}(;
        network_reductions = NetworkReduction[DegreeTwoReduction(;
            reduce_reactive_power_injectors = true,
        )],
        correct_bustypes = true,
    )
    @test_throws InfrastructureSystems.ConflictingInputsError PF.PowerFlowData(pf_ac, sys)
    # DC power flow ignores reactive injections, so the same reduction is allowed.
    pf_dc = PF.DCPowerFlow(;
        network_reductions = NetworkReduction[DegreeTwoReduction(;
            reduce_reactive_power_injectors = true,
        )],
    )
    @test PF.PowerFlowData(pf_dc, sys) isa PF.PowerFlowData
end
