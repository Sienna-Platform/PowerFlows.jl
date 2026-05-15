dc_power_flows = (
    PF.DCPowerFlow(),
    PF.PTDFDCPowerFlow(),
    PF.vPTDFDCPowerFlow(),
)
dc_reduction_types = Dict{String, Vector{PNM.NetworkReduction}}(
    "radial" => PNM.NetworkReduction[PNM.RadialReduction()],
    "degree 2" => PNM.NetworkReduction[PNM.DegreeTwoReduction()],
    "radial + degree 2" =>
        PNM.NetworkReduction[PNM.RadialReduction(), PNM.DegreeTwoReduction()],
)
@testset "DC power flow on 2k bus system: validate reduce-then-solve" begin
    sys = build_system(MatpowerTestSystems, "matpower_ACTIVSg2000_sys")
    @testset "$k reduction" for (k, v) in dc_reduction_types
        @testset "$pf_type power flow" for pf_type in
                                           InteractiveUtils.subtypes(PF.AbstractDCPowerFlow)
            dc_pf = pf_type(; correct_bustypes = true)
            unreduced = PF.PowerFlowData(dc_pf, sys)
            PF.solve_power_flow!(unreduced)
            @assert all(unreduced.converged)
            dc_pf_reduced =
                pf_type(; network_reductions = deepcopy(v), correct_bustypes = true)
            validate_reduced_power_flow(dc_pf_reduced, sys, v, unreduced)
            results = PF.solve_power_flow(dc_pf_reduced, sys, PF.FlowReporting.ARC_FLOWS)
            # no write-results-to-system solve_and_store_power_flow! for DC: should we add one?
        end
    end
end

# Ward reduction tested separately from Radial/Degree Two reductions because power flow results for retained buses are not expected to match exactly.
@testset "RTS: Ward Reduction and DC Power Flow" begin
    sys = build_system(PSISystems, "RTS_GMLC_DA_sys")
    bus_numbers = get_number.(get_components(ACBus, sys))
    study_buses = filter!(x -> digits(x)[end] == 1, bus_numbers)  #study buses are from area 1
    n_study_buses = length(study_buses)
    @testset "ward reduction $pf_type power flow" for pf_type in
                                                      InteractiveUtils.subtypes(
        PF.AbstractDCPowerFlow,
    )
        dc_pf = pf_type()
        unreduced = PF.PowerFlowData(dc_pf, sys)
        PF.solve_power_flow!(unreduced)
        @assert all(unreduced.converged)
        dc_pf_reduced = pf_type(;
            network_reductions = PNM.NetworkReduction[PNM.WardReduction(study_buses)],
        )
        results = PF.solve_power_flow(
            dc_pf_reduced,
            sys,
            PF.FlowReporting.ARC_FLOWS,
        )
        @test nrow(results["1"]["bus_results"]) == n_study_buses
    end
end

@testset "Validation test: Ward Reduction and DC Power Flow" begin
    sys = build_system(PSITestSystems, "c_sys5")
    b6 = ACBus(;
        number = 6,
        name = "b6",
        available = true,
        bustype = PowerSystems.ACBusTypes.PQ,
        angle = 0.0,
        magnitude = 1.0,
        voltage_limits = (min = 0.9, max = 1.1),
        base_voltage = 230.0,
    )
    add_component!(sys, b6)
    b2 = get_component(ACBus, sys, "nodeB")
    b4 = get_component(ACBus, sys, "nodeD")
    arc_2_6 = Arc(; from = b2, to = b6)
    arc_4_6 = Arc(; from = b4, to = b6)
    line_2_6 = Line(;
        name = "l_2_6",
        available = true,
        active_power_flow = 0.0,
        reactive_power_flow = 0.0,
        arc = arc_2_6,
        r = 0.01,
        x = 0.2,
        b = (from = 0.0, to = 0.0),
        rating = 0.0,
        angle_limits = (min = -pi, max = pi),
    )
    line_4_6 = Line(;
        name = "l_4_6",
        available = true,
        active_power_flow = 0.0,
        reactive_power_flow = 0.0,
        arc = arc_4_6,
        r = 0.01,
        x = 0.2,
        b = (from = 0.0, to = 0.0),
        rating = 0.0,
        angle_limits = (min = -pi, max = pi),
    )
    add_component!(sys, line_2_6)
    add_component!(sys, line_4_6)
    unreduced = PF.PowerFlowData(PF.DCPowerFlow(), sys)
    PF.solve_power_flow!(unreduced)
    @assert all(unreduced.converged)
    ward_reduction = PNM.NetworkReduction[PNM.WardReduction([1, 2, 3, 4, 5])]
    dc_pf_reduced =
        PF.DCPowerFlow(;
            network_reductions = ward_reduction,
        )
    validate_reduced_power_flow(
        dc_pf_reduced,
        sys,
        ward_reduction,
        unreduced,
    )
end

@testset "DC PowerFlow with Ward and multiple islands" begin
    sys = build_system(PSISystems, "2Area 5 Bus System")
    show_components(ACBus, sys, [:number])
    ward_reduction = PNM.NetworkReduction[PNM.WardReduction([1, 2, 3, 4])]

    ybus = Ybus(sys; network_reductions = ward_reduction)
    @show ybus.subnetwork_axes
    data = PF.PowerFlowData(PF.DCPowerFlow(; network_reductions = ward_reduction), sys)
    PF.solve_power_flow!(data)
    @test all(data.converged)
end
