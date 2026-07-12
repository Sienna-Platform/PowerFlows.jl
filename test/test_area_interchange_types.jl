@testset "area interchange constructor validation" begin
    pf = ACPolarPowerFlow{NewtonRaphsonACPowerFlow}(; area_interchange_control = true)
    @test PowerFlows.get_area_interchange_control(pf) == true
    @test PowerFlows.get_interchange_tolerance(pf) == 0.05
    @test PowerFlows.get_tie_definition(pf) == :lines_only

    default_pf = ACPolarPowerFlow()
    @test PowerFlows.get_area_interchange_control(default_pf) == false

    @test_throws ArgumentError ACRectangularPowerFlow{NewtonRaphsonACPowerFlow}(;
        area_interchange_control = true,
    )
    @test_throws ArgumentError ACMixedPowerFlow{NewtonRaphsonACPowerFlow}(;
        area_interchange_control = true,
    )

    for S in (
        LevenbergMarquardtACPowerFlow,
        RobustHomotopyPowerFlow,
        GradientDescentACPowerFlow,
        FastDecoupledACPowerFlow,
    )
        @test_throws ArgumentError ACPolarPowerFlow{S}(; area_interchange_control = true)
    end

    @test_throws ArgumentError ACPolarPowerFlow(;
        area_interchange_control = true,
        tie_definition = :lines_and_loads,
    )

    floored_pf =
        @test_logs (:warn, r"interchange_tolerance") match_mode = :any ACPolarPowerFlow(;
            area_interchange_control = true,
            interchange_tolerance = 0.0,
        )
    @test PowerFlows.get_interchange_tolerance(floored_pf) == 0.02

    @test ACPolarPowerFlow(; area_interchange_control = false).area_interchange_control ==
          false
    @test ACRectangularPowerFlow(;
        area_interchange_control = false,
    ).area_interchange_control ==
          false
    @test ACMixedPowerFlow(; area_interchange_control = false).area_interchange_control ==
          false
    for S in (
        LevenbergMarquardtACPowerFlow,
        RobustHomotopyPowerFlow,
        GradientDescentACPowerFlow,
        FastDecoupledACPowerFlow,
    )
        @test ACPolarPowerFlow{S}(;
            area_interchange_control = false,
        ).area_interchange_control ==
              false
    end
end

@testset "area interchange SLACK bus ingestion" begin
    base_sys = PSB.build_system(PSB.PSITestSystems, "c_sys14"; add_forecasts = false)
    pf = ACPowerFlow()
    ref_data = PowerFlowData(pf, base_sys)
    @test solve_power_flow!(ref_data)

    # (a) SLACK on a bus with in-service generators normalizes to PV.
    sys_a = deepcopy(base_sys)
    bus_a = PSY.get_component(PSY.ACBus, sys_a, "Bus 2")
    PSY.set_bustype!(bus_a, PSY.ACBusTypes.SLACK)
    data_a = PowerFlowData(pf, sys_a)
    ix_a = PF.get_bus_lookup(data_a)[PSY.get_number(bus_a)]
    @test data_a.bus_type[ix_a, 1] == PSY.ACBusTypes.PV
    @test solve_power_flow!(data_a)
    @test all(isapprox.(data_a.bus_magnitude, ref_data.bus_magnitude; atol = 1e-8))
    @test all(isapprox.(data_a.bus_angles, ref_data.bus_angles; atol = 1e-8))

    # (b) SLACK on a bus with no in-service component capable of active power
    # injection errors at construction: pure-load bus...
    sys_b = deepcopy(base_sys)
    bus_b = PSY.get_component(PSY.ACBus, sys_b, "Bus 14")
    PSY.set_bustype!(bus_b, PSY.ACBusTypes.SLACK)
    @test_throws ArgumentError PowerFlowData(pf, sys_b)
    @test_throws r"SLACK-designated bus Bus 14" PowerFlowData(pf, sys_b)

    # ...and generator-backed bus whose generators are all out of service.
    sys_b2 = deepcopy(base_sys)
    bus_b2 = PSY.get_component(PSY.ACBus, sys_b2, "Bus 2")
    PSY.set_bustype!(bus_b2, PSY.ACBusTypes.SLACK)
    for gen in PSY.get_components(PSY.Generator, sys_b2)
        if PSY.get_number(PSY.get_bus(gen)) == PSY.get_number(bus_b2)
            PSY.set_available!(gen, false)
        end
    end
    @test_throws ArgumentError PowerFlowData(pf, sys_b2)
    @test_throws r"SLACK-designated bus Bus 2" PowerFlowData(pf, sys_b2)

    # (c) P-capable but not voltage-regulation-capable normalizes to PQ + warn.
    # The only PSY types in that gap (HybridSystem, InterconnectingConverter)
    # need heavyweight fixtures, so unit-test the classification helper.
    bt_c = @test_logs (:warn, r"SLACK-designated bus TestBus") PF._normalize_slack_bustype(
        pf,
        PSY.ACBusTypes.SLACK,
        99,
        "TestBus",
        Set{Int}(),
        Set([99]),
    )
    @test bt_c == PSY.ACBusTypes.PQ

    # (d) same classification for a DC evaluation model demotes silently: voltage
    # regulation is irrelevant to DC power flow.
    dc_pf = DCPowerFlow()
    bt_c_dc = @test_logs(
        min_level = Logging.Warn,
        PF._normalize_slack_bustype(
            dc_pf,
            PSY.ACBusTypes.SLACK,
            99,
            "TestBus",
            Set{Int}(),
            Set([99]),
        )
    )
    @test bt_c_dc == PSY.ACBusTypes.PQ

    # (e) the throw sub-case (rule 1, no active-power capability) still throws for a
    # DC model.
    @test_throws ArgumentError PowerFlowData(dc_pf, sys_b)
    @test_throws r"SLACK-designated bus Bus 14" PowerFlowData(dc_pf, sys_b)

    # Non-SLACK bus types pass through the helper unchanged.
    for bt in
        (PSY.ACBusTypes.REF, PSY.ACBusTypes.PV, PSY.ACBusTypes.PQ, PSY.ACBusTypes.ISOLATED)
        @test PF._normalize_slack_bustype(pf, bt, 99, "TestBus", Set{Int}(), Set{Int}()) ==
              bt
        @test PF._normalize_slack_bustype(
            dc_pf, bt, 99, "TestBus", Set{Int}(), Set{Int}()) == bt
    end
end

@testset "area interchange AreaInterchangeData tail length" begin
    areas = [
        PF.ControlledArea("Area1", 1, 0.1, 1),
        PF.ControlledArea("Area2", 5, -0.1, 2),
    ]
    aid = PF.AreaInterchangeData(
        areas, PF.AreaTie[], 0.05, zeros(2), zeros(2, 1),
        areas, PF.AreaTie[], zeros(2, 1), Dict{Int, Vector{PF.RelaxedAreaRecord}}(),
    )
    @test PF.area_tail_length(aid) == 2
    @test PF.n_controlled_areas(aid) == 2

    empty_aid =
        PF.AreaInterchangeData(
            PF.ControlledArea[],
            PF.AreaTie[],
            0.05,
            Float64[],
            zeros(Float64, 0, 1),
            PF.ControlledArea[],
            PF.AreaTie[],
            zeros(Float64, 0, 1),
            Dict{Int, Vector{PF.RelaxedAreaRecord}}(),
        )
    @test PF.area_tail_length(empty_aid) == 0
    @test PF.n_controlled_areas(empty_aid) == 0
end

@testset "area interchange empty tail is inert" begin
    sys = PSB.build_system(PSB.PSITestSystems, "c_sys14"; add_forecasts = false)
    for pf in (ACPolarPowerFlow(), ACRectangularPowerFlow(), ACMixedPowerFlow())
        data = PowerFlowData(pf, sys)
        @test PF.n_controlled_areas(data) == 0
        @test isempty(PF.get_area_interchange_data(data).areas)
        @test isempty(PF.get_area_interchange_data(data).ties)
        @test solve_power_flow!(data)
    end
end
