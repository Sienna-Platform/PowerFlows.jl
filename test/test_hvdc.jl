@testset "Test DC power flow with VSC line" begin
    for DC_type in (PF.DCPowerFlow, PF.PTDFDCPowerFlow, PF.vPTDFDCPowerFlow)
        @testset "DC Solver: $(DC_type)" begin
            # Create a simple system with a VSC line
            sys = System(100.0)
            b1 = _add_simple_bus!(sys, 1, ACBusTypes.REF, 230, 1.0, 0.0)
            b2 = _add_simple_bus!(sys, 2, ACBusTypes.PQ, 230, 1.0, 0.0)
            b3 = _add_simple_bus!(sys, 3, ACBusTypes.PQ, 230, 1.0, 0.0)

            _add_simple_source!(sys, b1, 1.0, 0.0)
            _add_simple_load!(sys, b2, 0.3, 0.1)
            _add_simple_load!(sys, b3, 0.4, 0.1)

            _add_simple_line!(sys, b1, b2, 0.01, 0.05, 0.02)
            _add_simple_line!(sys, b1, b3, 0.01, 0.05, 0.02)

            # Add VSC line between buses 2 and 3
            P_flow = 0.2  # 20 MW in per-unit
            loss_coeff = 0.02  # 2% loss
            vsc = _add_simple_vsc!(
                sys,
                b2,
                b3;
                active_power_flow = P_flow,
                loss_coefficient = loss_coeff,
            )

            pf = DC_type()
            data = PF.PowerFlowData(pf, sys)

            # Verify VSC is recognized
            @test !isempty(get_components(TwoTerminalVSCLine, sys))

            # Check that bus_hvdc_net_power is populated correctly
            # From bus (b2) should have negative injection (power withdrawn)
            # To bus (b3) should have positive injection minus losses (2x loss_coeff for both converters)
            expected_from = -P_flow
            expected_to = P_flow * (1 - 2 * loss_coeff)

            bus_lookup = PF.get_bus_lookup(data)
            bus2_ix = bus_lookup[2]
            bus3_ix = bus_lookup[3]

            @test isapprox(data.bus_hvdc_net_power[bus2_ix, 1], expected_from; atol = 1e-6)
            @test isapprox(data.bus_hvdc_net_power[bus3_ix, 1], expected_to; atol = 1e-6)

            # Power flow should converge
            solve_power_flow!(data)
            @test all(data.converged)
        end
    end
end

@testset "Test HVDC injections helper function" begin
    sys = build_system(MatpowerTestSystems, "matpower_case5_dc_sys")
    hvdc = only(get_components(TwoTerminalHVDC, sys))

    P_dc = 10.0
    set_active_power_flow!(hvdc, P_dc * PSY.MW)

    # fixed slope loss function: 1% loss
    loss_coeff = 0.01
    set_loss!(hvdc, LinearCurve(loss_coeff))
    (P_from, P_to) = PF.hvdc_injections_natural_units(hvdc)
    @test isapprox(-P_from, P_dc)
    @test isapprox(P_to, P_dc * (1 - loss_coeff))

    # constant loss: 0.5 MW loss
    loss_const = 0.5
    set_loss!(hvdc, LinearCurve(0.0, loss_const))
    (P_from, P_to) = PF.hvdc_injections_natural_units(hvdc)
    @test isapprox(-P_from, P_dc)
    @test isapprox(P_to, P_dc - loss_const)

    # piecewise linear loss: 1% loss up to 20 MW, then 2% marginal loss above that
    loss_curve = PiecewiseIncrementalCurve(0.0, 0.0, [0.0, 20.0, 100.0], [0.01, 0.02])
    set_loss!(hvdc, loss_curve)
    for P_dc_setpoint in (10.0, 30.0)
        set_active_power_flow!(hvdc, P_dc_setpoint * PSY.MW)
        (P_from, P_to) = PF.hvdc_injections_natural_units(hvdc)
        expected_loss =
            0.01 * min(P_dc_setpoint, 20.0) + 0.02 * max(0.0, P_dc_setpoint - 20.0)
        @test isapprox(-P_from, P_dc_setpoint)
        @test isapprox(P_to, P_dc_setpoint - expected_loss)
    end

    # test reversed flow error
    _, lcc = simple_lcc_system()
    set_active_power_flow!(lcc, -15.0 * PSY.MW)
    @test_throws ArgumentError PF.hvdc_injections_natural_units(lcc)
end

@testset "Test AC on generic HVDC" begin
    sys = build_system(MatpowerTestSystems, "matpower_case5_dc_sys")

    pf = ACPowerFlow{TrustRegionACPowerFlow}(; correct_bustypes = true)
    data = PF.PowerFlowData(pf, sys)
    solve_power_flow!(data)
    @test all(data.converged)
end

@testset "Test DC on generic HVDC" begin
    for DC_type in (PF.DCPowerFlow, PF.PTDFDCPowerFlow, PF.vPTDFDCPowerFlow)
        sys = build_system(MatpowerTestSystems, "matpower_case5_dc_sys")

        pf = DC_type()
        data = PF.PowerFlowData(pf, sys)
        solve_power_flow!(data)
        @test all(data.converged)
    end
end

function add_component_with_power!(sys::PSY.System, bus::PSY.ACBus, P::Float64)
    if P == 0.0
        println(
            "Skipping addition of component with zero power at bus $(PSY.get_number(bus))",
        )
        return
    elseif P > 0
        gen = ThermalStandard(;
            name = "gen_$(PSY.get_number(bus))_thermal_hvdc_$P",
            available = true,
            status = true,
            bus = bus,
            active_power = P, # TODO check units
            reactive_power = 0.0,
            # rest shouldn't actually matter for our test.
            rating = 1.0,
            active_power_limits = (min = 0, max = P),
            reactive_power_limits = (min = -1, max = 1),
            ramp_limits = nothing,
            operation_cost = ThermalGenerationCost(nothing),
            base_power = 100.0,
            time_limits = nothing,
            prime_mover_type = PrimeMovers.OT,
            fuel = ThermalFuels.OTHER,
            services = Device[],
            dynamic_injector = nothing,
            ext = Dict{String, Any}(),
        )
        add_component!(sys, gen)
        @assert get_active_power(gen, PSY.SU) == P
    else
        load = PowerLoad(;
            name = "load_$(PSY.get_number(bus))_hvdc_$(-P)",
            available = true,
            bus = bus,
            active_power = -P, # Per-unitized by device base_power
            reactive_power = 0.0, # Per-unitized by device base_power
            base_power = 100.0, # MVA
            max_active_power = -P,
            max_reactive_power = 0.0,
        )
        add_component!(sys, load)
        @assert get_active_power(load, PSY.SU) == -P
    end
end

function replace_generic_hvdcs!(sys)
    for hvdc in get_components(PSY.TwoTerminalGenericHVDCLine, sys)
        if PSY.get_active_power_flow(hvdc, PSY.SU) == 0.0
            set_active_power_flow!(hvdc, 0.1 * PSY.SU)
        end
        (P_from, P_to) = PF.hvdc_injections_natural_units(hvdc)
        P_from /= PSY.get_base_power(sys, PSY.NU)
        P_to /= PSY.get_base_power(sys, PSY.NU)
        arc = get_arc(hvdc)
        bus_from = arc.from
        bus_to = arc.to
        add_component_with_power!(sys, bus_from, P_from)
        add_component_with_power!(sys, bus_to, P_to)
        remove_component!(sys, hvdc)
    end
    return sys
end

function test_generic_hvdc_on_big_system(pf_type::Type{<:PF.PowerFlowEvaluationModel})
    sys_original = build_system(PSISystems, "HVDC_TWO_RTO_RTS_1Hr_sys")

    for hvdc in get_components(PSY.TwoTerminalGenericHVDCLine, sys_original)
        set_active_power_flow!(hvdc, 0.1 * PSY.SU)
    end

    pf = pf_type(; correct_bustypes = true)
    data_original = PF.PowerFlowData(pf, sys_original)
    solve_power_flow!(data_original)
    ref_bus_inds = findall(data_original.bus_type .== (PSY.ACBusTypes.REF,))
    @test all(data_original.bus_angles[ref_bus_inds] .== 0.0)

    sys_modified = deepcopy(sys_original)
    replace_generic_hvdcs!(sys_modified)

    data_modified = PF.PowerFlowData(pf, sys_modified)
    solve_power_flow!(data_modified)

    # verify assumptions
    @assert !isempty(get_components(PSY.TwoTerminalGenericHVDCLine, sys_original))
    @assert isempty(get_components(PSY.TwoTerminalGenericHVDCLine, sys_modified))
    # arc axes aren't the same: the HVDC arcs must somehow change control flow in PNM.
    # @assert all(PF.get_arc_axis(data_original) .== PF.get_arc_axis(data_modified))
    @assert all(PF.get_bus_axis(data_original) .== PF.get_bus_axis(data_modified))

    @test all(
        abs.(
            data_original.power_network_matrix.data .-
            data_modified.power_network_matrix.data
        ) .< TIGHT_TOLERANCE,
    )
    if pf_type() isa PF.AbstractDCPowerFlow
        # for AC, aux network matrix defaults to nothing.
        @test all(
            abs.(
                data_original.aux_network_matrix.data .-
                data_modified.aux_network_matrix.data
            ) .< TIGHT_TOLERANCE,
        )
        # no need to check arc flows: since angles and aux matrices match, flows match.
        # check that the b in our Ax = b is the same
        rhs_original = (
            data_original.bus_active_power_injections .-
            data_original.bus_active_power_withdrawals .+
            data_original.bus_hvdc_net_power
        )
        rhs_modified = (
            data_modified.bus_active_power_injections .-
            data_modified.bus_active_power_withdrawals
        )
        @test all(abs.(rhs_original - rhs_modified) .< TIGHT_TOLERANCE)
    end

    for fieldname in (:bus_angles, :bus_magnitude)
        @test all(
            abs.(
                getfield(data_original, fieldname) .-
                getfield(data_modified, fieldname)
            ) .< TIGHT_TOLERANCE,
        )
    end

    for fieldname in (:arc_active_power_flow_from_to, :arc_active_power_flow_to_from,
        :arc_reactive_power_flow_from_to, :arc_reactive_power_flow_to_from)
        for (original_arc_ind, arc) in enumerate(PF.get_arc_axis(data_original))
            modified_arc_ind = PF.get_arc_lookup(data_modified)[arc]
            @test isapprox(
                getfield(data_original, fieldname)[original_arc_ind],
                getfield(data_modified, fieldname)[modified_arc_ind],
                atol = TIGHT_TOLERANCE,
            )
        end
    end
end

# take a big system with a generic HVDC, replace its injections/withdrawals with
# generators, and verify that a power flow converges to the same solution
@testset "Test Generic HVDC on big network: AC power flow" begin
    test_generic_hvdc_on_big_system(PF.ACPowerFlow)
end

@testset "Test Generic HVDC on big network: DC power flow" begin
    test_generic_hvdc_on_big_system(PF.DCPowerFlow)
end

@testset "Test AC power flow with multiple LCC lines" begin
    # This test covers the bug fix for LCC matrix indexing in solve_ac_powerflow.jl
    # where [time_step, i] was incorrectly used instead of [i, time_step]
    # The bug only manifested when there were multiple LCC lines.
    raw_path = joinpath(TEST_DATA_DIR, "case5_2_lcc.raw")
    sys = make_system(PFP.PowerModelsData(raw_path); runchecks = false)

    # Verify we have multiple LCC lines
    lcc_components = collect(get_components(TwoTerminalLCCLine, sys))
    @test length(lcc_components) == 2

    # Run AC power flow
    pf_results = solve_power_flow(ACPowerFlow(), sys)
    @test !ismissing(pf_results)

    # Verify results structure
    @test haskey(pf_results, "bus_results")
    @test haskey(pf_results, "flow_results")
    @test haskey(pf_results, "lcc_results")

    # Verify LCC results have correct number of rows
    lcc_results = pf_results["lcc_results"]
    @test nrow(lcc_results) == 2

    # Verify all buses have valid voltage magnitudes
    bus_results = pf_results["bus_results"]
    @test all(bus_results.Vm .> 0.9)
    @test all(bus_results.Vm .< 1.1)
end
