@testset "PowerFlowData" begin
    sys = PSB.build_system(PSB.PSITestSystems, "c_sys14"; add_forecasts = false)
    @test PowerFlowData(ACPowerFlow{LUACPowerFlow}(; correct_bustypes = true), sys) isa
          PF.ACPowerFlowData
    @test PowerFlowData(
        ACPowerFlow{NewtonRaphsonACPowerFlow}(; correct_bustypes = true),
        sys,
    ) isa PF.ACPowerFlowData
    @test PowerFlowData(DCPowerFlow(; correct_bustypes = true), sys) isa PF.ABAPowerFlowData
    @test PowerFlowData(PTDFDCPowerFlow(; correct_bustypes = true), sys) isa
          PF.PTDFPowerFlowData
    @test PowerFlowData(vPTDFDCPowerFlow(; correct_bustypes = true), sys) isa
          PF.vPTDFPowerFlowData
end

@testset "PowerFlowData multiperiod" begin
    sys = PSB.build_system(PSB.PSITestSystems, "c_sys14"; add_forecasts = false)
    time_steps = 24
    @test PowerFlowData(
        DCPowerFlow(; time_steps = time_steps, correct_bustypes = true),
        sys,
    ) isa PF.ABAPowerFlowData
    @test PowerFlowData(
        PTDFDCPowerFlow(; time_steps = time_steps, correct_bustypes = true),
        sys,
    ) isa
          PF.PTDFPowerFlowData
    @test PowerFlowData(
        vPTDFDCPowerFlow(; time_steps = time_steps, correct_bustypes = true),
        sys,
    ) isa
          PF.vPTDFPowerFlowData
end

@testset "System <-> PowerFlowData round trip" begin
    # TODO currently only tested with ACPowerFlow
    # TODO test that update_system! errors if the PowerFlowData doesn't correspond to the system
    for ACSolver in AC_SOLVERS_TO_TEST
        @testset "AC Solver: $(ACSolver)" begin
            sys_original = build_system(PSISystems, "RTS_GMLC_DA_sys")
            set_units_base_system!(sys_original, PSY.UnitSystem.SYSTEM_BASE)
            data_original =
                PowerFlowData(
                    ACPowerFlow{ACSolver}(; correct_bustypes = true),
                    sys_original,
                )

            sys_modified = deepcopy(sys_original)
            set_units_base_system!(sys_modified, PSY.UnitSystem.SYSTEM_BASE)
            modify_rts_system!(sys_modified)
            data_modified =
                PowerFlowData(
                    ACPowerFlow{ACSolver}(; correct_bustypes = true),
                    sys_original,
                )
            modify_rts_power_flow!(data_modified)

            # update_system! with unmodified PowerFlowData should result in system that yields unmodified PowerFlowData
            # (NOTE does NOT necessarily yield original system due to power redistribution)
            sys_null_updated = deepcopy(sys_original)
            set_units_base_system!(sys_null_updated, PSY.UnitSystem.SYSTEM_BASE)
            PF.update_system!(sys_null_updated, data_original)
            data_null_updated =
                PowerFlowData(
                    ACPowerFlow{ACSolver}(; correct_bustypes = true),
                    sys_null_updated,
                )
            @test IS.compare_values(power_flow_match_fn, data_null_updated, data_original)

            # Modified versions should not be the same as unmodified versions
            @test !@test_logs((:error, r"values do not match"),
                match_mode = :any, min_level = Logging.Error,
                IS.compare_values(power_flow_match_fn, data_original, data_modified))
            @test !@test_logs((:error, r"values do not match"),
                match_mode = :any, min_level = Logging.Error,
                IS.compare_values(power_flow_match_fn, sys_original, sys_modified))

            # Constructing PowerFlowData from modified system should result in data_modified
            @test IS.compare_values(
                power_flow_match_fn,
                PowerFlowData(
                    ACPowerFlow{ACSolver}(; correct_bustypes = true),
                    sys_modified,
                ),
                data_modified)

            # The big one: update_system! with modified PowerFlowData should result in sys_modified,
            # modulo information that is inherently lost in the PowerFlowData representation
            sys_modify_updated = deepcopy(sys_original)
            PF.update_system!(sys_modify_updated, data_modified)
            sys_mod_redist = deepcopy(sys_modified)
            PF.update_system!(
                sys_mod_redist,
                PowerFlowData(
                    ACPowerFlow{ACSolver}(; correct_bustypes = true),
                    sys_mod_redist,
                ),
            )
            @test IS.compare_values(power_flow_match_fn, sys_modify_updated, sys_mod_redist)
        end
    end
end

"""Helper function that sets availability of all sources at a given bus."""
function set_availability_at_bus(
    sys::PSY.System,
    bus::PSY.ACBus,
    availability::Bool,
)
    set_available!.(
        PSY.get_components(
            d -> !isa(d, PSY.ElectricLoad) && get_number(get_bus(d)) == get_number(bus),
            PSY.StaticInjection,
            sys,
        ),
        (availability,),
    )
end

@testset "test correct_bustypes" begin
    sys = PSB.build_system(PSB.PSITestSystems, "c_sys14"; add_forecasts = false)
    buses = collect(PSY.get_components(PSY.ACBus, sys))
    # prevent against unknowningly having a PV bus with no available generators.
    for g in get_components(PSY.Generator, sys)
        set_available!(g, true)
    end
    first_pv = findfirst(bus -> PSY.get_bustype(bus) == PSY.ACBusTypes.PV, buses)
    pv_bus = buses[first_pv]
    set_availability_at_bus(sys, pv_bus, false)

    data_fixed = PF.PowerFlowData(PF.ACPowerFlow(; correct_bustypes = true), sys)
    pv_bus_row_fixed = PF.get_bus_lookup(data_fixed)[PSY.get_number(pv_bus)]
    # bus type in power flow data struct changes, but bus type in system shouldn't change.
    @test PF.get_bus_type(data_fixed)[pv_bus_row_fixed, 1] == PSY.ACBusTypes.PQ
    @test PSY.get_bustype(pv_bus) == PSY.ACBusTypes.PV

    for bus in buses
        if get_number(bus) != get_number(pv_bus)
            @test get_bustype(bus) == PF.get_bus_type(data_fixed)[
                PF.get_bus_lookup(data_fixed)[get_number(bus)],
                1,
            ]
        end
    end
end

@testset "clear_injection_data! resets NaN columns to flat start" begin
    sys = PSB.build_system(PSB.PSITestSystems, "c_sys14"; add_forecasts = false)
    data = PowerFlowData(ACPowerFlow(; correct_bustypes = true, time_steps = 2), sys)
    # Seed column 1 with NaNs, leave column 2 untouched.
    orig_angles_2 = copy(data.bus_angles[:, 2])
    orig_mag_2 = copy(data.bus_magnitude[:, 2])
    data.bus_angles[:, 1] .= NaN
    data.bus_magnitude[:, 1] .= NaN
    PF.clear_injection_data!(data)
    # NaN column reset to flat start.
    @test all(data.bus_angles[:, 1] .== 0.0)
    @test all(data.bus_magnitude[:, 1] .== 1.0)
    # Non-NaN column unchanged.
    @test data.bus_angles[:, 2] == orig_angles_2
    @test data.bus_magnitude[:, 2] == orig_mag_2
end

@testset "Wrong bus type" begin
    sys = PSB.build_system(PSB.PSITestSystems, "c_sys14"; add_forecasts = false)
    buses = collect(PSY.get_components(PSY.ACBus, sys))
    first_pv = findfirst(bus -> PSY.get_bustype(bus) == PSY.ACBusTypes.PV, buses)
    pv_bus = buses[first_pv]
    set_availability_at_bus(sys, pv_bus, true)
    @assert PSY.get_bustype(pv_bus) == PSY.ACBusTypes.PV
    # PV with no available generators => AC power flow errors.
    set_availability_at_bus(sys, pv_bus, false)
    @assert PSY.get_bustype(pv_bus) == PSY.ACBusTypes.PV
    @test_throws ArgumentError PF.PowerFlowData(
        PF.ACPowerFlow{NewtonRaphsonACPowerFlow}(),
        sys,
    )
    @assert PSY.get_bustype(pv_bus) == PSY.ACBusTypes.PV
    # on the other hand, DC power flow is fine: PV vs PQ doesn't matter.
    solve_power_flow(PF.DCPowerFlow(), sys, PF.FlowReporting.ARC_FLOWS)
    # No available generators at REF bus does error for DC power flow.
    ref_bus = findfirst(bus -> PSY.get_bustype(bus) == PSY.ACBusTypes.REF, buses)
    ref_bus = buses[ref_bus]
    set_availability_at_bus(sys, ref_bus, false)
    @test_throws ArgumentError PF.PowerFlowData(
        PF.DCPowerFlow(),
        sys,
    )
    set_availability_at_bus(sys, ref_bus, true)
    # change it to PQ: should work now.
    set_bustype!(pv_bus, PSY.ACBusTypes.PQ)
    @test PF.PowerFlowData(PF.ACPowerFlow{NewtonRaphsonACPowerFlow}(), sys) isa Any
end
