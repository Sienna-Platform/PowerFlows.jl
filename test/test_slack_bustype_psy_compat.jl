# PSY does not coerce `ACBusTypes.SLACK` to REF in the `ACBus` constructor, so a SLACK
# designation (PSS/E area-interchange ISW bus) reaches PowerFlows intact. These tests pin
# the three places that read the RAW `PSY.get_bustype` and so must understand it.

# Bus 2 of c_sys14 is a PV bus with a generator, i.e. a legal ISW bus.
const _SLACK_COMPAT_BUS = "Bus 2"

function _two_area_system_with_slack()
    sys = _make_two_area_system()
    PSY.set_bustype!(
        PSY.get_component(PSY.ACBus, sys, _SLACK_COMPAT_BUS),
        PSY.ACBusTypes.SLACK,
    )
    return sys
end

@testset "SLACK write-back predicate" begin
    # The normalization SLACK -> PV must not be written back, or the ISW designation is lost.
    @test !PF._bustype_write_back_needed(PSY.ACBusTypes.SLACK, PSY.ACBusTypes.PV)
    # A SLACK bus demoted to PQ mirrors the PV -> PQ Q-limit flip and is still recorded.
    @test PF._bustype_write_back_needed(PSY.ACBusTypes.SLACK, PSY.ACBusTypes.PQ)
    # Ordinary flips are untouched by the SLACK carve-out.
    @test PF._bustype_write_back_needed(PSY.ACBusTypes.PV, PSY.ACBusTypes.PQ)
    @test PF._bustype_write_back_needed(PSY.ACBusTypes.REF, PSY.ACBusTypes.PQ)
    @test !PF._bustype_write_back_needed(PSY.ACBusTypes.PV, PSY.ACBusTypes.PV)
    @test !PF._bustype_write_back_needed(PSY.ACBusTypes.REF, PSY.ACBusTypes.REF)
end

@testset "SLACK bustype survives a solve write-back" begin
    sys = _two_area_system_with_slack()
    bus = PSY.get_component(PSY.ACBus, sys, _SLACK_COMPAT_BUS)
    pf = PF.ACPowerFlow{NewtonRaphsonACPowerFlow}(; area_interchange_control = true)
    @test solve_and_store_power_flow!(pf, sys)
    # Without the carve-out this is PV, and `_area_slack_buses` would find no area slack on
    # the next PowerFlowData build -- silently de-enrolling the area from interchange control.
    @test PSY.get_bustype(bus) == PSY.ACBusTypes.SLACK
end

@testset "SLACK bus contributes headroom to distributed slack" begin
    pv_sys = _make_two_area_system()
    slack_sys = _two_area_system_with_slack()
    pf = PF.ACPowerFlow{NewtonRaphsonACPowerFlow}(;
        distribute_slack_proportional_to_headroom = true,
    )
    pv_data = PF.PowerFlowData(pf, pv_sys)
    slack_data = PF.PowerFlowData(pf, slack_sys)

    bus_number = PSY.get_number(
        PSY.get_component(PSY.ACBus, pv_sys, _SLACK_COMPAT_BUS),
    )
    pv_ix = PF.get_bus_lookup(pv_data)[bus_number]
    slack_ix = PF.get_bus_lookup(slack_data)[bus_number]

    # Guard against a vacuous comparison if the fixture ever loses its headroom.
    @test pv_data.bus_active_power_range[pv_ix, 1] > 0.0
    # Designating the bus SLACK must not change its headroom: it still normalizes to PV.
    @test slack_data.bus_active_power_range[slack_ix, 1] ==
          pv_data.bus_active_power_range[pv_ix, 1]
    @test slack_data.bus_slack_participation_factors[slack_ix, 1] > 0.0
end

@testset "SLACK exports as a PSS/E PV bus" begin
    # IDE=3 would emit a second swing bus; the ISW belongs in the AREA INTERCHANGE record.
    @test PF.PSSE_BUS_TYPE_MAP[PSY.ACBusTypes.SLACK] ==
          PF.PSSE_BUS_TYPE_MAP[PSY.ACBusTypes.PV]
end
