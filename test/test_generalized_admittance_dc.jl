# GA VSC DC substep (spec §3.8): the shared warm-start Q row must HOLD the AC-voltage
# converter's Q_c (the AC side owns it) instead of resetting it to q_set, so the DC-tail
# Newton reproduces the NR-converged converter and DC-node states.

function _ga_vsc_droop_system()
    sys = deepcopy(PSB.build_system(PSB.PSITestSystems, "c_sys14"; add_forecasts = false))
    pq = sort!(
        collect(
            PSY.get_components(
                b -> PSY.get_bustype(b) == PSY.ACBusTypes.PQ,
                PSY.ACBus,
                sys,
            ),
        );
        by = PSY.get_number,
    )
    arc = _get_or_make_arc(sys, pq[1], pq[2])
    vsc = PSY.TwoTerminalVSCLine(;
        name = "ga_vsc_droop",
        available = true,
        arc = arc,
        active_power_flow = 0.3,
        rating = 2.0,
        active_power_limits_from = (min = -2.0, max = 2.0),
        active_power_limits_to = (min = -2.0, max = 2.0),
        g = 50.0,
        dc_control_from = PSY.VSCDCControlModes.DC_VOLTAGE_DROOP,
        dc_voltage_droop_from = 0.02,
        dc_setpoint_from = 1.05,
        reactive_power_from = 0.0,
        dc_control_to = PSY.VSCDCControlModes.DC_VOLTAGE_DROOP,
        dc_voltage_droop_to = 0.03,
        dc_setpoint_to = 1.03,
        reactive_power_to = 0.0,
        converter_loss_to = PSY.LossCurve(
            PSY.QuadraticCurve(0.01, 0.02, 0.005),
            PSY.NaturalUnit(),
        ), input_basis = PSY.CU,
    )
    PSY.add_component!(sys, vsc)
    return sys
end

# Both terminals pin AC voltage (from = ControlVdcQ, to = ControlPVac): both Q rows must hold.
function _ga_vsc_ac_voltage_system()
    sys = deepcopy(PSB.build_system(PSB.PSITestSystems, "c_sys14"; add_forecasts = false))
    pq = sort!(
        collect(
            PSY.get_components(
                b -> PSY.get_bustype(b) == PSY.ACBusTypes.PQ,
                PSY.ACBus,
                sys,
            ),
        );
        by = PSY.get_number,
    )
    arc = _get_or_make_arc(sys, pq[1], pq[2])
    vsc = PSY.TwoTerminalVSCLine(;
        name = "ga_vsc_av",
        available = true,
        arc = arc,
        active_power_flow = 0.3,
        rating = 2.0,
        active_power_limits_from = (min = -2.0, max = 2.0),
        active_power_limits_to = (min = -2.0, max = 2.0),
        g = 50.0,
        dc_control_from = PSY.VSCDCControlModes.DC_VOLTAGE,
        ac_control_from = PSY.VSCACControlModes.AC_VOLTAGE,
        dc_setpoint_from = 1.05,
        ac_setpoint_from = 1.01,
        dc_control_to = PSY.VSCDCControlModes.DC_POWER,
        ac_control_to = PSY.VSCACControlModes.AC_VOLTAGE,
        dc_setpoint_to = 0.25,
        ac_setpoint_to = 1.0, input_basis = PSY.CU,
    )
    PSY.add_component!(sys, vsc)
    return sys
end

@testset "GA: VSC warm start holds AC-voltage Q and reproduces NR ($name)" for (
    name,
    build,
) in
                                                                               (
    ("droop", _ga_vsc_droop_system),
    ("ac_voltage", _ga_vsc_ac_voltage_system))
    sys = build()
    pf = ACPowerFlow{NewtonRaphsonACPowerFlow}(;
        solution_parameters = VSC_SOLUTION_PARAMETERS,
    )
    data_nr = PowerFlowData(pf, sys)
    @test solve_power_flow!(data_nr)
    dn = PF.get_dc_network(data_nr)
    data = PowerFlowData(pf, sys)
    dcn = PF.get_dc_network(data)
    dcn.q_c[:, 1] .= dn.q_c[:, 1]
    PF._vsc_warm_start!(dcn, data_nr.bus_magnitude[:, 1], 1; tol = 1e-12)
    @test maximum(abs.(dcn.p_c[:, 1] .- dn.p_c[:, 1])) < 1e-8
    @test maximum(abs.(dcn.node_vdc[:, 1] .- dn.node_vdc[:, 1])) < 1e-8
    @test maximum(abs.(dcn.q_c[:, 1] .- dn.q_c[:, 1])) < 1e-8
end
