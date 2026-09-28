# GA VSC parity (spec §3.8): every converter control mode — PQ, DC-voltage droop,
# AC-voltage at both ends, PV and REF AC terminals, multi-terminal DC — must match NR
# on the AC state, converter powers, and DC-node voltages.

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

# Converter with a lossy from terminal on the REF bus: the DC substep must settle P_c at
# the fixed REF |V|, and the AC-voltage/pv modes must not leak Q into the REF row.
function _ga_vsc_system_ref_terminal(; g = 45.0)
    sys = deepcopy(PSB.build_system(PSB.PSITestSystems, "c_sys14"; add_forecasts = false))
    pick(t) = first(
        sort!(
            collect(PSY.get_components(b -> PSY.get_bustype(b) == t, PSY.ACBus, sys));
            by = PSY.get_number,
        ),
    )
    arc = _get_or_make_arc(sys, pick(PSY.ACBusTypes.PQ), pick(PSY.ACBusTypes.REF))
    vsc = PSY.TwoTerminalVSCLine(;
        name = "ga_vsc_ref",
        available = true,
        arc = arc,
        active_power_flow = 0.3,
        rating = 2.0,
        active_power_limits_from = (min = -2.0, max = 2.0),
        active_power_limits_to = (min = -2.0, max = 2.0),
        g = g,
        dc_control_from = PSY.VSCDCControlModes.DC_VOLTAGE,
        ac_control_from = PSY.VSCACControlModes.AC_REACTIVE_POWER,
        dc_setpoint_from = 1.03,
        reactive_power_from = 0.0,
        dc_control_to = PSY.VSCDCControlModes.DC_POWER,
        ac_control_to = PSY.VSCACControlModes.AC_REACTIVE_POWER,
        dc_setpoint_to = 0.35,
        reactive_power_to = 0.05,
        converter_loss_to = PSY.LossCurve(
            PSY.QuadraticCurve(0.01, 0.02, 0.005),
            PSY.NaturalUnit(),
        ), input_basis = PSY.CU,
    )
    PSY.add_component!(sys, vsc)
    return sys
end

@testset "GA: VSC parity ($name)" for (name, build) in (
    ("pq", () -> _build_vsc_pq_system(; g = 50.0, p_set = 0.4, q_set = 0.1, vdc = 1.05)),
    ("droop", _ga_vsc_droop_system),
    ("ac_voltage", _ga_vsc_ac_voltage_system),
    ("pv_terminal", _vsc_system_pv_terminal),
    ("ref_terminal", _ga_vsc_system_ref_terminal),
    ("mtdc", _build_mtdc_system),
)
    data_nr, data_ga = ga_parity(
        build();
        pf_kwargs = (; solution_parameters = VSC_SOLUTION_PARAMETERS),
    )
    dn, dg = PF.get_dc_network(data_nr), PF.get_dc_network(data_ga)
    @test maximum(abs.(dn.p_c .- dg.p_c)) < 1e-6
    @test maximum(abs.(dn.q_c .- dg.q_c)) < 1e-6
    @test maximum(abs.(dn.node_vdc .- dg.node_vdc)) < 1e-6
end
