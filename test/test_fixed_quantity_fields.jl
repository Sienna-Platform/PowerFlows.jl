# Mode-selected control fields: each controlled quantity has its own `Union{Nothing, T}`
# field, and the device's control mode selects which one is populated.

function _two_bus_lcc_system()
    sys = System(100.0)
    b1 = _add_simple_bus!(sys, 1, ACBusTypes.REF, 230.0)
    b2 = _add_simple_bus!(sys, 2, ACBusTypes.PQ, 230.0)
    _add_simple_source!(sys, b1, 0.0, 0.0)
    _add_simple_load!(sys, b2, 0.1, 0.05)
    _add_simple_line!(sys, b1, b2, 0.01, 0.10, 0.0)
    lcc = _add_simple_lcc!(sys, b1, b2, 0.01, 0.01, 0.01)
    return sys, lcc
end

@testset "LCC transfer setpoint follows the control mode" begin
    sys, lcc = _two_bus_lcc_system()
    @test PF._lcc_power_transfer_setpoint(lcc) ≈ 0.5
    @test PF._lcc_export_setvl(lcc) ≈ 50.0

    PSY.set_power_transfer_setpoint!(lcc, -0.25 * PSY.SU)
    @test PF._lcc_power_transfer_setpoint(lcc) ≈ -0.25

    PSY.set_control_mode!(lcc, PSY.LCCControlMode.BLOCKED)
    PSY.set_power_transfer_setpoint!(lcc, nothing)
    @test PF._lcc_power_transfer_setpoint(lcc) == 0.0
    @test PF._lcc_export_setvl(lcc) == 0.0
    data = PowerFlowData(ACPowerFlow(), sys)
    @test all(iszero, PF.get_lcc_p_set(data))

    PSY.set_control_mode!(lcc, PSY.LCCControlMode.CURRENT)
    PSY.set_current_transfer_setpoint!(lcc, 500.0)
    @test PF._lcc_export_setvl(lcc) == 500.0
    @test_throws r"CURRENT" PF._lcc_power_transfer_setpoint(lcc)
    @test_throws ArgumentError PowerFlowData(ACPowerFlow(), sys)

    PSY.set_control_mode!(lcc, PSY.LCCControlMode.POWER)
    @test_throws r"power_transfer_setpoint" PF._lcc_power_transfer_setpoint(lcc)
end

@testset "VSC lowering errors when the selected setpoint is nothing" begin
    for (clear!, field) in (
        (vsc -> PSY.set_dc_power_setpoint_to!(vsc, nothing), "dc_power_setpoint"),
        (vsc -> PSY.set_dc_voltage_setpoint_from!(vsc, nothing), "dc_voltage_setpoint"),
    )
        sys = _build_vsc_pq_system()
        clear!(only(PSY.get_components(PSY.TwoTerminalVSCLine, sys)))
        pf = ACPowerFlow{NewtonRaphsonACPowerFlow}(;
            solution_parameters = VSC_SOLUTION_PARAMETERS,
        )
        @test_throws Regex(field) PowerFlowData(pf, sys)
    end

    sys = _build_vsc_pq_system(;
        ac_control_to = PSY.VSCACControlModes.AC_VOLTAGE,
        power_factor_setpoint_to = nothing,
        ac_voltage_setpoint_to = 1.02,
    )
    pf = ACPowerFlow{NewtonRaphsonACPowerFlow}(;
        solution_parameters = VSC_SOLUTION_PARAMETERS,
    )
    dcn = PF.get_dc_network(PowerFlowData(pf, sys))
    @test dcn.vac_set[2, 1] ≈ 1.02
    vsc = only(PSY.get_components(PSY.TwoTerminalVSCLine, sys))
    @test PF._vsc_export_acset(vsc, :to) ≈ 1.02
    @test PF._vsc_export_acset(vsc, :from) ≈ PSY.get_power_factor_setpoint_from(vsc)
    PSY.set_ac_voltage_setpoint_to!(vsc, nothing)
    @test_throws r"ac_voltage_setpoint" PowerFlowData(pf, sys)
end

@testset "PSS/E export writes the control band the objective selects" begin
    arc = Arc(; from = ACBus(nothing), to = ACBus(nothing))
    circuit(; kwargs...) =
        TransformerCircuit(; available = true, arc = arc, kwargs..., input_basis = PSY.CU)

    bands = PF._circuit_control_bands(circuit())
    @test bands.rm == (min = PF.PSSE_DEFAULT, max = PF.PSSE_DEFAULT)
    @test bands.vm == (min = PF.PSSE_DEFAULT, max = PF.PSSE_DEFAULT)

    bands = PF._circuit_control_bands(
        circuit(;
            control_objective = PSY.TransformerControlObjective.VOLTAGE,
            tap_ratio_limits = (min = 0.9, max = 1.1),
            controlled_voltage_limits = (min = 0.98, max = 1.02),
        ),
    )
    @test bands.rm == (min = 0.9, max = 1.1)
    @test bands.vm == (min = 0.98, max = 1.02)

    # The MVAr target band is written in natural units: 0.2 p.u. on the 100 MVA base.
    bands = PF._circuit_control_bands(
        circuit(;
            control_objective = PSY.TransformerControlObjective.REACTIVE_POWER_FLOW,
            tap_ratio_limits = (min = 0.9, max = 1.1),
            controlled_reactive_power_flow_limits = (min = -0.2, max = 0.2),
        ),
    )
    @test bands.rm == (min = 0.9, max = 1.1)
    @test bands.vm.min ≈ -20.0
    @test bands.vm.max ≈ 20.0

    # A band the objective selects but the data leaves unset writes blanks.
    bands = PF._circuit_control_bands(
        circuit(;
            control_objective = PSY.TransformerControlObjective.ACTIVE_POWER_FLOW,
            phase_angle_limits = (min = deg2rad(-10), max = deg2rad(10)),
        ),
    )
    @test bands.rm.min ≈ -10.0
    @test bands.rm.max ≈ 10.0
    @test bands.vm == (min = PF.PSSE_DEFAULT, max = PF.PSSE_DEFAULT)
end

@testset "PSS/E export writes the switched-shunt band the mode selects" begin
    shunt(mode; kwargs...) = SwitchedAdmittance(;
        name = "sa",
        available = true,
        bus = ACBus(nothing),
        Y_increase = [0.05im],
        control_mode = mode,
        kwargs...,
    )
    @test isnothing(PF._switched_shunt_band(shunt(PSY.SwitchedAdmittanceControlMode.FIXED)))
    @test PF._switched_shunt_band(
        shunt(
            PSY.SwitchedAdmittanceControlMode.DISCRETE_VOLTAGE;
            voltage_limits = (min = 0.97, max = 1.03),
        ),
    ) == (min = 0.97, max = 1.03)
    @test PF._switched_shunt_band(
        shunt(
            PSY.SwitchedAdmittanceControlMode.DISCRETE_REACTIVE_PLANT;
            voltage_limits = (min = 0.97, max = 1.03),
            reactive_power_range_limits = (min = 0.2, max = 0.8),
        ),
    ) == (min = 0.2, max = 0.8)
end

@testset "PSS/E export writes a phase-angle correction curve in degrees" begin
    curve = IS.PiecewiseLinearData([
        (x = deg2rad(-30.0), y = 1.2),
        (x = deg2rad(30.0), y = 1.1),
    ])
    angle_table = ImpedanceCorrectionData(;
        table_number = 1,
        phase_angle_correction_curve = curve,
        transformer_winding = WindingCategory.TR2W_WINDING,
        transformer_control_mode = ImpedanceCorrectionTransformerControlMode.PHASE_SHIFT_ANGLE,
    )
    points = PF._icd_export_points(angle_table)
    @test [p.x for p in points] ≈ [-30.0, 30.0]
    @test [p.y for p in points] ≈ [1.2, 1.1]

    tap_table = ImpedanceCorrectionData(;
        table_number = 2,
        tap_ratio_correction_curve = IS.PiecewiseLinearData([
            (x = 0.9, y = 1.05), (x = 1.1, y = 0.95),
        ]),
        transformer_winding = WindingCategory.TR2W_WINDING,
        transformer_control_mode = ImpedanceCorrectionTransformerControlMode.TAP_RATIO,
    )
    points = PF._icd_export_points(tap_table)
    @test [p.x for p in points] ≈ [0.9, 1.1]
    @test [p.y for p in points] ≈ [1.05, 0.95]
end
