# The PSS/E writer carries every remote and shared voltage control field: IREG/VS/RMPCT on
# generators, the signed CONT and CR/CX on transformer windings, ICR and IFR/ITR/IDR on the
# two-terminal DC line, REMOT/RMPCT on VSC converters, FCREG/REMOT on FACTS and SWREG/SWREM on
# switched shunts. Each case exports on both versions and re-imports through the parser, so the
# assertions are on the PowerSystems objects a user gets back.

const _REMOTE_CONTROL_EXPORT_DIR = joinpath(BASE_DIR, "test", "test_exports")

"""Re-import an export with its metadata, the way PowerSystemCaseBuilder's reimport does."""
function _reimport_export(raw_path, metadata_path)
    return System(raw_path, Dict(JSON3.read(metadata_path, Dict)))
end

"""A voltage-controlling two-winding transformer between `from_bus` and `to_bus`."""
function _add_control_transformer!(
    sys::System,
    name::AbstractString,
    from_bus::ACBus,
    to_bus::ACBus;
    control_objective = PSY.TransformerControlObjective.VOLTAGE,
    regulated_bus = nothing,
    regulated_bus_side = nothing,
    load_drop_compensation = 0.0 + 0.0im,
)
    tx = TwoWindingTransformer(;
        name = name,
        circuit = TransformerCircuit(;
            available = true,
            arc = Arc(; from = from_bus, to = to_bus),
            r = 0.001,
            x = 0.05,
            tap = 1.0,
            rating = 2.0,
            base_power = 100.0,
            control_objective = control_objective,
            regulated_bus = regulated_bus,
            regulated_bus_side = regulated_bus_side,
            load_drop_compensation = load_drop_compensation,
            control_limits = (min = 0.9, max = 1.1),
            controlled_quantity_limits = (min = 0.95, max = 1.05),
            input_basis = CU,
        ),
        input_basis = CU,
    )
    add_component!(sys, tx)
    return tx
end

"""The network of PowerFlowFileParser's synthetic remote-control fixtures, built directly."""
function _remote_control_system()
    sys = System(100.0)
    b1 = _add_simple_bus!(sys, 1, ACBusTypes.REF, 138, 1.0, 0.0)
    b2 = _add_simple_bus!(sys, 2, ACBusTypes.PV, 138, 1.01, 0.0)
    b3 = _add_simple_bus!(sys, 3, ACBusTypes.PQ, 138, 1.02, 0.0)
    b4 = _add_simple_bus!(sys, 4, ACBusTypes.PV, 138, 1.0, 0.0)
    b5 = _add_simple_bus!(sys, 5, ACBusTypes.PQ, 138, 1.0, 0.0)
    b6 = _add_simple_bus!(sys, 6, ACBusTypes.PQ, 138, 1.0, 0.0)
    b7 = _add_simple_bus!(sys, 7, ACBusTypes.PQ, 138, 1.0, 0.0)
    _add_simple_thermal_standard!(sys, b1, 0.8, 0.0)
    _add_simple_load!(sys, b3, 0.5, 0.1)
    _add_simple_load!(sys, b5, 0.3, 0.05)
    _add_simple_load!(sys, b7, 0.2, 0.05)
    _add_simple_line!(sys, b1, b2, 0.01, 0.1, 0.02)
    _add_simple_line!(sys, b2, b3, 0.01, 0.1, 0.02)
    _add_simple_line!(sys, b3, b4, 0.01, 0.1, 0.02)
    _add_simple_line!(sys, b4, b5, 0.01, 0.1, 0.02)
    _add_simple_line!(sys, b5, b7, 0.01, 0.1, 0.02)

    g2 = _add_simple_thermal_standard!(sys, b2, 0.3, 0.0)
    set_remote_regulated_bus!(g2, b3)
    set_voltage_setpoint!(g2, 1.03)
    g4 = _add_simple_thermal_standard!(sys, b4, 0.3, 0.0)
    set_remote_regulated_bus!(g4, b3)
    set_voltage_setpoint!(g4, 1.03)

    _add_control_transformer!(
        sys, "xfmr_3_6", b3, b6; regulated_bus = b3,
        load_drop_compensation = 0.01 + 0.02im,
    )
    _add_control_transformer!(
        sys, "xfmr_5_6", b5, b6; regulated_bus = b7,
        regulated_bus_side = PSY.TransformerRegulatedBusSide.OPPOSITE_WINDING,
    )
    dc_tap = _add_control_transformer!(
        sys, "xfmr_6_7_dc", b6, b7;
        control_objective = PSY.TransformerControlObjective.CONTROL_OF_DC_LINE,
    )
    _add_control_transformer!(
        sys, "xfmr_6_7_v", b6, b7;
        control_objective = PSY.TransformerControlObjective.VOLTAGE_DISABLED,
        regulated_bus = b4,
        regulated_bus_side = PSY.TransformerRegulatedBusSide.CONTROLLING_WINDING,
    )
    lcc = _add_simple_lcc!(sys, b5, b7, 5.0, 0.1, 0.1)
    set_rectifier_commutating_bus!(lcc, b4)
    set_rectifier_tap_transformer!(lcc, dc_tap)

    vsc = _add_simple_vsc!(sys, b4, b6)
    set_dc_control_from!(vsc, PSY.VSCDCControlModes.DC_VOLTAGE)
    set_dc_setpoint_from!(vsc, 1.0)
    set_rated_dc_voltage!(vsc, 150.0)
    set_ac_control_from!(vsc, PSY.VSCACControlModes.AC_VOLTAGE)
    set_ac_setpoint_from!(vsc, 1.03)
    set_remote_regulated_bus_from!(vsc, b3)

    facts = FACTSControlDevice(;
        name = "facts_7", available = true, bus = b7,
        control_mode = PSY.FACTSOperationModes.NML, voltage_setpoint = 1.0,
        input_basis = CU,
    )
    set_max_shunt_current!(facts, 200.0 * u"MVA")
    add_component!(sys, facts)
    shunt = SwitchedAdmittance(;
        name = "shunt_6", available = true, bus = b6, number_engaged = [1],
        number_of_steps = [2], Y_increase = [0.0 + 0.05im],
        admittance_limits = (min = 0.95, max = 1.05),
        control_mode = PSY.SwitchedAdmittanceControlMode.DISCRETE_VOLTAGE,
        remote_regulated_bus = b7,
    )
    add_component!(sys, shunt)

    sharing3 = ReactivePowerSharing(; name = "share3")
    add_supplemental_attribute!(sys, g2, sharing3; weight = 0.6)
    add_supplemental_attribute!(sys, g4, sharing3; weight = 0.4)
    add_supplemental_attribute!(
        sys, vsc, sharing3; weight = 0.5, terminal = PSY.VoltageControlTerminal.FROM,
    )
    sharing7 = ReactivePowerSharing(; name = "share7")
    add_supplemental_attribute!(sys, shunt, sharing7; weight = 2.0)
    add_supplemental_attribute!(sys, facts, sharing7; weight = 1.0)
    return sys
end

_number_of(::Nothing) = nothing
_number_of(bus::ACBus) = get_number(bus)

_circuit_named(sys, name) = get_circuit(get_component(TwoWindingTransformer, sys, name))

@testset "PSSE Exporter: remote and shared voltage control round trip ($version)" for version in
                                                                                      (
    :v33,
    :v35,
)
    begin
        sys = _remote_control_system()
        export_location =
            joinpath(_REMOTE_CONTROL_EXPORT_DIR, string(version), "remote_voltage_control")
        exporter = PSSEExporter(sys, version, export_location; write_comments = true)
        write_export(exporter, "basic"; overwrite = true)
        raw_path, metadata_path =
            get_psse_export_paths(joinpath(export_location, "basic"))
        sys2 = _reimport_export(raw_path, metadata_path)

        # IREG, VS and RMPCT on the generators
        g2 = get_component(ThermalStandard, sys2, "thermal_standard_2")
        g4 = get_component(ThermalStandard, sys2, "thermal_standard_4")
        g1 = get_component(ThermalStandard, sys2, "thermal_standard_1")
        @test _number_of(get_remote_regulated_bus(g2)) == 3
        @test _number_of(get_remote_regulated_bus(g4)) == 3
        @test isnothing(get_remote_regulated_bus(g1))
        @test get_voltage_setpoint(g2) == 1.03
        @test get_voltage_setpoint(g1) == 1.0
        group3 = only(get_supplemental_attributes(ReactivePowerSharing, g2))
        vsc = only(get_components(TwoTerminalVSCLine, sys2))
        @test Set(get_associated_components(sys2, group3)) == Set([g2, g4, vsc])
        # RMPCT is the share in percent, so the weights come back normalized to one, at the
        # precision of the record
        @test get_weight(group3, g2) ≈ 0.6 / 1.5 atol = 1e-6
        @test get_weight(group3, g4) ≈ 0.4 / 1.5 atol = 1e-6
        @test get_weight(group3, vsc) ≈ 0.5 / 1.5 atol = 1e-6
        @test get_terminal(group3, vsc) == PSY.VoltageControlTerminal.FROM
        @test _number_of(get_remote_regulated_bus_from(vsc)) == 3
        @test isnothing(get_remote_regulated_bus_to(vsc))

        # CONT with its sign, CR + jCX
        local_circuit = _circuit_named(sys2, "xfmr_3_6")
        @test _number_of(get_regulated_bus(local_circuit)) == 3
        @test get_regulated_bus_side(local_circuit) ==
              PSY.TransformerRegulatedBusSide.CONTROLLING_WINDING
        @test get_load_drop_compensation(local_circuit, PSY.SU) ≈ 0.01 + 0.02im
        remote_circuit = _circuit_named(sys2, "xfmr_5_6")
        @test _number_of(get_regulated_bus(remote_circuit)) == 7
        @test get_regulated_bus_side(remote_circuit) ==
              PSY.TransformerRegulatedBusSide.OPPOSITE_WINDING
        disabled_circuit = _circuit_named(sys2, "xfmr_6_7_v")
        @test _number_of(get_regulated_bus(disabled_circuit)) == 4
        @test get_regulated_bus_side(disabled_circuit) ==
              PSY.TransformerRegulatedBusSide.CONTROLLING_WINDING
        dc_tap = get_component(TwoWindingTransformer, sys2, "xfmr_6_7_dc")
        @test isnothing(get_regulated_bus(get_circuit(dc_tap)))

        # ICR and IFR/ITR/IDR
        lcc = only(get_components(TwoTerminalLCCLine, sys2))
        @test _number_of(get_rectifier_commutating_bus(lcc)) == 4
        @test isnothing(get_inverter_commutating_bus(lcc))
        @test get_rectifier_tap_transformer(lcc) === dc_tap
        @test isnothing(get_inverter_tap_transformer(lcc))

        # FCREG/REMOT, SWREG/SWREM and their shares
        facts = only(get_components(FACTSControlDevice, sys2))
        shunt = only(get_components(SwitchedAdmittance, sys2))
        @test isnothing(get_remote_regulated_bus(facts))
        @test _number_of(get_remote_regulated_bus(shunt)) == 7
        group7 = only(get_supplemental_attributes(ReactivePowerSharing, facts))
        @test Set(get_associated_components(sys2, group7)) == Set([facts, shunt])
        @test get_weight(group7, shunt) ≈ 2.0 / 3.0 atol = 1e-6
        @test get_weight(group7, facts) ≈ 1.0 / 3.0 atol = 1e-6
    end
end

@testset "PSSE Exporter: a three-winding circuit regulating its star bus writes CONT 0 ($version)" for version in
                                                                                                       (
    :v33,
    :v35,
)
    sys = System(100.0)
    b1 = _add_simple_bus!(sys, 1, ACBusTypes.REF, 230, 1.0, 0.0)
    b2 = _add_simple_bus!(sys, 2, ACBusTypes.PQ, 138, 1.0, 0.0)
    b3 = _add_simple_bus!(sys, 3, ACBusTypes.PQ, 69, 1.0, 0.0)
    _add_simple_thermal_standard!(sys, b1, 0.5, 0.0)
    _add_simple_load!(sys, b2, 0.2, 0.05)
    _add_simple_load!(sys, b3, 0.1, 0.02)
    tx = _add_simple_transformer_3w!(sys, b1, b2, b3, 1001; available_tertiary = true)
    # The PSS/E record carries the pairwise impedances, which the helper leaves unset.
    for (setter, value) in (
        (set_r_12!, 0.02), (set_x_12!, 0.14), (set_r_23!, 0.02), (set_x_23!, 0.11),
        (set_r_31!, 0.02), (set_x_31!, 0.13),
    )
        setter(tx, value * CU)
    end
    set_base_power_12!(tx, 100.0)
    set_base_power_23!(tx, 100.0)
    set_base_power_31!(tx, 100.0)
    circuit = get_primary_circuit(tx)
    # The star bus has no PSS/E number; PSS/E spells "the winding's far end" as CONT 0.
    set_control_objective!(circuit, PSY.TransformerControlObjective.VOLTAGE)
    set_regulated_bus!(circuit, get_star_bus(tx))

    export_location =
        joinpath(
            _REMOTE_CONTROL_EXPORT_DIR,
            string(version),
            "three_winding_star_regulation",
        )
    exporter = PSSEExporter(sys, version, export_location; write_comments = true)
    write_export(exporter, "basic"; overwrite = true)
    raw_path, metadata_path = get_psse_export_paths(joinpath(export_location, "basic"))
    winding_records = filter(
        line -> startswith(line, "1.0"),
        readlines(raw_path),
    )
    @test !isempty(winding_records)
    cont_index = version == :v35 ? 17 : 8
    @test all(strip.(split(record, ","))[cont_index] == "0" for record in winding_records)

    sys2 = _reimport_export(raw_path, metadata_path)
    tx2 = only(get_components(ThreeWindingTransformer, sys2))
    @test get_regulated_bus(get_primary_circuit(tx2)) === get_star_bus(tx2)
end

@testset "PSSE Exporter: update_exporter! keeps the sharing groups" begin
    sys = _remote_control_system()
    export_location =
        joinpath(_REMOTE_CONTROL_EXPORT_DIR, "v33", "remote_voltage_control_update")
    exporter = PSSEExporter(sys, :v33, export_location)
    write_export(exporter, "basic"; overwrite = true)
    update_exporter!(exporter, sys)
    write_export(exporter, "updated"; overwrite = true)
    sys2 = _reimport_export(get_psse_export_paths(joinpath(export_location, "updated"))...)
    # Without the groups every RMPCT is written as 100, which re-imports as equal shares.
    g2 = get_component(ThermalStandard, sys2, "thermal_standard_2")
    g4 = get_component(ThermalStandard, sys2, "thermal_standard_4")
    group3 = only(get_supplemental_attributes(ReactivePowerSharing, g2))
    @test get_weight(group3, g2) ≈ 0.6 / 1.5 atol = 1e-6
    @test get_weight(group3, g4) ≈ 0.4 / 1.5 atol = 1e-6
    shunt = only(get_components(SwitchedAdmittance, sys2))
    group7 = only(get_supplemental_attributes(ReactivePowerSharing, shunt))
    @test get_weight(group7, shunt) ≈ 2.0 / 3.0 atol = 1e-6
end
