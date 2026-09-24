# End-to-end round trip of PowerFlowFileParser's synthetic remote voltage control cases: the
# RAW file is imported the way a user does it (parser -> OpenAPI document -> PowerSystems),
# the PowerSystems objects are checked, the system is written back with the PSS/E exporter,
# re-imported and checked again, and the control fields of the exported RAW are compared with
# the original record by record.

import PowerFlowFileParser

const _REMOTE_CONTROL_EXPORT_DIR = joinpath(BASE_DIR, "test", "test_exports")

"""Re-import an export with its metadata, the way PowerSystemCaseBuilder's reimport does."""
function _reimport_export(raw_path, metadata_path)
    return System(raw_path, Dict(JSON3.read(metadata_path, Dict)))
end

const _REMOTE_CONTROL_FIXTURE_DIR =
    joinpath(pkgdir(PowerFlowFileParser), "test", "fixtures")

_remote_control_fixture(version) =
    joinpath(_REMOTE_CONTROL_FIXTURE_DIR, "synthetic_$(version)_remote_control.raw")

_at_bus(components, number) =
    only(c for c in components if get_number(get_bus(c)) == number)

_circuit_between(sys, from, to, objective) = only(
    get_circuit(t) for t in get_components(TwoWindingTransformer, sys) if
    get_number(get_from(get_arc(t))) == from && get_number(get_to(get_arc(t))) == to &&
    get_control_objective(get_circuit(t)) == objective
)

"""Every voltage control field the fixture carries, on the PowerSystems objects."""
function _check_remote_control_fixture(sys::System)
    thermal = collect(get_components(ThermalStandard, sys))
    bus2_gens = filter(g -> get_number(get_bus(g)) == 2, thermal)
    g21 = only(filter(g -> !isnothing(get_remote_regulated_bus(g)), bus2_gens))
    g22 = only(filter(g -> isnothing(get_remote_regulated_bus(g)), bus2_gens))
    g41 = _at_bus(thermal, 4)
    g11 = _at_bus(thermal, 1)

    # IREG names the remote bus; IREG equal to the own bus and IREG = 0 both mean own-bus
    # control. VS is the setpoint at the regulated bus.
    @test get_number(get_remote_regulated_bus(g21)) == 3
    @test get_number(get_remote_regulated_bus(g41)) == 3
    @test isnothing(get_remote_regulated_bus(g22))
    @test isnothing(get_remote_regulated_bus(g11))
    @test get_number(get_regulated_bus(g22)) == 2
    @test get_voltage_setpoint(g21) == 1.02
    @test get_voltage_setpoint(g22) == 1.01

    # Two generators and a VSC converter hold bus 3 with unequal RMPCT.
    vsc = only(get_components(TwoTerminalVSCLine, sys))
    @test get_number(get_remote_regulated_bus_from(vsc)) == 3
    @test isnothing(get_remote_regulated_bus_to(vsc))
    group3 = only(get_supplemental_attributes(ReactivePowerSharing, g21))
    @test Set(get_associated_components(sys, group3)) == Set([g21, g41, vsc])
    # Ratios survive at the precision of the RMPCT record.
    @test get_weight(group3, g21) / get_weight(group3, g41) ≈ 60 / 40 rtol = 1e-6
    @test get_weight(group3, vsc) / get_weight(group3, g41) ≈ 50 / 40 rtol = 1e-6
    @test get_terminal(group3, vsc) == PSY.VoltageControlTerminal.FROM
    @test isempty(get_supplemental_attributes(ReactivePowerSharing, g22))
    @test isempty(get_supplemental_attributes(ReactivePowerSharing, g11))

    # A switched shunt with a remote target shares bus 7 with a FACTS device on that bus.
    shunts = collect(get_components(SwitchedAdmittance, sys))
    shunt6 = _at_bus(shunts, 6)
    shunt5 = _at_bus(shunts, 5)
    facts = only(get_components(FACTSControlDevice, sys))
    @test get_number(get_remote_regulated_bus(shunt6)) == 7
    @test get_number(get_remote_regulated_bus(shunt5)) == 2
    @test isnothing(get_remote_regulated_bus(facts))
    group7 = only(get_supplemental_attributes(ReactivePowerSharing, facts))
    @test Set(get_associated_components(sys, group7)) == Set([shunt6, facts])
    @test get_weight(group7, shunt6) == get_weight(group7, facts)

    # CONT of both signs, CR + jCX, and a winding whose objective regulates no voltage.
    local_circuit =
        _circuit_between(sys, 3, 6, PSY.TransformerControlObjective.VOLTAGE)
    @test get_number(get_regulated_bus(local_circuit)) == 3
    @test get_regulated_bus_side(local_circuit) ==
          PSY.TransformerRegulatedBusSide.CONTROLLING_WINDING
    @test get_load_drop_compensation(local_circuit, PSY.SU) ≈ 0.01 + 0.02im
    remote_circuit =
        _circuit_between(sys, 5, 6, PSY.TransformerControlObjective.VOLTAGE)
    @test get_number(get_regulated_bus(remote_circuit)) == 7
    @test get_regulated_bus_side(remote_circuit) ==
          PSY.TransformerRegulatedBusSide.OPPOSITE_WINDING
    @test iszero(get_load_drop_compensation(remote_circuit, PSY.SU))
    disabled_circuit =
        _circuit_between(sys, 6, 7, PSY.TransformerControlObjective.VOLTAGE_DISABLED)
    @test get_number(get_regulated_bus(disabled_circuit)) == 4
    @test get_regulated_bus_side(disabled_circuit) ==
          PSY.TransformerRegulatedBusSide.CONTROLLING_WINDING
    dc_circuit =
        _circuit_between(sys, 6, 7, PSY.TransformerControlObjective.CONTROL_OF_DC_LINE)
    @test isnothing(get_regulated_bus(dc_circuit))

    # ICR names the rectifier's commutating bus and IFR/ITR/IDR its tap transformer.
    lcc = only(get_components(TwoTerminalLCCLine, sys))
    @test get_number(get_rectifier_commutating_bus(lcc)) == 4
    @test isnothing(get_inverter_commutating_bus(lcc))
    @test get_circuit(get_rectifier_tap_transformer(lcc)) === dc_circuit
    @test isnothing(get_inverter_tap_transformer(lcc))
    return
end

"""
The voltage control fields of a parsed RAW file in a form that is invariant to what the
exporter may legitimately change: bus numbers are kept, circuit identifiers are replaced
by the transformer's control tuple, own-bus control is spelled as the own bus whether the
record says 0 or the bus itself, and RMPCT is normalized to each regulated bus's total, since
PSS/E reads the percentages relative to one another.
"""
function _raw_control_fields(path::AbstractString)
    pm = PowerFlowFileParser.parse_file(path)
    own_or_remote(remote, own) = iszero(remote) ? Int(own) : Int(remote)

    transformers = Dict{Tuple{Int, Int, String}, Tuple{Int, Int, Int, Float64, Float64}}()
    for d in values(pm["branch"])
        d["transformer"] || continue
        ckt = strip(String(d["source_id"][5]))
        transformers[(d["f_bus"], d["t_bus"], ckt)] =
            (d["f_bus"], d["t_bus"], Int(d["CONT1"]), d["CR1"], d["CX1"])
    end

    shares = Dict{Tuple, Tuple{Int, Float64}}()
    for d in values(pm["gen"])
        key = ("gen", Int(d["gen_bus"]), strip(String(d["source_id"][3])))
        shares[key] = (own_or_remote(d["regulated_bus_number"], d["gen_bus"]), d["rmpct"])
    end
    for d in values(pm["switched_shunt"])
        key = ("shunt", Int(d["shunt_bus"]))
        shares[key] = (own_or_remote(d["regulated_bus_number"], d["shunt_bus"]), d["rmpct"])
    end
    for d in values(pm["facts"])
        key = ("facts", Int(d["source_id"][2]))
        shares[key] =
            (own_or_remote(d["regulated_bus_number"], d["source_id"][2]), d["rmpct"])
    end
    for d in values(pm["vscline"])
        shares[("vsc", Int(d["f_bus"]), :from)] =
            (own_or_remote(d["remote_bus_number_from"], d["f_bus"]), d["rmpct_from"])
        shares[("vsc", Int(d["t_bus"]), :to)] =
            (own_or_remote(d["remote_bus_number_to"], d["t_bus"]), d["rmpct_to"])
    end
    totals = Dict{Int, Float64}()
    for (bus, rmpct) in values(shares)
        totals[bus] = get(totals, bus, 0.0) + rmpct
    end
    normalized = Dict(
        key => (bus, round(rmpct / totals[bus]; digits = 6)) for
        (key, (bus, rmpct)) in shares
    )

    tap_transformer(tuple) =
        if iszero(tuple[1])
            nothing
        else
            transformers[(Int(tuple[1]), Int(tuple[2]), strip(String(tuple[3])))]
        end
    lcc = only(values(pm["dcline"]))
    lcc_fields = (
        rectifier_bus = Int(lcc["rectifier_commutating_bus_number"]),
        inverter_bus = Int(lcc["inverter_commutating_bus_number"]),
        rectifier_tap = tap_transformer(lcc["rectifier_tap_transformer"]),
        inverter_tap = tap_transformer(lcc["inverter_tap_transformer"]),
    )
    return (;
        transformers = Set(values(transformers)),
        shares = normalized,
        lcc = lcc_fields,
    )
end

@testset "PSSE Exporter: remote voltage control fixture round trip ($version)" for version in
                                                                                   (
    :v33,
    :v35,
)
    fixture = _remote_control_fixture(version)
    sys = PowerSystemCaseBuilder.system_from_openapi(
        PowerFlowFileParser.PowerModelsData(fixture),
    )
    _check_remote_control_fixture(sys)

    export_location =
        joinpath(
            _REMOTE_CONTROL_EXPORT_DIR,
            string(version),
            "remote_voltage_control_fixture",
        )
    exporter = PSSEExporter(sys, version, export_location; write_comments = true)
    write_export(exporter, "fixture"; overwrite = true)
    raw_path, metadata_path = get_psse_export_paths(joinpath(export_location, "fixture"))
    _check_remote_control_fixture(_reimport_export(raw_path, metadata_path))

    original = _raw_control_fields(fixture)
    exported = _raw_control_fields(raw_path)
    @test exported.transformers == original.transformers
    @test exported.shares == original.shares
    @test exported.lcc == original.lcc
end
