# A series capacitor inserted into "Line1" of c_sys14: the line `from -> mid`, then the
# compensator `mid -> to`. The line's charging is removed in both cases so that only the
# series impedance differs between the split and the folded line.

const COMPENSATION = -0.3  # capacitor reactance as a fraction of the line's

function uncharged_c_sys14()
    sys = PSB.build_system(PSB.PSITestSystems, "c_sys14")
    line = PSY.get_component(PSY.Line, sys, "Line1")
    PSY.set_b!(line, (from = 0.0 * u"SU", to = 0.0 * u"SU"))
    return sys, line
end

function compensated_c_sys14()
    sys, line = uncharged_c_sys14()
    from, to = PSY.get_from(PSY.get_arc(line)), PSY.get_to(PSY.get_arc(line))
    mid = PSY.ACBus(;
        number = 100, name = "mid", available = true, bustype = PSY.ACBusTypes.PQ,
        angle = 0.0, magnitude = 1.0, voltage_limits = (min = 0.9, max = 1.1),
        base_voltage = PSY.get_base_voltage(from), input_basis = u"CU",
    )
    PSY.add_component!(sys, mid)
    line_arc, compensator_arc =
        PSY.Arc(; from = from, to = mid), PSY.Arc(; from = mid, to = to)
    PSY.add_component!(sys, line_arc)
    PSY.add_component!(sys, compensator_arc)
    PSY.set_arc!(line, line_arc)
    PSY.add_component!(
        sys,
        PSY.SeriesCompensator(;
            name = "compensator", available = true, active_power_flow = 0.0,
            reactive_power_flow = 0.0, arc = compensator_arc, r = 0.0,
            x = COMPENSATION * PSY.get_x(line, u"SU"), rating = PSY.get_rating(line, u"SU"),
            input_basis = u"CU",
        ),
    )
    return sys
end

function folded_c_sys14()
    sys, line = uncharged_c_sys14()
    PSY.set_x!(line, (1 + COMPENSATION) * PSY.get_x(line, u"SU") * u"SU")
    return sys
end

bus_voltages(results) = Dict(
    row.bus_number => (row.Vm, row.θ) for row in eachrow(results["bus_results"]))

@testset "AC power flow through a FIXED SeriesCompensator matches the folded line" begin
    compensated = bus_voltages(solve_power_flow(ACPowerFlow(), compensated_c_sys14()))
    folded = bus_voltages(solve_power_flow(ACPowerFlow(), folded_c_sys14()))
    # Two independent solves agree to the solver's convergence tolerance.
    for (bus, (vm, θ)) in folded
        @test compensated[bus][1] ≈ vm atol = 1e-6
        @test compensated[bus][2] ≈ θ atol = 1e-6
    end
end

@testset "PSS/E export writes a SeriesCompensator as an R = 0, X < 0 branch" begin
    sys = compensated_c_sys14()
    dir = mktempdir()
    exporter = PSSEExporter(sys, :v33, dir)
    write_export(exporter, "compensated"; overwrite = true)
    raw_path, metadata_path = get_psse_export_paths(joinpath(dir, "compensated"))
    sys2 = PSY.System(raw_path, Dict(JSON3.read(metadata_path, Dict)))
    compensator = only(PSY.get_components(PSY.SeriesCompensator, sys2))
    @test PSY.get_control_mode(compensator) == PSY.SeriesCompensatorControlMode.FIXED
    @test PSY.get_x(compensator, u"SU") ≈ PSY.get_x(
        PSY.get_component(PSY.SeriesCompensator, sys, "compensator"), u"SU")
end
