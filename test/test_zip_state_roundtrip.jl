# Issue #439: the data <-> state/residual conversion helpers must account for ZIP
# withdrawals, not just constant-power ones.

const ZIP_ROUNDTRIP_ATOL = 1e-7

# P+I+Z load on every bus, so REF/PV/PQ state blocks all see a ZIP withdrawal. The other
# ZIP fixtures only load REF/PQ buses, which is how the PV-slot bugs hid.
function _build_zip_3bus_system()
    sys = System(100.0)
    b1 = _add_simple_bus!(sys, 1, ACBusTypes.REF, 230, 1.02, 0.0)
    b2 = _add_simple_bus!(sys, 2, ACBusTypes.PV, 230, 1.01, 0.0)
    b3 = _add_simple_bus!(sys, 3, ACBusTypes.PQ, 230, 1.0, 0.0)
    _add_simple_line!(sys, b1, b2, 5e-3, 5e-2, 1e-2)
    _add_simple_line!(sys, b2, b3, 5e-3, 5e-2, 1e-2)
    _add_simple_source!(sys, b1, 0.0, 0.0)
    _add_simple_thermal_standard!(sys, b2, 0.4, 0.1)
    for bus in (b1, b2, b3)
        _add_simple_zip_load!(
            sys,
            bus;
            constant_power_active_power = 1.0,
            constant_power_reactive_power = 0.4,
            constant_current_active_power = 0.8,
            constant_current_reactive_power = 0.3,
            constant_impedance_active_power = 0.6,
            constant_impedance_reactive_power = 0.2,
        )
    end
    return sys
end

@testset "ZIP loads: polar update_state! / update_data! round-trip" begin
    pf = ACPowerFlow{NewtonRaphsonACPowerFlow}()
    data = PowerFlowData(pf, _build_zip_3bus_system())
    @test PF.solve_power_flow!(data)

    # Rebuilding the state from converged `data` must give a converged residual.
    residual = PF.ACPowerFlowResidual(data, 1)
    x = zeros(Float64, length(residual.Rv))
    PF.update_state!(x, data, 1)
    residual(data, x, 1)
    @test maximum(abs, residual.Rv) < ZIP_ROUNDTRIP_ATOL

    # `update_data!` inverts `update_state!`, but only for actual state variables:
    # P at REF, Q at REF and PV. The rest are inputs.
    bt = view(data.bus_type, :, 1)
    ref_ix = findall(==(PSY.ACBusTypes.REF), bt)
    gen_ix = findall(!=(PSY.ACBusTypes.PQ), bt)
    P_inj = copy(view(data.bus_active_power_injections, :, 1))
    Q_inj = copy(view(data.bus_reactive_power_injections, :, 1))
    Vm = copy(view(data.bus_magnitude, :, 1))
    θ = copy(view(data.bus_angles, :, 1))
    data.bus_active_power_injections[ref_ix, 1] .= NaN
    data.bus_reactive_power_injections[gen_ix, 1] .= NaN
    PF.update_data!(data, x, 1)
    @test maximum(abs, data.bus_active_power_injections[ref_ix, 1] - P_inj[ref_ix]) <
          ZIP_ROUNDTRIP_ATOL
    @test maximum(abs, data.bus_reactive_power_injections[gen_ix, 1] - Q_inj[gen_ix]) <
          ZIP_ROUNDTRIP_ATOL
    @test maximum(abs, data.bus_magnitude[:, 1] - Vm) < ZIP_ROUNDTRIP_ATOL
    @test maximum(abs, data.bus_angles[:, 1] - θ) < ZIP_ROUNDTRIP_ATOL
end

@testset "ZIP loads: rectangular/mixed state fill round-trip" begin
    for (pf, residual_type, fill_state!) in (
        (
            ACRectangularPowerFlow{NewtonRaphsonACPowerFlow}(),
            PF.ACRectangularCIResidual,
            PF.rect_initial_state!,
        ),
        (
            ACMixedPowerFlow{NewtonRaphsonACPowerFlow}(),
            PF.ACMixedCPBResidual,
            PF.mixed_initial_state!,
        ),
    )
        data = PowerFlowData(pf, _build_zip_3bus_system())
        @test PF.solve_power_flow!(data)
        residual = residual_type(data, 1)
        x = Vector{Float64}(undef, length(residual.Rv))
        fill_state!(x, data, residual.bus_state_offset, residual.bus_block_size, 1)
        residual(data, x, 1)
        @test maximum(abs, residual.Rv) < ZIP_ROUNDTRIP_ATOL
    end
end

# Reported injections come from `*_finalize_bus_injections!`; polar is the reference.
@testset "ZIP loads: rectangular/mixed reported injections match polar" begin
    pf_p = ACPowerFlow{NewtonRaphsonACPowerFlow}()
    data_p = PowerFlowData(pf_p, _build_zip_3bus_system())
    @test PF.solve_power_flow!(data_p)
    for pf in (
        ACRectangularPowerFlow{NewtonRaphsonACPowerFlow}(),
        ACMixedPowerFlow{NewtonRaphsonACPowerFlow}(),
    )
        data = PowerFlowData(pf, _build_zip_3bus_system())
        @test PF.solve_power_flow!(data)
        @test maximum(abs, data.bus_magnitude - data_p.bus_magnitude) <
              ZIP_ROUNDTRIP_ATOL
        @test maximum(
            abs,
            data.bus_active_power_injections - data_p.bus_active_power_injections,
        ) < ZIP_ROUNDTRIP_ATOL
        @test maximum(
            abs,
            data.bus_reactive_power_injections - data_p.bus_reactive_power_injections,
        ) < ZIP_ROUNDTRIP_ATOL
    end
end

# The polar helpers only cover bus quantities: `update_data!` never writes the LCC/VSC/
# area tail back, and `partition_state` sees only bus types. Both warn rather than
# silently returning a partial answer.
@testset "non-bus tail: update_data! and partition_state warn" begin
    sys = System(100.0)
    b1 = _add_simple_bus!(sys, 1, ACBusTypes.REF, 230, 1.1, 0.0)
    b2 = _add_simple_bus!(sys, 2, ACBusTypes.PQ, 230, 1.1, 0.0)
    b3 = _add_simple_bus!(sys, 3, ACBusTypes.PQ, 230, 1.1, 0.0)
    _add_simple_load!(sys, b2, 10, 5)
    _add_simple_load!(sys, b3, 60, 20)
    _add_simple_line!(sys, b1, b2, 5e-3, 5e-3, 1e-3)
    _add_simple_line!(sys, b1, b3, 5e-3, 5e-3, 1e-3)
    _add_simple_source!(sys, b1, 0.0, 0.0)
    lcc = _add_simple_lcc!(sys, b2, b3, 0.05, 0.05, 0.08)
    PSY.set_inverter_extinction_angle!(lcc, 1.0)

    data = PowerFlowData(ACPowerFlow{NewtonRaphsonACPowerFlow}(), sys)
    @test PF.solve_power_flow!(data)
    x = zeros(Float64, 2 * size(data.bus_type, 1) + 4)
    PF.update_state!(x, data, 1)
    @test_logs (:warn, r"not written back") match_mode = :any PF.update_data!(data, x, 1)
    @test_logs (:warn, r"non-bus tail") match_mode = :any PF.partition_state(
        x, view(data.bus_type, :, 1),
    )
    # No tail, no warning.
    no_lcc =
        PowerFlowData(ACPowerFlow{NewtonRaphsonACPowerFlow}(), _build_zip_3bus_system())
    x2 = zeros(Float64, 2 * size(no_lcc.bus_type, 1))
    PF.update_state!(x2, no_lcc, 1)
    @test_logs min_level = Logging.Warn PF.update_data!(no_lcc, x2, 1)
    @test_logs min_level = Logging.Warn PF.partition_state(x2, view(no_lcc.bus_type, :, 1))
end
