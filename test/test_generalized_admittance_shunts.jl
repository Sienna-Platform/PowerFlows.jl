# GA nodal power model (P/I/Z split), the fixed PQ/PV shunts, and the flat-start q⁰ heuristic.

@testset "GA: nodal power, ZIP split, PQ shunts on c_sys14" begin
    sys = PSB.build_system(PSB.PSITestSystems, "c_sys14"; add_forecasts = false)
    pq1 = first(
        sort!(
            collect(
                PSY.get_components(
                    b -> PSY.get_bustype(b) == PSY.ACBusTypes.PQ,
                    PSY.ACBus,
                    sys,
                ),
            );
            by = PSY.get_number,
        ),
    )
    _add_simple_zip_load!(
        sys,
        pq1;
        constant_current_active_power = 0.1,
        constant_current_reactive_power = 0.05,
        constant_impedance_active_power = 0.2,
        constant_impedance_reactive_power = 0.07,
    )
    data = PowerFlowData(ACPowerFlow{GeneralizedAdmittanceACPowerFlow}(), sys)
    part = PF.GAPartition(data, 1, Dict{Int, Float64}())
    np = PF.GANodalPower(data, part, PF.GAConverterTerms(size(data.bus_type, 1)), 1)
    y = PF._ga_initial_shunts(PF.GABlocks(data, part), np, part, data, 1)
    @test any(!iszero, np.sI) && any(!iszero, np.sZ)
    for (j, ix) in enumerate(part.l_ix)
        sP = complex(
            data.bus_active_power_withdrawals[ix, 1] -
            data.bus_active_power_injections[ix, 1] - data.bus_hvdc_net_power[ix, 1],
            data.bus_reactive_power_withdrawals[ix, 1] -
            data.bus_reactive_power_injections[ix, 1],
        )
        @test np.sP[j] ≈ sP
        if j > PF.n_v(part)
            ur = data.bus_magnitude[ix, 1]
            @test y[j] ≈ conj(sP + np.sI[j] * ur) / ur^2 + conj(np.sZ[j])
        end
    end
end

@testset "GA: flat-start q⁰ on a hand-built 3-bus system" begin
    sys = PSY.System(100.0)
    b1 = _add_simple_bus!(sys, 1, PSY.ACBusTypes.REF, 230, 1.0)
    b2 = _add_simple_bus!(sys, 2, PSY.ACBusTypes.PV, 230, 1.05)
    b3 = _add_simple_bus!(sys, 3, PSY.ACBusTypes.PQ, 230, 1.0)
    _add_simple_source!(sys, b1)
    _add_simple_thermal_standard!(sys, b2, 0.5, 0.0)
    _add_simple_load!(sys, b3, 80.0, 30.0)
    _add_simple_line!(sys, b1, b2, 0.0, 0.1, 0.0)
    _add_simple_line!(sys, b1, b3, 0.0, 0.1, 0.0)
    _add_simple_line!(sys, b2, b3, 0.0, 0.2, 0.0)
    data = PowerFlowData(ACPowerFlow{GeneralizedAdmittanceACPowerFlow}(), sys)
    part = PF.GAPartition(data, 1, Dict{Int, Float64}())
    np = PF.GANodalPower(data, part, PF.GAConverterTerms(3), 1)
    b = PF.GABlocks(data, part)
    y = PF._ga_initial_shunts(b, np, part, data, 1)
    q0 = only(PF._ga_flat_start_q0(b, part, y, data.bus_magnitude[part.s_ix, 1]))
    q3 = data.bus_reactive_power_withdrawals[only(part.q_ix), 1]
    u3 = (10 * 1.0 + 5 * 1.05) / (15 + q3) # spec §3.6 by hand: B33 = -15 - q3
    @test q0 ≈ 1.05 * (-15 * 1.05 + 10 * 1.0 + 5 * u3) atol = 1e-10
    @test q0 < 0.0
    @test y[1] ≈ complex(real(PF._ga_s(np, 1, 1.05)), -q0) / 1.05^2
end
