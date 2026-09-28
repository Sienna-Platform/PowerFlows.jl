# GA partition and sparse Y-bus blocks: the s/v/q bus-set split and the fixed-pattern blocks
# (net Yℓℓ / Yqq with stored shunt slots) that the generalized-admittance iteration solves.

# Point-to-point VSC line between the first two PQ buses of c_sys14, BOTH terminals pinning AC
# voltage (from = ControlVdcQ, to = ControlPVac), so both converters enter set v.
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
        name = "ga_vsc",
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

@testset "GA: partition and blocks on c_sys14" begin
    sys = PSB.build_system(PSB.PSITestSystems, "c_sys14"; add_forecasts = false)
    data = PowerFlowData(ACPowerFlow{GeneralizedAdmittanceACPowerFlow}(), sys)
    part = PF.GAPartition(data, 1, Dict{Int, Float64}())
    ref, pv, pq = PF.bus_type_idx(data, 1)
    @test (part.s_ix, part.v_ix, part.q_ix) == (ref, pv, pq)
    @test part.l_ix == vcat(pv, pq)
    @test part.n_pv == length(pv)
    @test part.Vset == data.bus_magnitude[pv, 1]
    b = PF.GABlocks(data, part)
    Y = Matrix{ComplexF64}(PNM.get_data(data.power_network_matrix))
    nv = PF.n_v(part)
    @test Matrix(b.Yll) ≈ Y[part.l_ix, part.l_ix]
    @test Matrix(b.Yvq) ≈ Y[pv, pq]
    @test Matrix(b.Yqv) ≈ Y[pq, pv]
    @test Matrix(b.Yls) ≈ Y[part.l_ix, ref]
    y = ComplexF64.(1:PF.n_l(part)) .* (0.1 - 0.05im)
    for scale in (1.0, 2.0)
        PF._ga_set_shunts!(b, scale .* y, nv)
        @test Matrix(b.Yll) ≈ Y[part.l_ix, part.l_ix] + Diagonal(scale .* y)
        @test Matrix(b.Yqq) ≈ Y[pq, pq] + Diagonal(scale .* y[(nv + 1):end])
    end
end

@testset "GA: VSC AC-voltage buses join set v, bus types unchanged" begin
    data = PowerFlowData(
        ACPowerFlow{GeneralizedAdmittanceACPowerFlow}(;
            solution_parameters = VSC_SOLUTION_PARAMETERS), _ga_vsc_ac_voltage_system())
    targets = PF._ga_vsc_ac_voltage_targets(data, 1)
    @test length(targets) == 2
    part = PF.GAPartition(data, 1, targets)
    extra = part.v_ix[(part.n_pv + 1):end]
    @test sort(extra) == sort(collect(keys(targets)))
    @test part.Vset[(part.n_pv + 1):end] == [targets[ix] for ix in extra]
    @test all(data.bus_type[ix, 1] === PSY.ACBusTypes.PQ for ix in extra)
end
