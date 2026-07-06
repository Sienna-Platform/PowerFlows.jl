# VSC results write-back: `solve_and_store_power_flow!` must write the solved DC-network state
# (`p_c`, `q_c`, `node_vdc`) back to `TwoTerminalVSCLine` / `InterconnectingConverter` components.
# Only Vdc + PQ control modes are exercised here. Inputs (`active_power_flow`, IC `active_power`)
# start at 0.0 (≠ solved values) so the write-back is observable.

# Converter index in `dcn` whose AC bus number matches `number` (unique in these systems).
_conv_ix_by_bus_number(dcn, number::Int) =
    only(findall(==(number), dcn.converter_ac_bus_number))

@testset "VSC results: point-to-point line write-back via solve_and_store_power_flow!" begin
    pf = ACPowerFlow{NewtonRaphsonACPowerFlow}(; solver_settings = VSC_SETTINGS)
    build() = _build_vsc_pq_system(;
        name = "vsc_results",
        active_power_flow = 0.0,
        set_system_base = true,
    )

    # reference solve on an identical fresh system to obtain the solved DC-network state
    data = PowerFlowData(pf, build())
    @test solve_power_flow!(data)
    dcn = PF.get_dc_network(data)

    sys = build()
    vsc = PSY.get_component(PSY.TwoTerminalVSCLine, sys, "vsc_results")
    from_number = PSY.get_number(PSY.get_from(PSY.get_arc(vsc)))
    to_number = PSY.get_number(PSY.get_to(PSY.get_arc(vsc)))
    @test iszero(PSY.get_active_power_flow(vsc))
    @test solve_and_store_power_flow!(pf, sys)

    cf = _conv_ix_by_bus_number(dcn, from_number)
    ct = _conv_ix_by_bus_number(dcn, to_number)
    p_c_from = dcn.p_c[cf, 1]

    # active_power_flow moved off its 0.0 input and equals −p_c_from (from→to link flow)
    @test !isapprox(PSY.get_active_power_flow(vsc), 0.0; atol = 1e-3)
    @test isapprox(PSY.get_active_power_flow(vsc), -p_c_from; atol = 1e-6)
    # the to converter holds its 0.4 power order, so the link carries ≈ 0.4 + DC-line losses
    @test PSY.get_active_power_flow(vsc) > 0.4

    # reactive terminal injections equal the solved q_c
    @test isapprox(PSY.get_reactive_power_from(vsc), dcn.q_c[cf, 1]; atol = 1e-6)
    @test isapprox(PSY.get_reactive_power_to(vsc), dcn.q_c[ct, 1]; atol = 1e-6)

    # dc_current is finite, positive from→to, and consistent with P_dc / V_dc at the from node
    nf = dcn.converter_dc_node_ix[cf]
    Vm_from = data.bus_magnitude[dcn.converter_ac_bus_ix[cf], 1]
    Vdc_from = dcn.node_vdc[nf, 1]
    Idc = PSY.get_dc_current(vsc)
    @test isfinite(Idc)
    @test isapprox(Idc, -PF._vsc_pdc(dcn, cf, Vm_from, 1) / Vdc_from; atol = 1e-6)
    @test Idc > 0.0
    # sign consistency: dc_current and active_power_flow are both from→to
    @test sign(Idc) == sign(PSY.get_active_power_flow(vsc))
end

@testset "VSC results: MTDC InterconnectingConverter write-back" begin
    pf = ACPowerFlow{NewtonRaphsonACPowerFlow}(; solver_settings = VSC_SETTINGS)

    data = PowerFlowData(pf, _build_mtdc_system())
    @test solve_power_flow!(data)
    dcn = PF.get_dc_network(data)

    sys = _build_mtdc_system()
    @test solve_and_store_power_flow!(pf, sys)

    for ic in PSY.get_components(PSY.InterconnectingConverter, sys)
        c = _conv_ix_by_bus_number(dcn, PSY.get_number(PSY.get_bus(ic)))
        @test dcn.node_number[dcn.converter_dc_node_ix[c]] ==
              PSY.get_number(PSY.get_dc_bus(ic))
        Vm = data.bus_magnitude[dcn.converter_ac_bus_ix[c], 1]
        # active_power (DC-side, positive = drawn from the DC bus into the AC side) = P_dc
        @test isapprox(PSY.get_active_power(ic), PF._vsc_pdc(dcn, c, Vm, 1); atol = 1e-6)
    end

    # the DC-slack converter (ic1) balances the two 0.30/0.20 power orders plus DC losses,
    # so its active_power must have MOVED from the 0.0 input
    ic1 = PSY.get_component(PSY.InterconnectingConverter, sys, "ic1")
    @test !isapprox(PSY.get_active_power(ic1), 0.0; atol = 1e-3)
    @test PSY.get_active_power(ic1) < -0.5
    # the power-controlled converters hold their orders (lossless converters: P_dc = p_c)
    ic2 = PSY.get_component(PSY.InterconnectingConverter, sys, "ic2")
    ic3 = PSY.get_component(PSY.InterconnectingConverter, sys, "ic3")
    @test isapprox(PSY.get_active_power(ic2), 0.30; atol = 1e-6)
    @test isapprox(PSY.get_active_power(ic3), 0.20; atol = 1e-6)
end
