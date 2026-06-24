# VSC HVDC power-flow tests. I0: lowering of a point-to-point TwoTerminalVSCLine into the internal
# DCNetwork (isolated 2-node). Later increments add the residual/Jacobian/solver tests.

# Modeling VSC/DC components as joint AC↔DC unknowns is opt-in (they are ignored in AC power flow
# otherwise). Every solve in this file enables it.
const VSC_SETTINGS = Dict{Symbol, Any}(:model_dc_network => true)

# Reuse an existing Arc between two buses if present (PSY enforces Arc-name uniqueness), else make
# and add one. HVDC lines may share an Arc with an existing AC branch.
function _get_or_make_arc(sys, from_bus, to_bus)
    existing = PSY.get_components(
        a -> PSY.get_from(a) === from_bus && PSY.get_to(a) === to_bus,
        PSY.Arc,
        sys,
    )
    isempty(existing) || return first(existing)
    arc = PSY.Arc(; from = from_bus, to = to_bus)
    PSY.add_component!(sys, arc)
    return arc
end

# Build c_sys5 and add one point-to-point VSC line: the `from` converter controls DC voltage
# (DC slack), the `to` converter controls (P, Q). This is the physically well-posed config: one
# terminal fixes V_dc, the other sets power.
function _build_vsc_system(; g = 50.0)
    sys = deepcopy(PSB.build_system(PSB.PSITestSystems, "c_sys5"; add_forecasts = false))
    buses = sort!(collect(PSY.get_components(PSY.ACBus, sys)); by = PSY.get_number)
    from_bus = buses[1]
    to_bus = buses[4]
    arc = _get_or_make_arc(sys, from_bus, to_bus)
    vsc = PSY.TwoTerminalVSCLine(;
        name = "vsc_test",
        available = true,
        arc = arc,
        active_power_flow = 0.5,
        rating = 1.0,
        active_power_limits_from = (min = -1.0, max = 1.0),
        active_power_limits_to = (min = -1.0, max = 1.0),
        g = g,
        # from converter: DC-voltage control (DC slack), no AC-voltage control
        dc_control_from = PSY.VSCDCControlModes.DC_VOLTAGE,
        ac_control_from = PSY.VSCACControlModes.AC_REACTIVE_POWER,
        dc_setpoint_from = 1.0,
        ac_setpoint_from = 1.0,
        # to converter: power control (P, Q)
        dc_control_to = PSY.VSCDCControlModes.DC_POWER,
        ac_control_to = PSY.VSCACControlModes.AC_REACTIVE_POWER,
        dc_setpoint_to = 0.5,
        ac_setpoint_to = 1.0,
    )
    PSY.add_component!(sys, vsc)
    return sys
end

@testset "VSC I0: TwoTerminalVSCLine lowers to an isolated 2-node DCNetwork" begin
    sys = _build_vsc_system(; g = 50.0)
    data = PowerFlowData(
        ACPowerFlow{NewtonRaphsonACPowerFlow}(; solver_settings = VSC_SETTINGS),
        sys,
    )
    dcn = PF.get_dc_network(data)

    @test PF.n_vsc_converters(dcn) == 2
    @test PF.n_dc_nodes(dcn) == 2
    @test PF.n_dc_branches(dcn) == 1
    # exactly one slack (the from / DC-voltage-controlling terminal), one free node
    @test count(dcn.node_is_slack) == 1
    @test PF.n_vsc_free_nodes(dcn) == 1
    # tail length = 2 per converter + one V_dc per DC node (every node carries a V_dc state)
    @test PF.vsc_tail_length(dcn) == 6

    # control modes: from = ControlVdc (DC slack), to = ControlPQ
    @test dcn.converter_mode[1] == PF.ControlVdc
    @test dcn.converter_mode[2] == PF.ControlPQ
    @test dcn.node_is_slack[1]
    @test !dcn.node_is_slack[2]

    # dense DC conductance: [g -g; -g g]
    @test dcn.G_dc ≈ [50.0 -50.0; -50.0 50.0]
    @test dcn.branch_g == [50.0]
end

@testset "VSC I0: pure-AC system has an empty DCNetwork (regression-safe)" begin
    sys = deepcopy(PSB.build_system(PSB.PSITestSystems, "c_sys5"; add_forecasts = false))
    data = PowerFlowData(
        ACPowerFlow{NewtonRaphsonACPowerFlow}(; solver_settings = VSC_SETTINGS),
        sys,
    )
    dcn = PF.get_dc_network(data)
    @test PF.n_vsc_converters(dcn) == 0
    @test PF.vsc_tail_length(dcn) == 0
    @test !PF.has_dc_network(dcn)
end

@testset "VSC I0: pure-droop point-to-point line is anchored (no slack error)" begin
    sys = deepcopy(PSB.build_system(PSB.PSITestSystems, "c_sys5"; add_forecasts = false))
    buses = sort!(collect(PSY.get_components(PSY.ACBus, sys)); by = PSY.get_number)
    arc = _get_or_make_arc(sys, buses[2], buses[3])
    vsc = PSY.TwoTerminalVSCLine(;
        name = "vsc_droop",
        available = true,
        arc = arc,
        active_power_flow = 0.2,
        rating = 1.0,
        active_power_limits_from = (min = -1.0, max = 1.0),
        active_power_limits_to = (min = -1.0, max = 1.0),
        g = 40.0,
        dc_control_from = PSY.VSCDCControlModes.DC_VOLTAGE_DROOP,
        dc_voltage_droop_from = 0.05,
        dc_setpoint_from = 1.0,
        dc_control_to = PSY.VSCDCControlModes.DC_VOLTAGE_DROOP,
        dc_voltage_droop_to = 0.05,
        dc_setpoint_to = 1.0,
    )
    PSY.add_component!(sys, vsc)
    data = PowerFlowData(
        ACPowerFlow{NewtonRaphsonACPowerFlow}(; solver_settings = VSC_SETTINGS),
        sys,
    )
    dcn = PF.get_dc_network(data)
    @test dcn.converter_mode[1] == PF.ControlPVdcDroop
    @test dcn.converter_mode[2] == PF.ControlPVdcDroop
    # both droop ⇒ no slack node, but the subnet is anchored by droop ⇒ no error, 2 free nodes
    @test count(dcn.node_is_slack) == 0
    @test PF.n_vsc_free_nodes(dcn) == 2
end

# I1: a point-to-point VSC (from = DC-voltage control / DC slack, to = P,Q control) on two PQ buses,
# zero converter loss. Solve with polar NR and verify convergence + setpoints + Jacobian.
function _build_vsc_pq_system(;
    g = 50.0,
    p_set = 0.4,
    q_set = 0.1,
    vdc = 1.05,
    q_set_from = 0.05,
)
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
    from_bus = pq[1]
    to_bus = pq[2]
    arc = _get_or_make_arc(sys, from_bus, to_bus)
    vsc = PSY.TwoTerminalVSCLine(;
        name = "vsc_i1",
        available = true,
        arc = arc,
        active_power_flow = p_set,
        rating = 2.0,
        active_power_limits_from = (min = -2.0, max = 2.0),
        active_power_limits_to = (min = -2.0, max = 2.0),
        g = g,
        # from: DC-voltage control (slack), reactive setpoint
        dc_control_from = PSY.VSCDCControlModes.DC_VOLTAGE,
        ac_control_from = PSY.VSCACControlModes.AC_REACTIVE_POWER,
        dc_setpoint_from = vdc,
        reactive_power_from = q_set_from,
        # to: power control (P, Q)
        dc_control_to = PSY.VSCDCControlModes.DC_POWER,
        ac_control_to = PSY.VSCACControlModes.AC_REACTIVE_POWER,
        dc_setpoint_to = p_set,
        reactive_power_to = q_set,
    )
    PSY.add_component!(sys, vsc)
    return sys
end

@testset "VSC I1: polar NR solves a point-to-point VSC and meets setpoints" begin
    sys = _build_vsc_pq_system(; g = 50.0, p_set = 0.4, q_set = 0.1, vdc = 1.05)
    data = PowerFlowData(
        ACPowerFlow{NewtonRaphsonACPowerFlow}(; solver_settings = VSC_SETTINGS),
        sys,
    )
    @test solve_power_flow!(data)
    dcn = PF.get_dc_network(data)
    # converter 1 = from (ControlVdc, DC slack); converter 2 = to (ControlPQ)
    @test dcn.converter_mode[1] == PF.ControlVdc
    @test dcn.converter_mode[2] == PF.ControlPQ
    # ControlPQ converter holds its P, Q setpoints
    @test isapprox(dcn.p_c[2, 1], 0.4; atol = 1e-7)
    @test isapprox(dcn.q_c[2, 1], 0.1; atol = 1e-7)
    # ControlVdc converter pins its DC-node voltage and holds its reactive setpoint
    @test isapprox(dcn.node_vdc[1, 1], 1.05; atol = 1e-7)
    @test isapprox(dcn.q_c[1, 1], 0.05; atol = 1e-7)
    # full residual (incl. DC-KCL) is ~0 at the solution
    residual = PF.ACPowerFlowResidual(data, 1)
    x = PF.calculate_x0(data, 1)
    residual(x, 1)
    @test maximum(abs, residual.Rv) < 1e-7
end

@testset "VSC I1: analytic polar Jacobian matches finite differences" begin
    sys = _build_vsc_pq_system(; g = 40.0, p_set = 0.3, q_set = -0.05, vdc = 1.02)
    data = PowerFlowData(
        ACPowerFlow{NewtonRaphsonACPowerFlow}(; solver_settings = VSC_SETTINGS),
        sys,
    )
    @test solve_power_flow!(data)
    # rebuild residual/Jacobian at the converged state and check the analytic J vs FD
    residual = PF.ACPowerFlowResidual(data, 1)
    jac = PF.ACPowerFlowJacobian(residual, 1)
    x = PF.calculate_x0(data, 1)
    residual(x, 1)
    jac(1)
    verify_jacobian_asymptotic(residual, jac.Jv, x, 1; label = "VSC polar I1")
end

# Flexible builder: one VSC line between the first two PQ buses of c_sys14, control fields passed
# through as keyword args.
function _vsc_system(; g = 50.0, vsc_kwargs...)
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
        name = "vsc",
        available = true,
        arc = arc,
        active_power_flow = 0.3,
        rating = 2.0,
        active_power_limits_from = (min = -2.0, max = 2.0),
        active_power_limits_to = (min = -2.0, max = 2.0),
        g = g,
        vsc_kwargs...,
    )
    PSY.add_component!(sys, vsc)
    return (sys, PSY.get_number(pq[1]), PSY.get_number(pq[2]))
end

@testset "VSC I2: DC-voltage droop — both converters follow V_dc = V_set − k·P_c" begin
    sys, _, _ = _vsc_system(;
        g = 50.0,
        dc_control_from = PSY.VSCDCControlModes.DC_VOLTAGE_DROOP,
        dc_voltage_droop_from = 0.02,
        dc_setpoint_from = 1.04,
        reactive_power_from = 0.0,
        dc_control_to = PSY.VSCDCControlModes.DC_VOLTAGE_DROOP,
        dc_voltage_droop_to = 0.03,
        dc_setpoint_to = 1.04,
        reactive_power_to = 0.0,
    )
    data = PowerFlowData(
        ACPowerFlow{NewtonRaphsonACPowerFlow}(; solver_settings = VSC_SETTINGS),
        sys,
    )
    @test solve_power_flow!(data)
    dcn = PF.get_dc_network(data)
    @test dcn.converter_mode[1] == PF.ControlPVdcDroop
    for c in 1:2
        node = dcn.converter_dc_node_ix[c]
        @test isapprox(
            dcn.node_vdc[node, 1],
            dcn.vdc_set[c, 1] - dcn.droop_k[c] * dcn.p_c[c, 1];
            atol = 1e-7,
        )
    end
end

@testset "VSC I2: AC-voltage control (Vdc+Vac from, P+Vac to) holds bus magnitudes" begin
    sys, from_no, to_no = _vsc_system(;
        g = 50.0,
        dc_control_from = PSY.VSCDCControlModes.DC_VOLTAGE,
        ac_control_from = PSY.VSCACControlModes.AC_VOLTAGE,
        dc_setpoint_from = 1.05,
        ac_setpoint_from = 1.01,
        dc_control_to = PSY.VSCDCControlModes.DC_POWER,
        ac_control_to = PSY.VSCACControlModes.AC_VOLTAGE,
        dc_setpoint_to = 0.25,
        ac_setpoint_to = 1.0,
    )
    data = PowerFlowData(
        ACPowerFlow{NewtonRaphsonACPowerFlow}(; solver_settings = VSC_SETTINGS),
        sys,
    )
    @test solve_power_flow!(data)
    dcn = PF.get_dc_network(data)
    @test dcn.converter_mode[1] == PF.ControlVdcQ
    @test dcn.converter_mode[2] == PF.ControlPVac
    bus_lookup = PF.get_bus_lookup(data)
    @test isapprox(dcn.node_vdc[1, 1], 1.05; atol = 1e-7)                 # from pins V_dc
    @test isapprox(data.bus_magnitude[bus_lookup[from_no], 1], 1.01; atol = 1e-7)  # from pins |V_ac|
    @test isapprox(dcn.p_c[2, 1], 0.25; atol = 1e-7)                      # to holds P
    @test isapprox(data.bus_magnitude[bus_lookup[to_no], 1], 1.0; atol = 1e-7)     # to pins |V_ac|
end

@testset "VSC I2: lossy converter — analytic Jacobian matches finite differences" begin
    sys, _, _ = _vsc_system(;
        g = 45.0,
        dc_control_from = PSY.VSCDCControlModes.DC_VOLTAGE,
        ac_control_from = PSY.VSCACControlModes.AC_REACTIVE_POWER,
        dc_setpoint_from = 1.03,
        reactive_power_from = 0.0,
        dc_control_to = PSY.VSCDCControlModes.DC_POWER,
        ac_control_to = PSY.VSCACControlModes.AC_REACTIVE_POWER,
        dc_setpoint_to = 0.35,
        reactive_power_to = 0.05,
        converter_loss_to = PSY.QuadraticCurve(0.01, 0.02, 0.005),
    )
    data = PowerFlowData(
        ACPowerFlow{NewtonRaphsonACPowerFlow}(; solver_settings = VSC_SETTINGS),
        sys,
    )
    @test solve_power_flow!(data)
    residual = PF.ACPowerFlowResidual(data, 1)
    jac = PF.ACPowerFlowJacobian(residual, 1)
    x = PF.calculate_x0(data, 1)
    residual(x, 1)
    jac(1)
    verify_jacobian_asymptotic(residual, jac.Jv, x, 1; label = "VSC polar lossy")
end

@testset "VSC I4: NR / TrustRegion / LevenbergMarquardt agree on a VSC solve (polar)" begin
    solvers = (
        NewtonRaphsonACPowerFlow,
        PF.TrustRegionACPowerFlow,
        PF.LevenbergMarquardtACPowerFlow,
    )
    refs = Vector{Tuple{Float64, Float64, Float64}}()
    for S in solvers
        sys = _build_vsc_pq_system(; g = 50.0, p_set = 0.4, q_set = 0.1, vdc = 1.05)
        data = PowerFlowData(PF.ACPolarPowerFlow{S}(; solver_settings = VSC_SETTINGS), sys)
        @test solve_power_flow!(data)
        dcn = PF.get_dc_network(data)
        push!(refs, (dcn.p_c[2, 1], dcn.q_c[2, 1], dcn.node_vdc[2, 1]))
    end
    for r in refs[2:end]
        @test all(isapprox.(r, refs[1]; atol = 1e-6))
    end
end

# I3: the VSC model must give the same answer under every AC formulation.
@testset "VSC I3: polar, rectangular, and mixed all agree on a VSC solve" begin
    sol = Dict{String, Any}()
    for (name, PF_T) in (
        ("polar", PF.ACPolarPowerFlow{NewtonRaphsonACPowerFlow}),
        ("rect", PF.ACRectangularPowerFlow{NewtonRaphsonACPowerFlow}),
        ("mixed", PF.ACMixedPowerFlow{NewtonRaphsonACPowerFlow}),
    )
        sys = _build_vsc_pq_system(; g = 50.0, p_set = 0.4, q_set = 0.1, vdc = 1.05)
        data = PowerFlowData(PF_T(; solver_settings = VSC_SETTINGS), sys)
        @test solve_power_flow!(data)
        dcn = PF.get_dc_network(data)
        sol[name] = (
            copy(data.bus_magnitude[:, 1]),
            dcn.p_c[2, 1],
            dcn.q_c[2, 1],
            dcn.node_vdc[2, 1],
        )
    end
    for other in ("rect", "mixed")
        @test isapprox(sol["polar"][1], sol[other][1]; atol = 1e-6)
        @test isapprox(sol["polar"][2], sol[other][2]; atol = 1e-6)
        @test isapprox(sol["polar"][3], sol[other][3]; atol = 1e-6)
        @test isapprox(sol["polar"][4], sol[other][4]; atol = 1e-6)
    end
end

@testset "VSC I3: mixed Jacobian matches finite differences (incl. loss)" begin
    sys, _, _ = _vsc_system(;
        g = 45.0,
        dc_control_from = PSY.VSCDCControlModes.DC_VOLTAGE,
        ac_control_from = PSY.VSCACControlModes.AC_REACTIVE_POWER,
        dc_setpoint_from = 1.03,
        reactive_power_from = 0.0,
        dc_control_to = PSY.VSCDCControlModes.DC_POWER,
        ac_control_to = PSY.VSCACControlModes.AC_REACTIVE_POWER,
        dc_setpoint_to = 0.35,
        reactive_power_to = 0.05,
        converter_loss_to = PSY.QuadraticCurve(0.01, 0.02, 0.005),
    )
    pf = PF.ACMixedPowerFlow{NewtonRaphsonACPowerFlow}(; solver_settings = VSC_SETTINGS)
    data = PowerFlowData(pf, sys)
    @test solve_power_flow!(data)
    residual, jac, x = PF.initialize_power_flow_variables(pf, data, 1)
    residual(x, 1)
    jac(1)
    verify_jacobian_asymptotic(residual, jac.Jv, x, 1; label = "VSC mixed")
end

@testset "VSC I3: rectangular Jacobian matches finite differences (incl. loss)" begin
    sys, _, _ = _vsc_system(;
        g = 45.0,
        dc_control_from = PSY.VSCDCControlModes.DC_VOLTAGE,
        ac_control_from = PSY.VSCACControlModes.AC_REACTIVE_POWER,
        dc_setpoint_from = 1.03,
        reactive_power_from = 0.0,
        dc_control_to = PSY.VSCDCControlModes.DC_POWER,
        ac_control_to = PSY.VSCACControlModes.AC_REACTIVE_POWER,
        dc_setpoint_to = 0.35,
        reactive_power_to = 0.05,
        converter_loss_to = PSY.QuadraticCurve(0.01, 0.02, 0.005),
    )
    pf = PF.ACRectangularPowerFlow{NewtonRaphsonACPowerFlow}(;
        solver_settings = VSC_SETTINGS,
    )
    data = PowerFlowData(pf, sys)
    @test solve_power_flow!(data)
    residual, jac, x = PF.initialize_power_flow_variables(pf, data, 1)
    residual(x, 1)
    jac(1)
    verify_jacobian_asymptotic(residual, jac.Jv, x, 1; label = "VSC rect")
end

# I5 / M2: genuine multi-terminal DC grid — DCBus nodes + InterconnectingConverter (AC↔DC) +
# TModelHVDCLine DC branches. Three terminals: one DC-voltage slack, two power-controlled.
function _build_mtdc_system()
    sys = deepcopy(PSB.build_system(PSB.PSITestSystems, "c_sys14"; add_forecasts = false))
    PSY.set_units_base_system!(sys, "SYSTEM_BASE")
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
    ac = pq[1:3]
    dcbuses = PSY.DCBus[]
    for k in 1:3
        dcb = PSY.DCBus(;
            number = 100 + k,
            name = "dc$k",
            available = true,
            magnitude = 1.0,
            voltage_limits = (min = 0.8, max = 1.2),
            base_voltage = 230.0,
        )
        PSY.add_component!(sys, dcb)
        push!(dcbuses, dcb)
    end
    # converters: dc1 = Vdc slack (1.05), dc2/dc3 = power orders
    configs = (
        (dc_control = PSY.VSCDCControlModes.DC_VOLTAGE, dc_setpoint = 1.05),
        (dc_control = PSY.VSCDCControlModes.DC_POWER, dc_setpoint = 0.30),
        (dc_control = PSY.VSCDCControlModes.DC_POWER, dc_setpoint = 0.20),
    )
    for k in 1:3
        ic = PSY.InterconnectingConverter(;
            name = "ic$k",
            available = true,
            bus = ac[k],
            dc_bus = dcbuses[k],
            active_power = 0.0,
            rating = 3.0,
            active_power_limits = (min = -3.0, max = 3.0),
            base_power = 100.0,
            dc_control = configs[k].dc_control,
            ac_control = PSY.VSCACControlModes.AC_REACTIVE_POWER,
            dc_setpoint = configs[k].dc_setpoint,
        )
        PSY.add_component!(sys, ic)
    end
    # DC branches dc1–dc2 and dc2–dc3
    for (a, c) in ((1, 2), (2, 3))
        arc = PSY.Arc(; from = dcbuses[a], to = dcbuses[c])
        PSY.add_component!(sys, arc)
        dcl = PSY.TModelHVDCLine(;
            name = "dcline$(a)$(c)",
            available = true,
            active_power_flow = 0.0,
            arc = arc,
            r = 0.01,
            l = 0.0,
            c = 0.0,
            active_power_limits_from = (min = -5.0, max = 5.0),
            active_power_limits_to = (min = -5.0, max = 5.0),
        )
        PSY.add_component!(sys, dcl)
    end
    return sys
end

@testset "VSC I5: 3-terminal MTDC lowers and solves across all formulations" begin
    sys = _build_mtdc_system()
    data0 = PowerFlowData(
        ACPowerFlow{NewtonRaphsonACPowerFlow}(; solver_settings = VSC_SETTINGS),
        sys,
    )
    dcn0 = PF.get_dc_network(data0)
    @test PF.n_vsc_converters(dcn0) == 3
    @test PF.n_dc_nodes(dcn0) == 3
    @test PF.n_dc_branches(dcn0) == 2
    @test count(dcn0.node_is_slack) == 1   # only ic1 controls V_dc

    sol = Dict{String, Any}()
    for (name, PF_T) in (
        ("polar", PF.ACPolarPowerFlow{NewtonRaphsonACPowerFlow}),
        ("rect", PF.ACRectangularPowerFlow{NewtonRaphsonACPowerFlow}),
        ("mixed", PF.ACMixedPowerFlow{NewtonRaphsonACPowerFlow}),
    )
        sys_k = _build_mtdc_system()
        data = PowerFlowData(PF_T(; solver_settings = VSC_SETTINGS), sys_k)
        @test solve_power_flow!(data)
        dcn = PF.get_dc_network(data)
        # Converter order follows component iteration (hash order), so assert order-independently:
        # the single Vdc-controlling converter pins its DC node at the setpoint, ...
        slack_nodes = findall(dcn.node_is_slack)
        @test length(slack_nodes) == 1
        @test isapprox(dcn.node_vdc[slack_nodes[1], 1], 1.05; atol = 1e-6)
        # ... and the two power-controlled converters hold their P orders {0.30, 0.20}.
        pq_p = sort([
            dcn.p_c[c, 1]
            for c in 1:3 if dcn.converter_mode[c] == PF.ControlPQ
        ])
        @test isapprox(pq_p, [0.20, 0.30]; atol = 1e-6)
        sol[name] = (copy(dcn.node_vdc[:, 1]), copy(dcn.p_c[:, 1]), copy(dcn.q_c[:, 1]))
    end
    for other in ("rect", "mixed")
        @test isapprox(sol["polar"][1], sol[other][1]; atol = 1e-6)
        @test isapprox(sol["polar"][2], sol[other][2]; atol = 1e-6)
        @test isapprox(sol["polar"][3], sol[other][3]; atol = 1e-6)
    end
end
