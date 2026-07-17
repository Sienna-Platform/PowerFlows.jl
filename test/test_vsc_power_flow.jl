# VSC HVDC power-flow tests. I0: lowering of a point-to-point TwoTerminalVSCLine into the internal
# DCNetwork (isolated 2-node). Later increments add the residual/Jacobian/solver tests.

# DC-network modeling is on by default; `VSC_SETTINGS` (test_utils/common.jl) passes it
# explicitly for clarity.

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

# I1: a point-to-point VSC (from = DC-voltage control / DC slack, to = P,Q control) on two PQ
# buses, zero converter loss (`_build_vsc_pq_system`, test_utils/common.jl). Solve with polar NR
# and verify convergence + setpoints + Jacobian.
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

@testset "VSC I2: DC-voltage droop — both converters follow V_dc = V_set + k·P_c" begin
    sys, _, _ = _vsc_system(;
        g = 50.0,
        dc_control_from = PSY.VSCDCControlModes.DC_VOLTAGE_DROOP,
        dc_voltage_droop_from = 0.02,
        dc_setpoint_from = 1.05,
        reactive_power_from = 0.0,
        dc_control_to = PSY.VSCDCControlModes.DC_VOLTAGE_DROOP,
        dc_voltage_droop_to = 0.03,
        dc_setpoint_to = 1.03,
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
            dcn.vdc_set[c, 1] + dcn.droop_k[c] * dcn.p_c[c, 1];
            atol = 1e-7,
        )
        # Beerten droop direction: a converter injecting into the DC grid (p_c < 0) sits below
        # its setpoint; a withdrawing one (p_c > 0) sits above.
        if abs(dcn.p_c[c, 1]) >= 1e-9
            @test sign(dcn.node_vdc[node, 1] - dcn.vdc_set[c, 1]) == sign(dcn.p_c[c, 1])
        end
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

# Regression: the polar VSC Jacobian must be bus-type aware. Column `2ix-1` is the |V_ac| state
# only for PQ buses; for PV it is Q_gen and for REF it is P_gen (see state_indexing_helpers.jl).
# A lossy converter whose AC terminal is a PV (or REF) bus has a nonzero ∂KCL/∂|V_ac| loss term —
# writing it into column `2ix-1` (which is not |V_ac| there) corrupts the Jacobian. |V_ac| is fixed
# at PV/REF buses, so that derivative must not enter the Jacobian at all.
function _vsc_system_pv_terminal(; g = 45.0)
    sys = deepcopy(PSB.build_system(PSB.PSITestSystems, "c_sys14"; add_forecasts = false))
    pick(t) = first(
        sort!(
            collect(PSY.get_components(b -> PSY.get_bustype(b) == t, PSY.ACBus, sys));
            by = PSY.get_number,
        ),
    )
    from_bus = pick(PSY.ACBusTypes.PQ)        # DC-voltage slack converter on a PQ bus
    to_bus = pick(PSY.ACBusTypes.PV)          # lossy power-control converter on a PV bus
    arc = _get_or_make_arc(sys, from_bus, to_bus)
    vsc = PSY.TwoTerminalVSCLine(;
        name = "vsc_pv",
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
        converter_loss_to = PSY.QuadraticCurve(0.01, 0.02, 0.005),
    )
    PSY.add_component!(sys, vsc)
    return sys
end

@testset "VSC: analytic polar Jacobian matches FD for a lossy converter on a PV bus" begin
    sys = _vsc_system_pv_terminal(; g = 45.0)
    data = PowerFlowData(
        ACPowerFlow{NewtonRaphsonACPowerFlow}(; solver_settings = VSC_SETTINGS),
        sys,
    )
    # check at the flat start to isolate the Jacobian structure from solver convergence
    residual = PF.ACPowerFlowResidual(data, 1)
    jac = PF.ACPowerFlowJacobian(residual, 1)
    x = PF.calculate_x0(data, 1)
    residual(x, 1)
    jac(1)
    verify_jacobian_asymptotic(residual, jac.Jv, x, 1; label = "VSC polar lossy-on-PV")
end

# A point-to-point VSC with g = 0 is an open DC link: `_build_G_dc` yields an all-zero DC
# conductance matrix, so the DC nodal balance is singular and the joint AC↔DC solve cannot
# converge (no DC path for a DC-power setpoint). Such a degenerate line must be excluded from the
# DCNetwork (and the AC solve falls back to ignoring it, as before VSC support), not break the solve.
@testset "VSC: a zero-conductance (open) DC link is excluded from joint modeling" begin
    sys, _, _ = _vsc_system(;
        g = 0.0,
        dc_control_from = PSY.VSCDCControlModes.DC_VOLTAGE,
        ac_control_from = PSY.VSCACControlModes.AC_REACTIVE_POWER,
        dc_setpoint_from = 1.03,
        dc_control_to = PSY.VSCDCControlModes.DC_POWER,
        ac_control_to = PSY.VSCACControlModes.AC_REACTIVE_POWER,
        dc_setpoint_to = 0.3,
    )
    data = PowerFlowData(
        ACPowerFlow{NewtonRaphsonACPowerFlow}(; solver_settings = VSC_SETTINGS),
        sys,
    )
    dcn = PF.get_dc_network(data)
    @test !PF.has_dc_network(dcn)   # the g = 0 line is not lowered into the DCNetwork
    @test solve_power_flow!(data)   # the AC power flow still converges
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

# ── Coverage: control modes whose analytic Jacobian / cross-formulation parity were previously
# exercised only for polar-NR convergence. `∂r1/∂P_c = -k` (droop) and the AC-voltage pin rows
# (`-2e/-2f` in rect/mixed) appear in no earlier FD-parity test.
const _ALL_AC_FORMULATIONS = (
    ("polar", PF.ACPolarPowerFlow{NewtonRaphsonACPowerFlow}),
    ("rect", PF.ACRectangularPowerFlow{NewtonRaphsonACPowerFlow}),
    ("mixed", PF.ACMixedPowerFlow{NewtonRaphsonACPowerFlow}),
)

# Two droop converters anchor the DC subnet via the droop relation (no V_dc slack), so this
# exercises the droop control row and its `∂r1/∂P_c = -k`, `∂r1/∂V_dc = 1` partials.
function _vsc_droop_system()
    sys, _, _ = _vsc_system(;
        g = 50.0,
        dc_control_from = PSY.VSCDCControlModes.DC_VOLTAGE_DROOP,
        dc_voltage_droop_from = 0.02,
        dc_setpoint_from = 1.05,
        reactive_power_from = 0.0,
        dc_control_to = PSY.VSCDCControlModes.DC_VOLTAGE_DROOP,
        dc_voltage_droop_to = 0.03,
        dc_setpoint_to = 1.03,
        reactive_power_to = 0.0,
        converter_loss_to = PSY.QuadraticCurve(0.01, 0.02, 0.005),
    )
    return sys
end

# from pins V_dc and AC voltage (ControlVdcQ), to holds P and pins AC voltage (ControlPVac): both
# `r2` rows are the `|V_ac|^2 - V_set^2` pin, exercising `∂r2/∂|V_ac|` (polar) / `-2e,-2f` (rect/mixed).
function _vsc_ac_voltage_system()
    sys, _, _ = _vsc_system(;
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
    return sys
end

@testset "VSC: DC-voltage droop — analytic Jacobian matches FD ($name)" for (name, PF_T) in
                                                                            _ALL_AC_FORMULATIONS
    pf = PF_T(; solver_settings = VSC_SETTINGS)
    data = PowerFlowData(pf, _vsc_droop_system())
    @test solve_power_flow!(data)
    residual, jac, x = PF.initialize_power_flow_variables(pf, data, 1)
    residual(x, 1)
    jac(1)
    verify_jacobian_asymptotic(residual, jac.Jv, x, 1; label = "VSC droop $name")
end

@testset "VSC: DC-voltage droop — polar, rectangular, and mixed all agree" begin
    sol = Dict{String, Any}()
    for (name, PF_T) in _ALL_AC_FORMULATIONS
        data = PowerFlowData(PF_T(; solver_settings = VSC_SETTINGS), _vsc_droop_system())
        @test solve_power_flow!(data)
        dcn = PF.get_dc_network(data)
        sol[name] = (
            copy(data.bus_magnitude[:, 1]),
            copy(dcn.p_c[:, 1]),
            copy(dcn.q_c[:, 1]),
            copy(dcn.node_vdc[:, 1]),
        )
    end
    for other in ("rect", "mixed")
        for k in 1:4
            @test isapprox(sol["polar"][k], sol[other][k]; atol = 1e-6)
        end
    end
end

@testset "VSC: AC-voltage control — analytic Jacobian matches FD ($name)" for (
    name,
    PF_T,
) in
                                                                              _ALL_AC_FORMULATIONS
    pf = PF_T(; solver_settings = VSC_SETTINGS)
    data = PowerFlowData(pf, _vsc_ac_voltage_system())
    @test solve_power_flow!(data)
    residual, jac, x = PF.initialize_power_flow_variables(pf, data, 1)
    residual(x, 1)
    jac(1)
    verify_jacobian_asymptotic(residual, jac.Jv, x, 1; label = "VSC ac-voltage $name")
end

@testset "VSC: AC-voltage control — polar, rectangular, and mixed all agree" begin
    sol = Dict{String, Any}()
    for (name, PF_T) in _ALL_AC_FORMULATIONS
        data =
            PowerFlowData(PF_T(; solver_settings = VSC_SETTINGS), _vsc_ac_voltage_system())
        @test solve_power_flow!(data)
        dcn = PF.get_dc_network(data)
        sol[name] = (
            copy(data.bus_magnitude[:, 1]),
            copy(dcn.p_c[:, 1]),
            copy(dcn.q_c[:, 1]),
            copy(dcn.node_vdc[:, 1]),
        )
    end
    for other in ("rect", "mixed")
        for k in 1:4
            @test isapprox(sol["polar"][k], sol[other][k]; atol = 1e-6)
        end
    end
end

# Multi-time-step: DCNetwork state is (n_conv, n_time); PSI drives VSC solves across many periods.
# Verify every step converges and pins its setpoints, and that the per-step AC solutions actually
# differ (so the tail is solved per-step, not copied from t=1).
@testset "VSC: multi-time-step solve holds setpoints at every step ($name)" for (
    name,
    PF_T,
) in
                                                                                _ALL_AC_FORMULATIONS
    time_steps = 4
    sys, from_no, to_no = _vsc_system(;
        g = 50.0,
        dc_control_from = PSY.VSCDCControlModes.DC_VOLTAGE,
        ac_control_from = PSY.VSCACControlModes.AC_REACTIVE_POWER,
        dc_setpoint_from = 1.04,
        reactive_power_from = 0.0,
        dc_control_to = PSY.VSCDCControlModes.DC_POWER,
        ac_control_to = PSY.VSCACControlModes.AC_REACTIVE_POWER,
        dc_setpoint_to = 0.3,
        reactive_power_to = 0.05,
    )
    pf = PF_T(; time_steps = time_steps, solver_settings = VSC_SETTINGS)
    data = PowerFlowData(pf, sys)
    prepare_ts_data!(data, time_steps)
    @test solve_power_flow!(data)
    dcn = PF.get_dc_network(data)
    for t in 1:time_steps
        node_from = dcn.converter_dc_node_ix[1]
        @test isapprox(dcn.node_vdc[node_from, t], dcn.vdc_set[1, t]; atol = 1e-7)  # from pins V_dc
        @test isapprox(dcn.p_c[2, t], dcn.p_set[2, t]; atol = 1e-7)                 # to holds P
    end
    # The varying per-step AC injections must produce distinct solutions, proving genuine per-step
    # solves rather than a replicated t=1 answer.
    bus_lookup = PF.get_bus_lookup(data)
    from_ix = bus_lookup[from_no]
    @test !isapprox(data.bus_magnitude[from_ix, 1], data.bus_magnitude[from_ix, time_steps])
end

# I5 / M2: genuine multi-terminal DC grid (`_build_mtdc_system`, test_utils/common.jl) — DCBus
# nodes + InterconnectingConverter (AC↔DC) + TModelHVDCLine DC branches. Three terminals: one
# DC-voltage slack, two power-controlled.
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

# Results layer: a joint AC↔DC solve surfaces converter/DC-line state in dedicated tables —
# `vsc_results` (point-to-point TwoTerminalVSCLine), `mtdc_results` (InterconnectingConverter), and
# `mtdc_line_results` (TModelHVDCLine). A non-applicable table comes back empty (uniform schema).
@testset "VSC results: MTDC populates converter and DC-line tables" begin
    sys = _build_mtdc_system()
    pf = ACPowerFlow{NewtonRaphsonACPowerFlow}(; solver_settings = VSC_SETTINGS)
    results = solve_power_flow(pf, sys)
    @test DataFrames.nrow(results["vsc_results"]) == 0   # no point-to-point VSC lines

    mtdc = results["mtdc_results"]
    @test DataFrames.nrow(mtdc) == 3
    @test Set(mtdc.converter_name) == Set(["ic1", "ic2", "ic3"])
    # exactly one DC-voltage converter, pinned at V_dc = 1.05
    @test count(m -> m == string(PF.ControlVdc), mtdc.mode) == 1
    @test count(v -> isapprox(v, 1.05; atol = 1e-6), mtdc.Vdc) == 1

    mtdc_line = results["mtdc_line_results"]
    @test DataFrames.nrow(mtdc_line) == 2
    @test Set(mtdc_line.line_name) == Set(["dcline12", "dcline23"])
    @test all(>=(0.0), mtdc_line.P_losses)   # I²r loss is non-negative
end

@testset "VSC results: point-to-point line populates the vsc table" begin
    sys, from_no, to_no = _vsc_system(;
        g = 50.0,
        dc_control_from = PSY.VSCDCControlModes.DC_VOLTAGE,
        dc_setpoint_from = 1.03,
        dc_control_to = PSY.VSCDCControlModes.DC_POWER,
        dc_setpoint_to = 0.3,
    )
    pf = ACPowerFlow{NewtonRaphsonACPowerFlow}(; solver_settings = VSC_SETTINGS)
    results = solve_power_flow(pf, sys)
    @test DataFrames.nrow(results["mtdc_results"]) == 0
    @test DataFrames.nrow(results["mtdc_line_results"]) == 0

    vsc = results["vsc_results"]
    @test DataFrames.nrow(vsc) == 1
    @test vsc.bus_from[1] == from_no
    @test vsc.bus_to[1] == to_no
    @test isapprox(vsc.Vdc_from[1], 1.03; atol = 1e-6)   # DC-voltage terminal holds V_dc
    @test vsc.P_losses[1] >= -1e-8
end

# R3: converter AC terminals must be marked irreducible so network reduction can never remove a
# VSC/IC bus (a reduced-away terminal would silently drop the converter). The protection set comes
# from `_dc_converter_ac_buses`, passed to PNM as `irreducible_buses` for every PowerFlowData.
@testset "VSC: converter AC terminals are collected as irreducible buses" begin
    sys, from_no, to_no = _vsc_system(;
        g = 50.0,
        dc_control_from = PSY.VSCDCControlModes.DC_VOLTAGE,
        dc_setpoint_from = 1.03,
        dc_control_to = PSY.VSCDCControlModes.DC_POWER,
        dc_setpoint_to = 0.3,
    )
    irr = PF._dc_converter_ac_buses(sys)
    @test from_no in irr
    @test to_no in irr
    # a g = 0 (open, unmodeled) VSC line contributes no protected buses
    sys0, _, _ = _vsc_system(;
        g = 0.0,
        dc_control_from = PSY.VSCDCControlModes.DC_VOLTAGE,
        dc_setpoint_from = 1.03,
        dc_control_to = PSY.VSCDCControlModes.DC_POWER,
        dc_setpoint_to = 0.3,
    )
    @test isempty(PF._dc_converter_ac_buses(sys0))
    # MTDC: every interconnecting-converter AC bus is collected
    msys = _build_mtdc_system()
    ic_buses = Set(
        PSY.get_number(PSY.get_bus(ic)) for
        ic in PSY.get_components(PSY.InterconnectingConverter, msys)
    )
    @test !isempty(ic_buses)
    @test issubset(ic_buses, PF._dc_converter_ac_buses(msys))
end

# N4: PSS/E .raw has no DC-voltage-droop representation, so a droop converter is exported as MW
# (power) control — TYPE 2 — with its DC setpoint as the scheduled active power, not the droop
# reference voltage; only a strict DC_VOLTAGE terminal is the TYPE 1 DC-voltage controller, and the
# DC line's base voltage is sourced from a voltage-referencing terminal (DC_VOLTAGE or droop).
@testset "VSC export: a droop converter maps to MW control (TYPE 2)" begin
    @test PF._vsc_export_dc_type(PSY.VSCDCControlModes.DC_VOLTAGE) == 1
    @test PF._vsc_export_dc_type(PSY.VSCDCControlModes.DC_POWER) == 2
    @test PF._vsc_export_dc_type(PSY.VSCDCControlModes.DC_VOLTAGE_DROOP) == 2
    @test PF._has_dc_voltage_reference(PSY.VSCDCControlModes.DC_VOLTAGE)
    @test PF._has_dc_voltage_reference(PSY.VSCDCControlModes.DC_VOLTAGE_DROOP)
    @test !PF._has_dc_voltage_reference(PSY.VSCDCControlModes.DC_POWER)

    sys, _, _ = _vsc_system(;
        g = 50.0,
        dc_control_from = PSY.VSCDCControlModes.DC_VOLTAGE,
        dc_setpoint_from = 1.04,
        dc_control_to = PSY.VSCDCControlModes.DC_VOLTAGE_DROOP,
        dc_voltage_droop_to = 0.05,
        dc_setpoint_to = 1.0,
    )
    vsc = first(PSY.get_components(PSY.TwoTerminalVSCLine, sys))
    base = PSY.get_base_power(sys)
    # strict DC_VOLTAGE terminal keeps its voltage setpoint as DCSET
    @test PF._vsc_export_dcset(vsc, :from, base) == PSY.get_dc_setpoint_from(vsc)
    # droop terminal's DCSET is the scheduled active-power demand (MW, to side receives -P_flow)
    @test isapprox(
        PF._vsc_export_dcset(vsc, :to, base),
        -PSY.get_active_power_flow(vsc) * base;
        atol = 1.0,
    )
end

# Two InterconnectingConverters on the same AC bus contend for control of that bus's voltage with no
# defined arbitration, so lowering rejects it (`_validate_ic_ac_buses`). ICs may still share a DC
# bus (a normal MTDC topology). `shared_ac = true` puts ic2/ic3 on the same AC bus pq[2] (rejected);
# `false` gives them distinct AC buses but the same DC node dc2 (a valid parallel-on-DC topology).
function _build_parallel_ic_system(; shared_ac::Bool = true)
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
    dcbuses = PSY.DCBus[]
    for k in 1:2
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
    if shared_ac
        ac3 = pq[2]
    else
        ac3 = pq[3]
    end
    # ic1 = Vdc slack on (pq[1], dc1); ic2 on (pq[2], dc2); ic3 on (ac3, dc2). ic2/ic3 always share
    # the DC node dc2 — `shared_ac` decides whether they also share the AC bus.
    configs = (
        (ac = pq[1], dc = dcbuses[1], mode = PSY.VSCDCControlModes.DC_VOLTAGE, set = 1.05),
        (ac = pq[2], dc = dcbuses[2], mode = PSY.VSCDCControlModes.DC_POWER, set = 0.15),
        (ac = ac3, dc = dcbuses[2], mode = PSY.VSCDCControlModes.DC_POWER, set = 0.10),
    )
    for (k, cfg) in enumerate(configs)
        ic = PSY.InterconnectingConverter(;
            name = "ic$k",
            available = true,
            bus = cfg.ac,
            dc_bus = cfg.dc,
            active_power = 0.0,
            rating = 3.0,
            active_power_limits = (min = -3.0, max = 3.0),
            base_power = 100.0,
            dc_control = cfg.mode,
            ac_control = PSY.VSCACControlModes.AC_REACTIVE_POWER,
            dc_setpoint = cfg.set,
            loss_function = PSY.QuadraticCurve(0.005, 0.01, 0.002),
        )
        PSY.add_component!(sys, ic)
    end
    arc = PSY.Arc(; from = dcbuses[1], to = dcbuses[2])
    PSY.add_component!(sys, arc)
    dcl = PSY.TModelHVDCLine(;
        name = "dcline12",
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
    return sys
end

@testset "VSC: two InterconnectingConverters on one AC bus are rejected" begin
    bad = _build_parallel_ic_system(; shared_ac = true)
    @test_throws ErrorException PowerFlowData(
        ACPowerFlow{NewtonRaphsonACPowerFlow}(; solver_settings = VSC_SETTINGS),
        bad,
    )
    # sharing only the DC node (distinct AC buses) is a valid MTDC topology and still solves
    ok = _build_parallel_ic_system(; shared_ac = false)
    data = PowerFlowData(
        ACPowerFlow{NewtonRaphsonACPowerFlow}(; solver_settings = VSC_SETTINGS),
        ok,
    )
    @test solve_power_flow!(data)
end

# Regression: a converter whose AC terminal is the REF bus. The rect/mixed REF rows are current
# balance with (P_gen, Q_gen) as the bus states — the converter couples to (P_c, Q_c) but NOT to
# the (P_gen, Q_gen) columns (e,f are fixed at REF), which the shared Jacobian writer must gate.
function _vsc_system_ref_terminal(; g = 45.0)
    sys = deepcopy(PSB.build_system(PSB.PSITestSystems, "c_sys14"; add_forecasts = false))
    pick(t) = first(
        sort!(
            collect(PSY.get_components(b -> PSY.get_bustype(b) == t, PSY.ACBus, sys));
            by = PSY.get_number,
        ),
    )
    arc = _get_or_make_arc(sys, pick(PSY.ACBusTypes.PQ), pick(PSY.ACBusTypes.REF))
    vsc = PSY.TwoTerminalVSCLine(;
        name = "vsc_ref",
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
        converter_loss_to = PSY.QuadraticCurve(0.01, 0.02, 0.005),
    )
    PSY.add_component!(sys, vsc)
    return sys
end

@testset "VSC: lossy converter on the REF bus — Jacobian and formulation parity" begin
    sol = Dict{String, Any}()
    for (label, PF_T) in (
        ("polar", PF.ACPolarPowerFlow{NewtonRaphsonACPowerFlow}),
        ("rect", PF.ACRectangularPowerFlow{NewtonRaphsonACPowerFlow}),
        ("mixed", PF.ACMixedPowerFlow{NewtonRaphsonACPowerFlow}),
    )
        sys = _vsc_system_ref_terminal(; g = 45.0)
        pf = PF_T(; solver_settings = VSC_SETTINGS)
        data = PowerFlowData(pf, sys)
        @test solve_power_flow!(data)
        residual, jac, x = PF.initialize_power_flow_variables(pf, data, 1)
        residual(x, 1)
        jac(1)
        verify_jacobian_asymptotic(
            residual, jac.Jv, x, 1;
            label = "VSC REF terminal $(label)",
        )
        dcn = PF.get_dc_network(data)
        sol[label] =
            (copy(data.bus_magnitude[:, 1]), copy(dcn.p_c[:, 1]), copy(dcn.q_c[:, 1]))
    end
    for other in ("rect", "mixed")
        @test isapprox(sol["polar"][1], sol[other][1]; atol = 1e-6)
        @test isapprox(sol["polar"][2], sol[other][2]; atol = 1e-6)
        @test isapprox(sol["polar"][3], sol[other][3]; atol = 1e-6)
    end
end

# Regression: a converter on a PV bus in the MIXED formulation. MCPB PV rows are (real-power
# balance, |V|² pin) — the converter must enter the power row as −P_c and leave the pin row
# untouched; treating them like rect current rows converges to a silently WRONG solution
# (the FD check passes because the Jacobian is consistent with the wrong residual — only
# cross-formulation parity catches it).
@testset "VSC: lossy converter on a PV bus — all three formulations agree" begin
    sol = Dict{String, Any}()
    for (label, PF_T) in (
        ("polar", PF.ACPolarPowerFlow{NewtonRaphsonACPowerFlow}),
        ("rect", PF.ACRectangularPowerFlow{NewtonRaphsonACPowerFlow}),
        ("mixed", PF.ACMixedPowerFlow{NewtonRaphsonACPowerFlow}),
    )
        sys = _vsc_system_pv_terminal(; g = 45.0)
        pf = PF_T(; solver_settings = VSC_SETTINGS)
        data = PowerFlowData(pf, sys)
        @test solve_power_flow!(data)
        residual, jac, x = PF.initialize_power_flow_variables(pf, data, 1)
        residual(x, 1)
        jac(1)
        verify_jacobian_asymptotic(
            residual, jac.Jv, x, 1;
            label = "VSC PV terminal $(label)",
        )
        dcn = PF.get_dc_network(data)
        sol[label] = (
            copy(data.bus_magnitude[:, 1]),
            copy(dcn.p_c[:, 1]),
            copy(dcn.node_vdc[:, 1]),
        )
    end
    for other in ("rect", "mixed")
        @test isapprox(sol["polar"][1], sol[other][1]; atol = 1e-6)
        @test isapprox(sol["polar"][2], sol[other][2]; atol = 1e-6)
        @test isapprox(sol["polar"][3], sol[other][3]; atol = 1e-6)
    end
end

# Lowering-time validation: an AC-voltage-controlling converter pins |V_ac|, which is singular at
# a bus whose magnitude is already regulated (PV/REF), and two AC-voltage converters on one bus
# duplicate the pin. Both must fail fast at PowerFlowData construction, not as a mid-solve
# singular Jacobian.
@testset "VSC: AC-voltage control is rejected on regulated buses and duplicate pins" begin
    # Vac converter on a PV bus → error at construction
    sys_pv =
        deepcopy(PSB.build_system(PSB.PSITestSystems, "c_sys14"; add_forecasts = false))
    pick(sys, t) = first(
        sort!(
            collect(PSY.get_components(b -> PSY.get_bustype(b) == t, PSY.ACBus, sys));
            by = PSY.get_number,
        ),
    )
    arc = _get_or_make_arc(sys_pv, pick(sys_pv, PSY.ACBusTypes.PQ),
        pick(sys_pv, PSY.ACBusTypes.PV))
    vsc = PSY.TwoTerminalVSCLine(;
        name = "vsc_bad_pv",
        available = true,
        arc = arc,
        active_power_flow = 0.3,
        rating = 2.0,
        active_power_limits_from = (min = -2.0, max = 2.0),
        active_power_limits_to = (min = -2.0, max = 2.0),
        g = 45.0,
        dc_control_from = PSY.VSCDCControlModes.DC_VOLTAGE,
        ac_control_from = PSY.VSCACControlModes.AC_REACTIVE_POWER,
        dc_setpoint_from = 1.03,
        dc_control_to = PSY.VSCDCControlModes.DC_POWER,
        ac_control_to = PSY.VSCACControlModes.AC_VOLTAGE,
        dc_setpoint_to = 0.3,
        ac_setpoint_to = 1.0,
    )
    PSY.add_component!(sys_pv, vsc)
    @test_throws ErrorException PowerFlowData(
        ACPowerFlow{NewtonRaphsonACPowerFlow}(; solver_settings = VSC_SETTINGS),
        sys_pv,
    )

    # two Vac converters on the same PQ bus → error at construction
    sys_dup =
        deepcopy(PSB.build_system(PSB.PSITestSystems, "c_sys14"; add_forecasts = false))
    pq = sort!(
        collect(
            PSY.get_components(
                b -> PSY.get_bustype(b) == PSY.ACBusTypes.PQ,
                PSY.ACBus,
                sys_dup,
            ),
        );
        by = PSY.get_number,
    )
    for (k, to_bus) in enumerate((pq[2], pq[3]))
        arc_k = _get_or_make_arc(sys_dup, pq[1], to_bus)
        vsc_k = PSY.TwoTerminalVSCLine(;
            name = "vsc_dup$k",
            available = true,
            arc = arc_k,
            active_power_flow = 0.2,
            rating = 2.0,
            active_power_limits_from = (min = -2.0, max = 2.0),
            active_power_limits_to = (min = -2.0, max = 2.0),
            g = 45.0,
            dc_control_from = PSY.VSCDCControlModes.DC_VOLTAGE,
            ac_control_from = PSY.VSCACControlModes.AC_VOLTAGE,
            dc_setpoint_from = 1.03,
            ac_setpoint_from = 1.0,
            dc_control_to = PSY.VSCDCControlModes.DC_POWER,
            ac_control_to = PSY.VSCACControlModes.AC_REACTIVE_POWER,
            dc_setpoint_to = 0.2,
        )
        PSY.add_component!(sys_dup, vsc_k)
    end
    @test_throws ErrorException PowerFlowData(
        ACPowerFlow{NewtonRaphsonACPowerFlow}(; solver_settings = VSC_SETTINGS),
        sys_dup,
    )
end

# Solver guards: RobustHomotopy has no DC-tail (VSC/MTDC) support (must reject at construction);
# the FDDecoupled variant handles the tail via a sequential sub-solve, which cannot honor
# AC-voltage control rows (must reject those too).
@testset "VSC: RobustHomotopy rejects DC networks; FDDecoupled rejects AC-voltage control" begin
    sys = _build_vsc_pq_system(; g = 50.0, p_set = 0.4, q_set = 0.1, vdc = 1.05)
    data = PowerFlowData(
        ACPowerFlow{PF.RobustHomotopyPowerFlow}(; solver_settings = VSC_SETTINGS),
        sys,
    )
    @test_throws ArgumentError solve_power_flow!(data)

    sys_vac, _, _ = _vsc_system(;
        g = 50.0,
        dc_control_from = PSY.VSCDCControlModes.DC_VOLTAGE,
        ac_control_from = PSY.VSCACControlModes.AC_VOLTAGE,
        dc_setpoint_from = 1.05,
        ac_setpoint_from = 1.01,
        dc_control_to = PSY.VSCDCControlModes.DC_POWER,
        ac_control_to = PSY.VSCACControlModes.AC_REACTIVE_POWER,
        dc_setpoint_to = 0.25,
    )
    data_vac = PowerFlowData(
        ACPowerFlow{PF.FastDecoupledACPowerFlow}(; solver_settings = VSC_SETTINGS),
        sys_vac,
    )
    @test_throws ArgumentError solve_power_flow!(data_vac)
end

# The FDDecoupled sequential VSC sub-solve: a LOSSY converter couples the DC tail to the AC
# voltages, so the tail must be re-solved each cycle (`_fd_vsc_substep!`) — frozen tail states
# previously left FDDecoupled unable to converge on any lossy VSC system.
@testset "VSC: FDDecoupled converges on a lossy VSC system and matches NR" begin
    lossy_kwargs = (;
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
    sol = Dict{String, Any}()
    for (label, S) in
        (("nr", NewtonRaphsonACPowerFlow), ("fd", PF.FastDecoupledACPowerFlow))
        sys, _, _ = _vsc_system(; lossy_kwargs...)
        data = PowerFlowData(ACPowerFlow{S}(; solver_settings = VSC_SETTINGS), sys)
        @test solve_power_flow!(data)
        dcn = PF.get_dc_network(data)
        sol[label] = (
            copy(data.bus_magnitude[:, 1]),
            copy(dcn.p_c[:, 1]),
            copy(dcn.q_c[:, 1]),
            copy(dcn.node_vdc[:, 1]),
        )
    end
    for i in 1:4
        @test isapprox(sol["nr"][i], sol["fd"][i]; atol = 1e-6)
    end
end

@testset "get_hvdc_results: multiperiod NamedTuple tables" begin
    time_steps = 3
    sys, from_no, to_no = _vsc_system(;
        g = 50.0,
        dc_control_from = PSY.VSCDCControlModes.DC_VOLTAGE,
        ac_control_from = PSY.VSCACControlModes.AC_REACTIVE_POWER,
        dc_setpoint_from = 1.04,
        reactive_power_from = 0.0,
        dc_control_to = PSY.VSCDCControlModes.DC_POWER,
        ac_control_to = PSY.VSCACControlModes.AC_REACTIVE_POWER,
        dc_setpoint_to = 0.3,
        reactive_power_to = 0.05,
    )
    pf = ACPowerFlow{NewtonRaphsonACPowerFlow}(;
        time_steps = time_steps, solver_settings = VSC_SETTINGS)
    data = PowerFlowData(pf, sys)
    prepare_ts_data!(data, time_steps)
    @test solve_power_flow!(data)
    res = get_hvdc_results(sys, data)
    @test DataFrames.nrow(res.vsc) == time_steps   # one VSC line, one row per step
    @test isempty(res.lcc)
    @test isempty(res.mtdc_converters)
    @test isempty(res.mtdc_lines)
    # Per-step slices equal the single-time-step builders.
    dcn = PF.get_dc_network(data)
    nrd = PNM.get_network_reduction_data(PF.get_power_network_matrix(data))
    base = PSY.get_base_power(sys)
    for t in 1:time_steps
        slice = res.vsc[res.vsc.time_step .== t, DataFrames.Not(:time_step)]
        direct = PF.vsc_results_dataframe(sys, data, dcn, nrd, base, t)
        @test slice == direct
    end
    # The P_to_from column closes the loss identity at every step.
    @test all(
        isapprox.(res.vsc.P_losses, res.vsc.P_from_to .+ res.vsc.P_to_from; atol = 1e-9),
    )
end

@testset "get_hvdc_results: MTDC converter and DC-line tables" begin
    sys = _build_mtdc_system()
    pf = ACPowerFlow{NewtonRaphsonACPowerFlow}(; solver_settings = VSC_SETTINGS)
    data = PowerFlowData(pf, sys)
    @test solve_power_flow!(data)
    res = get_hvdc_results(sys, data)
    @test DataFrames.nrow(res.mtdc_converters) == 3
    @test DataFrames.nrow(res.mtdc_lines) == 2
    @test all(res.mtdc_converters.time_step .== 1)
    @test isempty(res.vsc)
    @test isempty(res.lcc)
end
