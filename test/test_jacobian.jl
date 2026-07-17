function verify_jacobian(
    sys::PSY.System;
    pf::PF.ACPowerFlow = PF.ACPowerFlow{NewtonRaphsonACPowerFlow}(;
        correct_bustypes = true,
    ),
    label::String = "",
    perturbation::Float64 = 0.02,
    seed::Int = 42,
)
    data = PF.PowerFlowData(pf, sys)
    time_step = 1
    residual = PF.ACPowerFlowResidual(data, time_step)
    J = PF.ACPowerFlowJacobian(residual, time_step)
    x0 = PF.calculate_x0(data, time_step)
    # Verify away from the flat-start state. At flat start θ=0 for every bus,
    # which silently zeroes all `sin(Δθ)` cross-terms — a sign flip in the
    # symbolic Jacobian for those entries would not be detected. A small
    # deterministic perturbation breaks the symmetry.
    if perturbation > 0
        Random.seed!(seed)
        x0 .+= perturbation .* randn(length(x0))
    end
    residual(x0, time_step)
    J(time_step)
    verify_jacobian_asymptotic(
        residual, deepcopy(J.Jv), x0, time_step; label = label,
    )
end

@testset "Jacobian verification" begin
    sys = PSB.build_system(PSITestSystems, "c_sys14")
    verify_jacobian(sys; label = "polar c_sys14")
end

@testset "Jacobian verification with LCC" begin
    sys = System(100.0)
    b1 = _add_simple_bus!(sys, 1, ACBusTypes.REF, 230, 1.1, 0.0)
    b2 = _add_simple_bus!(sys, 2, ACBusTypes.PQ, 230, 1.1, 0.0)
    b3 = _add_simple_bus!(sys, 3, ACBusTypes.PQ, 230, 1.1, 0.0)
    ld2 = _add_simple_load!(sys, b2, 10, 5)
    ld3 = _add_simple_load!(sys, b3, 60, 20)
    l12 = _add_simple_line!(sys, b1, b2, 5e-3, 5e-3, 1e-3)
    l13 = _add_simple_line!(sys, b1, b3, 5e-3, 5e-3, 1e-3)
    s1 = _add_simple_source!(sys, b1, 0.0, 0.0)
    lcc = _add_simple_lcc!(sys, b2, b3, 0.05, 0.05, 0.08)
    # `_add_simple_lcc!` already sets rectifier_delay_angle = 0.01 > 0; the
    # default inverter_extinction_angle is 0.0 which makes ϕ_i hit the
    # acos clamp boundary (sin(ϕ_i) = 0) at x0 — that's a separate boundary
    # test (see "Jacobian verification with LCC at inverter ϕ clamp"
    # below). Bump α_i here to verify the interior regime.
    PSY.set_inverter_extinction_angle!(lcc, 1.0)
    # Smaller perturbation here so the LCC α tail entries (α_r ≈ 0.087,
    # α_i = 1.0) stay clear of the min-thyristor-angle clamp.
    verify_jacobian(sys; label = "polar 3-bus LCC", perturbation = 0.01)
end

@testset "Jacobian verification with LCC, inverter-side setpoint" begin
    # A negative transfer setpoint puts the P-setpoint constraint on the
    # inverter: F_t_fb = −P_lcc_to − P_set, so the F_t_fb Jacobian row must
    # carry ∂/∂(V_tb, tap_i, α_i) instead of ∂/∂(V_fb, tap_r, α_r). Before
    # the fix the row still held the rectifier-side derivatives — the
    # asymptotic verifier catches that as order-1 decay on those columns.
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
    PSY.set_inverter_extinction_angle!(lcc, 1.0)   # interior, off the ϕ clamp
    PSY.set_transfer_setpoint!(lcc, -50.0)          # setpoint at inverter
    verify_jacobian(sys; label = "polar 3-bus LCC, inverter-side setpoint",
        perturbation = 0.01)
end

@testset "Jacobian verification with LCC at a PV terminal" begin
    # An LCC terminal at a PV bus: state is (Q_gen, θ), with V fixed at V_set.
    # The bus Q-balance still depends on tap_r/α_r through the LCC's Q
    # contribution, so ∂Q/∂tap and ∂Q/∂α must be filled in the Jacobian
    # for the PV terminal — they previously weren't, leaving these entries
    # stuck at 0 from the sparsity pattern.
    sys = System(100.0)
    b1 = _add_simple_bus!(sys, 1, ACBusTypes.REF, 230, 1.1, 0.0)
    b2 = _add_simple_bus!(sys, 2, ACBusTypes.PV, 230, 1.05, 0.0)   # PV terminal
    b3 = _add_simple_bus!(sys, 3, ACBusTypes.PQ, 230, 1.1, 0.0)
    _add_simple_thermal_standard!(sys, b2, 0.2, 0.1)  # generator at PV bus
    _add_simple_load!(sys, b2, 10, 5)
    _add_simple_load!(sys, b3, 60, 20)
    _add_simple_line!(sys, b1, b2, 5e-3, 5e-3, 1e-3)
    _add_simple_line!(sys, b1, b3, 5e-3, 5e-3, 1e-3)
    _add_simple_source!(sys, b1, 0.0, 0.0)
    lcc = _add_simple_lcc!(sys, b2, b3, 0.05, 0.05, 0.08)
    PSY.set_inverter_extinction_angle!(lcc, 1.0)
    verify_jacobian(sys; label = "polar 3-bus LCC, PV rectifier terminal",
        perturbation = 0.01)
end

@testset "Jacobian verification with LCC at inverter ϕ clamp" begin
    # Drive the inverter into the `raw < -1` clamp of `_calculate_ϕ_lcc`:
    # need cos(α_i) + x_t·I_dc/(√2·V·tap) > 1. Large inverter x_t plus a
    # small extinction angle does it. This exercises the `sin(ϕ) → 0`
    # boundary guards in the dP/dV, dP/dt helpers — without those guards
    # the analytic Jacobian disagrees with the residual at the inverter,
    # and the asymptotic verifier would catch it as order-1 decay.
    sys = System(100.0)
    b1 = _add_simple_bus!(sys, 1, ACBusTypes.REF, 230, 1.1, 0.0)
    b2 = _add_simple_bus!(sys, 2, ACBusTypes.PQ, 230, 1.1, 0.0)
    b3 = _add_simple_bus!(sys, 3, ACBusTypes.PQ, 230, 1.1, 0.0)
    _add_simple_load!(sys, b2, 10, 5)
    _add_simple_load!(sys, b3, 60, 20)
    _add_simple_line!(sys, b1, b2, 5e-3, 5e-3, 1e-3)
    _add_simple_line!(sys, b1, b3, 5e-3, 5e-3, 1e-3)
    _add_simple_source!(sys, b1, 0.0, 0.0)
    # xc_i = 0.20 (large), α_i = 0.1 rad (small) → inverter raw < -1, ϕ = π.
    lcc = _add_simple_lcc!(sys, b2, b3, 0.05, 0.05, 0.20)
    PSY.set_inverter_extinction_angle!(lcc, 0.1)
    PSY.set_rectifier_delay_angle!(lcc, 0.1)
    verify_jacobian(
        sys; label = "polar 3-bus LCC, inverter ϕ-clamp", perturbation = 0.01,
    )
end

@testset "Jacobian verification with LCC, realistic inverter (interior, tap≠1, NBR>1)" begin
    # The regime large interconnection-scale planning cases hit and the tap=1 / α≈0 tests
    # above never exercised:
    # extinction/delay angles ~15-18°, transformer taps off nominal, 2 bridges per side.
    # With the corrected inverter commutation (drop SUBTRACTS), ϕ_i stays interior
    # (sin ϕ_i > 0) so the converter carries reactive power, and the −xtr_i sign on the
    # inverter's commutation-chain derivatives must make the analytic Jacobian match the
    # residual. A wrong inverter commutation sign shows up as order-1 decay here.
    sys = System(100.0)
    b1 = _add_simple_bus!(sys, 1, ACBusTypes.REF, 230, 1.05, 0.0)
    b2 = _add_simple_bus!(sys, 2, ACBusTypes.PQ, 230, 1.02, 0.0)
    b3 = _add_simple_bus!(sys, 3, ACBusTypes.PQ, 230, 1.0, 0.0)
    _add_simple_load!(sys, b2, 10, 5)
    _add_simple_load!(sys, b3, 60, 20)
    _add_simple_line!(sys, b1, b2, 5e-3, 5e-3, 1e-3)
    _add_simple_line!(sys, b1, b3, 5e-3, 5e-3, 1e-3)
    _add_simple_source!(sys, b1, 0.0, 0.0)
    lcc = _add_simple_lcc!(sys, b2, b3, 0.02, 0.03, 0.04)
    PSY.set_rectifier_delay_angle!(lcc, deg2rad(15))
    PSY.set_inverter_extinction_angle!(lcc, deg2rad(18))
    PSY.set_rectifier_tap_setting!(lcc, 0.9)
    PSY.set_inverter_tap_setting!(lcc, 0.95)
    PSY.set_rectifier_bridges!(lcc, 2)
    PSY.set_inverter_bridges!(lcc, 2)
    verify_jacobian(sys; label = "polar 3-bus LCC, realistic interior inverter",
        perturbation = 0.005)
end

@testset "LCC inverter reactive power is nonzero at the solution (regression)" begin
    # Regression for the inverter ϕ-commutation-sign defect: before the fix a realistic
    # small-γ inverter had its commutation drop ADDED, driving raw = −(cos γ + comm) < −1,
    # so ϕ_i clamped to π (sin ϕ_i = 0). That zeroed the inverter's reactive draw and let
    # the terminal voltage run away. With the fix the inverter stays interior and consumes
    # reactive power. Assert the solved inverter is well off the clamp.
    sys = System(100.0)
    b1 = _add_simple_bus!(sys, 1, ACBusTypes.REF, 230, 1.0, 0.0)
    b2 = _add_simple_bus!(sys, 2, ACBusTypes.PQ, 230, 1.0, 0.0)
    b3 = _add_simple_bus!(sys, 3, ACBusTypes.PQ, 230, 1.0, 0.0)
    _add_simple_load!(sys, b2, 10, 5)
    _add_simple_load!(sys, b3, 60, 20)
    _add_simple_line!(sys, b1, b2, 5e-3, 5e-3, 1e-3)
    _add_simple_line!(sys, b1, b3, 5e-3, 5e-3, 1e-3)
    _add_simple_source!(sys, b1, 0.0, 0.0)
    lcc = _add_simple_lcc!(sys, b2, b3, 0.02, 0.04, 0.08)
    PSY.set_rectifier_delay_angle!(lcc, deg2rad(15))
    PSY.set_inverter_extinction_angle!(lcc, deg2rad(17))
    PSY.set_inverter_tap_setting!(lcc, 0.95)
    pf = PF.ACPowerFlow{NewtonRaphsonACPowerFlow}(; correct_bustypes = true)
    data = PF.PowerFlowData(pf, sys)
    @test PF.solve_power_flow!(data)
    # Off the acos clamp: sin(ϕ_i) = 0 would mean zero reactive contribution.
    @test sin(data.lcc.inverter.phi[1, 1]) > 0.1
    # Nonzero DC current carrying the transfer.
    @test data.lcc.i_dc[1, 1] > 0.0
end

@testset "Jacobian verification with ZIP load" begin
    sys = System(100.0)
    b1 = _add_simple_bus!(sys, 1, ACBusTypes.REF, 230, 1.0, 0.0)
    b2 = _add_simple_bus!(sys, 2, ACBusTypes.PQ, 230, 1.0, 0.0)
    _add_simple_line!(sys, b1, b2, 5e-3, 5e-3, 1e-3)
    _add_simple_source!(sys, b1, 0.0, 0.0)
    _add_simple_zip_load!(
        sys, b2;
        constant_power_active_power = 0.5,
        constant_power_reactive_power = 0.2,
        constant_current_active_power = 2.0,
        constant_current_reactive_power = 1.0,
        constant_impedance_active_power = 1.5,
        constant_impedance_reactive_power = 0.8,
    )
    verify_jacobian(sys; label = "polar ZIP")
end

@testset "Jacobian verification with distributed slack" begin
    sys = PSB.build_system(PSITestSystems, "c_sys14")
    generators = collect(get_components(ThermalStandard, sys))
    # Assign distinct nonzero participation factors to all generators (REF and PV buses).
    # This exercises the cross-terms ∂F_P_k/∂x[2*ref-1] = -c_k for PV buses
    # and the corrected REF diagonal ∂F_P_ref/∂x[2*ref-1] = -c_ref.
    gspf = Dict(
        (ThermalStandard, get_name(g)) => Float64(i)
        for (i, g) in enumerate(generators)
    )
    pf = PF.ACPowerFlow{NewtonRaphsonACPowerFlow}(;
        correct_bustypes = true,
        generator_slack_participation_factors = gspf,
    )
    verify_jacobian(sys; pf = pf, label = "polar c_sys14 distributed-slack")
end
