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
    # Both thyristor angles must be strictly positive at x0; otherwise the
    # arccos in _calculate_ϕ_lcc hits the clamp boundary and sin(ϕ) = 0,
    # making the analytic Q-derivatives singular at the verification point.
    # _add_simple_lcc! already sets rectifier_delay_angle = 0.01 > 0; the
    # default inverter_extinction_angle is 0.0, so bump it here.
    PSY.set_inverter_extinction_angle!(lcc, 1.0)
    # Smaller perturbation here so the LCC α tail entries (α_r ≈ 0.087,
    # α_i = 1.0) stay clear of the min-thyristor-angle clamp.
    verify_jacobian(sys; label = "polar 3-bus LCC", perturbation = 0.01)
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
