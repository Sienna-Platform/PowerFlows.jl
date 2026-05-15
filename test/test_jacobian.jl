function verify_jacobian(
    sys::PSY.System;
    pf::PF.ACPowerFlow = PF.ACPowerFlow{NewtonRaphsonACPowerFlow}(;
        correct_bustypes = true,
    ),
)
    data = PF.PowerFlowData(pf, sys)
    time_step = 1
    residual = PF.ACPowerFlowResidual(data, time_step)
    J = PF.ACPowerFlowJacobian(residual, time_step)
    n_lccs =
        length(collect(PSY.get_components(PSY.get_available, PSY.TwoTerminalLCCLine, sys)))
    n = 2 * length(collect(get_components(ACBus, sys))) + 4 * n_lccs
    x0 = PF.calculate_x0(data, time_step)
    residual(x0, time_step)
    J(time_step)
    J_x0 = deepcopy(J.Jv)
    Δx_start, Δx_stop = 1e-3, 1e-6
    for j in 1:n
        u = zeros(n)
        u[j] = 1
        Δx_mag = Δx_start
        close_enough = false
        while !close_enough && Δx_mag >= Δx_stop
            inputValues = [x0 - Δx_mag * u, x0 + Δx_mag * u]
            outputValues = Vector{Vector{Float64}}()
            for inputVal in inputValues
                residual(inputVal, time_step)
                push!(outputValues, deepcopy(residual.Rv))
            end
            ΔF = (outputValues[2] .- outputValues[1]) ./ 2
            floating_point_issues = all(isapprox.(ΔF, 0.0; atol = eps(Float32)))
            if floating_point_issues
                break
            end
            ∂F_∂u_numerical = ΔF ./ Δx_mag
            ∂F_∂u_symbolic = J_x0 * u

            if !isapprox(∂F_∂u_numerical, ∂F_∂u_symbolic; rtol = 1e-6, atol = eps(Float32))
                Δx_mag /= 10.0
            else
                close_enough = true
            end
        end
        @test close_enough
    end
end

@testset "Jacobian verification" begin
    sys = PSB.build_system(PSITestSystems, "c_sys14")
    verify_jacobian(sys)
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
    PSY.set_inverter_extinction_angle!(lcc, 1.0)
    verify_jacobian(sys)
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
    verify_jacobian(sys)
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
    verify_jacobian(sys; pf = pf)
end
