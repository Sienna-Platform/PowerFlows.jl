# Post-solve VSC capability-limit warning: limits are lowered but not enforced, so a converged
# solve outside capability must emit one consolidated `@warn` per time step.

@testset "VSC limit warning: Q setpoint outside reactive limits (NR and TR)" begin
    for solver in (NewtonRaphsonACPowerFlow, TrustRegionACPowerFlow)
        sys = _build_vsc_pq_system(;
            name = "vsc_limits",
            q_set = 0.4,
            reactive_power_limits_to = (min = -0.1, max = 0.1),
        )
        data = PowerFlowData(
            ACPowerFlow{solver}(; solver_settings = VSC_SETTINGS),
            sys,
        )
        converged = @test_logs (:warn, r"Q = .* outside") match_mode = :any begin
            solve_power_flow!(data)
        end
        @test converged
        # the ControlPQ converter pins its (violating) setpoint
        dcn = PF.get_dc_network(data)
        @test isapprox(dcn.q_c[2, 1], 0.4; atol = 1e-7)
    end
end

@testset "VSC limit warning: apparent power exceeds s_max" begin
    sys = _build_vsc_pq_system(;
        name = "vsc_limits",
        p_set = 0.4,
        q_set = 0.1,
        rating_to = 0.3,
    )
    data = PowerFlowData(
        ACPowerFlow{NewtonRaphsonACPowerFlow}(; solver_settings = VSC_SETTINGS),
        sys,
    )
    converged = @test_logs (:warn, r"S = .* exceeds s_max") match_mode = :any begin
        solve_power_flow!(data)
    end
    @test converged
end

@testset "VSC limit warning: P setpoint outside active limits" begin
    sys = _build_vsc_pq_system(;
        name = "vsc_limits",
        p_set = 0.4,
        active_power_limits_to = (min = -0.2, max = 0.2),
    )
    data = PowerFlowData(
        ACPowerFlow{NewtonRaphsonACPowerFlow}(; solver_settings = VSC_SETTINGS),
        sys,
    )
    converged = @test_logs (:warn, r"P = .* outside") match_mode = :any begin
        solve_power_flow!(data)
    end
    @test converged
end

@testset "VSC limit warning: silent when the solution is within capability" begin
    sys = _build_vsc_pq_system(;
        name = "vsc_limits",
        p_set = 0.4,
        q_set = 0.1,
        rating_to = 2.0,
        reactive_power_limits_to = (min = -0.5, max = 0.5),
    )
    data = PowerFlowData(
        ACPowerFlow{NewtonRaphsonACPowerFlow}(; solver_settings = VSC_SETTINGS),
        sys,
    )
    # @test_logs with no patterns: asserts no Warn-or-above output
    converged = @test_logs min_level = Logging.Warn solve_power_flow!(data)
    @test converged
end
