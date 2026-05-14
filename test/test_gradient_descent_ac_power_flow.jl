@testset "GradientDescentACPowerFlow" begin
    @testset "Type construction" begin
        pf = ACPowerFlow{GradientDescentACPowerFlow}()
        @test pf isa ACPowerFlow{GradientDescentACPowerFlow}
    end

    @testset "AdamConfig hyperparameter passthrough" begin
        settings = Dict{Symbol, Any}(
            :learning_rate => 0.05,
            :beta1 => 0.85,
            :beta2 => 0.99,
            :epsilon => 1e-7,
        )
        cfg = PF.AdamConfig(settings)
        @test cfg.learning_rate == 0.05
        @test cfg.beta1 == 0.85
        @test cfg.beta2 == 0.99
        @test cfg.epsilon == 1e-7
    end

    @testset "AdamConfig defaults" begin
        cfg = PF.AdamConfig()
        @test cfg.learning_rate == 0.01
        @test cfg.beta1 == 0.9
        @test cfg.beta2 == 0.999
        @test cfg.epsilon == 1e-8
    end

    @testset "Basic convergence on 14-bus" begin
        sys = PSB.build_system(PSB.PSITestSystems, "c_sys14"; add_forecasts = false)
        # Solve with Newton-Raphson as reference
        pf_nr = ACPowerFlow{NewtonRaphsonACPowerFlow}()
        result_nr = solve_power_flow(pf_nr, sys)

        # Rebuild system (solve_power_flow may modify sys)
        sys2 = PSB.build_system(PSB.PSITestSystems, "c_sys14"; add_forecasts = false)

        # Solve with Gradient Descent (Adam)
        pf_gd = ACPowerFlow{GradientDescentACPowerFlow}(;
            solver_settings = Dict{Symbol, Any}(:learning_rate => 0.01),
        )
        result_gd = solve_power_flow(pf_gd, sys2; maxIterations = 15000)
        @test !ismissing(result_gd)

        # Compare bus voltage results
        nr_bus = result_nr["bus_results"]
        gd_bus = result_gd["bus_results"]
        @test maximum(abs.(nr_bus.Vm .- gd_bus.Vm)) < 1e-6
        @test maximum(abs.(nr_bus.θ .- gd_bus.θ)) < 1e-6
    end

    @testset "Allocation-free gradient computation" begin
        sys = PSB.build_system(PSB.PSITestSystems, "c_sys14"; add_forecasts = false)
        pf = ACPowerFlow{GradientDescentACPowerFlow}()
        data = PowerFlowData(pf, sys)
        time_step = 1
        residual = PF.ACPowerFlowResidual(data, time_step)
        x0 = PF.calculate_x0(data, time_step)
        residual(x0, time_step)
        J = PF.ACPowerFlowJacobian(residual, time_step)
        J(time_step)
        state = PF.AdamState(length(x0))

        # Warm up
        PF.compute_gradient!(state, J, residual)

        # Test allocation
        allocs = @allocated PF.compute_gradient!(state, J, residual)
        @test iszero(allocs)
    end

    @testset "Allocation-free Adam step" begin
        n = 100
        state = PF.AdamState(n)
        rand_g = randn(n)
        state.g .= rand_g
        cfg = PF.AdamConfig()
        x = randn(n)

        # Warm up
        PF.adam_step!(x, state, cfg)

        # Test allocation
        allocs = @allocated PF.adam_step!(x, state, cfg)
        @test iszero(allocs)
    end
end
