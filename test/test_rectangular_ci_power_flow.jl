function _rect_pf_settings()
    return Dict{Symbol, Any}(:validate_voltage_magnitudes => false)
end

@testset "Rectangular CI Power Flow: convergence" begin
    @testset "c_sys5 converges" begin
        sys = PSB.build_system(PSB.PSITestSystems, "c_sys5")
        pf_rect = ACPowerFlow{RectangularCurrentInjectionACPowerFlow}(;
            solver_settings = _rect_pf_settings())
        @test PF.solve_and_store_power_flow!(pf_rect, sys)
    end

    @testset "c_sys14 converges" begin
        sys = PSB.build_system(PSB.PSITestSystems, "c_sys14"; add_forecasts = false)
        pf_rect = ACPowerFlow{RectangularCurrentInjectionACPowerFlow}(;
            solver_settings = _rect_pf_settings())
        @test PF.solve_and_store_power_flow!(pf_rect, sys)
    end
end

@testset "Rectangular CI Power Flow: parity with polar NR" begin
    fixtures = [
        ("c_sys5", false),
        ("c_sys14", false),
    ]
    for (name, with_forecasts) in fixtures
        @testset "$name" begin
            sys_p = if with_forecasts
                PSB.build_system(PSB.PSITestSystems, name)
            else
                PSB.build_system(PSB.PSITestSystems, name; add_forecasts = false)
            end
            sys_r = deepcopy(sys_p)
            pf_p = ACPowerFlow{NewtonRaphsonACPowerFlow}()
            pf_r = ACPowerFlow{RectangularCurrentInjectionACPowerFlow}(;
                solver_settings = _rect_pf_settings())
            res_p = solve_power_flow(pf_p, sys_p)
            res_r = solve_power_flow(pf_r, sys_r)
            @test res_p !== missing
            @test res_r !== missing
            @test maximum(abs.(res_p["bus_results"].Vm - res_r["bus_results"].Vm)) < 1e-7
            @test maximum(abs.(res_p["bus_results"].θ - res_r["bus_results"].θ)) < 1e-7
        end
    end
end

@testset "Rectangular CI Power Flow: defensive errors on unsupported config" begin
    sys = PSB.build_system(PSB.PSITestSystems, "c_sys5")
    base_settings = _rect_pf_settings()

    @testset "robust_power_flow=true rejected" begin
        pf = ACPowerFlow{RectangularCurrentInjectionACPowerFlow}(;
            robust_power_flow = true,
            solver_settings = base_settings)
        @test_throws ArgumentError solve_power_flow(pf, deepcopy(sys))
    end

    @testset "calculate_loss_factors=true rejected" begin
        pf = ACPowerFlow{RectangularCurrentInjectionACPowerFlow}(;
            calculate_loss_factors = true,
            solver_settings = base_settings)
        @test_throws ArgumentError solve_power_flow(pf, deepcopy(sys))
    end

    @testset "calculate_voltage_stability_factors=true rejected" begin
        pf = ACPowerFlow{RectangularCurrentInjectionACPowerFlow}(;
            calculate_voltage_stability_factors = true,
            solver_settings = base_settings)
        @test_throws ArgumentError solve_power_flow(pf, deepcopy(sys))
    end

    @testset "step_strategy=:levenberg_marquardt rejected with explanatory msg" begin
        settings = merge(base_settings,
            Dict{Symbol, Any}(:step_strategy => :levenberg_marquardt))
        pf = ACPowerFlow{RectangularCurrentInjectionACPowerFlow}(;
            solver_settings = settings)
        @test_throws ArgumentError solve_power_flow(pf, deepcopy(sys))
    end

    @testset "step_strategy=:robust_homotopy rejected" begin
        settings =
            merge(base_settings, Dict{Symbol, Any}(:step_strategy => :robust_homotopy))
        pf = ACPowerFlow{RectangularCurrentInjectionACPowerFlow}(;
            solver_settings = settings)
        @test_throws ArgumentError solve_power_flow(pf, deepcopy(sys))
    end

    @testset "step_strategy=:gradient_descent rejected" begin
        settings =
            merge(base_settings, Dict{Symbol, Any}(:step_strategy => :gradient_descent))
        pf = ACPowerFlow{RectangularCurrentInjectionACPowerFlow}(;
            solver_settings = settings)
        @test_throws ArgumentError solve_power_flow(pf, deepcopy(sys))
    end

    @testset "step_strategy=:typo rejected with helpful msg" begin
        settings = merge(base_settings,
            Dict{Symbol, Any}(:step_strategy => :trust_regioon))
        pf = ACPowerFlow{RectangularCurrentInjectionACPowerFlow}(;
            solver_settings = settings)
        @test_throws ArgumentError solve_power_flow(pf, deepcopy(sys))
    end
end

@testset "Rectangular CI Power Flow: step strategy variants" begin
    # Verify Iwamoto and Trust Region wrappers converge through the rectangular CI
    # residual/Jacobian. The drivers (_simple_step, _iwamoto_step, _trust_region_step)
    # are generic over the residual/Jacobian functor interface, so all four step
    # strategies should work without any rectangular-specific code in the drivers.
    fixtures = [("c_sys5", false), ("c_sys14", false)]
    strategies = [
        ("plain NR", Dict{Symbol, Any}()),
        ("NR + Iwamoto", Dict{Symbol, Any}(:iwamoto => true)),
        ("Trust Region", Dict{Symbol, Any}(:step_strategy => :trust_region)),
        ("TR + Iwamoto FB",
            Dict{Symbol, Any}(
                :step_strategy => :trust_region,
                :iwamoto_fallback => true,
            )),
    ]
    for (name, with_forecasts) in fixtures
        @testset "$name" begin
            sys_p = if with_forecasts
                PSB.build_system(PSB.PSITestSystems, name)
            else
                PSB.build_system(PSB.PSITestSystems, name; add_forecasts = false)
            end
            pf_p = ACPowerFlow{NewtonRaphsonACPowerFlow}()
            res_p = solve_power_flow(pf_p, deepcopy(sys_p))
            for (label, extra_settings) in strategies
                @testset "$label" begin
                    sys_r = deepcopy(sys_p)
                    settings = merge(extra_settings, _rect_pf_settings())
                    pf_r = ACPowerFlow{RectangularCurrentInjectionACPowerFlow}(;
                        solver_settings = settings)
                    res_r = solve_power_flow(pf_r, sys_r)
                    @test res_r !== missing
                    @test maximum(
                        abs.(res_p["bus_results"].Vm - res_r["bus_results"].Vm),
                    ) < 1e-7
                    @test maximum(
                        abs.(res_p["bus_results"].θ - res_r["bus_results"].θ),
                    ) < 1e-7
                end
            end
        end
    end
end
