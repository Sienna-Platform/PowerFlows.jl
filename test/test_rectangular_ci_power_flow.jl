function _rect_pf_settings()
    return Dict{Symbol, Any}(:validate_voltage_magnitudes => false)
end

@testset "Rectangular CI Power Flow: convergence" begin
    @testset "c_sys5 converges" begin
        sys = PSB.build_system(PSB.PSITestSystems, "c_sys5")
        pf_rect = ACRectangularPowerFlow{NewtonRaphsonACPowerFlow}(;
            solver_settings = _rect_pf_settings())
        @test PF.solve_and_store_power_flow!(pf_rect, sys)
    end

    @testset "c_sys14 converges" begin
        sys = PSB.build_system(PSB.PSITestSystems, "c_sys14"; add_forecasts = false)
        pf_rect = ACRectangularPowerFlow{NewtonRaphsonACPowerFlow}(;
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
            pf_r = ACRectangularPowerFlow{NewtonRaphsonACPowerFlow}(;
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

@testset "Rectangular CI Power Flow: unsupported config rejected" begin
    # Removed fields: passing them is a constructor MethodError.
    @test_throws MethodError ACRectangularPowerFlow{NewtonRaphsonACPowerFlow}(;
        robust_power_flow = true)
    @test_throws MethodError ACRectangularPowerFlow{NewtonRaphsonACPowerFlow}(;
        calculate_loss_factors = true)
    @test_throws MethodError ACRectangularPowerFlow{NewtonRaphsonACPowerFlow}(;
        calculate_voltage_stability_factors = true)
    # Sanity: these kwargs ARE valid on the polar type — the MethodErrors above
    # prove the fields were removed from the rectangular type, not mistyped.
    @test ACPolarPowerFlow{NewtonRaphsonACPowerFlow}(;
        robust_power_flow = true) isa ACPolarPowerFlow
    @test ACPolarPowerFlow{NewtonRaphsonACPowerFlow}(;
        calculate_loss_factors = true) isa ACPolarPowerFlow
    @test ACPolarPowerFlow{NewtonRaphsonACPowerFlow}(;
        calculate_voltage_stability_factors = true) isa ACPolarPowerFlow
    # Polar-only solvers rejected at construction.
    @test_throws ArgumentError ACRectangularPowerFlow{LevenbergMarquardtACPowerFlow}()
    @test_throws ArgumentError ACRectangularPowerFlow{RobustHomotopyPowerFlow}()
    @test_throws ArgumentError ACRectangularPowerFlow{GradientDescentACPowerFlow}()
end

@testset "Rectangular CI Power Flow: step strategy variants" begin
    # Verify Iwamoto and Trust Region wrappers converge through the rectangular CI
    # residual/Jacobian. The drivers (_simple_step, _iwamoto_step, _trust_region_step)
    # are generic over the residual/Jacobian functor interface, so all four step
    # strategies should work without any rectangular-specific code in the drivers.
    fixtures = [("c_sys5", false), ("c_sys14", false)]
    strategies = [
        ("plain NR", NewtonRaphsonACPowerFlow, Dict{Symbol, Any}()),
        ("NR + Iwamoto", NewtonRaphsonACPowerFlow,
            Dict{Symbol, Any}(:iwamoto => true)),
        ("Trust Region", TrustRegionACPowerFlow, Dict{Symbol, Any}()),
        ("TR + Iwamoto FB", TrustRegionACPowerFlow,
            Dict{Symbol, Any}(:iwamoto_fallback => true)),
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
            for (label, solver, extra_settings) in strategies
                @testset "$label" begin
                    sys_r = deepcopy(sys_p)
                    settings = merge(extra_settings, _rect_pf_settings())
                    pf_r = ACRectangularPowerFlow{solver}(;
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
