function _rect_lcc_settings()
    return Dict{Symbol, Any}(:validate_voltage_magnitudes => false)
end

@testset "Rectangular CI LCC: residual zero at polar-converged state" begin
    raw_path = joinpath(TEST_DATA_DIR, "case5_2_lcc.raw")
    sys = System(raw_path)
    pf_p = ACPowerFlow{NewtonRaphsonACPowerFlow}()
    @test PF.solve_and_store_power_flow!(pf_p, sys)
    pf_r = ACPowerFlow{RectangularCurrentInjectionACPowerFlow}(;
        solver_settings = _rect_lcc_settings())
    data = PF.PowerFlowData(pf_r, sys)
    R = PF.ACRectangularCIResidual(data, 1)
    x = Vector{Float64}(undef, length(R.Rv))
    PF.rect_initial_state!(x, data, R.bus_state_offset, R.bus_block_size, 1)
    R(x, 1)
    @test LinearAlgebra.norm(R.Rv, Inf) < 1e-7
end

@testset "Rectangular CI LCC: asymptotic Jacobian verification on case5_2_lcc" begin
    raw_path = joinpath(TEST_DATA_DIR, "case5_2_lcc.raw")
    sys = System(raw_path)
    pf_p = ACPowerFlow{NewtonRaphsonACPowerFlow}()
    PF.solve_and_store_power_flow!(pf_p, sys)
    pf_r = ACPowerFlow{RectangularCurrentInjectionACPowerFlow}(;
        solver_settings = _rect_lcc_settings())
    data = PF.PowerFlowData(pf_r, sys)
    R = PF.ACRectangularCIResidual(data, 1)
    J = PF.ACRectangularCIJacobian(R, 1)
    x = Vector{Float64}(undef, length(R.Rv))
    PF.rect_initial_state!(x, data, R.bus_state_offset, R.bus_block_size, 1)
    # Verify away from the converged state. NB: case5_2_lcc has x_t = 0 for
    # both converter sides, which forces ϕ ≡ ±α — so the α-approximation in
    # rect's Jacobian coincides with the true-ϕ residual math regardless of
    # perturbation. A future test on an LCC system with x_t > 0 would
    # exercise the α-vs-true-ϕ divergence properly.
    Random.seed!(42)
    x .+= 0.02 .* randn(length(x))
    R(x, 1)
    J(1)
    verify_jacobian_asymptotic(R, copy(J.Jv), x, 1; label = "rect CI LCC case5_2")
end

@testset "Rectangular CI LCC: solve parity with polar" begin
    raw_path = joinpath(TEST_DATA_DIR, "case5_2_lcc.raw")
    sys_p = System(raw_path)
    sys_r = System(raw_path)
    pf_p = ACPowerFlow{NewtonRaphsonACPowerFlow}()
    pf_r = ACPowerFlow{RectangularCurrentInjectionACPowerFlow}(;
        solver_settings = _rect_lcc_settings())
    res_p = solve_power_flow(pf_p, sys_p)
    res_r = solve_power_flow(pf_r, sys_r)
    @test res_p !== missing
    @test res_r !== missing
    @test maximum(abs.(res_p["bus_results"].Vm - res_r["bus_results"].Vm)) < 1e-7
    @test maximum(abs.(res_p["bus_results"].θ - res_r["bus_results"].θ)) < 1e-7
end

@testset "Rectangular CI LCC: step strategy variants" begin
    raw_path = joinpath(TEST_DATA_DIR, "case5_2_lcc.raw")
    pf_p = ACPowerFlow{NewtonRaphsonACPowerFlow}()
    res_p = solve_power_flow(pf_p, System(raw_path))
    for (label, extra_settings) in [
        ("plain NR", Dict{Symbol, Any}()),
        ("NR + Iwamoto", Dict{Symbol, Any}(:iwamoto => true)),
        ("Trust Region", Dict{Symbol, Any}(:step_strategy => :trust_region)),
        ("TR + Iwamoto FB",
            Dict{Symbol, Any}(
                :step_strategy => :trust_region,
                :iwamoto_fallback => true,
            )),
    ]
        @testset "$label" begin
            settings = merge(extra_settings, _rect_lcc_settings())
            pf_r = ACPowerFlow{RectangularCurrentInjectionACPowerFlow}(;
                solver_settings = settings)
            res_r = solve_power_flow(pf_r, System(raw_path))
            @test res_r !== missing
            @test maximum(abs.(res_p["bus_results"].Vm - res_r["bus_results"].Vm)) < 1e-7
            @test maximum(abs.(res_p["bus_results"].θ - res_r["bus_results"].θ)) < 1e-7
        end
    end
end
