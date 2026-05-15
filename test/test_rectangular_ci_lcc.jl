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

function _rect_lcc_verify(sys::System; label::String, perturbation::Float64 = 0.02)
    pf_r = ACPowerFlow{RectangularCurrentInjectionACPowerFlow}(;
        correct_bustypes = true, solver_settings = _rect_lcc_settings())
    data = PF.PowerFlowData(pf_r, sys)
    R = PF.ACRectangularCIResidual(data, 1)
    J = PF.ACRectangularCIJacobian(R, 1)
    x = Vector{Float64}(undef, length(R.Rv))
    PF.rect_initial_state!(x, data, R.bus_state_offset, R.bus_block_size, 1)
    if perturbation > 0
        Random.seed!(42)
        x .+= perturbation .* randn(length(x))
    end
    R(x, 1)
    J(1)
    verify_jacobian_asymptotic(R, copy(J.Jv), x, 1; label = label)
end

@testset "Rectangular CI LCC: asymptotic verification, nonzero xc (interior)" begin
    # Simple 3-bus LCC system with x_t > 0 and ϕ_i kept off the clamp by a
    # moderate extinction angle. This is the regime where rect's old
    # α-approximation Jacobian disagreed with the true-ϕ residual.
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
    PSY.set_inverter_extinction_angle!(lcc, 1.0)   # well clear of ϕ-clamp
    _rect_lcc_verify(sys; label = "rect CI LCC nonzero-xc interior")
end

@testset "Rectangular CI LCC: asymptotic verification at inverter ϕ clamp" begin
    # Same fixture as the polar inverter-ϕ-clamp test: large x_t_i + small
    # extinction angle pushes raw_i < -1 and clamps ϕ_i at π. Exercises the
    # sin(ϕ) → 0 boundary guards in the ∂ϕ helpers.
    sys = System(100.0)
    b1 = _add_simple_bus!(sys, 1, ACBusTypes.REF, 230, 1.1, 0.0)
    b2 = _add_simple_bus!(sys, 2, ACBusTypes.PQ, 230, 1.1, 0.0)
    b3 = _add_simple_bus!(sys, 3, ACBusTypes.PQ, 230, 1.1, 0.0)
    _add_simple_load!(sys, b2, 10, 5)
    _add_simple_load!(sys, b3, 60, 20)
    _add_simple_line!(sys, b1, b2, 5e-3, 5e-3, 1e-3)
    _add_simple_line!(sys, b1, b3, 5e-3, 5e-3, 1e-3)
    _add_simple_source!(sys, b1, 0.0, 0.0)
    lcc = _add_simple_lcc!(sys, b2, b3, 0.05, 0.05, 0.20)
    PSY.set_inverter_extinction_angle!(lcc, 0.1)
    PSY.set_rectifier_delay_angle!(lcc, 0.1)
    _rect_lcc_verify(sys; label = "rect CI LCC inverter ϕ-clamp",
        perturbation = 0.01)
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
