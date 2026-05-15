function _rect_lcc_settings()
    return Dict{Symbol, Any}(:validate_voltage_magnitudes => false)
end

@testset "Rectangular CI LCC: residual zero at polar-converged state" begin
    raw_path = joinpath(TEST_DATA_DIR, "case5_2_lcc.raw")
    sys = System(raw_path)
    pf_p = ACPowerFlow{NewtonRaphsonACPowerFlow}()
    @test PF.solve_and_store_power_flow!(pf_p, sys)
    pf_r = ACRectangularPowerFlow{NewtonRaphsonACPowerFlow}(;
        solver_settings = _rect_lcc_settings())
    data = PF.PowerFlowData(pf_r, sys)
    R = PF.ACRectangularCIResidual(data, 1)
    x = Vector{Float64}(undef, length(R.Rv))
    PF.rect_initial_state!(x, data, R.bus_state_offset, R.bus_block_size, 1)
    R(x, 1)
    @test LinearAlgebra.norm(R.Rv, Inf) < 1e-7
end

@testset "Rectangular CI LCC: FD parity on case5_2_lcc" begin
    raw_path = joinpath(TEST_DATA_DIR, "case5_2_lcc.raw")
    sys = System(raw_path)
    pf_p = ACPowerFlow{NewtonRaphsonACPowerFlow}()
    PF.solve_and_store_power_flow!(pf_p, sys)
    pf_r = ACRectangularPowerFlow{NewtonRaphsonACPowerFlow}(;
        solver_settings = _rect_lcc_settings())
    data = PF.PowerFlowData(pf_r, sys)
    R = PF.ACRectangularCIResidual(data, 1)
    J = PF.ACRectangularCIJacobian(R, 1)
    x = Vector{Float64}(undef, length(R.Rv))
    PF.rect_initial_state!(x, data, R.bus_state_offset, R.bus_block_size, 1)
    R(x, 1)
    J(1)
    ε = 1e-6
    n = length(R.Rv)
    Jfd = zeros(n, n)
    for k in 1:n
        xp = copy(x)
        xp[k] += ε
        R(xp, 1)
        Jfd[:, k] = copy(R.Rv) / (2ε)
        xm = copy(x)
        xm[k] -= ε
        R(xm, 1)
        Jfd[:, k] -= copy(R.Rv) / (2ε)
    end
    R(x, 1)
    J(1)
    @test maximum(abs.(Array(J.Jv) - Jfd)) < 1e-4
end

@testset "Rectangular CI LCC: solve parity with polar" begin
    raw_path = joinpath(TEST_DATA_DIR, "case5_2_lcc.raw")
    sys_p = System(raw_path)
    sys_r = System(raw_path)
    pf_p = ACPowerFlow{NewtonRaphsonACPowerFlow}()
    pf_r = ACRectangularPowerFlow{NewtonRaphsonACPowerFlow}(;
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
    for (label, solver, extra_settings) in [
        ("plain NR", NewtonRaphsonACPowerFlow, Dict{Symbol, Any}()),
        ("NR + Iwamoto", NewtonRaphsonACPowerFlow,
            Dict{Symbol, Any}(:iwamoto => true)),
        ("Trust Region", TrustRegionACPowerFlow, Dict{Symbol, Any}()),
        ("TR + Iwamoto FB", TrustRegionACPowerFlow,
            Dict{Symbol, Any}(:iwamoto_fallback => true)),
    ]
        @testset "$label" begin
            settings = merge(extra_settings, _rect_lcc_settings())
            pf_r = ACRectangularPowerFlow{solver}(;
                solver_settings = settings)
            res_r = solve_power_flow(pf_r, System(raw_path))
            @test res_r !== missing
            @test maximum(abs.(res_p["bus_results"].Vm - res_r["bus_results"].Vm)) < 1e-7
            @test maximum(abs.(res_p["bus_results"].θ - res_r["bus_results"].θ)) < 1e-7
        end
    end
end
