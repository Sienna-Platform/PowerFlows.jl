# GA solver cache: build into the `solver_cache` slot, KLU factor/refactor of the complex
# Yℓℓ and Yqq blocks, PNM in-place solve, and reuse on a shunt-only (same-pattern) refresh.

@testset "GA: KLU factor, solve, reuse and refactor on c_sys14" begin
    sys = PSB.build_system(PSB.PSITestSystems, "c_sys14"; add_forecasts = false)
    data = PowerFlowData(ACPowerFlow{GeneralizedAdmittanceACPowerFlow}(), sys)
    part = PF.GAPartition(data, 1, Dict{Int, Float64}())
    np = PF.GANodalPower(data, part, PF.GAConverterTerms(size(data.bus_type, 1)), 1)
    cache = PF._get_or_build_ga_cache!(data, part)
    @test data.solver_cache[] === cache
    y = PF._ga_initial_shunts(cache.blocks, np, part, data, 1)
    rhs = ComplexF64.(randn(PF.n_l(part)) .+ im .* randn(PF.n_l(part)))
    for yk in (y, 1.1 .* y)
        PF._ga_factor!(cache, yk, part)
        x = copy(rhs)
        PNM.solve!(cache.Fl, x)
        @test Matrix(cache.blocks.Yll) * x ≈ rhs
        xq = rhs[1:PF.n_q(part)]
        PNM.solve!(cache.Fq, xq)
        @test Matrix(cache.blocks.Yqq) * xq ≈ rhs[1:PF.n_q(part)]
    end
    @test PF._get_or_build_ga_cache!(data, part) === cache
end

@testset "GA: singular factorization error names the bus" begin
    part = PF.GAPartition([1], [2], [3, 4], [2, 3, 4], [1.0], 1)
    @test_throws r"bus index 4" PF._ga_factor_error(
        LinearAlgebra.SingularException(3),
        part,
        "Yℓℓ",
    )
    @test_throws r"bus index 4" PF._ga_factor_error(
        LinearAlgebra.SingularException(2),
        part,
        "Yqq",
    )
    @test_throws ArgumentError PF._ga_factor_error(ArgumentError("x"), part, "Yℓℓ")
end
