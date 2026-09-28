# GA iteration kernel (spec §3.5): the sparse kernel must step identically to the dense
# literal reference (spec §3.5, eqs. 10–25) and be allocation-free after warm-up.

function _ga_kernel_setup(sys)
    data = PowerFlowData(ACPowerFlow{GeneralizedAdmittanceACPowerFlow}(), sys)
    part = PF.GAPartition(data, 1, Dict{Int, Float64}())
    np = PF.GANodalPower(data, part, PF.GAConverterTerms(size(data.bus_type, 1)), 1)
    cache = PF._get_or_build_ga_cache!(data, part)
    y = PF._ga_initial_shunts(cache.blocks, np, part, data, 1)
    PF._ga_factor!(cache, y, part)
    PF._ga_u0!(cache.ws, cache, PF._ga_slack_voltages(data, part, 1))
    fill!(cache.ws.i, 0.0im)
    return data, part, np, cache, y
end

function _ga_check_against_dense(sys, n_iter)
    data, part, np, cache, y = _ga_kernel_setup(sys)
    steps = ga_dense_reference(ga_dense_problem(data), y; maxiter = n_iter, tol = 0.0)
    for k in 1:n_iter
        gap = PF._ga_iterate!(cache.ws, cache, np, y, part.Vset, PF.n_v(part))
        @test maximum(abs.(cache.ws.u .- steps[k].u)) < 1e-9
        @test maximum(abs.(cache.ws.i .- steps[k].i_next)) < 1e-9
        @test isapprox(gap, steps[k].gap; rtol = 1e-7, atol = 1e-12)
    end
    return part, np, cache, y
end

@testset "GA: sparse iteration matches the dense reference and does not allocate" begin
    sys = PSB.build_system(PSB.PSITestSystems, "c_sys14"; add_forecasts = false)
    part, np, cache, y = _ga_check_against_dense(sys, 5)
    nv = PF.n_v(part)
    @test (@allocated PF._ga_iterate!(cache.ws, cache, np, y, part.Vset, nv)) == 0
end

@testset "GA: kernel with no PV buses" begin
    sys = PSB.build_system(PSB.PSITestSystems, "c_sys14"; add_forecasts = false)
    for b in
        PSY.get_components(b -> PSY.get_bustype(b) == PSY.ACBusTypes.PV, PSY.ACBus, sys)
        PSY.set_bustype!(b, PSY.ACBusTypes.PQ)
    end
    part, _, _, _ = _ga_check_against_dense(sys, 3)
    @test iszero(PF.n_v(part))
end
