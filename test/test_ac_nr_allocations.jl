@testset "AC NR allocation regression" begin
    # Use a system big enough to expose hot-path allocations but small enough
    # for CI. 2000-bus matches the system size already exercised by other tests.
    sys = PSB.build_system(PSB.MatpowerTestSystems, "matpower_ACTIVSg2000_sys")
    pf = ACPowerFlow{PF.NewtonRaphsonACPowerFlow}(; correct_bustypes = true)
    pf_data = PF.PowerFlowData(pf, sys)
    residual = PF.ACPowerFlowResidual(pf_data, 1)
    J = PF.ACPowerFlowJacobian(residual, 1)
    x0 = PF.calculate_x0(pf_data, 1)
    residual(x0, 1)    # warm
    J(1)               # warm

    # --- per-call upper bounds (chosen ~2x current best-case after fixes) ---
    # Baseline before fixes is ~140 KB on 2000-bus; target post-fix is < 2 KB.
    @test (@allocated residual(x0, 1)) < 2_000
    # Jacobian update is already lean; tight bound catches future regressions.
    @test (@allocated J(1)) < 200

    # --- _do_refinement! mul! path: A * Δx_nr should be zero-alloc when using mul! ---
    cache = PF.make_linear_solver_cache(PF.PNM.KLUSolver(), J.Jv)
    PF.full_factor!(cache, J.Jv)
    b = randn(size(J.Jv, 1))
    PF.solve!(cache, b)
    out = similar(b)
    LinearAlgebra.mul!(out, J.Jv, b)  # warm
    @test (@allocated LinearAlgebra.mul!(out, J.Jv, b)) == 0
end

@testset "Rectangular CI allocation regression" begin
    # Mirrors the polar NR test above. The rectangular CI residual/Jacobian
    # were authored with preallocated scratch arrays (P_eff_cache, e_state,
    # etc.) so the per-call cost is already ~0; these tests lock that in.
    sys = PSB.build_system(PSB.MatpowerTestSystems, "matpower_ACTIVSg2000_sys")
    pf = ACRectangularPowerFlow{NewtonRaphsonACPowerFlow}(;
        correct_bustypes = true,
        solver_settings = Dict{Symbol, Any}(:validate_voltage_magnitudes => false),
    )
    pf_data = PF.PowerFlowData(pf, sys)
    residual = PF.ACRectangularCIResidual(pf_data, 1)
    x0 = Vector{Float64}(undef, length(residual.Rv))
    PF.rect_initial_state!(
        x0, pf_data, residual.bus_state_offset, residual.bus_block_size, 1,
    )
    residual(x0, 1)  # warm
    J = PF.ACRectangularCIJacobian(residual, 1)
    J(1)  # warm

    # Baselines on 2000-bus: residual ~96 B, Jacobian ~144 B. Tight bounds
    # catch any future change that reintroduces per-iteration allocations.
    @test (@allocated residual(x0, 1)) < 500
    @test (@allocated J(1)) < 500

    # The linear-solve step also goes through the PNM linear-solver cache for the
    # rectangular Jacobian; mul! against the rect Jv must be zero-alloc.
    cache = PF.make_linear_solver_cache(PF.PNM.KLUSolver(), J.Jv)
    PF.full_factor!(cache, J.Jv)
    b = randn(size(J.Jv, 1))
    PF.solve!(cache, b)
    out = similar(b)
    LinearAlgebra.mul!(out, J.Jv, b)  # warm
    @test (@allocated LinearAlgebra.mul!(out, J.Jv, b)) == 0
end

@testset "AC Jacobian-structure cache regression" begin
    # The Jacobian sparse structure (~3.2 MB on 2000-bus) is memoized in
    # data.solver_cache and reused across repeated solves on the same data (and
    # across the Q-limit inner loop). A repeated solve must (a) allocate far less
    # than the ~4 MB it would cost rebuilding the structure, and (b) produce the
    # SAME result as the first solve (structure reuse must not change the answer).
    sys = PSB.build_system(PSB.MatpowerTestSystems, "matpower_ACTIVSg2000_sys")
    pf = ACPowerFlow{PF.NewtonRaphsonACPowerFlow}(; correct_bustypes = true)
    data = PF.PowerFlowData(pf, sys)
    PF.solve_power_flow!(data)               # warm: builds + caches the structure
    v1 = copy(data.bus_magnitude[:, 1])
    θ1 = copy(data.bus_angles[:, 1])
    a = @allocated PF.solve_power_flow!(data)   # reuse the cached structure
    @test a < 2_000_000                          # was ~4.2 MB before the cache
    @test data.bus_magnitude[:, 1] == v1         # structure reuse is answer-preserving
    @test data.bus_angles[:, 1] == θ1
end

@testset "DC PCM-reuse allocation regression" begin
    # A repeated DC solve on the same `data` (the PCM loop: fixed network, changing
    # injections) must reuse the cached factorization AND the typed `DCSolveScratch`
    # (incl. the concrete `valid_ix`) so the per-solve gather/scatter does not
    # re-materialize the `Not(ref)` InvertedIndex. Baseline before the fix was
    # ~35 KB/solve on 2000-bus (the InvertedIndex gather + scatter); post-fix the
    # reuse path is dominated by the function-barrier boundary, well under 5 KB.
    sys = PSB.build_system(PSB.MatpowerTestSystems, "matpower_ACTIVSg2000_sys")
    data = PF.PowerFlowData(DCPowerFlow(), sys)
    PF.solve_power_flow!(data)   # warm: builds + factors the cache and scratch
    PF.solve_power_flow!(data)   # 2nd: cache + scratch reused
    @test (@allocated PF.solve_power_flow!(data)) < 5_000
end
