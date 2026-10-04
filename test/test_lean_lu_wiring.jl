const _LEAN_KLU = SolutionParameters(; linear_solver = "KLU")
const _KW = PNM.KLUWrapper

function _lean_sys14_data(; T = 1, kwargs...)
    sys = PSB.build_system(PSB.PSITestSystems, "c_sys14"; add_forecasts = false)
    data = PowerFlowData(
        ACPowerFlow{NewtonRaphsonACPowerFlow}(;
            time_steps = T, solution_parameters = _LEAN_KLU, kwargs...),
        sys,
    )
    T > 1 && prepare_ts_data!(data, T)
    return data
end

function _without_lean(f)
    PF._USE_LEAN_LU[] = false
    try
        return f()
    finally
        PF._USE_LEAN_LU[] = true
    end
end

_lean_slot(data) = data.ac_jacobian_structure_cache[].lean
_lean_ar(data) = (PF._lean_counts(data).attempts, PF._lean_counts(data).rejects)

function _lean_plan_hash(plan)
    return hash((plan.p, plan.q, plan.cp, plan.dpos, plan.row, plan.a_row, plan.rcond0))
end

@testset "lean LU: polar NR on KLU matches plain KLU" begin
    for kw in (
        (; check_reactive_power_limits = true, correct_bustypes = true),
        (; generator_slack_participation_factors = nothing),
    )
        lean = _lean_sys14_data(; T = 24, kw...)
        @test solve_power_flow!(lean)
        klu = _lean_sys14_data(; T = 24, kw...)
        @test _without_lean(() -> solve_power_flow!(klu))

        cache = lean.polar_nr_cache[].linSolveCache
        @test _KW.has_lean_plan(cache)
        @test !_KW.has_lean_plan(klu.polar_nr_cache[].linSolveCache)
        (; attempts, rejects, solve_failures, late_analyses) = PF._lean_counts(lean)
        @test attempts > 0
        @test iszero(rejects) && iszero(solve_failures)
        # A solve that stays lean never runs the cache's own klu_analyze.
        @test iszero(late_analyses) && cache.symbolic == C_NULL
        @test PF._lean_counts(klu) == PF._NO_LEAN_COUNTS
        @test lean.converged == klu.converged
        @test lean.bus_type == klu.bus_type
        @test isapprox(lean.bus_magnitude, klu.bus_magnitude; atol = 1e-8)
        @test isapprox(lean.bus_angles, klu.bus_angles; atol = 1e-8)
    end
end

@testset "lean LU: one plan per Jacobian structure" begin
    data = _lean_sys14_data(; T = 4)
    types = data.bus_type[:, 1]
    @test solve_power_flow!(data)
    slot = _lean_slot(data)
    @test slot.tried && slot.valid
    @test slot.bus_types == types
    first_cache = data.polar_nr_cache[].linSolveCache
    @test first_cache.lean_plan === slot.plan

    # A rebuilt polar cache on the same structure shares the plan, never rebuilds it. The
    # load change keeps the warm start from converging without a factorization.
    data.polar_nr_cache[] = nothing
    data.bus_active_power_withdrawals .*= 1.01
    @test solve_power_flow!(data)
    rebuilt = data.polar_nr_cache[].linSolveCache
    @test rebuilt !== first_cache
    @test rebuilt.lean_plan === slot.plan
    @test _lean_slot(data) === slot
end

@testset "lean LU: threaded time steps share the parent's plan" begin
    data = _lean_sys14_data(; T = 24)
    @test solve_power_flow!(data; threads = 4)
    slot = _lean_slot(data)
    @test slot.tried && slot.valid
    h = _lean_plan_hash(slot.plan)
    @test solve_power_flow!(data; threads = 4)
    @test _lean_slot(data) === slot
    @test _lean_plan_hash(slot.plan) == h

    worker = PF._column_worker(data, 1:24, 1:6)
    @test worker.ac_jacobian_structure_cache[] === data.ac_jacobian_structure_cache[]
    @test isnothing(worker.polar_nr_cache[])
end

@testset "lean LU: Numeric drop, re-pivot and condest on a lean cache" begin
    data = _lean_sys14_data()
    @test solve_power_flow!(data)
    cache = data.polar_nr_cache[].linSolveCache
    Jv = data.polar_nr_cache[].J.Jv
    PF.numeric_refactor!(cache, Jv)
    @test cache.lean_active

    # A bus-type change frees the factors; the next refactor goes lean again.
    PF._drop_numeric!(cache)
    @test !cache.lean_active && !PNM.is_factored(cache)
    PF.numeric_refactor!(cache, Jv)
    @test cache.lean_active
    b = collect(1.0:size(Jv, 1))
    x_lean = PF.solve!(cache, copy(b))

    # The re-pivot guard's factorization is KLU's own, on the cache's first (deferred) analysis,
    # so tsolve!/condest! work after it.
    @test cache.symbolic == C_NULL
    before = PF._lean_counts(data)
    PF._repivot!(cache, Jv)
    @test !cache.lean_active && cache.numeric != C_NULL && cache.symbolic != C_NULL
    @test _KW.has_lean_plan(cache)
    after = PF._lean_counts(data)
    @test after.solve_failures == before.solve_failures + 1
    @test after.late_analyses == before.late_analyses + 1
    @test isapprox(PF.solve!(cache, copy(b)), x_lean; rtol = 1e-10)
    @test isapprox(PF.tsolve!(cache, copy(b)), transpose(Matrix(Jv)) \ b; rtol = 1e-8)

    # ... and the rest of that solve stays on it; the next solve goes lean again.
    PF.numeric_refactor!(cache, Jv)
    @test !cache.lean_active && cache.numeric != C_NULL
    PF._resume_lean!(cache)
    PF.numeric_refactor!(cache, Jv)
    @test cache.lean_active
    @test isfinite(PF._diag_condest(cache, Jv))
    PF.numeric_refactor!(cache, Jv)
    @test cache.lean_active
end

@testset "lean LU: a rejected refactor finishes the solve on KLU" begin
    data = _lean_sys14_data()
    @test solve_power_flow!(data)
    cache = data.polar_nr_cache[].linSolveCache
    Jv = copy(data.polar_nr_cache[].J.Jv)
    plan = cache.lean_plan
    # Shrink the plan's first pivot far below its reject ratio; KLU re-pivots around it.
    bad = copy(Jv)
    bad[plan.p[1], plan.q[1]] *= 1e-14
    (; attempts, rejects) = PF._lean_counts(data)
    PF._resume_lean!(cache)
    PF.numeric_refactor!(cache, bad)
    @test !cache.lean_active && cache.numeric != C_NULL
    @test _lean_ar(data) == (attempts + 1, rejects + 1)
    b = collect(1.0:size(Jv, 1))
    x = PF.solve!(cache, copy(b))
    @test norm(bad * x - b) <= 1e-8 * norm(b)
    # Paused: KLU refactors on the fallback's order, no further lean attempts.
    PF.numeric_refactor!(cache, Jv)
    @test !cache.lean_active
    @test _lean_ar(data) == (attempts + 1, rejects + 1)
    PF._resume_lean!(cache)
    PF.numeric_refactor!(cache, Jv)
    @test cache.lean_active
end

@testset "lean LU: a PV bus promoted to REF stays lean" begin
    function promote!(data, k)
        @test solve_power_flow!(data)
        data.bus_type[k, 1] = PSY.ACBusTypes.REF
        data.bus_active_power_withdrawals .*= 1.01
        @test solve_power_flow!(data)
        return data
    end
    data = _lean_sys14_data()
    k = findlast(==(PSY.ACBusTypes.PV), data.bus_type[:, 1])
    promote!(data, k)
    # Unaligned, the plan's pivot for the PV bus's Q column lands on an exact zero.
    @test PF._lean_counts(data).rejects == 0
    entry = data.polar_nr_cache[]
    plan = entry.lean.plan
    @test entry.linSolveCache.lean_plan !== plan
    @test entry.linSolveCache.lean_plan.p === plan.p
    klu = _without_lean(() -> promote!(_lean_sys14_data(), k))
    @test isapprox(data.bus_magnitude, klu.bus_magnitude; atol = 1e-12)
    @test isapprox(data.bus_angles, klu.bus_angles; atol = 1e-12)

    # Back on the plan's types, the plan's own order is restored without allocating.
    data.bus_type[k, 1] = PSY.ACBusTypes.PV
    align(c, e, d) = @allocated PF._align_lean_plan!(c, e.lean, view(d.bus_type, :, 1))
    align(entry.linSolveCache, entry, data)
    @test entry.linSolveCache.lean_plan === plan
    @test iszero(align(entry.linSolveCache, entry, data))
    data.bus_type[k, 1] = PSY.ACBusTypes.REF
    @test iszero(align(entry.linSolveCache, entry, data))
end

@testset "lean LU: never attached off KLU" begin
    if Sys.isapple()
        sys = PSB.build_system(PSB.PSITestSystems, "c_sys14"; add_forecasts = false)
        pf = ACPowerFlow{NewtonRaphsonACPowerFlow}(;
            solution_parameters = SolutionParameters(; linear_solver = "AppleAccelerateLU"))
        data = PowerFlowData(pf, sys)
        @test solve_power_flow!(data)
        @test !_lean_slot(data).tried
        @test PF._lean_counts(data) == PF._NO_LEAN_COUNTS
    end
    data = _lean_sys14_data()
    _without_lean(() -> solve_power_flow!(data))
    @test !_lean_slot(data).tried
end

@testset "lean LU: a same-pattern copy inherits the base's plan" begin
    base = _lean_sys14_data()
    @test solve_power_flow!(base)
    plan = _lean_slot(base).plan
    work = _lean_sys14_data()
    @test work.power_network_matrix !== base.power_network_matrix
    PF._inherit_jacobian_structure!(work, base)
    @test _lean_slot(work) === _lean_slot(base)
    work.bus_active_power_withdrawals .*= 1.01
    @test solve_power_flow!(work)
    @test work.polar_nr_cache[].linSolveCache.lean_plan === plan

    # An unbuilt plan is never shared, so copies cannot race to build it.
    fresh = _lean_sys14_data()
    _without_lean(() -> solve_power_flow!(fresh))
    other = _lean_sys14_data()
    PF._inherit_jacobian_structure!(other, fresh)
    @test _lean_slot(other) !== _lean_slot(fresh)
    @test !_lean_slot(other).tried

    sys5 = PSB.build_system(PSB.PSITestSystems, "c_sys5"; add_forecasts = false)
    c5 = PowerFlowData(
        ACPowerFlow{NewtonRaphsonACPowerFlow}(; solution_parameters = _LEAN_KLU), sys5)
    @test_throws ErrorException PF._inherit_jacobian_structure!(c5, base)
end

@testset "lean LU: a singular flat-start Jacobian is counted and warned every time" begin
    J = SparseArrays.sparse(
        PF.J_INDEX_TYPE[1, 2, 1, 2], PF.J_INDEX_TYPE[1, 1, 2, 2], [1.0, 0.0, 0.0, 0.0])
    for _ in 1:2
        slot = PF.LeanPlanSlot()
        n0 = PF._LEAN_SINGULAR_PLANS[]
        warned = (:warn, r"flat-start Jacobian is singular")
        @test_logs warned PF._build_lean_plan!(slot, J, 1)
        @test !slot.valid
        @test PF._LEAN_SINGULAR_PLANS[] == n0 + 1
    end
end

@testset "lean LU: a stored plan skips the residual and Jacobian builds" begin
    data = _lean_sys14_data()
    @test solve_power_flow!(data)
    memo = data.ac_jacobian_structure_cache[]
    @test PF._lean_plan_tried(memo, data)
    pf = PF.get_pf(data)
    PF._prepare_lean_plan!(pf, data, 1, PNM.KLUSolver())
    @test (@allocated PF._prepare_lean_plan!(pf, data, 1, PNM.KLUSolver())) == 0
    @test data.ac_jacobian_structure_cache[] === memo
end

@testset "lean LU: accepted lean factors skip residual refinement" begin
    # With a negative threshold every refinement check fails, re-pivoting the factorization.
    never = SolutionParameters(; linear_solver = "KLU", refinement_threshold = -1.0)
    lean = _lean_sys14_data(; solution_parameters = never)
    @test solve_power_flow!(lean)
    (; attempts, rejects, solve_failures) = PF._lean_counts(lean)
    @test attempts > 0 && iszero(rejects)
    @test iszero(solve_failures)
    klu = _lean_sys14_data(; solution_parameters = never)
    @test_logs (:warn, r"Jacobian is singular") match_mode = :any _without_lean(
        () -> solve_power_flow!(klu))
end

# The plan's scatter map with two off-diagonal entries of each column swapped: a plan for the
# wrong matrix whose pivots stay nonzero, accepted whatever their ratio.
function _wrong_matrix_plan(plan)
    a_row = copy(plan.a_row)
    for c in 1:Int(plan.n)
        k = findfirst(==(c), plan.q)
        off = [e for e in plan.a_colptr[c]:(plan.a_colptr[c + 1] - 1) if a_row[e] != k]
        if length(off) >= 2
            a_row[off[1]], a_row[off[2]] = a_row[off[2]], a_row[off[1]]
        end
    end
    return _KW.LeanLUPlan(plan.n, 1e-300, plan.p, plan.q, plan.cp, plan.dpos, plan.row,
        a_row, plan.dep_lb, plan.dep_le, plan.a_colptr, plan.a_rowval)
end

function _wrong_plan_data(; kwargs...)
    data = _lean_sys14_data(; kwargs...)
    slot = PF._lean_plan_slot!(data, 1)
    slot.plan = _wrong_matrix_plan(slot.plan)
    return data
end

@testset "lean LU: a solve failing on a reused pivot order is rerun cold" begin
    bad = _wrong_plan_data()
    @test solve_power_flow!(bad)
    @test PF._cold_retries(bad) == 1
    @test PF._lean_counts(bad).attempts > 0
    klu = _lean_sys14_data()
    @test _without_lean(() -> solve_power_flow!(klu))
    @test iszero(PF._cold_retries(klu))
    @test bad.bus_magnitude == klu.bus_magnitude
    @test bad.bus_angles == klu.bus_angles
    @test bad.bus_type == klu.bus_type

    # Each Newton solve of the Q-limit loop is retried on its own.
    q = (; check_reactive_power_limits = true, correct_bustypes = true)
    bad = _wrong_plan_data(; q...)
    @test solve_power_flow!(bad)
    @test PF._cold_retries(bad) >= 1
    klu = _lean_sys14_data(; q...)
    @test _without_lean(() -> solve_power_flow!(klu))
    @test bad.bus_magnitude == klu.bus_magnitude
    @test bad.bus_type == klu.bus_type

    # A healthy lean solve, and a fresh plain-KLU one, never retry.
    lean = _lean_sys14_data()
    @test solve_power_flow!(lean)
    @test iszero(PF._cold_retries(lean))

    # Retried once, then reported as failed when the cold solve fails too.
    capped = SolutionParameters(; linear_solver = "KLU", maxIterations = 2)
    failed = (:error, r"did not converge")
    bad = _wrong_plan_data(; solution_parameters = capped)
    @test_logs failed match_mode = :any @test !solve_power_flow!(bad)
    @test PF._cold_retries(bad) == 1
    klu = _lean_sys14_data(; solution_parameters = capped)
    @test_logs failed match_mode = :any @test !_without_lean(() -> solve_power_flow!(klu))
end

# The plan with 30% of its scatter map randomized: the lean solve diverges far enough that the
# telescoped ZIP loads lose their set points unless the retry restores them.
function _scrambled_plan(plan, rng)
    a_row = copy(plan.a_row)
    for e in eachindex(a_row)
        if rand(rng) < 0.3
            a_row[e] = rand(rng, 1:Int(plan.n))
        end
    end
    return _KW.LeanLUPlan(plan.n, 1e-300, plan.p, plan.q, plan.cp, plan.dpos, plan.row,
        a_row, plan.dep_lb, plan.dep_le, plan.a_colptr, plan.a_rowval)
end

function _zip_sys14_data(; kwargs...)
    data = _lean_sys14_data(; kwargs...)
    for ix in axes(data.bus_type, 1)
        if data.bus_type[ix, 1] == PSY.ACBusTypes.PQ
            data.bus_active_power_constant_impedance_withdrawals[ix, 1] = 0.2
            data.bus_reactive_power_constant_impedance_withdrawals[ix, 1] = 0.05
            data.bus_active_power_constant_current_withdrawals[ix, 1] = 0.1
        end
    end
    return data
end

@testset "lean LU: a cold retry restores the ZIP loads of the solve's start" begin
    sp = SolutionParameters(; linear_solver = "KLU", maxIterations = 30)
    klu = _zip_sys14_data(; solution_parameters = sp)
    @test _without_lean(() -> solve_power_flow!(klu))
    for seed in (5, 18)
        bad = _zip_sys14_data(; solution_parameters = sp)
        slot = PF._lean_plan_slot!(bad, 1)
        slot.plan = _scrambled_plan(slot.plan, MersenneTwister(seed))
        @test with_logger(() -> solve_power_flow!(bad), NullLogger())
        @test PF._cold_retries(bad) == 1
        @test bad.bus_magnitude == klu.bus_magnitude
        @test bad.bus_angles == klu.bus_angles
    end
end
