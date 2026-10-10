const _LEAN_KLU = SolutionParameters(; linear_solver = "KLU")
const _KW = PNM.KLUWrapper

function _lean_sys14_data(; T = 1, n_threads = 1, kwargs...)
    sys = PSB.build_system(PSB.PSITestSystems, "c_sys14"; add_forecasts = false)
    data = PowerFlowData(
        ACPowerFlow{NewtonRaphsonACPowerFlow}(;
            time_steps = T,
            solution_parameters = SolutionParameters(; linear_solver = "KLU", n_threads),
            kwargs...),
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
    data = _lean_sys14_data(; T = 24, n_threads = 4)
    @test solve_power_flow!(data)
    slot = _lean_slot(data)
    @test slot.tried && slot.valid
    h = _lean_plan_hash(slot.plan)
    @test solve_power_flow!(data)
    @test _lean_slot(data) === slot
    @test _lean_plan_hash(slot.plan) == h

    worker = PF._column_worker(data, 1:24, 1:6, PF.WorkerSlot())
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

    # After a re-pivot, refactors stay on KLU until `_resume_lean!`.
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
    align(entry.linSolveCache, entry, data)
    @test entry.linSolveCache.lean_plan !== plan
    @test entry.linSolveCache.lean_plan.p === plan.p
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

@testset "lean LU: a singular lean-plan Jacobian is warned every time" begin
    J = SparseArrays.sparse(
        PF.J_INDEX_TYPE[1, 2, 1, 2], PF.J_INDEX_TYPE[1, 1, 2, 2], [1.0, 0.0, 0.0, 0.0])
    for _ in 1:2
        slot = PF.LeanPlanSlot()
        warned = (:warn, r"Jacobian the lean plan is built on is singular")
        @test_logs warned PF._build_lean_plan!(slot, J, 1)
        @test !slot.valid
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

@testset "lean LU: a fresh solve plans on its own Jacobian and leaves it as it was" begin
    data = _lean_sys14_data()
    residual = PF.ACPowerFlowResidual(data, 1)
    J = PF.ACPowerFlowJacobian(data, residual, 1)
    J(data, 1)
    s = J.bus_state
    before = (copy(s.Vm), copy(s.θ), copy(s.phasor), copy(SparseArrays.nonzeros(J.Jv)))
    slot = PF.LeanPlanSlot()
    PF._plan_at_flat_in_place!(slot, J, data, 1)
    @test slot.tried && slot.valid
    @test (s.Vm, s.θ, s.phasor, SparseArrays.nonzeros(J.Jv)) == before
    @test _lean_plan_hash(slot.plan) == _lean_plan_hash(PF._lean_plan_slot!(data, 1).plan)

    planned = _lean_sys14_data()
    PF._lean_plan_slot!(planned, 1)
    @test solve_power_flow!(planned)
    fresh = _lean_sys14_data()
    @test solve_power_flow!(fresh)
    @test _lean_plan_hash(_lean_slot(fresh).plan) ==
          _lean_plan_hash(_lean_slot(planned).plan)
    @test fresh.bus_magnitude == planned.bus_magnitude
    @test fresh.bus_angles == planned.bus_angles
    @test PF._lean_counts(fresh) == PF._lean_counts(planned)
end

@testset "lean LU: an LCC system plans before the solve, serial and threaded alike" begin
    sys = PSB.build_system(PSB.PSITestSystems, "c_sys14_hvdc_lcc")
    function lcc_data(n_threads)
        pf = ACPowerFlow{NewtonRaphsonACPowerFlow}(;
            time_steps = 3, correct_bustypes = true,
            solution_parameters = SolutionParameters(; linear_solver = "KLU", n_threads))
        return PowerFlowData(pf, sys)
    end
    planned = lcc_data(1)
    PF._lean_plan_slot!(planned, 1)
    serial = lcc_data(1)
    threaded = lcc_data(2)
    for data in (planned, serial, threaded)
        @test solve_power_flow!(data)
    end
    for data in (serial, threaded)
        @test _lean_plan_hash(_lean_slot(data).plan) ==
              _lean_plan_hash(_lean_slot(planned).plan)
        @test data.bus_magnitude == planned.bus_magnitude
        @test data.bus_angles == planned.bus_angles
    end
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

_with_a_row(plan, a_row) =
    _KW.LeanLUPlan(plan.n, 1e-300, plan.p, plan.q, plan.cp, plan.dpos,
        plan.row, a_row, plan.dep_lb, plan.dep_le, plan.a_colptr, plan.a_rowval)

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
    return _with_a_row(plan, a_row)
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

    # A healthy lean solve never retries.
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

@testset "lean LU: a solve that pivoted fresh at its start is not rerun" begin
    capped = SolutionParameters(; linear_solver = "KLU", maxIterations = 1)
    bad = _lean_sys14_data(; solution_parameters = capped)
    slot = PF._lean_plan_slot!(bad, 1)
    p = slot.plan
    # An unreachable pivot ratio: the first lean refactor is rejected for a klu_factor at x0.
    slot.plan = _KW.LeanLUPlan(p.n, 1e300, p.p, p.q, p.cp, p.dpos, p.row, p.a_row,
        p.dep_lb, p.dep_le, p.a_colptr, p.a_rowval)
    @test !with_logger(() -> solve_power_flow!(bad), NullLogger())
    @test _lean_ar(bad) == (1, 1)
    @test iszero(PF._cold_retries(bad))
    @test PF.get_iterations(bad)[1] == 1
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
    return _with_a_row(plan, a_row)
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

@testset "lean LU: FD and GA handoffs run on the structure's plan" begin
    sys = PSB.build_system(PSB.PSITestSystems, "c_sys14"; add_forecasts = false)
    function staged(S, handoff = NewtonRaphsonACPowerFlow)
        params = SolutionParameters(;
            linear_solver = "KLU", handoff_solver = handoff, handoff_tol = 1e-2)
        return PowerFlowData(ACPowerFlow{S}(; solution_parameters = params), sys)
    end
    nr = _lean_sys14_data()
    @test solve_power_flow!(nr)

    fd = staged(PF.FastDecoupledXB)
    vm0, va0 = copy(fd.bus_magnitude), copy(fd.bus_angles)
    @test solve_power_flow!(fd)
    @test _lean_slot(fd).tried
    h = fd.solver_cache[].handoff[].cache
    @test h.lean_plan === _lean_slot(fd).plan
    @test isapprox(fd.bus_magnitude, nr.bus_magnitude; atol = 1e-9, rtol = 0)
    @test isapprox(fd.bus_angles, nr.bus_angles; atol = 1e-9, rtol = 0)
    (; attempts) = PF._lean_counts(h)
    copyto!(fd.bus_magnitude, vm0)
    copyto!(fd.bus_angles, va0)
    @test solve_power_flow!(fd)
    @test fd.solver_cache[].handoff[].cache === h
    @test PF._lean_counts(h).attempts > attempts
    @test PF._lean_counts(h).rejects == 0

    fd_pq = staged(PF.FastDecoupledXB)
    nr_pq = staged(NewtonRaphsonACPowerFlow)
    vm0, va0 = copy(fd_pq.bus_magnitude), copy(fd_pq.bus_angles)
    @test solve_power_flow!(fd_pq)
    @test solve_power_flow!(nr_pq)
    h_pq = fd_pq.solver_cache[].handoff[].cache
    counts = PF._lean_counts(h_pq)
    pv_ix = findfirst(==(PSY.ACBusTypes.PV), fd_pq.bus_type[:, 1])
    for data in (fd_pq, nr_pq)
        data.bus_type[pv_ix, 1] = PSY.ACBusTypes.PQ
        copyto!(data.bus_magnitude, vm0)
        copyto!(data.bus_angles, va0)
    end
    @test PNM.is_factored(h_pq)
    PF._handoff_linear_cache!(
        fd_pq.solver_cache[], PF.get_pf(fd_pq), fd_pq, nothing, 1, "KLU")
    @test !PNM.is_factored(h_pq)
    @test solve_power_flow!(fd_pq)
    @test solve_power_flow!(nr_pq)
    @test fd_pq.solver_cache[].handoff[].cache === h_pq
    counts_pq = PF._lean_counts(h_pq)
    @test counts_pq.attempts > counts.attempts
    @test counts_pq.rejects == counts.rejects
    @test isapprox(fd_pq.bus_magnitude, nr_pq.bus_magnitude; atol = 1e-9, rtol = 0)
    @test isapprox(fd_pq.bus_angles, nr_pq.bus_angles; atol = 1e-9, rtol = 0)

    ga = staged(GeneralizedAdmittanceACPowerFlow)
    @test solve_power_flow!(ga)
    @test _lean_slot(ga).tried
    @test isapprox(ga.bus_magnitude, nr.bus_magnitude; atol = 1e-9, rtol = 0)

    for (handoff, planned) in ((NewtonRaphsonACPowerFlow, true), (PF.NoHandoff, false))
        data = staged(PF.FastDecoupledXB, handoff)
        PF._prepare_lean_plan!(PF.get_pf(data), data, 1, PNM.KLUSolver())
        @test PF._lean_plan_tried(data.ac_jacobian_structure_cache[], data) == planned
    end
end

function _lean_rm_data(F; T = 1, n_threads = 1, params = (;), kwargs...)
    sys = PSB.build_system(PSB.PSITestSystems, "c_sys14"; add_forecasts = false)
    data = PowerFlowData(
        F{NewtonRaphsonACPowerFlow}(;
            time_steps = T, correct_bustypes = true,
            solution_parameters = SolutionParameters(;
                linear_solver = "KLU", n_threads, params...),
            kwargs...),
        sys,
    )
    n = size(data.bus_type, 1)
    profile = [1 + 0.02 * cos(0.7 * i + 1.3 * t) for i in 1:n, t in 1:T]
    data.bus_active_power_withdrawals .*= profile
    data.bus_reactive_power_withdrawals .*= profile
    return data
end

@testset "lean LU: rectangular and mixed NR on KLU match plain KLU" begin
    for F in (ACRectangularPowerFlow, ACMixedPowerFlow)
        lean = _lean_rm_data(F; T = 24)
        @test solve_power_flow!(lean)
        klu = _lean_rm_data(F; T = 24)
        @test _without_lean(() -> solve_power_flow!(klu))
        cache = lean.solver_cache[]
        @test _KW.has_lean_plan(cache.linSolveCache)
        @test cache.linSolveCache.lean_plan === cache.lean.plan
        @test !_KW.has_lean_plan(klu.solver_cache[].linSolveCache)
        (; attempts, rejects, solve_failures, cold_retries) = PF._lean_counts(lean)
        @test attempts > 0
        @test iszero(rejects) && iszero(solve_failures) && iszero(cold_retries)
        @test PF._lean_counts(klu) == PF._NO_LEAN_COUNTS
        @test lean.converged == klu.converged
        @test lean.bus_type == klu.bus_type
        @test isapprox(lean.bus_magnitude, klu.bus_magnitude; atol = 1e-10)
        @test isapprox(lean.bus_angles, klu.bus_angles; atol = 1e-10)

        lean = _lean_rm_data(F; T = 24, check_reactive_power_limits = true)
        klu = _lean_rm_data(F; T = 24, check_reactive_power_limits = true)
        @test solve_power_flow!(lean)
        @test _without_lean(() -> solve_power_flow!(klu))
        @test lean.converged == klu.converged
        @test lean.bus_type == klu.bus_type
        @test isapprox(lean.bus_magnitude, klu.bus_magnitude; atol = 1e-10)
        @test isapprox(lean.bus_angles, klu.bus_angles; atol = 1e-10)
    end
end

@testset "lean LU: rectangular and mixed pause the plan off its bus types" begin
    # A PV→PQ flip keeps both patterns: the cache, and its plan, are reused.
    for F in (ACRectangularPowerFlow, ACMixedPowerFlow)
        data = _lean_rm_data(F)
        @test solve_power_flow!(data)
        cache = data.solver_cache[]
        k = findfirst(==(PSY.ACBusTypes.PV), data.bus_type[:, 1])
        data.bus_type[k, 1] = PSY.ACBusTypes.PQ
        data.bus_active_power_withdrawals .*= 1.01
        (; attempts) = PF._lean_counts(cache)
        @test solve_power_flow!(data)
        @test data.solver_cache[] === cache
        @test PF._lean_counts(cache).attempts == attempts
        @test _KW.has_lean_plan(cache.linSolveCache)
        @test cache.lean.tried
        data.bus_type[k, 1] = PSY.ACBusTypes.PV
        data.bus_active_power_withdrawals .*= 1.01
        @test solve_power_flow!(data)
        @test data.solver_cache[] === cache
        @test PF._lean_counts(cache).attempts > attempts
        @test iszero(PF._lean_counts(cache).rejects)
    end
end

@testset "lean LU: a rejected rectangular plan is retired" begin
    data = _lean_rm_data(ACRectangularPowerFlow; T = 2)
    @test solve_power_flow!(data)
    cache = data.solver_cache[]
    @test cache.lean.valid
    lin = cache.linSolveCache
    R = PF.ACRectangularCIResidual(data, 1)
    x = Vector{Float64}(undef, length(R.Rv))
    PF.rect_initial_state!(x, data, R.bus_state_offset, R.bus_block_size, 1)
    R(data, x, 1)
    bad = copy(PF.ACRectangularCIJacobian(data, R, 1).Jv)
    # Shrink the plan's first pivot far below its reject ratio.
    bad[lin.lean_plan.p[1], lin.lean_plan.q[1]] *= 1e-14
    PF._resume_lean!(lin)
    PF.numeric_refactor!(lin, bad)
    @test PF._lean_counts(cache).rejects == 1
    data.bus_active_power_withdrawals .*= 1.01
    @test solve_power_flow!(data)
    @test !data.solver_cache[].lean.valid
    klu = _lean_rm_data(ACRectangularPowerFlow; T = 2)
    @test _without_lean(() -> solve_power_flow!(klu))
    klu.bus_active_power_withdrawals .*= 1.01
    @test _without_lean(() -> solve_power_flow!(klu))
    @test isapprox(data.bus_magnitude, klu.bus_magnitude; atol = 1e-10)
    @test isapprox(data.bus_angles, klu.bus_angles; atol = 1e-10)
end

@testset "lean LU: a failed solve on a paused plan is not rerun" begin
    data = _lean_rm_data(ACMixedPowerFlow; params = (; maxIterations = 1))
    vm0, va0 = copy(data.bus_magnitude), copy(data.bus_angles)
    with_logger(() -> solve_power_flow!(data), NullLogger())
    cache = data.solver_cache[]
    @test cache.lean.valid
    k = findfirst(==(PSY.ACBusTypes.PV), data.bus_type[:, 1])
    data.bus_type[k, 1] = PSY.ACBusTypes.PQ
    copyto!(data.bus_magnitude, vm0)
    copyto!(data.bus_angles, va0)
    retries = PF._cold_retries(data)
    @test !with_logger(() -> solve_power_flow!(data), NullLogger())
    @test data.solver_cache[] === cache
    @test PF._cold_retries(data) == retries
    @test PF.get_iterations(data)[1] == 1
end

@testset "lean LU: rectangular and mixed workers and copies share the seed's plan" begin
    for F in (ACRectangularPowerFlow, ACMixedPowerFlow)
        threaded = _lean_rm_data(F; T = 24, n_threads = 2)
        @test solve_power_flow!(threaded)
        a, b = (s.solver_cache[] for s in threaded.worker_slots)
        @test a.lean.valid && b.lean === a.lean
        @test b.linSolveCache.lean_plan === a.lean.plan
        serial = _lean_rm_data(F; T = 24)
        @test solve_power_flow!(serial)
        @test threaded.converged == serial.converged
        @test isapprox(threaded.bus_magnitude, serial.bus_magnitude; atol = 1e-10)
        @test isapprox(threaded.bus_angles, serial.bus_angles; atol = 1e-10)

        work = _lean_rm_data(F; T = 24)
        PF._inherit_jacobian_structure!(work, serial)
        @test work.solver_cache[].lean === serial.solver_cache[].lean
        work.bus_active_power_withdrawals .*= 1.01
        @test solve_power_flow!(work)
        @test work.solver_cache[].linSolveCache.lean_plan ===
              serial.solver_cache[].lean.plan
        @test PF._lean_counts(work).attempts > 0
    end
end
