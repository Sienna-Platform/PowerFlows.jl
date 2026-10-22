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

_lean_slot(data) = data.ac_jacobian_structure_cache[].lean

# Settle the plan on a converged solve, then put the state back so the next solve runs
# every Newton step on that plan.
function _settled_slot!(data)
    vm0, va0 = copy(data.bus_magnitude), copy(data.bus_angles)
    with_logger(() -> solve_power_flow!(data), NullLogger())
    copyto!(data.bus_magnitude, vm0)
    copyto!(data.bus_angles, va0)
    data.polar_nr_cache[] = nothing
    return _lean_slot(data)
end

# A twin with the same settings settles the plan, so `data` starts from its own untouched
# state on the plan it inherits.
function _on_settled_plan(make = _lean_sys14_data; kwargs...)
    twin = make(; (k => v for (k, v) in kwargs if k != :solution_parameters)...)
    slot = _settled_slot!(twin)
    data = make(; kwargs...)
    PF._inherit_jacobian_structure!(data, twin)
    return data, slot
end

# The first step settles the plan and runs on plain KLU. A second solve moves the polar cache
# onto the plan.
function _lean_cache_data(; kwargs...)
    data = _lean_sys14_data(; kwargs...)
    @test solve_power_flow!(data)
    data.bus_active_power_withdrawals .*= 1.01
    @test solve_power_flow!(data)
    return data
end
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
        # Only the first step, which runs before the plan exists, analyzes on plain KLU.
        @test iszero(late_analyses)
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

    worker = PF._column_worker(data, 1:24, PF.WorkerSlot())
    @test worker.ac_jacobian_structure_cache[].lean === slot
    @test isnothing(worker.polar_nr_cache[])

    # Every worker holds a memo keyed on its own data and the one shared slot.
    for s in data.worker_slots
        @test s.polar_nr_cache[].lean === slot
    end
    @test PF._slot_tried(data.ac_jacobian_structure_cache[])
end

@testset "lean LU: Numeric drop, re-pivot and condest on a lean cache" begin
    data = _lean_cache_data()
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

    # The re-pivot guard's factorization is KLU's own, on the cache's analysis, so
    # tsolve!/condest! work after it.
    before = PF._lean_counts(data)
    PF._repivot!(cache, Jv)
    @test !cache.lean_active && cache.numeric != C_NULL && cache.symbolic != C_NULL
    @test _KW.has_lean_plan(cache)
    after = PF._lean_counts(data)
    @test after.solve_failures == before.solve_failures + 1
    @test after.late_analyses == before.late_analyses
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
    data = _lean_cache_data()
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

@testset "lean LU: a settled slot skips the residual and Jacobian builds" begin
    data = _lean_sys14_data()
    @test solve_power_flow!(data)
    memo = data.ac_jacobian_structure_cache[]
    @test memo.lean.tried
    pf = PF.get_pf(data)
    types = data.bus_type[:, 1]
    PF._settle_lean_plan!(pf, PNM.KLUSolver(), data, 1, true, types)
    @test (@allocated PF._settle_lean_plan!(pf, PNM.KLUSolver(), data, 1, true, types)) == 0
    @test data.ac_jacobian_structure_cache[] === memo
end

@testset "lean LU: the plan is built once, after the first converged step" begin
    data = _lean_sys14_data(; T = 4)
    @test solve_power_flow!(data)
    slot = _lean_slot(data)
    @test slot.tried && slot.valid
    @test slot.bus_types == data.bus_type[:, 1]
    settled = slot.plan
    # Step 1 ran on plain KLU: its cache holds a KLU symbolic analysis and the plan.
    cache = data.polar_nr_cache[].linSolveCache
    @test cache.symbolic != C_NULL
    @test cache.lean_plan === settled
    data.bus_active_power_withdrawals .*= 1.01
    @test solve_power_flow!(data)
    @test slot.plan === settled
    @test PF._lean_counts(data).rejects == 0
    @test PF._lean_counts(data).attempts > 0

    # A step that did not converge leaves the slot untried.
    capped = SolutionParameters(; linear_solver = "KLU", maxIterations = 1)
    failed = _lean_sys14_data(; solution_parameters = capped)
    @test !with_logger(() -> solve_power_flow!(failed), NullLogger())
    @test !_lean_slot(failed).tried
end

@testset "lean LU: an LCC system settles the same plan, serial and threaded alike" begin
    sys = PSB.build_system(PSB.PSITestSystems, "c_sys14_hvdc_lcc")
    function lcc_data(n_threads)
        pf = ACPowerFlow{NewtonRaphsonACPowerFlow}(;
            time_steps = 3, correct_bustypes = true,
            solution_parameters = SolutionParameters(; linear_solver = "KLU", n_threads))
        return PowerFlowData(pf, sys)
    end
    serial = lcc_data(1)
    threaded = lcc_data(2)
    for data in (serial, threaded)
        @test solve_power_flow!(data)
    end
    @test _lean_plan_hash(_lean_slot(threaded).plan) ==
          _lean_plan_hash(_lean_slot(serial).plan)
    @test threaded.bus_magnitude == serial.bus_magnitude
    @test threaded.bus_angles == serial.bus_angles
end

@testset "lean LU: accepted lean factors skip residual refinement" begin
    # With a negative threshold every refinement check fails, re-pivoting the factorization.
    never = SolutionParameters(; linear_solver = "KLU", refinement_threshold = -1.0)
    lean, _ = _on_settled_plan(; solution_parameters = never)
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
    data, slot = _on_settled_plan(; kwargs...)
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
    bad, slot = _on_settled_plan(; solution_parameters = capped)
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
        bad, slot = _on_settled_plan(_zip_sys14_data; solution_parameters = sp)
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
    @test !_KW.has_lean_plan(h)
    @test isapprox(fd.bus_magnitude, nr.bus_magnitude; atol = 1e-9, rtol = 0)
    @test isapprox(fd.bus_angles, nr.bus_angles; atol = 1e-9, rtol = 0)
    (; attempts) = PF._lean_counts(h)
    copyto!(fd.bus_magnitude, vm0)
    copyto!(fd.bus_angles, va0)
    @test solve_power_flow!(fd)
    @test fd.solver_cache[].handoff[].cache === h
    # The re-solve moved the cache to the plan the first solve settled.
    @test h.lean_plan === _lean_slot(fd).plan
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
        solve_power_flow!(data)
        @test PF._lean_slot(data.ac_jacobian_structure_cache[]).tried == planned
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
    # A PV→PQ flip keeps the pattern in both formulations. The plan pauses while the
    # bus types differ from the plan's. It resumes when they match again.
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
        @test _KW.has_lean_plan(data.solver_cache[].linSolveCache)
        @test data.solver_cache[].lean.tried
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
    PF.rect_initial_state!(x, data, R.bus_state_offset, 1)
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
        @test count(s -> !isnothing(s.solver_cache[]), threaded.worker_slots) > 1
        @test !PF._head_needed(
            PF.get_pf(threaded), PNM.KLUSolver(), threaded, threaded.worker_slots[1])
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

@testset "lean LU: a Q-limit flip on the first step still plans, on the start types" begin
    q = (; check_reactive_power_limits = true, correct_bustypes = true)
    data = _lean_sys14_data(; T = 3, q...)
    # Force a flip: a PV bus with a zero reactive band.
    k = findfirst(==(PSY.ACBusTypes.PV), data.bus_type[:, 1])
    for t in 1:3
        data.bus_reactive_power_bounds[k, t] = (0.0, 0.0)
    end
    start = data.bus_type[:, 1]
    @test with_logger(() -> solve_power_flow!(data), NullLogger())
    @test data.bus_type[k, 1] == PSY.ACBusTypes.PQ
    slot = _lean_slot(data)
    @test slot.tried && slot.valid
    @test slot.bus_types == start
    @test PF._lean_counts(data).rejects == 0
    klu = _lean_sys14_data(; T = 3, q...)
    for t in 1:3
        klu.bus_reactive_power_bounds[k, t] = (0.0, 0.0)
    end
    @test _without_lean(() -> with_logger(() -> solve_power_flow!(klu), NullLogger()))
    @test data.bus_type == klu.bus_type
    @test isapprox(data.bus_magnitude, klu.bus_magnitude; atol = 1e-8)
end

@testset "lean LU: a converged 0-iteration step builds the plan from the stored state" begin
    kw = (; generator_slack_participation_factors = nothing)
    data = _lean_sys14_data(; kw...)
    @test solve_power_flow!(data)
    fresh = _lean_sys14_data(; kw...)
    # The solved injections too: the stored state then meets the tolerance as it stands.
    for field in (:bus_magnitude, :bus_angles, :bus_active_power_injections,
        :bus_reactive_power_injections)
        copyto!(getfield(fresh, field), getfield(data, field))
    end
    @test solve_power_flow!(fresh)
    @test iszero(PF.get_iterations(fresh)[1])
    @test _lean_slot(fresh).tried && _lean_slot(fresh).valid
    @test _lean_plan_hash(_lean_slot(fresh).plan) == _lean_plan_hash(_lean_slot(data).plan)
end
