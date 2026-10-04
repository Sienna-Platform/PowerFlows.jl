# Threaded multi-period AC solves (`SolutionParameters(; n_threads > 1)`) must reproduce the
# sequential solve on every per-step output. A worker warm-starts only from its own chunk's
# steps, so a first solve matches the sequential one. A worker that plans its own pivot order
# (a private network matrix or area set) differs only by factorization round-off.

const THREADED_PARITY_ATOL = 1e-9

# With one thread the tasks run but never concurrently, so races could not show up: fail
# rather than pass vacuously. Threads come from `WORKER_EXEFLAGS` in runtests.jl.
@testset "threaded tests have threads" begin
    @test Threads.nthreads() >= 2
end

_threaded_params(n_threads; kwargs...) =
    SolutionParameters(; linear_solver = "KLU", n_threads = n_threads, kwargs...)

@testset "n_threads validation" begin
    @test_throws ArgumentError ACPolarPowerFlow(;
        solution_parameters = SolutionParameters(; n_threads = 0))
    @test_throws ArgumentError vPTDFDCPowerFlow(; n_threads = 0)
    @test PF.get_n_threads(vPTDFDCPowerFlow(; n_threads = 3)) == 3
    for backend in (PNM.AppleAccelerateLUSolver(), PNM.MKLPardisoSolver())
        @test_throws r"requires the KLU" PF._check_concurrent_factorization(backend)
    end
    if Sys.isapple()
        @test_throws r"requires the KLU" ACPolarPowerFlow(;
            solution_parameters = SolutionParameters(;
                linear_solver = "AppleAccelerateLU", n_threads = 2))
    end
    for F in (ACPolarPowerFlow, ACRectangularPowerFlow, ACMixedPowerFlow),
        ACSolver in (
            NewtonRaphsonACPowerFlow,
            TrustRegionACPowerFlow,
            LevenbergMarquardtACPowerFlow,
        )

        pf = F{ACSolver}(; solution_parameters = _threaded_params(2))
        @test PF.get_n_threads(pf) == 2
    end
    # Any backend is fine when not threaded.
    @test PF.get_n_threads(ACPolarPowerFlow()) == 1
end

"""Solve `build_sys()` sequentially and with `n_threads` tasks; return both datas. `setup!`
fills the per-step inputs of a fresh `data`."""
function _solve_sequential_and_threaded(
    build_sys,
    setup!;
    time_steps::Int,
    n_threads::Int = 4,
    ACSolver = NewtonRaphsonACPowerFlow,
    params_kwargs = (;),
    pf_kwargs...,
)
    return map((1, n_threads)) do n
        pf = ACPolarPowerFlow{ACSolver}(;
            time_steps = time_steps,
            solution_parameters = _threaded_params(n; params_kwargs...),
            pf_kwargs...,
        )
        data = PowerFlowData(pf, build_sys())
        setup!(data, time_steps)
        solve_power_flow!(data)
        data
    end
end

_is_real_matrix(::Matrix{<:Real}) = true
_is_real_matrix(::Any) = false
_matrix_fields(x) = [f for f in fieldnames(typeof(x)) if _is_real_matrix(getfield(x, f))]

"""Every per-step output of `seq` and `thr` agrees: all real matrices at the top level, in
`lcc` and in the area-interchange data, plus `converged` and the relaxed-area records."""
function _test_threaded_parity(seq, thr; atol = THREADED_PARITY_ATOL)
    @test seq.converged == thr.converged
    for get_container in (
        d -> d,
        d -> d.lcc,
        d -> d.lcc.rectifier,
        d -> d.lcc.inverter,
        d -> d.area_interchange,
    )
        a, b = get_container(seq), get_container(thr)
        for f in _matrix_fields(a)
            f === :delta_p && continue  # per-worker scratch; results read pristine_delta_p
            # The field name rides along so a failure says which output diverged.
            @test (
                f,
                isapprox(getfield(a, f), getfield(b, f);
                    atol = atol, nans = true),
            ) == (f, true)
        end
    end
    @test seq.area_interchange.relaxed == thr.area_interchange.relaxed
    return
end

@testset "threaded parity: c_sys14, 24 steps" begin
    for ACSolver in (NewtonRaphsonACPowerFlow, TrustRegionACPowerFlow)
        seq, thr = _solve_sequential_and_threaded(
            () ->
                PSB.build_system(PSB.PSITestSystems, "c_sys14"; add_forecasts = false),
            prepare_ts_data!;
            time_steps = 24, ACSolver = ACSolver,
        )
        @test all(seq.converged)
        _test_threaded_parity(seq, thr)
        # Workers share one Jacobian structure and pivot order, so a first solve is bitwise equal
        # to the serial solve.
        @test isequal(seq.bus_magnitude, thr.bus_magnitude)
        @test isequal(seq.bus_angles, thr.bus_angles)
    end
end

@testset "seeded workers match serial bitwise (c_sys14 T=24 and ACTIVSg2000 T=16)" begin
    for (build, prepare!, T) in (
        (
            () -> PSB.build_system(PSB.PSITestSystems, "c_sys14"; add_forecasts = false),
            prepare_ts_data!, 24,
        ),
        (
            () -> PSB.build_system(PSB.MatpowerTestSystems, "matpower_ACTIVSg2000_sys"),
            _replicate_col1_to_all_steps!, 16,
        ),
    )
        seq, thr = _solve_sequential_and_threaded(
            build, prepare!; time_steps = T, correct_bustypes = true)
        @test all(seq.converged)
        @test all(thr.converged)
        @test isequal(seq.bus_magnitude, thr.bus_magnitude)
        @test isequal(seq.bus_angles, thr.bus_angles)
        @test seq.iterations == thr.iterations
    end
end

function _slot_data(T, n_threads)
    pf = ACPolarPowerFlow(;
        time_steps = T,
        solution_parameters = _threaded_params(n_threads),
    )
    d = PowerFlowData(
        pf, PSB.build_system(PSB.PSITestSystems, "c_sys14"; add_forecasts = false))
    prepare_ts_data!(d, T)
    return d
end

@testset "seeding shares the first worker's index maps" begin
    d = _slot_data(8, 2)
    @test solve_power_flow!(d)
    a, b = (s.polar_nr_cache[] for s in d.worker_slots)
    @test a.J.od_jnz === b.J.od_jnz
end

@testset "repeated threaded solves reuse worker caches" begin
    T = 16
    d = _slot_data(T, 4)
    ref = _slot_data(T, 1)
    flat = (copy(d.bus_magnitude), copy(d.bus_angles))
    @test solve_power_flow!(d)
    caches = [s.polar_nr_cache[] for s in d.worker_slots]
    @test length(caches) == 4
    @test solve_power_flow!(ref)
    # New inputs, so a skipped head step or chunk leaves stale columns that the match catches.
    for x in (d, ref)
        x.bus_active_power_withdrawals .*= 1.01
        x.bus_reactive_power_withdrawals .*= 1.01
        x.bus_magnitude .= flat[1]
        x.bus_angles .= flat[2]
    end
    @test solve_power_flow!(d)
    @test all(d.converged)
    @test all(s.polar_nr_cache[] === c for (s, c) in zip(d.worker_slots, caches))
    @test solve_power_flow!(ref)
    @test maximum(abs, d.bus_magnitude .- ref.bus_magnitude) < 1e-8
    @test maximum(abs, d.bus_angles .- ref.bus_angles) < 1e-8
    @test all(d.iterations .<= ref.iterations)
end

@testset "invalidating the partition invalidates every worker slot" begin
    d = _slot_data(8, 2)
    @test solve_power_flow!(d)
    PF._invalidate_partition!(d)
    @test all(isempty(s.polar_nr_cache[].bus_type_snapshot) for s in d.worker_slots)
end

@testset "a worker's own data holds no slots" begin
    d = _slot_data(8, 2)
    slot = PF.WorkerSlot()
    w = PF._column_worker(d, 1:8, 1:4, slot)
    @test isempty(w.worker_slots)
    @test w.polar_nr_cache === slot.polar_nr_cache
end

@testset "a slot whose worker raised is cleared" begin
    d = _slot_data(8, 2)
    @test solve_power_flow!(d)
    slot = d.worker_slots[2]
    @test !isnothing(slot.polar_nr_cache[])
    @test_throws Exception PF._solve_slot!(
        fill(false, 8), PF._column_worker(d, 1:8, 5:8, slot), slot,
        d.pf, 1:8, 5:99, (;))
    @test isnothing(slot.polar_nr_cache[])
    @test isnothing(slot.solver_cache[])
    @test solve_power_flow!(d)
end

@testset "a raising step clears its own slot only" begin
    d = _slot_data(8, 2)
    # No reference bus at step 7 makes the solve raise inside the second chunk's task.
    d.bus_type[:, 7] .= PSY.ACBusTypes.PQ
    @test_throws Exception solve_power_flow!(d)
    @test !isnothing(d.worker_slots[1].polar_nr_cache[])
    @test isnothing(d.worker_slots[2].polar_nr_cache[])
    @test isnothing(d.worker_slots[2].solver_cache[])
end

@testset "threaded parity: Q limits" begin
    seq, thr = _solve_sequential_and_threaded(
        () -> PSB.build_system(PSB.PSITestSystems, "c_sys14"; add_forecasts = false),
        _replicate_col1_to_all_steps!;
        time_steps = 6, check_reactive_power_limits = true, correct_bustypes = true,
    )
    @test all(seq.converged)
    _test_threaded_parity(seq, thr)
end

@testset "threaded parity: switched shunts" begin
    seq, thr = _solve_sequential_and_threaded(
        _make_multiperiod_shunt_system, _set_multiperiod_shunt_loads!;
        time_steps = 6, control_discrete_devices = true,
    )
    @test all(seq.converged)
    _test_threaded_parity(seq, thr)
    @test isequal(
        PF.get_controlled_device_results(seq),
        PF.get_controlled_device_results(thr),
    )
end

# Taps update the ComplexF32 Y-bus by deltas. The sequential solve accumulates round-off across
# steps (~8e-6 pu in reactive injections here), but each worker starts from a fresh copy. Thus,
# with one step per task, each threaded step must match a single-step solve at that step's load.
@testset "threaded taps match single-step solves" begin
    time_steps = 6
    pf = ACPolarPowerFlow(;
        time_steps = time_steps,
        control_discrete_devices = true,
        solution_parameters = _threaded_params(time_steps),
    )
    data = PowerFlowData(pf, _make_multiperiod_tap_system())
    _set_multiperiod_tap_loads!(data, time_steps)
    @test solve_power_flow!(data)
    results = PF.get_controlled_device_results(data)
    for t in 1:time_steps
        pf_ref = ACPolarPowerFlow(;
            control_discrete_devices = true,
            solution_parameters = _threaded_params(1),
        )
        ref = PowerFlowData(pf_ref, _make_multiperiod_tap_system())
        _set_single_tap_load!(ref, t)
        @test solve_power_flow!(ref)
        for f in (:bus_magnitude, :bus_angles, :bus_reactive_power_injections,
            :arc_reactive_power_flow_from_to, :arc_reactive_power_flow_to_from)
            @test (
                f,
                t,
                isapprox(getfield(data, f)[:, t], getfield(ref, f)[:, 1];
                    atol = THREADED_PARITY_ATOL),
            ) == (f, t, true)
        end
        @test results[results.time_step .== t, :].final ==
              PF.get_controlled_device_results(ref).final
    end
end

@testset "threaded parity: LCC with discrete control" begin
    seq, thr = _solve_sequential_and_threaded(
        build_lcc_control_system,
        _replicate_col1_to_all_steps!;
        time_steps = 4, control_discrete_devices = true,
    )
    @test all(seq.converged)
    _test_threaded_parity(seq, thr)
end

@testset "threaded parity: area interchange" begin
    seq, thr = _solve_sequential_and_threaded(
        _three_area_transfer_fixture,
        _replicate_col1_to_all_steps!;
        time_steps = 6, area_interchange_control = true,
    )
    @test all(seq.converged)
    _test_threaded_parity(seq, thr)
end

# Races are intermittent: repeat the threaded solve and require identical results each time.
@testset "threaded solves are repeatable" begin
    function threaded_state()
        data = _slot_data(24, 4)
        @test solve_power_flow!(data)
        return (copy(data.bus_magnitude), copy(data.bus_angles),
            copy(data.arc_active_power_flow_from_to))
    end
    reference = threaded_state()
    for _ in 1:9
        @test threaded_state() == reference
    end
end

@testset "_copy_for_task shares read-only state only" begin
    pf = ACPolarPowerFlow(;
        solution_parameters = SolutionParameters(; linear_solver = "KLU"))
    data = PowerFlowData(pf,
        PSB.build_system(PSB.PSITestSystems, "c_sys14"; add_forecasts = false))
    @test solve_power_flow!(data)
    entry = data.polar_nr_cache[]
    slot = data.ac_jacobian_structure_cache[].lean
    dup = PF._copy_for_task(entry, slot, data, 1)
    for f in (:od_ptr, :od_to, :od_ybus_nz, :od_jnz, :diag_jnz, :diag_ybus_nz)
        @test getfield(dup.J, f) === getfield(entry.J, f)
    end
    @test dup.arc_flows.arcs === entry.arc_flows.arcs
    @test dup.arc_flows.fb_ix === entry.arc_flows.fb_ix
    @test dup.arc_flows.tb_ix === entry.arc_flows.tb_ix
    @test dup.lean === entry.lean
    @test dup.linSolveCache !== entry.linSolveCache
    @test dup.J.Jv !== entry.J.Jv
    @test dup.J.Jv == entry.J.Jv
    @test dup.residual.Rv !== entry.residual.Rv
    @test dup.J.bus_state === dup.residual.bus_state
    @test dup.J.bus_state !== entry.J.bus_state
    @test dup.stateVector.x !== entry.stateVector.x
    @test isnothing(dup.stateVector.fallback_cache[])
    @test isnothing(dup.stateVector.fallback_matrix[])
    @test dup.arc_flows.V !== entry.arc_flows.V
    @test dup.x0 !== entry.x0
    @test dup.partition !== entry.partition
    @test dup.J.bus_slack_participation_factors ===
          dup.residual.bus_slack_participation_factors
    @test dup.J.bus_slack_participation_factors !==
          entry.J.bus_slack_participation_factors
    @test PNM.KLUWrapper.has_lean_plan(dup.linSolveCache)
end

@testset "a copy never takes a swapped plan" begin
    pf = ACPolarPowerFlow(;
        solution_parameters = SolutionParameters(; linear_solver = "KLU"))
    data = PowerFlowData(pf,
        PSB.build_system(PSB.PSITestSystems, "c_sys14"; add_forecasts = false))
    @test solve_power_flow!(data)
    k = findlast(==(PSY.ACBusTypes.PV), data.bus_type[:, 1])
    data.bus_type[k, 1] = PSY.ACBusTypes.REF
    data.bus_active_power_withdrawals .*= 1.01
    @test solve_power_flow!(data)
    entry = data.polar_nr_cache[]
    slot = entry.lean
    q0 = copy(slot.plan.q)
    @test entry.linSolveCache.lean_plan.q != slot.plan.q
    dup = PF._copy_for_task(entry, slot, data, 1)
    lin, seed = dup.linSolveCache, entry.linSolveCache
    @test lin.lean_plan.p === slot.plan.p
    @test lin.lean_q !== seed.lean_q
    @test lin.lean_plan.q !== seed.lean_plan.q
    @test slot.plan.q == q0
    @test entry.linSolveCache.lean_plan.q != slot.plan.q

    data.bus_type[k, 1] = PSY.ACBusTypes.PV
    dup2 = PF._copy_for_task(entry, slot, data, 1)
    @test dup2.linSolveCache.lean_plan === slot.plan
end
