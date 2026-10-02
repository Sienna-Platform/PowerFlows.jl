# Threaded multi-period AC solves (`SolutionParameters(; n_threads > 1)`) must reproduce the
# sequential solve on every per-step output. On a fresh `PowerFlowData` the sequential solve
# has no earlier converged step to warm-start from, so both start every step from the system
# setpoints and differ only by factorization round-off.

const THREADED_PARITY_ATOL = 1e-9

# With one thread the tasks run but never concurrently, so races could not show up: fail
# rather than pass vacuously. Threads come from `WORKER_EXEFLAGS` in runtests.jl.
@testset "threaded tests have threads" begin
    @test Threads.nthreads() >= 2
end

_threaded_params(n_threads; kwargs...) =
    SolutionParameters(; linear_solver = "LeanKLU", n_threads = n_threads, kwargs...)

@testset "n_threads validation" begin
    @test_throws ArgumentError ACPolarPowerFlow(;
        solution_parameters = SolutionParameters(; n_threads = 0))
    @test_throws r"LeanKLU" ACPolarPowerFlow(;
        solution_parameters = SolutionParameters(; linear_solver = "KLU", n_threads = 2))
    @test_throws r"ACPolarPowerFlow" ACRectangularPowerFlow(;
        solution_parameters = _threaded_params(2))
    @test_throws r"ACPolarPowerFlow" ACMixedPowerFlow(;
        solution_parameters = _threaded_params(2))
    @test_throws r"NewtonRaphsonACPowerFlow" ACPolarPowerFlow{
        LevenbergMarquardtACPowerFlow,
    }(;
        solution_parameters = _threaded_params(2))
    for ACSolver in (NewtonRaphsonACPowerFlow, TrustRegionACPowerFlow)
        pf = ACPolarPowerFlow{ACSolver}(; solution_parameters = _threaded_params(2))
        @test PF.get_n_threads(pf) == 2
    end
    # A KLU (or any) backend is fine when not threaded.
    @test PF.get_n_threads(ACPolarPowerFlow()) == 1
end

@testset "_copy_for_task shares read-only state only" begin
    @test PF._copy_for_task(nothing) === nothing
    pf = ACPolarPowerFlow(; solution_parameters = _threaded_params(1))
    data = PowerFlowData(pf,
        PSB.build_system(PSB.PSITestSystems, "c_sys14"; add_forecasts = false))
    @test solve_power_flow!(data)
    entry = data.polar_nr_cache[]
    c = PF._copy_for_task(entry)
    @test c.linSolveCache.plan === entry.linSolveCache.plan
    @test c.linSolveCache !== entry.linSolveCache
    @test c.J.od_jnz === entry.J.od_jnz
    @test c.J.Jv !== entry.J.Jv
    @test c.residual.Rv !== entry.residual.Rv
    @test c.stateVector.x !== entry.stateVector.x
    @test isnothing(c.stateVector.fallback_cache[])
    @test c.J.bus_slack_participation_factors ===
          c.residual.bus_slack_participation_factors
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

_matrix_fields(x) = [f for f in fieldnames(typeof(x)) if getfield(x, f) isa Matrix{<:Real}]

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
    end
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

# Not compared against the sequential solve: taps update the ComplexF32 Y-bus by deltas, so the
# sequential solve accumulates round-off across steps (~8e-6 pu in reactive injections here)
# while each worker starts from a fresh copy. Instead, with one step per task, each threaded
# step must match a single-step solve at that step's load.
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
    build = () -> PSB.build_system(PSB.PSITestSystems, "c_sys14"; add_forecasts = false)
    reference = nothing
    for _ in 1:10
        pf = ACPolarPowerFlow(; time_steps = 24, solution_parameters = _threaded_params(4))
        data = PowerFlowData(pf, build())
        prepare_ts_data!(data, 24)
        @test solve_power_flow!(data)
        state = (copy(data.bus_magnitude), copy(data.bus_angles),
            copy(data.arc_active_power_flow_from_to))
        if isnothing(reference)
            reference = state
        else
            @test state == reference
        end
    end
end
