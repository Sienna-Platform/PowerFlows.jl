const _THREADED_FIELDS = (
    :bus_magnitude, :bus_angles, :bus_type,
    :bus_active_power_injections, :bus_reactive_power_injections,
    :arc_active_power_flow_from_to, :arc_reactive_power_flow_from_to,
    :arc_active_power_flow_to_from, :arc_reactive_power_flow_to_from,
    :arc_angle_differences, :converged, :iterations,
)

# Threading refuses every backend but KLU, whatever the platform default.
_klu(n_threads = 1) = SolutionParameters(; linear_solver = "KLU", n_threads)
const _KLU = _klu()

function _threaded_data(build, time_steps, n_threads, perturb!)
    data = build(time_steps, n_threads)
    perturb!(data)
    return data
end

function _test_threads_bitwise(build, time_steps, perturb!; resolve::Bool)
    serial = _threaded_data(build, time_steps, 1, perturb!)
    @test solve_power_flow!(serial)
    resolve && @test solve_power_flow!(serial)
    for n_threads in (2, 4, 16)
        threaded = _threaded_data(build, time_steps, n_threads, perturb!)
        @test solve_power_flow!(threaded)
        resolve && @test solve_power_flow!(threaded)
        for f in _THREADED_FIELDS
            @test isequal(getfield(serial, f), getfield(threaded, f))
        end
        dc_s, dc_t = PF.get_dc_network(serial), PF.get_dc_network(threaded)
        for f in (:p_c, :q_c, :node_vdc)
            @test isequal(getfield(dc_s, f), getfield(dc_t, f))
        end
    end
end

@testset "threaded time steps equal the serial solve: c_sys14 T=24" begin
    for ACSolver in (NewtonRaphsonACPowerFlow, TrustRegionACPowerFlow)
        build = function (T, n_threads)
            sys = PSB.build_system(PSB.PSITestSystems, "c_sys14"; add_forecasts = false)
            return PowerFlowData(
                ACPowerFlow{ACSolver}(;
                    time_steps = T, solution_parameters = _klu(n_threads)),
                sys,
            )
        end
        # Re-solve from the converged state too: every step starts at its own solution.
        _test_threads_bitwise(build, 24, d -> prepare_ts_data!(d, 24); resolve = true)
    end
end

@testset "iterations per time step" begin
    sys = PSB.build_system(PSB.PSITestSystems, "c_sys14"; add_forecasts = false)
    data = PowerFlowData(
        ACPowerFlow{NewtonRaphsonACPowerFlow}(; time_steps = 2, solution_parameters = _KLU),
        sys,
    )
    @test solve_power_flow!(data)
    @test all(>(0), PowerFlows.get_iterations(data))
    # From the converged state every step starts within tolerance.
    @test solve_power_flow!(data)
    @test all(iszero, PowerFlows.get_iterations(data))
end

@testset "threaded time steps equal the serial solve: ACTIVSg2000 T=8" begin
    sys = PSB.build_system(PSB.MatpowerTestSystems, "matpower_ACTIVSg2000_sys")
    build = function (T, n_threads)
        pf = ACPowerFlow{NewtonRaphsonACPowerFlow}(;
            time_steps = T, correct_bustypes = true,
            solution_parameters = _klu(n_threads))
        return PowerFlowData(pf, sys)
    end
    perturb! = function (data)
        for t in 1:size(data.bus_active_power_withdrawals, 2)
            data.bus_active_power_withdrawals[:, t] .*= 1.0 + 0.002 * t
            data.bus_reactive_power_withdrawals[:, t] .*= 1.0 + 0.002 * t
        end
    end
    _test_threads_bitwise(build, 8, perturb!; resolve = true)
end

# Workers share the DC network: the GA flat start must not restore other workers' columns.
@testset "threaded time steps equal the serial solve: GA flat start with VSC" begin
    build = function (T, n_threads)
        params = SolutionParameters(;
            linear_solver = "KLU", model_dc_network = true, n_threads)
        pf = ACPowerFlow{NewtonRaphsonACPowerFlow}(;
            time_steps = T, ga_flat_start = true, solution_parameters = params)
        return PowerFlowData(pf, _build_vsc_system())
    end
    perturb! = function (data)
        for t in 1:size(data.bus_active_power_withdrawals, 2)
            data.bus_active_power_withdrawals[:, t] .*= 1.0 + 0.02 * t
            data.bus_reactive_power_withdrawals[:, t] .*= 1.0 + 0.02 * t
        end
    end
    _test_threads_bitwise(build, 8, perturb!; resolve = false)
end

@testset "threaded time steps: time_steps subset and n_threads > steps" begin
    sys = PSB.build_system(PSB.PSITestSystems, "c_sys14"; add_forecasts = false)
    pf =
        ACPowerFlow{NewtonRaphsonACPowerFlow}(; time_steps = 24, solution_parameters = _KLU)
    serial = PowerFlowData(pf, sys)
    prepare_ts_data!(serial, 24)
    pf8 = ACPowerFlow{NewtonRaphsonACPowerFlow}(;
        time_steps = 24, solution_parameters = _klu(8))
    threaded = PowerFlowData(pf8, sys)
    prepare_ts_data!(threaded, 24)
    steps = [2, 5, 6, 11, 20]
    @test solve_power_flow!(serial; time_steps = steps)
    @test solve_power_flow!(threaded; time_steps = steps)
    @test isequal(serial.bus_magnitude, threaded.bus_magnitude)
    @test isequal(serial.converged, threaded.converged)
    @test count(threaded.converged) == length(steps)
end

@testset "threaded time steps reject repeated time steps" begin
    sys = PSB.build_system(PSB.PSITestSystems, "c_sys14"; add_forecasts = false)
    data = PowerFlowData(
        ACPowerFlow{NewtonRaphsonACPowerFlow}(;
            time_steps = 2, solution_parameters = _klu(2)),
        sys,
    )
    @test_throws ArgumentError solve_power_flow!(data; time_steps = [1, 1])
end

@testset "threaded time steps refuse a per-call non-KLU backend" begin
    sys = PSB.build_system(PSB.PSITestSystems, "c_sys14"; add_forecasts = false)
    plain = PowerFlowData(
        ACPowerFlow{NewtonRaphsonACPowerFlow}(;
            time_steps = 2, solution_parameters = _klu(2)),
        sys,
    )
    # AppleAccelerateLU resolves only on Apple. test_threaded_ac_power_flow.jl checks the
    # backend tags on every platform.
    if Sys.isapple()
        @test_throws r"requires the KLU" solve_power_flow!(
            plain; linear_solver = "AppleAccelerateLU")
    end
    @test_throws r"n_threads" solve_power_flow!(plain; threads = 2)
end
