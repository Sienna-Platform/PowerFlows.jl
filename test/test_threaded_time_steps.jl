const _THREADED_FIELDS = (
    :bus_magnitude, :bus_angles, :bus_type,
    :bus_active_power_injections, :bus_reactive_power_injections,
    :arc_active_power_flow_from_to, :arc_reactive_power_flow_from_to,
    :arc_active_power_flow_to_from, :arc_reactive_power_flow_to_from,
    :arc_angle_differences, :converged, :iterations,
)

# Threading refuses every backend but KLU, whatever the platform default.
const _KLU = SolutionParameters(; linear_solver = "KLU")

function _threaded_data(build, time_steps, perturb!)
    data = build(time_steps)
    perturb!(data)
    return data
end

function _test_threads_bitwise(build, time_steps, perturb!; resolve::Bool)
    serial = _threaded_data(build, time_steps, perturb!)
    @test solve_power_flow!(serial)
    resolve && @test solve_power_flow!(serial)
    for threads in (2, 4, 16)
        threaded = _threaded_data(build, time_steps, perturb!)
        @test solve_power_flow!(threaded; threads = threads)
        resolve && @test solve_power_flow!(threaded; threads = threads)
        for f in _THREADED_FIELDS
            @test isequal(getfield(serial, f), getfield(threaded, f))
        end
    end
end

@testset "threaded time steps equal the serial solve: c_sys14 T=24" begin
    for ACSolver in (NewtonRaphsonACPowerFlow, TrustRegionACPowerFlow)
        build = function (T)
            sys = PSB.build_system(PSB.PSITestSystems, "c_sys14"; add_forecasts = false)
            return PowerFlowData(
                ACPowerFlow{ACSolver}(; time_steps = T, solution_parameters = _KLU), sys,
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
    build = function (T)
        pf = ACPowerFlow{NewtonRaphsonACPowerFlow}(;
            time_steps = T, correct_bustypes = true, solution_parameters = _KLU)
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

@testset "threaded time steps: time_steps subset and threads > steps" begin
    sys = PSB.build_system(PSB.PSITestSystems, "c_sys14"; add_forecasts = false)
    pf =
        ACPowerFlow{NewtonRaphsonACPowerFlow}(; time_steps = 24, solution_parameters = _KLU)
    serial = PowerFlowData(pf, sys)
    prepare_ts_data!(serial, 24)
    threaded = PowerFlowData(pf, sys)
    prepare_ts_data!(threaded, 24)
    steps = [2, 5, 6, 11, 20]
    @test solve_power_flow!(serial; time_steps = steps)
    @test solve_power_flow!(threaded; time_steps = steps, threads = 8)
    @test isequal(serial.bus_magnitude, threaded.bus_magnitude)
    @test isequal(serial.converged, threaded.converged)
    @test count(threaded.converged) == length(steps)
    @test_throws ErrorException solve_power_flow!(threaded; threads = 0)
end

@testset "threaded time steps refuse shared per-solve state" begin
    shunt = PowerFlowData(
        ACPowerFlow{NewtonRaphsonACPowerFlow}(;
            control_discrete_devices = true, time_steps = 3),
        _make_multiperiod_shunt_system(),
    )
    @test_throws r"discrete device control" solve_power_flow!(shunt; threads = 2)

    area = PowerFlowData(
        ACPolarPowerFlow{NewtonRaphsonACPowerFlow}(;
            area_interchange_control = true, time_steps = 2),
        _three_area_transfer_fixture(; slack_area3 = true),
    )
    @test_throws r"area interchange" solve_power_flow!(area; threads = 2)

    lcc_sys, _ = simple_lcc_system()
    lcc = PowerFlowData(
        ACPowerFlow{NewtonRaphsonACPowerFlow}(; time_steps = 2, correct_bustypes = true),
        lcc_sys,
    )
    @test_throws r"LCC" solve_power_flow!(lcc; threads = 2)

    sys = PSB.build_system(PSB.PSITestSystems, "c_sys14"; add_forecasts = false)
    plain = PowerFlowData(ACPowerFlow{NewtonRaphsonACPowerFlow}(; time_steps = 2), sys)
    @test_throws r"KLU" solve_power_flow!(plain; threads = 2, linear_solver = "MKLPardiso")
    # One step never reaches the threaded path.
    @test solve_power_flow!(lcc; threads = 2, time_steps = [1])
end
