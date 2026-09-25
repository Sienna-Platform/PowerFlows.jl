@testset "solver_kwargs exposes solver parameters and hides the controls" begin
    kwargs = PF.solver_kwargs(SolutionParameters())
    # `maxIterations` is a static field of the returned NamedTuple; an unresolved value
    # reports the sentinel, resolved to a solver's default by the model constructor.
    @test kwargs.maxIterations == PF.UNSET_MAX_ITERATIONS
    @test kwargs.tol == PF.DEFAULT_NR_TOL
    @test kwargs.validate_voltage_magnitudes == PF.DEFAULT_VALIDATE_VOLTAGES
    @test kwargs.λ_0 == PF.DEFAULT_λ_0
    @test kwargs.Δt_k == PF.DEFAULT_Δt_k

    # Network controls are read through their accessors, never splatted into a solver.
    for field in (
        :check_reactive_power_limits,
        :enhanced_flat_start,
        :control_discrete_devices,
        :area_interchange_control,
        :interchange_tolerance,
        :tie_definition,
        :model_dc_network,
    )
        @test !haskey(kwargs, field)
    end

    @test PF.solver_kwargs(SolutionParameters(; maxIterations = 7)).maxIterations == 7
end

@testset "SolutionParameters reaches the model and its accessors" begin
    params = SolutionParameters(;
        tol = 1e-7,
        maxIterations = 12,
        check_reactive_power_limits = true,
        control_discrete_devices = true,
        enhanced_flat_start = false,
    )
    pf = ACPolarPowerFlow{NewtonRaphsonACPowerFlow}(; solution_parameters = params)

    @test PF.get_solver_kwargs(pf).tol == 1e-7
    @test PF.get_solver_kwargs(pf).maxIterations == 12

    # DC models carry no parameters of their own, so the accessors fall back to defaults.
    @test PF.get_solution_parameters(DCPowerFlow()) == SolutionParameters()
    @test !PF.get_control_discrete_devices(DCPowerFlow())
end

@testset "Legacy keywords still configure a model" begin
    # The named keywords remain a supported spelling and fold into the stored parameters.
    pf = ACPolarPowerFlow{NewtonRaphsonACPowerFlow}(;
        check_reactive_power_limits = true,
        control_discrete_devices = true,
        enhanced_flat_start = false,
    )
    @test PF.get_check_reactive_power_limits(pf)
    @test PF.get_control_discrete_devices(pf)
    @test !PF.get_enhanced_flat_start(pf)

    # An explicit keyword wins over `solution_parameters`.
    pf_mixed = ACPolarPowerFlow{NewtonRaphsonACPowerFlow}(;
        solution_parameters = SolutionParameters(; control_discrete_devices = true),
        control_discrete_devices = false,
    )
    @test !PF.get_control_discrete_devices(pf_mixed)
end

@testset "check_reactive_power_limits overrides per call" begin
    # `_solve_with_q_limits!` must read the per-call override, not just the stored parameter.
    # c_sys14 bus 8's Q violates its limit unless the flag is honored.
    sys = PSB.build_system(PSB.PSITestSystems, "c_sys14"; add_forecasts = false)
    pf = ACPolarPowerFlow{NewtonRaphsonACPowerFlow}(; correct_bustypes = true)
    data = PF.PowerFlowData(pf, sys)
    bt0 = copy(data.bus_type[:, 1])
    @test PF.solve_power_flow!(data; check_reactive_power_limits = true)
    flips = findall(data.bus_type[:, 1] .!= bt0)
    @test !isempty(flips)
    for i in flips
        bt0[i] == PSY.ACBusTypes.PV || continue
        q = data.bus_reactive_power_injections[i, 1]
        (qmin, qmax) = data.bus_reactive_power_bounds[i, 1]
        @test qmin - 1e-6 <= q <= qmax + 1e-6
    end
end

@testset "Parameter validation still runs against the solver type" begin
    # Discrete control is NR/TR only.
    @test_throws ArgumentError ACPolarPowerFlow{LevenbergMarquardtACPowerFlow}(;
        solution_parameters = SolutionParameters(; control_discrete_devices = true),
    )
    # Area interchange is polar only.
    @test_throws ArgumentError ACRectangularPowerFlow{NewtonRaphsonACPowerFlow}(;
        solution_parameters = SolutionParameters(; area_interchange_control = true),
    )
    # Only the tie-line definition is implemented.
    @test_throws ArgumentError ACPolarPowerFlow{NewtonRaphsonACPowerFlow}(;
        solution_parameters = SolutionParameters(;
            area_interchange_control = true,
            tie_definition = :lines_and_loads,
        ),
    )
    # A non-positive interchange tolerance is floored, and the model stores the floored
    # value rather than the one the caller asked for.
    pf = ACPolarPowerFlow{NewtonRaphsonACPowerFlow}(;
        solution_parameters = SolutionParameters(;
            area_interchange_control = true,
            interchange_tolerance = -1.0,
        ),
    )
    @test PF.get_interchange_tolerance(pf) == PF.MIN_INTERCHANGE_TOLERANCE
end

@testset "maxIterations < 1 errors loudly instead of silently non-converging" begin
    sys = PSB.build_system(PSB.PSITestSystems, "c_sys14"; add_forecasts = false)
    pf = ACPolarPowerFlow{NewtonRaphsonACPowerFlow}(; correct_bustypes = true)
    data = PF.PowerFlowData(pf, sys)
    @test_throws ErrorException PF.solve_power_flow!(data; maxIterations = -1)
end
