@testset "SolutionParameters defaults match the solver defaults" begin
    params = SolutionParameters()
    @test params.tol == PF.DEFAULT_NR_TOL
    # `nothing` so each solver keeps its own iteration budget: 50 for the Newton-type
    # solvers, 150 for fast decoupled.
    @test isnothing(params.maxIterations)
    @test params.enhanced_flat_start
    @test !params.check_reactive_power_limits
    @test !params.control_discrete_devices
    @test !params.area_interchange_control
    @test params.tie_definition === :lines_only
    @test params.interchange_tolerance == PF.DEFAULT_INTERCHANGE_TOLERANCE
    @test params.model_dc_network
    @test params.fd_blowup == PF.DEFAULT_FD_BLOWUP
    @test params.fd_dvlim == PF.DEFAULT_FD_DVLIM
    @test params.fd_ndvfct == PF.DEFAULT_FD_NDVFCT
end

@testset "solver_kwargs exposes solver parameters and hides the controls" begin
    kwargs = PF.solver_kwargs(SolutionParameters())
    # An unset iteration cap must not be splatted: passing `nothing` would override each
    # solver's own default with `nothing`.
    @test !haskey(kwargs, :maxIterations)
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

    @test PF.get_solution_parameters(pf) == params
    @test PF.get_check_reactive_power_limits(pf)
    @test PF.get_control_discrete_devices(pf)
    @test !PF.get_enhanced_flat_start(pf)
    @test PF.get_solver_kwargs(pf).tol == 1e-7
    @test PF.get_solver_kwargs(pf).maxIterations == 12

    # DC models carry no parameters of their own, so the accessors fall back to defaults.
    @test PF.get_solution_parameters(DCPowerFlow()) == SolutionParameters()
    @test !PF.get_control_discrete_devices(DCPowerFlow())
end

@testset "Legacy keywords and solver_settings still configure a model" begin
    # The named keywords remain a supported spelling and fold into the stored parameters.
    pf = ACPolarPowerFlow{NewtonRaphsonACPowerFlow}(;
        check_reactive_power_limits = true,
        control_discrete_devices = true,
        enhanced_flat_start = false,
    )
    @test PF.get_check_reactive_power_limits(pf)
    @test PF.get_control_discrete_devices(pf)
    @test !PF.get_enhanced_flat_start(pf)

    # The deprecated dictionary still reaches the solver.
    pf_dict = ACPolarPowerFlow{NewtonRaphsonACPowerFlow}(;
        solver_settings = Dict{Symbol, Any}(:iwamoto => true, :maxIterations => 11),
    )
    @test PF.get_solver_kwargs(pf_dict).iwamoto
    @test PF.get_solver_kwargs(pf_dict).maxIterations == 11

    # An explicit keyword wins over the dictionary and over `solution_parameters`.
    pf_mixed = ACPolarPowerFlow{NewtonRaphsonACPowerFlow}(;
        solution_parameters = SolutionParameters(; control_discrete_devices = true),
        control_discrete_devices = false,
    )
    @test !PF.get_control_discrete_devices(pf_mixed)

    # A dictionary key that names no parameter is dropped loudly rather than silently
    # splatted into a solver that would ignore it.
    @test_logs (:warn,) match_mode = :any ACPolarPowerFlow{NewtonRaphsonACPowerFlow}(;
        solver_settings = Dict{Symbol, Any}(:not_a_parameter => 1),
    )
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
