"""
    SolutionParameters(; kwargs...)

The parameter set for a power flow solve: convergence targets, step-control limits,
the network controls to enforce, and the linear-solver backend.

This is the single parameter interface to PowerFlows. The same type carries parameters
chosen in Julia and parameters read from an industrial case file (see
[`read_solution_parameters`](@ref)), so a case can be solved with the settings it was
distributed with and exported back with the settings it was actually solved with.

Attach one to an evaluation model with the `solution_parameters` keyword:

```julia
pf = ACPolarPowerFlow{NewtonRaphsonACPowerFlow}(;
    solution_parameters = SolutionParameters(; tol = 1e-8, control_discrete_devices = true),
)
```

Every field name matches the keyword the corresponding solver already accepts, so a
parameter may also be overridden per call — `solve_power_flow!(data; tol = 1e-8)` wins
over the stored value for that solve only.

# Convergence
- `tol::Float64`: convergence threshold on the ∞-norm of the per-unit mismatch.
- `maxIterations::Union{Nothing, Int}`: iteration cap. `nothing` (the default) leaves each
  solver on its own default — `DEFAULT_NR_MAX_ITER` for Newton-type solvers,
  `DEFAULT_FD_MAX_ITER` for fast decoupled.

# Network controls
- `check_reactive_power_limits::Bool`: enforce generator reactive limits by switching PV
  buses to PQ between solves.
- `enhanced_flat_start::Bool`: fall back to the enhanced flat start when the initial
  residual is large.
- `control_discrete_devices::Bool`: run discrete device control (tap changers, switched
  shunts) via λ-continuation.
- `area_interchange_control::Bool`: embed per-area net-interchange control in the solve.
- `interchange_tolerance::Float64`: interchange tolerance (pu), used for validation and
  reporting. Non-positive values are floored to `MIN_INTERCHANGE_TOLERANCE`.
- `tie_definition::Symbol`: how area ties are identified. Only `:lines_only` is
  implemented.
- `model_dc_network::Bool`: lower VSC/DC network equations into the joint AC–DC solve.

# Voltage validation
- `validate_voltage_magnitudes::Bool`, `vm_validation_range::MinMax`.

# Newton / trust region / Levenberg-Marquardt
- `refinement_threshold`, `refinement_eps`, `iwamoto`, `stop_at_fold`.
- `factor`, `eta`, `autoscale`, `iwamoto_fallback`.
- `λ_0`, `marquardt_scaling` (`nothing` selects the per-formulation default).

# Fast decoupled
- `handoff_solver` (`nothing` for pure FD), `handoff_tol`, `refreeze_on_stall`,
  `fd_non_divergent`, `fd_blowup`, `fd_dvlim`, `fd_vm_abort`, `fd_ndvfct`,
  `fd_max_step_halvings`.

# Robust homotopy
- `Δt_k`: continuation step size.

# Gradient descent (Adam)
- `learning_rate`, `beta1`, `beta2`, `epsilon`.

# Backend
- `linear_solver`: name of the sparse linear-solver backend, or `nothing` to take the
  `PowerNetworkMatrices` preference default.

Per-call data (`x0`) is not a parameter and is not carried here — pass it at the call site.
"""
Base.@kwdef struct SolutionParameters
    # `maxIterations === nothing` keeps each solver's own default: 50 for Newton-type
    # solvers, 150 for fast decoupled.
    tol::Float64 = DEFAULT_NR_TOL
    maxIterations::Union{Nothing, Int} = nothing

    # Read through the `get_*` accessors — never splatted into a solver call.
    check_reactive_power_limits::Bool = false
    enhanced_flat_start::Bool = true
    control_discrete_devices::Bool = false
    area_interchange_control::Bool = false
    interchange_tolerance::Float64 = DEFAULT_INTERCHANGE_TOLERANCE
    tie_definition::Symbol = :lines_only
    model_dc_network::Bool = true

    validate_voltage_magnitudes::Bool = DEFAULT_VALIDATE_VOLTAGES
    vm_validation_range::MinMax = DEFAULT_VALIDATION_RANGE

    refinement_threshold::Float64 = DEFAULT_REFINEMENT_THRESHOLD
    refinement_eps::Float64 = DEFAULT_REFINEMENT_EPS
    iwamoto::Bool = false
    stop_at_fold::Bool = false

    factor::Float64 = DEFAULT_TRUST_REGION_FACTOR
    eta::Float64 = DEFAULT_TRUST_REGION_ETA
    autoscale::Bool = DEFAULT_AUTOSCALE
    iwamoto_fallback::Bool = DEFAULT_IWAMOTO_FALLBACK

    # `marquardt_scaling === nothing` selects the per-formulation default (off for polar,
    # on for rectangular).
    λ_0::Float64 = DEFAULT_λ_0
    marquardt_scaling::Union{Bool, Nothing} = nothing

    # `handoff_solver` is typed as `DataType`, not `ACPowerFlowSolverType`, because that
    # type is defined after this file in the include order; `_validate_fd_handoff_solver`
    # checks the value anyway.
    handoff_solver::Union{Nothing, DataType} = nothing
    handoff_tol::Float64 = DEFAULT_FD_HANDOFF_TOL
    refreeze_on_stall::Bool = DEFAULT_FD_REFREEZE_ON_STALL
    fd_non_divergent::Bool = DEFAULT_FD_NON_DIVERGENT
    fd_blowup::Float64 = DEFAULT_FD_BLOWUP
    fd_dvlim::Float64 = DEFAULT_FD_DVLIM
    fd_vm_abort::Float64 = DEFAULT_FD_VM_ABORT
    fd_ndvfct::Float64 = DEFAULT_FD_NDVFCT
    fd_max_step_halvings::Int = DEFAULT_FD_MAX_STEP_HALVINGS

    Δt_k::Float64 = DEFAULT_Δt_k

    # Defaults mirror `AdamConfig`.
    learning_rate::Float64 = 0.01
    beta1::Float64 = 0.9
    beta2::Float64 = 0.999
    epsilon::Float64 = 1e-8

    linear_solver::Union{Nothing, AbstractString} = nothing
end

# Excluded from `get_solver_kwargs` so the kwargs surface a solver sees matches what it saw
# when these were separate struct fields on the evaluation model.
const _SOLUTION_PARAMETER_CONTROL_FIELDS = (
    :check_reactive_power_limits,
    :enhanced_flat_start,
    :control_discrete_devices,
    :area_interchange_control,
    :interchange_tolerance,
    :tie_definition,
    :model_dc_network,
)

const _SOLUTION_PARAMETER_SOLVER_FIELDS = Tuple(
    name for name in fieldnames(SolutionParameters)
    if !(name in _SOLUTION_PARAMETER_CONTROL_FIELDS)
)

# `maxIterations` is the sentinel field: emitting `nothing` would override the solver's own
# default, so it's dropped when unset. Both field lists are precomputed to avoid rebuilding
# a tuple on every solve.
const _SOLUTION_PARAMETER_SOLVER_FIELDS_NO_ITER = Tuple(
    name for name in _SOLUTION_PARAMETER_SOLVER_FIELDS if name !== :maxIterations
)

"""
    solver_kwargs(params::SolutionParameters) -> NamedTuple

The solver-facing parameters as a `NamedTuple`, ready to splat into a solver call.
Network-control fields are excluded — those are read through their accessors — and
`maxIterations` is omitted when unset so each solver keeps its own default.
"""
function solver_kwargs(params::SolutionParameters)
    names = if isnothing(params.maxIterations)
        _SOLUTION_PARAMETER_SOLVER_FIELDS_NO_ITER
    else
        _SOLUTION_PARAMETER_SOLVER_FIELDS
    end
    return NamedTuple{names}(map(n -> getfield(params, n), names))
end

"""
    SolutionParameters(settings::AbstractDict) -> SolutionParameters

Build a `SolutionParameters` from a legacy `solver_settings` dictionary. Keys that name a
field are applied to the defaults; any other key is dropped with a warning, since it would
previously have been splatted into a solver and silently ignored there.
"""
SolutionParameters(settings::AbstractDict) =
    _override(SolutionParameters(), _settings_overrides(settings))

"""
    _override(x, overrides::AbstractDict) -> typeof(x)

A copy of `x` with the named fields replaced. Generic over any struct type with a
`T(field_values...)` constructor. Used by the evaluation-model constructors to store a
validated value (a floored `interchange_tolerance`, say), to fold the legacy keyword
spellings in, and by `SolutionRecordValues`'s copy-with-overrides — without rebuilding the
struct by hand.
"""
function _override(x::T, overrides::AbstractDict) where {T}
    isempty(overrides) && return x
    values = map(fieldnames(T)) do name
        if haskey(overrides, name)
            overrides[name]
        else
            getfield(x, name)
        end
    end
    return T(values...)
end

_override(x; kwargs...) = _override(x, Dict{Symbol, Any}(kwargs))

function _settings_overrides(settings)
    overrides = Dict{Symbol, Any}()
    isnothing(settings) && return overrides
    for (key, value) in settings
        sym = Symbol(key)
        if sym in fieldnames(SolutionParameters)
            overrides[sym] = value
        else
            @warn(
                "solver_settings key :$sym does not name a SolutionParameters field and " *
                "was dropped. Pass it as a keyword to the solve call instead.",
                maxlog = 1,
            )
        end
    end
    return overrides
end

"""
    _fold_legacy_parameters(params, solver_settings; legacy_kwargs...) -> SolutionParameters

Merge the deprecated ways of specifying solve parameters into `params`, in increasing
order of precedence: `params` itself, then the `solver_settings` dictionary, then any
explicitly-passed legacy keyword (a `nothing` value means "not passed").

The named keywords (`check_reactive_power_limits`, `control_discrete_devices`, ...) remain
supported spellings and are not deprecated; only the untyped `solver_settings` dictionary
is, so only it raises a `depwarn`.
"""
function _fold_legacy_parameters(
    params::SolutionParameters,
    solver_settings;
    check_reactive_power_limits::Union{Nothing, Bool} = nothing,
    enhanced_flat_start::Union{Nothing, Bool} = nothing,
    control_discrete_devices::Union{Nothing, Bool} = nothing,
    area_interchange_control::Union{Nothing, Bool} = nothing,
    interchange_tolerance::Union{Nothing, Float64} = nothing,
    tie_definition::Union{Nothing, Symbol} = nothing,
)
    if !isnothing(solver_settings)
        Base.depwarn(
            "`solver_settings` is deprecated; pass " *
            "`solution_parameters = SolutionParameters(...)` instead.",
            :solver_settings,
        )
    end
    overrides = _settings_overrides(solver_settings)
    legacy = (;
        check_reactive_power_limits,
        enhanced_flat_start,
        control_discrete_devices,
        area_interchange_control,
        interchange_tolerance,
        tie_definition,
    )
    for (name, value) in pairs(legacy)
        isnothing(value) || (overrides[name] = value)
    end
    return _override(params, overrides)
end
