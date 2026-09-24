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

Every solver-facing field name matches the keyword the corresponding solver already
accepts, so a parameter may also be overridden per call — `solve_power_flow!(data; tol =
1e-8)` wins over the stored value for that solve only. The network-control fields
(`check_reactive_power_limits`, `enhanced_flat_start`, `control_discrete_devices`,
`area_interchange_control`, `interchange_tolerance`, `tie_definition`,
`model_dc_network`) are read from the stored parameters, not from a per-call keyword,
because most of them shape `PowerFlowData` at construction time — a keyword passed to
`solve_power_flow!` after that has nothing left to change. `check_reactive_power_limits`
is the one exception: it is re-read on every Q-limit retry, so a per-call override does
take effect.

# Convergence
- `tol::Float64`: convergence threshold on the ∞-norm of the per-unit mismatch.
- `maxIterations::Int`: iteration cap. Left unset, a formulation constructor (e.g.
  [`ACPolarPowerFlow`](@ref)) resolves it to the solver's own default —
  `DEFAULT_NR_MAX_ITER` for Newton-type solvers, `DEFAULT_FD_MAX_ITER` for fast decoupled.

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
- `λ_0`, `marquardt_scaling::Bool`: Marquardt diagonal column scaling. The formulation
  constructor (e.g. [`ACRectangularPowerFlow`](@ref)) resolves its own default
  (`true` for rectangular + LM, `false` elsewhere) unless a `marquardt_scaling`
  keyword is given explicitly there.

# Fast decoupled
- `handoff_solver` (`NoHandoff` for pure FD), `handoff_tol`, `refreeze_on_stall`,
  `fd_non_divergent`, `fd_blowup`, `fd_dvlim`, `fd_vm_abort`, `fd_ndvfct`,
  `fd_max_step_halvings`.

# Robust homotopy
- `Δt_k`: continuation step size.

# Gradient descent (Adam)
- `learning_rate`, `beta1`, `beta2`, `epsilon`.

# Backend
- `linear_solver::String`: name of the sparse linear-solver backend. Defaults to the
  `PowerNetworkMatrices` preference default, resolved once at construction.

Per-call data (`x0`) is not a parameter and is not carried here — pass it at the call site.
"""

"""Sentinel [`ACPowerFlowSolverType`](@ref)-shaped marker for "no fast-decoupled handoff
solver configured" — the [`SolutionParameters`](@ref) `handoff_solver` default. A concrete
singleton type (not `nothing`) keeps the field concretely typed; FD dispatches on the value
(`_fd_maybe_handoff!(::Type{NoHandoff}, …)` vs. the solver-type method) instead of an
`isnothing` check."""
struct NoHandoff end

Base.@kwdef struct SolutionParameters
    tol::Float64 = DEFAULT_NR_TOL
    # `UNSET_MAX_ITERATIONS` keeps each solver's own default: 50 for Newton-type solvers,
    # 150 for fast decoupled. A formulation constructor resolves it via
    # `_default_max_iterations`; a concrete `Int` (not `nothing`) keeps the field stable.
    maxIterations::Int = UNSET_MAX_ITERATIONS

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

    λ_0::Float64 = DEFAULT_λ_0
    marquardt_scaling::Bool = false

    # `handoff_solver` is typed as `DataType`, not `ACPowerFlowSolverType`, because that
    # type is defined after this file in the include order; `_validate_fd_handoff_solver`
    # checks the value anyway. Defaults to the `NoHandoff` sentinel (not `nothing`) so the
    # field stays concrete.
    handoff_solver::DataType = NoHandoff
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

    linear_solver::String = PNM._default_linear_solver()
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

"""
    solver_kwargs(params::SolutionParameters) -> NamedTuple

The solver-facing parameters as a `NamedTuple`, ready to splat into a solver call.
Network-control fields are excluded — those are read through their accessors.

Field access is written out literally (not `map(getfield, names)`) so the return type
infers as a concrete `NamedTuple` rather than `Any`.
"""
function solver_kwargs(params::SolutionParameters)
    return (;
        tol = params.tol,
        maxIterations = params.maxIterations,
        validate_voltage_magnitudes = params.validate_voltage_magnitudes,
        vm_validation_range = params.vm_validation_range,
        refinement_threshold = params.refinement_threshold,
        refinement_eps = params.refinement_eps,
        iwamoto = params.iwamoto,
        stop_at_fold = params.stop_at_fold,
        factor = params.factor,
        eta = params.eta,
        autoscale = params.autoscale,
        iwamoto_fallback = params.iwamoto_fallback,
        λ_0 = params.λ_0,
        marquardt_scaling = params.marquardt_scaling,
        handoff_solver = params.handoff_solver,
        handoff_tol = params.handoff_tol,
        refreeze_on_stall = params.refreeze_on_stall,
        fd_non_divergent = params.fd_non_divergent,
        fd_blowup = params.fd_blowup,
        fd_dvlim = params.fd_dvlim,
        fd_vm_abort = params.fd_vm_abort,
        fd_ndvfct = params.fd_ndvfct,
        fd_max_step_halvings = params.fd_max_step_halvings,
        Δt_k = params.Δt_k,
        learning_rate = params.learning_rate,
        beta1 = params.beta1,
        beta2 = params.beta2,
        epsilon = params.epsilon,
        linear_solver = params.linear_solver,
    )
end

# Guards against a new SolutionParameters field silently missing from the literal list above.
@assert Set(keys(solver_kwargs(SolutionParameters()))) ==
        Set(setdiff(fieldnames(SolutionParameters), _SOLUTION_PARAMETER_CONTROL_FIELDS))

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

"""
    _apply_legacy_kwargs(params; legacy_kwargs...) -> SolutionParameters

Fold the per-constructor control keywords (`check_reactive_power_limits`,
`control_discrete_devices`, ...) into `params`. These remain supported spellings —
a `nothing` value means "not passed", so the stored parameter is kept.
"""
function _apply_legacy_kwargs(
    params::SolutionParameters;
    check_reactive_power_limits::Union{Nothing, Bool} = nothing,
    enhanced_flat_start::Union{Nothing, Bool} = nothing,
    control_discrete_devices::Union{Nothing, Bool} = nothing,
    area_interchange_control::Union{Nothing, Bool} = nothing,
    interchange_tolerance::Union{Nothing, Float64} = nothing,
    tie_definition::Union{Nothing, Symbol} = nothing,
)
    legacy = (;
        check_reactive_power_limits,
        enhanced_flat_start,
        control_discrete_devices,
        area_interchange_control,
        interchange_tolerance,
        tie_definition,
    )
    overrides = Dict{Symbol, Any}()
    for (name, value) in pairs(legacy)
        isnothing(value) || (overrides[name] = value)
    end
    return _override(params, overrides)
end
