"""An abstract supertype for all types of power flows.
Subtypes: [`AbstractACPowerFlow`](@ref), [`AbstractDCPowerFlow`](@ref), and
[`PSSEExportPowerFlow`](@ref). The last isn't a power flow in the usual sense, but it is 
implemented that way (with writing the export file as solving the power flow) for interface reasons."""
abstract type PowerFlowEvaluationModel end

"""An abstract supertype for all AC power flow solver/step strategies.
Subtypes: [`NewtonRaphsonACPowerFlow`](@ref), [`TrustRegionACPowerFlow`](@ref),
[`LevenbergMarquardtACPowerFlow`](@ref), [`RobustHomotopyPowerFlow`](@ref), and
[`GradientDescentACPowerFlow`](@ref).

The solver is orthogonal to the formulation; see [`AbstractACPowerFlow`](@ref),
[`ACPolarPowerFlow`](@ref), [`ACRectangularPowerFlow`](@ref).
"""
abstract type ACPowerFlowSolverType end

"""An abstract supertype for AC power flow evaluation models, parametrized by the
solver type `S <: ACPowerFlowSolverType`. Concrete subtypes select the *formulation*:
[`ACPolarPowerFlow`](@ref) uses the polar voltage state; a rectangular
current-injection formulation is provided separately. The solver and the
formulation are orthogonal."""
abstract type AbstractACPowerFlow{S <: ACPowerFlowSolverType} <: PowerFlowEvaluationModel end

"""An abstract supertype for the persistent per-solve caches stored in
`PowerFlowData.solver_cache[]`. Concrete subtypes ([`DCSolverCache`](@ref) for the DC/PTDF path,
`FastDecoupledCache` for the polar fast-decoupled solver) are type-disjoint, so the slot's type
discriminates which path populated it — no sentinel tag is needed and a cross-use is a plain
`MethodError` rather than a silent reuse."""
abstract type SolverCache end

"""Memoized AC-Jacobian sparse structure, stored in its OWN `PowerFlowData` field (not the shared
`solver_cache` slot): the NR/TR AC Jacobian and a [`SolverCache`](@ref) can both be live in one
solve — e.g. a FastDecoupled solve that hands off to NR uses a `FastDecoupledCache` *and* this
structure — so the two must not contend for a single slot. Cache key is the network-matrix
identity + slack nonzero pattern (`nzind`); see `_get_or_build_jacobian_structure`."""
struct ACJacobianStructureCache
    matrix::PNM.AC_Ybus_Matrix
    nzind::Vector{Int}
    structure::SparseMatrixCSC{Float64, J_INDEX_TYPE}
    area_data::AreaInterchangeData
end

# Centralized so the multi-line warning text can't drift between the two
# formulation constructors.
function _validate_slack_distribution_settings(
    distribute_slack_proportional_to_headroom::Bool,
    generator_slack_participation_factors,
    time_steps::Int,
)
    if distribute_slack_proportional_to_headroom &&
       !isnothing(generator_slack_participation_factors)
        error(
            "Cannot use both distribute_slack_proportional_to_headroom and generator_slack_participation_factors.",
        )
    end
    # This scenario can be handled fine from PSI, we just don't handle it in PF alone.
    if distribute_slack_proportional_to_headroom && time_steps > 1
        @warn(
            "distribute_slack_proportional_to_headroom with multiple time steps: " *
            "headroom (Pmax - Pset) is computed once from system data and applied " *
            "to all time steps. Time-varying active power limits and generator " *
            "setpoints are not supported.",
        )
    end
    return
end

# The fast-decoupled `FDDecoupled` variant (classic B′/B″ half-iterations) is polar-only — its
# half-steps don't span the rectangular/mixed state. Rejected at construction on the non-polar
# formulations so the misconfiguration fails immediately with a descriptive message. `FDDecoupled`
# and `FastDecoupledACPowerFlow` are defined later in this file; the reference resolves at call time.
function _reject_fd_decoupled_on_nonpolar(
    ::Type{ACSolver},
    formulation::String,
) where {
    ACSolver <: ACPowerFlowSolverType,
}
    if ACSolver <: FastDecoupledACPowerFlow{FDDecoupled}
        throw(
            ArgumentError(
                "$(ACSolver) is not supported by $(formulation): the FDDecoupled variant " *
                "(classic B′/B″ half-iterations) is polar-only. Use " *
                "FastDecoupledACPowerFlow{FDFixedJacobian, …} on this formulation, or run " *
                "FDDecoupled on ACPolarPowerFlow.",
            ),
        )
    end
    return
end

# Discrete device control is validated only for NR/TR inner solvers (FD reuses stale B′/B″
# after tap moves; LM/GD/Homotopy are unvalidated). All device families support multiple
# time steps: shunts/FACTS via the per-ts state store, taps via the reset-to-baseline Y-bus
# design (see `ControlledDeviceSet`/`load_device_state!`). Centralized so the three
# formulation constructors cannot drift. NR/TR types are defined later in this file —
# references resolve at call time.
function _validate_discrete_control_settings(
    control_discrete_devices::Bool,
    ::Type{ACSolver},
) where {ACSolver <: ACPowerFlowSolverType}
    control_discrete_devices || return
    if !(ACSolver <: Union{NewtonRaphsonACPowerFlow, TrustRegionACPowerFlow})
        throw(
            ArgumentError(
                "control_discrete_devices=true requires a NewtonRaphsonACPowerFlow or " *
                "TrustRegionACPowerFlow solver; got $(ACSolver). Other solvers are not " *
                "validated as continuation inner solvers (FastDecoupled would reuse " *
                "stale B′/B″ factorizations after tap moves).",
            ),
        )
    end
    return
end

# Area interchange control (Phase 1) is validated only for NR/TR inner solvers: LM is
# deferred to Phase 2 validation of the augmented least-squares residual; Robust Homotopy,
# Gradient Descent, and Fast Decoupled have no natural home for the interchange border.
# Centralized so the three formulation constructors cannot drift. NR/TR types are defined
# later in this file — references resolve at call time. Multi-period is allowed (no
# time_steps guard). Returns the (possibly floored) interchange_tolerance.
function _validate_area_interchange_settings(
    ::Type{ACSolver},
    area_interchange_control::Bool,
    interchange_tolerance::Float64,
    tie_definition::Symbol,
) where {ACSolver <: ACPowerFlowSolverType}
    area_interchange_control || return interchange_tolerance
    if !(ACSolver <: Union{NewtonRaphsonACPowerFlow, TrustRegionACPowerFlow})
        throw(
            ArgumentError(
                "area_interchange_control=true requires a NewtonRaphsonACPowerFlow or " *
                "TrustRegionACPowerFlow solver; got $(ACSolver). Levenberg-Marquardt is " *
                "deferred to Phase 2 validation of the augmented least-squares residual; " *
                "Robust Homotopy, Gradient Descent, and Fast Decoupled are not supported.",
            ),
        )
    end
    if tie_definition !== :lines_only
        throw(
            ArgumentError(
                "tie_definition=$(tie_definition) is not supported; only :lines_only is " *
                "implemented. :lines_and_loads (PSS/E control code 2) is reserved for a " *
                "future phase.",
            ),
        )
    end
    if interchange_tolerance <= 0
        @warn(
            "interchange_tolerance=$(interchange_tolerance) is non-positive; flooring to " *
            "MIN_INTERCHANGE_TOLERANCE=$(MIN_INTERCHANGE_TOLERANCE).",
        )
        return MIN_INTERCHANGE_TOLERANCE
    end
    return interchange_tolerance
end

# Phase 1 area interchange control is polar-only: reject immediately on the non-polar
# formulations regardless of solver. Centralized so the rejection message can't drift.
function _reject_area_interchange_on_nonpolar(
    area_interchange_control::Bool,
    formulation::String,
)
    area_interchange_control || return
    throw(
        ArgumentError(
            "area_interchange_control=true is not supported by $(formulation): Phase 1 " *
            "of area interchange control is polar-only. Use ACPolarPowerFlow.",
        ),
    )
end

"""
    NewtonRaphsonACPowerFlow <: ACPowerFlowSolverType

An [`ACPowerFlowSolverType`](@ref) corresponding to a basic Newton-Raphson iterative method.
The Newton step is taken verbatim at each iteration: no line search is performed.

Iwamoto step control can be enabled via `solver_settings = Dict(:iwamoto => true)` in
[`ACPowerFlow`](@ref). When enabled, each iteration checks whether the full Newton step
reduces the residual norm. If it does, the full step is accepted (overhead: 3 dot products).
If not, an optimal damping multiplier `μ` is computed by solving a cubic and the step
`x += μ·Δx` is applied instead, preventing divergence on ill-conditioned or
poorly-initialized systems.

Based on: Iwamoto & Tamura, "A Load Flow Calculation Method for Ill-Conditioned Power
Systems," IEEE Trans. PAS, 1981.

See also: [`ACPowerFlow`](@ref).
"""
struct NewtonRaphsonACPowerFlow <: ACPowerFlowSolverType end

"""
    TrustRegionACPowerFlow <: ACPowerFlowSolverType

An [`ACPowerFlowSolverType`](@ref) corresponding to the [Powell dogleg](https://en.wikipedia.org/wiki/Powell%27s_dog_leg_method) iterative method. 
This is a bit more robust than the basic Newton-Raphson method and comparably lightweight.

See also: [`ACPowerFlow`](@ref).
"""
struct TrustRegionACPowerFlow <: ACPowerFlowSolverType end

"""
    LevenbergMarquardtACPowerFlow <: ACPowerFlowSolverType

An [`ACPowerFlowSolverType`](@ref) corresponding to the [Levenberg-Marquardt](https://en.wikipedia.org/wiki/Levenberg–Marquardt_algorithm) iterative method.
This is more robust than the basic Newton-Raphson method, but also more computationally
intensive. Due to the difficulty of tuning meta parameters, this method may occasionally
fail to converge where other methods would succeed.

Works with both the polar ([`ACPolarPowerFlow`](@ref)) and rectangular
current-injection ([`ACRectangularPowerFlow`](@ref)) formulations.

Marquardt diagonal column scaling (`√λ·D` damping instead of `√λ·I`) can be
toggled via `solver_settings = Dict(:marquardt_scaling => true|false)`. When
unset it defaults **on** for [`ACRectangularPowerFlow`](@ref) — whose state
columns `(e, f, Q, P_gen)` are differently scaled, so identity damping is
ill-conditioned — and **off** for [`ACPolarPowerFlow`](@ref), leaving the polar
solver numerically unchanged.

See also: [`ACPowerFlow`](@ref).
"""
struct LevenbergMarquardtACPowerFlow <: ACPowerFlowSolverType end

"""
    RobustHomotopyPowerFlow <: ACPowerFlowSolverType

An [`ACPowerFlowSolverType`](@ref) corresponding to a homotopy iterative method, based on the
paper [\"Improving the robustness of Newton-based power flow methods to cope with poor
initial points\"](https://ieeexplore.ieee.org/document/6666905). This is significantly more
robust than Newton-Raphson, but also slower by an order of magnitude or two.

See also: [`ACPowerFlow`](@ref).
"""
struct RobustHomotopyPowerFlow <: ACPowerFlowSolverType end

"""Abstract supertype for the [`FastDecoupledACPowerFlow`](@ref) iteration variants
([`FDDecoupled`](@ref), [`FDFixedJacobian`](@ref)). Carried as the first type parameter of the
solver so the iteration scheme is selected by multiple dispatch rather than a runtime flag."""
abstract type FDVariant end

"""Classic fast-decoupled variant: constant B′/B″ half-iterations.
Polar formulation only. See [`FDVariant`](@ref), [`FastDecoupledACPowerFlow`](@ref)."""
struct FDDecoupled <: FDVariant end

"""Frozen-Jacobian ("dishonest Newton") variant: the full per-formulation Jacobian is factored
once at `x0` and reused every iteration. Works with all three AC formulations.
See [`FDVariant`](@ref), [`FastDecoupledACPowerFlow`](@ref)."""
struct FDFixedJacobian <: FDVariant end

"""Abstract supertype for the B′/B″ construction scheme of the [`FDDecoupled`](@ref) variant
([`FDSchemeXB`](@ref), [`FDSchemeBX`](@ref)). Carried as the second type parameter of
[`FastDecoupledACPowerFlow`](@ref) and dispatched on during matrix assembly. Only meaningful for
[`FDDecoupled`](@ref)."""
abstract type FDScheme end

"""`XB` scheme: B′ neglects branch resistance, B″ keeps it. See [`FDScheme`](@ref)."""
struct FDSchemeXB <: FDScheme end

"""`BX` scheme: B″ neglects branch resistance, B′ keeps it. See [`FDScheme`](@ref)."""
struct FDSchemeBX <: FDScheme end

"""
    FastDecoupledACPowerFlow{V<:FDVariant, S<:FDScheme} <: ACPowerFlowSolverType
    FastDecoupledACPowerFlow  (bare: per-formulation defaults)

An [`ACPowerFlowSolverType`](@ref) implementing fixed-slope decoupled Newton-Raphson (the classic
fast decoupled power flow). Constant approximate
Jacobian factor(s) are built once and reused across all iterations *and* time steps, while the
exact mismatches are evaluated every iteration. This trades the quadratic convergence rate of
Newton-Raphson for a linear rate at a fraction of the per-iteration cost — ideal for repeated
solves, contingency screening, and as a cheap initializer for the exact-Newton family.

Works with all three AC formulations ([`ACPolarPowerFlow`](@ref),
[`ACRectangularPowerFlow`](@ref), [`ACMixedPowerFlow`](@ref)).

# Type parameters (the iteration options)
- `V <: FDVariant`: [`FDDecoupled`](@ref) (classic B′/B″ half-iterations; polar only) or
    [`FDFixedJacobian`](@ref) (frozen full-formulation Jacobian; all formulations). When the bare
    `FastDecoupledACPowerFlow` is used, the variant defaults per formulation: `FDDecoupled` for
    polar, `FDFixedJacobian` for rectangular/mixed.
- `S <: FDScheme`: B′/B″ scheme, [`FDSchemeXB`](@ref) (default) or [`FDSchemeBX`](@ref). Only
    meaningful for the [`FDDecoupled`](@ref) variant.

```julia
ACPowerFlow{FastDecoupledACPowerFlow}()                                   # per-formulation defaults
ACPowerFlow{FastDecoupledACPowerFlow{FDDecoupled, FDSchemeBX}}()          # explicit variant + scheme
ACRectangularPowerFlow{FastDecoupledACPowerFlow{FDFixedJacobian, FDSchemeXB}}()
```

# Settings (via `solver_settings` and/or call kwargs)
- `handoff_solver`: `nothing` (pure FD; default) or [`NewtonRaphsonACPowerFlow`](@ref) /
    [`TrustRegionACPowerFlow`](@ref) / [`LevenbergMarquardtACPowerFlow`](@ref) for final
    refinement to `tol`.
- `handoff_tol::Float64`: FD-stage exit ∞-norm when a handoff solver is configured.

See also: [`ACPowerFlow`](@ref), [`NewtonRaphsonACPowerFlow`](@ref).
"""
struct FastDecoupledACPowerFlow{V <: FDVariant, S <: FDScheme} <: ACPowerFlowSolverType end

"""Alias for the classic decoupled fast power flow with the XB scheme,
[`FastDecoupledACPowerFlow`](@ref)`{`[`FDDecoupled`](@ref)`, `[`FDSchemeXB`](@ref)`}`. Use as a
solver type parameter, e.g. `ACPowerFlow{FastDecoupledXB}()`."""
const FastDecoupledXB = FastDecoupledACPowerFlow{FDDecoupled, FDSchemeXB}

"""Alias for the frozen-Jacobian fast decoupled power flow,
[`FastDecoupledACPowerFlow`](@ref)`{`[`FDFixedJacobian`](@ref)`, `[`FDSchemeXB`](@ref)`}` (the
scheme is nominal — the fixed-Jacobian variant builds no B′/B″). Works on all three formulations,
e.g. `ACRectangularPowerFlow{FastDecoupledFixed}()`."""
const FastDecoupledFixed = FastDecoupledACPowerFlow{FDFixedJacobian, FDSchemeXB}

"""
    ACPowerFlow{ACSolver}(; kwargs...) where {ACSolver <: ACPowerFlowSolverType}
    ACPowerFlow(; kwargs...)

An evaluation model for a standard
[AC power flow](https://en.wikipedia.org/wiki/Power-flow_study#Power-flow_problem_formulation)
with the specified solver type.

# Arguments
- `ACSolver` (type parameter): The AC iterative solver tag, a subtype of [`ACPowerFlowSolverType`](@ref)
    (for example [`NewtonRaphsonACPowerFlow`](@ref), [`TrustRegionACPowerFlow`](@ref),
    [`LevenbergMarquardtACPowerFlow`](@ref), or [`RobustHomotopyPowerFlow`](@ref)).
    If not specified, defaults to [`NewtonRaphsonACPowerFlow`](@ref).
- `check_reactive_power_limits::Bool`: Whether to check reactive power limits during the power flow solution.
    Default is `false`.
- `exporter::Union{Nothing, PowerFlowEvaluationModel}`: An optional exporter for the power flow results.
    If not `nothing`, it should be a [`PSSEExportPowerFlow`](@ref). Default is `nothing`.
- `calculate_loss_factors::Bool`: Whether to calculate loss factors during the power flow solution.
    Default is `false`.
- `calculate_voltage_stability_factors::Bool`: Whether to calculate voltage stability factors.
    Default is `false`.
- `generator_slack_participation_factors`: An optional parameter that specifies the participation
    factors for generator slack in the power flow solution. If `nothing`, all slack is picked up by
    the reference bus. If a `Dict{Tuple{DataType, String}, Float64}`, it should map
    `(component_type, component_name)` tuples to participation factors. If a `Vector` of such
    dictionaries, different participation factors can be used for different time steps. Default is `nothing`.
- `enhanced_flat_start::Bool`: Whether to use enhanced flat start initialization. Default is `true`.
- `robust_power_flow::Bool`: Whether to use run a DC power flow as a fallback if the initial residual is large.
    Default is `false`.
- `skip_redistribution::Bool`: Whether to skip slack redistribution. Default is `false`.
- `network_reductions::Vector{PNM.NetworkReduction}`: Network reductions to apply.
    Default is an empty vector. A `PNM.ZeroImpedanceBranchReduction` placed here is routed to
    PNM's dedicated zero-impedance step; set its `resistance_tolerance` to also merge
    near-zero-impedance branches that carry a tiny nonzero resistance (PSS/E-style).
- `time_steps::Int`: Number of time steps to solve. Default is `1`.
- `time_step_names::Vector{String}`: Names for each time step. Default is an empty vector.
- `correct_bustypes::Bool`: Whether to automatically correct bus types based on available generation.
    Default is `false`.
- `control_discrete_devices::Bool`: Whether to run discrete device control (tap changers, switched
    shunts) via λ-continuation. Default is `false`.
- `area_interchange_control::Bool`: Whether to embed PSS/E-style per-area net-interchange
    control in the AC Newton system (Phase 1: polar formulation, NR/TR solvers only).
    Default is `false`.
- `interchange_tolerance::Float64`: PTOL analogue (pu); used for validation and reporting only —
    the embedded formulation targets each area's PDES exactly. Non-positive values are floored to
    `MIN_INTERCHANGE_TOLERANCE` with a warning. Default is `DEFAULT_INTERCHANGE_TOLERANCE`.
- `tie_definition::Symbol`: How area ties are identified. Only `:lines_only` is implemented;
    `:lines_and_loads` (PSS/E control code 2) is reserved. Default is `:lines_only`.
- `solver_settings::Dict{Symbol, Any}`: Additional keyword arguments to pass to the solver.
    Default is an empty dictionary.
"""
struct ACPolarPowerFlow{ACSolver <: ACPowerFlowSolverType} <: AbstractACPowerFlow{ACSolver}
    check_reactive_power_limits::Bool
    exporter::Union{Nothing, PowerFlowEvaluationModel}
    calculate_loss_factors::Bool
    calculate_voltage_stability_factors::Bool
    log_solver_diagnostics::Bool
    generator_slack_participation_factors::Union{
        Nothing,
        Dict{Tuple{DataType, String}, Float64},
        Vector{Dict{Tuple{DataType, String}, Float64}},
    }
    enhanced_flat_start::Bool
    robust_power_flow::Bool
    skip_redistribution::Bool
    distribute_slack_proportional_to_headroom::Bool
    network_reductions::Vector{PNM.NetworkReduction}
    time_steps::Int
    time_step_names::Vector{String}
    correct_bustypes::Bool
    control_discrete_devices::Bool
    area_interchange_control::Bool
    interchange_tolerance::Float64
    tie_definition::Symbol
    solver_settings::Dict{Symbol, Any}
end

"""
    ACPowerFlow{ACSolver}(
        check_reactive_power_limits::Bool = false,
        exporter::Union{Nothing, PowerFlowEvaluationModel} = nothing,
        calculate_loss_factors::Bool = false,
        generator_slack_participation_factors::Union{
            Nothing,
            Dict{Tuple{DataType, String}, Float64},
            Vector{Dict{Tuple{DataType, String}, Float64}},
        } = nothing,
    ) where {ACSolver <: ACPowerFlowSolverType}

An evaluation model for a standard 
[AC power flow](https://en.wikipedia.org/wiki/Power-flow_study#Power-flow_problem_formulation) 
with the specified solver type.


# Arguments
- `ACSolver` (type parameter): The AC iterative solver tag, a subtype of [`ACPowerFlowSolverType`](@ref)
    (for example [`NewtonRaphsonACPowerFlow`](@ref) or [`TrustRegionACPowerFlow`](@ref)).
    Default is [`NewtonRaphsonACPowerFlow`](@ref).
- `check_reactive_power_limits::Bool`: Whether to check reactive power limits during the power flow solution.
    Default is `false`.
- `exporter::Union{Nothing, PowerFlowEvaluationModel}`: An optional exporter for the power flow results. 
    If not `nothing`, it should be a [`PSSEExportPowerFlow`](@ref).
- `calculate_loss_factors::Bool`: Whether to calculate loss factors during the power flow solution.
    Default is `false`.
- `generator_slack_participation_factors::Union{Nothing, Dict{Tuple{DataType, String}, Float64}, Vector{Dict{Tuple{DataType, String}, Float64}}}`:
    An optional parameter that specifies the participation factors for generator slack in the power flow solution.
    If `nothing`, all slack is picked up by the reference bus. If a `Dict`, it should map `(component_type, component_name)`
    tuples to participation factors. If a `Vector`, it should contain multiple such dictionaries, 
    allowing for different participation factors for different time steps.
"""
function ACPolarPowerFlow{ACSolver}(;
    check_reactive_power_limits::Bool = false,
    exporter::Union{Nothing, PowerFlowEvaluationModel} = nothing,
    calculate_loss_factors::Bool = false,
    calculate_voltage_stability_factors::Bool = false,
    log_solver_diagnostics::Bool = false,
    generator_slack_participation_factors::Union{
        Nothing,
        Dict{Tuple{DataType, String}, Float64},
        Vector{Dict{Tuple{DataType, String}, Float64}},
    } = nothing,
    enhanced_flat_start::Bool = true,
    robust_power_flow::Bool = false,
    skip_redistribution::Bool = false,
    distribute_slack_proportional_to_headroom::Bool = false,
    network_reductions::Vector{PNM.NetworkReduction} = PNM.NetworkReduction[],
    time_steps::Int = 1,
    time_step_names::Vector{String} = String[],
    correct_bustypes::Bool = false,
    control_discrete_devices::Bool = false,
    area_interchange_control::Bool = false,
    interchange_tolerance::Float64 = DEFAULT_INTERCHANGE_TOLERANCE,
    tie_definition::Symbol = :lines_only,
    solver_settings::AbstractDict = Dict{Symbol, Any}(),
) where {ACSolver <: ACPowerFlowSolverType}
    settings = Dict{Symbol, Any}(solver_settings)
    if calculate_loss_factors && ACSolver == LevenbergMarquardtACPowerFlow
        error("Loss factor calculation is not supported by the Levenberg-Marquardt solver.")
    end
    _validate_slack_distribution_settings(
        distribute_slack_proportional_to_headroom,
        generator_slack_participation_factors,
        time_steps,
    )
    _validate_discrete_control_settings(control_discrete_devices, ACSolver)
    validated_interchange_tolerance = _validate_area_interchange_settings(
        ACSolver,
        area_interchange_control,
        interchange_tolerance,
        tie_definition,
    )
    return ACPolarPowerFlow{ACSolver}(
        check_reactive_power_limits,
        exporter,
        calculate_loss_factors,
        calculate_voltage_stability_factors,
        log_solver_diagnostics,
        generator_slack_participation_factors,
        enhanced_flat_start,
        robust_power_flow,
        skip_redistribution,
        distribute_slack_proportional_to_headroom,
        network_reductions,
        time_steps,
        time_step_names,
        correct_bustypes,
        control_discrete_devices,
        area_interchange_control,
        validated_interchange_tolerance,
        tie_definition,
        settings,
    )
end

# Default constructor: ACPolarPowerFlow() defaults to NewtonRaphsonACPowerFlow solver
ACPolarPowerFlow(; kwargs...) = ACPolarPowerFlow{NewtonRaphsonACPowerFlow}(; kwargs...)

"""`ACPowerFlow` is an alias for [`ACPolarPowerFlow`](@ref), kept for backward
compatibility with PowerSimulations.jl and external callers. It is a plain type
alias (no deprecation warning); polar remains the default AC formulation."""
const ACPowerFlow = ACPolarPowerFlow

get_enhanced_flat_start(pf::AbstractACPowerFlow) = pf.enhanced_flat_start
get_distribute_slack_proportional_to_headroom(::PowerFlowEvaluationModel) = false
get_distribute_slack_proportional_to_headroom(pf::AbstractACPowerFlow) =
    pf.distribute_slack_proportional_to_headroom
get_slack_participation_factors(pf::AbstractACPowerFlow) =
    pf.generator_slack_participation_factors
get_network_reductions(pf::AbstractACPowerFlow) = pf.network_reductions
get_time_steps(pf::AbstractACPowerFlow) = pf.time_steps
get_time_step_names(pf::AbstractACPowerFlow) = pf.time_step_names
get_correct_bustypes(pf::AbstractACPowerFlow) = pf.correct_bustypes
get_solver_kwargs(pf::AbstractACPowerFlow) = pf.solver_settings

# Polar-only fields: rectangular has no equivalent, so default to false.
get_robust_power_flow(::AbstractACPowerFlow) = false
get_robust_power_flow(pf::ACPolarPowerFlow) = pf.robust_power_flow
get_calculate_loss_factors(::AbstractACPowerFlow) = false
get_calculate_loss_factors(pf::ACPolarPowerFlow) = pf.calculate_loss_factors
get_calculate_voltage_stability_factors(::AbstractACPowerFlow) = false
get_calculate_voltage_stability_factors(pf::ACPolarPowerFlow) =
    pf.calculate_voltage_stability_factors
# Works on every AC formulation: the diagnostic needs only J (1st derivatives).
get_log_solver_diagnostics(::PowerFlowEvaluationModel) = false
get_log_solver_diagnostics(pf::AbstractACPowerFlow) = pf.log_solver_diagnostics

get_control_discrete_devices(pf::AbstractACPowerFlow) = pf.control_discrete_devices
get_control_discrete_devices(::PowerFlowEvaluationModel) = false

get_area_interchange_control(pf::AbstractACPowerFlow) = pf.area_interchange_control
get_area_interchange_control(::PowerFlowEvaluationModel) = false
get_interchange_tolerance(pf::AbstractACPowerFlow) = pf.interchange_tolerance
get_interchange_tolerance(::PowerFlowEvaluationModel) = DEFAULT_INTERCHANGE_TOLERANCE
get_tie_definition(pf::AbstractACPowerFlow) = pf.tie_definition
get_tie_definition(::PowerFlowEvaluationModel) = :lines_only

"""
    ACRectangularPowerFlow{ACSolver}(; kwargs...) where {ACSolver <: ACPowerFlowSolverType}
    ACRectangularPowerFlow(; kwargs...)

An evaluation model for the AC power flow solved with the augmented
current-injection (Da Costa) formulation in rectangular coordinates.

State per bus: PQ `(eᵢ, fᵢ)`, PV `(eᵢ, fᵢ, Qᵢ)`, REF `(P_genᵢ, Q_genᵢ)` with
`(eᵢ, fᵢ)` fixed. Residual is the complex current mismatch
`ΔIᵢ = I_specᵢ − Y_bus·V`. Off-diagonal Jacobian blocks ≡ Y_bus 2×2 real blocks
and are constant across iterations.

`ACSolver` defaults to [`NewtonRaphsonACPowerFlow`](@ref). Supported solvers:
[`NewtonRaphsonACPowerFlow`](@ref), [`TrustRegionACPowerFlow`](@ref), and
[`LevenbergMarquardtACPowerFlow`](@ref). Robust Homotopy and Gradient Descent
operate on the polar formulation only and are rejected at construction.

Unlike [`ACPolarPowerFlow`](@ref), this model has no
`calculate_voltage_stability_factors`, `calculate_loss_factors`, or
`robust_power_flow` options — those post-processing/fallback paths assume the
polar state layout and have no current-injection equivalent.

# Arguments
- `check_reactive_power_limits::Bool`: Default `false`.
- `exporter::Union{Nothing, PowerFlowEvaluationModel}`: Default `nothing`.
- `generator_slack_participation_factors`: Same semantics as
    [`ACPolarPowerFlow`](@ref). Default `nothing`.
- `enhanced_flat_start::Bool`: Default `true`.
- `skip_redistribution::Bool`: Default `false`.
- `distribute_slack_proportional_to_headroom::Bool`: Default `false`.
- `network_reductions::Vector{PNM.NetworkReduction}`: Default empty.
- `time_steps::Int`: Default `1`.
- `time_step_names::Vector{String}`: Default empty.
- `correct_bustypes::Bool`: Default `false`.
- `control_discrete_devices::Bool`: Whether to run discrete device control via λ-continuation.
    Default `false`.
- `area_interchange_control::Bool`: Not supported on this formulation (Phase 1 is polar-only);
    passing `true` throws `ArgumentError`. Default `false`.
- `solver_settings::Dict{Symbol, Any}`: Default empty.
"""
struct ACRectangularPowerFlow{ACSolver <: ACPowerFlowSolverType} <:
       AbstractACPowerFlow{ACSolver}
    check_reactive_power_limits::Bool
    exporter::Union{Nothing, PowerFlowEvaluationModel}
    generator_slack_participation_factors::Union{
        Nothing,
        Dict{Tuple{DataType, String}, Float64},
        Vector{Dict{Tuple{DataType, String}, Float64}},
    }
    enhanced_flat_start::Bool
    log_solver_diagnostics::Bool
    skip_redistribution::Bool
    distribute_slack_proportional_to_headroom::Bool
    network_reductions::Vector{PNM.NetworkReduction}
    time_steps::Int
    time_step_names::Vector{String}
    correct_bustypes::Bool
    control_discrete_devices::Bool
    area_interchange_control::Bool
    interchange_tolerance::Float64
    tie_definition::Symbol
    solver_settings::Dict{Symbol, Any}
end

function ACRectangularPowerFlow{ACSolver}(;
    check_reactive_power_limits::Bool = false,
    exporter::Union{Nothing, PowerFlowEvaluationModel} = nothing,
    generator_slack_participation_factors::Union{
        Nothing,
        Dict{Tuple{DataType, String}, Float64},
        Vector{Dict{Tuple{DataType, String}, Float64}},
    } = nothing,
    enhanced_flat_start::Bool = true,
    log_solver_diagnostics::Bool = false,
    skip_redistribution::Bool = false,
    distribute_slack_proportional_to_headroom::Bool = false,
    network_reductions::Vector{PNM.NetworkReduction} = PNM.NetworkReduction[],
    time_steps::Int = 1,
    time_step_names::Vector{String} = String[],
    correct_bustypes::Bool = false,
    control_discrete_devices::Bool = false,
    area_interchange_control::Bool = false,
    interchange_tolerance::Float64 = DEFAULT_INTERCHANGE_TOLERANCE,
    tie_definition::Symbol = :lines_only,
    solver_settings::Dict{Symbol, Any} = Dict{Symbol, Any}(),
) where {ACSolver <: ACPowerFlowSolverType}
    if ACSolver <: Union{
        RobustHomotopyPowerFlow,
        GradientDescentACPowerFlow,
    }
        throw(
            ArgumentError(
                "$(ACSolver) is not supported by ACRectangularPowerFlow. " *
                "Robust Homotopy and Gradient Descent operate on the polar " *
                "formulation only. Use ACRectangularPowerFlow{NewtonRaphsonACPowerFlow}, " *
                "{TrustRegionACPowerFlow}, or {LevenbergMarquardtACPowerFlow}, " *
                "or run the solver on ACPolarPowerFlow.",
            ),
        )
    end
    _reject_fd_decoupled_on_nonpolar(ACSolver, "ACRectangularPowerFlow")
    _reject_area_interchange_on_nonpolar(
        area_interchange_control,
        "ACRectangularPowerFlow",
    )
    _validate_slack_distribution_settings(
        distribute_slack_proportional_to_headroom,
        generator_slack_participation_factors,
        time_steps,
    )
    _validate_discrete_control_settings(control_discrete_devices, ACSolver)
    return ACRectangularPowerFlow{ACSolver}(
        check_reactive_power_limits,
        exporter,
        generator_slack_participation_factors,
        enhanced_flat_start,
        log_solver_diagnostics,
        skip_redistribution,
        distribute_slack_proportional_to_headroom,
        network_reductions,
        time_steps,
        time_step_names,
        correct_bustypes,
        control_discrete_devices,
        area_interchange_control,
        interchange_tolerance,
        tie_definition,
        solver_settings,
    )
end

ACRectangularPowerFlow(; kwargs...) =
    ACRectangularPowerFlow{NewtonRaphsonACPowerFlow}(; kwargs...)

"""
    ACMixedPowerFlow{ACSolver}(; kwargs...) where {ACSolver <: ACPowerFlowSolverType}
    ACMixedPowerFlow(; kwargs...)

An evaluation model for the AC power flow solved with the Mixed
Current-Power Balance (MCPB) formulation in rectangular coordinates.

State per bus: PQ `(eᵢ, fᵢ)` with the divided complex current balance
`(I_specᵢ − Y_bus·V)ᵢ / V̄ᵢ`, PV `(eᵢ, fᵢ)` with the real power balance plus
the `|Vᵢ|²` magnitude constraint, REF `(P_genᵢ, Q_genᵢ)` with `(eᵢ, fᵢ)` fixed.
There are 2 variables per bus, so the system size is `2n`.

`ACSolver` defaults to [`NewtonRaphsonACPowerFlow`](@ref). Supported solvers:
[`NewtonRaphsonACPowerFlow`](@ref), [`TrustRegionACPowerFlow`](@ref), and
[`LevenbergMarquardtACPowerFlow`](@ref). Robust Homotopy and Gradient Descent
are rejected at construction — they operate on the polar formulation only.

Unlike [`ACPolarPowerFlow`](@ref), this model has no
`calculate_voltage_stability_factors`, `calculate_loss_factors`, or
`robust_power_flow` options — those post-processing/fallback paths assume the
polar state layout and have no mixed current-power equivalent.

# Arguments
- `check_reactive_power_limits::Bool`: Default `false`.
- `exporter::Union{Nothing, PowerFlowEvaluationModel}`: Default `nothing`.
- `generator_slack_participation_factors`: Same semantics as
    [`ACPolarPowerFlow`](@ref). Default `nothing`.
- `enhanced_flat_start::Bool`: Default `true`.
- `skip_redistribution::Bool`: Default `false`.
- `distribute_slack_proportional_to_headroom::Bool`: Default `false`.
- `network_reductions::Vector{PNM.NetworkReduction}`: Default empty.
- `time_steps::Int`: Default `1`.
- `time_step_names::Vector{String}`: Default empty.
- `correct_bustypes::Bool`: Default `false`.
- `control_discrete_devices::Bool`: Whether to run discrete device control via λ-continuation.
    Default `false`.
- `area_interchange_control::Bool`: Not supported on this formulation (Phase 1 is polar-only);
    passing `true` throws `ArgumentError`. Default `false`.
- `solver_settings::Dict{Symbol, Any}`: Default empty.
"""
struct ACMixedPowerFlow{ACSolver <: ACPowerFlowSolverType} <:
       AbstractACPowerFlow{ACSolver}
    check_reactive_power_limits::Bool
    exporter::Union{Nothing, PowerFlowEvaluationModel}
    generator_slack_participation_factors::Union{
        Nothing,
        Dict{Tuple{DataType, String}, Float64},
        Vector{Dict{Tuple{DataType, String}, Float64}},
    }
    enhanced_flat_start::Bool
    log_solver_diagnostics::Bool
    skip_redistribution::Bool
    distribute_slack_proportional_to_headroom::Bool
    network_reductions::Vector{PNM.NetworkReduction}
    time_steps::Int
    time_step_names::Vector{String}
    correct_bustypes::Bool
    control_discrete_devices::Bool
    area_interchange_control::Bool
    interchange_tolerance::Float64
    tie_definition::Symbol
    solver_settings::Dict{Symbol, Any}
end

function ACMixedPowerFlow{ACSolver}(;
    check_reactive_power_limits::Bool = false,
    exporter::Union{Nothing, PowerFlowEvaluationModel} = nothing,
    generator_slack_participation_factors::Union{
        Nothing,
        Dict{Tuple{DataType, String}, Float64},
        Vector{Dict{Tuple{DataType, String}, Float64}},
    } = nothing,
    enhanced_flat_start::Bool = true,
    log_solver_diagnostics::Bool = false,
    skip_redistribution::Bool = false,
    distribute_slack_proportional_to_headroom::Bool = false,
    network_reductions::Vector{PNM.NetworkReduction} = PNM.NetworkReduction[],
    time_steps::Int = 1,
    time_step_names::Vector{String} = String[],
    correct_bustypes::Bool = false,
    control_discrete_devices::Bool = false,
    area_interchange_control::Bool = false,
    interchange_tolerance::Float64 = DEFAULT_INTERCHANGE_TOLERANCE,
    tie_definition::Symbol = :lines_only,
    solver_settings::Dict{Symbol, Any} = Dict{Symbol, Any}(),
) where {ACSolver <: ACPowerFlowSolverType}
    if ACSolver <: Union{
        RobustHomotopyPowerFlow,
        GradientDescentACPowerFlow,
    }
        throw(
            ArgumentError(
                "$(ACSolver) is not supported by ACMixedPowerFlow. " *
                "Robust Homotopy and Gradient Descent do not operate on the " *
                "mixed current-power formulation. Use " *
                "ACMixedPowerFlow{NewtonRaphsonACPowerFlow}, " *
                "{TrustRegionACPowerFlow}, or {LevenbergMarquardtACPowerFlow}, " *
                "or run the solver on ACPolarPowerFlow.",
            ),
        )
    end
    _reject_fd_decoupled_on_nonpolar(ACSolver, "ACMixedPowerFlow")
    _reject_area_interchange_on_nonpolar(area_interchange_control, "ACMixedPowerFlow")
    _validate_slack_distribution_settings(
        distribute_slack_proportional_to_headroom,
        generator_slack_participation_factors,
        time_steps,
    )
    _validate_discrete_control_settings(control_discrete_devices, ACSolver)
    return ACMixedPowerFlow{ACSolver}(
        check_reactive_power_limits,
        exporter,
        generator_slack_participation_factors,
        enhanced_flat_start,
        log_solver_diagnostics,
        skip_redistribution,
        distribute_slack_proportional_to_headroom,
        network_reductions,
        time_steps,
        time_step_names,
        correct_bustypes,
        control_discrete_devices,
        area_interchange_control,
        interchange_tolerance,
        tie_definition,
        solver_settings,
    )
end

ACMixedPowerFlow(; kwargs...) =
    ACMixedPowerFlow{NewtonRaphsonACPowerFlow}(; kwargs...)

"""An abstract supertype for all DC power flow evaluation models.
Subtypes: [`DCPowerFlow`](@ref), [`PTDFDCPowerFlow`](@ref), and [`vPTDFDCPowerFlow`](@ref)."""
abstract type AbstractDCPowerFlow <: PowerFlowEvaluationModel end

# only make sense for AC power flows, but convenient to have for code reuse reasons.
get_slack_participation_factors(pf::AbstractDCPowerFlow) =
    pf.generator_slack_participation_factors
get_distribute_slack_proportional_to_headroom(pf::AbstractDCPowerFlow) =
    pf.distribute_slack_proportional_to_headroom
get_calculate_loss_factors(::AbstractDCPowerFlow) = false
get_calculate_voltage_stability_factors(::AbstractDCPowerFlow) = false

# Getters for fields shared across DC power flow types
# (slightly duplicative: could create common supertype between AC and DC)
get_network_reductions(pf::AbstractDCPowerFlow) = pf.network_reductions
get_time_steps(pf::AbstractDCPowerFlow) = pf.time_steps
get_time_step_names(pf::AbstractDCPowerFlow) = pf.time_step_names
get_correct_bustypes(pf::AbstractDCPowerFlow) = pf.correct_bustypes

# the exporter field is not used in PowerFlows.jl, only in PowerSimulations.jl,
# which calls flatten_power_flow_evaluation_model then evaluates the two sequentially.
"""
    DCPowerFlow(; kwargs...)

An evaluation model for a standard DC power flow.

This provides a fast approximate solution to the AC power flow problem, by solving for the 
bus voltage angles under some simplifying assumptions (lossless lines, constant voltage 
magnitudes, etc.). Branch flows are then calculated from the voltage angles. For details, see 
[Wikipedia](https://en.wikipedia.org/wiki/Power-flow_study#DC_power_flow)
or section 4 of the [MATPOWER docs](https://matpower.org/docs/MATPOWER-manual-4.1.pdf).

# Arguments
- `exporter::Union{Nothing, PowerFlowEvaluationModel}`: An optional exporter for the power flow results.
    If not `nothing`, it should be a [`PSSEExportPowerFlow`](@ref). Default is `nothing`.
- `network_reductions::Vector{PNM.NetworkReduction}`: Network reductions to apply.
    Default is an empty vector. A `PNM.ZeroImpedanceBranchReduction` placed here is routed to
    PNM's dedicated zero-impedance step; set its `resistance_tolerance` to also merge
    near-zero-impedance branches that carry a tiny nonzero resistance (PSS/E-style).
- `time_steps::Int`: Number of time steps to solve. Default is `1`.
- `time_step_names::Vector{String}`: Names for each time step. Default is an empty vector.
- `correct_bustypes::Bool`: Whether to automatically correct bus types based on available generation.
    Default is `false`.
- `lossy_flows::Bool`: Controls how branch flows and losses are computed after solving
    for bus angles. When `true`, flows are computed from the full π-model arc admittance
    matrices (`Y_ft`, `Y_tf`), giving asymmetric `P_from_to` and `P_to_from`; losses are
    then `P_from_to + P_to_from` (exact real-power balance). When `false` (default),
    flows are computed from the lossless `BA·θ` formula (symmetric), and losses are
    approximated as `R·P²`.
- `generator_slack_participation_factors`: An optional parameter that specifies the participation
    factors for generator slack in the power flow solution. Same semantics as [`ACPolarPowerFlow`](@ref).
    Default is `nothing`.
- `distribute_slack_proportional_to_headroom::Bool`: Whether to distribute the slack proportional to
    generator headroom. Default is `false`.
- `skip_redistribution::Bool`: Whether to skip slack redistribution. Default is `false`.
"""
struct DCPowerFlow <: AbstractDCPowerFlow
    exporter::Union{Nothing, PowerFlowEvaluationModel}
    generator_slack_participation_factors::Union{
        Nothing,
        Dict{Tuple{DataType, String}, Float64},
        Vector{Dict{Tuple{DataType, String}, Float64}},
    }
    distribute_slack_proportional_to_headroom::Bool
    skip_redistribution::Bool
    network_reductions::Vector{PNM.NetworkReduction}
    time_steps::Int
    time_step_names::Vector{String}
    correct_bustypes::Bool
    lossy_flows::Bool
end

function DCPowerFlow(;
    exporter::Union{Nothing, PowerFlowEvaluationModel} = nothing,
    generator_slack_participation_factors::Union{
        Nothing,
        Dict{Tuple{DataType, String}, Float64},
        Vector{Dict{Tuple{DataType, String}, Float64}},
    } = nothing,
    distribute_slack_proportional_to_headroom::Bool = false,
    skip_redistribution::Bool = false,
    network_reductions::Vector{PNM.NetworkReduction} = PNM.NetworkReduction[],
    time_steps::Int = 1,
    time_step_names::Vector{String} = String[],
    correct_bustypes::Bool = false,
    lossy_flows::Bool = false,
)
    _validate_slack_distribution_settings(
        distribute_slack_proportional_to_headroom,
        generator_slack_participation_factors,
        time_steps,
    )
    return DCPowerFlow(
        exporter,
        generator_slack_participation_factors,
        distribute_slack_proportional_to_headroom,
        skip_redistribution,
        network_reductions,
        time_steps,
        time_step_names,
        correct_bustypes,
        lossy_flows,
    )
end

"""
    PTDFDCPowerFlow(; kwargs...)

An evaluation model that calculates line flows using the Power Transfer Distribution Factor
Matrix.

This approximates the branch flows in the power grid, under some simplifying
assumptions (lossless lines, constant voltage magnitudes, etc.). In contrast to [`DCPowerFlow`](@ref), 
branch flows are computed directly from bus power injections, without use of the voltage 
angles. See section 4 of the [MATPOWER docs](https://matpower.org/docs/MATPOWER-manual-4.1.pdf) 
for details.

# Arguments
- `exporter::Union{Nothing, PowerFlowEvaluationModel}`: An optional exporter for the power flow results.
    If not `nothing`, it should be a [`PSSEExportPowerFlow`](@ref). Default is `nothing`.
- `calculate_loss_factors::Bool`: Whether to calculate DC loss factors after solving.
    Uses the approximation `∂Loss/∂P = 2 · PTDFᵀ · diag(R) · PTDF · P`. Default is `false`.
- `network_reductions::Vector{PNM.NetworkReduction}`: Network reductions to apply.
    Default is an empty vector. A `PNM.ZeroImpedanceBranchReduction` placed here is routed to
    PNM's dedicated zero-impedance step; set its `resistance_tolerance` to also merge
    near-zero-impedance branches that carry a tiny nonzero resistance (PSS/E-style).
- `time_steps::Int`: Number of time steps to solve. Default is `1`.
- `time_step_names::Vector{String}`: Names for each time step. Default is an empty vector.
- `correct_bustypes::Bool`: Whether to automatically correct bus types based on available generation.
    Default is `false`.
- `generator_slack_participation_factors`: An optional parameter that specifies the participation
    factors for generator slack in the power flow solution. Same semantics as [`ACPolarPowerFlow`](@ref).
    Default is `nothing`.
- `distribute_slack_proportional_to_headroom::Bool`: Whether to distribute the slack proportional to
    generator headroom. Default is `false`.
- `skip_redistribution::Bool`: Whether to skip slack redistribution. Default is `false`.
"""
struct PTDFDCPowerFlow <: AbstractDCPowerFlow
    exporter::Union{Nothing, PowerFlowEvaluationModel}
    calculate_loss_factors::Bool
    generator_slack_participation_factors::Union{
        Nothing,
        Dict{Tuple{DataType, String}, Float64},
        Vector{Dict{Tuple{DataType, String}, Float64}},
    }
    distribute_slack_proportional_to_headroom::Bool
    skip_redistribution::Bool
    network_reductions::Vector{PNM.NetworkReduction}
    time_steps::Int
    time_step_names::Vector{String}
    correct_bustypes::Bool
end

function PTDFDCPowerFlow(;
    exporter::Union{Nothing, PowerFlowEvaluationModel} = nothing,
    calculate_loss_factors::Bool = false,
    generator_slack_participation_factors::Union{
        Nothing,
        Dict{Tuple{DataType, String}, Float64},
        Vector{Dict{Tuple{DataType, String}, Float64}},
    } = nothing,
    distribute_slack_proportional_to_headroom::Bool = false,
    skip_redistribution::Bool = false,
    network_reductions::Vector{PNM.NetworkReduction} = PNM.NetworkReduction[],
    time_steps::Int = 1,
    time_step_names::Vector{String} = String[],
    correct_bustypes::Bool = false,
)
    _validate_slack_distribution_settings(
        distribute_slack_proportional_to_headroom,
        generator_slack_participation_factors,
        time_steps,
    )
    return PTDFDCPowerFlow(
        exporter,
        calculate_loss_factors,
        generator_slack_participation_factors,
        distribute_slack_proportional_to_headroom,
        skip_redistribution,
        network_reductions,
        time_steps,
        time_step_names,
        correct_bustypes,
    )
end

"""
    vPTDFDCPowerFlow(; kwargs...)

An evaluation model that calculates line flows using a virtual Power Transfer Distribution
Factor Matrix.

This is a replacement for the [`PTDFDCPowerFlow`](@ref) for large grids,
where creating and storing the full PTDF matrix would be infeasible or slow. See the
[PowerNetworkMatrices.jl docs](https://sienna-platform.github.io/PowerNetworkMatrices.jl/stable/) for details.

# Arguments
- `exporter::Union{Nothing, PowerFlowEvaluationModel}`: An optional exporter for the power flow results.
    If not `nothing`, it should be a [`PSSEExportPowerFlow`](@ref). Default is `nothing`.
- `calculate_loss_factors::Bool`: Whether to calculate DC loss factors after solving.
    Uses the approximation `∂Loss/∂P = 2 · PTDFᵀ · diag(R) · PTDF · P`. Default is `false`.
- `network_reductions::Vector{PNM.NetworkReduction}`: Network reductions to apply.
    Default is an empty vector. A `PNM.ZeroImpedanceBranchReduction` placed here is routed to
    PNM's dedicated zero-impedance step; set its `resistance_tolerance` to also merge
    near-zero-impedance branches that carry a tiny nonzero resistance (PSS/E-style).
- `time_steps::Int`: Number of time steps to solve. Default is `1`.
- `time_step_names::Vector{String}`: Names for each time step. Default is an empty vector.
- `correct_bustypes::Bool`: Whether to automatically correct bus types based on available generation.
    Default is `false`.
- `generator_slack_participation_factors`: An optional parameter that specifies the participation
    factors for generator slack in the power flow solution. Same semantics as [`ACPolarPowerFlow`](@ref).
    Default is `nothing`.
- `distribute_slack_proportional_to_headroom::Bool`: Whether to distribute the slack proportional to
    generator headroom. Default is `false`.
- `skip_redistribution::Bool`: Whether to skip slack redistribution. Default is `false`.
"""
struct vPTDFDCPowerFlow <: AbstractDCPowerFlow
    exporter::Union{Nothing, PowerFlowEvaluationModel}
    calculate_loss_factors::Bool
    generator_slack_participation_factors::Union{
        Nothing,
        Dict{Tuple{DataType, String}, Float64},
        Vector{Dict{Tuple{DataType, String}, Float64}},
    }
    distribute_slack_proportional_to_headroom::Bool
    skip_redistribution::Bool
    network_reductions::Vector{PNM.NetworkReduction}
    time_steps::Int
    time_step_names::Vector{String}
    correct_bustypes::Bool
end

function vPTDFDCPowerFlow(;
    exporter::Union{Nothing, PowerFlowEvaluationModel} = nothing,
    calculate_loss_factors::Bool = false,
    generator_slack_participation_factors::Union{
        Nothing,
        Dict{Tuple{DataType, String}, Float64},
        Vector{Dict{Tuple{DataType, String}, Float64}},
    } = nothing,
    distribute_slack_proportional_to_headroom::Bool = false,
    skip_redistribution::Bool = false,
    network_reductions::Vector{PNM.NetworkReduction} = PNM.NetworkReduction[],
    time_steps::Int = 1,
    time_step_names::Vector{String} = String[],
    correct_bustypes::Bool = false,
)
    _validate_slack_distribution_settings(
        distribute_slack_proportional_to_headroom,
        generator_slack_participation_factors,
        time_steps,
    )
    return vPTDFDCPowerFlow(
        exporter,
        calculate_loss_factors,
        generator_slack_participation_factors,
        distribute_slack_proportional_to_headroom,
        skip_redistribution,
        network_reductions,
        time_steps,
        time_step_names,
        correct_bustypes,
    )
end

get_calculate_loss_factors(pf::PTDFDCPowerFlow) = pf.calculate_loss_factors
get_calculate_loss_factors(pf::vPTDFDCPowerFlow) = pf.calculate_loss_factors
get_lossy_flows(pf::DCPowerFlow) = pf.lossy_flows

# See also: PSSEExportPowerFlow in psse_export.jl
