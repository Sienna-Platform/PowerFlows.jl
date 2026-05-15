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

"""
    ACPowerFlow{ACSolver}(; kwargs...) where {ACSolver <: ACPowerFlowSolverType}
    ACPowerFlow(; kwargs...)

An evaluation model for a standard
[AC power flow](https://en.wikipedia.org/wiki/Power-flow_study#Power-flow_problem_formulation)
with the specified solver type.

# Arguments
- `ACSolver`: The type of AC power flow solver to use, which must be a subtype of [`ACPowerFlowSolverType`](@ref).
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
    Default is an empty vector.
- `time_steps::Int`: Number of time steps to solve. Default is `1`.
- `time_step_names::Vector{String}`: Names for each time step. Default is an empty vector.
- `correct_bustypes::Bool`: Whether to automatically correct bus types based on available generation.
    Default is `false`.
- `solver_settings::Dict{Symbol, Any}`: Additional keyword arguments to pass to the solver.
    Default is an empty dictionary.
"""
struct ACPolarPowerFlow{ACSolver <: ACPowerFlowSolverType} <: AbstractACPowerFlow{ACSolver}
    check_reactive_power_limits::Bool
    exporter::Union{Nothing, PowerFlowEvaluationModel}
    calculate_loss_factors::Bool
    calculate_voltage_stability_factors::Bool
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
- `ACSolver`: The type of AC power flow solver to use, which must be a subtype of [`ACPowerFlowSolverType`](@ref).
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
    solver_settings::Dict{Symbol, Any} = Dict{Symbol, Any}(),
) where {ACSolver <: ACPowerFlowSolverType}
    if calculate_loss_factors && ACSolver == LevenbergMarquardtACPowerFlow
        error("Loss factor calculation is not supported by the Levenberg-Marquardt solver.")
    end
    _validate_slack_distribution_settings(
        distribute_slack_proportional_to_headroom,
        generator_slack_participation_factors,
        time_steps,
    )
    return ACPolarPowerFlow{ACSolver}(
        check_reactive_power_limits,
        exporter,
        calculate_loss_factors,
        calculate_voltage_stability_factors,
        generator_slack_participation_factors,
        enhanced_flat_start,
        robust_power_flow,
        skip_redistribution,
        distribute_slack_proportional_to_headroom,
        network_reductions,
        time_steps,
        time_step_names,
        correct_bustypes,
        solver_settings,
    )
end

# Default constructor: ACPolarPowerFlow() defaults to NewtonRaphsonACPowerFlow solver
ACPolarPowerFlow(; kwargs...) = ACPolarPowerFlow{NewtonRaphsonACPowerFlow}(; kwargs...)

"""`ACPowerFlow` is the deprecated former name of [`ACPolarPowerFlow`](@ref).
It remains as an alias for backward compatibility (PowerSimulations.jl and
external callers) and will be removed in a future breaking release."""
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

"""
    ACRectangularPowerFlow{ACSolver}(; kwargs...) where {ACSolver <: ACPowerFlowSolverType}
    ACRectangularPowerFlow(; kwargs...)

An evaluation model for the AC power flow solved with the augmented
current-injection (Da Costa) formulation in rectangular coordinates.

State per bus: PQ `(eᵢ, fᵢ)`, PV `(eᵢ, fᵢ, Qᵢ)`, REF `(P_genᵢ, Q_genᵢ)` with
`(eᵢ, fᵢ)` fixed. Residual is the complex current mismatch
`ΔIᵢ = I_specᵢ − Y_bus·V`. Off-diagonal Jacobian blocks ≡ Y_bus 2×2 real blocks
and are constant across iterations.

`ACSolver` defaults to [`NewtonRaphsonACPowerFlow`](@ref); only
[`NewtonRaphsonACPowerFlow`](@ref) and [`TrustRegionACPowerFlow`](@ref) are
supported. Levenberg-Marquardt, Robust Homotopy, and Gradient Descent operate
on the polar formulation only and are rejected at construction.

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
    skip_redistribution::Bool
    distribute_slack_proportional_to_headroom::Bool
    network_reductions::Vector{PNM.NetworkReduction}
    time_steps::Int
    time_step_names::Vector{String}
    correct_bustypes::Bool
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
    skip_redistribution::Bool = false,
    distribute_slack_proportional_to_headroom::Bool = false,
    network_reductions::Vector{PNM.NetworkReduction} = PNM.NetworkReduction[],
    time_steps::Int = 1,
    time_step_names::Vector{String} = String[],
    correct_bustypes::Bool = false,
    solver_settings::Dict{Symbol, Any} = Dict{Symbol, Any}(),
) where {ACSolver <: ACPowerFlowSolverType}
    if ACSolver <: Union{
        LevenbergMarquardtACPowerFlow,
        RobustHomotopyPowerFlow,
        GradientDescentACPowerFlow,
    }
        throw(
            ArgumentError(
                "$(ACSolver) is not supported by ACRectangularPowerFlow. " *
                "Levenberg-Marquardt, Robust Homotopy, and Gradient Descent " *
                "operate on the polar formulation only. Use " *
                "ACRectangularPowerFlow{NewtonRaphsonACPowerFlow} or " *
                "{TrustRegionACPowerFlow}, or run the solver on " *
                "ACPolarPowerFlow. (Rectangular LM is tracked as a separate " *
                "follow-up project.)",
            ),
        )
    end
    _validate_slack_distribution_settings(
        distribute_slack_proportional_to_headroom,
        generator_slack_participation_factors,
        time_steps,
    )
    return ACRectangularPowerFlow{ACSolver}(
        check_reactive_power_limits,
        exporter,
        generator_slack_participation_factors,
        enhanced_flat_start,
        skip_redistribution,
        distribute_slack_proportional_to_headroom,
        network_reductions,
        time_steps,
        time_step_names,
        correct_bustypes,
        solver_settings,
    )
end

ACRectangularPowerFlow(; kwargs...) =
    ACRectangularPowerFlow{NewtonRaphsonACPowerFlow}(; kwargs...)

"""An abstract supertype for all DC power flow evaluation models.
Subtypes: [`DCPowerFlow`](@ref), [`PTDFDCPowerFlow`](@ref), and [`vPTDFDCPowerFlow`](@ref)."""
abstract type AbstractDCPowerFlow <: PowerFlowEvaluationModel end

# only make sense for AC power flows, but convenient to have for code reuse reasons.
get_slack_participation_factors(::AbstractDCPowerFlow) = nothing
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
    Default is an empty vector.
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
"""
@kwdef struct DCPowerFlow <: AbstractDCPowerFlow
    exporter::Union{Nothing, PowerFlowEvaluationModel} = nothing
    network_reductions::Vector{PNM.NetworkReduction} = PNM.NetworkReduction[]
    time_steps::Int = 1
    time_step_names::Vector{String} = String[]
    correct_bustypes::Bool = false
    lossy_flows::Bool = false
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
    Default is an empty vector.
- `time_steps::Int`: Number of time steps to solve. Default is `1`.
- `time_step_names::Vector{String}`: Names for each time step. Default is an empty vector.
- `correct_bustypes::Bool`: Whether to automatically correct bus types based on available generation.
    Default is `false`.
"""
@kwdef struct PTDFDCPowerFlow <: AbstractDCPowerFlow
    exporter::Union{Nothing, PowerFlowEvaluationModel} = nothing
    calculate_loss_factors::Bool = false
    network_reductions::Vector{PNM.NetworkReduction} = PNM.NetworkReduction[]
    time_steps::Int = 1
    time_step_names::Vector{String} = String[]
    correct_bustypes::Bool = false
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
    Default is an empty vector.
- `time_steps::Int`: Number of time steps to solve. Default is `1`.
- `time_step_names::Vector{String}`: Names for each time step. Default is an empty vector.
- `correct_bustypes::Bool`: Whether to automatically correct bus types based on available generation.
    Default is `false`.
"""
@kwdef struct vPTDFDCPowerFlow <: AbstractDCPowerFlow
    exporter::Union{Nothing, PowerFlowEvaluationModel} = nothing
    calculate_loss_factors::Bool = false
    network_reductions::Vector{PNM.NetworkReduction} = PNM.NetworkReduction[]
    time_steps::Int = 1
    time_step_names::Vector{String} = String[]
    correct_bustypes::Bool = false
end

get_calculate_loss_factors(pf::PTDFDCPowerFlow) = pf.calculate_loss_factors
get_calculate_loss_factors(pf::vPTDFDCPowerFlow) = pf.calculate_loss_factors
get_lossy_flows(pf::DCPowerFlow) = pf.lossy_flows

# See also: PSSEExportPowerFlow in psse_export.jl
