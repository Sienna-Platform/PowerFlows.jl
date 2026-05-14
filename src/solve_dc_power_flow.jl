"""
Adjust the power injections vector to account for the power flows through LCCs.

Relies on the fact that we calculate those flows during initialization and save them
to the `active_power_flow_from_to` and `active_power_flow_to_from` fields of the
`LCCParameters` struct.
"""
function adjust_power_injections_for_lccs!(power_injections::Matrix{Float64},
    lcc_params::LCCParameters,
)
    for (i, bus_inds) in enumerate(lcc_params.bus_indices)
        from_bus_ix, to_bus_ix = bus_inds
        rectifier_power = lcc_params.arc_active_power_flow_from_to[i]
        # inverter_power here takes into account losses.
        inverter_power = lcc_params.arc_active_power_flow_to_from[i]
        power_injections[from_bus_ix, :] .-= rectifier_power
        power_injections[to_bus_ix, :] .+= inverter_power
    end
    return
end

"""
    solve_power_flow!(data::PTDFPowerFlowData)
Evaluates the PTDF power flow and writes the result to the fields of the
[`PTDFPowerFlowData`](@ref) structure.

This function modifies the following fields of `data`, setting them to the computed values:
- `data.bus_angles`: the bus angles for each bus in the system.
- `data.branch_active_power_flow_from_to`: the active power flow from the "from" bus to the "to" bus of each branch
- `data.branch_active_power_flow_to_from`: the active power flow from the "to" bus to the "from" bus of each branch

Additionally, it sets `data.converged` to `true`, indicating that the power flow calculation was successful.
"""
function solve_power_flow!(
    data::PTDFPowerFlowData,
)
    solver_cache = KLULinSolveCache(data.aux_network_matrix.data)
    full_factor!(solver_cache, data.aux_network_matrix.data)
    # get net power injections
    power_injections = data.bus_active_power_injections .- data.bus_active_power_withdrawals
    power_injections .+= data.bus_hvdc_net_power
    # evaluate flows
    data.arc_active_power_flow_from_to .=
        data.power_network_matrix.data' * power_injections
    data.arc_active_power_flow_to_from .= -data.arc_active_power_flow_from_to
    # HVDC flows stored separately and already calculated: see initialize_power_flow_data!
    valid_ix = get_valid_ix(data)
    p_inj = power_injections[valid_ix, :]
    solve!(solver_cache, p_inj)
    data.bus_angles[valid_ix, :] .= p_inj
    _compute_arc_angle_differences_from_data!(data)
    Rs = _get_arc_resistances(data)
    data.arc_active_power_losses .= Rs .* data.arc_active_power_flow_from_to .^ 2
    data.converged .= true
    if get_calculate_loss_factors(data)
        data.loss_factors .= dc_loss_factors(data, power_injections)
    end
    return
end

"""
    solve_power_flow!(data::vPTDFPowerFlowData)

Evaluates the virtual PTDF power flow and writes the results to the fields
of the [`vPTDFPowerFlowData`](@ref) structure.


This function modifies the following fields of `data`, setting them to the computed values:
- `data.bus_angles`: the bus angles for each bus in the system.
- `data.branch_active_power_flow_from_to`: the active power flow from the "from" bus to the "to" bus of each branch
- `data.branch_active_power_flow_to_from`: the active power flow from the "to" bus to the "from" bus of each branch

Additionally, it sets `data.converged` to `true`, indicating that the power flow calculation was successful.
"""
function solve_power_flow!(
    data::vPTDFPowerFlowData,
)
    solver_cache = KLULinSolveCache(data.aux_network_matrix.data)
    full_factor!(solver_cache, data.aux_network_matrix.data)
    power_injections = data.bus_active_power_injections .- data.bus_active_power_withdrawals
    power_injections .+= data.bus_hvdc_net_power
    data.arc_active_power_flow_from_to .=
        my_mul_mt(data.power_network_matrix, power_injections)
    data.arc_active_power_flow_to_from .= -data.arc_active_power_flow_from_to
    # HVDC flows stored separately and already calculated: see initialize_power_flow_data!
    valid_ix = get_valid_ix(data)
    p_inj = power_injections[valid_ix, :]
    solve!(solver_cache, p_inj)
    data.bus_angles[valid_ix, :] .= p_inj
    _compute_arc_angle_differences_from_data!(data)
    Rs = _get_arc_resistances(data)
    data.arc_active_power_losses .= Rs .* data.arc_active_power_flow_from_to .^ 2
    data.converged .= true
    if get_calculate_loss_factors(data)
        data.loss_factors .= dc_loss_factors(data, power_injections)
    end
    return
end

# TODO: solve just for some lines with vPTDF

"""
    solve_power_flow!(data::ABAPowerFlowData)

Evaluates the DC power flow and writes the results (branch flows) to the fields
of the [`ABAPowerFlowData`](@ref) structure.


This function modifies the following fields of `data`, setting them to the computed values:
- `data.bus_angles`: the bus angles for each bus in the system.
- `data.branch_active_power_flow_from_to`: the active power flow from the "from" bus to the "to" bus of each branch
- `data.branch_active_power_flow_to_from`: the active power flow from the "to" bus to the "from" bus of each branch

Additionally, it sets `data.converged` to `true`, indicating that the power flow calculation was successful.

# Loss estimation

Losses are estimated differently depending on whether lossy flows are enabled
(`DCPowerFlow(; lossy_flows = true)`):

- **Lossless (default):** Flows are estimated from `BA' * θ` and losses are approximated as
  `Rₖ * Pₖ²` (the classical DC loss approximation).
- **Lossy:** Flows are computed from the arc admittance matrices to match PSS/e's DCPF
  formulation: `Sft = V_f * conj(Y_ft * V)`, `Stf = V_t * conj(Y_tf * V)`. Losses are
  then `Pft + Ptf` (the exact real-power balance across each arc).
"""
# DC flow: ABA and BA case
function solve_power_flow!(
    data::ABAPowerFlowData,
)
    solver_cache = KLULinSolveCache(data.power_network_matrix.data)
    full_factor!(solver_cache, data.power_network_matrix.data)

    power_injections = data.bus_active_power_injections - data.bus_active_power_withdrawals
    power_injections .+= data.bus_hvdc_net_power
    valid_ix = get_valid_ix(data)
    p_inj = power_injections[valid_ix, :]
    solve!(solver_cache, p_inj)
    data.bus_angles[valid_ix, :] .= p_inj

    if data.arc_lossy_admittance_from_to !== nothing
        # DC assumption: all bus voltage magnitudes are 1.0 p.u., so V = e^(jθ).
        V = exp.(1im .* data.bus_angles)
        arcs = get_arc_axis(data)
        bus_lookup = get_bus_lookup(data)
        fb_ix = [bus_lookup[first(arc)] for arc in arcs]
        tb_ix = [bus_lookup[last(arc)] for arc in arcs]
        Sft = V[fb_ix, :] .* conj.(data.arc_lossy_admittance_from_to * V)
        Stf = V[tb_ix, :] .* conj.(data.arc_lossy_admittance_to_from * V)
        data.arc_active_power_flow_from_to .= real.(Sft)
        data.arc_active_power_flow_to_from .= real.(Stf)
        # True losses come directly from the admittance calculation.
        data.arc_active_power_losses .=
            data.arc_active_power_flow_from_to .+ data.arc_active_power_flow_to_from
    else
        data.arc_active_power_flow_from_to .=
            data.aux_network_matrix.data' * data.bus_angles
        data.arc_active_power_flow_to_from .= -data.arc_active_power_flow_from_to
        Rs = _get_arc_resistances(data)
        data.arc_active_power_losses .= Rs .* data.arc_active_power_flow_from_to .^ 2
    end
    _compute_arc_angle_differences_from_data!(data)
    data.converged .= true
    return
end

# SINGLE PERIOD ##############################################################

"""
    solve_power_flow(
        pf::T,
        sys::PSY.System,
        flow_reporting::FlowReporting
    ) where T <: AbstractDCPowerFlow


Evaluates the provided DC power flow method `pf` on the [PowerSystems.System](@extref) `sys`,
returning a dictionary of `DataFrame`s containing the calculated flows and bus angles.
The `flow_reporting` input determines if flows are reported for arcs (`FlowReporting.ARC_FLOWS`)
or for branches (`FlowReporting.BRANCH_FLOWS`).

Configuration options like `time_steps`, `time_step_names`, `network_reductions`, and
`correct_bustypes` should be set on the power flow object (e.g., `DCPowerFlow(; time_steps=2)`).

Provided for convenience: this interface bypasses the need to create a [`PowerFlowData`](@ref)
struct, but that's still what's happening under the hood.

# Example
```julia
using PowerFlows, PowerSystemCaseBuilder
sys = build_system(PSITestSystems, "c_sys5")
d = solve_power_flow(DCPowerFlow(), sys, FlowReporting.ARC_FLOWS)
display(d["1"]["flow_results"])
display(d["1"]["bus_results"])
```
"""
function solve_power_flow(
    pf::T,
    sys::PSY.System,
    flow_reporting::FlowReporting,
) where {T <: AbstractDCPowerFlow}
    with_units_base(sys, PSY.UnitSystem.SYSTEM_BASE) do
        data = PowerFlowData(pf, sys)
        solve_power_flow!(data)
        return write_results(data, sys, flow_reporting)
    end
end

# MULTI PERIOD ###############################################################

"""
    solve_power_flow(
        data::Union{PTDFPowerFlowData, vPTDFPowerFlowData, ABAPowerFlowData},
        sys::PSY.System,
        flow_reporting::FlowReporting,
    )

Evaluates the power flows on the system's branches by means of the method associated with
the `PowerFlowData` structure `data`, which can be one of `PTDFPowerFlowData`,
`vPTDFPowerFlowData`, or `ABAPowerFlowData`.
Returns a dictionary of `DataFrame`s, each containing the flows and bus voltages for
the input `PSY.System` at that time step.
The `flow_reporting` argument determines if flows are reported for arcs (`FlowReporting.ARC_FLOWS`)
or for branches (`FlowReporting.BRANCH_FLOWS`).

# Arguments:
- `data::Union{PTDFPowerFlowData, vPTDFPowerFlowData, ABAPowerFlowData}`:
        `PowerFlowData` structure containing the system's data per each time_step
        considered, as well as the associated matrix for the power flow.
- `sys::PSY.System`:
        container gathering the system data.
- `flow_reporting::FlowReporting`:
        Format for reporting flows

Note that `data` must have been created from the [System](@extref PowerSystems.System)
`sys` using one of the [`PowerFlowData`](@ref) constructors.

# Example
```julia
using PowerFlows, PowerSystemCaseBuilder
sys = build_system(PSITestSystems, "c_sys14")
data = PowerFlowData(PTDFDCPowerFlow(; time_steps = 2), sys)
d = solve_power_flow(data, sys, FlowReporting.ARC_FLOWS)
display(d["2"]["flow_results"])
```
"""
function solve_power_flow(
    data::Union{PTDFPowerFlowData, vPTDFPowerFlowData, ABAPowerFlowData},
    sys::PSY.System,
    flow_reporting::FlowReporting;
)
    solve_power_flow!(data)
    return write_results(data, sys, flow_reporting)
end

"""
    _get_arc_resistances(data::Union{PTDFPowerFlowData, vPTDFPowerFlowData, ABAPowerFlowData}) -> Vector{Float64}

Look up the equivalent resistance of each arc from the network reduction data.
Delegates to [`_get_arc_branch_params`](@ref) and returns only the resistance vector.
"""
function _get_arc_resistances(
    data::Union{PTDFPowerFlowData, vPTDFPowerFlowData, ABAPowerFlowData},
)
    rs, _, _, _ = _get_arc_branch_params(data)
    return rs
end

"""
    dc_loss_factors(
        data::Union{PTDFPowerFlowData, vPTDFPowerFlowData},
        P::Matrix{Float64},
    ) -> Matrix{Float64}

Compute the gradient of total system active power losses with respect to
bus injections using the DC power flow approximation:

    ∂Loss/∂P = 2 · PTDFᵀ · diag(R) · PTDF · P

This is equivalent to the per-element form:

    ∂Loss/∂Pᵢ = Σₖ 2·Rₖ·PTDFₖᵢ·Σⱼ PTDFₖⱼ·Pⱼ

# Arguments
- `data::Union{PTDFPowerFlowData, vPTDFPowerFlowData}`: solved power flow data containing
  the PTDF matrix and network reduction data for looking up branch resistances.
- `P::Matrix{Float64}`: bus injection matrix of size `(num_buses, num_timesteps)`.

# Returns
- `Matrix{Float64}`: loss factor matrix of size `(num_buses, num_timesteps)`, where each
  entry `[i, t]` is the marginal change in total system losses per unit injection at bus `i`
  in time step `t`.
"""
function dc_loss_factors(
    data::PTDFPowerFlowData,
    P::Matrix{Float64},
)
    Rs = _get_arc_resistances(data)
    ptdf_t = data.power_network_matrix.data
    # PERF could be optimized: remove the Diagonal call.
    return 2 * ptdf_t * LinearAlgebra.Diagonal(Rs) * ptdf_t' * P
end

function dc_loss_factors(
    data::vPTDFPowerFlowData,
    P::Matrix{Float64},
)
    Rs = _get_arc_resistances(data)
    ptdf = data.power_network_matrix
    arc_ax = get_arc_axis(data)
    n_buses = size(P, 1)
    n_ts = size(P, 2)
    result = zeros(n_buses, n_ts)
    flows_k = Vector{Float64}(undef, n_ts)
    # Single pass: fetch each PTDF row once, compute flows vectorized, then accumulate.
    for (k, arc) in enumerate(arc_ax)
        row_k = ptdf[arc, :]
        r_k = Rs[k]
        mul!(flows_k, P', row_k)
        for t in 1:n_ts
            @inbounds w = 2.0 * r_k * flows_k[t]
            @inbounds @simd for j in 1:n_buses
                result[j, t] += row_k[j] * w
            end
        end
    end
    return result
end
