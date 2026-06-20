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

# DC/PTDF persistent cache stored in `data.solver_cache[]`. The factored network matrix `matrix`
# and the `backend` form the invalidation key (rebuild when either changes); `cache` is the actual
# factorization and `scratch` the per-solve buffers. Subtypes `SolverCache` so it is type-disjoint
# from the AC `FastDecoupledCache` that shares the same slot (no sentinel tag needed).
struct DCSolverCache{M, B, C, S} <: SolverCache
    matrix::M
    backend::B
    cache::C
    scratch::S
end

# Reuse on a matching key, else `nothing` to signal a rebuild. Dispatch on the cached entry's type
# rather than an `isa`/sentinel check: an empty slot returns `nothing`; a stray non-DC `SolverCache`
# (cross-use with the AC path, impossible today since the data types are disjoint) is a loud
# `MethodError` instead of a silent mis-read.
_reuse_dc_cache(::Nothing, M, backend) = nothing
_reuse_dc_cache(e::DCSolverCache, M, backend) =
    if (e.matrix === M && typeof(e.backend) === typeof(backend))
        (e.cache, e.scratch)
    else
        nothing
    end

# Reuse a cached factorization of `M` while the matrix object and backend are unchanged;
# rebuild otherwise. Assumes the network matrix is not mutated in place.
function _get_or_build_solver_cache!(
    data::PowerFlowData,
    backend,
    M::SparseMatrixCSC{Float64},
)
    reused = _reuse_dc_cache(data.solver_cache[], M, backend)
    reused === nothing || return reused
    cache = make_linear_solver_cache(backend, M)
    full_factor!(cache, M)
    scratch = _make_dc_scratch(data)
    data.solver_cache[] = DCSolverCache(M, backend, cache, scratch)
    return cache, scratch
end

# Per-solve scratch buffers + network-fixed precomputes; built once with the cache. The signed
# arc-bus incidence is built once at `PowerFlowData` construction via `PNM.IncidenceMatrix` (see
# `_signed_arc_bus_incidence`) and reused here.
function _make_dc_scratch(data::PowerFlowData)
    valid_ix = get_valid_ix(data)
    # InvertedIndex has no `length`; size via a view.
    p_inj_dims = size(view(data.bus_active_power_injections, valid_ix, :))
    return (
        power_injections = similar(data.bus_active_power_injections),
        p_inj = Matrix{Float64}(undef, p_inj_dims),
        rs = _get_arc_resistances(data),
        arc_bus_incidence = data.arc_bus_incidence,
    )
end

_convert_to_range(ix::Integer) = ix:ix
_convert_to_range(::Colon) = Colon()

# PERF: cache ref-to-bus-indices dict.
"""
    _shift_angles_to_stored_reference!(data, time_steps = :)

The DC solve computes angles relative to 0 at each subnetwork's ref bus; if the
subnetwork's ref bus has nonzero angle, shift the angles accordingly.
"""
function _shift_angles_to_stored_reference!(
    data::Union{PTDFPowerFlowData, vPTDFPowerFlowData, ABAPowerFlowData, ACPowerFlowData},
    time_steps::Union{Colon, Integer} = Colon(),
)
    # normalize a scalar index to a range: in-place broadcasting cannot assign
    # through the zero-dimensional view that scalar indexing produces.
    timestep_range = _convert_to_range(time_steps)
    bus_lookup = get_bus_lookup(data)
    for (ref_bus, ax) in subnetwork_axes(data)
        ref_row = bus_lookup[ref_bus]
        θ_ref = @view data.bus_angles[ref_row, timestep_range]
        all(iszero, θ_ref) && continue
        for bus in first(ax)
            bus == ref_bus && continue
            @views data.bus_angles[bus_lookup[bus], timestep_range] .+= θ_ref
        end
    end
end

"""
    solve_power_flow!(data::PTDFPowerFlowData)

Evaluates the PTDF power flow and writes the result to the fields of the
[`PTDFPowerFlowData`](@ref) structure (a type alias of [`PowerFlowData`](@ref)).

This function modifies the following fields of `data`, setting them to the computed values:
- `data.bus_angles`: the bus angles for each bus in the system.
- `data.branch_active_power_flow_from_to`: the active power flow from the "from" bus to the "to" bus of each branch
- `data.branch_active_power_flow_to_from`: the active power flow from the "to" bus to the "from" bus of each branch

Additionally, it sets `data.converged` to `true`, indicating that the power flow calculation was successful.
"""
function solve_power_flow!(
    data::PTDFPowerFlowData;
    linear_solver::Union{Nothing, AbstractString} = nothing,
)
    backend = resolve_linear_solver_backend(linear_solver)
    solver_cache, scratch =
        _get_or_build_solver_cache!(data, backend, data.aux_network_matrix.data)
    power_injections = scratch.power_injections
    @. power_injections =
        data.bus_active_power_injections - data.bus_active_power_withdrawals
    power_injections .+= data.bus_hvdc_net_power
    mul!(
        data.arc_active_power_flow_from_to,
        transpose(data.power_network_matrix.data),
        power_injections,
    )
    @. data.arc_active_power_flow_to_from = -data.arc_active_power_flow_from_to
    # HVDC flows stored separately and already calculated: see initialize_power_flow_data!
    valid_ix = get_valid_ix(data)
    p_inj = scratch.p_inj
    @views p_inj .= power_injections[valid_ix, :]
    solve!(solver_cache, p_inj)
    @views data.bus_angles[valid_ix, :] .= p_inj
    _shift_angles_to_stored_reference!(data)
    mul!(data.arc_angle_differences, scratch.arc_bus_incidence, data.bus_angles)
    @. data.arc_active_power_losses = scratch.rs * data.arc_active_power_flow_from_to^2
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
    data::vPTDFPowerFlowData;
    linear_solver::Union{Nothing, AbstractString} = nothing,
)
    backend = resolve_linear_solver_backend(linear_solver)
    solver_cache, _ =
        _get_or_build_solver_cache!(data, backend, data.aux_network_matrix.data)
    power_injections =
        @. data.bus_active_power_injections - data.bus_active_power_withdrawals
    power_injections .+= data.bus_hvdc_net_power
    data.arc_active_power_flow_from_to .=
        my_mul_mt(data.power_network_matrix, power_injections)
    @. data.arc_active_power_flow_to_from = -data.arc_active_power_flow_from_to
    # HVDC flows stored separately and already calculated: see initialize_power_flow_data!
    valid_ix = get_valid_ix(data)
    p_inj = power_injections[valid_ix, :]
    solve!(solver_cache, p_inj)
    data.bus_angles[valid_ix, :] .= p_inj
    _shift_angles_to_stored_reference!(data)
    _compute_arc_angle_differences_from_data!(data)
    Rs = _get_arc_resistances(data)
    @. data.arc_active_power_losses = Rs * data.arc_active_power_flow_from_to^2
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
    data::ABAPowerFlowData;
    linear_solver::Union{Nothing, AbstractString} = nothing,
)
    backend = resolve_linear_solver_backend(linear_solver)
    solver_cache, scratch =
        _get_or_build_solver_cache!(data, backend, data.power_network_matrix.data)

    # Reuse preallocated buffers from the cache scratch so a PCM-loop solve allocates
    # nothing on the common (lossless) DC path beyond the bus-angle writeback view.
    power_injections = scratch.power_injections
    @. power_injections =
        data.bus_active_power_injections - data.bus_active_power_withdrawals
    power_injections .+= data.bus_hvdc_net_power
    valid_ix = get_valid_ix(data)
    p_inj = scratch.p_inj
    @views p_inj .= power_injections[valid_ix, :]
    solve!(solver_cache, p_inj)
    @views data.bus_angles[valid_ix, :] .= p_inj
    _shift_angles_to_stored_reference!(data)

    if data.arc_lossy_admittance_from_to !== nothing
        # DC assumption: all bus voltage magnitudes are 1.0 p.u., so V = e^(jθ).
        V = @. exp(1im * data.bus_angles)
        arcs = get_arc_axis(data)
        bus_lookup = get_bus_lookup(data)
        fb_ix = [bus_lookup[first(arc)] for arc in arcs]
        tb_ix = [bus_lookup[last(arc)] for arc in arcs]
        # Explicit dots (not `@.`) because the RHS embeds a matrix-vector product
        # (`admittance * V`), which `@.` would wrongly broadcast as element-wise.
        Sft = V[fb_ix, :] .* conj.(data.arc_lossy_admittance_from_to * V)
        Stf = V[tb_ix, :] .* conj.(data.arc_lossy_admittance_to_from * V)
        @. data.arc_active_power_flow_from_to = real(Sft)
        @. data.arc_active_power_flow_to_from = real(Stf)
        # True losses come directly from the admittance calculation.
        @. data.arc_active_power_losses =
            data.arc_active_power_flow_from_to + data.arc_active_power_flow_to_from
    else
        mul!(
            data.arc_active_power_flow_from_to,
            transpose(data.aux_network_matrix.data),
            data.bus_angles,
        )
        @. data.arc_active_power_flow_to_from = -data.arc_active_power_flow_from_to
        @. data.arc_active_power_losses =
            scratch.rs * data.arc_active_power_flow_from_to^2
    end
    # Δθ = A·θ as a single sparse SpMV using the cached signed incidence — replaces
    # the per-call rebuild of fb_ix/tb_ix index vectors in
    # `_compute_arc_angle_differences_from_data!` (~0.38 MB/call).
    mul!(data.arc_angle_differences, scratch.arc_bus_incidence, data.bus_angles)
    data.converged .= true
    return
end

# SINGLE PERIOD ##############################################################

"""
    solve_power_flow(
        pf::T,
        sys::PSY.System,
        flow_reporting::FlowReporting = FlowReporting.ARC_FLOWS,
    ) where T <: AbstractDCPowerFlow


Evaluates the provided DC power flow method `pf` on the [PowerSystems.System](@extref) `sys`,
returning a dictionary of `DataFrame`s containing the calculated flows and bus angles.
The `flow_reporting` input determines if flows are reported for arcs (`FlowReporting.ARC_FLOWS`,
the default) or for branches (`FlowReporting.BRANCH_FLOWS`).

Configuration options like `time_steps`, `time_step_names`, `network_reductions`, and
`correct_bustypes` should be set on the power flow object (e.g., `DCPowerFlow(; time_steps=2)`).

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
    flow_reporting::FlowReporting = FlowReporting.ARC_FLOWS;
    linear_solver::Union{Nothing, AbstractString} = nothing,
) where {T <: AbstractDCPowerFlow}
    with_units_base(sys, PSY.UnitSystem.SYSTEM_BASE) do
        data = PowerFlowData(pf, sys)
        solve_power_flow!(data; linear_solver)
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

Note that `data` must have been created from the [`PowerSystems.System`](@extref)
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
    linear_solver::Union{Nothing, AbstractString} = nothing,
)
    solve_power_flow!(data; linear_solver)
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
