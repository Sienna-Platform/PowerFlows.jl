abstract type PowerFlowContainer end

"""
Trait signifying whether the `PowerFlowContainer` can represent multi-period data. Must be
implemented for all concrete subtypes.
"""
supports_multi_period(x::PowerFlowContainer) =
    throw(
        IS.NotImplementedError(
            "supports_multi_period must be implemented for $(typeof(x))"),
    )

"A `PowerFlowContainer` that represents its data as a `PSY.System`."
abstract type SystemPowerFlowContainer <: PowerFlowContainer end

get_system(container::SystemPowerFlowContainer) = container.system

"""
    PowerFlowData{M <: PNM.PowerNetworkMatrix, N <: Union{PNM.PowerNetworkMatrix, Nothing}}

Structure containing all the data required for the evaluation of the power
flows and angles, as well as these ones.

All fields starting with `bus_` are ordered according to `bus_lookup`, and all fields 
starting with `arc_` are ordered according to `arc_lookup`: one row per bus/arc, 
one column per time period. Here, buses should be understood as \"buses remaining, after 
the network reduction.\" Similarly, we use \"arcs\" instead of \"branches\" to distinguish 
between network elements (post-reduction) and system objects (pre-reduction).

Generally, do not construct this directly. Instead, use one of the later constructors to 
pass in a [`PowerFlowEvaluationModel`](@ref) and a [`PowerSystems.System`](@extref). 
`aux\\_network\\_matrix` and `power\\_network\\_matrix` will then be set to the appropriate 
matrices that are needed for computing that type of power flow. See also [`ACPowerFlowData`](@ref),
[`ABAPowerFlowData`](@ref), [`PTDFPowerFlowData`](@ref), and [`vPTDFPowerFlowData`](@ref): 
these are all aliases for [`PowerFlowData`](@ref)`{N, M}` with specific `N`,`M`, that are used for 
the respective type of power flow evaluations.

# Fields:
- `bus_active_power_injections::Matrix{Float64}`:
        matrix containing the bus active power injections.
- `bus_reactive_power_injections::Matrix{Float64}`:
        matrix containing the bus reactive power injections.
- `bus_active_power_withdrawals::Matrix{Float64}`:
        matrix containing the bus reactive power withdrawals.
- `bus_reactive_power_withdrawals::Matrix{Float64}`:
        matrix containing the bus reactive power withdrawals.
- `bus_active_power_constant_current_withdrawals::Matrix{Float64}`:
        matrix containing the bus active power constant current
        withdrawals.
- `bus_reactive_power_constant_current_withdrawals::Matrix{Float64}`:
        matrix containing the bus reactive power constant current
        withdrawals.
- `bus_active_power_constant_impedance_withdrawals::Matrix{Float64}`:
        matrix containing the bus active power constant impedance
        withdrawals.
- `bus_reactive_power_constant_impedance_withdrawals::Matrix{Float64}`:  
        matrix containing the bus reactive power constant impedance
        withdrawals.
- `bus_reactive_power_bounds::Matrix{Float64}`:
        matrix containing upper and lower bounds for the reactive supply at each
        bus at each time period.
- `bus_type::Matrix{PSY.ACBusTypes}`:
        matrix containing type of buses present in the system.
- `bus_magnitude::Matrix{Float64}`:
        matrix containing the bus voltage magnitudes.
- `bus_angles::Matrix{Float64}`:
        matrix containing the bus voltage angles.
- `arc_active_power_flow_from_to::Matrix{Float64}`:
        matrix containing the active power flows measured at the `from` bus.
- `arc_reactive_power_flow_from_to::Matrix{Float64}`:
        matrix containing the reactive power flows measured at the `from` bus.
- `arc_active_power_flow_to_from::Matrix{Float64}`:
        matrix containing the active power flows measured at the `to` bus.
- `arc_reactive_power_flow_to_from::Matrix{Float64}`:
        matrix containing the reactive power flows measured at the `to` bus.
- `arc_angle_differences::Matrix{Float64}`:
        matrix containing the voltage angle difference (θ_from − θ_to) across each arc.
- `generic_hvdc_flows::Dict{Tuple{Int, Int}, Tuple{Float64, Float64}}`:
        dictionary mapping each generic HVDC line (represented as a tuple of the from and to bus
        numbers) to a tuple of `(P_from_to, P_to_from)` active power flows.
- `bus_hvdc_net_power::Matrix{Float64}`:
        "(b, t)" matrix containing the net power injections from all HVDC lines at each bus.
        b: number of buses, t: number of time period. Only contains HVDCs handled as
        separate injection/withdrawal pairs: LCCs and generic for DC, or just generic for AC.
- `time_step_map::Dict{Int, S}`:
        dictionary mapping the number of the time periods (corresponding to the
        column number of the previously mentioned matrices) and their names.
- `power_network_matrix::M`:
        matrix used for the evaluation of either the power flows or bus angles,
        depending on the method considered.
- `aux_network_matrix::N`:
        matrix used for the evaluation of either the power flows or bus angles,
        depending on the method considered.
- `neighbors::Vector{Set{Int}}`: Vector with the sets of adjacent buses.
"""
struct PowerFlowData{
    T <: PowerFlowEvaluationModel,
    M <: PNM.PowerNetworkMatrix,
    N <: Union{PNM.PowerNetworkMatrix, Nothing},
} <: PowerFlowContainer
    pf::T
    bus_active_power_injections::Matrix{Float64}
    bus_reactive_power_injections::Matrix{Float64}
    bus_active_power_withdrawals::Matrix{Float64}
    bus_reactive_power_withdrawals::Matrix{Float64}
    bus_active_power_constant_current_withdrawals::Matrix{Float64}
    bus_reactive_power_constant_current_withdrawals::Matrix{Float64}
    bus_active_power_constant_impedance_withdrawals::Matrix{Float64}
    bus_reactive_power_constant_impedance_withdrawals::Matrix{Float64}
    bus_reactive_power_bounds::Matrix{Tuple{Float64, Float64}}
    bus_slack_participation_factors::SparseMatrixCSC{Float64, Int}
    bus_active_power_range::Matrix{Float64}
    computed_generator_slack_participation_factors::Vector{
        Dict{Tuple{DataType, String}, Float64},
    }
    bus_type::Matrix{PSY.ACBusTypes}
    bus_magnitude::Matrix{Float64}
    bus_angles::Matrix{Float64}
    arc_active_power_flow_from_to::Matrix{Float64}
    arc_reactive_power_flow_from_to::Matrix{Float64}
    arc_active_power_flow_to_from::Matrix{Float64}
    arc_reactive_power_flow_to_from::Matrix{Float64}
    arc_angle_differences::Matrix{Float64}
    generic_hvdc_flows::Dict{Tuple{Int, Int}, Tuple{Float64, Float64}}
    bus_hvdc_net_power::Matrix{Float64}
    time_step_map::Dict{Int, String}
    power_network_matrix::M
    aux_network_matrix::N
    # Signed arc-bus incidence (rows = arcs; +1 from-bus, -1 to-bus), built once from
    # `PNM.IncidenceMatrix`. Used by DC solves for Δθ = A·θ; `nothing` for AC/vPTDF.
    arc_bus_incidence::Union{SparseMatrixCSC{Int8, Int}, Nothing}
    neighbors::Vector{Set{Int}}
    converged::BitVector
    loss_factors::Union{Matrix{Float64}, Nothing}
    voltage_stability_factors::Union{Matrix{Float64}, Nothing}
    arc_active_power_losses::Union{Matrix{Float64}, Nothing}
    lcc::LCCParameters
    arc_lossy_admittance_from_to::Union{SparseMatrixCSC{YBUS_ELTYPE, Int}, Nothing}
    arc_lossy_admittance_to_from::Union{SparseMatrixCSC{YBUS_ELTYPE, Int}, Nothing}
    controlled_devices::Union{Nothing, ControlledDeviceSet}
    # Persistent solver cache, reused across repeated solves on the same data (e.g. a PCM loop:
    # fixed network, changing injections) so factorizations are computed once and solve buffers
    # are not reallocated. Lazily populated in place on the first solve (the `Base.Ref` avoids
    # reconstructing `data`). Holds a [`SolverCache`](@ref) — an abstract supertype forward-declared
    # in `power_flow_types.jl` so the field type resolves before the concrete subtypes are defined.
    # Two TYPE-DISJOINT subtypes share this one slot:
    #   * DC path (`ABA`/PTDF data): a [`DCSolverCache`](@ref) holding the factored network matrix +
    #     backend (the invalidation key — rebuild when either changes, see
    #     `_get_or_build_solver_cache!`), the `PFLinearSolverCache`, and the per-solve scratch.
    #   * AC path, FastDecoupled solver (`ACPowerFlowData`): a `FastDecoupledCache` holding the
    #     factored B′ (once per data/scheme/backend) and per-PQ-set factored B″ submatrices
    #     (see `_get_or_build_fd_cache!`).
    # Each getter dispatches on the cached subtype, so an empty slot or a cross-use fails loudly
    # (a `MethodError`) instead of being silently mis-read — no sentinel tag needed.
    solver_cache::Base.RefValue{Union{Nothing, SolverCache}}
    # Memoized NR/TR AC-Jacobian sparse structure. Its OWN slot (not `solver_cache`) because the
    # AC Jacobian and a `solver_cache` entry can both be live in one solve (FastDecoupled handing
    # off to NR), so they must not contend. Lazily populated; see `_get_or_build_jacobian_structure`.
    ac_jacobian_structure_cache::Base.RefValue{Union{Nothing, ACJacobianStructureCache}}
end

# aliases for specific type parameter combinations.
"""A type alias for a `PowerFlowData` struct whose type parameters are
configured for an AC power flow method (`ACPolarPowerFlow` or
`ACRectangularPowerFlow`, i.e. any `AbstractACPowerFlow`)."""
const ACPowerFlowData = PowerFlowData{
    <:AbstractACPowerFlow,
    PNM.AC_Ybus_Matrix,
    <:Union{
        PNM.DC_ABA_Matrix_Factorized,
        Nothing,
    },
}
get_metadata_matrix(pfd::ACPowerFlowData) = pfd.power_network_matrix

"""A type alias for a `PowerFlowData` struct whose type parameters
are configured for the `PTDFDCPowerFlow` method ."""
const PTDFPowerFlowData = PowerFlowData{
    PTDFDCPowerFlow,
    PNM.DC_PTDF_Matrix,
    PNM.DC_ABA_Matrix_Factorized,
}

"""A type alias for a `PowerFlowData` struct whose type parameters
are configured for the `vPTDFDCPowerFlow` method."""
const vPTDFPowerFlowData = PowerFlowData{
    vPTDFDCPowerFlow,
    <:PNM.DC_vPTDF_Matrix,
    PNM.DC_ABA_Matrix_Factorized,
}
get_metadata_matrix(pfd::Union{PTDFPowerFlowData, vPTDFPowerFlowData}) =
    pfd.power_network_matrix

"""A type alias for a `PowerFlowData` struct whose type parameters
are configured for the `DCPowerFlow` method."""
const ABAPowerFlowData = PowerFlowData{
    DCPowerFlow,
    PNM.DC_ABA_Matrix_Factorized,
    PNM.DC_BA_Matrix,
}
get_metadata_matrix(pfd::ABAPowerFlowData) = pfd.aux_network_matrix

# true getters for fields:
get_pf(pfd::PowerFlowData) = pfd.pf
get_bus_active_power_injections(pfd::PowerFlowData) = pfd.bus_active_power_injections
get_bus_reactive_power_injections(pfd::PowerFlowData) = pfd.bus_reactive_power_injections
get_bus_active_power_withdrawals(pfd::PowerFlowData) = pfd.bus_active_power_withdrawals
get_bus_active_power_constant_current_withdrawals(pfd::PowerFlowData) =
    pfd.bus_active_power_constant_current_withdrawals
get_bus_active_power_constant_impedance_withdrawals(pfd::PowerFlowData) =
    pfd.bus_active_power_constant_impedance_withdrawals
get_bus_reactive_power_withdrawals(pfd::PowerFlowData) = pfd.bus_reactive_power_withdrawals
get_bus_reactive_power_constant_current_withdrawals(pfd::PowerFlowData) =
    pfd.bus_reactive_power_constant_current_withdrawals
get_bus_reactive_power_constant_impedance_withdrawals(pfd::PowerFlowData) =
    pfd.bus_reactive_power_constant_impedance_withdrawals
get_bus_reactive_power_bounds(pfd::PowerFlowData) = pfd.bus_reactive_power_bounds
get_bus_slack_participation_factors(pfd::PowerFlowData) =
    pfd.bus_slack_participation_factors
get_bus_type(pfd::PowerFlowData) = pfd.bus_type
get_bus_magnitude(pfd::PowerFlowData) = pfd.bus_magnitude
get_bus_angles(pfd::PowerFlowData) = pfd.bus_angles
get_arc_active_power_flow_from_to(pfd::PowerFlowData) =
    pfd.arc_active_power_flow_from_to
get_arc_reactive_power_flow_from_to(pfd::PowerFlowData) =
    pfd.arc_reactive_power_flow_from_to
get_arc_active_power_flow_to_from(pfd::PowerFlowData) =
    pfd.arc_active_power_flow_to_from
get_arc_reactive_power_flow_to_from(pfd::PowerFlowData) =
    pfd.arc_reactive_power_flow_to_from
get_arc_angle_differences(pfd::PowerFlowData) = pfd.arc_angle_differences
get_time_step_map(pfd::PowerFlowData) = pfd.time_step_map
get_power_network_matrix(pfd::PowerFlowData) = pfd.power_network_matrix
get_aux_network_matrix(pfd::PowerFlowData) = pfd.aux_network_matrix
get_neighbor(pfd::PowerFlowData) = pfd.neighbors
supports_multi_period(::PowerFlowData) = true
get_converged(pfd::PowerFlowData) = pfd.converged
get_loss_factors(pfd::PowerFlowData) = pfd.loss_factors
get_voltage_stability_factors(pfd::PowerFlowData) = pfd.voltage_stability_factors
get_arc_active_power_losses(pfd::PowerFlowData) = pfd.arc_active_power_losses
get_controlled_devices(pfd::PowerFlowData) = pfd.controlled_devices

# Field getter for expanded slack participation factors (one dict per time step)
# Named "computed" to distinguish from the user-supplied pf.generator_slack_participation_factors
get_computed_gspf(pfd::PowerFlowData) = pfd.computed_generator_slack_participation_factors

# Delegating getters: delegate to the stored PowerFlowEvaluationModel
get_calculate_loss_factors(pfd::PowerFlowData) = get_calculate_loss_factors(pfd.pf)
get_calculate_voltage_stability_factors(pfd::PowerFlowData) =
    get_calculate_voltage_stability_factors(pfd.pf)
get_log_solver_diagnostics(pfd::PowerFlowData) = get_log_solver_diagnostics(pfd.pf)
get_network_reductions(pfd::PowerFlowData) = get_network_reductions(pfd.pf)
"""
    get_time_steps(pfd::PowerFlowData)

Number of time steps configured on the embedded [`PowerFlowEvaluationModel`](@ref).
"""
get_time_steps(pfd::PowerFlowData) = get_time_steps(pfd.pf)
get_time_step_names(pfd::PowerFlowData) = get_time_step_names(pfd.pf)
get_correct_bustypes(pfd::PowerFlowData) = get_correct_bustypes(pfd.pf)

# LCC getters.
get_lcc_setpoint_at_rectifier(pfd::PowerFlowData) = pfd.lcc.setpoint_at_rectifier
get_lcc_p_set(pfd::PowerFlowData) = pfd.lcc.p_set
get_lcc_dc_line_resistance(pfd::PowerFlowData) = pfd.lcc.dc_line_resistance
get_lcc_rectifier_tap(pfd::PowerFlowData) = pfd.lcc.rectifier.tap
get_lcc_inverter_tap(pfd::PowerFlowData) = pfd.lcc.inverter.tap
get_lcc_rectifier_thyristor_angle(pfd::PowerFlowData) = pfd.lcc.rectifier.thyristor_angle
get_lcc_inverter_thyristor_angle(pfd::PowerFlowData) = pfd.lcc.inverter.thyristor_angle
get_lcc_rectifier_phi(pfd::PowerFlowData) = pfd.lcc.rectifier.phi
get_lcc_inverter_phi(pfd::PowerFlowData) = pfd.lcc.inverter.phi
get_lcc_rectifier_bus(pfd::PowerFlowData) = pfd.lcc.rectifier.bus
get_lcc_inverter_bus(pfd::PowerFlowData) = pfd.lcc.inverter.bus
get_lcc_rectifier_transformer_reactance(pfd::PowerFlowData) =
    pfd.lcc.rectifier.transformer_reactance
get_lcc_inverter_transformer_reactance(pfd::PowerFlowData) =
    pfd.lcc.inverter.transformer_reactance
get_lcc_rectifier_min_thyristor_angle(pfd::PowerFlowData) =
    pfd.lcc.rectifier.min_thyristor_angle
get_lcc_inverter_min_thyristor_angle(pfd::PowerFlowData) =
    pfd.lcc.inverter.min_thyristor_angle
get_lcc_i_dc(pfd::PowerFlowData) = pfd.lcc.i_dc
# pseudo getter.
get_lcc_count(data::PowerFlowData) = length(data.lcc.rectifier.bus)

# auxiliary getters for the fields of PowerNetworkMatrices we're storing:
# most things we patch through to calls on the metadata matrix:
"""
    get_bus_lookup(pfd::PowerFlowData)

Bus number → row index lookup for matrices stored in `pfd` (via the metadata
[`PowerNetworkMatrices`](https://sienna-platform.github.io/PowerNetworkMatrices.jl/stable/)
matrix). Use this when mapping device buses from a [`PowerSystems.System`](@extref)
onto [`PowerFlowData`](@ref) injection and withdrawal arrays.
"""
get_bus_lookup(pfd::PowerFlowData) = PNM.get_bus_lookup(get_metadata_matrix(pfd))
get_bus_axis(pfd::PowerFlowData) = PNM.get_bus_axis(get_metadata_matrix(pfd))
get_arc_lookup(pfd::PowerFlowData) = PNM.get_arc_lookup(get_metadata_matrix(pfd))
get_arc_axis(pfd::PowerFlowData) = PNM.get_arc_axis(get_metadata_matrix(pfd))
get_network_reduction_data(pfd::PowerFlowData) =
    PNM.get_network_reduction_data(get_metadata_matrix(pfd))
get_valid_ix(pdf::PowerFlowData) = Not(PNM.get_ref_bus_position(get_metadata_matrix(pdf)))

# the ybus matrix itself doesn't have an arc axis, so we have to special-case it.
get_arc_lookup(pfd::ACPowerFlowData) =
    PNM.get_arc_lookup(pfd.power_network_matrix.arc_admittance_from_to)
get_arc_axis(pfd::ACPowerFlowData) =
    PNM.get_arc_axis(pfd.power_network_matrix.arc_admittance_from_to)

# used for shifting angles relative to REF bus after DC solve.
subnetwork_axes(data::PTDFPowerFlowData) = data.aux_network_matrix.subnetwork_axes
subnetwork_axes(data::vPTDFPowerFlowData) = data.aux_network_matrix.subnetwork_axes
subnetwork_axes(data::ABAPowerFlowData) = data.power_network_matrix.subnetwork_axes
subnetwork_axes(data::ACPowerFlowData) = get_aux_network_matrix(data).subnetwork_axes

# so we can initialize things to the correct size inside the below constructor.
# No `PowerFlowData` instance, so can't call get_arc_axis or similar to get the size.
arc_count(::AbstractACPowerFlow,
    power_network_matrix::PNM.PowerNetworkMatrix,
    ::Union{PNM.PowerNetworkMatrix, Nothing}) = length(PNM.get_arc_axis(power_network_matrix.arc_admittance_from_to))
bus_count(::AbstractACPowerFlow,
    power_network_matrix::PNM.PowerNetworkMatrix,
    ::Union{PNM.PowerNetworkMatrix, Nothing}) = length(PNM.get_bus_axis(power_network_matrix))

arc_count(::Union{PTDFDCPowerFlow, vPTDFDCPowerFlow},
    power_network_matrix::PNM.PowerNetworkMatrix,
    ::Union{PNM.PowerNetworkMatrix, Nothing}) =
    length(PNM.get_arc_axis(power_network_matrix))
bus_count(::Union{PTDFDCPowerFlow, vPTDFDCPowerFlow},
    power_network_matrix::PNM.PowerNetworkMatrix,
    ::Union{PNM.PowerNetworkMatrix, Nothing}) =
    length(PNM.get_bus_axis(power_network_matrix))

arc_count(::DCPowerFlow,
    power_network_matrix::PNM.PowerNetworkMatrix,
    aux_network_matrix::Union{PNM.PowerNetworkMatrix, Nothing}) =
    length(PNM.get_arc_axis(aux_network_matrix))
bus_count(::DCPowerFlow,
    power_network_matrix::PNM.PowerNetworkMatrix,
    aux_network_matrix::Union{PNM.PowerNetworkMatrix, Nothing}) =
    length(PNM.get_bus_axis(aux_network_matrix))

_make_arc_active_power_losses(::AbstractDCPowerFlow, n_arcs, n_time_steps) =
    zeros(n_arcs, n_time_steps)
_make_arc_active_power_losses(::PowerFlowEvaluationModel, n_arcs, n_time_steps) = nothing

"""
Sets the two `PowerNetworkMatrix` fields and a few others (`time_steps`, `time_step_map`),
then creates arrays of default values (usually zeros) for the rest.
"""
function PowerFlowData(
    pf::T,
    power_network_matrix::M,
    aux_network_matrix::N,
    n_lccs::Int;
    neighbors = Vector{Set{Int}}(),
    arc_lossy_admittance_from_to::Union{SparseMatrixCSC{YBUS_ELTYPE, Int}, Nothing} = nothing,
    arc_lossy_admittance_to_from::Union{SparseMatrixCSC{YBUS_ELTYPE, Int}, Nothing} = nothing,
    controlled_devices::Union{Nothing, ControlledDeviceSet} = nothing,
    arc_bus_incidence::Union{SparseMatrixCSC{Int8, Int}, Nothing} = nothing,
) where {
    T <: PowerFlowEvaluationModel,
    M <: PNM.PowerNetworkMatrix,
    N <: Union{PNM.PowerNetworkMatrix, Nothing},
}
    n_time_steps = get_time_steps(pf)
    time_step_names = get_time_step_names(pf)
    if n_time_steps != 0
        if length(time_step_names) == 0
            time_step_names = [string(i) for i in 1:n_time_steps]
        elseif length(time_step_names) != n_time_steps
            error("time_step_names field must have same length as n_time_steps")
        end
    end
    time_step_map = Dict(zip([i for i in 1:n_time_steps], time_step_names))

    n_buses = bus_count(pf, power_network_matrix, aux_network_matrix)
    n_arcs = arc_count(pf, power_network_matrix, aux_network_matrix)
    calculate_loss_factors = get_calculate_loss_factors(pf)
    calculate_voltage_stability_factors = get_calculate_voltage_stability_factors(pf)

    lcc_parameters = LCCParameters(n_time_steps, n_lccs)
    return PowerFlowData(
        pf,
        zeros(n_buses, n_time_steps), # bus_active_power_injections
        zeros(n_buses, n_time_steps), # bus_reactive_power_injections
        zeros(n_buses, n_time_steps), # bus_active_power_withdrawals
        zeros(n_buses, n_time_steps), # bus_reactive_power_withdrawals
        zeros(n_buses, n_time_steps), # bus_active_power_constant_current_withdrawals
        zeros(n_buses, n_time_steps), # bus_reactive_power_constant_current_withdrawals
        zeros(n_buses, n_time_steps), # bus_active_power_constant_impedance_withdrawals
        zeros(n_buses, n_time_steps), # bus_reactive_power_constant_impedance_withdrawals
        fill((-Inf, Inf), (n_buses, n_time_steps)), # bus_reactive_power_bounds
        spzeros(n_buses, n_time_steps), # bus_slack_participation_factors
        zeros(n_buses, n_time_steps), # bus_active_power_range
        Vector{Dict{Tuple{DataType, String}, Float64}}(), # computed_generator_slack_participation_factors
        fill(PSY.ACBusTypes.PQ, (n_buses, n_time_steps)), # bus_type
        ones(n_buses, n_time_steps), # bus_magnitude
        zeros(n_buses, n_time_steps), # bus_angles
        zeros(n_arcs, n_time_steps), # arc_active_power_flow_from_to
        zeros(n_arcs, n_time_steps), # arc_reactive_power_flow_from_to
        zeros(n_arcs, n_time_steps), # arc_active_power_flow_to_from
        zeros(n_arcs, n_time_steps), # arc_reactive_power_flow_to_from
        zeros(n_arcs, n_time_steps), # arc_angle_differences
        Dict{Tuple{Int, Int}, Tuple{Float64, Float64}}(), # generic_hvdc_flows
        zeros(n_buses, n_time_steps), # bus_hvdc_net_power
        time_step_map,
        power_network_matrix,
        aux_network_matrix,
        arc_bus_incidence,
        neighbors,
        falses(n_time_steps), # converged
        calculate_loss_factors ? zeros(n_buses, n_time_steps) : nothing, # loss_factors
        calculate_voltage_stability_factors ? zeros(n_buses, n_time_steps) : nothing, # voltage_stability_factors
        _make_arc_active_power_losses(pf, n_arcs, n_time_steps), # arc_active_power_losses
        lcc_parameters,
        arc_lossy_admittance_from_to,
        arc_lossy_admittance_to_from,
        controlled_devices,
        Base.RefValue{Union{Nothing, SolverCache}}(nothing), # solver_cache (lazily populated)
        Base.RefValue{Union{Nothing, ACJacobianStructureCache}}(nothing), # ac_jacobian_structure_cache
    )
end

function get_bus_active_power_total_withdrawals(pfd::PowerFlowData, ix::Int, time_step::Int)
    return pfd.bus_active_power_withdrawals[ix, time_step] +
           pfd.bus_active_power_constant_current_withdrawals[ix, time_step] *
           pfd.bus_magnitude[ix, time_step] +
           pfd.bus_active_power_constant_impedance_withdrawals[ix, time_step] *
           pfd.bus_magnitude[ix, time_step]^2
end

function get_bus_reactive_power_total_withdrawals(
    pfd::PowerFlowData,
    ix::Int,
    time_step::Int,
)
    return pfd.bus_reactive_power_withdrawals[ix, time_step] +
           pfd.bus_reactive_power_constant_current_withdrawals[ix, time_step] *
           pfd.bus_magnitude[ix, time_step] +
           pfd.bus_reactive_power_constant_impedance_withdrawals[ix, time_step] *
           pfd.bus_magnitude[ix, time_step]^2
end

function clear_injection_data!(pfd::PowerFlowData)
    # anything overwritten with NaNs in the case of non-convergence should be reset here.
    pfd.bus_active_power_injections .= 0.0
    pfd.bus_reactive_power_injections .= 0.0
    pfd.bus_active_power_withdrawals .= 0.0
    pfd.bus_active_power_constant_current_withdrawals .= 0.0
    pfd.bus_active_power_constant_impedance_withdrawals .= 0.0
    pfd.bus_reactive_power_withdrawals .= 0.0
    pfd.bus_reactive_power_constant_current_withdrawals .= 0.0
    pfd.bus_reactive_power_constant_impedance_withdrawals .= 0.0
    for col in eachcol(pfd.bus_angles)
        any(isnan, col) && (col .= 0.0)
    end
    for col in eachcol(pfd.bus_magnitude)
        any(isnan, col) && (col .= 1.0)
    end
    return
end

function _calculate_neighbors(
    Yb::PNM.Ybus{
        Tuple{Vector{Int64}, Vector{Int64}},
        Tuple{Dict{Int64, Int64}, Dict{Int64, Int64}},
    },
)
    I, J, V = SparseArrays.findnz(Yb.data)
    neighbors = [Set{Int}([i]) for i in 1:length(Yb.axes[1])]
    for nz in eachindex(V)
        push!(neighbors[I[nz]], J[nz])
        push!(neighbors[J[nz]], I[nz])
    end
    return neighbors
end

# A DegreeTwoReduction that reduces reactive-power injectors folds away shunts and
# synchronous condensers, discarding the reactive injections the AC solution depends
# on; the default of `true` is only safe for DC power flow.
_assert_ac_reduction_supported(::PNM.NetworkReduction) = nothing
function _assert_ac_reduction_supported(nr::PNM.DegreeTwoReduction)
    PNM.get_reduce_reactive_power_injectors(nr) || return nothing
    throw(
        IS.ConflictingInputsError(
            "DegreeTwoReduction with `reduce_reactive_power_injectors = true` is not \
             supported with AC power flow: reducing reactive-power injectors (e.g. a \
             shunt FixedAdmittance or a SynchronousCondenser) discards reactive \
             injections the AC solution depends on. Pass \
             `DegreeTwoReduction(; reduce_reactive_power_injectors = false)` instead.",
        ),
    )
end

# NOTE: remove this once network reductions are fully implemented
function network_reduction_message(
    nrs::Vector{PNM.NetworkReduction},
    m::PowerFlowEvaluationModel,
)
    if m isa ACPowerFlow && any(isa.(nrs, (PNM.WardReduction,)))
        throw(
            IS.NotImplementedError(
                "Ward reduction with AC power flow is not supported yet.",
            ),
        )
    end
    if m isa ACPowerFlow
        foreach(_assert_ac_reduction_supported, nrs)
    end
    if m isa AbstractDCPowerFlow && any(isa.(nrs, (PNM.WardReduction,)))
        @warn "Use Ward reduction with DC power flow with caution. Branch flows for branches in parallel with equivalent branches added by Ward reduction may be incorrect."
    end
    if any(isa.(nrs, (PNM.DegreeTwoReduction,)))
        @warn "Degree 2 network reductions mis-report branch power flows, but bus voltage results are correct. Use with caution."
    end
    return
end

function make_and_initialize_power_flow_data(
    pf::PowerFlowEvaluationModel,
    sys::PSY.System,
    power_network_matrix::M,
    aux_network_matrix::N;
    neighbors = Vector{Set{Int}}(),
    arc_lossy_admittance_from_to::Union{SparseMatrixCSC{YBUS_ELTYPE, Int}, Nothing} = nothing,
    arc_lossy_admittance_to_from::Union{SparseMatrixCSC{YBUS_ELTYPE, Int}, Nothing} = nothing,
    controlled_devices::Union{Nothing, ControlledDeviceSet} = nothing,
    arc_bus_incidence::Union{SparseMatrixCSC{Int8, Int}, Nothing} = nothing,
) where {M <: PNM.PowerNetworkMatrix, N <: Union{PNM.PowerNetworkMatrix, Nothing}}
    check_unit_setting(sys)
    if controlled_devices === nothing && get_control_discrete_devices(pf)
        @warn "control_discrete_devices=true, but no controlled_devices were supplied \
            to make_and_initialize_power_flow_data — discrete device control will NOT \
            run. Construct via PowerFlowData(pf, sys), or pass a ControlledDeviceSet \
            built with build_controlled_device_set." maxlog = 1
    end
    removed_buses =
        PNM.get_removed_buses(PNM.get_network_reduction_data(power_network_matrix))
    lcc_filter =
        lcc ->
            PSY.get_number(PSY.get_from(PSY.get_arc(lcc))) ∉ removed_buses &&
                PSY.get_number(PSY.get_to(PSY.get_arc(lcc))) ∉ removed_buses
    n_lccs = length(PSY.get_available_components(lcc_filter, PSY.TwoTerminalLCCLine, sys))
    data = PowerFlowData(
        pf,
        power_network_matrix,
        aux_network_matrix,
        n_lccs;
        neighbors = neighbors,
        arc_lossy_admittance_from_to = arc_lossy_admittance_from_to,
        arc_lossy_admittance_to_from = arc_lossy_admittance_to_from,
        controlled_devices = controlled_devices,
        arc_bus_incidence = arc_bus_incidence,
    )
    @assert length(data.lcc.setpoint_at_rectifier) == n_lccs
    initialize_power_flow_data!(data, pf, sys; correct_bustypes = get_correct_bustypes(pf))
    return data
end

# Build the signed arc-bus incidence from PNM's `IncidenceMatrix`, permuted to align its rows/cols
# with `metadata_matrix`'s axes (= `get_arc_axis(data)`/`get_bus_lookup(data)` at solve time);
# the explicit permutation guards against PNM axis drift. `Int8` entries match PNM.
function _signed_arc_bus_incidence(ybus::PNM.Ybus, metadata_matrix::PNM.PowerNetworkMatrix)
    inc = PNM.IncidenceMatrix(ybus)
    arc_lookup = PNM.get_arc_lookup(inc)
    bus_lookup = PNM.get_bus_lookup(inc)
    arc_perm = [arc_lookup[arc] for arc in PNM.get_arc_axis(metadata_matrix)]
    bus_perm = [bus_lookup[bus] for bus in PNM.get_bus_axis(metadata_matrix)]
    return inc.data[arc_perm, bus_perm]
end

# PNM applies the zero-impedance branch reduction through a dedicated `zero_impedance_reduction`
# kwarg (and rejects one passed in `network_reductions`). A `ZeroImpedanceBranchReduction` is still a
# `NetworkReduction`, so PowerFlows lets users put it in the usual `network_reductions` field and
# routes it to that kwarg here. Dispatch (not `isa`) classifies the entry.
_is_zero_impedance_reduction(::PNM.ZeroImpedanceBranchReduction) = true
_is_zero_impedance_reduction(::PNM.NetworkReduction) = false

# Split the user's reductions into (everything else, the zero-impedance reduction). When the user
# supplied none, return PNM's default `ZeroImpedanceBranchReduction()` so the always-applied
# zero-impedance step keeps its default parameters.
function _route_zero_impedance_reduction(reductions::Vector{PNM.NetworkReduction})
    idx = findfirst(_is_zero_impedance_reduction, reductions)
    isnothing(idx) && return reductions, PNM.ZeroImpedanceBranchReduction()
    others = PNM.NetworkReduction[
        r for r in reductions if !_is_zero_impedance_reduction(r)
    ]
    return others, reductions[idx]
end

"""
    PowerFlowData(
        pf::AbstractACPowerFlow{<:ACPowerFlowSolverType},
        sys::PSY.System
    ) -> ACPowerFlowData{<:ACPowerFlowSolverType}

Creates the structure for an AC power flow calculation, given the
[`PowerSystems.System`](@extref) `sys`. Configuration options like `time_steps`,
`time_step_names`, `network_reductions`, and `correct_bustypes` are taken from the
[`AbstractACPowerFlow`](@ref) object (either [`ACPolarPowerFlow`](@ref) or
[`ACRectangularPowerFlow`](@ref)).

Calling this function will not evaluate the power flows and angles. This version is
used to solve AC power flows and returns an [`ACPowerFlowData`](@ref) object.

# Arguments:
- `pf::AbstractACPowerFlow`:
        the settings for the AC power flow solver, including `time_steps`, `time_step_names`,
        `network_reductions`, and `correct_bustypes`.
- `sys::PSY.System`:
        A [`PowerSystems.System`](@extref) object that represents the power
        grid under consideration.

WARNING: functions for the evaluation of the multi-period AC PF still to be implemented.
"""
function PowerFlowData(
    pf::AbstractACPowerFlow{<:ACPowerFlowSolverType},
    sys::PSY.System,
)
    network_reductions = get_network_reductions(pf)
    network_reduction_message(network_reductions, pf)
    reductions, zero_impedance_reduction =
        _route_zero_impedance_reduction(network_reductions)
    power_network_matrix = PNM.Ybus(
        sys;
        network_reductions = reductions,
        make_arc_admittance_matrices = true,
        include_constant_impedance_loads = false,
        zero_impedance_reduction = zero_impedance_reduction,
    )
    neighbors = _calculate_neighbors(power_network_matrix)

    if get_robust_power_flow(pf)
        aux_network_matrix =
            PNM.ABA_Matrix(power_network_matrix; factorize = true)
    else
        aux_network_matrix = nothing
    end

    controlled_devices = if get_control_discrete_devices(pf)
        if !isempty(PSY.get_available_components(PSY.TwoTerminalLCCLine, sys))
            throw(
                ArgumentError(
                    "control_discrete_devices=true is not supported on systems with " *
                    "LCC HVDC lines: the continuation's rollback does not yet cover " *
                    "the per-time-step LCC state.",
                ),
            )
        end
        bus_lookup = PNM.get_bus_lookup(power_network_matrix)
        nrd = PNM.get_network_reduction_data(power_network_matrix)
        set = build_controlled_device_set(
            sys,
            bus_lookup,
            power_network_matrix;
            reverse_bus_search_map = PNM.get_reverse_bus_search_map(nrd),
            include_experimental = get(
                pf.solver_settings, :experimental_controls, false)::Bool,
        )
        isempty(set) ? nothing : set
    else
        nothing
    end

    return make_and_initialize_power_flow_data(
        pf,
        sys,
        power_network_matrix,
        aux_network_matrix;
        neighbors = neighbors,
        controlled_devices = controlled_devices,
    )
end

# DC Power Flow Data based on ABA and BA matrices
"""
    PowerFlowData(
        pf::DCPowerFlow,
        sys::PSY.System
    ) -> ABAPowerFlowData

Creates a `PowerFlowData` structure configured for a standard DC power flow calculation,
given the [`PowerSystems.System`](@extref) `sys`. Configuration options like
`time_steps`, `time_step_names`, `network_reductions`, and `correct_bustypes` are taken
from the [`DCPowerFlow`](@ref) object.

Calling this function will not evaluate the power flows and angles.
Note that first input is of type [`DCPowerFlow`](@ref): this version is
used to solve DC power flows, and returns an [`ABAPowerFlowData`](@ref) object.

# Arguments:
- [`pf::DCPowerFlow`](@ref PowerFlows.DCPowerFlow):
        Run a DC power flow: internally, store the ABA matrix as `power_network_matrix` and
        the BA matrix as `aux_network_matrix`. Configuration options are taken from this object.
- `sys::PSY.System`:
        A [`PowerSystems.System`](@extref) object that represents the power
        grid under consideration.
"""
function PowerFlowData(
    pf::DCPowerFlow,
    sys::PSY.System,
)
    network_reductions = get_network_reductions(pf)
    network_reduction_message(network_reductions, pf)
    reductions, zero_impedance_reduction =
        _route_zero_impedance_reduction(network_reductions)
    ybus = PNM.Ybus(
        sys;
        network_reductions = reductions,
        make_arc_admittance_matrices = pf.lossy_flows,
        zero_impedance_reduction = zero_impedance_reduction,
    )
    power_network_matrix = PNM.ABA_Matrix(ybus; factorize = true)
    aux_network_matrix = PNM.BA_Matrix(ybus)
    # `get_arc_axis(data)`/`get_bus_lookup(data)` read the BA (metadata) matrix for this method.
    arc_bus_incidence = _signed_arc_bus_incidence(ybus, aux_network_matrix)

    if pf.lossy_flows
        # Reorder rows of the arc admittance matrices to match the BA matrix arc
        # axis ordering (which is what get_arc_axis(data) returns at solve time).
        ba_arcs = PNM.get_arc_axis(aux_network_matrix)
        ybus_arc_lookup = PNM.get_arc_lookup(ybus.arc_admittance_from_to)
        perm = [ybus_arc_lookup[arc] for arc in ba_arcs]
        arc_lossy_from_to = ybus.arc_admittance_from_to.data[perm, :]
        arc_lossy_to_from = ybus.arc_admittance_to_from.data[perm, :]
    else
        arc_lossy_from_to = nothing
        arc_lossy_to_from = nothing
    end

    return make_and_initialize_power_flow_data(
        pf,
        sys,
        power_network_matrix,
        aux_network_matrix;
        arc_lossy_admittance_from_to = arc_lossy_from_to,
        arc_lossy_admittance_to_from = arc_lossy_to_from,
        arc_bus_incidence = arc_bus_incidence,
    )
end

# DC Power Flow Data with PTDF matrix
"""
    PowerFlowData(
        pf::PTDFDCPowerFlow,
        sys::PSY.System
    ) -> PTDFPowerFlowData

Creates a `PowerFlowData` structure configured for a Partial Transfer
Distribution Factor Matrix DC power flow calculation, given the
[`PowerSystems.System`](@extref) `sys`. Configuration options like
`time_steps`, `time_step_names`, `network_reductions`, and `correct_bustypes` are taken
from the [`PTDFDCPowerFlow`](@ref) object.

Calling this function will not evaluate the power flows and angles.
Note that first input is of type [`PTDFDCPowerFlow`](@ref): this version is used to solve
DC power flows via the Power Transfer Distribution Factor (PTDF) matrix. This function
returns a [`PTDFPowerFlowData`](@ref) object.

# Arguments:
- [`pf::PTDFDCPowerFlow`](@ref PowerFlows.PTDFDCPowerFlow):
        Run a DC power flow with PTDF matrix: internally, store the PTDF matrix
        as `power_network_matrix` and the ABA matrix as `aux_network_matrix`.
        Configuration options are taken from this object.
- `sys::PSY.System`:
        A [`PowerSystems.System`](@extref) object that represents the power
        grid under consideration.
"""
function PowerFlowData(
    pf::PTDFDCPowerFlow,
    sys::PSY.System,
)
    network_reductions = get_network_reductions(pf)
    network_reduction_message(network_reductions, pf)
    reductions, zero_impedance_reduction =
        _route_zero_impedance_reduction(network_reductions)
    # get the network matrices
    ybus = PNM.Ybus(sys;
        network_reductions = reductions,
        zero_impedance_reduction = zero_impedance_reduction)
    power_network_matrix = PNM.PTDF(ybus)
    aux_network_matrix = PNM.ABA_Matrix(ybus; factorize = true)
    # `get_arc_axis(data)`/`get_bus_lookup(data)` read the PTDF (metadata) matrix for this method.
    arc_bus_incidence = _signed_arc_bus_incidence(ybus, power_network_matrix)
    return make_and_initialize_power_flow_data(
        pf,
        sys,
        power_network_matrix,
        aux_network_matrix;
        arc_bus_incidence = arc_bus_incidence,
    )
end

# DC Power Flow Data with virtual PTDF matrix
"""
    PowerFlowData(
        pf::vPTDFDCPowerFlow,
        sys::PSY.System
    ) -> vPTDFPowerFlowData

Creates a `PowerFlowData` structure configured for a virtual Partial Transfer
Distribution Factor Matrix DC power flow calculation, given the
[`PowerSystems.System`](@extref) `sys`. Configuration options like
`time_steps`, `time_step_names`, `network_reductions`, and `correct_bustypes` are taken
from the [`vPTDFDCPowerFlow`](@ref) object.

Calling this function will not evaluate the power flows and angles.
Note that first input is of type [`vPTDFDCPowerFlow`](@ref): this version is used to solve
DC power flows using a virtual Power Transfer Distribution Factor (PTDF) matrix. This
function returns a [`vPTDFPowerFlowData`](@ref) object.

# Arguments:
- [`pf::vPTDFDCPowerFlow`](@ref vPTDFDCPowerFlow):
        Run a virtual PTDF power flow: internally, store the virtual PTDF matrix as
        `power_network_matrix` and the ABA matrix as `aux_network_matrix`.
        Configuration options are taken from this object.
- `sys::PSY.System`:
        A [`PowerSystems.System`](@extref) object that represents the power
        grid under consideration.
"""
function PowerFlowData(
    pf::vPTDFDCPowerFlow,
    sys::PSY.System,
)
    network_reductions = get_network_reductions(pf)
    network_reduction_message(network_reductions, pf)
    reductions, zero_impedance_reduction =
        _route_zero_impedance_reduction(network_reductions)

    # get the network matrices
    ybus = PNM.Ybus(sys;
        network_reductions = reductions,
        zero_impedance_reduction = zero_impedance_reduction)
    power_network_matrix = PNM.VirtualPTDF(ybus) # evaluates an empty virtual PTDF
    aux_network_matrix = PNM.ABA_Matrix(ybus; factorize = true)

    return make_and_initialize_power_flow_data(
        pf,
        sys,
        power_network_matrix,
        aux_network_matrix,
    )
end

"""Compute arc angle differences for all arcs and all time steps, looking up
bus indices from the arc axis and bus lookup stored in `data`. Used by DC solvers."""
function _compute_arc_angle_differences_from_data!(
    data::PowerFlowData{T, M, N},
) where {
    T <: PowerFlowEvaluationModel,
    M <: PNM.PowerNetworkMatrix,
    N <: Union{PNM.PowerNetworkMatrix, Nothing},
}
    arcs = get_arc_axis(data)
    bus_lookup = get_bus_lookup(data)
    fb_ix = [bus_lookup[bus_no] for bus_no in first.(arcs)]
    tb_ix = [bus_lookup[bus_no] for bus_no in last.(arcs)]
    @views data.arc_angle_differences .=
        data.bus_angles[fb_ix, :] .- data.bus_angles[tb_ix, :]
    return
end

"""Compute arc angle differences using precomputed from/to bus index vectors
over specified time steps. Used by the AC solver where `fb_ix`/`tb_ix` are
already available from the branch flow calculation."""
function _compute_arc_angle_differences_from_indices!(
    data::PowerFlowData{T, M, N},
    fb_ix::Vector{Int},
    tb_ix::Vector{Int},
    time_steps::Vector{Int},
) where {
    T <: PowerFlowEvaluationModel,
    M <: PNM.PowerNetworkMatrix,
    N <: Union{PNM.PowerNetworkMatrix, Nothing},
}
    @views data.arc_angle_differences[:, time_steps] .=
        data.bus_angles[fb_ix, time_steps] .- data.bus_angles[tb_ix, time_steps]
    return
end

"""
Create an appropriate `PowerFlowContainer` for the given `PowerFlowEvaluationModel` and initialize it from the given `PSY.System`.

Configuration options like `time_steps`, `time_step_names`, `network_reductions`, and
`correct_bustypes` are taken from the `PowerFlowEvaluationModel` object.

# Arguments:
- `pfem::PowerFlowEvaluationModel`: power flow model to construct a container for (e.g., `DCPowerFlow()`)
- `sys::PSY.System`: the [PowerSystems.System](@extref) from which to initialize the
    power flow container
"""
function make_power_flow_container end

make_power_flow_container(
    pfem::AbstractACPowerFlow{<:ACPowerFlowSolverType},
    sys::PSY.System,
) = PowerFlowData(pfem, sys)

make_power_flow_container(pfem::DCPowerFlow, sys::PSY.System) =
    PowerFlowData(pfem, sys)

make_power_flow_container(pfem::PTDFDCPowerFlow, sys::PSY.System) =
    PowerFlowData(pfem, sys)

make_power_flow_container(pfem::vPTDFDCPowerFlow, sys::PSY.System) =
    PowerFlowData(pfem, sys)
