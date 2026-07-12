"""
    struct ACPowerFlowResidual

A struct to keep track of the residuals in the Newton-Raphson AC power flow calculation.

# Fields
- `data::ACPowerFlowData`: The grid model data.
- `Rv::Vector{Float64}`: A vector of the values of the residuals.
- `P_net::Vector{Float64}`: A vector of net active power injections.
- `Q_net::Vector{Float64}`: A vector of net reactive power injections.
- `P_net_set::Vector{Float64}`: A vector of the set-points for active power injections (their initial values before power flow calculation).
- `bus_slack_participation_factors::SparseVector{Float64, Int}`: A sparse vector of the slack participation factors aggregated at the bus level.
- `subnetworks::Dict{Int64, Vector{Int64}}`: The dictionary that identifies subnetworks (connected components), with the key defining the REF bus, values defining the corresponding buses in the subnetwork.
- `P_slack_buf::Vector{Float64}`: Scratch buffer of length `n_buses` used by `_update_residual_values!` to write the per-subnetwork slack distribution in place, avoiding a per-iteration allocation when indexing `bus_slack_participation_factors` by `subnetwork_buses`.
- `validate_indices::Vector{Int}`: precomputed `x`-indices of PQ-bus |V| entries for the per-iteration voltage-magnitude diagnostic.
"""
struct ACPowerFlowResidual
    data::ACPowerFlowData
    Rv::Vector{Float64}
    P_net::Vector{Float64}
    Q_net::Vector{Float64}
    P_net_set::Vector{Float64}
    bus_slack_participation_factors::SparseVector{Float64, Int}
    subnetworks::Dict{Int64, Vector{Int64}}
    bus_active_constant_I::Vector{Float64}
    bus_reactive_constant_I::Vector{Float64}
    bus_active_constant_Z::Vector{Float64}
    bus_reactive_constant_Z::Vector{Float64}
    P_slack_buf::Vector{Float64}
    validate_indices::Vector{Int}
end

"""
    ACPowerFlowResidual(data::ACPowerFlowData, time_step::Int64)

Create an instance of `ACPowerFlowResidual` for a given time step.

# Arguments
- `data::ACPowerFlowData`: The power flow data representing the power system model.
- `time_step::Int64`: The time step for which the power flow calculation is executed.

# Returns
- `ACPowerFlowResidual`: An instance containing the residual values, net bus active power injections, 
    and net bus reactive power injections.
"""
function ACPowerFlowResidual(data::ACPowerFlowData, time_step::Int64)
    n_buses = first(size(data.bus_type))
    P_net = Vector{Float64}(undef, n_buses)
    Q_net = Vector{Float64}(undef, n_buses)

    P_net_set = zeros(Float64, n_buses)
    bus_type = view(data.bus_type, :, time_step)

    # ref_bus is set to the first REF bus found - will be used for the total slack power
    subnetworks =
        _find_subnetworks_for_reference_buses(data.power_network_matrix.data, bus_type)

    for ix in 1:n_buses
        P_net[ix] =
            data.bus_active_power_injections[ix, time_step] -
            get_bus_active_power_total_withdrawals(data, ix, time_step) +
            data.bus_hvdc_net_power[ix, time_step]
        Q_net[ix] =
            data.bus_reactive_power_injections[ix, time_step] -
            get_bus_reactive_power_total_withdrawals(data, ix, time_step)
        P_net_set[ix] = P_net[ix]
    end

    validate_indices = _pq_validate_indices(bus_type)

    bus_slack_participation_factors =
        _build_bus_slack_participation_factors(data, bus_type, subnetworks, time_step)

    bus_active_constant_I =
        copy(view(data.bus_active_power_constant_current_withdrawals, :, time_step))
    bus_reactive_constant_I =
        copy(view(data.bus_reactive_power_constant_current_withdrawals, :, time_step))
    bus_active_constant_Z =
        copy(view(data.bus_active_power_constant_impedance_withdrawals, :, time_step))
    bus_reactive_constant_Z =
        copy(view(data.bus_reactive_power_constant_impedance_withdrawals, :, time_step))

    return ACPowerFlowResidual(
        data,
        Vector{Float64}(undef,
            2 * n_buses + state_tail_length(data, get_dc_network(data))),
        P_net,
        Q_net,
        P_net_set,
        bus_slack_participation_factors,
        subnetworks,
        bus_active_constant_I,
        bus_reactive_constant_I,
        bus_active_constant_Z,
        bus_reactive_constant_Z,
        Vector{Float64}(undef, n_buses),
        validate_indices,
    )
end

"""
    (Residual::ACPowerFlowResidual)(Rv::Vector{Float64}, x::Vector{Float64}, time_step::Int64)

Evaluate the AC power flow residuals and store the result in `Rv` using the provided
state vector `x` and the current time step `time_step`.
The residuals are updated inplace in the struct and additionally copied to the provided array.
This function implements the functor approach for the `ACPowerFlowResidual` struct.
This makes the struct callable.
Calling the `ACPowerFlowResidual` will also update the values of P, Q, V, Θ in the `data` struct.

# Arguments
- `Rv::Vector{Float64}`: The vector to store the calculated residuals.
- `x::Vector{Float64}`: The state vector.
- `time_step::Int64`: The current time step.
"""
function (Residual::ACPowerFlowResidual)(
    Rv::Vector{Float64},
    x::Vector{Float64},
    time_step::Int64,
)
    _update_residual_values!(
        Residual.Rv,
        x,
        Residual.P_net,
        Residual.Q_net,
        Residual.P_net_set,
        Residual.bus_slack_participation_factors,
        Residual.subnetworks,
        Residual.bus_active_constant_I,
        Residual.bus_reactive_constant_I,
        Residual.bus_active_constant_Z,
        Residual.bus_reactive_constant_Z,
        Residual.data,
        time_step,
        Residual.P_slack_buf,
    )
    copyto!(Rv, Residual.Rv)
    return
end

"""
    (Residual::ACPowerFlowResidual)(x::Vector{Float64}, time_step::Int64)

Update the AC power flow residuals inplace and store the result in the attribute `Rv` of the struct.
The inputs are the values of state vector `x` and the current time step `time_step`.
This function implements the functor approach for the `ACPowerFlowResidual` struct.
This makes the struct callable.
Calling the `ACPowerFlowResidual` will also update the values of P, Q, V, Θ in the `data` struct.

# Arguments
- `x::Vector{Float64}`: The state vector values.
- `time_step::Int64`: The current time step.
"""
function (Residual::ACPowerFlowResidual)(x::Vector{Float64}, time_step::Int64)
    _update_residual_values!(
        Residual.Rv,
        x,
        Residual.P_net,
        Residual.Q_net,
        Residual.P_net_set,
        Residual.bus_slack_participation_factors,
        Residual.subnetworks,
        Residual.bus_active_constant_I,
        Residual.bus_reactive_constant_I,
        Residual.bus_active_constant_Z,
        Residual.bus_reactive_constant_Z,
        Residual.data,
        time_step,
        Residual.P_slack_buf,
    )
    return
end

function _setpq(
    ix::Int,
    P_net::Vector{Float64},
    Q_net::Vector{Float64},
    data::ACPowerFlowData,
    time_step::Int64,
)
    data.bus_active_power_injections[ix, time_step] =
        P_net[ix] + get_bus_active_power_total_withdrawals(data, ix, time_step) -
        data.bus_hvdc_net_power[ix, time_step]
    data.bus_reactive_power_injections[ix, time_step] =
        Q_net[ix] + get_bus_reactive_power_total_withdrawals(data, ix, time_step)
end

# dispatching on Val for performance reasons.
function _set_state_variables_at_bus!(
    ix::Int,
    P_net::Vector{Float64},
    Q_net::Vector{Float64},
    P_net_set::Vector{Float64},
    P_slack::Float64,
    StateVector::Vector{Float64},
    data::ACPowerFlowData,
    time_step::Int64,
    ::Val{PSY.ACBusTypes.REF})
    # When bustype == REFERENCE PSY.ACBus, state variables are Active and Reactive Power Generated
    P_net[ix] = P_net_set[ix] + P_slack
    Q_net[ix] = StateVector[2 * ix]
    _setpq(ix, P_net, Q_net, data, time_step)
end

function _set_state_variables_at_bus!(
    ix::Int,
    P_net::Vector{Float64},
    Q_net::Vector{Float64},
    P_net_set::Vector{Float64},
    P_slack::Float64,
    StateVector::Vector{Float64},
    data::ACPowerFlowData,
    time_step::Int64,
    ::Val{PSY.ACBusTypes.PV})
    # When bustype == PV PSY.ACACBus, state variables are Reactive Power Generated and Voltage Angle
    # We still update both P and Q values in case the PV bus participates in distributed slack
    P_net[ix] = P_net_set[ix] + P_slack
    Q_net[ix] = StateVector[2 * ix - 1]
    _setpq(ix, P_net, Q_net, data, time_step)
    data.bus_angles[ix, time_step] = StateVector[2 * ix]
end

function _set_state_variables_at_bus!(
    ix::Int,
    P_net::Vector{Float64},
    Q_net::Vector{Float64},
    ::Vector{Float64},
    ::Float64,
    StateVector::Vector{Float64},
    bus_active_constant_I::Vector{Float64},
    bus_reactive_constant_I::Vector{Float64},
    bus_active_constant_Z::Vector{Float64},
    bus_reactive_constant_Z::Vector{Float64},
    data::ACPowerFlowData,
    time_step::Int64,
    ::Val{PSY.ACBusTypes.PQ})
    vm_1 = data.bus_magnitude[ix, time_step]
    vm_2 = StateVector[2 * ix - 1]
    data.bus_magnitude[ix, time_step] = vm_2
    data.bus_angles[ix, time_step] = StateVector[2 * ix]
    # update P_net and Q_net for ZIP loads
    P_net[ix] +=
        bus_active_constant_I[ix] * (vm_1 - vm_2) +
        bus_active_constant_Z[ix] * (vm_1^2 - vm_2^2)
    Q_net[ix] +=
        bus_reactive_constant_I[ix] * (vm_1 - vm_2) +
        bus_reactive_constant_Z[ix] * (vm_1^2 - vm_2^2)
    _setpq(ix, P_net, Q_net, data, time_step)
end

"""
    _update_residual_values!(
        F::Vector{Float64},
        x::Vector{Float64},
        P_net::Vector{Float64},
        Q_net::Vector{Float64},
        data::ACPowerFlowData,
        time_step::Int64,
    )

Update the residual values for the Newton-Raphson AC power flow calculation. This function is used internally in the
`ACPowerFlowResidual` struct. This function also updates the values of P, Q, V, Θ in the `data` struct.

# Arguments
- `F::Vector{Float64}`: Vector of the values of the residuals.
- `x::Vector{Float64}`: State vector values.
- `P_net::Vector{Float64}`: Vector of net active power injections at each bus.
- `Q_net::Vector{Float64}`: Vector of net reactive power injections at each bus.
- `P_net_set::Vector{Float64}`: Vector of the set-points for active power injections (their initial values before power flow calculation).
- `bus_slack_participation_factors::SparseVector{Float64, Int}`: Sparse vector of the slack participation factors aggregated at the bus level.
- `ref_bus::Int`: The index of the reference bus to be used for the total slack power.
- `data::ACPowerFlowData`: Data structure representing the grid model for the AC power flow calculation.
- `time_step::Int64`: The current time step for which the residual values are being updated.
"""
function _update_residual_values!(
    F::Vector{Float64},
    x::Vector{Float64},
    P_net::Vector{Float64},
    Q_net::Vector{Float64},
    P_net_set::Vector{Float64},
    bus_slack_participation_factors::SparseVector{Float64, Int},
    subnetworks::Dict{Int64, Vector{Int64}},
    bus_active_constant_I::Vector{Float64},
    bus_reactive_constant_I::Vector{Float64},
    bus_active_constant_Z::Vector{Float64},
    bus_reactive_constant_Z::Vector{Float64},
    data::ACPowerFlowData,
    time_step::Int64,
    P_slack_buf::Vector{Float64},
)
    # update P_net, Q_net, data.bus_angles, data.bus_magnitude based on X
    Yb = data.power_network_matrix.data
    num_lcc = size(data.lcc.p_set, 1)
    n_buses_total = first(size(data.bus_type))
    dcn = get_dc_network(data)
    vsc_off = 2 * n_buses_total + 4 * num_lcc
    bus_types = view(data.bus_type, :, time_step)

    for (ref_bus, subnetwork_buses) in subnetworks
        slack_scalar = x[2 * ref_bus - 1] - P_net_set[ref_bus]
        n_sub = length(subnetwork_buses)
        # Write per-bus slack into P_slack_buf[1:n_sub]. SparseVector indexed
        # by a Vector{Int} allocates a fresh Vector; iterate manually instead.
        @inbounds for k in 1:n_sub
            ix = subnetwork_buses[k]
            P_slack_buf[k] = slack_scalar * bus_slack_participation_factors[ix]
        end

        @inbounds for k in 1:n_sub
            ix = subnetwork_buses[k]
            bt = bus_types[ix]
            p_bus_slack = P_slack_buf[k]
            # creating Val(bt) at runtime is slow, requires allocating: split into cases
            # explicitly, so instead it's Val(compile-time constant).
            if bt == PSY.ACBusTypes.PQ
                _set_state_variables_at_bus!(
                    ix, P_net, Q_net, P_net_set, p_bus_slack, x,
                    bus_active_constant_I, bus_reactive_constant_I,
                    bus_active_constant_Z, bus_reactive_constant_Z,
                    data, time_step, Val(PSY.ACBusTypes.PQ),
                )
            elseif bt == PSY.ACBusTypes.PV
                _set_state_variables_at_bus!(
                    ix, P_net, Q_net, P_net_set, p_bus_slack, x,
                    data, time_step, Val(PSY.ACBusTypes.PV),
                )
            elseif bt == PSY.ACBusTypes.REF
                _set_state_variables_at_bus!(
                    ix, P_net, Q_net, P_net_set, p_bus_slack, x,
                    data, time_step, Val(PSY.ACBusTypes.REF),
                )
            end
        end
    end

    if num_lcc > 0
        lcc_end = vsc_off
        data.lcc.rectifier.tap[:, time_step] = x[(lcc_end - 4 * num_lcc + 1):4:lcc_end]
        data.lcc.inverter.tap[:, time_step] = x[(lcc_end - 4 * num_lcc + 2):4:lcc_end]
        data.lcc.rectifier.thyristor_angle[:, time_step] =
            x[(lcc_end - 4 * num_lcc + 3):4:lcc_end]
        data.lcc.inverter.thyristor_angle[:, time_step] =
            x[(lcc_end - 4 * num_lcc + 4):4:lcc_end]
        _update_ybus_lcc!(data, time_step)
    end
    if has_dc_network(dcn)
        _read_vsc_state!(dcn, x, vsc_off, time_step)
    end

    # compute active, reactive power balances using the just updated values.
    Vm = view(data.bus_magnitude, :, time_step)
    θ = view(data.bus_angles, :, time_step)
    # F is active and reactive power balance equations at all buses
    F .= 0.0
    # normal ybus.
    Yb_vals = SparseArrays.nonzeros(Yb)
    Yb_rowvals = SparseArrays.rowvals(Yb)
    @inbounds for bus_to in axes(Yb, 1)
        Vm_to = Vm[bus_to]
        θ_to = θ[bus_to]
        for j in SparseArrays.nzrange(Yb, bus_to)
            yb = Yb_vals[j]
            bus_from = Yb_rowvals[j]
            gb = real(yb)
            bb = imag(yb)
            vv = Vm[bus_from] * Vm_to
            if bus_from == bus_to
                F[2 * bus_from - 1] += vv * gb
                F[2 * bus_from] += -vv * bb
            else
                # `sincos` computes both at once (cheaper than separate `cos`/`sin`); the
                # trig over every off-diagonal nonzero dominates the residual evaluation.
                sinΔθ, cosΔθ = sincos(θ[bus_from] - θ_to)
                F[2 * bus_from - 1] += vv * (gb * cosΔθ + bb * sinΔθ)
                F[2 * bus_from] += vv * (gb * sinΔθ - bb * cosΔθ)
            end
        end
    end
    # we read off entries from the LCC branch admittances instead of maintaining
    # a separate ybus matrix for the LCCs. Few LCCs so efficient enough.
    if num_lcc > 0
        for (bus_indices, self_admittances) in
            zip(data.lcc.bus_indices, data.lcc.branch_admittances)
            for (bus_ix, y_val) in zip(bus_indices, self_admittances)
                gb = real(y_val)
                bb = imag(y_val)
                F[2 * bus_ix - 1] += Vm[bus_ix] * Vm[bus_ix] * gb
                F[2 * bus_ix] += -Vm[bus_ix] * Vm[bus_ix] * bb
            end
        end
    end

    # Strided broadcast `F[1:2:N] .-= P_net` allocates a copy of the slice on
    # each call; iterate explicitly to keep this allocation-free.
    @inbounds for ix in eachindex(P_net)
        F[2 * ix - 1] -= P_net[ix]
        F[2 * ix] -= Q_net[ix]
    end

    # PSS/E-style embedded area net-interchange control: ΔP_a couples into the P-balance
    # row at each controlled area's slack bus, at the same seam the distributed-slack
    # P_slack term enters above conceptually. Applied directly to `F` (not folded into
    # `P_net[ix] += ΔP` before the balance loop) because `P_net[ix]` is NOT reset to a
    # clean baseline every call once the slack bus is PQ (a PQ bus's ZIP-load dispatch does
    # `P_net[ix] += ...` incrementally, relying on `P_net[ix]` carrying over from the
    # PREVIOUS evaluation on this same `Residual` object — see
    # `_set_state_variables_at_bus!(::Val{PQ})`). Adding ΔP into that same persistent
    # buffer would silently accumulate an extra ΔP on every repeated evaluation after a
    # PV->PQ Q-limit flip of the area slack bus. `F` is reset (`F .= 0.0`) at the top of
    # every call, so applying ΔP here is exactly-once and bus-type-invariant (spec §4 item
    # 1 / §5.6), matching the Jacobian's constant `-1.0` stamp at this same row.
    if n_controlled_areas(data) > 0
        area_off = area_tail_offset(data, dcn)
        @inbounds for area in data.area_interchange.areas
            ΔP = x[area_off + area.tail_ix]
            # Mirror ΔP_a onto `data`, column-indexed by `time_step` (same seam as the LCC
            # tap / `_read_vsc_state!` tail write-back below) so a warm re-solve's `x0` can
            # recover the converged value for THIS time step without contaminating others —
            # see `AreaInterchangeData.delta_p` and `update_state!`.
            data.area_interchange.delta_p[area.tail_ix, time_step] = ΔP
            F[2 * area.slack_bus_ix - 1] -= ΔP
        end
    end

    if num_lcc > 0
        _set_lcc_tail_residuals!(F, data, vsc_off - 4 * num_lcc, time_step)
    end
    if has_dc_network(dcn)
        _apply_vsc_bus_injections_polar!(F, dcn, time_step)
        _set_vsc_tail_residuals!(F, dcn, Vm, vsc_off, time_step)
    end
    if n_controlled_areas(data) > 0
        area_off = area_tail_offset(data, dcn)
        _set_area_tail_residuals!(F, x, data, area_off, time_step)
    end
    return
end

function _find_subnetworks_for_reference_buses(
    Ybus::SparseMatrixCSC,
    bus_type::AbstractArray{PSY.ACBusTypes},
)
    subnetworks = PNM.find_subnetworks(Ybus, collect(eachindex(bus_type)))
    ref_buses = findall(x -> x == PSY.ACBusTypes.REF, bus_type)
    bus_groups = Dict{Int, Vector{Int}}()
    for (bus_key, subnetwork_buses) in subnetworks
        ref_bus = intersect(ref_buses, subnetwork_buses)
        if length(ref_bus) >= 1
            bus_groups[first(ref_bus)] = collect(subnetwork_buses)
        else
            throw(
                ArgumentError(
                    "No REF bus found in the subnetwork with $(length(subnetwork_buses)) buses defined by bus key $bus_key",
                ),
            )
        end
    end
    return bus_groups
end

"""
    _build_bus_slack_participation_factors(data, bus_type, subnetworks, time_step)

Collect the per-bus generator-slack-participation factors (REF and PV buses
only), validate that the sum is positive and no value is negative, and
normalize so that each subnetwork's participating buses sum to 1. Returns
a `SparseVector{Float64, Int}` of length `n_buses`.

Shared between the polar `ACPowerFlowResidual` and the rectangular
current-injection (CI) residual (`ACRectangularCIResidual`) constructors —
both need identical slack-distribution semantics.
"""
function _build_bus_slack_participation_factors(
    data::ACPowerFlowData,
    bus_type::AbstractVector{PSY.ACBusTypes},
    subnetworks::Dict{Int64, Vector{Int64}},
    time_step::Int64,
)
    n_buses = length(bus_type)
    spf_idx = Int[]
    spf_val = Float64[]
    sum_sl_weights = 0.0
    for (ix, bt) in zip(1:n_buses, bus_type)
        bt ∈ (PSY.ACBusTypes.REF, PSY.ACBusTypes.PV) || continue
        (spf_v = data.bus_slack_participation_factors[ix, time_step]) == 0.0 && continue
        push!(spf_idx, ix)
        push!(spf_val, spf_v)
        sum_sl_weights += spf_v
    end
    sum_sl_weights == 0.0 &&
        throw(ArgumentError("sum of slack_participation_factors cannot be zero"))
    any(spf_val .< 0.0) &&
        throw(ArgumentError("slack_participation_factors cannot be negative"))
    bus_slack_participation_factors = sparsevec(spf_idx, spf_val, n_buses)
    for subnetwork_buses in values(subnetworks)
        bspf_subnetwork = view(bus_slack_participation_factors, subnetwork_buses)
        sum_bspf = sum(bspf_subnetwork)
        sum_bspf == 0.0 && throw(
            ArgumentError(
                "sum of slack_participation_factors per subnetwork cannot be zero",
            ),
        )
        bspf_subnetwork ./= sum_bspf
    end
    return bus_slack_participation_factors
end
