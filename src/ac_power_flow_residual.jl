"""Per-bus |V|, θ and cis(θ) of the last evaluated polar iterate, shared by the
`ACPowerFlowResidual` and its `ACPowerFlowJacobian`. `data_stale` is `true` while `data` lacks
that iterate's voltages and injections (the NR loop's fused kernel defers them to
[`_write_back_bus_state!`](@ref)); while it is `false`, `data` is authoritative and is reloaded
before each evaluation."""
mutable struct PolarBusState
    Vm::Vector{Float64}
    θ::Vector{Float64}
    phasor::Vector{ComplexF64}
    data_stale::Bool
end

PolarBusState(n::Int) = PolarBusState(
    Vector{Float64}(undef, n),
    Vector{Float64}(undef, n),
    Vector{ComplexF64}(undef, n),
    false,
)

"""
    struct ACPowerFlowResidual

A struct to keep track of the residuals in the Newton-Raphson AC power flow calculation.

# Fields
- `Rv::Vector{Float64}`: A vector of the values of the residuals.
- `P_net::Vector{Float64}`: A vector of net active power injections.
- `Q_net::Vector{Float64}`: A vector of net reactive power injections.
- `P_net_set::Vector{Float64}`: A vector of the set-points for active power injections (their initial values before power flow calculation).
- `bus_slack_participation_factors::Vector{Float64}`: Dense per-bus slack participation factors, normalized per subnetwork. Shared with the `ACPowerFlowJacobian` and refilled in place on cache reuse.
- `subnetworks::Dict{Int64, Vector{Int64}}`: The dictionary that identifies subnetworks (connected components), with the key defining the REF bus, values defining the corresponding (sorted) buses in the subnetwork.
- `validate_indices::Vector{Int}`: precomputed `x`-indices of PQ-bus |V| entries for the per-iteration voltage-magnitude diagnostic.
- `bus_state::PolarBusState`: per-bus |V|, θ and `cis(θ)` of the last evaluated iterate, shared with the `ACPowerFlowJacobian`.
- `solve_start::Matrix{Float64}`: `P_net`, `Q_net`, |V| and θ at a solve's start, by column, for a rerun from the same start.
"""
struct ACPowerFlowResidual
    Rv::Vector{Float64}
    P_net::Vector{Float64}
    Q_net::Vector{Float64}
    P_net_set::Vector{Float64}
    bus_slack_participation_factors::Vector{Float64}
    subnetworks::Dict{Int64, Vector{Int64}}
    bus_active_constant_I::Vector{Float64}
    bus_reactive_constant_I::Vector{Float64}
    bus_active_constant_Z::Vector{Float64}
    bus_reactive_constant_Z::Vector{Float64}
    validate_indices::Vector{Int}
    bus_state::PolarBusState
    solve_start::Matrix{Float64}
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
    bus_type = view(data.bus_type, :, time_step)

    # ref_bus is set to the first REF bus found - will be used for the total slack power
    subnetworks =
        _find_subnetworks_for_reference_buses(data.power_network_matrix.data, bus_type)
    validate_indices = _pq_validate_indices(bus_type)
    bus_slack_participation_factors = zeros(n_buses)
    _fill_bus_slack_participation_factors!(
        bus_slack_participation_factors, data, bus_type, subnetworks, time_step)

    residual = ACPowerFlowResidual(
        Vector{Float64}(undef,
            2 * n_buses + state_tail_length(data, get_dc_network(data))),
        Vector{Float64}(undef, n_buses),
        Vector{Float64}(undef, n_buses),
        Vector{Float64}(undef, n_buses),
        bus_slack_participation_factors,
        subnetworks,
        Vector{Float64}(undef, n_buses),
        Vector{Float64}(undef, n_buses),
        Vector{Float64}(undef, n_buses),
        Vector{Float64}(undef, n_buses),
        validate_indices,
        PolarBusState(n_buses),
        Matrix{Float64}(undef, n_buses, 4),
    )
    _refresh_residual_setpoints!(residual, data, time_step)
    return residual
end

# Fills `P_net`/`Q_net`/`P_net_set` and the four constant-I/Z withdrawal vectors from `data` at
# `time_step`, in place. `P_net` is (re)set to the freshly computed value, not accumulated onto —
# the PQ ZIP path in `_update_residual_values!` telescopes onto whatever is here, so every caller
# (construction, `_refresh_polar_residual!`'s cache reuse, the sensitivity context's per-pass
# refresh) must rebuild it fresh from `data`, not fold onto a stale value.
"""Always succeeds for the polar residual; returns `true`."""
function _refresh_residual_setpoints!(
    residual::ACPowerFlowResidual, data::ACPowerFlowData, time_step::Int64,
)::Bool
    _load_bus_state!(residual.bus_state, data, time_step)
    @inbounds for ix in eachindex(residual.P_net)
        p =
            data.bus_active_power_injections[ix, time_step] -
            get_bus_active_power_total_withdrawals(data, ix, time_step) +
            data.bus_hvdc_net_power[ix, time_step]
        residual.P_net[ix] = p
        residual.P_net_set[ix] = p
        residual.Q_net[ix] =
            data.bus_reactive_power_injections[ix, time_step] -
            get_bus_reactive_power_total_withdrawals(data, ix, time_step)
    end
    residual.bus_active_constant_I .=
        view(data.bus_active_power_constant_current_withdrawals, :, time_step)
    residual.bus_reactive_constant_I .=
        view(data.bus_reactive_power_constant_current_withdrawals, :, time_step)
    residual.bus_active_constant_Z .=
        view(data.bus_active_power_constant_impedance_withdrawals, :, time_step)
    residual.bus_reactive_constant_Z .=
        view(data.bus_reactive_power_constant_impedance_withdrawals, :, time_step)
    return true
end

"""
    (Residual::ACPowerFlowResidual)(data::ACPowerFlowData, Rv::Vector{Float64}, x::Vector{Float64}, time_step::Int64)

Evaluate the AC power flow residuals and store the result in `Rv` using the provided
state vector `x` and the current time step `time_step`.
The residuals are updated inplace in the struct and additionally copied to the provided array.
This function implements the functor approach for the `ACPowerFlowResidual` struct.
This makes the struct callable.
Calling the `ACPowerFlowResidual` will also update the values of P, Q, V, Θ in the `data` struct.

# Arguments
- `data::ACPowerFlowData`: The grid model data.
- `Rv::Vector{Float64}`: The vector to store the calculated residuals.
- `x::Vector{Float64}`: The state vector.
- `time_step::Int64`: The current time step.
"""
function (Residual::ACPowerFlowResidual)(
    data::ACPowerFlowData,
    Rv::Vector{Float64},
    x::Vector{Float64},
    time_step::Int64,
)
    _update_residual_values!(Residual, x, data, time_step)
    copyto!(Rv, Residual.Rv)
    return
end

"""
    (Residual::ACPowerFlowResidual)(data::ACPowerFlowData, x::Vector{Float64}, time_step::Int64)

Update the AC power flow residuals inplace and store the result in the attribute `Rv` of the struct.
The inputs are the values of state vector `x` and the current time step `time_step`.
This function implements the functor approach for the `ACPowerFlowResidual` struct.
This makes the struct callable.
Calling the `ACPowerFlowResidual` will also update the values of P, Q, V, Θ in the `data` struct.

# Arguments
- `data::ACPowerFlowData`: The grid model data.
- `x::Vector{Float64}`: The state vector values.
- `time_step::Int64`: The current time step.
"""
function (Residual::ACPowerFlowResidual)(
    data::ACPowerFlowData, x::Vector{Float64}, time_step::Int64,
)
    _update_residual_values!(Residual, x, data, time_step)
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

function _load_bus_state!(s::PolarBusState, data::ACPowerFlowData, time_step::Int64)
    copyto!(s.Vm, view(data.bus_magnitude, :, time_step))
    copyto!(s.θ, view(data.bus_angles, :, time_step))
    s.data_stale = false
    return
end

# `data` may have been edited since the last evaluation unless it is waiting on a write-back.
function _sync_from_data!(s::PolarBusState, data::ACPowerFlowData, time_step::Int64)
    s.data_stale || _load_bus_state!(s, data, time_step)
    return
end

function _copy_voltages_to_data!(s::PolarBusState, data::ACPowerFlowData, time_step::Int64)
    copyto!(view(data.bus_magnitude, :, time_step), s.Vm)
    copyto!(view(data.bus_angles, :, time_step), s.θ)
    return
end

"""Write the last evaluated iterate's |V|, θ and REF/PV/PQ injections into `data`'s
`time_step` column, if the fused kernel deferred them. Every polar driver reaches this through
`_finalize_formulation!`."""
function _write_back_bus_state!(
    R::ACPowerFlowResidual,
    data::ACPowerFlowData,
    time_step::Int64,
)
    s = R.bus_state
    s.data_stale || return
    _copy_voltages_to_data!(s, data, time_step)
    bus_types = view(data.bus_type, :, time_step)
    @inbounds for ix in eachindex(bus_types)
        bt = bus_types[ix]
        if bt == PSY.ACBusTypes.PQ || bt == PSY.ACBusTypes.PV || bt == PSY.ACBusTypes.REF
            _setpq(ix, R.P_net, R.Q_net, data, time_step)
        end
    end
    s.data_stale = false
    return
end

"""Hand `data` the iterate at once. Every caller outside the NR loop's fused kernel needs this
contract of the residual functor."""
struct WriteBackNow end
"""Leave `data` stale until [`_write_back_bus_state!`](@ref); only the NR loop uses this."""
struct WriteBackDeferred end

_publish_bus_state!(::WriteBackNow, R::ACPowerFlowResidual, data, time_step::Int64) =
    _write_back_bus_state!(R, data, time_step)

# The LCC, VSC and area tails read |V| and θ from `data`; injections can still wait.
function _publish_bus_state!(
    ::WriteBackDeferred,
    R::ACPowerFlowResidual,
    data::ACPowerFlowData,
    time_step::Int64,
)
    if state_tail_length(data, get_dc_network(data)) > 0
        _copy_voltages_to_data!(R.bus_state, data, time_step)
    end
    return
end

# dispatching on Val for performance reasons.
function _set_state_variables_at_bus!(
    ix::Int,
    P_net::Vector{Float64},
    Q_net::Vector{Float64},
    P_net_set::Vector{Float64},
    P_slack::Float64,
    StateVector::Vector{Float64},
    ::PolarBusState,
    ::Val{PSY.ACBusTypes.REF})
    # When bustype == REFERENCE PSY.ACBus, state variables are Active and Reactive Power Generated
    P_net[ix] = P_net_set[ix] + P_slack
    Q_net[ix] = StateVector[2 * ix]
    return
end

function _set_state_variables_at_bus!(
    ix::Int,
    P_net::Vector{Float64},
    Q_net::Vector{Float64},
    P_net_set::Vector{Float64},
    P_slack::Float64,
    StateVector::Vector{Float64},
    s::PolarBusState,
    ::Val{PSY.ACBusTypes.PV})
    # When bustype == PV, state variables are Reactive Power Generated and Voltage Angle
    # We still update both P and Q values in case the PV bus participates in distributed slack
    P_net[ix] = P_net_set[ix] + P_slack
    Q_net[ix] = StateVector[2 * ix - 1]
    s.θ[ix] = StateVector[2 * ix]
    return
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
    s::PolarBusState,
    ::Val{PSY.ACBusTypes.PQ})
    vm_1 = s.Vm[ix]
    vm_2 = StateVector[2 * ix - 1]
    s.Vm[ix] = vm_2
    s.θ[ix] = StateVector[2 * ix]
    # update P_net and Q_net for ZIP loads
    P_net[ix] +=
        bus_active_constant_I[ix] * (vm_1 - vm_2) +
        bus_active_constant_Z[ix] * (vm_1^2 - vm_2^2)
    Q_net[ix] +=
        bus_reactive_constant_I[ix] * (vm_1 - vm_2) +
        bus_reactive_constant_Z[ix] * (vm_1^2 - vm_2^2)
    return
end

"""
    _update_residual_values!(R::ACPowerFlowResidual, x::Vector{Float64}, data::ACPowerFlowData, time_step::Int64)

Evaluate the polar residual `R.Rv` at `x` (the F-only kernel, used at trial points). Also
writes P, Q, V, Θ for `time_step` into `data`. [`_update_residual_and_jacobian!`](@ref) is the
fused variant that fills the Jacobian in the same sweep.
"""
function _update_residual_values!(
    R::ACPowerFlowResidual,
    x::Vector{Float64},
    data::ACPowerFlowData,
    time_step::Int64,
)
    _update_residual_state!(R, x, data, time_step, WriteBackNow())
    F = R.Rv
    F .= 0.0
    Yb = data.power_network_matrix.data
    Vm = R.bus_state.Vm
    e = R.bus_state.phasor
    Yb_vals = SparseArrays.nonzeros(Yb)
    Yb_rowvals = SparseArrays.rowvals(Yb)
    @inbounds for bus_to in axes(Yb, 1)
        Vm_to = Vm[bus_to]
        e_to = conj(e[bus_to])
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
                # cis(θ_from − θ_to) from the per-bus phasors: no trig per nonzero.
                c = e[bus_from] * e_to
                cosΔθ = real(c)
                sinΔθ = imag(c)
                F[2 * bus_from - 1] += vv * (gb * cosΔθ + bb * sinΔθ)
                F[2 * bus_from] += vv * (gb * sinΔθ - bb * cosΔθ)
            end
        end
    end
    _finish_residual!(R, x, data, time_step)
    return
end

# Reads the state `x` into `R` (P_net/Q_net and `R.bus_state`), hands it to `data` per `mode`,
# writes the LCC and VSC tail states into `data` and refills the per-bus phasors.
function _update_residual_state!(
    R::ACPowerFlowResidual,
    x::Vector{Float64},
    data::ACPowerFlowData,
    time_step::Int64,
    mode::Union{WriteBackNow, WriteBackDeferred},
)
    s = R.bus_state
    _sync_from_data!(s, data, time_step)
    P_net = R.P_net
    Q_net = R.Q_net
    P_net_set = R.P_net_set
    bus_slack_participation_factors = R.bus_slack_participation_factors
    bus_active_constant_I = R.bus_active_constant_I
    bus_reactive_constant_I = R.bus_reactive_constant_I
    bus_active_constant_Z = R.bus_active_constant_Z
    bus_reactive_constant_Z = R.bus_reactive_constant_Z
    num_lcc = size(data.lcc.p_set, 1)
    n_buses_total = first(size(data.bus_type))
    dcn = get_dc_network(data)
    vsc_off = 2 * n_buses_total + 4 * num_lcc
    bus_types = view(data.bus_type, :, time_step)

    for (ref_bus, subnetwork_buses) in R.subnetworks
        slack_scalar = x[2 * ref_bus - 1] - P_net_set[ref_bus]
        n_sub = length(subnetwork_buses)
        # Multi-swing island: each swing self-balances at its own P-slot
        # (P_net = x[2·ix−1]) instead of sharing one distributed island scalar;
        # single-swing islands keep the distributed-slack path unchanged.
        n_ref = 0
        @inbounds for k in 1:n_sub
            bus_types[subnetwork_buses[k]] == PSY.ACBusTypes.REF && (n_ref += 1)
        end
        multi_swing = n_ref > 1
        @inbounds for k in 1:n_sub
            ix = subnetwork_buses[k]
            bt = bus_types[ix]
            if multi_swing && bt == PSY.ACBusTypes.REF
                p_bus_slack = x[2 * ix - 1] - P_net_set[ix]
            else
                p_bus_slack = slack_scalar * bus_slack_participation_factors[ix]
            end
            # creating Val(bt) at runtime is slow, requires allocating: split into cases
            # explicitly, so instead it's Val(compile-time constant).
            if bt == PSY.ACBusTypes.PQ
                _set_state_variables_at_bus!(
                    ix, P_net, Q_net, P_net_set, p_bus_slack, x,
                    bus_active_constant_I, bus_reactive_constant_I,
                    bus_active_constant_Z, bus_reactive_constant_Z,
                    s, Val(PSY.ACBusTypes.PQ),
                )
            elseif bt == PSY.ACBusTypes.PV
                _set_state_variables_at_bus!(
                    ix, P_net, Q_net, P_net_set, p_bus_slack, x,
                    s, Val(PSY.ACBusTypes.PV),
                )
            elseif bt == PSY.ACBusTypes.REF
                _set_state_variables_at_bus!(
                    ix, P_net, Q_net, P_net_set, p_bus_slack, x,
                    s, Val(PSY.ACBusTypes.REF),
                )
            end
        end
    end
    s.data_stale = true
    _publish_bus_state!(mode, R, data, time_step)

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
    _fill_bus_phasor!(s)
    return
end

function _fill_bus_phasor!(s::PolarBusState)
    s.phasor .= cis.(s.θ)
    return
end

# Everything after the Ybus sweep: LCC self-admittances, the −P_net/−Q_net set points, the area
# ΔP coupling and the LCC/VSC/area tail rows. `F`'s bus rows already hold the Ybus injections.
function _finish_residual!(
    R::ACPowerFlowResidual,
    x::Vector{Float64},
    data::ACPowerFlowData,
    time_step::Int64,
)
    F = R.Rv
    P_net = R.P_net
    Q_net = R.Q_net
    num_lcc = size(data.lcc.p_set, 1)
    dcn = get_dc_network(data)
    vsc_off = 2 * first(size(data.bus_type)) + 4 * num_lcc
    Vm = R.bus_state.Vm
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

    # ΔP_a couples into the P-balance row at each area's slack bus. Applied directly to
    # `F` (not folded into `P_net[ix]`) because `P_net[ix]` persists across calls once the
    # slack bus is PQ (ZIP-load dispatch accumulates into it — see
    # `_set_state_variables_at_bus!(::Val{PQ})`); adding ΔP there would double-count after
    # a PV->PQ Q-limit flip. `F` is reset every call, so applying ΔP here is exactly-once,
    # matching the Jacobian's constant `-1.0` stamp at this row.
    if n_controlled_areas(data) > 0
        area_off = area_tail_offset(data, dcn)
        @inbounds for area in data.area_interchange.areas
            ΔP = x[area_off + area.tail_ix]
            # Mirror ΔP_a onto `data` (time_step-indexed, same seam as the LCC tap /
            # VSC tail write-back) so a warm re-solve's `x0` recovers this time step's
            # converged value without contaminating others.
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

"""Union-find and bucket buffers for [`_find_subnetworks_for_reference_buses!`](@ref). `roots`
holds each island's union-find root, then its REF bus; `pool` keeps the bus vectors of earlier
partitions for reuse."""
struct SubnetworkScratch
    uf::Vector{Int}
    group::Vector{Int}
    roots::Vector{Int}
    buses::Vector{Vector{Int}}
    pool::Vector{Vector{Int}}
end

SubnetworkScratch(n_buses::Int) = SubnetworkScratch(
    Vector{Int}(undef, n_buses), Vector{Int}(undef, n_buses), Int[], Vector{Int}[],
    Vector{Int}[])

"""Partition the buses into Ybus islands, keyed by each island's first REF bus, with sorted
members, into `subnetworks`, reusing its bus vectors and `s`; allocates nothing once `s` has
seen as many islands. Warns on islanded buses like `PNM.find_subnetworks` and throws an
`ArgumentError` for an island without a REF bus."""
function _find_subnetworks_for_reference_buses!(
    subnetworks::Dict{Int64, Vector{Int64}},
    s::SubnetworkScratch,
    Ybus::SparseMatrixCSC,
    bus_type::AbstractArray{PSY.ACBusTypes.Value},
)
    rows = SparseArrays.rowvals(Ybus)
    vals = SparseArrays.nonzeros(Ybus)
    uf = s.uf
    for ix in eachindex(bus_type)
        if PNM._live_entry_count(vals, SparseArrays.nzrange(Ybus, ix)) <= 1
            @warn "Bus $ix is islanded"
        end
        uf[ix] = ix
    end
    # Same union order as `PNM.find_subnetworks`, so the roots (named in the error) match.
    for ix in eachindex(bus_type), j in SparseArrays.nzrange(Ybus, ix)
        iszero(vals[j]) || PNM.union_sets!(uf, ix, rows[j])
    end
    empty!(s.buses)
    empty!(s.roots)
    fill!(s.group, 0)
    for ix in eachindex(bus_type)
        root = PNM.get_representative(uf, ix)
        g = s.group[root]
        if iszero(g)
            if isempty(s.pool)
                members = Int[]
            else
                members = empty!(pop!(s.pool))
            end
            push!(s.buses, members)
            push!(s.roots, root)
            g = length(s.buses)
            s.group[root] = g
        end
        push!(s.buses[g], ix)
    end
    # Validate every island before touching `subnetworks`, so a throw leaves it intact.
    for (k, buses) in enumerate(s.buses)
        ref_bus = 0
        for ix in buses
            if bus_type[ix] == PSY.ACBusTypes.REF
                ref_bus = ix
                break
            end
        end
        iszero(ref_bus) && throw(
            ArgumentError(
                "No REF bus found in the subnetwork with $(length(buses)) buses defined by bus key $(s.roots[k])",
            ),
        )
        s.roots[k] = ref_bus
    end
    append!(s.pool, values(subnetworks))
    empty!(subnetworks)
    for (ref_bus, buses) in zip(s.roots, s.buses)
        subnetworks[ref_bus] = buses
    end
    return subnetworks
end

_find_subnetworks_for_reference_buses(
    Ybus::SparseMatrixCSC,
    bus_type::AbstractArray{PSY.ACBusTypes.Value},
) = _find_subnetworks_for_reference_buses!(
    Dict{Int, Vector{Int}}(), SubnetworkScratch(length(bus_type)), Ybus, bus_type)

"""
    _fill_bus_slack_participation_factors!(spf, data, bus_type, subnetworks, time_step)

Write into the dense `spf` the per-bus generator-slack-participation factors (REF and PV buses
only), validate that the sum is positive and no value is negative, and normalize so that each
subnetwork's participating buses sum to 1.
"""
function _fill_bus_slack_participation_factors!(
    spf::Vector{Float64},
    data::ACPowerFlowData,
    bus_type::AbstractVector{PSY.ACBusTypes.Value},
    subnetworks::Dict{Int64, Vector{Int64}},
    time_step::Int64,
)
    fill!(spf, 0.0)
    factors = data.bus_slack_participation_factors
    rows = SparseArrays.rowvals(factors)
    vals = SparseArrays.nonzeros(factors)
    sum_sl_weights = 0.0
    negative = false
    for j in SparseArrays.nzrange(factors, time_step)
        ix = rows[j]
        bt = bus_type[ix]
        if (bt == PSY.ACBusTypes.REF || bt == PSY.ACBusTypes.PV) && !iszero(vals[j])
            spf[ix] = vals[j]
            sum_sl_weights += vals[j]
            negative |= vals[j] < 0.0
        end
    end
    iszero(sum_sl_weights) &&
        throw(ArgumentError("sum of slack_participation_factors cannot be zero"))
    negative && throw(ArgumentError("slack_participation_factors cannot be negative"))
    for subnetwork_buses in values(subnetworks)
        # Multi-swing island: each swing carries its own slack (see
        # `_update_residual_values!`); spreading slack onto non-swing buses is undefined
        # there, so reject it.
        n_ref_sub = count(ix -> bus_type[ix] == PSY.ACBusTypes.REF, subnetwork_buses)
        if n_ref_sub > 1
            any(
                ix -> bus_type[ix] != PSY.ACBusTypes.REF && !iszero(spf[ix]),
                subnetwork_buses,
            ) && throw(
                ArgumentError(
                    "distributed slack (participation factors on non-swing buses) is not " *
                    "supported in an island containing $n_ref_sub swing (REF) buses: each " *
                    "swing carries its own slack. Reduce the island to a single swing or " *
                    "remove the non-swing participation factors.",
                ),
            )
        end
        sum_bspf = 0.0
        for ix in subnetwork_buses
            sum_bspf += spf[ix]
        end
        iszero(sum_bspf) && throw(
            ArgumentError(
                "sum of slack_participation_factors per subnetwork cannot be zero",
            ),
        )
        for ix in subnetwork_buses
            spf[ix] /= sum_bspf
        end
    end
    return
end

"""
    _build_bus_slack_participation_factors(data, bus_type, subnetworks, time_step)

[`_fill_bus_slack_participation_factors!`](@ref) as a `SparseVector{Float64, Int}` of length
`n_buses`, for the rectangular current-injection and mixed CPB residuals.
"""
function _build_bus_slack_participation_factors(
    data::ACPowerFlowData,
    bus_type::AbstractVector{PSY.ACBusTypes.Value},
    subnetworks::Dict{Int64, Vector{Int64}},
    time_step::Int64,
)
    spf = zeros(length(bus_type))
    _fill_bus_slack_participation_factors!(spf, data, bus_type, subnetworks, time_step)
    return SparseArrays.sparse(spf)
end
