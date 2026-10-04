"""
    struct ACPowerFlowJacobian

A struct that represents the Jacobian matrix for AC power flow calculations.

This struct uses the functor pattern, meaning instances of `ACPowerFlowJacobian` store the data (Jacobian matrix) internally
and can be called as a function at the same time. Calling the instance as a function updates the stored Jacobian matrix.

Does not store the grid model `data`: it is threaded explicitly through the functor and
constructor calls instead, to avoid a reference cycle through `data.polar_nr_cache` (see
`PolarNRCache`).

# Fields
- `Jv::SparseArrays.SparseMatrixCSC{Float64, $J_INDEX_TYPE}`: The Jacobian matrix, which is updated by `_update_jacobian_matrix_values!`.
- `bus_slack_participation_factors::Vector{Float64}`: Normalized per-bus slack participation factors for the current time step (the `ACPowerFlowResidual`'s vector, shared). Used for the distributed slack Jacobian entries.
- `subnetworks::Dict{Int64, Vector{Int64}}`: Subnetwork mapping from REF bus to bus list (from the `ACPowerFlowResidual`). Used for the distributed slack Jacobian entries.
- `independent_ref::Set{Int}`: Multi-swing REF bus indices, from `_multi_swing_ref_indices`. Recomputed in place by `_refresh_polar_residual!` when the partition or the REF set changes.
- `slack_jnz::Vector{Int32}`: per bus, the `nonzeros(Jv)` offset of `∂F_P/∂x[2·ref−1]` for its island's REF, or 0 when the pattern has no such slot (and for the REF itself). See `_slack_jnz!`.
- `bus_state::PolarBusState`: per-bus |V|, θ and `cis(θ)` (the residual's, shared).
"""
struct ACPowerFlowJacobian
    Jv::SparseArrays.SparseMatrixCSC{Float64, J_INDEX_TYPE}  # This is the Jacobian matrix, updated in place by `_update_jacobian_matrix_values!`
    bus_slack_participation_factors::Vector{Float64}
    subnetworks::Dict{Int64, Vector{Int64}}
    independent_ref::Set{Int}
    slack_jnz::Vector{Int32}
    bus_active_constant_I::Vector{Float64}
    bus_reactive_constant_I::Vector{Float64}
    bus_active_constant_Z::Vector{Float64}
    bus_reactive_constant_Z::Vector{Float64}
    bus_state::PolarBusState
    # nzval-offset caches built once at construction; see _build_polar_nz_caches.
    # Off-diagonal Ybus entries are grouped by row (bus_from), so the fill keeps each bus's
    # diagonal sums in registers and writes nonzeros(Jv) directly. Int32 on every platform
    # halves the kernel's index traffic; a J too large for it errors at the conversion.
    od_ptr::Vector{Int32}           # n_buses + 1: entries of bus_from i are od_ptr[i]:od_ptr[i+1]-1
    od_to::Vector{Int32}            # bus_to per off-diagonal Ybus entry
    od_ybus_nz::Vector{Int32}       # nonzeros(Yb) index for that entry (g, b)
    od_jnz::Matrix{Int32}           # 4 × n_od: J nzval offsets for (p,vm),(q,vm),(p,va),(q,va)
    diag_jnz::Matrix{Int32}         # 4 × n_buses: J nzval offsets for the self block, same slot order
    diag_ybus_nz::Vector{Int32}     # n_buses: nonzeros(Yb) index for Yb[i,i]
end

"""
    (J::ACPowerFlowJacobian)(data::ACPowerFlowData, time_step::Int64)

Update the Jacobian matrix `Jv` using `_update_jacobian_matrix_values!` and the provided data and time step.

Defining this method allows an instance of `ACPowerFlowJacobian` to be called as a function, following the functor pattern.

# Arguments
- `data::ACPowerFlowData`: The grid model data used for power flow calculations.
- `time_step::Int64`: The time step for the calculations.

# Example
```julia
residual = ACPowerFlowResidual(data, time_step)
J = ACPowerFlowJacobian(data, residual, time_step)
J(data, time_step)  # Updates the Jacobian matrix Jv
```
"""
function (J::ACPowerFlowJacobian)(data::ACPowerFlowData, time_step::Int64)
    _sync_from_data!(J.bus_state, data, time_step)
    _fill_bus_phasor!(J.bus_state)
    _update_jacobian_matrix_values!(J, data, time_step)
    return
end

"""
    (J::ACPowerFlowJacobian)(data::ACPowerFlowData, J::SparseArrays.SparseMatrixCSC{Float64, $J_INDEX_TYPE}, time_step::Int64)

Use the `ACPowerFlowJacobian` to update the provided Jacobian matrix `J` inplace.

Update the internally stored Jacobian matrix `Jv` using `_update_jacobian_matrix_values!` and the provided data and time step, and write the updated Jacobian values to `J`.

This method allows an instance of ACPowerFlowJacobian to be called as a function, following the functor pattern.

# Arguments
- `data::ACPowerFlowData`: The grid model data used for power flow calculations.
- `Jv::SparseArrays.SparseMatrixCSC{Float64, $J_INDEX_TYPE}`: A sparse matrix to be updated with new values of the Jacobian matrix.
- `time_step::Int64`: The time step for the calculations.

# Example
```julia
residual = ACPowerFlowResidual(data, time_step)
J = ACPowerFlowJacobian(data, residual, time_step)
Jv = SparseArrays.sparse(Float64[], J_INDEX_TYPE[], J_INDEX_TYPE[])
J(data, Jv, time_step)  # Updates the Jacobian matrix Jv and writes it to J
```
"""
function (J::ACPowerFlowJacobian)(
    data::ACPowerFlowData,
    Jv::SparseArrays.SparseMatrixCSC{Float64, J_INDEX_TYPE},
    time_step::Int64,
)
    J(data, time_step)
    copyto!(Jv, J.Jv)
    return
end

"""
    ACPowerFlowJacobian(data::ACPowerFlowData, residual::ACPowerFlowResidual, time_step::Int64) -> ACPowerFlowJacobian

Constructor for `ACPowerFlowJacobian`. The returned instance has its sparsity
pattern initialized and shares the residual's slack-participation, subnetwork,
and ZIP-coefficient caches — the residual must be constructed first against the
same `data` and `time_step`.

# Arguments
- `data::ACPowerFlowData`: The grid model data used for power flow calculations.
- `residual::ACPowerFlowResidual`: The companion residual; supplies
  `bus_slack_participation_factors`, `subnetworks`, and the per-bus ZIP load
  coefficient vectors.
- `time_step::Int64`: The time step for the calculations.

# Example
```julia
residual = ACPowerFlowResidual(data, time_step)
J = ACPowerFlowJacobian(data, residual, time_step)
J(data, time_step)  # Updates the Jacobian matrix stored internally in J.
J.Jv  # Access the Jacobian matrix stored internally in J.
```
"""
# The distributed-slack slots that the Ybus pattern lacks: `(bus_k, ref)` for each bus with a
# nonzero participation factor in `data` that is not a neighbor of its island's REF. Bus type is
# ignored, so a PV→PQ flip keeps the pattern. The polar J pattern depends only on these slots,
# the Ybus pattern (network edits write zeros, so it does not change) and the LCC/VSC/area data.
# The result is sorted, so it compares as a cache key.
function _extra_slack_slots(
    data::ACPowerFlowData,
    subnetworks::Dict{Int64, Vector{Int64}},
    time_step::Int64,
)
    factors = data.bus_slack_participation_factors
    slots = Tuple{Int, Int}[]
    for (ref_bus, subnetwork_buses) in subnetworks
        for bus_k in subnetwork_buses
            iszero(factors[bus_k, time_step]) && continue
            ref_bus in data.neighbors[bus_k] && continue
            push!(slots, (bus_k, ref_bus))
        end
    end
    return sort!(slots)
end

"""Fill `slack_jnz[k]` with the `nonzeros(Jv)` offset of `(2k−1, 2·ref−1)` for every bus `k` of
each island other than its REF `ref`, or 0 when `Jv` has no such slot. The slots it pointed to
before are zeroed first: after a partition change nothing else writes a slot to a bus's old REF."""
function _slack_jnz!(
    slack_jnz::Vector{Int32},
    Jv::SparseMatrixCSC{Float64, J_INDEX_TYPE},
    subnetworks::Dict{Int64, Vector{Int64}},
)
    Jvnz = SparseArrays.nonzeros(Jv)
    for o in slack_jnz
        if !iszero(o)
            Jvnz[o] = 0.0
        end
    end
    fill!(slack_jnz, 0)
    rowvals = SparseArrays.rowvals(Jv)
    for (ref_bus, subnetwork_buses) in subnetworks
        rng = SparseArrays.nzrange(Jv, 2 * ref_bus - 1)
        for bus_k in subnetwork_buses
            bus_k == ref_bus && continue
            row = 2 * bus_k - 1
            k = searchsortedfirst(view(rowvals, rng), row)
            if k <= length(rng) && rowvals[rng[k]] == row
                slack_jnz[bus_k] = rng[k]
            end
        end
    end
    return
end

"""Whether `J`'s pattern holds a distributed-slack slot for every participating bus outside the
multi-swing islands."""
function _slack_slots_cover(
    slack_jnz::Vector{Int32},
    bus_slack_participation_factors::Vector{Float64},
    subnetworks::Dict{Int64, Vector{Int64}},
    independent_ref::Set{Int},
)
    for (ref_bus, subnetwork_buses) in subnetworks
        ref_bus in independent_ref && continue
        for bus_k in subnetwork_buses
            if bus_k != ref_bus && !iszero(bus_slack_participation_factors[bus_k]) &&
               iszero(slack_jnz[bus_k])
                return false
            end
        end
    end
    return true
end

# Memoize the expensive Jacobian sparse-structure build (~3.2 MB on 2000 buses) so it is built
# once and reused across the Q-limit loop, repeated solves and value-only network edits. The key
# is the network-matrix identity and the area interchange data. The cached slack slots must cover
# `_extra_slack_slots`, so an outage that drops a slot still reuses the structure. Returns a full
# `copy` so each `ACPowerFlowJacobian` owns its buffer, or `nothing` to signal a rebuild. Lives
# in its own `data.ac_jacobian_structure_cache` field ([`ACJacobianStructureCache`](@ref)) so it
# never collides with the FastDecoupled/DC caches in `data.solver_cache[]`.
_reuse_ac_jac_structure(::Nothing, matrix, slots, area_data) = nothing
function _reuse_ac_jac_structure(e::ACJacobianStructureCache, matrix, slots, area_data)
    if e.matrix === matrix && e.area_data === area_data && issubset(slots, e.slack_slots)
        return copy(e.structure)
    end
    return nothing
end

function _get_or_build_jacobian_structure(
    data::ACPowerFlowData,
    subnetworks::Dict{Int64, Vector{Int64}},
    time_step::Int64,
)
    slots = _extra_slack_slots(data, subnetworks, time_step)
    reused = _reuse_ac_jac_structure(
        data.ac_jacobian_structure_cache[], data.power_network_matrix, slots,
        data.area_interchange)
    isnothing(reused) || return reused
    Jv0 = _create_jacobian_matrix_structure(data, slots)
    # Cache a pristine copy; `Jv0` is about to be mutated by the Newton loop. `area_data`
    # is stored by IDENTITY (not copied) — a rebuilt `PowerFlowData` gets a fresh
    # `AreaInterchangeData`, forcing a rebuild; a Q-limit flip keeps the same object, so
    # reuse still works.
    data.ac_jacobian_structure_cache[] =
        ACJacobianStructureCache(
            data.power_network_matrix, slots, copy(Jv0), data.area_interchange)
    return Jv0
end

function ACPowerFlowJacobian(
    data::ACPowerFlowData,
    residual::ACPowerFlowResidual,
    time_step::Int64,
)
    Jv0 = _get_or_build_jacobian_structure(data, residual.subnetworks, time_step)
    od_ptr, od_to, od_ybus_nz, od_jnz, diag_jnz, diag_ybus_nz =
        _build_polar_nz_caches(data, Jv0)
    slack_jnz = zeros(Int32, length(residual.bus_slack_participation_factors))
    _slack_jnz!(slack_jnz, Jv0, residual.subnetworks)
    return ACPowerFlowJacobian(
        Jv0,
        residual.bus_slack_participation_factors,
        residual.subnetworks,
        _multi_swing_ref_indices(data.bus_type, residual.subnetworks, time_step),
        slack_jnz,
        residual.bus_active_constant_I,
        residual.bus_reactive_constant_I,
        residual.bus_active_constant_Z,
        residual.bus_reactive_constant_Z,
        residual.bus_state,
        od_ptr,
        od_to,
        od_ybus_nz,
        od_jnz,
        diag_jnz,
        diag_ybus_nz,
    )
end

"""
Build the once-per-construction nzval-offset caches that drive the polar
Jacobian fill. Off-diagonal Ybus entries `Yb[bus_from, bus_to]` are grouped by
`bus_from` (CSR order, `od_ptr`), so the fill sums each bus's diagonal terms in
registers. For each off-diagonal entry we record the four `J.Jv` nzval offsets of
its 2×2 block; for each diagonal we record the self block offsets and the
`Yb[i,i]` position.
"""
function _build_polar_nz_caches(
    data::ACPowerFlowData,
    Jv::SparseArrays.SparseMatrixCSC{Float64, J_INDEX_TYPE},
)
    Yb = data.power_network_matrix.data
    num_buses = first(size(data.bus_type))
    Yrows = SparseArrays.rowvals(Yb)
    od_ptr = zeros(Int32, num_buses + 1)
    for bus_to in 1:num_buses
        for j in SparseArrays.nzrange(Yb, bus_to)
            bus_from = Yrows[j]
            bus_from != bus_to && (od_ptr[bus_from + 1] += 1)
        end
    end
    od_ptr[1] = 1
    for i in 1:num_buses
        od_ptr[i + 1] += od_ptr[i]
    end
    n_od = od_ptr[end] - 1
    od_to = Vector{Int32}(undef, n_od)
    od_ybus_nz = Vector{Int32}(undef, n_od)
    od_jnz = Matrix{Int32}(undef, 4, n_od)
    diag_jnz = Matrix{Int32}(undef, 4, num_buses)
    diag_ybus_nz = zeros(Int32, num_buses)  # 0 = no Yb[i,i] nonzero (self-admittance ≡ 0)
    # Diagonal J slots always exist (neighbors include self); fill them even if the
    # bus has no Ybus diagonal entry, matching the old getindex-returns-0 behavior.
    for bus_from in 1:num_buses
        col_vm = 2 * bus_from - 1
        col_va = 2 * bus_from
        diag_jnz[1, bus_from] = _jv_nz_index(Jv, 2 * bus_from - 1, col_vm)
        diag_jnz[2, bus_from] = _jv_nz_index(Jv, 2 * bus_from, col_vm)
        diag_jnz[3, bus_from] = _jv_nz_index(Jv, 2 * bus_from - 1, col_va)
        diag_jnz[4, bus_from] = _jv_nz_index(Jv, 2 * bus_from, col_va)
    end
    next = od_ptr[1:num_buses]
    for bus_to in 1:num_buses
        col_to_vm = 2 * bus_to - 1
        col_to_va = 2 * bus_to
        for j in SparseArrays.nzrange(Yb, bus_to)
            bus_from = Yrows[j]
            row_from_p = 2 * bus_from - 1
            row_from_q = 2 * bus_from
            if bus_from == bus_to
                diag_ybus_nz[bus_from] = j
            else
                k = next[bus_from]
                next[bus_from] += 1
                od_to[k] = bus_to
                od_ybus_nz[k] = j
                od_jnz[1, k] = _jv_nz_index(Jv, row_from_p, col_to_vm)
                od_jnz[2, k] = _jv_nz_index(Jv, row_from_q, col_to_vm)
                od_jnz[3, k] = _jv_nz_index(Jv, row_from_p, col_to_va)
                od_jnz[4, k] = _jv_nz_index(Jv, row_from_q, col_to_va)
            end
        end
    end
    return od_ptr, od_to, od_ybus_nz, od_jnz, diag_jnz, diag_ybus_nz
end

"""
Create the Jacobian matrix structure for a reference bus (REF). Currently unused: we \
fill all four values even for PV buses with structiural zeros using the same function as for PQ buses.
"""
function _create_jacobian_matrix_structure_bus!(rows::Vector{J_INDEX_TYPE},
    columns::Vector{J_INDEX_TYPE},
    values::Vector{Float64},
    bus_from::Int,
    bus_to::Int,
    row_from_p::Int,
    row_from_q::Int,
    col_to_vm::Int,
    col_to_va::Int,
    ::Val{PSY.ACBusTypes.REF})
    if bus_from == bus_to
        # Active PF w/r Local Active Power
        push!(rows, row_from_p)
        push!(columns, col_to_vm)
        push!(values, 0.0)
        # Reactive PF w/r Local Reactive Power
        push!(rows, row_from_q)
        push!(columns, col_to_va)
        push!(values, 0.0)
    end
    return
end

"""
Create the Jacobian matrix structure for a PV bus. Currently unused: we \
fill all four values even for PV buses with structiural zeros using the same function as for PQ buses.
"""
function _create_jacobian_matrix_structure_bus!(rows::Vector{J_INDEX_TYPE},
    columns::Vector{J_INDEX_TYPE},
    values::Vector{Float64},
    bus_from::Int,
    bus_to::Int,
    row_from_p::Int,
    row_from_q::Int,
    col_to_vm::Int,
    col_to_va::Int,
    ::Val{PSY.ACBusTypes.PV})
    # Active PF w/r Voltage Angle
    push!(rows, row_from_p)
    push!(columns, col_to_va)
    push!(values, 0.0)
    # Reactive PF w/r Voltage Angle
    push!(rows, row_from_q)
    push!(columns, col_to_va)
    push!(values, 0.0)
    if bus_from == bus_to
        # Reactive PF w/r Local Reactive Power
        push!(rows, row_from_q)
        push!(columns, col_to_vm)
        push!(values, 0.0)
    end
    return
end

"""
Create the Jacobian matrix structure for a PQ bus. Using this for all buses because
    a) for REF buses it doesn't matter if there are 2 values or 4 values - there are not many of them in the grid
    b) for PV buses we fill all four values because we can have a PV -> PQ transition and then we need to fill all four values
"""
function _create_jacobian_matrix_structure_bus!(rows::Vector{J_INDEX_TYPE},
    columns::Vector{J_INDEX_TYPE},
    values::Vector{Float64},
    bus_from::Int,
    bus_to::Int,
    row_from_p::Int,
    row_from_q::Int,
    col_to_vm::Int,
    col_to_va::Int,
    # ::Val{PSY.ACBusTypes.PQ}
)
    # Active PF w/r Voltage Magnitude
    push!(rows, row_from_p)
    push!(columns, col_to_vm)
    push!(values, 0.0)
    # Reactive PF w/r Voltage Magnitude
    push!(rows, row_from_q)
    push!(columns, col_to_vm)
    push!(values, 0.0)
    # Active PF w/r Voltage Angle
    push!(rows, row_from_p)
    push!(columns, col_to_va)
    push!(values, 0.0)
    # Reactive PF w/r Voltage Angle
    push!(rows, row_from_q)
    push!(columns, col_to_va)
    push!(values, 0.0)
    return
end

"""
    _create_jacobian_matrix_structure_lcc(
        data::ACPowerFlowData,
        rows::Vector{$J_INDEX_TYPE},
        columns::Vector{$J_INDEX_TYPE},
        values::Vector{Float64},
        num_buses::Int
    )

Create the Jacobian matrix structure for LCC HVDC systems.

# Description

The function iterates over each LCC system and adds the non-zero entries to the Jacobian matrix structure.
The state vector for every LCC contains 4 variables: tap position and thyristor angle for both the rectifier and inverter sides.
The indices of non-zero entries correspond to the positions of these variables in the extended state vector.

For an LCC system connecting bus ``i`` (rectifier side) and bus ``j`` (inverter side), the state variables are:
- ``t_i``: tap position at rectifier
- ``t_j``: tap position at inverter
- ``\\alpha_i``: thyristor angle at rectifier
- ``\\alpha_j``: thyristor angle at inverter

The residuals include:
- ``F_{t_i}``: Active power balance at rectifier (controls ``P_i`` to match setpoint)
- ``F_{t_j}``: Total active power balance across LCC system
- ``F_{\\alpha_i}``: Rectifier thyristor angle constraint (maintains ``\\alpha_i`` at minimum)
- ``F_{\\alpha_j}``: Inverter thyristor angle constraint (maintains ``\\alpha_j`` at minimum)

# Example Structure

For a system with 2 buses connected by one LCC where bus 1 is the rectifier side and bus 2 is the inverter side,
the Jacobian matrix would have non-zero entries at positions like:

```math
\\begin{array}{c|cccccccc}
 & V_1 & \\delta_1 & V_2 & \\delta_2 & t_1 & t_2 & \\alpha_1 & \\alpha_2 \\\\
\\hline
P_1 & \\frac{\\partial P_1}{\\partial V_1} & & & & \\frac{\\partial P_1}{\\partial t_1} & & \\frac{\\partial P_1}{\\partial \\alpha_1} & \\\\
Q_1 & \\frac{\\partial Q_1}{\\partial V_1} & & & & \\frac{\\partial Q_1}{\\partial t_1} & & \\frac{\\partial Q_1}{\\partial \\alpha_1} & \\\\
P_2 & & & & & & & & \\\\
Q_2 & & & & & & & & \\\\
F_{t_1} & \\frac{\\partial F_{t_1}}{\\partial V_1} & & & & \\frac{\\partial F_{t_1}}{\\partial t_1} & & \\frac{\\partial F_{t_1}}{\\partial \\alpha_1} & \\\\
F_{t_2} & \\frac{\\partial F_{t_2}}{\\partial V_1} & & \\frac{\\partial F_{t_2}}{\\partial V_2} & & \\frac{\\partial F_{t_2}}{\\partial t_1} & \\frac{\\partial F_{t_2}}{\\partial t_2} & \\frac{\\partial F_{t_2}}{\\partial \\alpha_1} & \\frac{\\partial F_{t_2}}{\\partial \\alpha_2} \\\\
F_{\\alpha_1} & & & & & & & \\frac{\\partial F_{\\alpha_1}}{\\partial \\alpha_1} & \\\\
F_{\\alpha_2} & & & & & & & & \\frac{\\partial F_{\\alpha_2}}{\\partial \\alpha_2}
\\end{array}
```

This function sets up the indices of these non-zero entries in the sparse Jacobian matrix structure.

# Arguments
- `data::ACPowerFlowData`: The power flow data containing LCC system information.
- `rows::Vector{$J_INDEX_TYPE}`: Vector to store row indices of non-zero Jacobian entries.
- `columns::Vector{$J_INDEX_TYPE}`: Vector to store column indices of non-zero Jacobian entries.
- `values::Vector{Float64}`: Vector to store initial values of non-zero Jacobian entries.
- `num_buses::Int`: Total number of buses in the system.
"""
function _create_jacobian_matrix_structure_lcc(
    data::ACPowerFlowData,
    rows::Vector{J_INDEX_TYPE},
    columns::Vector{J_INDEX_TYPE},
    values::Vector{Float64},
    num_buses::Int,
)
    for (i, (fb, tb)) in enumerate(data.lcc.bus_indices)
        idx_p_fb = 2 * fb - 1
        idx_q_fb = 2 * fb
        idx_p_tb = 2 * tb - 1
        idx_q_tb = 2 * tb
        offset_lcc = num_buses * 2 + (i - 1) * 4
        idx_tap_from = offset_lcc + 1
        idx_tap_to = offset_lcc + 2
        idx_angle_from = offset_lcc + 3
        idx_angle_to = offset_lcc + 4

        rcv = [
            (idx_p_fb, idx_p_fb, 0.0),  # ∂Pᵢ/∂Vᵢ
            (idx_q_fb, idx_p_fb, 0.0),  # ∂Qᵢ/∂Vᵢ
            (idx_p_fb, idx_tap_from, 0.0),  # ∂Pᵢ/∂tᵢ
            (idx_p_fb, idx_angle_from, 0.0),  # ∂Pᵢ/∂αᵢ
            (idx_q_fb, idx_tap_from, 0.0),  # ∂Qᵢ/∂tᵢ
            (idx_q_fb, idx_angle_from, 0.0),  # ∂Qᵢ/∂αᵢ
            (idx_p_tb, idx_p_tb, 0.0),  # ∂Pⱼ/∂Vⱼ
            (idx_q_tb, idx_p_tb, 0.0),  # ∂Qⱼ/∂Vⱼ
            (idx_p_tb, idx_tap_to, 0.0),  # ∂Pⱼ/∂tⱼ
            (idx_p_tb, idx_angle_to, 0.0),  # ∂Pⱼ/∂αⱼ
            (idx_q_tb, idx_tap_to, 0.0),  # ∂Qⱼ/∂tⱼ
            (idx_q_tb, idx_angle_to, 0.0),  # ∂Qⱼ/∂αⱼ
            (idx_tap_from, idx_p_fb, 0.0),  # ∂Fₜᵢ/∂Vᵢ
            (idx_tap_to, idx_p_fb, 0.0),  # ∂Fₜⱼ/∂Vᵢ
            (idx_tap_to, idx_p_tb, 0.0),  # ∂Fₜⱼ/∂Vⱼ
            # Inverter-side slots for the P-setpoint row F_t_fb, used when the
            # set point is at the inverter (F_t_fb = −P_lcc_to − P_set).
            (idx_tap_from, idx_p_tb, 0.0),  # ∂Fₜᵢ/∂Vⱼ
            (idx_tap_from, idx_tap_to, 0.0),  # ∂Fₜᵢ/∂tⱼ
            (idx_tap_from, idx_angle_to, 0.0),  # ∂Fₜᵢ/∂αⱼ
            (idx_tap_from, idx_tap_from, 0.0),  # ∂Fₜᵢ/∂tᵢ
            (idx_tap_from, idx_angle_from, 0.0),  # ∂Fₜᵢ/∂αᵢ
            (idx_tap_to, idx_tap_from, 0.0),  # ∂Fₜⱼ/∂tᵢ
            (idx_tap_to, idx_tap_to, 0.0),  # ∂Fₜⱼ/∂tⱼ
            (idx_tap_to, idx_angle_from, 0.0),  # ∂Fₜⱼ/∂αᵢ
            (idx_tap_to, idx_angle_to, 0.0),  # ∂Fₜⱼ/∂αⱼ
            (idx_angle_from, idx_angle_from, 1.0),  # ∂Fₐᵢ/∂αᵢ
            (idx_angle_to, idx_angle_to, 1.0),  # ∂Fₐⱼ/∂αⱼ
        ]
        for (r, c, v) in rcv
            push!(rows, r)
            push!(columns, c)
            push!(values, v)
        end
    end
    return
end

"""
    _create_jacobian_matrix_structure(data::ACPowerFlowData, extra_slack_slots::Vector{Tuple{Int, Int}}) -> SparseMatrixCSC{Float64, $J_INDEX_TYPE}

Create the structure of the Jacobian matrix for an AC power flow problem.

# Arguments
- `data::ACPowerFlowData`: The power flow model.
- `extra_slack_slots::Vector{Tuple{Int, Int}}`: The distributed-slack slots the Ybus pattern lacks, from `_extra_slack_slots`.

# Returns
- `SparseMatrixCSC{Float64, $J_INDEX_TYPE}`: A sparse matrix with structural zeros representing the structure of the Jacobian matrix.

# Description

This function initializes the structure of the Jacobian matrix for an AC power flow problem.
The Jacobian matrix is used in power flow analysis to represent the partial derivatives of bus active and reactive power injections with respect to bus voltage magnitudes and angles.

Unlike some commonly used approaches where the Jacobian matrix is constructed as four submatrices, each grouping values for the four types of partial derivatives,
this function groups the partial derivatives by bus. The structure is organized as groups of 4 values per bus.

This approach is more memory-efficient. Furthermore, this structure results in a more efficient factorization because the values are more likely to be grouped close to the diagonal.
Refer to Electric Energy Systems: Analysis and Operation by Antonio Gomez-Exposito and Fernando L. Alvarado for more details.

For each bus in the system, the function iterates over its neighboring buses and determines the type of each neighboring bus (`REF`, `PV`, or `PQ`).
Depending on the bus type, the function adds the appropriate entries to the Jacobian matrix structure.

- For `REF` buses, entries are added for local active and reactive power.
- For `PV` buses, entries are added for active and reactive power with respect to angle, and for local reactive power.
- For `PQ` buses, entries are added for active and reactive power with respect to voltage magnitude and angle.

# Example Structure

For a system with 3 buses where bus 1 is `REF`, bus 2 is `PV`, and bus 3 is `PQ`:

Let ``\\Delta P_j``, ``\\Delta Q_j`` be the active, reactive power balance at the ``j``th bus. Let ``P_j`` and ``Q_j`` be the
active and reactive power generated at the ``j``th bus (`REF` and `PV` only). The state vector is
``x = [P_1, Q_1, Q_2, \\theta_2, V_3, \\theta_3]``, and the residual vector is ``F(x) = [\\Delta P_1, \\Delta Q_1, \\Delta P_2, \\Delta Q_2, \\Delta P_3, \\Delta Q_3]``.

The Jacobian matrix ``J = \\nabla F(x)`` has the structure:

```math
J = \\begin{bmatrix}
\\frac{\\partial \\vec{F}}{\\partial P_1} &
\\frac{\\partial \\vec{F}}{\\partial Q_1} &
\\frac{\\partial \\vec{F}}{\\partial Q_2} &
\\frac{\\partial \\vec{F}}{\\partial \\theta_2} &
\\frac{\\partial \\vec{F}}{\\partial V_3} &
\\frac{\\partial \\vec{F}}{\\partial \\theta_3}
\\end{bmatrix}
```

In reality, for large networks, this matrix would be sparse, and each 2×2 block would only be nonzero
when there's a line between the respective buses.

The function writes the bus blocks directly into CSC arrays (`_bus_block_pattern`),
assembles the few slack, LCC, VSC and area tail entries with `sparse`, and merges the two
(`_merge_patterns`).
"""
function _create_jacobian_matrix_structure(
    data::ACPowerFlowData,
    extra_slack_slots::Vector{Tuple{Int, Int}},
)
    rows = J_INDEX_TYPE[]
    columns = J_INDEX_TYPE[]
    values = Float64[]
    num_buses = first(size(data.bus_type))
    # Distributed slack: each participating bus k has ∂F_P_k/∂x[2*ref-1] = -c_k, a slot the
    # Ybus pattern lacks when k is not a neighbor of the ref bus.
    for (bus_k, ref_bus) in extra_slack_slots
        push!(rows, J_INDEX_TYPE(2 * bus_k - 1))
        push!(columns, J_INDEX_TYPE(2 * ref_bus - 1))
        push!(values, 0.0)
    end
    _create_jacobian_matrix_structure_lcc(data, rows, columns, values, num_buses)
    _create_jacobian_matrix_structure_vsc(data, rows, columns, values, num_buses)
    _create_jacobian_matrix_structure_area(data, rows, columns, values)
    if isempty(rows)
        return _bus_block_pattern(data.neighbors, num_buses, 2 * num_buses)
    end
    m = max(2 * num_buses, Int(maximum(rows)))
    n = max(2 * num_buses, Int(maximum(columns)))
    return _merge_patterns(
        _bus_block_pattern(data.neighbors, num_buses, n),
        SparseArrays.sparse(rows, columns, values, m, n),
    )
end

# The all-PQ 2×2 bus blocks of every `neighbors` pair as an all-zero CSC with `n` columns: a
# bus pair puts rows (2f-1, 2f) in columns (2t-1, 2t) for each t in neighbors[f]. Scanning f in
# order fills each column's rows already sorted, so no sort or duplicate pass is needed.
function _bus_block_pattern(neighbors::Vector{Set{Int}}, num_buses::Int, n::Int)
    count = zeros(Int, num_buses)
    for bus_from in 1:num_buses, bus_to in neighbors[bus_from]
        count[bus_to] += 1
    end
    colptr = Vector{J_INDEX_TYPE}(undef, n + 1)
    colptr[1] = 1
    for bus_to in 1:num_buses
        colptr[2 * bus_to] = colptr[2 * bus_to - 1] + 2 * count[bus_to]
        colptr[2 * bus_to + 1] = colptr[2 * bus_to] + 2 * count[bus_to]
    end
    colptr[(2 * num_buses + 2):end] .= colptr[2 * num_buses + 1]
    nnz_total = colptr[2 * num_buses + 1] - 1
    rowval = Vector{J_INDEX_TYPE}(undef, nnz_total)
    next = colptr[1:2:(2 * num_buses)]
    for bus_from in 1:num_buses, bus_to in neighbors[bus_from]
        p = next[bus_to]
        q = p + 2 * count[bus_to]
        rowval[p] = 2 * bus_from - 1
        rowval[p + 1] = 2 * bus_from
        rowval[q] = 2 * bus_from - 1
        rowval[q + 1] = 2 * bus_from
        next[bus_to] = p + 2
    end
    return SparseArrays.SparseMatrixCSC(
        2 * num_buses, n, colptr, rowval, zeros(Float64, nnz_total))
end

# Union of two sorted, duplicate-free CSC patterns with `size(B)` (`A` may have fewer rows),
# summing the values where both hold an entry, as `sparse` combines duplicates.
function _merge_patterns(
    A::SparseArrays.SparseMatrixCSC{Float64, J_INDEX_TYPE},
    B::SparseArrays.SparseMatrixCSC{Float64, J_INDEX_TYPE},
)
    m, n = size(B)
    ra, va = SparseArrays.rowvals(A), SparseArrays.nonzeros(A)
    rb, vb = SparseArrays.rowvals(B), SparseArrays.nonzeros(B)
    colptr = Vector{J_INDEX_TYPE}(undef, n + 1)
    rowval = Vector{J_INDEX_TYPE}(undef, SparseArrays.nnz(A) + SparseArrays.nnz(B))
    nzval = Vector{Float64}(undef, length(rowval))
    colptr[1] = 1
    p = 1
    for j in 1:n
        ia, ea = Int(A.colptr[j]), Int(A.colptr[j + 1])
        ib, eb = Int(B.colptr[j]), Int(B.colptr[j + 1])
        while ia < ea || ib < eb
            if ib == eb || (ia < ea && ra[ia] < rb[ib])
                rowval[p], nzval[p] = ra[ia], va[ia]
                ia += 1
            elseif ia == ea || rb[ib] < ra[ia]
                rowval[p], nzval[p] = rb[ib], vb[ib]
                ib += 1
            else
                rowval[p], nzval[p] = ra[ia], va[ia] + vb[ib]
                ia += 1
                ib += 1
            end
            p += 1
        end
        colptr[j + 1] = p
    end
    resize!(rowval, p - 1)
    resize!(nzval, p - 1)
    return SparseArrays.SparseMatrixCSC(m, n, colptr, rowval, nzval)
end

# Structural slots for the VSC tail (polar). Bus×converter injection, the two control rows per
# converter, and the DC-KCL row per node (G_dc pattern + converter coupling). All-zero placeholders;
# filled by `_set_entries_for_vsc`. Bus magnitude column (`2ix-1`) slots carry the AC-voltage / loss
# coupling (zero unless the converter controls |V_ac| or has losses).
function _create_jacobian_matrix_structure_vsc(
    data::ACPowerFlowData,
    rows::Vector{J_INDEX_TYPE},
    columns::Vector{J_INDEX_TYPE},
    values::Vector{Float64},
    num_buses::Int,
)
    dcn = get_dc_network(data)
    has_dc_network(dcn) || return
    nconv = n_vsc_converters(dcn)
    nnode = n_dc_nodes(dcn)
    num_lcc = size(data.lcc.p_set, 1)
    vsc_off = 2 * num_buses + 4 * num_lcc
    base = vsc_off + 2 * nconv
    function push3(r, c)
        push!(rows, J_INDEX_TYPE(r))
        push!(columns, J_INDEX_TYPE(c))
        push!(values, 0.0)
        return
    end
    for c in 1:nconv
        ix = dcn.converter_ac_bus_ix[c]
        k = dcn.converter_dc_node_ix[c]
        pc = vsc_off + 2 * c - 1
        qc = vsc_off + 2 * c
        vk = base + k
        push3(2 * ix - 1, pc)   # ∂P_bal/∂P_c
        push3(2 * ix, qc)       # ∂Q_bal/∂Q_c
        push3(pc, pc)           # ∂r1/∂P_c
        push3(pc, vk)           # ∂r1/∂V_dc
        push3(qc, qc)           # ∂r2/∂Q_c
        push3(qc, 2 * ix - 1)   # ∂r2/∂|V_ac| (Vac modes)
        push3(vk, pc)           # ∂KCL/∂P_c
        push3(vk, qc)           # ∂KCL/∂Q_c (loss)
        push3(vk, 2 * ix - 1)   # ∂KCL/∂|V_ac| (loss)
    end
    for k in 1:nnode
        push3(base + k, base + k)  # DC-KCL diagonal (G_dc[k,k] + converter coupling)
    end
    for b in 1:n_dc_branches(dcn)
        f = dcn.branch_from[b]
        t = dcn.branch_to[b]
        push3(base + f, base + t)  # DC-KCL off-diagonal
        push3(base + t, base + f)
    end
    return
end

# Fill the VSC tail Jacobian entries (polar). Called each iteration after the bus and LCC entries.
function _set_entries_for_vsc(
    data::ACPowerFlowData,
    Jv::SparseArrays.SparseMatrixCSC{Float64, J_INDEX_TYPE},
    num_buses::Int,
    time_step::Int,
)
    dcn = get_dc_network(data)
    has_dc_network(dcn) || return
    nconv = n_vsc_converters(dcn)
    nnode = n_dc_nodes(dcn)
    num_lcc = size(data.lcc.p_set, 1)
    vsc_off = 2 * num_buses + 4 * num_lcc
    base = vsc_off + 2 * nconv
    Vm = view(data.bus_magnitude, :, time_step)
    G = dcn.G_dc
    for b in 1:n_dc_branches(dcn)
        f = dcn.branch_from[b]
        t = dcn.branch_to[b]
        Jv[base + f, base + t] = G[f, t]
        Jv[base + t, base + f] = G[t, f]
    end
    for k in 1:nnode
        Jv[base + k, base + k] = G[k, k]
    end
    # Pre-zero the shared ∂KCL/∂|V_ac| slots before accumulating: two converters can share BOTH
    # the DC node and the AC bus (parallel converters), in which case `sparse` merged their
    # structural slots into one — an `=` write would clobber the first converter's contribution.
    # Unconditional (not gated on the bus currently being PQ): the slot is always structurally
    # allocated (see below), and a bus that was PQ on a previous call but is PV now must have its
    # stale contribution cleared here, since the accumulation loop below only writes it for PQ.
    for c in 1:nconv
        Jv[base + dcn.converter_dc_node_ix[c], 2 * dcn.converter_ac_bus_ix[c] - 1] = 0.0
    end
    for c in 1:nconv
        ix = dcn.converter_ac_bus_ix[c]
        k = dcn.converter_dc_node_ix[c]
        pc = vsc_off + 2 * c - 1
        qc = vsc_off + 2 * c
        vk = base + k
        mode = dcn.converter_mode[c]
        Vmix = Vm[ix]
        Vdc = dcn.node_vdc[k, time_step]
        (Pdc, dP, dQ, dVm) = _vsc_pdc_derivatives(dcn, c, Vmix, time_step)
        Jv[2 * ix - 1, pc] = -1.0
        Jv[2 * ix, qc] = -1.0
        Jv[pc, pc] = _vsc_dr1_dP(mode, dcn, c)
        Jv[pc, vk] = _vsc_dr1_dVdc(mode)
        Jv[qc, qc] = _vsc_dr2_dQ(mode)
        Jv[vk, pc] = dP / Vdc
        Jv[vk, qc] = dQ / Vdc
        Jv[vk, vk] += -Pdc / (Vdc * Vdc)
        # Column `2ix-1` is the |V_ac| state only at PQ buses; at PV/REF |V_ac| is fixed, so the
        # converter's |V_ac|-coupling derivatives (AC-voltage control + loss) do not enter the
        # Jacobian. The structure allocates the slot as PQ regardless (PV→PQ transitions), so
        # leaving it unwritten here keeps a correct structural zero.
        if data.bus_type[ix, time_step] == PSY.ACBusTypes.PQ
            Jv[qc, 2 * ix - 1] = _vsc_dr2_dVm(mode, Vmix)
            Jv[vk, 2 * ix - 1] += dVm / Vdc
        end
    end
    return
end

function _set_entries_for_lcc(data::ACPowerFlowData,
    Jv::SparseArrays.SparseMatrixCSC{Float64, J_INDEX_TYPE},
    num_buses::Int,
    time_step::Int)
    for (i, (fb, tb)) in enumerate(data.lcc.bus_indices)
        idx_p_fb = 2 * fb - 1
        idx_q_fb = 2 * fb
        idx_p_tb = 2 * tb - 1
        idx_q_tb = 2 * tb
        offset_lcc = num_buses * 2 + (i - 1) * 4
        idx_tap_from = offset_lcc + 1
        idx_tap_to = offset_lcc + 2
        idx_angle_from = offset_lcc + 3
        idx_angle_to = offset_lcc + 4

        # F_α = α − α_min has a constant unit self-derivative; write it each iteration so
        # every nonzero is owned by the update path, not seeded only at construction.
        Jv[idx_angle_from, idx_angle_from] = 1.0
        Jv[idx_angle_to, idx_angle_to] = 1.0

        alpha_r = data.lcc.rectifier.thyristor_angle[i, time_step]
        alpha_i = data.lcc.inverter.thyristor_angle[i, time_step]
        phi_r = data.lcc.rectifier.phi[i, time_step]
        phi_i = data.lcc.inverter.phi[i, time_step]
        xtr_r = data.lcc.rectifier.transformer_reactance[i]
        xtr_i = data.lcc.inverter.transformer_reactance[i]
        Vm_fb = data.bus_magnitude[fb, time_step]
        Vm_tb = data.bus_magnitude[tb, time_step]
        bus_type_fb = data.bus_type[fb, time_step]
        bus_type_tb = data.bus_type[tb, time_step]

        if iszero(data.lcc.i_dc[i, time_step])
            # 0-current converter: P_lcc ≡ 0, so it contributes nothing to the bus
            # rows and its P-setpoint / DC-line-balance rows are vacuous. Zero its
            # bus-coupling entries and pin the two tap states with identity rows
            # (matching _write_lcc_tail!), keeping the block nonsingular without
            # changing the sparsity structure.
            Jv[idx_p_fb, idx_tap_from] = 0.0
            Jv[idx_p_fb, idx_angle_from] = 0.0
            Jv[idx_q_fb, idx_tap_from] = 0.0
            Jv[idx_q_fb, idx_angle_from] = 0.0
            Jv[idx_p_tb, idx_tap_to] = 0.0
            Jv[idx_p_tb, idx_angle_to] = 0.0
            Jv[idx_q_tb, idx_tap_to] = 0.0
            Jv[idx_q_tb, idx_angle_to] = 0.0
            if bus_type_fb == PSY.ACBusTypes.PQ
                Jv[idx_tap_from, idx_p_fb] = 0.0
                Jv[idx_tap_to, idx_p_fb] = 0.0
            end
            if bus_type_tb == PSY.ACBusTypes.PQ
                Jv[idx_tap_from, idx_p_tb] = 0.0
                Jv[idx_tap_to, idx_p_tb] = 0.0
            end
            Jv[idx_tap_from, idx_tap_from] = 1.0   # ∂(tap_r − tap_set)/∂tap_r
            Jv[idx_tap_from, idx_angle_from] = 0.0
            Jv[idx_tap_from, idx_tap_to] = 0.0
            Jv[idx_tap_from, idx_angle_to] = 0.0
            Jv[idx_tap_to, idx_tap_from] = 0.0
            Jv[idx_tap_to, idx_tap_to] = 1.0       # ∂(tap_i − tap_set)/∂tap_i
            Jv[idx_tap_to, idx_angle_from] = 0.0
            Jv[idx_tap_to, idx_angle_to] = 0.0
            continue
        end

        s = _lcc_jacobian_scalars(data, i, time_step, Vm_fb, Vm_tb)

        dP_dV_fb = s.dP_dV_fb
        dP_dV_tb = s.dP_dV_tb
        dP_dt_fb = s.dP_dt_fb
        dP_dt_tb = s.dP_dt_tb

        # Bus-row × tail-column entries (∂{P,Q}/∂{tap, α}) are written
        # unconditionally — the bus residual rows exist for all bus types,
        # and tap/α are state variables regardless of which AC terminal is
        # PQ/PV/REF. ∂{P,Q}/∂V is gated by PQ (V is a state only there);
        # likewise the tail × bus-V chain rule.
        Jv[idx_p_fb, idx_tap_from] = dP_dt_fb # ∂P_fb/∂t_fb
        Jv[idx_p_fb, idx_angle_from] = s.dP_dα_fb # ∂P_fb/∂α_fb
        Jv[idx_q_fb, idx_tap_from] =
            _calculate_dQ_dt_lcc(s.tap_r, s.i_dc, xtr_r, Vm_fb, phi_r) # ∂Q_fb/∂t_fb
        Jv[idx_q_fb, idx_angle_from] =
            _calculate_dQ_dα_lcc(s.tap_r, s.i_dc, xtr_r, Vm_fb, phi_r, alpha_r) # ∂Q_fb/∂α_fb
        Jv[idx_p_tb, idx_tap_to] = dP_dt_tb # ∂P_tb/∂t_tb
        Jv[idx_p_tb, idx_angle_to] = s.dP_dα_tb # ∂P_tb/∂α_tb
        # Inverter dQ: −xtr_i flips the commutation-chain term to the inverter sign
        # (see _lcc_jacobian_scalars); the leading sin ϕ_i term is x_t-free.
        Jv[idx_q_tb, idx_tap_to] =
            _calculate_dQ_dt_lcc(s.tap_i, s.i_dc, -xtr_i, Vm_tb, phi_i) # ∂Q_tb/∂t_tb
        # φ_i convention flips sign of ∂φ_i/∂α_i vs the rectifier; negate helper output.
        Jv[idx_q_tb, idx_angle_to] =
            -_calculate_dQ_dα_lcc(s.tap_i, s.i_dc, xtr_i, Vm_tb, phi_i, alpha_i) # ∂Q_tb/∂α_tb

        if bus_type_fb == PSY.ACBusTypes.PQ
            Jv[idx_p_fb, idx_p_fb] += dP_dV_fb # ∂P_fb/∂V_fb
            Jv[idx_q_fb, idx_p_fb] +=
                _calculate_dQ_dV_lcc(s.tap_r, s.i_dc, xtr_r, Vm_fb, phi_r) # ∂Q_fb/∂V_fb
            # ∂F_t_fb/∂V_fb is nonzero only with a rectifier-side set point;
            # the scalar is pre-zeroed otherwise.
            Jv[idx_tap_from, idx_p_fb] = s.d_Ft_fb_d_V_fb # ∂F_t_fb/∂V_fb
            Jv[idx_tap_to, idx_p_fb] = dP_dV_fb # ∂F_t_tb/∂V_fb
        end

        if bus_type_tb == PSY.ACBusTypes.PQ
            Jv[idx_p_tb, idx_p_tb] += dP_dV_tb # ∂P_tb/∂V_tb
            Jv[idx_q_tb, idx_p_tb] +=
                _calculate_dQ_dV_lcc(s.tap_i, s.i_dc, -xtr_i, Vm_tb, phi_i) # ∂Q_tb/∂V_tb (−xtr_i: inverter commutation sign)
            # ∂F_t_fb/∂V_tb is nonzero only with an inverter-side set point.
            Jv[idx_tap_from, idx_p_tb] = s.d_Ft_fb_d_V_tb # ∂F_t_fb/∂V_tb
            Jv[idx_tap_to, idx_p_tb] = dP_dV_tb # ∂F_t_tb/∂V_tb
        end

        # P-setpoint row F_t_fb: rectifier-side (tap_r, α_r) and inverter-side
        # (tap_i, α_i) slots are written unconditionally; the scalars helper
        # zeroes whichever side the set point is not on.
        Jv[idx_tap_from, idx_tap_from] = s.d_Ft_fb_d_tap_r
        Jv[idx_tap_from, idx_angle_from] = s.d_Ft_fb_d_alpha_r
        Jv[idx_tap_from, idx_tap_to] = s.d_Ft_fb_d_tap_i
        Jv[idx_tap_from, idx_angle_to] = s.d_Ft_fb_d_alpha_i
        Jv[idx_tap_to, idx_tap_from] = s.d_Ft_tb_d_tap_r
        Jv[idx_tap_to, idx_tap_to] = s.d_Ft_tb_d_tap_i
        Jv[idx_tap_to, idx_angle_from] = s.d_Ft_tb_d_alpha_r
        Jv[idx_tap_to, idx_angle_to] = s.d_Ft_tb_d_alpha_i
    end
    return
end

"""Bus indices of REF buses sharing an island with another REF (multi-swing). Each
self-balances its own P-slot (`∂F_P/∂x[2i−1] = −1`) instead of the distributed island
scalar; single-swing islands are excluded and keep the distributed-slack path."""
_multi_swing_ref_indices(
    bus_type::AbstractMatrix{PSY.ACBusTypes.Value},
    subnetworks::Dict{Int64, Vector{Int64}},
    time_step::Int64,
) = _multi_swing_ref_indices!(Set{Int}(), bus_type, subnetworks, time_step)

function _multi_swing_ref_indices!(
    independent::Set{Int},
    bus_type::AbstractMatrix{PSY.ACBusTypes.Value},
    subnetworks::Dict{Int64, Vector{Int64}},
    time_step::Int64,
)
    empty!(independent)
    for subnetwork_buses in values(subnetworks)
        n_ref = count(ix -> bus_type[ix, time_step] == PSY.ACBusTypes.REF, subnetwork_buses)
        n_ref > 1 || continue
        for ix in subnetwork_buses
            bus_type[ix, time_step] == PSY.ACBusTypes.REF && push!(independent, ix)
        end
    end
    return independent
end

"""Marks a Jacobian-only sweep: the bus-row residual writes compile away."""
struct NoResidualRows end

@inline _write_bus_rows!(::NoResidualRows, ::Int, ::Float64, ::Float64) = nothing
@inline function _write_bus_rows!(F::Vector{Float64}, i::Int, fp::Float64, fq::Float64)
    F[2 * i - 1] = fp
    F[2 * i] = fq
    return
end

"""
    _update_residual_and_jacobian!(R::ACPowerFlowResidual, J::ACPowerFlowJacobian, x::Vector{Float64}, data::ACPowerFlowData, time_step::Int64)

Fused polar kernel: evaluate the residual `R.Rv` at `x` and fill `J.Jv` at the same iterate in
one sweep over Ybus. Equals `R(data, x, time_step)` then `J(data, time_step)` up to the
summation order of the residual rows, except that `data` receives the iterate's voltages and
injections only at the next [`_write_back_bus_state!`](@ref).
"""
function _update_residual_and_jacobian!(
    R::ACPowerFlowResidual,
    J::ACPowerFlowJacobian,
    x::Vector{Float64},
    data::ACPowerFlowData,
    time_step::Int64,
)
    _update_residual_state!(R, x, data, time_step, WriteBackDeferred())
    # The sweep assigns every bus row; only the tail rows need clearing.
    fill!(view(R.Rv, (2 * first(size(data.bus_type)) + 1):length(R.Rv)), 0.0)
    _polar_ybus_sweep!(J, R.Rv, data, time_step)
    # The residual tails refresh LCC/VSC/area state the Jacobian tails read.
    _finish_residual!(R, x, data, time_step)
    _set_jacobian_tails!(J, data, time_step)
    return
end

function _update_jacobian_matrix_values!(
    J::ACPowerFlowJacobian,
    data::ACPowerFlowData,
    time_step::Int64,
)
    _polar_ybus_sweep!(J, NoResidualRows(), data, time_step)
    _set_jacobian_tails!(J, data, time_step)
    return
end

"""Fill the Ybus part of Jv and the distributed-slack cross-terms from `J.bus_state` (|V| and
cis(θ), refilled by the caller). With `F::Vector{Float64}`, also write the Ybus part of the
residual bus rows.

INVARIANT: every call writes every structural nonzero of the sweep, also the slots that are 0
for PV/REF neighbors and the constant REF/PV diagonal-block entries. A reused `Jv` depends on
this after a bus-type change. All writes go through the offset caches (`od_jnz`, `diag_jnz`,
`slack_jnz`). cis(θ_from − θ_to) is `phasor[from] * conj(phasor[to])`, so the sweep does no
trig per Ybus nonzero."""
function _polar_ybus_sweep!(
    J::ACPowerFlowJacobian,
    F::Union{Vector{Float64}, NoResidualRows},
    data::ACPowerFlowData,
    time_step::Int64,
)
    Jv = J.Jv
    od_ptr = J.od_ptr
    od_to = J.od_to
    od_ybus_nz = J.od_ybus_nz
    od_jnz = J.od_jnz
    diag_jnz = J.diag_jnz
    diag_ybus_nz = J.diag_ybus_nz
    e = J.bus_state.phasor
    bus_slack_participation_factors = J.bus_slack_participation_factors
    independent_ref = J.independent_ref
    bus_active_constant_I = J.bus_active_constant_I
    bus_reactive_constant_I = J.bus_reactive_constant_I
    bus_active_constant_Z = J.bus_active_constant_Z
    bus_reactive_constant_Z = J.bus_reactive_constant_Z
    Yb = data.power_network_matrix.data
    Yb_vals = SparseArrays.nonzeros(Yb)
    Jvnz = SparseArrays.nonzeros(Jv)
    Vm = J.bus_state.Vm
    bus_types = view(data.bus_type, :, time_step)
    num_buses = first(size(data.bus_type))

    @inbounds for bus_from in 1:num_buses
        Vm_from = Vm[bus_from]
        e_from = e[bus_from]
        # Off-diagonal parts of the self block and of the P and Q injections.
        dp_dθ = 0.0
        dq_dθ = 0.0
        dp_dv = 0.0
        dq_dv = 0.0
        fp = 0.0
        fq = 0.0
        for k in od_ptr[bus_from]:(od_ptr[bus_from + 1] - 1)
            bus_to = od_to[k]
            y = Yb_vals[od_ybus_nz[k]]
            g_ij = real(y)
            b_ij = imag(y)
            Vm_to = Vm[bus_to]
            c = e_from * conj(e[bus_to])
            cosθ = real(c)
            sinθ = imag(c)
            p_vm_common = g_ij * cosθ + b_ij * sinθ
            q_vm_common = g_ij * sinθ - b_ij * cosθ
            vv = Vm_from * Vm_to
            p_va_common = vv * q_vm_common          # Vm_f·Vm_t·(g·sin − b·cos)
            q_va_common = vv * (-g_ij * cosθ - b_ij * sinθ)
            fp += vv * p_vm_common
            fq += vv * q_vm_common
            # Diagonal accumulation is bus_to-type-independent (REF/PV/PQ identical).
            dp_dv += Vm_to * p_vm_common
            dp_dθ -= p_va_common
            dq_dv += Vm_to * q_vm_common
            dq_dθ -= q_va_common
            # Off-diagonal slot values depend on bus_to type: PQ writes all four; PV
            # zeroes the (·, Vm) columns (Vm_to not a state); REF zeroes all four
            # (its columns hold P_gen/Q_gen). Every slot is written each call.
            bt = bus_types[bus_to]
            if bt == PSY.ACBusTypes.PQ
                Jvnz[od_jnz[1, k]] = Vm_from * p_vm_common  # Jv[p, vm]
                Jvnz[od_jnz[2, k]] = Vm_from * q_vm_common  # Jv[q, vm]
                Jvnz[od_jnz[3, k]] = p_va_common            # Jv[p, va]
                Jvnz[od_jnz[4, k]] = q_va_common            # Jv[q, va]
            elseif bt == PSY.ACBusTypes.PV
                Jvnz[od_jnz[1, k]] = 0.0
                Jvnz[od_jnz[2, k]] = 0.0
                Jvnz[od_jnz[3, k]] = p_va_common
                Jvnz[od_jnz[4, k]] = q_va_common
            else  # REF
                Jvnz[od_jnz[1, k]] = 0.0
                Jvnz[od_jnz[2, k]] = 0.0
                Jvnz[od_jnz[3, k]] = 0.0
                Jvnz[od_jnz[4, k]] = 0.0
            end
        end
        yii = if iszero(diag_ybus_nz[bus_from])
            zero(eltype(Yb_vals))
        else
            Yb_vals[diag_ybus_nz[bus_from]]
        end
        vv_ii = Vm_from * Vm_from
        _write_bus_rows!(F, bus_from, fp + vv_ii * real(yii), fq - vv_ii * imag(yii))

        # diag_jnz rows: 1: Jv[p, vm], 2: Jv[q, vm], 3: Jv[p, va], 4: Jv[q, va].
        bt = bus_types[bus_from]
        if bt == PSY.ACBusTypes.PQ
            Jvnz[diag_jnz[3, bus_from]] = dp_dθ
            Jvnz[diag_jnz[4, bus_from]] = dq_dθ
            d3 = dp_dv + 2 * real(yii) * Vm_from  # ∂P∂V_from
            d4 = dq_dv - 2 * imag(yii) * Vm_from  # ∂Q∂V_from
            # ZIP chain rule: P_net(V) = P₀ − const_I_P·V − const_Z_P·V², so ∂F_P/∂V
            # picks up −∂P_net/∂V = +const_I_P + 2·const_Z_P·V (same shape on Q).
            d3 +=
                bus_active_constant_I[bus_from] +
                2 * bus_active_constant_Z[bus_from] * Vm_from
            d4 +=
                bus_reactive_constant_I[bus_from] +
                2 * bus_reactive_constant_Z[bus_from] * Vm_from
            Jvnz[diag_jnz[1, bus_from]] = d3  # ∂P∂V_from
            Jvnz[diag_jnz[2, bus_from]] = d4  # ∂Q∂V_from
        elseif bt == PSY.ACBusTypes.PV
            Jvnz[diag_jnz[1, bus_from]] = 0.0
            Jvnz[diag_jnz[2, bus_from]] = -1.0
            Jvnz[diag_jnz[3, bus_from]] = dp_dθ
            Jvnz[diag_jnz[4, bus_from]] = dq_dθ
        else  # REF
            if bus_from in independent_ref
                # Multi-swing island: this swing self-balances at its own P-slot, so
                # ∂F_P/∂x[2i−1] = −1 (not the distributed −c_ref).
                Jvnz[diag_jnz[1, bus_from]] = -1.0
            else
                Jvnz[diag_jnz[1, bus_from]] = -bus_slack_participation_factors[bus_from]
            end
            Jvnz[diag_jnz[2, bus_from]] = 0.0
            Jvnz[diag_jnz[3, bus_from]] = 0.0
            Jvnz[diag_jnz[4, bus_from]] = -1.0
        end
    end

    # Distributed slack cross-terms: for each bus k (other than the REF bus), the active power
    # residual depends on the REF bus state variable x[2*ref-1] through the slack distribution:
    # ∂F_P_k/∂x[2*ref-1] = -c_k. Every slot is written, so a factor that drops to zero (or an
    # island turning multi-swing, where each swing self-balances) leaves no stale value.
    slack_jnz = J.slack_jnz
    @inbounds for (ref_bus, subnetwork_buses) in J.subnetworks
        multi_swing = ref_bus in independent_ref
        for bus_k in subnetwork_buses
            o = slack_jnz[bus_k]
            iszero(o) && continue
            if multi_swing
                Jvnz[o] = 0.0
            else
                Jvnz[o] = -bus_slack_participation_factors[bus_k]
            end
        end
    end
    return
end

function _set_jacobian_tails!(
    J::ACPowerFlowJacobian,
    data::ACPowerFlowData,
    time_step::Int64,
)
    num_buses = first(size(data.bus_type))
    _set_entries_for_lcc(data, J.Jv, num_buses, time_step)
    _set_entries_for_vsc(data, J.Jv, num_buses, time_step)
    _set_entries_for_area(data, J.Jv, time_step)
    return
end

"""
    calculate_loss_factors(data::ACPowerFlowData, Jv::SparseMatrixCSC{Float64, $J_INDEX_TYPE}, time_step::Int)

Calculate and store the active power loss factors in the `loss_factors` matrix of the `ACPowerFlowData` structure for a given time step.

The loss factors are computed using the Jacobian matrix `Jv` and the vector `dSbus_dV_ref`, which contains the
partial derivatives of slack power with respect to bus voltages. The function interprets changes in
slack active power injections as indicative of changes in grid active power losses.
KLU is used to factorize the sparse Jacobian matrix to solve for the loss factors.

# Arguments
- `data::ACPowerFlowData`: The data structure containing power flow information, including the `loss_factors` matrix.
- `Jv::SparseMatrixCSC{Float64, $J_INDEX_TYPE}`: The sparse Jacobian matrix of the power flow system.
- `time_step::Int`: The time step index for which the loss factors are calculated.
"""
function _calculate_loss_factors(
    data::ACPowerFlowData,
    Jv::SparseMatrixCSC{Float64, J_INDEX_TYPE},
    time_step::Int,
)
    bus_numbers = 1:first(size(data.bus_type))
    ref_mask = data.bus_type[:, time_step] .== (PSY.ACBusTypes.REF,)
    if count(ref_mask) > 1
        error(
            "Loss factors with multiple REF buses isn't supported.",
        )
    end
    pvpq_mask = .!ref_mask
    ref = findfirst(ref_mask)
    new_ref_mask = falses(size(ref_mask))
    new_ref_mask[ref] = true
    pvpq_mask = .!(new_ref_mask)
    pvpq_coord_mask = repeat(pvpq_mask; inner = 2)
    J_t = sparse(transpose(Jv[pvpq_coord_mask, pvpq_coord_mask]))
    dSbus_dV_ref = collect(Jv[2 .* ref .- 1, pvpq_coord_mask])[:]
    lf_cache = make_linear_solver_cache(PNM.KLUSolver(), J_t)
    full_factor!(lf_cache, J_t)
    lf = copy(dSbus_dV_ref)
    solve!(lf_cache, lf)
    # only take the dPref_dP loss factors, ignore dPref_dQ
    data.loss_factors[pvpq_mask, time_step] .= lf[1:2:end]
    data.loss_factors[new_ref_mask, time_step] .= -1.0
end

"""
    calculate_voltage_stability_factors(data::ACPowerFlowData, J::ACPowerFlowJacobian, time_step::Integer)

Calculate and store the voltage stability factors in the `voltage_stability_factors` matrix of the `ACPowerFlowData` structure for a given time step.
The voltage stability factors are computed using the Jacobian matrix `J` in block format after a converged power flow calculation.
The results are stored in the `voltage_stability_factors` matrix in the `data` instance.
The factor for the grid as a whole (σ) is stored in the position of the REF bus.
The values of the singular vector `v` indicate the sensitivity of the buses and are stored in the positions of the PQ buses.
The values of `v` for PV buses are set to zero.
The function uses the method described in \"Fast calculation of a voltage stability index\" by PA Lof et. al.
# Arguments
- `data::ACPowerFlowData`: The instance containing the grid model data.
- `J::ACPowerFlowJacobian`: The Jacobian matrix cache.
- `time_step::Integer`: The calculated time step.
"""
function _calculate_voltage_stability_factors(
    data::ACPowerFlowData,
    Jv::SparseMatrixCSC{Float64, J_INDEX_TYPE},
    time_step::Integer,
)
    ref, pv, pq = bus_type_idx(data, time_step)
    pvpq = [pv; pq]
    rows, cols = _block_J_indices(pvpq, pq)
    σ, _, right = _singular_value_decomposition(Jv[rows, cols], length(pvpq))
    # Store σ at REF bus, set remaining REF buses (if any) to zero
    data.voltage_stability_factors[first(ref), time_step] = σ
    data.voltage_stability_factors[ref[2:end], time_step] .= 0.0
    # PV buses have zero sensitivity, PQ buses get the right singular vector
    data.voltage_stability_factors[pv, time_step] .= 0.0
    data.voltage_stability_factors[pq, time_step] .= right
    return
end

"""
    _block_J_indices(data::ACPowerFlowData, time_step::Int) -> (Vector{$J_INDEX_TYPE}, Vector{$J_INDEX_TYPE})

Get the indices to reindex the Jacobian matrix from the interleaved form to the block form:

```math
\\begin{bmatrix}
\\frac{\\partial P}{\\partial \\theta} & \\frac{\\partial P}{\\partial V} \\\\
\\frac{\\partial Q}{\\partial \\theta} & \\frac{\\partial Q}{\\partial V}
\\end{bmatrix}
```

# Arguments
- `pvpq::Vector{$J_INDEX_TYPE}`: Indices of the buses that are PV or PQ buses.
- `pq::Vector{$J_INDEX_TYPE}`: Indices of the buses that are PQ buses.

# Returns
- `rows::Vector{$J_INDEX_TYPE}`: Row indices for the block Jacobian matrix.
- `cols::Vector{$J_INDEX_TYPE}`: Column indices for the block Jacobian matrix.
"""
function _block_J_indices(pvpq::Vector{<:Integer}, pq::Vector{<:Integer})
    rows = vcat(2 .* pvpq .- 1, 2 .* pq)
    cols = vcat(2 .* pvpq, 2 .* pq .- 1)

    return rows, cols
end

"""
    _singular_value_decomposition(J::SparseMatrixCSC{Float64, $J_INDEX_TYPE}, npvpq::Integer; tol::Float64 = 1e-9, max_iter::Integer = 100,)

Estimate the smallest singular value `σ` and corresponding left and right singular vectors `u` and `v` of a sparse matrix `G_s` (a sub-matrix of `J`).
This function uses an iterative method involving LU factorization of the Jacobian matrix to estimate the smallest singular value of `G_s`.
The algorithm alternates between updating `u` and `v`, normalizing, and checking for convergence based on the change in the estimated singular value `σ`.
The function uses the method described in `Algorithm 3` of \"Fast calculation of a voltage stability index\" by PA Lof et. al.

# Arguments
- `J::SparseMatrixCSC{Float64, $J_INDEX_TYPE}`: The sparse block-form Jacobian matrix.
- `npvpq::Integer`: Number of PV and PQ buses in J.

# Keyword Arguments
- `tol::Float64=1e-9`: Convergence tolerance for the iterative algorithm.
- `max_iter::Integer=100`: Maximum number of iterations.

# Returns
- `σ::Float64`: The estimated smallest singular value.
- `left::Vector{Float64}`: The estimated left singular vector (referred to as `u` in the cited paper).
- `right::Vector{Float64}`: The estimated right singular vector (referred to as `v` in the cited paper).
"""
function _singular_value_decomposition(
    Jv::SparseMatrixCSC{Float64, J_INDEX_TYPE},
    npvpq::Integer;
    tol::Float64 = 1e-9,
    max_iter::Integer = 100,
)
    # Voltage-stability factors solve `Aᵀ x = b` reusing the existing factorization of
    # `A` (rather than factoring `Aᵀ` separately). Only the KLU backend exposes that
    # transposed-solve-from-an-A-factorization, so this routine is KLU-only by construction.
    factorized_block_J = make_linear_solver_cache(PNM.KLUSolver(), Jv)
    full_factor!(factorized_block_J, Jv)
    n = size(Jv, 1)
    voltage_angle_indices = 1:npvpq

    right = ones(n)
    right_angle_section = view(right, voltage_angle_indices)
    fill!(right_angle_section, 0.0)  # Set the part of `right` corresponding to voltage angles to zero
    right ./= norm(right, 2)

    left = ones(n)
    left_angle_section = view(left, voltage_angle_indices)
    fill!(left_angle_section, 0.0)  # Set the part of `left` corresponding to voltage angles to zero

    σ = 1e6  # min. singular value
    k = 1

    while k <= max_iter
        copyto!(left, right)
        tsolve!(factorized_block_J, left)
        fill!(left_angle_section, 0.0)
        norm_left = norm(left, 2)

        σ_1 = 1 / norm_left
        delta_σ = σ_1 - σ
        σ = σ_1

        ldiv!(left, norm_left, left)

        if abs(delta_σ) < tol
            break
        end

        copyto!(right, left)
        solve!(factorized_block_J, right)
        fill!(right_angle_section, 0.0)
        norm_right = norm(right, 2)

        σ_2 = 1 / norm_right
        delta_σ = σ_2 - σ
        σ = σ_2

        ldiv!(right, norm_right, right)

        if abs(delta_σ) < tol
            break
        end

        k += 1
    end
    return σ, left[(npvpq + 1):end], right[(npvpq + 1):end]
end
