"""
    struct ACPowerFlowJacobian

A struct that represents the Jacobian matrix for AC power flow calculations.

This struct uses the functor pattern, meaning instances of `ACPowerFlowJacobian` store the data (Jacobian matrix) internally
and can be called as a function at the same time. Calling the instance as a function updates the stored Jacobian matrix.

# Fields
- `data::ACPowerFlowData`: The grid model data used for power flow calculations.
- `Jf!::Function`: A function that calculates the Jacobian matrix inplace.
- `Jv::SparseArrays.SparseMatrixCSC{Float64, $J_INDEX_TYPE}`: The Jacobian matrix, which is updated by the function `Jf!`.
- `diag_elements::MVector{4, Float64}`: Temporary storage for diagonal elements during Jacobian update.
- `bus_slack_participation_factors::SparseVector{Float64, Int}`: Normalized per-bus slack participation factors for the current time step (from the `ACPowerFlowResidual`). Used for the distributed slack Jacobian entries.
- `subnetworks::Dict{Int64, Vector{Int64}}`: Subnetwork mapping from REF bus to bus list (from the `ACPowerFlowResidual`). Used for the distributed slack Jacobian entries.
"""
struct ACPowerFlowJacobian
    data::ACPowerFlowData
    Jf!::Function   # This is the function that calculates the Jacobian matrix and updates Jv inplace
    Jv::SparseArrays.SparseMatrixCSC{Float64, J_INDEX_TYPE}  # This is the Jacobian matrix, that is updated by the function Jf
    diag_elements::MVector{4, Float64}  # Temporary storage for diagonal elements during Jacobian update
    bus_slack_participation_factors::SparseVector{Float64, Int}
    subnetworks::Dict{Int64, Vector{Int64}}
    bus_active_constant_I::Vector{Float64}
    bus_reactive_constant_I::Vector{Float64}
    bus_active_constant_Z::Vector{Float64}
    bus_reactive_constant_Z::Vector{Float64}
end

"""
    (J::ACPowerFlowJacobian)(time_step::Int64)

Update the Jacobian matrix `Jv` using the function `Jf!` and the provided data and time step.

Defining this method allows an instance of `ACPowerFlowJacobian` to be called as a function, following the functor pattern.

# Arguments
- `time_step::Int64`: The time step for the calculations.

# Example
```julia
residual = ACPowerFlowResidual(data, time_step)
J = ACPowerFlowJacobian(residual, time_step)
J(time_step)  # Updates the Jacobian matrix Jv
```
"""
function (J::ACPowerFlowJacobian)(time_step::Int64)
    J.Jf!(J.Jv, J.data, time_step, J.diag_elements,
        J.bus_slack_participation_factors, J.subnetworks,
        J.bus_active_constant_I, J.bus_reactive_constant_I,
        J.bus_active_constant_Z, J.bus_reactive_constant_Z)
    return
end

"""
    (J::ACPowerFlowJacobian)(J::SparseArrays.SparseMatrixCSC{Float64, $J_INDEX_TYPE}, time_step::Int64)

Use the `ACPowerFlowJacobian` to update the provided Jacobian matrix `J` inplace.

Update the internally stored Jacobian matrix `Jv` using the function `Jf!` and the provided data and time step, and write the updated Jacobian values to `J`.

This method allows an instance of ACPowerFlowJacobian to be called as a function, following the functor pattern.

# Arguments
- `J::SparseArrays.SparseMatrixCSC{Float64, $J_INDEX_TYPE}`: A sparse matrix to be updated with new values of the Jacobian matrix.
- `time_step::Int64`: The time step for the calculations.

# Example
```julia
residual = ACPowerFlowResidual(data, time_step)
J = ACPowerFlowJacobian(residual, time_step)
Jv = SparseArrays.sparse(Float64[], J_INDEX_TYPE[], J_INDEX_TYPE[])
J(Jv, time_step)  # Updates the Jacobian matrix Jv and writes it to J
```
"""
function (J::ACPowerFlowJacobian)(
    Jv::SparseArrays.SparseMatrixCSC{Float64, J_INDEX_TYPE},
    time_step::Int64,
)
    J.Jf!(J.Jv, J.data, time_step, J.diag_elements,
        J.bus_slack_participation_factors, J.subnetworks,
        J.bus_active_constant_I, J.bus_reactive_constant_I,
        J.bus_active_constant_Z, J.bus_reactive_constant_Z)
    copyto!(Jv, J.Jv)
    return
end

"""
    ACPowerFlowJacobian(residual::ACPowerFlowResidual, time_step::Int64) -> ACPowerFlowJacobian

Constructor for `ACPowerFlowJacobian`. The returned instance has its sparsity
pattern initialized and shares the residual's slack-participation, subnetwork,
and ZIP-coefficient caches — the residual must be constructed first against the
same `data` and `time_step`.

# Arguments
- `residual::ACPowerFlowResidual`: The companion residual; supplies `data`,
  `bus_slack_participation_factors`, `subnetworks`, and the per-bus ZIP load
  coefficient vectors.
- `time_step::Int64`: The time step for the calculations.

# Example
```julia
residual = ACPowerFlowResidual(data, time_step)
J = ACPowerFlowJacobian(residual, time_step)
J(time_step)  # Updates the Jacobian matrix stored internally in J.
J.Jv  # Access the Jacobian matrix stored internally in J.
```
"""
function ACPowerFlowJacobian(
    residual::ACPowerFlowResidual,
    time_step::Int64,
)
    Jv0 = _create_jacobian_matrix_structure(
        residual.data,
        residual.bus_slack_participation_factors,
        residual.subnetworks,
        time_step,
    )
    return ACPowerFlowJacobian(
        residual.data,
        _update_jacobian_matrix_values!,
        Jv0,
        MVector{4, Float64}(undef),
        residual.bus_slack_participation_factors,
        residual.subnetworks,
        residual.bus_active_constant_I,
        residual.bus_reactive_constant_I,
        residual.bus_active_constant_Z,
        residual.bus_reactive_constant_Z,
    )
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
    _create_jacobian_matrix_structure(data::ACPowerFlowData, time_step::Int64) -> SparseMatrixCSC{Float64, $J_INDEX_TYPE}

Create the structure of the Jacobian matrix for an AC power flow problem.

# Arguments
- `data::ACPowerFlowData`: The power flow model.
- `time_step::Int64`: The specific time step for which the Jacobian matrix structure is created.

# Returns
- `SparseMatrixCSC{Float64, $J_INDEX_TYPE}`: A sparse matrix with structural zeros representing the structure of the Jacobian matrix.

# Description

This function initializes the structure of the Jacobian matrix for an AC power flow problem.
The Jacobian matrix is used in power flow analysis to represent the partial derivatives of bus active and reactive power injections with respect to bus voltage magnitudes and angles.

Unlike some commonly used approaches where the Jacobian matrix is constructed as four submatrices, each grouping values for the four types of partial derivatives,
this function groups the partial derivatives by bus. The structure is organized as groups of 4 values per bus.

This approach is more memory-efficient. Furthermore, this structure results in a more efficient factorization because the values are more likely to be grouped close to the diagonal.
Refer to Electric Energy Systems: Analysis and Operation by Antonio Gomez-Exposito and Fernando L. Alvarado for more details.

The function initializes three arrays (`rows`, `columns`, and `values`) to store the row indices, column indices, and values of the non-zero elements of the Jacobian matrix, respectively.

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

Finally, the function constructs a sparse matrix from the collected indices and values and returns it.
"""
function _create_jacobian_matrix_structure(
    data::ACPowerFlowData,
    bus_slack_participation_factors::SparseVector{Float64, Int},
    subnetworks::Dict{Int64, Vector{Int64}},
    time_step::Int64,
)
    # Create Jacobian structure
    # Initialize arrays to store the row indices, column indices, and values of the non-zero elements of the Jacobian matrix
    rows = J_INDEX_TYPE[]      # I
    columns = J_INDEX_TYPE[]   # J
    values = Float64[]  # V

    num_buses = first(size(data.bus_type))
    num_lccs = size(data.lcc.p_set, 1)

    num_lines = length(get_arc_lookup(data))
    sizehint!(rows, 4 * num_lines + 15 * num_lccs)
    sizehint!(columns, 4 * num_lines + 15 * num_lccs)
    sizehint!(values, 4 * num_lines + 15 * num_lccs)

    for bus_from in 1:num_buses
        row_from_p = 2 * bus_from - 1  # Row index for the value that is related to active power
        row_from_q = 2 * bus_from      # Row index for the value that is related to reactive power
        for bus_to in data.neighbors[bus_from]
            col_to_vm = 2 * bus_to - 1  # Column index for the value related to voltage magnitude
            col_to_va = 2 * bus_to      # Column index for the value related to voltage angle
            # We ignore the bus type and initialize the structure as if all buses were PQ -
            # mainly because we can have a PV -> PQ transition, and the number of REF buses is small
            # bus_type = data.bus_type[bus_to, time_step]
            _create_jacobian_matrix_structure_bus!(
                rows,
                columns,
                values,
                bus_from,
                bus_to,
                row_from_p,
                row_from_q,
                col_to_vm,
                col_to_va,
                # Val(bus_type),
            )
        end
    end

    # Add structural entries for distributed slack: each participating bus k has
    # ∂F_P_k/∂x[2*ref-1] = -c_k. If bus k is not a neighbor of the ref bus,
    # this entry doesn't exist yet in the sparsity pattern.
    for (ref_bus, subnetwork_buses) in subnetworks
        for bus_k in subnetwork_buses
            bus_slack_participation_factors[bus_k] == 0.0 && continue
            if !(ref_bus in data.neighbors[bus_k])
                push!(rows, J_INDEX_TYPE(2 * bus_k - 1))
                push!(columns, J_INDEX_TYPE(2 * ref_bus - 1))
                push!(values, 0.0)
            end
        end
    end

    _create_jacobian_matrix_structure_lcc(data, rows, columns, values, num_buses)
    Jv0 = SparseArrays.sparse(rows, columns, values)
    return Jv0
end

function _set_entries_for_neighbor(::SparseArrays.SparseMatrixCSC{Float64, J_INDEX_TYPE},
    Y_from_to::ComplexF32,
    Vm_from::Float64,
    Vm_to::Float64,
    θ_from_to::Float64,
    ::Int,
    ::Int,
    ::Int,
    ::Int,
    diag_elements::MVector{4, Float64},
    ::Val{PSY.ACBusTypes.REF})
    # State variables are Active and Reactive Power Generated
    # F[2*i-1] := p[i] = p_flow[i] + p_load[i] - x[2*i-1]
    # F[2*i] := q[i] = q_flow[i] + q_load[i] - x[2*i]
    # x does not appear in p_flow and q_flow
    g_ij, b_ij = real(Y_from_to), imag(Y_from_to)
    # still need to do diagonal terms: those are based off
    # the bus type of from_bus, when we're dispatching on bustype of to_bus.
    diag_elements[1] -= Vm_from * Vm_to * (g_ij * sin(θ_from_to) - b_ij * cos(θ_from_to))  # ∂P∂θ_from
    diag_elements[2] -= Vm_from * Vm_to * (-g_ij * cos(θ_from_to) - b_ij * sin(θ_from_to))  # ∂Q∂θ_from
    diag_elements[3] += Vm_to * (g_ij * cos(θ_from_to) + b_ij * sin(θ_from_to))  # ∂P∂V_from
    diag_elements[4] += Vm_to * (g_ij * sin(θ_from_to) - b_ij * cos(θ_from_to))  # ∂Q∂V_from
    return
end

function _set_entries_for_neighbor(Jv::SparseArrays.SparseMatrixCSC{Float64, J_INDEX_TYPE},
    Y_from_to::ComplexF32,
    Vm_from::Float64,
    Vm_to::Float64,
    θ_from_to::Float64,
    row_from_p::Int,
    row_from_q::Int,
    ::Int,
    col_to_va::Int,
    diag_elements::MVector{4, Float64},
    ::Val{PSY.ACBusTypes.PV},
)
    # State variables are Reactive Power Generated and Voltage Angle
    # F[2*i-1] := p[i] = p_flow[i] + p_load[i] - p_gen[i]
    # F[2*i] := q[i] = q_flow[i] + q_load[i] - x[2*i]
    # x[2*i] (associated with q_gen) does not appear in q_flow
    # x[2*i] (associated with q_gen) does not appear in the active power balance
    g_ij, b_ij = real(Y_from_to), imag(Y_from_to)
    # Jac: Active PF against other angles θ[bus_to]
    p_va_common_term = Vm_from * Vm_to * (g_ij * sin(θ_from_to) - b_ij * cos(θ_from_to))
    Jv[row_from_p, col_to_va] = p_va_common_term
    diag_elements[1] -= p_va_common_term # ∂P∂θ_from
    # Jac: Reactive PF w/r to different angle θ[bus_to]
    q_va_common_term = Vm_from * Vm_to * (-g_ij * cos(θ_from_to) - b_ij * sin(θ_from_to))
    Jv[row_from_q, col_to_va] = q_va_common_term
    diag_elements[2] -= q_va_common_term # ∂Q∂θ_from

    # still need to do all diagonal terms: those are based off
    # the bus type of from_bus, when we're dispatching on bustype of to_bus.
    diag_elements[3] += Vm_to * (g_ij * cos(θ_from_to) + b_ij * sin(θ_from_to))  # ∂P∂V_from
    diag_elements[4] += Vm_to * (g_ij * sin(θ_from_to) - b_ij * cos(θ_from_to))  # ∂Q∂V_from
    return
end

function _set_entries_for_neighbor(Jv::SparseArrays.SparseMatrixCSC{Float64, J_INDEX_TYPE},
    Y_from_to::ComplexF32,
    Vm_from::Float64,
    Vm_to::Float64,
    θ_from_to::Float64,
    row_from_p::Int,
    row_from_q::Int,
    col_to_vm::Int,
    col_to_va::Int,
    diag_elements::MVector{4, Float64},
    ::Val{PSY.ACBusTypes.PQ},
)
    # State variables are Voltage Magnitude and Voltage Angle
    # both state variables appear in both outputs.
    g_ij, b_ij = real(Y_from_to), imag(Y_from_to)
    # Active PF w/r to different voltage magnitude Vm[bus_to]
    p_vm_common_term = g_ij * cos(θ_from_to) + b_ij * sin(θ_from_to)
    Jv[row_from_p, col_to_vm] = Vm_from * p_vm_common_term
    diag_elements[3] += Vm_to * p_vm_common_term # ∂P∂V_from
    # Active PF w/r to different angle θ[bus_to]
    p_va_common_term = Vm_from * Vm_to * (g_ij * sin(θ_from_to) - b_ij * cos(θ_from_to))
    Jv[row_from_p, col_to_va] = p_va_common_term
    diag_elements[1] -= p_va_common_term # ∂P∂θ_from
    # Reactive PF w/r to different voltage magnitude Vm[bus_to]
    q_vm_common_term = g_ij * sin(θ_from_to) - b_ij * cos(θ_from_to)
    Jv[row_from_q, col_to_vm] = Vm_from * q_vm_common_term
    diag_elements[4] += Vm_to * q_vm_common_term # ∂Q∂V_from
    # Jac: Reactive PF w/r to different angle θ[bus_to]
    q_va_common_term = Vm_from * Vm_to * (-g_ij * cos(θ_from_to) - b_ij * sin(θ_from_to))
    Jv[row_from_q, col_to_va] = q_va_common_term
    diag_elements[2] -= q_va_common_term # ∂Q∂θ_from
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
        Jv[idx_q_tb, idx_tap_to] =
            _calculate_dQ_dt_lcc(s.tap_i, s.i_dc, xtr_i, Vm_tb, phi_i) # ∂Q_tb/∂t_tb
        # φ_i convention flips sign of ∂φ_i/∂α_i vs the rectifier; negate helper output.
        Jv[idx_q_tb, idx_angle_to] =
            -_calculate_dQ_dα_lcc(s.tap_i, s.i_dc, xtr_i, Vm_tb, phi_i, alpha_i) # ∂Q_tb/∂α_tb

        if bus_type_fb == PSY.ACBusTypes.PQ
            Jv[idx_p_fb, idx_p_fb] += dP_dV_fb # ∂P_fb/∂V_fb
            Jv[idx_q_fb, idx_p_fb] +=
                _calculate_dQ_dV_lcc(s.tap_r, s.i_dc, xtr_r, Vm_fb, phi_r) # ∂Q_fb/∂V_fb
            Jv[idx_tap_from, idx_p_fb] = dP_dV_fb # ∂F_t_fb/∂V_fb
            Jv[idx_tap_to, idx_p_fb] = dP_dV_fb # ∂F_t_tb/∂V_fb
        end

        if bus_type_tb == PSY.ACBusTypes.PQ
            Jv[idx_p_tb, idx_p_tb] += dP_dV_tb # ∂P_tb/∂V_tb
            Jv[idx_q_tb, idx_p_tb] +=
                _calculate_dQ_dV_lcc(s.tap_i, s.i_dc, xtr_i, Vm_tb, phi_i) # ∂Q_tb/∂V_tb
            Jv[idx_tap_to, idx_p_tb] = dP_dV_tb # ∂F_t_tb/∂V_tb
        end

        Jv[idx_tap_from, idx_tap_from] = s.d_Ft_fb_d_tap_r
        Jv[idx_tap_from, idx_angle_from] = s.d_Ft_fb_d_alpha_r
        Jv[idx_tap_to, idx_tap_from] = s.d_Ft_tb_d_tap_r
        Jv[idx_tap_to, idx_tap_to] = s.d_Ft_tb_d_tap_i
        Jv[idx_tap_to, idx_angle_from] = s.d_Ft_tb_d_alpha_r
        Jv[idx_tap_to, idx_angle_to] = s.d_Ft_tb_d_alpha_i
    end
    return
end

"""Used to update Jv based on the bus voltages, angles, etc. in data."""
function _update_jacobian_matrix_values!(
    Jv::SparseArrays.SparseMatrixCSC{Float64, J_INDEX_TYPE},
    data::ACPowerFlowData,
    time_step::Int64,
    diag_elements::MVector{4, Float64},
    bus_slack_participation_factors::SparseVector{Float64, Int},
    subnetworks::Dict{Int64, Vector{Int64}},
    bus_active_constant_I::Vector{Float64},
    bus_reactive_constant_I::Vector{Float64},
    bus_active_constant_Z::Vector{Float64},
    bus_reactive_constant_Z::Vector{Float64},
)
    Yb = data.power_network_matrix.data
    Vm = view(data.bus_magnitude, :, time_step)
    θ = view(data.bus_angles, :, time_step)
    num_buses = first(size(data.bus_type))

    for bus_from in 1:num_buses
        row_from_p = 2 * bus_from - 1
        row_from_q = 2 * bus_from

        # Reset diagonal elements for this bus
        fill!(diag_elements, 0.0)

        Vm_from = Vm[bus_from]
        for bus_to in data.neighbors[bus_from]
            if bus_to != bus_from
                col_to_vm = 2 * bus_to - 1
                col_to_va = 2 * bus_to
                bus_type = data.bus_type[bus_to, time_step]
                θ_from_to = θ[bus_from] - θ[bus_to]
                Vm_to = Vm[bus_to]
                Y_from_to = Yb[bus_from, bus_to]
                # 3 case if-else with Val(constant) is faster than 1 case with Val(bus_type)
                if bus_type == PSY.ACBusTypes.PQ
                    _set_entries_for_neighbor(Jv,
                        Y_from_to,
                        Vm_from,
                        Vm_to,
                        θ_from_to,
                        row_from_p,
                        row_from_q,
                        col_to_vm,
                        col_to_va,
                        diag_elements,
                        Val(PSY.ACBusTypes.PQ))
                elseif bus_type == PSY.ACBusTypes.PV
                    _set_entries_for_neighbor(Jv,
                        Y_from_to,
                        Vm_from,
                        Vm_to,
                        θ_from_to,
                        row_from_p,
                        row_from_q,
                        col_to_vm,
                        col_to_va,
                        diag_elements,
                        Val(PSY.ACBusTypes.PV))
                elseif bus_type == PSY.ACBusTypes.REF
                    _set_entries_for_neighbor(Jv,
                        Y_from_to,
                        Vm_from,
                        Vm_to,
                        θ_from_to,
                        row_from_p,
                        row_from_q,
                        col_to_vm,
                        col_to_va,
                        diag_elements,
                        Val(PSY.ACBusTypes.REF))
                end
            end
        end
        col_from_vm = 2 * bus_from - 1
        col_from_va = 2 * bus_from
        # set entries in diagonal blocks
        if data.bus_type[bus_from, time_step] == PSY.ACBusTypes.PQ
            Jv[row_from_p, col_from_va] = diag_elements[1]  # ∂P∂θ_from
            Jv[row_from_q, col_from_va] = diag_elements[2]  # ∂Q∂θ_from
            diag_elements[3] += 2 * real(Yb[bus_from, bus_from]) * Vm[bus_from]  # ∂P∂V_from
            diag_elements[4] -= 2 * imag(Yb[bus_from, bus_from]) * Vm[bus_from]  # ∂Q∂V_from
            # ZIP chain rule: P_net(V) = P₀ − const_I_P·V − const_Z_P·V², so ∂F_P/∂V
            # picks up −∂P_net/∂V = +const_I_P + 2·const_Z_P·V (same shape on Q).
            diag_elements[3] +=
                bus_active_constant_I[bus_from] +
                2 * bus_active_constant_Z[bus_from] * Vm[bus_from]
            diag_elements[4] +=
                bus_reactive_constant_I[bus_from] +
                2 * bus_reactive_constant_Z[bus_from] * Vm[bus_from]
            Jv[row_from_p, col_from_vm] = diag_elements[3]  # ∂P∂V_from
            Jv[row_from_q, col_from_vm] = diag_elements[4]  # ∂Q∂V_from
        elseif data.bus_type[bus_from, time_step] == PSY.ACBusTypes.PV
            Jv[row_from_q, col_from_vm] = -1.0
            Jv[row_from_p, col_from_va] = diag_elements[1]  # ∂P∂θ_from
            Jv[row_from_q, col_from_va] = diag_elements[2]  # ∂Q∂θ_from
        elseif data.bus_type[bus_from, time_step] == PSY.ACBusTypes.REF
            Jv[row_from_p, col_from_vm] = -bus_slack_participation_factors[bus_from]
            Jv[row_from_q, col_from_va] = -1.0
        end
    end

    # Distributed slack cross-terms: for each participating bus k (other than the
    # REF bus), the active power residual depends on the REF bus state variable
    # x[2*ref-1] through the slack distribution: ∂F_P_k/∂x[2*ref-1] = -c_k.
    for (ref_bus, subnetwork_buses) in subnetworks
        col_ref = 2 * ref_bus - 1
        for bus_k in subnetwork_buses
            bus_k == ref_bus && continue
            c_k = bus_slack_participation_factors[bus_k]
            c_k == 0.0 && continue
            Jv[2 * bus_k - 1, col_ref] = -c_k
        end
    end

    _set_entries_for_lcc(data, Jv, num_buses, time_step)
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
