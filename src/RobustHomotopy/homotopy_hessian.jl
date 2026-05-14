struct HomotopyHessian
    # PERF: data is stored in triplicate: here, inside pfResidual, and inside J.
    data::ACPowerFlowData
    pfResidual::ACPowerFlowResidual
    J::ACPowerFlowJacobian
    PQ_V_mags::BitVector # true iff that coordinate in the state vector is V_mag at a PQ bus
    grad::Vector{Float64}
    Hv::SparseMatrixCSC{Float64, J_INDEX_TYPE}
end

"""Does `A += B' * B`, in a way that preserves the sparse structure of `A`, if possible.
A workaround for the fact that Julia seems to run `dropzeros!(A)` automatically if I just 
do `A .+= B' * B`."""
function A_plus_eq_BT_B!(A::SparseMatrixCSC, B::SparseMatrixCSC)
    M = B' * B # shouldn't this be allocating too?
    @assert M.colptr == A.colptr && M.rowval == A.rowval
    A.nzval .+= M.nzval
    return
end

"""Compute value of gradient and Hessian at x."""
function (hess::HomotopyHessian)(x::Vector{Float64}, t_k::Float64, time_step::Int)
    hess.pfResidual(x, time_step)
    Rv = hess.pfResidual.Rv
    hess.J(time_step)
    Jv = hess.J.Jv
    _update_hessian_matrix_values!(hess.Hv, Rv, hess.data, time_step)
    A_plus_eq_BT_B!(hess.Hv, Jv)
    SparseArrays.nonzeros(hess.Hv) .*= t_k
    for (bus_ix, bt) in enumerate(get_bus_type(hess.data)[:, time_step]) # PERF: allocating
        if bt == PSY.ACBusTypes.PQ
            hess.Hv[2 * bus_ix - 1, 2 * bus_ix - 1] += (1 - t_k)
        end
    end
    # PERF: allocating
    hess.grad .= (1 - t_k) * hess.PQ_V_mags .* (x - ones(size(x, 1))) + t_k * Jv' * Rv
    return
end

function F_value(hess::HomotopyHessian, t_k::Float64, x::Vector{Float64}, time_step::Int)
    hess.pfResidual(x, time_step)
    Rv = hess.pfResidual.Rv
    φ_vector = x[hess.PQ_V_mags] .- 1.0 # PERF: allocating
    F_value = (1 - t_k) * 0.5 * dot(φ_vector, φ_vector) + t_k * 0.5 * dot(Rv, Rv)
    return F_value
end

# slightly confusing that I have the field grad, and the argument grad.
function gradient_value!(grad::Vector{Float64},
    hess::HomotopyHessian,
    t_k::Float64,
    x::Vector{Float64},
    time_step::Int,
)
    hess.pfResidual(x, time_step)
    hess.J(time_step) # PERF bottleneck. Look into a different line search strategy?
    # or otherwise reduce the number of gradient computations?
    # for a 10k bus system, computing J takes over 10x longer than computing F.
    Jv = hess.J.Jv
    mask = hess.PQ_V_mags
    # PERF: allocating
    grad .= (1 - t_k) * (mask .* (x - ones(size(x, 1)))) + t_k * Jv' * hess.pfResidual.Rv
    return grad
end

function homotopy_x0(data::ACPowerFlowData, time_step::Int)
    x = calculate_x0(data, time_step)
    for (bus_ix, bt) in enumerate(get_bus_type(data)[:, time_step]) # PERF: allocating
        if bt == PSY.ACBusTypes.PQ
            x[2 * bus_ix - 1] = 1.0
        end
    end
    return x
end

function HomotopyHessian(data::ACPowerFlowData, time_step::Int)
    pfResidual = ACPowerFlowResidual(data, time_step)
    Hv = _create_hessian_matrix_structure(data, time_step)
    J = ACPowerFlowJacobian(pfResidual, time_step)
    nbuses = size(get_bus_type(data), 1)
    PQ_mask = get_bus_type(data)[:, time_step] .== (PSY.ACBusTypes.PQ,)
    PQ_V_mags = collect(Iterators.flatten(zip(PQ_mask, falses(nbuses))))
    return HomotopyHessian(data, pfResidual, J, PQ_V_mags, zeros(2 * nbuses), Hv)
end

function _create_hessian_matrix_structure(data::ACPowerFlowData, ::Int64)
    rows = J_INDEX_TYPE[]      # I
    columns = J_INDEX_TYPE[]   # J
    values = Float64[]  # V

    # an over-estimate: I want ordered pairs of vertices that are 2 or fewer
    # steps apart, whereas this counts directed paths of 2 edges.
    numEdgePairs = sum(x -> (length(x) - 1)^2, get_neighbor(data); init = 0.0)
    numEdgePairs += length(get_arc_axis(data))
    sizehint!(rows, 4 * numEdgePairs)
    sizehint!(columns, 4 * numEdgePairs)
    sizehint!(values, 4 * numEdgePairs)

    # H is J'*J + c*I
    # J' * J is dot products of pairs of columns.
    # so look at pairs of columns and check if there's a row in which both are nonzero.
    # i.e. look at pairs of buses and see if they have a neighbor in common.

    visited = Set{Tuple{Int, Int}}()
    # non structurally zero block correspond to pairs of buses with a neighbor in common.
    # instead of checking pairs of buses O(n^2 d), we travel 2 hops from each bus O(n d^2)
    # (d = degree of typical vertex, n = number of vertices; d << n.)
    for (ind, nbrs) in enumerate(get_neighbor(data))
        for nbr in nbrs
            for nbrOfNbr in get_neighbor(data)[nbr]
                if !((ind, nbrOfNbr) in visited)
                    bus_from, bus_to = ind, nbrOfNbr
                    # PERF: J.Jv^T * J.Jv would have fewer nonzero entries if J.Jv's sparse
                    # structure took into account the bus type
                    for (i, j) in Iterators.product((2 * bus_from - 1, 2 * bus_from),
                        (2 * bus_to - 1, 2 * bus_to))
                        push!(rows, i) # PERF: allocating
                        push!(columns, j) # PERF: allocating
                        push!(values, 0.0) # (oddly enough, this line is non-allocating)
                    end
                    push!(visited, (ind, nbrOfNbr))
                end
            end
        end
    end
    return SparseArrays.sparse(rows, columns, values)
end

"""
    _update_hessian_matrix_values!(
        Hv::SparseMatrixCSC{Float64, $J_INDEX_TYPE},
        F_value::Vector{Float64},
        data::ACPowerFlowData,
        time_step::Int64
    )

Update the Hessian matrix values for the robust homotopy power flow solver.

# Description

This function sets `Hv` equal to:

```math
\\sum_{k=1}^{2n} F_k(x) H_{F_k}(x)
```

where ``F_k`` denotes the ``k``th power balance equation and ``H_{F_k}`` denotes its Hessian matrix.

This computes only the terms in the Hessian that come from the second derivatives of the power balance equations. 
The full Hessian of the objective function also includes a ``J^T J`` term, which is computed separately.

# Sparse Structure

The Hessian is organized into 2×2 blocks, each corresponding to a pair of buses. For a pair of buses ``i`` and ``k`` 
connected by a branch, the sparse structure of their block depends on the bus types:

```math
\\begin{array}{c|cc|cc|cc}
 & \\text{REF} & & \\text{PV} & & \\text{PQ} & \\\\
 & P_i & Q_i & Q_i & V_i & V_i & \\theta_i \\\\
\\hline
\\text{REF: } P_k & & & & & & \\\\
Q_k & & & & & & \\\\
\\hline
\\text{PV: } Q_k & & & & & & \\\\
V_k & & & & \\bullet & \\bullet & \\bullet \\\\
\\hline
\\text{PQ: } V_k & & & & \\bullet & \\bullet & \\bullet \\\\
\\theta_k & & & & \\bullet & \\bullet & \\bullet
\\end{array}
```

where ``\\bullet`` represents a potentially non-zero entry.

Diagonal blocks (where ``i = k``) follow the same pattern as if each bus is its own neighbor.
Off-diagonal blocks for pairs of buses not connected by a branch are structurally zero.

# Arguments
- `Hv::SparseMatrixCSC{Float64, $J_INDEX_TYPE}`: The Hessian matrix to be updated (modified in-place).
- `F_value::Vector{Float64}`: Current values of the power balance residuals.
- `data::ACPowerFlowData`: The power flow data containing bus and network information.
- `time_step::Int64`: The time step for which to compute the Hessian.
"""
function _update_hessian_matrix_values!(
    Hv::SparseArrays.SparseMatrixCSC{Float64, J_INDEX_TYPE},
    F_value::Vector{Float64},
    data::ACPowerFlowData,
    time_step::Int64)
    Yb = data.power_network_matrix.data
    Vm = view(data.bus_magnitude, :, time_step)
    θ = view(data.bus_angles, :, time_step)
    num_buses = first(size(data.bus_type))
    SparseArrays.nonzeros(Hv) .= 0.0
    for i in 1:num_buses
        bt_i = data.bus_type[i, time_step]
        Pi_θiθi, Qi_θiθi = 0.0, 0.0
        Pi_Viθi, Qi_Viθi = 0.0, 0.0
        has_θi = (bt_i == PSY.ACBusTypes.PQ) || (bt_i == PSY.ACBusTypes.PV)
        for k in data.neighbors[i]
            if i != k
                bt_k = data.bus_type[k, time_step]
                Gik, Bik = real(Yb[i, k]), imag(Yb[i, k])
                has_θk = (bt_k == PSY.ACBusTypes.PQ) || (bt_k == PSY.ACBusTypes.PV)
                # the partials where all 3 indices are different vanish
                # naively count 8 with 2 distinct indices: {∂Vₖ, ∂θₖ} x {∂Vₖ, ∂θₖ, ∂Vᵢ, ∂θᵢ}
                # but can reduce to 6: ∂²/∂Vₖ∂θₖ = ∂²/∂θₖ∂Vₖ, and ∂²Δ{Pᵢ, Qᵢ}/∂²Vₖ is 0.
                # start with the 4 involving ∂θₖ, then do remaining the 2 involving ∂Vₖ
                if has_θk
                    # ∂²Δ{Pᵢ, Qᵢ}/∂²θₖ
                    Pi_θkθk =
                        Vm[i] * Vm[k] * ( # = Vm[k] * Qi_θkVk
                            -Gik * cos(θ[i] - θ[k])
                            -
                            Bik * sin(θ[i] - θ[k])
                        )
                    Qi_θkθk =
                        Vm[i] * Vm[k] * ( # = -Vm[k] * Pi_θkVk
                            -Gik * sin(θ[i] - θ[k])
                            +
                            Bik * cos(θ[i] - θ[k])
                        )
                    θkθks = Pi_θkθk * F_value[2 * i - 1] + Qi_θkθk * F_value[2 * i]
                    Hv[2 * k, 2 * k] += θkθks
                end
                if bt_k == PSY.ACBusTypes.PQ
                    # ∂²Δ{Pᵢ, Qᵢ}/∂θₖ∂Vₖ
                    Pi_θkVk = Vm[i] * (
                        Gik * sin(θ[i] - θ[k])
                        -
                        Bik * cos(θ[i] - θ[k])
                    )
                    Qi_θkVk = Vm[i] * (
                        -Gik * cos(θ[i] - θ[k])
                        -
                        Bik * sin(θ[i] - θ[k])
                    )
                    θkVks = Pi_θkVk * F_value[2 * i - 1] + Qi_θkVk * F_value[2 * i]
                    Hv[2 * k - 1, 2 * k] += θkVks
                    Hv[2 * k, 2 * k - 1] += θkVks
                end
                if has_θi
                    Pi_θkθi =
                        Vm[i] * Vm[k] * (
                            Gik * cos(θ[i] - θ[k]) +
                            Bik * sin(θ[i] - θ[k])
                        )
                    Qi_θkθi =
                        Vm[i] * Vm[k] * (
                            Gik * sin(θ[i] - θ[k])
                            -
                            Bik * cos(θ[i] - θ[k])
                        )
                    # contribution towards sum in ∂²Δ{Pᵢ, Qᵢ}/∂θᵢ∂θᵢ
                    Pi_θiθi -= Pi_θkθi
                    Qi_θiθi -= Qi_θkθi
                    if has_θk
                        # ∂²Δ{Pᵢ, Qᵢ}/∂θₖ∂θᵢ
                        θiθks = Pi_θkθi * F_value[2 * i - 1] + Qi_θkθi * F_value[2 * i]
                        Hv[2 * i, 2 * k] += θiθks
                        Hv[2 * k, 2 * i] += θiθks
                    end
                end
                if bt_i == PSY.ACBusTypes.PQ
                    Pi_θkVi = Vm[k] * ( # = Vm[k] * Qi_VkVi 
                        Gik * sin(θ[i] - θ[k])
                        -
                        Bik * cos(θ[i] - θ[k])
                    )
                    Qi_θkVi = Vm[k] * ( # = -Vm[k] * Pi_VkVi 
                        -Gik * cos(θ[i] - θ[k])
                        -
                        Bik * sin(θ[i] - θ[k])
                    )
                    # contribution towards sum in ∂²Δ{Pᵢ, Qᵢ}/∂θᵢ∂Vᵢ
                    Pi_Viθi -= Pi_θkVi
                    Qi_Viθi -= Qi_θkVi
                    if has_θk
                        # ∂²Δ{Pᵢ, Qᵢ}/∂θₖ∂Vᵢ
                        Viθks = Pi_θkVi * F_value[2 * i - 1] + Qi_θkVi * F_value[2 * i]
                        Hv[2 * i - 1, 2 * k] += Viθks
                        Hv[2 * k, 2 * i - 1] += Viθks
                    end
                end
                if bt_k == PSY.ACBusTypes.PQ && has_θi
                    # ∂²Δ{Pᵢ, Qᵢ}/∂Vₖ∂θᵢ
                    Pi_Vkθi = Vm[i] * ( # = -Vm[i] * Qi_VkVi 
                        -Gik * sin(θ[i] - θ[k])
                        +
                        Bik * cos(θ[i] - θ[k])
                    )
                    Qi_Vkθi = Vm[i] * ( # = Vm[i] * Pi_VkVi 
                        Gik * cos(θ[i] - θ[k])
                        +
                        Bik * sin(θ[i] - θ[k])
                    )
                    θiVks = Pi_Vkθi * F_value[2 * i - 1] + Qi_Vkθi * F_value[2 * i]
                    Hv[2 * i, 2 * k - 1] += θiVks
                    Hv[2 * k - 1, 2 * i] += θiVks
                end
                if bt_k == PSY.ACBusTypes.PQ && bt_i == PSY.ACBusTypes.PQ
                    # ∂²Δ{Pᵢ, Qᵢ}/∂Vₖ∂Vᵢ
                    Pi_VkVi = Gik * cos(θ[i] - θ[k]) + Bik * sin(θ[i] - θ[k])
                    Qi_VkVi = Gik * sin(θ[i] - θ[k]) - Bik * cos(θ[i] - θ[k])
                    ViVks = Pi_VkVi * F_value[2 * i - 1] + Qi_VkVi * F_value[2 * i]
                    Hv[2 * i - 1, 2 * k - 1] += ViVks
                    Hv[2 * k - 1, 2 * i - 1] += ViVks
                end
            end
        end
        # now, do the diagonal terms that depend only on i: these are sums [except for ∂²Vᵢ],
        # but we've been accumulating the sums as we go.

        # ∂²Δ{Pᵢ, Qᵢ}/∂²θᵢ: PQ and PV
        if has_θi
            θiθis = Pi_θiθi * F_value[2 * i - 1] + Qi_θiθi * F_value[2 * i]
            Hv[2 * i, 2 * i] += θiθis
        end

        # ∂²Δ{Pᵢ, Qᵢ}/∂Vᵢ∂θᵢ and ∂²Δ{Pᵢ, Qᵢ}/∂²Vᵢ: PQ only.
        if bt_i == PSY.ACBusTypes.PQ
            Viθis = Pi_Viθi * F_value[2 * i - 1] + Qi_Viθi * F_value[2 * i]
            Hv[2 * i, 2 * i - 1] += Viθis
            Hv[2 * i - 1, 2 * i] += Viθis

            Pi_ViVi = 2 * real(Yb[i, i])
            Qi_ViVi = -2 * imag(Yb[i, i])

            ViVis = Pi_ViVi * F_value[2 * i - 1] + Qi_ViVi * F_value[2 * i]
            Hv[2 * i - 1, 2 * i - 1] += ViVis
        end
    end
    return
end
