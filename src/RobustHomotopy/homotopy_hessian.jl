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
    IS.@assert_op M.colptr == A.colptr
    IS.@assert_op M.rowval == A.rowval
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

function _bus_V(data::ACPowerFlowData, bus_ix::Int, time_step::Int)
    if data.bus_type[bus_ix, time_step] == PSY.ACBusTypes.PQ
        return 1.0
    else
        return data.bus_magnitude[bus_ix, time_step]
    end
end

function homotopy_x0(data::ACPowerFlowData, time_step::Int)
    x = calculate_x0(data, time_step)
    for (bus_ix, bt) in enumerate(get_bus_type(data)[:, time_step]) # PERF: allocating
        if bt == PSY.ACBusTypes.PQ
            x[2 * bus_ix - 1] = 1.0
        end
    end
    # Force every LCC's ϕ_s to start strictly interior. The homotopy
    # pulls V_PQ to 1.0; if α_s starts at α_s_min and the LCC's β_s/(V·t)
    # is comparable to ~1, the arccos argument can fall onto the clamp
    # boundary (-1 or +1), where Q_s's second derivatives are singular
    # and the Hessian assembly produces an ill-scaled search direction.
    # Bumping α_s just enough to keep ϕ_s interior (plus a small margin)
    # avoids the degenerate starting state. The residual F_{α_s} = α_s -
    # α_{s,min} then drives α_s back toward its min as the homotopy
    # progresses.
    n_lcc = size(data.lcc.p_set, 1)
    if n_lcc > 0
        num_buses = first(size(data.bus_type))
        for i in 1:n_lcc
            offset_lcc = num_buses * 2 + (i - 1) * 4
            fb, tb = data.lcc.bus_indices[i]
            # need V for β/(V·t) threshold computation.
            # at PQ buses, `x[2·bus−1]` is the voltage magnitude (which we just set to 1.0)
            # but at PV/REF, have to go check data.voltage_magnitude for the setpoint.
            V_fb, V_tb = _bus_V(data, fb, time_step), _bus_V(data, tb, time_step)
            t_r, t_i = x[offset_lcc + 1], x[offset_lcc + 2]
            I_dc = data.lcc.i_dc[i, time_step]
            β_r = data.lcc.rectifier.transformer_reactance[i] * I_dc / sqrt(2)
            β_i = data.lcc.inverter.transformer_reactance[i] * I_dc / sqrt(2)
            α_r_min = data.lcc.rectifier.min_thyristor_angle[i]
            α_i_min = data.lcc.inverter.min_thyristor_angle[i]
            margin = 0.05
            # u_r = cos α_r - β_r/(V·t) ∈ (-1, 1) requires α_r off the
            # rectifier-clamp threshold; u_i = -cos α_i - β_i/(V·t) ∈
            # (-1, 1) requires α_i > acos(1 - β_i/(V·t)).
            min_α_r_interior =
                if β_r ≥ V_fb * t_r
                    acos(clamp(β_r / (V_fb * t_r) - 1.0, -1.0, 1.0)) + margin
                else
                    margin
                end
            min_α_i_interior =
                acos(clamp(1.0 - β_i / (V_tb * t_i), -1.0, 1.0)) + margin
            x[offset_lcc + 3] = max(α_r_min, min_α_r_interior)
            x[offset_lcc + 4] = max(α_i_min, min_α_i_interior)
        end
    end
    return x
end

function HomotopyHessian(data::ACPowerFlowData, time_step::Int)
    pfResidual = ACPowerFlowResidual(data, time_step)
    J = ACPowerFlowJacobian(pfResidual, time_step)
    # Allocate Hv with the maximal sparsity pattern of J' * J. Sparse `*`
    # currently preserves structural zeros, but that isn't a documented
    # SparseArrays contract, so we defensively fill nzval with ones to force
    # the maximal pattern. We then restore J.Jv's original nzval — some
    # entries (e.g. LCC angle-constraint diagonals of 1.0) are set at
    # structure creation and not rewritten by subsequent J(time_step) calls.
    # The per-call IS.@assert_op in A_plus_eq_BT_B! guards against any future
    # change in Julia that would drop structural zeros at runtime.
    original_J_nzval = copy(SparseArrays.nonzeros(J.Jv))
    fill!(SparseArrays.nonzeros(J.Jv), 1.0)
    Hv = J.Jv' * J.Jv
    SparseArrays.nonzeros(Hv) .= 0.0
    copyto!(SparseArrays.nonzeros(J.Jv), original_J_nzval)
    nbuses = size(get_bus_type(data), 1)
    n_lcc = size(data.lcc.p_set, 1)
    n_state = 2 * nbuses + 4 * n_lcc
    PQ_mask = get_bus_type(data)[:, time_step] .== (PSY.ACBusTypes.PQ,)
    # PQ_V_mags marks the V_mag coordinate at each PQ bus; LCC state slots
    # (tap, thyristor angle) are excluded — the homotopy continuation
    # `(1 − t_k)·(x − 1)` only pulls bus voltages toward 1.0.
    PQ_V_mags = Vector{Bool}(undef, n_state)
    PQ_V_mags[1:(2 * nbuses)] .= collect(Iterators.flatten(zip(PQ_mask, falses(nbuses))))
    PQ_V_mags[(2 * nbuses + 1):n_state] .= false
    return HomotopyHessian(data, pfResidual, J, PQ_V_mags, zeros(n_state), Hv)
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
    _update_hessian_lcc_contributions!(Hv, F_value, data, time_step)
    return
end

"""
    _update_hessian_lcc_contributions!(Hv, F, data, time_step)

Add per-LCC contributions to the residual-Hessian sum `∑_k F_k ∇² F_k`.

For each LCC, the residual rows that depend on LCC state are the bus
`(P, Q)`-balance rows at both AC terminals plus the two tail rows
`(F_{t_r}, F_{t_i})`. (The two `α`-constraint tail rows are linear, so
`∇² F = 0`.) The bus-row contributions to the Hessian come from the LCC
self-admittance terms `P_s(V_s, t_s, α_s)` and `Q_s(V_s, t_s, α_s)`,
which the network-only Hessian assembly above does not include. The tail
rows are linear combinations of `P_r` and `P_i`, so they also reduce to
the same `∇² P_s` blocks.

The Hessian additions are block-diagonal between the rectifier
`(V_{f_b}, t_r, α_r)` and inverter `(V_{t_b}, t_i, α_i)` coordinates of
each LCC: `P_r, Q_r` have no `inverter`-state dependence and vice versa.
The sparsity pattern of these entries is already covered by `J' * J`
(every rectifier-side column has structural support at rows
`{P_{f_b}, Q_{f_b}, F_{t_r}, F_{t_i}}`, so all 3×3 cross-terms exist).
"""
function _update_hessian_lcc_contributions!(
    Hv::SparseArrays.SparseMatrixCSC{Float64, J_INDEX_TYPE},
    F::Vector{Float64},
    data::ACPowerFlowData,
    time_step::Int64,
)
    n_lcc = size(data.lcc.p_set, 1)
    n_lcc == 0 && return
    num_buses = first(size(data.bus_type))
    Vm = view(data.bus_magnitude, :, time_step)
    for (i, (fb, tb)) in enumerate(data.lcc.bus_indices)
        bt_fb = data.bus_type[fb, time_step]
        bt_tb = data.bus_type[tb, time_step]
        offset_lcc = num_buses * 2 + (i - 1) * 4
        idx_V_fb = 2 * fb - 1
        idx_V_tb = 2 * tb - 1
        idx_t_r = offset_lcc + 1
        idx_t_i = offset_lcc + 2
        idx_α_r = offset_lcc + 3
        idx_α_i = offset_lcc + 4

        tap_r = data.lcc.rectifier.tap[i, time_step]
        tap_i = data.lcc.inverter.tap[i, time_step]
        α_r = data.lcc.rectifier.thyristor_angle[i, time_step]
        α_i = data.lcc.inverter.thyristor_angle[i, time_step]
        ϕ_r = data.lcc.rectifier.phi[i, time_step]
        ϕ_i = data.lcc.inverter.phi[i, time_step]
        x_t_r = data.lcc.rectifier.transformer_reactance[i]
        x_t_i = data.lcc.inverter.transformer_reactance[i]
        I_dc = max(data.lcc.i_dc[i, time_step], 1e-9)
        V_fb = Vm[fb]
        V_tb = Vm[tb]

        F_P_fb = F[2 * fb - 1]
        F_Q_fb = F[2 * fb]
        F_P_tb = F[2 * tb - 1]
        F_Q_tb = F[2 * tb]
        F_t_r = F[idx_t_r]
        F_t_i = F[idx_t_i]

        # The P-setpoint tail row F_{t_r} carries +P_lcc_from when the
        # setpoint is at the rectifier and -P_lcc_to otherwise (see
        # `_write_lcc_tail!`). Its ∇²F therefore attaches to the rectifier
        # curvature `d2P_r` in the first case and to the inverter curvature
        # `d2P_i` (negated, since the residual carries -P_lcc_to) in the
        # second. This mirrors the side-aware Jacobian assembly in
        # `_lcc_jacobian_scalars`; the two must agree or the
        # Hessian-of-residual term would not match J^T J's quadratic. The
        # DC-line-balance row F_{t_i} = P_lcc_from + P_lcc_to - R·I_dc² and
        # the bus-balance rows F_{P_fb}/F_{P_tb} depend on their own side
        # unconditionally.
        setpoint_at_rect = data.lcc.setpoint_at_rectifier[i]
        coef_Pr = F_P_fb + F_t_i + (setpoint_at_rect ? F_t_r : 0.0)
        coef_Qr = F_Q_fb
        coef_Pi = F_P_tb + F_t_i + (setpoint_at_rect ? 0.0 : -F_t_r)
        coef_Qi = F_Q_tb

        d2P_r = _d2P_lcc(V_fb, tap_r, α_r, I_dc, +1)
        d2Q_r = _d2Q_lcc(V_fb, tap_r, α_r, x_t_r, I_dc, ϕ_r, +1)
        d2P_i = _d2P_lcc(V_tb, tap_i, α_i, I_dc, -1)
        d2Q_i = _d2Q_lcc(V_tb, tap_i, α_i, x_t_i, I_dc, ϕ_i, -1)

        # Rectifier 3×3 block on (V_fb, t_r, α_r).
        if bt_fb == PSY.ACBusTypes.PQ
            VV = coef_Pr * d2P_r.VV + coef_Qr * d2Q_r.VV
            Vt_r = coef_Pr * d2P_r.Vt + coef_Qr * d2Q_r.Vt
            Vα_r = coef_Pr * d2P_r.Vα + coef_Qr * d2Q_r.Vα
            Hv[idx_V_fb, idx_V_fb] += VV
            Hv[idx_V_fb, idx_t_r] += Vt_r
            Hv[idx_t_r, idx_V_fb] += Vt_r
            Hv[idx_V_fb, idx_α_r] += Vα_r
            Hv[idx_α_r, idx_V_fb] += Vα_r
        end
        tt_r = coef_Pr * d2P_r.tt + coef_Qr * d2Q_r.tt
        tα_r = coef_Pr * d2P_r.tα + coef_Qr * d2Q_r.tα
        αα_r = coef_Pr * d2P_r.αα + coef_Qr * d2Q_r.αα
        Hv[idx_t_r, idx_t_r] += tt_r
        Hv[idx_t_r, idx_α_r] += tα_r
        Hv[idx_α_r, idx_t_r] += tα_r
        Hv[idx_α_r, idx_α_r] += αα_r

        # Inverter 3×3 block on (V_tb, t_i, α_i).
        if bt_tb == PSY.ACBusTypes.PQ
            VV = coef_Pi * d2P_i.VV + coef_Qi * d2Q_i.VV
            Vt_i = coef_Pi * d2P_i.Vt + coef_Qi * d2Q_i.Vt
            Vα_i = coef_Pi * d2P_i.Vα + coef_Qi * d2Q_i.Vα
            Hv[idx_V_tb, idx_V_tb] += VV
            Hv[idx_V_tb, idx_t_i] += Vt_i
            Hv[idx_t_i, idx_V_tb] += Vt_i
            Hv[idx_V_tb, idx_α_i] += Vα_i
            Hv[idx_α_i, idx_V_tb] += Vα_i
        end
        tt_i = coef_Pi * d2P_i.tt + coef_Qi * d2Q_i.tt
        tα_i = coef_Pi * d2P_i.tα + coef_Qi * d2Q_i.tα
        αα_i = coef_Pi * d2P_i.αα + coef_Qi * d2Q_i.αα
        Hv[idx_t_i, idx_t_i] += tt_i
        Hv[idx_t_i, idx_α_i] += tα_i
        Hv[idx_α_i, idx_t_i] += tα_i
        Hv[idx_α_i, idx_α_i] += αα_i
    end
    return
end
