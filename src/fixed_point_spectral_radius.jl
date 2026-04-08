"""
    acpf_hvvp(data, time_step, v, u) -> Vector{Float64}

Compute the bilinear form of every component Hessian of the AC power flow residual
with two vectors `v` and `u`:

```math
w[k] = v^\\top \\left(\\frac{\\partial^2 F_k}{\\partial x \\partial x^\\top}\\right) u
```

where ``F_k`` is the ``k``-th component of the AC power flow residual at the current
state stored in `data` (i.e., `data.bus_magnitude` and `data.bus_angles`).

This is matrix-free: the residual's 3-tensor of second derivatives is never
materialized. See [`_create_jacobian_matrix_structure`](@ref) for the
bus-type-dependent state vector convention.

LCC state variables are not yet supported; the returned vector covers only the
`2 * num_buses` bus state entries.
"""
function acpf_hvvp(
    data::ACPowerFlowData,
    time_step::Int,
    v::AbstractVector{Float64},
    u::AbstractVector{Float64},
)
    Yb = data.power_network_matrix.data
    Vm = view(data.bus_magnitude, :, time_step)
    θ = view(data.bus_angles, :, time_step)
    num_buses = first(size(data.bus_type))
    n = 2 * num_buses
    @assert length(v) >= n && length(u) >= n
    w = zeros(Float64, n)

    for i in 1:num_buses
        bt_i = data.bus_type[i, time_step]
        has_θi = (bt_i == PSY.ACBusTypes.PQ) || (bt_i == PSY.ACBusTypes.PV)
        Pi_θiθi, Qi_θiθi = 0.0, 0.0
        Pi_Viθi, Qi_Viθi = 0.0, 0.0

        for k in data.neighbors[i]
            k == i && continue
            bt_k = data.bus_type[k, time_step]
            has_θk = (bt_k == PSY.ACBusTypes.PQ) || (bt_k == PSY.ACBusTypes.PV)
            Gik, Bik = real(Yb[i, k]), imag(Yb[i, k])
            θik = θ[i] - θ[k]
            s, c = sin(θik), cos(θik)

            if has_θi
                d2P_θkθi = Vm[i] * Vm[k] * (Gik * c + Bik * s)
                d2Q_θkθi = Vm[i] * Vm[k] * (Gik * s - Bik * c)
                # contribution towards sum in ∂²Δ{Pᵢ,Qᵢ}/∂θᵢ∂θᵢ
                Pi_θiθi -= d2P_θkθi
                Qi_θiθi -= d2Q_θkθi

                if has_θk
                    # ∂²Δ{Pᵢ,Qᵢ}/∂θₖ∂θᵢ
                    # really ∂²ΔPᵢ/∂θₖ∂θᵢ * v[θₖ] * u[θᵢ] + ∂²ΔQᵢ/∂θᵢ∂θₖ * v[θᵢ] * u[θₖ]
                    # but ∂²ΔPᵢ/∂θₖ∂θᵢ equals ∂²ΔPᵢ/∂θᵢ∂θₖ.
                    w[2 * i - 1] += d2P_θkθi * (v[2 * k] * u[2 * i] + v[2 * i] * u[2 * k])
                    w[2 * i] += d2Q_θkθi * (v[2 * k] * u[2 * i] + v[2 * i] * u[2 * k])
                end
            end

            if bt_i == PSY.ACBusTypes.PQ
                d2P_θkVi = Vm[k] * (Gik * s - Bik * c)
                d2Q_θkVi = Vm[k] * (-Gik * c - Bik * s)
                # contribution towards sum in ∂²Δ{Pᵢ,Qᵢ}/∂θᵢ∂Vᵢ
                Pi_Viθi -= d2P_θkVi
                Qi_Viθi -= d2Q_θkVi

                if has_θk
                    # ∂²Δ{Pᵢ,Qᵢ}/∂θₖ∂Vᵢ
                    w[2 * i - 1] +=
                        d2P_θkVi * (v[2 * k] * u[2 * i - 1] + v[2 * i - 1] * u[2 * k])
                    w[2 * i] +=
                        d2Q_θkVi * (v[2 * k] * u[2 * i - 1] + v[2 * i - 1] * u[2 * k])
                end
            end

            if has_θk
                # ∂²Δ{Pᵢ,Qᵢ}/∂²θₖ
                d2P_θkθk = Vm[i] * Vm[k] * (-Gik * c - Bik * s)
                d2Q_θkθk = Vm[i] * Vm[k] * (-Gik * s + Bik * c)
                w[2 * i - 1] += d2P_θkθk * v[2 * k] * u[2 * k]
                w[2 * i] += d2Q_θkθk * v[2 * k] * u[2 * k]
            end

            if bt_k == PSY.ACBusTypes.PQ
                # ∂²Δ{Pᵢ,Qᵢ}/∂θₖ∂Vₖ
                d2P_θkVk = Vm[i] * (Gik * s - Bik * c)
                d2Q_θkVk = Vm[i] * (-Gik * c - Bik * s)
                w[2 * i - 1] +=
                    d2P_θkVk * (v[2 * k] * u[2 * k - 1] + v[2 * k - 1] * u[2 * k])
                w[2 * i] +=
                    d2Q_θkVk * (v[2 * k] * u[2 * k - 1] + v[2 * k - 1] * u[2 * k])

                if bt_i == PSY.ACBusTypes.PQ
                    # ∂²Δ{Pᵢ,Qᵢ}/∂Vₖ∂Vᵢ
                    d2P_VkVi = Gik * c + Bik * s
                    d2Q_VkVi = Gik * s - Bik * c
                    w[2 * i - 1] +=
                        d2P_VkVi *
                        (v[2 * k - 1] * u[2 * i - 1] + v[2 * i - 1] * u[2 * k - 1])
                    w[2 * i] +=
                        d2Q_VkVi *
                        (v[2 * k - 1] * u[2 * i - 1] + v[2 * i - 1] * u[2 * k - 1])
                end

                if has_θi
                    # ∂²Δ{Pᵢ,Qᵢ}/∂Vₖ∂θᵢ
                    d2P_Vkθi = -Vm[i] * (Gik * s - Bik * c)
                    d2Q_Vkθi = -Vm[i] * (-Gik * c - Bik * s)
                    w[2 * i - 1] +=
                        d2P_Vkθi * (v[2 * k - 1] * u[2 * i] + v[2 * i] * u[2 * k - 1])
                    w[2 * i] +=
                        d2Q_Vkθi * (v[2 * k - 1] * u[2 * i] + v[2 * i] * u[2 * k - 1])
                end
            end
        end

        # diagonal terms in i: accumulated sums [except ∂²Vᵢ]
        if has_θi
            # ∂²Δ{Pᵢ,Qᵢ}/∂²θᵢ
            w[2 * i - 1] += Pi_θiθi * v[2 * i] * u[2 * i]
            w[2 * i] += Qi_θiθi * v[2 * i] * u[2 * i]
        end

        if bt_i == PSY.ACBusTypes.PQ
            # ∂²Δ{Pᵢ,Qᵢ}/∂Vᵢ∂θᵢ
            w[2 * i - 1] +=
                Pi_Viθi * (v[2 * i - 1] * u[2 * i] + v[2 * i] * u[2 * i - 1])
            w[2 * i] +=
                Qi_Viθi * (v[2 * i - 1] * u[2 * i] + v[2 * i] * u[2 * i - 1])

            # ∂²Δ{Pᵢ,Qᵢ}/∂²Vᵢ
            d2P_ViVi = 2.0 * real(Yb[i, i])
            d2Q_ViVi = -2.0 * imag(Yb[i, i])
            w[2 * i - 1] += d2P_ViVi * v[2 * i - 1] * u[2 * i - 1]
            w[2 * i] += d2Q_ViVi * v[2 * i - 1] * u[2 * i - 1]
        end
    end

    return w
end

"""
    compute_fixed_point_spectral_radius(data, time_step; x0, tol, maxiter, krylovdim)
        -> (ρ::Float64, info, condest::Float64)

Estimate the spectral radius of the Jacobian of the Newton fixed-point map

```math
g(x) = x - J(x)^{-1} F(x)
```

at the state `x0` (default: the result of `calculate_x0(data, time_step)`). The
spectral radius ``\\rho(\\partial g / \\partial x)`` is a local convergence
diagnostic for Newton-Raphson: ``\\rho < 1`` is necessary (and locally sufficient)
for the fixed-point iteration to converge from a small enough neighborhood of
``x``. Values close to or above 1 indicate that NR is at risk of stalling or
diverging from this starting point.

Differentiating ``g`` and using ``\\partial F / \\partial x = J`` gives the
identity

```math
(G v)_k = \\bigl(J^{-1} w\\bigr)_k,
\\qquad w_k = v^\\top H_k u,
\\qquad u = J^{-1} F(x)
```

where ``H_k`` is the Hessian of the ``k``-th residual component. ``G`` is never
materialized: the matvec ``G v`` is computed as one [`acpf_hvvp`](@ref) call
(producing ``w``) followed by two `KLU` back-solves (one for ``u``, one for
``J^{-1} w``). KrylovKit's
matrix-free Lanczos / Arnoldi solver then extracts the eigenvalue of largest
magnitude.

# Notes

- At a true power-flow solution, ``F(x) = 0`` so ``u = 0`` and ``\\rho(G) = 0``
  trivially. The diagnostic is meaningful at *non-solution* iterates such as the
  flat-start `x0`.
- LCC state variables are not yet supported.
"""
function compute_fixed_point_spectral_radius(
    data::ACPowerFlowData,
    time_step::Int;
    x0::Union{Vector{Float64}, Nothing} = nothing,
    tol::Float64 = 1e-6,
    maxiter::Int = 200,
    krylovdim::Int = 30,
)
    if size(data.lcc.p_set, 1) > 0
        throw(
            ArgumentError(
                "compute_fixed_point_spectral_radius does not yet support LCC HVDC systems",
            ),
        )
    end

    residual = ACPowerFlowResidual(data, time_step)
    jac = ACPowerFlowJacobian(
        data,
        residual.bus_slack_participation_factors,
        residual.subnetworks,
        time_step,
    )
    x = isnothing(x0) ? calculate_x0(data, time_step) : copy(x0)
    residual(x, time_step)
    jac(time_step)
    return _fixed_point_spectral_radius!(
        data, residual, jac, time_step;
        tol = tol, maxiter = maxiter, krylovdim = krylovdim,
    )
end

"""In-place spectral radius computation that reuses an already-evaluated
`residual` and `jac` (both must have been called at the current state). Returns
`(ρ, info, condest)`, where `condest` is a Hager 1-norm estimate of the
condition number of `jac.Jv` computed from the same KLU factor used for the
spectral radius matvecs. Used by the per-iteration monitor inside
`_run_power_flow_method`."""
function _fixed_point_spectral_radius!(
    data::ACPowerFlowData,
    residual::ACPowerFlowResidual,
    jac::ACPowerFlowJacobian,
    time_step::Int;
    tol::Float64 = 1e-6,
    maxiter::Int = 200,
    krylovdim::Int = 30,
)
    n = 2 * size(data.bus_type, 1)
    F = KLU.klu(jac.Jv)
    u = F \ copy(residual.Rv)
    matvec(v::AbstractVector) = F \ acpf_hvvp(data, time_step, v, u)
    # Deterministic init for reproducibility across runs / CI logs.
    v_init = ones(Float64, n) ./ sqrt(n)
    vals, _, info = KrylovKit.eigsolve(
        matvec, v_init, 1, :LM;
        tol = tol, maxiter = maxiter, krylovdim = krylovdim,
    )
    ρ = isempty(vals) ? NaN : abs(vals[1])
    condest = KLU.condest(F)
    return ρ, info, condest
end
