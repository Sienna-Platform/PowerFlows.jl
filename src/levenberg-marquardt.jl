"""Pre-allocated workspace for the Levenberg-Marquardt solver.

Holds the augmented matrix `[J; √λ·I]` with a fixed sparsity pattern,
a mapping to update its entries in-place, and a cached QR factorization."""
mutable struct LMWorkspace
    A::SparseMatrixCSC{Float64, Int64}
    # PERF: if we're doing dense access, I'd expect a bitmask to be faster.
    #       Or only store λ_diag_indices (sparse) and infer J entries as the complement.
    # Indices into A.nzval for the J block entries (same order as J.Jv.nzval)
    j_nzval_indices::Vector{Int}
    # Indices into A.nzval for the √λ diagonal entries (length n)
    λ_diag_indices::Vector{Int}
    # PERF: have not investigated other factorization methods. SPQR does not allow 
    #       for in-place updates or re-use of symbolic factorization, but with non
    #       square matrices--J^T J form is more unstable--we don't have a lot of options.
    # Cached QR factorization
    F::SparseArrays.SPQR.QRSparse{Float64, Int64}
end

"""Build the augmented matrix `[J; I]` once, recording which `A.nzval` entries
correspond to J values vs the λ diagonal."""
function LMWorkspace(Jv::SparseMatrixCSC{Float64, J_INDEX_TYPE})
    m, n = size(Jv)

    # Convert J to Int64 indices for SPQR compatibility, then vcat with identity.
    Jv64 = SparseMatrixCSC{Float64, Int64}(
        Jv.m, Jv.n,
        Vector{Int64}(Jv.colptr),
        Vector{Int64}(Jv.rowval),
        copy(Jv.nzval),
    )
    Iλ = sparse(Int64.(1:n), Int64.(1:n), ones(n), n, n)
    A = vcat(Jv64, Iλ)

    # Identify which A.nzval entries come from J vs the diagonal.
    j_nzval_indices = Vector{Int}(undef, length(Jv.nzval))
    λ_diag_indices = Vector{Int}(undef, n)

    j_idx = 0
    for col in 1:n
        for a_idx in A.colptr[col]:(A.colptr[col + 1] - 1)
            row = A.rowval[a_idx]
            if row <= m
                j_idx += 1
                j_nzval_indices[j_idx] = a_idx
            elseif row == m + col
                λ_diag_indices[col] = a_idx
            end
        end
    end
    @assert j_idx == length(Jv.nzval) "Expected $(length(Jv.nzval)) J entries, found $j_idx"

    F = LinearAlgebra.qr(A)

    return LMWorkspace(A, j_nzval_indices, λ_diag_indices, F)
end

"""Copy current Jacobian values into the augmented matrix."""
function copy_jacobian!(ws::LMWorkspace, Jv::SparseMatrixCSC{Float64, J_INDEX_TYPE})
    nzv = Jv.nzval
    for (i, a_idx) in enumerate(ws.j_nzval_indices)
        ws.A.nzval[a_idx] = nzv[i]
    end
    return
end

"""Update the √λ diagonal and re-factorize."""
function update_lambda!(ws::LMWorkspace, λ::Float64)
    sqrtλ = sqrt(λ)
    for a_idx in ws.λ_diag_indices
        ws.A.nzval[a_idx] = sqrtλ
    end
    ws.F = LinearAlgebra.qr(ws.A)
    return
end

"""Driver for the LevenbergMarquardtACPowerFlow method: sets up the data
structures (e.g. residual), runs the power flow method via calling `_run_power_flow_method`
on them, then handles post-processing (e.g. loss factors)."""
function _newton_power_flow(
    pf::ACPolarPowerFlow{LevenbergMarquardtACPowerFlow},
    data::ACPowerFlowData,
    time_step::Int64;
    tol::Float64 = DEFAULT_NR_TOL,
    maxIterations::Int = DEFAULT_NR_MAX_ITER,
    validate_voltage_magnitudes::Bool = DEFAULT_VALIDATE_VOLTAGES,
    vm_validation_range::MinMax = DEFAULT_VALIDATION_RANGE,
    λ_0::Float64 = DEFAULT_λ_0,
    x0::Union{Vector{Float64}, Nothing} = nothing,
    _ignored...,
)
    init_kwargs = if isnothing(x0)
        (; validate_voltage_magnitudes, vm_validation_range)
    else
        (; validate_voltage_magnitudes, vm_validation_range, x0)
    end
    residual, J, x0 = initialize_power_flow_variables(
        pf, data, time_step; init_kwargs...)
    converged = norm(residual.Rv, Inf) < tol
    i = 0
    if !converged
        ws = LMWorkspace(J.Jv)
        converged, i = _run_power_flow_method(
            time_step,
            x0,
            residual,
            J,
            ws;
            tol, maxIterations, λ_0,
        )
    end
    return _finalize_power_flow(
        converged, i, "LevenbergMarquardtACPowerFlow", residual, data, J.Jv, time_step)
end

function _run_power_flow_method(
    time_step::Int,
    x::Vector{Float64},
    residual::ACPowerFlowResidual,
    J::ACPowerFlowJacobian,
    ws::LMWorkspace;
    maxIterations::Int = DEFAULT_NR_MAX_ITER,
    tol::Float64 = DEFAULT_NR_TOL,
    λ_0::Float64 = DEFAULT_λ_0,
    _ignored...,
)
    μ::Float64 = λ_0
    λ::Float64 = 0.0
    i, converged = 0, false
    residual(x, time_step)
    resSize = dot(residual.Rv, residual.Rv)
    linf = norm(residual.Rv, Inf)
    @debug "initially: sum of squares $(siground(resSize)), L ∞ norm $(siground(linf)), λ = $λ"
    while i < maxIterations && !converged && isfinite(λ)
        λ, μ = update_damping_factor!(x, residual, J, μ, time_step, ws)
        converged = isfinite(λ) && norm(residual.Rv, Inf) < tol
        i += 1
    end
    if !isfinite(λ)
        @error "λ is not finite ($(λ))"
    elseif i == maxIterations
        @error "The LevenbergMarquardtACPowerFlow solver didn't coverge in $maxIterations iterations."
    end

    return converged, i
end

# LM implementation based on standard Levenberg-Marquardt method.
# See Nocedal & Wright (2006), sections 10.3 and 11.2.

"""Compute one LM trial step. Assumes `residual` and `J` are already evaluated
at `x` by the caller. Returns the gain ratio ρ."""
function compute_error(
    x::Vector{Float64},
    residual::ACPowerFlowResidual,
    J::ACPowerFlowJacobian,
    λ::Float64,
    time_step::Int,
    residualSize::Float64,
    ws::LMWorkspace,
)
    # Update augmented matrix with current J values and λ, then factorize once.
    copy_jacobian!(ws, J.Jv)
    update_lambda!(ws, λ)

    n = size(J.Jv, 2)
    b_x = vcat(-residual.Rv, zeros(n))
    Δx = ws.F \ b_x

    temp_x = residual.Rv .+ J.Jv * Δx

    x_trial = x .+ Δx
    residual(x_trial, time_step) # M(x_c + Δx)
    newResidualSize = dot(residual.Rv, residual.Rv)

    predicted_reduction = residualSize - dot(temp_x, temp_x)
    actual_reduction = residualSize - newResidualSize

    # Guard against zero/negative predicted reduction.
    if predicted_reduction <= 0.0 || !isfinite(predicted_reduction)
        residual(x, time_step)
        return 0.0
    end

    ρ = actual_reduction / predicted_reduction

    if ρ > 1e-4
        x .+= Δx
    else
        # Bad step: restore data state to match x (not x_trial).
        residual(x, time_step)
    end

    return ρ
end

function update_damping_factor!(
    x::Vector{Float64},
    residual::ACPowerFlowResidual,
    J::ACPowerFlowJacobian,
    μ::Float64,
    time_step::Int,
    ws::LMWorkspace,
)
    residual(x, time_step)
    residualSize = dot(residual.Rv, residual.Rv)
    J(time_step)

    λ = μ * sqrt(residualSize)
    ρ = compute_error(x, residual, J, λ, time_step, residualSize, ws)
    coef = 4.0
    if ρ > 0.75
        μ = max(μ / coef, 1e-8)
    elseif ρ >= 0.25
        # intentional no-op
    else
        μ *= coef
    end

    return (λ, μ)
end
