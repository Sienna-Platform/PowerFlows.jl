"""Pre-allocated workspace for the Levenberg-Marquardt solver.

Holds the augmented matrix `[J; √λ·D]` with a fixed sparsity pattern, a mapping
to update its entries in-place, and a cached QR factorization. `D` is the
Marquardt column scaling (identity when disabled)."""
mutable struct LMWorkspace
    A::SparseMatrixCSC{Float64, Int64}
    # Indices into A.nzval for the J block entries (same order as J.Jv.nzval)
    j_nzval_indices::Vector{Int}
    # Indices into A.nzval for the √λ diagonal entries (length n)
    λ_diag_indices::Vector{Int}
    # SPQR: enables a cached symbolic factorization with in-place numeric
    # updates on the augmented [J; √λ·I]; the J^TJ normal-equations form is
    # less stable for the rectangular system.
    # Cached QR factorization
    F::SparseArrays.SPQR.QRSparse{Float64, Int64}
    # Preallocated augmented RHS [-Rv; 0] (length m + n); bottom n stay zero.
    b::Vector{Float64}
    # Marquardt diagonal scaling (length n). All-ones ⇒ √λ·I.
    D::Vector{Float64}
    marquardt_scaling::Bool
    # Per-iteration scratch: predicted-residual buffer (length m) and trial
    # iterate (length n). Reused in-place each step.
    temp_x::Vector{Float64}
    x_trial::Vector{Float64}
end

"""Build the augmented matrix `[J; D]` once, recording which `A.nzval` entries
correspond to J values vs the damping diagonal."""
function LMWorkspace(
    Jv::SparseMatrixCSC{Float64, J_INDEX_TYPE};
    marquardt_scaling::Bool = false,
)
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
        for a_idx in SparseArrays.nzrange(A, col)
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

    b = zeros(m + n)
    F = LinearAlgebra.qr(A)
    D = marquardt_scaling ? zeros(n) : ones(n)

    ws = LMWorkspace(
        A, j_nzval_indices, λ_diag_indices, F, b, D, marquardt_scaling,
        zeros(m), zeros(n))
    if marquardt_scaling
        update_column_scale!(ws, Jv)
    end
    return ws
end

"""Copy current Jacobian values into the augmented matrix."""
function copy_jacobian!(ws::LMWorkspace, Jv::SparseMatrixCSC{Float64, J_INDEX_TYPE})
    nzv = Jv.nzval
    for (i, a_idx) in enumerate(ws.j_nzval_indices)
        ws.A.nzval[a_idx] = nzv[i]
    end
    return
end

"""Update `ws.D`, the per-column damping scale: each entry is the running
maximum (across iterations) of the corresponding Jacobian column's 2-norm. It
is used as the Levenberg-Marquardt diagonal damping `√λ·D` in
[`update_lambda!`](@ref). A column whose running max is still zero is floored
to `1.0`, keeping `D > 0` so the damped block stays nonsingular."""
function update_column_scale!(
    ws::LMWorkspace,
    Jv::SparseMatrixCSC{Float64, J_INDEX_TYPE},
)
    nzv = Jv.nzval
    @inbounds for col in 1:size(Jv, 2)
        s = 0.0
        for k in SparseArrays.nzrange(Jv, col)
            v = nzv[k]
            s += v * v
        end
        cnorm = sqrt(s)
        d = ws.D[col]
        d = ifelse(cnorm > d, cnorm, d)
        ws.D[col] = d == 0.0 ? 1.0 : d
    end
    return
end

"""Update the √λ·D damping diagonal and re-factorize."""
function update_lambda!(ws::LMWorkspace, λ::Float64)
    sqrtλ = sqrt(λ)
    @inbounds for col in eachindex(ws.λ_diag_indices)
        ws.A.nzval[ws.λ_diag_indices[col]] = sqrtλ * ws.D[col]
    end
    ws.F = LinearAlgebra.qr(ws.A)
    return
end

"""Marquardt column scaling default per formulation: the rectangular CI state
columns `(e, f, Q, P_gen)` differ in natural scale, so identity damping is
ill-conditioned there — default it on. The polar state is well-scaled; keep it
off so the polar solver is bit-identical to before."""
_default_marquardt_scaling(::AbstractACPowerFlow) = false
_default_marquardt_scaling(::ACRectangularPowerFlow) = true

"""Driver for the LevenbergMarquardtACPowerFlow method: sets up the data
structures (e.g. residual), runs the power flow method via calling `_run_power_flow_method`
on them, then handles post-processing (e.g. loss factors)."""
function _newton_power_flow(
    pf::AbstractACPowerFlow{LevenbergMarquardtACPowerFlow},
    data::ACPowerFlowData,
    time_step::Int64;
    tol::Float64 = DEFAULT_NR_TOL,
    maxIterations::Int = DEFAULT_NR_MAX_ITER,
    validate_voltage_magnitudes::Bool = DEFAULT_VALIDATE_VOLTAGES,
    vm_validation_range::MinMax = DEFAULT_VALIDATION_RANGE,
    λ_0::Float64 = DEFAULT_λ_0,
    marquardt_scaling::Union{Bool, Nothing} = nothing,
    x0::Union{Vector{Float64}, Nothing} = nothing,
    stop_at_fold::Bool = false,
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
        use_scaling = something(marquardt_scaling, _default_marquardt_scaling(pf))
        ws = LMWorkspace(J.Jv; marquardt_scaling = use_scaling)
        converged, i = _run_power_flow_method(
            time_step,
            x0,
            residual,
            J,
            ws;
            tol, maxIterations, λ_0, stop_at_fold,
        )
    end
    # x0 was mutated in place to the converged state by _run_power_flow_method
    # (or is the already-converged initial state if the loop was skipped).
    _finalize_formulation!(pf, data, x0, residual, time_step)
    return _finalize_power_flow(
        converged, i, "LevenbergMarquardtACPowerFlow", residual, data, J.Jv, time_step)
end

function _run_power_flow_method(
    time_step::Int,
    x::Vector{Float64},
    residual::Union{ACPowerFlowResidual, ACRectangularCIResidual,
        ACMixedCPBResidual},
    J::Union{ACPowerFlowJacobian, ACRectangularCIJacobian, ACMixedCPBJacobian},
    ws::LMWorkspace;
    maxIterations::Int = DEFAULT_NR_MAX_ITER,
    tol::Float64 = DEFAULT_NR_TOL,
    λ_0::Float64 = DEFAULT_λ_0,
    stop_at_fold::Bool = false,
    _ignored...,
)
    μ::Float64 = λ_0
    λ::Float64 = 0.0
    i, converged = 0, false
    residual(x, time_step)
    resSize = dot(residual.Rv, residual.Rv)
    linf = norm(residual.Rv, Inf)
    @debug "initially: sum of squares $(siground(resSize)), L ∞ norm $(siground(linf)), λ = $λ"
    monitor, diag_state = setup_solver_diagnostics(J, stop_at_fold)
    # LM factorizes the augmented [J; √λ·D], not J, so the diagnostic keeps its own
    # KLU factor of J (symbolic once here, refreshed each iteration by the hook).
    diag_cache =
        isnothing(diag_state) ? nothing :
        make_linear_solver_cache(PNM.KLUSolver(), J.Jv)
    isnothing(diag_state) || symbolic_factor!(diag_cache, J.Jv)
    # J is fresh from initialize_power_flow_variables, so the first iteration
    # must not re-fill it.
    step_accepted = false
    while i < maxIterations && !converged && isfinite(λ) && μ < DEFAULT_μ_MAX
        λ, μ, step_accepted =
            update_damping_factor!(x, residual, J, μ, time_step, ws, step_accepted)
        if !isnothing(diag_state)
            # One-iterate lag: update_damping_factor! evaluated J at the pre-step
            # iterate but residual.Rv is already post-step, so κ̂/λ_min describe the
            # linearization J while the reported ‖F‖∞ is after the step. Not realigned.
            run_solver_diagnostics!(
                diag_state, "LM iter $i", residual, J, time_step,
                diag_cache, monitor, stop_at_fold) &&
                return false, i
        end
        converged = isfinite(λ) && norm(residual.Rv, Inf) < tol
        i += 1
    end
    if !converged
        if !isfinite(λ)
            @error "λ is not finite ($(λ))"
        elseif μ >= DEFAULT_μ_MAX
            @error "The LevenbergMarquardtACPowerFlow damping factor μ hit the cap (DEFAULT_μ_MAX=$(DEFAULT_μ_MAX)) after $i iterations; aborting (likely divergence)."
        elseif i == maxIterations
            @error "The LevenbergMarquardtACPowerFlow solver didn't coverge in $maxIterations iterations."
        end
    end

    return converged, i
end

# LM implementation based on standard Levenberg-Marquardt method.
# See Nocedal & Wright (2006), sections 10.3 and 11.2.

"""Compute one LM trial step. Assumes `residual` and `J` are already evaluated
at `x` by the caller. Returns `(ρ, accepted)`: `accepted` is true iff the step
was taken (`x` mutated to `x + Δx`)."""
function compute_error(
    x::Vector{Float64},
    residual::Union{ACPowerFlowResidual, ACRectangularCIResidual,
        ACMixedCPBResidual},
    J::Union{ACPowerFlowJacobian, ACRectangularCIJacobian, ACMixedCPBJacobian},
    λ::Float64,
    time_step::Int,
    residualSize::Float64,
    ws::LMWorkspace,
)
    copy_jacobian!(ws, J.Jv)
    ws.marquardt_scaling && update_column_scale!(ws, J.Jv)
    update_lambda!(ws, λ)

    m = length(residual.Rv)
    @assert m == length(ws.b) - size(J.Jv, 2) "residual/J size mismatch vs preallocated LM buffer (m=$m, buf=$(length(ws.b)), n=$(size(J.Jv, 2)))"
    @views ws.b[1:m] .= .-residual.Rv   # bottom n entries stay zero from construction
    # SPQR's QRSparse exposes no in-place ldiv!, so the solve allocates Δx.
    Δx = ws.F \ ws.b

    # temp_x = residual.Rv + J.Jv * Δx. mul! into the buffer, then add Rv; this
    # must not alias residual.Rv because residual() may overwrite Rv below.
    LinearAlgebra.mul!(ws.temp_x, J.Jv, Δx)
    ws.temp_x .+= residual.Rv

    ws.x_trial .= x .+ Δx
    residual(ws.x_trial, time_step) # M(x_c + Δx)
    newResidualSize = dot(residual.Rv, residual.Rv)

    predicted_reduction = residualSize - dot(ws.temp_x, ws.temp_x)
    actual_reduction = residualSize - newResidualSize

    # Guard against zero/negative predicted reduction.
    if predicted_reduction <= 0.0 || !isfinite(predicted_reduction)
        residual(x, time_step)
        return (0.0, false)
    end

    ρ = actual_reduction / predicted_reduction

    if ρ > 1e-4
        x .+= Δx
        return (ρ, true)
    else
        # Bad step: restore data state to match x (not x_trial).
        residual(x, time_step)
        return (ρ, false)
    end
end

function update_damping_factor!(
    x::Vector{Float64},
    residual::Union{ACPowerFlowResidual, ACRectangularCIResidual,
        ACMixedCPBResidual},
    J::Union{ACPowerFlowJacobian, ACRectangularCIJacobian, ACMixedCPBJacobian},
    μ::Float64,
    time_step::Int,
    ws::LMWorkspace,
    previous_step_accepted::Bool,
)
    # residual.Rv is already current at x: every exit of compute_error (and the
    # pre-loop init) leaves it evaluated at the held x.
    residualSize = dot(residual.Rv, residual.Rv)
    # J is current unless the previous step moved x; refresh only then.
    previous_step_accepted && J(time_step)

    λ = μ * sqrt(residualSize)
    ρ, accepted = compute_error(x, residual, J, λ, time_step, residualSize, ws)
    coef = 4.0
    if ρ > 0.75
        μ = max(μ / coef, 1e-8)
    elseif ρ >= 0.25
        # intentional no-op
    else
        μ = min(μ * coef, DEFAULT_μ_MAX)
    end

    return (λ, μ, accepted)
end
