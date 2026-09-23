"""Pre-allocated workspace for the Levenberg-Marquardt solver.

Solves each trial step from the normal equations `N = JᵀJ + λ·D²`. `N`'s
sparsity pattern is fixed for the life of a solve, so its CHOLMOD symbolic
factorization is computed once and reused across every λ update and
iteration; only the numeric factorization reruns. `D` is the Marquardt column
scaling (identity when disabled), floored to keep every entry positive, which
makes `N` positive definite whenever `λ > 0`. `_lm_qr_fallback` solves the
augmented system `[J; √λ·D]` by QR instead, for the rare case `N` is not
positive definite."""
mutable struct LMWorkspace
    N::SparseMatrixCSC{Float64, J_INDEX_TYPE}    # JᵀJ + λ·D², fixed pattern
    jtj_p1::Vector{Int}
    jtj_p2::Vector{Int}
    jtj_offsets::Vector{Int}    # see _build_jtj_nz_cache
    diag_nz::Vector{Int}        # N.nzval index of each diagonal entry i
    mat::FixedStructureCHOLMOD{Float64, J_INDEX_TYPE}
    F::SparseArrays.CHOLMOD.Factor{Float64, J_INDEX_TYPE}
    rhs::Vector{Float64}        # -Jᵀ·Rv, length n
    # Marquardt diagonal scaling (length n). All-ones ⇒ λ·I.
    D::Vector{Float64}
    marquardt_scaling::Bool
    # Per-iteration scratch: temp_x = Rv + J·Δx (m); x_trial = x + Δx (n).
    temp_x::Vector{Float64}
    x_trial::Vector{Float64}
end

"""Build the fixed-pattern normal-equations matrix `N = JᵀJ` (values zeroed)
once, its CHOLMOD symbolic factorization, and the JᵀJ row-pair refill cache."""
function LMWorkspace(
    Jv::SparseMatrixCSC{Float64, J_INDEX_TYPE};
    marquardt_scaling::Bool = false,
)
    m, n = size(Jv)

    # Force the maximal J'*J pattern the same way HomotopyHessian does, then
    # restore Jv's real values (see homotopy_hessian.jl's HomotopyHessian ctor).
    original_nzval = copy(Jv.nzval)
    fill!(Jv.nzval, 1.0)
    N = Jv' * Jv
    SparseArrays.nonzeros(N) .= 0.0
    copyto!(Jv.nzval, original_nzval)

    jtj_p1, jtj_p2, jtj_offsets = _build_jtj_nz_cache(Jv, N)
    diag_nz = [_nz_index(N, i, i) for i in 1:n]

    mat = FixedStructureCHOLMOD(N)
    F = symbolic_factor(mat)
    D = marquardt_scaling ? zeros(n) : ones(n)

    ws = LMWorkspace(
        N, jtj_p1, jtj_p2, jtj_offsets, diag_nz, mat, F,
        Vector{Float64}(undef, n), D, marquardt_scaling,
        Vector{Float64}(undef, m), Vector{Float64}(undef, n))
    if marquardt_scaling
        update_column_scale!(ws, Jv)
    end
    return ws
end

"""Update `ws.D`, the per-column damping scale: each entry is the running
maximum (across iterations) of the corresponding Jacobian column's 2-norm. It
is used as the Levenberg-Marquardt diagonal damping `λ·D²` in
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

"""Refresh `ws.N = JᵀJ + λ·D²` in place from the current `Jv` (via the cached
row-pair map) and run a numeric CHOLMOD factorization reusing the symbolic
factorization computed once in the `LMWorkspace` constructor."""
function update_lambda!(
    ws::LMWorkspace,
    Jv::SparseMatrixCSC{Float64, J_INDEX_TYPE},
    λ::Float64,
)
    Nnz = SparseArrays.nonzeros(ws.N)
    fill!(Nnz, 0.0)
    _refresh_JtJ!(ws.N, Jv, ws.jtj_p1, ws.jtj_p2, ws.jtj_offsets)
    @inbounds for i in eachindex(ws.diag_nz)
        Nnz[ws.diag_nz[i]] += λ * ws.D[i]^2
    end
    set_values!(ws.mat, Nnz)
    numeric_factor!(ws.F, ws.mat)
    return
end

"""Solve one LM trial step `(JᵀJ + λ·D²)Δx = -Jᵀ·Rv`. Falls back to a fresh
sparse QR of the augmented system `[J; √λ·D]` (uncached; not meant to be hot)
if the normal-equations factorization is not positive definite."""
function _lm_solve_step!(
    ws::LMWorkspace,
    Jv::SparseMatrixCSC{Float64, J_INDEX_TYPE},
    Rv::Vector{Float64},
    λ::Float64,
)
    ok = try
        update_lambda!(ws, Jv, λ)
        LinearAlgebra.issuccess(ws.F)
    catch e
        e isa SparseArrays.CHOLMOD.CHOLMODException ||
            e isa SparseArrays.CHOLMOD.PosDefException || rethrow(e)
        false
    end
    if ok
        LinearAlgebra.mul!(ws.rhs, Jv', Rv)
        ws.rhs .*= -1
        return ws.F \ ws.rhs
    end
    @warn "LM normal-equations factorization was not positive definite; falling \
        back to a sparse QR solve of the augmented system for this step." maxlog = 5
    return _lm_qr_fallback(Jv, Rv, ws.D, λ)
end

function _lm_qr_fallback(
    Jv::SparseMatrixCSC{Float64, J_INDEX_TYPE},
    Rv::Vector{Float64},
    D::Vector{Float64},
    λ::Float64,
)
    m, n = size(Jv)
    Jv64 = SparseMatrixCSC{Float64, Int64}(
        Jv.m, Jv.n, Vector{Int64}(Jv.colptr), Vector{Int64}(Jv.rowval), copy(Jv.nzval))
    Iλ = sparse(Int64.(1:n), Int64.(1:n), sqrt(λ) .* D, n, n)
    A = vcat(Jv64, Iλ)
    b = zeros(m + n)
    b[1:m] .= .-Rv
    return LinearAlgebra.qr(A) \ b
end

"""Marquardt column scaling default per formulation, dispatched on the formulation TYPE (so
it can be resolved at evaluation-model construction time, before an instance exists — see the
`marquardt_scaling` keyword on [`ACPolarPowerFlow`](@ref)/[`ACRectangularPowerFlow`](@ref)/
[`ACMixedPowerFlow`](@ref)). The rectangular CI state columns `(e, f, Q, P_gen)` differ in
natural scale, so identity damping is ill-conditioned there — default it on. The polar and
mixed states are well-scaled; keep it off so those solvers are bit-identical to before."""
_default_marquardt_scaling(::Type{<:AbstractACPowerFlow}) = false
_default_marquardt_scaling(::Type{<:ACRectangularPowerFlow}) = true

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
        use_scaling = something(marquardt_scaling, _default_marquardt_scaling(typeof(pf)))
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
    # LM factorizes JᵀJ + λ·D² (or, on the rare QR fallback, the augmented
    # [J; √λ·D]), not J itself, so the diagnostic keeps its own KLU factor of J
    # (symbolic once here, refreshed each iteration by the hook).
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
    ws.marquardt_scaling && update_column_scale!(ws, J.Jv)
    Δx = _lm_solve_step!(ws, J.Jv, residual.Rv, λ)

    # temp_x = Rv + J·Δx
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
