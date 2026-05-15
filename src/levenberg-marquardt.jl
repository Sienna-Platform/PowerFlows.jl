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
    maxTestλs::Int = DEFAULT_MAX_TEST_λs,
    _ignored...,
)
    residual, J, x0 = initialize_power_flow_variables(
        pf, data, time_step;
        validate_voltage_magnitudes, vm_validation_range)
    converged = norm(residual.Rv, Inf) < tol
    i = 0
    if !converged
        converged, i = _run_power_flow_method(
            time_step,
            x0,
            residual,
            J;
            tol, maxIterations, λ_0, maxTestλs,
        )
    end
    return _finalize_power_flow(
        converged, i, "LevenbergMarquardtACPowerFlow", residual, data, J.Jv, time_step)
end

function _run_power_flow_method(
    time_step::Int,
    x::Vector{Float64},
    residual::ACPowerFlowResidual,
    J::ACPowerFlowJacobian;
    maxIterations::Int = DEFAULT_NR_MAX_ITER,
    tol::Float64 = DEFAULT_NR_TOL,
    λ_0::Float64 = DEFAULT_λ_0,
    maxTestλs::Int = DEFAULT_MAX_TEST_λs,
    _ignored...,
)
    λ::Float64 = λ_0
    i, converged = 0, false
    residual(x, time_step)
    resSize = dot(residual.Rv, residual.Rv)
    linf = norm(residual.Rv, Inf)
    @debug "initially: sum of squares $(siground(resSize)), L ∞ norm $(siground(linf)), λ = $λ"
    while i < maxIterations && !converged && !isnan(λ)
        λ = update_damping_factor!(x, residual, J, time_step, maxTestλs)
        converged = !isnan(λ) && norm(residual.Rv, Inf) < tol
        i += 1
    end
    if isnan(λ)
        @error "λ is NaN"
    elseif i == maxIterations
        @error "The LevenbergMarquardtACPowerFlow solver didn't coverge in $maxIterations iterations."
    end

    return converged, i
end

# LM implementation and parameters values largely based upon this paper:
# # https://www.sciencedirect.com/science/article/pii/S0142061518336342
function compute_error(
    x::Vector{Float64},
    residual::ACPowerFlowResidual,
    J::ACPowerFlowJacobian,
    λ::Float64,
    time_step::Int,
    residualSize::Float64,
)
    residual(x, time_step) # M(x_c)
    J(time_step)

    n = size(J.Jv, 2)
    Iλ = sparse(1:n, 1:n, sqrt(λ) .* ones(n), n, n) # less error-prone compared to A = vcat(J.Jv, sqrt(λ) * sparse(LinearAlgebra.I, size(J.Jv)))
    A = [J.Jv; Iλ]

    b_x = vcat(-residual.Rv, zeros(size(J.Jv, 2)))
    Δx = A \ b_x

    temp_x = residual.Rv .+ J.Jv * Δx

    x_trial = x .+ Δx
    residual(x_trial, time_step) # M(y_c)

    b_y = vcat(-residual.Rv, zeros(size(J.Jv, 2)))
    Δy = A \ b_y
    temp_y = residual.Rv .+ J.Jv * Δy

    newResidualSize_y = dot(residual.Rv, residual.Rv)
    residual(x_trial .+ Δy, time_step) # M(x_c+Δx+Δy)
    newResidualSize = dot(residual.Rv, residual.Rv)

    b_z = vcat(-residual.Rv, zeros(size(J.Jv, 2)))
    Δz = A \ b_z
    temp_z = residual.Rv .+ J.Jv * Δz
    newResidualSize_z = dot(residual.Rv, residual.Rv)

    residual(x_trial .+ Δy .+ Δz, time_step) # M(x_c+Δx+Δy+Δz)
    newResidualSize = dot(residual.Rv, residual.Rv)

    predicted_reduction = (
        residualSize - dot(temp_x, temp_x) + newResidualSize_y - dot(temp_y, temp_y) +
        newResidualSize_z - dot(temp_z, temp_z)
    )
    actual_reduction = (residualSize - newResidualSize)
    ρ = actual_reduction / predicted_reduction

    if ρ > 1e-4
        x .+= (Δx .+ Δy .+ Δz)
    end

    return ρ
end

function update_damping_factor!(
    x::Vector{Float64},
    residual::ACPowerFlowResidual,
    J::ACPowerFlowJacobian,
    time_step::Int,
    maxTestλs::Int,
)
    residual(x, time_step)
    residualSize = dot(residual.Rv, residual.Rv)
    J(time_step)

    # Now update \lambda
    test_lambda::Int = 1
    # https://www.sciencedirect.com/science/article/pii/S0142061518336342
    λ = DEFAULT_λ_0 * sqrt(residualSize)
    while test_lambda < maxTestλs
        ρ = compute_error(x, residual, J, λ, time_step, residualSize)
        if ρ > 0.75 # good step
            factor = 4.0
            λ_temp = λ / factor
            λ = max(λ_temp, 1.0e-8)
            break
        elseif ρ >= 0.25 # okay step
            break
        else # bad step
            factor = 4.0
            λ *= factor
        end

        test_lambda += 1
    end
    if test_lambda == maxTestλs
        @error "Unable to improve: gave up after increasing damping factor $maxTestλs times."
        λ = NaN
    end

    return λ
end
