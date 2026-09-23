"""Throw if any island carries more than one swing (REF) bus. `HomotopyHessian` has no
independent-per-swing slack handling, so its curvature silently disagrees with the residual's
multi-swing self-balancing rows; fail loudly rather than return a mis-specified solution."""
function _reject_multi_swing_islands(data::ACPowerFlowData, time_step::Int64)
    bus_type = view(data.bus_type, :, time_step)
    subnetworks =
        _find_subnetworks_for_reference_buses(data.power_network_matrix.data, bus_type)
    multi_swing = _multi_swing_ref_indices(data.bus_type, subnetworks, time_step)
    if !isempty(multi_swing)
        throw(
            ArgumentError(
                "RobustHomotopyPowerFlow does not support multiple swing (REF) buses in " *
                "one island ($(length(multi_swing)) found); use the polar " *
                "NewtonRaphsonACPowerFlow, TrustRegionACPowerFlow, or " *
                "FastDecoupledACPowerFlow solver for multi-swing systems.",
            ),
        )
    end
    return
end

function _newton_power_flow(pf::ACPolarPowerFlow{<:RobustHomotopyPowerFlow},
    data::ACPowerFlowData,
    time_step::Int64;
    Δt_k::Float64 = DEFAULT_Δt_k,
    _ignored...,
)
    _reject_multi_swing_islands(data, time_step)
    homHess = HomotopyHessian(data, time_step)
    x = homotopy_x0(data, time_step)
    t_k = 0.0

    # the sparse structure of the Hessian is different at t_k = 0.0 and t_k > 0.0
    # so we need to increase t_k once before we initialize the solver.
    t_k += Δt_k
    homHess(data, x, t_k, time_step)

    # options: KLUHessianSolver, CholeskyHessianSolver (fastest)
    hSolver = CholeskyHessianSolver(homHess.Hv)
    symbolic_factor!(hSolver, homHess.Hv)

    success = true
    total_iters = 0
    while true # go onto next t_k even if search doesn't terminate within max iterations.
        converged_t_k, iters =
            _second_order_newton(homHess, data, t_k, time_step, x, hSolver)
        total_iters += iters
        if t_k == 1.0
            success = converged_t_k
            break
        end
        t_k = min(t_k + Δt_k, 1.0)
    end
    r_L2 = norm(homHess.pfResidual.Rv, 2)
    r_Linf = norm(homHess.pfResidual.Rv, Inf)
    @info("Final residual size: $(r_L2) L2, $(r_Linf) L∞.")
    if !success
        @error(
            "The RobustHomotopyPowerFlow solver failed to converge after $total_iters iterations."
        )
    else
        @info("The RobustHomotopyPowerFlow solver converged after $total_iters iterations.")
        if get_calculate_loss_factors(data)
            _calculate_loss_factors(data, homHess.J.Jv, time_step)
        end
        if get_calculate_voltage_stability_factors(data)
            _calculate_voltage_stability_factors(data, homHess.J.Jv, time_step)
        end
    end
    return success
end

sig3(x::Float64) = round(x; sigdigits = 3)

function info_helper(homHess::HomotopyHessian, t_k::Float64, F_val::Float64, msg::String)
    r_val = norm(homHess.pfResidual.Rv, Inf)
    @info "t_k = $(sig3(t_k)): $msg, F_k $(sig3(F_val)), residual $(sig3(r_val))"
end

function _second_order_newton(homHess::HomotopyHessian,
    data::ACPowerFlowData,
    t_k::Float64,
    time_step::Int,
    x::Vector{Float64},
    hSolver::HessianSolver;
    maxIterations::Int = DEFAULT_NR_MAX_ITER,
    tol::Float64 = DEFAULT_NR_TOL,
)
    i, converged, stop = 0, false, false
    F_val = F_value(homHess, data, t_k, x, time_step)
    last_tk = t_k == 1.0
    δ = zeros(size(x, 1)) # PERF: allocating
    while i < maxIterations && !converged && !stop
        stop = _second_order_newton_step(
            homHess,
            data,
            t_k,
            time_step,
            x,
            hSolver,
            δ,
        )
        F_val = F_value(homHess, data, t_k, x, time_step)
        converged = (last_tk ? norm(homHess.pfResidual.Rv, Inf) : abs(F_val)) < tol
        i += 1
        if converged
            info_helper(homHess, t_k, F_val, "converged")
        elseif i == maxIterations && !stop
            info_helper(homHess, t_k, F_val, "max iterations")
        end
    end
    return converged, i
end

function _second_order_newton_step(homHess::HomotopyHessian,
    data::ACPowerFlowData,
    t_k::Float64,
    time_step::Int,
    x::Vector{Float64},
    hSolver::HessianSolver,
    δ::Vector{Float64},
)
    F_val = F_value(homHess, data, t_k, x, time_step)
    last_step = t_k == 1.0
    homHess(data, x, t_k, time_step)
    if !last_step && dot(homHess.grad, homHess.grad) < GRAD_ZERO &&
       LinearAlgebra.isposdef(homHess.Hv) # stop case 1: hit local minimum.
        info_helper(homHess, t_k, F_val, "local minimum")
        return true
    end
    modify_and_numeric_factor!(hSolver, homHess.Hv)
    δ .= homHess.grad
    solve!(hSolver, δ)
    δ .*= -1

    # Create objective function
    ϕ = α -> F_value(homHess, data, t_k, x + α * δ, time_step)

    # Perform line search
    φ_0 = F_val
    dφ_0 = dot(homHess.grad, δ)
    (α_star, F_val) = BackTracking()(ϕ, 1.0, φ_0, dφ_0)

    if !last_step && norm(δ * α_star) < INSUFFICIENT_CHANGE_IN_X
        # stop case 2: slow progress.
        info_helper(homHess, t_k, F_val, "slow progress")
        return true
    end
    x .+= δ * α_star
    return false
end
