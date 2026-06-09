@testset "RH method: hessian" begin
    time_step = 1
    sys = PSB.build_system(PSB.PSITestSystems, "c_sys14")
    pf = ACPowerFlow{NewtonRaphsonACPowerFlow}()
    data = PowerFlowData(pf, sys)

    hess = PF.HomotopyHessian(data, time_step)
    t_k = 1.0

    residual = PF.ACPowerFlowResidual(data, time_step)
    J = PF.ACPowerFlowJacobian(residual, time_step)

    # when t_k is 1, homotopy hessian H(x) is Jacobian matrix of G(x) := J(x)^T*F(x)
    # check that as Δx -> 0, [G(x) - G(x+Δx)] - H(x)*Δx -> 0 at O(norm(Δx)^2)
    x0 = PF.calculate_x0(data, time_step)
    n = size(x0, 1)
    u = rand(Float64, n) .- 0.5
    u /= LinearAlgebra.norm(u)
    hess(x0, t_k, time_step)
    errors = []
    Δx_mags = collect(10.0^k for k in -3:-1:-6)
    for Δx_mag in Δx_mags
        x1 = deepcopy(x0)
        x1 .+= Δx_mag * u
        inputValues = [x0, x1]
        outputValues = Vector{Vector{Float64}}()
        for inputVal in inputValues
            residual(inputVal, time_step)
            J(time_step)
            push!(outputValues, J.Jv' * residual.Rv)
        end
        ΔFtJ = outputValues[2] - outputValues[1]
        push!(errors, norm(ΔFtJ - hess.Hv * (x1 - x0)) / Δx_mag)
    end
    # if correct, errors should be going to 0, linearly in Δx_mag;
    # if incorrect, then errors should be comparable, O(1)
    ratios = [err / Δx_mag for (err, Δx_mag) in zip(errors, Δx_mags)]
    @test all(isapprox(r, ratios[1]; rtol = 0.2) for r in ratios)
end

"""Wraps `(pfResidual, J)` so that `verify_jacobian_asymptotic` can be
used to check the homotopy Hessian against the gradient `g(x) := J(x)^T
F(x)`. At `t_k = 1`, the homotopy Hessian equals `∇g(x)`, so the same
asymptotic-FD machinery used for Jacobians applies — and we get
column-by-column diagnostics for free."""
mutable struct _GradAsResidual
    pfResidual::PF.ACPowerFlowResidual
    J::PF.ACPowerFlowJacobian
    Rv::Vector{Float64}
end
function (gr::_GradAsResidual)(x::Vector{Float64}, time_step::Int)
    gr.pfResidual(x, time_step)
    gr.J(time_step)
    gr.Rv .= gr.J.Jv' * gr.pfResidual.Rv
    return
end

@testset "RH method: hessian on simple LCC system (asymptotic check)" begin
    time_step = 1
    sys, _ = simple_lcc_system()
    # The NR-converged state of simple_lcc_system has α_r = α_i = 0 (on
    # the min limit, killing sin α factors), sin ϕ_i ≈ 1e-16 (inverter on
    # clamp, killing _d2Q_lcc), and ‖F‖ ≈ 1e-14 (so ∑ F_k ∇²F_k ≈ 0
    # regardless of LCC second-derivative correctness). Testing Hv there
    # would only exercise J^T J. Instead, perturb the base state into a
    # genuine interior point where F_k is O(1), α_s is well away from the
    # min limit, and sin ϕ_s is bounded away from 0.
    pf = ACPowerFlow{NewtonRaphsonACPowerFlow}()
    data = PowerFlowData(pf, sys)
    solve_power_flow!(data; pf = pf)

    residual = PF.ACPowerFlowResidual(data, time_step)
    J = PF.ACPowerFlowJacobian(residual, time_step)

    # Perturb every coordinate well off the NR-converged state so every
    # residual entry is O(1) — the four LCC ∇²F blocks are each weighted
    # by individual F_k values, and we want all of them nonzero to ensure
    # we exercise the full LCC second-derivative addition. The α offsets
    # are deterministic (0.6 rad off the min limit, well inside the
    # interior); the other coordinates use a fixed-seed random perturb
    # to avoid hand-tuning offsets that accidentally cancel inside F_t_r
    # or F_t_i.
    x0 = copy(PF.calculate_x0(data, time_step))
    Random.seed!(2)
    x0[1:8] .+= 0.3 .* (rand(8) .- 0.5)
    x0[end - 1] += 0.6   # α_r
    x0[end] += 0.6       # α_i

    residual(x0, time_step)
    @test minimum(abs, residual.Rv) > 0.05
    @test sin(data.lcc.rectifier.phi[1, time_step]) > 0.1
    @test sin(data.lcc.inverter.phi[1, time_step]) > 0.1

    hess = PF.HomotopyHessian(data, time_step)
    hess(x0, 1.0, time_step)   # populates hess.Hv

    grad_residual = _GradAsResidual(residual, J, similar(x0))
    verify_jacobian_asymptotic(
        grad_residual,
        Matrix(hess.Hv),   # dense for arbitrary J·e_j slicing
        x0,
        time_step;
        label = "homotopy Hessian on simple_lcc_system (interior pt)",
    )
end

@testset "RH method: sparse structure" begin
    # hessian's sparse structure shouldn't change.
    time_step = 1
    sys = PSB.build_system(PSB.PSITestSystems, "c_sys14")
    pf = ACPowerFlow{NewtonRaphsonACPowerFlow}()
    data = PowerFlowData(pf, sys)
    hess = PF.HomotopyHessian(data, time_step)
    t_k = 0.0
    x0 = PF.homotopy_x0(data, time_step)
    hess(x0, t_k, time_step)

    rowval, colptr = copy(hess.Hv.rowval), copy(hess.Hv.colptr)

    t_k = 0.5
    hess(x0, t_k, time_step)
    @test hess.Hv.rowval == rowval && hess.Hv.colptr == colptr

    t_k = 1.0
    hess(x0, t_k, time_step)
    @test hess.Hv.rowval == rowval && hess.Hv.colptr == colptr
end

@testset "RH method: gradient" begin
    time_step = 1
    sys = PSB.build_system(PSB.PSITestSystems, "c_sys14")
    pf = ACPowerFlow{NewtonRaphsonACPowerFlow}()
    data = PowerFlowData(pf, sys)
    hess = PF.HomotopyHessian(data, time_step)
    t_k = 0.0
    x0 = PF.homotopy_x0(data, time_step)
    g0 = similar(x0)
    PF.gradient_value!(g0, hess, t_k, x0, time_step)
    for (ind, bt) in enumerate(PF.get_bus_type(data)[:, time_step])
        @test g0[2 * ind - 1] == (bt == PSY.ACBusTypes.PQ ? x0[2 * ind - 1] - 1 : 0.0)
        @test g0[2 * ind] == 0.0
    end

    t_k = 0.5
    g1 = similar(x0)
    PF.gradient_value!(g1, hess, t_k, x0, time_step)

    n = size(x0, 1)
    u = rand(Float64, n) .- 0.5
    u /= LinearAlgebra.norm(u)
    Δx_mags = collect(10.0^k for k in -3:-1:-6)
    errors = []
    for Δx_mag in Δx_mags
        x1 = deepcopy(x0)
        x1 .+= Δx_mag * u
        inputValues = [x0, x1]
        outputValues = Vector{Float64}()
        for inputVal in inputValues
            push!(outputValues, PF.F_value(hess, t_k, inputVal, time_step))
        end
        ΔF = outputValues[2] - outputValues[1]
        push!(errors, norm(ΔF - dot(g1, x1 - x0)) / Δx_mag)
    end
    ratios = [err / Δx_mag for (err, Δx_mag) in zip(errors, Δx_mags)]
    @test all(isapprox(r, ratios[1]; rtol = 0.2) for r in ratios)
end
