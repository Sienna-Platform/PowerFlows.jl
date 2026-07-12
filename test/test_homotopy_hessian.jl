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

@testset "RH method: hessian on inverter-setpoint LCC (asymptotic check)" begin
    # Same interior-point check as above, but with the P-setpoint metered at
    # the inverter (`transfer_setpoint < 0` ⇒ `setpoint_at_rectifier = false`).
    # Then the P-setpoint tail row F_t_r = -P_lcc_to - P_set carries inverter
    # curvature, so the homotopy Hessian's LCC contribution must attach F_t_r's
    # ∇²F to `d2P_i` (negated) instead of `d2P_r`. This regression would fail
    # (max error O(0.05)) if `_update_hessian_lcc_contributions!` did not branch
    # on `setpoint_at_rectifier`.
    time_step = 1
    sys, lcc = simple_lcc_system()
    set_transfer_setpoint!(lcc, -abs(get_transfer_setpoint(lcc)))
    pf = ACPowerFlow{NewtonRaphsonACPowerFlow}()
    data = PowerFlowData(pf, sys)
    solve_power_flow!(data; pf = pf)
    @test all(.!data.lcc.setpoint_at_rectifier)

    residual = PF.ACPowerFlowResidual(data, time_step)
    J = PF.ACPowerFlowJacobian(residual, time_step)

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
    hess(x0, 1.0, time_step)

    grad_residual = _GradAsResidual(residual, J, similar(x0))
    verify_jacobian_asymptotic(
        grad_residual,
        Matrix(hess.Hv),
        x0,
        time_step;
        label = "homotopy Hessian on inverter-setpoint LCC (interior pt)",
    )
end

@testset "RH method: hessian at intermediate t" begin
    # At 0 < t < 1 the assembled Hessian Hv must be the Jacobian (in x) of the
    # homotopy gradient G(x) = ∇ F_value = (1−t)·diag(PQ)·(x−1) + t·Jᵀ F. Verify
    # [G(x+Δx) − G(x)] − Hv·Δx → 0 at O(‖Δx‖²). This pins the sign/scaling of the
    # F-weighted second-derivative term together with the JᵀJ and (1−t) terms.
    time_step = 1
    sys = PSB.build_system(PSB.PSITestSystems, "c_sys14")
    pf = ACPowerFlow{NewtonRaphsonACPowerFlow}()
    data = PowerFlowData(pf, sys)
    hess = PF.HomotopyHessian(data, time_step)
    t_k = 0.5

    x0 = PF.calculate_x0(data, time_step)
    n = size(x0, 1)
    u = rand(Float64, n) .- 0.5
    u /= LinearAlgebra.norm(u)
    hess(x0, t_k, time_step)
    Hv = copy(hess.Hv)
    errors = Float64[]
    Δx_mags = collect(10.0^k for k in -3:-1:-6)
    for Δx_mag in Δx_mags
        x1 = x0 .+ Δx_mag .* u
        g0 = similar(x0)
        g1 = similar(x0)
        PF.gradient_value!(g0, hess, t_k, x0, time_step)
        PF.gradient_value!(g1, hess, t_k, x1, time_step)
        push!(errors, norm((g1 - g0) - Hv * (x1 - x0)) / Δx_mag)
    end
    ratios = [err / Δx_mag for (err, Δx_mag) in zip(errors, Δx_mags)]
    @test all(isapprox(r, ratios[1]; rtol = 0.2) for r in ratios)
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

@testset "homotopy_x0 rectifier start is interior" begin
    # Deep-commutation regime β_r ≥ V·t: force it directly on the data so the
    # test doesn't depend on how simple_lcc_system()'s defaults happen to land.
    time_step = 1
    sys, _ = simple_lcc_system()
    pf = ACPowerFlow{NewtonRaphsonACPowerFlow}()
    data = PowerFlowData(pf, sys)
    for i in 1:length(data.lcc.i_dc[:, time_step])
        # β_r = x_t·I_dc/√2; with V ≈ t ≈ 1 at flat start, this puts
        # β_r/(V·t) = 1.5, squarely inside the deep-commutation regime.
        data.lcc.i_dc[i, time_step] = 1.0
        data.lcc.rectifier.transformer_reactance[i] = 1.5 * sqrt(2)
    end
    x = PF.homotopy_x0(data, time_step)

    n_lcc = size(data.lcc.p_set, 1)
    num_buses = first(size(PF.get_bus_type(data)))
    for i in 1:n_lcc
        # Offset computation copied verbatim from homotopy_x0's own loop.
        offset_lcc = num_buses * 2 + (i - 1) * 4
        t_r = x[offset_lcc + 1]
        α_r = x[offset_lcc + 3]
        V_fb = PF._bus_V(data, first(data.lcc.bus_indices[i]), time_step)
        β_r =
            data.lcc.rectifier.transformer_reactance[i] * data.lcc.i_dc[i, time_step] /
            sqrt(2)
        u_r = cos(α_r) - β_r / (V_fb * t_r)
        @test -1.0 < u_r < 1.0
    end
end

@testset "homotopy_x0 rectifier start is non-negative under deep commutation" begin
    # β_r/(V·t) ≈ 3: the interior window for α_r is empty (max_α_r_interior < 0).
    # The fallback must clamp to a non-negative angle, never a negative one.
    time_step = 1
    sys, _ = simple_lcc_system()
    pf = ACPowerFlow{NewtonRaphsonACPowerFlow}()
    data = PowerFlowData(pf, sys)
    for i in 1:length(data.lcc.i_dc[:, time_step])
        data.lcc.i_dc[i, time_step] = 1.0
        data.lcc.rectifier.transformer_reactance[i] = 3.0 * sqrt(2)
    end
    x = PF.homotopy_x0(data, time_step)

    n_lcc = size(data.lcc.p_set, 1)
    num_buses = first(size(PF.get_bus_type(data)))
    for i in 1:n_lcc
        offset_lcc = num_buses * 2 + (i - 1) * 4
        @test x[offset_lcc + 3] >= 0.0
    end
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
