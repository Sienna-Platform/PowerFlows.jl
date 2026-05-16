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
