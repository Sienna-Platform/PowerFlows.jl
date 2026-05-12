function _fd_jacobian(R::PF.ACRectangularCIResidual, x::Vector{Float64}, time_step::Int)
    ε = 1e-6
    n = length(R.Rv)
    Jfd = zeros(n, n)
    for k in 1:n
        xp = copy(x)
        xp[k] += ε
        R(xp, time_step)
        Fp = copy(R.Rv)
        xm = copy(x)
        xm[k] -= ε
        R(xm, time_step)
        Fm = copy(R.Rv)
        Jfd[:, k] = (Fp - Fm) / (2ε)
    end
    R(x, time_step)
    return Jfd
end

@testset "Rectangular CI Jacobian: FD parity" begin
    @testset "c_sys5 at polar-converged state" begin
        sys = PSB.build_system(PSB.PSITestSystems, "c_sys5")
        pf_polar = ACPowerFlow{NewtonRaphsonACPowerFlow}()
        PF.solve_and_store_power_flow!(pf_polar, sys)
        pf_rect = ACPowerFlow{RectangularCurrentInjectionACPowerFlow}()
        data = PF.PowerFlowData(pf_rect, sys)
        R = PF.ACRectangularCIResidual(data, 1)
        J = PF.ACRectangularCIJacobian(R, 1)
        x = Vector{Float64}(undef, length(R.Rv))
        PF.rect_initial_state!(x, data, R.bus_state_offset, R.bus_block_size, 1)
        R(x, 1)
        J(1)
        Jfd = _fd_jacobian(R, x, 1)
        J(1)
        @test maximum(abs.(Array(J.Jv) - Jfd)) < 1e-4
    end

    @testset "c_sys14 at polar-converged state" begin
        sys = PSB.build_system(PSB.PSITestSystems, "c_sys14"; add_forecasts = false)
        pf_polar = ACPowerFlow{NewtonRaphsonACPowerFlow}()
        PF.solve_and_store_power_flow!(pf_polar, sys)
        pf_rect = ACPowerFlow{RectangularCurrentInjectionACPowerFlow}()
        data = PF.PowerFlowData(pf_rect, sys)
        R = PF.ACRectangularCIResidual(data, 1)
        J = PF.ACRectangularCIJacobian(R, 1)
        x = Vector{Float64}(undef, length(R.Rv))
        PF.rect_initial_state!(x, data, R.bus_state_offset, R.bus_block_size, 1)
        R(x, 1)
        J(1)
        Jfd = _fd_jacobian(R, x, 1)
        J(1)
        @test maximum(abs.(Array(J.Jv) - Jfd)) < 1e-4
    end

    @testset "c_sys5 at perturbed (non-converged) state" begin
        sys = PSB.build_system(PSB.PSITestSystems, "c_sys5")
        pf_polar = ACPowerFlow{NewtonRaphsonACPowerFlow}()
        PF.solve_and_store_power_flow!(pf_polar, sys)
        pf_rect = ACPowerFlow{RectangularCurrentInjectionACPowerFlow}()
        data = PF.PowerFlowData(pf_rect, sys)
        R = PF.ACRectangularCIResidual(data, 1)
        J = PF.ACRectangularCIJacobian(R, 1)
        x = Vector{Float64}(undef, length(R.Rv))
        PF.rect_initial_state!(x, data, R.bus_state_offset, R.bus_block_size, 1)
        Random.seed!(42)
        x .+= 0.05 .* randn(length(x))
        R(x, 1)
        J(1)
        Jfd = _fd_jacobian(R, x, 1)
        J(1)
        @test maximum(abs.(Array(J.Jv) - Jfd)) < 1e-3
    end
end

@testset "Rectangular CI Jacobian: off-diagonal Y_bus blocks are constant" begin
    sys = PSB.build_system(PSB.PSITestSystems, "c_sys5")
    pf_polar = ACPowerFlow{NewtonRaphsonACPowerFlow}()
    PF.solve_and_store_power_flow!(pf_polar, sys)
    pf_rect = ACPowerFlow{RectangularCurrentInjectionACPowerFlow}()
    data = PF.PowerFlowData(pf_rect, sys)
    R = PF.ACRectangularCIResidual(data, 1)
    J = PF.ACRectangularCIJacobian(R, 1)
    x = Vector{Float64}(undef, length(R.Rv))
    PF.rect_initial_state!(x, data, R.bus_state_offset, R.bus_block_size, 1)
    R(x, 1)
    J(1)
    J_first = copy(J.Jv)

    Random.seed!(123)
    x .+= 0.01 .* randn(length(x))
    R(x, 1)
    J(1)
    J_second = copy(J.Jv)

    # For non-REF, non-PV-Q columns at off-diagonal block positions, Y_bus entries
    # should be identical across iterations.
    bus_types = data.bus_type[:, 1]
    for col in 1:length(bus_types), row in 1:length(bus_types)
        row == col && continue
        bus_types[col] == PSY.ACBusTypes.REF && continue
        row_off = Int(R.bus_state_offset[row])
        col_off = Int(R.bus_state_offset[col])
        for dr in 0:1, dc in 0:1
            @test J_first[row_off + dr, col_off + dc] ==
                  J_second[row_off + dr, col_off + dc]
        end
    end
end
