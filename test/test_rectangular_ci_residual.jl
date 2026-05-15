@testset "Rectangular CI Residual: parity with polar" begin
    @testset "c_sys5: residual zero at polar-converged state" begin
        sys = PSB.build_system(PSB.PSITestSystems, "c_sys5")
        pf_polar = ACPowerFlow{NewtonRaphsonACPowerFlow}()
        @test PF.solve_and_store_power_flow!(pf_polar, sys)
        pf_rect = ACPowerFlow{RectangularCurrentInjectionACPowerFlow}()
        data = PF.PowerFlowData(pf_rect, sys)
        R = PF.ACRectangularCIResidual(data, 1)
        x = Vector{Float64}(undef, length(R.Rv))
        PF.rect_initial_state!(x, data, R.bus_state_offset, R.bus_block_size, 1)
        R(x, 1)
        @test LinearAlgebra.norm(R.Rv, Inf) < 1e-7
    end

    @testset "c_sys14: residual zero at polar-converged state" begin
        sys = PSB.build_system(PSB.PSITestSystems, "c_sys14"; add_forecasts = false)
        pf_polar = ACPowerFlow{NewtonRaphsonACPowerFlow}()
        @test PF.solve_and_store_power_flow!(pf_polar, sys)
        pf_rect = ACPowerFlow{RectangularCurrentInjectionACPowerFlow}()
        data = PF.PowerFlowData(pf_rect, sys)
        R = PF.ACRectangularCIResidual(data, 1)
        x = Vector{Float64}(undef, length(R.Rv))
        PF.rect_initial_state!(x, data, R.bus_state_offset, R.bus_block_size, 1)
        R(x, 1)
        @test LinearAlgebra.norm(R.Rv, Inf) < 1e-7
    end
end

@testset "Rectangular CI Residual: flat start has nonzero residual" begin
    sys = PSB.build_system(PSB.PSITestSystems, "c_sys5")
    pf_rect = ACPowerFlow{RectangularCurrentInjectionACPowerFlow}()
    data = PF.PowerFlowData(pf_rect, sys)
    # Force flat start: |V|=1, θ=0
    data.bus_magnitude[:, 1] .= 1.0
    data.bus_angles[:, 1] .= 0.0
    R = PF.ACRectangularCIResidual(data, 1)
    x = Vector{Float64}(undef, length(R.Rv))
    PF.rect_initial_state!(x, data, R.bus_state_offset, R.bus_block_size, 1)
    R(x, 1)
    # Something to converge from
    @test LinearAlgebra.norm(R.Rv, Inf) > 1e-3
end

@testset "Rectangular CI Residual: state layout sanity" begin
    sys = PSB.build_system(PSB.PSITestSystems, "c_sys5")
    pf_rect = ACPowerFlow{RectangularCurrentInjectionACPowerFlow}()
    data = PF.PowerFlowData(pf_rect, sys)
    R = PF.ACRectangularCIResidual(data, 1)
    # Block sizes: 2 for PQ/REF, 3 for PV
    bt = view(data.bus_type, :, 1)
    expected = sum(b == PSY.ACBusTypes.PV ? 3 : 2 for b in bt) +
               4 * size(data.lcc.p_set, 1)
    @test length(R.Rv) == expected
end
