@testset "Rectangular CI Jacobian: asymptotic verification" begin
    @testset "c_sys5 at polar-converged + perturbation" begin
        sys = PSB.build_system(PSB.PSITestSystems, "c_sys5")
        pf_polar = ACPowerFlow{NewtonRaphsonACPowerFlow}()
        PF.solve_and_store_power_flow!(pf_polar, sys)
        pf_rect = ACRectangularPowerFlow{NewtonRaphsonACPowerFlow}()
        data = PF.PowerFlowData(pf_rect, sys)
        R = PF.ACRectangularCIResidual(data, 1)
        J = PF.ACRectangularCIJacobian(data, R, 1)
        x = Vector{Float64}(undef, length(R.Rv))
        PF.rect_initial_state!(x, data, R.bus_state_offset, R.bus_block_size, 1)
        # Avoid verifying at the special converged state — see note in
        # verify_jacobian (test_jacobian.jl) about hidden zeros.
        Random.seed!(42)
        x .+= 0.02 .* randn(length(x))
        R(data, x, 1)
        J(data, 1)
        verify_jacobian_asymptotic(R, data, copy(J.Jv), x, 1; label = "rect CI c_sys5")
    end

    @testset "c_sys14 at polar-converged + perturbation" begin
        sys = PSB.build_system(PSB.PSITestSystems, "c_sys14"; add_forecasts = false)
        pf_polar = ACPowerFlow{NewtonRaphsonACPowerFlow}()
        PF.solve_and_store_power_flow!(pf_polar, sys)
        pf_rect = ACRectangularPowerFlow{NewtonRaphsonACPowerFlow}()
        data = PF.PowerFlowData(pf_rect, sys)
        R = PF.ACRectangularCIResidual(data, 1)
        J = PF.ACRectangularCIJacobian(data, R, 1)
        x = Vector{Float64}(undef, length(R.Rv))
        PF.rect_initial_state!(x, data, R.bus_state_offset, R.bus_block_size, 1)
        Random.seed!(42)
        x .+= 0.02 .* randn(length(x))
        R(data, x, 1)
        J(data, 1)
        verify_jacobian_asymptotic(R, data, copy(J.Jv), x, 1; label = "rect CI c_sys14")
    end

    @testset "ZIP constant-current load at perturbed state" begin
        sys = System(100.0)
        b1 = _add_simple_bus!(sys, 1, PSY.ACBusTypes.REF, 230, 1.1, 0.0)
        b2 = _add_simple_bus!(sys, 2, PSY.ACBusTypes.PQ, 230, 1.1, 0.0)
        _add_simple_line!(sys, b1, b2, 5e-3, 5e-3, 1e-3)
        _add_simple_source!(sys, b1, 0.0, 0.0)
        _add_simple_zip_load!(
            sys, b2;
            constant_power_active_power = 0.5,
            constant_power_reactive_power = 0.2,
            constant_current_active_power = 2.0,
            constant_current_reactive_power = 1.0,
        )
        pf_polar = ACPowerFlow{NewtonRaphsonACPowerFlow}()
        PF.solve_and_store_power_flow!(pf_polar, sys)
        pf_rect = ACRectangularPowerFlow{NewtonRaphsonACPowerFlow}(;
            correct_bustypes = true,
            solution_parameters = SolutionParameters(; validate_voltage_magnitudes = false),
        )
        data = PF.PowerFlowData(pf_rect, sys)
        R = PF.ACRectangularCIResidual(data, 1)
        J = PF.ACRectangularCIJacobian(data, R, 1)
        x = Vector{Float64}(undef, length(R.Rv))
        PF.rect_initial_state!(x, data, R.bus_state_offset, R.bus_block_size, 1)
        Random.seed!(7)
        x .+= 0.02 .* randn(length(x))
        R(data, x, 1)
        J(data, 1)
        verify_jacobian_asymptotic(
            R,
            data,
            copy(J.Jv),
            x,
            1;
            label = "rect CI ZIP perturbed",
        )
    end

    @testset "c_sys5 at perturbed (non-converged) state" begin
        sys = PSB.build_system(PSB.PSITestSystems, "c_sys5")
        pf_polar = ACPowerFlow{NewtonRaphsonACPowerFlow}()
        PF.solve_and_store_power_flow!(pf_polar, sys)
        pf_rect = ACRectangularPowerFlow{NewtonRaphsonACPowerFlow}()
        data = PF.PowerFlowData(pf_rect, sys)
        R = PF.ACRectangularCIResidual(data, 1)
        J = PF.ACRectangularCIJacobian(data, R, 1)
        x = Vector{Float64}(undef, length(R.Rv))
        PF.rect_initial_state!(x, data, R.bus_state_offset, R.bus_block_size, 1)
        Random.seed!(42)
        x .+= 0.05 .* randn(length(x))
        R(data, x, 1)
        J(data, 1)
        verify_jacobian_asymptotic(
            R,
            data,
            copy(J.Jv),
            x,
            1;
            label = "rect CI c_sys5 perturbed",
        )
    end
end

@testset "Rectangular Jacobian: pattern independent of the PQ/PV split" begin
    # A Q-limit flip (PV → PQ) only rewrites the bus's second row, so the linear-solver cache
    # keeps its symbolic analysis.
    sys = PSB.build_system(PSB.PSITestSystems, "c_sys14"; add_forecasts = false)
    data = PF.PowerFlowData(ACRectangularPowerFlow{NewtonRaphsonACPowerFlow}(), sys)
    function rect_J(data)
        R = PF.ACRectangularCIResidual(data, 1)
        x = Vector{Float64}(undef, length(R.Rv))
        PF.rect_initial_state!(x, data, R.bus_state_offset, R.bus_block_size, 1)
        R(data, x, 1)
        return PF.ACRectangularCIJacobian(data, R, 1).Jv
    end
    J_pv = rect_J(data)
    pv = findfirst(==(PSY.ACBusTypes.PV), data.bus_type[:, 1])
    data.bus_type[pv, 1] = PSY.ACBusTypes.PQ
    J_pq = rect_J(data)
    @test J_pq.colptr == J_pv.colptr
    @test J_pq.rowval == J_pv.rowval
    @test J_pv[2 * pv, 2 * pv - 1] ==
          2 * data.bus_magnitude[pv, 1] * cos(data.bus_angles[pv, 1])
    @test J_pq[2 * pv, 2 * pv - 1] != J_pv[2 * pv, 2 * pv - 1]
end

@testset "Rectangular CI Jacobian: two swings in one island (multi-swing)" begin
    # `_rect_two_swing_system` / `_rect_pf_settings` live in test_utils/cross_file_fixtures.jl.
    sys = _rect_two_swing_system()
    pf_rect = ACRectangularPowerFlow{NewtonRaphsonACPowerFlow}(;
        solution_parameters = _rect_pf_settings())
    data = PF.PowerFlowData(pf_rect, sys)
    R = PF.ACRectangularCIResidual(data, 1)
    J = PF.ACRectangularCIJacobian(data, R, 1)
    x = Vector{Float64}(undef, length(R.Rv))
    PF.rect_initial_state!(x, data, R.bus_state_offset, R.bus_block_size, 1)
    # Avoid the special converged state (see verify_jacobian note in
    # test_jacobian.jl about hidden zeros).
    Random.seed!(42)
    x .+= 0.01 .* randn(length(x))
    R(data, x, 1)
    J(data, 1)
    verify_jacobian_asymptotic(R, data, copy(J.Jv), x, 1; label = "rect CI two-swing")
end
