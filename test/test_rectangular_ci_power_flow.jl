function _rect_pf_settings()
    return Dict{Symbol, Any}(:validate_voltage_magnitudes => false)
end

@testset "Rectangular CI Power Flow: convergence" begin
    @testset "c_sys5 converges" begin
        sys = PSB.build_system(PSB.PSITestSystems, "c_sys5")
        pf_rect = ACRectangularPowerFlow{NewtonRaphsonACPowerFlow}(;
            solver_settings = _rect_pf_settings())
        @test PF.solve_and_store_power_flow!(pf_rect, sys)
    end

    @testset "c_sys14 converges" begin
        sys = PSB.build_system(PSB.PSITestSystems, "c_sys14"; add_forecasts = false)
        pf_rect = ACRectangularPowerFlow{NewtonRaphsonACPowerFlow}(;
            solver_settings = _rect_pf_settings())
        @test PF.solve_and_store_power_flow!(pf_rect, sys)
    end
end

@testset "Rectangular CI Power Flow: non-convergence returns missing" begin
    sys = PSB.build_system(PSB.PSITestSystems, "c_sys14"; add_forecasts = false)
    # maxIterations = 1 from flat start cannot converge c_sys14; the solver must
    # report non-convergence (results = missing) rather than error or hang.
    pf = ACRectangularPowerFlow{NewtonRaphsonACPowerFlow}(;
        solver_settings = merge(_rect_pf_settings(),
            Dict{Symbol, Any}(:maxIterations => 1)))
    @test_logs(
        (:error, r".*solver failed to converge"),
        match_mode = :any,
        @test ismissing(solve_power_flow(pf, sys))
    )
end

@testset "Rectangular CI Power Flow (LM): non-convergence returns missing" begin
    sys = PSB.build_system(PSB.PSITestSystems, "c_sys14"; add_forecasts = false)
    # maxIterations = 1 from flat start cannot converge c_sys14; the solver must
    # report non-convergence (results = missing) rather than error or hang.
    pf = ACRectangularPowerFlow{LevenbergMarquardtACPowerFlow}(;
        solver_settings = merge(_rect_pf_settings(),
            Dict{Symbol, Any}(:maxIterations => 1)))
    @test_logs(
        (:error, r".*solver failed to converge"),
        match_mode = :any,
        @test ismissing(solve_power_flow(pf, sys))
    )
end

@testset "Rectangular CI Power Flow: parity with polar NR" begin
    fixtures = [
        ("c_sys5", false),
        ("c_sys14", false),
    ]
    for (name, with_forecasts) in fixtures
        @testset "$name" begin
            sys_p = if with_forecasts
                PSB.build_system(PSB.PSITestSystems, name)
            else
                PSB.build_system(PSB.PSITestSystems, name; add_forecasts = false)
            end
            sys_r = deepcopy(sys_p)
            pf_p = ACPowerFlow{NewtonRaphsonACPowerFlow}()
            pf_r = ACRectangularPowerFlow{NewtonRaphsonACPowerFlow}(;
                solver_settings = _rect_pf_settings())
            res_p = solve_power_flow(pf_p, sys_p)
            res_r = solve_power_flow(pf_r, sys_r)
            @test res_p !== missing
            @test res_r !== missing
            @test maximum(abs.(res_p["bus_results"].Vm - res_r["bus_results"].Vm)) < 1e-7
            @test maximum(abs.(res_p["bus_results"].θ - res_r["bus_results"].θ)) < 1e-7
        end
    end
end

@testset "Rectangular CI Power Flow: unsupported config rejected" begin
    # Removed fields: passing them is a constructor MethodError.
    @test_throws MethodError ACRectangularPowerFlow{NewtonRaphsonACPowerFlow}(;
        robust_power_flow = true)
    @test_throws MethodError ACRectangularPowerFlow{NewtonRaphsonACPowerFlow}(;
        calculate_loss_factors = true)
    @test_throws MethodError ACRectangularPowerFlow{NewtonRaphsonACPowerFlow}(;
        calculate_voltage_stability_factors = true)
    # Sanity: these kwargs ARE valid on the polar type — the MethodErrors above
    # prove the fields were removed from the rectangular type, not mistyped.
    @test ACPolarPowerFlow{NewtonRaphsonACPowerFlow}(;
        robust_power_flow = true) isa ACPolarPowerFlow
    @test ACPolarPowerFlow{NewtonRaphsonACPowerFlow}(;
        calculate_loss_factors = true) isa ACPolarPowerFlow
    @test ACPolarPowerFlow{NewtonRaphsonACPowerFlow}(;
        calculate_voltage_stability_factors = true) isa ACPolarPowerFlow
    # Polar-only solvers rejected at construction.
    @test_throws ArgumentError ACRectangularPowerFlow{RobustHomotopyPowerFlow}()
    @test_throws ArgumentError ACRectangularPowerFlow{GradientDescentACPowerFlow}()
    # Levenberg-Marquardt is now supported on the rectangular formulation.
    @test ACRectangularPowerFlow{LevenbergMarquardtACPowerFlow}() isa
          ACRectangularPowerFlow
end

@testset "Rectangular CI Power Flow: step strategy variants" begin
    # Verify Iwamoto and Trust Region wrappers converge through the rectangular CI
    # residual/Jacobian. The drivers (_simple_step, _iwamoto_step, _trust_region_step)
    # are generic over the residual/Jacobian functor interface, so all four step
    # strategies should work without any rectangular-specific code in the drivers.
    fixtures = [("c_sys5", false), ("c_sys14", false)]
    strategies = [
        ("plain NR", NewtonRaphsonACPowerFlow, Dict{Symbol, Any}()),
        ("NR + Iwamoto", NewtonRaphsonACPowerFlow,
            Dict{Symbol, Any}(:iwamoto => true)),
        ("Trust Region", TrustRegionACPowerFlow, Dict{Symbol, Any}()),
        ("TR + Iwamoto FB", TrustRegionACPowerFlow,
            Dict{Symbol, Any}(:iwamoto_fallback => true)),
        ("Levenberg-Marquardt", LevenbergMarquardtACPowerFlow,
            Dict{Symbol, Any}()),
    ]
    for (name, with_forecasts) in fixtures
        @testset "$name" begin
            sys_p = if with_forecasts
                PSB.build_system(PSB.PSITestSystems, name)
            else
                PSB.build_system(PSB.PSITestSystems, name; add_forecasts = false)
            end
            pf_p = ACPowerFlow{NewtonRaphsonACPowerFlow}()
            res_p = solve_power_flow(pf_p, deepcopy(sys_p))
            for (label, solver, extra_settings) in strategies
                @testset "$label" begin
                    sys_r = deepcopy(sys_p)
                    settings = merge(extra_settings, _rect_pf_settings())
                    pf_r = ACRectangularPowerFlow{solver}(;
                        solver_settings = settings)
                    res_r = solve_power_flow(pf_r, sys_r)
                    @test res_r !== missing
                    @test maximum(
                        abs.(res_p["bus_results"].Vm - res_r["bus_results"].Vm),
                    ) < 1e-7
                    @test maximum(
                        abs.(res_p["bus_results"].θ - res_r["bus_results"].θ),
                    ) < 1e-7
                end
            end
        end
    end
end

@testset "Rectangular CI: LM matches polar LM" begin
    for name in ("c_sys5", "c_sys14")
        @testset "$name" begin
            sys = PSB.build_system(PSB.PSITestSystems, name; add_forecasts = false)
            pf_polar = ACPolarPowerFlow{LevenbergMarquardtACPowerFlow}()
            res_polar = solve_power_flow(pf_polar, deepcopy(sys))
            pf_rect = ACRectangularPowerFlow{LevenbergMarquardtACPowerFlow}(;
                solver_settings = _rect_pf_settings())
            res_rect = solve_power_flow(pf_rect, deepcopy(sys))
            @test res_rect !== missing
            @test maximum(
                abs.(res_polar["bus_results"].Vm - res_rect["bus_results"].Vm),
            ) < 1e-7
            @test maximum(
                abs.(res_polar["bus_results"].θ - res_rect["bus_results"].θ),
            ) < 1e-7
        end
    end
end

@testset "ACTIVSg2000 (LM): polar and rectangular match polar NR" begin
    sys = PSB.build_system(PSB.MatpowerTestSystems, "matpower_ACTIVSg2000_sys")
    PSY.set_units_base_system!(sys, "SYSTEM_BASE")

    # Tight tolerance + generous iteration budget: LM refactorizes the sparse QR
    # every iteration and needs more iterations than NR on a 2000-bus system.
    lm_settings = Dict{Symbol, Any}(:tol => 1e-10, :maxIterations => 100)
    ref_settings = Dict{Symbol, Any}(:tol => 1e-10)

    # Reference: Newton-Raphson on the polar formulation (trusted for ACTIVSg2000
    # elsewhere in the suite).
    pf_ref = ACPolarPowerFlow{NewtonRaphsonACPowerFlow}(;
        correct_bustypes = true, solver_settings = ref_settings)
    res_ref = solve_power_flow(pf_ref, sys)

    pf_lm_polar = ACPolarPowerFlow{LevenbergMarquardtACPowerFlow}(;
        correct_bustypes = true, solver_settings = lm_settings)
    res_lm_polar = solve_power_flow(pf_lm_polar, sys)

    pf_lm_rect = ACRectangularPowerFlow{LevenbergMarquardtACPowerFlow}(;
        correct_bustypes = true,
        solver_settings = merge(_rect_pf_settings(), lm_settings))
    res_lm_rect = solve_power_flow(pf_lm_rect, sys)

    @test res_lm_polar !== missing
    @test res_lm_rect !== missing

    # Polar LM and rectangular LM both reproduce the NR reference solution.
    @test norm(res_lm_polar["bus_results"].Vm .- res_ref["bus_results"].Vm, Inf) < 1e-5
    @test norm(res_lm_polar["bus_results"].θ .- res_ref["bus_results"].θ, Inf) < 1e-5
    @test norm(res_lm_rect["bus_results"].Vm .- res_ref["bus_results"].Vm, Inf) < 1e-5
    @test norm(res_lm_rect["bus_results"].θ .- res_ref["bus_results"].θ, Inf) < 1e-5
    # Polar LM and rectangular LM agree with each other.
    @test norm(
        res_lm_polar["bus_results"].Vm .- res_lm_rect["bus_results"].Vm, Inf) < 1e-5
    @test norm(
        res_lm_polar["bus_results"].θ .- res_lm_rect["bus_results"].θ, Inf) < 1e-5
end

@testset "Rectangular CI: multi-period previous-solution warm start" begin
    sys = PSB.build_system(PSB.PSITestSystems, "c_sys5")
    pf = ACRectangularPowerFlow{NewtonRaphsonACPowerFlow}(;
        time_steps = 2,
        time_step_names = ["1", "2"],
    )
    data = PF.PowerFlowData(pf, sys)

    # Solve step 1, mark it converged so step 2's improve_x0 sees a `prev`.
    res1, J1, x1 = PF.initialize_power_flow_variables(pf, data, 1)
    data.converged[1] = true

    # improve_x0 at step 2 must run the previous-solution branch without error
    # and return a finite, correctly-sized state vector.
    res2 = PF.ACRectangularCIResidual(data, 2)
    x2 = PF.improve_x0(pf, data, res2, 2)
    @test length(x2) == length(res2.Rv)
    @test all(isfinite, x2)

    # End-to-end multi-period solve converges for both steps.
    data_e = PF.PowerFlowData(pf, sys)
    @test PF.solve_power_flow!(data_e)
    @test all(data_e.converged)
end

@testset "LM Marquardt diagonal scaling option" begin
    # Formulation-dispatched default: rectangular on, polar off.
    @test PF._default_marquardt_scaling(
        ACPolarPowerFlow{LevenbergMarquardtACPowerFlow}()) == false
    @test PF._default_marquardt_scaling(
        ACRectangularPowerFlow{LevenbergMarquardtACPowerFlow}()) == true

    # Known J with distinct column 2-norms: col1 = 5, col2 = 12. Index type must
    # match `LMWorkspace`'s `J_INDEX_TYPE` (Int64 on Apple, Int32 elsewhere).
    J = SparseArrays.sparse(
        PF.J_INDEX_TYPE[1, 2, 3], PF.J_INDEX_TYPE[1, 1, 2], [3.0, 4.0, 12.0], 3, 2)

    ws_on = PF.LMWorkspace(J; marquardt_scaling = true)
    @test ws_on.marquardt_scaling == true
    @test ws_on.D ≈ [5.0, 12.0]

    ws_off = PF.LMWorkspace(J; marquardt_scaling = false)
    @test ws_off.marquardt_scaling == false
    @test ws_off.D == ones(2)               # identity damping ⇒ polar unaffected
    @test PF.LMWorkspace(J).marquardt_scaling == false  # defaults to off

    # update_lambda! writes √λ·D into the damping diagonal.
    λ = 4.0
    PF.update_lambda!(ws_on, λ)
    PF.update_lambda!(ws_off, λ)
    @test ws_on.A.nzval[ws_on.λ_diag_indices[1]] ≈ sqrt(λ) * 5.0
    @test ws_on.A.nzval[ws_on.λ_diag_indices[2]] ≈ sqrt(λ) * 12.0
    @test ws_off.A.nzval[ws_off.λ_diag_indices[1]] ≈ sqrt(λ)  # == identity
    @test ws_off.A.nzval[ws_off.λ_diag_indices[2]] ≈ sqrt(λ)

    # Integration: rectangular LM converges with the default (scaling on) and
    # with the explicit override (scaling off); both match the polar NR
    # reference, and polar LM is unaffected by either setting.
    sys = PSB.build_system(PSB.PSITestSystems, "c_sys14"; add_forecasts = false)
    res_ref = solve_power_flow(
        ACPolarPowerFlow{NewtonRaphsonACPowerFlow}(), deepcopy(sys))

    res_rect_default = solve_power_flow(
        ACRectangularPowerFlow{LevenbergMarquardtACPowerFlow}(;
            solver_settings = _rect_pf_settings()), deepcopy(sys))
    res_rect_off = solve_power_flow(
        ACRectangularPowerFlow{LevenbergMarquardtACPowerFlow}(;
            solver_settings = merge(_rect_pf_settings(),
                Dict{Symbol, Any}(:marquardt_scaling => false))),
        deepcopy(sys))

    for res in (res_rect_default, res_rect_off)
        @test res !== missing
        @test maximum(
            abs.(res_ref["bus_results"].Vm - res["bus_results"].Vm)) < 1e-7
        @test maximum(
            abs.(res_ref["bus_results"].θ - res["bus_results"].θ)) < 1e-7
    end
end

@testset "Rectangular/Mixed squared voltage-magnitude validation" begin
    range = (min = 0.5, max = 1.5)   # min² = 0.25, max² = 2.25

    @testset "_validate_squared_voltage_magnitudes helper" begin
        bus_types = [PSY.ACBusTypes.PQ, PSY.ACBusTypes.PV, PSY.ACBusTypes.REF]
        offsets = [1, 3, 5]   # 2-slot blocks: (e,f) at off, off+1

        # Precompute step keeps only PQ/PV offsets, drops REF.
        validate_offsets = PF._pqpv_validate_offsets(bus_types, offsets)
        @test validate_offsets == [1, 3]

        # PQ in range (|V|²=1), PV out of range (|V|²=4 > 2.25), REF ignored.
        x_bad = [1.0, 0.0, 2.0, 0.0, 9.9, 9.9]
        @test_logs (:warn, r"voltage magnitudes outside of range") match_mode = :any PF._validate_squared_voltage_magnitudes(
            x_bad, validate_offsets, range, 1)

        # All free buses in range ⇒ silent (REF still ignored even if wild).
        x_ok = [1.0, 0.0, 1.0, 0.2, 9.9, 9.9]
        @test_logs min_level = Logging.Warn PF._validate_squared_voltage_magnitudes(
            x_ok, validate_offsets, range, 1)
    end

    @testset "rectangular CI residual dispatch" begin
        sys = PSB.build_system(PSB.PSITestSystems, "c_sys5")
        pf = ACRectangularPowerFlow{NewtonRaphsonACPowerFlow}(;
            solver_settings = _rect_pf_settings())
        data = PF.PowerFlowData(pf, sys)
        residual, _, x = PF.initialize_power_flow_variables(pf, data, 1)
        bus_types = PF.get_bus_type(data)

        # Flat/initial state is ~1 p.u. ⇒ in range ⇒ silent.
        @test_logs min_level = Logging.Warn PF._validate_state_magnitudes(
            residual, x, range, 1)

        # Drive the first PQ or PV bus's |V|² out of range via its (e,f) slots.
        b = findfirst(bt -> bt != PSY.ACBusTypes.REF, bus_types)
        off = Int(residual.bus_state_offset[b])
        x[off] = 3.0
        x[off + 1] = 0.0
        @test_logs (:warn, r"voltage magnitudes outside of range") match_mode = :any PF._validate_state_magnitudes(
            residual, x, range, 1)
    end
end

@testset "Rectangular CI: V_FLOOR2 guards degenerate (e,f)" begin
    # Mirrors the mixed CPB V_FLOOR2 hardening: collapsing a PQ bus voltage to
    # zero must not produce Inf/NaN in the rectangular residual or Jacobian
    # (the 1/|V|² current-balance terms are otherwise unguarded).
    sys = PSB.build_system(PSB.PSITestSystems, "c_sys5")
    pf = ACRectangularPowerFlow{NewtonRaphsonACPowerFlow}(;
        solver_settings = _rect_pf_settings())
    data = PF.PowerFlowData(pf, sys)
    residual, J, x = PF.initialize_power_flow_variables(pf, data, 1)
    bus_types = PF.get_bus_type(data)

    b = findfirst(bt -> bt == PSY.ACBusTypes.PQ, bus_types)
    off = Int(residual.bus_state_offset[b])
    x[off] = 0.0
    x[off + 1] = 0.0

    residual(x, 1)
    J(1)
    @test all(isfinite, residual.Rv)
    @test all(isfinite, J.Jv.nzval)
end
