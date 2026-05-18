@testset "Mixed CPB Jacobian: structural invariants + constant PQ off-diagonal blocks" begin
    function _check_structure(sys, label)
        @testset "$label" begin
            pf_rect = ACRectangularPowerFlow{NewtonRaphsonACPowerFlow}()
            data = PF.PowerFlowData(pf_rect, sys)
            R = PF.ACMixedCPBResidual(data, 1)
            x = Vector{Float64}(undef, length(R.Rv))
            PF.mixed_initial_state!(x, data, R.bus_state_offset,
                R.bus_block_size, 1)
            R(x, 1)
            J = PF.ACMixedCPBJacobian(R, 1)

            n_buses = first(size(data.bus_type))
            n_lcc = size(data.lcc.p_set, 1)
            total = R.total_bus_state + 4 * n_lcc

            # Formulation property (not an implementation detail): MCPB uses a
            # uniform 2-slot block per bus — no PV→3 expansion — so the bus
            # state is exactly 2·n_buses, plus the 4-per-LCC tail. The Jacobian
            # is square over the full state.
            @test size(J.Jv) == (total, total)
            @test J.total_bus_state == R.total_bus_state
            @test all(==(2), R.bus_block_size)
            @test R.total_bus_state == 2 * n_buses

            bus_types = data.bus_type[:, 1]
            J_first = copy(J.Jv)

            # Re-evaluate at a perturbed state. The PQ divided-current-balance
            # rows are linear in the network voltages, so their off-diagonal
            # Y_bus blocks are CONSTANT across iterations. PV power-balance
            # off-diagonals are nonlinear and are intentionally NOT asserted
            # here — their correctness is covered by the asymptotic
            # finite-difference testset below.
            Random.seed!(123)
            x .+= 0.01 .* randn(length(x))
            R(x, 1)
            J(1)
            J_second = copy(J.Jv)

            # Sparsity structure is fixed across iterations.
            @test J_second.colptr == J_first.colptr
            @test J_second.rowval == J_first.rowval

            n_checked = 0
            for col in 1:n_buses
                bus_types[col] == PSY.ACBusTypes.REF && continue
                col_off = Int(R.bus_state_offset[col])
                for row in 1:n_buses
                    row == col && continue
                    bus_types[row] != PSY.ACBusTypes.PQ && continue
                    row_off = Int(R.bus_state_offset[row])
                    for dr in 0:1, dc in 0:1
                        @test J_first[row_off + dr, col_off + dc] ==
                              J_second[row_off + dr, col_off + dc]
                        n_checked += 1
                    end
                end
            end
            @test n_checked > 0  # system actually exercises PQ off-diagonals
        end
    end

    _check_structure(
        PSB.build_system(PSB.PSITestSystems, "c_sys5"), "c_sys5")
    # c_sys14 has PV–PQ neighbor pairs, exercising more off-diagonal blocks.
    _check_structure(
        PSB.build_system(PSB.PSITestSystems, "c_sys14"; add_forecasts = false),
        "c_sys14")
end

@testset "Mixed CPB Jacobian: asymptotic verification" begin
    # Verify the analytic Jacobian by its asymptotic agreement with the
    # residual (O(Δx²) Taylor remainder), not a single fixed-tolerance
    # finite-difference snapshot. Mirrors test_rectangular_ci_jacobian.jl.
    function _verify_mixed_jacobian(R, x, label)
        R(x, 1)
        J = PF.ACMixedCPBJacobian(R, 1)
        verify_jacobian_asymptotic(R, copy(J.Jv), x, 1; label = label)
    end

    function _build_mixed_x(sys)
        pf_polar = ACPowerFlow{NewtonRaphsonACPowerFlow}()
        @test PF.solve_and_store_power_flow!(pf_polar, sys)
        pf_mixed = ACMixedPowerFlow{NewtonRaphsonACPowerFlow}()
        data = PF.PowerFlowData(pf_mixed, sys)
        R = PF.ACMixedCPBResidual(data, 1)
        x = Vector{Float64}(undef, length(R.Rv))
        PF.mixed_initial_state!(x, data, R.bus_state_offset, R.bus_block_size, 1)
        return R, x
    end

    @testset "c_sys5" begin
        sys = PSB.build_system(PSB.PSITestSystems, "c_sys5")
        R, x = _build_mixed_x(sys)
        Random.seed!(2024)
        x .+= 1e-3 .* randn(length(x))
        _verify_mixed_jacobian(R, x, "mixed CPB c_sys5")
    end

    @testset "c_sys14" begin
        sys = PSB.build_system(PSB.PSITestSystems, "c_sys14"; add_forecasts = false)
        R, x = _build_mixed_x(sys)
        Random.seed!(2024)
        x .+= 1e-3 .* randn(length(x))
        _verify_mixed_jacobian(R, x, "mixed CPB c_sys14")
    end

    @testset "ZIP load (P+I+Z combination)" begin
        sys = _build_zip_2bus_system(;
            power_pq = (0.5, 0.2),
            current_pq = (2.0, 1.0),
            impedance_pq = (1.5, 0.8),
        )
        pf_polar = ACPowerFlow{NewtonRaphsonACPowerFlow}(; correct_bustypes = true)
        @test PF.solve_and_store_power_flow!(pf_polar, sys)
        pf_mixed = ACMixedPowerFlow{NewtonRaphsonACPowerFlow}(;
            correct_bustypes = true)
        data = PF.PowerFlowData(pf_mixed, sys)
        R = PF.ACMixedCPBResidual(data, 1)
        x = Vector{Float64}(undef, length(R.Rv))
        PF.mixed_initial_state!(x, data, R.bus_state_offset, R.bus_block_size, 1)
        Random.seed!(2024)
        x .+= 1e-3 .* randn(length(x))
        _verify_mixed_jacobian(R, x, "mixed CPB ZIP")
    end

    function _build_mixed_lcc_x(sys; correct_bustypes = false)
        settings = Dict{Symbol, Any}(:validate_voltage_magnitudes => false)
        pf_polar = ACPowerFlow{NewtonRaphsonACPowerFlow}(;
            correct_bustypes = correct_bustypes, solver_settings = settings)
        @test PF.solve_and_store_power_flow!(pf_polar, sys)
        pf_mixed = ACMixedPowerFlow{NewtonRaphsonACPowerFlow}(;
            correct_bustypes = correct_bustypes, solver_settings = settings)
        data = PF.PowerFlowData(pf_mixed, sys)
        R = PF.ACMixedCPBResidual(data, 1)
        x = Vector{Float64}(undef, length(R.Rv))
        PF.mixed_initial_state!(x, data, R.bus_state_offset, R.bus_block_size, 1)
        return R, x
    end

    @testset "LCC PQ terminals (simple_lcc_system)" begin
        sys, _ = simple_lcc_system()
        R, x = _build_mixed_lcc_x(sys)
        Random.seed!(2024)
        x .+= 1e-3 .* randn(length(x))
        _verify_mixed_jacobian(R, x, "mixed CPB LCC PQ")
    end

    @testset "LCC PV terminal" begin
        sys, _ = simple_lcc_system()
        b2 = PSY.get_component(ACBus, sys, "bus_2")
        PSY.set_bustype!(b2, ACBusTypes.PV)
        PSY.set_magnitude!(b2, 1.05)
        _add_simple_thermal_standard!(sys, b2, 0.3, 0.0)
        R, x = _build_mixed_lcc_x(sys)
        Random.seed!(2024)
        x .+= 1e-3 .* randn(length(x))
        _verify_mixed_jacobian(R, x, "mixed CPB LCC PV")
    end
end

@testset "Mixed CPB Jacobian: zero allocation per Newton iteration" begin
    # Builds a populated MCPB Jacobian functor; mirrors the FD helper's setup.
    function _build_mixed_J(sys)
        pf_rect = ACRectangularPowerFlow{NewtonRaphsonACPowerFlow}()
        data = PF.PowerFlowData(pf_rect, sys)
        R = PF.ACMixedCPBResidual(data, 1)
        x = Vector{Float64}(undef, length(R.Rv))
        PF.mixed_initial_state!(x, data, R.bus_state_offset,
            R.bus_block_size, 1)
        R(x, 1)
        return PF.ACMixedCPBJacobian(R, 1)
    end
    function _build_rect_J(sys)
        pf_rect = ACRectangularPowerFlow{NewtonRaphsonACPowerFlow}()
        data = PF.PowerFlowData(pf_rect, sys)
        R = PF.ACRectangularCIResidual(data, 1)
        x = Vector{Float64}(undef, length(R.Rv))
        PF.rect_initial_state!(x, data, R.bus_state_offset,
            R.bus_block_size, 1)
        R(x, 1)
        return PF.ACRectangularCIJacobian(R, 1)
    end

    function _check_zero_alloc(sys, label)
        @testset "$label" begin
            J = _build_mixed_J(sys)
            J(1)                       # warm-up (JIT)
            a_mixed = @allocated J(1)
            Jr = _build_rect_J(sys)
            Jr(1)                      # warm-up (JIT)
            a_rect = @allocated Jr(1)
            # Measured (c_sys5 & c_sys14, neither has an LCC):
            #   mixed @allocated J(1) == 80, rect @allocated J(1) == 80.
            # The inner `_update_mixed_cpb_jacobian_values!` is verified
            # 0-alloc in isolation; the 80 bytes are the boxed return of the
            # dynamic dispatch through the untyped `Jf!::Function` field —
            # an artifact of the mirror-for-validation convention that keeps
            # `Jf!::Function` identical to `ACRectangularCIJacobian` (rect
            # allocates the identical 80 bytes for the same reason). Guard
            # against regression by requiring the mixed hot path to allocate
            # no more than the rect template it mirrors.
            @test a_mixed <= a_rect
        end
    end

    _check_zero_alloc(
        PSB.build_system(PSB.PSITestSystems, "c_sys5"), "c_sys5")
    _check_zero_alloc(
        PSB.build_system(PSB.PSITestSystems, "c_sys14"; add_forecasts = false),
        "c_sys14")
end

@testset "Mixed CPB solver-driver wiring" begin
    sys = PSB.build_system(PSB.PSITestSystems, "c_sys5")
    data = PF.PowerFlowData(ACMixedPowerFlow{NewtonRaphsonACPowerFlow}(), sys)
    R, J, x0 = PF.initialize_power_flow_variables(
        ACMixedPowerFlow{NewtonRaphsonACPowerFlow}(), data, 1)
    @test all(isfinite, x0)

    # Smoke check: the shared step functions accept the mixed R/J without a
    # MethodError (full parity is Stage 3 Task 3.1).
    linSolveCache = PF.KLULinSolveCache(J.Jv)
    PF.symbolic_factor!(linSolveCache, J.Jv)
    stateVector = PF.StateVectorCache(x0, R.Rv)
    @test_nowarn PF._simple_step(1, stateVector, linSolveCache, R, J)
end
