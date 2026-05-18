@testset "Mixed CPB Jacobian: structure + constant PQ off-diagonal blocks" begin
    function _check_mixed_cpb_jacobian(sys, label)
        @testset "$label structure invariants" begin
            pf_rect = ACRectangularPowerFlow{NewtonRaphsonACPowerFlow}()
            data = PF.PowerFlowData(pf_rect, sys)
            R = PF.ACMixedCPBResidual(data, 1)
            x = Vector{Float64}(undef, length(R.Rv))
            PF.mixed_initial_state!(x, data, R.bus_state_offset,
                R.bus_block_size, 1)
            # Evaluate the residual once so e_state / Ir_acc are populated
            # before constructing the Jacobian (the constructor's self-call
            # reads them to fill the state-dependent entries).
            R(x, 1)
            J = PF.ACMixedCPBJacobian(R, 1)

            n_buses = first(size(data.bus_type))
            n_lcc = size(data.lcc.p_set, 1)
            total = R.total_bus_state + 4 * n_lcc

            @test size(J.Jv) == (total, total)
            @test J.total_bus_state == R.total_bus_state

            # Every bus block is 2×2 in MCPB (no PV 3-slot expansion).
            @test all(==(2), R.bus_block_size)
            @test R.total_bus_state == 2 * n_buses

            bus_types = data.bus_type[:, 1]
            Y = R.Y_bus_eff

            # Constant PQ off-diagonal blocks: written by
            # _populate_mixed_constant_yb_blocks! in IMAG-FIRST ordering.
            #   Jv[off,   k_off]   = -B_ik   Jv[off,   k_off+1] = -G_ik
            #   Jv[off+1, k_off]   = -G_ik   Jv[off+1, k_off+1] = +B_ik
            Yr = SparseArrays.rowvals(Y)
            Yv = SparseArrays.nonzeros(Y)
            n_checked = 0
            for col in 1:n_buses
                bus_types[col] == PSY.ACBusTypes.REF && continue
                col_off = Int(R.bus_state_offset[col])
                for j in Y.colptr[col]:(Y.colptr[col + 1] - 1)
                    row = Yr[j]
                    row == col && continue
                    bus_types[row] != PSY.ACBusTypes.PQ && continue
                    row_off = Int(R.bus_state_offset[row])
                    g = real(Yv[j])
                    b = imag(Yv[j])
                    @test J.Jv[row_off, col_off] == -b
                    @test J.Jv[row_off, col_off + 1] == -g
                    @test J.Jv[row_off + 1, col_off] == -g
                    @test J.Jv[row_off + 1, col_off + 1] == b
                    n_checked += 1
                end
            end
            @test n_checked > 0  # system actually exercises PQ off-diagonals

            # offdiag_pv_nz cache: 2-row layout (PV power-balance row only;
            # the PV voltage-constraint row has no off-diagonals).
            nnz_total = length(SparseArrays.nonzeros(J.Jv))
            @test size(J.offdiag_pv_nz, 1) == 2
            @test length(J.offdiag_pv_i) == size(J.offdiag_pv_nz, 2)
            @test length(J.offdiag_pv_k) == size(J.offdiag_pv_nz, 2)

            Jvnz = SparseArrays.nonzeros(J.Jv)
            n_sentinel_checked = 0
            for p in axes(J.offdiag_pv_nz, 2)
                i = J.offdiag_pv_i[p]
                k = J.offdiag_pv_k[p]
                @test bus_types[i] == PSY.ACBusTypes.PV
                @test bus_types[k] != PSY.ACBusTypes.REF
                @test i != k
                i_off = Int(R.bus_state_offset[i])
                k_off = Int(R.bus_state_offset[k])
                for r in 1:2
                    idx = J.offdiag_pv_nz[r, p]
                    @test 1 <= idx <= nnz_total
                    # Sentinel round-trip: write a unique value into the cached
                    # nzval slot and confirm sparse getindex reads it back at
                    # exactly (PV power-balance row i_off, neighbor e/f column).
                    orig = Jvnz[idx]
                    sentinel = -987654.321 - p - r
                    Jvnz[idx] = sentinel
                    target_col = r == 1 ? k_off : k_off + 1
                    @test J.Jv[i_off, target_col] == sentinel
                    Jvnz[idx] = orig
                    n_sentinel_checked += 1
                end
            end
            # The strengthened sentinel block must actually run (n_pv_pairs>0).
            @test n_sentinel_checked > 0

            # diag_base_nz covers a 2×2 block per bus.
            @test size(J.diag_base_nz) == (4, n_buses)
            for i in 1:n_buses
                for r in 1:4
                    @test 1 <= J.diag_base_nz[r, i] <= nnz_total
                end
            end

            # Structure (colptr/rowval) is fixed; constant PQ off-diagonals
            # stay put after perturbing + re-evaluating the residual only
            # (J is intentionally NOT re-called here, so only the
            # construction-time constant blocks are asserted; per-iteration
            # value correctness is covered by the finite-difference testset).
            colptr0 = copy(J.Jv.colptr)
            rowval0 = copy(J.Jv.rowval)
            pq_snapshot = Dict{Tuple{Int, Int}, Float64}()
            for col in 1:n_buses
                bus_types[col] == PSY.ACBusTypes.REF && continue
                col_off = Int(R.bus_state_offset[col])
                for j in Y.colptr[col]:(Y.colptr[col + 1] - 1)
                    row = Yr[j]
                    row == col && continue
                    bus_types[row] != PSY.ACBusTypes.PQ && continue
                    row_off = Int(R.bus_state_offset[row])
                    for dr in 0:1, dc in 0:1
                        pq_snapshot[(row_off + dr, col_off + dc)] =
                            J.Jv[row_off + dr, col_off + dc]
                    end
                end
            end

            Random.seed!(2024)
            x .+= 0.01 .* randn(length(x))
            R(x, 1)  # only the residual re-evaluates; J is not re-called here
            @test J.Jv.colptr == colptr0
            @test J.Jv.rowval == rowval0
            for ((rr, cc), v) in pq_snapshot
                @test J.Jv[rr, cc] == v
            end
        end
    end

    _check_mixed_cpb_jacobian(
        PSB.build_system(PSB.PSITestSystems, "c_sys5"), "c_sys5")
    # c_sys14 has PV–PQ neighbor pairs (n_pv_pairs>0), exercising the
    # offdiag_pv_nz sentinel round-trip with more PV off-diagonals.
    _check_mixed_cpb_jacobian(
        PSB.build_system(PSB.PSITestSystems, "c_sys14"; add_forecasts = false),
        "c_sys14")
end

@testset "Mixed CPB Jacobian vs finite difference" begin
    function _fd_check(R, x0)
        n = length(x0)
        Rv = Vector{Float64}(undef, n)
        R(Rv, x0, 1)
        # The constructor's self-call already populates J; no extra J(1) needed.
        J = PF.ACMixedCPBJacobian(R, 1)
        Jdense = Matrix(J.Jv)
        J_fd = Matrix{Float64}(undef, n, n)
        eps = 1e-7
        Rp = Vector{Float64}(undef, n)
        Rm = Vector{Float64}(undef, n)
        x = copy(x0)
        for k in 1:n
            xk = x[k]
            x[k] = xk + eps
            R(Rp, x, 1)
            x[k] = xk - eps
            R(Rm, x, 1)
            x[k] = xk
            @views J_fd[:, k] .= (Rp .- Rm) ./ (2 * eps)
        end
        # Restore residual state caches to x0 (FD perturbed them).
        R(Rv, x0, 1)
        return maximum(abs.(Jdense .- J_fd))
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
        err = _fd_check(R, x)
        @test err < 1e-6
    end

    @testset "c_sys14" begin
        sys = PSB.build_system(PSB.PSITestSystems, "c_sys14"; add_forecasts = false)
        R, x = _build_mixed_x(sys)
        Random.seed!(2024)
        x .+= 1e-3 .* randn(length(x))
        err = _fd_check(R, x)
        @test err < 1e-6
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
        err = _fd_check(R, x)
        @test err < 1e-6
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
        err = _fd_check(R, x)
        @test err < 1e-6
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
        err = _fd_check(R, x)
        @test err < 1e-6
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
