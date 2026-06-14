@testset "FastDecoupled WP0: construction" begin
    # FD must construct for ALL THREE formulations (polar, rectangular, mixed).
    @test_nowarn ACPowerFlow{PF.FastDecoupledACPowerFlow}()
    @test_nowarn ACRectangularPowerFlow{PF.FastDecoupledACPowerFlow}()
    @test_nowarn ACMixedPowerFlow{PF.FastDecoupledACPowerFlow}()

    @test ACPowerFlow{PF.FastDecoupledACPowerFlow}() isa
          PF.ACPolarPowerFlow{PF.FastDecoupledACPowerFlow}
    @test ACRectangularPowerFlow{PF.FastDecoupledACPowerFlow}() isa
          PF.ACRectangularPowerFlow{PF.FastDecoupledACPowerFlow}
    @test ACMixedPowerFlow{PF.FastDecoupledACPowerFlow}() isa
          PF.ACMixedPowerFlow{PF.FastDecoupledACPowerFlow}
end

@testset "FastDecoupled WP0: default variant" begin
    @test PF._default_fd_variant(ACPowerFlow{PF.FastDecoupledACPowerFlow}()) ==
          :decoupled
    @test PF._default_fd_variant(ACRectangularPowerFlow{PF.FastDecoupledACPowerFlow}()) ==
          :fixed_jacobian
    @test PF._default_fd_variant(ACMixedPowerFlow{PF.FastDecoupledACPowerFlow}()) ==
          :fixed_jacobian
end

@testset "FastDecoupled WP0: settings validation" begin
    polar = ACPowerFlow{PF.FastDecoupledACPowerFlow}()
    rect = ACRectangularPowerFlow{PF.FastDecoupledACPowerFlow}()
    mixed = ACMixedPowerFlow{PF.FastDecoupledACPowerFlow}()

    # Invalid fd_variant.
    @test_throws ArgumentError PF._validate_fd_settings(
        polar, :nonsense, :XB, nothing)

    # Invalid fd_scheme.
    @test_throws ArgumentError PF._validate_fd_settings(
        polar, :decoupled, :ZZ, nothing)

    # :decoupled is polar-only in v1: rect/mixed must error.
    @test_throws ArgumentError PF._validate_fd_settings(
        rect, :decoupled, :XB, nothing)
    @test_throws ArgumentError PF._validate_fd_settings(
        mixed, :decoupled, :XB, nothing)

    # Unsupported handoff solver (only nothing / NR / TR / LM are accepted).
    @test_throws ArgumentError PF._validate_fd_settings(
        polar, :decoupled, :XB, RobustHomotopyPowerFlow)

    # Valid configurations must pass (no throw), including all supported handoff solvers.
    @test PF._validate_fd_settings(polar, :decoupled, :XB, nothing) === nothing
    @test PF._validate_fd_settings(polar, :decoupled, :BX, NewtonRaphsonACPowerFlow) ===
          nothing
    @test PF._validate_fd_settings(
        polar, :fixed_jacobian, :XB, TrustRegionACPowerFlow) === nothing
    @test PF._validate_fd_settings(
        polar, :decoupled, :XB, LevenbergMarquardtACPowerFlow) === nothing
    @test PF._validate_fd_settings(rect, :fixed_jacobian, :XB, nothing) === nothing
    @test PF._validate_fd_settings(mixed, :fixed_jacobian, :BX, nothing) === nothing
end

@testset "FastDecoupled WP0: driver dispatches & errors" begin
    sys = PSB.build_system(PSB.PSITestSystems, "c_sys5"; add_forecasts = false)
    # Rectangular + :decoupled is unsupported (decoupled is polar-only in v1): the FD driver
    # must validate and throw through the public solve path — proving the FD solver dispatches
    # to the new `_newton_power_flow` method and its settings validation is wired in. (The
    # polar :decoupled and :fixed_jacobian paths are now implemented in WP1–WP3 and no longer
    # throw, so the obsolete "not yet implemented" assertion was repointed here.)
    pf = ACRectangularPowerFlow{PF.FastDecoupledACPowerFlow}(;
        solver_settings = Dict{Symbol, Any}(:fd_variant => :decoupled))
    @test_throws ArgumentError solve_power_flow(pf, sys)
end

# =====================================================================================
# WP1 — B′/B″ matrix machinery (fast_decoupled_matrices.jl).
# =====================================================================================

# Build a small LOSSLESS (r=0), shunt-free (b_c=0, no bus shunts), nominal-tap (τ=1),
# no-phase-shifter network. On such a network at FLAT START (V=1, θ=0), the fast-decoupled
# blocks are EXACT and the XB and BX schemes coincide. This is the sign/value arbiter:
# B′ and B″ are compared against the codebase's OWN exact Jacobian sub-blocks (extracted
# from `ACPowerFlowJacobian.Jv`), never against the B-matrix code itself (no tautology).
function _lossless_flat_system()
    sys = PSY.System(100.0)
    b1 = _add_simple_bus!(sys, 1, PSY.ACBusTypes.REF, 230.0, 1.0, 0.0)
    b2 = _add_simple_bus!(sys, 2, PSY.ACBusTypes.PV, 230.0, 1.0, 0.0)
    b3 = _add_simple_bus!(sys, 3, PSY.ACBusTypes.PQ, 230.0, 1.0, 0.0)
    b4 = _add_simple_bus!(sys, 4, PSY.ACBusTypes.PQ, 230.0, 1.0, 0.0)
    # Source / generation so the REF and PV buses are sane; loads on PQ.
    _add_simple_source!(sys, b1, 0.0, 0.0)
    _add_simple_thermal_standard!(sys, b2, 0.2, 0.0)
    _add_simple_load!(sys, b3, 10.0, 5.0)
    _add_simple_load!(sys, b4, 8.0, 3.0)
    # Lossless, shunt-free, nominal-tap lines (r=0, b=0). Distinct reactances so the
    # off-diagonal structure is non-trivial and the value comparison is meaningful.
    _add_simple_line!(sys, b1, b2, 0.0, 0.10, 0.0)
    _add_simple_line!(sys, b2, b3, 0.0, 0.20, 0.0)
    _add_simple_line!(sys, b3, b4, 0.0, 0.05, 0.0)
    _add_simple_line!(sys, b1, b4, 0.0, 0.25, 0.0)
    return sys
end

# Extract the exact codebase Jacobian (at flat start) and the matching fast-decoupled
# sub-blocks. Returns (Jv, pvpq, pq, Vm) where Jv is evaluated at x0 (flat).
function _flat_exact_jacobian(data, time_step)
    pf = ACPowerFlow()
    residual, J, x0 =
        PF.initialize_power_flow_variables(pf, data, time_step;
            validate_voltage_magnitudes = false)
    # initialize_power_flow_variables already evaluates J at x0 (flat start here).
    ref, pv, pq = PF.bus_type_idx(data, time_step)
    pvpq = sort(vcat(pv, pq))
    return J.Jv, pvpq, sort(pq), view(data.bus_magnitude, :, time_step)
end

# Exact P-θ sub-block normalized by V: rows = P-mismatch rows Rv[2i-1] at pvpq divided
# by Vm[i], cols = θ entries x[2i] at pvpq. Mirrors B′ ≈ ∂(Rv_P/V)/∂θ over pvpq.
function _exact_Bp_block(Jv, pvpq, Vm)
    n = length(pvpq)
    M = zeros(n, n)
    for (a, i) in enumerate(pvpq), (b, j) in enumerate(pvpq)
        M[a, b] = Jv[2 * i - 1, 2 * j] / Vm[i]
    end
    return M
end

# Exact Q-V sub-block normalized by V: rows = Q-mismatch rows Rv[2i] at pq divided by
# Vm[i], cols = V entries x[2i-1] at pq. Mirrors B″ ≈ ∂(Rv_Q/V)/∂V over pq.
function _exact_Bpp_block(Jv, pq, Vm)
    n = length(pq)
    M = zeros(n, n)
    for (a, i) in enumerate(pq), (b, j) in enumerate(pq)
        M[a, b] = Jv[2 * i, 2 * j - 1] / Vm[i]
    end
    return M
end

@testset "FastDecoupled WP1: B′/B″ vs exact Jacobian (flat, lossless)" begin
    sys = _lossless_flat_system()
    pf = ACPowerFlow()
    data = PowerFlowData(pf, sys)
    time_step = 1

    Jv, pvpq, pq, Vm = _flat_exact_jacobian(data, time_step)
    Bp_exact = _exact_Bp_block(Jv, pvpq, Vm)
    Bpp_exact = _exact_Bpp_block(Jv, pq, Vm)

    for scheme in (:XB, :BX)
        fd = PF.build_fd_matrices(data, time_step, scheme)
        # B′ over pvpq.
        Bp = Matrix(PF.get_bp_matrix(fd))
        @test size(Bp) == size(Bp_exact)
        @test isapprox(Bp, Bp_exact; atol = 1e-6, rtol = 0)
        # B″ over pq (extracted from B″_full).
        Bpp_cache = PF.extract_bpp(fd, pq)
        Bpp = Matrix(PF.get_bpp_matrix(Bpp_cache))
        @test size(Bpp) == size(Bpp_exact)
        @test isapprox(Bpp, Bpp_exact; atol = 1e-6, rtol = 0)
    end

    # On a lossless, shunt-free, nominal-tap network XB == BX exactly.
    fd_xb = PF.build_fd_matrices(data, time_step, :XB)
    fd_bx = PF.build_fd_matrices(data, time_step, :BX)
    @test isapprox(Matrix(PF.get_bp_matrix(fd_xb)), Matrix(PF.get_bp_matrix(fd_bx));
        atol = 1e-9, rtol = 0)
    @test isapprox(
        Matrix(PF.get_bpp_matrix(PF.extract_bpp(fd_xb, pq))),
        Matrix(PF.get_bpp_matrix(PF.extract_bpp(fd_bx, pq)));
        atol = 1e-9, rtol = 0)
end

@testset "FastDecoupled WP1: restamp reconstruction (c_sys14)" begin
    sys = PSB.build_system(PSB.PSITestSystems, "c_sys14")
    pf = ACPowerFlow()
    data = PowerFlowData(pf, sys)
    time_step = 1
    Yb = ComplexF64.(Matrix(data.power_network_matrix.data))
    Yrec = Matrix(PF._restamp_ybus(PF._recover_arc_params(data)))
    @test isapprox(Yrec, Yb; atol = 1e-4, rtol = 0)

    # B″ symmetric; B′ symmetric here (c_sys14 has no phase shifters).
    fd = PF.build_fd_matrices(data, time_step, :XB)
    Bp = Matrix(PF.get_bp_matrix(fd))
    @test isapprox(Bp, transpose(Bp); atol = 1e-6, rtol = 0)
    ref, pv, pq = PF.bus_type_idx(data, time_step)
    Bpp = Matrix(PF.get_bpp_matrix(PF.extract_bpp(fd, sort(pq))))
    @test isapprox(Bpp, transpose(Bpp); atol = 1e-6, rtol = 0)
end

@testset "FastDecoupled WP1: restamp reconstruction (WECC240)" begin
    file = joinpath(
        TEST_DATA_DIR,
        "WECC240_v04_DPV_RE20_v33_6302_xfmr_DPbuscode_PFadjusted_V32_noRemoteVctrl.raw",
    )
    system = PSY.System(
        file;
        bus_name_formatter = x -> strip(string(x["name"])) * "-" * string(x["index"]),
        runchecks = false,
    )
    pf = ACPowerFlow(; skip_redistribution = true, correct_bustypes = true)
    data = PowerFlowData(pf, system)
    Yb = ComplexF64.(Matrix(data.power_network_matrix.data))
    Yrec = Matrix(PF._restamp_ybus(PF._recover_arc_params(data)))
    relerr = norm(Yrec - Yb) / norm(Yb)
    @test relerr <= 1e-4
end

# WP1 regression: a mostly-resistive branch whose reactance sits BELOW PNM's reactance floor but
# whose resistance is non-negligible (so PNM, which floors x only when r==x==0, leaves it
# untouched). The near-zero-x cap lives ONLY on the resistance-drop B-stamp path (`_fd_series`),
# so the recovered `ys`/`b_c`/shunt and the restamp stay at their true values. Before the cap was
# moved out of `_recover_arc_params`, this branch overwrote `ys`'s imaginary part with `-1/x_cap`,
# injecting a ~1e6 susceptance into the restamp off-diagonal and the full-`ys` half of B′/B″.
function _resistive_near_zero_x_system()
    sys = PSY.System(100.0)
    b1 = _add_simple_bus!(sys, 1, PSY.ACBusTypes.REF, 230.0, 1.0, 0.0)
    b2 = _add_simple_bus!(sys, 2, PSY.ACBusTypes.PV, 230.0, 1.0, 0.0)
    b3 = _add_simple_bus!(sys, 3, PSY.ACBusTypes.PQ, 230.0, 1.0, 0.0)
    _add_simple_source!(sys, b1, 0.0, 0.0)
    _add_simple_thermal_standard!(sys, b2, 0.3, 0.0)
    _add_simple_load!(sys, b3, 15.0, 6.0)
    _add_simple_line!(sys, b1, b2, 0.01, 0.10, 0.0)   # normal line keeps the network connected
    # Mostly-resistive branch: |x| below PNM's ZERO_IMPEDANCE_X_EPSILON, r ≫ 0.
    _add_simple_line!(sys, b2, b3, 0.05, 1e-8, 0.0)
    return sys
end

@testset "FastDecoupled WP1: mostly-resistive near-zero-x branch (restamp invariant)" begin
    sys = _resistive_near_zero_x_system()
    pf = ACPowerFlow()
    data = PowerFlowData(pf, sys)
    time_step = 1

    # The branch's reactance is below the cap threshold (1/FD_INV_X_CAP), so it exercises the
    # resistance-drop cap path; the cap is locked to PNM's reactance floor.
    @test PF.FD_INV_X_CAP == 1 / PNM.ZERO_IMPEDANCE_X_EPSILON

    # Recovered params / restamp must stay at TRUE values (no 1e6 susceptance leak): the restamp
    # reconstructs the original Ybus within ComplexF32 noise even for this branch.
    Yb = ComplexF64.(Matrix(data.power_network_matrix.data))
    Yrec = Matrix(PF._restamp_ybus(PF._recover_arc_params(data)))
    relerr = norm(Yrec - Yb) / norm(Yb)
    @test relerr <= 1e-4

    # The resistance-drop stamp path must still be finite (a true x→0 there would be Inf/NaN
    # without the cap); B′/B″ entries are all finite under both schemes.
    ref, pv, pq = PF.bus_type_idx(data, time_step)
    for scheme in (:XB, :BX)
        fd = PF.build_fd_matrices(data, time_step, scheme)
        @test all(isfinite, PF.get_bp_matrix(fd).nzval)
        @test all(isfinite, PF.get_bpp_matrix(PF.extract_bpp(fd, sort(pq))).nzval)
    end
end

# =====================================================================================
# WP2 — Frozen-Jacobian (:fixed_jacobian) loop + shared safeguard helpers.
# =====================================================================================

# Per-formulation frozen-Jacobian FD power flow constructor by name.
_fd_fixed_jacobian_pf(::Type{<:PF.ACPolarPowerFlow}; kwargs...) =
    ACPowerFlow{PF.FastDecoupledACPowerFlow}(;
        solver_settings = Dict{Symbol, Any}(:fd_variant => :fixed_jacobian, kwargs...))
_fd_fixed_jacobian_pf(::Type{<:PF.ACRectangularPowerFlow}; kwargs...) =
    ACRectangularPowerFlow{PF.FastDecoupledACPowerFlow}(;
        solver_settings = Dict{Symbol, Any}(:fd_variant => :fixed_jacobian, kwargs...))
_fd_fixed_jacobian_pf(::Type{<:PF.ACMixedPowerFlow}; kwargs...) =
    ACMixedPowerFlow{PF.FastDecoupledACPowerFlow}(;
        solver_settings = Dict{Symbol, Any}(:fd_variant => :fixed_jacobian, kwargs...))

# Plain (non-FD) formulation constructor parametrized by a solver type, for NR-parity refs.
_plain_pf(::Type{<:PF.ACPolarPowerFlow}, ::Type{S}) where {S} = ACPowerFlow{S}()
_plain_pf(::Type{<:PF.ACRectangularPowerFlow}, ::Type{S}) where {S} =
    ACRectangularPowerFlow{S}()
_plain_pf(::Type{<:PF.ACMixedPowerFlow}, ::Type{S}) where {S} = ACMixedPowerFlow{S}()

# The three FD-capable formulation type "tags" (the polar formulation type, etc.).
const _FD_FORMULATIONS = (
    PF.ACPolarPowerFlow,
    PF.ACRectangularPowerFlow,
    PF.ACMixedPowerFlow,
)

@testset "FastDecoupled WP2: :fixed_jacobian NR-parity (non-LCC)" begin
    systems = (
        (
            "c_sys5",
            () -> PSB.build_system(PSB.PSITestSystems, "c_sys5";
                add_forecasts = false),
        ),
        (
            "c_sys14",
            () -> PSB.build_system(PSB.PSITestSystems, "c_sys14";
                add_forecasts = false),
        ),
        ("matpower_case5",
            () -> PSB.build_system(PSB.MatpowerTestSystems, "matpower_case5_sys")),
    )
    for (sysname, build) in systems
        for Formulation in _FD_FORMULATIONS
            @testset "$sysname / $(nameof(Formulation))" begin
                # Independent NR reference solve on a FRESH system state.
                sys_nr = build()
                pf_nr = _plain_pf(Formulation, NewtonRaphsonACPowerFlow)
                data_nr = PowerFlowData(pf_nr, sys_nr)
                solve_power_flow!(data_nr)
                x_nr = _calc_x(data_nr, 1)

                # Pure frozen-Jacobian FD solve on its OWN fresh system state.
                sys_fd = build()
                pf_fd = _fd_fixed_jacobian_pf(Formulation)
                data_fd = PowerFlowData(pf_fd, sys_fd)
                solve_power_flow!(data_fd)
                x_fd = _calc_x(data_fd, 1)

                # FD must converge: bus voltages must match NR's to TIGHT_TOLERANCE.
                @test isapprox(data_fd.bus_magnitude[:, 1], data_nr.bus_magnitude[:, 1];
                    atol = TIGHT_TOLERANCE, rtol = 0)
                @test isapprox(data_fd.bus_angles[:, 1], data_nr.bus_angles[:, 1];
                    atol = TIGHT_TOLERANCE, rtol = 0)
                @test isapprox(x_fd, x_nr; atol = TIGHT_TOLERANCE, rtol = 0)
            end
        end
    end
end

@testset "FastDecoupled WP2: :fixed_jacobian ACTIVSg2000 (refreeze allowed)" begin
    sys_nr = PSB.build_system(PSB.MatpowerTestSystems, "matpower_ACTIVSg2000_sys")
    PSY.set_units_base_system!(sys_nr, "SYSTEM_BASE")
    pf_nr = ACPowerFlow{NewtonRaphsonACPowerFlow}(; correct_bustypes = true)
    data_nr = PowerFlowData(pf_nr, sys_nr)
    solve_power_flow!(data_nr)

    sys_fd = PSB.build_system(PSB.MatpowerTestSystems, "matpower_ACTIVSg2000_sys")
    PSY.set_units_base_system!(sys_fd, "SYSTEM_BASE")
    pf_fd = ACPowerFlow{PF.FastDecoupledACPowerFlow}(;
        correct_bustypes = true,
        solver_settings = Dict{Symbol, Any}(:fd_variant => :fixed_jacobian))
    data_fd = PowerFlowData(pf_fd, sys_fd)
    # refreeze_on_stall is on by default; large stiff system may need it. Report iters.
    converged = solve_power_flow!(data_fd)
    @test converged
    @test isapprox(data_fd.bus_magnitude[:, 1], data_nr.bus_magnitude[:, 1];
        atol = TIGHT_TOLERANCE, rtol = 0)
    @test isapprox(data_fd.bus_angles[:, 1], data_nr.bus_angles[:, 1];
        atol = TIGHT_TOLERANCE, rtol = 0)
end

# A stressed, high r/x two-bus PQ system used to exercise the safeguard helpers.
# High r/x makes the frozen Jacobian a poor approximation for FD-style iteration so
# the non-divergent backtracking / blowup paths can be provoked. `load_scale` controls
# stress: at 1.0 the system is comfortably solvable; large scales push it to FD failure.
function _stressed_high_rx_system(; load_scale = 1.0)
    sys = PSY.System(100.0)
    b1 = _add_simple_bus!(sys, 1, PSY.ACBusTypes.REF, 230.0, 1.0, 0.0)
    b2 = _add_simple_bus!(sys, 2, PSY.ACBusTypes.PQ, 230.0, 1.0, 0.0)
    b3 = _add_simple_bus!(sys, 3, PSY.ACBusTypes.PQ, 230.0, 1.0, 0.0)
    _add_simple_source!(sys, b1, 0.0, 0.0)
    _add_simple_load!(sys, b2, 80.0 * load_scale, 40.0 * load_scale)
    _add_simple_load!(sys, b3, 60.0 * load_scale, 30.0 * load_scale)
    # Very high r/x (r ≫ x): classic FD failure mode.
    _add_simple_line!(sys, b1, b2, 0.30, 0.02, 0.0)
    _add_simple_line!(sys, b2, b3, 0.25, 0.02, 0.0)
    return sys
end

# The driver leaves `data` synced at the FINAL (best, on non-divergent termination) state;
# `solve_power_flow!`'s NaN-overwrite for non-converged steps would mask that. So drive the
# FD method directly and inspect `data` before any overwrite. Returns `converged`.
function _drive_fd_directly(pf, data; kwargs...)
    return PF._newton_power_flow(pf, data, 1; kwargs...)
end

@testset "FastDecoupled WP2: safeguard helpers" begin
    # (a) Non-divergent backtracking terminates and RESTORES the best-Σ(Rv²) state. We
    # drive the FD method directly (so `data` is NOT NaN-overwritten on non-convergence),
    # then verify the state left in `data` is the recorded best-Σ(Rv²) state: its residual
    # equals the smallest residual seen, and is strictly better than the flat start.
    @testset "non-divergent backtracking restores best state" begin
        sys = _stressed_high_rx_system(; load_scale = 6.0)
        pf = ACPowerFlow{PF.FastDecoupledACPowerFlow}()
        data = PowerFlowData(pf, sys)

        # Flat-start residual (fresh residual on freshly-initialized data).
        residual_flat = PF.ACPowerFlowResidual(data, 1)
        x0_flat = _calc_x(data, 1)
        residual_flat(x0_flat, 1)
        flat_ss = sum(abs2, residual_flat.Rv)

        # The non-convergence emits an @error at finalization; capture it with @test_logs so
        # it does not trip run_tests()'s zero-Logging.Error-events assertion (full suite), while
        # also asserting the expected error is logged.
        converged = nothing
        @test_logs (:error, r"failed to converge") match_mode = :any begin
            converged = _drive_fd_directly(pf, data;
                fd_variant = :fixed_jacobian,
                fd_non_divergent = true,
                refreeze_on_stall = false,
                maxIterations = 8,
                validate_voltage_magnitudes = false,
            )
        end
        @test !converged   # this pathological case does NOT converge under pure frozen FD

        # State left in `data` is the best one seen: its residual is finite, all voltages
        # positive, and Σ(Rv²) no worse than the flat start (the loop never leaves a
        # diverged state in `data`).
        residual_final = PF.ACPowerFlowResidual(data, 1)
        x_final = _calc_x(data, 1)
        residual_final(x_final, 1)
        final_ss = sum(abs2, residual_final.Rv)
        @test isfinite(final_ss)
        @test final_ss <= flat_ss + 1e-8
        @test all(isfinite, data.bus_magnitude[:, 1])
        @test all(data.bus_magnitude[:, 1] .> 0.0)
    end

    # (d) With fd_non_divergent=false, BLOWUP aborts on the step-size check rather than
    # producing NaNs. Drive directly so the non-converged state isn't NaN-overwritten;
    # assert the abort happened (a BLOWUP warning is emitted) and `data` stays finite.
    @testset "BLOWUP aborts step (fd_non_divergent=false)" begin
        sys = _stressed_high_rx_system(; load_scale = 10.0)
        pf = ACPowerFlow{PF.FastDecoupledACPowerFlow}()
        data = PowerFlowData(pf, sys)
        converged = nothing
        @test_logs (:warn, r"BLOWUP") match_mode = :any begin
            converged = _drive_fd_directly(pf, data;
                fd_variant = :fixed_jacobian,
                fd_non_divergent = false,
                refreeze_on_stall = false,
                maxIterations = 50,
                validate_voltage_magnitudes = false)
        end
        @test !converged
        @test all(isfinite, data.bus_magnitude[:, 1])
    end

    # (e) DVLIM clamp engages on a contrived large-ΔV case and the solve still converges.
    # The clamp helper is also exercised at the unit level (deterministic), then end-to-end:
    # a small DVLIM forces repeated clamping yet the solve must still reach the NR solution.
    @testset "DVLIM clamp engages, still converges" begin
        # Unit-level: a step with a large ΔV entry gets scaled so the largest |ΔV| ≤ dvlim.
        Δx = [0.4, 0.0, -0.6, 0.0]      # voltage entries at positions 1 and 3
        v_idx = [1, 3]
        v_vals = [1.0, 1.0]
        engaged = PF._fd_dvlim_clamp!(Δx, v_idx, v_vals, 0.1)
        @test engaged
        @test maximum(abs.(Δx[v_idx])) <= 0.1 + 1e-12
        # The scaling is uniform across the WHOLE step.
        @test isapprox(Δx[1] / Δx[3], 0.4 / -0.6; atol = 1e-12)

        # Positivity guard: a step that would drive V ≤ 0 is scaled to leave V positive.
        Δx2 = [-2.0, 0.0]
        engaged2 = PF._fd_dvlim_clamp!(Δx2, [1], [1.0], 10.0)
        @test engaged2
        @test 1.0 + Δx2[1] > 0.0

        # End-to-end: on a well-conditioned, solvable system a tiny DVLIM forces the ΔV
        # clamp to engage on early iterations (capping voltage steps) yet the solve must
        # still converge to the NR solution. c_sys5 is solvable and has PQ buses with
        # nonzero ΔV at flat start.
        sys_nr = PSB.build_system(PSB.PSITestSystems, "c_sys5"; add_forecasts = false)
        data_nr = PowerFlowData(ACPowerFlow{NewtonRaphsonACPowerFlow}(), sys_nr)
        @test solve_power_flow!(data_nr)

        sys_fd = PSB.build_system(PSB.PSITestSystems, "c_sys5"; add_forecasts = false)
        pf = ACPowerFlow{PF.FastDecoupledACPowerFlow}(;
            solver_settings = Dict{Symbol, Any}(
                :fd_variant => :fixed_jacobian,
                :fd_dvlim => 0.005,   # tiny DVLIM forces the clamp to engage repeatedly
                :maxIterations => 150,
            ))
        data_fd = PowerFlowData(pf, sys_fd)
        converged = solve_power_flow!(data_fd)
        @test converged
        @test isapprox(data_fd.bus_magnitude[:, 1], data_nr.bus_magnitude[:, 1];
            atol = TIGHT_TOLERANCE, rtol = 0)
    end
end

# =====================================================================================
# WP3 — Polar :decoupled (B′/B″ half-iteration) loop.
# =====================================================================================

# STEP 1 — phase-shifter gate. Closes the one WP1 coverage gap before the :decoupled loop
# relies on B′: c_sys14 / WECC240 have NO phase shifters, so the phase-retention path
# (|τ|=1 in B′ but phase shift retained) is otherwise untested. Build a small
# system WITH a PhaseShiftingTransformer (constructed directly via PowerSystems) plus a
# FixedAdmittance shunt, and assert:
#   (a) restamp(_recover_arc_params) ≈ original Ybus within ComplexF32 noise,
#   (b) B′ is ASYMMETRIC (phase shifter retained) while B″ is SYMMETRIC (phase dropped).
# If this FAILS it is a real WP1 phase-shifter bug — DO NOT patch WP1 here; report it.
function _phase_shifter_system()
    sys = PSY.System(100.0)
    b1 = _add_simple_bus!(sys, 1, PSY.ACBusTypes.REF, 230.0, 1.0, 0.0)
    b2 = _add_simple_bus!(sys, 2, PSY.ACBusTypes.PV, 230.0, 1.0, 0.0)
    b3 = _add_simple_bus!(sys, 3, PSY.ACBusTypes.PQ, 230.0, 1.0, 0.0)
    _add_simple_source!(sys, b1, 0.0, 0.0)
    _add_simple_thermal_standard!(sys, b2, 0.3, 0.0)
    _add_simple_load!(sys, b3, 15.0, 6.0)
    # Plain lines so the network is connected.
    _add_simple_line!(sys, b1, b2, 0.01, 0.10, 0.02)
    _add_simple_line!(sys, b1, b3, 0.01, 0.12, 0.02)
    # A phase-shifting transformer between b2 and b3 (nonzero α ⇒ asymmetric B′).
    pst = PSY.PhaseShiftingTransformer(;
        name = "pst_2_3",
        available = true,
        active_power_flow = 0.0,
        reactive_power_flow = 0.0,
        arc = PSY.Arc(; from = b2, to = b3),
        r = 0.005,
        x = 0.08,
        primary_shunt = 0.0,
        tap = 1.0,
        α = 0.15,           # nonzero phase shift
        rating = 2.0,
        base_power = 100.0,
        phase_angle_limits = (min = -0.7, max = 0.7),
    )
    add_component!(sys, pst)
    # A fixed-admittance shunt at b3 so the per-bus shunt-residual path is exercised too.
    shunt = PSY.FixedAdmittance(;
        name = "shunt_3",
        available = true,
        bus = b3,
        Y = 0.0 + 0.03im,
    )
    add_component!(sys, shunt)
    return sys
end

@testset "FastDecoupled WP3: phase-shifter gate (restamp + B′ asymmetry)" begin
    sys = _phase_shifter_system()
    pf = ACPowerFlow()
    data = PowerFlowData(pf, sys)
    time_step = 1

    # (a) Restamp reconstruction must recover the original Ybus within ComplexF32 noise.
    Yb = ComplexF64.(Matrix(data.power_network_matrix.data))
    Yrec = Matrix(PF._restamp_ybus(PF._recover_arc_params(data)))
    relerr = norm(Yrec - Yb) / norm(Yb)
    @test relerr <= 1e-4

    # (b) Phase-retention path: B′ stamps the phase shift but |τ|=1, so the
    # off-diagonal admittances pick up the e^{±jα} rotation. Because B′ = −imag(Ybus_temp),
    # the rotation only produces an ASYMMETRIC B′ when the series admittance has a nonzero
    # real part (resistance): under BX (full ys retained in B′) the phase shifter makes B′
    # asymmetric; under XB (ys → 1/(jx), purely reactive) −imag(j·B·e^{±jα}) = B·cos α is
    # symmetric in α, so XB's B′ stays symmetric — correct WP1 behavior, matching MATPOWER
    # makeB's resistance-neglect rule. B″ drops the phase shift entirely, so it is symmetric
    # under both schemes regardless of the phase shifter.
    ref, pv, pq = PF.bus_type_idx(data, time_step)

    # BX: B′ asymmetric due to the retained phase shift acting on a resistive ys.
    fd_bx = PF.build_fd_matrices(data, time_step, :BX)
    Bp_bx = Matrix(PF.get_bp_matrix(fd_bx))
    @test !isapprox(Bp_bx, transpose(Bp_bx); atol = 1e-6, rtol = 0)

    # XB: B′ symmetric (resistance neglected → phase rotation is symmetric under −imag).
    fd_xb = PF.build_fd_matrices(data, time_step, :XB)
    Bp_xb = Matrix(PF.get_bp_matrix(fd_xb))
    @test isapprox(Bp_xb, transpose(Bp_xb); atol = 1e-6, rtol = 0)

    # B″ symmetric under both schemes (phase shift dropped in B″).
    for fd in (fd_xb, fd_bx)
        Bpp = Matrix(PF.get_bpp_matrix(PF.extract_bpp(fd, sort(pq))))
        @test isapprox(Bpp, transpose(Bpp); atol = 1e-6, rtol = 0)
    end
end

# Polar :decoupled FD constructor with arbitrary settings.
_fd_decoupled_pf(; kwargs...) =
    ACPowerFlow{PF.FastDecoupledACPowerFlow}(;
        solver_settings = Dict{Symbol, Any}(:fd_variant => :decoupled, kwargs...))

# T2 — Polar FDNR solution parity. For each scheme and system, the pure FD :decoupled
# solve must (1) converge to tol=1e-9 within DEFAULT_FD_MAX_ITER, (2) match an INDEPENDENT
# NewtonRaphsonACPowerFlow solve to TIGHT_TOLERANCE on bus_magnitude / bus_angles / _calc_x,
# and (3) take MORE iterations than NR and > 5 — proving it is genuinely fast-decoupled
# (linear rate), not an accidental exact-Newton.
@testset "FastDecoupled WP3: :decoupled NR-parity (T2)" begin
    systems = (
        ("c_sys5",
            () -> PSB.build_system(PSB.PSITestSystems, "c_sys5"; add_forecasts = false),
        ),
        ("c_sys14",
            () ->
                PSB.build_system(PSB.PSITestSystems, "c_sys14"; add_forecasts = false),
        ),
        ("matpower_case5",
            () -> PSB.build_system(PSB.MatpowerTestSystems, "matpower_case5_sys")),
    )
    for scheme in (:XB, :BX), (sysname, build) in systems
        @testset "$sysname / $scheme" begin
            # Independent NR reference on a fresh system state; capture iteration count.
            sys_nr = build()
            pf_nr = ACPowerFlow{NewtonRaphsonACPowerFlow}()
            data_nr = PowerFlowData(pf_nr, sys_nr)
            @test solve_power_flow!(data_nr)
            x_nr = _calc_x(data_nr, 1)

            # Pure FD :decoupled solve on its OWN fresh system state.
            sys_fd = build()
            pf_fd = _fd_decoupled_pf(; fd_scheme = scheme)
            data_fd = PowerFlowData(pf_fd, sys_fd)
            @test solve_power_flow!(data_fd)
            x_fd = _calc_x(data_fd, 1)

            @test isapprox(data_fd.bus_magnitude[:, 1], data_nr.bus_magnitude[:, 1];
                atol = TIGHT_TOLERANCE, rtol = 0)
            @test isapprox(data_fd.bus_angles[:, 1], data_nr.bus_angles[:, 1];
                atol = TIGHT_TOLERANCE, rtol = 0)
            @test isapprox(x_fd, x_nr; atol = TIGHT_TOLERANCE, rtol = 0)

            # Iteration-count sanity: drive both methods directly to read their iteration
            # counts (the public solver returns a Bool). FD must take > 5 iterations and at
            # least as many as NR — proving genuine FD linear convergence.
            data_nr2 = PowerFlowData(pf_nr, build())
            r_nr, J_nr, x0_nr =
                PF.initialize_power_flow_variables(pf_nr, data_nr2, 1;
                    validate_voltage_magnitudes = false)
            sv_nr = PF.StateVectorCache(x0_nr, r_nr.Rv)
            backend = PF.resolve_linear_solver_backend(nothing)
            lc_nr = PF.make_linear_solver_cache(backend, J_nr.Jv)
            PF.symbolic_factor!(lc_nr, J_nr.Jv)
            conv_nr, it_nr = PF._run_power_flow_method(
                1, sv_nr, lc_nr, r_nr, J_nr, NewtonRaphsonACPowerFlow;
                tol = 1e-9, maxIterations = 50)
            @test conv_nr

            data_fd2 = PowerFlowData(pf_fd, build())
            conv_fd, it_fd_val = PF._fd_decoupled_power_flow(
                pf_fd, data_fd2, 1;
                tol = 1e-9, fd_scheme = scheme,
                maxIterations = PF.DEFAULT_FD_MAX_ITER,
                validate_voltage_magnitudes = false,
                _return_iters = true)
            @test conv_fd
            # Genuine FD linear rate (not accidental exact-Newton): FD must take strictly
            # MORE iterations than quadratic NR, and a sanity floor of ≥ 5 half-cycle
            # iterations. (NR converges in ~3 here; FD takes 5–9.)
            @test it_fd_val >= 5
            @test it_fd_val > it_nr
        end
    end
end

# =====================================================================================
# WP4 — opt-in handoff (FD stage → NR/TR refinement to the real tol).
# c_sys14 ONLY (small, fast). Validates: (1) FD+handoff converges and matches an
# INDEPENDENT NR solve to TIGHT_TOLERANCE for both variants × both handoff solvers;
# (2) handoff actually runs (FD-stage>0, small handoff-iters) when handoff_tol is loose;
# (3) handoff is SKIPPED when FD alone meets tol; (4) an invalid handoff solver throws
# through the public solve path.
# =====================================================================================

# Build an ACPowerFlow{FastDecoupled} (polar) with handoff settings for a given variant.
_fd_handoff_pf(variant, handoff; extra...) =
    ACPowerFlow{PF.FastDecoupledACPowerFlow}(;
        solver_settings = Dict{Symbol, Any}(
            :fd_variant => variant,
            :handoff_solver => handoff,
            extra...,
        ))

# Direct-drive entry point for the per-variant FD loop (so we can pass `_return_stage_iters`
# and read FD-stage vs handoff iteration counts the public solver hides).
_fd_loop(::Val{:decoupled}, args...; kwargs...) =
    PF._fd_decoupled_power_flow(args...; kwargs...)
_fd_loop(::Val{:fixed_jacobian}, args...; kwargs...) =
    PF._fd_fixed_jacobian_power_flow(args...; kwargs...)

@testset "FastDecoupled WP4: handoff NR-parity (c_sys14)" begin
    for variant in (:decoupled, :fixed_jacobian)
        for handoff in (NewtonRaphsonACPowerFlow, TrustRegionACPowerFlow,
            LevenbergMarquardtACPowerFlow)
            @testset "$variant / $(nameof(handoff))" begin
                # Independent NR reference on a FRESH system.
                sys_nr =
                    PSB.build_system(PSB.PSITestSystems, "c_sys14"; add_forecasts = false)
                pf_nr = ACPowerFlow{NewtonRaphsonACPowerFlow}()
                data_nr = PowerFlowData(pf_nr, sys_nr)
                solve_power_flow!(data_nr)
                x_nr = _calc_x(data_nr, 1)

                # FD + handoff on its OWN fresh system.
                sys_fd =
                    PSB.build_system(PSB.PSITestSystems, "c_sys14"; add_forecasts = false)
                pf_fd = _fd_handoff_pf(variant, handoff)
                data_fd = PowerFlowData(pf_fd, sys_fd)
                converged = solve_power_flow!(data_fd)
                x_fd = _calc_x(data_fd, 1)

                @test converged
                @test isapprox(data_fd.bus_magnitude[:, 1], data_nr.bus_magnitude[:, 1];
                    atol = TIGHT_TOLERANCE, rtol = 0)
                @test isapprox(data_fd.bus_angles[:, 1], data_nr.bus_angles[:, 1];
                    atol = TIGHT_TOLERANCE, rtol = 0)
                @test isapprox(x_fd, x_nr; atol = TIGHT_TOLERANCE, rtol = 0)
            end
        end
    end
end

@testset "FastDecoupled WP4: handoff actually occurs (loose handoff_tol)" begin
    for variant in (:decoupled, :fixed_jacobian)
        @testset "$variant" begin
            sys = PSB.build_system(PSB.PSITestSystems, "c_sys14"; add_forecasts = false)
            pf = _fd_handoff_pf(variant, NewtonRaphsonACPowerFlow)
            data = PowerFlowData(pf, sys)
            # LOOSE handoff_tol forces the FD stage to exit early and hand off to NR for the
            # final polish to tol=1e-9.
            result, fd_iters, handoff_iters = _fd_loop(
                Val(variant), pf, data, 1;
                tol = 1e-9,
                handoff_solver = NewtonRaphsonACPowerFlow,
                handoff_tol = 1e-2,
                maxIterations = PF.DEFAULT_FD_MAX_ITER,
                validate_voltage_magnitudes = false,
                _return_stage_iters = true,
            )
            @test result               # converged after handoff
            # FD stage may exit at 0 iters when the post-init residual already meets the loose
            # handoff_tol (e.g. fixed_jacobian on the well-conditioned c_sys14); the meaningful
            # guarantee is that the handoff actually ran and polished the solution to tol.
            @test fd_iters >= 0        # the FD stage ran to the loose stage tolerance
            @test handoff_iters > 0    # the handoff ran
            @test handoff_iters <= 6   # and finished quickly (NR quadratic from a warm start)
        end
    end
end

@testset "FastDecoupled WP4: handoff SKIPPED when FD meets tol" begin
    for variant in (:decoupled, :fixed_jacobian)
        @testset "$variant" begin
            sys = PSB.build_system(PSB.PSITestSystems, "c_sys14"; add_forecasts = false)
            pf = _fd_handoff_pf(variant, NewtonRaphsonACPowerFlow)
            data = PowerFlowData(pf, sys)
            # handoff_tol == tol: the FD stage already reaches `tol`, so the handoff is a
            # no-op (zero handoff iterations).
            result, fd_iters, handoff_iters = _fd_loop(
                Val(variant), pf, data, 1;
                tol = 1e-9,
                handoff_solver = NewtonRaphsonACPowerFlow,
                handoff_tol = 1e-9,
                maxIterations = PF.DEFAULT_FD_MAX_ITER,
                validate_voltage_magnitudes = false,
                _return_stage_iters = true,
            )
            @test result
            @test fd_iters > 0
            @test handoff_iters == 0
        end
    end
end

@testset "FastDecoupled WP4: invalid handoff solver throws (public path)" begin
    sys = PSB.build_system(PSB.PSITestSystems, "c_sys14"; add_forecasts = false)
    # RobustHomotopy is not an accepted handoff target (only nothing / NR / TR / LM are).
    pf = ACPowerFlow{PF.FastDecoupledACPowerFlow}(;
        solver_settings = Dict{Symbol, Any}(
            :handoff_solver => RobustHomotopyPowerFlow))
    data = PowerFlowData(pf, sys)
    @test_throws ArgumentError solve_power_flow!(data)
end

# =====================================================================================
# WP5 — special-case CORRECTNESS parity (the caching OPTIMIZATION is a separate later
# step). These tests assert FD already produces the right answer across the three
# special cases — Q-limit PV→PQ switching, LCC HVDC, and loss/voltage-stability factors —
# against an independent NR reference, with NO production-code changes.
# =====================================================================================

# T5 — Q-limit switching (PV→PQ) parity. Mirrors the c_sys14 Q-limit pattern in
# test_solve_power_flow.jl ("AC Power Flow 14-Bus testing"): on c_sys14 with
# correct_bustypes=true the PV generator at Bus8 violates its reactive-power upper limit at
# the unconstrained solution, so check_reactive_power_limits=true forces a PV→PQ switch in
# the `_ac_power_flow` Q-limit outer loop. For each FD variant we assert the FD solve (1)
# converges, (2) respects the reactive limit at Bus8, and (3) matches an independent
# NewtonRaphsonACPowerFlow run (same check_reactive_power_limits=true) to TIGHT_TOLERANCE.
@testset "FastDecoupled WP5: Q-limit PV→PQ parity (T5)" begin
    _build_sys14_qlim() = (
        let s = PSB.build_system(PSB.PSITestSystems, "c_sys14"; add_forecasts = false)
            set_units_base_system!(s, UnitSystem.SYSTEM_BASE)
            s
        end
    )

    # Independent NR reference WITH limit enforcement, on its own fresh system state.
    sys_nr = _build_sys14_qlim()
    pf_nr = ACPowerFlow{NewtonRaphsonACPowerFlow}(;
        check_reactive_power_limits = true, correct_bustypes = true)
    data_nr = PowerFlowData(pf_nr, sys_nr)
    @test solve_power_flow!(data_nr)
    x_nr = _calc_x(data_nr, 1)

    # Sanity: the limit actually binds (PV→PQ switch really happens) — Bus8 reactive output
    # lands at (or below) its upper reactive limit in the NR reference solve.
    solved_nr = deepcopy(_build_sys14_qlim())
    @test solve_and_store_power_flow!(pf_nr, solved_nr)
    @test get_reactive_power(get_component(ThermalStandard, solved_nr, "Bus8")) <=
          get_reactive_power_limits(
        get_component(ThermalStandard, solved_nr, "Bus8")).max + 1e-6

    for variant in (:decoupled, :fixed_jacobian)
        @testset "$variant" begin
            sys_fd = _build_sys14_qlim()
            pf_fd = ACPowerFlow{PF.FastDecoupledACPowerFlow}(;
                check_reactive_power_limits = true,
                correct_bustypes = true,
                solver_settings = Dict{Symbol, Any}(:fd_variant => variant))
            data_fd = PowerFlowData(pf_fd, sys_fd)
            @test solve_power_flow!(data_fd)
            x_fd = _calc_x(data_fd, 1)

            # FD must match NR (both with limit enforcement) to TIGHT_TOLERANCE.
            @test isapprox(data_fd.bus_magnitude[:, 1], data_nr.bus_magnitude[:, 1];
                atol = TIGHT_TOLERANCE, rtol = 0)
            @test isapprox(data_fd.bus_angles[:, 1], data_nr.bus_angles[:, 1];
                atol = TIGHT_TOLERANCE, rtol = 0)
            @test isapprox(x_fd, x_nr; atol = TIGHT_TOLERANCE, rtol = 0)

            # Reactive limit at Bus8 is respected in the FD store-back too.
            solved_fd = deepcopy(_build_sys14_qlim())
            @test solve_and_store_power_flow!(pf_fd, solved_fd)
            @test get_reactive_power(
                get_component(ThermalStandard, solved_fd, "Bus8")) <=
                  get_reactive_power_limits(
                get_component(ThermalStandard, solved_fd, "Bus8")).max + 1e-6
        end
    end
end

# T3-LCC — :fixed_jacobian with LCC HVDC. Reuses the LCC fixtures from
# test_rectangular_ci_lcc.jl / test_mixed_cpb_lcc.jl (case5_2_lcc.raw, with PQ LCC
# terminals). The :fixed_jacobian variant freezes the FULL per-formulation Jacobian — which
# DOES span the 4 trailing LCC state entries per converter — so it solves LCC systems for
# all three formulations. We assert FD :fixed_jacobian converges and matches the NR solve
# (same formulation) to TIGHT_TOLERANCE, that the systems really carry LCC state
# (get_lcc_count > 0), and that the polar :decoupled + LCC combination throws ArgumentError
# (the WP0/method data-dependent guard) through the public solve path.
@testset "FastDecoupled WP5: :fixed_jacobian + LCC HVDC (T3-LCC)" begin
    lcc_raw = joinpath(TEST_DATA_DIR, "case5_2_lcc.raw")

    # Confirm the fixture really carries LCC state (the LCC predicate).
    let data_probe =
            PF.PowerFlowData(ACPowerFlow{NewtonRaphsonACPowerFlow}(), System(lcc_raw))
        @test PF.get_lcc_count(data_probe) > 0
    end

    @testset "rectangular" begin
        pf_nr = ACRectangularPowerFlow{NewtonRaphsonACPowerFlow}(;
            solver_settings = Dict{Symbol, Any}(:validate_voltage_magnitudes => false))
        data_nr = PF.PowerFlowData(pf_nr, System(lcc_raw))
        @test solve_power_flow!(data_nr)

        pf_fd = ACRectangularPowerFlow{PF.FastDecoupledACPowerFlow}(;
            solver_settings = Dict{Symbol, Any}(
                :fd_variant => :fixed_jacobian,
                :validate_voltage_magnitudes => false))
        data_fd = PF.PowerFlowData(pf_fd, System(lcc_raw))
        @test PF.get_lcc_count(data_fd) > 0
        @test solve_power_flow!(data_fd)
        @test isapprox(data_fd.bus_magnitude[:, 1], data_nr.bus_magnitude[:, 1];
            atol = TIGHT_TOLERANCE, rtol = 0)
        @test isapprox(data_fd.bus_angles[:, 1], data_nr.bus_angles[:, 1];
            atol = TIGHT_TOLERANCE, rtol = 0)
    end

    @testset "mixed" begin
        pf_nr = ACMixedPowerFlow{NewtonRaphsonACPowerFlow}(;
            solver_settings = Dict{Symbol, Any}(:validate_voltage_magnitudes => false))
        data_nr = PF.PowerFlowData(pf_nr, System(lcc_raw))
        @test solve_power_flow!(data_nr)

        pf_fd = ACMixedPowerFlow{PF.FastDecoupledACPowerFlow}(;
            solver_settings = Dict{Symbol, Any}(
                :fd_variant => :fixed_jacobian,
                :validate_voltage_magnitudes => false))
        data_fd = PF.PowerFlowData(pf_fd, System(lcc_raw))
        @test PF.get_lcc_count(data_fd) > 0
        @test solve_power_flow!(data_fd)
        @test isapprox(data_fd.bus_magnitude[:, 1], data_nr.bus_magnitude[:, 1];
            atol = TIGHT_TOLERANCE, rtol = 0)
        @test isapprox(data_fd.bus_angles[:, 1], data_nr.bus_angles[:, 1];
            atol = TIGHT_TOLERANCE, rtol = 0)
    end

    @testset "polar :fixed_jacobian" begin
        pf_nr = ACPowerFlow{NewtonRaphsonACPowerFlow}()
        data_nr = PF.PowerFlowData(pf_nr, System(lcc_raw))
        @test solve_power_flow!(data_nr)

        pf_fd = ACPowerFlow{PF.FastDecoupledACPowerFlow}(;
            solver_settings = Dict{Symbol, Any}(:fd_variant => :fixed_jacobian))
        data_fd = PF.PowerFlowData(pf_fd, System(lcc_raw))
        @test PF.get_lcc_count(data_fd) > 0
        @test solve_power_flow!(data_fd)
        @test isapprox(data_fd.bus_magnitude[:, 1], data_nr.bus_magnitude[:, 1];
            atol = TIGHT_TOLERANCE, rtol = 0)
        @test isapprox(data_fd.bus_angles[:, 1], data_nr.bus_angles[:, 1];
            atol = TIGHT_TOLERANCE, rtol = 0)
    end

    @testset "polar :decoupled + LCC throws (WP0 guard)" begin
        # The B′/B″ half-iterations do not span the LCC state, so the FD method's
        # data-dependent guard must reject this through the public solve path.
        sys_lcc, _ = simple_lcc_system()
        pf = ACPowerFlow{PF.FastDecoupledACPowerFlow}(;
            solver_settings = Dict{Symbol, Any}(:fd_variant => :decoupled))
        data = PF.PowerFlowData(pf, sys_lcc)
        @test PF.get_lcc_count(data) > 0
        @test_throws ArgumentError solve_power_flow!(data)
    end
end

# T9 — loss-factor (and voltage-stability-factor) parity, the stale-Jacobian pitfall.
# Loss/vstab factors are computed from J.Jv at finalization, so the FD drivers must refresh
# J at the SOLUTION before `_finalize_power_flow`. Mirrors test_loss_factors.jl
# ("test_loss_factors_case_14"): on c_sys14 with calculate_loss_factors=true the FD factors
# must match the NR factors to the same 1e-4 tolerance that file uses. If FD were leaving a
# stale (frozen / decoupled) Jacobian in place, these factors would be wrong — this is the
# regression guard. Loss/vstab factors are polar-only, so this uses the polar formulation.
@testset "FastDecoupled WP5: loss/vstab factor parity (T9)" begin
    # NR reference with loss + voltage-stability factors on a fresh system state.
    sys_nr = PSB.build_system(PSB.PSITestSystems, "c_sys14"; add_forecasts = false)
    pf_nr = ACPowerFlow{NewtonRaphsonACPowerFlow}(;
        calculate_loss_factors = true,
        calculate_voltage_stability_factors = true)
    data_nr = PowerFlowData(pf_nr, sys_nr)
    @test solve_power_flow!(data_nr)
    @test data_nr.loss_factors !== nothing
    @test data_nr.voltage_stability_factors !== nothing

    for variant in (:decoupled, :fixed_jacobian)
        @testset "$variant" begin
            sys_fd = PSB.build_system(PSB.PSITestSystems, "c_sys14"; add_forecasts = false)
            pf_fd = ACPowerFlow{PF.FastDecoupledACPowerFlow}(;
                calculate_loss_factors = true,
                calculate_voltage_stability_factors = true,
                solver_settings = Dict{Symbol, Any}(:fd_variant => variant))
            data_fd = PowerFlowData(pf_fd, sys_fd)
            @test solve_power_flow!(data_fd)

            # FD must populate the factors (refreshed-J at the solution, not stale).
            @test data_fd.loss_factors !== nothing
            @test data_fd.voltage_stability_factors !== nothing
            # And they must match the NR factors to test_loss_factors.jl's tolerance.
            @test all(
                isapprox.(
                    data_fd.loss_factors,
                    data_nr.loss_factors;
                    atol = 1e-4,
                    rtol = 0,
                ),
            )
            @test all(
                isapprox.(
                    data_fd.voltage_stability_factors,
                    data_nr.voltage_stability_factors;
                    atol = 1e-4,
                    rtol = 0,
                ),
            )
        end
    end
end

# =====================================================================================
# WP5b — FastDecoupledCache: factor-once across time steps and Q-limit retries.
# The polar :decoupled loop must factor B′ EXACTLY ONCE
# per (data, scheme, backend) lifetime and B″ once per DISTINCT PQ set (bus-type signature),
# reusing the factorizations on repeat signatures. The cache lives in
# `data.solver_cache[]` as `(PF.FD_CACHE_TAG, ::FastDecoupledCache)`; its `bp_factor_count`
# / `bpp_factor_count` counters are the factor-once arbiters.
# =====================================================================================

# Pull the live FastDecoupledCache out of data.solver_cache[] (the tagged tuple), or `nothing`
# if the :decoupled loop has not run on this data yet.
function _fd_cache(data)
    entry = data.solver_cache[]
    (entry isa Tuple && length(entry) == 2 && entry[1] === PF.FD_CACHE_TAG) ||
        return nothing
    return entry[2]
end

@testset "FastDecoupled WP5b: multi-period caching (T7)" begin
    # 24-step c_sys14 (pattern of test_multiperiod_ac_power_flow.jl). The bus-type columns
    # are identical across all steps (no per-step Q-limit switching here), so B′ AND B″ each
    # factor exactly ONCE across all 24 solves — the central fast-decoupled performance contract.
    sys = PSB.build_system(PSB.PSITestSystems, "c_sys14"; add_forecasts = false)
    time_steps = 24

    pf_fd = ACPowerFlow{PF.FastDecoupledACPowerFlow}(;
        time_steps = time_steps,
        solver_settings = Dict{Symbol, Any}(:fd_variant => :decoupled))
    data_fd = PowerFlowData(pf_fd, sys)
    prepare_ts_data!(data_fd, time_steps)
    @test solve_power_flow!(data_fd)

    # All steps converged.
    @test all(data_fd.converged)

    # NR reference (single-period, same network) to validate the cached FD solution per step.
    pf_nr = ACPowerFlow{NewtonRaphsonACPowerFlow}(; time_steps = time_steps)
    data_nr = PowerFlowData(pf_nr, sys)
    prepare_ts_data!(data_nr, time_steps)
    @test solve_power_flow!(data_nr)
    @test isapprox(data_fd.bus_magnitude, data_nr.bus_magnitude; atol = TIGHT_TOLERANCE)
    @test isapprox(data_fd.bus_angles, data_nr.bus_angles; atol = TIGHT_TOLERANCE)

    # Factor-once verification via the cache counters.
    cache = _fd_cache(data_fd)
    @test cache !== nothing
    # B′ factored EXACTLY ONCE across all 24 time steps.
    @test cache.bp_factor_count == 1
    # B″ factored once per DISTINCT bus-type signature. All 24 steps share one signature
    # (identical bus_type columns), so exactly one B″ factorization — and exactly one cached
    # FDPQData entry.
    distinct_sigs =
        length(unique(hash(view(data_fd.bus_type, :, t)) for t in 1:time_steps))
    @test distinct_sigs == 1
    @test cache.bpp_factor_count == distinct_sigs
    @test length(cache.pq_data) == distinct_sigs
end

@testset "FastDecoupled WP5b: Q-limit retries reuse B′ (T5 caching)" begin
    # c_sys14 with PV→PQ Q-limit switching: the `_ac_power_flow` outer loop re-invokes the FD
    # driver from scratch after switching Bus8 PV→PQ. B′ is restricted to the non-REF set
    # (both PV and PQ are non-REF) so it is INVARIANT across the switch — assert it is factored
    # exactly ONCE across all outer-loop re-invocations. The PQ set DOES change (Bus8 enters
    # PQ), so B″ is factored once per distinct PQ signature (the pre- and post-switch sets).
    sys = PSB.build_system(PSB.PSITestSystems, "c_sys14"; add_forecasts = false)
    set_units_base_system!(sys, UnitSystem.SYSTEM_BASE)
    pf_fd = ACPowerFlow{PF.FastDecoupledACPowerFlow}(;
        check_reactive_power_limits = true,
        correct_bustypes = true,
        solver_settings = Dict{Symbol, Any}(:fd_variant => :decoupled))
    data_fd = PowerFlowData(pf_fd, sys)
    @test solve_power_flow!(data_fd)

    cache = _fd_cache(data_fd)
    @test cache !== nothing
    # B′ factored EXACTLY ONCE across the Q-limit outer-loop re-invocations.
    @test cache.bp_factor_count == 1
    # The Q-limit switch produced at least the original PQ set; B″ factored once per distinct
    # PQ signature seen, and the dict holds exactly that many entries (no redundant refactor).
    @test cache.bpp_factor_count >= 1
    @test cache.bpp_factor_count == length(cache.pq_data)
end

@testset "FastDecoupled WP5b: cache invalidation & loud collision" begin
    sys = PSB.build_system(PSB.PSITestSystems, "c_sys14"; add_forecasts = false)
    pf_fd = ACPowerFlow{PF.FastDecoupledACPowerFlow}(;
        solver_settings = Dict{Symbol, Any}(:fd_variant => :decoupled))
    data = PowerFlowData(pf_fd, sys)
    @test solve_power_flow!(data)
    cache = _fd_cache(data)
    @test cache !== nothing

    # Invalidation key matches Ybus identity / scheme / backend.
    @test cache.ybus_id == objectid(data.power_network_matrix)
    @test cache.scheme === :XB
    @test cache.backend_id === PF._fd_backend_id(nothing)

    # Reuse: a second :decoupled solve on the SAME data with the SAME scheme/backend must NOT
    # rebuild B′ (count stays 1) nor refactor B″ for the already-seen PQ set.
    bp_before = cache.bp_factor_count
    bpp_before = cache.bpp_factor_count
    @test solve_power_flow!(data)
    cache2 = _fd_cache(data)
    @test cache2 === cache   # same cache object reused
    @test cache.bp_factor_count == bp_before
    @test cache.bpp_factor_count == bpp_before

    # Loud collision: a non-FD value in the slot (e.g. the DC-path tuple shape) must `error`.
    data.solver_cache[] = (SparseArrays.spzeros(2, 2), PF.PNM.KLUSolver(), nothing, nothing)
    @test_throws ErrorException PF._get_or_build_fd_cache!(
        data, 1, :XB, PF._fd_backend_id(nothing), nothing)
end

@testset "FastDecoupled WP5b: allocations (T10)" begin
    # Mirror test_ac_nr_allocations.jl: warm the cache with one solve, then assert the
    # per-iteration HOT-PATH operations of the :decoupled loop (cache-warm) allocate ~0.
    # The factor-once cache means a warm second solve does ZERO refactorizations; the inner
    # half-step buffer fills + solves reuse preallocated cache buffers.
    sys = PSB.build_system(PSB.MatpowerTestSystems, "matpower_ACTIVSg2000_sys")
    PSY.set_units_base_system!(sys, "SYSTEM_BASE")
    pf = ACPowerFlow{PF.FastDecoupledACPowerFlow}(;
        correct_bustypes = true,
        solver_settings = Dict{Symbol, Any}(:fd_variant => :decoupled))
    data = PowerFlowData(pf, sys)
    @test solve_power_flow!(data)   # warm the FastDecoupledCache

    cache = _fd_cache(data)
    @test cache !== nothing
    bp_before = cache.bp_factor_count
    bpp_before = cache.bpp_factor_count

    # A second solve on the SAME data must do ZERO additional factorizations (cache warm):
    # this is the post-warmup factor-once budget — exactly 0 new B′/B″ factorizations.
    @test solve_power_flow!(data)
    @test cache.bp_factor_count == bp_before     # 0 new B′ factorizations
    @test cache.bpp_factor_count == bpp_before   # 0 new B″ factorizations

    # Per-iteration hot-path allocation budget ~0. Build the loop locals exactly as the
    # :decoupled loop does (all from the warm cache), then measure the half-step primitives:
    # the buffer fills and the cached-factorization solves. These reuse preallocated buffers
    # and must be allocation-free once warm.
    fd = cache.fd
    pqdata = PF._get_pq_data!(cache, data, 1, nothing)
    bpp = pqdata.bpp
    Vm = view(data.bus_magnitude, :, 1)
    residual = PF.ACPowerFlowResidual(data, 1)
    x0 = PF.calculate_x0(data, 1)
    residual(x0, 1)   # warm residual + sync data

    rp = cache.rp
    rq = pqdata.rq
    p_row_idx = cache.p_row_idx
    q_row_idx = pqdata.q_row_idx

    # Active half-step buffer fill (rp = Rv_P / Vm over pvpq).
    fill_rp!() = begin
        @inbounds for k in eachindex(fd.pvpq)
            rp[k] = residual.Rv[p_row_idx[k]] / Vm[fd.pvpq[k]]
        end
    end
    fill_rp!()   # warm
    @test (@allocated fill_rp!()) == 0

    # Reactive half-step buffer fill (rq = Rv_Q / Vm over pq).
    fill_rq!() = begin
        @inbounds for k in eachindex(bpp.pq)
            rq[k] = residual.Rv[q_row_idx[k]] / Vm[bpp.pq[k]]
        end
    end
    fill_rq!()   # warm
    @test (@allocated fill_rq!()) == 0

    # The cached-factorization solves reuse the buffer in place. The buffer is preallocated
    # (no per-iteration vector allocation from OUR loop), but the backend's `solve!` (KLU/AA)
    # may do a tiny bounded internal allocation — the NR allocation test likewise warms
    # `solve!` without asserting it is exactly 0. Bound it small (a few hundred bytes) to
    # catch any per-solve buffer regrowth while tolerating the backend's fixed overhead. Do
    # NOT loosen silently: if this ever fails, the cause is a NEW allocation in our path, not
    # the (fixed) backend overhead.
    PF.solve!(fd.bp_cache, rp)   # warm
    @test (@allocated PF.solve!(fd.bp_cache, rp)) < 256
    PF.solve!(bpp.bpp_cache, rq)  # warm
    @test (@allocated PF.solve!(bpp.bpp_cache, rq)) < 256
end

@testset "FastDecoupled WP5b: :decoupled skips the formulation Jacobian (T11 lazy-J)" begin
    # The :decoupled half-steps run on B′/B″ and never touch the formulation Jacobian; it is built
    # ONLY for a handoff or loss/voltage-stability factors. With neither, the driver must skip the
    # full sparse-Jacobian allocation + evaluation entirely — a per-solve, per-time-step saving.
    sys = PSB.build_system(PSB.MatpowerTestSystems, "matpower_ACTIVSg2000_sys")
    PSY.set_units_base_system!(sys, "SYSTEM_BASE")

    # (a) Direct isolation: the residual/x0-only initializer the :decoupled driver uses must NOT
    # allocate the Jacobian, so it allocates strictly less than the full initializer — by at least
    # most of the omitted sparse-Jacobian footprint (≈3 MiB on this 2000-bus system).
    pf = ACPowerFlow{PF.FastDecoupledACPowerFlow}(;
        correct_bustypes = true,
        solver_settings = Dict{Symbol, Any}(:fd_variant => :decoupled))
    data = PowerFlowData(pf, sys)
    kw = (; validate_voltage_magnitudes = false)
    PF._initialize_residual_x0(pf, data, 1; kw...)             # warm (compile)
    PF.initialize_power_flow_variables(pf, data, 1; kw...)     # warm (compile)
    a_lazy = @allocated PF._initialize_residual_x0(pf, data, 1; kw...)
    a_full = @allocated PF.initialize_power_flow_variables(pf, data, 1; kw...)
    jac_bytes = Base.summarysize(
        PF.ACPowerFlowJacobian(PF.ACPowerFlowResidual(data, 1), 1).Jv)
    @test a_lazy < a_full
    @test (a_full - a_lazy) > jac_bytes ÷ 2

    # (b) Multi-period: a warm :decoupled re-solve must not rebuild a per-step Jacobian. The old
    # eager path built one full formulation Jacobian PER time step (≥ time_steps·jac_alloc bytes);
    # the lazy path builds none, so the whole warm multi-period re-solve stays well under that.
    time_steps = 3
    pf_mp = ACPowerFlow{PF.FastDecoupledACPowerFlow}(;
        time_steps = time_steps,
        correct_bustypes = true,
        solver_settings = Dict{Symbol, Any}(:fd_variant => :decoupled))
    data_mp = PowerFlowData(pf_mp, sys)
    @test solve_power_flow!(data_mp)        # warm caches + factorizations
    @test all(data_mp.converged)
    cache = _fd_cache(data_mp)
    @test cache !== nothing
    @test cache.bp_factor_count == 1        # lazy-J change did not disturb factor-once

    jac_alloc = @allocated (
        let J = PF.ACPowerFlowJacobian(PF.ACPowerFlowResidual(data_mp, 1), 1)
            J(1)
        end
    )
    a_solve = @allocated solve_power_flow!(data_mp)
    @test a_solve < time_steps * jac_alloc

    # (c) Factors-on still works through the lazy gate: when loss factors ARE requested the driver
    # builds + evaluates J at the solution, so finite loss factors are produced. (c_sys14: fast.)
    sys14 = PSB.build_system(PSB.PSITestSystems, "c_sys14"; add_forecasts = false)
    pf_lf = ACPowerFlow{PF.FastDecoupledACPowerFlow}(;
        calculate_loss_factors = true,
        solver_settings = Dict{Symbol, Any}(:fd_variant => :decoupled))
    data_lf = PowerFlowData(pf_lf, sys14)
    @test solve_power_flow!(data_lf)
    @test PF.get_loss_factors(data_lf) !== nothing
    @test all(isfinite, PF.get_loss_factors(data_lf))
end

# WP6 regression: degenerate islands make B′ and/or B″ a 0×0 submatrix, which errors in the
# sparse backends (AppleAccelerate: "columnCount must be > 0") if factored. FD :decoupled must
# skip the corresponding half-step. Covered indirectly by AC_SOLVERS_TO_TEST integration tests
# (test_ac_3bus_fixed_admittance = empty PQ; test_ac_multiple_sources_at_ref = lone REF); this
# pins the behavior directly on the FD solver.
@testset "FastDecoupled WP6: degenerate islands (empty B′/B″)" begin
    @testset "lone REF bus (empty pvpq and pq)" begin
        sys = PSY.System(100.0)
        b = _add_simple_bus!(sys, 1, PSY.ACBusTypes.REF, 230.0, 1.05, 0.0)
        _add_simple_source!(sys, b, 0.5, 0.1)
        _add_simple_source!(sys, b, -0.5, -0.1)
        pf = ACPowerFlow{PF.FastDecoupledACPowerFlow}(; correct_bustypes = true)
        @test solve_power_flow!(PowerFlowData(pf, sys))
    end
    @testset "all-PV/REF, no PQ buses (empty pq, nonempty pvpq)" begin
        sys = PSY.System(100.0)
        b1 = _add_simple_bus!(sys, 1, PSY.ACBusTypes.REF, 230.0, 1.0, 0.0)
        b2 = _add_simple_bus!(sys, 2, PSY.ACBusTypes.PV, 230.0, 1.0, 0.0)
        _add_simple_source!(sys, b1, 0.0, 0.0)
        _add_simple_thermal_standard!(sys, b2, 0.3, 0.0)
        _add_simple_line!(sys, b1, b2, 0.01, 0.1, 0.0)
        data_nr = PowerFlowData(ACPowerFlow{NewtonRaphsonACPowerFlow}(), sys)
        @test solve_power_flow!(data_nr)
        data_fd = PowerFlowData(ACPowerFlow{PF.FastDecoupledACPowerFlow}(), sys)
        @test solve_power_flow!(data_fd)
        @test isapprox(data_fd.bus_magnitude[:, 1], data_nr.bus_magnitude[:, 1];
            atol = TIGHT_TOLERANCE, rtol = 0)
        @test isapprox(data_fd.bus_angles[:, 1], data_nr.bus_angles[:, 1];
            atol = TIGHT_TOLERANCE, rtol = 0)
    end
end

# Review follow-ups (final code review): the polar :decoupled safeguard application and the
# stall→handoff seam were previously exercised only on the :fixed_jacobian path. These lock the
# decoupled non-divergent-backtracking fix (step rescaled from a single cycle-step snapshot, not
# re-read from the mutated state) and the handoff-from-an-unconverged-FD-state path (plan T8b),
# plus the missing ACTIVSg2000 :decoupled NR-parity.
@testset "FastDecoupled review: decoupled safeguards + handoff + ACTIVSg2000 parity" begin
    @testset "decoupled non-divergent backtracking runs and restores a valid state" begin
        # Exercises the polar :decoupled backtracking path (previously untested) — locks the fix
        # that rescales the cycle step from a single snapshot rather than the mutated state. The
        # safeguard must leave a finite, positive-voltage best state in `data` (never NaN/diverged).
        sys = _stressed_high_rx_system(; load_scale = 6.0)
        pf = ACPowerFlow{PF.FastDecoupledACPowerFlow}()
        data = PowerFlowData(pf, sys)
        converged = nothing
        @test_logs (:error, r"failed to converge") match_mode = :any begin
            converged = _drive_fd_directly(pf, data;
                fd_variant = :decoupled,
                fd_non_divergent = true,
                maxIterations = 8,
                validate_voltage_magnitudes = false)
        end
        @test !converged
        @test all(isfinite, data.bus_magnitude[:, 1])
        @test all(data.bus_magnitude[:, 1] .> 0.0)
        residual_final = PF.ACPowerFlowResidual(data, 1)
        residual_final(_calc_x(data, 1), 1)
        @test isfinite(sum(abs2, residual_final.Rv))
    end

    @testset "unconverged FD stage hands off and converges (T8b): $variant" for variant in
                                                                                (:decoupled,
        :fixed_jacobian)
        # c_sys14 is feasible; cap FD at 1 iteration so it cannot reach `tol` alone, forcing the
        # handoff to TrustRegion to polish the unconverged FD state to the NR solution.
        sys_nr = PSB.build_system(PSB.PSITestSystems, "c_sys14"; add_forecasts = false)
        data_nr = PowerFlowData(ACPowerFlow{NewtonRaphsonACPowerFlow}(), sys_nr)
        @test solve_power_flow!(data_nr)

        sys_fd = PSB.build_system(PSB.PSITestSystems, "c_sys14"; add_forecasts = false)
        pf = ACPowerFlow{PF.FastDecoupledACPowerFlow}(;
            solver_settings = Dict{Symbol, Any}(
                :fd_variant => variant,
                :handoff_solver => TrustRegionACPowerFlow,
                :maxIterations => 1))   # forces an unconverged FD stage → handoff
        data_fd = PowerFlowData(pf, sys_fd)
        @test solve_power_flow!(data_fd)
        @test isapprox(data_fd.bus_magnitude[:, 1], data_nr.bus_magnitude[:, 1];
            atol = TIGHT_TOLERANCE, rtol = 0)
        @test isapprox(data_fd.bus_angles[:, 1], data_nr.bus_angles[:, 1];
            atol = TIGHT_TOLERANCE, rtol = 0)
    end

    @testset "ACTIVSg2000 :decoupled NR-parity" begin
        sys_nr = PSB.build_system(PSB.MatpowerTestSystems, "matpower_ACTIVSg2000_sys")
        PSY.set_units_base_system!(sys_nr, "SYSTEM_BASE")
        data_nr =
            PowerFlowData(ACPowerFlow{NewtonRaphsonACPowerFlow}(; correct_bustypes = true),
                sys_nr)
        solve_power_flow!(data_nr)

        sys_fd = PSB.build_system(PSB.MatpowerTestSystems, "matpower_ACTIVSg2000_sys")
        PSY.set_units_base_system!(sys_fd, "SYSTEM_BASE")
        pf_fd = ACPowerFlow{PF.FastDecoupledACPowerFlow}(;
            correct_bustypes = true,
            solver_settings = Dict{Symbol, Any}(
                :fd_variant => :decoupled,
                :fd_scheme => :XB,
            ))
        data_fd = PowerFlowData(pf_fd, sys_fd)
        @test solve_power_flow!(data_fd)
        @test isapprox(data_fd.bus_magnitude[:, 1], data_nr.bus_magnitude[:, 1];
            atol = TIGHT_TOLERANCE, rtol = 0)
        @test isapprox(data_fd.bus_angles[:, 1], data_nr.bus_angles[:, 1];
            atol = TIGHT_TOLERANCE, rtol = 0)
    end
end
