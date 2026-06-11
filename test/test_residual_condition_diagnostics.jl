# The Schur-eigenvalue solve and the fold bail-out need only a back-solve, so they
# work on any backend; only κ̂ is KLU-only (reported as `n/a` otherwise). Tests that
# assert a numeric κ̂ pin `linear_solver = "KLU"` for determinism across platforms
# (on Apple the default is AppleAccelerate); the AppleAccelerate path is covered
# explicitly below.
const _KLU_SETTINGS = Dict{Symbol, Any}(:linear_solver => "KLU")

# Build a Schur operator at the flat start of `sys` under `backend` and return its
# smallest eigenvalue alongside the dense ground truth (smallest-magnitude
# eigenvalue of S = A − B·D⁻¹·C, recovered as inv(inv(J)[1:nb, 1:nb])).
function _schur_eig_and_truth(pf, sys; time_step = 1, backend = PNM.KLUSolver())
    data = PowerFlowData(pf, sys)
    residual = PF.ACPowerFlowResidual(data, time_step)
    jac = PF.ACPowerFlowJacobian(residual, time_step)
    x0 = PF.calculate_x0(data, time_step)
    residual(x0, time_step)
    jac(time_step)

    cache = PF.make_linear_solver_cache(backend, jac.Jv)
    PF.symbolic_factor!(cache, jac.Jv)
    PF.numeric_refactor!(cache, jac.Jv)

    n_state = size(jac.Jv, 1)
    n_lcc = size(data.lcc.p_set, 1)
    n_bus = n_state - 4 * n_lcc
    op = PF.SchurInverseOperator(cache, n_bus, Vector{Float64}(undef, n_state))
    λ, converged = PF._schur_min_eigenvalue(op)
    @test converged

    Jinv = inv(Matrix(jac.Jv))
    S = inv(Jinv[1:n_bus, 1:n_bus])
    ev = eigvals(S)
    return λ, ev[argmin(abs.(ev))], n_lcc
end

# Solve under `pf` and return the per-iteration diagnostic log lines.
function _solver_diagnostic_lines(pf, sys)
    data = PowerFlowData(pf, sys)
    tl = Test.TestLogger(; min_level = Logging.Info)
    Logging.with_logger(tl) do
        solve_power_flow!(data)
    end
    return [r.message for r in tl.logs if occursin(r"iter \d+", r.message)]
end

@testset "Schur min-eigenvalue matches dense ground truth (no LCC)" begin
    sys = PSB.build_system(PSB.PSITestSystems, "c_sys14")
    pf = ACPowerFlow{NewtonRaphsonACPowerFlow}(; correct_bustypes = true)
    λ, λ_true, n_lcc = _schur_eig_and_truth(pf, sys)
    @test n_lcc == 0                       # with no LCC, S = J
    @test abs(λ - λ_true) / abs(λ_true) < 1e-6
end

@testset "Schur min-eigenvalue matches dense ground truth (LCC)" begin
    # On an LCC system the Schur complement projects out the converter states;
    # the matvec must still match the dense inv(inv(J)[1:nb, 1:nb]) eigenvalue.
    sys = System(joinpath(TEST_DATA_DIR, "case5_2_lcc.raw"))
    pf = ACPowerFlow{NewtonRaphsonACPowerFlow}()
    λ, λ_true, n_lcc = _schur_eig_and_truth(pf, sys)
    @test n_lcc == 2
    @test abs(λ - λ_true) / abs(λ_true) < 1e-6
end

@testset "Schur min-eigenvalue is backend-agnostic (AppleAccelerate)" begin
    # The Schur matvec is just a back-solve, so AppleAccelerate must give the same
    # eigenvalue as KLU. Only meaningful where AppleAccelerate is available.
    if Sys.isapple()
        sys = PSB.build_system(PSB.PSITestSystems, "c_sys14")
        pf = ACPowerFlow{NewtonRaphsonACPowerFlow}(; correct_bustypes = true)
        λ, λ_true, _ = _schur_eig_and_truth(pf, sys;
            backend = PNM.AppleAccelerateLUSolver())
        @test abs(λ - λ_true) / abs(λ_true) < 1e-6
    end
end

@testset "log_solver_diagnostics is off by default" begin
    sys = PSB.build_system(PSB.PSITestSystems, "c_sys14")
    pf = ACPowerFlow{NewtonRaphsonACPowerFlow}(; correct_bustypes = true,
        solver_settings = _KLU_SETTINGS)
    @test isempty(_solver_diagnostic_lines(pf, sys))
end

@testset "log_solver_diagnostics emits ‖F‖/κ̂/λ_min/contraction lines" begin
    sys = PSB.build_system(PSB.PSITestSystems, "c_sys14")
    for solver in (NewtonRaphsonACPowerFlow, TrustRegionACPowerFlow,
        LevenbergMarquardtACPowerFlow)
        pf = ACPowerFlow{solver}(; correct_bustypes = true,
            log_solver_diagnostics = true, solver_settings = _KLU_SETTINGS)
        lines = _solver_diagnostic_lines(pf, sys)
        @test length(lines) >= 2
        for line in lines
            @test occursin("‖F‖_∞ = ", line)
            @test occursin("κ̂(J) = ", line)
            @test occursin("λ_min(S) = ", line)
            @test occursin(r"at bus \d+", line)
            # Under KLU, κ̂ must be a real number, never the n/a fallback — guards
            # against the _diag_condest dispatch silently routing KLU to NaN.
            @test occursin(r"κ̂\(J\) = [0-9]", line)
            @test !occursin("κ̂(J) = n/a", line)
        end
        # The contraction ratio appears from the second logged iteration onward.
        @test any(l -> occursin("contraction = ", l), lines)
    end
end

@testset "log_solver_diagnostics works for rectangular-CI and mixed-CPB" begin
    sys = PSB.build_system(PSB.PSITestSystems, "c_sys14")
    for PFType in (PF.ACRectangularPowerFlow, PF.ACMixedPowerFlow)
        pf = PFType{NewtonRaphsonACPowerFlow}(; correct_bustypes = true,
            log_solver_diagnostics = true, solver_settings = _KLU_SETTINGS)
        lines = _solver_diagnostic_lines(pf, sys)
        @test length(lines) >= 2
        for line in lines
            @test occursin("λ_min(S) = ", line)
            @test occursin("κ̂(J) = ", line)
        end
    end
end

@testset "log_solver_diagnostics works on LCC systems" begin
    sys = System(joinpath(TEST_DATA_DIR, "case5_2_lcc.raw"))
    pf = ACPowerFlow{NewtonRaphsonACPowerFlow}(; log_solver_diagnostics = true,
        solver_settings = _KLU_SETTINGS)
    lines = _solver_diagnostic_lines(pf, sys)
    @test length(lines) >= 2
    for line in lines
        @test occursin("λ_min(S) = ", line)
    end
end

@testset "diagnostics run on AppleAccelerate, reporting κ̂ as n/a" begin
    # The Schur eigenvalue needs only a back-solve, so AppleAccelerate works; only
    # κ̂ is unavailable and must be reported as `n/a` rather than erroring. Only
    # meaningful where the AppleAccelerate backend is available (Apple platforms).
    if Sys.isapple()
        sys = PSB.build_system(PSB.PSITestSystems, "c_sys14")
        pf = ACPowerFlow{NewtonRaphsonACPowerFlow}(; correct_bustypes = true,
            log_solver_diagnostics = true,
            solver_settings = Dict{Symbol, Any}(:linear_solver => "AppleAccelerateLU"))
        lines = _solver_diagnostic_lines(pf, sys)
        @test length(lines) >= 2
        for line in lines
            @test occursin("λ_min(S) = ", line)          # back-solve path works
            @test occursin("κ̂(J) = n/a", line)           # condest unavailable
        end
    end
end

@testset "stop_at_fold returns without erroring on a well-conditioned case" begin
    # c_sys14 converges with a stable-sign Jacobian, so the bail-out never fires;
    # this just exercises the plumbing (kwarg → loop → run_solver_diagnostics!).
    sys = PSB.build_system(PSB.PSITestSystems, "c_sys14")
    for solver in (NewtonRaphsonACPowerFlow, TrustRegionACPowerFlow)
        pf = ACPowerFlow{solver}(; correct_bustypes = true,
            solver_settings = Dict{Symbol, Any}(
                :linear_solver => "KLU", :stop_at_fold => true))
        data = PowerFlowData(pf, sys)
        @test solve_power_flow!(data)
    end
end
