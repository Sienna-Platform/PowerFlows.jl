# The Schur-eigenvalue solve and the fold bail-out need only a back-solve, so they
# work on any backend; only κ̂ is KLU-only (reported as `n/a` otherwise). Tests that
# assert a numeric κ̂ pin `linear_solver = "KLU"` for determinism across platforms
# (on Apple the default is AppleAccelerate); the AppleAccelerate path is covered
# explicitly below.
const _KLU_SETTINGS = SolutionParameters(; linear_solver = "KLU")

# Build a Schur operator at the flat start of `sys` under `backend` and return its
# smallest eigenvalue alongside the dense ground truth (smallest-magnitude
# eigenvalue of S = A − B·D⁻¹·C, recovered as inv(inv(J)[1:nb, 1:nb])).
function _schur_eig_and_truth(pf, sys; time_step = 1, backend = PNM.KLUSolver())
    data = PowerFlowData(pf, sys)
    residual = PF.ACPowerFlowResidual(data, time_step)
    jac = PF.ACPowerFlowJacobian(data, residual, time_step)
    x0 = PF.calculate_x0(data, time_step)
    residual(data, x0, time_step)
    jac(data, time_step)

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

function _assert_describe_total(residual, data, n_bus_eqs::Int)
    for ix in 1:length(residual.Rv)
        label = PF._describe_residual_entry(residual, data, 1, ix)
        @test label isa AbstractString
        @test !isempty(label)
        ix > n_bus_eqs && @test !startswith(label, "bus ")
    end
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
    sys = make_system(
        PFP.PowerModelsData(joinpath(TEST_DATA_DIR, "case5_2_lcc.raw"));
        runchecks = false,
    )
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
        solution_parameters = _KLU_SETTINGS)
    @test isempty(_solver_diagnostic_lines(pf, sys))
end

@testset "log_solver_diagnostics emits ‖F‖/κ̂/λ_min/contraction lines" begin
    sys = PSB.build_system(PSB.PSITestSystems, "c_sys14")
    for solver in (NewtonRaphsonACPowerFlow, TrustRegionACPowerFlow,
        LevenbergMarquardtACPowerFlow)
        pf = ACPowerFlow{solver}(; correct_bustypes = true,
            log_solver_diagnostics = true, solution_parameters = _KLU_SETTINGS)
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
            log_solver_diagnostics = true, solution_parameters = _KLU_SETTINGS)
        lines = _solver_diagnostic_lines(pf, sys)
        @test length(lines) >= 2
        for line in lines
            @test occursin("λ_min(S) = ", line)
            @test occursin("κ̂(J) = ", line)
        end
    end
end

@testset "log_solver_diagnostics works on LCC systems" begin
    sys = make_system(
        PFP.PowerModelsData(joinpath(TEST_DATA_DIR, "case5_2_lcc.raw"));
        runchecks = false,
    )
    pf = ACPowerFlow{NewtonRaphsonACPowerFlow}(; log_solver_diagnostics = true,
        solution_parameters = _KLU_SETTINGS)
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
            solution_parameters = SolutionParameters(;
                linear_solver = "AppleAccelerateLU"))
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
            solution_parameters = SolutionParameters(;
                linear_solver = "KLU", stop_at_fold = true))
        data = PowerFlowData(pf, sys)
        @test solve_power_flow!(data)
    end
end

# ---------------------------------------------------------------------------
# Bordered fold monitor: g = 1/(d − cᵀJ⁻¹b) = det(J)/det(M).
# ---------------------------------------------------------------------------

# A one-parameter family of Jacobians with a KNOWN singularity, on a single fixed
# sparsity pattern (so the sweep is a pure numeric refactor, as in a solver loop).
# det is AFFINE in one diagonal entry, so sweeping A[1,1] crosses zero exactly once
# and the crossing point is recoverable from two samples.
function _singular_matrix_family(; backend = PNM.KLUSolver())
    sys = PSB.build_system(PSB.PSITestSystems, "c_sys14")
    pf = ACPowerFlow{NewtonRaphsonACPowerFlow}(; correct_bustypes = true)
    data = PF.PowerFlowData(pf, sys)
    residual = PF.ACPowerFlowResidual(data, 1)
    jac = PF.ACPowerFlowJacobian(data, residual, 1)
    x0 = PF.calculate_x0(data, 1)
    residual(data, x0, 1)
    jac(data, 1)

    n = size(jac.Jv, 1)
    # An odd shift keeps every diagonal entry structurally stored (a cancelling
    # sum would be pruned, and the pattern must not change across the sweep).
    shift = 0.3141592653589793
    A = jac.Jv + shift * SparseArrays.sparse(LinearAlgebra.I, n, n)
    dptr = [
        only(filter(k -> A.rowval[k] == i, A.colptr[i]:(A.colptr[i + 1] - 1)))
        for i in 1:n
    ]
    for i in 1:n
        A.nzval[dptr[i]] -= shift
    end
    diag_11 = A.nzval[dptr[1]]
    set_t! = t -> (A.nzval[dptr[1]] = diag_11 + t)

    cache = PF.make_linear_solver_cache(backend, A)
    PF.symbolic_factor!(cache, A)
    g_at = function (t)
        set_t!(t)
        PF.numeric_refactor!(cache, A)
        return A, cache
    end
    set_t!(0.0)
    d0 = LinearAlgebra.det(Matrix(A))
    set_t!(1.0)
    d1 = LinearAlgebra.det(Matrix(A))
    return (; A, n, cache, set_t!, g_at, t_star = d0 / (d0 - d1))
end

@testset "bordering vectors are deterministic, normalized and independent" begin
    # Reproducible logs depend on the bordering being identical run to run, and the
    # zero-vs-pole test depends on the borderings not being the same vector.
    v1, v2, v3 = (Vector{Float64}(undef, 64) for _ in 1:3)
    PF._fill_border_vector!(v1, 3)
    PF._fill_border_vector!(v2, 3)
    PF._fill_border_vector!(v3, 4)
    @test v1 == v2                       # same index, same vector
    @test v1 != v3                       # different index, different vector
    @test isapprox(LinearAlgebra.norm(v1), 1.0; atol = 1e-12)
    @test all(isfinite, v1)

    # The monitor's own slots must differ from each other.
    mon = PF.BorderedFoldMonitor(64)
    @test length(mon.b) == PF.FOLD_N_BORDERINGS >= 2
    @test allunique(mon.b)
    @test allunique(mon.c)
end

@testset "sign(g) tracks sign(det J) across a singularity" begin
    fam = _singular_matrix_family()
    mon = PF.BorderedFoldMonitor(fam.n)
    # A window tight around the singularity, so the sweep contains the zero and no
    # pole; the exact crossing itself is skipped (det = 0 there, g = NaN).
    ts = [
        t for t in range(fam.t_star - 0.3, fam.t_star + 0.3; length = 21)
        if abs(t - fam.t_star) > 1e-12
    ]
    offsets = Int[]
    signs_g, signs_d = Int[], Int[]
    for t in ts
        _, cache = fam.g_at(t)
        g = PF._fold_monitor_value!(mon, cache)
        d = LinearAlgebra.det(Matrix(fam.A))
        @test isfinite(g)
        push!(signs_g, Int(sign(g)))
        push!(signs_d, Int(sign(d)))
        push!(offsets, Int(sign(g) * sign(d)))
    end
    # det J and g each flip signs once in the interval
    @test count(i -> signs_g[i] != signs_g[i - 1], 2:length(signs_g)) == 1
    @test count(i -> signs_d[i] != signs_d[i - 1], 2:length(signs_d)) == 1
    # one flips signs exactly when the other flips signs.
    @test length(unique(offsets)) == 1
end

@testset "a flip on every bordering is a fold" begin
    # Independent borderings agree only when det J itself crossed zero.
    mon = PF.BorderedFoldMonitor(8)
    tl = Test.TestLogger(; min_level = Logging.Warn)
    bailed = Logging.with_logger(tl) do
        PF._decide_det_sign_switch!(mon, "t1", [1.0, 1.0], true)   # first signs
        PF._decide_det_sign_switch!(mon, "t2", [-1.0, -1.0], true) # both flip
    end
    @test bailed
    @test any(m -> occursin("sign(det J) flipped on all", m), [r.message for r in tl.logs])

    # `bail = false` classifies and logs identically but never aborts.
    mon2 = PF.BorderedFoldMonitor(8)
    Logging.with_logger(Logging.NullLogger()) do
        PF._decide_det_sign_switch!(mon2, "t1", [1.0, 1.0], false)
        @test !PF._decide_det_sign_switch!(mon2, "t2", [-1.0, -1.0], false)
    end
end

@testset "a lone flip is a degenerate bordering, not a fold" begin
    # Only det(M) of that one bordering crossed zero: re-pick it, never cry fold.
    mon = PF.BorderedFoldMonitor(8)
    b1_first = copy(mon.b[1])
    bailed = Logging.with_logger(Logging.NullLogger()) do
        PF._decide_det_sign_switch!(mon, "t1", [1.0, 1.0], true)
        PF._decide_det_sign_switch!(mon, "t2", [-1.0, 1.0], true)
    end
    @test !bailed
    @test mon.b[1] != b1_first           # the flipping bordering was re-picked
    @test mon.signs[1] == 0              # and its sign history forgotten
    @test mon.signs[2] == 1              # the other one is untouched
    @test mon.enabled
end

@testset "a non-finite g re-picks that bordering; all non-finite is a fold" begin
    mon = PF.BorderedFoldMonitor(8)
    b1_first = copy(mon.b[1])
    bailed = Logging.with_logger(Logging.NullLogger()) do
        PF._decide_det_sign_switch!(mon, "t1", [Inf, 1.0], true)
    end
    @test !bailed                        # the live bordering still covers it
    @test mon.b[1] != b1_first

    mon2 = PF.BorderedFoldMonitor(8)
    tl = Test.TestLogger(; min_level = Logging.Warn)
    bailed2 = Logging.with_logger(tl) do
        PF._decide_det_sign_switch!(mon2, "t1", [NaN, NaN], true)
    end
    @test bailed2                        # nothing left to see with: bail
    @test any(m -> occursin("every fold-monitor bordering is degenerate", m),
        [r.message for r in tl.logs])
end

@testset "repeated bordering poles disable the monitor rather than cry fold" begin
    mon = PF.BorderedFoldMonitor(8)
    b_first = copy(mon.b[1])
    @test mon.enabled
    for _ in 1:(PF.FOLD_MAX_BORDER_REPICKS)
        Logging.with_logger(Logging.NullLogger()) do
            PF._handle_border_pole!(mon, "test", 1)
        end
    end
    @test mon.enabled                    # still re-picking
    @test mon.b[1] != b_first            # with a genuinely different bordering
    Logging.with_logger(Logging.NullLogger()) do
        PF._handle_border_pole!(mon, "test", 1)
    end
    @test !mon.enabled                   # gives up instead of reporting a fold
    # A disabled monitor never bails, whatever it is fed.
    @test !PF._decide_det_sign_switch!(mon, "test", [-1.0, -1.0], true)
end

@testset "the monitor line reports sign(det J)" begin
    sys = PSB.build_system(PSB.PSITestSystems, "c_sys14")
    pf = ACPowerFlow{NewtonRaphsonACPowerFlow}(; correct_bustypes = true,
        log_solver_diagnostics = true, solution_parameters = _KLU_SETTINGS)
    lines = _solver_diagnostic_lines(pf, sys)
    @test length(lines) >= 2
    for line in lines
        @test occursin(r"sign\(det J\) = [+−]", line)
    end
end

@testset "stop_at_fold aborts on a stressed system with a det-sign warning" begin
    # c_sys14 with every load scaled well past the nose: the solve must not report
    # convergence, and must say WHY in terms of sign(det J).
    sys = PSB.build_system(PSB.PSITestSystems, "c_sys14")
    pf = ACPowerFlow{NewtonRaphsonACPowerFlow}(; correct_bustypes = true,
        solution_parameters = SolutionParameters(;
            linear_solver = "KLU", stop_at_fold = true))
    data = PF.PowerFlowData(pf, sys)
    data.bus_active_power_withdrawals .*= 6.0
    data.bus_reactive_power_withdrawals .*= 6.0
    tl = Test.TestLogger(; min_level = Logging.Warn)
    converged = Logging.with_logger(tl) do
        solve_power_flow!(data)
    end
    @test !converged
    msgs = [r.message for r in tl.logs]
    @test any(m -> occursin("sign(det J) flipped", m), msgs)
    @test any(m -> occursin("Fold / voltage-collapse signature", m), msgs)
end

@testset "diagnostics never perturb the solve" begin
    # The monitor only back-solves against the existing factorization, so the iterates
    # must stay bit-identical to a solve with the fold bail-out off.
    sys = PSB.build_system(PSB.PSITestSystems, "c_sys14")
    function final_state(; stop_at_fold, scale, maxiter)
        pf = ACPowerFlow{NewtonRaphsonACPowerFlow}(; correct_bustypes = true,
            solution_parameters = SolutionParameters(; linear_solver = "KLU",
                stop_at_fold = stop_at_fold, maxIterations = maxiter))
        data = PF.PowerFlowData(pf, sys)
        data.bus_active_power_withdrawals .*= scale
        data.bus_reactive_power_withdrawals .*= scale
        Logging.with_logger(Logging.NullLogger()) do
            solve_power_flow!(data)
        end
        return vcat(vec(copy(data.bus_magnitude)), vec(copy(data.bus_angles)))
    end
    for scale in (3.0, 6.0, 12.0), maxiter in (2, 7)
        off = final_state(; stop_at_fold = false, scale, maxiter)
        on = final_state(; stop_at_fold = true, scale, maxiter)
        @test isequal(off, on)           # `isequal` so NaN == NaN on aborted solves
    end
end

# Residual-entry labelling. Only the leading 2·n_bus rows are bus quantities; the rest is the
# LCC, VSC and area-interchange tail, in that order.

@testset "residual entry resolver is total over the polar area-interchange tail" begin
    sys = _three_area_transfer_fixture(; slack_area3 = true)
    data = PowerFlowData(ACPolarPowerFlow(; area_interchange_control = true), sys)
    residual = PF.ACPowerFlowResidual(data, 1)
    n_bus_eqs = 2 * size(data.bus_type, 1)

    # The fixture must carry a tail, or this test proves nothing.
    @test PF.n_controlled_areas(data) > 0
    @test length(residual.Rv) > n_bus_eqs

    _assert_describe_total(residual, data, n_bus_eqs)

    @test occursin(
        "NI−PDES",
        PF._describe_residual_entry(
            residual, data, 1, length(residual.Rv)),
    )
    @test startswith(PF._describe_residual_entry(residual, data, 1, 1), "bus ")
    @test occursin("(P)", PF._describe_residual_entry(residual, data, 1, 1))
    @test occursin("(Q)", PF._describe_residual_entry(residual, data, 1, 2))
end

@testset "residual entry resolver is total over the LCC tail" begin
    sys, _ = simple_lcc_system()
    data = PowerFlowData(ACPolarPowerFlow(), sys)
    residual = PF.ACPowerFlowResidual(data, 1)
    n_bus_eqs = 2 * size(data.bus_type, 1)

    @test size(data.lcc.p_set, 1) > 0
    @test length(residual.Rv) > n_bus_eqs
    _assert_describe_total(residual, data, n_bus_eqs)
    @test occursin("LCC", PF._describe_residual_entry(residual, data, 1, n_bus_eqs + 1))
end

@testset "residual entry resolver is total over the VSC tail" begin
    sys = _build_vsc_pq_system()
    data = PowerFlowData(ACPolarPowerFlow(), sys)
    residual = PF.ACPowerFlowResidual(data, 1)
    n_bus_eqs = 2 * size(data.bus_type, 1)
    dcn = PF.get_dc_network(data)

    @test PF.n_vsc_converters(dcn) > 0
    @test length(residual.Rv) > n_bus_eqs
    _assert_describe_total(residual, data, n_bus_eqs)
    # First VSC row is a converter control row; the DC-node KCL rows follow the converters.
    @test occursin("VSC converter",
        PF._describe_residual_entry(residual, data, 1, n_bus_eqs + 1))
    @test occursin(
        "DC node",
        PF._describe_residual_entry(
            residual, data, 1, n_bus_eqs + 2 * PF.n_vsc_converters(dcn) + 1),
    )
end

@testset "improve_x0 warns rather than throwing when a tail row dominates" begin
    sys = _three_area_transfer_fixture(; slack_area3 = true)
    # An absurd schedule makes each area's NI−PDES row dwarf every bus mismatch, so the mean
    # test trips and the largest entry lands in the tail.
    for ai in PSY.get_components(PSY.AreaInterchange, sys)
        PSY.set_active_power_flow!(ai, 5.0e4 * PSY.SU)
    end
    pf = ACPolarPowerFlow(; area_interchange_control = true)
    data = PowerFlowData(pf, sys)
    residual = PF.ACPowerFlowResidual(data, 1)
    n_bus_eqs = 2 * size(data.bus_type, 1)

    # Both trigger conditions must hold, or this test would pass for the wrong reason.
    x0 = PF.calculate_x0(data, 1)
    residual(data, x0, 1)
    @test sum(abs, residual.Rv) > PF.LARGE_RESIDUAL * length(residual.Rv)
    @test argmax(abs.(residual.Rv)) > n_bus_eqs

    logger = Test.TestLogger()
    x0 = Logging.with_logger(logger) do
        PF.improve_x0(pf, data, residual, 1)
    end
    @test length(x0) == length(residual.Rv)

    warns = filter(
        r -> r.level == Logging.Warn && occursin("large initial residual", r.message),
        logger.logs)
    @test length(warns) == 1
    @test occursin("area", first(warns).message)
    @test !occursin("Largest residual at bus", first(warns).message)
end
