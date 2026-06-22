# profile_power_flow_solvers.jl
#
# Purpose:
#   Phase-attributed + component-attributed profiling of the PowerFlows AC/DC
#   solvers on the large `matpower_ACTIVSg10k_sys` (~10k buses), comparing the
#   KLU and AppleAccelerate linear-solver backends to locate performance hot
#   spots.
#
#   AC solves run from a FLAT START (PQ |V|→1.0, all θ→0); the data is built once
#   and re-flat-started before each solve. enhanced_flat_start (the default) is
#   kept, so the flat guess is realistically improved and the solve converges —
#   but starting from flat still forces many more Newton iterations than the
#   system's near-solution warm start. The full-solve timing and the deep profile
#   therefore reflect the ITERATION work (residual/Jacobian/linear-solve) rather
#   than the one-time data build.
#
#   For each (solver, backend) it times the high-level phases (data build,
#   residual build, Jacobian build, full solve-from-flat) and, for the AC
#   formulation, the per-iteration components (residual evaluation, Jacobian
#   assembly, numeric refactorization, triangular solve) — the latter isolates
#   where KLU and AppleAccelerate differ. It then re-runs each full solve under
#   `Profile.@profile` and writes flat + tree text reports (and a PProf
#   `.pb.gz` flamegraph when PProf is available) to this directory.
#
# Run command (from repo root):
#   julia --project=test scripts/profiling/profile_power_flow_solvers.jl
#
#   Smoke-test on a small system first (fast, verifies the harness):
#   PF_PROFILE_SYSTEM=c_sys14 julia --project=test \
#       scripts/profiling/profile_power_flow_solvers.jl
#
# Note on project activation:
#   Uses the repo `test` project because PowerSystemCaseBuilder / PowerSystems
#   are dev'd there (same convention as the PNM profiling scripts).

using PowerFlows
import PowerFlows as PF
import PowerSystems as PSY
import PowerNetworkMatrices as PNM
import PowerSystemCaseBuilder as PSB
using PowerSystemCaseBuilder: build_system, MatpowerTestSystems, PSITestSystems
using Printf
using Profile
using Statistics: median
import SparseArrays
import Logging

# Silence the per-build network-reduction info chatter and the (benign) voltage
# range-validation warnings so the timing tables are readable; errors still surface.
Logging.disable_logging(Logging.Warn)

# Optional PProf flamegraph export. Import once at top level (load world) so the
# later `PProf.pprof` call is free of Julia 1.12 world-age complaints.
const HAS_PPROF = try
    @eval import PProf
    true
catch
    false
end

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
const SYSTEM_NAME = get(ENV, "PF_PROFILE_SYSTEM", "matpower_ACTIVSg10k_sys")
const PASSES = 5
const WARMUP = 1
# AC solves run from a flat start (see `_flatstart!`) and therefore do many Newton
# iterations — much heavier than a warm-started solve — so use fewer passes/repeats.
const SOLVE_PASSES = 3
const PROFILE_REPEATS = 8    # repeat each solve under @profile to gather samples
const PROFILE_DELAY = 0.0005 # seconds between profile samples
const OUT_DIR = @__DIR__

# matpower / ACTIVSg cases live in MatpowerTestSystems; everything else (the small
# smoke-test systems like c_sys14) lives in PSITestSystems.
_system_group(name::AbstractString) =
    if (occursin("matpower", name) || occursin("ACTIVSg", name))
        MatpowerTestSystems
    else
        PSITestSystems
    end

# KLU is universal; AppleAccelerate only where the platform provides it.
function available_backends()
    backends = String["KLU"]
    Sys.isapple() && push!(backends, "AppleAccelerateLU")
    return backends
end

# ─────────────────────────────────────────────────────────────────────────────
# Timing helpers (same style as the PNM profiling scripts)
# ─────────────────────────────────────────────────────────────────────────────
struct TimeStats
    median_ns::Float64
    min_ns::Float64
    bytes::Int
end

function fmt_time(t_ns::Float64)
    t_ns < 1e3 && return @sprintf("%.0f ns", t_ns)
    t_ns < 1e6 && return @sprintf("%.2f µs", t_ns / 1e3)
    t_ns < 1e9 && return @sprintf("%.2f ms", t_ns / 1e6)
    return @sprintf("%.3f s", t_ns / 1e9)
end

function fmt_bytes(b::Int)
    b < 1024 && return @sprintf("%d B", b)
    b < 1024^2 && return @sprintf("%.1f KiB", b / 1024)
    b < 1024^3 && return @sprintf("%.1f MiB", b / 1024^2)
    return @sprintf("%.2f GiB", b / 1024^3)
end

cell(s::TimeStats) = @sprintf("%-11s (min %-11s, %s)",
    fmt_time(s.median_ns), fmt_time(s.min_ns), fmt_bytes(s.bytes))

"""Run `f` `warmup` times (discarded), then `passes` timed runs; report median &
min elapsed and a single `@allocated` measurement."""
function collect_stats(f::F; passes::Int = PASSES, warmup::Int = WARMUP) where {F}
    for _ in 1:warmup
        f()
    end
    times = Vector{Float64}(undef, passes)
    for i in 1:passes
        times[i] = @elapsed f()
    end
    bytes = @allocated f()
    return TimeStats(median(times) * 1e9, minimum(times) * 1e9, bytes)
end

# ─────────────────────────────────────────────────────────────────────────────
# Load system once
# ─────────────────────────────────────────────────────────────────────────────
println("Loading system: ", SYSTEM_NAME)
const SYS = build_system(_system_group(SYSTEM_NAME), SYSTEM_NAME)
PSY.set_units_base_system!(SYS, "SYSTEM_BASE")
println("  buses    = ", length(PSY.get_components(PSY.Bus, SYS)))
println("  branches = ", length(PSY.get_components(PSY.ACBranch, SYS)))
println("  backends = ", join(available_backends(), ", "))
println()

# (label, solver type, extra solver_settings merged into the pf). The FD entries exercise both
# the polar :decoupled B′/B″ loop and the formulation-agnostic :fixed_jacobian (frozen J) loop.
const AC_SOLVERS = [
    ("AC-NewtonRaphson", PF.NewtonRaphsonACPowerFlow, Dict{Symbol, Any}()),
    ("AC-TrustRegion", PF.TrustRegionACPowerFlow, Dict{Symbol, Any}()),
    ("AC-FastDecoupled-decoupled", PF.FastDecoupledACPowerFlow,
        Dict{Symbol, Any}(:fd_variant => :decoupled)),
    ("AC-FastDecoupled-fixedjac", PF.FastDecoupledACPowerFlow,
        Dict{Symbol, Any}(:fd_variant => :fixed_jacobian)),
]

# ─────────────────────────────────────────────────────────────────────────────
# Flat start: reset the variable state (PQ |V| → 1.0, all θ → 0) so the solver
# converges from scratch and exercises many Newton iterations. PV/REF voltage
# setpoints are left untouched, so the underlying problem is unchanged.
# ─────────────────────────────────────────────────────────────────────────────
function _flatstart!(data, pq_idx::Vector{Int})
    fill!(view(data.bus_angles, :, 1), 0.0)
    bm = view(data.bus_magnitude, :, 1)
    @inbounds for i in pq_idx
        bm[i] = 1.0
    end
    return data
end

# ─────────────────────────────────────────────────────────────────────────────
# AC phase + component timing for one (solver, backend)
# ─────────────────────────────────────────────────────────────────────────────
function profile_ac(label, solver, backend, extra_settings = Dict{Symbol, Any}())
    # Keep enhanced_flat_start (the default) so the flat guess is realistically
    # improved and the solve converges; resetting to flat below still forces many
    # more Newton iterations than the system's near-solution warm start, without
    # manufacturing a divergence.
    settings = merge(Dict{Symbol, Any}(:linear_solver => backend), extra_settings)
    pf = ACPowerFlow{solver}(;
        correct_bustypes = true,
        solver_settings = settings,
    )

    # Build data ONCE and reuse it across (re-flat-started) solves, so the full-solve
    # timing and the deep profile reflect the Newton iteration rather than the
    # one-time data build.
    data = PF.PowerFlowData(pf, SYS)
    pq_idx = findall(==(PSY.ACBusTypes.PQ), @view data.bus_type[:, 1])

    # P1 still measures the honest one-time data-build cost (fresh build).
    p_data = collect_stats(() -> PF.PowerFlowData(pf, SYS))

    _flatstart!(data, pq_idx)
    residual = PF.ACPowerFlowResidual(data, 1)
    p_res = collect_stats(() -> PF.ACPowerFlowResidual(data, 1))
    p_jac = collect_stats(() -> PF.ACPowerFlowJacobian(residual, 1))

    # Full solve FROM A FLAT START — re-flat-start each call so it iterates fully.
    solve_from_flat = function ()
        _flatstart!(data, pq_idx)
        return PF.solve_power_flow!(data)
    end
    p_solve = collect_stats(solve_from_flat; passes = SOLVE_PASSES, warmup = 1)
    _flatstart!(data, pq_idx)
    converged = PF.solve_power_flow!(data)

    # Per-iteration components on the system's WARM state (non-singular Jacobian).
    # The flat-start Jacobian can be singular on large systems (the full-solve path
    # handles that via the regularized fallback, but a direct numeric_refactor! would
    # throw); per-call component cost is state-independent, so the warm state is both
    # representative and crash-free.
    cdata = PF.PowerFlowData(pf, SYS)
    cresidual = PF.ACPowerFlowResidual(cdata, 1)
    J = PF.ACPowerFlowJacobian(cresidual, 1)
    x = PF.calculate_x0(cdata, 1)
    cresidual(x, 1)
    J(1)
    tag = PF.resolve_linear_solver_backend(backend)
    cache = PF.make_linear_solver_cache(tag, J.Jv)
    PF.symbolic_factor!(cache, J.Jv)
    PF.numeric_refactor!(cache, J.Jv)
    rbuf = copy(cresidual.Rv)

    c_res = collect_stats(() -> cresidual(x, 1))
    c_jac = collect_stats(() -> J(1))
    c_refac = collect_stats(() -> PF.numeric_refactor!(cache, J.Jv))
    c_solve = collect_stats(() -> (copyto!(rbuf, cresidual.Rv); PF.solve!(cache, rbuf)))

    println("── $label  [backend = $backend]  (flat start, converged = $converged) ──")
    println("  Phases:")
    println("    P1 PowerFlowData build      : ", cell(p_data))
    println("    P2 Residual build           : ", cell(p_res))
    println("    P3 Jacobian build           : ", cell(p_jac))
    println("    P4 Full solve! (from flat)  : ", cell(p_solve))
    println("  Per-iteration components:")
    println("    residual evaluation         : ", cell(c_res))
    println("    Jacobian assembly           : ", cell(c_jac))
    println("    numeric_refactor!           : ", cell(c_refac), "   <- backend")
    println("    triangular solve!           : ", cell(c_solve), "   <- backend")
    println()
    return solve_from_flat
end

# ─────────────────────────────────────────────────────────────────────────────
# DC phase timing for one backend
# ─────────────────────────────────────────────────────────────────────────────
function profile_dc(backend)
    mk_data() = PF.PowerFlowData(DCPowerFlow(), SYS)
    p_data = collect_stats(mk_data)
    p_solve = collect_stats(
        () -> PF.solve_power_flow!(mk_data(); linear_solver = backend))
    println("── DC  [backend = $backend] ───────────────────────────────────")
    println("    P1 PowerFlowData build : ", cell(p_data))
    println("    P2 Full solve! (factor+solve): ", cell(p_solve))
    println()
    return () -> PF.solve_power_flow!(mk_data(); linear_solver = backend)
end

# ─────────────────────────────────────────────────────────────────────────────
# Deep profile: re-run a solve closure under Profile.@profile, write reports.
# ─────────────────────────────────────────────────────────────────────────────
function deep_profile(label, solve_closure)
    Profile.clear()
    Profile.init(; n = 10_000_000, delay = PROFILE_DELAY)
    Profile.@profile for _ in 1:PROFILE_REPEATS
        solve_closure()
    end
    safe = replace(label, r"[^A-Za-z0-9]+" => "_")
    flat = joinpath(OUT_DIR, "pf_profile_$(safe).flat.txt")
    tree = joinpath(OUT_DIR, "pf_profile_$(safe).tree.txt")
    open(flat, "w") do io
        Profile.print(io; format = :flat, mincount = 10, sortedby = :count)
    end
    open(tree, "w") do io
        Profile.print(io; format = :tree, mincount = 10)
    end
    println("  wrote $(basename(flat)), $(basename(tree))")
    if HAS_PPROF
        out = joinpath(OUT_DIR, "pf_profile_$(safe).pb.gz")
        PProf.pprof(; out = out, web = false)
        println("  wrote $(basename(out)) (open with PProf.refresh / pprof)")
    else
        println("  (PProf not installed; skipped flamegraph)")
    end
    return nothing
end

# ─────────────────────────────────────────────────────────────────────────────
# Drive
# ─────────────────────────────────────────────────────────────────────────────
const BACKENDS = available_backends()
solve_closures = Dict{String, Function}()

println("=========================  PHASE / COMPONENT TIMING  ========================\n")
for backend in BACKENDS
    solve_closures["DC|$backend"] = profile_dc(backend)
    for (label, solver, extra_settings) in AC_SOLVERS
        solve_closures["$label|$backend"] =
            profile_ac(label, solver, backend, extra_settings)
    end
end

println("=============================  DEEP PROFILES  ===============================\n")
for (label, closure) in sort(collect(solve_closures); by = first)
    println("Profiling: ", label, "  (", PROFILE_REPEATS, " repeats)")
    deep_profile(label, closure)
    println()
end

println("Done. Reports written to: ", OUT_DIR)
