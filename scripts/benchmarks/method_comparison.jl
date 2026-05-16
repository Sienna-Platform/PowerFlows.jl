"""
Head-to-head comparison of all AC power flow methods across grid sizes and
initial-condition quality.  Captures convergence, iteration count, initial /
final residuals, and wall-clock time. Set `OVERWRITE_NON_CONVERGED = false`
in `src/definitions.jl`, else final residual size will show NaN whenever
convergence fails.

Perturbation model: for each trial, the true NR solution is perturbed by adding
`K * scale * randn()` independently to each state variable, where `scale` depends
on the physical quantity: `VM_SCALE` for voltage magnitudes (PQ buses), `VA_SCALE`
for voltage angles (PQ and PV buses), and `PQ_SCALE` for active/reactive power
(REF and PV buses). The same K value does NOT produce the same total perturbation
norm across systems: the expected L2 norm of the perturbation is `~K sqrt(n)` for a
system of size `n`, with a constant that depends on the bus type distribution.
**IMPORTANT**: K values are comparable across solvers on the same system, but not
across systems.

Usage:
    julia --project=scripts/benchmarks scripts/benchmarks/method_comparison.jl
"""

using PowerFlows, PowerSystemCaseBuilder, PowerSystems
using Logging, LinearAlgebra, Random, DataFrames, Printf, Statistics, CSV

const PSB = PowerSystemCaseBuilder
const PSY = PowerSystems
const PF = PowerFlows

# ─────────────────────────────────────────────────────────────────────────────
# Custom logger that intercepts the @info messages emitted by the solver
# and extracts iteration count and residual norms.
# ─────────────────────────────────────────────────────────────────────────────
mutable struct MetricsCapture
    iterations::Int
    initial_residual_L2::Float64
    initial_residual_Linf::Float64
    final_residual_L2::Float64
    final_residual_Linf::Float64
    converged_from_log::Union{Bool, Nothing}
end
MetricsCapture() = MetricsCapture(0, NaN, NaN, NaN, NaN, nothing)

struct CapturingLogger <: AbstractLogger
    metrics::MetricsCapture
    min_level::LogLevel
end
CapturingLogger(m::MetricsCapture) = CapturingLogger(m, Logging.Debug)

Logging.min_enabled_level(l::CapturingLogger) = l.min_level
Logging.shouldlog(l::CapturingLogger, level, _module, group, id) = true
Logging.catch_exceptions(::CapturingLogger) = true

function Logging.handle_message(l::CapturingLogger, level, message, _module, group, id,
    filepath, line; kwargs...)
    msg = string(message)

    # "Initial residual size: 1.23 L2, 4.56 L∞"
    m = match(r"Initial residual size:\s*([\d.eE+-]+)\s*L2,\s*([\d.eE+-]+)\s*L", msg)
    if m !== nothing
        l.metrics.initial_residual_L2 = parse(Float64, m.captures[1])
        l.metrics.initial_residual_Linf = parse(Float64, m.captures[2])
    end

    # "Final residual size: 1.23 L2, 4.56 L∞."
    m = match(r"Final residual size:\s*([\d.eE+-]+)\s*L2,\s*([\d.eE+-]+)\s*L", msg)
    if m !== nothing
        l.metrics.final_residual_L2 = parse(Float64, m.captures[1])
        l.metrics.final_residual_Linf = parse(Float64, m.captures[2])
    end

    # "The NewtonRaphsonACPowerFlow solver converged after 5 iterations."
    m = match(r"solver converged after (\d+) iteration", msg)
    if m !== nothing
        l.metrics.iterations = parse(Int, m.captures[1])
        l.metrics.converged_from_log = true
    end

    # "The … solver failed to converge after N iterations."
    m = match(r"solver failed to converge after (\d+) iteration", msg)
    if m !== nothing
        l.metrics.iterations = parse(Int, m.captures[1])
        l.metrics.converged_from_log = false
    elseif occursin("failed to converge", msg)
        l.metrics.converged_from_log = false
    end

    return nothing
end

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
const SYSTEMS = [
    (PSB.PSITestSystems, "c_sys14", "14-bus", Dict(:add_forecasts => false)),
    (PSB.MatpowerTestSystems, "matpower_ACTIVSg2000_sys", "2000-bus", Dict{Symbol, Any}()),
    (PSB.MatpowerTestSystems, "matpower_ACTIVSg10k_sys", "10k-bus", Dict{Symbol, Any}()),
]

const SOLVERS = [
    (PF.NewtonRaphsonACPowerFlow, "NR", Dict{Symbol, Any}()),
    (PF.NewtonRaphsonACPowerFlow, "NR+Iwamoto", Dict{Symbol, Any}(:iwamoto => true)),
    (PF.TrustRegionACPowerFlow, "TR+Iwamoto", Dict{Symbol, Any}(:iwamoto => true)),
    (PF.LevenbergMarquardtACPowerFlow, "LM", Dict{Symbol, Any}()),
    (PF.RobustHomotopyPowerFlow, "Homotopy", Dict{Symbol, Any}()),
    (PF.GradientDescentACPowerFlow, "GradDescent", Dict{Symbol, Any}()),
]

# Perturbation magnitudes (multiplied by randn to produce x0 offsets).
# Two special cases run once (deterministic):
#   "system data" — use system setpoints as x0 (no override, whatever's in the System object)
#   "flat start"  — true flat start: Vm=1.0 pu, Va=0.0 rad for all buses
const PERTURBATIONS = [
    (NaN, "system data"),    # default initialization (no x0 override)
    (-1.0, "flat start"),    # true flat start: Vm=1, Va=0 (sentinel value)
    (0.001, "K=0.001"),
    (0.005, "K=0.005"),
    (0.01, "K=0.01"),
    (0.025, "K=0.025"),
    (0.05, "K=0.05"),
    (0.075, "K=0.075"),
    (0.1, "K=0.1"),
    (0.15, "K=0.15"),
    (0.2, "K=0.2"),
    (0.25, "K=0.25"),
    (0.3, "K=0.3"),
]

const N_TRIALS = 20  # repeats per (system, solver, perturbation) triple

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

"""Suppress all logs below `Error` for the duration of `f()`."""
function quietly(f)
    with_logger(SimpleLogger(stderr, Logging.Error)) do
        f()
    end
end

"""Get the true solution for a system by solving with NR from flat start."""
function get_true_solution(sys)
    pf = ACPowerFlow(; correct_bustypes = true)
    data = quietly(() -> PF.PowerFlowData(pf, sys))
    n = size(PF.get_bus_type(data), 1)
    bus_types = PF.get_bus_type(data)[:, 1]
    quietly(() -> PF.solve_power_flow!(data; pf = pf))
    x_solved = zeros(2n)
    PF.update_state!(x_solved, data, 1)
    return x_solved, n, bus_types
end

# Base perturbation scales per physical quantity.
const VM_SCALE = 0.1   # pu — voltage magnitude
const VA_SCALE = 0.2   # rad — voltage angle (~11 degrees)
const PQ_SCALE = 0.1   # pu — active/reactive power

"""Build a perturbed x0 from x_solved. All state variables are perturbed:
Vm and Va at PQ buses, Q and Va at PV buses, P and Q at REF buses.
K is a multiplier on the base scales."""
function perturb_x0(x_solved, bus_types, K, rng)
    n = length(bus_types)
    x0 = copy(x_solved)
    for i in 1:n
        bt = bus_types[i]
        if bt == PSY.ACBusTypes.PQ
            x0[2i - 1] += K * VM_SCALE * randn(rng)  # Vm
            x0[2i] += K * VA_SCALE * randn(rng)   # Va
        elseif bt == PSY.ACBusTypes.PV
            x0[2i - 1] += K * PQ_SCALE * randn(rng)   # Q
            x0[2i] += K * VA_SCALE * randn(rng)    # Va
        elseif bt == PSY.ACBusTypes.REF
            x0[2i - 1] += K * PQ_SCALE * randn(rng)   # P
            x0[2i] += K * PQ_SCALE * randn(rng)    # Q
        end
    end
    return x0
end

"""Run a single benchmark trial.  Returns a NamedTuple of metrics."""
function run_trial(sys, solver_type, solver_settings, x_solved, n, bus_types, K;
    seed::Int = 42)
    # Build PF object
    pf = ACPowerFlow{solver_type}(;
        correct_bustypes = true,
        solver_settings = solver_settings,
    )
    data = quietly(() -> PF.PowerFlowData(pf, sys))

    # Build x0
    kwargs = Dict{Symbol, Any}()
    if isnan(K)
        # "system data" — no x0 override, use whatever's in the System object
    elseif K < 0.0
        # "flat start" — true flat start: Vm=1.0, Va=0.0 for all buses
        x0 = zeros(2n)
        for i in 1:length(bus_types)
            x0[2i - 1] = 1.0  # Vm = 1.0 pu (or P/Q = 1.0 for REF/PV, but solver overwrites)
            x0[2i] = 0.0  # Va = 0.0 rad
        end
        kwargs[:x0] = x0
    else
        rng = Random.MersenneTwister(seed)
        x0 = perturb_x0(x_solved, bus_types, K, rng)
        kwargs[:x0] = x0
    end

    # Compute initial power flow residual at the actual starting point,
    # consistently across all methods (including homotopy).
    residual_obj = PF.ACPowerFlowResidual(data, 1)
    if haskey(kwargs, :x0)
        residual_obj(kwargs[:x0], 1)
    else
        x0_default = PF.calculate_x0(data, 1)
        residual_obj(x0_default, 1)
    end
    init_res_L2 = norm(residual_obj.Rv, 2)
    init_res_Linf = norm(residual_obj.Rv, Inf)

    # Solve under capturing logger
    metrics = MetricsCapture()
    logger = CapturingLogger(metrics)
    local converged::Bool
    local elapsed::Float64

    elapsed = @elapsed begin
        converged = with_logger(logger) do
            PF.solve_power_flow!(data; pf = pf, kwargs...)
        end
    end

    # Extract final state and compare to reference solution
    x_final = zeros(2n)
    PF.update_state!(x_final, data, 1)
    solution_dist_Linf = norm(x_final .- x_solved, Inf)
    solution_dist_L2 = norm(x_final .- x_solved, 2)

    # Compute final power flow residual directly — don't rely on log capture,
    # which may be missing (homotopy) or absent on non-convergence.
    residual_obj(x_final, 1)
    final_res_L2 = norm(residual_obj.Rv, 2)
    final_res_Linf = norm(residual_obj.Rv, Inf)

    iters = metrics.iterations

    # Residual reduction ratio: how much did the residual shrink?
    res_reduction =
        (init_res_Linf > 0 && final_res_Linf > 0) ?
        init_res_Linf / final_res_Linf : NaN

    return (
        converged = converged,
        iterations = iters,
        initial_residual_L2 = init_res_L2,
        initial_residual_Linf = init_res_Linf,
        final_residual_L2 = final_res_L2,
        final_residual_Linf = final_res_Linf,
        elapsed = elapsed,
        solution_dist_Linf = solution_dist_Linf,
        solution_dist_L2 = solution_dist_L2,
        res_reduction = res_reduction,
    )
end

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
function main()
    results = DataFrame(;
        system = String[],
        solver = String[],
        perturbation = String[],
        trial = Int[],
        converged = Bool[],
        iterations = Int[],
        initial_res_L2 = Float64[],
        initial_res_Linf = Float64[],
        final_res_L2 = Float64[],
        final_res_Linf = Float64[],
        elapsed_s = Float64[],
        sol_dist_Linf = Float64[],
        sol_dist_L2 = Float64[],
        res_reduction = Float64[],
    )

    for (group, sysname, label, build_kwargs) in SYSTEMS
        println("\n", "="^70)
        println("System: $label ($sysname)")
        println("="^70)

        sys = quietly(() -> PSB.build_system(group, sysname; build_kwargs...))
        x_solved, n, bus_types = get_true_solution(sys)
        println("  Buses: $n,  true solution obtained.\n")

        for (solver_type, solver_label, solver_settings) in SOLVERS
            gave_up = false
            for (K, perturb_label) in PERTURBATIONS
                if gave_up
                    println(
                        "  %-14s  %-12s  ⏭  skipped (solver gave up at lower K)" |>
                        s -> @sprintf(
                            "  %-14s  %-12s  ⏭  skipped (solver gave up at lower K)",
                            solver_label, perturb_label),
                    )
                    continue
                end

                n_converged = 0
                n_trials = (isnan(K) || K < 0.0) ? 1 : N_TRIALS
                for trial in 1:n_trials
                    seed = trial

                    local result
                    try
                        result = run_trial(
                            sys, solver_type, solver_settings,
                            x_solved, n, bus_types, K; seed = seed,
                        )
                    catch e
                        if e isa InterruptException
                            println(
                                "\n  ⏭  Interrupted, skipping remaining trials for $solver_label / $perturb_label",
                            )
                            break
                        end
                        @warn "FAILED: $label / $solver_label / $perturb_label / trial $trial" exception =
                            (e, catch_backtrace())
                        result = (
                            converged = false, iterations = 0,
                            initial_residual_L2 = NaN, initial_residual_Linf = NaN,
                            final_residual_L2 = NaN, final_residual_Linf = NaN,
                            elapsed = NaN,
                            solution_dist_Linf = NaN, solution_dist_L2 = NaN,
                            res_reduction = NaN,
                        )
                    end

                    n_converged += result.converged

                    push!(
                        results,
                        (
                            label, solver_label, perturb_label, trial,
                            result.converged, result.iterations,
                            result.initial_residual_L2, result.initial_residual_Linf,
                            result.final_residual_L2, result.final_residual_Linf,
                            result.elapsed,
                            result.solution_dist_Linf, result.solution_dist_L2,
                            result.res_reduction,
                        ),
                    )

                    status = result.converged ? "OK" : "FAIL"
                    dist_flag = result.solution_dist_Linf > 1e-4 ? " ⚠DIFF" : ""
                    reduc_str = if isnan(result.res_reduction)
                        "N/A"
                    else
                        @sprintf("%.1e", result.res_reduction)
                    end
                    @printf(
                        "  %-14s  %-12s  trial %d  %4s  %3d iter  %.4f s  init‖r‖∞=%.2e  final‖r‖∞=%.2e  ‖Δx‖∞=%.2e  reduce=%s%s\n",
                        solver_label, perturb_label, trial, status,
                        result.iterations, result.elapsed,
                        result.initial_residual_Linf, result.final_residual_Linf,
                        result.solution_dist_Linf, reduc_str, dist_flag)
                end

                # If no trials converged at this K, skip all higher K values.
                # Exception: GradDescent is not expected to converge; skip the early exit.
                if n_converged == 0 && !isnan(K) && K > 0.0 &&
                   solver_type != PF.GradientDescentACPowerFlow
                    gave_up = true
                    println(
                        "  %-14s  %-12s  → 0/$N_TRIALS converged, skipping higher K" |>
                        s ->
                            @sprintf("  %-14s  %-12s  → 0/%d converged, skipping higher K",
                                solver_label, perturb_label, N_TRIALS),
                    )
                end
            end
        end
    end

    # ── Summary table ──────────────────────────────────────────────────────
    println("\n\n", "="^70)
    println("SUMMARY (median over $N_TRIALS trials)")
    println("="^70)

    summary = combine(
        groupby(results, [:system, :solver, :perturbation]),
        :converged => (x -> all(x)) => :all_converged,
        :iterations => median => :med_iter,
        :elapsed_s => median => :med_time,
        :initial_res_Linf => median => :med_init_Linf,
        :final_res_Linf => median => :med_final_Linf,
        :sol_dist_Linf => maximum => :max_sol_dist,
        :res_reduction => median => :med_reduction,
    )

    @printf(
        "\n%-10s  %-14s  %-12s  Conv  Iter   Time(s)    Init‖r‖∞     Final‖r‖∞    max‖Δx‖∞   Reduction\n",
        "System", "Solver", "Perturbation")
    println("-"^120)
    for row in eachrow(summary)
        conv = row.all_converged ? "YES" : "NO"
        reduc_str =
            isnan(row.med_reduction) ? "       N/A" : @sprintf("%10.1e", row.med_reduction)
        @printf(
            "%-10s  %-14s  %-12s  %-4s  %4.0f   %8.4f   %10.2e   %10.2e   %10.2e   %s\n",
            row.system, row.solver, row.perturbation, conv,
            row.med_iter, row.med_time, row.med_init_Linf, row.med_final_Linf,
            row.max_sol_dist, reduc_str)
    end

    # Save full results
    outfile = joinpath(@__DIR__, "method_comparison_results.csv")
    CSV.write(outfile, results)
    println("\nFull results saved to $outfile")

    return results
end

main()
