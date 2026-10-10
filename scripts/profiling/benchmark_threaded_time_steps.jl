# benchmark_threaded_time_steps.jl
#
# Purpose:
#   Wall-clock speedup of a threaded multi-period AC solve (`n_threads > 1`) over the serial
#   solve. Each system solves `PF_BENCH_TIME_STEPS` steps with loads and generation scaled
#   per step (±5% sinusoid), so every step needs real Newton iterations. Serial KLU
#   (`n_threads = 1`) is the baseline for KLU at each `n_threads`. Every timed solve uses a
#   fresh `PowerFlowData` (build excluded). The script keeps the best of `PF_BENCH_REPS` runs
#   after one warmup.
#
# Run command (from repo root; start Julia with at least the largest `n_threads`):
#   julia --project=test --threads=8 scripts/profiling/benchmark_threaded_time_steps.jl
#
#   Knobs: PF_BENCH_SYSTEMS (comma-separated MatpowerTestSystems names),
#   PF_BENCH_TIME_STEPS, PF_BENCH_THREADS (comma-separated), PF_BENCH_REPS.

using PowerFlows
import PowerFlows as PF
import PowerSystemCaseBuilder as PSB
using Logging

const SYSTEMS = split(
    get(ENV, "PF_BENCH_SYSTEMS", "matpower_ACTIVSg2000_sys,matpower_ACTIVSg10k_sys"), ",")
const TIME_STEPS = parse(Int, get(ENV, "PF_BENCH_TIME_STEPS", "24"))
const N_THREADS = parse.(Int, split(get(ENV, "PF_BENCH_THREADS", "1,2,4,8"), ","))
const REPS = parse(Int, get(ENV, "PF_BENCH_REPS", "3"))

function vary_steps!(data)
    for t in 1:TIME_STEPS
        s = 1 + 0.05 * sin(2π * t / TIME_STEPS)
        for m in (data.bus_active_power_withdrawals, data.bus_reactive_power_withdrawals,
            data.bus_active_power_injections)
            m[:, t] .= m[:, 1] .* s
        end
    end
    return
end

function best_time(sys, n_threads)
    best = Inf
    for rep in 0:REPS  # rep 0 is warmup
        pf = ACPolarPowerFlow(;
            time_steps = TIME_STEPS,
            correct_bustypes = true,
            solution_parameters = SolutionParameters(; linear_solver = "KLU", n_threads),
        )
        data = PF.PowerFlowData(pf, sys)
        vary_steps!(data)
        t = @elapsed converged = PF.solve_power_flow!(data)
        converged || error("n_threads = $n_threads did not converge")
        rep > 0 && (best = min(best, t))
    end
    return best
end

maximum(N_THREADS) > Threads.nthreads() &&
    @warn "Julia has $(Threads.nthreads()) thread(s); start it with --threads=$(maximum(N_THREADS))."

for name in SYSTEMS
    sys = with_logger(NullLogger()) do
        PSB.build_system(PSB.MatpowerTestSystems, String(name))
    end
    serial = with_logger(() -> best_time(sys, 1), NullLogger())
    println("$name, $TIME_STEPS steps, serial KLU: $(round(serial; digits = 3)) s")
    for n in filter(>(1), N_THREADS)
        t = with_logger(() -> best_time(sys, n), NullLogger())
        println("  KLU n_threads = $n: $(round(t; digits = 3)) s  ",
            "($(round(serial / t; digits = 2))x vs serial)")
    end
end
