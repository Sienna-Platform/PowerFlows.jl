"""
Fast-decoupled `:fixed_jacobian` re-solve time with and without `FDFixedJacobianCache`.

The cache reuses the symbolic factorization across repeated solves on the same
`PowerFlowData`; the numeric refactor runs every solve either way. `no-cache` overrides
`_get_or_build_fdj_cache!` to call `full_factor!` on every solve. Loads are scaled by
±0.1% before each solve, since an unperturbed re-solve converges in 0 iterations. Each
(system, backend, mode) case runs in its own process so the override cannot leak.

Usage:
    julia --project=test scripts/benchmarks/fdj_cache_benchmark.jl
    julia --project=test scripts/benchmarks/fdj_cache_benchmark.jl <cache|no-cache> <system> <backend>
"""

using PowerFlows, PowerSystemCaseBuilder, PowerSystems
using Logging, Printf, Statistics

const PSB = PowerSystemCaseBuilder
const PSY = PowerSystems
const PF = PowerFlows

const FDJ = PF.FastDecoupledACPowerFlow{PF.FDFixedJacobian, PF.FDSchemeXB}
const NREPS = 30
const PERTURBATION = 1.001
const SYSTEMS = Dict(
    "c_sys14" => (PSB.PSITestSystems, "c_sys14", (; add_forecasts = false)),
    "RTS" => (PSB.PSISystems, "RTS_GMLC_DA_sys", (;)),
    "2k" => (PSB.MatpowerTestSystems, "matpower_ACTIVSg2000_sys", (;)),
    "10k" => (PSB.MatpowerTestSystems, "matpower_ACTIVSg10k_sys", (;)),
)
const SYSTEM_ORDER = ["c_sys14", "RTS", "2k", "10k"]
const BACKENDS = ["KLU", "AppleAccelerateLU"]

function run_reps(data, nreps)
    times = Float64[]
    allocs = Int[]
    for r in 1:nreps
        f = 1 / PERTURBATION
        if isodd(r)
            f = PERTURBATION
        end
        data.bus_active_power_withdrawals .*= f
        GC.gc(false)
        t0 = time_ns()
        a = @allocated ok = PF.solve_power_flow!(data)
        push!(times, (time_ns() - t0) / 1e6)
        push!(allocs, a)
        if !ok
            error("solve failed on rep $r")
        end
    end
    return times, allocs
end

function run_case(mode, system, backend)
    if mode == "no-cache"
        @eval PF function _get_or_build_fdj_cache!(
            cache::FDFixedJacobianCache, data::ACPowerFlowData, key::FDJCacheKey,
            backend, Jv::SparseMatrixCSC{Float64, J_INDEX_TYPE},
        )
            return _build_fdj_cache!(data, key, backend, Jv)
        end
    end
    mod, name, kw = SYSTEMS[system]
    sys = PSB.build_system(mod, name; kw...)
    pf = ACPowerFlow{FDJ}(;
        correct_bustypes = true,
        solution_parameters = PF.SolutionParameters(; linear_solver = backend),
    )
    data = PF.PowerFlowData(pf, sys)
    if !Base.invokelatest(PF.solve_power_flow!, data)
        error("initial solve did not converge")
    end
    Base.invokelatest(run_reps, data, 3)
    t, a = Base.invokelatest(run_reps, data, NREPS)
    nb = length(collect(PSY.get_components(PSY.ACBus, sys)))
    @printf("RESULT %s %s %s %d %.3f %.1f\n",
        system, backend, mode, nb, median(t), median(a) / 1024)
    return
end

function run_all()
    println("| System | Buses | Backend | Cached (ms) | Uncached (ms) | Speedup |")
    println("|---|---:|---|---:|---:|---:|")
    for system in SYSTEM_ORDER, backend in BACKENDS
        medians = Dict{String, Float64}()
        nb = 0
        for mode in ("cache", "no-cache")
            cmd = `$(Base.julia_cmd()) --project=$(Base.active_project()) $(@__FILE__) $mode $system $backend`
            out = read(cmd, String)
            for line in split(out, '\n')
                if startswith(line, "RESULT")
                    fields = split(line)
                    nb = parse(Int, fields[5])
                    medians[mode] = parse(Float64, fields[6])
                end
            end
        end
        c = medians["cache"]
        n = medians["no-cache"]
        @printf("| %s | %d | %s | %.3f | %.3f | %.1f× |\n", system, nb, backend, c, n, n / c)
        flush(stdout)
    end
    return
end

global_logger(ConsoleLogger(stderr, Logging.Warn))
if isempty(ARGS)
    run_all()
else
    run_case(ARGS[1], ARGS[2], ARGS[3])
end
