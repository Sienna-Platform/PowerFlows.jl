"""
3×3 comparison: {Polar, Rectangular, Mixed} AC formulations ×
{NewtonRaphson, TrustRegion, Levenberg-Marquardt} solvers. After one warm-up
solve, times N_RUNS (10) solves and reports the median and [min, max] range,
plus the (deterministic) iteration count and convergence per combination.

Feeds the "Choosing a formulation and solver" recommendation table in the
quick-start tutorial. Run after the test suite is green.

Usage:
    julia --project=scripts/benchmarks scripts/benchmarks/formulation_solver_comparison.jl
Optional: ENV `FSC_10K=1` also runs the 10k-bus system (slow).
"""

using PowerFlows, PowerSystemCaseBuilder, PowerSystems
using Logging, LinearAlgebra, Printf, Statistics

const PSB = PowerSystemCaseBuilder
const PSY = PowerSystems
const PF = PowerFlows

mutable struct Cap
    iters::Int
    converged::Union{Bool, Nothing}
    final_Linf::Float64
end
Cap() = Cap(0, nothing, NaN)

struct CapLogger <: AbstractLogger
    c::Cap
end
Logging.min_enabled_level(::CapLogger) = Logging.Debug
Logging.shouldlog(::CapLogger, level, _m, g, id) = true
Logging.catch_exceptions(::CapLogger) = true
function Logging.handle_message(l::CapLogger, level, message, _m, g, id, fp, ln;
    kwargs...)
    msg = string(message)
    m = match(r"Final residual size:\s*([\d.eE+-]+)\s*L2,\s*([\d.eE+-]+)\s*L", msg)
    m !== nothing && (l.c.final_Linf = parse(Float64, m.captures[2]))
    m = match(r"solver converged after (\d+) iteration", msg)
    if m !== nothing
        l.c.iters = parse(Int, m.captures[1])
        l.c.converged = true
    end
    m = match(r"solver failed to converge after (\d+) iteration", msg)
    if m !== nothing
        l.c.iters = parse(Int, m.captures[1])
        l.c.converged = false
    end
    return nothing
end

const FORMULATIONS = [
    ("Polar", ACPolarPowerFlow),
    ("Rectangular", ACRectangularPowerFlow),
    ("Mixed", ACMixedPowerFlow),
]
const SOLVERS = [
    ("NR", NewtonRaphsonACPowerFlow),
    ("TR", TrustRegionACPowerFlow),
    ("LM", LevenbergMarquardtACPowerFlow),
]

const N_RUNS = 10

function bench(pf, sys, label)
    solve_power_flow(pf, sys)              # warm-up (compile + caches)
    c = Cap()
    times = Float64[]
    for _ in 1:N_RUNS
        t = @elapsed begin
            with_logger(CapLogger(c)) do
                solve_power_flow(pf, sys)
            end
        end
        push!(times, t)
    end
    # Iterations / convergence are deterministic; time is the median of N_RUNS
    # with the [min, max] range reported.
    lo, hi = extrema(times)
    @printf(
        "  %-22s conv=%-5s iters=%-4d median=%8.4f s  range=[%.4f, %.4f] s  L∞=%.1e\n",
        label, string(c.converged), c.iters, median(times), lo, hi, c.final_Linf)
    return
end

function run_system(group, name, build_kwargs, extra_settings)
    println("\n=== $name ===")
    sys = PSB.build_system(group, name; build_kwargs...)
    try
        PSY.set_units_base_system!(sys, "SYSTEM_BASE")
    catch
    end
    for (fname, F) in FORMULATIONS, (sname, S) in SOLVERS
        settings = if F === ACPolarPowerFlow
            extra_settings
        else
            merge(Dict{Symbol, Any}(:validate_voltage_magnitudes => false),
                extra_settings)
        end
        pf = F{S}(; correct_bustypes = true, solver_settings = settings)
        bench(pf, sys, "$fname / $sname")
    end
    return
end

function main()
    run_system(PSB.PSITestSystems, "c_sys14",
        Dict{Symbol, Any}(:add_forecasts => false), Dict{Symbol, Any}())
    run_system(PSB.MatpowerTestSystems, "matpower_ACTIVSg2000_sys",
        Dict{Symbol, Any}(),
        Dict{Symbol, Any}(:tol => 1e-9, :maxIterations => 200))
    if get(ENV, "FSC_10K", "0") == "1"
        run_system(PSB.MatpowerTestSystems, "matpower_ACTIVSg10k_sys",
            Dict{Symbol, Any}(),
            Dict{Symbol, Any}(:tol => 1e-9, :maxIterations => 300))
    end
    return
end

main()
