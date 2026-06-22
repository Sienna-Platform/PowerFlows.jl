# Fast/Fixed Decoupled Newton-Raphson performance benchmark.
#
# This is TARGET EVIDENCE, not a CI assertion: it reports wall-time for Fast Decoupled vs
# Newton-Raphson on a ~2000-bus system for (a) a single solve and (b) a 24-step multi-period
# re-solve (where the FD B′/B″ factor-once cache is amortised across the time-step loop).
# Per-solve iteration counts appear in the @info convergence logs.
#
# Run:
#   julia --project=test test/performance/fd_benchmark.jl
#
# Expected qualitative result: FD's per-(half-)iteration cost is a fraction of an
# NR iteration; FD start-up is slower (the fixed matrices are built once), so single-solve FD
# may not beat NR, but the 24-step re-solve amortises the fixed-matrix build and the cached
# factorizations, making FD competitive or faster for repeated solves / PCM-style workloads.

using PowerSystems
using PowerSystemCaseBuilder
using PowerFlows
using Logging
import PowerFlows as PF
import PowerSystemCaseBuilder as PSB

configure_logging(; console_level = Logging.Info)

const SYS_GROUP = PSB.MatpowerTestSystems
const SYS_NAME = "matpower_ACTIVSg2000_sys"

_build() = build_system(SYS_GROUP, SYS_NAME)

# `PowerFlowData` populates only time-step 1 from the system. Replicate it across all steps so the
# multi-period solve performs `steps` genuine (identical) solves — the cache-amortization workload
# (B′ factored once, reused across the whole time-step loop) rather than near-trivial zero-injection
# solves for steps 2..n.
function _replicate_first_step!(data, steps)
    for f in (:bus_active_power_injections, :bus_reactive_power_injections,
        :bus_active_power_withdrawals, :bus_reactive_power_withdrawals)
        hasproperty(data, f) || continue
        m = getproperty(data, f)
        (m isa AbstractMatrix && size(m, 2) >= steps) || continue
        for t in 2:steps
            @views m[:, t] .= m[:, 1]
        end
    end
    return data
end

# Time a single solve, after one warm-up solve to remove compilation latency.
function bench_single(solver, settings)
    pf = ACPowerFlow{solver}(; correct_bustypes = true, solver_settings = settings)
    PF.solve_power_flow!(PF.PowerFlowData(pf, _build()))            # warm-up (compile)
    data = PF.PowerFlowData(pf, _build())
    return @elapsed PF.solve_power_flow!(data)
end

# Time a `steps`-period solve on a single PowerFlowData (the FD cache persists across the
# time-step loop within `solve_power_flow!`). After one warm-up.
function bench_multiperiod(solver, settings, steps)
    pf = ACPowerFlow{solver}(;
        correct_bustypes = true, time_steps = steps, solver_settings = settings)
    PF.solve_power_flow!(_replicate_first_step!(PF.PowerFlowData(pf, _build()), steps))  # warm-up
    data = _replicate_first_step!(PF.PowerFlowData(pf, _build()), steps)
    return @elapsed PF.solve_power_flow!(data)
end

const CASES = [
    ("NewtonRaphson", PF.NewtonRaphsonACPowerFlow, Dict{Symbol, Any}()),
    ("FastDecoupled(FDDecoupled,XB)",
        PF.FastDecoupledACPowerFlow{PF.FDDecoupled, PF.FDSchemeXB}, Dict{Symbol, Any}(),
    ),
    ("FastDecoupled(FDDecoupled,BX)",
        PF.FastDecoupledACPowerFlow{PF.FDDecoupled, PF.FDSchemeBX}, Dict{Symbol, Any}(),
    ),
    ("FastDecoupled(FDFixedJacobian)",
        PF.FastDecoupledACPowerFlow{PF.FDFixedJacobian, PF.FDSchemeXB},
        Dict{Symbol, Any}()),
]

_try(f) =
    try
        round(f(); digits = 3)
    catch e
        @error "benchmark case failed" exception = (e, catch_backtrace())
        "FAILED"
    end

println("\n=== Fast Decoupled vs Newton-Raphson on $(SYS_NAME) ===")
println(rpad("solver", 34), rpad("single solve (s)", 18), "24-step solve (s)")
for (label, solver, settings) in CASES
    ts = _try(() -> bench_single(solver, settings))
    tm = _try(() -> bench_multiperiod(solver, settings, 24))
    println(rpad(label, 34), rpad(string(ts), 18), string(tm))
end
println(
    "\n(Iteration counts are in the @info \"... converged after N iterations\" logs above.)",
)
