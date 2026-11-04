"""
Discrete-control wall-clock check on a real PSS/E case (e.g. a 10k-bus RAW file).

Parses the RAW file, then solves an AC polar power flow with reactive-power limits
enforced and discrete device adjustment (taps, switched shunts, FACTS) enabled. Reports
convergence, wall-clock, inner-solve and factorization counts, and the controlled-device
outcomes.

The first solve includes compilation. `--reps=2` (default) rebuilds `PowerFlowData` from
the parsed system and solves again, so the second line is the real runtime. A fresh
`PowerFlowData` per rep avoids the 0-iteration warm start.

Usage (test env: it has PowerFlowFileParser and resolves PowerFlows to the working tree):
    julia --project=test scripts/benchmarks/discrete_control_raw_case.jl path/to/case.raw [nr|tr] [--reps=N]
"""

using PowerFlows, PowerSystems, Logging, Printf
import PowerFlowFileParser as PFP
import PowerSystemCaseBuilder: system_from_openapi

const PF = PowerFlows

function parse_args(args)
    paths = filter(a -> !startswith(a, "--") && a ∉ ("nr", "tr"), args)
    length(paths) == 1 || error("usage: discrete_control_raw_case.jl case.raw [nr|tr] [--reps=N]")
    solver = "tr" in args ? TrustRegionACPowerFlow : NewtonRaphsonACPowerFlow
    reps_arg = findfirst(a -> startswith(a, "--reps="), args)
    reps = isnothing(reps_arg) ? 2 : parse(Int, split(args[reps_arg], "=")[2])
    return only(paths), solver, reps
end

function solve_once(sys, solver)
    pf = ACPolarPowerFlow{solver}(;
        check_reactive_power_limits = true,
        control_discrete_devices = true,
        correct_bustypes = true,
    )
    data = PF.PowerFlowData(pf, sys)
    # NR prints a residual line per inner solve; keep warnings (freezes, non-settled stages).
    t = @elapsed with_logger(ConsoleLogger(stderr, Logging.Warn)) do
        PF.solve_power_flow!(data)
    end
    return data, t
end

function report(data, t, rep)
    cd = data.controlled_devices
    n = isnothing(cd) ? (0, 0, 0) : (length(cd.taps), length(cd.shunts), length(cd.facts))
    @printf("rep %d: converged=%s  time=%.2f s  inner_solves=%d  symbolic=%d  numeric=%d\n",
        rep, all(data.converged), t,
        PF.get_control_inner_solve_count(data),
        PF.get_control_symbolic_factor_count(data),
        PF.get_control_numeric_refactor_count(data))
    @printf("        controlled devices: taps=%d  shunts=%d  facts=%d\n", n...)
    return
end

function main(args)
    path, solver, reps = parse_args(args)
    println("parsing $path")
    t_parse = @elapsed sys = with_logger(ConsoleLogger(stderr, Logging.Error)) do
        system_from_openapi(PFP.PowerModelsData(path); runchecks = false)
    end
    @printf("parsed in %.1f s: %d buses; solver = %s\n",
        t_parse, length(get_components(ACBus, sys)), solver)
    data = nothing
    for rep in 1:reps
        data, t = solve_once(sys, solver)
        report(data, t, rep)
    end
    all(data.converged) || return
    res = PF.get_controlled_device_results(data)
    println("\ncontrolled-device results (last rep):")
    show(stdout, MIME("text/plain"), res; allrows = false)
    println()
    return
end

main(ARGS)
