"""
Locate where the discrete-control continuation spends its time on a real PSS/E case.

Usage (test env, single thread):
    julia --project=test -t 1 scripts/profiling/profile_discrete_control_raw.jl \\
        path/to/case.raw [nr|tr] [--out=DIR] [--skip-baseline]

Steps:
  1. Parse the RAW file and time `PowerFlowData` construction (incl. the device-set build).
  2. Baseline: one solve with Q limits but NO device control (also compiles the NR path).
  3. The controlled solve under the sampling profiler, with a counting logger that prints a
     progress line every ~15 s.

Ctrl-C during step 3 stops the solve and still writes the report, so there is no need to
wait for it to finish.

The report contains function names, source locations and counts only — no bus, device or
network data from the case. Warning messages from the solve (which can contain device names)
are written to `warnings.txt` in the output directory and are not printed.
"""

using PowerFlows, PowerSystems, Logging, Printf, Profile, Dates
import PowerFlowFileParser as PFP
import PowerSystemCaseBuilder: system_from_openapi

const PF = PowerFlows

# ── args ────────────────────────────────────────────────────────────────────────────────
function parse_args(args)
    flags = filter(a -> startswith(a, "--"), args)
    rest = filter(a -> !startswith(a, "--"), args)
    solver = "tr" in rest ? TrustRegionACPowerFlow : NewtonRaphsonACPowerFlow
    paths = filter(a -> a ∉ ("nr", "tr"), rest)
    length(paths) == 1 ||
        error("usage: profile_discrete_control_raw.jl case.raw [nr|tr] [--out=DIR] [--skip-baseline]")
    out_flag = findfirst(f -> startswith(f, "--out="), flags)
    out = isnothing(out_flag) ?
          "discrete_control_profile_" * Dates.format(now(), "yyyymmdd_HHMMSS") :
          split(flags[out_flag], "="; limit = 2)[2]
    return (; path = only(paths), solver, out, baseline = "--skip-baseline" ∉ flags)
end

make_pf(solver, control) = ACPolarPowerFlow{solver}(;
    check_reactive_power_limits = true,
    control_discrete_devices = control,
    correct_bustypes = true,
)

# ── counting logger ─────────────────────────────────────────────────────────────────────
# Counts NR solves/iterations from PowerFlows' own @debug lines and groups warnings by
# source location. Nothing is printed except the periodic progress line.
mutable struct CountingLogger <: AbstractLogger
    t0::Float64
    last_beat::Float64
    nr_converged::Int
    nr_failed::Int
    nr_iters::Int
    max_iters::Int
    q_oscillation::Int
    warn_counts::Dict{String, Int}
    warn_first::Dict{String, String}
    data::Any   # PowerFlowData, for the live inner-solve counter
end
CountingLogger() = CountingLogger(time(), time(), 0, 0, 0, 0, 0,
    Dict{String, Int}(), Dict{String, String}(), nothing)

Logging.min_enabled_level(::CountingLogger) = Logging.Debug
Logging.catch_exceptions(::CountingLogger) = true
# Debug is accepted ONLY from the NR/TR driver file (the "converged/failed after N iterations"
# lines). Accepting every PowerFlows @debug makes Julia build those message strings, which
# inflates the very functions being profiled (e.g. `_check_q_limit_bounds!`).
function Logging.shouldlog(::CountingLogger, level, _module, group, id)
    return level >= Logging.Warn ||
           (Base.moduleroot(_module) === PowerFlows && group === :power_flow_method)
end

const RX_CONV = r"solver converged after (\d+) iterations"
const RX_FAIL = r"solver failed to converge after (\d+) iterations"

function Logging.handle_message(l::CountingLogger, level, message, _module, group, id,
    file, line; kwargs...)
    msg = string(message)
    if level >= Logging.Warn
        key = "$(basename(string(file))):$line"
        l.warn_counts[key] = get(l.warn_counts, key, 0) + 1
        get!(l.warn_first, key, msg)
        occursin("reactive power limits still oscillating", msg) && (l.q_oscillation += 1)
    elseif (m = match(RX_CONV, msg)) !== nothing
        _nr_done!(l, parse(Int, m[1]), true)
    elseif (m = match(RX_FAIL, msg)) !== nothing
        _nr_done!(l, parse(Int, m[1]), false)
    end
    return
end

function _nr_done!(l::CountingLogger, iters::Int, ok::Bool)
    ok ? (l.nr_converged += 1) : (l.nr_failed += 1)
    l.nr_iters += iters
    l.max_iters = max(l.max_iters, iters)
    if time() - l.last_beat > 15
        l.last_beat = time()
        inner = isnothing(l.data) ? 0 : PF.get_control_inner_solve_count(l.data)
        @printf(stderr,
            "  [%6.0f s] NR solves: %d ok / %d failed, %d iterations; control inner solves: %d\n",
            time() - l.t0, l.nr_converged, l.nr_failed, l.nr_iters, inner)
    end
    return
end

# ── profile attribution ─────────────────────────────────────────────────────────────────
# Inclusive share of samples whose stack contains each marker frame. `file` restricts a
# generic name (e.g. `solve!`) to one package's source.
const MARKERS = [
    ("continuation (total)", :_control_continuation!, ""),
    ("  probe phase", :_probe_device_signs!, ""),
    ("    blocked sensitivity solves (all callers)", :_linear_plant_signs!, ""),
    ("    finite-difference probe (solves!)", :_plant_sign, ""),
    ("  sensitivity ctx build", :_sensitivity_context, ""),
    ("  sensitivity ctx refresh/rebuild", :_refresh_or_rebuild_context, ""),
    ("  batched pass", :_batched_pass!, ""),
    ("    batched gain refresh", :_refresh_gains_group!, ""),
    ("  SEQUENTIAL fallback pass", :_step_voltage_groups!, ""),
    ("    per-device continuation walk", :_continuation_to!, ""),
    ("    bisection sub-stepping", :_bisection_walk!, ""),
    ("  snap + restore", :snap_and_restore!, ""),
    ("    per-device restore walk", :_restore_one!, ""),
    ("solve w/ Q limits (all callers)", :_solve_with_q_limits!, ""),
    ("  Newton/TR solve", :_newton_power_flow, ""),
    ("  Q-limit check", :_check_q_limit_bounds!, ""),
    ("Jacobian construction", :ACPowerFlowJacobian, ""),
    ("  Jacobian structure build", :_get_or_build_jacobian_structure, ""),
    ("Jacobian value update", :_update_jacobian_matrix_values!, ""),
    ("NR step (Δx)", :_set_Δx_nr!, ""),
    ("  iterative refinement", :_do_refinement!, ""),
    ("PNM symbolic factor", :symbolic_factor!, "PowerNetworkMatrices"),
    ("PNM numeric refactor", :numeric_refactor!, "PowerNetworkMatrices"),
    ("PNM full factor", :full_factor!, "PowerNetworkMatrices"),
    ("PNM triangular solve", :solve!, "PowerNetworkMatrices"),
    ("device apply_parameter!", :apply_parameter!, ""),
    ("state snapshot/restore", :_restore_state!, ""),
    ("arc admittance sync", :_sync_arc_admittances!, ""),
]
const COMPILE_HINTS = ("typeinf", "jl_compile", "jl_generate_fptr", "emit_", "jl_type_infer")

_frame_name(fr) = string(fr.func)
function _matches(fr, name::Symbol, file::String)
    s = _frame_name(fr)
    hit = s == string(name) || startswith(s, "#$(name)#") || startswith(s, "$(name)##")
    return hit && (isempty(file) || occursin(file, string(fr.file)))
end

function attribute(io)
    data = Profile.fetch(; include_meta = false)
    cache = Dict{UInt64, Vector{Base.StackTraces.StackFrame}}()
    frames(ip) = get!(() -> Profile.lookup(ip), cache, ip)
    hits = zeros(Int, length(MARKERS))
    self = Dict{String, Int}()
    nsamp = 0
    ncompile = 0
    seen = falses(length(MARKERS))
    stack = Base.StackTraces.StackFrame[]
    for ip in data
        if ip != 0
            append!(stack, frames(ip))
            continue
        end
        # Only samples taken inside the solve: Julia ≥ 1.12 also samples the (idle)
        # interactive thread even under `-t 1`, which would dilute every percentage.
        in_solve = any(fr -> _matches(fr, :solve_power_flow!, "PowerFlows"), stack)
        if !in_solve
            empty!(stack)
            continue
        end
        nsamp += 1
        fill!(seen, false)
        compiling = false
        for fr in stack
            compiling |= fr.from_c && any(h -> occursin(h, _frame_name(fr)), COMPILE_HINTS)
            for (k, (_, name, file)) in enumerate(MARKERS)
                seen[k] |= _matches(fr, name, file)
            end
        end
        hits .+= seen
        ncompile += compiling
        # innermost Julia frame = self time
        j = findfirst(fr -> !fr.from_c, stack)
        if !isnothing(j)
            fr = stack[j]
            key = "$(fr.func)  $(basename(string(fr.file))):$(fr.line)"
            self[key] = get(self, key, 0) + 1
        end
        empty!(stack)
    end
    nsamp == 0 && (println(io, "no profile samples collected"); return)
    pct(n) = 100 * n / nsamp
    @printf(io, "\n%d samples inside solve_power_flow!; %.1f%% inside JIT compilation\n\n", nsamp, pct(ncompile))
    println(io, "inclusive time by phase (share of all samples):")
    for (k, (label, _, _)) in enumerate(MARKERS)
        @printf(io, "  %-40s %6.1f%%\n", label, pct(hits[k]))
    end
    println(io, "\ntop self-time frames:")
    for (key, n) in first(sort(collect(self); by = last, rev = true), 25)
        @printf(io, "  %6.1f%%  %s\n", pct(n), key)
    end
    return
end

# ── main ────────────────────────────────────────────────────────────────────────────────
function main(args)
    a = parse_args(args)
    mkpath(a.out)
    report = IOBuffer()
    say(x...) = (println(x...); println(report, x...))

    Threads.nthreads() == 1 ||
        @warn "running with $(Threads.nthreads()) threads; idle-thread samples dilute the profile (use -t 1)"
    say("case: $(basename(a.path))   solver: $(a.solver)")
    t = @elapsed sys = with_logger(ConsoleLogger(stderr, Logging.Error)) do
        system_from_openapi(PFP.PowerModelsData(a.path); runchecks = false)
    end
    say(@sprintf("parse: %.1f s, %d buses", t, length(get_components(ACBus, sys))))

    if a.baseline
        t_build = @elapsed data0 = PF.PowerFlowData(make_pf(a.solver, false), sys)
        cl0 = CountingLogger()
        t0 = @elapsed ok0 = with_logger(() -> PF.solve_power_flow!(data0), cl0)
        say(@sprintf(
            "baseline (Q limits, no control): build %.1f s, solve %.1f s (incl. compile), converged=%s, NR solves=%d, iterations=%d",
            t_build, t0, all(data0.converged), cl0.nr_converged + cl0.nr_failed, cl0.nr_iters))
    end

    t_build = @elapsed data = with_logger(ConsoleLogger(stderr, Logging.Error)) do
        PF.PowerFlowData(make_pf(a.solver, true), sys)
    end
    cd = data.controlled_devices
    ndev = isnothing(cd) ? (0, 0, 0) : (length(cd.taps), length(cd.shunts), length(cd.facts))
    say(@sprintf("controlled build: %.1f s; enrolled taps=%d shunts=%d facts=%d",
        t_build, ndev...))
    isnothing(cd) && (say("no enrolled devices; nothing to profile"); return)

    cl = CountingLogger()
    cl.data = data
    Profile.clear()
    Profile.init(; n = 10^7, delay = 0.005)
    Base.exit_on_sigint(false)
    interrupted = false
    println("profiling controlled solve (Ctrl-C to stop early and still get the report)…")
    t_solve = @elapsed try
        with_logger(cl) do
            Profile.@profile PF.solve_power_flow!(data)
        end
    catch e
        e isa InterruptException || rethrow()
        interrupted = true
    end
    Base.exit_on_sigint(true)

    say(@sprintf("\ncontrolled solve: %.1f s%s, converged=%s", t_solve,
        interrupted ? " (INTERRUPTED)" : "", all(data.converged)))
    say(@sprintf("  control inner solves: %d   symbolic factors: %d   numeric refactors: %d",
        PF.get_control_inner_solve_count(data), PF.get_control_symbolic_factor_count(data),
        PF.get_control_numeric_refactor_count(data)))
    nr = cl.nr_converged + cl.nr_failed
    say(@sprintf("  NR/TR solves: %d (%d failed), iterations: %d total, %.1f mean, %d max",
        nr, cl.nr_failed, cl.nr_iters, cl.nr_iters / max(nr, 1), cl.max_iters))
    say(@sprintf("  Q-limit oscillation caps hit: %d", cl.q_oscillation))
    say("  warnings by source location (messages in warnings.txt):")
    for (k, n) in sort(collect(cl.warn_counts); by = last, rev = true)
        say(@sprintf("    %6d  %s", n, k))
    end

    buf = IOBuffer()
    attribute(buf)
    say(String(take!(buf)))
    open(joinpath(a.out, "warnings.txt"), "w") do io
        for (k, msg) in cl.warn_first
            println(io, "$k  (×$(cl.warn_counts[k]))\n  $msg\n")
        end
    end
    open(joinpath(a.out, "profile_flat.txt"), "w") do io
        Profile.print(IOContext(io, :displaysize => (10^4, 300));
            format = :flat, sortedby = :count, mincount = 20)
    end
    open(joinpath(a.out, "profile_tree.txt"), "w") do io
        Profile.print(IOContext(io, :displaysize => (10^4, 300));
            noisefloor = 2.0, mincount = 50)
    end
    write(joinpath(a.out, "summary.txt"), take!(report))
    println("\nreports written to $(abspath(a.out))")
    return
end

main(ARGS)
