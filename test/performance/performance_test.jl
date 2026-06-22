precompile = @timed using PowerFlows

function is_running_on_ci()
    return get(ENV, "CI", "false") == "true" || haskey(ENV, "GITHUB_ACTIONS")
end

using Dates

pushed_to_args = false
if length(ARGS) == 0
    pushed_to_args = true
    if is_running_on_ci()
        push!(ARGS, "CI Test at $(Dates.now())")
    else
        push!(ARGS, "Local Test at $(Dates.now())")
    end
end

open("precompile_time_$(ARGS[1]).txt", "w") do io
    write(io, string(precompile.time))
end

using PowerSystems
using PowerSystemCaseBuilder
using PowerFlows
using Logging
import PowerFlows as PF

configure_logging(; console_level = Logging.Info)
systems = [
    (MatpowerTestSystems, "matpower_ACTIVSg10k_sys"),
]

function record_time(label, time)
    open("solve_time_$(ARGS[1]).csv", "a") do io
        write(io, "$(label),$(time)\n")
    end
end

function record_failure(label)
    open("solve_time_$(ARGS[1]).csv", "a") do io
        write(io, "$(label),FAILED\n")
    end
end

# Two timed solves of a freshly-built evaluation model on a fresh `PowerFlowData` each (avoids
# warm-start contamination); records the wall time per pass, or one FAILED row on error. `make_pf`
# is a thunk so the AC formulation/solver/settings vary per call site; `bench_dc!` takes the
# stateless DC model directly.
function bench_ac!(name, solver_label, make_pf, sys)
    try
        for pass in ("First", "Second")
            pf = make_pf()
            pf_data = PF.PowerFlowData(pf, sys)
            _, time_solve, _, _ = @timed PF.solve_power_flow!(pf_data; pf = pf)
            record_time("$(name)-$(solver_label) $(pass) Solve", time_solve)
        end
    catch e
        @error exception = (e, catch_backtrace())
        record_failure("$(name)-$(solver_label) Solve")
    end
end

function bench_dc!(name, solver_label, dc_pf, sys)
    try
        for pass in ("First", "Second")
            pf_data = PF.PowerFlowData(dc_pf, sys)
            _, time_solve, _, _ = @timed PF.solve_power_flow!(pf_data)
            record_time("$(name)-$(solver_label) $(pass) Solve", time_solve)
        end
    catch e
        @error exception = (e, catch_backtrace())
        record_failure("$(name)-$(solver_label) Solve")
    end
end

# Decoupled BX scheme has no exported alias (unlike FastDecoupledXB/Fixed); name it once here.
const FD_BX = PF.FastDecoupledACPowerFlow{PF.FDDecoupled, PF.FDSchemeBX}

# Polar AC solvers, as (label, solver, settings). All three FastDecoupled configurations (decoupled
# XB, decoupled BX, and fixed-Jacobian) are exercised here (and on the Eastern Interconnect below) —
# the factor-once advantage is only relevant at scale, so small systems aren't worth timing. The
# decoupled XB/BX schemes are ill-conditioned on this synthetic case (many off-nominal taps), so
# pure decoupled crawls toward the default 1e-9 tolerance (~1500 iters); relax their tolerance so
# they record a meaningful timing instead of hitting the iteration cap. The robust FD→Newton handoff
# is the production path for tight tolerance. The other solvers keep the default tolerance.
polar_ac_solvers = [
    ("NewtonRaphsonACPowerFlow", PF.NewtonRaphsonACPowerFlow, Dict{Symbol, Any}()),
    ("NewtonRaphsonACPowerFlow(iwamoto)", PF.NewtonRaphsonACPowerFlow,
        Dict{Symbol, Any}(:iwamoto => true)),
    ("TrustRegionACPowerFlow(iwamoto)", PF.TrustRegionACPowerFlow,
        Dict{Symbol, Any}(:iwamoto => true)),
    ("RobustHomotopyPowerFlow", PF.RobustHomotopyPowerFlow, Dict{Symbol, Any}()),
    ("FastDecoupledFixed", PF.FastDecoupledFixed, Dict{Symbol, Any}()),
    ("FastDecoupledXB(tol=1e-2)", PF.FastDecoupledXB, Dict{Symbol, Any}(:tol => 1e-2)),
    ("FastDecoupledBX(tol=1e-2)",
        FD_BX,
        Dict{Symbol, Any}(:tol => 1e-2)),
]

# Rectangular Current-Injection (Da Costa) and Mixed Current-Power Balance (MCPB) formulations,
# each across plain NR, NR+Iwamoto, Trust Region, TR+Iwamoto fallback (MCPB adds a single LM run).
_RECT_CI_VARIANTS = [
    ("ACRectangularPowerFlow{NR}", PF.NewtonRaphsonACPowerFlow, Dict{Symbol, Any}()),
    ("ACRectangularPowerFlow{NR}(iwamoto)", PF.NewtonRaphsonACPowerFlow,
        Dict{Symbol, Any}(:iwamoto => true)),
    ("ACRectangularPowerFlow{TR}", PF.TrustRegionACPowerFlow, Dict{Symbol, Any}()),
    ("ACRectangularPowerFlow{TR}(iwamoto_fallback)", PF.TrustRegionACPowerFlow,
        Dict{Symbol, Any}(:iwamoto_fallback => true)),
]
_MIXED_CPB_VARIANTS = [
    ("ACMixedPowerFlow{NR}", PF.NewtonRaphsonACPowerFlow, Dict{Symbol, Any}()),
    ("ACMixedPowerFlow{NR}(iwamoto)", PF.NewtonRaphsonACPowerFlow,
        Dict{Symbol, Any}(:iwamoto => true)),
    ("ACMixedPowerFlow{TR}", PF.TrustRegionACPowerFlow, Dict{Symbol, Any}()),
    ("ACMixedPowerFlow{TR}(iwamoto_fallback)", PF.TrustRegionACPowerFlow,
        Dict{Symbol, Any}(:iwamoto_fallback => true)),
    ("ACMixedPowerFlow{LM}", PF.LevenbergMarquardtACPowerFlow, Dict{Symbol, Any}()),
]

dc_solvers = [
    (DCPowerFlow(; correct_bustypes = true), "DCPowerFlow"),
    (PTDFDCPowerFlow(; correct_bustypes = true), "PTDFDCPowerFlow"),
    (vPTDFDCPowerFlow(; correct_bustypes = true), "vPTDFDCPowerFlow"),
]

for (group, name) in systems
    sys = build_system(group, name)
    for (label, solver, settings) in polar_ac_solvers
        bench_ac!(name, label,
            () ->
                ACPowerFlow{solver}(; correct_bustypes = true, solver_settings = settings),
            sys)
    end
    for (label, solver, settings) in _RECT_CI_VARIANTS
        bench_ac!(name, label,
            () -> PF.ACRectangularPowerFlow{solver}(;
                correct_bustypes = true, solver_settings = settings),
            sys)
    end
    for (label, solver, settings) in _MIXED_CPB_VARIANTS
        bench_ac!(name, label,
            () -> PF.ACMixedPowerFlow{solver}(;
                correct_bustypes = true, solver_settings = settings),
            sys)
    end
    for (dc_pf, label) in dc_solvers
        bench_dc!(name, label, dc_pf, sys)
    end
end

# Large-scale validation system: synthetic Eastern Interconnect (~78k buses).
# Only the memory-light solvers are exercised here. PTDF/vPTDF build dense
# sensitivity matrices that exhaust RAM at this scale (~19 GB and >250 s for a
# single solve), and the Hessian-based RobustHomotopy/Rectangular/Mixed variants
# are likewise prohibitive; including them OOM-kills even a 34 GB machine. The
# restricted set (DC + Newton-Raphson + Trust Region + FastDecoupled) peaks near
# 6 GB and runs in about a minute. Set PF_PERF_SKIP_LARGE_SYSTEMS=true to skip on
# low-RAM runners.
large_systems = [
    (PSSEParsingTestSystems, "Base_Eastern_Interconnect_515GW"),
]
large_dc_solvers = [(DCPowerFlow(; correct_bustypes = true), "DCPowerFlow")]
# FastDecoupledXB (decoupled B′/B″) solves the EI's LCC via the sequential AC–DC method and
# converges at the default 1e-9 tolerance in ~18 iterations (the EI's low-|x| branches are
# near-zero-impedance with low r/x, which the decoupling tolerates). No tolerance relaxation needed
# here — unlike the stiffer ACTIVSg10k synthetic case above.
large_ac_solvers = [
    ("NewtonRaphsonACPowerFlow", PF.NewtonRaphsonACPowerFlow, Dict{Symbol, Any}()),
    ("TrustRegionACPowerFlow", PF.TrustRegionACPowerFlow, Dict{Symbol, Any}()),
    ("FastDecoupledFixed", PF.FastDecoupledFixed, Dict{Symbol, Any}()),
    ("FastDecoupledXB", PF.FastDecoupledXB, Dict{Symbol, Any}()),
    ("FastDecoupledBX",
        FD_BX, Dict{Symbol, Any}(),
    ),
]
if get(ENV, "PF_PERF_SKIP_LARGE_SYSTEMS", "false") != "true"
    for (group, name) in large_systems
        sys = build_system(group, name)
        for (dc_pf, label) in large_dc_solvers
            bench_dc!(name, label, dc_pf, sys)
        end
        for (label, solver, settings) in large_ac_solvers
            bench_ac!(name, label,
                () -> ACPowerFlow{solver}(;
                    correct_bustypes = true, solver_settings = settings),
                sys)
        end
    end
end

if !is_running_on_ci()
    println("Precompile time: $(precompile.time) s")
    csv_file = "solve_time_$(ARGS[1]).csv"
    if isfile(csv_file)
        function _category(label)
            occursin("ACRectangular", label) && return "Rectangular CI"
            occursin("ACMixed", label) && return "Mixed CPB"
            (occursin("DCPowerFlow", label) || occursin("PTDF", label)) &&
                return "DC"
            return "Polar AC"
        end
        order = ["Polar AC", "Rectangular CI", "Mixed CPB", "DC"]
        buckets = Dict(c => String[] for c in order)
        for line in eachline(csv_file)
            label = first(split(line, ","))
            # Drop the redundant "<system>-" prefix; group by formulation family.
            row = replace(line, "$(systems[1][2])-" => "")
            push!(buckets[_category(label)], row)
        end
        println("\nSolve times:")
        for cat in order
            isempty(buckets[cat]) && continue
            println("\n  [", cat, "]")
            for row in buckets[cat]
                println("\t", row)
            end
        end
    end
    pushed_to_args && pop!(ARGS)
end
