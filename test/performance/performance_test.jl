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

solvers = [PF.NewtonRaphsonACPowerFlow, PF.RobustHomotopyPowerFlow]
for (group, name) in systems
    for solver in solvers
        sys = build_system(group, name)
        try
            pf = ACPowerFlow{solver}(; correct_bustypes = true)
            pf_data = PF.PowerFlowData(pf, sys)
            _, time_solve_1, _, _ = @timed PF.solve_power_flow!(pf_data; pf = pf)
            record_time("$(name)-$(solver) First Solve", time_solve_1)
            pf = ACPowerFlow{solver}(; correct_bustypes = true)
            pf_data = PF.PowerFlowData(pf, sys)
            _, time_solve_2, _, _ = @timed PF.solve_power_flow!(pf_data; pf = pf)
            record_time("$(name)-$(solver) Second Solve", time_solve_2)
        catch e
            @error exception = (e, catch_backtrace())
            record_failure("$(name)-$(solver) Solve")
        end
    end
end

# Rectangular Current-Injection (Da Costa) NR — augmented current-injection
# formulation; key benefit is constant Y_bus off-diagonal Jacobian blocks.
# Tested with all four step strategies: plain NR, NR+Iwamoto, Trust Region,
# Trust Region + Iwamoto fallback.
const _RECT_CI_VARIANTS = [
    ("ACRectangularPowerFlow{NR}", PF.NewtonRaphsonACPowerFlow,
        Dict{Symbol, Any}()),
    ("ACRectangularPowerFlow{NR}(iwamoto)", PF.NewtonRaphsonACPowerFlow,
        Dict{Symbol, Any}(:iwamoto => true)),
    ("ACRectangularPowerFlow{TR}", PF.TrustRegionACPowerFlow,
        Dict{Symbol, Any}()),
    ("ACRectangularPowerFlow{TR}(iwamoto_fallback)", PF.TrustRegionACPowerFlow,
        Dict{Symbol, Any}(:iwamoto_fallback => true)),
]
for (group, name) in systems
    sys = build_system(group, name)
    for (solver_label, solver, extra_settings) in _RECT_CI_VARIANTS
        try
            pf = PF.ACRectangularPowerFlow{solver}(;
                correct_bustypes = true,
                solver_settings = extra_settings)
            pf_data = PF.PowerFlowData(pf, sys)
            _, time_solve_1, _, _ = @timed PF.solve_power_flow!(pf_data; pf = pf)
            record_time("$(name)-$(solver_label) First Solve", time_solve_1)
            pf = PF.ACRectangularPowerFlow{solver}(;
                correct_bustypes = true,
                solver_settings = extra_settings)
            pf_data = PF.PowerFlowData(pf, sys)
            _, time_solve_2, _, _ = @timed PF.solve_power_flow!(pf_data; pf = pf)
            record_time("$(name)-$(solver_label) Second Solve", time_solve_2)
        catch e
            @error exception = (e, catch_backtrace())
            record_failure("$(name)-$(solver_label) Solve")
        end
    end
end

# Mixed Current-Power Balance (MCPB) formulation — current-balance equations
# at PQ buses, power-balance at PV/REF, augmented generator P/Q states. Mirrors
# the rectangular CI variant matrix (plain NR, NR+Iwamoto, Trust Region,
# Trust Region + Iwamoto fallback) plus a single Levenberg-Marquardt run.
const _MIXED_CPB_VARIANTS = [
    ("ACMixedPowerFlow{NR}", PF.NewtonRaphsonACPowerFlow,
        Dict{Symbol, Any}()),
    ("ACMixedPowerFlow{NR}(iwamoto)", PF.NewtonRaphsonACPowerFlow,
        Dict{Symbol, Any}(:iwamoto => true)),
    ("ACMixedPowerFlow{TR}", PF.TrustRegionACPowerFlow,
        Dict{Symbol, Any}()),
    ("ACMixedPowerFlow{TR}(iwamoto_fallback)", PF.TrustRegionACPowerFlow,
        Dict{Symbol, Any}(:iwamoto_fallback => true)),
    ("ACMixedPowerFlow{LM}", PF.LevenbergMarquardtACPowerFlow,
        Dict{Symbol, Any}()),
]
for (group, name) in systems
    sys = build_system(group, name)
    for (solver_label, solver, extra_settings) in _MIXED_CPB_VARIANTS
        try
            pf = PF.ACMixedPowerFlow{solver}(;
                correct_bustypes = true,
                solver_settings = extra_settings)
            pf_data = PF.PowerFlowData(pf, sys)
            _, time_solve_1, _, _ = @timed PF.solve_power_flow!(pf_data; pf = pf)
            record_time("$(name)-$(solver_label) First Solve", time_solve_1)
            pf = PF.ACMixedPowerFlow{solver}(;
                correct_bustypes = true,
                solver_settings = extra_settings)
            pf_data = PF.PowerFlowData(pf, sys)
            _, time_solve_2, _, _ = @timed PF.solve_power_flow!(pf_data; pf = pf)
            record_time("$(name)-$(solver_label) Second Solve", time_solve_2)
        catch e
            @error exception = (e, catch_backtrace())
            record_failure("$(name)-$(solver_label) Solve")
        end
    end
end

# Iwamoto step control (NR variant with damping)
for (group, name) in systems
    sys = build_system(group, name)
    solver_label = "NewtonRaphsonACPowerFlow(iwamoto)"
    try
        pf = ACPowerFlow{PF.NewtonRaphsonACPowerFlow}(;
            correct_bustypes = true,
            solver_settings = Dict{Symbol, Any}(:iwamoto => true))
        pf_data = PF.PowerFlowData(pf, sys)
        _, time_solve_1, _, _ = @timed PF.solve_power_flow!(pf_data; pf = pf)
        record_time("$(name)-$(solver_label) First Solve", time_solve_1)
        pf = ACPowerFlow{PF.NewtonRaphsonACPowerFlow}(;
            correct_bustypes = true,
            solver_settings = Dict{Symbol, Any}(:iwamoto => true))
        pf_data = PF.PowerFlowData(pf, sys)
        _, time_solve_2, _, _ = @timed PF.solve_power_flow!(pf_data; pf = pf)
        record_time("$(name)-$(solver_label) Second Solve", time_solve_2)
    catch e
        @error exception = (e, catch_backtrace())
        record_failure("$(name)-$(solver_label)")
    end
end

# Trust Region with Iwamoto step control
for (group, name) in systems
    sys = build_system(group, name)
    solver_label = "TrustRegionACPowerFlow(iwamoto)"
    try
        pf = ACPowerFlow{PF.TrustRegionACPowerFlow}(;
            correct_bustypes = true,
            solver_settings = Dict{Symbol, Any}(:iwamoto => true))
        pf_data = PF.PowerFlowData(pf, sys)
        _, time_solve_1, _, _ = @timed PF.solve_power_flow!(pf_data; pf = pf)
        record_time("$(name)-$(solver_label) First Solve", time_solve_1)
        pf = ACPowerFlow{PF.TrustRegionACPowerFlow}(;
            correct_bustypes = true,
            solver_settings = Dict{Symbol, Any}(:iwamoto => true))
        pf_data = PF.PowerFlowData(pf, sys)
        _, time_solve_2, _, _ = @timed PF.solve_power_flow!(pf_data; pf = pf)
        record_time("$(name)-$(solver_label) Second Solve", time_solve_2)
    catch e
        @error exception = (e, catch_backtrace())
        record_failure("$(name)-$(solver_label)")
    end
end

# DC Power Flow solvers
dc_solvers = [
    (DCPowerFlow(; correct_bustypes = true), "DCPowerFlow"),
    (PTDFDCPowerFlow(; correct_bustypes = true), "PTDFDCPowerFlow"),
    (vPTDFDCPowerFlow(; correct_bustypes = true), "vPTDFDCPowerFlow"),
]
for (group, name) in systems
    sys = build_system(group, name)
    for (dc_pf, solver_label) in dc_solvers
        try
            pf_data = PF.PowerFlowData(dc_pf, sys)
            _, time_solve_1, _, _ = @timed PF.solve_power_flow!(pf_data)
            record_time("$(name)-$(solver_label) First Solve", time_solve_1)
            pf_data = PF.PowerFlowData(dc_pf, sys)
            _, time_solve_2, _, _ = @timed PF.solve_power_flow!(pf_data)
            record_time("$(name)-$(solver_label) Second Solve", time_solve_2)
        catch e
            @error exception = (e, catch_backtrace())
            record_failure("$(name)-$(solver_label)")
        end
    end
end

# Large-scale validation system: synthetic Eastern Interconnect (~78k buses).
# Only the memory-light solvers are exercised here. PTDF/vPTDF build dense
# sensitivity matrices that exhaust RAM at this scale (~19 GB and >250 s for a
# single solve), and the Hessian-based RobustHomotopy/Rectangular/Mixed variants
# are likewise prohibitive; including them OOM-kills even a 34 GB machine. The
# restricted set (DC + Newton-Raphson + Trust Region) peaks near 6 GB and runs
# in about a minute. Set PF_PERF_SKIP_LARGE_SYSTEMS=true to skip on low-RAM
# runners.
large_systems = [
    (PSSEParsingTestSystems, "Base_Eastern_Interconnect_515GW"),
]
large_dc_solvers = [(DCPowerFlow(; correct_bustypes = true), "DCPowerFlow")]
large_ac_solvers = [PF.NewtonRaphsonACPowerFlow, PF.TrustRegionACPowerFlow]
if get(ENV, "PF_PERF_SKIP_LARGE_SYSTEMS", "false") != "true"
    for (group, name) in large_systems
        sys = build_system(group, name)
        for (dc_pf, solver_label) in large_dc_solvers
            try
                pf_data = PF.PowerFlowData(dc_pf, sys)
                _, time_solve_1, _, _ = @timed PF.solve_power_flow!(pf_data)
                record_time("$(name)-$(solver_label) First Solve", time_solve_1)
                pf_data = PF.PowerFlowData(dc_pf, sys)
                _, time_solve_2, _, _ = @timed PF.solve_power_flow!(pf_data)
                record_time("$(name)-$(solver_label) Second Solve", time_solve_2)
            catch e
                @error exception = (e, catch_backtrace())
                record_failure("$(name)-$(solver_label)")
            end
        end
        for solver in large_ac_solvers
            try
                pf = ACPowerFlow{solver}(; correct_bustypes = true)
                pf_data = PF.PowerFlowData(pf, sys)
                _, time_solve_1, _, _ = @timed PF.solve_power_flow!(pf_data; pf = pf)
                record_time("$(name)-$(solver) First Solve", time_solve_1)
                pf = ACPowerFlow{solver}(; correct_bustypes = true)
                pf_data = PF.PowerFlowData(pf, sys)
                _, time_solve_2, _, _ = @timed PF.solve_power_flow!(pf_data; pf = pf)
                record_time("$(name)-$(solver) Second Solve", time_solve_2)
            catch e
                @error exception = (e, catch_backtrace())
                record_failure("$(name)-$(solver) Solve")
            end
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
