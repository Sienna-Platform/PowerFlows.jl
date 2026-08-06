# Convergence testbed: take a known-good base system, perturb it into the kinds
# of operating points a PCM hand-off can produce (generator outages, load swings,
# gen/load imbalance, transmission outages), and see whether the AC power flow
# still converges.
#
# The point is to have a reproducible, knob-by-knob stand-in for the real PCM
# setpoints so convergence-recovery strategies (and, later, the residual/condition
# diagnostics on this branch) can be validated against scenarios whose failure
# mode you already understand.
#
# Usage from a Julia REPL with PowerFlows + PowerSystems + PSB available:
#
#   include("scripts/convergence_testbed.jl")
#   base = build_base_system()                 # build once, deepcopy per scenario
#   results = run_all(default_scenarios(base); base = base)
#   print_summary(results)
#
#   # Single scenario, with the per-iteration diagnostics turned on:
#   r = run_scenario(scale_all_loads(1.4), base; log_diagnostics = true)
#
#   # Roll your own:
#   run_scenario(generator_outage(; n = 3), base)
#
# Every scenario runs against a fresh deepcopy of `base`, so they're independent
# and order doesn't matter.

using PowerFlows
using PowerSystems
using PowerSystemCaseBuilder
import LinearAlgebra

const PSY = PowerSystems
const PF = PowerFlows
const PSB = PowerSystemCaseBuilder

# ---------------------------------------------------------------------------
# Base system
# ---------------------------------------------------------------------------

"""
    build_base_system(; name) -> PSY.System

Build the test system once. Default is the 2000-bus synthetic ACTIVSg case.
Build this a single time and `deepcopy` it per scenario (see `run_scenario`);
re-`build_system`-ing per scenario is far slower and buys nothing.
"""
function build_base_system(; name::String = "matpower_ACTIVSg2000_sys")
    return PSB.build_system(PSB.MatpowerTestSystems, name)
end

# ---------------------------------------------------------------------------
# Scenario abstraction
# ---------------------------------------------------------------------------

"""
A named perturbation. `apply!` mutates a system in place and returns a short
string describing what it actually did (e.g. which units it tripped), which gets
folded into the result for the summary table.
"""
struct Scenario
    name::String
    description::String
    apply!::Function   # (sys::PSY.System) -> String (detail of what was applied)
end

Scenario(name, description, f::Function) = Scenario(String(name), String(description), f)

# `do`-block form: `Scenario(name, desc) do sys ... end` lowers to passing the
# closure as the *first* argument, so accept that ordering too.
Scenario(f::Function, name, description) = Scenario(String(name), String(description), f)

# ---------------------------------------------------------------------------
# Component helpers
# ---------------------------------------------------------------------------

# Reference (slack) bus numbers — never trip generation sitting on these, or the
# "failure" is just "I deleted the slack," which tells you nothing.
function _reference_bus_numbers(sys::PSY.System)
    nums = Set{Int}()
    for b in PSY.get_components(PSY.ACBus, sys)
        PSY.get_bustype(b) == PSY.ACBusTypes.REF && push!(nums, PSY.get_number(b))
    end
    return nums
end

# Online generators (any Generator subtype) not sitting on a reference bus,
# sorted by active power descending so "trip the n largest" is well defined.
function _online_generators(sys::PSY.System)
    refs = _reference_bus_numbers(sys)
    gens = PSY.Generator[]
    for g in PSY.get_components(PSY.Generator, sys)
        PSY.get_available(g) || continue
        PSY.get_number(PSY.get_bus(g)) in refs && continue
        push!(gens, g)
    end
    sort!(gens; by = PSY.get_active_power, rev = true)
    return gens
end

# `StaticLoad` (not `ElectricLoad`) deliberately: it excludes `FixedAdmittance`,
# which is a shunt with no P/Q setpoint to scale.
_online_loads(sys::PSY.System) =
    collect(PSY.get_components(x -> PSY.get_available(x), PSY.StaticLoad, sys))

# Total active power of a load. `StandardLoad` has no scalar P — it splits into
# constant/impedance/current ZIP components — so sum them; everything else
# (PowerLoad, ExponentialLoad, …) exposes a plain `get_active_power`.
_load_active_power(l::PSY.ElectricLoad) = PSY.get_active_power(l)
_load_active_power(l::PSY.StandardLoad) =
    PSY.get_constant_active_power(l) +
    PSY.get_impedance_active_power(l) +
    PSY.get_current_active_power(l)

# Scale both P and Q of a load in place by `factor`. ZIP-aware for StandardLoad.
function _scale_load!(l::PSY.ElectricLoad, factor::Real)
    PSY.set_active_power!(l, PSY.get_active_power(l) * factor)
    PSY.set_reactive_power!(l, PSY.get_reactive_power(l) * factor)
    return
end
function _scale_load!(l::PSY.StandardLoad, factor::Real)
    PSY.set_constant_active_power!(l, PSY.get_constant_active_power(l) * factor)
    PSY.set_constant_reactive_power!(l, PSY.get_constant_reactive_power(l) * factor)
    PSY.set_impedance_active_power!(l, PSY.get_impedance_active_power(l) * factor)
    PSY.set_impedance_reactive_power!(l, PSY.get_impedance_reactive_power(l) * factor)
    PSY.set_current_active_power!(l, PSY.get_current_active_power(l) * factor)
    PSY.set_current_reactive_power!(l, PSY.get_current_reactive_power(l) * factor)
    return
end

# Scale only active power (used by the imbalance scenario). ZIP-aware.
function _scale_load_active!(l::PSY.ElectricLoad, factor::Real)
    PSY.set_active_power!(l, PSY.get_active_power(l) * factor)
    return
end
function _scale_load_active!(l::PSY.StandardLoad, factor::Real)
    PSY.set_constant_active_power!(l, PSY.get_constant_active_power(l) * factor)
    PSY.set_impedance_active_power!(l, PSY.get_impedance_active_power(l) * factor)
    PSY.set_current_active_power!(l, PSY.get_current_active_power(l) * factor)
    return
end

# AC lines that are in service. Transformers are excluded so a "line outage"
# doesn't silently island a whole voltage level by yanking a tie transformer.
function _online_lines(sys::PSY.System)
    lines = PSY.Line[]
    for l in PSY.get_components(PSY.Line, sys)
        PSY.get_available(l) && push!(lines, l)
    end
    return lines
end

# ---------------------------------------------------------------------------
# Scenario library
# ---------------------------------------------------------------------------
#
# Each `apply!` returns a machine-parseable `detail` NamedTuple with a uniform
# shape: `(; kind::Symbol, buses::Vector{Int}, params::NamedTuple)`. `buses` always
# holds the directly-perturbed bus numbers (empty for system-wide scenarios), so a
# caller can pull the affected buses the same way regardless of scenario `kind`.
# `params` carries scenario-specific scalars/names. `_detail_str` renders it for
# the human-facing summary table.

# Bus number a static-injection component sits at.
_bus_no(c) = PSY.get_number(PSY.get_bus(c))

"""Compact human-readable rendering of a structured `detail` NamedTuple."""
function _detail_str(d::NamedTuple)
    d.kind === :baseline && return "no change"
    d.kind === :gen_outage &&
        return "tripped $(length(d.buses)) gen(s) $(d.params.names), P_lost=$(d.params.p_lost) p.u."
    d.kind === :load_scale &&
        return "scaled $(d.params.n_loads) load(s) by $(d.params.factor)"
    d.kind === :load_spike &&
        return "spiked $(length(d.buses)) load(s) $(d.params.names) by $(d.params.factor)"
    d.kind === :imbalance &&
        return "load P $(d.params.base_p) → $(d.params.new_p) p.u. (slack absorbs $(d.params.gap))"
    d.kind === :line_outage &&
        return "tripped $(length(d.params.names)) line(s) $(d.params.names)"
    return string(d)
end
_detail_str(s::AbstractString) = s   # tolerate a plain string (e.g. apply! errored)

"""Identity perturbation — sanity check that the base case converges."""
baseline() =
    Scenario("baseline", "unmodified base system",
        _ -> (; kind = :baseline, buses = Int[], params = (;)))

"""
    generator_outage(; n = 1) -> Scenario

Trip the `n` largest non-slack online generators (set them unavailable) without
redispatching. The lost MW have to be picked up by the slack bus / Q-limited PV
buses — exactly the "a unit dropped and nothing rebalanced it" case.
"""
function generator_outage(; n::Int = 1)
    name = n == 1 ? "gen_outage_1" : "gen_outage_$n"
    return Scenario(name, "trip $n largest non-slack generator(s)") do sys
        gens = _online_generators(sys)
        tripped = gens[1:min(n, length(gens))]
        for g in tripped
            PSY.set_available!(g, false)
        end
        mw = round(sum(PSY.get_active_power, tripped; init = 0.0); sigdigits = 4)
        (;
            kind = :gen_outage,
            buses = _bus_no.(tripped),
            params = (; names = PSY.get_name.(tripped), p_lost = mw),
        )
    end
end

"""
    scale_all_loads(factor) -> Scenario

Multiply every load's P and Q by `factor` with no matching change in generation.
factor > 1 stresses voltage support and leans on the slack; large enough factors
push the case past the nose of the PV curve and it stops converging.
"""
function scale_all_loads(factor::Real)
    return Scenario("load_x$(factor)", "scale all loads (P and Q) by $factor") do sys
        loads = _online_loads(sys)
        for l in loads
            _scale_load!(l, factor)
        end
        # System-wide: no single set of "perturbed" buses.
        (; kind = :load_scale, buses = Int[], params = (; factor, n_loads = length(loads)))
    end
end

"""
    load_spike(; n_buses = 5, factor = 3.0) -> Scenario

Localized stress: multiply P and Q by `factor` on the `n_buses` largest loads
only. Models a concentrated demand surge rather than a uniform system-wide swing.
"""
function load_spike(; n_buses::Int = 5, factor::Real = 3.0)
    return Scenario(
        "load_spike_$(n_buses)x$(factor)",
        "spike P/Q by $factor on the $n_buses largest loads",
    ) do sys
        loads = _online_loads(sys)
        sort!(loads; by = _load_active_power, rev = true)
        hit = loads[1:min(n_buses, length(loads))]
        for l in hit
            _scale_load!(l, factor)
        end
        (;
            kind = :load_spike,
            buses = _bus_no.(hit),
            params = (; names = PSY.get_name.(hit), factor),
        )
    end
end

"""
    gen_load_imbalance(load_factor) -> Scenario

Scale loads by `load_factor` but freeze generation setpoints, so the entire
mismatch lands on the slack bus. This is the canonical "PCM dispatch doesn't sum
to load" hand-off problem; small imbalances converge with a strained slack, big
ones don't.
"""
function gen_load_imbalance(load_factor::Real)
    return Scenario(
        "imbalance_load_x$(load_factor)",
        "scale loads by $load_factor, generation frozen (mismatch → slack)",
    ) do sys
        loads = _online_loads(sys)
        base_p = sum(_load_active_power, loads; init = 0.0)
        for l in loads
            _scale_load_active!(l, load_factor)
        end
        new_p = sum(_load_active_power, loads; init = 0.0)
        (;
            kind = :imbalance,
            buses = Int[],   # system-wide; mismatch lands on the slack
            params = (;
                factor = load_factor,
                base_p = round(base_p; sigdigits = 4),
                new_p = round(new_p; sigdigits = 4),
                gap = round(new_p - base_p; sigdigits = 4),
            ),
        )
    end
end

"""
    line_outage(; n = 1) -> Scenario

Trip the first `n` in-service AC lines (set unavailable). Transformers are left
alone so this doesn't accidentally island a voltage level. With `n` large this is
a coarse stand-in for the transmission-failure scenarios coming later; even `n=1`
can disconnect a radial tail and prevent convergence.
"""
function line_outage(; n::Int = 1)
    return Scenario("line_outage_$n", "trip first $n in-service AC line(s)") do sys
        lines = _online_lines(sys)
        tripped = lines[1:min(n, length(lines))]
        for l in tripped
            PSY.set_available!(l, false)
        end
        endpoints = unique(
            Iterators.flatten(
                (PSY.get_number(PSY.get_from(PSY.get_arc(l))),
                    PSY.get_number(PSY.get_to(PSY.get_arc(l)))) for l in tripped
            ),
        )
        (;
            kind = :line_outage,
            buses = collect(endpoints),
            params = (; names = PSY.get_name.(tripped)),
        )
    end
end

"""Default scenario sweep covering each failure mode plus a stress ramp."""
function default_scenarios(::PSY.System = build_base_system())
    return Scenario[
        baseline(),
        generator_outage(; n = 1),
        generator_outage(; n = 5),
        scale_all_loads(1.1),
        scale_all_loads(1.4),
        scale_all_loads(1.8),
        load_spike(; n_buses = 5, factor = 4.0),
        gen_load_imbalance(1.05),
        gen_load_imbalance(1.5),
        line_outage(; n = 1),
        line_outage(; n = 10),
    ]
end

# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

# ∞-norm of the polar residual at the as-built starting point. A cheap "how far
# from balanced is this operating point before we even start iterating" metric;
# returns NaN if anything about building the residual throws.
function _initial_residual_inf(data::PF.ACPowerFlowData)
    try
        residual = PF.ACPowerFlowResidual(data, 1)
        x0 = PF.calculate_x0(data, 1)
        residual(x0, 1)
        return maximum(abs, residual.Rv)
    catch
        return NaN
    end
end

"""
    run_scenario(scn, base; solver, log_diagnostics, bail_on_fold, localize) -> NamedTuple

Deepcopy `base`, apply `scn`, and solve the AC power flow. Returns
`(; name, detail, converged, init_resid_inf, bottleneck, data, sys)`. `detail` is
the structured `(; kind, buses, params)` NamedTuple the scenario's `apply!`
returns — `detail.buses` is the directly-perturbed bus numbers. `data`/`sys` are
kept so a follow-up diagnostic pass (e.g. `classify_residual_nans`,
`report_residual_at_x0`) can be pointed straight at a failing scenario.

`bail_on_fold = true` turns on the eigenvalue-sign-switch bail-out: the solver
stops the moment `λ_min` of the bus-voltage Schur complement crosses zero (a fold /
voltage-collapse signature) rather than grinding to max iterations. This also
leaves `data` sitting at the *fold iterate* — the operating point where the
collapse mode is meaningful.

`localize = true` runs `PF.localize_bottleneck` on a failed scenario and returns it
as `bottleneck`. It forces `bail_on_fold = true`, because the localization is only
meaningful at the fold iterate — left to grind for 50 iterations the solver ends at
a blown-up point whose smallest singular value is a structural artifact, not the
scenario's collapse mode. `bottleneck` is `nothing` when the scenario converged or
`localize = false`.
"""
function run_scenario(
    scn::Scenario,
    base::PSY.System;
    solver = NewtonRaphsonACPowerFlow,
    log_diagnostics::Bool = false,
    bail_on_fold::Bool = false,
    localize::Bool = false,
)
    bail_on_fold = bail_on_fold || localize   # localization needs the fold iterate
    sys = deepcopy(base)
    detail = ""
    try
        detail = scn.apply!(sys)
    catch err
        @warn "scenario apply! failed" scenario = scn.name exception = err
        detail = "apply! errored: $err"
    end

    pf = ACPowerFlow{solver}(;
        correct_bustypes = true,
        log_solver_diagnostics = log_diagnostics,
        solver_settings = Dict{Symbol, Any}(:stop_at_fold => bail_on_fold),
    )
    data = PF.PowerFlowData(pf, sys)
    init_resid = _initial_residual_inf(data)

    converged = false
    try
        converged = PF.solve_power_flow!(data)
    catch err
        @warn "solve threw" scenario = scn.name exception = err
    end

    bottleneck = nothing
    if localize && !converged
        try
            bottleneck = PF.localize_bottleneck(data, sys)
        catch err
            @warn "localize_bottleneck failed" scenario = scn.name exception = err
        end
    end

    return (;
        name = scn.name,
        detail = detail,
        converged = converged,
        init_resid_inf = init_resid,
        bottleneck = bottleneck,
        data = data,
        sys = sys,
    )
end

"""Run a list of scenarios against the same base system. See `run_scenario`."""
function run_all(
    scenarios::Vector{Scenario},
    base::PSY.System = build_base_system();
    solver = NewtonRaphsonACPowerFlow,
    log_diagnostics::Bool = false,
    bail_on_fold::Bool = false,
    localize::Bool = false,
)
    results = NamedTuple[]
    for scn in scenarios
        @info "=== scenario: $(scn.name) — $(scn.description) ==="
        push!(
            results,
            run_scenario(
                scn, base;
                solver = solver,
                log_diagnostics = log_diagnostics,
                bail_on_fold = bail_on_fold,
                localize = localize,
            ),
        )
    end
    return results
end

"""Pretty-print the convergence outcome of a `run_all` result set."""
function print_summary(results)
    println("\n", "="^78)
    println(rpad("scenario", 26), rpad("converged", 11), rpad("‖F(x0)‖∞", 13), "detail")
    println("-"^78)
    for r in results
        flag = r.converged ? "yes" : "NO"
        ir =
            isnan(r.init_resid_inf) ? "n/a" : string(round(r.init_resid_inf; sigdigits = 4))
        println(rpad(r.name, 26), rpad(flag, 11), rpad(ir, 13), _detail_str(r.detail))
    end
    println("="^78)
    n_conv = count(r -> r.converged, results)
    println("$n_conv / $(length(results)) scenarios converged.\n")
    return
end
