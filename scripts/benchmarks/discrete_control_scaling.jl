"""
Scaling benchmark for the discrete-control λ-continuation.

Measures the continuation's cost as the number of controlled devices N grows, using the
repo's robust metric — **counts, not wall-clock**:

  * inner_solves      — full `_solve_with_q_limits!` calls per continuation
  * symbolic_factors  — KLU/AA SYMBOLIC factorizations (target of PolarNRCache reuse:
                        should stay ~O(1) per continuation, NOT track inner_solves)
  * numeric_refactors — per-NR-iteration numeric refactorizations

This is the harness the P0→P3 performance work is measured against. Run it BEFORE and
AFTER each phase; the phase "worked" iff the relevant count drops as designed:

  P1 (persist symbolic factor):  symbolic_factors  : ~inner_solves  →  ~1
  P2 (adjoint sensitivities):    inner_solves       : drops by ~N (probe phase linearized)
  P3 (batched passes):           inner_solves       : stepping term becomes ~independent of N

Fixture: a star of K decoupled controlled feeders, each mirroring the proven
`_make_solvable_tap_shunt_system` test fixture (REF ─tap─ PQ-load, REF ─line─ PQ-shunt).
Decoupled so convergence is guaranteed and the per-device cost is clean to read; N = 2·K
devices, B = 1 + 2·K buses. (A meshed/realistic PSS/E fixture is a worthwhile refinement.)

The warm-start "measurement trap" (CLAUDE.md): an unperturbed re-solve converges in 0
iterations. Each rep therefore perturbs the feeder loads before solving, and we report the
median over reps.

Usage:
    julia --project=scripts/benchmarks scripts/benchmarks/discrete_control_scaling.jl
"""

using PowerFlows, PowerSystems
using Logging, Random, Statistics, Printf

const PSY = PowerSystems
const PF = PowerFlows

# One controlled feeder off the REF bus: REF ─tap─ PQ(load), REF ─line─ PQ(shunt).
# `k` indexes the feeder; bus numbers are offset so they never collide.
function _add_feeder!(sys::PSY.System, ref::PSY.ACBus, k::Int)
    b_load = PSY.ACBus(; number = 2k, name = "load_bus_$k", available = true,
        bustype = PSY.ACBusTypes.PQ, angle = 0.0, magnitude = 1.0,
        voltage_limits = (0.0, 2.0), base_voltage = 230.0)
    b_sh = PSY.ACBus(; number = 2k + 1, name = "shunt_bus_$k", available = true,
        bustype = PSY.ACBusTypes.PQ, angle = 0.0, magnitude = 1.0,
        voltage_limits = (0.0, 2.0), base_voltage = 230.0)
    PSY.add_component!(sys, b_load)
    PSY.add_component!(sys, b_sh)
    PSY.add_component!(sys,
        PSY.PowerLoad(; name = "load_$k", available = true, bus = b_load,
            active_power = 0.5, reactive_power = 0.25, base_power = 100.0,
            max_active_power = 100.0, max_reactive_power = 100.0))
    PSY.add_component!(sys,
        PSY.PowerLoad(; name = "shload_$k", available = true, bus = b_sh,
            active_power = 0.05, reactive_power = 0.025, base_power = 100.0,
            max_active_power = 100.0, max_reactive_power = 100.0))
    PSY.add_component!(sys,
        PSY.Line(; name = "line_$k", available = true, active_power_flow = 0.0,
            reactive_power_flow = 0.0, arc = PSY.Arc(; from = ref, to = b_sh),
            r = 1e-2, x = 1e-2, b = (from = 0.0, to = 0.0),
            rating = 10.0, angle_limits = (min = -pi / 2, max = pi / 2)))
    PSY.add_component!(sys,
        PSY.TapTransformer(; name = "tap_$k", available = true, active_power_flow = 0.0,
            reactive_power_flow = 0.0, arc = PSY.Arc(; from = ref, to = b_load),
            r = 0.01, x = 0.10, primary_shunt = 0.0 + 0.0im, tap = 1.0, rating = 1.0,
            base_power = 100.0,
            control_objective = PSY.TransformerControlObjective.VOLTAGE))
    PSY.add_component!(sys,
        PSY.SwitchedAdmittance(; name = "shunt_$k", available = true, bus = b_sh,
            Y = 0.0 + 0.0im, initial_status = [0], number_of_steps = [4],
            Y_increase = [0.0 + 0.05im], admittance_limits = (min = 0.9, max = 1.1)))
    return nothing
end

function build_controlled_system(K::Int)
    sys = PSY.System(100.0)
    ref = PSY.ACBus(; number = 1, name = "ref", available = true,
        bustype = PSY.ACBusTypes.REF, angle = 0.0, magnitude = 1.0,
        voltage_limits = (0.0, 2.0), base_voltage = 230.0)
    PSY.add_component!(sys, ref)
    PSY.add_component!(sys,
        PSY.Source(; name = "source", available = true, bus = ref,
            active_power = 0.0, reactive_power = 0.0, R_th = 0.0, X_th = 1e-5))
    for k in 1:K
        _add_feeder!(sys, ref, k)
    end
    return sys
end

# Perturb every feeder load so the next continuation does real work (defeats the
# 0-iteration warm-start early return).
function _perturb_loads!(sys::PSY.System, rng)
    for ld in PSY.get_components(PSY.PowerLoad, sys)
        base = PSY.get_active_power(ld)
        PSY.set_active_power!(ld, base * (1.0 + 0.1 * (rand(rng) - 0.5)))
    end
    return nothing
end

function run_size(K::Int; reps::Int = 3, seed::Int = 1)
    rng = Random.MersenneTwister(seed)
    inner = Int[]
    symb = Int[]
    numf = Int[]
    conv = Bool[]
    for _ in 1:reps
        with_logger(NullLogger()) do
            sys = build_controlled_system(K)
            _perturb_loads!(sys, rng)
            pf = PF.ACPowerFlow(; control_discrete_devices = true)
            data = PF.PowerFlowData(pf, sys)
            PF.solve_power_flow!(data)
            push!(conv, all(data.converged))
            push!(inner, PF.get_control_inner_solve_count(data))
            push!(symb, PF.get_control_symbolic_factor_count(data))
            push!(numf, PF.get_control_numeric_refactor_count(data))
        end
    end
    return (; K, N = 2K, B = 1 + 2K,
        converged = all(conv),
        inner = round(Int, median(inner)),
        symbolic = round(Int, median(symb)),
        numeric = round(Int, median(numf)))
end

function main()
    sizes = [1, 2, 4, 8, 16, 32]
    @printf("%4s %5s %6s  %5s  %10s  %10s  %11s\n",
        "K", "N", "B", "conv", "inner", "symbolic", "numeric")
    println("-"^62)
    for K in sizes
        r = run_size(K)
        @printf("%4d %5d %6d  %5s  %10d  %10d  %11d\n",
            r.K, r.N, r.B, r.converged, r.inner, r.symbolic, r.numeric)
    end
    println("\nRead: `symbolic` ≈ `inner` is the pre-P1 baseline; P1 drives it toward ~1.")
    println("`inner` growth vs N shows the probe+stepping cost P2/P3 target.")
    return nothing
end

main()
