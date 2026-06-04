# Power Flow Overview for Developers

```@meta
CurrentModule = PowerFlows
```

`PowerFlows.jl` solves steady-state power flow on a [`PowerSystems.System`](@extref).
The public API is built around [`PowerFlowEvaluationModel`](@ref) values (what problem
to solve) and, for AC cases, an [`ACPowerFlowSolverType`](@ref) (how to solve it).
See [Evaluation Models vs. Solver Algorithms](@ref) for that split and for the three
AC formulations ([`ACPolarPowerFlow`](@ref), [`ACRectangularPowerFlow`](@ref),
[`ACMixedPowerFlow`](@ref)).

AC solves use sparse Jacobians factorized with [KLU.jl](https://github.com/JuliaSparse/KLU.jl).
The initial guess comes from the current bus setpoints in `sys`, optionally adjusted
by enhanced flat start and related options on the evaluation model.

## Two standalone usage modes

Most callers use one of the following patterns.

### 1. Solve and return results (`solve_power_flow`)

[`solve_power_flow`](@ref) reads injections and setpoints from `sys`, runs the
selected evaluation model, and returns a `Dict` of `DataFrame`s (`"bus_results"`,
`"flow_results"`, and `"lcc_results"` when LCC HVDC is present). The `System` is
not modified.

```@example dev_pf
using PowerFlows
using PowerSystems
using PowerSystemCaseBuilder

sys = build_system(PSITestSystems, "c_sys14"; runchecks = false)
pf = ACPowerFlow()
```

```@example dev_pf
results = solve_power_flow(pf, sys)
results["bus_results"]
```

For DC formulations, pass a [`FlowReporting`](@ref) mode so branch flows use the
same arc basis as the rest of the package:

```@example dev_pf
keys(solve_power_flow(DCPowerFlow(), sys, FlowReporting.ARC_FLOWS))
```

### 2. Solve and write back into the system (`solve_and_store_power_flow!`)

[`solve_and_store_power_flow!`](@ref) solves the same problem and, on success,
updates bus voltages, branch flows, and generator setpoints in `sys`. It returns
`true` or `false` for convergence — useful in scripts and validation loops.

```@example dev_pf
converged = solve_and_store_power_flow!(pf, sys)
```

Solver tolerances and iteration limits can be passed as keyword arguments (for example
`tol`, `maxIterations`); configuration such as `time_steps`, `network_reductions`,
and `correct_bustypes` belongs on the evaluation model constructor.

Typical uses: initializing a case before export, checking AC feasibility after a
scheduling step, or batch validation when you already have a `System` in memory.

## Capabilities relevant to developers

  - **AC formulations** — polar power balance (default), Da Costa rectangular current
    injection, and mixed current–power balance ([`ACMixedPowerFlow`](@ref)).
  - **AC solvers** — Newton–Raphson, trust region, Levenberg–Marquardt; robust homotopy
    and gradient descent on polar only.
  - **DC** — bus-angle DC, PTDF, and virtual PTDF; multi-period DC is supported.
  - **HVDC** — LCC line-commutated converters on all three AC formulations; VSC/HVDC
    models per package tests and docs.
  - **Post-processing** — optional reactive-power limit enforcement, PSS/e export via
    [`PSSEExportPowerFlow`](@ref), loss and voltage-stability factors (polar only).

For formulation and solver selection at scale, see
[How to choose an AC formulation and solver](@ref choose-ac-formulation-and-solver).

## Power flow in the loop ([PowerSimulations.jl](https://sienna-platform.github.io/PowerSimulations.jl/stable/))

There is a third integration pattern that does **not** call `solve_power_flow` from
user code directly, but still exercises the same solvers inside `PowerFlows.jl`.

In production-cost, unit commitment, and economic-dispatch workflows,
[PowerSimulations.jl](https://sienna-platform.github.io/PowerSimulations.jl/stable/)
(PSI) can run an AC (or DC) power
flow **after each optimization interval** while a simulation is executing. This is
often called *power flow in the loop* (or *in-the-loop* PF).

### How PSI wires it

 1. You pass a [`PowerFlowEvaluationModel`](@ref) (for example
    [`ACPowerFlow`](@ref) with an optional [`PSSEExportPowerFlow`](@ref) exporter)
    into a PSI [`NetworkModel`](@extref PowerSimulations.NetworkModel) via the
    `power_flow_evaluation` keyword.
 2. During `build!` / `execute!`, PSI constructs a [`PowerFlowData`](@ref) container
    (see `make_power_flow_container` in the source) and maps UC/ED decision variables
    (dispatch, load, storage, etc.) into PF injection data for each time step.
 3. At the appropriate point in the simulation, PSI calls `solve_power_flow!` on
    that container — the same in-place solve used internally by standalone AC
    solves — then copies voltages, angles, and branch flows into **auxiliary variables**
    on the optimization results (for example `PowerFlowVoltageMagnitude__ACBus`,
    `PowerFlowBranchActivePowerFromTo__Line`).

So from a `PowerFlows.jl` maintainer's perspective, in-the-loop usage is still
`PowerFlowEvaluationModel` + `solve_power_flow!` on `PowerFlowData`; PSI owns the
scheduling, input mapping, and exposure of results to the simulation interface.

### Why it matters for PF development

  - Changes to residual/Jacobian setup, multi-period behavior, LCC handling, or
    convergence reporting can surface in PSI simulations even when standalone
    `solve_power_flow` tests still pass.
  - PSI currently rejects some PF options in this path (for example network
    reductions on in-the-loop containers); see `check_network_reduction` in
    PowerSimulations.
  - The UC stage still optimizes over its own network model (commonly PTDF); the
    in-loop AC solve is a **post-optimization evaluation** on the committed dispatch,
    not a feedback loop into the MILP/LP. PSI `use_slacks` on the network model
    refers to slack variables in the **optimization** formulation, not to AC/PTDF
    reconciliation.

### Learn more in PowerSimulations

PSI documents the end-to-end workflow (template setup, auxiliary variables, comparing
PTDF UC flows to AC in-loop flows) in its tutorial
[Running Power Flow In The Loop with Unit Commitment](@extref uc-inloop-pf).

Standalone PF examples and formulation trade-offs remain in this package's
[Tutorials](@ref) and [Explanation](@ref) sections.

## Related reading

  - [Evaluation Models vs. Solver Algorithms](@ref)
  - [Mixed Current-Power Balance Formulation](@ref)
  - [LCC Model Implementation](@ref)
  - [Public API Reference](@ref) — evaluation models and `solve_*` entry points
