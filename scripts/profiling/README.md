# Power Flow Solver Profiling

`profile_power_flow_solvers.jl` performs phase- and component-attributed
profiling of the PowerFlows AC/DC solvers on the large
`matpower_ACTIVSg10k_sys` (~10k buses), comparing the **KLU** and
**AppleAccelerate** linear-solver backends to locate performance hot spots.

AC solves run from a **flat start** (PQ |V|→1.0, all θ→0); the data is built once
and re-flat-started before each solve. `enhanced_flat_start` (the default) is kept
so the solve converges realistically, but starting from flat still forces many
more Newton iterations than the system's near-solution warm start. The full-solve
timing and the deep profile therefore reflect the **iteration** work, not the
one-time data build. (DC is a single linear solve, so it is profiled as-is.)

For each `(solver, backend)` it reports:

- **Phases:** PowerFlowData build, residual build, Jacobian build, full
  solve-from-flat.
- **Per-iteration components** (AC): residual evaluation, Jacobian assembly,
  `numeric_refactor!`, triangular `solve!`. The last two isolate where the KLU
  and AppleAccelerate backends differ.

It then re-runs each full solve under `Profile.@profile` and writes flat + tree
text reports (`pf_profile_<label>.flat.txt` / `.tree.txt`) and, when `PProf` is
available, a `pf_profile_<label>.pb.gz` flamegraph to this directory.

Solvers covered: DC, AC Newton-Raphson, AC Trust Region.
Backends: KLU everywhere; AppleAccelerate additionally on Apple platforms.

## Run command

From the repository root:

```
julia --project=test scripts/profiling/profile_power_flow_solvers.jl
```

Smoke-test the harness on a small system first (seconds, not minutes):

```
PF_PROFILE_SYSTEM=c_sys14 julia --project=test \
    scripts/profiling/profile_power_flow_solvers.jl
```

## Reading the output

- The component table is the fastest way to spot the dominant cost: if
  `numeric_refactor!` + `solve!` dominate, the linear-solver backend is the
  lever; if Jacobian assembly or residual evaluation dominate, the hot spot is
  in PowerFlows' own evaluation code.
- The `.tree.txt` report shows the call-stack attribution; the `.pb.gz`
  flamegraph (PProf) is best for interactive drill-down.

## Notes

- `PF_PROFILE_SYSTEM` overrides the system name (group is inferred: `matpower`/
  `ACTIVSg` names load from `MatpowerTestSystems`, others from `PSITestSystems`).
- The 10k run is multi-minute; prefer running it in the background.
