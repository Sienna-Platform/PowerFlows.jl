# Solving a Power Flow

In this tutorial you'll solve power flows on a 5-bus test system using each of the
evaluation models PowerFlows.jl provides, then swap in different AC solver
algorithms — without changing the rest of the script.

The tutorial is built around one idea: PowerFlows.jl separates **what** you
solve (the *evaluation model* — DC, AC polar, AC rectangular, …) from **how** you
solve it (the *AC solver algorithm* — Newton-Raphson, trust region, Levenberg-Marquardt, …).
For a fuller treatment of that distinction, see the
[Evaluation Models vs. Solver Algorithms](@ref) explanation page.

To get started, ensure you have followed the [installation instructions](@ref "Installation").
Start Julia from the command line if you haven't already.

## Building a System

Load the needed packages. We're using a standard test system and want to keep
output clean, so we adjust the logging settings to filter out a few precautionary
warnings.

!!! tip "Activate the project environment first"
    If you are following this tutorial from within the cloned PowerFlows.jl repository,
    activate the local project environment before loading packages so that Julia uses the
    local version rather than any globally-installed version:
    ```julia
    import Pkg
    Pkg.activate(".")
    Pkg.instantiate()  # first time only: downloads packages listed in Manifest.toml
    ```
    `Pkg.instantiate()` is only needed the first time you activate the project (or after
    pulling changes that update `Manifest.toml`). It is safe to skip on subsequent sessions.
    If you installed PowerFlows.jl via `Pkg.add`, you can skip both steps.

```@repl basic_tutorial
using PowerSystemCaseBuilder
using PowerFlows
using PowerSystems
using Logging
disable_logging(Logging.Warn)
```

Create a [`System`](@extref PowerSystems.System) from
[PowerSystemCaseBuilder.jl](https://github.com/NREL-Sienna/PowerSystemCaseBuilder.jl):

```@repl basic_tutorial
sys = build_system(MatpowerTestSystems, "matpower_case5_sys")
```

!!! warning "Run the setup blocks first"
    If any `using` statement in the setup block failed because a package was not yet
    installed, install it with `Pkg.add("PackageName")` and then **re-run the entire
    setup block** — simply installing a package does not load it into the current session.

## The Two-Layer API

Throughout the tutorial we call exactly one function to solve a power flow:

```julia
results = solve_power_flow(model, sys)
```

The first argument, `model`, is a [`PowerFlowEvaluationModel`](@ref) that
describes the problem to solve and any options. Everything below — DC, AC polar,
AC rectangular — is a matter of building a different `model`. The second
argument, `sys`, never changes.

## DC Power Flow

[`DCPowerFlow`](@ref) solves for bus voltage angles using the bus admittance
matrix, then computes branch flows from the angle differences. Create a solver
and run it, passing a [`FlowReporting`](@ref) mode so flows are reported on the
same arc basis as the rest of the package:

```@repl basic_tutorial
pf_dc = DCPowerFlow()
dc_results = solve_power_flow(pf_dc, sys, FlowReporting.ARC_FLOWS)
```

The result is a `Dict{String, Dict{String, DataFrame}}`. The outer key is the
time-step name (`"1"` here, since DC supports multi-period). The inner
dictionary stores `"bus_results"`, `"flow_results"` (AC lines), and
`"lcc_results"` (HVDC lines — empty for this system).

```@repl basic_tutorial
dc_results["1"]["bus_results"]
```

Notice that `Vm` (voltage magnitude) is `1.0` for all buses, and `Q_gen` and
`Q_load` are zero. This is expected for DC power flow, which assumes flat
voltage magnitudes and ignores reactive power.

```@repl basic_tutorial
dc_results["1"]["flow_results"]
```

`Q_from_to` and `Q_to_from` are zero, and `P_losses` follows a first-order
`R · P²` estimate. Purely inductive branches will show zero losses.

## PTDF DC Power Flow

[`PTDFDCPowerFlow`](@ref) computes branch flows directly from bus power
injections using the Power Transfer Distribution Factor matrix, without solving
for voltage angles as an intermediate step:

```@repl basic_tutorial
pf_ptdf = PTDFDCPowerFlow()
ptdf_results = solve_power_flow(pf_ptdf, sys, FlowReporting.ARC_FLOWS)
ptdf_results["1"]["bus_results"]
```

The results match `DCPowerFlow`, as they should: the two are mathematically
equivalent. For very large systems where forming the full PTDF matrix would be
too expensive, see [`vPTDFDCPowerFlow`](@ref), which computes the same results
without storing the dense matrix.

## AC Power Flow — Polar Formulation

[`ACPowerFlow`](@ref) is an alias for [`ACPolarPowerFlow`](@ref). It solves the
full non-linear AC equations using a per-bus state `(Vᵢ, θᵢ)` and a real
power-mismatch residual:

```@repl basic_tutorial
pf_ac = ACPowerFlow()
ac_results = solve_power_flow(pf_ac, sys)
```

AC results are returned as a flat `Dict{String, DataFrame}` with the same keys
as before. (Sienna does not yet support multi-period AC power flows, so there is
no outer time-step layer.)

```@repl basic_tutorial
ac_results["bus_results"]
```

`Vm` now varies across buses (it is not all `1.0`), and `Q_gen` has non-zero
values.

```@repl basic_tutorial
ac_results["flow_results"]
```

`Q_from_to` and `Q_to_from` show reactive power flows, and `P_from_to` differs
from `P_to_from` due to losses.

## AC Power Flow — Rectangular (Current-Injection) Formulation

[`ACRectangularPowerFlow`](@ref) solves the *same* AC power flow as the polar
model, but in rectangular coordinates `(eᵢ, fᵢ)` with a complex-current
mismatch residual. Off-diagonal Jacobian blocks are constant 2×2 real blocks of
`Y_bus`, which can make refactorization cheaper:

```@repl basic_tutorial
pf_ac_rect = ACRectangularPowerFlow()
ac_rect_results = solve_power_flow(pf_ac_rect, sys)
ac_rect_results["bus_results"]
```

The result schema is identical to the polar case. Voltages, generations and
flows should match the polar solution to solver tolerance — the two
formulations are mathematically equivalent.

```@repl basic_tutorial
ac_rect_results["flow_results"]
```

!!! note "When to prefer rectangular"
    Pick the rectangular formulation when you specifically need the
    current-injection structure (e.g. cheap reuse of `Y_bus` blocks, integration
    with code that expects `(e, f)` coordinates). The polar formulation is the
    default because it is the only one that currently supports the
    loss-factor, voltage-stability, and DC-fallback options.

## Swapping the AC Solver Algorithm

The AC solver is **orthogonal** to the AC formulation: each AC model is
parameterized by an [`ACPowerFlowSolverType`](@ref) tag that selects the
iterative algorithm used to drive the residual to zero. The default is
[`NewtonRaphsonACPowerFlow`](@ref); alternatives are
[`TrustRegionACPowerFlow`](@ref), [`LevenbergMarquardtACPowerFlow`](@ref),
and [`RobustHomotopyPowerFlow`](@ref) (polar only).

Pass the solver tag as the type parameter of the model — nothing else changes:

```@repl basic_tutorial
pf_ac_tr = ACPowerFlow{TrustRegionACPowerFlow}()
tr_results = solve_power_flow(pf_ac_tr, sys)
tr_results["bus_results"]
```

```@repl basic_tutorial
pf_ac_lm = ACPowerFlow{LevenbergMarquardtACPowerFlow}()
lm_results = solve_power_flow(pf_ac_lm, sys)
lm_results["bus_results"]
```

All three converge to the same physical solution on this system. The choice
matters when default Newton-Raphson struggles: trust region and
Levenberg-Marquardt are increasingly robust at increasing computational cost.

The same solver tag works with the rectangular formulation:

```@repl basic_tutorial
pf_ac_rect_lm = ACRectangularPowerFlow{LevenbergMarquardtACPowerFlow}()
rect_lm_results = solve_power_flow(pf_ac_rect_lm, sys)
rect_lm_results["bus_results"]
```

## Comparing Results

The DC and AC active power flows differ slightly because AC power flow accounts
for resistive losses. Compare the `P_from_to` column across methods:

```@repl basic_tutorial
dc_results["1"]["flow_results"][!, [:flow_name, :P_from_to]]
```

```@repl basic_tutorial
ptdf_results["1"]["flow_results"][!, [:flow_name, :P_from_to]]
```

```@repl basic_tutorial
ac_results["flow_results"][!, [:flow_name, :P_from_to]]
```

```@repl basic_tutorial
ac_rect_results["flow_results"][!, [:flow_name, :P_from_to]]
```

DC and PTDF-DC are identical (they are mathematically equivalent). The polar
and rectangular AC results agree to numerical tolerance and differ from DC
because the AC solvers find the physically exact solution, including losses.

## When AC Power Flow Fails

Unlike DC, AC power flow is iterative and not guaranteed to converge. Systems
with high-impedance lines, poor initial voltage profiles, or insufficient
reactive-power support can defeat the solver. When that happens
`solve_power_flow` returns `missing` and logs an error. The recommended
escalation path is:

1. Default [`NewtonRaphsonACPowerFlow`](@ref), optionally with Iwamoto damping
   (`solver_settings = Dict(:iwamoto => true)`).
2. [`TrustRegionACPowerFlow`](@ref) — usually rescues mildly ill-conditioned
   cases at comparable cost.
3. [`LevenbergMarquardtACPowerFlow`](@ref) — more robust, more expensive,
   works on both polar and rectangular formulations.
4. [`RobustHomotopyPowerFlow`](@ref) — order-of-magnitude slower than Newton,
   but very robust to bad initial points (polar only).

You do not need to change anything but the solver tag — same `sys`, same
`solve_power_flow` call.

## Next Steps

- **How-tos**: see the how-to guides for writing results to PSS/e format,
  running multi-period power flows, and working with HVDC systems.
- **Explanation**: read
  [Evaluation Models vs. Solver Algorithms](@ref) for the conceptual model
  behind the API, and
  [Line-Commutated Converter (LCC) Implementations](@ref) for how
  line-commutated HVDC converters are modelled inside the AC power flow.
- **Reference**: browse the [Public API](@ref) for all available
  evaluation models, solver tags, and keyword arguments.
