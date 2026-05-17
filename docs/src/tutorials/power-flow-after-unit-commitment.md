# Validating a UC Dispatch with Multi-Period Power Flow

To get started, ensure you have followed the [installation instructions](@ref "Installation"). Start Julia from the command line if you haven't already.

In this tutorial, you'll run a 24-interval DC power flow over a set of dispatch results
from a unit commitment (UC) problem, then scan every interval for overloaded branches.

Unit commitment determines which generators are online and at what output level for each
hour of a planning horizon. UC models typically use a simplified network representation
(e.g., Power Transfer Distribution Factor linearization). Running a power flow at every dispatch interval lets you
check whether the committed schedule is network-feasible — no line overloads, angle
differences within limits, etc.

If you want to run power flow automatically *during* a [`PowerSimulations.jl`](https://nrel-sienna.github.io/PowerSimulations.jl/stable/) UC simulation rather than
after it, see the tutorial [Running Power Flow In The Loop with Unit Commitment](@ref).

## Setup

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

Load the needed packages:

```@repl uc_validation
using PowerSystemCaseBuilder
using PowerFlows
using PowerSystems
using DataFrames
using Logging
disable_logging(Logging.Warn)
```

Build a 14-bus test system:

```@repl uc_validation
sys = build_system(PSITestSystems, "c_sys14")
```

!!! warning "Run the setup blocks first"
    Every code block in this tutorial shares the same REPL session (`uc_validation`). If you
    skip or re-order blocks, variables like `sys`, `pf`, and `data` will be undefined and
    you will see `UndefVarError`. Always run the setup blocks above before proceeding to
    each subsequent section.

    If any `using` statement in the setup block failed because a package was not yet
    installed, install it with `Pkg.add("PackageName")` and then **re-run the entire
    setup block** — simply installing a package does not load it into the current session.

## Building a Multi-Period PowerFlowData Object

Multi-period power flow requires constructing a [`PowerFlowData`](@ref) object first.
This pre-allocates all matrices and working arrays for every time step at once, avoiding
repeated setup overhead.

Create a [`DCPowerFlow`](@ref) configured for 24 time steps:

```@repl uc_validation
pf = DCPowerFlow(; time_steps = 24)
data = PowerFlows.PowerFlowData(pf, sys)
```

`data` now holds pre-allocated injection and withdrawal arrays shaped `(n_buses, 24)`.

## Loading Dispatch Data

In a real workflow, per-interval bus injections and withdrawals come from your UC
results (e.g., extracted from a [`PowerSimulations.jl`](https://nrel-sienna.github.io/PowerSimulations.jl/stable/) `SimulationResults` object). Here we use the
default values already populated by `PowerFlowData` and apply a sinusoidal load
scale to simulate varying demand across the day:

```@repl uc_validation
base_injections  = copy(data.bus_active_power_injections[:, 1])
base_withdrawals = copy(data.bus_active_power_withdrawals[:, 1])

for t in 1:24
    load_scale = 0.8 + 0.2 * sin(2π * (t - 1) / 24)
    data.bus_active_power_injections[:, t]  .= base_injections
    data.bus_active_power_withdrawals[:, t] .= base_withdrawals .* load_scale
end
```

## Solving the Multi-Period Power Flow

Solve all 24 time steps in a single call. Using [`FlowReporting`](@ref).BRANCH_FLOWS reports
one row per physical branch by its [`PowerSystems.jl`](https://nrel-sienna.github.io/PowerSystems.jl/stable/) name, which lets us look up ratings directly
from the `System`:

```@repl uc_validation
results = solve_power_flow(data, sys, FlowReporting.BRANCH_FLOWS)
```

`results` is a `Dict{String, Dict{String, DataFrame}}`. The outer key is the time step
name (`"1"` through `"24"`). The inner dictionary has keys `"bus_results"`,
`"flow_results"`, and `"lcc_results"`.

## Inspecting Results

Look at bus results for hour 1:

```@repl uc_validation
results["1"]["bus_results"]
```

Notice that `Vm` (voltage magnitude) is 1.0 for all buses and `Q_gen`, `Q_load` are 0.
This is expected for DC power flow, which assumes flat voltage profiles and ignores
reactive power entirely.

Look at line flows for hour 1:

```@repl uc_validation
results["1"]["flow_results"]
```

Notice that `P_from_to` and `P_to_from` are **not** equal and opposite, and `P_losses`
is non-zero for resistive branches. Although DC power flow ignores reactive power and
assumes flat voltage magnitudes, PowerFlows.jl computes a first-order active power loss
estimate as $R \cdot P^2$, where $R$ is the branch resistance. Purely inductive branches
(zero resistance) will still show zero losses.

## Checking for Overloads

Build a branch rating lookup from the `System` — `flow_name` in the results matches
the [`PowerSystems.jl`](https://nrel-sienna.github.io/PowerSystems.jl/stable/) branch name when using [`FlowReporting`](@ref).BRANCH_FLOWS reporting:

```@repl uc_validation
ratings = Dict(
    PowerSystems.get_name(b) => PowerSystems.get_rating(b) * PowerSystems.get_base_power(sys)
    for b in PowerSystems.get_components(PowerSystems.ACBranch, sys)
)
```

Scan every interval for branches that exceed their rating:

```@repl uc_validation
overloads = DataFrame(
    time_step  = String[],
    flow_name  = String[],
    P_from_to  = Float64[],
    rating     = Float64[],
)

for (t, step_results) in results
    for row in eachrow(step_results["flow_results"])
        rating = get(ratings, row.flow_name, Inf)
        if abs(row.P_from_to) > rating
            push!(overloads, (t, row.flow_name, row.P_from_to, rating))
        end
    end
end

overloads
```

An empty DataFrame means the committed dispatch is network-feasible at every interval —
every branch stays within its thermal rating at every hour of the day.

If the DataFrame is non-empty, each row identifies a specific congestion event:

- **`time_step`**: the interval (hour) in which the overload occurred.
- **`flow_name`**: the name of the overloaded branch, matching the PowerSystems.jl component name.
- **`P_from_to`**: the actual active power flow on that branch in MW (signed, from the
  "from" bus to the "to" bus).
- **`rating`**: the branch's thermal rating in MW. The severity of the overload is
  `abs(P_from_to) - rating`.

In a real UC workflow, overloaded branches indicate that the UC model's network
representation was too loose — it allowed a dispatch that the full DC network model rejects.
Common responses include:

- **Add transmission constraints** to the UC model for the congested branches.
- **Re-run the UC** with those constraints active and repeat the power flow check.
- **Manually curtail** generation near the overloaded branch and re-check.

Iterating this process until the overloads DataFrame is empty gives you a
network-feasible dispatch schedule.
