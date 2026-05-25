# Power Flow

```@meta
CurrentModule = PowerFlows
```

`PowerFlows.jl` provides the capability to run a power flow using the [Newton-Raphson method](https://en.wikipedia.org/wiki/Newton%27s_method), optionally enforcing reactive power constraints of generators. This power flow routine can be used to check for AC feasibility of results of DC optimal power flow.

The power flow solver uses [KLU.jl](https://github.com/JuliaSparse/KLU.jl) for Jacobian matrix factorization. The solver uses the current
operating point in the buses to provide the initial guess. The initial guess is then
adjusted to contain voltage magnitudes within a feasible range of 0.8 p.u. - 1.2 p.u.

**Limitations**: The PowerFlow solver doesn't support systems with HVDC lines or
Phase Shifting transformers yet. The power flow solver can't handle systems with islands.

````@example generated_power_flow
using PowerFlows
using PowerSystems
using PowerSystemCaseBuilder

system_data = build_system(PSITestSystems, "c_sys14")
````

`PowerFlows.jl` has two modes of using the power flow solver.

 1. Solving the power flow for the current operating point in the system.
    Takes the data in the buses, the `active_power` and `reactive_power` fields
    in the static injection devices. Returns a dictionary with results in a DataFrame that
    can be exported or manipulated as needed.

 2. Solves the power flow and updated the devices in the system to the operating condition.
    This model will update the values of magnitudes and angles in the system's buses. It
    also updates the active and reactive power flows in the branches and devices connected
    to PV buses. It also updates the active and reactive power of the injection devices
    connected to the Slack bus, and updates only the reactive power of the injection devices
    connected to PV buses. If multiple devices are connected to the same bus, the power is
    divided proportional to the base power.
    This utility is useful to initialize systems before serializing or checking the
    addition of new devices is still AC feasible.

Solving the power flow with mode 1:

````@example generated_power_flow
pf = ACPowerFlow()
results = solve_power_flow(pf, system_data)
results["bus_results"]
````

Solving the power flow with mode 2:

Before running the power flow command these are the values of the
voltages:

````@example generated_power_flow
for b in get_components(Bus, system_data)
    println("$(get_name(b)) - Magnitude $(get_magnitude(b)) - Angle (rad) $(get_angle(b))")
end
````

[`solve_and_store_power_flow!`](@ref) return true or false to signal the successful result of the power
flow. This enables the integration of a power flow into functions and use the return as check.

````@example generated_power_flow
solve_and_store_power_flow!(pf, system_data; method = :newton)
````

After running the power flow command this are the values of the
voltages:

````@example generated_power_flow
for b in get_components(Bus, system_data)
    println("$(get_name(b)) - Magnitude $(get_magnitude(b)) - Angle (rad) $(get_angle(b))")
end
````
