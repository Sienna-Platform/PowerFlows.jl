#src EXECUTE = TRUE
# # Discrete Shunt Voltage Control

# On a heavily loaded network, the voltage at a weak bus can sag well below its
# nominal value: the reactive power that the loads demand outpaces what nearby
# generators and lines can deliver. A shunt reactive device installed at (or
# near) the weak bus can restore the voltage by injecting reactive power
# locally. This tutorial builds a stressed 14-bus system, then shows two ways
# PowerFlows can hold a bus voltage at a setpoint:
#
#   - a **continuous** shunt FACTS device (SVC/STATCOM-style), whose reactive
#     injection can take any value up to a current limit; and
#   - a **discrete** switched shunt (PSS/E-style block-switched capacitor
#     bank), whose reactive injection can only take one of a finite set of
#     steps.
#
# Both device families are co-solved with the network equations by passing
# `control_discrete_devices = true` to the AC solver: the device's parameter
# (susceptance, in either case) is adjusted between outer iterations until the
# controlled bus voltage settles at its target, or the device saturates at its
# own limit.

# ## Building and stressing the system
# To get started, load the needed packages.

using PowerSystemCaseBuilder
using PowerSystems
using PowerFlows

# Build the 14-bus test system with [`PowerSystemCaseBuilder.build_system`](@extref),
# then scale every `PowerLoad`'s active and reactive power by `load_scale` to
# push the network toward its reactive power limits:

function build_stressed_system(load_scale::Float64)
    sys = build_system(PSITestSystems, "c_sys14"; force_build = true, add_forecasts = false)
    set_units_base_system!(sys, "SYSTEM_BASE")
    for load in get_components(PowerLoad, sys)
        set_active_power!(load, get_active_power(load) * load_scale)
        set_reactive_power!(load, get_reactive_power(load) * load_scale)
    end
    return sys
end

load_scale = 3.0
sys = build_stressed_system(load_scale)

# Solve a baseline [`ACPowerFlow`](@ref) on the stressed system, with no
# control devices installed:

baseline_results = solve_power_flow(ACPowerFlow(), sys)
bus_results = baseline_results["bus_results"]

# Find the PQ bus with the lowest voltage magnitude. (PV and slack buses hold
# their setpoint by definition, so only PQ buses can sag.)

pq_bus_numbers =
    Set(
        get_number(b) for b in get_components(ACBus, sys) if get_bustype(b) == ACBusTypes.PQ
    )
pq_bus_results = filter(row -> row.bus_number in pq_bus_numbers, bus_results)
weakest = first(sort(pq_bus_results, :Vm), 1)

# At this load scale, the weakest PQ bus sags well below 1.0 pu:

weakest

weak_bus_number = weakest.bus_number[1]
weak_bus = get_bus(sys, weak_bus_number)

# ## Continuous FACTS control
# A `FACTSControlDevice` in `NML` mode is a continuously controllable shunt
# (an SVC or STATCOM): its susceptance can take any value between
# `-max_shunt_current` and `+max_shunt_current`, so it can inject exactly the
# reactive power needed to hold its bus at `voltage_setpoint`. Add one at
# `weak_bus`:

facts = FACTSControlDevice(;
    name = "svc_$(get_number(weak_bus))",
    available = true,
    bus = weak_bus,
    control_mode = FACTSOperationModes.NML,
    voltage_setpoint = 1.0,
    max_shunt_current = 100.0,
    shunt_control_type = FACTSShuntControlType.STATCOM,
)
add_component!(sys, facts)

# Solve with `control_discrete_devices = true` so the FACTS device's
# susceptance is co-solved with the network. This requires building a
# `PowerFlowData` and calling `solve_power_flow!` in place, rather than the
# `solve_power_flow` convenience wrapper used above:

pf = ACPolarPowerFlow(; control_discrete_devices = true)
data = PowerFlows.PowerFlowData(pf, sys)
PowerFlows.solve_power_flow!(data)

# The controlled bus now sits at its setpoint:

bus_lookup = PowerFlows.get_bus_lookup(data)
data.bus_magnitude[bus_lookup[weak_bus_number], 1]

# Detailed results for every enrolled control device are available from
# `PowerFlows.get_controlled_device_results`, one row per device:

res = PowerFlows.get_controlled_device_results(data)

# `lower_limit`/`upper_limit` are the susceptance band implied by
# `max_shunt_current`; `initial` is the susceptance the device started from
# (zero, since it wasn't installed before); `final` is the solved susceptance,
# in pu; `delivered_q_mvar` is the reactive power the device is actually
# injecting at the converged voltage; and `saturated` reports whether the
# device had to be clamped to `lower_limit`/`upper_limit` to converge — `false`
# here means the bus was held at setpoint without exhausting the device's
# capacity. Note that `final` is **not** snapped to any grid: this is a
# continuous device, so it lands on whatever real susceptance value the
# outer-loop continuation converges to.

# ## Discrete switched shunt
# A `SwitchedAdmittance` in `DISCRETE_VOLTAGE` mode models a block-switched
# capacitor bank: its susceptance can only take one of `number_of_steps`
# discrete increments of `Y_increase`, mirroring how a real substation shunt
# bank is switched. Build a fresh copy of the stressed system
# (so this device doesn't coexist with the FACTS device above), and add a
# 60-step, 1 MVar-per-step bank at the same weak bus. `admittance_limits` is a
# voltage deadband: once the bus voltage falls inside it, the continuation
# stops switching in more blocks.

sys2 = build_stressed_system(load_scale)
weak_bus2 = get_bus(sys2, weak_bus_number)
sa = SwitchedAdmittance(;
    name = "shunt_$(get_number(weak_bus2))",
    available = true,
    bus = weak_bus2,
    Y = 0.0 + 0.0im,
    initial_status = [0],
    number_of_steps = [60],
    Y_increase = [0.0 + (1.0 / get_base_power(sys2)) * im],
    admittance_limits = (min = 0.98, max = 1.02),
    control_mode = SwitchedAdmittanceControlMode.DISCRETE_VOLTAGE,
)
add_component!(sys2, sa)

data2 = PowerFlows.PowerFlowData(pf, sys2)
PowerFlows.solve_power_flow!(data2)

# The bus voltage settles inside the deadband, rather than exactly at 1.0:

bus_lookup2 = PowerFlows.get_bus_lookup(data2)
data2.bus_magnitude[bus_lookup2[weak_bus_number], 1]

# Reading the device results shows the contrast with the continuous case:

res2 = PowerFlows.get_controlled_device_results(data2)

# `final` here lands on an integer number of 1 MVar blocks (`final * base_power`
# is a whole number), rather than an arbitrary real value. `delivered_q_mvar`
# and `saturated` are `missing`/`false`: those two derived columns are
# currently only populated for `FACTSControlDevice`. For a switched shunt, the
# delivered reactive power at the converged voltage can be computed directly
# from `final`:

res2.final[1] * data2.bus_magnitude[bus_lookup2[weak_bus_number], 1]^2 *
get_base_power(sys2)

# ## Saturation
# If a FACTS device's `max_shunt_current` is too small for the correction the
# bus needs, it cannot hold the setpoint: it clamps to its own limit instead.
# Build another fresh copy of the stressed system and add a FACTS device with
# a tight current limit:

sys3 = build_stressed_system(load_scale)
weak_bus3 = get_bus(sys3, weak_bus_number)
facts_tight = FACTSControlDevice(;
    name = "svc_tight_$(get_number(weak_bus3))",
    available = true,
    bus = weak_bus3,
    control_mode = FACTSOperationModes.NML,
    voltage_setpoint = 1.0,
    max_shunt_current = 5.0,
    shunt_control_type = FACTSShuntControlType.STATCOM,
)
add_component!(sys3, facts_tight)

data3 = PowerFlows.PowerFlowData(pf, sys3)
PowerFlows.solve_power_flow!(data3)

# The bus stays below setpoint:

bus_lookup3 = PowerFlows.get_bus_lookup(data3)
data3.bus_magnitude[bus_lookup3[weak_bus_number], 1]

# and the results table flags the device as saturated, the shunt-device
# analogue of a generator hitting a reactive power limit:

res3 = PowerFlows.get_controlled_device_results(data3)

# ## When to use each
# Use a continuous FACTS device when the model calls for an SVC or STATCOM:
# fast, continuously variable reactive support that lands exactly on the
# solved susceptance. Use a discrete switched shunt to model an actual
# block-switched capacitor/reactor bank, where only whole steps can be
# switched in or out. Both are read from the same `get_controlled_device_results`
# table. When the installed capacity isn't enough to hold the bus at its target
# voltage, the FACTS device reports it directly through `saturated = true`; a
# switched shunt instead reveals it by switching in all of its steps (its `final`
# reaching the top of the susceptance band).
