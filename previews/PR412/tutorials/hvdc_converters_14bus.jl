#src EXECUTE = TRUE
# # HVDC Converters (VSC & LCC)

# An HVDC link moves power between two AC buses through a DC stage: an
# AC/DC converter at each end, joined by a DC line. PowerFlows co-solves this
# DC stage together with the surrounding AC network, so an AC power flow with
# an HVDC branch converges to one consistent solution rather than being
# solved as two separate problems. PowerFlows represents two converter
# technologies:
#
#   - **VSC** (voltage-source converter): self-commutated, so each terminal
#     independently controls its own active power (or DC voltage) and
#     reactive power (or AC voltage).
#   - **LCC** (line-commutated converter): thyristor-based, so the only
#     controls are each terminal's firing/extinction angle and transformer
#     tap, and the converter always draws reactive power from the AC system
#     it connects to.
#
# This tutorial builds one 14-bus system for each technology and solves them
# with the same [`ACPowerFlow`](@ref) solver used elsewhere in these docs.

# ## Loading the needed packages

using PowerSystemCaseBuilder
using PowerSystems
using PowerFlows

# ## VSC case
# Build the 14-bus test system with a single
# [`PowerSystems.TwoTerminalVSCLine`](@extref) replacing the AC line on the
# bus 2↔3 arc:

sys_vsc = build_system(
    PSITestSystems,
    "c_sys14_hvdc_vsc";
    force_build = true,
    add_forecasts = false,
)

# Solve it with [`ACPowerFlow`](@ref), the same convenience wrapper used for
# a plain AC-only system — no special solver settings are needed for a VSC
# line:

vsc_results = solve_power_flow(ACPowerFlow(), sys_vsc)

# The result `Dict` carries the usual AC keys (`"bus_results"`,
# `"flow_results"`) plus HVDC-specific tables. For a VSC line, the relevant
# one is `"vsc_results"`:

vsc_results["vsc_results"]

# `P_from_to` is the active power (MW) the `from`-side AC bus delivers into
# the converter; `Q_from_to`/`Q_to_from` are the reactive power each AC
# terminal exchanges with its own bus; `dc_current`, `Vdc_from`, `Vdc_to`
# describe the state of the (here two-node) internal DC network; and
# `P_losses` is the combined DC-side loss (converter losses plus the DC
# line's own resistive drop).
#
# The two converters play different roles. This system's `from` converter is
# configured with `dc_control_from = VSCDCControlModes.DC_VOLTAGE`: it holds
# the DC network at its scheduled voltage, acting as the DC-side slack. The
# `to` converter is configured with `dc_control_to = VSCDCControlModes.DC_POWER`:
# it is dispatched to a scheduled active-power transfer, and the `from`
# converter's power then adjusts to balance that transfer plus DC losses —
# which is exactly the relationship between `P_from_to` and `P_losses` above.
# Independently of this DC-side role, each converter also holds its own
# AC-side reactive-power setpoint (`ac_control_from`/`ac_control_to =
# VSCACControlModes.AC_REACTIVE_POWER` here), so `Q_from_to` and `Q_to_from`
# are unrelated numbers rather than mirror images of each other.

# Bus voltages solve normally alongside the DC network:

vsc_results["bus_results"]

# ## LCC case
# Build the matching system with a single
# [`PowerSystems.TwoTerminalLCCLine`](@extref) on the same 2↔3 arc, and solve
# it the same way:

sys_lcc = build_system(
    PSITestSystems,
    "c_sys14_hvdc_lcc";
    force_build = true,
    add_forecasts = false,
)
lcc_results = solve_power_flow(ACPowerFlow(), sys_lcc)

# The matching result table is `"lcc_results"`:

lcc_results["lcc_results"]

# An LCC has no independent control over its terminal voltages or reactive
# power. The rectifier and inverter each have one control angle —
# `rectifier_delay_angle` (firing angle) and `inverter_extinction_angle`
# (extinction angle) — plus a transformer tap, `rectifier_tap`/
# `inverter_tap`. PowerFlows keeps both thyristor angles at their minimum
# physical limits (`rectifier_delay_angle_limits`/
# `inverter_extinction_angle_limits` on the component) and instead uses the
# tap ratios to hold the line's scheduled DC power transfer — here the
# line's `transfer_setpoint` of 50 MW appears directly as `P_from_to`.
#
# Because thyristors can only be switched on, not off, on demand, an LCC
# always draws reactive power from the AC system at both ends to commutate —
# it cannot supply it. `Q_from_to` and `Q_to_from` are both positive
# (absorbed from the AC network), in contrast to the VSC case above, where
# each converter's reactive power is an independent setpoint that can be
# positive, negative, or zero.

lcc_results["bus_results"]

# ## When each is modeled
# Reach for VSC when a converter has independent control over active power,
# reactive power (or AC voltage), and DC voltage — the behavior of modern
# IGBT-based HVDC links. Reach for LCC to model classic thyristor-based
# HVDC, where the only controls are firing/extinction angles and transformer
# taps, and the converter is always a net reactive-power sink on the AC
# side. PowerFlows solves both inside the same AC power flow as the
# surrounding network, so no separate DC power-flow step is needed for
# either technology.
