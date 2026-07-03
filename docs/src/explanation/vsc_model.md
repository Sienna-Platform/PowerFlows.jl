# Voltage Source Converter (VSC) Model Implementation

## Implementations in PSSE

PSS/E represents a point-to-point VSC HVDC link with the steady-state *VSC DC Transmission Line*
record: two converter records, each carrying a DC-control `TYPE` (`1` = DC-voltage control,
`2` = MW/power control, `0` = out of service) and an AC-control `MODE` (`1` = AC-voltage control,
`2` = fixed power factor), plus a single fixed setpoint `DCSET` (scheduled DC voltage in kV for
`TYPE 1`, MW demand for `TYPE 2`). When both converters are in service exactly one must be `TYPE 1`.

The steady-state record has **no representation of DC-voltage *droop* control** — there is no droop
slope, gain, or deadband field, and the field set is stable across PSS/E v33/v34/v35. Droop is a
dynamics-level concern in PSS/E (`.dyr` models such as VSCDCT/CDC4T) or an external/multi-terminal
add-on. Consequently, when Sienna **exports** to PSS/E `.raw`, a droop-controlled converter is
written as MW control (`TYPE 2`) at its scheduled active power, and the droop characteristic is
dropped (see [`PSSEExportPowerFlow`](@ref)); only a strict DC-voltage terminal is exported as `TYPE 1`.

## Implementation in Sienna

Every PSY DC component lowers into a single internal `DCNetwork` that the AC power flow solves
jointly with the AC buses. A point-to-point `TwoTerminalVSCLine` becomes two converters, two DC
nodes, and one DC branch; a multi-terminal grid (`InterconnectingConverter` on `DCBus` nodes joined
by `TModelHVDCLine` branches) lowers into the same structure, so all residual/Jacobian kernels are
written once. The escape hatch `solver_settings = Dict(:model_dc_network => false)` restores the
historical behavior (DC components ignored in the AC solve).

The state vector is extended with a VSC tail: per converter the AC injections $(P_c, Q_c)$, and per
DC node the DC voltage $V_{dc}$. Each converter contributes two control rows ($r_1$, $r_2$) and each
DC node one DC-KCL row, matching the added states. VSC extensions are implemented for all three AC
formulations — [`ACPolarPowerFlow`](@ref), [`ACRectangularPowerFlow`](@ref), and
[`ACMixedPowerFlow`](@ref) — sharing the same converter-physics kernels; only the bus-coupling rows
differ (polar uses the bus power balance; rectangular/mixed use the bus current-injection balance).

### Control modes

A converter's per-terminal PSY DC-side and AC-side control enums lower into one `VSCControlMode`,
which selects the two control-row residuals:

| Mode               | $r_1$ (active / $V_{dc}$)     | $r_2$ (reactive / AC voltage)       |
|:------------------ |:----------------------------- |:----------------------------------- |
| `ControlPQ`        | $P_c - P_{set}$               | $Q_c - Q_{set}$                     |
| `ControlPVac`      | $P_c - P_{set}$               | $\lvert V_{ac}\rvert^2 - V_{set}^2$ |
| `ControlVdc`       | $V_{dc} - V_{set}$            | $Q_c - Q_{set}$                     |
| `ControlVdcQ`      | $V_{dc} - V_{set}$            | $\lvert V_{ac}\rvert^2 - V_{set}^2$ |
| `ControlPVdcDroop` | $(V_{dc} - V_{set}) - k\,P_c$ | $Q_c - Q_{set}$                     |

where $k$ is the DC-voltage droop gain. With $P_c$ the converter's AC-side injection, the droop row
$r_1 = (V_{dc} - V_{set}) - k\,P_c$ is equivalently $V_{dc} = V_{set} - k\,P_{dc,inj}$ where
$P_{dc,inj}$ is the power injected into the DC grid — the standard Beerten droop convention: a
converter injecting into the DC grid operates below its voltage reference, and a high DC voltage
drives converters to withdraw more. A mode pins the DC-node voltage (makes its node a DC slack)
iff it is a $V_{dc}$-control mode; a wholly droop-controlled DC subnet is anchored by the droop
relation instead. The AC-voltage row uses the raw $\lvert V_{ac}\rvert^2$ (floor-free, like the
rectangular PV pin) for a derivative consistent with the residual.

### DC network balance and converter losses

For each DC node the DC-KCL row enforces the nodal current balance through the dense DC nodal
conductance matrix $G_{dc}$ (built from the branch conductances) plus the converter current
injections:

```math
\sum_{j} G_{dc}[k, j]\, V_{dc}[j] + \sum_{c \,\in\, k} \frac{P_{dc,c}}{V_{dc}} = 0
```

The converter draws $P_{dc} = P_c + P_{loss}$ from its DC node, with a quadratic converter-loss
curve in the converter current magnitude $I_c$:

```math
P_{loss} = a + b\,I_c + c\,I_c^2, \qquad
I_c = \frac{S}{\lvert V_{ac}\rvert}, \qquad
S = \sqrt{P_c^2 + Q_c^2}
```

Both $S^2$ and $\lvert V_{ac}\rvert^2$ are floored at `V_FLOOR2` to keep $I_c$ and its derivatives
finite at the origin (the $P_c/S$, $Q_c/S$ terms are otherwise singular at $S = 0$).

### Jacobian

The control-row partials depend only on the control mode (and on $\lvert V_{ac}\rvert$ for the
AC-voltage and loss terms), so each is a small branch on `VSCControlMode` rather than method
dispatch:

```math
\begin{aligned}
\frac{\partial r_1}{\partial P_c} &= \begin{cases} 0 & V_{dc}\text{-control} \\ -k & \text{droop} \\ 1 & P\text{-control} \end{cases}
&\qquad
\frac{\partial r_1}{\partial V_{dc}} &= \begin{cases} 1 & V_{dc}\text{-control or droop} \\ 0 & \text{otherwise} \end{cases} \\[1em]
\frac{\partial r_2}{\partial Q_c} &= \begin{cases} 0 & \text{AC-voltage control} \\ 1 & \text{otherwise} \end{cases}
&\qquad
\frac{\partial r_2}{\partial \lvert V_{ac}\rvert} &= \begin{cases} 2\lvert V_{ac}\rvert & \text{AC-voltage control} \\ 0 & \text{otherwise} \end{cases}
\end{aligned}
```

The converter's $(P_c, Q_c)$ inject into the bus balance rows (with a $-1$ coupling), and the DC-KCL
row couples to $(P_c, Q_c, \lvert V_{ac}\rvert, V_{dc})$ through $P_{dc}/V_{dc}$. The
$\lvert V_{ac}\rvert$-coupling enters the Jacobian only where $\lvert V_{ac}\rvert$ is a state — i.e.
at PQ buses in the polar formulation (at PV/REF buses the AC magnitude is fixed, so those derivatives
vanish and no Jacobian entry is written). The loss-term DC-side derivatives are

```math
\frac{\partial P_{dc}}{\partial P_c} = 1 + \frac{dP_{loss}}{dI_c}\frac{\partial I_c}{\partial P_c},
\qquad
\frac{\partial P_{dc}}{\partial Q_c} = \frac{dP_{loss}}{dI_c}\frac{\partial I_c}{\partial Q_c},
\qquad
\frac{\partial P_{dc}}{\partial \lvert V_{ac}\rvert} = -\frac{dP_{loss}}{dI_c}\frac{I_c}{\lvert V_{ac}\rvert}
```

with $\dfrac{dP_{loss}}{dI_c} = b + 2 c\,I_c$, $\dfrac{\partial I_c}{\partial P_c} = \dfrac{P_c/S}{\lvert V_{ac}\rvert}$, and $\dfrac{\partial I_c}{\partial Q_c} = \dfrac{Q_c/S}{\lvert V_{ac}\rvert}$.
