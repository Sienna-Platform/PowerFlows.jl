# Area Interchange Control

## What it does

**Net-interchange control**: each participating `PSY.Area` is driven so that its net
active-power exchange with the rest of the system meets a scheduled target ``PDES_a`` (read from
the area's `scheduled_net_interchange` field). Control is *embedded* in the AC solver — the
schedules are met at convergence, not by an outer sweep — so it composes with reactive-power limits,
distributed slack, and multi-period solves. Both AC boundary branches and two-terminal HVDC
(LCC and VSC) boundary links count toward each area's net interchange.

### Terminology

The names below are PSS/E's, kept so that engineers comparing against a PSS/E case can map
between the two directly.

| Symbol / term  | PSS/E name              | Meaning here                                                                                      |
|:-------------- |:----------------------- |:------------------------------------------------------------------------------------------------- |
| ``NI_a``       | net interchange         | Sum of metered active power over area ``a``'s boundary ties, signed positive *out* of the area.   |
| ``PDES_a``     | desired net interchange | Area ``a``'s scheduled net-interchange target, read from its `scheduled_net_interchange` field.   |
| ``\Delta P_a`` | —                       | The solved active-power adjustment at area ``a``'s slack bus that makes ``NI_a`` meet ``PDES_a``. |
| ISW bus        | area slack / swing bus  | The one voltage-regulating bus in an area that absorbs ``\Delta P_a``. Marked `ACBusTypes.SLACK`. |
| PTOL           | interchange tolerance   | Allowed ``NI_a`` deviation. Here **reporting only** — see `interchange_tolerance` below.          |

## When to use it

  - **Planning-case parity**: multi-area interconnection cases (PSS/E-style area interchange
    control, control mode 1) where solved tie flows must honor the scheduled interchange, not
    just system-wide balance.
  - **Seams and transfer studies**: hold every other area to schedule while one schedule is
    varied, so the solved state isolates the transfer of interest.
  - **Schedule feasibility screening**: an unenforceable schedule is relaxed rather than failing
    the solve, and the results table reports the achieved-vs-scheduled gap as the infeasibility
    certificate.
  - **Multi-period studies**: each time step re-solves its own adjustments against per-step
    injections; relax decisions never leak across steps.

## Enabling it

Set the flag on a polar power flow; NR, TR, LM, and both Fast Decoupled variants are supported:

```julia
pf = ACPowerFlow{NewtonRaphsonACPowerFlow}(; area_interchange_control = true)
solve_power_flow(pf, sys)
```

| Keyword                    | Default       | Meaning                                                                                                                                                                                                                                                                                                    |
|:-------------------------- |:------------- |:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `area_interchange_control` | `false`       | Turn embedded control on.                                                                                                                                                                                                                                                                                  |
| `interchange_tolerance`    | `0.05`        | PTOL analogue (pu), **reporting/validation only** — the embedded rows target `PDES_a` exactly. Non-positive values floor to `0.02` with a warning.                                                                                                                                                         |
| `tie_definition`           | `:lines_only` | `:lines_only` (PSS/E control code 1) counts tie flows. `:lines_and_loads` (control code 2) also counts each area's `out_of_area_loads`: a load owned by area `o` at a bus in area `b` counts as an export of `b` to `o` (+P in ``NI_b``, −P in ``NI_o``), with its ZIP parts evaluated at the bus voltage. |

Area interchange control is **polar-only**. `GradientDescentACPowerFlow` and
`RobustHomotopyPowerFlow` are rejected at construction (gradient descent has no natural home for
the interchange border; robust homotopy would require exact second-order curvature for every tie
term, which is not maintained), as is any non-polar formulation.

## Formulation

Each enrolled area contributes one state and one residual row, appended as a tail after the LCC/VSC
tails:

  - **State** ``\Delta P_a``: an active-power adjustment injected at the area's slack (ISW) bus.
    It enters that bus's P-balance row in the same way the distributed-slack term does.
  - **Residual** ``r_a = NI_a - PDES_a``, where ``NI_a`` is the sum of metered active power over the
    area's boundary ties, signed positive out of the area.

Solving drives every ``r_a`` to the same tolerance as the bus mismatch rows, so the schedules hold
exactly at convergence while ``\Delta P_a`` reports how much the area's slack generation moved to
achieve them.

If the ISW bus hits a reactive limit and is switched PV → PQ mid-solve, area control keeps
working and nothing needs to be re-enrolled. ``\Delta P_a`` couples into the P-balance row at
index `2 * slack_bus_ix - 1`, which does not depend on bus type, so the Jacobian entry stays
put across the flip. ``\Delta P_a`` is also applied directly to the residual vector rather than
folded into the bus's accumulated net injection, precisely so it is counted exactly once after
the bus becomes PQ. What *is* checked at enrollment (not mid-solve) is that the ISW bus is
voltage-regulating to begin with: an area whose slack bus normalizes to PQ is de-enrolled.

### Control code 2 (`:lines_and_loads`, PSS/E AREAIN=2)

With `tie_definition = :lines_and_loads`, the net interchange also counts off-area loads:

``NI_a = \sum(\text{tie flows out of } a) - \sum P(\text{loads owned by } a \text{ at buses in other areas}) + \sum P(\text{loads owned by other areas at buses in } a)``

The owner serves its off-area load through the ties, so that tie flow is not an export. With zero
schedules and the tie flow out of the owner equal to ``P_{load}``, ``NI_{owner} = 0``. Only active
power counts, and the ZIP parts of the load are evaluated at the bus voltage.

  - **PDES source**: `Area.scheduled_net_interchange` (positive = export). `nothing` means the area is
    not enrolled. PSS/E's PTOL has no field in `PSY.Area`; PowerFlows keeps `interchange_tolerance`.
  - **Off-area loads**: declare them in `Area.out_of_area_loads` of the owning area. A listed load at a
    bus in the same area is a data error.
  - **PSS/E files**: the reader maps AREAIN=2 to `:lines_and_loads`. The exporter writes one AREA
    INTERCHANGE record per area with a schedule (ISW = the area's `SLACK` bus, PTOL = 10.0 MW) and
    writes the owning area in the AREA field of each LOAD record.

### Ties

**AC ties** are in-service AC branches whose endpoints lie in different areas (a
`ThreeWindingTransformer` decomposes into its star-node windings, so a winding crossing the
boundary is a correctly metered tie). Which end of an AC tie is metered follows the branch's
`ext["metered_end"]` (`"from"`/`"to"`, defaulting to from-metered when the key is absent), the
same rule DC ties use; an unrecognized value warns and falls back to from-metered rather than
being silently coerced. The tie-flow kernel reads each corridor's admittances
directly from the aggregate Y-bus. Because a nodal Y-bus diagonal sums *every* device at a bus, a
per-tie `diag_pollution` correction, cached at enrollment, recovers the corridor's own
self-admittance; this stays exact under a controlled tap on the corridor itself.

**DC ties** are two-terminal HVDC links (LCC or VSC) whose converter buses lie in different
areas. The metered-terminal converter active power enters ``NI_a`` with the same out-of-area sign
convention as AC ties, and metering follows the same `ext["metered_end"]` rule. The Jacobian carries the exact cross-derivatives of ``r_a`` with
respect to the DC state: for an LCC, ∂P/∂(terminal voltage, tap, thyristor angle) at the metered
terminal; a VSC's metered power is linear in its converter-power state. Series FACTS and
generalized network elements are **not supported models** in PowerSystems/PowerFlows, so ties
through them are out of scope by definition, not a gap in the tie enumeration.

### How each solver carries the interchange border

  - **Newton–Raphson / Trust Region**: the ``\Delta P_a`` columns and ``r_a`` rows are embedded
    directly in the augmented Jacobian.
  - **Levenberg–Marquardt**: the same augmented rows feed the least-squares normal equations; no
    separate machinery.
  - **Fast Decoupled, fixed-Jacobian variant**: the frozen Jacobian is the augmented one, so the
    border is factored once with everything else.
  - **Fast Decoupled, classic B′/B″ variant**: a bordered-Schur substep corrects ``\theta`` and
    ``\Delta P_a`` each cycle against the fixed B′ factor. The *step* uses only the ``\partial r_a/\partial\theta`` coupling (DC-tie and voltage cross-terms are dropped, consistent with the
    decoupling approximation), but the *target* is evaluated with the exact residual — including
    DC ties — so schedules are still met exactly at convergence.

## Enrollment guards

An area is **enrolled** only if it can be embedded-controlled. At `PowerFlowData` construction each
candidate is checked, and any failure emits a `@warn` and de-enrolls that area (its schedule then
falls back to the island's REF/distributed slack). An area must:

  - have a scheduled net interchange (an area whose `scheduled_net_interchange` is `nothing` is
    not controlled and floats with the system swing; this logs `@info`, not `@warn`);
  - have exactly one in-service, voltage-regulating slack bus that survives network reduction;
  - not contain a network reference (REF/swing) bus;
  - lie within a single electrical island;
  - hold less than `AREA_SLACK_ABSORPTION_LIMIT` (0.9) of system slack-participation weight;
  - have at least one in-service tie to another area.

A tie-endpoint tap or switched shunt that is not a corridor member is a `diag_pollution` staleness
hazard; it is flagged with a `@warn` (net-interchange tracking then assumes it holds its
enrollment-time value) rather than de-enrolled.

## Infeasibility: greedy relax

A schedule can be unenforceable given the network and tie capacity. On a non-converged time step with
areas still enrolled, the driver de-enrolls the area with the largest residual gap ``|r_a|``, emits
an `@error` naming it (with its achieved net interchange, AC and DC ties included), and re-solves
with the rest — repeating until the solve converges or no controlled area remains. Relaxation is
never silent. Relax decisions are **per time step**: the full enrollment (including the DC-tie set
and all solver caches) is restored before the next time step's own attempt, and likewise re-grown
on a later re-solve of the same data.

## Results

`solve_and_store_power_flow!` / `write_results` add an `"area_interchange_results"` DataFrame, one row
per area enrolled at construction:

| Column            | Meaning                                                                                  |
|:----------------- |:---------------------------------------------------------------------------------------- |
| `area`            | PSY area name.                                                                           |
| `ni_solved`       | Achieved net interchange (recomputed from the tie kernels, AC and DC ties).              |
| `pdes`            | Scheduled target.                                                                        |
| `delta_p`         | Solved ``\Delta P_a`` (`0.0` for a relaxed area).                                        |
| `schedule_status` | `:enforced` or `:relaxed`.                                                               |
| `beyond_limits`   | Whether the slack bus's **total** solved output exceeds its machine active-power limits. |

For a relaxed area, `ni_solved - pdes` is its infeasibility certificate.

`beyond_limits` compares the machines' summed `active_power_limits` against the slack bus's
whole solved active output — its scheduled injection **plus its distributed-slack share plus**
``\Delta P_a`` — so the two effects compound rather than being checked in isolation. An area
whose ``\Delta P_a`` fits on its own can still be flagged once its slack-participation share is
added. The flag never clamps anything; it is a diagnostic.
