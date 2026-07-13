# Area Interchange Control

## What it does

**Net-interchange control**: each participating `PSY.Area` is driven so that its net
active-power exchange with the rest of the system meets a scheduled target ``PDES_a`` (derived from
the system's `PSY.AreaInterchange` records). Control is *embedded* in the AC Newton system — the
schedules are met at convergence, not by an outer sweep — so it composes with reactive-power limits,
distributed slack, and multi-period solves.

## Enabling it

Set the flag on a polar power flow with an `NewtonRaphsonACPowerFlow` or `TrustRegionACPowerFlow`
solver:

```julia
pf = ACPowerFlow(; area_interchange_control = true)
solve_power_flow(pf, sys)
```

| Keyword                    | Default       | Meaning                                                                                                                                            |
|:-------------------------- |:------------- |:-------------------------------------------------------------------------------------------------------------------------------------------------- |
| `area_interchange_control` | `false`       | Turn embedded control on.                                                                                                                          |
| `interchange_tolerance`    | `0.05`        | PTOL analogue (pu), **reporting/validation only** — the embedded rows target `PDES_a` exactly. Non-positive values floor to `0.02` with a warning. |
| `tie_definition`           | `:lines_only` | Only `:lines_only` is implemented (`:lines_and_loads` / PSS/E control code 2 is reserved).                                                         |

Phase 1 is **polar-only** and validated on NR/TR; unsupported formulation×solver pairs
(Levenberg-Marquardt, Robust Homotopy, Gradient Descent, Fast Decoupled, and any non-polar
formulation) are rejected at construction.

## Formulation

Each enrolled area contributes one state and one residual row, appended as a tail after the LCC/VSC
tails:

  - **State** ``\Delta P_a``: an active-power adjustment injected at the area's slack bus (PSS/E ISW).
    It couples into that bus's P-balance row at the same seam as the distributed-slack term.
  - **Residual** ``r_a = NI_a - PDES_a``, where ``NI_a`` is the sum of metered active power over the
    area's boundary (tie) lines, signed positive out of the area.

The tie-flow kernel reads each corridor's admittances directly from the aggregate Y-bus. Because a
nodal Y-bus diagonal sums *every* device at a bus, a per-tie `diag_pollution` correction, cached at
enrollment, recovers the corridor's own self-admittance; this stays exact under a controlled tap on
the corridor itself.

## Enrollment guards

An area is **enrolled** only if it can be embedded-controlled. At `PowerFlowData` construction each
candidate is checked, and any failure emits a `@warn` and de-enrolls that area (its schedule then
falls back to the island's REF/distributed slack). An area must:

  - have exactly one in-service, voltage-regulating slack bus that survives network reduction;
  - not contain the network reference (REF) bus;
  - lie within a single electrical island;
  - hold less than `AREA_SLACK_ABSORPTION_LIMIT` (0.9) of system slack-participation weight;
  - have at least one in-service tie to another area.

A tie-endpoint tap or switched shunt that is not a corridor member is a `diag_pollution` staleness
hazard; it is flagged with a `@warn` (net-interchange tracking then assumes it holds its
enrollment-time value) rather than de-enrolled.

## Infeasibility: greedy relax

A schedule can be unenforceable given the network and tie capacity. On a non-converged time step with
areas still enrolled, the driver de-enrolls the area with the largest residual gap ``|r_a|``, emits
an `@error` naming it, and re-solves with the rest — repeating until the solve converges or no
controlled area remains. Relax decisions are **per time step**: the full enrollment is restored before
the next time step's own attempt.

## Results

`solve_and_store_power_flow!` / `write_results` add an `"area_interchange_results"` DataFrame, one row
per area enrolled at construction:

| Column            | Meaning                                                                       |
|:----------------- |:----------------------------------------------------------------------------- |
| `area`            | PSY area name.                                                                |
| `ni_solved`       | Achieved net interchange (recomputed from the tie kernel).                    |
| `pdes`            | Scheduled target.                                                             |
| `delta_p`         | Solved ``\Delta P_a`` (`0.0` for a relaxed area).                             |
| `schedule_status` | `:enforced` or `:relaxed`.                                                    |
| `beyond_limits`   | Whether ``\Delta P_a`` exceeds the slack bus's machine active-power headroom. |

For a relaxed area, `ni_solved - pdes` is its infeasibility certificate.
