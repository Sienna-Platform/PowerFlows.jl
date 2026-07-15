# Area Interchange Control

## What it does

**Net-interchange control**: each participating `PSY.Area` is driven so that its net
active-power exchange with the rest of the system meets a scheduled target ``PDES_a`` (derived from
the system's `PSY.AreaInterchange` records). Control is *embedded* in the AC solver — the
schedules are met at convergence, not by an outer sweep — so it composes with reactive-power limits,
distributed slack, and multi-period solves. Both AC boundary branches and two-terminal HVDC
(LCC and VSC) boundary links count toward each area's net interchange.

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

| Keyword                    | Default       | Meaning                                                                                                                                            |
|:-------------------------- |:------------- |:-------------------------------------------------------------------------------------------------------------------------------------------------- |
| `area_interchange_control` | `false`       | Turn embedded control on.                                                                                                                          |
| `interchange_tolerance`    | `0.05`        | PTOL analogue (pu), **reporting/validation only** — the embedded rows target `PDES_a` exactly. Non-positive values floor to `0.02` with a warning. |
| `tie_definition`           | `:lines_only` | Only `:lines_only` is implemented (`:lines_and_loads` / PSS/E control code 2 is reserved).                                                         |

Area interchange control is **polar-only**. `GradientDescentACPowerFlow` and
`RobustHomotopyPowerFlow` are rejected at construction (gradient descent has no natural home for
the interchange border; robust homotopy would require exact second-order curvature for every tie
term, which is not maintained), as is any non-polar formulation.

## Formulation

Each enrolled area contributes one state and one residual row, appended as a tail after the LCC/VSC
tails:

  - **State** ``\Delta P_a``: an active-power adjustment injected at the area's slack bus (the
    area-slack machine, PSS/E ISW). It couples into that bus's P-balance row at the same seam as
    the distributed-slack term.
  - **Residual** ``r_a = NI_a - PDES_a``, where ``NI_a`` is the sum of metered active power over the
    area's boundary ties, signed positive out of the area.

Solving drives every ``r_a`` to the same tolerance as the bus mismatch rows, so the schedules hold
exactly at convergence while ``\Delta P_a`` reports how much the area's slack generation moved to
achieve them.

### Ties

**AC ties** are in-service AC branches whose endpoints lie in different areas (a
`ThreeWindingTransformer` decomposes into its star-node windings, so a winding crossing the
boundary is a correctly metered tie). The tie-flow kernel reads each corridor's admittances
directly from the aggregate Y-bus. Because a nodal Y-bus diagonal sums *every* device at a bus, a
per-tie `diag_pollution` correction, cached at enrollment, recovers the corridor's own
self-admittance; this stays exact under a controlled tap on the corridor itself.

**DC ties** are two-terminal HVDC links (LCC or VSC) whose converter buses lie in different
areas. The metered-terminal converter active power enters ``NI_a`` with the same out-of-area sign
convention as AC ties. Metering follows the branch's `ext["metered_end"]` (`"from"`/`"to"`,
defaulting to from-metered). The Jacobian carries the exact cross-derivatives of ``r_a`` with
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
    ``\Delta P_a`` each cycle against the fixed B′ factor. The *step* uses only the ``\partial
    r_a/\partial\theta`` coupling (DC-tie and voltage cross-terms are dropped, consistent with the
    decoupling approximation), but the *target* is evaluated with the exact residual — including
    DC ties — so schedules are still met exactly at convergence.

## Enrollment guards

An area is **enrolled** only if it can be embedded-controlled. At `PowerFlowData` construction each
candidate is checked, and any failure emits a `@warn` and de-enrolls that area (its schedule then
falls back to the island's REF/distributed slack). An area must:

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

| Column            | Meaning                                                                       |
|:----------------- |:----------------------------------------------------------------------------- |
| `area`            | PSY area name.                                                                |
| `ni_solved`       | Achieved net interchange (recomputed from the tie kernels, AC and DC ties).   |
| `pdes`            | Scheduled target.                                                             |
| `delta_p`         | Solved ``\Delta P_a`` (`0.0` for a relaxed area).                             |
| `schedule_status` | `:enforced` or `:relaxed`.                                                    |
| `beyond_limits`   | Whether ``\Delta P_a`` exceeds the slack bus's machine active-power headroom. |

For a relaxed area, `ni_solved - pdes` is its infeasibility certificate.
