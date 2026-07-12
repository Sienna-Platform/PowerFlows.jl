# Discrete Control Devices via λ-Continuation

PowerFlows.jl supports three families of controlled devices, solved by a common
outer-loop continuation: **tap-changing transformers**, **switched (stepping)
shunts**, and **shunt FACTS devices** (SVC/STATCOM). This page explains what
those devices are, why an outer-loop continuation strategy is used instead of
embedding the control law in the Jacobian, how the sigmoid control law and its
steepness ramp work, and what happens when the continuous solution is snapped
to discrete settings.

## What and why

Commercial power flow tools (PSS/E, PSLF) simulate discrete control devices as
part of the steady-state solution: a tap-changing transformer adjusts its turns
ratio to regulate the voltage at a designated bus; a switched shunt switches
reactive admittance blocks in or out for the same purpose. Without modeling
these devices, the solved bus voltages can differ materially from what a
commercial tool reports, because the network seen by the solver is wrong.

The natural mathematical formulation (Agarwal, Pandey, Jereminov & Pileggi,
arXiv 1811.02000, §IV.C; Pandey PhD thesis 2018, §5.2.8.1) embeds a
differentiable sigmoid approximation of the discrete control law directly in
the residual and Jacobian, so the Newton iteration solves for device parameters
simultaneously with bus voltages. PowerFlows.jl deliberately does **not** use
that implicit embedding for the initial implementation. Instead it wraps the
existing inner solvers in an **outer-loop continuation** (the sigmoid law and
steepness schedule follow the paper; the outer-loop reinterpretation and its
convergence safeguards are original to PowerFlows.jl):

  - The inner solver ([`NewtonRaphsonACPowerFlow`](@ref),
    [`TrustRegionACPowerFlow`](@ref)) is called without any modification.
  - Between outer iterations only the following fields of `data` are mutated:
    Y-bus `nzval` entries for tap devices; the reactive constant-impedance
    withdrawal matrix for shunt/FACTS devices.
  - The outer loop is **formulation-agnostic**: it works identically for
    [`ACPolarPowerFlow`](@ref), [`ACRectangularPowerFlow`](@ref), and
    [`ACMixedPowerFlow`](@ref) because it calls `_solve_with_q_limits!`, the
    existing Q-limit loop, as a black box.

The dispatch point is `_ac_power_flow` in `solve_ac_power_flow.jl`: when
`data.controlled_devices` is non-empty it delegates to
`_control_continuation!`; otherwise the existing path is taken unchanged
(regression invariant).

## [Scope and guards](@id discrete-control-guards)

`control_discrete_devices = true` is validated at construction and rejected
with an `ArgumentError` for combinations that are not yet supported:

  - **Solvers:** only `NewtonRaphsonACPowerFlow` and `TrustRegionACPowerFlow`
    inner solvers. (FastDecoupled factors B′/B″ once and would silently reuse
    them after a tap move; LM/GD/Homotopy are unvalidated as continuation inner
    solvers.)
  - **Single time step:** `time_steps > 1` is rejected — device state does not
    yet track per-time-step baselines (taps mutate the shared Y-bus; shunt
    deltas are per-column).
  - **No LCC HVDC:** the continuation's rollback does not yet cover the
    per-time-step LCC state.

Per-device data problems (unresolvable controlled buses, degenerate tap/step
ranges, unsupported MODSW modes, implausible voltage setpoints) never abort
construction: the device is de-enrolled with a `@warn` and stays at its current
setting — the same *warn and lock* posture PSS/E takes for bad control data.

`ControlledFACTS` is enrolled unconditionally (no flag); its metadata sourcing
caveats are noted in [Metadata sourcing](@ref discrete-control-metadata).

## Device abstraction

Two abstract families sit under `AbstractControlledDevice`:

  - `AbstractBranchControl` — devices that mutate the branch's 2×2 Y-bus block:
    `ControlledTap` (voltage-controlling `TapTransformer`).
  - `AbstractShuntControl` — devices that mutate the bus reactive
    constant-impedance withdrawal: `ControlledSwitchedShunt`
    (voltage-controlling `SwitchedAdmittance`) and `ControlledFACTS`
    (SVC/STATCOM).

The runtime container is `ControlledDeviceSet`, which holds one concretely
typed `Vector` per family. All outer-loop traversal iterates each vector
separately, so each element has a concrete type and no dynamic dispatch occurs;
the per-device kernels
(`apply_parameter!`, `snap_to_discrete`) are allocation-free (the outer loop
itself allocates small snapshot buffers, which is immaterial next to the inner
solves).

For `ControlledTap`, `apply_parameter!` rewrites three of the four cached `nzval`
entries of the sparse Y-bus in place (Y11, Y12, Y21) using cached linear offsets
(`nz_offsets::NTuple{4,Int}`) resolved once at device-set construction; Y22 = Yt
is tap-independent and is skipped. The complex tap includes the winding-group
phase shift `α = PSY.get_α(tx)`, matching PowerNetworkMatrices' stamping
`t = p·e^{iα}`. The delta update `nzval[k] += Y_new − Y_old` preserves any
parallel-branch contributions already in the shared slot; the device's stored
`current` value (not the lossy `nzval`) is the authoritative source for the old
parameter. For
`ControlledSwitchedShunt`, `apply_parameter!` applies the analogous reactive
delta to `data.bus_reactive_power_constant_impedance_withdrawals[bus_ix, ts]`,
so co-located constant-impedance contributions on the same bus are preserved.

## The sigmoid control law

The continuous target for each device is a sigmoid function of the regulated
quantity ``y`` (the controlled-bus voltage magnitude):

```math
\sigma(\ell, h, S, y, y_\text{set}) = \frac{h - \ell}{1 + e^{S(y - y_\text{set})}} + \ell
```

The steepness ``S`` controls how closely the smooth sigmoid approximates the
step function.

```@eval
using Markdown, PowerFlows
Markdown.parse(
    "It starts at `INITIAL_CONTROL_STEEPNESS = $(PowerFlows.INITIAL_CONTROL_STEEPNESS)` and is " *
    "ramped by `CONTROL_STEEPNESS_GROWTH = $(PowerFlows.CONTROL_STEEPNESS_GROWTH)` after each " *
    "settling phase, up to `MAX_CONTROL_STEEPNESS = $(PowerFlows.MAX_CONTROL_STEEPNESS)` " *
    "(values from paper equation 10 / thesis).",
)
```

The ramp happens only after the devices have settled at the current ``S``, so the
solver is never asked to handle a stiff sigmoid before the network state is
compatible with it.

### Plant-sign orientation and gain tracking

The orientation of the control law comes **entirely from the measured plant
sensitivity**, not from the device's wiring. `_plant_sign` measures
``dy/dp`` by a small parameter perturbation at the converged base point (one
inner solve per device; the full pre-probe state — voltages, bus types, and
injections — is restored afterward, so a probe can never permanently flip a
marginal generator PV→PQ). `_control_target` then evaluates the sigmoid with
its limits ordered so the closed-loop gain ``\sigma'(y)\cdot dy/dp`` is
non-positive (negative feedback): measured ``dy/dp > 0`` selects the
decreasing orientation ``(\ell, h) = (p_\min, p_\max)``; measured
``dy/dp \le 0`` selects the increasing orientation ``(h, \ell)``. There is no
stored flip flag and no wiring-based (primary/secondary) orientation logic.

The gain is not frozen at its probed value: every accepted step refreshes it
with a **secant estimate** from the parameter/response pair the step just
produced (zero extra solves). Three conditions freeze a device at its current
parameter with a warning (PSS/E lock-and-continue): a failed probe solve
(orientation unknown), a full-range effectiveness below `CONTROL_GAIN_FLOOR`
(e.g. a PV-pinned controlled bus, which probes exactly 0 — stepping it would
slam the parameter to a rail with no feedback), and a detected **sign
reversal** of the sensitivity along the trajectory (OLTC reverse action —
continuing would be positive feedback). Devices sharing a controlled bus
split the correction (``\omega / n_\text{shared}``), since N co-located
controllers stepping the full error together have an in-phase gain ≈ N× the
measured self-gain.

## The continuation engine

The outer loop `_control_continuation!` runs a steepness ladder of ~7 stages,
each with its own budget of `MAX_CONTROL_PASSES_PER_STAGE = 20` passes. Each
pass:

 1. Computes the sigmoid target ``p^*`` for every device given the current
    regulated quantity and steepness ``S``.
 2. Applies an **adaptive under-relaxation** step
    ``p \leftarrow p + \omega(p^* - p)``.
 3. Applies each update via `_continuation_to!`, the incremental robust
    applicator.

**Under-relaxation.** The damped iteration ``p \leftarrow p + \omega(p^* - p)`` has
local slope ``m = 1 + \omega(g' - 1)``, where ``g' = \sigma'(y)\cdot dy/dp \le 0``
after sign correction. The factor ``\omega`` is chosen to keep ``m`` non-negative
(monotone, ``0 \le m < 1``) with target ``m \ge \theta``, i.e.

```math
\omega \leq \frac{1 - \theta}{1 + |g'|}
```

where `CONTROL_CONTRACTION = 0.5` is the contraction target ``\theta`` and ``g'`` is
the closed-loop gain bound ``|h - \ell| \cdot S/4 \cdot |dy/dp|``
(the maximum sigmoid derivative times the plant gain). Note this formula
already caps ``\omega \le 1-\theta = 0.5``, and the guarantee is conditional on
the plant gain measured at the base point remaining representative along the
trajectory — it is a safeguard, not an unconditional convergence proof.

**Deadband.** A switched shunt whose controlled voltage is anywhere inside its
parsed `[VSWLO, VSWHI]` band is held, not driven toward the band midpoint —
the PSS/E semantics. Other device families carry a point setpoint (the parser
persists no band for them) and always regulate.

**Incremental applicator.** `_continuation_to!` tries the full damped move
first — in the common case one warm-started inner solve accepts it. Only when
that fails does it fall back to bisection sub-stepping (starting at half the
interval, growing by `CONTROL_STEP_GROWTH = 1.5` on success and halving on
failure). If the step falls below `MIN_LAMBDA_STEP = 1e-3` the parameter stays
at the last converged point; a device that could not move at all despite a
requested move is frozen with a warning rather than counted as settled.
Intermediate steepness stages solve at a relaxed tolerance
(`CONTROL_STAGE_TOL = 1e-6`); the final stage and the snap/restore solves use
the full tolerance. `get_control_inner_solve_count(data)` reports the
continuation's inner-solve count — the metric its performance is measured by.

**Settling and steepness ramp.** A pass is considered settled when every device
moves by less than its scale-aware tolerance
``\max(10^{-5},\ 10^{-4}\cdot(p_\max - p_\min))``. Each steepness stage has
its own budget of `MAX_CONTROL_PASSES_PER_STAGE = 20` passes, so a
slow-settling early stage cannot starve the stiffer later stages; ``S`` is
advanced when the stage settles (or its budget runs out). If the final stage
does not settle, an honest `@warn` is emitted (regulation may be loose); the
solver does not claim failure on account of steepness alone.

**Oscillation safeguard.** If the direction of a device's parameter update
reverses sign more than `CONTROL_OSCILLATION_LIMIT = 3` times **within one
steepness stage** (the counter and direction memory reset at each ramp, since
a ramp legitimately reverses the update direction once; sub-tolerance moves
carry no direction information), the device is frozen at its current parameter
with a `@warn`. Frozen devices count as settled — one locked device does not
block the steepness ramp for the healthy ones. This mirrors the PSS/E behavior
of locking oscillating adjustments after a fixed iteration count.

## Snap and feasibility restoration

After the continuous outer loop converges (or reaches its iteration limit),
`snap_and_restore!` discretizes every device:

  - **Tap.** The continuous parameter is snapped to the nearest value in the
    pre-computed `levels` vector (a uniform grid of `NTP` tap positions from
    `p_min` to `p_max`).
  - **Shunt.** The target susceptance is snapped to the nearest point of the
    PSS/E cumulative block-activation chain: blocks switch on in listed order
    (and off in reverse), so the physically realizable totals are exactly the
    prefix sums of the block steps. Walking that chain is simultaneously
    optimal over the realizable set and order-respecting; `block_n` records
    the chosen per-block step counts, and `snap_to_discrete` makes no heap
    allocations. (Continuous devices — MODSW=2 shunts, FACTS — clamp instead
    of snapping.)

After snapping all devices, the inner solver is called on the discretized
network. If it converges, the procedure is complete. If not, each device is
individually restored toward its pre-snap continuous value using the same
bisection sub-stepping as `_continuation_to!` (`_restore_one!`). If any
device cannot be restored to a converged state, `data.converged[ts] = false`
is set and an `@error` is emitted with the device names; no non-physical
solution is silently returned.

## [Metadata sourcing](@id discrete-control-metadata)

Device parameters are sourced from the PSY component's first-class fields when
the installed PowerSystems.jl provides them (PSY ≥ 6 via PSY #1705 / the psy6
development branch — detected once at load time), and otherwise from the PSS/E
parser's `ext` dictionary with documented defaults. The `ext` keys the parser
actually writes are winding-suffixed; unsuffixed spellings are accepted as
user-facing overrides.

### `TapTransformer` → `ControlledTap`

Only transformers with `get_control_objective(tx) == VOLTAGE` are included.

| Parameter        | `ext` keys (parser first)            | First-class field (PSY ≥ 6)               | Default on PSY 5.x |
|:---------------- |:------------------------------------ |:----------------------------------------- |:------------------ |
| Controlled bus   | `CONT1`, `NREG`                      | `regulated_bus_number` (0 ⇒ local/to-bus) | to-bus             |
| Tap ratio min    | `RMI1`, `RMI`                        | `tap_limits.min`                          | `0.9`              |
| Tap ratio max    | `RMA1`, `RMA`                        | `tap_limits.max`                          | `1.1`              |
| Tap positions    | `NTP1`, `NTP`                        | `number_of_tap_positions`                 | `33`               |
| Voltage setpoint | `VSET` (never written by the parser) | `voltage_setpoint`                        | `1.0`              |

Precedence: a nonzero `regulated_bus_number` wins for the controlled bus;
for the numeric parameters the `ext` keys win over the first-class fields
(so parsed PSS/E data keeps working until the parser populates the fields).
The discrete tap levels are `range(p_min, p_max; length=NTP)`, collected once
at construction. The winding-group phase shift is taken from `PSY.get_α(tx)`.
Setpoints outside `[0.5, 1.5]` p.u. de-enroll the device with a warning.

### `SwitchedAdmittance` → `ControlledSwitchedShunt`

| Parameter         | Source                                                                                                           |
|:----------------- |:---------------------------------------------------------------------------------------------------------------- |
| Control mode      | `ext["MODSW"]`: 0 ⇒ locked (skipped); 1 ⇒ discrete; 2 ⇒ continuous; ≥3 ⇒ unsupported (warn + lock). Absent ⇒ 1.  |
| Controlled bus    | `ext["NREG"]` (v35) or `ext["SWREM"]` (v32/33), else own bus                                                     |
| Voltage setpoint  | midpoint of `get_admittance_limits` — the VSWLO/VSWHI band for parsed systems                                    |
| Susceptance range | spanned by the blocks: `[Σ min(steps·dB, 0), Σ max(steps·dB, 0)]` (plus the fixed base for API-built components) |
| Block structure   | `get_number_of_steps`, `get_Y_increase`, `get_initial_status`                                                    |

Two `Y`/`initial_status` conventions exist and are auto-detected via the
presence of the parser's `MODSW` key: the **PSS/E parser** stores
`Y = BINIT` (the *total* in-service admittance) and zeroes `initial_status`,
so the reachable range is spanned by the blocks alone with the current point
at BINIT; **API-built** components follow the PSY docstring (`Y` = fixed N=0
base, `initial_status` meaningful).

### `FACTSControlDevice` → `ControlledFACTS`

A `FACTSControlDevice` in a non-out-of-service control mode enrolls
unconditionally (no flag) as a *continuous* shunt: it varies a symmetric
susceptance `b` to hold its regulated bus at `voltage_setpoint`, injecting
`Q = b·|V|²`.

| Parameter          | `ext` keys (parser first)       | First-class field (PSY ≥ 6)                    | Default on PSY 5.x |
|:------------------ |:------------------------------- |:---------------------------------------------- |:------------------ |
| Regulated bus      | `FCREG` (v35), `REMOT` (v32/33) | `regulated_bus_number` (0 ⇒ local/sending bus) | local bus          |
| Voltage setpoint   | `VSET`                          | `voltage_setpoint`                             | `1.0`              |
| Shunt current cap  | `SHMX`                          | `max_shunt_current` (MVA at unity voltage)     | `9999`             |
| Reactive power cap | —                               | `max_reactive_power` (MVA)                     | `9999`             |
| Device class       | —                               | `shunt_control_type` (SVC / STATCOM)           | STATCOM            |

`SHMX` is the shunt *current* capability, **not** a reactive-power limit; the
series-branch current `IMX` and `RMPCT` are out of scope and left in `ext`.
`reactive_power_required` is a solver *output* (the delivered `Q`), not a parsed
input.

**Voltage-dependent limit.** The effective susceptance bound `b ∈ [−b_lim, b_lim]`
is refreshed each outer iteration from the measured regulated-bus voltage `V`,
with `rating = SHMX/S_base` and `q_cap = max_reactive_power/S_base`:

  - **SVC** (susceptance-limited): `b_lim = min(rating, q_cap/V²)`
  - **STATCOM** (current-limited): `b_lim = min(rating/V, q_cap/V²)`

so a STATCOM's reactive capability falls off linearly with voltage (`Q ≈ V·SHMX`
at the current limit) while an SVC's falls off as `V²`. At the bound the clamp
holds `b` there — the homotopy analogue of the PV→PQ Q-limit release. To keep the
refreshing bound from chattering, a device that has been oscillation-frozen holds
its last `b_lim` instead of continuing to track voltage.

**Saturation reporting.** After the solve each FACTS device is classified: one
pinned at `b_lim` while its regulated bus remains off `voltage_setpoint` is
flagged `saturated = true` (with a warning). `get_controlled_device_results`
carries that flag plus `delivered_q_mvar = b·|V|²·S_base` on the FACTS rows, and
under active controls the solved `Q` is written back to the component's
`reactive_power_required`.

## Reserved seams for future work

`stamp_control!(d::AbstractControlledDevice, args...)` is defined on the
abstract base and calls `error("implicit embedding not implemented for ...")`.
It is the attachment point for a future per-formulation implicit embedding
(sigmoid term added to the residual and corresponding Jacobian row), which
would allow the inner Newton iteration to solve for device parameters
simultaneously with bus voltages. Such an embedding would be added per
formulation type by overloading `stamp_control!`; the outer-loop code and the
device hierarchy would require no changes.
