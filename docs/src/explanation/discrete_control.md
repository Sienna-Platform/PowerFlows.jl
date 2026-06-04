# Discrete Control Devices via λ-Continuation

PowerFlows.jl supports two families of voltage-controlling discrete devices:
**tap-changing transformers** and **switched (stepping) shunts**. This page
explains what those devices are, why an outer-loop continuation strategy is
used instead of embedding the control law in the Jacobian, how the sigmoid
control law and its homotopy ramp work, and what happens when the continuous
solution is snapped to discrete settings.

## What and why

Commercial power flow tools (PSS/E, PSLF) simulate discrete control devices as
part of steady-state solution: a tap-changing transformer adjusts its turns
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
existing inner solvers in an **outer-loop λ-continuation**:

- The inner solver ([`NewtonRaphsonACPowerFlow`](@ref),
  [`TrustRegionACPowerFlow`](@ref)) is called without any modification.
- Between outer iterations only `data` is mutated: Y-bus `nzval` entries for
  tap devices; the reactive constant-impedance withdrawal matrix for shunt
  devices.
- The outer loop is **formulation-agnostic**: it works identically for
  [`ACPolarPowerFlow`](@ref), [`ACRectangularPowerFlow`](@ref), and
  [`ACMixedPowerFlow`](@ref) because it calls `_solve_with_q_limits!`, the
  existing Q-limit loop, as a black box.

The dispatch point is `_ac_power_flow` in `solve_ac_power_flow.jl`: when
`data.controlled_devices` is non-empty it delegates to
`_control_continuation!`; otherwise the existing path is taken unchanged
(regression invariant).

A reserved interface method `stamp_control!` exists on every device type but
is never called by the outer loop; it is an error stub that marks where a
future per-formulation implicit embedding would dispatch. Similarly,
`ControlledPhaseShifter` and `ControlledFACTS` are typed seams in the
hierarchy whose interface methods all call an internal `_seam_err` — they
occupy the right dispatch slots but are not implemented.

## Device abstraction

Two abstract families sit under `AbstractControlledDevice`:

- `AbstractBranchControl` — devices that mutate the branch's 2×2 Y-bus block.
  The only implemented leaf is `ControlledTap` (voltage-controlling
  `TapTransformer`).
- `AbstractShuntControl` — devices that mutate the bus reactive
  constant-impedance withdrawal. The only implemented leaf is
  `ControlledSwitchedShunt` (voltage-controlling `SwitchedAdmittance`).

The runtime container is `ControlledDeviceSet`, which holds
`Vector{ControlledTap}` and `Vector{ControlledSwitchedShunt}` as concretely
typed fields. All outer-loop traversal iterates `for d in set.taps` and `for
d in set.shunts`, so dispatch is monomorphic and the hot-path kernels are
allocation-free.

For `ControlledTap`, `apply_parameter!` rewrites three of the four cached `nzval`
entries of the sparse Y-bus in place (Y11, Y12, Y21) using cached linear offsets
(`nz_offsets::NTuple{4,Int}`) resolved once at device-set construction; Y22 = Yt
is tap-independent and is skipped. The delta update `nzval[k] += Y_new − Y_old`
preserves any parallel-branch contributions already in the shared slot;
`d.current` (not the lossy `nzval`) is the authoritative source for the old
parameter. For `ControlledSwitchedShunt`, `apply_parameter!` applies the
analogous reactive delta to
`data.bus_reactive_power_constant_impedance_withdrawals[bus_ix, ts]`, so
co-located constant-impedance contributions on the same bus are preserved.

## The sigmoid control law

The continuous target for each device is a sigmoid function of the controlled
bus voltage magnitude $|V|$:

$$\sigma(\ell, h, S, x, x_\text{set}) = \frac{h - \ell}{1 + e^{S(x - x_\text{set})}} + \ell$$

For a **tap transformer** controlled on its primary side (equation 46 of the
thesis), the sigmoid maps a low voltage to a high tap ratio and a high voltage
to a low tap ratio: $\ell = p_\min$, $h = p_\max$. When the controlled bus is
on the secondary side (equation 47), the limits are swapped
($\ell = p_\max$, $h = p_\min$), making the sigmoid increasing rather than
decreasing. The result is clamped to $[p_\min, p_\max]$.

For a **switched shunt** (equation 9), low voltage calls for more susceptance
and high voltage for less: $\ell = b_\min$, $h = b_\max$; the sigmoid is
decreasing in $|V|$.

The steepness $S$ controls how closely the smooth sigmoid approximates the
step function. It starts at `INITIAL_CONTROL_STEEPNESS = 100` and is ramped
by `CONTROL_STEEPNESS_GROWTH = 2.0` after each settling phase, up to
`MAX_CONTROL_STEEPNESS = 5000` (values from paper equation 10 / thesis). The
ramp happens only after the devices have settled at the current $S$, so the
solver is never asked to handle a stiff sigmoid before the network state is
compatible with it.

### Plant-sign correction

The sigmoid formulas above assume a particular orientation between the
controlled device and the controlled bus. Real networks can wire either end of
a transformer to either bus, so the closed-loop gain may be positive or negative
with the nominal sigmoid orientation. `_plant_sign` measures the sensitivity
$\partial|V|/\partial p$ by a small perturbation at the current parameter
value. If the resulting plant gain is positive, it means the nominal sigmoid
orientation (decreasing for primary-controlled tap) would produce positive
feedback; the `flip` flag is set and the sigmoid limits are swapped
(`_control_target` passes `(hi, lo)` instead of `(lo, hi)`), restoring
negative feedback regardless of device wiring.

## The continuation engine

The outer loop `_control_continuation!` runs up to
`MAX_CONTROL_OUTER_ITERATIONS = 100` passes. Each pass:

1. Computes the sigmoid target $p^*$ for every device given the current
   controlled-bus voltage and steepness $S$.
2. Applies an **adaptive under-relaxation** step
   $p \leftarrow p + \omega(p^* - p)$.
3. Applies each update via `_continuation_to!`, the incremental robust
   applicator.

**Under-relaxation.** The damped iteration $p \leftarrow p + \omega(p^* - p)$ has
local slope $m = 1 + \omega(g' - 1)$, where $g' = \sigma'(|V|)\cdot\partial|V|/\partial p \le 0$
after sign correction. The factor $\omega$ is chosen to keep $m$ non-negative
(monotone, $0 \le m < 1$) with target $m \ge \theta$, i.e.

$$\omega \leq \frac{1 - \theta}{1 + |g'|}$$

where `CONTROL_CONTRACTION = 0.5` is the contraction target $\theta$ and $g'$ is
the closed-loop gain bound $|h - \ell| \cdot S/4 \cdot |\partial|V|/\partial p|$
(the maximum sigmoid derivative times the plant gain). This guarantees monotone
convergence at every steepness without re-measuring the plant gain each
iteration. The cap `CONTROL_RELAXATION_MAX = 0.8` prevents $\omega$ from being
set too high when $g'$ is near zero.

**Incremental applicator.** `_continuation_to!` walks the parameter from its
current value to the relaxed target in sub-steps. The first sub-step is
`MIN_LAMBDA_STEP = 1e-3` of the total interval; successful NR calls allow the
step to grow (factor `CONTROL_STEP_GROWTH = 1.5`, capped at
`MAX_LAMBDA_STEP = 1.0`); a failed NR call halves the step. If the step falls
below `MIN_LAMBDA_STEP` the parameter stays at the last converged point and
the applicator returns the reached value.

**Settling and steepness ramp.** A pass is considered settled when every device
moves by less than `CONTROL_PARAM_TOL = 1e-5`. Steepness $S$ is advanced only
after settling, not after every iteration. If `MAX_CONTROL_OUTER_ITERATIONS`
is reached before full-steepness convergence, an honest `@warn` is emitted
(regulation may be loose); the solver does not claim failure on account of
steepness alone.

**Oscillation safeguard.** If the direction of a device's parameter update
reverses sign on more than `CONTROL_OSCILLATION_LIMIT = 3` consecutive outer
passes, the device is frozen at its current parameter and a `@warn` is emitted.
The frozen device returns `Inf` as its gap, so the outer loop does not count
it as settled while other devices continue to converge. This mirrors the
PSS/E behavior of locking oscillating adjustments after a fixed iteration
count.

## Snap and feasibility restoration

After the continuous outer loop converges (or reaches its iteration limit),
`snap_and_restore!` discretizes every device:

- **Tap.** The continuous parameter is snapped to the nearest value in the
  pre-computed `levels` vector (a uniform grid of `NTP` tap positions from
  `p_min` to `p_max`).
- **Shunt.** The target susceptance is placed using a block-greedy algorithm:
  blocks are processed largest-first; each block's step count is floored to
  avoid overshooting; a single ±1 bounded refinement pass corrects
  under-committed blocks. Both `block_order` and `block_n` are pre-allocated
  fields, so `snap_to_discrete` makes no heap allocations.

After snapping all devices, the inner solver is called on the discretized
network. If it converges, the procedure is complete. If not, each device is
individually restored toward its pre-snap continuous value using the same
bisection sub-stepping as `_continuation_to!` (`_restore_one!`). If any
device cannot be restored to a converged state, `data.converged[ts] = false`
is set and an `@error` is emitted with the device names; no non-physical
solution is silently returned.

## Metadata sourcing

Device parameters are sourced from the PSS/E parser `ext` dictionary on each
`TapTransformer` or `SwitchedAdmittance`, with documented defaults when keys
are absent.

### `TapTransformer` → `ControlledTap`

Only transformers with `get_control_objective(tx) == VOLTAGE` are included.

| Parameter | `ext` key | Default |
|---|---|---|
| Controlled bus | `NREG` or `RMIDNT` (bus number) | transformer to-bus (secondary) |
| Voltage setpoint | `VSET` | controlled bus `magnitude`, else `1.0` |
| Tap ratio min | `RMI` | `DEFAULT_TAP_RATIO_MIN = 0.9` |
| Tap ratio max | `RMA` | `DEFAULT_TAP_RATIO_MAX = 1.1` |
| Tap positions | `NTP` | `DEFAULT_TAP_POSITIONS = 33` |
| Controlled side | derived: controlled bus == from-bus → primary (eq. 46), else secondary (eq. 47) | — |

The discrete tap levels are `range(p_min, p_max; length=NTP)`, collected once
at construction.

### `SwitchedAdmittance` → `ControlledSwitchedShunt`

| Parameter | Source |
|---|---|
| Controlled bus | `ext["NREG"]` or `ext["RMIDNT"]`, else own bus |
| Voltage setpoint | midpoint of `get_admittance_limits` (VSWLO/VSWHI band) |
| Susceptance limits | `get_Y` base + Σ over blocks of `number_of_steps .* imag.(Y_increase)` |
| Block structure | `get_number_of_steps`, `get_Y_increase`, `get_initial_status` |

If the controlled bus number cannot be resolved to a network index,
`build_controlled_device_set` raises an error immediately; there is no silent
skip.

## Reserved seams for future work

`stamp_control!(d::AbstractControlledDevice, args...)` is defined on the
abstract base and calls `error("implicit embedding not implemented for ...")`.
It is the attachment point for a future per-formulation implicit embedding
(sigmoid term added to the residual and corresponding Jacobian row), which
would allow the inner Newton iteration to solve for device parameters
simultaneously with bus voltages. Such an embedding would be added per
formulation type by overloading `stamp_control!`; the outer-loop code and the
device hierarchy would require no changes.

`ControlledPhaseShifter <: AbstractBranchControl` and
`ControlledFACTS <: AbstractShuntControl` are stub types whose interface
methods all call `_seam_err`. They hold the correct position in the dispatch
hierarchy for active-power phase-shifting and FACTS device control, which are
not implemented in this scope.
