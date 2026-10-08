# Generalized-Admittance Power Flow

[`GeneralizedAdmittanceACPowerFlow`](@ref) is an AC *solver* (not a formulation) that follows
Artoisenet and Verstraete, "Sparse Generalized-Admittance AC Power Flow for Fast Contingency
Analysis and Remedial-Action Assessment", arXiv:2609.14132. It treats loads and generators as
shunt admittances in the non-slack block of the bus admittance matrix. A fixed point on
*corrective currents* then restores the exact power and voltage-magnitude constraints. The
admittance matrix is factored with KLU and each iteration needs only triangular solves.

This page uses the paper's equation numbers. It also describes the parts that the paper does
not cover: PV stiffening, Anderson mixing, shunt refresh, ZIP loads, LCC and VSC terminals, and
the handoff to a Newton-type solver.

## Notation

Buses are split into four index sets.

| Symbol              | Meaning                                                                                |
|:------------------- |:-------------------------------------------------------------------------------------- |
| ``s``               | Slack (REF) buses. Their voltage ``u_s`` is fixed.                                     |
| ``v``               | PV buses, followed by AC-voltage-controlling VSC buses. Ordered first inside ``\ell``. |
| ``q``               | PQ buses. Ordered after ``v``.                                                         |
| ``\ell = v \cup q`` | All non-slack buses, in the order ``(v, q)``.                                          |

The unknowns and parameters are:

| Symbol         | Meaning                                                                                     |
|:-------------- |:------------------------------------------------------------------------------------------- |
| ``u``          | Complex bus voltages. ``u_\ell`` is the iterate; ``u_s`` is fixed.                          |
| ``u^0``        | Response of the non-slack buses to the slack voltage at zero corrective current.            |
| ``i``          | Corrective currents at the ``\ell`` buses.                                                  |
| ``\tilde u_v`` | Voltage change at the ``v`` buses that the ``v`` currents cause.                            |
| ``y``          | Per-bus shunt admittances at the ``\ell`` buses.                                            |
| ``V^{set}``    | Voltage-magnitude targets at the ``v`` buses.                                               |
| ``\kappa``     | PV stiffness fraction of ``\lvert Y_{kk}\rvert`` (see [PV stiffening](@ref ga-stiffening)). |

The admittance blocks come from the network matrix ``Y^{net}``. Two groups exist, and the
difference matters.

  - **Blocks that include the shunts.** ``Y_{\ell\ell}`` and ``Y_{qq}`` equal the network block
    plus ``\operatorname{diag}(y)`` (paper eq. 3). The code rewrites their diagonal entries every
    time ``y`` changes.
  - **Network-only blocks.** ``Y_{vv}``, ``Y_{vq}``, ``Y_{qv}`` and ``Y_{\ell s}`` never contain
    shunts. The code adds the ``v``-bus shunt term ``y_v \tilde u_v`` explicitly where the paper
    uses the generalized ``Y_{vv}``.

Two KLU factorizations exist: one of ``Y_{\ell\ell}`` and one of ``Y_{qq}``. They keep the same
sparsity pattern for the whole solve, so a change of ``y`` costs only a numeric refactorization.

## Method

### Shunts and corrective currents

A load of power ``s`` at a bus with voltage magnitude ``\lvert u\rvert`` is equal to the shunt
``y = \bar s/\lvert u\rvert^2``. Add that shunt to the diagonal and the network equation holds
with zero corrective current. In practice ``\lvert u\rvert`` at PQ buses and the reactive power
``q`` at PV buses are not known before the solve. The method picks approximate shunts and lets
the corrective current absorb the error (paper eq. 7):

```math
Y_{\ell\ell} u_\ell + Y_{\ell s} u_s = i_\ell .
```

The initial shunts are, for PQ buses (paper eq. 8) and PV buses (paper eq. 9),

```math
y_q = \frac{\bar s_q(\lvert u_q\rvert)}{\lvert u_q\rvert^2},
\qquad
y_v = \frac{\operatorname{Re} s_v(V^{set}) - j q^0_v}{(V^{set})^2},
```

with ``\lvert u_q\rvert`` taken from the input state. The estimate ``q^0`` comes from an
auxiliary flat-start calculation: keep only the imaginary part of ``Y^{net}``, include the PQ
shunts, impose ``V^{set}`` with zero angles, and read the generator reactive power. Here
``s(\lvert u\rvert) = s_P + s_I\lvert u\rvert + s_Z\lvert u\rvert^2`` is the ZIP load model.
The paper treats constant-power loads only; the ZIP terms are an extension.

The slack is eliminated. With ``u_s`` fixed, the zero-current response is
(paper eq. 12)

```math
u^0 = -Y_{\ell\ell}^{-1} Y_{\ell s} u_s ,
```

and the voltages follow from (paper eq. 11)

```math
u_\ell = u^0 + Y_{\ell\ell}^{-1} i_\ell .
```

### One iteration

Each iteration maps the previous currents ``i`` to new voltages and new currents. The
iteration solves with ``Y_{\ell\ell}`` for two right-hand sides (``i`` and ``(0; i_q)``) and once
with ``Y_{qq}``.

 1. Update the voltages: ``u_\ell = u^0 + Y_{\ell\ell}^{-1} i`` (eq. 11 and 14).
 2. Project the PV magnitudes onto the targets and keep the angles (eq. 15):
    ``u_v \leftarrow V^{set} \odot u_v/\lvert u_v\rvert``.
 3. Isolate the part of ``u_v`` that the PV currents cause (eq. 16 and 17):
    ``\tilde u_v = u_v - u^0_v - [Y_{\ell\ell}^{-1}(0; i_q)]_v``.
 4. Compute the raw PV current that produces ``\tilde u_v`` (eq. 19), with ``S_q = Y_{qq}^{-1}``:
    ``i_v^{raw} = Y_{vv}\tilde u_v - Y_{vq} S_q Y_{qv}\tilde u_v + y_v \tilde u_v``.
 5. Update the PQ voltages (eq. 21):
    ``u_q = u^0_q + [Y_{\ell\ell}^{-1}(0; i_q)]_q - S_q Y_{qv}\tilde u_v``.
 6. Set the PQ corrective current so that the constant-power constraint holds (eq. 22):
    ``i_q = (\lvert u_q\rvert^2 y_q - \bar s_q)\, u_q/\lvert u_q\rvert^2``.
 7. Set the PV corrective current to the quadrature component of ``i_v^{raw}`` (eq. 25), so that
    the PV bus keeps its active power and only the reactive power adjusts. The reactive power
    ``q_v`` is extracted from the imaginary part of ``u_v^* i_v^{raw}`` (eq. 24).

The *gap* of the iteration is the largest mismatch of the constant-power constraint over all
``\ell`` buses. At convergence the shunts together with the corrective currents reproduce the
exact bus powers.

### [PV stiffening](@id ga-stiffening)

The choice of ``y`` does not change the solution, because the corrective current absorbs any
error. It does change the speed of the iteration, because it sets the contraction rate. A PV
corrective current ``j(q^0-q)/\bar u`` rotates with the bus angle. To first order, the loop gain
is ``\lvert Z_{th}\rvert\,\lvert\Delta y\rvert``, where ``Z_{th}`` is the Thévenin impedance at
the bus and ``\Delta y`` is the shunt error. On large grids the flat-start ``q^0`` is wrong by
tens of per unit at PV buses that have stiff ties, so the plain fixed point *repels*. The
ACTIVSg10k case shows a linearized spectral radius of about 11.7.

The stage therefore adds an inductive shunt ``-j\kappa\lvert Y_{kk}\rvert`` at each PV bus
(``Y_{kk}`` is the network diagonal entry). The shunt lowers ``\lvert Z_{th}\rvert`` and
restores contraction. The initial value is `GA_PV_STIFFNESS_FRACTION = 0.5`. This value is
empirical, not derived.

### Anderson mixing

The stage mixes the corrective currents with Anderson acceleration (type II, depth
`GA_ANDERSON_DEPTH = 5`). The map conjugates ``u``, so it is only ℝ-linear and not
ℂ-linear. The mixing coefficients are therefore real. The code finds them by least squares over
the stacked real and imaginary parts of the residual differences. If the history becomes
linearly dependent, the code drops it and takes a plain fixed-point step.

### Shunt refresh

A refresh replaces ``y`` with the *ideal* shunts of the current iterate (paper eq. 10). For PV
buses the code uses the extracted ``q_v``; for PQ buses it uses the present ``\lvert u\rvert``.
It then adds the PV stiffness ``\kappa`` again, refactors both KLU factors numerically, and
moves the currents by ``i \mathrel{+}= \Delta y \odot u``. This move offsets the change of
``Y_{\ell\ell}`` at the current iterate, so the next voltages stay close. A refresh also
resets the Anderson history.

## Stage logic

`_ga_run_stage!` runs the loop below. Rows 1 to 4 test the result of an iteration. Rows 5 and 6
choose between a refresh and an Anderson step. The checks run in the order shown.

| # | Trigger                                                                      | Active                       | Effect                                                                                                                                                         |
|:- |:---------------------------------------------------------------------------- |:---------------------------- |:-------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1 | `exit_gap` is not finite                                                     | always                       | Exit `GANonFinite`                                                                                                                                             |
| 2 | `exit_gap` ≤ `stage_tol` (see below)                                         | always                       | Exit `GAConverged`                                                                                                                                             |
| 3 | `gap` > `GA_DIVERGENCE_FACTOR` (``10^3``) × best gap since the last refresh  | always                       | Exit `GADiverged`                                                                                                                                              |
| 4 | Every 20 iterations: best gap > 0.9 × best gap at the previous check         | handoff only                 | Exit `GAStagnated`; the handoff continues from the best iterate                                                                                                |
| 5 | ``10\,\cdot`` gap ≤ gap at the last refresh (iteration 1 sets the reference) | always                       | Drop refresh with ``\kappa = 0``: ideal shunts, numeric refactor, ``i \mathrel{+}= \Delta y \odot u``, Anderson reset, reset of the segment and stall counters |
| 6 | 40 iterations without a 10 % gain on the stall reference                     | in practice, no handoff only | Stall refresh with ``\kappa = \min(\max(4\kappa, 0.05), 0.5)``, then the same refresh as row 5                                                                 |
| 7 | None of rows 5 and 6                                                         | always                       | Anderson step                                                                                                                                                  |
| 8 | Iteration ``k`` = `max_iter` (default 500)                                   | always                       | Exit `GAMaxIter`                                                                                                                                               |

The exit test depends on the handoff setting.

  - **No handoff.** `exit_gap` = max(per-bus gap, largest absolute island active-power sum) and
    `stage_tol` = `tol`. The island sum equals minus the REF active-power row of the polar
    residual after the explicit sync, so this test bounds that row too.
  - **Handoff.** `exit_gap` is the per-bus gap and `stage_tol` = `handoff_tol` (default
    ``10^{-3}``).

Row 6 matters only without a handoff. With a handoff, the stagnation check of row 4 runs every
20 iterations and exits first, before 40 stalled iterations can accumulate. The handoff solver
then takes over. Without a handoff, no solver rescues a slow run, so the stage keeps
iterating and re-stiffens the PV buses on a stall. The cap keeps ``\kappa`` at or below the
initial stiffness, because a larger ``\kappa`` slows the PV loop.

Each iteration also runs the VSC DC substep when the system has VSC converters (see below).
Its largest change of converter active power enters the gap.

### After the stage

 1. Write the best iterate (lowest `exit_gap`) into `PowerFlowData`, and settle the VSC
    state.
 2. Evaluate the polar residual and close the REF, PV, LCC and VSC state with the explicit-state
    sync. A second residual evaluation makes the residual the true residual of the best state.
 3. If a handoff solver is set and the residual does not already meet `tol`, refine the state
    with NR, TR or LM to `tol`.
 4. Without a handoff, check consistency. If the stage reported `GAConverged` but the polar
    residual exceeds `GA_CONSISTENCY_FACTOR` (10) × `tol`, the solver raises an error.

## Supported scope

  - **Polar only.** `ACPolarPowerFlow{GeneralizedAdmittanceACPowerFlow}`. The rectangular and
    mixed formulations reject this solver.
  - **Rejected settings.** `check_reactive_power_limits`, distributed slack (participation
    factors or headroom-proportional slack), discrete control, and area interchange. The first
    two are rejected by `_validate_solver_specific_settings`; the last two by the shared
    discrete-control and area-interchange validators, which do not list this solver.
  - **Required buses.** At least one REF bus and at least one PQ bus.
  - **LCC.** Each terminal is a constant ``P + jQ`` withdrawal. A closed form gives the values
    from the DC current, the setpoint and the minimum firing and extinction angles. The code
    folds them into the constant power term of the bus.
  - **VSC.** Converters are constant injections. When the system has a DC network, a DC substep
    in each iteration settles the converter active power and the DC voltage. AC-voltage VSC
    buses join the ``v`` set inside the partition only; `data.bus_type` is not changed.
  - **Handoff.** Opt-in through `handoff_solver` (`NewtonRaphsonACPowerFlow`,
    `TrustRegionACPowerFlow` or `LevenbergMarquardtACPowerFlow`). The default is the
    `NoHandoff` sentinel type.
