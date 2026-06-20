# Fast/Fixed Decoupled Power Flow

[`FastDecoupledACPowerFlow`](@ref) is a *solver* (not a formulation): it
replaces the per-iteration Jacobian refactorization of the Newton family with
constant matrices that are factored **once** and reused across every iteration
*and* every time step. It implements the fast decoupled power flow of Stott &
Alsac (1974) and van Amerongen (1989) — the algorithm PSS/E exposes as its
`FDNS` ("fixed-slope decoupled Newton-Raphson") activity. It works with all
three AC formulations ([`ACPolarPowerFlow`](@ref),
[`ACRectangularPowerFlow`](@ref), [`ACMixedPowerFlow`](@ref)).

The variant and the ``B'``/``B''`` scheme are **type parameters** of the solver —
`FastDecoupledACPowerFlow{V<:FDVariant, S<:FDScheme}` — so they are selected by
multiple dispatch rather than runtime flags. The two variants are:

  - **[`FDDecoupled`](@ref)** (polar only, the default for
    [`ACPolarPowerFlow`](@ref)): the classic ``B'``/``B''`` half-iteration scheme.
  - **[`FDFixedJacobian`](@ref)** (default for the rectangular and mixed
    formulations, also available on polar): a frozen full Jacobian — the
    "constant-matrix" or "dishonest" Newton method.

The bare `FastDecoupledACPowerFlow` (no type parameters) picks these
per-formulation defaults; write `FastDecoupledACPowerFlow{FDDecoupled, FDSchemeBX}`
to choose explicitly.

## The exactness-of-fixed-point argument

The single most important property of this solver — the one that de-risks the
whole design — is that the fast decoupled iteration evaluates the **exact**
residual ``R_v`` every iteration. The constant ``B'``/``B''`` matrices (or the
frozen Jacobian) only *precondition the update*; they never enter the
convergence test. Convergence is declared on exactly the same criterion as
every other solver in the package,

```math
\lVert R_v \rVert_\infty < \texttt{tol}, \qquad \texttt{tol} = 10^{-9}\ \text{(default)},
```

evaluated on the raw mismatch vector. An approximate Jacobian therefore changes
the *rate* of convergence (linear instead of Newton's quadratic), but the
converged *solution* is identical to the one Newton-Raphson finds, to the same
tolerance. A worse approximation simply costs more iterations; it does not move
the fixed point. This is why a deliberately crude constant matrix is safe: the
residual is the arbiter, the matrix is only a step direction.

The practical consequences:

  - Results match the Newton solvers to tolerance at the same `tol` (validated
    to `TIGHT_TOLERANCE`).
  - All downstream quantities derived from the converged state (bus voltages,
    line flows, reactive outputs, Q-limit switching, loss and
    voltage-stability factors) are computed from the *exact* state, not from
    the approximation.
  - The only thing the approximation can do is fail to converge (or converge
    slowly) on a stressed system — which the safeguards and the optional
    handoff are designed to handle.

## The ``B'``/``B''`` derivation (polar `:decoupled`)

The polar power-balance Jacobian has the block structure

```math
\begin{bmatrix} \Delta P \\ \Delta Q \end{bmatrix}
=
\begin{bmatrix} H & N \\ M & L \end{bmatrix}
\begin{bmatrix} \Delta \theta \\ \Delta V / V \end{bmatrix} .
```

Two empirical facts about transmission networks justify decoupling:

 1. Active power is strongly coupled to angle and weakly to magnitude
    (``\lVert N \rVert \ll \lVert H \rVert``); reactive power is strongly
    coupled to magnitude and weakly to angle (``\lVert M \rVert \ll \lVert L \rVert``). Dropping ``N`` and ``M`` splits the solve into two independent
    half-systems.
 2. Under the operating assumptions of fast decoupled load flow
    (``\cos\theta_{ik} \approx 1``, ``G_{ik}\sin\theta_{ik} \ll B_{ik}``,
    ``Q_i \ll B_{ii} V_i^2``), the remaining blocks ``H`` and ``L`` reduce to
    *constant*, voltage-independent susceptance matrices, after normalizing the
    mismatches by ``V``:

```math
\frac{\Delta P}{V} = B' \, \Delta\theta, \qquad
\frac{\Delta Q}{V} = B'' \, \Delta V .
```

``B'`` is restricted to the non-reference buses (PV ∪ PQ angle rows/columns);
``B''`` is restricted to the PQ buses (the only buses whose magnitude is
unknown). Each iteration is a **pair of half-steps** — first the ``P``–``\theta``
half-step with magnitudes held fixed, then the ``Q``–``V`` half-step with the
just-updated angles held fixed — and the exact mismatch is re-evaluated after
*each* half-step. This strict ``P\to Q`` alternation, with the mid-cycle
refresh, is what van Amerongen (1989) showed prevents the convergence *cycling*
seen on some systems; it is deliberately not "optimized" away.

In this package, ``B'`` and ``B''`` are defined as constant approximations of
the codebase's **own** Jacobian sub-blocks, and the update mirrors the Newton
solve-then-negate convention. The sign and value contract is settled by a unit
test, not by textbook convention: on a lossless, shunt-free, nominal-tap
network at flat start the decoupling is *exact*, so ``B'`` and ``B''`` must equal
the corresponding exact Jacobian blocks to machine precision (and there XB = BX).

### What each scheme neglects: XB vs BX

``B'`` and ``B''`` are each built by stamping a *temporary* network from the
recovered per-branch parameters and taking ``-\mathrm{Im}(Y_\text{bus})`` of the
restricted submatrix. The two matrices neglect different physical effects, and
the two schemes differ only in *where the series resistance is neglected*:

| Stamped into               | ``B'`` (``P``–``\theta``)                  | ``B''`` (``Q``–``V``)    |
|:-------------------------- |:------------------------------------------ |:------------------------ |
| Line charging ``b_c``      | **neglected** (set to 0)                   | retained                 |
| Bus shunts                 | **neglected** (set to 0)                   | retained                 |
| Off-nominal tap magnitude  | **neglected** (``\lvert\tau\rvert = 1``)   | retained                 |
| Phase shift ``\angle\tau`` | retained (makes ``B'`` mildly unsymmetric) | **neglected** (set to 0) |
| Series resistance ``r``    | **XB:** neglected here                     | **BX:** neglected here   |

  - **[`FDSchemeXB`](@ref)** (Stott–Alsac, the PSS/E-like default): resistance is
    neglected in ``B'`` (each series admittance is replaced by ``1/(jx)``), kept
    in ``B''``.
  - **[`FDSchemeBX`](@ref)** (van Amerongen): resistance is kept in ``B'``,
    neglected in ``B''``.

On normal-``r/x`` systems the two schemes take essentially the same iteration
count; `:BX` tends to be more robust when problematic ``r/x`` ratios are present.
``B'`` is symmetric except when phase shifters are present; ``B''`` is always
symmetric. Both are block-diagonal (and nonsingular per island) across a
multi-island network, so a single factorization covers every island.

These zeroing rules are rule-for-rule identical to MATPOWER's `makeB`/`fdpf`,
and the mismatch normalization (``\Delta P/V``, ``\Delta Q/V``; ``P`` rows at
PV ∪ PQ, ``Q`` rows at PQ; factor-once LU; negated updates) matches `fdpf.m`.
The one deliberate divergence from MATPOWER is the convergence test: MATPOWER
converges on the ``V``-normalized mismatches, while this package keeps the
package-wide raw ``\lVert R_v \rVert_\infty`` criterion — which is the **PSS/E
convention** (the POM's `TOLN` tests the largest raw MW/Mvar mismatch) and
identical to MATPOWER's at ``\lvert V\rvert \approx 1`` pu.

## The `:fixed_jacobian` variant

The frozen-Jacobian variant generalizes the fixed-slope idea to all three
formulations. The full formulation Jacobian is evaluated once at the initial
state, factored once, and that factorization is reused for every subsequent
update:

```math
J(x_0)\, \Delta x = R_v(x_k), \qquad x_{k+1} = x_k - \Delta x ,
```

with ``R_v(x_k)`` the **exact** residual at the current iterate but ``J`` frozen
at ``x_0``. This is the literature-established constant-matrix Newton method
(Moura & Moura, 2013), which is competitive with — and on high-``r/x`` systems
often better-iterating than — the XB/BX decoupled schemes.

It is the natural default for the rectangular and mixed formulations because
their off-diagonal Jacobian blocks are *already* the constant admittance-matrix
entries (see [Mixed Current-Power Balance Formulation](@ref)); freezing the
Jacobian then only affects the diagonal blocks. Because the LCC HVDC state
variables and the distributed-slack column live inside the frozen Jacobian,
**`:fixed_jacobian` supports LCC systems and distributed slack with no special
handling** — unlike the polar `:decoupled` half-iterations, which cannot span
the trailing LCC state variables and reject LCC systems at the driver entry. If
the frozen-Jacobian iteration stalls, the solver may (by default) re-evaluate
and re-factor the Jacobian once at the current iterate and continue — a single
"refreeze" before giving up.

## Caching across iterations and time steps

This is the central performance contract, and it mirrors PSS/E's FDNS exactly.
The ``B'`` matrix depends only on the network, never on the bus voltages, so it
is factored **once per (data, scheme, backend) lifetime** and reused across all
iterations, all Q-limit retries, and all time steps. The ``B''`` matrix depends
only on the PQ set, so it is refactored *only* when a bus switches between
voltage-regulating (PV) and reactive-power-limited (PQ) status — a small,
millisecond-scale refactorization even at ten thousand buses. This factor-once
cache applies to the polar `:decoupled` ``B'``/``B''`` only. The
`:fixed_jacobian` variant cannot reuse a factorization across time steps: its
frozen Jacobian is evaluated at the per-step starting point, which changes each
step, so it is refactored once per solve.

The effect on repeated-solve workloads (multi-period dispatch, contingency
screening) is dramatic: across time steps with identical bus-type columns, and
across Q-limit retries that return to a previously-seen PQ set, there are **zero
refactorizations**. The start-up cost (building and factoring the fixed
matrices) is higher than a single Newton iteration, but it is amortized over the
whole study.

## Handoff to an exact-Newton solver (opt-in)

Pure fast decoupled is the default and converges to the full `tol` on its own.
But because the FD-stage state is a valid iterate of the *exact* residual, it is
also an excellent warm start for an exact-Newton solver. The solver can
optionally hand off: run the cheap FD stage to a looser `handoff_tol`, then let
[`NewtonRaphsonACPowerFlow`](@ref), [`TrustRegionACPowerFlow`](@ref), or
[`LevenbergMarquardtACPowerFlow`](@ref) polish the solution to `tol` using the
*same* residual and state-vector objects (a
verified drop-in — no data is copied or re-initialized).

```julia
using PowerFlows

# Pure fast decoupled (polar, XB) — converges to tol = 1e-9 on its own:
pf = ACPowerFlow{FastDecoupledACPowerFlow}()
results = solve_power_flow(pf, system)

# FD stage to engineering tolerance, then Newton-Raphson to 1e-9:
pf = ACPowerFlow{FastDecoupledACPowerFlow}(;
    solver_settings = Dict(
        :handoff_solver => NewtonRaphsonACPowerFlow,
        :handoff_tol => 1e-3,
    ),
)
results = solve_power_flow(pf, system)

# BX scheme (a solver type parameter), hand off to Trust Region:
pf = ACPowerFlow{FastDecoupledACPowerFlow{FDDecoupled, FDSchemeBX}}(;
    solver_settings = Dict(
        :handoff_solver => TrustRegionACPowerFlow,
    ),
)
results = solve_power_flow(pf, system)

# Fixed-slope (frozen-Jacobian) on the mixed formulation — supports LCC, distributed slack:
pf = ACMixedPowerFlow{FastDecoupledACPowerFlow}()    # rect/mixed default to FDFixedJacobian
results = solve_power_flow(pf, system)
```

The `handoff_tol` default (`1e-3` pu) corresponds to PSS/E's default
convergence tolerance (`TOLN` = 0.1 MW/MVAr on a 100 MVA base): the FD stage
exits at commercial-tool tolerance and the refinement stage continues to the
package's `1e-9`. If the FD stage already reaches `tol`, the handoff is skipped.
This staging — FD as a conditioner, Newton as the closer — is standard
commercial practice: PowerWorld's *Robust Solution Process* (controls off →
fast decoupled → Newton-Raphson) and PSS/E's FDNS-then-FNSL activity sequencing
are the same pattern.

A cross-formulation composition (e.g. a polar FD solve warm-starting a
rectangular Newton solve) already works without any dedicated API: the residual
functor syncs ``V``/``\theta`` into the shared data container, and every
formulation's warm-start reads from that container. Run the two solves in
sequence.

### Behavior matrix

| Case                             | polar `:decoupled`                                                                                                                                          | `:fixed_jacobian` (any formulation)                                                   |
|:-------------------------------- |:----------------------------------------------------------------------------------------------------------------------------------------------------------- |:------------------------------------------------------------------------------------- |
| LCC HVDC present                 | `ArgumentError` at driver entry (half-iterations cannot span the LCC state variables) — message points to `:fixed_jacobian`, rectangular/mixed FD, or NR/TR | Supported (LCC rows live in the frozen Jacobian)                                      |
| Distributed slack                | Supported via a per-iteration rank-1 slack redistribution                                                                                                   | Supported (slack column is in the frozen Jacobian)                                    |
| Q-limit (PV→PQ) switching        | Supported; ``B''`` refactored via the cache, keyed on the PQ set; bus types switch only between driver invocations (the package-wide convention)            | Supported; the outer loop re-invokes the driver and re-freezes ``J`` at the new start |
| Loss / voltage-stability factors | Supported; the driver refreshes ``J`` at the converged state before computing them                                                                          | Same                                                                                  |
| Multiple islands                 | Supported (block-diagonal ``B'``/``B''``)                                                                                                                   | Supported                                                                             |

## Safeguards on stressed systems

Fast decoupled divergence on high-``r/x`` or heavily-stressed cases is *expected
algorithm behavior*, not a bug — the PSS/E POM documents it for the NSOL/FDNS
family ("the iteration usually reaches some mismatch level and then begins to
diverge, usually slowly"). The solver ships the PSS/E-parity safeguards rather
than a promise of unconditional convergence:

  - **Non-divergent backtracking** (default on): each cycle, if the
    sum-of-squared mismatches fails to improve, the step is halved and re-tried
    (up to ten halvings); on exhaustion the iteration terminates while
    **restoring the best-mismatch state seen** into the data — a good
    starting point for a handoff, and a useful localization of trouble spots.
  - **BLOWUP** abort on an oversized per-half-step ``\lvert\Delta\theta\rvert``
    or ``\lvert\Delta V/V\rvert`` (the only divergence guard when non-divergent
    control is disabled).
  - **DVLIM** clamping that uniformly scales the ``\Delta V`` vector so the
    largest applied magnitude change stays bounded, plus a ``\Delta V/V \le -1``
    positivity guard.
  - A **``\lvert V\rvert \approx 0`` abort** if any bus magnitude is driven to
    nearly zero.

When a handoff solver is configured, a stalled FD stage still hands its
best-recorded state to the Newton solver, which frequently converges from
there. In pure-FD mode a stall returns a non-converged result (no exception)
with an actionable error suggesting `:BX`, `:fixed_jacobian`, a handoff, or a
Newton solver.

## When to use which solver

| Solver                                  | Convergence                       | Per-iteration cost                                 | Use when                                                                                                                  |
|:--------------------------------------- |:--------------------------------- |:-------------------------------------------------- |:------------------------------------------------------------------------------------------------------------------------- |
| [`FastDecoupledACPowerFlow`](@ref)      | Linear (more, cheaper iterations) | Lowest — factor once, reuse                        | Repeated solves (multi-period, contingency screening); a cheap warm start for Newton; commercial-tool parity (PSS/E FDNS) |
| [`NewtonRaphsonACPowerFlow`](@ref)      | Quadratic                         | Refactor every iteration                           | The well-conditioned default for a single solve                                                                           |
| [`TrustRegionACPowerFlow`](@ref)        | Quadratic, globalized             | Refactor every iteration                           | Poor start, ill-conditioned, or high-impedance — more robust than NR at ≈ the same median cost                            |
| [`LevenbergMarquardtACPowerFlow`](@ref) | Robust least-squares              | Refactors an augmented system every step (slowest) | A fallback when NR/TR stall                                                                                               |
| [`RobustHomotopyPowerFlow`](@ref)       | Continuation                      | Highest (Hessian solves)                           | The most robust last resort for hard or non-convergent cases (polar only)                                                 |

Rules of thumb:

  - **Repeated solves at scale** are the prime use case for fast decoupled: the
    ``B'`` factorization is shared across every time step, so the higher
    start-up cost is amortized and the total wall time is competitive with — or
    better than — re-running Newton each step.
  - For a **single, well-conditioned solve**, plain Newton-Raphson is usually as
    fast or faster (its quadratic rate needs only a handful of refactorizations).
  - For a **stressed or high-``r/x``** system, prefer `:BX`, `:fixed_jacobian`,
    a handoff to Trust Region, or one of the robust solvers directly.
  - **LCC HVDC or distributed slack under fast decoupled:** use
    `:fixed_jacobian` (the polar `:decoupled` variant rejects LCC systems).

See also [How to choose an AC formulation and solver](@ref
choose-ac-formulation-and-solver) and
[Evaluation Models vs. Solver Algorithms](@ref).

## Scope

Commercial FDNS bundles automatic tap, switched-shunt, and area-interchange
adjustment loops, and remote/droop voltage control. These are out of scope for
this solver — as they are for every other solver in the package — and are listed
here as future work. Optimized FDNS for contingency analysis (reusing the
base-case ``B'`` across contingencies with compensation correction vectors) is
likewise future work that this implementation enables.

## References

  - B. Stott, O. Alsac, "Fast decoupled load flow", *IEEE Trans. Power
    Apparatus and Systems* PAS-93(3), 1974 — the XB scheme.
  - R. A. M. van Amerongen, "A general-purpose version of the fast decoupled
    load flow", *IEEE Trans. Power Systems* 4(2):760–770, 1989 — the BX scheme
    and the strict ``P\to Q`` alternation argument.
  - Siemens PTI, *PSS®E Program Operation Manual*, §6.5–6.7 (activities `FNSL`,
    `NSOL`, `FDNS`) — the fixed-slope decoupled Newton-Raphson algorithm, its
    half-iteration accounting, the fixed-``B'``/switching-``B''`` cache
    contract, and the non-divergent/BLOWUP/DVLIM safeguards.
  - MATPOWER, `makeB.m` and `fdpf.m` — the reference XB/BX stamping rules and
    factor-once iteration this implementation mirrors.
  - A. M. Moura, A. P. Moura, "Newton–Raphson power flow with constant
    matrices: a comparison with decoupled power flow methods", *Int. J.
    Electrical Power & Energy Systems* 46:108–115, 2013 — the constant-matrix
    (`:fixed_jacobian`) method.
  - C. C. de Oliveira, A. Bonini Neto, D. A. Alves, C. R. Minussi, C. A. Castro,
    "Alternative Current Injection Newton and Fast Decoupled Power Flow",
    *Energies* 16(6):2548, 2023 — the experimental decoupled current-injection
    variant (gated, not the default).
