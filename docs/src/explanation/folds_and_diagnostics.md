# Folds, Voltage Collapse, and Solver Diagnostics

An AC power flow can fail to converge because of a poor starting point — a
*numerical* difficulty, addressed by robust solvers such as
[`TrustRegionACPowerFlow`](@ref), [`LevenbergMarquardtACPowerFlow`](@ref), or
[`RobustHomotopyPowerFlow`](@ref) — or because the requested operating point
**has no solution**, a *physical* limit no solver can overcome. The second case
is a **fold**. This page explains the term and the per-iteration diagnostics
that recognize it.

## What is a fold?

Write the power flow equations as $F(x; p) = 0$, where $x$ collects the bus
voltage states and $p$ is a scalar that *stresses* the system — typically total
load. At light loading there is a well-behaved high-voltage solution and a
second, lower-voltage one; as $p$ increases the two drift toward each other.
Plotting a bus voltage magnitude against $p$ traces the **PV (nose) curve**:

```text
 |V|
  │   ── high-voltage (operable) branch
  │ ╱                       ╲
  │╱                          ● ← fold / nose: the two branches meet
  │                         ╱     (maximum loadability p*)
  │   ── low-voltage branch
  └────────────────────────────── p (loading)
                            p*
```

At the **fold** (the nose, loading $p^*$) the two solutions coalesce and, for
$p > p^*$, vanish: no (real) power flow solution exists. Mathematically this is
a **saddle-node bifurcation**, whose defining feature is that the Jacobian
$J = \partial F / \partial x$ becomes **singular** — its smallest eigenvalue
passes through zero. The fold is the classical static **voltage-collapse** limit.

## Why a fold breaks the solver

Newton-type methods solve a linear system with $J$ at every iteration. Near the
fold $J$ becomes ill-conditioned and finally singular: the step
$\Delta x = -J^{-1} F$ blows up and the iteration diverges, or the solver
crosses onto the **low-voltage branch** and "converges" to a physically
meaningless solution. Both look like ordinary non-convergence, but more
iterations or a more robust solver cannot help — the operating point is
infeasible. Recognizing the fold lets you report *"past the loadability
limit"* instead of returning a bad root or burning the iteration budget.

## How PowerFlows.jl detects a fold

The smallest-magnitude eigenvalue of the Jacobian is the standard proximity
indicator: it tends to zero at the fold, and its real part changes sign when an
iterate crosses onto the other branch.

PowerFlows.jl computes this eigenvalue on the bus-voltage **Schur complement**
$S$ of the Jacobian — the bus block left after projecting out the LCC/HVDC
converter states (with no converters, $S = J$). $S$ is never formed or
factorized: inverse iteration (KrylovKit) finds the largest-magnitude
eigenvalue $\mu$ of $S^{-1}$ through a factorization of the *full* $J$ — each
Arnoldi matvec is one back-solve — and reports $\lambda_{\min}(S) = 1/\mu$. The
per-iteration cost is one numeric refactorization of $J$ plus the eigensolve,
shared by the log line and the bail-out. (LM keeps a small dedicated KLU factor
of $J$, since its solver factorizes the augmented least-squares system
instead.) Because $J$ is non-symmetric, $\lambda_{\min}$ may be complex; the
fold signal is the sign of its real part.

Two opt-in settings expose this. Both are off by default and add cost only when
enabled.

### `log_solver_diagnostics` — observe the approach

A constructor keyword on the formulation. Each iteration emits an `@info` line
with $\lVert F\rVert_\infty$ (and the bus/equation where it is attained), the
condition estimate $\hat\kappa(J)$, $\lambda_{\min}(S)$, and the residual
contraction ratio. Watching $\lambda_{\min}(S)$ shrink toward zero is a direct
view of an iterate nearing a fold.

```julia
pf = ACPolarPowerFlow{NewtonRaphsonACPowerFlow}(; log_solver_diagnostics = true)
solve_power_flow(pf, sys)
# [ Info: NR iter 3: ‖F‖_∞ = 0.04211 at bus 8 (Q), κ̂(J) = 1830.0,
#         λ_min(S) = 0.5217 (|λ_min| = 0.5217), contraction = 0.231
```

$\hat\kappa(J)$ is available only on the KLU backend; other backends print
`n/a`. Pin `:linear_solver => "KLU"` in `solver_settings` if you need it on
platforms whose default is AppleAccelerate.

### `stop_at_fold` — abort at the fold

Passed through `solver_settings`. The solver stops as soon as it sees a fold
signature — the real part of $\lambda_{\min}(S)$ flips sign between iterations,
the Jacobian is outright singular, or the eigenvalue is indeterminate
(non-converged or non-finite, treated conservatively as a fold) — and returns
*not converged* with a warning, rather than continuing toward divergence or the
low-voltage branch.

```julia
pf = ACPolarPowerFlow{NewtonRaphsonACPowerFlow}(;
    solver_settings = Dict{Symbol, Any}(:stop_at_fold => true))
data = PowerFlowData(pf, sys)
converged = solve_power_flow!(data)   # false at a fold, with a voltage-collapse warning
```

`stop_at_fold` is supported by [`NewtonRaphsonACPowerFlow`](@ref),
[`TrustRegionACPowerFlow`](@ref), and [`LevenbergMarquardtACPowerFlow`](@ref),
on every AC formulation (polar, rectangular-CI, and mixed-CPB, with or without
HVDC).

## Caveats

  - The sign-flip test is a **practical signature, not a rigorous bifurcation
    test**: it stops a solve wandering into a collapse region; it does not
    certify the exact loadability limit.
  - The per-iteration cost is justified on hard or near-collapse cases — not
    something to leave on for routine, well-conditioned solves.
  - A `stop_at_fold` abort returns `false` like any other non-convergence; the
    distinguishing detail is the fold/voltage-collapse warning in the log.

## See also

[`NewtonRaphsonACPowerFlow`](@ref), [`TrustRegionACPowerFlow`](@ref),
[`LevenbergMarquardtACPowerFlow`](@ref), [`RobustHomotopyPowerFlow`](@ref),
[Evaluation Models vs. Solver Algorithms](@ref),
[How to choose an AC formulation and solver](@ref choose-ac-formulation-and-solver).
