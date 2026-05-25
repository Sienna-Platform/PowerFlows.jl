# Levenberg-Marquardt in Place of Gauss-Seidel

PowerFlows.jl does **not** ship a Gauss-Seidel (GS) solver. In classical power
systems practice GS was the robust, low-memory workhorse used to obtain a power
flow solution — especially as a starting point for hard cases or when a good
initial guess was unavailable — at the cost of slow (linear) convergence.
[`LevenbergMarquardtACPowerFlow`](@ref) (LM) fills the same *role* GS used to
play — a robust solver with a large convergence basin that tolerates poor
starting points — but with a strictly stronger convergence mechanism. This page
explains the substitution and how to configure LM so its behaviour resembles
GS as closely as the algorithm allows, for users who specifically want
Gauss-Seidel-like characteristics.

## Why LM supersedes Gauss-Seidel

The two methods belong to different algorithm families:

| Aspect            | Gauss-Seidel                                                  | Levenberg-Marquardt                                                     |
|:----------------- |:------------------------------------------------------------- |:----------------------------------------------------------------------- |
| Principle         | matrix-splitting fixed-point iteration on the nodal equations | damped Gauss-Newton minimization of $\tfrac{1}{2}\lVert F(x)\rVert_2^2$ |
| State update      | one bus at a time, reusing the newest values within a sweep   | all states simultaneously, via a factorized linear solve                |
| Derivative use    | none (no Jacobian)                                            | Jacobian $J$, refactored each iteration                                 |
| Convergence order | linear (first-order)                                          | between linear (heavily damped) and quadratic (Newton regime)           |
| Robustness        | large convergence basin, hard to diverge                      | large basin when damped; Newton speed near the solution                 |
| Tuning knob       | acceleration factor                                           | damping $\lambda$ (via `λ_0`), `marquardt_scaling`                      |

LM minimizes $\tfrac{1}{2}\lVert F(x)\rVert_2^2$ with the damped normal-equation
step

```math
(J^\top J + \lambda D^2)\,\Delta x = -J^\top F(x),
```

where $D$ is the optional Marquardt column scaling (identity by default on the
polar formulation). The damping $\lambda$ interpolates between two regimes:

  - $\lambda \to 0$: the step approaches the **Gauss-Newton / Newton** step —
    fast (quadratic) but fragile far from the solution.
  - $\lambda \to \infty$: the step approaches
    $\Delta x \approx -\tfrac{1}{\lambda} D^{-2} J^\top F(x)$ — a short, scaled
    **steepest-descent** step: cautious, first-order, with a very large
    convergence basin.

The heavily-damped regime is first-order, robust, and slow — the same
qualitative behaviour that makes Gauss-Seidel dependable on difficult or
flat-start cases. The decisive difference is that LM is **adaptive**: it stays
in the cautious regime only while needed and automatically transitions toward
Newton speed as it approaches the solution, so it does not pay GS's slow linear
tail. In this sense LM is a strict practical improvement over GS, not merely an
alternative.

## What cannot be matched

LM is a least-squares, Jacobian-based, simultaneous-update method; Gauss-Seidel
is a Jacobian-free fixed-point iteration with sequential per-bus updates. They
do not produce the same iterates, and **no configuration makes LM algebraically
equivalent to Gauss-Seidel**. What *can* be reproduced is GS's practical
character: a very large convergence basin, gradual descent, insensitivity to a
poor flat start, and slow first-order progress. The settings below push LM into
that regime.

## Configuring LM for Gauss-Seidel-like behaviour

All settings are passed through `solver_settings`.

  - **Large `λ_0`** (default `1e-5`; try `1e-1` to `1e1`). `λ_0` is the initial
    damping factor $\mu$; the working damping is $\lambda = \mu\lVert F\rVert$.
    A large value keeps LM in the heavily-damped, steepest-descent-like regime —
    small, cautious steps and a broad basin, with no aggressive Newton jumps —
    which is the closest analogue to a Gauss-Seidel sweep.
  - **`marquardt_scaling => true`.** Gauss-Seidel implicitly normalizes each bus
    update by that bus's self-admittance $Y_{ii}$ (it solves the $i$-th nodal
    equation for $V_i$, dividing by $Y_{ii}$). The Marquardt diagonal $D$ (column
    2-norms of $J$, which scale with the bus admittance coupling) is LM's closest
    structural echo of that per-bus diagonal normalization, so enabling it makes
    the damped step's per-coordinate scaling resemble the GS diagonal scaling.
    This is already the default for [`ACRectangularPowerFlow`](@ref); set it
    explicitly when chasing GS-like behaviour on [`ACPolarPowerFlow`](@ref).
  - **Looser `tol`** (default `1e-9`; try `1e-4` to `1e-6`). Gauss-Seidel is
    conventionally run to an engineering mismatch tolerance, not `1e-9`. A
    heavily-damped first-order iteration converges linearly and crawls at the
    tail, so a GS-typical tolerance keeps the iteration count realistic.
  - **Large `maxIterations`** (default `50`; try `200` to `1000`). First-order
    convergence implies many iterations — exactly as with Gauss-Seidel.

Example — polar formulation, Gauss-Seidel-like configuration:

```julia
pf = ACPolarPowerFlow{LevenbergMarquardtACPowerFlow}(;
    solver_settings = Dict{Symbol, Any}(
        :λ_0 => 1.0,                 # heavy damping → cautious first-order steps
        :marquardt_scaling => true,  # GS-like per-bus diagonal scaling
        :tol => 1e-6,                # engineering mismatch tolerance
        :maxIterations => 500,       # linear convergence needs many iterations
    ),
)
solve_power_flow(pf, sys)
```

## A caveat on cost, and when to pick LM

Each LM iteration refactorizes a sparse augmented system, so a GS-like
configuration (many heavily-damped iterations) is comparatively expensive on
large networks — on a 10k-bus system LM can take dozens of iterations where
Newton-type methods take a handful.

If you just need a power flow to converge from a reasonable start,
[`NewtonRaphsonACPowerFlow`](@ref) or [`TrustRegionACPowerFlow`](@ref) are
faster and the right default. LM earns its per-iteration cost on the hard case
it is built for: **ill-conditioned systems, or poor / flat initial conditions
where Newton-type iterations (NR, and the Newton-based steps inside TR) stall**.
TR (globalized Newton with a trust region) and
[`RobustHomotopyPowerFlow`](@ref) (homotopy continuation; currently not
compatible with HVDC) are *different* robustification strategies.

Treat LM's damped least-squares step
as a genuinely distinct tool for those hard cases: when NR/TR stall on an
ill-conditioned start, LM (optionally in the GS-like configuration above) is the
method to try.

See also: [`LevenbergMarquardtACPowerFlow`](@ref),
[`TrustRegionACPowerFlow`](@ref), [`RobustHomotopyPowerFlow`](@ref).
