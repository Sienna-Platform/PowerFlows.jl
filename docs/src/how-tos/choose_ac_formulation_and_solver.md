# [How to choose an AC formulation and solver](@id choose-ac-formulation-and-solver)

```@meta
CurrentModule = PowerFlows
```

AC power flow has two independent choices: the **formulation** (how the
network equations are written) and the **solver** (the iterative algorithm).
Each is a type parameter: [`ACPolarPowerFlow`](@ref)`{S}` (power balance, polar state),
[`ACRectangularPowerFlow`](@ref)`{S}` (Da Costa current injection), and
[`ACMixedPowerFlow`](@ref)`{S}` (mixed current/power balance, the most compact
state) — each combined with [`NewtonRaphsonACPowerFlow`](@ref) (`NR`),
[`TrustRegionACPowerFlow`](@ref) (`TR`), or
[`LevenbergMarquardtACPowerFlow`](@ref) (`LM`). [`ACPowerFlow`](@ref)`()` is
`ACPolarPowerFlow{NewtonRaphsonACPowerFlow}` — a good default.

For the conceptual split between evaluation models and solvers, see
[Evaluation Models vs. Solver Algorithms](@ref).

Warm-solve timings: median of 10 runs after warm-up, with the `[min, max]`
range. Hardware-dependent — compare medians across cells, not absolutes.

## 2000-bus (`ACTIVSg2000`, tol `1e-9`)

| Formulation \ Solver | NR                             | TR                                 | LM                             |
|:-------------------- |:------------------------------ |:---------------------------------- |:------------------------------ |
| Polar                | 4 it, 0.032 s `[0.031, 0.045]` | 3 it, 0.032 s `[0.031, 0.250]`     | 4 it, 0.104 s `[0.095, 0.348]` |
| Rectangular          | 4 it, 0.028 s `[0.028, 0.044]` | 3 it, **0.029 s** `[0.028, 0.042]` | 7 it, 0.150 s `[0.136, 0.398]` |
| Mixed                | 5 it, 0.030 s `[0.029, 0.039]` | 4 it, 0.029 s `[0.029, 0.041]`     | 3 it, 0.083 s `[0.075, 0.341]` |

## 10000-bus (`ACTIVSg10k`, tol `1e-9`)

| Formulation \ Solver | NR                                 | TR                             | LM                             |
|:-------------------- |:---------------------------------- |:------------------------------ |:------------------------------ |
| Polar                | 5 it, 0.214 s `[0.208, 0.447]`     | 4 it, 0.223 s `[0.210, 0.433]` | 5 it, 0.614 s `[0.492, 0.792]` |
| Rectangular          | 4 it, **0.183 s** `[0.178, 0.402]` | 3 it, 0.189 s `[0.180, 0.399]` | 56 it, 4.315 s `[4.15, 4.42]`  |
| Mixed                | 5 it, 0.186 s `[0.182, 0.209]`     | 4 it, 0.194 s `[0.182, 0.414]` | 5 it, 0.590 s `[0.451, 0.755]` |

All nine combinations converge to the same solution. On small systems the
differences are negligible (every combination solves a 14-bus case in
2–3 iterations in well under 1 ms median); the choice matters at scale.
The wide LM ranges reflect its per-iteration refactorization variance — the
median is the representative figure.

## Recommendations (based on the median)

  - **Default / general use:** [`ACPowerFlow`](@ref)`()` (Polar + NR). Well-trodden and
    the reference all other formulations are validated against.
  - **Fastest at scale:** [`ACRectangularPowerFlow`](@ref)`{NewtonRaphsonACPowerFlow}`
    or `{TrustRegionACPowerFlow}`. The rectangular formulation's off-diagonal
    Jacobian blocks are the constant admittance matrix, so the sparse
    factorization is reused across iterations — consistently ~15% faster than
    Polar/NR by median (2000 and 10k bus alike); [`ACMixedPowerFlow`](@ref) with NR/TR
    is within a few percent of it.
  - **Most robust (poor start, ill-conditioned, high-impedance):**
    [`TrustRegionACPowerFlow`](@ref) on any formulation — its median is on par
    with NR at every size while globalizing the step. For cases that still will
    not converge, [`RobustHomotopyPowerFlow`](@ref) (Polar only).
  - **Smallest state / most predictable:** [`ACMixedPowerFlow`](@ref) — `2n`
    unknowns, the most compact of the three, and the tightest timing spread at
    every scale (Mixed/NR 10k `[0.182, 0.209]`).
  - **Levenberg-Marquardt:** a robust least-squares fallback, the slowest per
    iteration (it refactorizes a sparse augmented system every step) and the
    highest variance — prefer NR/TR and reach for LM only when they stall. If
    you need LM, use **Mixed/LM** (the only LM that scales: 3–5 iterations and
    the lowest LM median at every size) or Polar/LM. **Avoid Rectangular/LM at
    scale**: its Marquardt scaling helps at ≤2000 buses (7 iters) but does not
    scale — 56 iterations / 4.3 s at 10k, ~20× Rectangular/NR. See
    [Levenberg-Marquardt in Place of Gauss-Seidel](@ref).

See also the explanation pages
[Mixed Current-Power Balance Formulation](@ref) and
[Levenberg-Marquardt in Place of Gauss-Seidel](@ref) for the underlying
trade-offs.
