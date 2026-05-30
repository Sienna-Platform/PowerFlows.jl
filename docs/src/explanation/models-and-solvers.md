# Evaluation Models vs. Solver Algorithms

PowerFlows.jl draws a sharp line between two concepts that are often blurred in
power-flow libraries:

 1. **What problem you are solving** — the [`PowerFlowEvaluationModel`](@ref).
 2. **How you solve it** — for AC problems, the [`ACPowerFlowSolverType`](@ref).

Keeping these orthogonal lets you swap the integration algorithm without
changing the model, and vice versa, and it is the reason every public
constructor looks the same: `solve_power_flow(model, sys)`.

## 1. The evaluation model: *what* to compute

A [`PowerFlowEvaluationModel`](@ref) is a value-type description of the power-flow
problem and of any post-processing or export the user wants alongside it.
Concrete evaluation models fall into three families:

| Family | Concrete types                                                                                                         | Purpose                                                                                                                                                                                                   |
|:------ |:---------------------------------------------------------------------------------------------------------------------- |:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| DC     | [`DCPowerFlow`](@ref), [`PTDFDCPowerFlow`](@ref), [`vPTDFDCPowerFlow`](@ref)                                           | Linear approximation: solves for voltage angles (or computes flows directly via PTDF). Fast, supports multi-period.                                                                                       |
| AC     | [`ACPolarPowerFlow`](@ref) (alias [`ACPowerFlow`](@ref)), [`ACRectangularPowerFlow`](@ref), [`ACMixedPowerFlow`](@ref) | Full non-linear AC power flow. Three **formulations** — polar power balance `(V, θ)`, Da Costa current injection `(e, f)`, and mixed current–power balance (MCPB, also rectangular state, `2n` unknowns). |
| Export | [`PSSEExportPowerFlow`](@ref)                                                                                          | Writes a PSS/e raw file as a "solve" step.                                                                                                                                                                |

Every AC model carries the bookkeeping for the AC problem: bus-type handling,
slack distribution, network reductions, time steps, the optional reactive-power
limit check, etc. None of that depends on the algorithm used to drive the
residual to zero.

### The three AC formulations

All three AC formulations are mathematically equivalent — they solve the same
power flow and should agree to numerical tolerance — but they differ in their
state vector, residual, and Jacobian structure:

  - [`ACPolarPowerFlow`](@ref) (the default and recommended starting point):
    per-bus state is `(Vᵢ, θᵢ)`; the residual is the active/reactive power
    mismatch. This is the formulation used by the classic Newton power flow and
    is the only one that currently supports the loss-factor, voltage-stability,
    and DC-fallback post-processing options.
  - [`ACRectangularPowerFlow`](@ref): per-bus state is `(eᵢ, fᵢ)` (with an extra
    Qᵢ at PV buses); the residual is the complex *current* mismatch
    `ΔIᵢ = I_specᵢ − Y_bus·V`. Off-diagonal Jacobian blocks are constant 2×2
    real blocks of `Y_bus`, which makes refactorization cheap. Pick this
    formulation if you want the current-injection structure or are integrating
    with code that expects rectangular coordinates.
  - [`ACMixedPowerFlow`](@ref): per-bus state is `(eᵢ, fᵢ)` with exactly `2n`
    unknowns — no auxiliary reactive-power variable at PV buses. The residual
    mixes divided current balance at PQ buses with real power balance plus a
    `|Vᵢ|²` constraint at PV buses. It does not outperform
    [`ACRectangularPowerFlow`](@ref) on speed, but its minimal state vector and
    predictable timing make it attractive for memory-bound large-scale studies;
    it is also the best-scaling LM formulation at 10k buses and above. Like
    rectangular, it lacks the polar-only post-processing options. See
    [Mixed Current-Power Balance Formulation](@ref) for the full derivation.

All three formulations support the LCC (line-commutated converter) HVDC model.
The LCC state and Jacobian rows are appended to the network state and share the
same true-φ derivation across polar, rectangular, and mixed code paths; see
[Line-Commutated Converter (LCC) Implementations](@ref).

## 2. The AC solver: *how* to drive the residual to zero

For an AC problem, the evaluation model is parameterized by an
[`ACPowerFlowSolverType`](@ref) — a tag that selects one of the iterative
algorithms PowerFlows.jl ships with:

| Solver                                       | When to use it                                 | Notes                                                                                                                                                                 |
|:-------------------------------------------- |:---------------------------------------------- |:--------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [`NewtonRaphsonACPowerFlow`](@ref) (default) | Well-conditioned systems, warm starts          | Pure Newton step. Optional Iwamoto damping via `solver_settings = Dict(:iwamoto => true)`.                                                                            |
| [`TrustRegionACPowerFlow`](@ref)             | Slightly ill-conditioned or unreliable starts  | Powell dogleg. Comparable cost to Newton.                                                                                                                             |
| [`LevenbergMarquardtACPowerFlow`](@ref)      | Hard-to-converge cases where TR also struggles | More robust, more expensive per iteration; meta-parameters can be sensitive. At scale, prefer LM on [`ACMixedPowerFlow`](@ref) over [`ACRectangularPowerFlow`](@ref). |
| [`RobustHomotopyPowerFlow`](@ref)            | Pathological initial points                    | Order(s) of magnitude slower than Newton but very robust. *Polar only.*                                                                                               |

Construction follows a single pattern — the solver is the type parameter of
the AC model:

```julia
ACPowerFlow()                                       # polar + Newton-Raphson (default)
ACPowerFlow{TrustRegionACPowerFlow}()               # polar + Powell dogleg
ACRectangularPowerFlow{LevenbergMarquardtACPowerFlow}()  # rectangular + LM
ACMixedPowerFlow()                                  # MCPB + Newton-Raphson (default)
ACMixedPowerFlow{LevenbergMarquardtACPowerFlow}()   # MCPB + LM (scales better than Rect/LM)
```

Any (formulation, solver) pair is valid except where noted:
[`RobustHomotopyPowerFlow`](@ref) and [`GradientDescentACPowerFlow`](@ref) are
polar-only because their continuation paths assume the polar state layout;
[`ACMixedPowerFlow`](@ref) rejects both at construction.

## 3. The user-facing call

Once you have an evaluation model, solving and storing results always look
the same:

```julia
results = solve_power_flow(model, sys)            # returns Dicts of DataFrames
solve_and_store_power_flow!(model, sys)           # writes results back into `sys`
```

The dispatcher inspects `model`, picks the right setup (`PowerFlowData`,
state vector, residual, Jacobian) and integration loop, and returns a uniform
result schema. From the caller's perspective the choice of formulation or
solver is just data.

## When you should care which is which

  - **Choosing a model**: driven by the *physics* you want (DC approximation
    vs. full AC, polar vs. rectangular vs. mixed state) and by which
    post-processing options you need. Loss factors, voltage-stability factors,
    and the DC robust fallback are [`ACPolarPowerFlow`](@ref)-only; the
    rectangular and mixed formulations share the same rectangular-state
    limitations. PSS/e export is independent of formulation.
  - **Choosing a solver**: driven by *numerics* — how well-conditioned the
    problem is and how close your initial guess is to the solution. If a
    default Newton-Raphson run does not converge, escalate to
    [`TrustRegionACPowerFlow`](@ref), then [`LevenbergMarquardtACPowerFlow`](@ref),
    then [`RobustHomotopyPowerFlow`](@ref) (polar only), without touching the
    rest of your setup. For LM on large systems, prefer
    [`ACMixedPowerFlow`](@ref) over [`ACRectangularPowerFlow`](@ref).

For benchmark-based selection at scale, see
[How to choose an AC formulation and solver](@ref choose-ac-formulation-and-solver).
For the MCPB theory and equations, see
[Mixed Current-Power Balance Formulation](@ref).
