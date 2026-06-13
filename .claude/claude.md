# CLAUDE.md — PowerFlows.jl

> **Development Guidelines:** Always load [Sienna.md](./Sienna.md) development preferences, style conventions, and best practices for projects using Sienna. Before running tests confirm that the [Sienna.md](./Sienna.md) file has been read.

PowerFlows.jl (NREL Sienna) solves AC and DC power flows over PowerSystems.jl `System`s. It
provides a unified interface to multiple solution methods plus utilities commonly found in
commercial software like Siemens PSS/e and GE PSLF (PSLF export not yet supported), and is
architected for large-scale systems (tens of thousands of buses) through extensive sparse matrix
operations and specialized linear solvers. Part of the Sienna-Platform ecosystem.
Julia ≥ 1.10. Package version: see `Project.toml` (currently 0.20.x).

## Commands

```bash
# Environment (test systems download via PowerSystemCaseBuilder on first use — needs network)
julia --project -e 'using Pkg; Pkg.instantiate()'

# Full test suite (ReTest + Aqua, via test/runtests.jl)
julia --project -e 'using Pkg; Pkg.test()'

# Interactive / filtered tests (preferred while developing)
julia --project -e 'using TestEnv; TestEnv.activate(); include("test/load_tests.jl");
                    using .PowerFlowsTests; run_tests()'   # run_tests accepts ReTest filters

# Format (CI fails on any resulting diff — run before committing)
julia -e 'include("scripts/formatter/formatter_code.jl")'

# Docs
julia --project=docs docs/make.jl
```

## Architecture (verified file:line anchors)

**Solvers ⊥ formulations.** `AbstractACPowerFlow{S<:ACPowerFlowSolverType}` has three concrete
*formulations* — `ACPolarPowerFlow{S}` (`src/power_flow_types.jl:150`; alias `ACPowerFlow` :256),
`ACRectangularPowerFlow{S}` (:315, Da Costa current injection), `ACMixedPowerFlow{S}` (:424,
mixed current-power balance) — each parametrized by an orthogonal *solver*: NewtonRaphson,
TrustRegion, LevenbergMarquardt, RobustHomotopy (polar-only), GradientDescent (polar-only),
FastDecoupled (in progress, all formulations). Rect/mixed constructors reject polar-only solvers
via `ArgumentError` (:351-364). DC models: `DCPowerFlow`, `PTDFDCPowerFlow`, `vPTDFDCPowerFlow`.

**Dispatch chain (AC):** `solve_power_flow` / `solve_and_store_power_flow!`
(`src/solve_ac_power_flow.jl`) → `solve_power_flow!(::ACPowerFlowData)` (:147, per-time-step
loop) → `_ac_power_flow` (:241, Q-limit outer loop) → per-solver `_newton_power_flow` methods
(generic NR/TR at `src/power_flow_method.jl:844`; LM, gradient-descent, homotopy, fast-decoupled
have their own methods) → shared `_run_power_flow_method(time_step, ::StateVectorCache,
::PFLinearSolverCache, residual, J, ::Type{Solver}; kwargs...) -> (converged, iters)`.

**Key files:** `src/PowerFlowData.jl` (data container; AC constructor :543), `src/power_flow_setup.jl`
(`initialize_power_flow_variables` → `(residual, J, x0)`; `improve_x0` warm-start chain),
`src/ac_power_flow_residual.jl` / `src/ac_power_flow_jacobian.jl` (+ `rectangular_ci_*`,
`mixed_cpb_*` counterparts), `src/state_indexing_helpers.jl` (state layout),
`src/linear_solver_backend.jl` (KLU/AppleAccelerate/Pardiso abstraction), `src/definitions.jl`
(all constants/defaults), `src/post_processing.jl` (results/write-back).

## Load-bearing conventions (violating these causes silent wrongness)

- **Step sign:** Newton steps solve `J·Δx = r` then **negate** (`rmul!(Δx,-1)`,
  `src/power_flow_method.jl:150`), then `x .+= Δx`. Mirror this; never re-derive signs from
  textbooks — write a test against the exact Jacobian instead.
- **Residual functor side effect:** `residual(x, t)` updates `residual.Rv` AND syncs `data`
  fields (V/θ/P/Q). Jacobian functors `J(t)` read `data`, not `x` → always call the residual
  before updating J (`src/power_flow_method.jl:520-524`).
- **Polar layout** (per bus i in Ybus order): `x[2i-1], x[2i]` = REF `(P,Q)` (P slot doubles as
  the subnetwork slack variable under distributed slack), PV `(Q,θ)`, PQ `(V,θ)`; 4 trailing
  entries per LCC HVDC. `Rv[2i-1]` = P mismatch, `Rv[2i]` = Q mismatch.
- **Convergence:** `norm(residual.Rv, Inf) < tol`, defaults `tol=1e-9`, `maxIterations=50`
  (`src/definitions.jl:15-16`).
- **Kwargs flow:** `pf.solver_settings::Dict{Symbol,Any}` is merged with call kwargs (explicit
  kwargs win, `src/solve_ac_power_flow.jl:153`); every driver must absorb unknown keys via
  `_ignored...`.
- **Q-limit loop:** `_ac_power_flow` re-invokes `_newton_power_flow` from scratch after PV→PQ
  switching — bus types never change *within* one driver invocation.
- **Linear algebra:** go through `src/linear_solver_backend.jl` wrappers (`make_linear_solver_cache`,
  `full_factor!`, `numeric_refactor!`, `solve!`) — never KLU directly. Sparse matrices for these
  must be `SparseMatrixCSC{Float64, J_INDEX_TYPE}` (Int32; Int64 on macOS).
- **PNM matrices store ComplexF32** (Ybus, arc admittances) — expect ~1e-4 noise vs Float64.
- **`data.solver_cache`** is DC-path-only today (`src/solve_dc_power_flow.jl:24-41`); the FD
  effort claims it for `ACPowerFlowData` with a tagged value — keep uses type-disjoint.
- **Loss/voltage-stability factors are computed from `J.Jv` at finalization**
  (`src/power_flow_method.jl:784-806`) — ensure J is evaluated at the *solution* before
  `_finalize_power_flow` (fixed-Jacobian/decoupled solvers must refresh it explicitly).
- `PowerFlowData` retains **no raw branch R/X/tap data** — derive network quantities from the
  stored `PNM.Ybus` (+ `arc_admittance_from_to/to_from`) so network reductions stay consistent.

## Main Objectives

1. **Multiple Solution Methods**: A unified interface to various power flow solution algorithms:
   - DC Power Flow (linear approximation)
   - AC Power Flow using Newton-Raphson
   - AC Power Flow using Trust Region methods
   - AC Power Flow using Levenberg-Marquardt
   - Robust Homotopy methods for difficult-to-converge systems
   - PTDF-based DC Power Flow methods
2. **Commercial Software Compatibility**: Export power flow results in formats compatible with
   PSS/e. PSLF not supported yet.
3. **Integration with Sienna**: Seamless integration with PowerSystems.jl and
   PowerSimulations.jl (PSI) for comprehensive power systems analysis.
4. **High-Performance at Scale**: Sparse matrix operations (SparseArrays.jl), specialized sparse
   direct solvers (KLU from SuiteSparse), sparsity-aware Jacobian/Hessian construction, in-place
   operations, cached factorizations, and parallel factorizations (CHOLMOD) in homotopy methods.
5. **Flexibility**: Multi-period analysis, HVDC systems and LCC converters, distributed slack bus
   models, arc types and network reductions, loss factor and stability factor calculations.

### Solution Methods Selection Guide

- **DCPowerFlow**: Fast linear approximation, good for screening studies
- **NewtonRaphsonACPowerFlow**: Standard method, fast convergence for well-conditioned systems
- **TrustRegionACPowerFlow**: More robust than Newton-Raphson, handles ill-conditioned cases better
- **LevenbergMarquardtACPowerFlow**: Robust nonlinear solver, good for difficult cases
- **RobustHomotopyPowerFlow**: Most robust method for hard-to-converge or non-convergent cases
- **PTDFDCPowerFlow** / **vPTDFDCPowerFlow**: DC power flow with pre-computed PTDFs

### Typical Usage Pattern

1. **Create System**: Use PowerSystems.jl to create or load a `System` object
2. **Choose Method**: Select an appropriate power flow method (DC, AC with specific algorithm)
3. **Solve**: Call `solve_power_flow` or `solve_power_flow!` (in-place)
4. **Analyze Results**: Extract voltages, flows, losses, etc. from the results
5. **Export** (optional): Export results in PSS/e format using `PSSEExporter`

## Repository Structure

### Core Source Code (`src/`)

- **`PowerFlows.jl`**: Main module file containing exports and include statements
- **Data & setup**: `PowerFlowData.jl` (core problem state), `power_flow_types.jl` (method types),
  `initialize_power_flow_data.jl`, `power_flow_setup.jl`, `power_flow_method.jl` (solve entry point)
- **DC power flow**: `solve_dc_power_flow.jl` (DC + PTDF methods)
- **AC power flow**: `solve_ac_power_flow.jl`, `ac_power_flow_residual.jl`,
  `ac_power_flow_jacobian.jl`, `levenberg-marquardt.jl`
- **Robust Homotopy** (`RobustHomotopy/`): `robust_homotopy_method.jl`, `homotopy_hessian.jl`,
  `HessianSolver/` (KLU, CHOLMOD, Cholesky)
- **Linear algebra** (`LinearSolverCache/`): `linear_solver_cache.jl`, `klu_linear_solver.jl`
- **Utilities**: `common.jl`, `definitions.jl` (constants), `state_indexing_helpers.jl`,
  `lcc_parameters.jl` & `lcc_utils.jl` (LCC HVDC), `powersystems_utils.jl`, `psi_utils.jl`,
  `post_processing.jl`, `psse_export.jl`

### Tests (`test/`)

- **`runtests.jl`**: Main test entry point
- **DC**: `test_dc_power_flow.jl`, `test_multiperiod_dc_power_flow.jl`, `test_reduced_dc_power_flow.jl`
- **AC**: `test_solve_power_flow.jl`, `test_multiperiod_ac_power_flow.jl`, `test_reduced_ac_power_flow.jl`
- **Specialized**: `test_hvdc.jl`, `test_distributed_slack.jl`, `test_robust_power_flow.jl`,
  `test_iterative_methods.jl`
- **Components**: `test_jacobian.jl`, `test_homotopy_hessian.jl`, `test_klu_linear_solver_cache.jl`
- **Utilities**: `test_psse_export.jl`, `test_post_processing.jl`, `test_power_flow_data.jl`,
  `test_loss_factors.jl`; **Integration**: `test_psi_utils.jl`
- **Performance**: `performance/performance_test.jl`; **Data**: `test_data/`; **Helpers**: `test_utils/`

### Documentation (`docs/`) — Diataxis framework

`src/` (Markdown sources), `make.jl` (build script), `build/` (generated, untracked).
`tutorials/`, `how-tos/`, `reference/` (with `api/` and `developers/`), `explanation/`.

### Scripts & Config

`scripts/formatter/` (JuliaFormatter); `Project.toml`, `Manifest.toml` (locked), `codecov.yml`,
`CONTRIBUTING.md`, `LICENSE` (BSD).

## Performance and Scalability Architecture

**Sparse-First Design**: All matrix operations use sparse representations — Y-bus, Jacobians,
Hessians, PTDF matrices.

**Specialized Linear Solvers**: KLU (primary sparse direct solver for AC, optimized for circuit
matrices), CHOLMOD (Cholesky for SPD systems in homotopy methods), custom Hessian solvers.

**Solver Caching and Reuse**: Symbolic factorization computed once and reused across Newton
iterations; numeric factorization updated only when needed.

**Memory Efficiency**: In-place operations (`!` functions), pre-allocated working arrays in
`PowerFlowData`, views over copies, type-stability discipline (see [Sienna.md](./Sienna.md)).

**Scalability Testing**: Validated on WECC, EI (Eastern Interconnect), and ACTIVSg2000 systems.

**Performance-critical components**: Jacobian construction
([ac_power_flow_jacobian.jl](../src/ac_power_flow_jacobian.jl)), linear solver cache
([LinearSolverCache/](../src/LinearSolverCache/)), residual evaluation
([ac_power_flow_residual.jl](../src/ac_power_flow_residual.jl)), homotopy Hessian
([RobustHomotopy/homotopy_hessian.jl](../src/RobustHomotopy/homotopy_hessian.jl)).

**Contributor discipline**: Profile before optimizing (`@time`, `@allocated`, `@code_warntype`),
preserve sparsity patterns, add in-place hot-path variants, document complexity for new
algorithms, test impact on large systems. See [Sienna.md](./Sienna.md).

## Integration with PowerSimulations.jl (PSI)

PowerFlows is used within PowerSimulations.jl for initialization of dynamic simulation problems,
network model validation, and post-processing of optimal power flow results. The `psi_utils.jl`
file provides the integration layer.

## Testing conventions

- Solver-parametrized testsets loop over `AC_SOLVERS_TO_TEST` (`test/PowerFlowsTests.jl:59-64`);
  per-solver skips need a comment (precedent: `test/test_loss_factors.jl:5-6`).
- Tolerances: `DIFF_INF_TOLERANCE=1e-4`, `DIFF_L2_TOLERANCE=1e-3`, `TIGHT_TOLERANCE=1e-7`.
- Systems via PowerSystemCaseBuilder: `c_sys5`, `c_sys14`, `matpower_case5_sys`,
  `matpower_ACTIVSg2000_sys`; PSS/E raw files (incl. WECC240) in `test/test_data/`.
- Helpers: `test/test_utils/common.jl` (`_calc_x`, simple-system builders),
  `jacobian_verification.jl`, `psse_results_compare.jl`. Allocation-test pattern:
  `test/test_ac_nr_allocations.jl`.
- Aqua runs in CI: every export needs a docstring; no stale deps.

## Style

- Follow `CONTRIBUTING.md`, [Sienna.md](./Sienna.md), and surrounding code. snake_case functions
  (internal ones `_`-prefixed), CamelCase types, solver types named `*ACPowerFlow`. Logging:
  `@debug` iteration detail, `@info` convergence/bus-type changes, `@warn` recoverable, `@error`
  non-convergence. Constants live in `src/definitions.jl`. Always run the formatter before committing.
- **Docstrings** follow the Diataxis framework; new exports require one (Aqua enforces this).

## Development Workflow

1. **Setup**: Clone, instantiate the environment with `]instantiate` in the Julia REPL
2. **Code**: Follow Sienna conventions (see [Sienna.md](./Sienna.md))
3. **Format**: Run the formatter (`scripts/formatter/`) before committing
4. **Test**: Run tests with `]test PowerFlows` or specific test files
5. **Document**: Add docstrings following Diataxis
6. **Submit**: Create a pull request following `CONTRIBUTING.md`

## Current effort: Fast/Fixed Decoupled Newton-Raphson (FDNR)

Authoritative plan: **`FDNR_IMPLEMENTATION_PLAN.md`** (repo root). Read it before touching
anything FD-related. Summary: add `FastDecoupledACPowerFlow` (PSS/E FDNS-equivalent) — polar
`:decoupled` B′/B″ variant (XB/BX schemes), `:fixed_jacobian` frozen-J variant for all three
formulations, opt-in handoff (`:handoff_solver`, default `nothing`) into the existing
`_run_power_flow_method` for NR/TR refinement, factor-once caching across iterations/time steps.

Non-negotiable rules for the multi-agent implementation:
1. Follow the plan's work-package DAG (WP0 → WP1∥WP2 → WP3∥WP4 → WP5 → WP6; WP7 gated).
   Interface contracts in plan §3–§4 are frozen; contract changes go through the integrator.
2. **T1 first** (plan §6): the B′/B″-vs-exact-Jacobian unit test is the sign/value arbiter and
   must pass before any iteration-loop work merges.
3. Do NOT add `FastDecoupledACPowerFlow` to the rect/mixed constructor rejection unions.
4. New defaults/constants go in `src/definitions.jl` as `DEFAULT_FD_*` (values in plan §3.2).
5. Industry alignment is documented in plan §11 — primary source: PSS/E 36.1.0 POM §6.5–6.7
   (FNSL/NSOL/FDNS), plus MATPOWER makeB/fdpf, PowerWorld FD→NR robust process, van Amerongen.
   Safeguards (non-divergent backtracking, BLOWUP, DVLIM) are PSS/E-parity; §11 also lists five
   *deliberate* divergences. Don't "fix" the divergences or "simplify" the safeguards.

## Dependencies & Resources

**Key external packages**: PowerSystems.jl (data models/network), PowerNetworkMatrices.jl
(admittance/incidence/PTDF), InfrastructureSystems.jl (shared core), KLU.jl (sparse direct
solver), DataFrames.jl (tabular results).

- **Documentation**: https://sienna-platform.github.io/PowerFlows.jl/dev/
- **Style Guide**: https://sienna-platform.github.io/InfrastructureSystems.jl/stable/style/
- **Issues**: GitHub Issues for bug reports and feature requests
