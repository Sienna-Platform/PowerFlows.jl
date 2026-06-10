# CLAUDE.md — PowerFlows.jl

PowerFlows.jl (NREL Sienna) solves AC and DC power flows over PowerSystems.jl `System`s.
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

- Follow `CONTRIBUTING.md` and surrounding code. snake_case functions (internal ones `_`-prefixed),
  CamelCase types, solver types named `*ACPowerFlow`. Logging: `@debug` iteration detail,
  `@info` convergence/bus-type changes, `@warn` recoverable, `@error` non-convergence.
  Constants live in `src/definitions.jl`. Always run the formatter before committing.

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
