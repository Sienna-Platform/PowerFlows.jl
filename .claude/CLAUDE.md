# PowerFlows.jl — Claude Guide (psy6 branch)

Platform-wide Sienna conventions (performance, type stability, formatter, environments, code style) live in `.claude/Sienna.md` — read it too. This file is repo-specific and does not restate them.

## Purpose & place in the stack

PowerFlows.jl provides AC and DC power-flow solution methods for large-scale systems (tens of thousands of buses), built on PowerSystems.jl. It is the steady-state network-solution layer of the psy6 stack. **The consumer in this line is PowerOperationsModels (POM)** — via POM's `ext/PowerFlowsExt` extension, which wraps PF models in `PowerFlowEvaluator <: IOM.AbstractEvaluator` so IOM stays PF-agnostic — for dynamic-init, network validation, and OPF post-processing. (The old PowerSimulations does not exist in this line; interfaces labeled "PSI-stable" below are the same protected surface, now consumed by POM.) Design priorities: sparse-first (SparseArrays), specialized sparse direct solvers via PowerNetworkMatrices (PNM) backends, factorization reuse, in-place/`!` hot paths. It also exports results in PSS/E `.raw` format (PSLF not supported).

Coupling (all wired via the psy6 shared env / `[sources]` — compat numbers stay unbumped until release):
- **PowerSystems.jl** (`PSY`, psy6 branch): `System` and component model; input to all solvers (`src/powersystems_utils.jl`).
- **PowerNetworkMatrices.jl** (`PNM`, psy6 branch): Y-bus, PTDF, incidence, network reductions, AND the linear-solver caches/factorization backends (see below). PNM owns network-reduction logic; PowerFlows must pass reduced tuples through it.
- **InfrastructureSystems.jl** (`IS`, IS4 branch): shared infra, `@assert_op`, serialization.

## Explicit units (psy6 — check every PSY read)

Under psy6, PSY getters on convertible fields take an explicit unit-system argument and the
power-flow layer works in **system base**: `PSY.get_x(br, PSY.SU)`. PNM aggregators already
return system base. A bare `PSY.get_*` on a convertible field in this repo is a defect; the
POM/PNM/PF consumer sweep for such bare getters is a known open work item — when touching a
file, fix the bare getters you see. Angle limits are radians (no base conversion). Wrong
flow/limit magnitudes after a refactor → suspect units first.

## Architecture & `src/` layout

Include order is authoritative — see `src/PowerFlows.jl`. New types/consts must be defined in a file included before its first use (e.g. `linear_solver_backend.jl` precedes `PowerFlowData.jl`; `definitions.jl` is first).

- **Module / config:** `PowerFlows.jl` (exports + includes), `definitions.jl` (consts), `common.jl`.
- **Data & types:** `power_flow_types.jl` (evaluation-model types), `PowerFlowData.jl` (problem state, pre-allocated working arrays), `initialize_power_flow_data.jl`, `state_indexing_helpers.jl`, `power_flow_setup.jl`, `power_flow_method.jl` (dispatch entry).
- **DC:** `solve_dc_power_flow.jl` (DC + PTDF/vPTDF), `dcpf_loss_injection.jl` (DC loss factors).
- **AC — polar:** `ac_power_flow_residual.jl`, `ac_power_flow_jacobian.jl`, `solve_ac_power_flow.jl` (unified NR/TR driver), `levenberg-marquardt.jl`, `gradient_descent_ac_power_flow.jl`.
- **AC — rectangular current-injection:** `rectangular_ci_setup.jl`, `rectangular_ci_power_flow_residual.jl`, `rectangular_ci_power_flow_jacobian.jl`.
- **AC — mixed current-power-balance (MCPB):** `mixed_cpb_setup.jl`, `mixed_cpb_power_flow_residual.jl`, `mixed_cpb_power_flow_jacobian.jl`.
- **Robust homotopy:** `RobustHomotopy/robust_homotopy_method.jl`, `homotopy_hessian.jl`, `HessianSolver/{hessian_solver,KLU_hessian_solver,fixed_structure_CHOLMOD,cholesky_solver}.jl`.
- **Linear-solver backends:** `linear_solver_backend.jl` (backend selection/dispatch over PNM caches; see invariants).
- **Diagnostics / results:** `residual_condition_diagnostics.jl`, `post_processing.jl`, `branch_flow_results.jl`.
- **HVDC/LCC:** `lcc_parameters.jl`, `lcc_utils.jl`.
- **Export:** `psse_export.jl` (PSS/E `.raw`).
- **Pardiso backend:** `ext/PowerFlowsPardisoExt.jl` (weakdep extension).

## Public API / entry points

Exported solver-model types and functions (see `src/PowerFlows.jl`):
- Solve: `solve_power_flow`, `solve_power_flow!` (in-place; not exported but PSI-stable), `solve_and_store_power_flow!`.
- DC models: `DCPowerFlow`, `PTDFDCPowerFlow`, `vPTDFDCPowerFlow` (all `<: AbstractDCPowerFlow`).
- AC formulation/solver types — **two-axis design**: a *formulation* type parameterized by an `S <: ACPowerFlowSolverType`:
  - Formulations: `ACPolarPowerFlow{S}`, `ACRectangularPowerFlow{S}`, `ACMixedPowerFlow{S}`, all `<: AbstractACPowerFlow{S}`. `const ACPowerFlow = ACPolarPowerFlow` (back-compat alias; POM uses it).
  - Solver types `S`: `NewtonRaphsonACPowerFlow`, `TrustRegionACPowerFlow`, `LevenbergMarquardtACPowerFlow`, `RobustHomotopyPowerFlow`, `GradientDescentACPowerFlow`, and `FastDecoupledACPowerFlow{V<:FDVariant,S<:FDScheme}` (the one *parametric* solver — variant/scheme are type params, not settings: `FDDecoupled`/`FDFixedJacobian` × `FDSchemeXB`/`FDSchemeBX`; bare `FastDecoupledACPowerFlow` picks per-formulation defaults).
  - Example: `ACRectangularPowerFlow{NewtonRaphsonACPowerFlow}`. The solver is the type parameter — there is no `:step_strategy`/`:formulation` settings flag.
- Export: `PSSEExportPowerFlow`, `PSSEExporter`, `update_exporter!`, `write_export`, `get_psse_export_paths`, `FlowReporting`.
- `PowerFlowData` and aliases, plus `write_results`, are not exported but are PSI-stable; treat as protected interface.

`PowerFlowData` is constructed per `AbstractACPowerFlow`/`AbstractDCPowerFlow` and holds pre-allocated state, the PNM network matrix, and cached factorization slots.

## Key conventions, invariants & gotchas

**Formulation vs solver split (design law).** Formulation = concrete type; solver = type param `S`. Seams are dispatched, not branched: `initialize_power_flow_variables` (formulation-dispatched), `_finalize_formulation!` hook (polar no-op; rect = `rect_finalize_bus_injections!`). NR and TR share one `_newton_power_flow(::AbstractACPowerFlow{S})`. LM/Homotopy/GD `_newton_power_flow` are pinned to `ACPolarPowerFlow` (and LM also to rectangular/mixed); illegal formulation×solver pairs are rejected at construction, not at runtime. Never branch on formulation with `isa`/`<:` — add a dispatch method.

**Rectangular CI numerics.** PQ 2 vars, PV 3 vars with a `|V|²` pin row, REF 2 gen vars. Off-diagonal Jacobian blocks are constant (≡ Y_bus blocks). `V_FLOOR2 = 1e-16` floors `e²+f²` in all `1/|V|²`/`1/|V|` residual+Jacobian terms — **but the PV `|V|²−V_set²` constraint row keeps RAW `e²+f²`** (its −2e/−2f Jacobian is floor-free); residual and derivative must stay consistent. `_update_ref_diag_block!` is shared with MCPB, so flooring hardens both.

**MCPB (mixed current-power-balance).** PQ buses use divided current balance (imag-first row order); PV buses use real-power balance + `|V|²` constraint with **only 2 vars/bus** (no Q state — the key difference from rectangular's 3). REF = (P_gen, Q_gen). System size is exactly 2n. Status: opt-in, NOT default; do not deprecate rectangular. Validated to polar parity. Performance: for NR/TR ≈ rectangular (no net win); for **LM, Mixed decisively beats Rectangular** (rectangular-LM fails to converge at ~10k buses; Mixed-LM converges like Polar-LM with the smallest 2n state). Jacobian kernels are called as concrete top-level functions, never stored as abstract `::Function` fields (that forces dynamic dispatch on the hot path).

**`validate_voltage_magnitudes`** exists for polar, rectangular, and mixed. For rect/mixed it checks squared bounds (`e²+f² ∈ [min²,max²]`) for PQ and PV (PV `(e,f)` are real state vars and `|V|²` can drift mid-iteration before the constraint row pins it); REF skipped. Toggle via `solver_settings[:validate_voltage_magnitudes]`.

**Linear-solver backends (PNM-owned).** PowerFlows no longer hand-rolls a KLU cache — KLU is **not** a direct dependency. Backends come from PNM (`PNM.KLULinSolveCache{Tv,Ti}`, `PNM.AAFactorCache` for AppleAccelerate) plus a PowerFlows-defined `PardisoLinSolveCache` in `ext/PowerFlowsPardisoExt.jl`. Selection = PNM preference default + per-solve kwarg (AC: `solver_settings[:linear_solver]`; DC: kwarg).
- Index width gates the AppleAccelerate path: `INDEX_TYPE = @static Sys.isapple() ? Int64 : Int32` (drives `J_INDEX_TYPE`/`REC_INDEX_TYPE`). AA's Apple `libSparse` ABI needs Int64 `columnStarts`; KLU uses Int32 elsewhere. Any cache-type `Union` must list BOTH `KLULinSolveCache{Float64,Int32}` and `{…,Int64}` (PNM's DC ABA factorization is always Int64 — omitting it MethodErrors on Linux).
- The KLU and AA backends do NOT share generic functions: `PNM.solve!/full_factor!/symbolic_factor!/numeric_refactor!/tsolve!` are KLU-only (`.KLUWrapper`); AA's live in `PNM.AccelerateWrapper.*`. PowerFlows defines local dispatch over both cache types. `AAFactorCache` is Int-only with NO transpose solve, so voltage-stability factors (need Aᵀ\b) stay KLU-only.
- MKLPardiso is x86_64-only; `resolve_linear_solver_backend` rejects it when `Sys.ARCH !== :x86_64`; Pardiso tests gate on `Pardiso.mkl_is_available()`.

**Singular-matrix handling is backend-agnostic.** Native singular signals are heterogeneous (KLU throws `SingularException`, AppleAccelerate silently returns finite garbage, Pardiso perturbs pivots). The authoritative guard is in `_set_Δx_nr!`: catch any factorization throw → fallback; then use the **relative residual** `‖JΔx−r‖/‖r‖` from `_do_refinement!`; if above `DEFAULT_REFINEMENT_THRESHOLD` (5%) → regularized KLU fallback. Native signals are diagnostics only.

**LCC / network reduction.** PNM's zero-impedance reduction merges LCC-terminal buses; `data.lcc.arcs` stores REDUCED tuples. Post-processing (`get_lcc_names`, `arc_to_lcc`) must key with reduced tuples via `get_arc_tuple(PSY.get_arc(lcc), nrd)` where `nrd = PNM.get_network_reduction_data(data.power_network_matrix)` — keying with raw tuples is a KeyError. Degree-2 parity tests must build systems with `reduce_reactive_power_injectors=false` (the default drops susceptive-FA shunts).

**Perf NR-cache reuse (polar).** `PolarNRCache` (slot `data.polar_nr_cache::RefValue{Union{Nothing,AbstractNRCache}}`) reuses residual/Jacobian/symbolic factorization across Q-limit retries and time steps; LCC or a changed subnetwork/slack forces a rebuild. `data.solver_cache::RefValue{Union{Nothing,SolverCache}}` holds the analogous per-solve cache — `DCSolverCache` (DC/PTDF) or `FastDecoupledCache` (FDNR factor-once B′/B″); the getters dispatch on the concrete subtype, so a cross-use is a loud `MethodError`, not a silent mis-read (no sentinel tag).

**Benchmark measurement trap.** Repeated `_ac_power_flow`/`solve_power_flow!` on the same `data` warm-starts to 0-iteration convergence (lazy early-return). Perturb injections per rep or you measure nothing. Use iteration count (not wall-clock) as the robust metric; the wall-clock timer is noisy. Background heavy compute (10k benchmark, full perf suite) — never block synchronously in a subagent.

**Known unfixed issues (flag before touching the results layer).**
- `write_results` is non-idempotent (it `+=` withdrawals).
- **NaN poisoning on non-convergence** (`solve_ac_power_flow.jl`, `OVERWRITE_NON_CONVERGED` path): a failed solve overwrites `PowerFlowData` with NaNs, so retries are non-idempotent and downstream consumers see poisoned state instead of an error. This is one of the platform's named silent-failure patterns — never extend it; new failure paths must error loudly with context.
- `PowerFlowData` is bare-field accessed across ~40 files (audit-flagged porous encapsulation) — prefer getters when touching these sites; don't add new direct reaches.
- No REF-bus-per-island pre-check before post-processing (`post_processing.jl`) — islanding surfaces late.

## Commands (verified against this clone)

This package uses **ReTest** and a `test/Project.toml` env (deps incl. PowerSystemCaseBuilder, ReTest, Pardiso, Aqua). Read the `sienna-test-environment` skill for the shared rules; PowerFlows specifics:

```sh
# Compile-check between edits (package env, fast):
julia --project -e 'using PowerFlows'

# One-time per clone: make --project=test resolve PowerFlows to the WORKING TREE
# (else it can resolve the registered copy in ~/.julia/packages and run stale source,
#  and new test/test_*.jl files are invisible to the glob in test/PowerFlowsTests.jl):
julia --project=test -e 'using Pkg; Pkg.develop(PackageSpec(path=pwd()))'
# Verify (must print the working-tree path, not ~/.julia/packages/...):
julia --project=test -e 'import Pkg; println(Base.find_package("PowerFlows"))'

# Run full suite:
julia --project=test test/runtests.jl

# Run a filtered subset via ReTest:
julia --project=test -e 'using PowerFlows; include("test/PowerFlowsTests.jl"); using .PowerFlowsTests, ReTest; retest(PowerFlowsTests, r"<regex>")'

# Docs:
julia --project=docs docs/make.jl

# Formatter (run after every task — Sienna rule):
julia --project=scripts/formatter -e 'include("scripts/formatter/formatter_code.jl")'
```

ReTest runs the whole suite and reports failures at the end (does not abort on first failure). Note `runtests.jl` aborts the whole run at the first exception outside a `@test`; "suite green" means the run REACHED the final `Main.PowerFlowsTests | <N>` summary with no Error column. Under recent PSY/IS, `PSY.System("file.raw")` may not parse PSS/E raw — use the PowerSystemCaseBuilder `PowerFlowFileParser` path for raw inputs in tests.

## Auto-generated files / do-not-edit

- `Manifest.toml`, `test/Manifest.toml` (gitignored), `docs/build/` (untracked).
- Do not edit files generated by the formatter's reflow; a file watcher may reformat between read and edit — re-Read immediately before Edit.

## Scripts

- `scripts/formatter/formatter_code.jl` (verified) — formatter.
- `scripts/benchmarks/{method_comparison,formulation_solver_comparison}.jl` — formulation×solver comparisons.
- `scripts/profiling/profile_power_flow_solvers.jl` — solver profiling.

## Version

Package `0.21.1`; Julia `^1.10`. Pardiso is a weakdep providing the optional MKLPardiso AC backend.
