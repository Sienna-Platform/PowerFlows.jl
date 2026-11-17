# PowerFlows.jl — Claude Guide (psy6 branch)

This file is repo-specific. The shared Sienna rules load from the import below; the `sienna-dev` skill holds the cross-package
context; the `sienna-dev` and `sienna-reviewer` subagents implement and review.

@Sienna.md

## Purpose & place in the stack

PowerFlows.jl provides AC and DC power-flow solution methods for large-scale systems (tens of thousands of buses), built on PowerSystems.jl. It is the steady-state network-solution layer of the psy6 stack. **The consumer in this line is PowerOperationsModels (POM)** — via POM's `ext/PowerFlowsExt` extension, which wraps PF models in `PowerFlowEvaluator <: IOM.AbstractEvaluator` so IOM stays PF-agnostic — for dynamic-init, network validation, and OPF post-processing. (PowerSimulations (PSI) is the simulation layer above IOM and POM. Interfaces labeled "PSI-stable" below are the same protected surface, now consumed by POM.) Design priorities: sparse-first (SparseArrays), specialized sparse direct solvers via PowerNetworkMatrices (PNM) backends, factorization reuse, in-place/`!` hot paths. It also exports results in PSS/E `.raw` format (PSLF not supported).

Coupling (all wired via `[sources]` — compat numbers stay unbumped until release):
- **PowerSystems.jl** (`PSY`, psy6 branch): `System` and component model; input to all solvers (`src/powersystems_utils.jl`).
- **PowerNetworkMatrices.jl** (`PNM`, psy6 branch): Y-bus, PTDF, incidence, network reductions, AND the linear-solver caches/factorization backends (see below). PNM owns network-reduction logic; PowerFlows must pass reduced tuples through it.
- **InfrastructureSystems.jl** (`IS`, IS4 branch): shared infra, `@assert_op`, serialization.

## Explicit units (psy6 — check every PSY read)

Under psy6, PSY getters on convertible fields take an explicit unit-system argument and the
power-flow layer works in **system base**: `PSY.get_x(br, u"SU")`. PNM aggregators already
return system base. A bare `PSY.get_*` on a convertible field in this repo is a defect; when you touch a file,
fix the bare getters you see. Angle limits are radians (no base conversion). Wrong
flow/limit magnitudes after a refactor → suspect units first.

## Architecture & `src/` layout

Include order is authoritative — see `src/PowerFlows.jl`. New types/consts must be defined in a file included before its first use (e.g. `linear_solver_backend.jl` precedes `PowerFlowData.jl`; `definitions.jl` is first).

File names map to solvers and formulations. The non-obvious parts:

- `solve_ac_power_flow.jl` is the unified NR/TR driver; `power_flow_method.jl` is the dispatch entry.
- `rectangular_ci_*.jl` hold the rectangular **power-mismatch** formulation, despite the `ci` in the name.
- `mixed_cpb_*.jl` hold MCPB; `fast_decoupled_*.jl` hold the fast-decoupled solver.
- `staged_handoff.jl` holds the stage handoff that NR, TR, and LM share.
- `GeneralizedAdmittance/ga_{partition,shunts,cache,iteration,lcc,dc,method}.jl` load in that order after `gradient_descent_ac_power_flow.jl`. `ga_method.jl` holds `_ga_solve` (driver) and the `_newton_power_flow` seam.
- `psi_utils.jl` holds the device power-contribution traits (`contributes_active_power`, `active_power_contribution_type`), not PSI code.
- PowerFlows has no Pardiso dependency. `PNM.PardisoLinSolveCache` comes from PowerMatricesCore, and its `MKLPardisoExt` loads on `import Pardiso`.

## Public API / entry points

Exported solver-model types and functions (see `src/PowerFlows.jl`):
- Solve: `solve_power_flow`, `solve_power_flow!` (in-place; not exported but PSI-stable), `solve_and_store_power_flow!`.
- DC models: `DCPowerFlow`, `PTDFDCPowerFlow`, `vPTDFDCPowerFlow` (all `<: AbstractDCPowerFlow`).
- AC formulation/solver types — **two-axis design**: a *formulation* type parameterized by an `S <: ACPowerFlowSolverType`:
  - Formulations: `ACPolarPowerFlow{S}`, `ACRectangularPowerFlow{S}`, `ACMixedPowerFlow{S}`, all `<: AbstractACPowerFlow{S}`. `const ACPowerFlow = ACPolarPowerFlow`. POM calls `ACPowerFlow()`; do not remove the alias without a POM change.
  - Solver types `S`: `NewtonRaphsonACPowerFlow`, `TrustRegionACPowerFlow`, `LevenbergMarquardtACPowerFlow`, `RobustHomotopyPowerFlow`, `GradientDescentACPowerFlow`, `GeneralizedAdmittanceACPowerFlow`, and `FastDecoupledACPowerFlow{V<:FDVariant,S<:FDScheme}` (the one *parametric* solver — variant/scheme are type params, not settings: `FDDecoupled`/`FDFixedJacobian` × `FDSchemeXB`/`FDSchemeBX`; bare `FastDecoupledACPowerFlow` picks per-formulation defaults).
  - Example: `ACRectangularPowerFlow{NewtonRaphsonACPowerFlow}`. The solver is the type parameter — there is no `:step_strategy`/`:formulation` settings flag.
- Export: `PSSEExportPowerFlow`, `PSSEExporter`, `update_exporter!`, `write_export`, `get_psse_export_paths`, `FlowReporting`.
- `PowerFlowData` and aliases, plus `write_results`, are not exported but are PSI-stable; treat as protected interface.

`PowerFlowData` is constructed per `AbstractACPowerFlow`/`AbstractDCPowerFlow` and holds pre-allocated state, the PNM network matrix, and cached factorization slots.

## Key conventions, invariants & gotchas

**Transformers are two types, and everything electrical lives on the circuit.** PSY exposes exactly `TwoWindingTransformer` and `ThreeWindingTransformer`, holding one or three `PSY.TransformerCircuit` objects that carry the arc, tap, α, `r`/`x`, ratings, control objective, and per-circuit `available`. There is no tap type, no phase-shifter type, and no 2W/3W split in the electrical model — reach the circuit with `PSY.get_circuit(tx)` (2W) or `PSY.get_circuits(tx)` (both arities; a 2W returns a 1-tuple) and read the same getters either way. Four rules:
- **Never reintroduce a per-behavior transformer type.** Tap and phase shift are not type-encoded. Read `PSY.get_tap(circuit)` and `PSY.get_alpha(circuit)` unconditionally, and use `PSY.is_phase_shifting(circuit)` where a predicate is needed — never a type check. That predicate is true when α ≠ 0 **or** the control objective is one of four active-power objectives, so it is broader than a plain α test.
- **Report at the circuit level, uniformly across arities.** A 3W may regulate on one circuit while the others are fixed, so the circuit — not the transformer — is the unit of control and of results. `get_controlled_device_results` reports family `"TransformerCircuit"` with `device_name` + `circuit_index` identifying the owning circuit for both arities (`circuit_index == 1` for a 2W).
- `PNM.ThreeWindingTransformerCircuit` wraps a `TransformerCircuit` directly and is not parameterized on a PSY transformer type.
- `get_series_susceptance` lives in **PNM**, not PSY. Call it as `PNM.get_series_susceptance(branch, units)`; there is no PSY fallback.

**Formulation vs solver split (design law).** Formulation = concrete type; solver = type param `S`. Seams are dispatched, not branched: `initialize_power_flow_variables` (formulation-dispatched), `_finalize_formulation!` hook (polar no-op; rect = `rect_finalize_bus_injections!`). NR and TR share one `_newton_power_flow(::AbstractACPowerFlow{S})`. LM/Homotopy/GD `_newton_power_flow` are pinned to `ACPolarPowerFlow` (and LM also to rectangular/mixed); illegal formulation×solver pairs are rejected at construction, not at runtime. Never branch on formulation with `isa`/`<:` — add a dispatch method.

**Rectangular power-mismatch numerics.** 2 state slots per bus; PQ and REF rows are (ΔP, ΔQ) with `ΔS = V·conj(Y_bus·V) − S_spec`; PV rows are (ΔP, `|V|²−V_set²`); REF slots hold net (P, Q). Off-diagonal Jacobian blocks are not constant. `V_FLOOR2 = 1e-16` floors `e²+f²` only in the const-I ZIP terms (`|V|` in `S_spec`) and the VSC kernels; the PV `|V|²−V_set²` row keeps RAW `e²+f²` (its −2e/−2f Jacobian is floor-free), so residual and derivative stay consistent. `_update_ref_diag_block!` is mixed-only (MCPB REF current-balance block).

**Generalized admittance (GA, `ACPolarPowerFlow{GeneralizedAdmittanceACPowerFlow}`).** Loads and generators are fixed shunts on the non-slack Y-bus block; a fixed-point loop of corrective currents restores the exact power and PV constraints.
- Polar only. Rectangular and mixed reject it; construction also rejects `check_reactive_power_limits`, distributed slack, discrete control and area interchange (`_validate_solver_specific_settings`).
- Handoff to NR/TR/LM is opt-in. "No handoff" is the `NoHandoff` sentinel type, never `nothing`; GA and FD share `_maybe_handoff!` and `_validate_handoff_solver(solver, label)`. Without a handoff the exit gap also bounds each island's signed P sum (`ws.psum`, `_ga_exit_gap`), which equals minus the synced REF P row; the stage tolerance is plain `tol`.
- **Convergence design (do not revert to the paper's plain fixed point).** The paper's flat-start q0 (§II-C) is off by tens of p.u. at stiffly tied PV buses on large grids, which makes the plain fixed point *repel* (ACTIVSg10k: linearized ρ ≈ 11.7). The stage therefore (1) adds an inductive `GA_PV_STIFFNESS_FRACTION·|Y_kk|` shunt at PV buses (`_ga_stiffen_pv!`), (2) Anderson-mixes the corrective currents with REAL coefficients (`GAAnderson`; the map conjugates u, so it is only ℝ-linear), and (3) refreshes the shunts to the ideal shunts of the iterate each time the per-bus gap falls `GA_REFRESH_DROP`-fold, re-stiffening on a stall (`_ga_refresh_shunts!`, `i += Δy ⊙ u`, numeric refactor). Heuristics run on the per-bus gap; the island sum only gates the exit. `test_generalized_admittance.jl` covers the ACTIVSg10k flat start, where NR from flat diverges.
- `GeneralizedAdmittanceCache` (in `data.solver_cache`) holds two complex KLU factorizations (`Yℓℓ`, `Yqq`) via `PNM.*` (not the Float64 wrappers in `linear_solver_backend.jl`). The sparsity pattern never changes, so shunt updates use `numeric_refactor!`. Sparse path: no `\`, `inv`, `lu`.
- LCC terminals are constant P+jQ withdrawals from a closed form; VSC converters are constant injections, with a DC substep (`ga_dc.jl`) when a DC network exists. `data.bus_type` is never mutated; AC-voltage VSC buses join the v set inside the partition only.
- The stage ends by writing the best iterate into `PowerFlowData` and closing REF/PV/LCC/VSC state through the shared explicit-state sync. It adds no NaN-overwrite path.

**MCPB (mixed current-power-balance).** PQ buses use divided current balance (imag-first row order); PV buses use real-power balance + `|V|²` constraint with **only 2 vars/bus** (no Q state). REF = (P_gen, Q_gen). System size is exactly 2n. MCPB is opt-in, not the default; do not deprecate rectangular. `test_mixed_cpb_polar_parity.jl` checks parity with polar. Jacobian kernels are called as concrete top-level functions, never stored as abstract `::Function` fields (that forces dynamic dispatch on the hot path).

**`validate_voltage_magnitudes`** exists for polar, rectangular, and mixed. For rect/mixed it checks squared bounds (`e²+f² ∈ [min²,max²]`) for PQ and PV (PV `(e,f)` are real state vars and `|V|²` can drift mid-iteration before the constraint row pins it); REF skipped. Toggle via the `validate_voltage_magnitudes` keyword of the formulation constructor (a `SolutionParameters` field; the old `solver_settings` Dict is gone).

**Linear-solver backends (PNM-owned).** PowerFlows has no KLU cache of its own, and KLU is **not** a direct dependency. Backends come from PNM, which re-exports them from PowerMatricesCore: `PNM.KLULinSolveCache{Tv,Ti}`, `PNM.AAFactorCache` for AppleAccelerate, and `PNM.PardisoLinSolveCache` (PowerMatricesCore's `MKLPardisoExt`, loaded on `import Pardiso`). Selection = PNM preference default + the `linear_solver` keyword (AC: a `SolutionParameters` field, resolved to a `String` at construction; DC: per-solve kwarg). `"Dense"` is rejected at resolution time.
- Index width gates the AppleAccelerate path: `INDEX_TYPE = @static Sys.isapple() ? Int64 : Int32` (drives `J_INDEX_TYPE`/`REC_INDEX_TYPE`). AA's Apple `libSparse` ABI needs Int64 `columnStarts`; KLU uses Int32 elsewhere. Any cache-type `Union` must list both `KLULinSolveCache{Float64,Int32}` and `{…,Int64}` (PNM's DC ABA factorization is always Int64 — omitting it MethodErrors on Linux).
- All backend caches share PNM's `PNM.solve!/full_factor!/symbolic_factor!/numeric_refactor!/is_factored`, plus `PNM.make_linear_solver_cache` and `PNM.factorization_ok`; PowerFlows forwards to them and keeps its own KLU `Float64` arms for the lean plan. `PNM.tsolve!` is KLU-only. `AAFactorCache` is Int-only with NO transpose solve, so voltage-stability factors (need Aᵀ\b) stay KLU-only.
- MKLPardiso is x86_64-only; `resolve_linear_solver_backend` rejects it when `Sys.ARCH !== :x86_64`; Pardiso tests gate on `Pardiso.mkl_is_available()`.

**Singular-matrix handling is backend-agnostic.** Native singular signals are heterogeneous (KLU throws `SingularException`, AppleAccelerate silently returns finite garbage, Pardiso perturbs pivots). The authoritative guard is in `_set_Δx_nr!`: catch any factorization throw → fallback; then use the **relative residual** `‖JΔx−r‖/‖r‖` from `_do_refinement!`; if above `DEFAULT_REFINEMENT_THRESHOLD` (5%) → regularized KLU fallback. Native signals are diagnostics only.

**LCC / network reduction.** PNM's zero-impedance reduction merges LCC-terminal buses; `data.lcc.arcs` stores REDUCED tuples. An arc does not identify an LCC (parallel LCCs share one, and a reduction can map several onto one), so post-processing uses `data.lcc.names`, never an arc-keyed lookup. `RadialReduction` and raw data can also give `fb == tb`, so the LCC tail-row × bus-column Jacobian slots are zeroed and then accumulated. Degree-2 parity tests must build systems with `reduce_reactive_power_injectors=false` (the default drops susceptive-FA shunts).

**Perf NR-cache reuse (polar).** `PolarNRCache` (slot `data.polar_nr_cache::RefValue{Union{Nothing,AbstractNRCache}}`) reuses residual/Jacobian/symbolic factorization across Q-limit retries and time steps; LCC or a changed subnetwork/slack forces a rebuild. Residuals, Jacobians and `HomotopyHessian` never store `data` — it is passed explicitly (`R(data, x, t)`, `J(data, t)`, `ACPowerFlowJacobian(data, residual, t)`), because the cache hangs off `data` and a stored back-reference would form a cycle; `test_nr_cache_reuse.jl` guards this. `data.solver_cache::RefValue{Union{Nothing,SolverCache}}` holds the analogous per-solve cache — `DCSolverCache` (DC/PTDF) or `FastDecoupledCache` (FDNR factor-once B′/B″); the getters dispatch on the concrete subtype, so a cross-use is a loud `MethodError`, not a silent mis-read (no sentinel tag).

**Benchmark measurement trap.** Repeated `_ac_power_flow`/`solve_power_flow!` on the same `data` warm-starts to 0-iteration convergence (lazy early-return). Perturb injections per rep or you measure nothing. Use iteration count (not wall-clock) as the robust metric; the wall-clock timer is noisy. Background heavy compute (10k benchmark, full perf suite) — never block synchronously in a subagent.

**Known unfixed issues (flag before touching the results layer).**
- `write_results` is non-idempotent (it `+=` withdrawals).
- **NaN poisoning on non-convergence** (`solve_ac_power_flow.jl`, `OVERWRITE_NON_CONVERGED` path): a failed solve overwrites `PowerFlowData` with NaNs, so retries are non-idempotent and downstream consumers see poisoned state instead of an error. This is one of the platform's named silent-failure patterns — never extend it; new failure paths must error loudly with context.
- `PowerFlowData` is bare-field accessed across many files — prefer getters when touching these sites; don't add new direct reaches.
- No REF-bus-per-island pre-check before post-processing (`post_processing.jl`) — islanding surfaces late.

## Commands (verified against this clone)

This package uses **ParallelTestRunner** and a `test/Project.toml` env (deps incl. PowerSystemCaseBuilder, ParallelTestRunner, Pardiso, Aqua) — one worker process per `test_*.jl` file, sharing nothing but `test/includes.jl`'s preamble. The shared test rules are in `.claude/Sienna.md` and the `sienna-dev` skill (`references/environments.md`); PowerFlows specifics:

```sh
# Compile-check between edits (package env, fast):
julia --project -e 'using PowerFlows'

# test/Project.toml path-pins this checkout; Pkg.instantiate() is the whole setup.
# Verify (must print the working-tree path, not ~/.julia/packages/...):
julia --project=test -e 'import Pkg; println(Base.find_package("PowerFlows"))'

# Run full suite:
julia --project=test test/runtests.jl

# Generalized admittance (one file, ~3 min, incl. an ACTIVSg10k flat start; top-level @testset "GeneralizedAdmittance"):
julia --project=test test/runtests.jl test_generalized_admittance

# Run a subset filtered by FILE name (startswith), cap parallelism, or list discoverable tests:
julia --project=test test/runtests.jl test_dc_power_flow
julia --project=test test/runtests.jl --jobs=4
julia --project=test test/runtests.jl --list

# Docs:
julia --project=docs docs/make.jl

# Formatter (run after every task — Sienna rule):
julia --project=scripts/formatter -e 'include("scripts/formatter/formatter_code.jl")'
```

Each test file runs as its own testset in its own worker process; the runner reports pass/fail per file and does not abort the whole run on one file's failure. PSY has no raw parser; build raw inputs through PowerFlowFileParser via PSB.

## Auto-generated files / do-not-edit

- `Manifest.toml`, `test/Manifest.toml` (gitignored), `docs/build/` (untracked).
- Do not edit files generated by the formatter's reflow; a file watcher may reformat between read and edit — re-Read immediately before Edit.

## Scripts

- `scripts/formatter/formatter_code.jl` (verified) — formatter.
- `scripts/benchmarks/{method_comparison,formulation_solver_comparison}.jl` — formulation×solver comparisons.
- `scripts/profiling/profile_power_flow_solvers.jl` — solver profiling.

## Version

Package `0.25.2`; Julia `^1.10`. PowerFlows has no Pardiso dependency; Pardiso is a test dependency only.
