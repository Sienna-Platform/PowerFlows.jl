# Fast/Fixed Decoupled Newton-Raphson (FDNR) AC Power Flow for PowerFlows.jl

Implementation plan for a multi-agent Claude Code session. All interface claims below were
verified by reading the code at the cited locations (branch base: current `main` checkout,
package version 0.20.1).

## 1. Context

PowerFlows.jl has five AC solver algorithms (`NewtonRaphsonACPowerFlow`, `TrustRegionACPowerFlow`,
`LevenbergMarquardtACPowerFlow`, `RobustHomotopyPowerFlow`, `GradientDescentACPowerFlow`) across
three AC formulations (`ACPolarPowerFlow`, `ACRectangularPowerFlow` (Da Costa current injection),
`ACMixedPowerFlow` (mixed current-power balance)). Every Newton-family solver refactorizes the
Jacobian at each iteration. Missing is the workhorse of commercial tools — PSS/E `FDNS`
("fixed-slope decoupled Newton-Raphson"), a.k.a. Stott–Alsac fast decoupled power flow: constant
approximate Jacobian blocks (B′/B″) factorized **once** and reused across all iterations *and*
time steps, trading quadratic for linear convergence at a fraction of the per-iteration cost.
Ideal for PCM-style repeated solves, contingency screening, and as a cheap initializer for the
exact-Newton family.

**Requirements (user-confirmed):**
1. FDNR semantics as in commercial tools (PSS/E FDNS).
2. **Decoupled/fixed-slope variants for all three formulations** (user's explicit choice):
   polar = classic B′/B″ with XB and BX schemes; rectangular CI and mixed CPB = fixed-slope
   (frozen-Jacobian, factor-once) variants, plus an *experimental, gated* true decoupled
   real/imag split for the CI family.
3. **Handoff capability, OFF by default** (user's explicit choice): opt-in configuration to hand
   the FD-stage state to another solver (NR/TrustRegion, LM as follow-on) for refinement to final
   tolerance. Pure FD must converge to full `tol` on its own by default.
4. Commercial-grade engineering: factor-once caching across iterations and time steps, Q-limit
   bus-type switching, stall/divergence safeguards, allocation-free iteration, logging, tests, docs.

**Correctness principle that de-risks the whole design** (state it in docs and rely on it in
reviews): the FD iteration evaluates the **exact** residual every iteration; the fixed B′/B″ (or
frozen J) only precondition the update. Any reasonable approximation changes the convergence
*rate*, never the converged *solution*. Convergence is declared on the same criterion as every
other solver: `norm(residual.Rv, Inf) < tol`.

## 2. Verified architecture contract (what the implementation plugs into)

Cited signatures are load-bearing; do not redesign them.

- **Type tree** (`src/power_flow_types.jl`): `AbstractACPowerFlow{S<:ACPowerFlowSolverType}` with
  concrete formulations `ACPolarPowerFlow{S}` (:150), `ACRectangularPowerFlow{S}` (:315),
  `ACMixedPowerFlow{S}` (:424). `const ACPowerFlow = ACPolarPowerFlow` (:256). Solver types are
  empty singleton structs (:67, :77, :99, :111). Rect/mixed constructors **reject**
  `RobustHomotopyPowerFlow`/`GradientDescentACPowerFlow` via `ArgumentError` (:351-364 and the
  mixed analog) — the new FD solver must NOT be added to those rejection unions (all three
  formulations support it). Settings flow via `solver_settings::Dict{Symbol,Any}` →
  `get_solver_kwargs(pf)` (:268) → merged with call kwargs in `solve_power_flow!`
  (`src/solve_ac_power_flow.jl:153`, explicit kwargs win) → passed to `_newton_power_flow`.
- **Dispatch chain**: `solve_power_flow!(data::ACPowerFlowData; kwargs...)`
  (`src/solve_ac_power_flow.jl:147`) loops time steps → `_ac_power_flow(data, pf, time_step;
  kwargs...)` (:241) wraps the **Q-limit outer loop** (`MAX_REACTIVE_POWER_ITERATIONS=10`;
  `_check_q_limit_bounds!` (:262) flips PV→PQ in `data.bus_type[:, t]` and clamps
  `data.bus_reactive_power_injections`) → `_newton_power_flow(pf, data, time_step; kwargs...)`.
  **Each Q-limit retry re-invokes `_newton_power_flow` from scratch — bus types never change
  inside one driver invocation.** Per-solver drivers exist as separate `_newton_power_flow`
  methods: generic NR/TR (`src/power_flow_method.jl:844`, constrained
  `where {T<:Union{TrustRegionACPowerFlow,NewtonRaphsonACPowerFlow}}` so a new method creates no
  ambiguity), LM (`src/levenberg-marquardt.jl:125`), gradient descent
  (`src/gradient_descent_ac_power_flow.jl:123`), homotopy (`src/RobustHomotopy/robust_homotopy_method.jl:1`).
- **Setup**: `initialize_power_flow_variables(pf, data, time_step; x0=nothing,
  validate_voltage_magnitudes, vm_validation_range, _ignored...)` → `(residual, J, x0_computed)`,
  formulation-dispatched (`src/power_flow_setup.jl:287` polar, :317 rect, :338 mixed). Includes
  `improve_x0` chain (flat start → previous-converged-timestep warm start → enhanced flat start →
  polar-only DC fallback when `robust_power_flow`). Polar updates J at x0 (`J(time_step)`, :306).
- **Iteration protocol** (`src/power_flow_method.jl`): residual functor `residual(x, t)` updates
  `residual.Rv` in-place AND syncs `data` fields (V, θ, P, Q) — Jacobian functors read `data`,
  not `x` (comment at :520-524). Step convention: solve `J·Δx = r` then **negate**
  (`rmul!(Δx_nr, -1.0)` at :150), `x .+= Δx`. `_run_power_flow_method(time_step,
  stateVector::StateVectorCache, linSolveCache::PFLinearSolverCache, residual, J,
  ::Type{Solver}; maxIterations, tol, ..., _ignored...)` → `(converged, i)` exists for NR (:642)
  and TR (:717) over the residual union `Union{ACPowerFlowResidual, ACRectangularCIResidual,
  ACMixedCPBResidual}` and Jacobian union `Union{ACPowerFlowJacobian, ACRectangularCIJacobian,
  ACMixedCPBJacobian}` — **drop-in handoff targets**. LM's `_run_power_flow_method`
  (`src/levenberg-marquardt.jl:166`) has a different signature (`x::Vector`, `ws::LMWorkspace`) —
  LM handoff needs a small adapter.
- **Finalization** (`src/power_flow_method.jl:784-842`): `_finalize_formulation!(pf, data,
  x_final, residual, time_step)` (rect/mixed redistribute subnetwork slack into data arrays) then
  `_finalize_power_flow(converged, i, name, residual, data, J.Jv, time_step)` (logging + optional
  loss factors / voltage-stability factors **computed from `J.Jv`** → FD must refresh J at the
  solution when those are requested, else silently wrong factors).
- **Polar state/residual layout** (`src/state_indexing_helpers.jl`,
  `src/ac_power_flow_residual.jl`): per bus i (Ybus matrix order), `x[2i-1], x[2i]` = REF:
  (P_net, Q_net) — under distributed slack the REF "P" slot is the subnetwork slack variable —
  PV: (Q_net, θ), PQ: (V, θ); then **4 trailing entries per LCC** (rectifier/inverter tap,
  thyristor angles; `update_state!` at :28). Residual rows: `Rv[2i-1]` = active-power mismatch,
  `Rv[2i]` = reactive (decoded in `improve_x0`, `src/power_flow_setup.jl:36-37`); `length(Rv) =
  2·n_buses + 4·n_lcc`. REF/PV "explicit" rows have ±1 coefficients on their own x entries; PV/REF
  `P_net[i] = P_net_set[i] + γᵢ·s` couples all subnetwork P rows to the slack variable `s`
  (rank-1; `_set_state_variables_at_bus!`, `src/ac_power_flow_residual.jl:192-224`).
  `residual.subnetworks::Dict{Int64,Vector{Int64}}` (REF bus → member buses) and
  `residual.bus_slack_participation_factors::SparseVector` are available on the residual.
- **Linear solver backend** (`src/linear_solver_backend.jl`): `PFLinearSolverCache` =
  KLU/AppleAccelerate/Pardiso union; ops `make_linear_solver_cache(tag, A)`, `symbolic_factor!`,
  `numeric_refactor!`, `full_factor!`, `solve!`; `resolve_linear_solver_backend(::Union{Nothing,
  AbstractString})`. **Matrices must be `SparseMatrixCSC{Float64, J_INDEX_TYPE}`**
  (`J_INDEX_TYPE` = Int32, Int64 on macOS; `src/definitions.jl:54-64`).
- **Network matrices**: `PowerFlowData` retains NO raw branch parameters. AC constructor
  (`src/PowerFlowData.jl:543-571`) builds `PNM.Ybus(sys; make_arc_admittance_matrices=true,
  include_constant_impedance_loads=false)`. Available: `data.power_network_matrix.data`
  (sparse **ComplexF32** Ybus), `.arc_admittance_from_to`/`.arc_admittance_to_from` (arc×bus
  sparse; row a has nonzeros only at the from/to bus columns; usage proof at
  `src/solve_ac_power_flow.jl:169-178, 227-228`), `PNM.get_bus_lookup`, `PNM.get_arc_axis`,
  `axes(data.power_network_matrix, 1)` = bus numbers, `bus_type_idx(data, t, types)` helper
  (`src/solve_ac_power_flow.jl:289`). Deriving B′/B″ from these (not from `PSY.System`) keeps
  consistency with `network_reductions`.
- **Cache slot**: `data.solver_cache::Base.RefValue{Any}` (`src/PowerFlowData.jl:150`) is used
  **only** by the DC path (grep-verified: all uses in `src/solve_dc_power_flow.jl`; the polar DC
  fallback uses `get_aux_network_matrix(data).K` instead, `src/power_flow_setup.jl:270`). DC
  functions only accept `ABAPowerFlowData`/PTDF data types → for `ACPowerFlowData` the slot is
  free. **FD claims it for ACPowerFlowData** (no struct change); update the field's doc comment.
- **Constants** (`src/definitions.jl`): `DEFAULT_NR_TOL=1e-9`, `DEFAULT_NR_MAX_ITER=50`,
  `LARGE_RESIDUAL`, `OVERWRITE_NON_CONVERGED=true`, naming conventions for new FD constants.
- **Tests**: ReTest-based; `AC_SOLVERS_TO_TEST = (NewtonRaphson, TrustRegion, LevenbergMarquardt,
  RobustHomotopy)` (`test/PowerFlowsTests.jl:59-64`); tolerances `DIFF_INF_TOLERANCE=1e-4`,
  `DIFF_L2_TOLERANCE=1e-3`, `TIGHT_TOLERANCE=1e-7`; systems c_sys5/c_sys14 (PSB),
  matpower_case5, ACTIVSg2000, WECC240 raw in `test/test_data`; per-solver skip precedent in
  `test/test_loss_factors.jl:5-6`; allocation-test pattern in `test/test_ac_nr_allocations.jl`;
  end-to-end example in `test/test_solve_power_flow.jl` (hard-coded `result_14`, asserts
  `norm(result_14 - x1, Inf) <= 3e-6`).
- **Docs**: autodocs from source files listed in `docs/src/reference/api/public.md`; explanation
  pages in `docs/src/explanation/` (e.g. `mixed_cpb_formulation.md`); tutorial with
  solver×formulation performance table at `docs/src/tutorials/solving_a_power_flow.jl`.

**Not verifiable in this environment** (no Julia depot): exact PNM accessor names beyond those
already exercised in-repo at the cited lines. Implementers must confirm against the instantiated
PNM version (compat `^0.23`) before relying on anything not cited above.

## 3. Public API design

### 3.1 New solver type

```julia
"""
    FastDecoupledACPowerFlow <: ACPowerFlowSolverType

Fixed-slope decoupled Newton-Raphson (the fast decoupled power flow of Stott & Alsac (1974),
van Amerongen (1989); equivalent in role to PSS/E's FDNS). Constant approximate Jacobian
factor(s) built once and reused across all iterations and time steps; exact mismatches each
iteration. Linear convergence rate, very low cost per iteration. Optional handoff to an exact
Newton-family solver for final refinement.
... (full docstring: settings table below, references, See also)
"""
struct FastDecoupledACPowerFlow <: ACPowerFlowSolverType end
```

In `src/power_flow_types.jl` next to the other solver singletons; exported from
`src/PowerFlows.jl`. Works with **all three formulations** (do not add it to the rect/mixed
rejection unions). Name rationale: literature-standard "fast decoupled"; docstring explicitly
cross-references "fixed(-slope) decoupled Newton-Raphson"/FDNS so PSS/E users find it.

### 3.2 Solver settings (via `solver_settings` dict and/or call kwargs)

| Key | Default | Meaning |
|---|---|---|
| `fd_variant::Symbol` | polar: `:decoupled`; rect/mixed: `:fixed_jacobian` | `:decoupled` = B′/B″ half-iterations (polar only in v1); `:fixed_jacobian` = frozen full formulation Jacobian, factor-once (all formulations) |
| `fd_scheme::Symbol` | `:XB` | B′/B″ scheme, `:XB` (Stott–Alsac, PSS/E-like) or `:BX` (van Amerongen). Only meaningful for polar `:decoupled` |
| `handoff_solver` | `nothing` | `nothing` (pure FD) or `NewtonRaphsonACPowerFlow` / `TrustRegionACPowerFlow` (v1; LM follow-on). Validated, descriptive `ArgumentError` otherwise |
| `handoff_tol::Float64` | `DEFAULT_FD_HANDOFF_TOL = 1e-3` | FD-stage exit ∞-norm when handoff configured |
| `refreeze_on_stall::Bool` | `true` | `:fixed_jacobian` only: on stall, re-evaluate + refactor the frozen J once, continue |
| `maxIterations::Int` | `DEFAULT_FD_MAX_ITER = 150` | FD-stage cap (linear convergence needs more, cheaper iterations than NR's 50) |
| `tol`, `validate_voltage_magnitudes`, `vm_validation_range`, `linear_solver` | existing defaults | shared semantics with other solvers |

New constants in `src/definitions.jl` (naming style of existing block at :15-45):
`DEFAULT_FD_MAX_ITER = 150`, `DEFAULT_FD_HANDOFF_TOL = 1e-3`, `DEFAULT_FD_STALL_WINDOW = 10`
(iterations without sufficient best-norm improvement), `DEFAULT_FD_STALL_RATIO = 0.99`
(improvement factor defining "progress"), `DEFAULT_FD_DIVERGENCE_FACTOR = 1e3` (abort when
`‖Rv‖∞ > factor·best`), `DEFAULT_FD_SCHEME = :XB`, `DEFAULT_FD_REFREEZE_ON_STALL = true`.

Usage examples (for docs):
```julia
pf = ACPowerFlow{FastDecoupledACPowerFlow}()                                   # pure FDNR, polar, XB
pf = ACPowerFlow{FastDecoupledACPowerFlow}(; solver_settings = Dict(
    :fd_scheme => :BX, :handoff_solver => NewtonRaphsonACPowerFlow, :handoff_tol => 1e-2))
pf = ACMixedPowerFlow{FastDecoupledACPowerFlow}()                              # fixed-slope MCPB
```

## 4. Algorithm specification

### 4.1 Polar `:decoupled` — classic FDNR

**Index sets** (per driver invocation; bus types frozen within it — see §2 Q-limit loop):
`pvpq = [i for bus i with type PV or PQ]`, `pq = [i: type PQ]` (use `bus_type_idx`). Maps:
θ entries `x[2i]` for i∈pvpq; V entries `x[2i-1]` for i∈pq; P rows `Rv[2i-1]` for i∈pvpq; Q rows
`Rv[2i]` for i∈pq. Precompute position vectors (`theta_x_idx`, `v_x_idx`, `p_row_idx`,
`q_row_idx`, and `bus→subvector` positions) in the FD workspace.

**Iteration** (one "iteration" = one P-θ + one Q-V half-step; standard successive scheme):
```
setup: residual, J, x0 from initialize_power_flow_variables (J needed for handoff/loss factors)
       fetch-or-build FD cache: B′ factored (full_factor!), B″ factored for current PQ set
       sync explicit rows (below); residual(x, t); best = ‖Rv‖∞
loop while i < maxIterations and not converged:
  1. P half-step: rp[k] = Rv[p_row_idx[k]] / Vm[pvpq[k]]          (V from data.bus_magnitude)
     solve!(Bp_cache, rp);  x[theta_x_idx] .-= rp                 (mirror solve-then-negate)
  2. sync explicit rows; residual(x, t)
  3. Q half-step: rq[k] = Rv[q_row_idx[k]] / Vm[pq[k]]
     solve!(Bpp_cache, rq); x[v_x_idx] .-= rq
  4. sync explicit rows; residual(x, t)
  5. converged = ‖Rv‖∞ < stage_tol   (stage_tol = handoff_tol if handing off else tol)
     stall/divergence bookkeeping on ‖Rv‖∞ (best-norm tracking, window/ratio/factor constants)
optional voltage validation per iteration via _validate_state_magnitudes (as NR loop :691-696)
```

**Sign convention contract**: B′/B″ are defined as constant approximations of the codebase's own
Jacobian sub-blocks — B′ ≈ ∂(Rv_P/V)/∂θ over pvpq, B″ ≈ ∂(Rv_Q/V)/∂V over pq — and the update
mirrors `_set_Δx_nr!`'s solve-then-negate. **Do not derive signs from textbook conventions; the
unit test in §6 (T1) is the arbiter** (on a lossless, shunt-free, nominal-tap network at flat
start, B′ and B″ must equal the corresponding exact Jacobian blocks to machine precision — on
such a network XB = BX and the decoupled blocks are exact).

**Explicit-row sync** (`_sync_explicit_state!`): REF/PV x entries are explicit functions given
(V, θ): for each subnetwork with slack variable `s = x[2·ref−1]`: `s_new = s − sign·Σ_{i∈subnet}
Rv[2i−1]` (Σγᵢ = 1 ⇒ one exact rank-1 update of the distributed-slack column; reduces to the
classic REF-row update when participation is REF-only); REF Q: `x[2·ref] −= sign·Rv[2·ref]`;
PV Q: `x[2i−1] −= sign·Rv[2i]` (coefficient ±1 ⇒ single-step exact given V, θ). `sign` per the
same contract as above (test-arbitered). This keeps `‖Rv‖∞` meaningful as the global convergence
criterion, keeps `data.bus_reactive_power_injections` current so the **Q-limit outer loop works
unchanged**, and implements distributed slack as a per-iteration redistribution (commercial
outer-adjustment style).

**B′/B″ assembly** (new `src/fast_decoupled_matrices.jl`; everything `Float64`/`J_INDEX_TYPE`):
1. *Per-arc parameter recovery* from `Yft`/`Ytf` rows (ComplexF64-promoted):
   `yff = Yft[a, f]`, `yft = Yft[a, t]`, `ytf = Ytf[a, f]`, `ytt = Ytf[a, t]`. With the standard
   π-model stamp `yff = (ys + j·b_c/2)/|τ|²`, `yft = −ys/conj(τ)`, `ytf = −ys/τ`,
   `ytt = ys + j·b_c/2`: recover `|τ| = sqrt(real(ytt/yff))` (guard ≥ ε), phase
   `θ_τ = −angle(ytf/yft)/2`, `τ = |τ|·e^{jθ_τ}`, `ys = −yft·conj(τ)`, `b_c = 2·imag(ytt − ys)`,
   and per-bus shunt `ysh_i = Ybus[i,i] − Σ_incident(arc self terms)`. Parallel branches are
   pre-aggregated per arc by PNM — the recovered equivalent is acceptable (rate-only effect).
   Guard NaN/Inf; treat near-zero-impedance arcs by capping `1/x` (constant, e.g. 1e7) with a
   `@debug` note.
2. *Validation hook built into the module*: `_restamp_ybus(recovered_params)` rebuilds the full
   Ybus from recovered parameters; tests assert ≈ original within ComplexF32 noise (≤ ~1e-4 rel).
3. *Scheme stamping* (MATPOWER `makeB` semantics, restated explicitly):
   - **B′** (rows/cols restricted to pvpq, i.e. non-REF): stamp a temp network with `b_c = 0`,
     bus shunts = 0, `|τ| = 1` (phase shift **retained** → mildly unsymmetric; KLU/LU fine);
     XB: replace each `ys` by `1/(j·x_a)` where `x_a = imag(1/ys)` (resistance neglected);
     BX: keep `ys`. Then `B′ = −imag(Ybus_temp)[pvpq, pvpq]`.
   - **B″** (rows/cols restricted to pq): temp network with phase shift = 0 (|τ| retained),
     `b_c` and bus shunts **included**; XB: keep `ys`; BX: `ys → 1/(j·x_a)`. Then
     `B″_full = −imag(Ybus_temp2)` assembled over **all** buses once; per driver invocation
     extract/factor the `[pq, pq]` submatrix.
   - Multi-island networks: REF rows/cols of every subnetwork are excluded by the pvpq/pq
     restriction; the matrices are block-diagonal and nonsingular per island — one factorization
     covers all islands.
4. API: `build_fd_matrices(data, time_step, scheme) -> FDMatrices` (recovered params cached;
   B′ assembled+factored; B″_full assembled; `extract_bpp(FDMatrices, pq_set)` → factored cache).

### 4.2 `:fixed_jacobian` — all three formulations

The frozen-Jacobian ("dishonest Newton") variant generalizes the fixed-slope idea and is the
default for rect/mixed (their off-diagonal Jacobian blocks are constant Ybus terms already —
docstring at `src/power_flow_types.jl:288-289` — so freezing mostly affects diagonal blocks):
```
residual, J, x0 = initialize_power_flow_variables(pf, data, time_step; ...)
ensure J holds values at x0 (polar setup calls J(t); rect/mixed: call J(t) once if constructor doesn't)
cache = make_linear_solver_cache(backend, J.Jv); full_factor!(cache, J.Jv)     # ONCE
loop: copyto!(sv.r, residual.Rv); _solve_Δx_nr_frozen!(sv, cache)   # solve + negate, NO refactor
      sv.x .+= sv.Δx_nr; residual(sv.x, t); convergence/stall/divergence checks
on stall && refreeze_on_stall (once): J(t); numeric_refactor!(cache, J.Jv); continue
```
Notes: do **not** call `_set_Δx_nr!` (it refactors every call — that's the thing being avoided);
add a tiny frozen-step helper that reuses `_solve_Δx_nr!` + `rmul!(−1)`. LCC variables and
distributed slack are inside the frozen J ⇒ **this variant supports LCC systems and distributed
slack with no special handling**. Polar also accepts `:fixed_jacobian` (useful comparison mode
and the LCC-capable polar path).

### 4.3 Experimental rect/mixed `:decoupled` (gated, WP7)

True real/imag current split for the CI family (when G≪B: ΔI_r couples mainly to Δf, ΔI_m to Δe
via the constant B matrix; one factorization can serve both half-systems). PV-bus handling is the
hard part: the augmented rect formulation carries Q variables and |V|² rows (mixed: power-balance
+ |V|² rows at PV) whose columns/rows don't decouple cleanly; MCPB PQ rows are divided by V̄ᵢ,
making even off-diagonals state-dependent. **Acceptance gate**: NR-parity on every system in §6
(T3) or it ships disabled/undocumented. Not on the critical path; nothing else depends on it.

### 4.4 Handoff stage (opt-in)

In the FD driver, when `handoff_solver !== nothing` and the FD stage exits (reached
`handoff_tol`, stalled, diverged, or hit maxIterations) without meeting `tol`:
```
J(time_step)                                   # refresh Jacobian values at current x
hcache = make_linear_solver_cache(backend, J.Jv); symbolic_factor!(hcache, J.Jv)
converged, i2 = _run_power_flow_method(time_step, sv, hcache, residual, J, handoff_solver;
                                       tol, maxIterations = DEFAULT_NR_MAX_ITER, NR/TR kwargs...)
@info "FDNR stage: i1 iterations to ‖Rv‖∞=...; handoff to <solver>: i2 iterations"
```
Same `residual`/`StateVectorCache` objects — verified drop-in for NR/TR (§2). If FD already meets
`tol`, skip handoff. Pure-FD mode (`handoff_solver === nothing`): stall/divergence ⇒ converged =
false with an actionable `@error` (suggest `:BX`, `:fixed_jacobian`, handoff, or NR). LM handoff:
small adapter constructing `LMWorkspace` (see `src/levenberg-marquardt.jl:125-180`) — follow-on
inside WP4, keep `ArgumentError` until done. Works for all formulations and both fd_variants.
Cross-formulation "handoff" (e.g. polar FD warm-starting a rect solve) already exists naturally —
the residual functor syncs V/θ into `data`, and every formulation's `improve_x0` warm-starts from
`data` — document the two-solve composition in the explanation page rather than building API.

### 4.5 Driver, caching, multi-period

New method following the gradient-descent/homotopy precedent (no dispatch ambiguity with the
NR/TR-constrained generic at `src/power_flow_method.jl:844`):
```julia
function _newton_power_flow(pf::AbstractACPowerFlow{FastDecoupledACPowerFlow},
    data::ACPowerFlowData, time_step::Int64;
    tol = DEFAULT_NR_TOL, maxIterations = DEFAULT_FD_MAX_ITER,
    fd_variant = _default_fd_variant(pf),       # formulation-dispatched
    fd_scheme = DEFAULT_FD_SCHEME, handoff_solver = nothing, handoff_tol = DEFAULT_FD_HANDOFF_TOL,
    refreeze_on_stall = DEFAULT_FD_REFREEZE_ON_STALL,
    validate_voltage_magnitudes = DEFAULT_VALIDATE_VOLTAGES, vm_validation_range = DEFAULT_VALIDATION_RANGE,
    x0 = nothing, linear_solver = nothing, _ignored...)
```
Body: validate settings (scheme/variant/handoff combos; polar-`:decoupled` + LCC ⇒
`ArgumentError` pointing at `:fixed_jacobian` or handoff-capable alternatives, see §4.6) →
`initialize_power_flow_variables` → early-exit if already converged (mirror :879-882) → run
variant loop → optional handoff → ensure `J(time_step)` is current when
`get_calculate_loss_factors(data) || get_calculate_voltage_stability_factors(data)` →
`_finalize_formulation!` → `_finalize_power_flow(converged, total_iters, "FastDecoupled(...)",
residual, data, J.Jv, time_step)`.

**Caching** (`FastDecoupledCache` struct in the new method file): stored in `data.solver_cache[]`
(free for `ACPowerFlowData`, §2) as a tagged value so any future collision fails loudly:
`(FD_CACHE_TAG, cache)`. Contents: recovered arc parameters; per-scheme B′ matrix + factored
`PFLinearSolverCache`; B″_full; a small map `pq_signature => factored B″ cache` (signature =
`hash(view(data.bus_type, :, t))` + scheme + backend); preallocated `rp`/`rq` buffers and index
vectors. Invalidation keys: Ybus object identity (`data.power_network_matrix === cached`),
backend type, scheme. Effects: across time steps with identical bus-type columns and across
Q-limit retries that return to a previously-seen PQ set, **zero refactorizations**; B′ is
factored exactly once per (data, scheme, backend) lifetime; a PV→PQ switch refactors only B″
(small, ~ms at 10k buses). The FD iteration loop itself must be allocation-free (buffers in the
cache; `@views`; no closures capturing boxed state).

### 4.6 Special cases (explicit behavior matrix)

| Case | polar `:decoupled` | `:fixed_jacobian` (any formulation) |
|---|---|---|
| LCC HVDC present (`get_lcc_count(data) > 0`; LCC vars = 4 trailing x entries + 4 residual rows each) | `ArgumentError` at driver entry: half-iterations don't span LCC vars so `‖Rv‖∞` can't converge. Message points to `:fixed_jacobian`, rect/mixed FD, or NR/TR. (Per-LCC 4×4 Gauss-Seidel sub-step = WP7 stretch.) | Supported (LCC rows live in frozen J) |
| Distributed slack (participation factors / headroom) | Supported via rank-1 slack sync (§4.1); dedicated test T6; if validation shows instability on test systems, tighten to `ArgumentError` + suggest `:fixed_jacobian` (decision recorded in WP3) | Supported |
| Q-limit switching (`check_reactive_power_limits`) | Supported; B″ refactor via cache keyed on PQ set | Supported (outer loop re-invokes driver; J re-frozen at new x0) |
| Loss factors / voltage-stability factors (polar-only features) | Supported; driver refreshes `J(t)` at solution | Same |
| Multiple islands | Supported (block-diagonal B′/B″) | Supported |
| `correct_bustypes`, ZIP loads, `skip_redistribution`, exporter | Orthogonal — handled in data construction/residual; no FD-specific code | Same |

## 5. File-by-file change list

**New files**
- `src/fast_decoupled_matrices.jl` — arc parameter recovery, restamp validation helper, XB/BX
  stamping, B′/B″ assembly + submatrix extraction, `FDMatrices`. Pure functions of
  (Ybus, arc matrices, bus types) — independently unit-testable.
- `src/fast_decoupled_method.jl` — `FastDecoupledCache`, `_default_fd_variant`, settings
  validation, `_sync_explicit_state!`, polar decoupled loop, frozen-Jacobian loop, stall/
  divergence helpers, handoff stage, the `_newton_power_flow` method.

**Edits**
- `src/power_flow_types.jl` — `FastDecoupledACPowerFlow` struct + docstring (next to :111). No
  change to rect/mixed rejection unions. No loss-factor guard needed (FD supports them).
- `src/definitions.jl` — `DEFAULT_FD_*` constants (§3.2).
- `src/PowerFlows.jl` — `include` the two new files (after `power_flow_method.jl`); add
  `export FastDecoupledACPowerFlow`.
- `src/PowerFlowData.jl` — extend the `solver_cache` doc comment (:139-150) to cover the AC/FD use.
- `test/PowerFlowsTests.jl` — add `FastDecoupledACPowerFlow` to `AC_SOLVERS_TO_TEST` (:59-64).
- `test/test_fast_decoupled.jl` (new), `test/performance/` addition, targeted edits where
  parametrized tests need an FD-specific skip (only with justification; LM-skip precedent).
- `docs/src/explanation/fast_decoupled.md` (new), `docs/src/reference/api/public.md` (add new
  source files to autodocs Pages), `docs/src/tutorials/solving_a_power_flow.jl` (solver table +
  recommendation row).

## 6. Test plan (file: `test/test_fast_decoupled.jl` unless noted)

- **T1 — B-matrix correctness (the sign/value arbiter)**: build a small lossless, shunt-free,
  nominal-tap system (helpers in `test/test_utils/common.jl`: `_add_simple_bus!`/`_line!` etc.);
  at flat start assert B′ == J[p_rows/V, θ_cols] and B″ == J[q_rows/V, V_cols] (extracted from
  `ACPowerFlowJacobian.Jv`) to ~1e-6; assert XB == BX there. On c_sys14 + WECC240 (taps, phase
  shifts, shunts): restamp-Ybus reconstruction ≈ original (ComplexF32 tolerance); B′ unsymmetric
  only when phase shifters present; B″ symmetric.
- **T2 — Polar FDNR solution parity**: for scheme in (:XB, :BX), systems c_sys5/c_sys14/
  matpower_case5/ACTIVSg2000 (+ WECC240 import path): pure FD converges to `tol=1e-9` within
  `DEFAULT_FD_MAX_ITER`; final state matches NR state to `TIGHT_TOLERANCE` (pattern of
  `test_solve_power_flow.jl` with `_calc_x`); iteration count sanity (≥ NR's, > 5 — proves it's
  actually FD, guards accidental exact-Newton).
- **T3 — Fixed-Jacobian parity**: all three formulations × test systems: converges, NR-parity at
  `TIGHT_TOLERANCE`; rect/mixed LCC systems (reuse cases from `test_rectangular_ci_lcc.jl` /
  `test_mixed_cpb_lcc.jl`); polar+LCC via `:fixed_jacobian`; polar `:decoupled`+LCC throws.
- **T4 — Handoff**: FD→NR and FD→TR on c_sys14/ACTIVSg2000: converges to 1e-9; FD-stage iterations
  > 0 and handoff iterations small (≤ ~6); `handoff_tol` respected; invalid handoff solver throws;
  handoff skipped when FD alone meets tol (set `handoff_tol = tol`).
- **T5 — Q-limits**: c_sys14 variant forcing PV→PQ switching with `check_reactive_power_limits=true`
  (mirror existing Q-limit tests): converged, limits respected, results match NR run; assert via
  cache introspection that B′ was factored once across the outer-loop retries.
- **T6 — Distributed slack**: extend `test/test_distributed_slack.jl` solver tuple with FD (both
  variants); parity with NR distributed-slack results.
- **T7 — Multi-period & caching**: `time_steps > 1` (pattern of `test_multiperiod_ac_power_flow.jl`):
  all steps converge; FD cache reused (B′ factored once; B″ per distinct bus-type signature) —
  expose a counter on `FastDecoupledCache` for testability.
- **T8 — Stall/divergence path**: pathological case (e.g. very high r/x ratio system built with
  test utils, XB scheme): FD stalls → with handoff: still converges via NR; without: returns
  false, no exception, NaN-overwrite behavior intact (`solve_power_flow!` :184-195).
- **T9 — Loss factors / vstab factors**: c_sys14 with `calculate_loss_factors=true`: FD factors ==
  NR factors (tolerance per `test_loss_factors.jl`) — catches the stale-J pitfall.
- **T10 — Allocations**: mirror `test_ac_nr_allocations.jl` for the FD loop (post-warmup
  iteration allocation budget ~0).
- **T11 — Suite integration**: full existing suite green with FD in `AC_SOLVERS_TO_TEST`. Audit
  every parametrized testsite for explicit kwargs (e.g. hardcoded `maxIterations`) that conflict
  with FD's linear rate; prefer fixing the test parametrically; skip with comment only as last
  resort (LM precedent at `test_loss_factors.jl:5`).
- **Performance** (`test/performance`): 2000-bus benchmark — wall-time per solve FD vs NR for
  (a) single solve, (b) 24-step multi-period re-solve; report iteration counts. Target evidence
  (not hard CI assert): FD per-iteration ≥ 5× cheaper; multi-period total time competitive or
  better despite more iterations.
- Optional dev-time cross-check (not CI): MATPOWER `runpf` with `FDXB`/`FDBX` on case5/case14 —
  compare iteration counts (±2) and solutions.

## 7. Documentation plan

- Docstrings: solver type (settings table, PSS/E FDNS mapping, when-to-use), every new public-ish
  function; Aqua runs doc/export checks — keep them complete.
- `docs/src/explanation/fast_decoupled.md`: math (B′/B″, XB vs BX, what's neglected where);
  fixed-slope variant for the CI formulations; exactness-of-fixed-point argument; handoff design
  + examples; behavior matrix from §4.6; guidance table (FD vs NR vs TR vs LM vs homotopy);
  references (Stott & Alsac 1974; van Amerongen 1989; PSS/E FDNS).
- Tutorial: add FD to the solver enumeration + performance table; one handoff example.

## 8. Multi-agent work packages

Interface contracts = the signatures in §3–§4. Agents must not change contracts unilaterally;
contract changes go through the integrator (WP0 owner).

| WP | Content | Depends on | Parallel? | Acceptance |
|---|---|---|---|---|
| **WP0** | Type, constants, exports, settings parsing/validation, driver skeleton (`ArgumentError("not yet implemented")` per variant), docstring stubs | — | first, small, single agent | Package loads; `ACPowerFlow{FastDecoupledACPowerFlow}()` constructs for all 3 formulations; settings validation unit tests |
| **WP1** | `fast_decoupled_matrices.jl` + T1 | WP0 | ∥ WP2 | T1 green incl. WECC240 restamp |
| **WP2** | Frozen-Jacobian loop (all formulations) + stall/divergence helpers + T3 (non-LCC parts) + T8 skeleton | WP0 | ∥ WP1 | T3 parity green on polar+rect+mixed |
| **WP3** | Polar `:decoupled` loop: index maps, half-steps, `_sync_explicit_state!`, distributed slack; T2, T6 | WP0+WP1 (+WP2's shared helpers) | after WP1 | T2/T6 green; records the distributed-slack keep-or-restrict decision |
| **WP4** | Handoff stage (NR/TR; LM adapter follow-on) + T4 | WP0+WP2 (extend to WP3 output) | ∥ WP3 (against WP2 first) | T4 green for both variants |
| **WP5** | `FastDecoupledCache` in `data.solver_cache`, multi-period reuse, Q-limit integration, LCC gating, loss-factor J refresh, allocation-free pass; T5/T7/T9/T10, LCC parts of T3 | WP2+WP3 | integration, single agent | T5/T7/T9/T10 green; counter-verified factor-once behavior |
| **WP6** | Docs + tutorial + performance benchmarks + `AC_SOLVERS_TO_TEST` + full-suite audit (T11) | all above | last | Full suite + docs build green |
| **WP7** *(optional, gated)* | Experimental rect/mixed `:decoupled`; per-LCC sub-step for polar decoupled | WP3+WP5 | independent | Ships only at NR-parity on all T3 systems; otherwise flag stays errored |

Suggested staffing: WP1 and WP2 in parallel after WP0; then WP3 ∥ WP4; WP5 integrates; WP6 closes.
One integrator agent owns merges, contract changes, and the final suite run.

**Execution notes for the session**: instantiate test env (`julia --project=test -e 'using Pkg;
Pkg.develop(path="."); Pkg.instantiate()'` or per CI workflow) — PSB systems download on first
build; run targeted tests via ReTest (`include("test/load_tests.jl")` pattern /
`PowerFlowsTests.run_tests("test_fast_decoupled")`-style filtering). Match local code style
(no repo-root `.JuliaFormatter.toml` found — mirror surrounding code; check CI for a format job).
Do not bump the version or touch CHANGELOG (none exists) — maintainer's call.

## 9. Verification (end-to-end)

1. `julia --project=test` → run new `test_fast_decoupled.jl` testset, then the full suite.
2. Manual smoke (REPL): c_sys14 → `solve_power_flow(ACPowerFlow{FastDecoupledACPowerFlow}(), sys)`
   → DataFrames match the NR run; repeat with `:BX`, handoff config, `ACMixedPowerFlow`,
   `time_steps=24` PCM-style loop watching `@info` iteration/caching logs.
3. ACTIVSg2000: pure FD and FD+handoff converge; compare wall times vs NR (expect parity to win
   for repeated solves; report numbers in the PR description).
4. Docs build (`docs/make.jl`) clean; Aqua checks pass (exports documented).

## 10. Risks & self-critique

- **Sign/orientation errors** (residual convention × B-matrix sign × solve-negate): highest
  blast-radius bug class. Mitigated by making T1 the arbiter (exact equality on a network where
  decoupling is exact) before any loop work lands (WP1 before WP3).
- **Arc recovery edge cases**: series capacitors (x<0 — keep sign, never `abs`), zero/near-zero
  impedance branches (cap susceptance, log), reduction-generated equivalent arcs, ComplexF32
  noise, τ-side convention mismatches. Mitigated by the restamp-reconstruction test on WECC240 +
  reduced-network cases; residual exactness bounds the damage to convergence rate.
- **FD divergence on high-r/x or stressed cases is *expected algorithm behavior***, not a bug:
  the deliverables are honest stall detection, useful error guidance, BX option, and handoff —
  not unconditional convergence. Document prominently.
- **Distributed slack under `:decoupled`** is the most novel piece (sequential rank-1 update).
  T6 validates; pre-agreed fallback = restrict with a clear error while `:fixed_jacobian` covers
  it. Decision must be recorded in WP3, not silently dropped.
- **Stale-J hazards**: loss factors / vstab factors / handoff must refresh `J(t)` first (T9, T4).
- **`data.solver_cache` reuse**: safe today (DC-only, type-disjoint); tagged tuple + loud failure
  on unexpected content protects against future drift.
- **Suite integration friction**: FD in `AC_SOLVERS_TO_TEST` will hit tests tuned for quadratic
  solvers (iteration caps, timing). Budget real time in WP6 for the audit; prefer parametric
  fixes over skips.
- **LM handoff signature mismatch** (`LMWorkspace`-based): deferred behind validation rather than
  hacked in v1.
- **Unverifiable-here PNM internals**: arc-admittance field semantics are evidenced only by
  in-repo usage (§2 citations); WP1's first task is to confirm against instantiated PNM source
  and adjust recovery formulas if the stamp convention differs (e.g. which side carries τ).
- **Scope discipline**: commercial FDNS also bundles tap/switched-shunt/area-interchange
  adjustments — explicitly **out of scope** here (as for all existing solvers); remote-bus
  voltage control likewise (Sienna models PV locally). Listed in docs as future work.
