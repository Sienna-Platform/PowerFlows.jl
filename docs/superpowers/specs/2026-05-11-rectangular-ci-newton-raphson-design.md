# Rectangular Current-Injection Newton-Raphson AC Power Flow — Design

**Date:** 2026-05-11
**Branch:** `jd/complex_nr_solver`
**Status:** Approved for implementation
**Authors:** brainstorm with Jose Daniel Lara

## 1. Motivation

PowerWorld's default AC power flow solver is named "Rectangular Newton-Raphson" and is the kernel of their Robust Solution Process. The marketing claim is better recovery from bad initial guesses; the technical foundation (from the Da Costa / Pereira / Garcia IEEE-TPS line of papers) is two-fold:

1. **Polynomial residual and Jacobian entries** — no trigonometric functions; smoother behavior near flat start where `cos θ → 1, sin θ → 0` flattens the polar Jacobian's angle-sensitivity.
2. **Constant off-diagonal Jacobian blocks ≡ Y_bus blocks** — only diagonal blocks update between Newton iterations. Per-iteration Jacobian update cost drops from `O(nnz(Y_bus))` to `O(N + n_LCC)`, which is the headline performance win for large grids (EI 50k+, ACTIVSg2000, WECC).

This work adds a parallel `RectangularCurrentInjectionACPowerFlow` solver type to `ACPowerFlow{<:ACPowerFlowSolverType}`, joining the existing 5 solvers. The polar Newton-Raphson remains the default.

## 2. Formulation

**Augmented current-injection** (Da Costa et al., IEEE TPS 2000):

For each bus `i`, `V_i = e_i + j f_i`, `|V_i|² = e_i² + f_i²`. Y_bus entries `Y_ij = G_ij + j B_ij`.

**Y_bus current injection** (linear in `(e, f)`):

```
I_inj,r,i = Σ_j (G_ij·e_j − B_ij·f_j)
I_inj,i,i = Σ_j (G_ij·f_j + B_ij·e_j)
```

The 2×2 partial `∂(I_inj,r, I_inj,i)/∂(e_j, f_j) = [[G_ij, −B_ij], [B_ij, G_ij]]` is constant.

**Specified current** from `S_i = P_i + j Q_i`:

```
I_spec,r,i = (P_i·e_i + Q_i·f_i) / |V_i|²
I_spec,i,i = (P_i·f_i − Q_i·e_i) / |V_i|²
```

**Residual** `ΔI_i = I_spec,i − I_inj,i` (split into real and imaginary parts).

### Per-bus-type residuals

| Bus type | Vars | Equations |
|---|---|---|
| PQ | (e, f) | ΔI_r, ΔI_i (P, Q known) |
| PV | (e, f, Q) | ΔI_r, ΔI_i, ΔV² = V_set² − (e² + f²) |
| REF | (P_gen, Q_gen); (e, f) fixed | ΔI_r, ΔI_i (both linear in P_gen, Q_gen) |

### ZIP loads

Constant-Z component folded into `Y_bus_eff` once at setup as fixed shunt admittance `(β_P − j·β_Q)/V₀²` on the diagonal. Constant-current component is retained as a `|V|`-dependent correction in the residual:

```
I_spec,r ⊕= −(α_P·e + α_Q·f)/|V|
I_spec,i ⊕= −(α_P·f − α_Q·e)/|V|
```

Constant-power component is the `(P_const, Q_const)` baseline.

### Distributed slack — mirrors today's polar convention

`x[bus_state_offset[ref]]` carries total `P_gen` at the REF bus including the slack increment. At participating bus `k` (`c_k > 0`):

```
P_eff(k) = P_net_set[k] + c_k · (x[offset[ref]] − P_net_set[ref])
```

Jacobian cross-term:

```
∂F_k,r / ∂x[offset[ref]] = c_k · e_k / |V_k|²
∂F_k,i / ∂x[offset[ref]] = c_k · f_k / |V_k|²
```

Rectangular analogue of today's `−c_k`.

### LCC HVDC

**Mechanism 1 stays.** `_update_ybus_lcc!` writes state-dependent LCC self-admittance `Y_lcc(t, α, |V|)` into Y_bus diagonals at the rectifier and inverter buses. No new Jacobian sparsity needed — the LCC contribution flows through the existing PQ/PV diagonal-block update path.

**LCC tail.** 4 state variables per LCC at the tail (`t_r, t_i, α_r, α_i`); 4 residual equations (P balance at rectifier, DC line P balance, two `α`-at-min constraints). Formulas reuse polar today; column indices translate `Vm slot → e slot`, `Va slot → f slot`. Derivative chain: `∂/∂e = (∂/∂Vm) · e/|V|`, `∂/∂f = (∂/∂Vm) · f/|V|`. The existing `_calculate_dQ_d{V,t,α}_lcc` helpers are reused verbatim at the call site.

## 3. State vector layout

Per-bus variable-size blocks. PQ/REF blocks are 2; PV blocks are 3. LCCs at the tail.

```
x = [ block_1 | block_2 | ... | block_N | lcc_tail ]
```

Two new lookup arrays (computed once at solver construction, recomputed only on PV↔PQ switching):

```julia
const REC_INDEX_TYPE = Int32
bus_state_offset::Vector{REC_INDEX_TYPE}   # length N+1, cumulative
bus_block_size::Vector{Int8}               # length N, ∈ {2, 3}
```

Indexing: `x[bus_state_offset[i]]` is bus `i`'s first var (`e_i` for PQ/PV, `P_gen` for REF). `x[bus_state_offset[i]+1]` is the second var. `x[bus_state_offset[i]+2]` is `Q_i` (PV only). LCC tail starts at `bus_state_offset[N+1]`.

## 4. Jacobian structure

### Sparsity pattern (built once at construction)

- Off-diagonal 2×2 (or 2×3 / 3×2 / 3×3) blocks at each `Y_bus[i, j]` nonzero. The `Q`-slot columns of PV neighbors have **structural zeros** for off-diagonal positions (current injection from neighbor doesn't depend on neighbor's `Q`).
- Diagonal block: 2×2 for PQ/REF, 3×3 for PV. The `∂ΔV²/∂Q = 0` entry of PV's 3×3 is kept as a structural zero so the pattern is stable across PV↔PQ switching.
- Distributed-slack cross-terms: one 2×1 column at `x[offset[ref]]` for each participating bus `k`.
- LCC tail: 17 entries per LCC (same as today's polar structure, indices translated).

Total `nnz ≈ 4·nnz(Y_bus) + 5·n_PV + 2·n_slack_participants + 17·n_LCC` — same order as today's polar Jacobian.

### Numerical update (per NR iteration)

- **Off-diagonal blocks: untouched.** Populated once at construction from `Y_bus_eff`.
- **REF diagonal blocks: untouched.** Populated once; both diagonal 2×2 (constant) and off-diagonal Y-blocks involving REF are constant.
- **PQ diagonal block (2×2):** state-dependent in `(e, f)`, written from cached `P_eff, Q_eff` and `|V|²`.
- **PV diagonal block (3×3):** state-dependent in `(e, f, Q)`, written from cached `P_eff` and current state.
- **Slack cross-terms:** state-dependent in `e_k, f_k, |V_k|²`.
- **LCC tail entries (17 per LCC):** state-dependent in `(t, α, |V_fb|, |V_tb|, I_dc, phi)`; reuses existing helpers via chain rule.

Per-iteration cost: `O(N + n_LCC)` writes. No Y_bus traversal.

## 5. PV → PQ switching

Outer-loop only (`_check_q_limit_bounds!`). On switch:

1. Save current `(e_i, f_i)` for all buses + LCC tail values
2. Recompute `bus_state_offset, bus_block_size` from updated `bus_type` (downstream offsets shift by −1)
3. Reconstruct residual + Jacobian structures (fast one-shot pattern build)
4. Repopulate `x` from saved values
5. Re-enter `_newton_power_flow`

Mirrors today's `_ac_power_flow` outer-loop flow exactly. The polar code calls `initialize_power_flow_variables` on each iteration of the Q-limit loop, and we follow the same pattern.

## 6. Integration with existing scaffolding

- `KLULinSolveCache{J_INDEX_TYPE}` — unchanged (Float64 KLU).
- `StateVectorCache` — unchanged.
- `_set_Δx_nr!`, `_do_refinement!`, `_dogleg!`, `_iwamoto_multiplier`, `_finalize_power_flow` — unchanged; they only operate on residuals and Jacobian sparse matrices.
- `_simple_step`, `_iwamoto_step`, `_trust_region_step` — generic over the residual/Jacobian functors. They work unmodified.
- `_run_power_flow_method` — dispatch on step strategy (`NewtonRaphsonACPowerFlow` vs `TrustRegionACPowerFlow`) stays. Dispatched orthogonally to formulation.
- `_newton_power_flow` — new dispatch for `ACPowerFlow{RectangularCurrentInjectionACPowerFlow}` that builds rectangular residual/Jacobian.

**Step-strategy split:** today, the step strategy (plain NR vs Iwamoto vs Trust Region) is selected by the solver type plus the `:iwamoto` flag. We preserve that convention. To run rectangular CI with Iwamoto: `ACPowerFlow{RectangularCurrentInjectionACPowerFlow}(; solver_settings = Dict(:iwamoto => true))`. To run with Trust Region we'll add (in a follow-up) a `:step_strategy => :trust_region` settings flag rather than a 2× solver-type explosion.

## 7. File layout

New files:

```
src/rectangular_ci_power_flow_residual.jl
src/rectangular_ci_power_flow_jacobian.jl
src/rectangular_ci_lcc.jl
src/state_indexing_helpers.jl                 [+ rect_* helpers]
src/power_flow_types.jl                       [+ RectangularCurrentInjectionACPowerFlow]
src/power_flow_method.jl                      [+ _newton_power_flow dispatch]
src/PowerFlows.jl                             [+ exports + include order]
```

```
test/test_rectangular_ci_residual.jl
test/test_rectangular_ci_jacobian.jl
test/test_rectangular_ci_power_flow.jl
test/test_rectangular_ci_lcc.jl
test/performance/performance_test.jl          [+ rectangular CI benchmarks]
test/test_distributed_slack.jl                [extended]
test/test_multiperiod_ac_power_flow.jl        [extended]
test/test_iterative_methods.jl                [extended]
test/test_hvdc.jl                             [extended]
```

## 8. Validation strategy

Per [[mirror-existing-for-validation]]: every polar NR test gets a sibling rectangular-CI test using the same fixtures and tolerances.

**Parity tests:** for each fixture, run both solvers and assert voltage / angle agreement within 1e-7, branch flow agreement within 1e-6, iteration count within ±3.

**Finite-difference Jacobian:** for each bus type and a synthetic LCC, sample 50 random states near nominal, compare analytic Jacobian to `(F(x+ε·eᵢ) − F(x−ε·eᵢ))/(2ε)` per entry to tolerance 1e-5. Mirrors `test_jacobian.jl`.

**Edge cases:** singular Jacobian fallback, NaN initial guess + `enhanced_flat_start`, Q-limit cycling, multi-subnetwork (islanded), Q at exactly min/max, empty system.

## 9. Performance benchmarking

Extend `test/performance/performance_test.jl` to run both polar NR and rectangular CI on:

- RTS-GMLC (~70 buses)
- IEEE 14, 30, 118, 300
- ACTIVSg2000
- WECC (when available)
- EI subset (the target)

Tracked: time per NR iteration (residual + Jacobian update + KLU solve), allocations per iteration (target: 0), total iterations, convergence success on hard fixtures.

## 10. Rollout (7 PRs, each mergeable independently)

```
PR 1: state offsets, types, ZIP-Z fold helper          [pure infra, no solver]
PR 2: ACRectangularCIResidual + tests                  [residual only]
PR 3: ACRectangularCIJacobian (no LCC tail) + tests    [structure + FD]
PR 4: LCC tail integration                             [test_hvdc parity]
PR 5: _newton_power_flow wiring + solver type          [full parity suite]
PR 6: Iwamoto + TR wrappers on rectangular CI          [iterative methods]
PR 7: benchmarks + docs                                [performance + Diataxis docs]
```

Each PR preserves polar NR as default. Each adds tests mirroring existing tests.

## 11. Open questions resolved during brainstorm

- ✅ Formulation: augmented current-injection (Da Costa).
- ✅ State ordering: per-bus variable blocks (PQ/REF 2-block, PV 3-block).
- ✅ ZIP-Z: folded into `Y_bus_eff` at setup.
- ✅ LCC mechanism 1 (Y_bus diagonal): same path as today.
- ✅ Distributed slack: REF-slot convention same as polar.
- ✅ Day-1 parity: LCC, distributed slack, ZIP, Q-limit switching, Iwamoto + TR wrappers.
- ✅ REF block: populate once, skip in per-iteration update.
- ✅ Structural-zero `∂ΔV²/∂Q`: kept in pattern for stable PV↔PQ.
- ✅ Validation principle: mirror existing implementation; iteration-count parity ±3.

## 12. Risk register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| LCC chain-rule derivatives wrong | medium | high | FD Jacobian tests with synthetic LCC + `test_hvdc.jl` parity |
| Iwamoto cubic-fit ill-behaved on rectangular Jacobian | medium | medium | Iteration-count tracking; ship plain NR first if needed |
| TR autoscale tuned for polar scales | medium | medium | Same as above; expose rectangular-tuned defaults if needed |
| Q-limit block-resize bug | low | high | Outer-loop rebuild mirrors today; explicit roundtrip test |
| ZIP-Z fold affects loss-factor / voltage-stability post-processing | low | medium | Parity tests on `test_loss_factors.jl` |
| Performance regression on small grids | low | low | RTS benchmark; `@inbounds` on offset lookup if needed |
| Distributed slack sign / convention mismatch | low | high | Unit test slack explicitly; FD verify |
| PV→PQ rebuild blowup on Q-limit-heavy cases | low | medium | Cache offsets where possible; benchmark on cycling cases |

## 13. Out of scope (follow-ups)

- Robust Homotopy variant of rectangular CI — needs rectangular-aware Hessian, separate design.
- Levenberg-Marquardt variant — driver lives in `levenberg-marquardt.jl`, separate wiring.
- Gradient Descent variant — same situation.
- Complex KLU substrate — rejected; Float64 KLU with `(e, f)` pairs is the chosen path.
- PSI integration testing — should work transparently; single smoke test in PR 7.

## 14. References

- [Da Costa, Pereira, Garcia — Developments in NR power flow formulation based on current injections, IEEE TPS, 2000](https://ieeexplore.ieee.org/document/801891/)
- [Augmented Newton–Raphson power flow based on current injections, EPSR 2001](https://www.sciencedirect.com/science/article/abs/pii/S0142061500000454)
- [Garcia et al., Three-phase power flow using current injection, IEEE TPS, 2000](https://www.researchgate.net/publication/3266183_Three-phase_power_flow_calculations_using_the_current_injection_method)
- [On a comparison of Newton-Raphson solvers for power flow problems, JCAM, 2019](https://www.sciencedirect.com/science/article/pii/S0377042719301876)
- PowerWorld — Weber, "Techniques for Conditioning Hard-to-Solve Cases" (loaded from `~/Downloads/WeberHardToSolveCases.pdf`).
- PowerWorld — S3, "Steady-State Power System Security Analysis with PowerWorld Simulator" (loaded from `~/Downloads/S03ConditioningHardToSolve.pdf`).
- Gómez-Expósito & Alvarado, *Electric Energy Systems: Analysis and Operation* — referenced in existing PowerFlows.jl Jacobian ordering rationale.
