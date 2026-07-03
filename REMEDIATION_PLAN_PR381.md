# Remediation Plan — PR #381 Discrete Reactive-Power Controls

> **Status (implemented on this branch):** PR-A ✅ (`831652c`), PR-B ✅ (`bacae62`),
> PR-C ✅ (`ef41bd8`), PR-D partial — D-0 harness, D-1 full-step-first, D-3 tolerance
> ladder landed; **D-2 (batched passes) deferred**: it interacts with the new secant
> gain tracking (joint Δy attribution corrupts per-device gains on coupled buses) and
> should be designed against CI-measured baselines from the D-0 harness. Single-solve
> probes (from PR-C's full-state restore) already halved probe cost. All changes are
> parse-checked only — this environment cannot install deps; CI must validate, and the
> formatter must be run once an environment allows it.

Companion to `REVIEW_PR381.md` (finding IDs reference its sections). Guiding principles:

1. **Simplest sound fix wins.** Where a guard/rejection resolves a finding as safely as new machinery, use the guard. Machinery must earn its way in with a measurement or a failing test.
2. **Restrict scope to what works end-to-end.** Better to ship taps + shunts that are production-solid than four device families with known-broken data sourcing.
3. **One behavior change per PR.** Four small PRs, each independently green and reviewable, instead of one omnibus.
4. **Every fix lands with the test that would have caught it.**

**Explicit non-goals (rejected as over-engineering for this effort):**
- No implicit sigmoid embedding in the Jacobian (`stamp_control!` stays a seam).
- No per-time-step device-state tracking — multi-period is *rejected at construction* instead (revisit when PSI actually needs it).
- No FastDecoupled cache-invalidation framework — the combination is *rejected at construction* instead.
- No RMPCT proportional-sharing controller — ω is divided among co-located devices and the deviation from PSS/E is documented.
- No adjoint-sensitivity plumbing yet — cheaper fixes first; add adjoints only if the post-fix probe cost still shows up in measurements.
- No persistent cross-solve residual/Jacobian cache in this effort (that is the future `PolarNRCache` work; leave an invalidation warning comment for it).
- No generic "multi-quantity steepness framework" — one per-device scalar normalization.

---

## PR-A — Make it run, shrink the surface (unblocks CI; ~1–2 days)

Goal: the package loads and the full test suite executes on released PSY; everything not production-ready is fenced off *loudly*.

| # | Fix | Finding | Why this and not more |
|---|-----|---------|----------------------|
| A-1 | Source tap metadata from the parser's real ext keys (`CONT1`, `COD1`, `RMI1`, `RMA1`, `NTP1`) via one small `_tap_metadata(tx)` helper; delete the calls to unreleased PSY accessors. Update test fixtures to populate `ext` instead of PSY-6 kwargs. | A1, D2 | Fixes the compat break **and** the wrong-key finding in one move. The helper isolates sourcing so the PSY-6 switch later is a one-function change. No `hasmethod` feature-detection — fragile and harder to read than a single sourcing function. |
| A-2 | Reject at `PowerFlowData` construction, with clear error messages: `time_steps > 1` + controls; LCC systems + controls; controls with any solver other than NR/TR (matches the repo's existing illegal-pair convention). | A4, M18, M20 | A guard is 10 lines and fully safe; per-ts device state, LCC snapshot columns, and FD cache invalidation are real machinery for combinations nobody has validated. Revisit each only when a user needs it. |
| A-3 | **Scope decision (recommended):** keep `ControlledTap` + `ControlledSwitchedShunt` production-enabled; gate `ControlledFACTS` + `ControlledPhaseShifter` behind `solver_settings[:experimental_controls] = true`, default off. | M8, D3 | The PAR's setpoint source (`get_active_power_flow` → 0.0 for raw-parsed) and angle band don't exist in parsed data yet, and its steepness scaling needs the Phase-C normalization. Fencing it is honest; building workarounds for missing upstream fields is exactly the over-engineering to avoid. FACTS is closer to ready but shares the untested probe path — keep them together behind the flag. |
| A-4 | Builder error posture → warn-and-skip-as-locked for per-device data problems (MODSW ≥ 3, block-less shunts, SHMX ≤ 0, unresolvable/reduced buses, NTP < 2). Resolve buses through the same `_get_bus_ix`/`reverse_bus_search_map` helper every other init path uses; one policy for all four families. Delete the `ntp < 2 → 33` pre-substitution. | M19, D6 | Matches PSS/E behavior and existing repo helpers — this is consistency, not new machinery. `error()` stays only for programmer errors. |
| A-5 | One-line numerics fixes: `alpha = PSY.get_α(tx)` at tap enrollment; CHOLMOD `@cholmod_param final_ll = true` around `numeric_factor!`; vset plausibility guard (warn+skip outside [0.5, 1.5] p.u.); shunt range from blocks alone (`b_min = Σ min(steps·dB,0)`, `b_max = Σ max(…,0)`, clamp `current` into it). | A5, F1, M12/D4, M9/D1 | Each is a few lines with an existing verified fix. |
| A-6 | Dead-code deletion: `_seam_err` stub method, `target_from_voltage` + `controlled_on_primary` + their testset, `INITIAL_LAMBDA_STEP`, `DEFAULT_TAP_RATIO_MIN/MAX`, the unreachable `CONTROL_RELAXATION_MAX` cap. Truth-up `discrete_control.md` (implemented device list per A-3 scoping, dydp-branch instead of "flip flag", real metadata tables, cumulative-counter semantics, scoped allocation claim). | G, docs | Deleting is the simplification. Docs must describe the code that ships. |

**Tests added in PR-A:** package loads + suite runs on compat floor (this is what CI now proves); construction-rejection tests for each guard; ext-key sourcing test (CONT1 remote bus, RMI1/RMA1/NTP1); warn-and-skip tests for each locked-device case.

---

## PR-B — Results are right (the user-visible blockers; ~2–3 days)

| # | Fix | Finding | Why this and not more |
|---|-----|---------|----------------------|
| B-1 | After the continuation converges for the time step (not on every apply), rewrite the `arc_admittance_from_to/to_from` entries for each *moved* device from its final parameters, before flows are computed. Cache those nzval offsets at device-set construction next to `nz_offsets`. | A2 | End-of-solve fixup is ~20 lines, off the hot path, and trivially testable. Per-apply updates or LCC-style bespoke flow recomputation add invariants for no benefit. |
| B-2 | Expose solved settings: `get_controlled_device_results(data)` returning name/type/initial/final parameter (+ tap level index / shunt block counts), and a `@warn` in the PSS/E exporter when controls were active ("exported device settings are pre-control"). | A3 | The accessor is the minimal honest surface. Writing back into the user's PSY `System` is a *behavioral decision* (mutating input data) — proposed as an opt-in `write_device_settings=true` on `solve_and_store_power_flow!` in PR-D, after the team confirms the semantics. Don't silently mutate now. |
| B-3 | Float64 shadow for mutated Y-bus slots: at construction store `other[k] = ComplexF64(nzval[oₖ]) − device_term(current)`; `apply_parameter!` becomes an absolute rewrite `nzval[oₖ] = ComplexF32(other[k] + device_term(p))`. | G (F32 drift) | ~10 lines in the same function as A5's alpha fix; kills the drift class *and* makes probe apply/revert exactly idempotent (needed by PR-C's rollback tests). The "periodic recompute" alternative is vaguer and needs a schedule. |
| B-4 | Snap-path correctness: snapshot voltages before snapping and restore on post-snap failure; replace the name-keyed `Dict` with per-group index-aligned `Vector{Float64}`; clamp `p` in `_restore_one!`. | M16, M17 | The vectors are *less* code than the Dict and remove both the collision and the allocation. |
| B-5 | Shunt snap → nearest point of the PSS/E cumulative prefix-sum chain (precomputed at construction, reuse `block_order` storage). Delete the greedy + ±1 refinement. | M7 | The chain is simpler than what it replaces, optimal over the realizable set, and order-respecting. Not an optimization — a correction. |

**Tests added in PR-B:** flow-parity test (controlled solve vs fresh solve of a system rebuilt at the snapped parameters — would have caught A2); settings-results accessor test; drift idempotency test (apply/revert N times → bitwise-stable nzval); snap chain vs brute force on mixed-sign banks; snap-failure fixture exercising `_restore_one!`.

---

## PR-C — Engine robustness (the math findings; ~3–4 days)

Design rule for this PR: **reuse the existing freeze mechanism for every new failure mode** (one escape hatch, not four), and prefer information the loop already has over new probes.

| # | Fix | Finding | Why this and not more |
|---|-----|---------|----------------------|
| C-1 | Extend the snapshot tuple to five columns: `bus_magnitude`, `bus_angles`, `bus_type`, `bus_active_power_injections`, `bus_reactive_power_injections`. Restore all five on any rollback. | M14, M15 | One tuple, one restore function — fixes sticky PV→PQ flips *and* distributed-slack setpoint drift together. No selective/partial restore logic. |
| C-2 | Secant gain refresh: after each device step, update `dVdp[idx] = Δy_observed/Δp_applied` (guarded by `|Δp| > tol`); if the sign flips vs the probe → freeze + warn. | M1 | Zero extra solves — reuses numbers the step just produced. This replaces the need for per-stage re-probing or adjoint sensitivities. |
| C-3 | Reliability floor: `|dVdp| < CONTROL_GAIN_FLOOR` (or `|y1−y0|` below a noise floor) ⇒ unreliable ⇒ freeze + existing warn path. Applies at probe time and at secant refresh. | M2, G (SNR) | One constant + one branch; routes into the machinery that already exists. Don't skip-by-bus-type (a PV bus can flip PQ later — the floor handles both causes). |
| C-4 | Scheduler repairs: (a) oscillation-frozen devices return 0.0 (settled) like probe-frozen ones; (b) reset `osc[idx]` on each steepness ramp; (c) ignore reversals with `|p_tgt − p_now| < tol`; (d) per-stage pass budget (`MAX_PASSES_PER_STAGE ≈ 20`) replacing the global 100; (e) `_continuation_to!` returns `(reached, moved)` and a requested-but-unapplied move ⇒ freeze + warn instead of counting as settled. | M4, M5, M6 | All five are edits inside the existing loop; no scheduler redesign. Together they make `regulation_complete` mean what it says. |
| C-5 | Scale-aware settle tolerance: `tol_d = max(CONTROL_PARAM_TOL, 1e-4·(hi−lo))` via one helper used by settle, `_continuation_to!`, and `_restore_one!`. | M4 (tol) | One function, three call sites. |
| C-6 | Co-located damping: at enrollment count devices per `controlled_ix`; divide ω by that count. Document that RMPCT-proportional sharing is intentionally not implemented. | M3, M13 (RMPCT) | An integer per device vs a coupling-matrix probe. The freeze path (C-2/C-3) remains the backstop for residual coupling. |
| C-7 | Deadband: shunts (and FACTS when re-enabled) get `vset_lo/vset_hi` from the parsed band; a device inside its band is not stepped (error = distance to nearest band edge otherwise). Taps keep midpoint targeting (PSS/E taps regulate to a band too — use the same guard if VMI1/VMA1 becomes available; until then midpoint is the only data we have). | M13/D5 | A 3-line guard, not a new control law: the sigmoid machinery is untouched, it just isn't invoked for in-band devices. Eliminates the over-switching class and most of the multi-controller fighting. |
| C-8 | Per-device steepness normalization (`S_eff = S / (|dVdp|·(hi−lo))`), landed with the flag-gated PAR/FACTS re-enable, plus the PAR shunt-conductance term and a real setpoint source (explicit user setpoint required until PSY persists the PSS/E flow window — enrolling with `p_target = 0.0` stays forbidden). | M8, D3, G | This is the gate for taking PAR/FACTS out from behind the experimental flag — not before. |

**Tests added in PR-C:** `@test_logs` for every warn (oscillation, unreliable probe, stuck device, per-stage budget); a coupled two-taps-one-bus fixture that must now converge; a PV-pinned controlled-bus fixture that must freeze (not rail); a fixture with a Q-limit-marginal generator (the Q-limit loop finally activates inside a control test); distributed-slack + controls regression (setpoints unchanged after a forced failed trial).

---

## PR-D — Performance (measure first, then three fixes; ~2–3 days)

Precondition: an **iteration-count harness** (count `_solve_with_q_limits!` calls per device per ts — counts, not wall-clock, per repo convention) landed first, so every claim below is measured.

| # | Fix | Finding | Expected effect | Why this and not more |
|---|-----|---------|-----------------|----------------------|
| D-1 | Full-step-first in `_continuation_to!`: try the whole relaxed move; halve on failure (keep `MIN_LAMBDA_STEP` as the give-up floor). | E | 16 → 1–2 solves per move (~10×) | Inverts one loop; the backtracking safety net is unchanged. |
| D-2 | Batch the pass: compute all device targets, apply all updates, **one** solve per pass; on failure, roll back all and halve a global pass-relaxation once before falling back to per-device stepping for that pass. | E | ~N× on pass cost | This is *less* total code than N sequential walks and matches the damped-joint-update picture the ω theory already assumes. The per-device path remains as the fallback, not a parallel implementation. |
| D-3 | Tolerance ladder: inner solves at `1e-6` until the final steepness stage, full tolerance for the last stage + snap/restore. | E | 1.5–2× | One kwarg threaded to the inner solve. |
| D-4 | Opt-in write-back `write_device_settings=true` in `solve_and_store_power_flow!` (from B-2's results surface), if the team confirms mutate-on-store semantics. | A3 follow-up | — | Deliberately last: it's a semantics decision, not a bug fix. |

**Explicitly deferred, with re-entry criteria:** adjoint probe sensitivities (only if the harness shows probe cost > ~20 % after D-1/D-2 — with batching, probes are 2N solves *once* per ts and likely fine); persistent residual/Jacobian/symbolic reuse across inner solves (belongs to the future `PolarNRCache` work — add the comment now that such a cache must invalidate constant-Z snapshots on `apply_parameter!`); structure-cache REF fingerprint (tiny, ride along with any PR touching that file).

---

## Sequencing and ownership of open decisions

```
PR-A (unblock, fence)  →  PR-B (results right)  →  PR-C (engine robust)  →  PR-D (fast)
      CI green              flows/settings            math holds              claim earned
```

Decisions needed from the author/team (each has a recommended default that the plan assumes):
1. **PAR/FACTS scoping** — recommended: experimental flag until C-8's gate is met (A-3).
2. **Write-back semantics** — recommended: results accessor now, opt-in mutation in PR-D (B-2/D-4).
3. **Deadband for shunts** — recommended: yes, it's the PSS/E behavior and a 3-line guard (C-7).
4. **PSY 6 timing** — A-1's ext-sourcing works today; when PSY ships #1705, swap `_tap_metadata` internals and bump compat in one small PR.

Rough total: ~8–12 working days across four reviewable PRs, each leaving `main` shippable.
