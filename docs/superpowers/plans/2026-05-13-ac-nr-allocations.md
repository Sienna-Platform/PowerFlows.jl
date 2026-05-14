# AC Newton-Raphson Allocation Reduction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Cut per-AC-NR-solve allocations from ~30 MB to under 10 MB on the 10k-bus benchmark, eliminating the GC-pause-driven flakiness that the perf-test single-shot `@timed` measurements are exposing.

**Architecture:** Identify hot-path allocations in `_update_residual_values!`, `_do_refinement!`, and the post-solve flow-computation block, then replace each with preallocated buffers in `ACPowerFlowResidual` (extended struct fields) or in-place `mul!` calls. The AC NR Jacobian update and KLU solve are already non-allocating; the residual is the dominant remaining source.

**Tech Stack:** Julia 1.10+, SparseArrays, LinearAlgebra (in-place `mul!`), PowerFlows internal types (`ACPowerFlowResidual`, `StateVectorCache`, `KLULinSolveCache`).

**Baseline measurements (matpower_ACTIVSg10k_sys, this branch, Apple Silicon):**
- `solve_power_flow!` whole call: **30.61 MB** / call
- `residual(x, time_step)` one call: **689 KB** / call (the main per-iteration cost)
- `J(time_step)` one call: 80 B / call ✓ already lean
- `solve!(cache, b)` in-place: 0 B / call ✓ already lean
- `A * Δx_nr` (inside `_do_refinement!`): **160 KB** / call (one fix away)

**Top per-call sites identified by `Profile.Allocs.@profile sample_rate=1.0`** (filenames given relative to `src/`):
| Site | Allocs / call | Hypothesis |
|---|---:|---|
| `ac_power_flow_residual.jl:282-285` (P_slack) | ~160 KB | `sv[idx]` + scalar `.*` vec → two fresh Vectors |
| `ac_power_flow_residual.jl:286-287` (bus_types[idx]) | ~40 KB | view indexed by Vector returns a fresh Vector |
| `ac_power_flow_residual.jl:378-379` (F slice .-=) | ~160 KB | strided LHS `.-=` allocates a copy on read |
| `power_flow_method.jl:55` (A * Δx_nr) | ~160 KB | non-`mul!` matrix-vector product |
| `solve_ac_power_flow.jl:223-228` (post-solve Sft/Stf) | ~1.2 MB | once per `solve_power_flow!`; lower priority |

---

## File Structure

**Modify:**
- `src/ac_power_flow_residual.jl` — add buffer fields to `ACPowerFlowResidual`; rewrite `_update_residual_values!` to use them.
- `src/power_flow_method.jl` — replace `A * Δx_nr` with `mul!` in `_do_refinement!`.
- `src/solve_ac_power_flow.jl` — optional later pass on post-solve flow computation.

**Create (test):**
- `test/test_ac_nr_allocations.jl` — `@allocated` regression tests that pin upper bounds on hot-path allocations.

**Modify (test):**
- `test/runtests.jl` — include the new test file.

No new public types or exports. The `ACPowerFlowResidual` struct gains internal scratch fields; the constructor signature stays unchanged.

---

## Task 1: Failing allocation regression test

Establish the test that proves the problem before fixing anything.

**Files:**
- Create: `test/test_ac_nr_allocations.jl`
- Modify: `test/runtests.jl` (add `include("test_ac_nr_allocations.jl")` in the appropriate testset group)

- [ ] **Step 1: Find where other tests are included in `runtests.jl`**

Run: `grep -n 'test_jacobian\|test_solve_power_flow' /Users/jlara/.julia/dev/PowerFlows/test/runtests.jl`

Note the include style and surrounding `@testset` block.

- [ ] **Step 2: Write the new test file**

Create `/Users/jlara/.julia/dev/PowerFlows/test/test_ac_nr_allocations.jl`:

```julia
@testset "AC NR allocation regression" begin
    # Use a system that's big enough to expose hot-path allocations but
    # small enough to run quickly in CI.
    sys = build_system(MatpowerTestSystems, "matpower_ACTIVSg2000_sys")
    pf = ACPowerFlow{PF.NewtonRaphsonACPowerFlow}(; correct_bustypes = true)
    pf_data = PF.PowerFlowData(pf, sys)
    residual = PF.ACPowerFlowResidual(pf_data, 1)
    J = PF.ACPowerFlowJacobian(
        pf_data,
        residual.bus_slack_participation_factors,
        residual.subnetworks,
        1,
    )
    x0 = PF.calculate_x0(pf_data, 1)
    residual(x0, 1)    # warm
    J(1)               # warm

    # --- per-call upper bounds (chosen ~2x current best-case after fixes) ---
    @test (@allocated residual(x0, 1)) < 2_000     # was ~140 KB on 2000-bus
    @test (@allocated J(1)) < 200                  # already ~80 B, leave headroom

    # --- _do_refinement! mul! path ---
    cache = PF.KLULinSolveCache(J.Jv)
    PF.full_factor!(cache, J.Jv)
    b = randn(size(J.Jv, 1))
    x = copy(b); PF.solve!(cache, x)
    out = similar(b)
    LinearAlgebra.mul!(out, J.Jv, x)  # warm
    @test (@allocated LinearAlgebra.mul!(out, J.Jv, x)) == 0
end
```

- [ ] **Step 3: Wire the test into `runtests.jl`**

Add `include("test_ac_nr_allocations.jl")` next to the other AC tests, inside the same testset block.

- [ ] **Step 4: Run the test, confirm it fails**

Run: `julia --project=test -e 'using ReTest; include("test/runtests.jl"); retest("AC NR allocation")'`

Expected: FAIL on the residual `@allocated` line (current value ~140 KB on 2000-bus, ~689 KB on 10k-bus). The `mul!` line passes already; J(1) passes already.

- [ ] **Step 5: Commit the failing test**

```bash
git add test/test_ac_nr_allocations.jl test/runtests.jl
git commit -m "test: AC NR allocation regression upper bounds (currently failing on residual)"
```

> ⚠️ **DO NOT COMMIT** unless the user explicitly asks — per project policy, stage only. Use `git add` then stop.

---

## Task 2: Preallocate the P_slack buffer in the residual

The biggest single hot-path allocation: `(scalar) .* bus_slack_participation_factors[subnetwork_buses]` allocates a fresh `Vector{Float64}` of length up to `n_buses` per call per subnetwork.

**Files:**
- Modify: `src/ac_power_flow_residual.jl` (struct definition lines 16-25, constructor lines 67-77, hot loop lines 281-328)

- [ ] **Step 1: Read current struct to confirm field order**

Read `/Users/jlara/.julia/dev/PowerFlows/src/ac_power_flow_residual.jl` lines 16-77 and reconfirm. Then read lines 275-330 to see the hot loop.

- [ ] **Step 2: Add `P_slack` scratch field to `ACPowerFlowResidual`**

In `src/ac_power_flow_residual.jl`, replace the struct definition (lines 16-25):

```julia
struct ACPowerFlowResidual
    data::ACPowerFlowData
    Rf!::Function
    Rv::Vector{Float64}
    P_net::Vector{Float64}
    Q_net::Vector{Float64}
    P_net_set::Vector{Float64}
    bus_slack_participation_factors::SparseVector{Float64, Int}
    subnetworks::Dict{Int64, Vector{Int64}}
    # Scratch buffer for the per-subnetwork slack distribution vector.
    # Sized to n_buses so the largest subnetwork fits without reallocation.
    P_slack_buf::Vector{Float64}
end
```

- [ ] **Step 3: Initialize the buffer in the constructor**

Edit the constructor's `return ACPowerFlowResidual(...)` call (around line 67) to add the new field:

```julia
    return ACPowerFlowResidual(
        data,
        _update_residual_values!,
        Vector{Float64}(undef, 2 * n_buses + 4 * n_lccs),
        P_net,
        Q_net,
        P_net_set,
        bus_slack_participation_factors,
        subnetworks,
        Vector{Float64}(undef, n_buses),
    )
```

- [ ] **Step 4: Compile-check**

Run: `julia --project -e 'using PowerFlows'`

Expected: no errors. Fix immediately if there are any.

- [ ] **Step 5: Rewrite the P_slack inner loop to use the scratch buffer**

In `src/ac_power_flow_residual.jl`, the hot loop currently reads (lines 281-287):

```julia
    for (ref_bus, subnetwork_buses) in subnetworks
        P_slack =
            (x[2 * ref_bus - 1] - P_net_set[ref_bus]) .*
            bus_slack_participation_factors[subnetwork_buses]

        for (ix, bt, p_bus_slack) in
            zip(subnetwork_buses, bus_types[subnetwork_buses], P_slack)
```

The function signature already passes `data::ACPowerFlowData` and `bus_slack_participation_factors::SparseVector{Float64, Int}` as arguments. We can't reach `P_slack_buf` from `_update_residual_values!` directly because it's called via the `Rf!` function pointer with those positional args. Two options:

**Option A (chosen):** Add a `P_slack_buf::Vector{Float64}` positional argument to `_update_residual_values!` and update the functor call site to pass `Residual.P_slack_buf`.

Read the functor at lines 94-110 of `ac_power_flow_residual.jl`:

```julia
function (Residual::ACPowerFlowResidual)(
    Rv::Vector{Float64},
    x::Vector{Float64},
    time_step::Int64,
)
    Residual.Rf!(
        Residual.Rv,
        x,
        Residual.P_net,
        Residual.Q_net,
        Residual.P_net_set,
        Residual.bus_slack_participation_factors,
        Residual.subnetworks,
        Residual.data,
        time_step,
    )
```

Edit to pass the scratch buffer (append at the end of the argument list to keep the existing positions stable):

```julia
function (Residual::ACPowerFlowResidual)(
    Rv::Vector{Float64},
    x::Vector{Float64},
    time_step::Int64,
)
    Residual.Rf!(
        Residual.Rv,
        x,
        Residual.P_net,
        Residual.Q_net,
        Residual.P_net_set,
        Residual.bus_slack_participation_factors,
        Residual.subnetworks,
        Residual.data,
        time_step,
        Residual.P_slack_buf,
    )
```

Verify the other functor overload (search for the second `(Residual::ACPowerFlowResidual)` if there is one) gets the same change.

- [ ] **Step 6: Update the `_update_residual_values!` signature**

Edit lines 265-275 of `ac_power_flow_residual.jl`:

```julia
function _update_residual_values!(
    F::Vector{Float64},
    x::Vector{Float64},
    P_net::Vector{Float64},
    Q_net::Vector{Float64},
    P_net_set::Vector{Float64},
    bus_slack_participation_factors::SparseVector{Float64, Int},
    subnetworks::Dict{Int64, Vector{Int64}},
    data::ACPowerFlowData,
    time_step::Int64,
    P_slack_buf::Vector{Float64},
)
```

- [ ] **Step 7: Rewrite the inner loop to use `P_slack_buf` in place**

Replace lines 281-328 of `ac_power_flow_residual.jl` (the subnetwork loop):

```julia
    for (ref_bus, subnetwork_buses) in subnetworks
        slack_scalar = x[2 * ref_bus - 1] - P_net_set[ref_bus]
        n_sub = length(subnetwork_buses)
        # Write per-bus slack into P_slack_buf[1:n_sub]. SparseVector indexing
        # by a Vector{Int} allocates; loop manually to keep this allocation-free.
        @inbounds for k in 1:n_sub
            ix = subnetwork_buses[k]
            P_slack_buf[k] = slack_scalar * bus_slack_participation_factors[ix]
        end

        @inbounds for k in 1:n_sub
            ix = subnetwork_buses[k]
            bt = bus_types[ix]
            p_bus_slack = P_slack_buf[k]
            if bt == PSY.ACBusTypes.PQ
                _set_state_variables_at_bus!(
                    ix, P_net, Q_net, P_net_set, p_bus_slack,
                    x, data, time_step, Val(PSY.ACBusTypes.PQ),
                )
            elseif bt == PSY.ACBusTypes.PV
                _set_state_variables_at_bus!(
                    ix, P_net, Q_net, P_net_set, p_bus_slack,
                    x, data, time_step, Val(PSY.ACBusTypes.PV),
                )
            elseif bt == PSY.ACBusTypes.REF
                _set_state_variables_at_bus!(
                    ix, P_net, Q_net, P_net_set, p_bus_slack,
                    x, data, time_step, Val(PSY.ACBusTypes.REF),
                )
            end
        end
    end
```

Note: this fixes two allocation sites at once — the `P_slack` Vector and the `bus_types[subnetwork_buses]` Vector (now we read `bus_types[ix]` inside the loop, which is a scalar index of a `view`, free).

- [ ] **Step 8: Run the allocation regression test, confirm partial pass**

Run: `julia --project=test -e 'using ReTest; include("test/runtests.jl"); retest("AC NR allocation")'`

Expected: residual `@allocated` drops significantly (from ~140 KB to ~80 KB on 2000-bus). Test may still fail if the F-slice `.-=` allocations remain (Task 3 below). Note the new number.

- [ ] **Step 9: Run the existing AC power-flow tests to catch regressions**

Run: `julia --project=test -e 'using ReTest; include("test/runtests.jl"); retest("AC.*power.*flow", "Newton")'`

Expected: PASS. The math is preserved (scalar × scalar → vector entry is equivalent to scalar `.*` vector).

- [ ] **Step 10: Format and stage**

Run: `julia --project=scripts/formatter -e 'include("scripts/formatter/formatter_code.jl")'`

Then: `git add src/ac_power_flow_residual.jl`

> Do not commit. Stage only.

---

## Task 3: Eliminate F-slice broadcast allocations

The two `F[1:2:(end - 4 * num_lcc)] .-= P_net` lines each allocate ~80 KB on each residual call.

**Files:**
- Modify: `src/ac_power_flow_residual.jl` (lines 378-379)

- [ ] **Step 1: Add a microbenchmark proving these lines allocate**

In a Julia REPL or scratch script:

```julia
n = 10_000
F = randn(2 * n)
P_net = randn(n)
@allocated F[1:2:(2*n)] .-= P_net
```

If this prints > 0, confirm the hypothesis (strided LHS `.-=` allocates). If 0, skip this task entirely.

- [ ] **Step 2: Replace strided broadcasts with explicit loops**

In `src/ac_power_flow_residual.jl`, replace lines 378-379:

```julia
    F[1:2:(end - 4 * num_lcc)] .-= P_net
    F[2:2:(end - 4 * num_lcc)] .-= Q_net
```

With:

```julia
    @inbounds for ix in eachindex(P_net)
        F[2 * ix - 1] -= P_net[ix]
        F[2 * ix]     -= Q_net[ix]
    end
```

This iterates over the same n_buses entries and writes the same locations. The bound is implicitly `length(P_net) == n_buses == (length(F) - 4 * num_lcc) ÷ 2` — already guaranteed by the constructor.

- [ ] **Step 3: Run the allocation test and AC NR tests**

Run: `julia --project=test -e 'using ReTest; include("test/runtests.jl"); retest("AC NR allocation", "Newton")'`

Expected: residual `@allocated` drops to <2 KB (the target). AC NR tests still pass.

- [ ] **Step 4: Format and stage**

Run: `julia --project=scripts/formatter -e 'include("scripts/formatter/formatter_code.jl")'`

Then: `git add src/ac_power_flow_residual.jl`

---

## Task 4: Replace `A * Δx_nr` with `mul!` in `_do_refinement!`

A 160 KB-per-call matrix-vector product that already has an obvious in-place replacement.

**Files:**
- Modify: `src/power_flow_method.jl` (lines 46-65)
- Possibly modify: `src/power_flow_method.jl` `StateVectorCache` struct definition (search for it)

- [ ] **Step 1: Find the StateVectorCache definition**

Run: `grep -n 'struct StateVectorCache\|StateVectorCache(' /Users/jlara/.julia/dev/PowerFlows/src/power_flow_method.jl`

Note its fields. We need a scratch vector the size of the state vector. Most candidates: `r_predict` is already used as a temporary in `_do_refinement!`, so we can reuse it.

- [ ] **Step 2: Rewrite `_do_refinement!` to use `mul!`**

Read lines 46-65 of `src/power_flow_method.jl`. Replace the body:

```julia
function _do_refinement!(stateVector::StateVectorCache,
    A::SparseMatrixCSC{Float64, J_INDEX_TYPE},
    cache::KLULinSolveCache{J_INDEX_TYPE},
    refinement_threshold::Float64,
    refinement_eps::Float64,
)
    # use stateVector.r_predict as temporary buffer.
    δ_temp = stateVector.r_predict
    LinearAlgebra.mul!(δ_temp, A, stateVector.Δx_nr)
    δ_temp .-= stateVector.r
    delta = norm(δ_temp, 1) / norm(stateVector.r, 1)
    if delta > refinement_threshold
        stateVector.Δx_nr .= solve_w_refinement(cache,
            A,
            stateVector.r,
            refinement_eps)
    end
    return
end
```

The change is one line: `copyto!(δ_temp, A * stateVector.Δx_nr)` → `LinearAlgebra.mul!(δ_temp, A, stateVector.Δx_nr)`.

Verify `LinearAlgebra` is already imported by this file (it is — confirm via `grep -n 'using LinearAlgebra\|import LinearAlgebra' src/power_flow_method.jl`; if not, add it).

- [ ] **Step 3: Run the test, AC NR tests**

Run: `julia --project=test -e 'using ReTest; include("test/runtests.jl"); retest("AC NR allocation", "Newton")'`

Expected: the `mul!` allocation assertion in the test (`@allocated LinearAlgebra.mul!(out, J.Jv, x)) == 0`) was always passing — this task ensures the production code follows the same pattern. AC NR tests still pass.

- [ ] **Step 4: Format and stage**

Run: `julia --project=scripts/formatter -e 'include("scripts/formatter/formatter_code.jl")'`

Then: `git add src/power_flow_method.jl`

---

## Task 5: End-to-end re-benchmark to verify cumulative gain

Validate that the overall AC NR `solve_power_flow!` allocation dropped meaningfully — not just the isolated hotspots.

**Files:** none modified; this is a measurement step.

- [ ] **Step 1: Run the existing distribution benchmark**

Run: `julia --project=test /tmp/perf_ac_distribution.jl` (the script from prior investigation; if unavailable, write a minimal one that builds the system, runs 10× warmed solves, and prints min/median/max + bytes).

Expected (target): per-call allocation drops from ~30 MB to under 12 MB. Median time should be unchanged or slightly faster (~50 ms on 10k-bus).

- [ ] **Step 2: Run on 2000-bus system for CI-equivalent check**

The CI uses 10k-bus. Verify locally that the regression assertion (`@allocated residual(x0, 1) < 2_000`) holds on 2000-bus.

- [ ] **Step 3: Capture the before/after numbers in the PR description**

Once the user is ready to push, the PR description should include:
- Before: residual call 689 KB, full solve 30.6 MB
- After: residual call <2 KB, full solve <12 MB (or actual measured)

---

## Task 6: Add an `@allocated` budget regression test for the full solve

Codify the gain so the next PR doesn't silently undo it.

**Files:**
- Modify: `test/test_ac_nr_allocations.jl` (append a new testset)

- [ ] **Step 1: Append a budget test**

Add to `test/test_ac_nr_allocations.jl`:

```julia
@testset "AC NR full-solve allocation budget" begin
    sys = build_system(MatpowerTestSystems, "matpower_ACTIVSg2000_sys")
    pf = ACPowerFlow{PF.NewtonRaphsonACPowerFlow}(; correct_bustypes = true)
    # Warm
    let pf_data = PF.PowerFlowData(pf, sys)
        PF.solve_power_flow!(pf_data; pf = pf)
    end
    pf_data = PF.PowerFlowData(pf, sys)
    bytes = @allocated PF.solve_power_flow!(pf_data; pf = pf)
    # 2000-bus baseline post-fix should be ~3 MB. 6 MB is 2x headroom for noise.
    @test bytes < 6_000_000
end
```

- [ ] **Step 2: Run and confirm pass**

Run: `julia --project=test -e 'using ReTest; include("test/runtests.jl"); retest("AC NR full-solve")'`

Expected: PASS.

- [ ] **Step 3: Format and stage**

Run: `julia --project=scripts/formatter -e 'include("scripts/formatter/formatter_code.jl")'`

Then: `git add test/test_ac_nr_allocations.jl`

---

## Self-Review Checklist

- [x] **Spec coverage:** Goal is "cut 30 MB to <10 MB". Task 2 cuts P_slack + bus_types[idx], Task 3 cuts F-slice .-=, Task 4 cuts A*Δx. Task 5 verifies, Task 6 codifies. Post-solve `Sft/Stf` block in `solve_ac_power_flow.jl:223-228` (1.2 MB, once per solve) is **explicitly out of scope** — adding to plan would push it past the bite-sized target; the goal is met without it. If a Task 5 measurement shows the goal isn't met, add a follow-up task then.
- [x] **Placeholder scan:** No "TBD"; every code block is complete; threshold numbers are derived from measured baselines (689 KB residual, 160 KB mul!, 80 B Jacobian).
- [x] **Type consistency:** `P_slack_buf::Vector{Float64}` named and used consistently across struct, constructor, functor, and `_update_residual_values!` signature. `bus_types` reference stays as `view(data.bus_type, :, time_step)`; the rewrite reads it scalar at `bus_types[ix]` not `bus_types[subnetwork_buses]`.

**Out of scope** (intentional, do not include):
- Post-solve flow computation in `solve_ac_power_flow.jl:223-228` — 1.2 MB once per solve, smaller relative gain.
- Iwamoto-path-specific allocations — already lean per profile.
- Trust-region path allocations — different code path; separate plan if needed.
- AppleAccelerate backend — separately confirmed not a win (see prior investigation).
