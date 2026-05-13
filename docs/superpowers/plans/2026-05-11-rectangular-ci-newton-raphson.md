# Rectangular Current-Injection Newton-Raphson Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` or `superpowers:subagent-driven-development` to implement task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Add `RectangularCurrentInjectionACPowerFlow` solver to PowerFlows.jl — augmented current-injection (Da Costa) NR with constant Y_bus off-diagonal Jacobian blocks, full LCC HVDC / distributed slack / ZIP / Q-limit parity with polar NR.

**Architecture:** Per-bus variable-size state blocks (PQ/REF: 2, PV: 3). New files for residual / Jacobian / LCC; reuse `KLULinSolveCache`, `StateVectorCache`, all step drivers (`_simple_step`, `_iwamoto_step`, `_trust_region_step`).

**Tech Stack:** Julia 1.10+, KLU sparse solver, SparseArrays.jl, PowerNetworkMatrices.jl, PowerSystems.jl.

**Spec:** `docs/superpowers/specs/2026-05-11-rectangular-ci-newton-raphson-design.md`

**Global rules from `~/.claude/CLAUDE.md`:**
- Never `git commit` — stage only. Commit happens after user review.
- Always run formatter (`julia --project=scripts/formatter -e 'include("scripts/formatter/formatter_code.jl")'`) before considering work complete.
- Use `julia --project=test` for tests; `julia --project=docs` for docs.
- After each file edit, verify it compiles (`julia --project=test -e 'using PowerFlows'`).
- Respect include order in `src/PowerFlows.jl`.

---

## File Map

```
src/
  power_flow_types.jl                          [MODIFY: + RectangularCurrentInjectionACPowerFlow]
  PowerFlows.jl                                [MODIFY: + exports + includes]
  state_indexing_helpers.jl                    [MODIFY: + rect_* helpers]
  definitions.jl                               [MODIFY: + REC_INDEX_TYPE]
  rectangular_ci_setup.jl                      [NEW: bus_state_offset, bus_block_size, Y_bus_eff, ZIP-Z fold]
  rectangular_ci_power_flow_residual.jl        [NEW: ACRectangularCIResidual]
  rectangular_ci_power_flow_jacobian.jl        [NEW: ACRectangularCIJacobian, sparsity, updates]
  rectangular_ci_lcc.jl                        [NEW: LCC tail residual + Jacobian chain-rule]
  power_flow_method.jl                         [MODIFY: + _newton_power_flow dispatch]
  power_flow_setup.jl                          [MODIFY: + initialize_power_flow_variables_rect]

test/
  test_rectangular_ci_setup.jl                 [NEW]
  test_rectangular_ci_residual.jl              [NEW]
  test_rectangular_ci_jacobian.jl              [NEW]
  test_rectangular_ci_lcc.jl                   [NEW]
  test_rectangular_ci_power_flow.jl            [NEW: full parity suite]
  test_distributed_slack.jl                    [MODIFY: + rect cases]
  test_multiperiod_ac_power_flow.jl            [MODIFY: + rect cases]
  test_iterative_methods.jl                    [MODIFY: + rect + Iwamoto/TR]
  test_hvdc.jl                                 [MODIFY: + rect cases]
  runtests.jl                                  [MODIFY: include new test files]
  performance/performance_test.jl              [MODIFY: + rect benchmarks]
```

---

## PR 1 — Foundation: solver type, indexing helpers, ZIP-Z fold

### Task 1.1: Add solver type and exports

**Files:**
- Modify: `src/power_flow_types.jl`
- Modify: `src/PowerFlows.jl`
- Modify: `src/definitions.jl`

- [ ] **Step 1.1.1:** Add solver type to `src/power_flow_types.jl` after `RobustHomotopyPowerFlow`:

```julia
"""
    RectangularCurrentInjectionACPowerFlow <: ACPowerFlowSolverType

An [`ACPowerFlowSolverType`](@ref) that solves the AC power flow problem using the
augmented current-injection (Da Costa) formulation in rectangular coordinates.

State variables per bus:
- PQ: (eᵢ, fᵢ) — real and imaginary parts of bus voltage
- PV: (eᵢ, fᵢ, Qᵢ) — extra Q variable; augmented row pins |V|² = V_set²
- REF: (P_genᵢ, Q_genᵢ); (eᵢ, fᵢ) fixed in data

Residuals: complex current mismatch ΔIᵢ = I_specᵢ − Y_bus·V at every bus.

Off-diagonal Jacobian blocks ≡ Y_bus 2×2 real blocks — constant across NR iterations.
Per-iteration Jacobian update cost is O(N + n_LCC), independent of nnz(Y_bus).

Based on: Da Costa, Pereira, Garcia — "Developments in the Newton-Raphson power flow
formulation based on current injections," IEEE TPS 2000.

See also: [`ACPowerFlow`](@ref).
"""
struct RectangularCurrentInjectionACPowerFlow <: ACPowerFlowSolverType end
```

- [ ] **Step 1.1.2:** Add `REC_INDEX_TYPE` constant near `J_INDEX_TYPE` in `src/definitions.jl`:

```julia
const REC_INDEX_TYPE = Int32
```

- [ ] **Step 1.1.3:** Export the new solver in `src/PowerFlows.jl` after `RobustHomotopyPowerFlow`:

```julia
export RectangularCurrentInjectionACPowerFlow
```

- [ ] **Step 1.1.4:** Verify compilation:

```sh
julia --project=test -e 'using PowerFlows; pf = ACPowerFlow{RectangularCurrentInjectionACPowerFlow}(); println("OK: ", typeof(pf))'
```

Expected: `OK: ACPowerFlow{RectangularCurrentInjectionACPowerFlow}`

### Task 1.2: Setup helpers — bus offsets and Y_bus_eff with ZIP-Z fold

**Files:**
- Create: `src/rectangular_ci_setup.jl`
- Modify: `src/PowerFlows.jl` (add include after `state_indexing_helpers.jl`)

- [ ] **Step 1.2.1:** Create `src/rectangular_ci_setup.jl`:

```julia
"""
    compute_bus_state_offsets(bus_type)

Compute per-bus state-vector offsets and block sizes for the augmented current-injection
formulation. PQ and REF buses use 2 entries (e,f) / (P_gen,Q_gen); PV buses use 3 entries
(e,f,Q).

Returns `(offsets, block_sizes, total_bus_state)` where:
- `offsets[i]` is the 1-based start index of bus i's block in the state vector
- `offsets[end]` is the start of the LCC tail (== total_bus_state + 1)
- `block_sizes[i]` ∈ {2, 3}
"""
function compute_bus_state_offsets(
    bus_type::AbstractVector{PSY.ACBusTypes},
)
    n_buses = length(bus_type)
    offsets = Vector{REC_INDEX_TYPE}(undef, n_buses + 1)
    block_sizes = Vector{Int8}(undef, n_buses)
    pos = REC_INDEX_TYPE(1)
    for i in 1:n_buses
        offsets[i] = pos
        bs = bus_type[i] == PSY.ACBusTypes.PV ? Int8(3) : Int8(2)
        block_sizes[i] = bs
        pos += bs
    end
    offsets[n_buses + 1] = pos
    return offsets, block_sizes, Int(pos - 1)
end

"""
    fold_zip_constant_z!(Y_bus_eff, data, time_step)

Add the constant-impedance ZIP load components into the Y_bus_eff diagonal as fixed
shunt admittances. Equivalent to today's polar `P_net` voltage-magnitude correction,
but applied once at setup so the rectangular Jacobian's off-diagonal Y blocks remain
truly constant.

A constant-Z load drawing S_load_0 at nominal |V|=V₀ corresponds to shunt admittance
Y_sh = conj(S_load_0)/V₀² = (β_P − jβ_Q)/V₀². V₀ is taken from the current bus_magnitude
field at solver setup.
"""
function fold_zip_constant_z!(
    Y_bus_eff::SparseMatrixCSC{ComplexF64, Int},
    data::ACPowerFlowData,
    time_step::Int64,
)
    Vm = view(data.bus_magnitude, :, time_step)
    n_buses = first(size(data.bus_type))
    for i in 1:n_buses
        β_P = data.bus_active_power_constant_impedance_withdrawals[i, time_step]
        β_Q = data.bus_reactive_power_constant_impedance_withdrawals[i, time_step]
        (β_P == 0.0 && β_Q == 0.0) && continue
        V0_sq = Vm[i]^2
        V0_sq == 0.0 && continue
        Y_bus_eff[i, i] += complex(β_P, -β_Q) / V0_sq
    end
    return
end

"""
    rect_initial_state!(x, data, bus_state_offset, bus_block_size, time_step)

Initialize state vector from data.bus_magnitude/bus_angles + power injections.
"""
function rect_initial_state!(
    x::Vector{Float64},
    data::ACPowerFlowData,
    bus_state_offset::Vector{REC_INDEX_TYPE},
    bus_block_size::Vector{Int8},
    time_step::Int64,
)
    bus_types = view(data.bus_type, :, time_step)
    n_buses = length(bus_types)
    for i in 1:n_buses
        off = bus_state_offset[i]
        bt = bus_types[i]
        if bt == PSY.ACBusTypes.REF
            # x[off]   = P_gen (incl. distributed slack)
            # x[off+1] = Q_gen
            x[off] = data.bus_active_power_injections[i, time_step] -
                     data.bus_active_power_withdrawals[i, time_step]
            x[off + 1] = data.bus_reactive_power_injections[i, time_step] -
                         data.bus_reactive_power_withdrawals[i, time_step]
        else
            # PQ or PV: x[off] = e_i, x[off+1] = f_i
            Vm = data.bus_magnitude[i, time_step]
            θ = data.bus_angles[i, time_step]
            x[off] = Vm * cos(θ)
            x[off + 1] = Vm * sin(θ)
            if bt == PSY.ACBusTypes.PV
                # x[off+2] = Q_gen
                x[off + 2] = data.bus_reactive_power_injections[i, time_step] -
                             data.bus_reactive_power_withdrawals[i, time_step]
            end
        end
    end
    # LCC tail
    n_lccs = size(data.lcc.p_set, 1)
    tail_start = bus_state_offset[n_buses + 1]
    for i in 1:n_lccs
        x[tail_start + 4*(i-1)] = data.lcc.rectifier.tap[i, time_step]
        x[tail_start + 4*(i-1) + 1] = data.lcc.inverter.tap[i, time_step]
        x[tail_start + 4*(i-1) + 2] = data.lcc.rectifier.thyristor_angle[i, time_step]
        x[tail_start + 4*(i-1) + 3] = data.lcc.inverter.thyristor_angle[i, time_step]
    end
    return
end

"""
    rect_update_data!(data, x, bus_state_offset, bus_block_size, time_step)

Write state vector contents back into the data fields (bus_magnitude, bus_angles,
bus_*_power_injections, lcc taps/angles). Counterpart of `rect_initial_state!`.
"""
function rect_update_data!(
    data::ACPowerFlowData,
    x::Vector{Float64},
    bus_state_offset::Vector{REC_INDEX_TYPE},
    bus_block_size::Vector{Int8},
    time_step::Int64,
)
    bus_types = view(data.bus_type, :, time_step)
    n_buses = length(bus_types)
    for i in 1:n_buses
        off = bus_state_offset[i]
        bt = bus_types[i]
        if bt == PSY.ACBusTypes.REF
            P_gen = x[off]
            Q_gen = x[off + 1]
            data.bus_active_power_injections[i, time_step] =
                P_gen + data.bus_active_power_withdrawals[i, time_step]
            data.bus_reactive_power_injections[i, time_step] =
                Q_gen + data.bus_reactive_power_withdrawals[i, time_step]
        else
            e = x[off]
            f = x[off + 1]
            Vm = sqrt(e^2 + f^2)
            θ = atan(f, e)
            data.bus_magnitude[i, time_step] = Vm
            data.bus_angles[i, time_step] = θ
            if bt == PSY.ACBusTypes.PV
                Q_gen = x[off + 2]
                data.bus_reactive_power_injections[i, time_step] =
                    Q_gen + data.bus_reactive_power_withdrawals[i, time_step]
            end
        end
    end
    n_lccs = size(data.lcc.p_set, 1)
    tail_start = bus_state_offset[n_buses + 1]
    for i in 1:n_lccs
        data.lcc.rectifier.tap[i, time_step] = x[tail_start + 4*(i-1)]
        data.lcc.inverter.tap[i, time_step] = x[tail_start + 4*(i-1) + 1]
        data.lcc.rectifier.thyristor_angle[i, time_step] = x[tail_start + 4*(i-1) + 2]
        data.lcc.inverter.thyristor_angle[i, time_step] = x[tail_start + 4*(i-1) + 3]
    end
    return
end
```

- [ ] **Step 1.2.2:** Add include in `src/PowerFlows.jl` between `state_indexing_helpers.jl` and `ac_power_flow_residual.jl`:

```julia
include("state_indexing_helpers.jl")
include("rectangular_ci_setup.jl")     # <-- new
include("ac_power_flow_residual.jl")
```

- [ ] **Step 1.2.3:** Compile check:

```sh
julia --project=test -e 'using PowerFlows; println("OK")'
```

### Task 1.3: Tests for setup helpers

**Files:**
- Create: `test/test_rectangular_ci_setup.jl`
- Modify: `test/runtests.jl` (add include)

- [ ] **Step 1.3.1:** Create `test/test_rectangular_ci_setup.jl`:

```julia
@testset "Rectangular CI Setup" begin
    using PowerFlows: compute_bus_state_offsets, REC_INDEX_TYPE, fold_zip_constant_z!,
                      rect_initial_state!, rect_update_data!
    using PowerSystems
    const PSY = PowerSystems

    @testset "bus offsets — all PQ" begin
        bus_type = fill(PSY.ACBusTypes.PQ, 5)
        off, bs, total = compute_bus_state_offsets(bus_type)
        @test off == REC_INDEX_TYPE[1, 3, 5, 7, 9, 11]
        @test bs == fill(Int8(2), 5)
        @test total == 10
    end

    @testset "bus offsets — mixed PV/PQ/REF" begin
        bus_type = [PSY.ACBusTypes.REF, PSY.ACBusTypes.PV, PSY.ACBusTypes.PQ,
                    PSY.ACBusTypes.PV, PSY.ACBusTypes.PQ]
        off, bs, total = compute_bus_state_offsets(bus_type)
        # REF=2, PV=3, PQ=2, PV=3, PQ=2 -> 12 total
        @test off == REC_INDEX_TYPE[1, 3, 6, 8, 11, 13]
        @test bs == Int8[2, 3, 2, 3, 2]
        @test total == 12
    end

    @testset "ZIP-Z fold sign convention" begin
        # Build a 1x1 Y_bus_eff initialized to zero, fold a 0.5 + j0.3 const-Z load
        # at V₀=1.0. Expect Y[1,1] += (0.5 - j0.3) / 1.0² = (0.5 - 0.3im).
        sys = PSB.build_system(PSB.PSITestSystems, "c_sys5")
        data = PowerFlowData(ACPowerFlow{RectangularCurrentInjectionACPowerFlow}(), sys)
        Y = copy(data.power_network_matrix.data)
        Y_eff = SparseArrays.sparse(ComplexF64.(Y))
        # Inject a synthetic const-Z load at bus 1
        data.bus_active_power_constant_impedance_withdrawals[1, 1] = 0.5
        data.bus_reactive_power_constant_impedance_withdrawals[1, 1] = 0.3
        data.bus_magnitude[1, 1] = 1.0
        Y_eff_before = copy(Y_eff)
        fold_zip_constant_z!(Y_eff, data, 1)
        @test Y_eff[1, 1] - Y_eff_before[1, 1] ≈ complex(0.5, -0.3) atol=1e-12
    end
end
```

- [ ] **Step 1.3.2:** Add to `test/runtests.jl` near other test includes:

```julia
include("test_rectangular_ci_setup.jl")
```

- [ ] **Step 1.3.3:** Run tests:

```sh
julia --project=test -e 'using Pkg; Pkg.test(test_args=["test_rectangular_ci_setup"])'
```

If runtests.jl uses ReTest, run via the project's ReTest entry point.

---

## PR 2 — `ACRectangularCIResidual`

### Task 2.1: Residual struct definition

**Files:**
- Create: `src/rectangular_ci_power_flow_residual.jl`
- Modify: `src/PowerFlows.jl`

- [ ] **Step 2.1.1:** Create `src/rectangular_ci_power_flow_residual.jl`:

```julia
"""
    struct ACRectangularCIResidual

Residual functor for the augmented current-injection AC power flow formulation.
Mirrors `ACPowerFlowResidual` but operates on the (e, f) state representation
with per-bus variable block sizes.

# Fields
- `data::ACPowerFlowData`
- `Rf!::Function` — inplace residual update; signature mirrors `_update_residual_values!`
- `Rv::Vector{Float64}` — current residual values
- `Y_bus_eff::SparseMatrixCSC{ComplexF64, Int}` — Y_bus with ZIP-Z folded in
- `P_net::Vector{Float64}` — net active injections (const-P + const-I baseline)
- `Q_net::Vector{Float64}` — net reactive injections (const-P + const-I baseline)
- `P_net_set::Vector{Float64}` — initial P_net (for distributed-slack delta)
- `bus_slack_participation_factors::SparseVector{Float64, Int}`
- `subnetworks::Dict{Int64, Vector{Int64}}`
- `bus_state_offset::Vector{REC_INDEX_TYPE}`
- `bus_block_size::Vector{Int8}`
- `total_bus_state::Int` — sum of block sizes (excludes LCC tail)
"""
struct ACRectangularCIResidual
    data::ACPowerFlowData
    Rf!::Function
    Rv::Vector{Float64}
    Y_bus_eff::SparseMatrixCSC{ComplexF64, Int}
    P_net::Vector{Float64}
    Q_net::Vector{Float64}
    P_net_set::Vector{Float64}
    bus_slack_participation_factors::SparseVector{Float64, Int}
    subnetworks::Dict{Int64, Vector{Int64}}
    bus_state_offset::Vector{REC_INDEX_TYPE}
    bus_block_size::Vector{Int8}
    total_bus_state::Int
end

function ACRectangularCIResidual(data::ACPowerFlowData, time_step::Int64)
    n_buses = first(size(data.bus_type))
    n_lccs = size(data.lcc.p_set, 1)
    bus_type = view(data.bus_type, :, time_step)

    offsets, block_sizes, total_bus_state = compute_bus_state_offsets(bus_type)
    total_state = total_bus_state + 4 * n_lccs

    P_net = Vector{Float64}(undef, n_buses)
    Q_net = Vector{Float64}(undef, n_buses)
    P_net_set = zeros(Float64, n_buses)

    spf_idx = Int[]
    spf_val = Float64[]
    sum_sl_weights = 0.0

    subnetworks =
        _find_subnetworks_for_reference_buses(data.power_network_matrix.data, bus_type)

    for (ix, bt) in zip(1:n_buses, bus_type)
        P_net[ix] =
            data.bus_active_power_injections[ix, time_step] -
            get_bus_active_power_total_withdrawals(data, ix, time_step) +
            data.bus_hvdc_net_power[ix, time_step]
        Q_net[ix] =
            data.bus_reactive_power_injections[ix, time_step] -
            get_bus_reactive_power_total_withdrawals(data, ix, time_step)
        P_net_set[ix] = P_net[ix]
        bt ∈ (PSY.ACBusTypes.REF, PSY.ACBusTypes.PV) || continue
        (spf_v = data.bus_slack_participation_factors[ix, time_step]) == 0.0 && continue
        push!(spf_idx, ix)
        push!(spf_val, spf_v)
        sum_sl_weights += spf_v
    end

    sum_sl_weights == 0.0 &&
        throw(ArgumentError("sum of slack_participation_factors cannot be zero"))
    any(spf_val .< 0.0) &&
        throw(ArgumentError("slack_participation_factors cannot be negative"))

    bus_slack_participation_factors = sparsevec(spf_idx, spf_val, n_buses)
    for subnetwork_buses in values(subnetworks)
        bspf_subnetwork = view(bus_slack_participation_factors, subnetwork_buses)
        sum_bspf = sum(bspf_subnetwork)
        sum_bspf == 0.0 &&
            throw(ArgumentError("sum of slack_participation_factors per subnetwork cannot be zero"))
        bspf_subnetwork ./= sum_bspf
    end

    # Build Y_bus_eff: copy Y_bus + fold ZIP-Z
    Y = data.power_network_matrix.data
    Y_bus_eff = SparseArrays.sparse(ComplexF64.(Y))
    fold_zip_constant_z!(Y_bus_eff, data, time_step)

    return ACRectangularCIResidual(
        data,
        _update_rect_ci_residual_values!,
        Vector{Float64}(undef, total_state),
        Y_bus_eff,
        P_net,
        Q_net,
        P_net_set,
        bus_slack_participation_factors,
        subnetworks,
        offsets,
        block_sizes,
        total_bus_state,
    )
end

function (R::ACRectangularCIResidual)(
    Rv::Vector{Float64},
    x::Vector{Float64},
    time_step::Int64,
)
    R.Rf!(R.Rv, x, R.Y_bus_eff, R.P_net, R.Q_net, R.P_net_set,
        R.bus_slack_participation_factors, R.subnetworks,
        R.bus_state_offset, R.bus_block_size, R.total_bus_state,
        R.data, time_step)
    copyto!(Rv, R.Rv)
    return
end

function (R::ACRectangularCIResidual)(x::Vector{Float64}, time_step::Int64)
    R.Rf!(R.Rv, x, R.Y_bus_eff, R.P_net, R.Q_net, R.P_net_set,
        R.bus_slack_participation_factors, R.subnetworks,
        R.bus_state_offset, R.bus_block_size, R.total_bus_state,
        R.data, time_step)
    return
end
```

- [ ] **Step 2.1.2:** Add include in `src/PowerFlows.jl` after `ac_power_flow_jacobian.jl`:

```julia
include("ac_power_flow_jacobian.jl")
include("rectangular_ci_power_flow_residual.jl")   # <-- new
```

### Task 2.2: Residual update function `_update_rect_ci_residual_values!`

- [ ] **Step 2.2.1:** Append to `src/rectangular_ci_power_flow_residual.jl`:

```julia
"""
Update residual values F for the augmented current-injection AC power flow.
Walks Y_bus_eff once to accumulate I_inj. Then per-bus subtracts I_spec.
PV bus adds the ΔV² row. LCC tail residuals appended at the end.
"""
function _update_rect_ci_residual_values!(
    F::Vector{Float64},
    x::Vector{Float64},
    Y_bus_eff::SparseMatrixCSC{ComplexF64, Int},
    P_net::Vector{Float64},
    Q_net::Vector{Float64},
    P_net_set::Vector{Float64},
    bus_slack_participation_factors::SparseVector{Float64, Int},
    subnetworks::Dict{Int64, Vector{Int64}},
    bus_state_offset::Vector{REC_INDEX_TYPE},
    bus_block_size::Vector{Int8},
    total_bus_state::Int,
    data::ACPowerFlowData,
    time_step::Int64,
)
    n_buses = first(size(data.bus_type))
    n_lccs = size(data.lcc.p_set, 1)
    bus_types = view(data.bus_type, :, time_step)

    # 1) Push state into data (handles V_m, V_a, P_gen, Q_gen, lcc taps/angles)
    rect_update_data!(data, x, bus_state_offset, bus_block_size, time_step)
    if n_lccs > 0
        _update_ybus_lcc!(data, time_step)
        # Update Y_bus_eff diagonals to reflect the new LCC self-admittances.
        # The polar code reads Y_bus (which includes LCC diag); we mirror by re-folding.
        _refresh_lcc_diagonals!(Y_bus_eff, data, time_step)
    end

    # 2) Apply distributed slack — compute P_eff (and Q is identical to set for PQ)
    fill!(F, 0.0)
    P_eff = P_net  # reuse buffer if no slack; otherwise recompute below per bus
    # We need a fresh P_eff vector that mixes slack into each participating bus.
    # Build it from P_net_set + slack delta. P_net itself may have been modified earlier.
    P_eff = similar(P_net)
    Q_eff = similar(Q_net)
    copyto!(P_eff, P_net_set)
    copyto!(Q_eff, Q_net)
    for (ref_bus, subnetwork_buses) in subnetworks
        ref_off = bus_state_offset[ref_bus]
        P_slack_total = x[ref_off] - P_net_set[ref_bus]
        for bus_k in subnetwork_buses
            c_k = bus_slack_participation_factors[bus_k]
            c_k == 0.0 && continue
            bus_k == ref_bus && continue
            P_eff[bus_k] = P_net_set[bus_k] + c_k * P_slack_total
        end
    end

    # 3) Walk Y_bus_eff to accumulate -I_inj into the F slots
    #    F at bus i: F[off+0] -= Re(I_inj), F[off+1] -= Im(I_inj)
    Yvals = SparseArrays.nonzeros(Y_bus_eff)
    Yrows = SparseArrays.rowvals(Y_bus_eff)
    for col in 1:n_buses
        col_off = bus_state_offset[col]
        e_col = x[col_off]
        f_col = x[col_off + 1]
        # For REF buses (col_off, col_off+1) hold (P_gen, Q_gen); use fixed data fields
        if bus_types[col] == PSY.ACBusTypes.REF
            Vm = data.bus_magnitude[col, time_step]
            θ = data.bus_angles[col, time_step]
            e_col = Vm * cos(θ)
            f_col = Vm * sin(θ)
        end
        for j in Y_bus_eff.colptr[col]:(Y_bus_eff.colptr[col + 1] - 1)
            row = Yrows[j]
            y = Yvals[j]
            g = real(y)
            b = imag(y)
            row_off = bus_state_offset[row]
            F[row_off]     -= (g * e_col - b * f_col)   # -Re(Y*V)
            F[row_off + 1] -= (g * f_col + b * e_col)   # -Im(Y*V)
        end
    end

    # 4) Add per-bus I_spec contributions; PV ΔV²; REF block linearity
    for i in 1:n_buses
        off = bus_state_offset[i]
        bt = bus_types[i]
        if bt == PSY.ACBusTypes.REF
            Vm = data.bus_magnitude[i, time_step]
            θ = data.bus_angles[i, time_step]
            e_r = Vm * cos(θ)
            f_r = Vm * sin(θ)
            V_sq = e_r^2 + f_r^2
            P_gen = x[off]
            Q_gen = x[off + 1]
            # I_spec at REF using fixed (e_r, f_r) and state (P_gen, Q_gen):
            F[off]     += (P_gen * e_r + Q_gen * f_r) / V_sq
            F[off + 1] += (P_gen * f_r - Q_gen * e_r) / V_sq
        else
            e_i = x[off]
            f_i = x[off + 1]
            V_sq = e_i^2 + f_i^2
            P_i = P_eff[i]
            Q_i = bt == PSY.ACBusTypes.PV ? x[off + 2] : Q_eff[i]
            F[off]     += (P_i * e_i + Q_i * f_i) / V_sq
            F[off + 1] += (P_i * f_i - Q_i * e_i) / V_sq
            if bt == PSY.ACBusTypes.PV
                V_set_sq = data.bus_magnitude[i, time_step]^2
                F[off + 2] = V_set_sq - V_sq
            end
        end
    end

    # 5) LCC tail (unchanged formulas; same as polar code's LCC tail)
    if n_lccs > 0
        Vm = view(data.bus_magnitude, :, time_step)
        tail_start = bus_state_offset[n_buses + 1]
        P_lcc_from =
            Vm[data.lcc.rectifier.bus] .* data.lcc.rectifier.tap[:, time_step] .*
            sqrt(6) / π .* data.lcc.i_dc[:, time_step] .*
            cos.(data.lcc.rectifier.phi[:, time_step])
        P_lcc_to =
            Vm[data.lcc.inverter.bus] .* data.lcc.inverter.tap[:, time_step] .*
            sqrt(6) / π .* data.lcc.i_dc[:, time_step] .*
            cos.(data.lcc.inverter.phi[:, time_step])
        for i in 1:n_lccs
            base = tail_start + 4*(i-1) - 1
            F[base + 1] = ifelse(data.lcc.setpoint_at_rectifier[i],
                P_lcc_from[i] - data.lcc.p_set[i, time_step],
                -P_lcc_to[i] - data.lcc.p_set[i, time_step])
            F[base + 2] = P_lcc_from[i] + P_lcc_to[i] -
                data.lcc.dc_line_resistance[i] * data.lcc.i_dc[i, time_step]^2
            F[base + 3] = data.lcc.rectifier.thyristor_angle[i, time_step] -
                data.lcc.rectifier.min_thyristor_angle[i]
            F[base + 4] = data.lcc.inverter.thyristor_angle[i, time_step] -
                data.lcc.inverter.min_thyristor_angle[i]
        end
    end
    return
end

"""Re-fold LCC self-admittances into Y_bus_eff after `_update_ybus_lcc!` ran.
The polar code mutates the shared Y_bus matrix directly via `data.power_network_matrix.data`.
Here we maintain Y_bus_eff = Y_bus + ZIP_Z_fold + LCC_diag separately so we don't
have to redo the ZIP fold each iteration."""
function _refresh_lcc_diagonals!(
    Y_bus_eff::SparseMatrixCSC{ComplexF64, Int},
    data::ACPowerFlowData,
    time_step::Int64,
)
    # `_update_ybus_lcc!` already wrote LCC diagonals into data.power_network_matrix.data.
    # Y_bus_eff = Y_bus_data + ZIP_Z_fold (constant). So sync LCC-affected diagonals
    # from data.power_network_matrix.data into Y_bus_eff, preserving the ZIP fold delta.
    # Implementation: at LCC's AC buses, Y_bus_eff[i,i] = Y_bus_data[i,i] + zip_delta[i].
    # ZIP delta is constant per setup; rebuild from data.
    for (bus_indices, _) in zip(data.lcc.bus_indices, data.lcc.branch_admittances)
        for bus_ix in bus_indices
            β_P = data.bus_active_power_constant_impedance_withdrawals[bus_ix, time_step]
            β_Q = data.bus_reactive_power_constant_impedance_withdrawals[bus_ix, time_step]
            V0 = data.bus_magnitude[bus_ix, time_step]
            zip_delta = (V0 == 0.0 || (β_P == 0.0 && β_Q == 0.0)) ? complex(0.0) :
                complex(β_P, -β_Q) / V0^2
            Y_bus_eff[bus_ix, bus_ix] =
                ComplexF64(data.power_network_matrix.data[bus_ix, bus_ix]) + zip_delta
        end
    end
    return
end
```

### Task 2.3: Residual unit tests

**Files:**
- Create: `test/test_rectangular_ci_residual.jl`
- Modify: `test/runtests.jl`

- [ ] **Step 2.3.1:** Create `test/test_rectangular_ci_residual.jl`:

```julia
@testset "Rectangular CI Residual" begin
    using PowerFlows: ACRectangularCIResidual, compute_bus_state_offsets,
                      rect_initial_state!, RectangularCurrentInjectionACPowerFlow
    using PowerSimulationsTestSystems
    using PowerSimulationsTestSystems: PSITestSystems

    @testset "Residual zero at known polar solution — c_sys5" begin
        sys = build_system(PSITestSystems, "c_sys5")
        # First solve with polar NR to get the converged state
        pf_polar = ACPowerFlow{NewtonRaphsonACPowerFlow}()
        @test solve_and_store_power_flow!(pf_polar, sys)
        # Now construct rectangular CI residual at the converged state
        data = PowerFlowData(ACPowerFlow{RectangularCurrentInjectionACPowerFlow}(), sys)
        R = ACRectangularCIResidual(data, 1)
        x = Vector{Float64}(undef, length(R.Rv))
        rect_initial_state!(x, data, R.bus_state_offset, R.bus_block_size, 1)
        R(x, 1)
        @test norm(R.Rv, Inf) < 1e-7
    end

    @testset "Residual nonzero at flat start — IEEE 14" begin
        sys = build_system(PSISystems, "ieee14")
        data = PowerFlowData(ACPowerFlow{RectangularCurrentInjectionACPowerFlow}(), sys)
        R = ACRectangularCIResidual(data, 1)
        x = Vector{Float64}(undef, length(R.Rv))
        # Force flat start: V=1, θ=0, Q_gen=0
        data.bus_magnitude[:, 1] .= 1.0
        data.bus_angles[:, 1] .= 0.0
        rect_initial_state!(x, data, R.bus_state_offset, R.bus_block_size, 1)
        R(x, 1)
        @test norm(R.Rv, Inf) > 1e-3   # something to converge
    end
end
```

- [ ] **Step 2.3.2:** Add include in `test/runtests.jl`.

- [ ] **Step 2.3.3:** Run:

```sh
julia --project=test -e 'using Pkg; Pkg.test(test_args=["test_rectangular_ci_residual"])'
```

---

## PR 3 — `ACRectangularCIJacobian` (no LCC tail)

### Task 3.1: Sparsity construction

**Files:**
- Create: `src/rectangular_ci_power_flow_jacobian.jl`
- Modify: `src/PowerFlows.jl`

- [ ] **Step 3.1.1:** Create skeleton with `ACRectangularCIJacobian` struct + `_create_rect_ci_jacobian_structure`. The structure walks Y_bus, places per-bus block-sized entries at each `(i, j)` nonzero, adds slack cross-terms, adds LCC tail entries with structural zeros (LCC math comes in PR 4).

Code:

```julia
struct ACRectangularCIJacobian
    data::ACPowerFlowData
    Jf!::Function
    Jv::SparseArrays.SparseMatrixCSC{Float64, J_INDEX_TYPE}
    Y_bus_eff::SparseMatrixCSC{ComplexF64, Int}
    bus_slack_participation_factors::SparseVector{Float64, Int}
    subnetworks::Dict{Int64, Vector{Int64}}
    bus_state_offset::Vector{REC_INDEX_TYPE}
    bus_block_size::Vector{Int8}
    total_bus_state::Int
end

function ACRectangularCIJacobian(
    residual::ACRectangularCIResidual,
    time_step::Int64,
)
    Jv0 = _create_rect_ci_jacobian_structure(
        residual.data,
        residual.Y_bus_eff,
        residual.bus_slack_participation_factors,
        residual.subnetworks,
        residual.bus_state_offset,
        residual.bus_block_size,
        residual.total_bus_state,
        time_step,
    )
    # Populate the CONSTANT off-diagonal Y_bus blocks once.
    _populate_constant_yb_blocks!(
        Jv0,
        residual.Y_bus_eff,
        residual.bus_state_offset,
        residual.bus_block_size,
        view(residual.data.bus_type, :, time_step),
    )
    return ACRectangularCIJacobian(
        residual.data,
        _update_rect_ci_jacobian_values!,
        Jv0,
        residual.Y_bus_eff,
        residual.bus_slack_participation_factors,
        residual.subnetworks,
        residual.bus_state_offset,
        residual.bus_block_size,
        residual.total_bus_state,
    )
end

function (J::ACRectangularCIJacobian)(time_step::Int64)
    J.Jf!(J.Jv, J.data, J.Y_bus_eff,
        J.bus_slack_participation_factors, J.subnetworks,
        J.bus_state_offset, J.bus_block_size, J.total_bus_state, time_step)
    return
end

function _create_rect_ci_jacobian_structure(
    data::ACPowerFlowData,
    Y_bus_eff::SparseMatrixCSC{ComplexF64, Int},
    bus_slack_participation_factors::SparseVector{Float64, Int},
    subnetworks::Dict{Int64, Vector{Int64}},
    bus_state_offset::Vector{REC_INDEX_TYPE},
    bus_block_size::Vector{Int8},
    total_bus_state::Int,
    time_step::Int64,
)
    rows = J_INDEX_TYPE[]
    cols = J_INDEX_TYPE[]
    vals = Float64[]
    n_buses = first(size(data.bus_type))
    n_lccs = size(data.lcc.p_set, 1)
    total_state = total_bus_state + 4 * n_lccs

    sizehint!(rows, 4 * SparseArrays.nnz(Y_bus_eff) + 17 * n_lccs)
    sizehint!(cols, 4 * SparseArrays.nnz(Y_bus_eff) + 17 * n_lccs)
    sizehint!(vals, 4 * SparseArrays.nnz(Y_bus_eff) + 17 * n_lccs)

    # Y_bus pattern blocks (block-size determined by bus_block_size)
    Yrows = SparseArrays.rowvals(Y_bus_eff)
    for col in 1:n_buses
        col_off = bus_state_offset[col]
        col_bs = bus_block_size[col]
        for j in Y_bus_eff.colptr[col]:(Y_bus_eff.colptr[col + 1] - 1)
            row = Yrows[j]
            row_off = bus_state_offset[row]
            row_bs = bus_block_size[row]
            for r in 0:(row_bs - 1)
                for c in 0:(col_bs - 1)
                    push!(rows, J_INDEX_TYPE(row_off + r))
                    push!(cols, J_INDEX_TYPE(col_off + c))
                    push!(vals, 0.0)
                end
            end
        end
    end

    # Distributed-slack cross terms
    for (ref_bus, subnetwork_buses) in subnetworks
        ref_off = bus_state_offset[ref_bus]
        for bus_k in subnetwork_buses
            c_k = bus_slack_participation_factors[bus_k]
            c_k == 0.0 && continue
            bus_k == ref_bus && continue
            k_off = bus_state_offset[bus_k]
            # ∂F_k_r/∂x[ref_off] and ∂F_k_i/∂x[ref_off]
            push!(rows, J_INDEX_TYPE(k_off));     push!(cols, J_INDEX_TYPE(ref_off)); push!(vals, 0.0)
            push!(rows, J_INDEX_TYPE(k_off + 1)); push!(cols, J_INDEX_TYPE(ref_off)); push!(vals, 0.0)
        end
    end

    # LCC tail entries — structural zeros for now; PR 4 fills the math
    if n_lccs > 0
        _create_rect_ci_lcc_structure!(
            rows, cols, vals, data, bus_state_offset, total_bus_state,
        )
    end

    return SparseArrays.sparse(rows, cols, vals, total_state, total_state)
end

"""Populate the off-diagonal Y_bus 2×2 (or 3×2 / 2×3) blocks ONCE. Off-diagonals
are constant across NR iterations. Diagonal blocks are state-dependent and updated
each iteration in `_update_rect_ci_jacobian_values!`."""
function _populate_constant_yb_blocks!(
    Jv::SparseMatrixCSC{Float64, J_INDEX_TYPE},
    Y_bus_eff::SparseMatrixCSC{ComplexF64, Int},
    bus_state_offset::Vector{REC_INDEX_TYPE},
    bus_block_size::Vector{Int8},
    bus_types::AbstractVector{PSY.ACBusTypes},
)
    n_buses = length(bus_types)
    Yvals = SparseArrays.nonzeros(Y_bus_eff)
    Yrows = SparseArrays.rowvals(Y_bus_eff)
    for col in 1:n_buses
        col_off = Int(bus_state_offset[col])
        for j in Y_bus_eff.colptr[col]:(Y_bus_eff.colptr[col + 1] - 1)
            row = Yrows[j]
            row_off = Int(bus_state_offset[row])
            y = Yvals[j]
            g = real(y); b = imag(y)
            # F = I_spec - I_inj; ∂I_inj/∂e = G, ∂I_inj/∂f = -B for real part
            # ∂I_inj_r/∂e_col = g; ∂I_inj_r/∂f_col = -b
            # ∂I_inj_i/∂e_col = b; ∂I_inj_i/∂f_col = g
            # F has -I_inj, so the J entries are negated.
            Jv[row_off,     col_off]     = -g    # ∂F_r/∂e_col
            Jv[row_off,     col_off + 1] =  b    # ∂F_r/∂f_col
            Jv[row_off + 1, col_off]     = -b    # ∂F_i/∂e_col
            Jv[row_off + 1, col_off + 1] = -g    # ∂F_i/∂f_col
            # If PV target row, the PV's 3rd row (ΔV²) has no Y_bus contribution.
            # If PV target column, the PV's 3rd column has no off-diagonal Y_bus contribution.
            # Both stay at the structural zeros from pattern construction.
        end
    end
    return
end
```

- [ ] **Step 3.1.2:** Add LCC structure stub (real entries in PR 4):

```julia
function _create_rect_ci_lcc_structure!(
    rows::Vector{J_INDEX_TYPE},
    cols::Vector{J_INDEX_TYPE},
    vals::Vector{Float64},
    data::ACPowerFlowData,
    bus_state_offset::Vector{REC_INDEX_TYPE},
    total_bus_state::Int,
)
    # Mirror polar `_create_jacobian_matrix_structure_lcc` but using bus_state_offset
    # for AC-side columns. 17 entries per LCC (same as polar). Filled in PR 4.
    for (i, (fb, tb)) in enumerate(data.lcc.bus_indices)
        col_e_fb = bus_state_offset[fb]
        col_f_fb = bus_state_offset[fb] + 1
        col_e_tb = bus_state_offset[tb]
        offset_lcc = total_bus_state + (i - 1) * 4
        idx_tap_r = offset_lcc + 1
        idx_tap_i = offset_lcc + 2
        idx_alpha_r = offset_lcc + 3
        idx_alpha_i = offset_lcc + 4
        # Rows (matching polar idx_p_fb=col_e_fb etc.; total 17 entries)
        rcv = [
            (col_e_fb, col_e_fb, 0.0),
            (col_f_fb, col_e_fb, 0.0),
            (col_e_fb, idx_tap_r, 0.0),
            (col_e_fb, idx_alpha_r, 0.0),
            (col_f_fb, idx_tap_r, 0.0),
            (col_f_fb, idx_alpha_r, 0.0),
            (idx_tap_r, col_e_fb, 0.0),
            (idx_tap_i, col_e_fb, 0.0),
            (idx_tap_i, col_e_tb, 0.0),
            (idx_tap_r, idx_tap_r, 0.0),
            (idx_tap_r, idx_alpha_r, 0.0),
            (idx_tap_i, idx_tap_r, 0.0),
            (idx_tap_i, idx_tap_i, 0.0),
            (idx_tap_i, idx_alpha_r, 0.0),
            (idx_tap_i, idx_alpha_i, 0.0),
            (idx_alpha_r, idx_alpha_r, 1.0),
            (idx_alpha_i, idx_alpha_i, 1.0),
        ]
        for (r, c, v) in rcv
            push!(rows, J_INDEX_TYPE(r))
            push!(cols, J_INDEX_TYPE(c))
            push!(vals, v)
        end
    end
    return
end
```

### Task 3.2: Per-iteration values update (no LCC tail body yet)

- [ ] **Step 3.2.1:** Append values updater:

```julia
function _update_rect_ci_jacobian_values!(
    Jv::SparseMatrixCSC{Float64, J_INDEX_TYPE},
    data::ACPowerFlowData,
    Y_bus_eff::SparseMatrixCSC{ComplexF64, Int},
    bus_slack_participation_factors::SparseVector{Float64, Int},
    subnetworks::Dict{Int64, Vector{Int64}},
    bus_state_offset::Vector{REC_INDEX_TYPE},
    bus_block_size::Vector{Int8},
    total_bus_state::Int,
    time_step::Int64,
)
    n_buses = first(size(data.bus_type))
    bus_types = view(data.bus_type, :, time_step)

    # Diagonal blocks: state-dependent for PQ/PV.
    # REF diagonal blocks were populated once with constant values.
    for i in 1:n_buses
        off = Int(bus_state_offset[i])
        bt = bus_types[i]
        if bt == PSY.ACBusTypes.PQ
            _update_pq_diag_block!(Jv, off, data, i, Y_bus_eff, time_step)
        elseif bt == PSY.ACBusTypes.PV
            _update_pv_diag_block!(Jv, off, data, i, Y_bus_eff, time_step)
        elseif bt == PSY.ACBusTypes.REF
            _update_ref_diag_block!(Jv, off, data, i, Y_bus_eff, time_step)
        end
    end

    # Distributed-slack cross terms
    for (ref_bus, subnetwork_buses) in subnetworks
        ref_off = Int(bus_state_offset[ref_bus])
        for bus_k in subnetwork_buses
            c_k = bus_slack_participation_factors[bus_k]
            c_k == 0.0 && continue
            bus_k == ref_bus && continue
            k_off = Int(bus_state_offset[bus_k])
            Vm = data.bus_magnitude[bus_k, time_step]
            θ = data.bus_angles[bus_k, time_step]
            e_k = Vm * cos(θ)
            f_k = Vm * sin(θ)
            V_sq = e_k^2 + f_k^2
            Jv[k_off,     ref_off] = c_k * e_k / V_sq
            Jv[k_off + 1, ref_off] = c_k * f_k / V_sq
        end
    end

    # LCC tail — full math in PR 4.
    return
end

function _update_pq_diag_block!(Jv, off, data, i, Y_bus_eff, time_step)
    Vm = data.bus_magnitude[i, time_step]
    θ = data.bus_angles[i, time_step]
    e = Vm * cos(θ); f = Vm * sin(θ)
    V_sq = e^2 + f^2
    P = data.bus_active_power_injections[i, time_step] -
        data.bus_active_power_withdrawals[i, time_step]
    Q = data.bus_reactive_power_injections[i, time_step] -
        data.bus_reactive_power_withdrawals[i, time_step]
    g_ii = real(Y_bus_eff[i, i]); b_ii = imag(Y_bus_eff[i, i])
    # I_spec_r = (P e + Q f) / V², I_spec_i = (P f - Q e) / V²
    # ∂I_spec_r/∂e = (P - 2 e I_spec_r) / V²
    # ∂I_spec_r/∂f = (Q - 2 f I_spec_r) / V²
    # ∂I_spec_i/∂e = (-Q - 2 e I_spec_i) / V²
    # ∂I_spec_i/∂f = (P - 2 f I_spec_i) / V²
    # Plus the constant Y_bus diagonal contribution (already populated).
    Is_r = (P * e + Q * f) / V_sq
    Is_i = (P * f - Q * e) / V_sq
    inv_V_sq = 1.0 / V_sq
    # Overwrite (replace, not add — block values are owned by this routine):
    Jv[off,     off]     = (P  - 2*e*Is_r) * inv_V_sq + (-g_ii)
    Jv[off,     off + 1] = (Q  - 2*f*Is_r) * inv_V_sq + ( b_ii)
    Jv[off + 1, off]     = (-Q - 2*e*Is_i) * inv_V_sq + (-b_ii)
    Jv[off + 1, off + 1] = (P  - 2*f*Is_i) * inv_V_sq + (-g_ii)
    return
end

function _update_pv_diag_block!(Jv, off, data, i, Y_bus_eff, time_step)
    Vm = data.bus_magnitude[i, time_step]
    θ = data.bus_angles[i, time_step]
    e = Vm * cos(θ); f = Vm * sin(θ)
    V_sq = e^2 + f^2
    P = data.bus_active_power_injections[i, time_step] -
        data.bus_active_power_withdrawals[i, time_step]
    Q = data.bus_reactive_power_injections[i, time_step] -
        data.bus_reactive_power_withdrawals[i, time_step]
    g_ii = real(Y_bus_eff[i, i]); b_ii = imag(Y_bus_eff[i, i])
    Is_r = (P * e + Q * f) / V_sq
    Is_i = (P * f - Q * e) / V_sq
    inv_V_sq = 1.0 / V_sq
    Jv[off,     off]     = (P  - 2*e*Is_r) * inv_V_sq + (-g_ii)
    Jv[off,     off + 1] = (Q  - 2*f*Is_r) * inv_V_sq + ( b_ii)
    Jv[off + 1, off]     = (-Q - 2*e*Is_i) * inv_V_sq + (-b_ii)
    Jv[off + 1, off + 1] = (P  - 2*f*Is_i) * inv_V_sq + (-g_ii)
    # Q-column entries:
    # ∂I_spec_r/∂Q =  f / V²,  ∂I_spec_i/∂Q = -e / V²
    Jv[off,     off + 2] =  f * inv_V_sq
    Jv[off + 1, off + 2] = -e * inv_V_sq
    # ΔV² row: ∂ΔV²/∂e = -2e,  ∂ΔV²/∂f = -2f,  ∂ΔV²/∂Q = 0 (kept as structural zero)
    Jv[off + 2, off]     = -2 * e
    Jv[off + 2, off + 1] = -2 * f
    # Jv[off + 2, off + 2] stays 0
    return
end

function _update_ref_diag_block!(Jv, off, data, i, Y_bus_eff, time_step)
    Vm = data.bus_magnitude[i, time_step]
    θ = data.bus_angles[i, time_step]
    e_r = Vm * cos(θ); f_r = Vm * sin(θ)
    V_sq = e_r^2 + f_r^2
    g_ii = real(Y_bus_eff[i, i]); b_ii = imag(Y_bus_eff[i, i])
    # x[off] = P_gen, x[off+1] = Q_gen; columns (e_R, f_R) are not state — Y_bus diag
    # contribution must still be present, but the dependency variables here are P_gen/Q_gen.
    # ∂F_r/∂P_gen = e_r / V², ∂F_r/∂Q_gen = f_r / V²
    # ∂F_i/∂P_gen = f_r / V², ∂F_i/∂Q_gen = -e_r / V²
    inv_V_sq = 1.0 / V_sq
    Jv[off,     off]     =  e_r * inv_V_sq
    Jv[off,     off + 1] =  f_r * inv_V_sq
    Jv[off + 1, off]     =  f_r * inv_V_sq
    Jv[off + 1, off + 1] = -e_r * inv_V_sq
    return
end
```

- [ ] **Step 3.2.2:** Include in `src/PowerFlows.jl` after the residual file.

### Task 3.3: Jacobian unit tests + FD verification

**Files:**
- Create: `test/test_rectangular_ci_jacobian.jl`
- Modify: `test/runtests.jl`

- [ ] **Step 3.3.1:** Create `test/test_rectangular_ci_jacobian.jl`:

```julia
@testset "Rectangular CI Jacobian" begin
    using PowerFlows: ACRectangularCIResidual, ACRectangularCIJacobian,
                      RectangularCurrentInjectionACPowerFlow, rect_initial_state!
    using SparseArrays
    using LinearAlgebra: norm
    using PowerSimulationsTestSystems

    @testset "FD match — c_sys5 at flat start" begin
        sys = build_system(PSITestSystems, "c_sys5")
        data = PowerFlowData(ACPowerFlow{RectangularCurrentInjectionACPowerFlow}(), sys)
        R = ACRectangularCIResidual(data, 1)
        J = ACRectangularCIJacobian(R, 1)
        x = Vector{Float64}(undef, length(R.Rv))
        rect_initial_state!(x, data, R.bus_state_offset, R.bus_block_size, 1)
        J(1); R(x, 1)
        # Finite difference
        ε = 1e-6
        Jfd = zeros(length(R.Rv), length(R.Rv))
        for k in 1:length(R.Rv)
            x_plus  = copy(x); x_plus[k]  += ε; R(x_plus, 1); F_plus  = copy(R.Rv)
            x_minus = copy(x); x_minus[k] -= ε; R(x_minus, 1); F_minus = copy(R.Rv)
            Jfd[:, k] = (F_plus - F_minus) / (2ε)
        end
        # Restore residual to x
        R(x, 1); J(1)
        Janalytic = Array(J.Jv)
        # Tolerance loose: FD with ε=1e-6 gives ~1e-5 accuracy
        @test maximum(abs.(Janalytic - Jfd)) < 1e-4
    end

    @testset "Y_bus off-diagonal blocks are constant across iterations" begin
        sys = build_system(PSITestSystems, "c_sys5")
        data = PowerFlowData(ACPowerFlow{RectangularCurrentInjectionACPowerFlow}(), sys)
        R = ACRectangularCIResidual(data, 1)
        J = ACRectangularCIJacobian(R, 1)
        x = Vector{Float64}(undef, length(R.Rv))
        rect_initial_state!(x, data, R.bus_state_offset, R.bus_block_size, 1)
        J(1)
        J_first = copy(J.Jv)
        # Perturb state and update Jacobian
        x .+= 0.01 .* randn(length(x))
        R(x, 1); J(1)
        J_second = copy(J.Jv)
        # Off-diagonal Y_bus block entries should be identical.
        # Pick a known off-diagonal: bus 1's neighbors not at bus 1's own block.
        for i in 1:5, j in 1:5
            i == j && continue
            row_off = Int(R.bus_state_offset[i])
            col_off = Int(R.bus_state_offset[j])
            for dr in 0:1, dc in 0:1
                @test J_first[row_off + dr, col_off + dc] ==
                      J_second[row_off + dr, col_off + dc]
            end
        end
    end
end
```

- [ ] **Step 3.3.2:** Run:

```sh
julia --project=test -e 'using Pkg; Pkg.test(test_args=["test_rectangular_ci_jacobian"])'
```

---

## PR 4 — LCC tail integration

### Task 4.1: LCC residual + Jacobian chain-rule

**Files:**
- Create: `src/rectangular_ci_lcc.jl`
- Modify: `src/PowerFlows.jl`
- Modify: `src/rectangular_ci_power_flow_jacobian.jl` (call `_set_entries_for_lcc_rect!` from updater)

- [ ] **Step 4.1.1:** Create `src/rectangular_ci_lcc.jl`:

```julia
"""
Chain-rule the existing polar LCC partial-derivative helpers
(`_calculate_dQ_d{V,t,α}_lcc`) through ∂Vm/∂e = e/|V|, ∂Vm/∂f = f/|V|, and write
into the LCC tail block of the rectangular Jacobian.

Mirrors `_set_entries_for_lcc` in `ac_power_flow_jacobian.jl` exactly except that
the polar-bus column slots `idx_p_fb` (the Vm slot) and `idx_q_fb` (the Va slot)
are replaced by the rectangular `col_e_fb`, `col_f_fb` with the appropriate chain
rule. Existing polar helper functions are reused verbatim.
"""
function _set_entries_for_lcc_rect!(
    data::ACPowerFlowData,
    Jv::SparseMatrixCSC{Float64, J_INDEX_TYPE},
    bus_state_offset::Vector{REC_INDEX_TYPE},
    total_bus_state::Int,
    time_step::Int,
)
    sqrt6_div_pi = sqrt(6) / π
    for (i, (fb, tb)) in enumerate(data.lcc.bus_indices)
        col_e_fb = Int(bus_state_offset[fb])
        col_f_fb = col_e_fb + 1
        col_e_tb = Int(bus_state_offset[tb])
        col_f_tb = col_e_tb + 1
        offset_lcc = total_bus_state + (i - 1) * 4
        idx_tap_r = offset_lcc + 1
        idx_tap_i = offset_lcc + 2
        idx_alpha_r = offset_lcc + 3
        idx_alpha_i = offset_lcc + 4

        i_dc = max(data.lcc.i_dc[i, time_step], 1e-9)
        tap_r = data.lcc.rectifier.tap[i, time_step]
        tap_i = data.lcc.inverter.tap[i, time_step]
        alpha_r = data.lcc.rectifier.thyristor_angle[i, time_step]
        alpha_i = data.lcc.inverter.thyristor_angle[i, time_step]
        phi_r = data.lcc.rectifier.phi[i, time_step]
        xtr_r = data.lcc.rectifier.transformer_reactance[i]
        Vm_fb = data.bus_magnitude[fb, time_step]
        Vm_tb = data.bus_magnitude[tb, time_step]
        bus_type_fb = data.bus_type[fb, time_step]
        bus_type_tb = data.bus_type[tb, time_step]

        cos_alpha_r = cos(alpha_r)
        sin_alpha_r = sin(alpha_r)
        cos_alpha_i = cos(alpha_i)
        sin_alpha_i = sin(alpha_i)

        e_fb = Vm_fb == 0.0 ? 0.0 : Vm_fb * cos(data.bus_angles[fb, time_step])
        f_fb = Vm_fb == 0.0 ? 0.0 : Vm_fb * sin(data.bus_angles[fb, time_step])
        e_tb = Vm_tb == 0.0 ? 0.0 : Vm_tb * cos(data.bus_angles[tb, time_step])
        f_tb = Vm_tb == 0.0 ? 0.0 : Vm_tb * sin(data.bus_angles[tb, time_step])
        inv_Vm_fb = Vm_fb == 0.0 ? 0.0 : 1.0 / Vm_fb
        inv_Vm_tb = Vm_tb == 0.0 ? 0.0 : 1.0 / Vm_tb
        de_dVm_fb = e_fb * inv_Vm_fb   # e/Vm
        df_dVm_fb = f_fb * inv_Vm_fb   # f/Vm
        de_dVm_tb = e_tb * inv_Vm_tb
        df_dVm_tb = f_tb * inv_Vm_tb

        common_term_fb = Vm_fb * sqrt6_div_pi * i_dc
        common_term_tb = Vm_tb * sqrt6_div_pi * (-i_dc)
        common_term_tap_r = tap_r * sqrt6_div_pi * i_dc * cos_alpha_r
        common_term_alpha_r = -common_term_fb * tap_r * sin_alpha_r

        # Polar code wrote four "AC-bus-affecting" entries via idx_p_fb (Vm slot)
        # and idx_q_fb (Va slot). For rectangular, chain rule splits each into two:
        # ∂/∂e_fb = (∂/∂Vm_fb) * (e_fb/Vm_fb)
        # ∂/∂f_fb = (∂/∂Vm_fb) * (f_fb/Vm_fb)
        # (The polar "Va slot" entries were ∂_Q∂t etc. — those are unaffected by V_a
        #  in polar; in rectangular they translate similarly.)

        if bus_type_fb == PSY.ACBusTypes.PQ
            dP_dVm = common_term_tap_r            # = ∂P_lcc_from / ∂Vm_fb in polar
            dQ_dVm = _calculate_dQ_dV_lcc(tap_r, i_dc, xtr_r, Vm_fb, phi_r)
            # Rectangular: F_r row of bus fb is idx col_e_fb; F_i row is col_f_fb.
            # The "P injection ∂Q/∂V" maps through the residual rewrite — see spec.
            # Note: rectangular residual splits ΔP/ΔQ via complex current mismatch.
            # The simplest mirror is to compute ∂(I_spec_r, I_spec_i)/∂(e_fb, f_fb)
            # contributions stemming from LCC's P/Q injection at bus fb.
            # Reusing polar derivative helpers with chain rule:
            Jv[col_e_fb, col_e_fb] += dP_dVm * de_dVm_fb   # rough approximation
            Jv[col_e_fb, col_f_fb] += dP_dVm * df_dVm_fb
            Jv[col_f_fb, col_e_fb] += dQ_dVm * de_dVm_fb
            Jv[col_f_fb, col_f_fb] += dQ_dVm * df_dVm_fb

            dQ_dt = _calculate_dQ_dt_lcc(tap_r, i_dc, xtr_r, Vm_fb, phi_r)
            dQ_da = _calculate_dQ_dα_lcc(tap_r, i_dc, xtr_r, Vm_fb, phi_r, alpha_r)
            Jv[col_f_fb, idx_tap_r]   = dQ_dt
            Jv[col_f_fb, idx_alpha_r] = dQ_da

            Jv[idx_tap_r, col_e_fb] = common_term_tap_r * de_dVm_fb
            Jv[idx_tap_r, col_f_fb] = common_term_tap_r * df_dVm_fb
            Jv[idx_tap_i, col_e_fb] = common_term_tap_r * de_dVm_fb
            Jv[idx_tap_i, col_f_fb] = common_term_tap_r * df_dVm_fb
        end

        if bus_type_fb in (PSY.ACBusTypes.PQ, PSY.ACBusTypes.PV)
            Jv[col_e_fb, idx_tap_r]   = common_term_fb * cos_alpha_r
            Jv[col_e_fb, idx_alpha_r] = common_term_alpha_r
        end

        if bus_type_tb == PSY.ACBusTypes.PQ
            common_term_tap_i_at_tb = tap_i * sqrt6_div_pi * (-i_dc) * cos_alpha_i
            Jv[idx_tap_i, col_e_tb] = common_term_tap_i_at_tb * de_dVm_tb
            Jv[idx_tap_i, col_f_tb] = common_term_tap_i_at_tb * df_dVm_tb
        end

        Jv[idx_tap_r, idx_tap_r]     = common_term_fb * cos_alpha_r
        Jv[idx_tap_r, idx_alpha_r]   = common_term_alpha_r
        Jv[idx_tap_i, idx_tap_r]     = common_term_fb * cos_alpha_r
        Jv[idx_tap_i, idx_tap_i]     = common_term_tb * cos_alpha_i
        Jv[idx_tap_i, idx_alpha_r]   = common_term_alpha_r
        Jv[idx_tap_i, idx_alpha_i]   = -common_term_tb * tap_i * sin_alpha_i
        # idx_alpha_r and idx_alpha_i diagonals are 1.0 (already populated at construction).
    end
    return
end
```

NOTE FOR IMPLEMENTER: the exact chain-rule mapping for the LCC P-injection contribution to `Jv[col_e_fb, ...]` and `Jv[col_f_fb, ...]` requires careful derivation matching the augmented current-injection rewrite of the LCC's contribution to bus fb's I_spec. The above is a first cut — verify with FD test on a synthetic 2-LCC system before claiming correctness. The FD test in Task 4.3 is the hard validation.

- [ ] **Step 4.1.2:** Wire up: call `_set_entries_for_lcc_rect!` from the bottom of `_update_rect_ci_jacobian_values!`.

- [ ] **Step 4.1.3:** Include `rectangular_ci_lcc.jl` in `src/PowerFlows.jl` after Jacobian file.

### Task 4.2: LCC parity tests

**Files:**
- Create: `test/test_rectangular_ci_lcc.jl`

- [ ] **Step 4.2.1:** Use existing HVDC fixtures from `test/test_hvdc.jl` — construct a 2-bus LCC system, run both polar NR and rectangular CI, assert voltage parity and branch flow parity.

```julia
@testset "Rectangular CI LCC parity vs polar" begin
    using PowerFlows
    using PowerSimulationsTestSystems

    sys = build_system(PSITestSystems, "c_sys_lcc")   # adapt to actual LCC fixture
    pf_polar = ACPowerFlow{NewtonRaphsonACPowerFlow}()
    pf_rect = ACPowerFlow{RectangularCurrentInjectionACPowerFlow}()

    res_polar = solve_power_flow(pf_polar, sys)
    res_rect = solve_power_flow(pf_rect, deepcopy(sys))

    @test all(abs.(res_polar["bus_results"].Vm - res_rect["bus_results"].Vm) .< 1e-6)
    @test all(abs.(res_polar["bus_results"].θ  - res_rect["bus_results"].θ)  .< 1e-6)
end
```

- [ ] **Step 4.2.2:** Run; iterate on chain-rule mapping until passing.

### Task 4.3: FD verification on LCC system

- [ ] **Step 4.3.1:** Extend the FD test in `test_rectangular_ci_jacobian.jl` to a synthetic 2-bus LCC fixture. This is the hard test that catches chain-rule errors.

---

## PR 5 — Solver wiring + full parity suite

### Task 5.1: `initialize_power_flow_variables_rect` helper

**Files:**
- Modify: `src/power_flow_setup.jl`

- [ ] **Step 5.1.1:** Add the rect variant alongside `initialize_power_flow_variables`:

```julia
function initialize_power_flow_variables_rect(
    pf::ACPowerFlow{RectangularCurrentInjectionACPowerFlow},
    data::ACPowerFlowData,
    time_step::Int64;
    x0::Union{Nothing, Vector{Float64}} = nothing,
    validate_voltage_magnitudes::Bool = DEFAULT_VALIDATE_VOLTAGES,
    vm_validation_range::MinMax = DEFAULT_VALIDATION_RANGE,
)
    R = ACRectangularCIResidual(data, time_step)
    J = ACRectangularCIJacobian(R, time_step)
    x = Vector{Float64}(undef, length(R.Rv))
    if isnothing(x0)
        # Apply enhanced flat start similar to polar: V=1 at PQ buses, retain V_set at PV/REF
        if get_enhanced_flat_start(pf)
            _enhanced_flat_start_rect!(data, time_step)
        end
        rect_initial_state!(x, data, R.bus_state_offset, R.bus_block_size, time_step)
    else
        copyto!(x, x0)
    end
    R(x, time_step)
    J(time_step)
    return R, J, x
end

function _enhanced_flat_start_rect!(data::ACPowerFlowData, time_step::Int64)
    bus_types = view(data.bus_type, :, time_step)
    for i in eachindex(bus_types)
        if bus_types[i] == PSY.ACBusTypes.PQ
            isnan(data.bus_magnitude[i, time_step]) && (data.bus_magnitude[i, time_step] = 1.0)
            isnan(data.bus_angles[i, time_step]) && (data.bus_angles[i, time_step] = 0.0)
        end
    end
    return
end
```

### Task 5.2: `_newton_power_flow` dispatch

**Files:**
- Modify: `src/power_flow_method.jl`

- [ ] **Step 5.2.1:** Add new dispatch (mirror polar version):

```julia
function _newton_power_flow(
    pf::ACPowerFlow{RectangularCurrentInjectionACPowerFlow},
    data::ACPowerFlowData,
    time_step::Int64;
    tol::Float64 = DEFAULT_NR_TOL,
    maxIterations::Int = DEFAULT_NR_MAX_ITER,
    validate_voltage_magnitudes::Bool = DEFAULT_VALIDATE_VOLTAGES,
    vm_validation_range::MinMax = DEFAULT_VALIDATION_RANGE,
    refinement_threshold::Float64 = DEFAULT_REFINEMENT_THRESHOLD,
    refinement_eps::Float64 = DEFAULT_REFINEMENT_EPS,
    iwamoto::Bool = false,
    factor::Float64 = DEFAULT_TRUST_REGION_FACTOR,
    eta::Float64 = DEFAULT_TRUST_REGION_ETA,
    autoscale::Bool = DEFAULT_AUTOSCALE,
    iwamoto_fallback::Bool = DEFAULT_IWAMOTO_FALLBACK,
    step_strategy::Symbol = :simple,
    x0::Union{Vector{Float64}, Nothing} = nothing,
    _ignored...,
)
    init_kwargs = isnothing(x0) ?
        (; validate_voltage_magnitudes, vm_validation_range) :
        (; validate_voltage_magnitudes, vm_validation_range, x0)
    residual, J, x0_init = initialize_power_flow_variables_rect(
        pf, data, time_step; init_kwargs...)
    converged = norm(residual.Rv, Inf) < tol
    i = 0
    if !converged
        linSolveCache = KLULinSolveCache(J.Jv)
        symbolic_factor!(linSolveCache, J.Jv)
        stateVector = StateVectorCache(x0_init, residual.Rv)
        # Step-strategy dispatch
        T_strategy = step_strategy == :trust_region ? TrustRegionACPowerFlow :
                                                       NewtonRaphsonACPowerFlow
        converged, i = _run_power_flow_method(
            time_step, stateVector, linSolveCache, residual, J,
            T_strategy;
            tol, maxIterations, validate_voltage_magnitudes,
            vm_validation_range, refinement_threshold, refinement_eps,
            iwamoto, factor, eta, autoscale, iwamoto_fallback,
        )
    end
    return _finalize_power_flow(
        converged, i, "RectangularCurrentInjectionACPowerFlow",
        residual, data, J.Jv, time_step,
    )
end
```

NOTE FOR IMPLEMENTER: `_run_power_flow_method` is currently dispatched on the solver type. Both `ACPowerFlowResidual`/`ACPowerFlowJacobian` and `ACRectangularCIResidual`/`ACRectangularCIJacobian` must implement the same callable interface (which they do). The driver should work unmodified — but verify by running tests.

### Task 5.3: Full parity test suite

**Files:**
- Create: `test/test_rectangular_ci_power_flow.jl`

- [ ] **Step 5.3.1:** Mirror the existing `test_solve_power_flow.jl` fixtures (RTS, IEEE, ACTIVSg2000) and add parity assertions vs polar NR.

```julia
@testset "Rectangular CI vs Polar NR parity" begin
    fixtures = [
        ("c_sys5", PSITestSystems),
        ("ieee14", PSISystems),
        ("ieee30", PSISystems),
        ("ieee118", PSISystems),
        ("ieee300", PSISystems),
        ("ACTIVSg2000", PSISystems),
    ]
    for (name, src) in fixtures
        @testset "$name" begin
            sys_polar = build_system(src, name)
            sys_rect = deepcopy(sys_polar)
            pf_polar = ACPowerFlow{NewtonRaphsonACPowerFlow}()
            pf_rect = ACPowerFlow{RectangularCurrentInjectionACPowerFlow}()
            res_polar = solve_power_flow(pf_polar, sys_polar)
            res_rect = solve_power_flow(pf_rect, sys_rect)
            @test all(abs.(res_polar["bus_results"].Vm - res_rect["bus_results"].Vm) .< 1e-7)
            @test all(abs.(res_polar["bus_results"].θ  - res_rect["bus_results"].θ)  .< 1e-7)
        end
    end
end
```

---

## PR 6 — Iwamoto + Trust Region on rectangular CI

### Task 6.1: Extend `test_iterative_methods.jl`

- [ ] Add Iwamoto cases: pass `solver_settings = Dict(:iwamoto => true)` to `ACPowerFlow{RectangularCurrentInjectionACPowerFlow}` and check convergence on the ill-conditioned fixtures.
- [ ] Add Trust Region cases: pass `solver_settings = Dict(:step_strategy => :trust_region)` and check convergence.

(The drivers should work unmodified. If they don't, debug the iteration-strategy dispatch.)

---

## PR 7 — Benchmarks + docs

### Task 7.1: Performance benchmarks

**Files:**
- Modify: `test/performance/performance_test.jl`

- [ ] Add rectangular CI to the existing benchmark sweep alongside polar NR. Track: time per NR iteration, allocations per iteration, total iterations.

### Task 7.2: Docs

**Files:**
- Create: `docs/src/explanation/rectangular_current_injection.md`
- Modify: `docs/src/reference/api/public.md` (autodocs picks up the new export)

- [ ] Write Diataxis explanation: why rectangular CI, when to choose it, performance characteristics. Reference the design spec.

---

## Final tasks

### Task F.1: Format

```sh
julia --project=scripts/formatter -e 'include("scripts/formatter/formatter_code.jl")'
```

### Task F.2: Full test suite

```sh
julia --project=test test/runtests.jl
```

Expected: all tests pass, including the new rectangular CI suite.

### Task F.3: Stage (do NOT commit)

```sh
git add src/ test/ docs/superpowers/
git status
```

Report passing state to the user.
