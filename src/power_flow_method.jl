"""Cache for non-linear methods.

# Fields
- `x::Vector{Float64}`: the current state vector.
- `r::Vector{Float64}`: the current residual.
- `Δx_nr::Vector{Float64}`: the step under the Newton-Raphson method.
The remainder of the fields are only used in the `TrustRegionACPowerFlow`:
- `r_predict::Vector{Float64}`: the predicted residual at `x+Δx_proposed`,
    under a linear approximation: i.e `J_x⋅(x+Δx_proposed)`.
- `Δx_proposed::Vector{Float64}`: the suggested step `Δx`, selected among `Δx_nr`,
    `Δx_cauchy`, and the dogleg interpolation between the two. The first is chosen when
    `x+Δx_nr` is inside the trust region, the second when both `x+Δx_cauchy`
    and `x+Δx_nr` are outside the trust region, and the third when `x+Δx_cauchy`
    is inside and `x+Δx_nr` outside. The dogleg step selects the point where the line
    from `x+Δx_cauchy` to `x+Δx_nr` crosses the boundary of the trust region.
- `Δx_cauchy::Vector{Float64}`: the step to the Cauchy point if the Cauchy point
    lies within the trust region, otherwise a step in that direction."""
struct StateVectorCache
    x::Vector{Float64}
    r::Vector{Float64} # residual
    r_predict::Vector{Float64} # predicted residual
    Δx_proposed::Vector{Float64} # proposed Δx: Cauchy, NR, or dogleg step.
    Δx_cauchy::Vector{Float64} # Cauchy step
    Δx_nr::Vector{Float64} # Newton-Raphson step
    d::Vector{Float64}
    r_scratch::Vector{Float64} # residual-length scratch for `_dogleg!`'s `Jv * g`
    # Persistent regularized singular-Jacobian fallback `-(JᵀJ + λI)`, reused across repeated
    # fallbacks: `fallback_matrix` keeps a fixed pattern so values are refreshed in place and
    # `fallback_cache`'s symbolic factorization is reused; both are rebuilt only on a pattern shift.
    fallback_cache::Base.RefValue{
        Union{Nothing, PNM.KLULinSolveCache{Float64, J_INDEX_TYPE}},
    }
    fallback_matrix::Base.RefValue{Union{Nothing, SparseMatrixCSC{Float64, J_INDEX_TYPE}}}
    # Whether F and J at the start came from the fused kernel, so a cold rerun evaluates alike.
    fused_start::Base.RefValue{Bool}
end

function StateVectorCache(x0::Vector{Float64}, f0::Vector{Float64})
    x = copy(x0)
    r = copy(f0)
    r_predict = copy(x0)
    Δx_proposed = copy(x0)
    Δx_cauchy = copy(x0)
    Δx_nr = copy(x0)
    return StateVectorCache(
        x, r, r_predict, Δx_proposed, Δx_cauchy, Δx_nr, ones(size(x0)), copy(f0),
        Base.RefValue{Union{Nothing, PNM.KLULinSolveCache{Float64, J_INDEX_TYPE}}}(nothing),
        Base.RefValue{Union{Nothing, SparseMatrixCSC{Float64, J_INDEX_TYPE}}}(nothing),
        Base.RefValue(false),
    )
end

# Reset the buffers a fresh StateVectorCache starts with, so a reused solve is bit-identical:
# `d` (TR autoscale recomputes it; NR leaves it untouched) and the singular-Jacobian fallback.
function _reset_for_reuse!(stateVector::StateVectorCache)
    fill!(stateVector.d, 1.0)
    stateVector.fallback_cache[] = nothing
    stateVector.fallback_matrix[] = nothing
    return
end

"""Arc→bus index maps and branch-flow buffers for `solve_power_flow!`. Valid while `arcs` is
(by identity) the arc axis of `data`'s arc admittance matrices."""
struct ArcFlowScratch
    arcs::Vector{Tuple{Int, Int}}
    fb_ix::Vector{Int}
    tb_ix::Vector{Int}
    V::Vector{ComplexF64}
    Sft::Vector{ComplexF64}
    Stf::Vector{ComplexF64}
end

function ArcFlowScratch(data::ACPowerFlowData)
    Yft = data.power_network_matrix.arc_admittance_from_to
    Ytf = data.power_network_matrix.arc_admittance_to_from
    @assert PNM.get_bus_lookup(Yft) == get_bus_lookup(data)
    arcs = PNM.get_arc_axis(Yft)
    @assert arcs == PNM.get_arc_axis(Ytf)
    n_bus = size(data.bus_angles, 1)
    @assert length(PNM.get_bus_axis(Yft)) == n_bus
    bus_lookup = get_bus_lookup(data)
    fb_ix = [bus_lookup[first(arc)] for arc in arcs]
    tb_ix = [bus_lookup[last(arc)] for arc in arcs]
    n_arc = length(arcs)
    return ArcFlowScratch(
        arcs, fb_ix, tb_ix,
        Vector{ComplexF64}(undef, n_bus),
        Vector{ComplexF64}(undef, n_arc),
        Vector{ComplexF64}(undef, n_arc),
    )
end

"""Polar NR/TR workspace stored in `data.polar_nr_cache`, reused across Q-limit retries, time
steps and contingencies. `bus_type_snapshot` holds the bus types the residual's partition was
last derived for (emptied by [`_invalidate_partition!`](@ref)). `residual` and `J` do not store
`data`: this
cache hangs off `data`, so a back-reference would form a cycle. `arc_flows` lets a reused
`solve_power_flow!` skip rebuilding its branch-flow scratch. `lean` is the slot whose plan the
KLU cache was given, for [`_align_lean_plan!`](@ref). `x0` and `partition` are the reuse path's
start-point and island-partition buffers."""
struct PolarNRCache{C <: PNM.LinearSolverCache} <: AbstractNRCache
    residual::ACPowerFlowResidual
    J::ACPowerFlowJacobian
    linSolveCache::C
    stateVector::StateVectorCache
    backend::PNM.LinearSolverType
    bus_type_snapshot::Vector{PSY.ACBusTypes.Value}
    arc_flows::ArcFlowScratch
    lean::LeanPlanSlot
    x0::Vector{Float64}
    partition::SubnetworkScratch
end

"""Copy of a polar NR cache for another worker. Shares the read-only Jacobian index maps, arc
index vectors and lean slot, and owns everything a solve writes. Its KLU cache takes the
slot's pristine plan, never `entry`'s current one: a swapped plan's `q` is `entry`'s own
`lean_q`, which `entry`'s next swap overwrites."""
function _copy_for_task(
    entry::PolarNRCache{<:PNM.KLULinSolveCache},
    slot::LeanPlanSlot,
    data::ACPowerFlowData,
    time_step::Int,
)
    J, sv, af = entry.J, entry.stateVector, entry.arc_flows
    substitutes = IdDict{Any, Any}(
        sv.fallback_cache => typeof(sv.fallback_cache)(nothing),
        sv.fallback_matrix => typeof(sv.fallback_matrix)(nothing),
    )
    for a in (J.od_ptr, J.od_to, J.od_ybus_nz, J.od_jnz, J.diag_jnz, J.diag_ybus_nz,
        af.arcs, af.fb_ix, af.tb_ix)
        substitutes[a] = a
    end
    lin = _polar_jacobian_cache(entry.backend, J.Jv)
    # PNM forbids deepcopy of a KLU cache, so copy the other fields one by one; the shared
    # `substitutes` keeps J's aliases of the residual's vectors inside the copy.
    dup = PolarNRCache(
        Base.deepcopy_internal(entry.residual, substitutes),
        Base.deepcopy_internal(J, substitutes),
        lin,
        Base.deepcopy_internal(sv, substitutes),
        entry.backend,
        copy(entry.bus_type_snapshot),
        Base.deepcopy_internal(af, substitutes),
        entry.lean,
        copy(entry.x0),
        Base.deepcopy_internal(entry.partition, substitutes),
    )
    _seed_lean_plan!(lin, dup.J.Jv, slot, data, time_step)
    return dup
end

# A worker is seeded only from a KLU polar cache; any other seed leaves it to build its own.
_seed_worker!(::ACPowerFlowData, ::Any, ::LeanPlanSlot, ::Int) = nothing
function _seed_worker!(
    worker::ACPowerFlowData,
    seed::PolarNRCache{<:PNM.KLULinSolveCache},
    slot::LeanPlanSlot,
    time_step::Int,
)
    worker.polar_nr_cache[] = _copy_for_task(seed, slot, worker, time_step)
    return
end

"""Mark the retained polar NR cache's island partition stale, so its next reuse re-derives it.
For callers that edit Ybus values in a way that can split or merge islands (zeroing a bridge's
admittances, restoring it) without necessarily changing bus types."""
function _invalidate_partition!(data::ACPowerFlowData)
    _invalidate_partition!(data.polar_nr_cache[])
    for slot in data.worker_slots
        _invalidate_partition!(slot.polar_nr_cache[])
    end
    return
end
_invalidate_partition!(::Nothing) = nothing
function _invalidate_partition!(entry::PolarNRCache)
    empty!(entry.bus_type_snapshot)
    return
end

# After a bus-type or partition change, `klu_refactor` on the inherited pivot order can hit a zero
# pivot (a PV bus's |V| column holds a single -1 where a PQ bus's holds dP/dV, dQ/dV). Free only
# the Numeric: the next `numeric_refactor!` then runs a fresh `klu_factor` on the kept Symbolic,
# or, with a lean plan, a lean refactor whose pivot-ratio check falls back to `klu_factor`.
function _drop_numeric!(c::PNM.KLULinSolveCache)
    PNM.KLUWrapper.drop_numeric!(c)
    return
end
# These backends pivot afresh on every numeric factorization.
_drop_numeric!(::PNM.AAFactorCache) = nothing
_drop_numeric!(::PardisoLinSolveCache) = nothing

# Test-only switch: `false` keeps every polar NR cache on plain KLU, for comparing against it.
const _USE_LEAN_LU = Ref(true)

"""The [`LeanPlanSlot`](@ref) of the Jacobian structure `data` uses at `time_step`, building
its plan on first use. The plan is factored from a separately built Jacobian evaluated at
|V| = 1, θ = 0 (as p3s does), so it depends only on the network, bus types, slack factors and
ZIP loads at `time_step`, never on an iterate: every cache, task and contingency sharing the
slot pivots identically."""
function _lean_plan_slot!(data::ACPowerFlowData, time_step::Int64)
    residual = ACPowerFlowResidual(data, time_step)
    J = ACPowerFlowJacobian(data, residual, time_step)
    # J's structure was just taken from (or stored into) this memo.
    slot = (data.ac_jacobian_structure_cache[]::ACJacobianStructureCache).lean
    slot.tried && return slot
    slot.tried = true
    slot.bus_types = data.bus_type[:, time_step]
    s = J.bus_state
    fill!(s.Vm, 1.0)
    fill!(s.θ, 0.0)
    fill!(s.phasor, one(ComplexF64))
    _update_jacobian_matrix_values!(J, data, time_step)
    _build_lean_plan!(slot, J.Jv, time_step)
    return slot
end

function _build_lean_plan!(
    slot::LeanPlanSlot,
    Jv::SparseMatrixCSC{Float64},
    time_step::Int64,
)
    try
        slot.plan = PNM.KLUWrapper.build_lean_plan(Jv)
        slot.valid = true
    catch e
        e isa LinearAlgebra.SingularException || rethrow()
        n = Threads.atomic_add!(_LEAN_SINGULAR_PLANS, 1) + 1
        @warn "The flat-start Jacobian is singular; this Jacobian structure solves without " *
              "the lean LU ($n such structures so far)." time_step
    end
    return
end

"""Flat-start Jacobians found singular by [`_lean_plan_slot!`](@ref), each a structure memo
whose solves run without the lean LU."""
const _LEAN_SINGULAR_PLANS = Threads.Atomic{Int}(0)

# Symbolic step of a fresh polar NR cache. On a lean plan the cache's own klu_analyze waits for
# its first KLU factorization (a reject, a re-pivot, or a diagnostic `pivoted_factor!`), which a
# solve that stays lean never reaches.
_symbolic_step!(::AbstractACPowerFlow, c, A, ::ACPowerFlowData, ::Int64) =
    symbolic_factor!(c, A)

function _symbolic_step!(
    ::ACPolarPowerFlow,
    c::PNM.KLULinSolveCache{Float64},
    A::SparseMatrixCSC{Float64},
    data::ACPowerFlowData,
    time_step::Int64,
)
    if !_USE_LEAN_LU[]
        PNM.symbolic_factor!(c, A)
        return
    end
    slot = (data.ac_jacobian_structure_cache[]::ACJacobianStructureCache).lean
    if !slot.tried
        slot = _lean_plan_slot!(data, time_step)
    end
    _seed_lean_plan!(c, A, slot, data, time_step)
    return
end

function _seed_lean_plan!(
    c::PNM.KLULinSolveCache{Float64},
    A::SparseMatrixCSC{Float64},
    slot::LeanPlanSlot,
    data::ACPowerFlowData,
    time_step::Int64,
)
    if slot.valid
        PNM.KLUWrapper.defer_symbolic!(c, A)
        PNM.KLUWrapper.set_lean_plan!(c, slot.plan)
        _align_lean_plan!(c, slot, view(data.bus_type, :, time_step))
    else
        PNM.symbolic_factor!(c, A)
    end
    return
end

# Each solve starts on the lean path, whatever paused it in the last one (`numeric_refactor!`).
function _resume_lean!(c::PNM.KLULinSolveCache)
    PNM.KLUWrapper.pause_lean!(c, false)
    return
end
_resume_lean!(::Union{PNM.AAFactorCache, PardisoLinSolveCache}) = nothing

# The plan pivots a PV bus's Q column on its Q row (that column's only entry) and a REF bus's P
# column on its P row. A bus promoted from PV to REF since the plan was built (an island
# reference), or demoted from REF to PV, puts an exact zero on a planned pivot: a certain lean
# reject and a fresh klu_factor. Exchanging the bus's two state columns in the lean steps keeps
# every pivot on its planned row.
function _align_lean_plan!(
    c::PNM.KLULinSolveCache{Float64},
    slot::LeanPlanSlot,
    bus_type::AbstractVector{PSY.ACBusTypes.Value},
)
    PNM.KLUWrapper.has_lean_plan(c) || return
    plan_types = slot.bus_types
    swaps(i) = _swaps_state_columns(plan_types[i], bus_type[i])
    pairs = ((2i - 1, 2i) for i in eachindex(bus_type, plan_types) if swaps(i))
    PNM.KLUWrapper.swap_lean_columns!(c, slot.plan, pairs)
    return
end
_align_lean_plan!(
    ::Union{PNM.AAFactorCache, PardisoLinSolveCache},
    ::LeanPlanSlot,
    ::AbstractVector{PSY.ACBusTypes.Value},
) = nothing

function _swaps_state_columns(planned::PSY.ACBusTypes.Value, now::PSY.ACBusTypes.Value)
    return (planned == PSY.ACBusTypes.PV && now == PSY.ACBusTypes.REF) ||
           (planned == PSY.ACBusTypes.REF && now == PSY.ACBusTypes.PV)
end

_lean_slot(memo::ACJacobianStructureCache) = memo.lean
_lean_slot(::Nothing) = LeanPlanSlot()

const _NO_LEAN_COUNTS = (;
    attempts = 0, rejects = 0, solve_failures = 0, late_analyses = 0, cold_retries = 0)

"""Lean-LU counters of `data`'s polar NR cache, `PNM.KLUWrapper.lean_counts`: refactors
attempted, rejected by the pivot-ratio test, accepted but failing their solve (re-pivoted), and
deferred analyses needed later; plus the solves rerun cold after failing on a reused pivot order
([`_newton_power_flow`](@ref)). Zeros without a cache, or on a non-KLU backend."""
_lean_counts(data::ACPowerFlowData) = _lean_counts(data.polar_nr_cache[])
_lean_counts(::Nothing) = _NO_LEAN_COUNTS
_lean_counts(entry::PolarNRCache) = _lean_counts(entry.linSolveCache)
_lean_counts(c::PNM.KLULinSolveCache) = merge(
    PNM.KLUWrapper.lean_counts(c), (; cold_retries = PNM.KLUWrapper.cold_retries(c)))
_lean_counts(::Union{PNM.AAFactorCache, PardisoLinSolveCache}) = _NO_LEAN_COUNTS

_cold_retries(data::ACPowerFlowData) = _lean_counts(data).cold_retries

# A solve starting on a lean plan or on a KLU numeric kept from an earlier solve pivots on an
# order chosen for other values; only such a solve is rerun cold when it fails.
_reuses_pivot_order(c::PNM.KLULinSolveCache) =
    PNM.KLUWrapper.has_lean_plan(c) || PNM.is_factored(c)
_reuses_pivot_order(::Union{PNM.AAFactorCache, PardisoLinSolveCache}) = false

# p3s accepts a lean factor on its pivot ratio alone (nr_klu.cpp:582-588). A poor lean step costs
# iterations, never a wrong answer: convergence is judged on the residual, and a solve that fails
# on lean factors is rerun cold.
_skips_refinement(c::PNM.KLULinSolveCache) = PNM.KLUWrapper.lean_active(c)
_skips_refinement(::Union{PNM.AAFactorCache, PardisoLinSolveCache}) = false

"""Seed `work`'s Jacobian-structure memo, and with it the lean-LU plan, from `base`'s. For a
working copy whose Ybus is a value copy of `base`'s with the same pattern (PowerTransmission-
SecurityAnalysis), so every copy pivots like `base` instead of on whatever it solves first. An
unbuilt plan is not shared: the copies would race to build it."""
_inherit_jacobian_structure!(work::ACPowerFlowData, base::ACPowerFlowData) =
    _inherit_jacobian_structure!(work, base.ac_jacobian_structure_cache[])
_inherit_jacobian_structure!(::ACPowerFlowData, ::Nothing) = nothing

function _inherit_jacobian_structure!(work::ACPowerFlowData, memo::ACJacobianStructureCache)
    y = work.power_network_matrix.data
    y0 = memo.matrix.data
    (
        SparseArrays.getcolptr(y) == SparseArrays.getcolptr(y0) &&
        SparseArrays.rowvals(y) == SparseArrays.rowvals(y0)
    ) ||
        error(
            "The working copy's Ybus pattern differs from the base's; cannot share its " *
            "Jacobian structure.",
        )
    lean = memo.lean
    if !lean.tried
        lean = LeanPlanSlot()
    end
    work.ac_jacobian_structure_cache[] = ACJacobianStructureCache(
        work.power_network_matrix, memo.slack_slots, memo.structure,
        work.area_interchange,
        lean)
    return
end

_arc_flow_scratch(::Nothing, data::ACPowerFlowData) = ArcFlowScratch(data)

function _arc_flow_scratch(entry::PolarNRCache, data::ACPowerFlowData)
    arcs = PNM.get_arc_axis(data.power_network_matrix.arc_admittance_from_to)
    if entry.arc_flows.arcs === arcs
        return entry.arc_flows
    end
    return ArcFlowScratch(data)
end

function _fill_flow_voltages!(
    V::Vector{ComplexF64},
    ::Nothing,
    data::ACPowerFlowData,
    time_step::Int,
)
    @views V .=
        data.bus_magnitude[:, time_step] .* exp.(1im .* data.bus_angles[:, time_step])
    return
end

# Reuses the last polar evaluation's cis(θ) when its θ is bitwise the column's (exp(iθ) and
# cis(θ) round alike), whichever solver last wrote the column.
function _fill_flow_voltages!(
    V::Vector{ComplexF64},
    entry::PolarNRCache,
    data::ACPowerFlowData,
    time_step::Int,
)
    s = entry.residual.bus_state
    θ = view(data.bus_angles, :, time_step)
    if !s.phasor_valid || !_bitwise_equal(s.θ, θ)
        return _fill_flow_voltages!(V, nothing, data, time_step)
    end
    @views V .= data.bus_magnitude[:, time_step] .* s.phasor
    return
end

function _bitwise_equal(a::Vector{Float64}, b::AbstractVector{Float64})
    length(a) == length(b) || return false
    @inbounds for i in eachindex(a, b)
        a[i] === b[i] || return false
    end
    return true
end

function _ref_set_changed(
    before::Vector{PSY.ACBusTypes.Value},
    after::AbstractVector{PSY.ACBusTypes.Value},
)
    for i in eachindex(before, after)
        if (before[i] == PSY.ACBusTypes.REF) != (after[i] == PSY.ACBusTypes.REF)
            return true
        end
    end
    return false
end

"""Refresh `entry` in place for `time_step`. The island partition is re-derived only after
[`_invalidate_partition!`](@ref) or a change of the REF set; any bus-type change refreshes the
PQ index set and drops the KLU Numeric so the next factorization pivots afresh on the kept
Symbolic. Slack factors are refilled in place every call. Returns `false` when a participating
bus has no distributed-slack slot in `J`'s pattern, so the caller must rebuild."""
function _refresh_polar_residual!(
    entry::PolarNRCache, data::ACPowerFlowData, time_step::Int64,
)
    residual = entry.residual
    J = entry.J
    bus_type = view(data.bus_type, :, time_step)
    snapshot = entry.bus_type_snapshot
    if bus_type != snapshot
        if isempty(snapshot) || _ref_set_changed(snapshot, bus_type)
            subnetworks = residual.subnetworks
            _find_subnetworks_for_reference_buses!(
                subnetworks, entry.partition, data.power_network_matrix.data, bus_type)
            _multi_swing_ref_indices!(
                J.independent_ref, data.bus_type, subnetworks, time_step)
            _slack_jnz!(J.slack_jnz, J.Jv, subnetworks)
        end
        _pq_validate_indices!(residual.validate_indices, bus_type)
        resize!(snapshot, length(bus_type))
        copyto!(snapshot, bus_type)
        _drop_numeric!(entry.linSolveCache)
    end
    spf = residual.bus_slack_participation_factors
    _fill_bus_slack_participation_factors!(
        spf, data, bus_type, residual.subnetworks, time_step)
    _slack_slots_cover(J.slack_jnz, spf, residual.subnetworks, J.independent_ref) ||
        return false
    _refresh_residual_setpoints!(residual, data, time_step)
    return true
end

"""Solve for the Newton-Raphson step, given the factorization object for `J.Jv`
(if non-singular) or its stand-in (if singular)."""
function _solve_Δx_nr!(stateVector::StateVectorCache, cache::PNM.LinearSolverCache)
    copyto!(stateVector.Δx_nr, stateVector.r)
    solve!(cache, stateVector.Δx_nr)
    return
end

"""Compute the relative residual `‖A·Δx_nr − r‖₁ / ‖r‖₁` of the linear solve and, if it
exceeds `refinement_threshold`, run iterative refinement and recompute it. Returns the
(post-refinement) relative residual; the caller uses it as a backend-agnostic singularity
signal (see [`_set_Δx_nr!`](@ref))."""
function _do_refinement!(stateVector::StateVectorCache,
    A::SparseMatrixCSC{Float64, J_INDEX_TYPE},
    cache::PNM.LinearSolverCache,
    refinement_threshold::Float64,
    refinement_eps::Float64,
)
    # use stateVector.r_predict as temporary buffer.
    δ_temp = stateVector.r_predict
    r_norm = sum(abs, stateVector.r)
    # A zero residual is an exact (already-converged) solve, not a singular Jacobian. Return a
    # zero relative residual rather than dividing 0/0 into a NaN, which the caller's
    # `!isfinite(residual)` guard would otherwise misread as a singularity.
    iszero(r_norm) && return 0.0
    mul!(δ_temp, A, stateVector.Δx_nr)
    δ_temp .-= stateVector.r
    delta = sum(abs, δ_temp) / r_norm
    if delta > refinement_threshold
        stateVector.Δx_nr .= solve_w_refinement(cache,
            A,
            stateVector.r,
            refinement_eps)
        mul!(δ_temp, A, stateVector.Δx_nr)
        δ_temp .-= stateVector.r
        delta = sum(abs, δ_temp) / r_norm
    end
    return delta
end

"""Factor `J.Jv` with `factor!`, solve for `Δx_nr`, and refine. Returns `false` when the
factorization is singular or the residual stays above `refinement_threshold`."""
function _factor_solve_ok!(factor!,
    stateVector::StateVectorCache,
    J::Union{ACPowerFlowJacobian, ACRectangularCIJacobian, ACMixedCPBJacobian},
    linSolveCache::PNM.LinearSolverCache,
    refinement_threshold::Float64,
    refinement_eps::Float64)
    try
        factor!(linSolveCache, J.Jv)
    catch e
        # KLU signals a singular factorization by throwing a `SingularException`;
        # AppleAccelerate and MKLPardiso do not (the residual guard below catches their
        # silent garbage solves). Any other exception is a genuine solver failure, not a
        # singular Jacobian, so rethrow it rather than masking it.
        e isa LinearAlgebra.SingularException || rethrow()
        return false
    end
    _solve_Δx_nr!(stateVector, linSolveCache)
    _skips_refinement(linSolveCache) && return true
    # Backend-agnostic singular-Jacobian guard: refinement returns the relative residual
    # and rescues merely ill-conditioned solves; KLU throws above, AA/Pardiso need this.
    residual = _do_refinement!(
        stateVector,
        J.Jv,
        linSolveCache,
        refinement_threshold,
        refinement_eps,
    )
    return isfinite(residual) && residual <= refinement_threshold
end

"""Sets the Newton-Raphson step. Usually, this is just `J.Jv \\ stateVector.r`, but
`J.Jv` might be singular."""
function _set_Δx_nr!(stateVector::StateVectorCache,
    J::Union{ACPowerFlowJacobian, ACRectangularCIJacobian, ACMixedCPBJacobian},
    data::ACPowerFlowData,
    linSolveCache::PNM.LinearSolverCache,
    solver::ACPowerFlowSolverType,
    refinement_threshold::Float64,
    refinement_eps::Float64)
    _count_numeric_refactor!(data)
    ok = _factor_solve_ok!(
        numeric_refactor!, stateVector, J, linSolveCache,
        refinement_threshold, refinement_eps,
    )
    if !ok && _repivots(linSolveCache)
        @debug "stale KLU pivot order; re-pivoting with a fresh factorization"
        _count_numeric_refactor!(data)
        ok = _factor_solve_ok!(
            _repivot!, stateVector, J, linSolveCache,
            refinement_threshold, refinement_eps,
        )
    end

    if !ok
        @warn("$solver hit a point where the Jacobian is singular.")
        # KLU is used because the fallback must reliably solve the regularized system. Refresh
        # values in place while the pattern holds (reusing the factorization); rebuild if it shifts.
        M_prev = stateVector.fallback_matrix[]
        cache_prev = stateVector.fallback_cache[]
        if !isnothing(M_prev) && !isnothing(cache_prev) &&
           _refresh_singular_J_fallback!(M_prev, J.Jv, stateVector.x)
            M = M_prev
            cache = cache_prev
            numeric_refactor!(cache, M)
        else
            M = _build_singular_J_fallback(J.Jv, stateVector.x)
            cache = make_linear_solver_cache(PNM.KLUSolver(), M)
            full_factor!(cache, M)
            stateVector.fallback_matrix[] = M
            stateVector.fallback_cache[] = cache
        end
        _solve_Δx_nr!(stateVector, cache)
        _do_refinement!(stateVector, M, cache, refinement_threshold, refinement_eps)
    end
    # Not rmul!: BLAS dscal wakes OpenBLAS's thread pool every Newton step, which then spins on
    # the cores PTSA and threaded time steps run their workers on.
    stateVector.Δx_nr .= .-stateVector.Δx_nr
    return
end

"""Fill `M` in place with `-(fjac2 + λI)`; `M` and `fjac2` share one sparsity pattern.

Loops over `M`'s stored pattern (not sparse broadcast) so structural zeros are not pruned."""
function _fill_singular_J_fallback!(M::SparseMatrixCSC{Float64, J_INDEX_TYPE},
    fjac2::SparseMatrixCSC{Float64, J_INDEX_TYPE},
    x::Vector{Float64})
    lambda = NR_SINGULAR_SCALING * sqrt(length(x) * eps()) * norm(fjac2, 1)
    Mnz = M.nzval
    Fnz = fjac2.nzval
    @inbounds for col in 1:size(M, 2)
        for p in M.colptr[col]:(M.colptr[col + 1] - 1)
            if M.rowval[p] == col
                Mnz[p] = -(Fnz[p] + lambda)
            else
                Mnz[p] = -Fnz[p]
            end
        end
    end
    return
end

"""Returns a freshly-allocated stand-in matrix `-(JᵀJ + λI)` for a singular `J`, on `JᵀJ`'s own
(full) pattern. The result defines the sparsity pattern that
[`_refresh_singular_J_fallback!`](@ref) reuses in place."""
function _build_singular_J_fallback(Jv::SparseMatrixCSC{Float64, J_INDEX_TYPE},
    x::Vector{Float64})
    fjac2 = Jv' * Jv
    M = copy(fjac2)
    _fill_singular_J_fallback!(M, fjac2, x)
    return M
end

"""Refresh `M = -(JᵀJ + λI)` in place (λ as in [`_build_singular_J_fallback`](@ref)). Returns
`false` without touching `M` when the `JᵀJ` pattern no longer matches `M`'s (i.e. `Jv`'s own
structural pattern changed), so the caller rebuilds."""
function _refresh_singular_J_fallback!(M::SparseMatrixCSC{Float64, J_INDEX_TYPE},
    Jv::SparseMatrixCSC{Float64, J_INDEX_TYPE},
    x::Vector{Float64})
    fjac2 = Jv' * Jv
    _same_sparsity(M, fjac2) || return false
    _fill_singular_J_fallback!(M, fjac2, x)
    return true
end

"""Sets `Δx_proposed` equal to the `Δx` by which we should update `x`. Decides
between the Cauchy step `Δx_cauchy`, Newton-Raphson step `Δx_nr`, and the dogleg
interpolation between the two, based on which fall within the trust region."""
function _dogleg!(Δx_proposed::Vector{Float64},
    Δx_cauchy::Vector{Float64},
    Δx_nr::Vector{Float64},
    r::Vector{Float64},
    Jv::SparseMatrixCSC{Float64, J_INDEX_TYPE},
    d::Vector{Float64},
    r_scratch::Vector{Float64},
    delta::Float64,
)
    nr_norm = wnorm(d, Δx_nr)
    @debug "Trust region: ||Δx_nr|| = $(siground(nr_norm)), δ = $(siground(delta))"

    if nr_norm <= delta
        copyto!(Δx_proposed, Δx_nr) # update Δx_proposed: newton-raphson case.
        @debug "Newton-Raphson step selected (inside trust region)"
    else
        # using Δx_proposed as a temporary buffer: alias to g for readability
        g = Δx_proposed
        LinearAlgebra.mul!(g, Jv', r)
        g .= g ./ d .^ 2
        LinearAlgebra.mul!(r_scratch, Jv, g)
        Δx_cauchy .= -wnorm(d, g)^2 / sum(abs2, r_scratch) .* g # Cauchy point

        cauchy_norm = wnorm(d, Δx_cauchy)
        @debug "Cauchy point: ||Δx_cauchy|| = $(siground(cauchy_norm))"

        if cauchy_norm >= delta
            # Δx_cauchy outside region => take step of length delta in direction of -g.
            LinearAlgebra.rmul!(g, -delta / wnorm(d, g))
            @debug "Cauchy step selected (truncated to trust region boundary)"
            # not needed because g is already an alias for Δx_proposed.
            # copyto!(Δx_proposed, g) # update Δx_proposed: cauchy point case
        else
            # Δx_cauchy inside region => next point is the spot where the line from
            # Δx_cauchy to Δx_nr crosses the boundary of the trust region.
            # this is the "dogleg" part.

            # using Δx_nr as temporary buffer: alias to Δx_diff for readability.
            Δx_nr .-= Δx_cauchy
            Δx_diff = Δx_nr

            b = wdot(d, Δx_cauchy, d, Δx_diff)
            a = wnorm(d, Δx_diff)^2
            tau = (-b + sqrt(b^2 - 4a * (wnorm(d, Δx_cauchy)^2 - delta^2))) / (2a)
            Δx_cauchy .+= tau .* Δx_diff
            copyto!(Δx_proposed, Δx_cauchy) # update Δx_proposed: dogleg case.
            @debug "Dogleg step selected (τ = $(siground(tau)))"
        end
    end
    return
end

"""Accept a trust region step: update cached residual and autoscale vector `d`.
The caller is responsible for recomputing the Jacobian via `J(data, time_step)` before
calling this, so that `Jv` reflects the new state."""
function _accept_trust_region_step!(
    stateVector::StateVectorCache,
    residual::Union{ACPowerFlowResidual, ACRectangularCIResidual, ACMixedCPBResidual},
    Jv::SparseMatrixCSC{Float64, J_INDEX_TYPE},
    autoscale::Bool,
)
    stateVector.r .= residual.Rv
    if autoscale
        for i in 1:length(stateVector.x)
            stateVector.d[i] = max(0.1 * stateVector.d[i], norm(view(Jv, :, i)))
        end
    end
    return
end

"""Attempt Iwamoto damping on a rejected trust region step.

Uses the already-evaluated trial-point residual to compute an optimal damped step.
Returns `true` if the damped step was accepted, `false` if reverted."""
function _iwamoto_fallback!(
    time_step::Int,
    stateVector::StateVectorCache,
    residual::Union{ACPowerFlowResidual, ACRectangularCIResidual, ACMixedCPBResidual},
    J::Union{ACPowerFlowJacobian, ACRectangularCIJacobian, ACMixedCPBJacobian},
    data::ACPowerFlowData,
    old_residual::Vector{Float64},
    old_residual_norm::Float64,
    autoscale::Bool,
)::Bool
    g0 = old_residual_norm
    # Quadratic model F(x+μΔx) = f₀ + μ·(J·Δx) + μ²·a along Δx_proposed. r_predict
    # (= f₀ + J·Δx) from the ρ test gives a = F(x+Δx) − r_predict for free (no extra matvec).
    c_fb, c_bb, c_fa, c_ba, c_aa =
        _iwamoto_quadratic_dots(old_residual, stateVector.r_predict, residual.Rv)
    μ = _iwamoto_multiplier(2.0 * c_fb, c_bb + 2.0 * c_fa, 2.0 * c_ba, c_aa)
    # Revert full step, apply damped step in a single fused pass.
    @. stateVector.x += (μ - 1.0) * stateVector.Δx_proposed
    residual(data, stateVector.x, time_step)
    g_damped = dot(residual.Rv, residual.Rv)
    if g_damped < g0
        @debug "Iwamoto fallback accepted: μ = $(siground(μ)), " *
               "g_damped/g₀ = $(siground(g_damped / g0))"
        J(data, time_step)
        _accept_trust_region_step!(stateVector, residual, J.Jv, autoscale)
        return true
    else
        # Damped step also failed — full revert.
        @. stateVector.x -= μ * stateVector.Δx_proposed
        copyto!(residual.Rv, old_residual)
        @debug "Iwamoto fallback rejected: μ = $(siground(μ)), " *
               "g_damped/g₀ = $(siground(g_damped / g0)); reverting"
        return false
    end
end

"""Does a single iteration of the `TrustRegionNRMethod`:
updates the `x` and `r` fields of the `stateVector` and computes
the value of the Jacobian at the new `x`, if needed. Unlike
`_simple_step`, this has a return value, the updated value of `delta``."""
function _trust_region_step(time_step::Int,
    stateVector::StateVectorCache,
    linSolveCache::PNM.LinearSolverCache,
    residual::Union{ACPowerFlowResidual, ACRectangularCIResidual, ACMixedCPBResidual},
    J::Union{ACPowerFlowJacobian, ACRectangularCIJacobian, ACMixedCPBJacobian},
    data::ACPowerFlowData,
    delta::Float64,
    delta_max::Float64,
    eta::Float64,
    autoscale::Bool,
    iwamoto_fallback::Bool,
)
    old_delta = delta
    _set_Δx_nr!(
        stateVector,
        J,
        data,
        linSolveCache,
        TrustRegionACPowerFlow(),
        DEFAULT_REFINEMENT_THRESHOLD,
        DEFAULT_REFINEMENT_EPS,
    )
    _dogleg!(
        stateVector.Δx_proposed,
        stateVector.Δx_cauchy,
        stateVector.Δx_nr,
        stateVector.r,
        J.Jv,
        stateVector.d,
        stateVector.r_scratch,
        delta,
    )
    # find proposed next point.
    stateVector.x .+= stateVector.Δx_proposed

    # use cache.Δx_nr as temporary buffer to store old residual
    # to avoid recomputing if we don't change x.
    oldResidual = stateVector.Δx_nr
    copyto!(oldResidual, residual.Rv)
    old_residual_norm = sum(abs2, stateVector.r)
    residual(data, stateVector.x, time_step)
    new_residual_norm = sum(abs2, residual.Rv)

    # Ratio of actual to predicted reduction
    LinearAlgebra.mul!(stateVector.r_predict, J.Jv, stateVector.Δx_proposed)
    stateVector.r_predict .+= stateVector.r
    predicted_reduction = old_residual_norm - sum(abs2, stateVector.r_predict)
    # The dogleg model reduction is non-negative by construction; a non-positive value
    # here is floating-point cancellation near convergence (‖r‖²≈0). Force a rejected-step
    # ρ to shrink the trust region — standard recovery, matching the LM solver's guard.
    rho = if predicted_reduction > 0.0
        (old_residual_norm - new_residual_norm) / predicted_reduction
    else
        @debug "Non-positive predicted reduction $(siground(predicted_reduction)); \
            rejecting step, shrinking trust region"
        -Inf
    end

    @debug "Trust region step: ρ = $(siground(rho)), η = $(siground(eta)), ||Δx|| = $(siground(norm(stateVector.Δx_proposed)))"

    step_accepted = false
    if rho > eta
        # Successful iteration
        @debug "Step accepted: sum of squares $(siground(dot(residual.Rv, residual.Rv))), L ∞ norm $(siground(norm(residual.Rv, Inf))), Δ = $(siground(delta)), ||Δx|| = $(siground(norm(stateVector.Δx_proposed)))"
        J(data, time_step)
        _accept_trust_region_step!(stateVector, residual, J.Jv, autoscale)
        step_accepted = true
    else
        # Unsuccessful step — try Iwamoto damping before reverting.
        if iwamoto_fallback
            iwamoto_accepted = _iwamoto_fallback!(
                time_step, stateVector, residual, J, data,
                oldResidual, old_residual_norm, autoscale)
            if iwamoto_accepted
                # Iwamoto accepted a damped step — shrink trust region since the
                # full proposed step was rejected by rho. Do not use rho-based
                # expansion logic because rho corresponds to the rejected full step.
                delta = min(delta / 2, delta_max)
                @debug "Trust region decreased (Iwamoto fallback accepted): δ $(siground(old_delta)) → $(siground(delta))"
                return delta
            end
        else
            stateVector.x .-= stateVector.Δx_proposed
            copyto!(residual.Rv, oldResidual)
            @debug "Step rejected: ρ = $(siground(rho)) ≤ η = $(siground(eta))"
        end
    end

    # Update size of trust region based on rho (only reached when the full step
    # was accepted via rho, or Iwamoto is disabled, or Iwamoto didn't help).
    if rho < HALVE_TRUST_REGION # rho < 0.1: insufficient improvement
        delta = delta / 2
        @debug "Trust region decreased: δ $(siground(old_delta)) → $(siground(delta)) (ρ < $(HALVE_TRUST_REGION))"
    elseif step_accepted && rho >= DOUBLE_TRUST_REGION # rho >= 0.9: good improvement
        delta = 2 * wnorm(stateVector.d, stateVector.Δx_proposed)
        @debug "Trust region increased (good): δ $(siground(old_delta)) → $(siground(delta)) (ρ ≥ $(DOUBLE_TRUST_REGION))"
    elseif step_accepted && rho >= MAX_DOUBLE_TRUST_REGION # rho >= 0.5: so-so improvement
        delta = max(delta, 2 * wnorm(stateVector.d, stateVector.Δx_proposed))
        @debug "Trust region increased (moderate): δ $(siground(old_delta)) → $(siground(delta)) (ρ ≥ $(MAX_DOUBLE_TRUST_REGION))"
    else
        @debug "Trust region unchanged: δ = $(siground(delta))"
    end
    delta = min(delta, delta_max)
    return delta
end

"""Inner products for the quadratic model `F(x+μΔx) = f₀ + μ·b + μ²·a` with
`b = J·Δx`, `a = F(x+Δx) − f₀ − b`. From `f₀`, `rpred = f₀ + b`, and `rv = F(x+Δx)`
returns `(f₀·b, b·b, f₀·a, b·a, a·a)`."""
@inline function _iwamoto_quadratic_dots(
    f0::Vector{Float64}, rpred::Vector{Float64}, rv::Vector{Float64},
)::NTuple{5, Float64}
    c_fb = 0.0
    c_bb = 0.0
    c_fa = 0.0
    c_ba = 0.0
    c_aa = 0.0
    @inbounds @simd for i in eachindex(f0, rpred, rv)
        b = rpred[i] - f0[i]
        a = rv[i] - rpred[i]
        c_fb += f0[i] * b
        c_bb += b * b
        c_fa += f0[i] * a
        c_ba += b * a
        c_aa += a * a
    end
    return c_fb, c_bb, c_fa, c_ba, c_aa
end

"""Iwamoto objective minus its μ-independent constant:
`g̃(μ) = q₁μ + q₂μ² + q₃μ³ + q₄μ⁴`. Dropping the constant preserves the minimizer."""
@inline function _iwamoto_objective(
    μ::Float64, q1::Float64, q2::Float64, q3::Float64, q4::Float64,
)::Float64
    return μ * (q1 + μ * (q2 + μ * (q3 + μ * q4)))
end

"""If μ ∈ [IWAMOTO_MU_MIN, IWAMOTO_MU_MAX] and g̃(μ) < best_g, return the
improved (μ, g̃(μ)); otherwise return (best_μ, best_g) unchanged."""
@inline function _try_iwamoto_candidate(
    μ::Float64,
    best_μ::Float64,
    best_g::Float64,
    q1::Float64,
    q2::Float64,
    q3::Float64,
    q4::Float64,
)::Tuple{Float64, Float64}
    if IWAMOTO_MU_MIN <= μ <= IWAMOTO_MU_MAX
        gval = _iwamoto_objective(μ, q1, q2, q3, q4)
        if gval < best_g
            return μ, gval
        end
    end
    return best_μ, best_g
end

"""Optimal Iwamoto multiplier μ ∈ [IWAMOTO_MU_MIN, IWAMOTO_MU_MAX] minimizing
`g̃(μ) = q₁μ + q₂μ² + q₃μ³ + q₄μ⁴` (coefficients from [`_iwamoto_quadratic_dots`](@ref)).
Stationary points solve the cubic `g̃'(μ) = 4q₄μ³ + 3q₃μ² + 2q₂μ + q₁ = 0`, found
analytically (depressed-cubic Cardano/trig form). Exact for the dogleg step;
reduces to classical Iwamoto & Tamura (1981) when `b = −f₀` (Newton step)."""
function _iwamoto_multiplier(q1::Float64, q2::Float64, q3::Float64, q4::Float64)::Float64
    # Initialize best candidate from domain boundaries.
    best_μ = IWAMOTO_MU_MIN
    best_g = _iwamoto_objective(IWAMOTO_MU_MIN, q1, q2, q3, q4)
    best_μ, best_g =
        _try_iwamoto_candidate(IWAMOTO_MU_MAX, best_μ, best_g, q1, q2, q3, q4)

    # Cubic coefficients: c₃μ³ + c₂μ² + c₁μ + c₀ = 0
    c3 = 4.0 * q4
    c2 = 3.0 * q3
    c1 = 2.0 * q2
    c0 = q1

    if abs(c3) < IWAMOTO_DEGENERACY_TOL
        # Degenerate: solve quadratic c₂μ² + c₁μ + c₀ = 0
        if abs(c2) > IWAMOTO_DEGENERACY_TOL
            disc = c1 * c1 - 4.0 * c2 * c0
            if disc >= 0.0
                sq = sqrt(disc)
                for μ in ((-c1 + sq) / (2.0 * c2), (-c1 - sq) / (2.0 * c2))
                    best_μ, best_g =
                        _try_iwamoto_candidate(μ, best_μ, best_g, q1, q2, q3, q4)
                end
            end
        elseif abs(c1) > IWAMOTO_DEGENERACY_TOL
            best_μ, best_g =
                _try_iwamoto_candidate(-c0 / c1, best_μ, best_g, q1, q2, q3, q4)
        end
        return best_μ
    end

    # Full cubic — depress to t³ + At + B = 0 via μ = t - p/3
    p = c2 / c3
    q = c1 / c3
    c0_n = c0 / c3
    p3 = p / 3.0
    A = q - p * p3
    B = c0_n - q * p3 + 2.0 * p3^3
    Δ = -4.0 * A^3 - 27.0 * B^2

    if Δ > 0.0
        # Three distinct real roots — trigonometric form (A < 0 guaranteed when Δ > 0).
        s = sqrt(-A / 3.0)
        m = 2.0 * s
        arg = clamp(-B / (2.0 * s * s * s), -1.0, 1.0)
        φ3 = acos(arg) / 3.0
        for k in 0:2
            best_μ, best_g = _try_iwamoto_candidate(
                m * cos(φ3 - 2.0 * π * k / 3.0) - p3,
                best_μ, best_g, q1, q2, q3, q4)
        end
    elseif Δ < 0.0
        # One real root — Cardano's formula.
        sqD = sqrt(max(-Δ / 108.0, 0.0))
        best_μ, best_g = _try_iwamoto_candidate(
            cbrt(-B / 2.0 + sqD) + cbrt(-B / 2.0 - sqD) - p3,
            best_μ, best_g, q1, q2, q3, q4)
    else
        # Δ ≈ 0 — repeated roots.
        if abs(A) < IWAMOTO_DEGENERACY_TOL
            # Triple root at t = 0.
            best_μ, best_g = _try_iwamoto_candidate(-p3, best_μ, best_g, q1, q2, q3, q4)
        else
            # Simple root t₁ = 3B/A and double root t₂ = -3B/(2A).
            for t in (3.0 * B / A, -3.0 * B / (2.0 * A))
                best_μ, best_g = _try_iwamoto_candidate(
                    t - p3, best_μ, best_g, q1, q2, q3, q4)
            end
        end
    end

    return best_μ
end

"""Classical Iwamoto & Tamura (1981) multiplier for the Newton step (`b = −f₀`),
with `g₀ = ‖f₀‖²`, `g₁ = f₀ᵀf₁`, `g₂ = ‖f₁‖²`, `f₁ = F(x+Δx)`."""
@inline function _iwamoto_multiplier(g0::Float64, g1::Float64, g2::Float64)::Float64
    return _iwamoto_multiplier(-2.0 * g0, g0 + 2.0 * g1, -2.0 * g1, g2)
end

"""Does a single iteration of `NewtonRaphsonACPowerFlow`. Updates the `r` and `x`
 fields of the `stateVector` and the residual at the new `x`. Returns whether the Jacobian was
 filled at the new `x` too (the fused polar kernel); otherwise the caller refills it there only
 if it reads it again."""
function _simple_step(time_step::Int,
    stateVector::StateVectorCache,
    linSolveCache::PNM.LinearSolverCache,
    residual::Union{ACPowerFlowResidual, ACRectangularCIResidual, ACMixedCPBResidual},
    J::Union{ACPowerFlowJacobian, ACRectangularCIJacobian, ACMixedCPBJacobian},
    data::ACPowerFlowData,
    refinement_threshold::Float64 = DEFAULT_REFINEMENT_THRESHOLD,
    refinement_eps::Float64 = DEFAULT_REFINEMENT_EPS,
)
    copyto!(stateVector.r, residual.Rv)
    _set_Δx_nr!(
        stateVector,
        J,
        data,
        linSolveCache,
        NewtonRaphsonACPowerFlow(),
        refinement_threshold,
        refinement_eps,
    )
    # update x
    stateVector.x .+= stateVector.Δx_nr
    # update data's fields (the bus angles/voltages) to match x, and update the residual.
    return _residual_at_step!(residual, J, data, stateVector.x, time_step)
end

function _residual_at_step!(
    residual::ACPowerFlowResidual,
    J::ACPowerFlowJacobian,
    data::ACPowerFlowData,
    x::Vector{Float64},
    time_step::Int,
)
    _update_residual_and_jacobian!(residual, J, x, data, time_step)
    return true
end

function _residual_at_step!(
    residual::Union{ACRectangularCIResidual, ACMixedCPBResidual},
    ::Union{ACRectangularCIJacobian, ACMixedCPBJacobian},
    data::ACPowerFlowData,
    x::Vector{Float64},
    time_step::Int,
)
    residual(data, x, time_step)
    return false
end

"""Does a single iteration of Newton-Raphson with Iwamoto step control.
Computes the Newton step, takes a full trial step, and checks whether the
residual norm decreased. If not, computes an optimal damping multiplier `μ`
and applies a damped step instead. When the damped step also fails to reduce
the residual, the step is reverted to avoid divergence.

Returns `true` if the step made progress (residual decreased), `false` if
the step was reverted. Consecutive reverts signal stagnation and the caller
should terminate early. Like [`_simple_step`](@ref), it leaves the Jacobian at the
pre-step iterate; the caller refills it after an accepted step."""
function _iwamoto_step(time_step::Int,
    stateVector::StateVectorCache,
    linSolveCache::PNM.LinearSolverCache,
    residual::Union{ACPowerFlowResidual, ACRectangularCIResidual, ACMixedCPBResidual},
    J::Union{ACPowerFlowJacobian, ACRectangularCIJacobian, ACMixedCPBJacobian},
    data::ACPowerFlowData,
    refinement_threshold::Float64 = DEFAULT_REFINEMENT_THRESHOLD,
    refinement_eps::Float64 = DEFAULT_REFINEMENT_EPS,
)::Bool
    # Save pre-step residual f into stateVector.r
    copyto!(stateVector.r, residual.Rv)
    # Compute Newton step Δx_nr = -J⁻¹f
    _set_Δx_nr!(
        stateVector,
        J,
        data,
        linSolveCache,
        NewtonRaphsonACPowerFlow(),
        refinement_threshold,
        refinement_eps,
    )
    # Take full trial step: x += Δx_nr
    stateVector.x .+= stateVector.Δx_nr
    # Evaluate trial residual b = F(x + Δx)
    residual(data, stateVector.x, time_step)

    # Compute gram scalars for Iwamoto criterion
    g0 = dot(stateVector.r, stateVector.r)
    g1 = dot(stateVector.r, residual.Rv)
    g2 = dot(residual.Rv, residual.Rv)

    if g2 < g0
        # Full step reduced residual — accept it (μ = 1).
        @debug "Iwamoto: full step accepted (g₂/g₀ = $(g2/g0))"
        return true
    end

    # Full step did not reduce residual — compute optimal μ.
    μ = _iwamoto_multiplier(g0, g1, g2)
    @debug "Iwamoto: damped step μ = $μ (g₂/g₀ = $(g2/g0))"
    # Undo full step and apply damped step.
    stateVector.x .-= stateVector.Δx_nr
    stateVector.x .+= μ .* stateVector.Δx_nr
    # Re-evaluate residual at damped point.
    residual(data, stateVector.x, time_step)
    # Check whether the damped step actually improved the residual.
    g_damped = dot(residual.Rv, residual.Rv)
    if g_damped >= g0
        # Damped step did not improve — revert to pre-step state.
        @debug "Iwamoto: damped step did not reduce residual " *
               "(g_damped/g₀ = $(g_damped/g0), μ = $μ); reverting"
        stateVector.x .-= μ .* stateVector.Δx_nr
        residual(data, stateVector.x, time_step)
        return false
    end
    # Damped step improved — accept it.
    return true
end

# Formulation-dispatched voltage-magnitude validation, driven entirely by the
# per-formulation index list precomputed once on the residual. Polar indexes
# the state as `[|V|, θ, …]` (`x[2i-1]` = |V|, PQ only); rectangular CI and
# mixed CPB states are `(e, f, …)` per-bus blocks validating `e²+f² ∈
# [min², max²]` over PQ/PV.
function _validate_state_magnitudes(
    r::ACPowerFlowResidual,
    x::Vector{Float64},
    range::MinMax,
    i::Int64,
)
    validate_voltage_magnitudes(x, r.validate_indices, range, i)
    return
end

function _validate_state_magnitudes(
    r::ACRectangularCIResidual,
    x::Vector{Float64},
    range::MinMax,
    i::Int64,
)
    _validate_squared_voltage_magnitudes(x, r.validate_offsets, range, i)
    return
end

function _validate_state_magnitudes(
    r::ACMixedCPBResidual,
    x::Vector{Float64},
    range::MinMax,
    i::Int64,
)
    _validate_squared_voltage_magnitudes(x, r.validate_offsets, range, i)
    return
end

"""Runs the full `NewtonRaphsonACPowerFlow`.
# Keyword arguments:
- `maxIterations::Int`: maximum iterations. Default: $DEFAULT_NR_MAX_ITER.
- `tol::Float64`: tolerance. The iterative search ends when `norm(abs.(residual)) < tol`.
    Default: $DEFAULT_NR_TOL.
- `refinement_threshold::Float64`: If the solution to `J_x Δx = r` satisfies
    `norm(J_x Δx - r, 1)/norm(r, 1) > refinement_threshold`, do iterative refinement to
    improve the accuracy. Default: $DEFAULT_REFINEMENT_THRESHOLD.
- `refinement_eps::Float64`: run iterative refinement on `J_x Δx = r` until
    `norm(Δx_{i}-Δx_{i+1}, 1)/norm(r,1) < refinement_eps`. Default:
    $DEFAULT_REFINEMENT_EPS """
function _run_power_flow_method(time_step::Int,
    stateVector::StateVectorCache,
    linSolveCache::PNM.LinearSolverCache,
    residual::Union{ACPowerFlowResidual, ACRectangularCIResidual, ACMixedCPBResidual},
    J::Union{ACPowerFlowJacobian, ACRectangularCIJacobian, ACMixedCPBJacobian},
    data::ACPowerFlowData,
    ::Type{NewtonRaphsonACPowerFlow};
    maxIterations::Int = DEFAULT_NR_MAX_ITER,
    tol::Float64 = DEFAULT_NR_TOL,
    refinement_threshold::Float64 = DEFAULT_REFINEMENT_THRESHOLD,
    refinement_eps::Float64 = DEFAULT_REFINEMENT_EPS,
    validate_voltage_magnitudes::Bool = DEFAULT_VALIDATE_VOLTAGES,
    vm_validation_range::MinMax = DEFAULT_VALIDATION_RANGE,
    iwamoto::Bool = false,
    stop_at_fold::Bool = false,
    _ignored...,  # absorb unknown keys from caller without error
)
    validate_vms = validate_voltage_magnitudes
    i, converged = 0, false
    consecutive_reverts = 0
    monitor, diag_state = setup_solver_diagnostics(J, data, stop_at_fold)
    # J at the converged iterate is read only by the diagnostics and the loss and
    # voltage-stability factors; otherwise its fill there is wasted.
    keep_converged_J =
        !isnothing(diag_state) || get_calculate_loss_factors(data) ||
        get_calculate_voltage_stability_factors(data)
    while i < maxIterations && !converged
        i += 1
        made_progress = true
        J_filled = false
        if iwamoto
            made_progress = _iwamoto_step(
                time_step,
                stateVector,
                linSolveCache,
                residual,
                J,
                data,
                refinement_threshold,
                refinement_eps,
            )
            if made_progress
                consecutive_reverts = 0
            else
                consecutive_reverts += 1
                if consecutive_reverts >= IWAMOTO_MAX_REVERTS
                    @debug "Iwamoto: $consecutive_reverts consecutive reverted steps; terminating early"
                    break
                end
            end
        else
            J_filled = _simple_step(
                time_step,
                stateVector,
                linSolveCache,
                residual,
                J,
                data,
                refinement_threshold,
                refinement_eps,
            )
        end
        converged = norm(residual.Rv, Inf) < tol
        # A reverted Iwamoto step leaves x, and so J, unchanged.
        if made_progress && !J_filled && (!converged || keep_converged_J)
            J(data, time_step)
        end
        validate_vms && _validate_state_magnitudes(
            residual,
            stateVector.x,
            vm_validation_range,
            i,
        )
        if !isnothing(diag_state)
            # J.Jv and residual.Rv are at the same iterate here, so one refactor feeds both
            # the log line and the bail-out.
            run_solver_diagnostics!(
                diag_state, "NR iter $i", residual, J, data, time_step,
                linSolveCache, monitor, stop_at_fold) &&
                return false, i
        end
    end
    return converged, i
end

"""Runs the full `TrustRegionNRMethod`.
# Keyword arguments:
- `maxIterations::Int`: maximum iterations. Default: $DEFAULT_NR_MAX_ITER.
- `tol::Float64`: tolerance. The iterative search ends when `maximum(abs.(residual)) < tol`.
    Default: $DEFAULT_NR_TOL.
- `factor::Float64`: the trust region starts out with radius `factor*norm(x_0, 1)`,
    where `x_0` is our initial guess, taken from `data`. Default: $DEFAULT_TRUST_REGION_FACTOR.
- `eta::Float64`: improvement threshold. If the observed improvement in our residual
    exceeds `eta` times the predicted improvement, we accept the new `x_i`.
    Default: $DEFAULT_TRUST_REGION_ETA.
- `iwamoto_fallback::Bool`: when a trust region step is rejected, attempt Iwamoto
    damping to salvage the step before reverting. Default: $DEFAULT_IWAMOTO_FALLBACK."""
function _run_power_flow_method(time_step::Int,
    stateVector::StateVectorCache,
    linSolveCache::PNM.LinearSolverCache,
    residual::Union{ACPowerFlowResidual, ACRectangularCIResidual, ACMixedCPBResidual},
    J::Union{ACPowerFlowJacobian, ACRectangularCIJacobian, ACMixedCPBJacobian},
    data::ACPowerFlowData,
    ::Type{TrustRegionACPowerFlow};
    maxIterations::Int = DEFAULT_NR_MAX_ITER,
    tol::Float64 = DEFAULT_NR_TOL,
    factor::Float64 = DEFAULT_TRUST_REGION_FACTOR,
    eta::Float64 = DEFAULT_TRUST_REGION_ETA,
    autoscale::Bool = DEFAULT_AUTOSCALE,
    iwamoto_fallback::Bool = DEFAULT_IWAMOTO_FALLBACK,
    validate_voltage_magnitudes::Bool = DEFAULT_VALIDATE_VOLTAGES,
    vm_validation_range::MinMax = DEFAULT_VALIDATION_RANGE,
    stop_at_fold::Bool = false,
    _ignored...,  # absorb unknown keys from caller without error
)
    validate_vms = validate_voltage_magnitudes

    if eta > 1.0 || eta < 0.0
        @warn("η = $eta is outside [0, 1]") # eta is set to 2.0 in one test.
    end

    if autoscale
        for i in 1:length(stateVector.x)
            stateVector.d[i] = norm(view(J.Jv, :, i))
            if iszero(stateVector.d[i])
                stateVector.d[i] = 1.0
            end
        end
    end

    delta = norm(stateVector.x) > 0 ? factor * norm(stateVector.x) : factor
    delta_max = DEFAULT_TRUST_REGION_DELTA_MAX_FACTOR * delta
    i, converged = 0, false
    residualSize = dot(residual.Rv, residual.Rv)
    linf = norm(residual.Rv, Inf)
    @debug "initially: sum of squares $(siground(residualSize)), L ∞ norm $(siground(linf)), Δ $(siground(delta))"

    monitor, diag_state = setup_solver_diagnostics(J, data, stop_at_fold)
    while i < maxIterations && !converged
        delta = _trust_region_step(
            time_step,
            stateVector,
            linSolveCache,
            residual,
            J,
            data,
            delta,
            delta_max,
            eta,
            autoscale,
            iwamoto_fallback,
        )
        validate_vms && _validate_state_magnitudes(
            residual,
            stateVector.x,
            vm_validation_range,
            i,
        )
        if !isnothing(diag_state)
            # After `_trust_region_step` (incl. reject and iwamoto-fallback), J.Jv and
            # residual.Rv are at the same iterate, so one refactor feeds both.
            run_solver_diagnostics!(
                diag_state, "TR iter $i", residual, J, data, time_step,
                linSolveCache, monitor, stop_at_fold) &&
                return false, i
        end
        converged = norm(residual.Rv, Inf) < tol
        if !converged
            i += 1
        end
    end
    return converged, i
end

"""Log final residual, report convergence, compute optional post-processing factors,
and return `true`/`false`. Shared by all AC power flow drivers."""
# `Jv === nothing`: the fast-decoupled :decoupled driver skipped the formulation Jacobian (neither
# loss nor voltage-stability factors were requested), so there is nothing to compute — just report
# convergence. Dispatch keeps this path free of the factor machinery entirely.
function _finalize_power_flow(
    converged::Bool,
    i::Int,
    solver_name::String,
    residual::Union{ACPowerFlowResidual, ACRectangularCIResidual, ACMixedCPBResidual},
    data::ACPowerFlowData,
    ::Nothing,
    time_step::Int64,
)
    data.iterations[time_step] += i
    return _report_power_flow_convergence(converged, i, solver_name, residual)
end

function _finalize_power_flow(
    converged::Bool,
    i::Int,
    solver_name::String,
    residual::Union{ACPowerFlowResidual, ACRectangularCIResidual, ACMixedCPBResidual},
    data::ACPowerFlowData,
    Jv::SparseMatrixCSC{Float64, J_INDEX_TYPE},
    time_step::Int64,
)
    data.iterations[time_step] += i
    if converged
        _warn_small_lcc_angles(data, time_step)
        if get_calculate_loss_factors(data)
            _calculate_loss_factors(data, Jv, time_step)
        end
        if get_calculate_voltage_stability_factors(data)
            _calculate_voltage_stability_factors(data, Jv, time_step)
        end
    end
    return _report_power_flow_convergence(converged, i, solver_name, residual)
end

"""Log the final residual size and convergence/non-convergence, returning `converged`. Shared by
both `_finalize_power_flow` methods (Jacobian and Jacobian-free). Non-convergence is reported at
debug level because this is the innermost layer: callers (e.g. discrete-control continuation
trials, the area-interchange de-enroll loop) may treat the failure as an expected trial to roll
back rather than a terminal error; the entry point that returns the failure to the user logs it
once at error level."""
function _report_power_flow_convergence(
    converged::Bool,
    i::Int,
    solver_name::String,
    residual::Union{ACPowerFlowResidual, ACRectangularCIResidual, ACMixedCPBResidual},
)
    @debug("Final residual size: $(norm(residual.Rv, 2)) L2, $(norm(residual.Rv, Inf)) L∞.")
    if converged
        @debug("The $solver_name solver converged after $i iterations.")
        return true
    end
    @debug("The $solver_name solver failed to converge after $i iterations.")
    return false
end

"""Warn if any LCC's converged thyristor angle lies outside the physical
operating window `(LCC_SMALL_ANGLE_THRESHOLD, π/2 − LCC_SMALL_ANGLE_THRESHOLD)`
(≈ 5° to 85° by default). Real PSS/E LCCs operate well inside this range
(rectifier α_r ≈ 10-20°, inverter γ_i ≈ 14-18°). Either extreme sits near
an `arccos` clamp boundary where `Q_s`'s second derivatives are singular
(`1/sin³ϕ`): low α puts the rectifier-side `u_r → +1` or the inverter-side
`u_i → −1`; high α (approaching π/2) puts them on the other boundary, and
beyond π/2 the converter's rectifying/inverting role would reverse.
Hessian-based solvers (LM, RobustHomotopy) degrade in either regime, and
even direct Newton hitting one of these bounds is a sign the input data
is non-physical."""
function _warn_small_lcc_angles(data::ACPowerFlowData, time_step::Int)
    n_lcc = size(data.lcc.p_set, 1)
    iszero(n_lcc) && return
    lo = LCC_SMALL_ANGLE_THRESHOLD
    hi = π / 2 - LCC_SMALL_ANGLE_THRESHOLD
    for i in 1:n_lcc
        α_r = data.lcc.rectifier.thyristor_angle[i, time_step]
        α_i = data.lcc.inverter.thyristor_angle[i, time_step]
        out_of_range = α_r < lo || α_r > hi || α_i < lo || α_i > hi
        if out_of_range
            (fb, tb) = data.lcc.arcs[i]
            @warn(
                "LCC $i (arc $(fb) → $(tb)): converged thyristor angles " *
                "α_r = $(rad2deg(α_r))°, α_i = $(rad2deg(α_i))° — one or " *
                "both outside the physical-realism window " *
                "($(rad2deg(lo))°, $(rad2deg(hi))°). Typical PSS/E LCCs " *
                "operate at α_r ≈ 10-20° and γ_i ≈ 14-18°. Values near " *
                "0° or π/2 sit at the LCC arccos clamp boundary " *
                "(singular Q_s Hessian). Check the configured " *
                "rectifier_delay_angle_limits and " *
                "inverter_extinction_angle_limits.", maxlog = PF_MAX_LOG,
            )
        end
    end
    return
end

"""Formulation-specific post-Newton step. Polar writes the deferred iterate (|V|, θ and bus
injections) into `data`; the rectangular CI formulation distributes the converged subnetwork
slack into the bus injection arrays."""
_finalize_formulation!(
    ::ACPolarPowerFlow,
    data::ACPowerFlowData,
    ::Vector{Float64},
    residual::ACPowerFlowResidual,
    time_step::Int64,
) = _write_back_bus_state!(residual, data, time_step)

function _finalize_formulation!(
    ::ACRectangularPowerFlow,
    data::ACPowerFlowData,
    x::Vector{Float64},
    residual::ACRectangularCIResidual,
    time_step::Int64,
)
    rect_finalize_bus_injections!(
        data, x, residual.bus_state_offset, residual.P_net_set,
        residual.bus_slack_participation_factors, residual.subnetworks,
        residual.independent_ref, time_step,
    )
    return
end

function _finalize_formulation!(
    ::ACMixedPowerFlow,
    data::ACPowerFlowData,
    x::Vector{Float64},
    residual::ACMixedCPBResidual,
    time_step::Int64,
)
    mixed_finalize_bus_injections!(
        data, x, residual.bus_state_offset,
        residual.bus_slack_participation_factors, residual.subnetworks,
        residual.independent_ref,
        residual.e_state, residual.f_state,
        time_step,
    )
    return
end

# Build + symbolically factor a fresh linear-solver cache. Polar workspace reuse lives in
# `PolarNRCache`/`_newton_workspace!` and does not route through here, so this never writes the
# shared `data.polar_nr_cache` slot; the continuation path calls it for a one-off cache.
# The caller (`_sensitivity_context`) counts this factorization itself — it is the ONE symbolic
# build of the continuation's probe phase, reused by every batched-pass refresh — so this does not
# also count it (that double-counted the same factorization).
function _nr_linear_solver_cache!(
    data::ACPowerFlowData,
    J,
    backend,
    ::AbstractVector{Float64},
)
    linSolveCache = make_linear_solver_cache(backend, J.Jv)
    symbolic_factor!(linSolveCache, J.Jv)
    return linSolveCache
end

function _nr_initialize_with_jacobian_deferred(
    pf::ACPolarPowerFlow, data::ACPowerFlowData, time_step::Int64; kwargs...,
)
    residual, x0 = _initialize_residual_x0(pf, data, time_step; kwargs...)
    return residual, nothing, x0
end

# No candidate start to compare: F and J at x0 come from one fused sweep. `fused` is false when
# `_fused_x0!` fell back to `improve_x0!`, leaving J stale.
function _fused_polar_init(
    pf::ACPolarPowerFlow, data::ACPowerFlowData, time_step::Int64; kwargs...,
)
    residual = ACPowerFlowResidual(data, time_step)
    x0 = calculate_x0(data, time_step)
    J = ACPowerFlowJacobian(data, residual, time_step)
    fused = _fused_x0!(x0, pf, data, residual, J, time_step)
    _log_initial_residual(residual)
    if get(kwargs, :validate_voltage_magnitudes, DEFAULT_VALIDATE_VOLTAGES)
        validate_voltage_magnitudes(x0, residual.validate_indices,
            get(kwargs, :vm_validation_range, DEFAULT_VALIDATION_RANGE), 0)
    end
    return residual, J, x0, fused
end

"""Evaluate F and J at `x0` in one fused sweep. On a large residual with a fallback start
enabled, run `improve_x0!` instead and return `false`: J is then stale."""
function _fused_x0!(
    x0::Vector{Float64},
    pf::ACPolarPowerFlow,
    data::ACPowerFlowData,
    residual::ACPowerFlowResidual,
    J::ACPowerFlowJacobian,
    time_step::Int64,
)
    _update_residual_and_jacobian!(residual, J, x0, data, time_step)
    large = _large_residual(residual)
    if large && (get_enhanced_flat_start(pf) || get_robust_power_flow(pf))
        # `bus_state` already holds x0, so this re-evaluation adds exactly 0 on PQ buses.
        improve_x0!(x0, pf, data, residual, time_step)
        return false
    end
    # The same log as `improve_x0!` when it would change nothing.
    @debug "skipping enhanced flat start"
    @debug "skipping running DC power flow fallback"
    large && _warn_large_initial_residual(residual, data, time_step)
    return true
end

# Rectangular/mixed: J is structure-only (no value evaluation), cheap enough to build eagerly.
# These formulations do not call J(data, time_step) in their setup, so the cost is just the
# sparse-structure allocation (~1-2 MB), not the full evaluation.
function _nr_initialize_with_jacobian_deferred(
    pf::ACRectangularPowerFlow{T},
    data::ACPowerFlowData,
    time_step::Int64;
    x0::Union{Vector{Float64}, Nothing} = nothing,
    validate_voltage_magnitudes::Bool = DEFAULT_VALIDATE_VOLTAGES,
    vm_validation_range::MinMax = DEFAULT_VALIDATION_RANGE,
    _ignored...,
) where {T <: ACPowerFlowSolverType}
    residual = ACRectangularCIResidual(data, time_step)
    if isnothing(x0)
        x0_computed = improve_x0(pf, data, residual, time_step)
    else
        x0_computed = copy(x0)
        @warn "Using caller-provided x0; skipping improve_x0."
        residual(data, x0_computed, time_step)
    end
    _log_initial_residual(residual)
    J = ACRectangularCIJacobian(data, residual, time_step)
    return residual, J, x0_computed
end

function _nr_initialize_with_jacobian_deferred(
    pf::ACMixedPowerFlow{T},
    data::ACPowerFlowData,
    time_step::Int64;
    x0::Union{Vector{Float64}, Nothing} = nothing,
    validate_voltage_magnitudes::Bool = DEFAULT_VALIDATE_VOLTAGES,
    vm_validation_range::MinMax = DEFAULT_VALIDATION_RANGE,
    _ignored...,
) where {T <: ACPowerFlowSolverType}
    residual = ACMixedCPBResidual(data, time_step)
    if isnothing(x0)
        x0_computed = improve_x0(pf, data, residual, time_step)
    else
        x0_computed = copy(x0)
        @warn "Using caller-provided x0; skipping improve_x0."
        residual(data, x0_computed, time_step)
    end
    _log_initial_residual(residual)
    J = ACMixedCPBJacobian(data, residual, time_step)
    return residual, J, x0_computed
end

# Build the Jacobian when the deferred path (polar) needs it after a failed convergence check.
# Rectangular/mixed already have J from setup.
function _nr_build_jacobian(
    ::ACPolarPowerFlow, data::ACPowerFlowData, residual::ACPowerFlowResidual, ::Nothing,
    time_step::Int64,
)
    J = ACPowerFlowJacobian(data, residual, time_step)
    J(data, time_step)
    return J
end
_nr_build_jacobian(
    ::AbstractACPowerFlow,
    ::ACPowerFlowData,
    residual,
    J,
    time_step::Int64,
) = J

"""Shared fresh-build body for `_newton_workspace!`: initialize the residual (deferring the
Jacobian per `_nr_initialize_with_jacobian_deferred`), return early on a 0-iteration warm start,
otherwise build `J`, a symbolically-factored linear-solver cache, and a fresh `StateVectorCache`.
Returns `(residual, J_or_nothing, x0_init, linSolveCache_or_nothing, stateVector_or_nothing,
converged)`. Counts the symbolic factorization it performs (a no-op outside discrete control)."""
function _fresh_newton_workspace(
    pf::AbstractACPowerFlow,
    data::ACPowerFlowData,
    time_step::Int64,
    backend,
    tol::Float64,
    init_kwargs::NamedTuple,
)
    residual, J_deferred, x0_init =
        _nr_initialize_with_jacobian_deferred(pf, data, time_step; init_kwargs...)
    converged = norm(residual.Rv, Inf) < tol
    converged && return residual, J_deferred, x0_init, nothing, nothing, true
    J = _nr_build_jacobian(pf, data, residual, J_deferred, time_step)
    return _fresh_solver_state(pf, data, time_step, backend, residual, J, x0_init, false)
end

# Polar with no candidate start: one fused F/J sweep replaces the deferred-J path.
function _fresh_newton_workspace(
    pf::ACPolarPowerFlow,
    data::ACPowerFlowData,
    time_step::Int64,
    backend,
    tol::Float64,
    init_kwargs::NamedTuple,
)
    if haskey(init_kwargs, :x0) || !_x0_has_no_candidates(pf, data, time_step)
        return @invoke _fresh_newton_workspace(
            pf::AbstractACPowerFlow, data, time_step, backend, tol, init_kwargs)
    end
    residual, J, x0_init, fused = _fused_polar_init(pf, data, time_step; init_kwargs...)
    converged = norm(residual.Rv, Inf) < tol
    converged && return residual, nothing, x0_init, nothing, nothing, true
    fused || J(data, time_step)
    return _fresh_solver_state(pf, data, time_step, backend, residual, J, x0_init, fused)
end

function _fresh_solver_state(
    pf::AbstractACPowerFlow,
    data::ACPowerFlowData,
    time_step::Int64,
    backend,
    residual,
    J,
    x0_init::Vector{Float64},
    fused::Bool,
)
    linSolveCache = _polar_jacobian_cache(backend, J.Jv)
    _symbolic_step!(pf, linSolveCache, J.Jv, data, time_step)
    _count_symbolic_factor!(data)
    stateVector = StateVectorCache(x0_init, residual.Rv)
    stateVector.fused_start[] = fused
    return residual, J, x0_init, linSolveCache, stateVector, false
end

"""Rectangular/mixed NR/TR linear-solver cache stored in `data.solver_cache`. Reused when the
rebuilt Jacobian has the recorded sparsity pattern (`colptr`, `rowval`, `m`, `n`);
`bus_type_snapshot` holds the bus types of the last factorization."""
mutable struct RectMixedNRCache{C <: PNM.LinearSolverCache} <: SolverCache
    colptr::Vector{J_INDEX_TYPE}
    rowval::Vector{J_INDEX_TYPE}
    m::Int
    n::Int
    backend::PNM.LinearSolverType
    linSolveCache::C
    stateVector::StateVectorCache
    bus_type_snapshot::Vector{PSY.ACBusTypes.Value}
end

function _build_rect_mixed_cache!(
    data::ACPowerFlowData,
    time_step::Int64,
    backend,
    Jv::SparseMatrixCSC{Float64, J_INDEX_TYPE},
    x0::Vector{Float64},
    r0::Vector{Float64},
)
    linSolveCache = make_linear_solver_cache(backend, Jv)
    symbolic_factor!(linSolveCache, Jv)
    _count_symbolic_factor!(data)
    stateVector = StateVectorCache(x0, r0)
    data.solver_cache[] = RectMixedNRCache(
        copy(Jv.colptr), copy(Jv.rowval), size(Jv, 1), size(Jv, 2), backend,
        linSolveCache, stateVector, copy(view(data.bus_type, :, time_step)),
    )
    return linSolveCache, stateVector
end

# No cache yet, or another solver's cache (e.g. FastDecoupled ran on this `data` first): rect/mixed
# share the slot with FD across an ordinary solver switch, so rebuild rather than error.
_get_or_build_rect_mixed_cache!(
    ::Union{Nothing, SolverCache}, data, time_step, backend, Jv, x0, r0,
) = _build_rect_mixed_cache!(data, time_step, backend, Jv, x0, r0)

function _get_or_build_rect_mixed_cache!(
    cache::RectMixedNRCache,
    data::ACPowerFlowData,
    time_step::Int64,
    backend,
    Jv::SparseMatrixCSC{Float64, J_INDEX_TYPE},
    x0::Vector{Float64},
    r0::Vector{Float64},
)
    if typeof(cache.backend) === typeof(backend) && _same_sparsity(cache, Jv)
        bus_type = view(data.bus_type, :, time_step)
        if bus_type != cache.bus_type_snapshot
            copyto!(cache.bus_type_snapshot, bus_type)
            _drop_numeric!(cache.linSolveCache)
        end
        stateVector = cache.stateVector
        copyto!(stateVector.x, x0)
        copyto!(stateVector.r, r0)
        _reset_for_reuse!(stateVector)
        return cache.linSolveCache, stateVector
    end
    return _build_rect_mixed_cache!(data, time_step, backend, Jv, x0, r0)
end

"""Newton workspace for one `_newton_power_flow` call. Returns
`(residual, J, x0_init, linSolveCache, stateVector, converged)`; `linSolveCache` and
`stateVector` are `nothing` when the initial point already converged."""
function _newton_workspace!(
    pf::AbstractACPowerFlow,
    data::ACPowerFlowData,
    time_step::Int64,
    backend,
    tol::Float64,
    init_kwargs::NamedTuple,
)
    return _fresh_newton_workspace(pf, data, time_step, backend, tol, init_kwargs)
end

function _newton_workspace!(
    pf::Union{ACRectangularPowerFlow, ACMixedPowerFlow},
    data::ACPowerFlowData,
    time_step::Int64,
    backend,
    tol::Float64,
    init_kwargs::NamedTuple,
)
    residual, J_deferred, x0_init =
        _nr_initialize_with_jacobian_deferred(pf, data, time_step; init_kwargs...)
    converged = norm(residual.Rv, Inf) < tol
    converged && return residual, J_deferred, x0_init, nothing, nothing, true
    J = _nr_build_jacobian(pf, data, residual, J_deferred, time_step)
    linSolveCache, stateVector = _get_or_build_rect_mixed_cache!(
        data.solver_cache[], data, time_step, backend, J.Jv, x0_init, residual.Rv)
    return residual, J, x0_init, linSolveCache, stateVector, false
end

# Dispatch on the slot content keeps the reuse path concretely inferred.
function _newton_workspace!(
    pf::ACPolarPowerFlow,
    data::ACPowerFlowData,
    time_step::Int64,
    backend,
    tol::Float64,
    init_kwargs::NamedTuple,
)
    return _polar_newton_workspace!(
        data.polar_nr_cache[], pf, data, time_step, backend, tol, init_kwargs)
end

# No cache yet (or the previous entry was invalidated on the last call): build fresh and, unless
# the initial point already converged (matching the historical lazy build), store it for reuse.
function _polar_newton_workspace!(
    ::Nothing,
    pf::ACPolarPowerFlow,
    data::ACPowerFlowData,
    time_step::Int64,
    backend,
    tol::Float64,
    init_kwargs::NamedTuple,
)
    residual, J, x0_init, linSolveCache, stateVector, converged =
        _fresh_newton_workspace(pf, data, time_step, backend, tol, init_kwargs)
    data.polar_nr_cache[] = if converged
        nothing
    else
        PolarNRCache(
            residual, J, linSolveCache, stateVector, backend,
            copy(view(data.bus_type, :, time_step)),
            # A rebuild keeps the replaced entry's scratch while the arc axis is the same.
            _arc_flow_scratch(data.polar_nr_cache[], data),
            _lean_slot(data.ac_jacobian_structure_cache[]),
            # A copy: `x0_init` may be the caller's `x0`.
            copy(x0_init),
            SubnetworkScratch(size(data.bus_type, 1)))
    end
    return residual, J, x0_init, linSolveCache, stateVector, converged
end

# A cache entry is present: try to reuse it, falling back to a fresh build (dispatching back to
# the `::Nothing` method) on any invalidation — caller-provided x0, a different backend, or a
# structural change `_refresh_polar_residual!` can't absorb in place.
function _polar_newton_workspace!(
    entry::PolarNRCache,
    pf::ACPolarPowerFlow,
    data::ACPowerFlowData,
    time_step::Int64,
    backend,
    tol::Float64,
    init_kwargs::NamedTuple,
)
    can_reuse =
        typeof(entry.backend) === typeof(backend) &&
        # The reuse path always recomputes the start point via `improve_x0`, so it can't honor
        # a caller-provided `x0`; excluding it here keeps that path from being silently ignored.
        !haskey(init_kwargs, :x0) &&
        _refresh_polar_residual!(entry, data, time_step)
    can_reuse ||
        return _polar_newton_workspace!(
            nothing,
            pf,
            data,
            time_step,
            backend,
            tol,
            init_kwargs,
        )

    residual = entry.residual
    J = entry.J
    x0_init = entry.x0
    update_state!(x0_init, data, time_step)
    # As a fresh start would: fused F and J unless `improve_x0!` has a candidate to compare.
    if _x0_has_no_candidates(pf, data, time_step)
        fused = _fused_x0!(x0_init, pf, data, residual, J, time_step)
    else
        improve_x0!(x0_init, pf, data, residual, time_step)
        fused = false
    end
    _log_initial_residual(residual)
    if get(init_kwargs, :validate_voltage_magnitudes, DEFAULT_VALIDATE_VOLTAGES)
        validate_voltage_magnitudes(
            x0_init,
            residual.validate_indices,
            get(init_kwargs, :vm_validation_range, DEFAULT_VALIDATION_RANGE),
            0,
        )
    end
    converged = norm(residual.Rv, Inf) < tol
    # Off the fused path, defer the Jacobian fill past the convergence check: a 0-iteration
    # warm start must not pay for it. `nothing` lets the caller rebuild only if it needs J.
    converged && return residual, nothing, x0_init, nothing, nothing, true
    fused || J(data, time_step)
    # Reuse the linear-solver cache (the Symbolic holds: the pattern is bus-type-agnostic) and the
    # state-vector buffers; refresh only the per-solve values.
    linSolveCache = entry.linSolveCache
    _resume_lean!(linSolveCache)
    _align_lean_plan!(linSolveCache, entry.lean, view(data.bus_type, :, time_step))
    stateVector = entry.stateVector
    copyto!(stateVector.x, x0_init)
    copyto!(stateVector.r, residual.Rv)
    _reset_for_reuse!(stateVector)
    stateVector.fused_start[] = fused
    return residual, J, x0_init, linSolveCache, stateVector, false
end

# The PQ ZIP update telescopes `P_net`/`Q_net` from the previous |V|, so after a diverged attempt
# re-evaluating at x0 alone leaves cancellation error in the loads: restore them with |V| and θ.
function _save_solve_start!(R::ACPowerFlowResidual)
    S = R.solve_start
    copyto!(view(S, :, 1), R.P_net)
    copyto!(view(S, :, 2), R.Q_net)
    copyto!(view(S, :, 3), R.bus_state.Vm)
    copyto!(view(S, :, 4), R.bus_state.θ)
    return
end

function _restore_solve_start!(R::ACPowerFlowResidual)
    S = R.solve_start
    copyto!(R.P_net, view(S, :, 1))
    copyto!(R.Q_net, view(S, :, 2))
    copyto!(R.bus_state.Vm, view(S, :, 3))
    copyto!(R.bus_state.θ, view(S, :, 4))
    R.bus_state.phasor_valid = false
    # The restored state, not `data` (holding the failed iterate), is authoritative.
    R.bus_state.data_stale = true
    return
end

# These residuals rebuild every injection from `data` on each evaluation.
_save_solve_start!(::Union{ACRectangularCIResidual, ACMixedCPBResidual}) = nothing
_restore_solve_start!(::Union{ACRectangularCIResidual, ACMixedCPBResidual}) = nothing

# Residual, Jacobian and state back at `x0`, every buffer as a reused workspace starts a solve.
function _restart_from!(
    stateVector::StateVectorCache,
    residual,
    J,
    data::ACPowerFlowData,
    x0::Vector{Float64},
    time_step::Int64,
)
    _restore_solve_start!(residual)
    _evaluate_start!(residual, J, data, x0, time_step, stateVector.fused_start[])
    copyto!(stateVector.x, x0)
    copyto!(stateVector.r, residual.Rv)
    _reset_for_reuse!(stateVector)
    return
end

function _evaluate_start!(residual, J, data::ACPowerFlowData, x0::Vector{Float64},
    time_step::Int64, ::Bool)
    residual(data, x0, time_step)
    J(data, time_step)
    return
end

function _evaluate_start!(
    residual::ACPowerFlowResidual,
    J::ACPowerFlowJacobian,
    data::ACPowerFlowData,
    x0::Vector{Float64},
    time_step::Int64,
    fused::Bool,
)
    if fused
        _update_residual_and_jacobian!(residual, J, x0, data, time_step)
    else
        residual(data, x0, time_step)
        J(data, time_step)
    end
    return
end

function _newton_power_flow(
    pf::AbstractACPowerFlow{T},
    data::ACPowerFlowData,
    time_step::Int64;
    # shared kwargs
    tol::Float64 = DEFAULT_NR_TOL,
    maxIterations::Int = DEFAULT_NR_MAX_ITER,
    validate_voltage_magnitudes::Bool = DEFAULT_VALIDATE_VOLTAGES,
    vm_validation_range::MinMax = DEFAULT_VALIDATION_RANGE,
    # NR-specific
    refinement_threshold::Float64 = DEFAULT_REFINEMENT_THRESHOLD,
    refinement_eps::Float64 = DEFAULT_REFINEMENT_EPS,
    iwamoto::Bool = false,
    # TR-specific
    factor::Float64 = DEFAULT_TRUST_REGION_FACTOR,
    eta::Float64 = DEFAULT_TRUST_REGION_ETA,
    autoscale::Bool = DEFAULT_AUTOSCALE,
    iwamoto_fallback::Bool = DEFAULT_IWAMOTO_FALLBACK,
    # NR and TR: fold / voltage-collapse bail-out (any backend; κ̂ is KLU-only)
    stop_at_fold::Bool = false,
    # initialize_power_flow_variables
    x0::Union{Vector{Float64}, Nothing} = nothing,
    # linear solver backend, resolved by `PNM.resolve_linear_solver`. Canonical names:
    # "KLU" | "AppleAccelerateLU" | "MKLPardiso" (PNM is the source of truth for any
    # aliases); `nothing` uses PNM's platform default.
    linear_solver::Union{Nothing, AbstractString} = nothing,
    _ignored...,
) where {T <: Union{TrustRegionACPowerFlow, NewtonRaphsonACPowerFlow}}

    # setup: common code
    init_kwargs = if isnothing(x0)
        (; validate_voltage_magnitudes, vm_validation_range)
    else
        (; validate_voltage_magnitudes, vm_validation_range, x0)
    end
    backend = resolve_linear_solver_backend(linear_solver)
    # `J_or_nothing` is `nothing` exactly when the initial point already converged: the Jacobian is
    # deferred past the convergence check so a 0-iteration warm start never pays for it.
    residual, J_or_nothing, x0_init, linSolveCache, stateVector, converged =
        _newton_workspace!(pf, data, time_step, backend, tol, init_kwargs)

    i = 0
    x_final = x0_init
    if !converged
        J = J_or_nothing
        run_method() = _run_power_flow_method(
            time_step,
            stateVector,
            linSolveCache,
            residual,
            J,
            data,
            T;
            tol,
            maxIterations,
            validate_voltage_magnitudes,
            vm_validation_range,
            refinement_threshold,
            refinement_eps,
            iwamoto,
            factor,
            eta,
            autoscale,
            iwamoto_fallback,
            stop_at_fold,
        )
        reused = _reuses_pivot_order(linSolveCache)
        reused && _save_solve_start!(residual)
        converged, i = run_method()
        if !converged && reused
            # p3s nr_klu.cpp:798-820: rerun once from x0 as a fresh KLU solve would, so a reused
            # pivot order never changes a solve's status.
            @debug "solve failed on a reused pivot order; retrying on a fresh factorization" time_step
            PNM.KLUWrapper.cold_restart!(linSolveCache)
            _restart_from!(stateVector, residual, J, data, x0_init, time_step)
            failed = i
            converged, i = run_method()
            i += failed
        end
        x_final = stateVector.x
        _finalize_formulation!(pf, data, x_final, residual, time_step)
        return _finalize_power_flow(
            converged, i, string(T), residual, data, J.Jv, time_step)
    end
    _finalize_formulation!(pf, data, x_final, residual, time_step)
    # 0-iteration warm start: skip the Jacobian-dependent post-processing UNLESS the caller
    # opted into loss / voltage-stability factors — those need J even at 0 iterations, or a
    # first solve that lands within tol would leave them at their zero-initialized values.
    if get_calculate_loss_factors(data) || get_calculate_voltage_stability_factors(data)
        J = _nr_build_jacobian(pf, data, residual, J_or_nothing, time_step)
        return _finalize_power_flow(
            converged, i, string(T), residual, data, J.Jv, time_step)
    end
    return _finalize_power_flow(
        converged, i, string(T), residual, data, nothing, time_step)
end
