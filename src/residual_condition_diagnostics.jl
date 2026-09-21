#=
Per-iteration solver diagnostics (`log_solver_diagnostics`) and a fold /
voltage-collapse bail-out (`stop_at_fold`).

λ_min is taken on the bus-voltage Schur complement S = A − B·D⁻¹·C of the blocked
Jacobian J = [A B; C D], whose non-bus tail (LCC + VSC + area interchange,
`state_tail_length(data, dcn)` rows/cols) is the trailing block. The (1,1) block of
J⁻¹ is exactly S⁻¹, so v ↦ (J⁻¹·[v; 0])[1:nb] applies S⁻¹ from the *existing*
factorization of J — no second matrix or factorization. With no tail, S = J. The
monitor line and the bail-out share one refactor and one eigensolve via
`run_solver_diagnostics!`.
=#

"""Round to 4 significant figures (one more digit than `siground`'s 3)."""
_sf4(x) = round(x; sigdigits = 4)

"""Applies S⁻¹ via a back-solve of the full `J`: pads `v` with zeros in the
LCC-tail slots, applies `J⁻¹`, returns the leading `n_bus` block."""
struct SchurInverseOperator{C}
    cache::C
    n_bus::Int
    buffer::Vector{Float64}   # padded RHS, length = full state
end

"""Condition estimate κ̂(J), or `NaN` when the backend exposes none. The NaN
fallback is restricted to the non-KLU `PFLinearSolverCache` members so the concrete
`KLULinSolveCache` doesn't shadow the KLU method onto the NaN path."""
_diag_condest(cache::PNM.KLULinSolveCache) = condest!(cache)
_diag_condest(::Union{PNM.AAFactorCache, PardisoLinSolveCache}) = NaN

function (op::SchurInverseOperator)(v::AbstractVector{Float64})
    b = op.buffer
    @inbounds begin
        copyto!(view(b, 1:(op.n_bus)), v)
        fill!(view(b, (op.n_bus + 1):length(b)), 0.0)
    end
    solve!(op.cache, b)
    # KrylovKit stores each returned vector, so hand back a fresh copy of the
    # bus block rather than the reused buffer.
    return b[1:(op.n_bus)]
end

"""Smallest-magnitude eigenvalue of the Schur complement `S` by inverse iteration:
KrylovKit finds the largest-magnitude eigenvalue `μ` of `S⁻¹` and returns `1/μ`.
`S` is non-symmetric, so the result may be complex. Returns `(λ_min, converged)`,
with `converged = false` (and `λ_min = NaN ± NaN im`) on any failure."""
function _schur_min_eigenvalue(
    op::SchurInverseOperator;
    tol::Float64 = 1e-6,
    maxiter::Int = 200,
    krylovdim::Int = 30,
)::Tuple{ComplexF64, Bool}
    n = op.n_bus
    v0 = fill(1.0 / sqrt(n), n)   # deterministic init for reproducible logs
    vals, _, info = KrylovKit.eigsolve(op, v0, 1, :LM; tol, maxiter, krylovdim)
    if info.converged < 1 || isempty(vals)
        return complex(NaN, NaN), false
    end
    μ = vals[1]
    # A (near-)zero dominant eigenvalue of S⁻¹ makes 1/μ overflow; treat it as
    # not-converged rather than reporting an Inf eigenvalue of S.
    if abs(μ) <= eps(Float64)
        return complex(NaN, NaN), false
    end
    return inv(ComplexF64(μ)), true
end

"""Format a (possibly complex) eigenvalue to 4 significant figures as `a` or
`a ± b im`."""
function _fmt_eig(z::Number)
    iz = imag(z)
    return if iz == 0
        "$(_sf4(real(z)))"
    else
        "$(_sf4(real(z))) $(iz < 0 ? "-" : "+") $(_sf4(abs(iz)))im"
    end
end

"""The system bus number for the `bus_ix`-th bus (reduced ordering)."""
_diag_bus_number(data::ACPowerFlowData, bus_ix::Int) =
    axes(data.power_network_matrix, 1)[bus_ix]

const _LCC_RESIDUAL_ROW_NAMES =
    ("P-setpoint", "DC-line balance", "rectifier α-limit", "inverter α-limit")

"""Describe a residual entry that falls in the LCC tail (4 rows per LCC)."""
function _describe_lcc_residual_entry(data::ACPowerFlowData, tail_ix::Int)
    i = div(tail_ix - 1, 4) + 1
    row = mod1(tail_ix, 4)
    from_no, to_no = data.lcc.arcs[i]
    return "LCC $(from_no)→$(to_no) ($(_LCC_RESIDUAL_ROW_NAMES[row]))"
end

"""Describe a residual entry that falls in the VSC tail: two control rows per converter
(`r1` active-power/V_dc, `r2` reactive-power/|V_ac|) followed by one DC-node KCL row per DC
node -- the layout [`_set_vsc_tail_residuals!`](@ref) writes."""
function _describe_vsc_residual_entry(
    data::ACPowerFlowData,
    dcn::DCNetwork,
    tail_ix::Int,
)
    nconv = n_vsc_converters(dcn)
    if tail_ix <= 2 * nconv
        c = div(tail_ix - 1, 2) + 1
        row = isodd(tail_ix) ? "P/V_dc control" : "Q/|V_ac| control"
        bus_no = _diag_bus_number(data, dcn.converter_ac_bus_ix[c])
        return "VSC converter $c at bus $bus_no ($row)"
    end
    return "DC node $(tail_ix - 2 * nconv) (DC KCL)"
end

"""Describe a residual entry in the non-bus tail, `tail_ix` being 1-based within the tail.
Bands are `[LCC][VSC][area]`, sized from the same terms as
[`state_tail_length`](@ref); every index in `1:state_tail_length(...)` lands in exactly one."""
function _describe_tail_residual_entry(data::ACPowerFlowData, tail_ix::Int)
    n_lcc_rows = 4 * size(data.lcc.p_set, 1)
    tail_ix <= n_lcc_rows && return _describe_lcc_residual_entry(data, tail_ix)
    dcn = get_dc_network(data)
    n_vsc_rows = vsc_tail_length(dcn)
    tail_ix <= n_lcc_rows + n_vsc_rows &&
        return _describe_vsc_residual_entry(data, dcn, tail_ix - n_lcc_rows)
    return _describe_area_residual_entry(data, tail_ix - n_lcc_rows - n_vsc_rows)
end

"""Describe a residual entry that falls in the area-interchange tail (1 row per
controlled area, keyed by `tail_ix` rather than vector position so a mid-solve
de-enrollment renumbering can't desync this from `_set_area_tail_residuals!`)."""
function _describe_area_residual_entry(data::ACPowerFlowData, tail_ix::Int)
    for area in data.area_interchange.areas
        area.tail_ix == tail_ix && return "area $(area.name) (NI−PDES)"
    end
    error(
        "area_interchange tail_ix=$tail_ix not found among " *
        "$(length(data.area_interchange.areas)) controlled areas",
    )
end

"""`(bus index, 1-based row within that bus's block)` for variable-block
formulations, from the `bus_state_offset` table."""
function _locate_variable_block(offsets::AbstractVector, ix::Int)
    b = searchsortedlast(offsets, ix)
    return b, ix - Int(offsets[b]) + 1
end

# Formulation-aware label for the entry where ‖F‖∞ is attained. The bus block is
# laid out first; the polar-only tail is `[LCC][VSC][area]` (area rows LAST, see
# `area_tail_offset`) — rectangular/mixed never carry an area tail (area-interchange
# control is rejected at construction for those formulations).
function _describe_residual_entry(
    ::ACPowerFlowResidual,
    data::ACPowerFlowData,
    time_step::Int,
    ix::Int,
)
    n_bus_eqs = 2 * size(data.bus_type, 1)
    if ix <= n_bus_eqs
        bus_ix = div(ix - 1, 2) + 1
        return "bus $(_diag_bus_number(data, bus_ix)) ($(isodd(ix) ? "P" : "Q"))"
    end
    return _describe_tail_residual_entry(data, ix - n_bus_eqs)
end

function _describe_residual_entry(
    r::ACRectangularCIResidual,
    data::ACPowerFlowData,
    ::Int,
    ix::Int,
)
    if ix <= r.total_bus_state
        b, row = _locate_variable_block(r.bus_state_offset, ix)
        labels = ("ΔI_re", "ΔI_im", "|V|²−V_set²")   # PV uses the 3rd row
        return "bus $(_diag_bus_number(data, b)) ($(labels[row]))"
    end
    return _describe_tail_residual_entry(data, ix - r.total_bus_state)
end

function _describe_residual_entry(
    r::ACMixedCPBResidual,
    data::ACPowerFlowData,
    time_step::Int,
    ix::Int,
)
    if ix <= r.total_bus_state
        b, row = _locate_variable_block(r.bus_state_offset, ix)
        bt = data.bus_type[b, time_step]
        labels = if bt == PSY.ACBusTypes.PV
            ("ΔP", "|V|²−V_set²")
        elseif bt == PSY.ACBusTypes.PQ
            ("ΔI_im", "ΔI_re")
        else  # REF
            ("ΔI_re", "ΔI_im")
        end
        return "bus $(_diag_bus_number(data, b)) ($(labels[row]))"
    end
    return _describe_tail_residual_entry(data, ix - r.total_bus_state)
end

# ---------------------------------------------------------------------------
# Fold / voltage-collapse bail-out state and the shared per-iteration hook.
# ---------------------------------------------------------------------------

"""Deterministic pseudo-random unit vector: a *generic* direction is all the
bordering needs, and determinism keeps the logged monitor values reproducible across
runs and Julia versions (`Random`'s streams are not version-stable)."""
_fill_border_vector!(v::Vector{Float64}, k::Int) =
    normalize!(v .= sin.((1:length(v)) .* (k * FOLD_BORDER_STRIDE)))

"""
    BorderedFoldMonitor(n_state)

Fold monitor tracking `sign(det J)` through bordered systems. For fixed vectors
`b`, `c` and scalar `d`, border `J` as

    M = [ J   b
          cᵀ  d ]

The Schur complement gives `det(M) = det(J) · (d − cᵀJ⁻¹b)`, so with

    s = d − cᵀJ⁻¹b        g = 1/s = det(J) / det(M)

`g` is smooth along the continuation and vanishes exactly when `J` is singular:
`det M` is a fixed smooth function, nonzero near a simple fold for generic `b`, `c`,
so `sign(g)` differs from `sign(det J)` only by the constant factor `sign(det M)`.
Therefore **flips of `g` are flips of `det J`**: tracking `g` tells us which branch
we are on.

`g` can also flip through a **pole**, where `det M` — not `det J` — crossed zero; the
bordering degenerated and `g` says nothing about `J`. Two *independent* borderings
settle that without any extra machinery: a zero of `det J` flips both on the same
step, a pole flips only the bordering that degenerated. A lone flip re-picks that
bordering and the solve continues, up to `FOLD_MAX_BORDER_REPICKS` times before the
monitor disables itself.

Cost per iteration: `FOLD_N_BORDERINGS` back-solves against the *existing*
factorization plus a dot product each."""
mutable struct BorderedFoldMonitor
    b::Vector{Vector{Float64}}
    c::Vector{Vector{Float64}}
    y::Vector{Float64}          # work buffer: holds J⁻¹b after `solve!`
    gs::Vector{Float64}         # this iteration's g per bordering
    signs::Vector{Int8}         # sign(g) last seen per bordering; 0 = nothing yet
    attempts::Vector{Int}       # bordering index per slot; bumped on every re-pick
    enabled::Bool               # false once the borderings degenerated too often
end

function BorderedFoldMonitor(n_state::Int)
    mon = BorderedFoldMonitor(
        [Vector{Float64}(undef, n_state) for _ in 1:FOLD_N_BORDERINGS],
        [Vector{Float64}(undef, n_state) for _ in 1:FOLD_N_BORDERINGS],
        Vector{Float64}(undef, n_state), Vector{Float64}(undef, FOLD_N_BORDERINGS),
        zeros(Int8, FOLD_N_BORDERINGS), zeros(Int, FOLD_N_BORDERINGS), true,
    )
    for k in 1:FOLD_N_BORDERINGS
        _repick_bordering!(mon, k)
    end
    return mon
end

"""Draw a fresh bordering into slot `k` and forget its sign: after a re-pick
`sign(det M)` is a different constant, so old signs are not comparable."""
function _repick_bordering!(mon::BorderedFoldMonitor, k::Int)
    mon.attempts[k] += 1
    # Distinct strides per (slot, attempt) so the two borderings are never the same
    # vector — independence is the whole zero-vs-pole test.
    _fill_border_vector!(mon.b[k], 2 * (k + FOLD_N_BORDERINGS * mon.attempts[k]))
    _fill_border_vector!(mon.c[k], 2 * (k + FOLD_N_BORDERINGS * mon.attempts[k]) + 1)
    mon.signs[k] = Int8(0)
    return mon
end

"""`g = 1/(d − cᵀJ⁻¹b)` for bordering `k` at the current iterate, via one back-solve
against `cache`'s existing factorization. Non-finite (`±Inf`/`NaN`) means the
bordering is degenerate or the back-solve failed."""
function _fold_monitor_value!(
    mon::BorderedFoldMonitor,
    cache::PFLinearSolverCache,
    k::Int = 1,
)
    copyto!(mon.y, mon.b[k])
    solve!(cache, mon.y)
    return inv(FOLD_BORDER_D - dot(mon.c[k], mon.y))
end

"""Update the monitor with this iteration's `g` values and decide the bail-out.
Returns `true` to abort the search. Every bordering flipping sign is a singular `J`
— the fold; a lone flip (or a non-finite `g`) is that bordering degenerating, which
re-picks it and continues. With `bail = false` the same classification is logged but
never aborts."""
function _decide_det_sign_switch!(
    mon::BorderedFoldMonitor,
    label::AbstractString,
    gs::AbstractVector{Float64},
    bail::Bool,
)::Bool
    mon.enabled || return false

    flipped = UInt8(0)          # bit k set = bordering k flipped this iteration
    n_flipped = 0
    n_finite = 0
    n_voting = 0                # borderings with an established previous sign
    for k in eachindex(gs)
        g = gs[k]
        if !isfinite(g)
            # s == 0 (|g| = Inf) or a failed back-solve: this bordering says nothing
            # about J. Re-pick it; the others still cover this iteration.
            _handle_border_pole!(mon, label, k)
            mon.enabled || return false
            continue
        end
        n_finite += 1
        current = Int8(sign(g))
        current == 0 && continue         # exactly zero: hold the previous sign
        prev = mon.signs[k]
        mon.signs[k] = current
        prev == 0 && continue            # first observation for this bordering
        n_voting += 1
        if current != prev
            flipped |= UInt8(1) << (k - 1)
            n_flipped += 1
        end
    end

    if n_finite == 0
        @warn "$label: every fold-monitor bordering is degenerate; read as a " *
              "fold$(bail ? ", aborting." : ".")"
        return bail
    end
    n_flipped == 0 && return false

    # A bordering re-picked last iteration has no previous sign to vote with, so the
    # verdict is taken over the ones that do.
    if n_flipped < n_voting
        # Independent borderings disagree, so det(J) did not cross zero: the ones that
        # flipped hit a pole of their own det(M). Re-pick them and keep going.
        for k in eachindex(gs)
            iszero(flipped & (UInt8(1) << (k - 1))) || _handle_border_pole!(mon, label, k)
        end
        return false
    end

    @warn "$label: sign(det J) flipped on all $(n_voting) borderings. Fold / " *
          "voltage-collapse signature$(bail ? ", aborting." : ".")"
    return bail
end

"""Handle a degenerate bordering: `det M` — not `J` — went singular. Re-pick slot `k`
and keep going; after `FOLD_MAX_BORDER_REPICKS` failures disable the monitor rather
than report a fold that was never observed."""
function _handle_border_pole!(mon::BorderedFoldMonitor, label::AbstractString, k::Int)
    if mon.attempts[k] > FOLD_MAX_BORDER_REPICKS
        mon.enabled = false
        @warn "$label: bordering $k degenerated $(mon.attempts[k])×; disabling fold " *
              "detection — solve continues with NO fold bail-out."
        return
    end
    @debug "$label: bordering $k passed through a pole of det(M) — degenerate " *
           "bordering, not a property of J. Re-picking it."
    _repick_bordering!(mon, k)
    return
end

"""Per-solve scratch for [`run_solver_diagnostics!`](@ref): previous ‖F‖∞ (`prev_F`),
the bordered `sign(det J)` fold monitor (`fold`), and a reusable padded RHS
(`buffer`) so the Schur operator allocates nothing per iteration."""
mutable struct SolverDiagnosticsState
    prev_F::Float64
    fold::BorderedFoldMonitor
    buffer::Vector{Float64}
end

SolverDiagnosticsState(n_state::Int) = SolverDiagnosticsState(
    NaN, BorderedFoldMonitor(n_state), Vector{Float64}(undef, n_state))

"""Set up a solver loop's diagnostics: returns `(monitor, diag_state)`, allocating
the scratch only when a diagnostic or the bail-out is on so the default solve path
allocates nothing. `diag_state` is `nothing` when neither is requested."""
function setup_solver_diagnostics(
    J::Union{ACPowerFlowJacobian, ACRectangularCIJacobian, ACMixedCPBJacobian},
    bail::Bool,
)
    monitor = get_log_solver_diagnostics(J.data)
    diag_state = (monitor || bail) ? SolverDiagnosticsState(size(J.Jv, 1)) : nothing
    return monitor, diag_state
end

"""Run one iteration's diagnostics against the current `J`/residual. Does the
*single* per-iteration refactor of `cache` on `J.Jv` (NR/TR pass `linSolveCache`, LM
its own KLU `diag_cache`), then the bordered `sign(det J)` monitor (one back-solve
per bordering) and, when the log line is on, the Schur eigensolve behind `λ_min(S)`.
Returns `true` iff the caller should abort. A `SingularException` is itself a fold
signature: under `bail` it aborts, under monitor-only it reports `singular` and
continues; any other exception is rethrown."""
function run_solver_diagnostics!(
    state::SolverDiagnosticsState,
    label::AbstractString,
    residual::Union{ACPowerFlowResidual, ACRectangularCIResidual, ACMixedCPBResidual},
    J::Union{ACPowerFlowJacobian, ACRectangularCIJacobian, ACMixedCPBJacobian},
    time_step::Int,
    cache::PFLinearSolverCache,
    monitor::Bool,
    bail::Bool,
)::Bool
    # KLU throws `SingularException` on a singular J; AppleAccelerate does not (it
    # silently returns garbage but still factors), so only KLU reaches the catch.
    singular = false
    try
        numeric_refactor!(cache, J.Jv)
    catch e
        e isa LinearAlgebra.SingularException || rethrow()
        singular = true
    end

    data = J.data
    if singular
        if bail
            @warn "$label: the Jacobian is singular; this is a fold / " *
                  "voltage-collapse signature, aborting the search."
            return true
        end
        # Monitor-only: report the singularity rather than crashing, and leave
        # `state.prev_F` untouched so the next contraction ratio is meaningful.
        abs_max, ix = findmax(abs, residual.Rv)
        @info "$label: ‖F‖_∞ = $(_sf4(abs_max)) at " *
              "$(_describe_residual_entry(residual, data, time_step, ix)), " *
              "κ̂(J) = singular, λ_min(S) = singular, sign(det J) = singular"
        return false
    end

    # The bordered monitor is a back-solve per bordering, so it runs whenever
    # diagnostics are on; only `bail` decides whether a fold signature aborts.
    fold = state.fold
    if fold.enabled
        for k in eachindex(fold.gs)
            fold.gs[k] = _fold_monitor_value!(fold, cache, k)
        end
    else
        fill!(fold.gs, NaN)
    end

    if monitor
        # Trailing block is the FULL non-bus tail (LCC + VSC + area interchange), not
        # just LCC: n_state on a VSC/area-interchange system is larger than
        # 2*nbuses + 4*n_lcc.
        n_state = size(J.Jv, 1)
        n_bus = n_state - state_tail_length(data, get_dc_network(data))
        op = SchurInverseOperator(cache, n_bus, state.buffer)
        λ_min, eig_converged = _schur_min_eigenvalue(op)

        abs_max, ix = findmax(abs, residual.Rv)
        κ = _diag_condest(cache)
        λ_str = if eig_converged
            "$(_fmt_eig(λ_min)) (|λ_min| = $(_sf4(abs(λ_min))))"
        else
            "not-converged"
        end
        parts = [
            "‖F‖_∞ = $(_sf4(abs_max)) at " *
            "$(_describe_residual_entry(residual, data, time_step, ix))",
            "κ̂(J) = $(isnan(κ) ? "n/a (KLU-only)" : string(_sf4(κ)))",
            "λ_min(S) = $λ_str",
            # sign(g) = sign(det J)·sign(det M); det M is a fixed constant, so only
            # FLIPS of this sign are meaningful, not the sign itself.
            "sign(det J) = $(_fmt_det_sign(first(fold.gs)))",
        ]
        if !isnan(state.prev_F) && state.prev_F > 0
            push!(parts, "contraction = $(_sf4(abs_max / state.prev_F))")
        end
        @info "$label: " * join(parts, ", ")
        state.prev_F = abs_max
    end

    return _decide_det_sign_switch!(fold, label, fold.gs, bail)
end

"""`+`/`−` for the monitor's sign, or `n/a` when `g` is unavailable."""
_fmt_det_sign(g::Float64) = !isfinite(g) ? "n/a" : (g > 0 ? "+" : (g < 0 ? "−" : "0"))

"""
    _report_area_interchange_failure(data, time_step)

Terminal-failure diagnostic for embedded area net-interchange control, called when the
greedy relax loop exhausts the enrolled set without converging. The WORKING set is empty
at that point, so this reports against the PRISTINE tie/area set at the last attempted
iterate's bus state, naming the area with the largest-magnitude interchange-row residual.
No-op if area interchange control was never enrolled.
"""
function _report_area_interchange_failure(data::ACPowerFlowData, time_step::Int)
    aid = data.area_interchange
    isempty(aid.pristine_areas) && return
    gaps = [
        _area_net_interchange(
            aid.pristine_ties, aid.pristine_dc_ties, area.tail_ix, data, time_step,
        ) - area.pdes
        for area in aid.pristine_areas
    ]
    abs_max, ix = findmax(abs, gaps)
    area = aid.pristine_areas[ix]
    @warn "Area interchange: Newton did not converge after the greedy relax loop " *
          "de-enrolled every controlled area (network non-convergence, not a relaxed " *
          "schedule); the largest interchange-row residual at the last attempted " *
          "iterate is area \"$(area.name)\" with |r| = $(_sf4(abs_max)) " *
          "(target PDES = $(area.pdes))."
    return
end
