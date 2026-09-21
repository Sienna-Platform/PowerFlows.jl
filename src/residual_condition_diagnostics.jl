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

# FIXME a bit over-engineered.
"""Deterministic pseudo-random unit vector, filled by an LCG so the bordering — and
therefore every logged monitor value — is reproducible across runs, platforms, and
Julia versions (no `Random` dependency, whose streams are not version-stable)."""
function _fill_border_vector!(v::Vector{Float64}, seed::UInt64)
    s = seed * 0x9E3779B97F4A7C15 + 0x1
    @inbounds for i in eachindex(v)
        s = s * 0x5851F42D4C957F2D + 0x14057B7EF767814F
        v[i] = 2.0 * ((s >> 11) * 2.0^-53) - 1.0   # uniform in [-1, 1)
    end
    v ./= norm(v)
    return v
end

"""
    BorderedFoldMonitor(n_state)

Fold monitor tracking `sign(det J)` through a bordered system. For fixed vectors
`b`, `c` and scalar `d`, border `J` as

    M = [ J   b
          cᵀ  d ]

The Schur complement gives `det(M) = det(J) · (d − cᵀJ⁻¹b)`, so with

    s = d − cᵀJ⁻¹b        g = 1/s = det(J) / det(M)

`g` is smooth along the continuation and vanishes exactly when `J` is singular:
`det M` is a fixed smooth function, nonzero near a simple fold for generic `b`, `c`,
so `sign(g)` differs from `sign(det J)` only by the constant factor `sign(det M)`. Therefore
**flips of `g` are flips of `det J`**: we can monitor which branch we're on via tracking `g`.

Cost per iteration: one back-solve `J y = b` against the *existing* factorization
plus a dot product.

`g` can flip sign two ways:

  - through **zero**: a genuine singularity of `J` — the fold;
  - through a **pole**: `det M` crossed zero. The bordering degenerated and `g` says
    nothing about `J`. The monitor re-picks `b`, `c` and restarts its sign tracking
    (it cannot recompute over past iterates, which are gone), up to
    `FOLD_MAX_BORDER_REPICKS` times before disabling itself.

The two are told apart by bisecting the *step* that produced the flip
(`_bracket_fold_flip!`): as the bracket tightens onto the event one determinant
vanishes there while the other stays continuous and nonzero, so `|g|` shrinks
geometrically onto a zero and grows onto a pole. Only that limiting *trend* is used.
The magnitude of `|g|` carries no usable information on its own — `|g| =
|det J|/|det M|`, and nothing pins `|det M|`, so a small `|g|` may mean a vanishing
`det J` or merely a swollen `det M`.

Bisection needs `residual` and `J` evaluated at the passed iterate: NR/TR supply one,
LM does not (its `J` lags the residual by a step). Without it a flip cannot be
classified at all and is read conservatively as a fold.
"""
mutable struct BorderedFoldMonitor
    b::Vector{Float64}
    c::Vector{Float64}
    d::Float64
    y::Vector{Float64}      # work buffer: holds J⁻¹b after `solve!`
    attempt::Int            # bordering index; bumped on every re-pick
    sign::Int8              # sign(g) last seen; 0 = nothing seen yet
    prev_abs_g::Float64     # |g| at the previous iteration (NaN = none)
    enabled::Bool           # false once the bordering has degenerated too often
    prev_x::Vector{Float64} # previous iterate, the low end of a bracketed step
    has_prev_x::Bool
    x_work::Vector{Float64} # interpolated iterate used while bracketing
end

function BorderedFoldMonitor(n_state::Int)
    mon = BorderedFoldMonitor(
        Vector{Float64}(undef, n_state), Vector{Float64}(undef, n_state),
        FOLD_BORDER_D, Vector{Float64}(undef, n_state),
        0, Int8(0), NaN, true,
        Vector{Float64}(undef, n_state), false, Vector{Float64}(undef, n_state),
    )
    _repick_bordering!(mon)
    return mon
end

"""Draw a fresh bordering `(b, c)` and reset the sign/magnitude history: after a
re-pick `sign(det M)` is a different constant, so old signs are not comparable."""
function _repick_bordering!(mon::BorderedFoldMonitor)
    mon.attempt += 1
    seed = FOLD_BORDER_SEED * UInt64(mon.attempt)
    _fill_border_vector!(mon.b, seed)
    _fill_border_vector!(mon.c, seed + 0x9E37)
    mon.sign = Int8(0)
    mon.prev_abs_g = NaN
    return mon
end

"""`g = 1/(d − cᵀJ⁻¹b)` at the current iterate, via one back-solve against `cache`'s
existing factorization. Returns `Inf` when the bordering is exactly degenerate
(`s == 0`) and `NaN` if the back-solve produced a non-finite `s`."""
function _fold_monitor_value!(mon::BorderedFoldMonitor, cache::PFLinearSolverCache)
    copyto!(mon.y, mon.b)
    solve!(cache, mon.y)
    s = mon.d - dot(mon.c, mon.y)
    isfinite(s) || return NaN
    return inv(s)   # ±Inf when s == 0: a pole, handled as a degenerate bordering
end

"""Update the monitor with this iteration's `g` and decide the bail-out. Returns
`true` to abort the search. A sign flip through zero (or an indeterminate `g`) is a
fold; a flip through a pole is a degenerate bordering, which re-picks `(b, c)` and
continues. An ambiguous flip is resolved by `bracket` (see
[`_bracket_fold_flip!`](@ref)) when the caller supplies one. With `bail = false` the
same classification is logged but never aborts."""
function _decide_det_sign_switch!(
    mon::BorderedFoldMonitor,
    label::AbstractString,
    g::Float64,
    bail::Bool,
    bracket::Union{Function, Nothing} = nothing,
)::Bool
    mon.enabled || return false

    if isnan(g)
        @warn "$label: fold monitor g = det(J)/det(M) indeterminate (non-finite " *
              "back-solve); read as a fold$(bail ? ", aborting." : ".")"
        return bail
    end

    abs_g = abs(g)
    current = Int8(sign(g))
    prev, prev_abs = mon.sign, mon.prev_abs_g

    # s == 0, |g| = Inf: a pole by definition.
    if !isfinite(abs_g)
        _handle_border_pole!(mon, label, "|g| = Inf (cᵀJ⁻¹b hit d exactly)")
        return false
    end

    # No flip: first observation, or unchanged sign.
    if prev == 0 || current == 0 || current == prev
        current == 0 || (mon.sign = current)
        mon.prev_abs_g = abs_g
        return false
    end

    # Flipped. Zero (fold) or pole (degenerate bordering)? Bisect the step: as the
    # bracket tightens onto the event, |g| → 0 at a zero and → ∞ at a pole whatever
    # det(M) is doing.
    event = _bracket_fold_flip!(mon, label, bracket, prev_abs, abs_g, prev)
    if event === :pole
        _handle_border_pole!(mon, label, _fold_evidence(event))
        return false
    end
    mon.sign = current
    mon.prev_abs_g = abs_g

    @warn "$label: sign(det J) flipped " *
          "($(prev > 0 ? "+" : "−") → $(current > 0 ? "+" : "−")); " *
          "$(_fold_evidence(event)). Fold / voltage-collapse " *
          "signature$(bail ? ", aborting." : ".")"
    return bail
end

"""Phrase the evidence behind a flip verdict."""
_fold_evidence(event::Symbol) =
    if event === :zero
        "bisection drove |g| DOWN — through zero"
    elseif event === :pole
        "bisection drove |g| UP"
    elseif event === :unavailable
        "no bracketing (unsupported by this solver) — read conservatively as zero"
    else  # :unresolved
        "bisection inconclusive — read conservatively as zero"
    end

"""Classify a sign flip of `g` by bisecting the step that produced it.

`bracket(θ)` re-evaluates `g` at the interpolated iterate
`x_prev + θ·(x − x_prev)`; the caller is responsible for restoring `residual`/`J`
afterwards. Bisection keeps the sub-interval that still brackets the sign change, so
after `FOLD_BRACKET_STEPS` steps both ends sit next to the event. Near a simple
zero, `|g| ∼ C·|θ − θ*|` shrinks geometrically as the bracket tightens; near a
simple pole, `|g| ∼ C/|θ − θ*|` grows.

Returns `:zero`, `:pole`, `:unavailable` (no bracket callable, or no previous
iterate), or `:unresolved` (bisection did not separate the two)."""
function _bracket_fold_flip!(
    mon::BorderedFoldMonitor,
    label::AbstractString,
    bracket::Union{Function, Nothing},
    abs_lo::Float64,
    abs_hi::Float64,
    sign_lo::Int8,
)::Symbol
    (isnothing(bracket) && return :unavailable)
    mon.has_prev_x || return :unavailable

    θ_lo, θ_hi = 0.0, 1.0
    a_lo, a_hi = abs_lo, abs_hi
    for _ in 1:FOLD_BRACKET_STEPS
        θ_mid = 0.5 * (θ_lo + θ_hi)
        g_mid = bracket(θ_mid)
        # An exactly singular J mid-step IS the zero we were looking for; a
        # non-finite g there is the pole.
        isnan(g_mid) && return :zero
        isfinite(g_mid) || return :pole
        a_mid = abs(g_mid)
        if Int8(sign(g_mid)) == sign_lo
            θ_lo, a_lo = θ_mid, a_mid
        else
            θ_hi, a_hi = θ_mid, a_mid
        end
    end

    outer, inner = min(abs_lo, abs_hi), max(a_lo, a_hi)
    inner < outer && return :zero
    min(a_lo, a_hi) > max(abs_lo, abs_hi) && return :pole
    @debug "$label: bisection did not separate a zero from a pole " *
           "(flanking |g| ∈ [$(_sf4(min(abs_lo, abs_hi))), $(_sf4(max(abs_lo, abs_hi)))], " *
           "bracketed |g| ∈ [$(_sf4(min(a_lo, a_hi))), $(_sf4(inner))])"
    return :unresolved
end

"""Handle a pole of `g`: `det M` crossed zero, so the bordering — not `J` — went
singular. Re-pick `(b, c)` and keep going; after `FOLD_MAX_BORDER_REPICKS` failures
disable the monitor rather than report a fold that was never observed."""
function _handle_border_pole!(
    mon::BorderedFoldMonitor,
    label::AbstractString,
    detail::AbstractString,
)
    if mon.attempt > FOLD_MAX_BORDER_REPICKS
        mon.enabled = false
        @warn "$label: bordering degenerated $(mon.attempt)× ($detail); " *
              "disabling fold detection — solve continues with NO fold bail-out."
        return
    end
    @warn "$label: g passed through a pole ($detail): det(M) crossed zero — " *
          "degenerate bordering, not a property of J. Re-picking (b, c)."
    _repick_bordering!(mon)
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
its own KLU `diag_cache`), then the bordered `sign(det J)` monitor (one back-solve)
and, when the log line is on, the Schur eigensolve behind `λ_min(S)`. Returns `true`
iff the caller should abort. A `SingularException` is itself a fold signature: under
`bail` it aborts, under monitor-only it reports `singular` and continues; any other
exception is rethrown.

Pass `x` (the current iterate) only when `residual` and `J` are both evaluated *at*
it: that lets an ambiguous sign flip be resolved by bracketing the step, which
re-evaluates both at interpolated iterates and restores them afterwards. NR/TR
satisfy this; LM does not (its `J` lags the residual by a step) and passes
`nothing`."""
function run_solver_diagnostics!(
    state::SolverDiagnosticsState,
    label::AbstractString,
    residual::Union{ACPowerFlowResidual, ACRectangularCIResidual, ACMixedCPBResidual},
    J::Union{ACPowerFlowJacobian, ACRectangularCIJacobian, ACMixedCPBJacobian},
    time_step::Int,
    cache::PFLinearSolverCache,
    monitor::Bool,
    bail::Bool,
    x::Union{AbstractVector{Float64}, Nothing} = nothing,
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

    # The bordered monitor is one back-solve, so it runs whenever diagnostics are on;
    # only `bail` decides whether a fold signature aborts the search.
    fold = state.fold
    g = fold.enabled ? _fold_monitor_value!(fold, cache) : NaN

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
            "sign(det J) = $(_fmt_det_sign(g))",
        ]
        if !isnan(state.prev_F) && state.prev_F > 0
            push!(parts, "contraction = $(_sf4(abs_max / state.prev_F))")
        end
        @info "$label: " * join(parts, ", ")
        state.prev_F = abs_max
    end

    # Bracketing re-evaluates `residual`/`J` at interpolated iterates, so it is only
    # offered when the caller vouches that both are at `x`, and it always restores
    # them (and the factorization) before returning.
    # `bracketed` keeps the restore off the hot path: bracketing only runs on an
    # ambiguous sign flip, so the common iteration pays nothing for this.
    bracketed = Ref(false)
    bracket = if (isnothing(x) || !fold.has_prev_x)
        nothing
    else
        function (θ::Float64)
            bracketed[] = true
            return _fold_value_at_interpolated_iterate!(
                fold, residual, J, time_step, cache, x, θ)
        end
    end
    abort = try
        _decide_det_sign_switch!(fold, label, g, bail, bracket)
    finally
        bracketed[] && _restore_iterate!(residual, J, time_step, cache, x)
    end

    if !isnothing(x)
        copyto!(fold.prev_x, x)
        fold.has_prev_x = true
    end
    return abort
end

"""Evaluate the bordered monitor `g` at `x_prev + θ·(x − x_prev)`. Leaves `residual`,
`J`, and `cache` at that interpolated iterate — [`_restore_iterate!`](@ref) puts them
back. Returns `NaN` when the interpolated `J` is outright singular, which
[`_bracket_fold_flip!`](@ref) reads as the zero it was hunting for."""
function _fold_value_at_interpolated_iterate!(
    mon::BorderedFoldMonitor,
    residual::Union{ACPowerFlowResidual, ACRectangularCIResidual, ACMixedCPBResidual},
    J::Union{ACPowerFlowJacobian, ACRectangularCIJacobian, ACMixedCPBJacobian},
    time_step::Int,
    cache::PFLinearSolverCache,
    x::AbstractVector{Float64},
    θ::Float64,
)
    @. mon.x_work = mon.prev_x + θ * (x - mon.prev_x)
    residual(mon.x_work, time_step)
    J(time_step)
    try
        numeric_refactor!(cache, J.Jv)
    catch e
        e isa LinearAlgebra.SingularException || rethrow()
        return NaN
    end
    return _fold_monitor_value!(mon, cache)
end

"""Put `residual`, `J`, and `cache`'s factorization back at the iterate `x` after
bracketing perturbed them. The solver loop reads `residual.Rv` for its convergence
test immediately after the diagnostics hook, so this is not optional."""
function _restore_iterate!(
    residual::Union{ACPowerFlowResidual, ACRectangularCIResidual, ACMixedCPBResidual},
    J::Union{ACPowerFlowJacobian, ACRectangularCIJacobian, ACMixedCPBJacobian},
    time_step::Int,
    cache::PFLinearSolverCache,
    x::AbstractVector{Float64},
)
    residual(x, time_step)
    J(time_step)
    try
        numeric_refactor!(cache, J.Jv)
    catch e
        # A singular J at the restored iterate is the caller's problem to report, not
        # something to raise from inside a diagnostic's cleanup.
        e isa LinearAlgebra.SingularException || rethrow()
    end
    return
end

"""`+`/`−` for the monitor's sign, or `n/a` when `g` is unavailable."""
_fmt_det_sign(g::Float64) = isnan(g) ? "n/a" : (g > 0 ? "+" : (g < 0 ? "−" : "0"))

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
