#=
Per-iteration solver diagnostics (`log_solver_diagnostics`) and a fold /
voltage-collapse bail-out (`stop_at_fold`).

λ_min is taken on the bus-voltage Schur complement S = A − B·D⁻¹·C of the blocked
Jacobian J = [A B; C D], whose LCC tail (4·n_lcc rows/cols) is the trailing block.
The (1,1) block of J⁻¹ is exactly S⁻¹, so v ↦ (J⁻¹·[v; 0])[1:nb] applies S⁻¹ from
the *existing* factorization of J — no second matrix or factorization. With no
LCCs, S = J. The monitor line and the bail-out share one refactor and one
eigensolve via `run_solver_diagnostics!`.
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

"""`(bus index, 1-based row within that bus's block)` for variable-block
formulations, from the `bus_state_offset` table."""
function _locate_variable_block(offsets::AbstractVector, ix::Int)
    b = searchsortedlast(offsets, ix)
    return b, ix - Int(offsets[b]) + 1
end

# Formulation-aware label for the entry where ‖F‖∞ is attained. The bus block is
# laid out first, the LCC tail last, in every formulation.
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
    return _describe_lcc_residual_entry(data, ix - n_bus_eqs)
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
    return _describe_lcc_residual_entry(data, ix - r.total_bus_state)
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
    return _describe_lcc_residual_entry(data, ix - r.total_bus_state)
end

# ---------------------------------------------------------------------------
# Fold / voltage-collapse bail-out state and the shared per-iteration hook.
# ---------------------------------------------------------------------------

# UNSEEN = pre-first-observation. A non-finite/non-converged λ_min is deliberately
# not a sign here; the bail decision treats it as a conservative abort.
IS.@scoped_enum(EigvalSign, UNSEEN = 0, NEGATIVE = -1, POSITIVE = 1)

"""Per-solve scratch for [`run_solver_diagnostics!`](@ref): previous ‖F‖∞ (`prev_F`),
last-seen sign of `real(λ_min)` (`eig_sign`), and a reusable padded RHS (`buffer`) so
the Schur operator allocates nothing per iteration."""
mutable struct SolverDiagnosticsState
    prev_F::Float64
    eig_sign::EigvalSign
    buffer::Vector{Float64}
end

SolverDiagnosticsState(n_state::Int) =
    SolverDiagnosticsState(NaN, EigvalSign.UNSEEN, Vector{Float64}(undef, n_state))

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

"""Update `state.eig_sign` from `λ_min` and decide the fold bail-out. A non-converged
or non-finite `real(λ_min)` is a conservative bail (warn + abort), never a silent
no-op. An exact-zero real part keeps the prior sign. Returns `true` to abort."""
function _decide_eig_sign_switch!(
    state::SolverDiagnosticsState,
    label::AbstractString,
    λ_min::ComplexF64,
    converged::Bool,
)::Bool
    s = real(λ_min)
    if !converged || !isfinite(s)
        @warn "$label: λ_min(S) is indeterminate " *
              "(converged = $converged, λ_min = $(_fmt_eig(λ_min))); treating it as " *
              "a fold / voltage-collapse signature and aborting the search."
        return true
    end
    prev = state.eig_sign
    current = s > 0 ? EigvalSign.POSITIVE : (s < 0 ? EigvalSign.NEGATIVE : prev)
    state.eig_sign = current

    # A switch needs a real, previously-seen sign that differs from the new one.
    switched = prev != EigvalSign.UNSEEN && current != prev
    switched || return false

    @warn "$label: λ_min(S) real part switched sign " *
          "($(prev == EigvalSign.POSITIVE ? "+" : "−") → " *
          "$(current == EigvalSign.POSITIVE ? "+" : "−")), " *
          "λ_min = $(_fmt_eig(λ_min)). This is a fold / voltage-collapse " *
          "signature; aborting the search."
    return true
end

"""Applies `(JᵀJ)⁻¹ = J⁻¹·J⁻ᵀ` via two back-solves against a KLU factorization of
`J` (`J` real ⇒ adjoint = transpose). Its dominant eigenvalue is `1/σ_min²`, so
Lanczos on it recovers `J`'s smallest singular value and right singular vector.
KLU-only by construction: it needs the transposed solve `tsolve!` from an existing
factorization of `J`, which AppleAccelerate doesn't expose (cf. the analogous
voltage-stability-factor solve in `ac_power_flow_jacobian.jl`)."""
struct GramInverseOperator{C}
    cache::C   # KLU factorization of J
    buffer::Vector{Float64}
end

function (op::GramInverseOperator)(v::AbstractVector{Float64})
    y = copyto!(op.buffer, v)
    tsolve!(op.cache, y)          # Jᵀ y = v  ⇒ y = J⁻ᵀ v
    solve!(op.cache, y)           # J z = y   ⇒ z = J⁻¹ y = (JᵀJ)⁻¹ v
    return copy(y)
end

"""Bus label for a *state*-vector index (mirrors the residual block layout, but
the per-row variable differs, so only the bus is reported)."""
# LCC *state* variables per converter (cf. state_indexing_helpers.jl), as opposed
# to the residual rows (`_LCC_RESIDUAL_ROW_NAMES`). i_dc is precomputed, not a state.
const _LCC_STATE_NAMES = ("tap_r", "tap_i", "α_r", "α_i")

function _describe_lcc_state_entry(data::ACPowerFlowData, tail_ix::Int)
    i = div(tail_ix - 1, 4) + 1
    from_no, to_no = data.lcc.arcs[i]
    return "LCC $(from_no)→$(to_no) ($(_LCC_STATE_NAMES[mod1(tail_ix, 4)]))"
end

function _state_bus_label(::ACPowerFlowResidual, data::ACPowerFlowData, ix::Int)
    n_bus_eqs = 2 * size(data.bus_type, 1)
    ix <= n_bus_eqs && return "bus $(_diag_bus_number(data, div(ix - 1, 2) + 1))"
    return _describe_lcc_state_entry(data, ix - n_bus_eqs)
end

function _state_bus_label(
    r::Union{ACRectangularCIResidual, ACMixedCPBResidual},
    data::ACPowerFlowData,
    ix::Int,
)
    if ix <= r.total_bus_state
        b, _ = _locate_variable_block(r.bus_state_offset, ix)
        return "bus $(_diag_bus_number(data, b))"
    end
    return _describe_lcc_state_entry(data, ix - r.total_bus_state)
end

"""Top `k` entries of `v` by magnitude, as `(label(ix), v[ix])` pairs."""
function _top_entries(v::AbstractVector{Float64}, k::Int, label)
    idx = partialsortperm(v, 1:min(k, length(v)); by = abs, rev = true)
    return [(label(i), round(v[i]; sigdigits = 4)) for i in idx]
end

"""Smallest singular triplet `(σ_min, v_min, u_min)` of the AC power flow Jacobian
`J` at `x0` (default: flat start), plus the `residual`/`jac` objects so callers can
reuse the labeling helpers. `v_min` is the right singular vector (state space), `u_min`
the left (residual space). Computed by Lanczos on `(JᵀJ)⁻¹` (dominant eigenvalue
`1/σ_min²`) against a *single* KLU factorization of `J`, applied for both the `J⁻¹`
and `J⁻ᵀ` solves — no second matrix, no second factorization."""
function _min_singular_triplet(
    data::ACPowerFlowData,
    time_step::Int;
    x0::Union{Vector{Float64}, Nothing} = nothing,
    tol::Float64 = 1e-8,
    maxiter::Int = 300,
    krylovdim::Int = 40,
)
    residual = ACPowerFlowResidual(data, time_step)
    jac = ACPowerFlowJacobian(residual, time_step)
    x = isnothing(x0) ? calculate_x0(data, time_step) : copy(x0)
    residual(x, time_step)
    jac(time_step)

    cache = make_linear_solver_cache(PNM.KLUSolver(), jac.Jv)
    full_factor!(cache, jac.Jv)
    n = size(jac.Jv, 1)
    op = GramInverseOperator(cache, Vector{Float64}(undef, n))
    v0 = fill(1.0 / sqrt(n), n)
    vals, vecs, _ = KrylovKit.eigsolve(
        op, v0, 1, :LM; tol = tol, maxiter = maxiter, krylovdim = krylovdim,
        issymmetric = true,
    )
    σ_min = 1.0 / sqrt(abs(vals[1]))          # 1/σ_min² is the dominant eigenvalue
    v_min = vecs[1] ./ norm(vecs[1])          # right singular vector (state space)
    u = jac.Jv * v_min
    u_min = u ./ norm(u)                      # left singular vector (residual space)
    return (; σ_min, v_min, u_min, residual, jac)
end

"""
    find_jacobian_null_space(data, time_step; x0, k, tol, maxiter, krylovdim)
        -> (σ_min, state_entries, residual_entries)

Locate the near-null space of the AC power flow Jacobian `J` at `x0` (default:
the flat start). Computes the smallest singular value `σ_min` and its singular
vectors via Lanczos on `(JᵀJ)⁻¹` (dominant eigenvalue `1/σ_min²`), reusing a KLU
factorization of `J`. A tiny `σ_min` is the true rank-deficiency behind a huge
condition estimate `κ̂ ≈ σ_max/σ_min`; the returned vectors say *where* it lives.

Returns:
- `σ_min`: smallest singular value of `J`.
- `state_entries`: the `k` largest-magnitude entries of the right singular vector
  `v_min` (state directions `J` barely moves) as `(bus label, weight)` pairs.
- `residual_entries`: the `k` largest of the left singular vector
  `u_min = J·v_min / σ_min` (residual equations that are nearly unreachable), as
  `(formulation-aware label, weight)` pairs.

For a localized singularity (e.g. a near-zero-impedance branch) both lists
concentrate on the few offending buses. The diagnostic is state-independent for a
*structural* singularity, so the default flat start localizes it fine.
"""
function find_jacobian_null_space(
    data::ACPowerFlowData,
    time_step::Int;
    x0::Union{Vector{Float64}, Nothing} = nothing,
    k::Int = 10,
    tol::Float64 = 1e-8,
    maxiter::Int = 300,
    krylovdim::Int = 40,
)
    (; σ_min, v_min, u_min, residual) =
        _min_singular_triplet(data, time_step; x0, tol, maxiter, krylovdim)
    state_entries = _top_entries(v_min, k, ix -> _state_bus_label(residual, data, ix))
    residual_entries =
        _top_entries(
            u_min,
            k,
            ix -> _describe_residual_entry(residual, data, time_step, ix),
        )
    return σ_min, state_entries, residual_entries
end

# ---------------------------------------------------------------------------
# Bottleneck localizer: turn the critical mode into a cutset + reactive-reserve
# statement. Baseline-free — the weak boundary is a structural property of the
# operating region, not of whatever perturbation exposed it.
# ---------------------------------------------------------------------------

# original-bus-number → surviving (reduced) bus number, from the forward
# `bus_reduction_map: survivor → Set{absorbed}`. Buses that survive map to
# themselves. Trivial (no reduction) ⇒ empty dict; callers fall back to identity.
function _orig_to_survivor(data::ACPowerFlowData)
    reduction = get_network_reduction_data(data).bus_reduction_map
    o2s = Dict{Int, Int}()
    for (survivor, absorbed) in reduction
        o2s[survivor] = survivor
        for a in absorbed
            o2s[a] = survivor
        end
    end
    return o2s
end

# Reactive-power limits as `(min, max)`, or `nothing` if the device has none
# (e.g. a renewable/source without Q limits). Tolerant across device types.
function _gen_q_limits(g)
    try
        lims = PSY.get_reactive_power_limits(g)
        return lims === nothing ? nothing : lims
    catch
        return nothing
    end
end

const _LCC_ROW_SYMBOLS = (:P_setpoint, :dc_balance, :rectifier_alpha, :inverter_alpha)

"""Structured (machine-parseable) classification of a polar residual entry as
`(kind, id, quantity)`: `(:bus, bus_number, :P|:Q)` for the bus block, or
`(:lcc, (from, to), row_symbol)` for the LCC tail. The companion to
[`_describe_residual_entry`](@ref), which returns the human-readable string."""
function _classify_residual_entry(
    ::ACPowerFlowResidual,
    data::ACPowerFlowData,
    ::Int,
    ix::Int,
)
    n_bus_eqs = 2 * size(data.bus_type, 1)
    if ix <= n_bus_eqs
        bus_ix = div(ix - 1, 2) + 1
        return (:bus, _diag_bus_number(data, bus_ix), isodd(ix) ? :P : :Q)
    end
    tail = ix - n_bus_eqs
    i = div(tail - 1, 4) + 1
    return (:lcc, data.lcc.arcs[i], _LCC_ROW_SYMBOLS[mod1(tail, 4)])
end

"""How many leading entries of a descending-sorted magnitude vector are
"significant" — i.e. within a factor `significance` of the peak. This replaces a
hard top-`k` with a data-driven cut: a lone dominant entry (the next is `1/200`
the peak) yields 1, a flat profile (many comparable entries) yields many. Always
returns at least 1 and never more than `k_max`."""
function _significant_count(sorted_desc::AbstractVector{<:Real};
    significance::Float64, k_max::Int)
    isempty(sorted_desc) && return 0
    peak = float(sorted_desc[1])
    peak <= 0 && return 1
    n = 0
    for v in sorted_desc
        v >= significance * peak || break
        n += 1
        n >= k_max && break
    end
    return max(n, 1)
end

"""
    localize_bottleneck(data, sys, time_step; x0, k, q_margin_frac, verbose)
        -> (; σ_min, pocket, cutset, binding, exhausted_q)

Turn the near-singular critical mode of the AC power flow Jacobian into a
*systemic bottleneck* statement — baseline-free, since the weak boundary is a
structural property of the operating region, not of the perturbation that exposed
it. Reuses a single KLU factorization via [`_min_singular_triplet`](@ref); the only
extra cost is an `O(branches)` walk to project the mode onto the network.

How many entries each list holds is *data-driven*, not a fixed count: an entry is
kept while it stays within a factor `significance` (default 0.1, i.e. one order of
magnitude) of the list's peak, capped at `k`. A lone dominant bus/branch yields a
one-element list; a flat profile (a genuine multi-branch corridor, or a whole area
equally infeasible) yields several. See [`_significant_count`](@ref).

Polar formulation only (interleaved `[Vm, θ]` per-bus state). Returns:
- `pocket`: `(bus_number, participation)`. Participation is `‖[ΔVm, Δθ]‖` of that
  bus in the right singular vector `v_min` — the buses that collapse together.
- `cutset`: `(branch_name, from, to, Δθ_mode, x)`. Branches across which the mode
  *tears* — i.e. that separate the collapsing group from the rest of the network.
  `x` is the branch series reactance (context only; a high-`x` entry is a weak
  corridor, a low-`x` one means a near-radial pocket hanging off a stiff bus).
- `binding`: the left singular vector `u_min` as machine-parseable tuples
  `(kind, id, quantity, weight)` — `(:bus, bus_number, :P|:Q, w)` or
  `(:lcc, (from, to), row_symbol, w)` — the power-balance equations nearly
  infeasible (the mismatch that can't close). The size is data-driven: it runs to
  wherever the weights drop off, with `binding_k` only a runaway *backstop*
  (default 100, not a target). Unlike the pocket — a sharp right-vector peak — the
  binding is often a *plateau*: a coherent extended region (a whole sub-network at
  its active-power limit) whose `u_min` weights are all comparable, so the natural
  cut is the plateau's edge, not a small top-N. Setting `binding_k` too low slices
  through such a plateau and undercounts the region.
- `exhausted_q`: available generators at pocket buses whose reactive output sits
  within `q_margin_frac` of a Q limit — why the pocket can't hold voltage.

`verbose` (default `true`) also `@info`-logs a readable summary.
"""
function localize_bottleneck(
    data::ACPowerFlowData,
    sys::PSY.System,
    time_step::Int = 1;
    x0::Union{Vector{Float64}, Nothing} = nothing,
    significance::Float64 = 0.1,
    k::Int = 15,
    binding_k::Int = 100,
    q_margin_frac::Float64 = 0.02,
    verbose::Bool = true,
)
    (; σ_min, v_min, u_min, residual) = _min_singular_triplet(data, time_step; x0)
    residual isa ACPowerFlowResidual || error(
        "localize_bottleneck supports the polar formulation only; got $(typeof(residual)).",
    )

    n_bus = size(data.bus_type, 1)
    # Per-bus participation (both state slots) and signed angle coordinate. Polar
    # layout is interleaved: state[2i-1] = Vm, state[2i] = θ for bus i.
    part = [hypot(v_min[2i - 1], v_min[2i]) for i in 1:n_bus]
    θ_mode = [v_min[2i] for i in 1:n_bus]   # eigenvector sign is arbitrary; we use diffs

    bus_numbers = collect(axes(data.power_network_matrix, 1))   # reduced ix → number
    num_to_ix = Dict(no => i for (i, no) in enumerate(bus_numbers))

    # Pocket: the significantly-participating buses (cut where participation falls
    # an order of magnitude below the peak, rather than a fixed count).
    order = sortperm(part; rev = true)
    np = _significant_count(part[order]; significance, k_max = min(k, n_bus))
    pocket = [(bus_numbers[i], round(part[i]; sigdigits = 4)) for i in order[1:np]]

    # Cutset: branches ranked by how hard the mode tears across them. A large
    # |Δθ_mode| means the two coherent groups pull apart there — it locates the
    # boundary of the collapse pocket without forming Ybus. (Not a high-reactance
    # filter: a radial pocket tears across its one low-x tie just as sharply.)
    o2s = _orig_to_survivor(data)
    surv_ix(no) = get(num_to_ix, get(o2s, no, no), 0)
    cut = NamedTuple{
        (:name, :from, :to, :dθ, :x),
        Tuple{String, Int, Int, Float64, Float64},
    }[]
    for b in PSY.get_components(PSY.ACBranch, sys)
        occursin("ThreeWinding", String(nameof(typeof(b)))) && continue   # no single arc
        applicable(PSY.get_arc, b) || continue
        arc = PSY.get_arc(b)
        f0 = PSY.get_number(PSY.get_from(arc))
        t0 = PSY.get_number(PSY.get_to(arc))
        fi, ti = surv_ix(f0), surv_ix(t0)
        (fi == 0 || ti == 0 || fi == ti) && continue   # off-network or intra-cluster
        dθ = abs(θ_mode[fi] - θ_mode[ti])
        xval = applicable(PSY.get_x, b) ? PSY.get_x(b) : NaN
        push!(cut, (name = PSY.get_name(b), from = f0, to = t0, dθ = dθ, x = xval))
    end
    sort!(cut; by = c -> c.dθ, rev = true)
    nc = _significant_count([c.dθ for c in cut]; significance, k_max = k)
    cutset = [
        (c.name, c.from, c.to, round(c.dθ; sigdigits = 4), round(c.x; sigdigits = 4))
        for c in cut[1:min(nc, length(cut))]
    ]

    # Binding mismatch: which power-balance equations are nearly unreachable, as
    # machine-parseable (kind, id, quantity, weight) tuples. Size is data-driven —
    # it runs to where the weights drop off; `binding_k` is only a runaway backstop.
    # The binding is typically a *plateau* (a coherent extended region all at its
    # P-limit, weights ~equal) rather than a sharp peak, so `significance` can't trim
    # it and the natural cut is the plateau's edge. A too-small backstop slices the
    # plateau and undercounts the region.
    nb = _significant_count(sort(abs.(u_min); rev = true); significance, k_max = binding_k)
    binding = map(partialsortperm(u_min, 1:nb; by = abs, rev = true)) do ix
        kind, id, quantity = _classify_residual_entry(residual, data, time_step, ix)
        (kind, id, quantity, round(u_min[ix]; sigdigits = 4))
    end

    # Reactive reserve in the pocket. Expand each surviving pocket bus to the
    # originals reduced into it, so generators on absorbed buses are still caught.
    reduction = get_network_reduction_data(data).bus_reduction_map
    pocket_numbers = Set(bus_numbers[i] for i in order[1:np])
    for s in collect(pocket_numbers)
        haskey(reduction, s) && union!(pocket_numbers, reduction[s])
    end
    exhausted_q = Tuple{String, Int, Float64, Float64, Float64}[]
    for g in PSY.get_components(PSY.Generator, sys)
        PSY.get_available(g) || continue
        PSY.get_number(PSY.get_bus(g)) in pocket_numbers || continue
        lims = _gen_q_limits(g)
        lims === nothing && continue
        q = PSY.get_reactive_power(g)
        hi = lims.max - q
        lo = q - lims.min
        margin = max(abs(lims.max), abs(lims.min), 1.0) * q_margin_frac
        (hi <= margin || lo <= margin) || continue
        push!(
            exhausted_q,
            (PSY.get_name(g), PSY.get_number(PSY.get_bus(g)), q, lims.min, lims.max),
        )
    end

    result = (; σ_min, pocket, cutset, binding, exhausted_q)
    verbose && _report_bottleneck(result)
    return result
end

"""`@info`-log a readable summary of a [`localize_bottleneck`](@ref) result."""
function _report_bottleneck(b)
    @info "Bottleneck localization: σ_min(J) = $(round(b.σ_min; sigdigits = 4))"
    @info "  Collapse pocket (bus ⇒ participation): $(b.pocket)"
    @info "  Weak cutset into the pocket (branch: from→to, Δθ_mode, x):"
    for (name, f, t, dθ, x) in b.cutset
        @info "    $name: $f→$t   Δθ_mode = $dθ   x = $x"
    end
    @info "  Binding mismatch (nearly-infeasible equations): $(b.binding)"
    if isempty(b.exhausted_q)
        @info "  No pocket generators near a Q-limit (bottleneck is transfer/structural, not local Q)."
    else
        @info "  Pocket generators out of reactive reserve (name, bus, Q, Qmin, Qmax):"
        for e in b.exhausted_q
            @info "    $e"
        end
    end
    return
end

"""Run one iteration's diagnostics against the current `J`/residual. Does the
*single* per-iteration refactor of `cache` on `J.Jv` (NR/TR pass `linSolveCache`, LM
its own KLU `diag_cache`) and, on success, the *single* eigensolve shared by the
monitor line (`monitor`) and the fold bail-out (`bail`). Returns `true` iff the
caller should abort. A `SingularException` is itself a fold signature: under `bail`
it aborts, under monitor-only it reports `singular` and continues; any other
exception is rethrown."""
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
              "κ̂(J) = singular, λ_min(S) = singular"
        return false
    end

    n_lcc = size(data.lcc.p_set, 1)
    n_state = size(J.Jv, 1)
    n_bus = n_state - 4 * n_lcc
    op = SchurInverseOperator(cache, n_bus, state.buffer)
    λ_min, converged = _schur_min_eigenvalue(op)

    if monitor
        abs_max, ix = findmax(abs, residual.Rv)
        κ = _diag_condest(cache)
        λ_str = if converged
            "$(_fmt_eig(λ_min)) (|λ_min| = $(_sf4(abs(λ_min))))"
        else
            "not-converged"
        end
        parts = [
            "‖F‖_∞ = $(_sf4(abs_max)) at " *
            "$(_describe_residual_entry(residual, data, time_step, ix))",
            "κ̂(J) = $(isnan(κ) ? "n/a (KLU-only)" : string(_sf4(κ)))",
            "λ_min(S) = $λ_str",
        ]
        if !isnan(state.prev_F) && state.prev_F > 0
            push!(parts, "contraction = $(_sf4(abs_max / state.prev_F))")
        end
        @info "$label: " * join(parts, ", ")
        state.prev_F = abs_max
    end

    if bail
        return _decide_eig_sign_switch!(state, label, λ_min, converged)
    end
    return false
end
