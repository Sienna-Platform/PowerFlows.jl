# Post-solve stagnation diagnostics. Self-contained: given a Jacobian, residual
# vector, and (for the Hessian terms) `data`/`time_step`, classify the cause of
# stagnation into an `ACPowerFlowSolveStatus` and produce a one-line log
# summary. Hessian-vector-vector products go through `acpf_hvvp`, which is
# polar-only today; non-polar formulations would need their own hvvp
# implementation to plug in here.

# Classification thresholds. Tuned against CATS multi-period traces; treat as
# illustrative starting points, not load-bearing.
const STAGNATION_STATIONARY_THRESHOLD = 1e-2   # ‖JᵀF‖/(‖J‖_F·‖F‖) below → non-root stationary
const STAGNATION_FOLD_THRESHOLD = 0.5          # ‖½H[Δx,Δx]‖_∞/‖F‖_∞ above → fold
const STAGNATION_SUBSPACE_THRESHOLD = 0.9      # subspace alignment above → trapped on near-singularity
# Default number of bottom singular vectors probed by the subspace alignment.
const STAGNATION_SUBSPACE_K = 3

# ---------------------------------------------------------------------------
# Formulation-aware residual-entry labeling for diagnostics.
#
# Decoding a global residual index `ix` into a readable "bus N (quantity)" /
# "LCC #k row" label must dispatch on the residual type, because the layout
# differs:
#   * polar (ACPowerFlowResidual): [P₁,Q₁,…,Pₙ,Qₙ | LCC tail], fixed 2/bus.
#   * rect CI (ACRectangularCIResidual): variable per-bus blocks (PQ/REF = 2 →
#     ΔI real/imag; PV = 3 → ΔI real/imag + |V|²−V_set²), indexed by
#     bus_state_offset.
#   * mixed (ACMixedCPBResidual): 2/bus, but the two rows depend on bus type.
# Both non-polar layouts share rect's `[bus state | 4·n_LCC tail]` split.
# ---------------------------------------------------------------------------

const _LCC_RESIDUAL_ROW_NAMES =
    ("P-setpoint", "DC-line balance", "rectifier α-limit", "inverter α-limit")

_diag_bus_number(data, bus_ix::Int) = axes(data.power_network_matrix, 1)[bus_ix]

function _describe_lcc_residual_entry(data, tail_ix::Int)
    i = div(tail_ix - 1, 4) + 1            # 1-based LCC index
    row = mod1(tail_ix, 4)
    (from_no, to_no) = data.lcc.arcs[i]
    return "LCC #$i ($from_no→$to_no) $(_LCC_RESIDUAL_ROW_NAMES[row])"
end

# Owning bus (matrix index) and 0-based row within its block, for the
# variable-block (rect/mixed) formulations. `bus_state_offset[b]` is the 1-based
# block start; `searchsortedlast` returns the block containing `ix`.
function _locate_variable_block(bus_state_offset::AbstractVector, ix::Int)
    b = searchsortedlast(bus_state_offset, ix)
    return b, ix - Int(bus_state_offset[b])
end

"""
    _describe_residual_entry(residual, data, time_step, ix) -> String

Formulation-aware label for global residual index `ix`: the owning bus number
(or LCC arc) and the physical quantity of that equation. Used by the
stagnation / limit-cycle / per-iteration diagnostics so they don't mislabel
non-polar residuals — whose entries are current/voltage mismatches on
variable-width blocks, not the polar fixed 2-per-bus P/Q."""
function _describe_residual_entry(::ACPowerFlowResidual, data, time_step::Int, ix::Int)
    n_bus_eqs = 2 * size(data.bus_type, 1)
    ix > n_bus_eqs && return _describe_lcc_residual_entry(data, ix - n_bus_eqs)
    bus_ix = div(ix + 1, 2)
    qty = isodd(ix) ? "P" : "Q"
    return "bus $(_diag_bus_number(data, bus_ix)) ($qty)"
end

function _describe_residual_entry(
    r::ACRectangularCIResidual, data, time_step::Int, ix::Int,
)
    ix > r.total_bus_state &&
        return _describe_lcc_residual_entry(data, ix - r.total_bus_state)
    bus_ix, row = _locate_variable_block(r.bus_state_offset, ix)
    # 2-block (PQ/REF): real/imag current mismatch. 3-block (PV) adds |V|²−V_set².
    qty = row == 0 ? "ΔI_re" : row == 1 ? "ΔI_im" : "|V|²−V_set²"
    return "bus $(_diag_bus_number(data, bus_ix)) ($qty)"
end

function _describe_residual_entry(r::ACMixedCPBResidual, data, time_step::Int, ix::Int)
    ix > r.total_bus_state &&
        return _describe_lcc_residual_entry(data, ix - r.total_bus_state)
    bus_ix, row = _locate_variable_block(r.bus_state_offset, ix)
    bt = data.bus_type[bus_ix, time_step]
    # MCPB layout: PQ = divided-current balance, imag-first; PV = power balance
    # then |V|²−V_set²; REF = rect's real/imag current rows.
    qty = if bt == PSY.ACBusTypes.PV
        row == 0 ? "ΔP" : "|V|²−V_set²"
    elseif bt == PSY.ACBusTypes.PQ
        row == 0 ? "ΔI_im" : "ΔI_re"     # imag-first
    else                                  # REF
        row == 0 ? "ΔI_re" : "ΔI_im"
    end
    return "bus $(_diag_bus_number(data, bus_ix)) ($qty)"
end

# Stability-gate parameters for `_check_stagnation!`. Two gates run in parallel:
# the **strict** gate (`ρ` and `κ̂` within ~10%) detects a true fixed point of
# the iteration map; the **loose** gate (~2×) detects period-≥2 limit cycles
# where ρ/κ̂ alternate between bounded clusters but never settle. `‖F‖_∞` uses
# the same 10% band for both — it's flat in either case.
const STAGNATION_WINDOW = 5
const STAGNATION_F_BAND = 1.1            # max/min ‖F‖_∞ in window
const STAGNATION_ρ_BAND_STRICT = 1.1     # ρ within 10% ⇒ fixed point
const STAGNATION_κ_BAND_STRICT = 1.1     # κ̂ within 10% ⇒ fixed point
const STAGNATION_ρ_BAND_LOOSE = 2.0      # ρ within 2× ⇒ limit cycle
const STAGNATION_κ_BAND_LOOSE = 2.0      # κ̂ within 2× ⇒ limit cycle

"""Update rolling windows of `‖F‖_∞`, `ρ`, and `κ̂(J)` and classify the recent
behavior of the iteration. Returns one of:

- `:fixed_point` — `‖F‖`, `ρ`, `κ̂` all within tight bands across the window.
  The iterate has converged (in the iteration-map sense) to a single point.
  Run [`_stagnation_diagnostic`](@ref) to determine the geometric character
  (local min / saddle / fold).
- `:limit_cycle` — `‖F‖` within tight band but `ρ` or `κ̂` only within the
  loose band. The iterate is in a period-≥2 cycle (alternates between bounded
  clusters but never settles). Typical near a fold/saddle-node bifurcation;
  point-wise classifiers like gradient/curvature tests don't apply.
- `:none` — not (yet) stagnated.

Pass `nothing` for `ρ`/`κ` when `compute_fixed_point_spectral_radius = false`;
the `‖F‖_∞` window is always required, and without `ρ`/`κ` data the helper
defaults to `:fixed_point` when `‖F‖` is flat (can't distinguish further).

Mutates the three windows in place."""
function _check_stagnation!(
    F_window::Vector{Float64},
    ρ_window::Vector{Float64},
    κ_window::Vector{Float64},
    F_inf::Float64,
    ρ::Union{Float64, Nothing},
    κ::Union{Float64, Nothing},
)
    _push_capped!(window, x) = begin
        push!(window, x)
        length(window) > STAGNATION_WINDOW && popfirst!(window)
    end
    _push_capped!(F_window, F_inf)
    if !isnothing(ρ) && isfinite(ρ)
        _push_capped!(ρ_window, ρ)
    end
    if !isnothing(κ) && isfinite(κ)
        _push_capped!(κ_window, κ)
    end
    _stable(window, band) =
        length(window) >= STAGNATION_WINDOW &&
        let (lo, hi) = extrema(window)
            hi / max(lo, eps()) < band
        end
    _stable(F_window, STAGNATION_F_BAND) || return :none
    # If we have no spectral data, we can only check F; report fixed_point.
    have_ρ = !isempty(ρ_window)
    have_κ = !isempty(κ_window)
    if !have_ρ && !have_κ
        return :fixed_point
    end
    # Loose gate first — anything failing loose isn't stagnated at all.
    have_ρ &&
        (_stable(ρ_window, STAGNATION_ρ_BAND_LOOSE) || return :none)
    have_κ &&
        (_stable(κ_window, STAGNATION_κ_BAND_LOOSE) || return :none)
    # Strict gate: all tracked spectral quantities within tight bands.
    strict_ρ = !have_ρ || _stable(ρ_window, STAGNATION_ρ_BAND_STRICT)
    strict_κ = !have_κ || _stable(κ_window, STAGNATION_κ_BAND_STRICT)
    return (strict_ρ && strict_κ) ? :fixed_point : :limit_cycle
end

"""Deflated inverse iteration on `JᵀJ` (via the KLU factor) to estimate the
bottom-`k` left-singular vectors of `J`. Returns the per-vector cosine
similarities `|⟨u_i, F⟩| / ‖F‖`, the subspace alignment
`√(Σ |⟨u_i, F⟩|²) / ‖F‖`, and the corresponding right singular vectors `V`
(needed for the directional-curvature test). A subspace alignment near 1 with
individual cosines all moderate is the canonical signature of a multi-
dimensional near-singular subspace.

`n_starts` runs the procedure with multiple random initial vectors and returns
the result with the largest subspace alignment. Mostly a stability check —
the bottom-`k` subspace is mathematically unique, so disagreement across
starts indicates inverse iteration hasn't fully converged."""
function _residual_subspace_alignment(
    Jv::SparseMatrixCSC{Float64, J_INDEX_TYPE},
    Rv::AbstractVector{Float64};
    k::Int = STAGNATION_SUBSPACE_K,
    n_iter::Int = 10,
    n_starts::Int = 3,
)
    nF = norm(Rv)
    nF > 0 || return Float64[], NaN, Vector{Vector{Float64}}()
    F = KLU.klu(Jv)
    n = size(Jv, 1)
    best_cosines = Float64[]
    best_subspace = -Inf
    best_V = Vector{Vector{Float64}}()
    for _ in 1:n_starts
        V = Vector{Vector{Float64}}()
        cosines = Float64[]
        for _ in 1:k
            v = randn(n)
            for vj in V
                v .-= dot(v, vj) .* vj
            end
            nv = norm(v)
            nv > 0 || break
            v ./= nv
            converged_inner = true
            for _ in 1:n_iter
                w = F' \ v
                v = F \ w
                for vj in V
                    v .-= dot(v, vj) .* vj
                end
                nv = norm(v)
                if nv == 0
                    converged_inner = false
                    break
                end
                v ./= nv
            end
            converged_inner || break
            push!(V, v)
            u = Jv * v
            nu = norm(u)
            nu > 0 || break
            u ./= nu
            push!(cosines, abs(dot(u, Rv)) / nF)
        end
        subspace = isempty(cosines) ? -Inf : sqrt(sum(c^2 for c in cosines))
        if subspace > best_subspace
            best_subspace = subspace
            best_cosines = cosines
            best_V = V
        end
    end
    best_subspace == -Inf && return Float64[], NaN, Vector{Vector{Float64}}()
    return best_cosines, best_subspace, best_V
end

"""Format a limit-cycle diagnostic line. Skips the point-wise classifier
(gradient / curvature / fold) since those tests are meaningless when the
iterate isn't actually at a single point. Reports the ρ and κ̂ ranges observed
across the window (which capture the cycle's amplitude) plus condest and the
top mismatches. Always returns `status = ACPowerFlowSolveStatus.LIMIT_CYCLE`."""
function _limit_cycle_diagnostic(
    residual,
    data::ACPowerFlowData,
    time_step::Int,
    ρ_window::AbstractVector{Float64},
    κ_window::AbstractVector{Float64},
)
    Rv = residual.Rv
    sf(x) = round(x; sigdigits = 4)
    ρ_part = if isempty(ρ_window)
        ""
    else
        ρ_lo, ρ_hi = extrema(ρ_window)
        ", ρ alternates in [$(sf(ρ_lo)), $(sf(ρ_hi))] (×$(sf(ρ_hi/max(ρ_lo,eps()))))"
    end
    κ_part = if isempty(κ_window)
        ""
    else
        κ_lo, κ_hi = extrema(κ_window)
        ", κ̂ alternates in [$(sf(κ_lo)), $(sf(κ_hi))] (×$(sf(κ_hi/max(κ_lo,eps()))))"
    end
    k_top = min(3, length(Rv))
    top_ix = partialsortperm(Rv, 1:k_top; by = abs, rev = true)
    parts = [
        "$(_describe_residual_entry(residual, data, time_step, ix)) = $(sf(Rv[ix]))"
        for ix in top_ix
    ]
    status = ACPowerFlowSolveStatus.LIMIT_CYCLE
    msg =
        ρ_part * κ_part *
        "; top $k_top mismatches: " * join(parts, ", ") *
        " [status: $status]"
    return msg, status
end

"""Classify why a Newton-type AC power flow solver got stuck, and produce a
log-friendly summary. Self-contained: takes only `(data, time_step, Rv, Jv)`
plus optional `condest`; no dependency on solver-internal structures.

Returns `(message::String, status::ACPowerFlowSolveStatus)`. The status is one
of `NON_ROOT_LOCAL_MIN`, `NON_ROOT_SADDLE`, `NON_ROOT_STATIONARY`,
`SINGULAR_SUBSPACE`, `FOLD`, or `STAGNATED_OTHER`, picked in priority order
(most actionable first):

1. **NON_ROOT_(LOCAL_MIN|SADDLE|STATIONARY)** — `‖JᵀF‖/(‖J‖_F·‖F‖) <
   $(STAGNATION_STATIONARY_THRESHOLD)`: Gauss-Newton stationary point. Sign of
   the directional curvature `v_minᵀ ∇²(½‖F‖²) v_min` further distinguishes:
   `+` → genuine local min (infeasibility), `−` → saddle (a different x0
   might find a root), undeterminable → fall back to `NON_ROOT_STATIONARY`.
2. **SINGULAR_SUBSPACE** — subspace alignment ≥
   $(STAGNATION_SUBSPACE_THRESHOLD): `F` lies in the bottom-`k` singular
   subspace of `J`; trapped at or past a fold of the state manifold.
3. **FOLD** — `‖½H[Δx,Δx]‖_∞/‖F‖_∞ ≥ $(STAGNATION_FOLD_THRESHOLD)`: a full
   Newton step would not reduce `‖F‖` because nonlinearity dominates.
4. **STAGNATED_OTHER** — none of the above fire; mechanism unclear.

The Hessian-vector-vector product is hardcoded to the polar formulation via
[`acpf_hvvp`](@ref), so for non-polar formulations (rectangular CI, mixed CPB)
the FOLD and directional-curvature checks are skipped entirely — they would
need a formulation-specific hvvp routine. The gradient/subspace tests, which
use only `J` and `F`, still run for every formulation."""
function _stagnation_diagnostic(
    residual,
    data::ACPowerFlowData,
    time_step::Int,
    Jv::SparseMatrixCSC{Float64, J_INDEX_TYPE};
    condest::Union{Float64, Nothing} = nothing,
    k::Int = STAGNATION_SUBSPACE_K,
)
    Rv = residual.Rv
    # acpf_hvvp is polar-only; running it on a non-polar Δx (different basis,
    # possibly different length) would either throw or — if the dimensions
    # happen to coincide — return a silently wrong number. Gate the FOLD and
    # directional-curvature checks on the polar formulation.
    is_polar = residual isa ACPowerFlowResidual
    sf(x) = round(x; sigdigits = 4)
    cond_part = isnothing(condest) || !isfinite(condest) ? "" :
                ", κ̂(J) = $(sf(condest))"
    cosines, subspace, V = _residual_subspace_alignment(Jv, Rv; k = k)
    align_part = if isempty(cosines)
        ""
    else
        cos_list = join((sf(c) for c in cosines), ", ")
        ", |⟨F, u_i⟩|/‖F‖ for i=1..$(length(cosines)) = ($cos_list), " *
        "subspace alignment = $(sf(subspace))"
    end
    # Stationary-point check for ½‖F‖²: gradient is JᵀF; at a Gauss-Newton local
    # minimum, JᵀF ≈ 0. Normalize by ‖J‖_F·‖F‖ (Cauchy-Schwarz upper bound), so
    # the result is in [0, 1] and ≪ 1 means we're at a non-root stationary point.
    JtF = Jv' * Rv
    nF = norm(Rv)
    nJ_F = norm(Jv.nzval)  # Frobenius norm of J
    grad_ratio = (nF > 0 && nJ_F > 0) ? norm(JtF) / (nJ_F * nF) : NaN
    grad_part = isfinite(grad_ratio) ?
                ", ‖JᵀF‖/(‖J‖_F·‖F‖) = $(sf(grad_ratio))" : ""
    # Fold check: predicted residual after a full Newton step is
    # F + J·Δx + ½ H[Δx,Δx,·] = ½ H[Δx,Δx,·]. Ratio ≈ 0 → Newton converges
    # cleanly; ≳ 1 → curvature kills the linear step.
    quad_ratio = NaN
    quad_part = if !is_polar
        ""
    else
        try
            F_klu = KLU.klu(Jv)
            Δx = -(F_klu \ Vector(Rv))
            h = acpf_hvvp(data, time_step, Δx, Δx)
            nh = norm(h, Inf)
            nFinf = norm(Rv, Inf)
            quad_ratio = (nFinf > 0) ? 0.5 * nh / nFinf : NaN
            isfinite(quad_ratio) ?
            ", ‖½H[Δx,Δx]‖_∞/‖F‖_∞ = $(sf(quad_ratio))" : ""
        catch _
            ""
        end
    end
    # Second-order test along v_min (right singular vector of smallest σ).
    # ∇²(½‖F‖²) = JᵀJ + Σ F_k Hᵏ; directional curvature at v_min is
    # ‖J·v_min‖² + Fᵀ·H[v_min, v_min, ·]. Near singularity the first term ≈
    # σ_min² is tiny, so sign is decided by the F·H part. + → robust local min,
    # − → saddle.
    curv_ratio = NaN
    curv_part = if !is_polar
        ""
    elseif !isempty(V)
        try
            v_min = V[1]
            Jv_min = Jv * v_min
            h_vv = acpf_hvvp(data, time_step, v_min, v_min)
            curvature = dot(Jv_min, Jv_min) + dot(Rv, h_vv)
            scale = max(dot(Jv_min, Jv_min), abs(dot(Rv, h_vv)), eps())
            curv_ratio = curvature / scale
            if isfinite(curv_ratio)
                ", v_minᵀ∇²(½‖F‖²)v_min (normalized) = $(sf(curv_ratio))"
            else
                ""
            end
        catch _
            ""
        end
    else
        ""
    end
    # Classify
    status = if isfinite(grad_ratio) && grad_ratio < STAGNATION_STATIONARY_THRESHOLD
        if isfinite(curv_ratio) && curv_ratio < 0
            ACPowerFlowSolveStatus.NON_ROOT_SADDLE
        elseif isfinite(curv_ratio) && curv_ratio > 0
            ACPowerFlowSolveStatus.NON_ROOT_LOCAL_MIN
        else
            ACPowerFlowSolveStatus.NON_ROOT_STATIONARY
        end
    elseif isfinite(subspace) && subspace >= STAGNATION_SUBSPACE_THRESHOLD
        ACPowerFlowSolveStatus.SINGULAR_SUBSPACE
    elseif isfinite(quad_ratio) && quad_ratio >= STAGNATION_FOLD_THRESHOLD
        ACPowerFlowSolveStatus.FOLD
    else
        ACPowerFlowSolveStatus.STAGNATED_OTHER
    end
    k_top = min(3, length(Rv))
    top_ix = partialsortperm(Rv, 1:k_top; by = abs, rev = true)
    parts = [
        "$(_describe_residual_entry(residual, data, time_step, ix)) = $(sf(Rv[ix]))"
        for ix in top_ix
    ]
    # Non-polar formulations skip the curvature/FOLD tests (acpf_hvvp is
    # polar-only), so make that explicit rather than letting their absence read
    # as "curvature was inconclusive".
    formulation_part =
        is_polar ? "" :
        ", curvature/FOLD checks skipped (non-polar formulation)"
    msg =
        cond_part * align_part * grad_part * quad_part * curv_part *
        formulation_part *
        "; top $k_top mismatches: " * join(parts, ", ") *
        " [status: $status]"
    return msg, status
end
