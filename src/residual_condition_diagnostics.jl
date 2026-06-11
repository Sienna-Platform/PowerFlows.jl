#=
Per-iteration solver diagnostics, enabled by the `log_solver_diagnostics` flag.
Each line reports the residual infinity norm ‖F‖∞ (and the bus/equation where it
is attained), a condition estimate κ̂(J), the Jacobian eigenvalue closest to the
origin, and the observed residual contraction ratio ‖Fₖ‖/‖Fₖ₋₁‖.

The eigenvalue is computed on the bus-voltage Schur complement S = A − B·D⁻¹·C,
where the Jacobian is blocked as J = [A B; C D] with the LCC tail (4·n_lcc rows
and columns) in the trailing block. The (1,1) block of J⁻¹ is exactly S⁻¹, so the
matvec v ↦ (J⁻¹·[v; 0])[1:nb] applies S⁻¹ using the *existing* factorization of
the full J — no second matrix, no second factorization. When there are no LCCs,
S = J and the matvec is a plain J⁻¹ back-solve.
=#

"""Applies the bus-voltage Schur inverse `S⁻¹` to a vector by back-solving against
an existing factorization of the full Jacobian `J`: it pads the input with zeros
in the LCC-tail slots, applies `J⁻¹` in place via [`_apply_Jinv!`](@ref), and
returns the leading `n_bus` block."""
struct SchurInverseOperator{C}
    cache::C                  # factorization of the full J; see _apply_Jinv!
    n_bus::Int                # size of the leading bus-voltage block
    buffer::Vector{Float64}   # length = full state; reused as the padded RHS
end

"""Apply `J⁻¹` to `b` in place by back-solving against the cached factorization."""
_apply_Jinv!(cache::PFLinearSolverCache, b::Vector{Float64}) = solve!(cache, b)

# PERF: LAPACK's dlacn2 would be the drop-in replacement, but Julia doesn't expose it.
"""Condition estimate κ̂(J), or `NaN` when the backend doesn't expose one."""
_diag_condest(cache::PNM.KLULinSolveCache) = condest!(cache)
_diag_condest(::PFLinearSolverCache) = NaN

function (op::SchurInverseOperator)(v::AbstractVector{Float64})
    b = op.buffer
    @inbounds begin
        copyto!(view(b, 1:(op.n_bus)), v)
        fill!(view(b, (op.n_bus + 1):length(b)), 0.0)
    end
    _apply_Jinv!(op.cache, b)
    # KrylovKit stores each returned vector, so hand back a fresh copy of the
    # bus block rather than the reused buffer.
    return b[1:(op.n_bus)]
end

"""Smallest-magnitude eigenvalue of the bus-voltage Schur complement `S`, by
inverse iteration: KrylovKit finds the largest-magnitude eigenvalue `μ` of `S⁻¹`
(applied by `op`) and we return `1/μ`. `S` is non-symmetric, so the result may be
complex."""
function _schur_min_eigenvalue(
    op::SchurInverseOperator;
    tol::Float64 = 1e-6,
    maxiter::Int = 200,
    krylovdim::Int = 30,
)
    n = op.n_bus
    v0 = fill(1.0 / sqrt(n), n)   # deterministic init for reproducible logs
    vals, _, _ = KrylovKit.eigsolve(op, v0, 1, :LM; tol, maxiter, krylovdim)
    return isempty(vals) ? complex(NaN, NaN) : inv(vals[1])
end

"""Format a (possibly complex) eigenvalue to 4 significant figures as `a` or
`a ± b im`."""
function _fmt_eig(z::Number, sf)
    iz = imag(z)
    return if iz == 0
        "$(sf(real(z)))"
    else
        "$(sf(real(z))) $(iz < 0 ? "-" : "+") $(sf(abs(iz)))im"
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

"""Compute and `@info`-log one per-iteration diagnostic line. `cache` must hold
the current factorization of the Jacobian `J`. Reports ‖F‖∞ and where it is attained,
κ̂(J) [or n/a], λ_min of the bus-voltage Schur complement, and the contraction ratio
relative to `prev_F_inf`. Returns the current ‖F‖∞ so the caller can carry it forward.
All numerics are rounded to 4 significant figures."""
function _log_solver_diagnostics(
    label::AbstractString,
    residual,
    data::ACPowerFlowData,
    time_step::Int,
    cache::PFLinearSolverCache,
    n_state::Int,
    prev_F_inf::Float64,
)
    n_lcc = size(data.lcc.p_set, 1)
    n_bus = n_state - 4 * n_lcc
    abs_max, ix = findmax(abs, residual.Rv)

    κ = _diag_condest(cache)
    op = SchurInverseOperator(cache, n_bus, Vector{Float64}(undef, n_state))
    λ_min = _schur_min_eigenvalue(op)

    sf(x) = round(x; sigdigits = 4)
    parts = [
        "‖F‖_∞ = $(sf(abs_max)) at " *
        "$(_describe_residual_entry(residual, data, time_step, ix))",
        "κ̂(J) = $(isnan(κ) ? "n/a (KLU-only)" : string(sf(κ)))",
        "λ_min(S) = $(_fmt_eig(λ_min, sf)) (|λ_min| = $(sf(abs(λ_min))))",
    ]
    if !isnan(prev_F_inf) && prev_F_inf > 0
        push!(parts, "contraction = $(sf(abs_max / prev_F_inf))")
    end
    @info "$label: " * join(parts, ", ")
    return abs_max
end

# ---------------------------------------------------------------------------
# Fold / voltage-collapse bail-out: stop the search when the Jacobian's
# eigenvalue closest to the origin switches sign across an iteration.
# ---------------------------------------------------------------------------

# `UNSEEN` is the pre-first-observation state
# zero real part (essentially never, in floating point) just keeps the prior sign.
IS.@scoped_enum(EigvalSign, UNSEEN = 0, NEGATIVE = -1, POSITIVE = 1)

"""
    detect_eig_sign_switch(prev, label, data, time_step, cache, n_state)
        -> (switched::Bool, current::EigvalSign)

Given an up-to-date factorization `cache` of the Jacobian, compute `λ_min` of
the bus-voltage Schur complement and decide whether the sign of its real part has
flipped since the previous iteration's sign `prev`.

On a switch this logs a warning so the caller can abort the search. Returns
`(switched, current_sign))`. Assumes `cache` has been numerically factored already."""
function detect_eig_sign_switch(
    prev::EigvalSign,
    label::AbstractString,
    data::ACPowerFlowData,
    ::Int,
    cache::PFLinearSolverCache,
    n_state::Int,
)
    n_lcc = size(data.lcc.p_set, 1)
    n_bus = n_state - 4 * n_lcc
    op = SchurInverseOperator(cache, n_bus, Vector{Float64}(undef, n_state))
    λ_min = _schur_min_eigenvalue(op)
    s = real(λ_min)
    # exact 0: keep prior sign
    current = s > 0 ? EigvalSign.POSITIVE : (s < 0 ? EigvalSign.NEGATIVE : prev)

    # A switch needs a real, previously-seen sign that differs from the new one.
    switched = prev != EigvalSign.UNSEEN && current != prev
    switched || return false, current

    sf(x) = round(x; sigdigits = 4)
    @warn "$label: λ_min(S) real part switched sign " *
          "($(prev == EigvalSign.POSITIVE ? "+" : "−") → " *
          "$(current == EigvalSign.POSITIVE ? "+" : "−")), " *
          "λ_min = $(_fmt_eig(λ_min, sf)). This is a fold / voltage-collapse " *
          "signature; aborting the search."
    return true, current
end
