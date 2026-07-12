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

    n_state = size(J.Jv, 1)
    # Trailing block is the FULL non-bus tail (LCC + VSC + area interchange), not just
    # LCC: n_state on a VSC/area-interchange system is larger than 2*nbuses + 4*n_lcc.
    n_bus = n_state - state_tail_length(data, get_dc_network(data))
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

"""
    _report_area_interchange_failure(data, time_step)

Terminal-failure diagnostic (design spec §6 failure signature) for embedded area
net-interchange control. Called by the greedy relax loop
(`_ac_power_flow_with_area_relax!`, `solve_ac_power_flow.jl`) when it exhausts the enrolled
set without converging — the FINAL failed solve has zero WORKING controlled areas left, so
this reports against the PRISTINE tie set instead, over every area that was ORIGINALLY
enrolled, at the last attempted iterate's bus state (`data.bus_magnitude`/`bus_angles`, left
in place by the failed solve — see `_finalize_formulation!`). Reports the
largest-magnitude interchange-row residual and its area's name, so a genuine network
non-convergence (as opposed to a since-relaxed schedule) still names where the interchange
tail was furthest from satisfied at the point the search gave up. A no-op if area
interchange control was never enrolled on this `data`.
"""
function _report_area_interchange_failure(data::ACPowerFlowData, time_step::Int)
    aid = data.area_interchange
    isempty(aid.pristine_areas) && return
    gaps = [
        _area_net_interchange(aid.pristine_ties, area.tail_ix, data, time_step) - area.pdes
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
