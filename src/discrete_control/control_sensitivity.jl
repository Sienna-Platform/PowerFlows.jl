# Analytic sensitivity for the discrete-control continuation: dy/dp from the factored power
# flow Jacobian instead of a finite-difference probe.

function _sensitivity_residual_jacobian(::ACPolarPowerFlow, data, ts::Int)
    residual = ACPowerFlowResidual(data, ts)
    # Not `initialize_power_flow_variables`: `improve_x0` would move x off the converged base.
    x = _sensitivity_x0(residual, data, ts)
    residual(data, x, ts)
    J = ACPowerFlowJacobian(data, residual, ts)
    J(data, ts)
    return residual, J
end

# `pf` is always one of the three formulations below — every `AbstractACPowerFlow` subtype
# is one of them — so there is no broader fallback method here.
function _sensitivity_context(
    pf::Union{ACPolarPowerFlow, ACRectangularPowerFlow, ACMixedPowerFlow},
    data,
    ts::Int;
    kwargs...,
)
    backend = resolve_linear_solver_backend(get(kwargs, :linear_solver, nothing))
    residual, J = _sensitivity_residual_jacobian(pf, data, ts)
    lin_cache =
        _nr_linear_solver_cache!(data, J, backend, residual.bus_slack_participation_factors)
    try
        numeric_refactor!(lin_cache, J.Jv)
    catch e
        e isa LinearAlgebra.SingularException || rethrow()
        return FiniteDifferenceProbes()
    end
    _singular_base_solve(lin_cache, J) && return FiniteDifferenceProbes()
    # One registration per continuation — `_refresh_sensitivity_context!` reuses this
    # topology-invariant factorization on every batched pass without counting again.
    _count_symbolic_factor!(data)
    n = length(residual.Rv)
    return _SensitivityContext(
        lin_cache, residual, J, zeros(n), zeros(n), copy(view(data.bus_type, :, ts)))
end

# KLU throws on a genuinely singular matrix (caught above), but AppleAccelerate and
# MKLPardiso silently return finite garbage — the repo's backend-agnostic guard against
# that is the relative-residual check `_set_Δx_nr!`/`_do_refinement!` already apply to the
# main NR solve; reuse it here on a synthetic probe solve rather than trusting only
# `SingularException`.
function _singular_base_solve(lin_cache::PFLinearSolverCache, J)
    n = size(J.Jv, 1)
    probe = ones(n)
    sol = copy(probe)
    solve!(lin_cache, sol)
    sv = StateVectorCache(sol, probe)
    residual = _do_refinement!(
        sv, J.Jv, lin_cache, DEFAULT_REFINEMENT_THRESHOLD, DEFAULT_REFINEMENT_EPS)
    return !isfinite(residual) || residual > DEFAULT_REFINEMENT_THRESHOLD
end

# Gates `use_batched`: only a live context (every formulation has a `_refresh_sensitivity_context!`
# method) supports the batched per-pass refresh; the FD-probe fallback does not.
_supports_batched_refresh(::FiniteDifferenceProbes) = false
_supports_batched_refresh(::_SensitivityContext) = true

# `_update_residual_values!`'s PQ case telescopes `P_net` from the residual's LAST evaluation, so
# these must be rebuilt fresh from `data` before every evaluation or the correction drifts.
function _refresh_residual_inputs!(residual::ACPowerFlowResidual, data, ts::Int)::Bool
    _refresh_residual_setpoints!(residual, data, ts)
    return true
end

_sensitivity_x0(::ACPowerFlowResidual, data, ts::Int) = calculate_x0(data, ts)

# `ACPowerFlowJacobian`'s p-dependent fields are the SAME vectors as the residual's (passed by
# reference at construction), already current after `_refresh_residual_inputs!`: nothing to do.
_refresh_jacobian_yb_caches!(J, data, ::ACPowerFlowResidual, ts::Int) = return

function _refresh_sensitivity_context!(ctx::_SensitivityContext, data, ts::Int)::Bool
    view(data.bus_type, :, ts) == ctx.bus_type || return false
    residual = ctx.residual
    _refresh_residual_inputs!(residual, data, ts) || return false
    _refresh_jacobian_yb_caches!(ctx.J, data, residual, ts)
    x = _sensitivity_x0(residual, data, ts)
    ctx.residual(data, x, ts)
    ctx.J(data, ts)
    try
        numeric_refactor!(ctx.lin_cache, ctx.J.Jv)
    catch e
        e isa LinearAlgebra.SingularException || rethrow()
        return false
    end
    return !_singular_base_solve(ctx.lin_cache, ctx.J)
end

# `_refresh_sensitivity_context!` refuses in-place reuse across a PV/PQ Q-limit flip (its
# baked-in subnetwork/slack layout goes stale) or a numeric refactor that turns singular.
# Rebuild fresh (a new base-state factorization, counted like any other) instead of leaving
# batching disabled for the rest of the continuation, as a stale `ctx` never replaced would.
function _refresh_or_rebuild_context(
    ctx::_SensitivityContext, pf, data, ts::Int; kwargs...,
)
    _refresh_sensitivity_context!(ctx, data, ts) && return ctx
    return _sensitivity_context(pf, data, ts; kwargs...)
end

# ∂Y/∂p of the from-side terms (t_c = p·cis(α)); Y_tt = yt is p-independent (see `_branch_terms`).
function _tap_dY(d::ControlledTap)
    p, a = d.current, d.alpha
    dYff = -2.0 * d.yt / p^3
    dYft = d.yt * cis(a) / p^2
    dYtf = d.yt * cis(-a) / p^2
    return dYff, dYft, dYtf
end

# Row convention: F[2b−1] active, F[2b] reactive balance at bus b (see `_update_residual_values!`).
# Returns `false` when no analytic form exists here; caller then falls back to the FD probe.
function _dF_dp!(
    rhs::Vector{Float64},
    d::ControlledTap,
    ::ACPowerFlowResidual,
    data,
    ts::Int,
)
    fill!(rhs, 0.0)
    f, t = d.from_ix, d.to_ix
    Vf = data.bus_magnitude[f, ts] * cis(data.bus_angles[f, ts])
    Vt = data.bus_magnitude[t, ts] * cis(data.bus_angles[t, ts])
    dYff, dYft, dYtf = _tap_dY(d)
    # ∂S_i/∂p = V_i·conj(Σ_k ∂Y_ik/∂p·V_k); only Y_ff,Y_ft (row f) and Y_tf (row t) change.
    dSf = Vf * conj(dYff * Vf + dYft * Vt)
    dSt = Vt * conj(dYtf * Vf)
    @inbounds begin
        rhs[2 * f - 1] = real(dSf)
        rhs[2 * f] = imag(dSf)
        rhs[2 * t - 1] = real(dSt)
        rhs[2 * t] = imag(dSt)
    end
    return true
end

function _dF_dp!(
    rhs::Vector{Float64},
    d::Union{ControlledSwitchedShunt, ControlledFACTS},
    ::ACPowerFlowResidual,
    data,
    ts::Int,
)
    fill!(rhs, 0.0)
    b = d.bus_ix
    Vm = data.bus_magnitude[b, ts]
    # Constant-Z reactive withdrawal w enters Q_net as −w·Vm²; apply_parameter! sets ∂w/∂susc = −1,
    # so ∂Q_net/∂susc = +Vm² and ∂F[2b]/∂susc = −∂Q_net/∂susc = −Vm² (reactive row only).
    @inbounds rhs[2 * b] = -Vm^2
    return true
end

# Polar: x[2b−1] is Vm directly, so dVm/dp = −sol[2b−1] (dx/dp = −J⁻¹·∂F/∂p).
_dVm_from_sol(::ACPowerFlowResidual, sol::Vector{Float64}, cbus::Int, data, ts::Int) =
    -sol[2 * cbus - 1]

function _linear_plant_sign(d, data, ts::Int, ctx::_SensitivityContext)
    _dF_dp!(ctx.rhs, d, ctx.residual, data, ts) || return 0.0, false
    cbus = controlled_bus_ix(d)
    # Voltage is a free state only at PQ (PV/REF pin it), so dVm/dp = 0 there by construction —
    # matches the FD probe's behavior and freezes the device via the caller's gain floor.
    if data.bus_type[cbus, ts] != PSY.ACBusTypes.PQ
        return 0.0, true
    end
    copyto!(ctx.sol, ctx.rhs)
    solve!(ctx.lin_cache, ctx.sol)        # sol = J⁻¹·(∂F/∂p)
    return _dVm_from_sol(ctx.residual, ctx.sol, cbus, data, ts), true
end

# ── Rectangular CI and MCPB ───────────────────────────────────────────────────────────────
# Shared primitive: ΔI_i = ∂(Y_bus_eff·V)_i/∂p, from which each formulation's rows follow.
# Cross-check identity: ∂F_rect/∂p = −ΔI_i, ∂F_polar/∂p = V_i·conj(ΔI_i).

const _RectOrMixedResidual = Union{ACRectangularCIResidual, ACMixedCPBResidual}

# `Y_bus_eff` must be rebuilt fresh from `data` each refresh, not re-folded onto the old copy —
# tap/shunt moves edit the source Y-bus and withdrawals in place. The structure check catches a
# `fold_zip_constant_z!`-inserted diagonal a plain `nonzeros` copy would silently misalign; on
# mismatch, fall back to FD probes.
function _refresh_residual_inputs!(r::_RectOrMixedResidual, data, ts::Int)::Bool
    Y = data.power_network_matrix.data
    if SparseArrays.getcolptr(r.Y_bus_eff) != SparseArrays.getcolptr(Y) ||
       SparseArrays.rowvals(r.Y_bus_eff) != SparseArrays.rowvals(Y)
        return false
    end
    SparseArrays.nonzeros(r.Y_bus_eff) .= ComplexF64.(SparseArrays.nonzeros(Y))
    fold_zip_constant_z!(r.Y_bus_eff, data, ts)
    return true
end

# Unlike `ACPowerFlowJacobian`, these cache `Y_bus_eff`-derived values at construction, so a
# tap/shunt move leaves them stale — rerun the constructor's population steps against the refresh.
function _refresh_jacobian_yb_caches!(J, data, r::ACRectangularCIResidual, ts::Int)
    @inbounds for i in eachindex(J.Y_diag)
        J.Y_diag[i] = r.Y_bus_eff[i, i]
    end
    _populate_constant_yb_blocks!(
        J.Jv, r.Y_bus_eff, r.bus_state_offset, view(data.bus_type, :, ts))
    return
end
function _refresh_jacobian_yb_caches!(J, data, r::ACMixedCPBResidual, ts::Int)
    @inbounds for i in eachindex(J.Y_diag)
        J.Y_diag[i] = r.Y_bus_eff[i, i]
    end
    _populate_mixed_constant_yb_blocks!(
        J.Jv, r.Y_bus_eff, r.bus_state_offset, view(data.bus_type, :, ts))
    @inbounds for p in eachindex(J.offdiag_pv_y)
        J.offdiag_pv_y[p] = r.Y_bus_eff[J.offdiag_pv_i[p], J.offdiag_pv_k[p]]
    end
    return
end

function _sensitivity_x0(r::ACRectangularCIResidual, data, ts::Int)
    x = zeros(length(r.Rv))
    rect_initial_state!(x, data, r.bus_state_offset, r.bus_block_size, ts)
    return x
end
function _sensitivity_x0(r::ACMixedCPBResidual, data, ts::Int)
    x = zeros(length(r.Rv))
    mixed_initial_state!(x, data, r.bus_state_offset, r.bus_block_size, ts)
    return x
end

_state_offset(r::_RectOrMixedResidual, i::Int) = Int(r.bus_state_offset[i])

# NOT `data.bus_magnitude`/`bus_angles`: those hold V_set at PV buses, not |V_state|. Use
# `e_state`/`f_state` — the values the Jacobian was built from — so ∂F/∂p and J stay consistent.
_bus_voltage(r::_RectOrMixedResidual, i::Int) = complex(r.e_state[i], r.f_state[i])

# ── ∂I/∂p, per device family ──────────────────────────────────────────────────────────────
# The buses a parameter move touches, each with its `ΔI`. Formulation-free: this is the
# physics of the device, not of the residual. Returns a tuple so it stays stack-allocated.

# `apply_parameter!` writes `bus_reactive_..._withdrawals += current − b`, so ∂β_Q/∂p = −1, and
# `fold_zip_constant_z!` folds that as `Y_bb += complex(β_P, −β_Q)` ⇒ ∂Y_bb/∂p = +j ⇒ ΔI = j·V.
_dI_dp(d::Union{ControlledSwitchedShunt, ControlledFACTS}, r::_RectOrMixedResidual) =
    ((d.bus_ix, im * _bus_voltage(r, d.bus_ix)),)

function _dI_dp(d::ControlledTap, r::_RectOrMixedResidual)
    f, t = d.from_ix, d.to_ix
    Vf, Vt = _bus_voltage(r, f), _bus_voltage(r, t)
    dYff, dYft, dYtf = _tap_dY(d)
    return ((f, dYff * Vf + dYft * Vt), (t, dYtf * Vf))
end

# ── ΔI → residual rows, per formulation ───────────────────────────────────────────────────

# F = I_spec − Y·V: real in slot 0, imag in slot 1 at every bus type. A PV bus's third row
# is the `|V|²` constraint, which carries no p dependence.
function _stamp_dI!(
    rhs::Vector{Float64},
    r::ACRectangularCIResidual,
    i::Int,
    dI::ComplexF64,
    ::PSY.ACBusTypes.Value,
)
    off = _state_offset(r, i)
    @inbounds begin
        rhs[off] = -real(dI)
        rhs[off + 1] = -imag(dI)
    end
    return
end

# MCPB mixes three row conventions (`_update_mixed_cpb_residual_values!`):
#   PQ  — divided current balance, IMAG-first (rows swap; (e, f) columns do not).
#   PV  — real-power balance + `|V|²` row (p-independent; reactive injection cannot move either).
#   REF — rect-verbatim, real-first.
function _stamp_dI!(
    rhs::Vector{Float64},
    r::ACMixedCPBResidual,
    i::Int,
    dI::ComplexF64,
    bt::PSY.ACBusTypes.Value,
)
    off = _state_offset(r, i)
    @inbounds if bt == PSY.ACBusTypes.PV
        rhs[off] = real(_bus_voltage(r, i) * conj(dI))
        rhs[off + 1] = 0.0
    elseif bt == PSY.ACBusTypes.PQ
        rhs[off] = -imag(dI)
        rhs[off + 1] = -real(dI)
    else
        rhs[off] = -real(dI)
        rhs[off + 1] = -imag(dI)
    end
    return
end

# One body for both formulations: the device supplies ΔI, the formulation supplies the stamp.
function _dF_dp!(
    rhs::Vector{Float64},
    d,
    r::_RectOrMixedResidual,
    data,
    ts::Int,
)
    fill!(rhs, 0.0)
    bus_types = view(data.bus_type, :, ts)
    for (i, dI) in _dI_dp(d, r)
        _stamp_dI!(rhs, r, i, dI, bus_types[i])
    end
    return true
end

# dVm/dp = (e·de/dp + f·df/dp)/Vm from |V|² = e²+f², with dx/dp = −sol. `V_FLOOR2` matches the
# residuals' own guard — never binds at a converged base.
function _dVm_from_sol(
    r::_RectOrMixedResidual,
    sol::Vector{Float64},
    cbus::Int,
    data,
    ts::Int,
)
    off = _state_offset(r, cbus)
    e, f = r.e_state[cbus], r.f_state[cbus]
    Vm = sqrt(max(e^2 + f^2, V_FLOOR2))
    return -(e * sol[off] + f * sol[off + 1]) / Vm
end

function _sensitivity_residual_jacobian(::ACRectangularPowerFlow, data, ts::Int)
    residual = ACRectangularCIResidual(data, ts)
    x = _sensitivity_x0(residual, data, ts)
    residual(data, x, ts)
    J = ACRectangularCIJacobian(data, residual, ts)
    J(data, ts)
    return residual, J
end

function _sensitivity_residual_jacobian(::ACMixedPowerFlow, data, ts::Int)
    residual = ACMixedCPBResidual(data, ts)
    x = _sensitivity_x0(residual, data, ts)
    residual(data, x, ts)
    J = ACMixedCPBJacobian(data, residual, ts)
    J(data, ts)
    return residual, J
end
