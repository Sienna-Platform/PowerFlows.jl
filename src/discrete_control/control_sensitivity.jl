# Analytic sensitivity for the discrete-control continuation: dy/dp from the factored power
# flow Jacobian instead of a finite-difference probe.
#
# This lives in its own file, included AFTER every formulation's residual and Jacobian, because
# the methods here dispatch on those concrete types in their SIGNATURES (which resolve at
# definition time). `control_discrete_devices/control_continuation.jl` is included before them,
# so the dispatch cannot live there.
#
# Adding a formulation means adding four methods here — `_sensitivity_residual_jacobian`,
# `_dF_dp!`, `_dVm_from_sol`, and (to earn batched passes) `_refresh_sensitivity_context!` plus
# `_refreshable` — and touching no device code and no continuation logic.

# The residual/Jacobian pair for a formulation, built and evaluated at the state currently in
# `data`. One method per formulation; `_sensitivity_context` below is generic over them.
#
# Deliberately NOT `initialize_power_flow_variables`: all of its methods route through
# `improve_x0` (previous-time-step warm-start comparison, enhanced flat start, DC fallback), so
# it would build the context at a warm-start CANDIDATE rather than at the converged base — and
# every gain read off it would be wrong by an unbounded amount.
function _sensitivity_residual_jacobian(::ACPolarPowerFlow, data, ts::Int)
    residual = ACPowerFlowResidual(data, ts)
    x = calculate_x0(data, ts)
    residual(x, ts)                       # evaluate at current state; fills P_net/Q_net
    J = ACPowerFlowJacobian(residual, ts)
    J(ts)                                 # Jacobian values at current state
    return residual, J
end

# No analytic form for this formulation ⇒ the caller uses FD probes. Kept as the escape for any
# formulation added later without a `_sensitivity_residual_jacobian` method.
_sensitivity_context(::AbstractACPowerFlow, data, ts::Int; kwargs...) = nothing
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
        return                    # singular base Jacobian ⇒ fall back to FD probes
    end
    # One full sensitivity-context build (fresh residual + Jacobian structure + refactor) per
    # continuation. `_refresh_sensitivity_context!` re-evaluates VALUES into these same objects
    # on every subsequent batched pass and does NOT count here — the Jacobian structure (and thus
    # the persisted symbolic factorization this counter tracks) is topology-invariant across the
    # continuation, so only this one build should ever register.
    _count_symbolic_factor!(data)
    n = length(residual.Rv)
    return _SensitivityContext(
        lin_cache, residual, J, zeros(n), zeros(n), copy(view(data.bus_type, :, ts)))
end

# Whether a live context can be kept current across batched passes, i.e. whether its
# formulation has a `_refresh_sensitivity_context!` that re-syncs every p-dependent cache.
# Gates `use_batched`. A formulation that supplies analytic probe sensitivities but no refresh
# stays on the sequential path: batching it would read a Jacobian that went stale the moment a
# device moved, and be wrong on every pass after the first.
_supports_batched_refresh(::Nothing) = false
_supports_batched_refresh(ctx::_SensitivityContext) = _refreshable(ctx.residual)
# Default false, so a formulation that gains analytic probe sensitivities does NOT silently
# gain batched passes as well: opting in requires adding a `_refresh_sensitivity_context!`
# method AND flipping this. Getting the default backwards would batch a formulation whose
# context goes stale on the first device move.
_refreshable(::Any) = false
_refreshable(::ACPowerFlowResidual) = true

function _refresh_sensitivity_context!(ctx::_SensitivityContext, data, ts::Int)::Bool
    view(data.bus_type, :, ts) == ctx.bus_type || return false
    residual = ctx.residual
    copyto!(
        residual.bus_active_constant_I,
        view(data.bus_active_power_constant_current_withdrawals, :, ts),
    )
    copyto!(
        residual.bus_reactive_constant_I,
        view(data.bus_reactive_power_constant_current_withdrawals, :, ts),
    )
    copyto!(
        residual.bus_active_constant_Z,
        view(data.bus_active_power_constant_impedance_withdrawals, :, ts),
    )
    copyto!(
        residual.bus_reactive_constant_Z,
        view(data.bus_reactive_power_constant_impedance_withdrawals, :, ts),
    )
    @inbounds for ix in eachindex(residual.P_net)
        residual.P_net[ix] =
            data.bus_active_power_injections[ix, ts] -
            get_bus_active_power_total_withdrawals(data, ix, ts) +
            data.bus_hvdc_net_power[ix, ts]
        residual.Q_net[ix] =
            data.bus_reactive_power_injections[ix, ts] -
            get_bus_reactive_power_total_withdrawals(data, ix, ts)
        residual.P_net_set[ix] = residual.P_net[ix]
    end
    x = calculate_x0(data, ts)
    ctx.residual(x, ts)
    ctx.J(ts)
    try
        numeric_refactor!(ctx.lin_cache, ctx.J.Jv)
    catch e
        e isa LinearAlgebra.SingularException || rethrow()
        return false
    end
    return true
end

# ∂F/∂p into `rhs` (zeroed first). Returns `true` iff the family has an analytic polar form here;
# the `false` path is reserved for a future family without one — the caller then uses the FD probe
# (see `_linear_plant_sign`). Row convention: F[2b−1] active, F[2b] reactive balance at bus b
# (see `_update_residual_values!`).
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
    p, a = d.current, d.alpha
    # ∂Y/∂p of the from-side terms (t_c = p·cis(a)); Y_tt = yt is p-independent (see `_branch_terms`).
    dYff = -2.0 * d.yt / p^3
    dYft = d.yt * cis(a) / p^2
    dYtf = d.yt * cis(-a) / p^2
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

# Linearized dy/dp for a voltage device via the factored Jacobian. `y = Vm(controlled_bus)`
# = x[2·cbus−1], and dx/dp = −J⁻¹·(∂F/∂p). Returns `(dy/dp, true)`, or `(0.0, false)` when the
# family has no analytic form (caller then uses the FD probe).
# dVm(cbus)/dp from `sol = J⁻¹·(∂F/∂p)`, given `dx/dp = −sol`. One method per formulation:
# each knows where its own state keeps the controlled bus's voltage.
# Polar: x[2b−1] is Vm directly.
_dVm_from_sol(::ACPowerFlowResidual, sol::Vector{Float64}, cbus::Int, data, ts::Int) =
    -sol[2 * cbus - 1]

function _linear_plant_sign(d, data, ts::Int, ctx::_SensitivityContext)
    _dF_dp!(ctx.rhs, d, ctx.residual, data, ts) || return 0.0, false
    cbus = controlled_bus_ix(d)
    # The controlled bus's voltage is a free state variable ONLY at a PQ bus (polar keeps Q_gen
    # there at PV and P at REF; the rectangular/mixed formulations pin |V|² with their own
    # constraint row). At a PV/REF controlled bus the voltage is pinned by the bus model, so
    # dVm/dp = 0 exactly: return a reliable zero so the caller's gain floor freezes the device,
    # matching the FD probe's behavior.
    if data.bus_type[cbus, ts] != PSY.ACBusTypes.PQ
        return 0.0, true
    end
    copyto!(ctx.sol, ctx.rhs)
    solve!(ctx.lin_cache, ctx.sol)        # sol = J⁻¹·(∂F/∂p)
    return _dVm_from_sol(ctx.residual, ctx.sol, cbus, data, ts), true
end

# ── Rectangular CI and MCPB ───────────────────────────────────────────────────────────────
#
# Both formulations carry a complex state `V = e + jf` and a residual built on a CURRENT
# balance, where polar's is a POWER balance. One primitive spans all three: the complex
# derivative of the network current injected at a bus,
#
#     ΔI_i = ∂(Y_bus_eff·V)_i / ∂p
#
# from which each formulation's rows follow by its own sign and ordering convention. The
# identity that ties the two families together, and the cross-check for any new stamp, is
#
#     ∂F_rect/∂p = −ΔI_i          ∂F_polar/∂p = V_i · conj(ΔI_i)
#
# Applied to polar, the right-hand form reproduces the two hand-written methods above exactly:
# a shunt gives `V·conj(jV) = −j|V|²` ⇒ reactive row `−Vm²`, and a tap gives `V_f·conj(ΔI_f)`
# ⇒ the `dSf` split. That is why those bodies are trusted and these are derived from them.

const _RectOrMixedResidual = Union{ACRectangularCIResidual, ACMixedCPBResidual}

_state_offset(r::_RectOrMixedResidual, i::Int) = Int(r.bus_state_offset[i])

# The bus voltage as the residual's own state sees it. NOT `data.bus_magnitude`/`bus_angles`:
# those hold V_set at PV buses rather than |V_state|, while `e_state`/`f_state` are the values
# the Jacobian was built from — so using them keeps ∂F/∂p and J consistent by construction.
_bus_voltage(r::_RectOrMixedResidual, i::Int) = complex(r.e_state[i], r.f_state[i])

# ── ∂I/∂p, per device family ──────────────────────────────────────────────────────────────
# The buses a parameter move touches, each with its `ΔI`. Formulation-free: this is the
# physics of the device, not of the residual. Returns a tuple so it stays stack-allocated.

# A shunt/FACTS parameter is a susceptance at one bus. `apply_parameter!` writes
# `bus_reactive_power_constant_impedance_withdrawals += current − b`, so ∂β_Q/∂p = −1, and
# `fold_zip_constant_z!` folds that into the Y-bus as `Y_bb += complex(β_P, −β_Q)`. Hence
# ∂Y_bb/∂p = (−j)(−1) = +j and ΔI = j·V.
_dI_dp(d::Union{ControlledSwitchedShunt, ControlledFACTS}, r::_RectOrMixedResidual) =
    ((d.bus_ix, im * _bus_voltage(r, d.bus_ix)),)

# A tap perturbs three Y-bus entries (t_c = p·cis(α); Y_tt is p-independent), so it touches
# both terminals. Same ∂Y/∂p as the polar tap method above.
function _dI_dp(d::ControlledTap, r::_RectOrMixedResidual)
    f, t = d.from_ix, d.to_ix
    Vf, Vt = _bus_voltage(r, f), _bus_voltage(r, t)
    p, a = d.current, d.alpha
    dYff = -2.0 * d.yt / p^3
    dYft = d.yt * cis(a) / p^2
    dYtf = d.yt * cis(-a) / p^2
    return ((f, dYff * Vf + dYft * Vt), (t, dYtf * Vf))
end

# ── ΔI → residual rows, per formulation ───────────────────────────────────────────────────

# Rectangular CI: `F = I_spec − Y·V`, real part in slot 0 and imag in slot 1, uniformly at
# every bus type (`_update_rect_ci_residual_values!`). A PV bus's third row is the `|V|²`
# constraint, which carries no p dependence.
function _stamp_dI!(
    rhs::Vector{Float64},
    r::ACRectangularCIResidual,
    i::Int,
    dI::ComplexF64,
    ::PSY.ACBusTypes,
)
    off = _state_offset(r, i)
    @inbounds begin
        rhs[off] = -real(dI)
        rhs[off + 1] = -imag(dI)
    end
    return
end

# MCPB mixes three conventions in one vector (`_update_mixed_cpb_residual_values!`):
#   PQ  — divided current balance, IMAG-first (the two rect slots swapped so a nonzero B_ii
#         lands on the block diagonal). Rows swap; the (e, f) COLUMNS do not.
#   PV  — real-power balance `e·Ir + f·Ii − P_spec` plus a `|V|²` row. A purely reactive
#         injection cannot move either: `Re(V·conj(jV)) = 0`, and the `|V|²` row is
#         p-independent. Written generically so a tap (which does move real power) is right.
#   REF — rect-verbatim, real-first.
function _stamp_dI!(
    rhs::Vector{Float64},
    r::ACMixedCPBResidual,
    i::Int,
    dI::ComplexF64,
    bt::PSY.ACBusTypes,
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

# `|V|² = e² + f²` ⇒ dVm/dp = (e·de/dp + f·df/dp)/Vm, with dx/dp = −sol. The PQ state slots are
# `(e, f)` at `off`, `off+1` in both formulations, and `_linear_plant_sign` only reaches here at
# a PQ controlled bus. `V_FLOOR2` matches the residuals' own guard; at a converged base it never
# binds.
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
    # `Rv` is the full residual/state length (bus blocks + the LCC/VSC/area tail), so it sizes
    # `x` without re-deriving the tail.
    x = zeros(length(residual.Rv))
    rect_initial_state!(x, data, residual.bus_state_offset, residual.bus_block_size, ts)
    residual(x, ts)                       # evaluate at current state; fills e_state/f_state
    J = ACRectangularCIJacobian(residual, ts)
    J(ts)
    return residual, J
end

function _sensitivity_residual_jacobian(::ACMixedPowerFlow, data, ts::Int)
    residual = ACMixedCPBResidual(data, ts)
    x = zeros(length(residual.Rv))
    mixed_initial_state!(x, data, residual.bus_state_offset, residual.bus_block_size, ts)
    residual(x, ts)
    J = ACMixedCPBJacobian(residual, ts)
    J(ts)
    return residual, J
end
