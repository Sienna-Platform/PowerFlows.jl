"""
    _calculate_ϕ_lcc(α::Float64, I_dc::Float64, x_t::Float64, Vm::Float64) -> Float64

Compute the phase angle ϕ for LCC converter calculations.
"""
function _calculate_ϕ_lcc(
    t::Float64,
    α::Float64,
    I_dc::Float64,
    x_t::Float64,
    Vm::Float64,
)::Float64
    raw = sign(I_dc) * (cos(α) - (x_t * I_dc) / (sqrt(2) * Vm * t))
    if raw < -1.0 || raw > 1.0
        @warn "LCC ϕ argument outside [-1, 1] (got $raw); clamping. \
               Derivative formulas in lcc_utils.jl are singular on this boundary \
               and the analytic Jacobian disagrees with the residual past it — \
               Newton-Raphson may struggle. Check α, I_dc, x_t, t, Vm." maxlog = 5
    end
    return acos(clamp(raw, -1.0, 1.0))
end

"""
    _calculate_y_lcc(t::Float64, I_dc::Float64, Vm::Float64, ϕ::Float64) -> ComplexF64

Compute the admittance value Y for LCC converter calculations.
"""
function _calculate_y_lcc(t::Float64, I_dc::Float64, Vm::Float64, ϕ::Float64)::ComplexF64
    return t / Vm * sqrt(6) / π * I_dc * exp(-1im * ϕ)
end

"""
    _calculate_dP_dV_lcc(t, I_dc, x_t, Vm, ϕ) -> Float64

True-ϕ derivative of `P_lcc = Vm · t · √6/π · I_dc · cos(ϕ(Vm, t, α))` with
respect to `Vm`. Two regimes:

  * Interior (ϕ unclamped): `∂ϕ/∂Vm = -∂raw/∂Vm / sin(ϕ)` is nonzero; the
    `sin(ϕ)` factor from the chain rule cancels the `-sin(ϕ)` from
    differentiating `cos(ϕ)`, giving the second (chain) term below.
  * Clamp (sin(ϕ) ≈ 0, i.e. ϕ ∈ {0, π}): ϕ is locally pinned (`∂ϕ/∂x = 0`)
    and the residual sees only the leading `Vm · cos(ϕ)` dependence on
    `Vm`. The chain term must be dropped — otherwise the analytic Jacobian
    disagrees with the residual at the clamp, exactly analogous to the
    `sin(ϕ)→0` guard in `_calculate_dQ_dV_lcc`.

Caller passes `I_dc > 0` and the side-specific ϕ. Rectifier: `phi_r`.
Inverter: `phi_i` (already encodes the sign convention via
`_calculate_ϕ_lcc(-I_dc, …)`; positive `I_dc` is still passed here).
"""
function _calculate_dP_dV_lcc(
    t::Float64,
    I_dc::Float64,
    x_t::Float64,
    Vm::Float64,
    ϕ::Float64,
)::Float64
    leading = t * sqrt(6) / π * I_dc * cos(ϕ)
    sin(ϕ) < LCC_sinϕ_TOLERANCE && return leading  # clamped: ∂ϕ/∂Vm = 0
    return leading + sqrt(6) / π * I_dc^2 * x_t / (sqrt(2) * Vm)
end

"""
    _calculate_dP_dt_lcc(t, I_dc, x_t, Vm, ϕ) -> Float64

True-ϕ derivative of `P_lcc` with respect to the transformer tap `t`. Same
two-regime structure as `_calculate_dP_dV_lcc`: chain term only when
unclamped (`sin(ϕ) ≥ LCC_sinϕ_TOLERANCE`); leading `Vm · cos(ϕ)` term
always present.
"""
function _calculate_dP_dt_lcc(
    t::Float64,
    I_dc::Float64,
    x_t::Float64,
    Vm::Float64,
    ϕ::Float64,
)::Float64
    leading = Vm * sqrt(6) / π * I_dc * cos(ϕ)
    sin(ϕ) < LCC_sinϕ_TOLERANCE && return leading  # clamped: ∂ϕ/∂t = 0
    return leading + sqrt(6) / π * I_dc^2 * x_t / (sqrt(2) * t)
end

"""
    _dphi_dV_lcc(x_t, I_dc, V, t, ϕ) -> Float64

`∂ϕ/∂V` with `sin(ϕ) → 0` clamp guard returning 0. In the interior,
`∂ϕ/∂V = -∂raw/∂V / sin(ϕ) = -x_t·I_dc / (√2·V²·t·sin(ϕ))`. At the
clamp ϕ is pinned (`∂ϕ/∂V = 0`). Same form on both sides — the inverter
passes the same positive `I_dc`, only its `ϕ` differs.
"""
function _dphi_dV_lcc(
    x_t::Float64,
    I_dc::Float64,
    V::Float64,
    t::Float64,
    ϕ::Float64,
)::Float64
    sϕ = sin(ϕ)
    sϕ < LCC_sinϕ_TOLERANCE && return 0.0
    return -x_t * I_dc / (sqrt(2) * V^2 * t * sϕ)
end

"""
    _dphi_dt_lcc(x_t, I_dc, V, t, ϕ) -> Float64

`∂ϕ/∂t` (tap) with clamp guard. `-x_t·I_dc / (√2·V·t²·sin(ϕ))` in the
interior, 0 at the clamp.
"""
function _dphi_dt_lcc(
    x_t::Float64,
    I_dc::Float64,
    V::Float64,
    t::Float64,
    ϕ::Float64,
)::Float64
    sϕ = sin(ϕ)
    sϕ < LCC_sinϕ_TOLERANCE && return 0.0
    return -x_t * I_dc / (sqrt(2) * V * t^2 * sϕ)
end

"""
    _dphi_dα_lcc(α, ϕ) -> Float64

`∂ϕ/∂α` (rectifier sign) with clamp guard. `sin(α)/sin(ϕ)` in the
interior, 0 at the clamp. Inverter convention flips the sign — callers
on the inverter side negate the helper output.
"""
function _dphi_dα_lcc(α::Float64, ϕ::Float64)::Float64
    sϕ = sin(ϕ)
    sϕ < LCC_sinϕ_TOLERANCE && return 0.0
    return sin(α) / sϕ
end

"""
    _calculate_dP_dα_lcc(t, I_dc, Vm, α, ϕ) -> Float64

True-ϕ derivative of `P_lcc` with respect to the firing/extinction angle
`α`, rectifier sign convention. In the interior, `∂ϕ/∂α = sin(α)/sin(ϕ)`
and combines with the `-sin(ϕ)` from differentiating `cos(ϕ)` to give the
closed form below (no `sin(ϕ)` in the result). At the clamp, `∂ϕ/∂α = 0`
and the true derivative is zero — same boundary handling as the dQ
helpers. Inverter callers must negate the helper output (the inverter ϕ
convention flips `∂ϕ_i/∂α_i`).
"""
function _calculate_dP_dα_lcc(
    t::Float64,
    I_dc::Float64,
    Vm::Float64,
    α::Float64,
    ϕ::Float64,
)::Float64
    sin(ϕ) < LCC_sinϕ_TOLERANCE && return 0.0
    return -Vm * t * sqrt(6) / π * I_dc * sin(α)
end

"""
    _calculate_dQ_dV_lcc(t::Float64, I_dc::Float64, x_t::Float64, Vm::Float64, ϕ::Float64) -> Float64

Compute the derivative of reactive power Q with respect to voltage magnitude Vm for LCC converter calculations.
"""
function _calculate_dQ_dV_lcc(
    t::Float64,
    I_dc::Float64,
    x_t::Float64,
    Vm::Float64,
    ϕ::Float64,
)::Float64
    sϕ = sin(ϕ)
    # On the clamp boundary (sin(ϕ) = 0), φ is locally pinned and the residual
    # is constant in this direction, so the true derivative is 0 even though
    # the analytic formula has a 1/sin(ϕ) singularity.
    sϕ < LCC_sinϕ_TOLERANCE && return 0.0
    return t * sqrt(6) / π * I_dc * sϕ -
           sqrt(6) / π * cos(ϕ) * sign(I_dc) * I_dc^2 * x_t /
           (sqrt(2) * Vm * sϕ)
end

"""
    _calculate_dQ_dt_lcc(t::Float64, I_dc::Float64, x_t::Float64, Vm::Float64, ϕ::Float64) -> Float64

Compute the derivative of reactive power Q with respect to transformer tap t for LCC converter calculations.
"""
function _calculate_dQ_dt_lcc(
    t::Float64,
    I_dc::Float64,
    x_t::Float64,
    Vm::Float64,
    ϕ::Float64,
)::Float64
    sϕ = sin(ϕ)
    sϕ < LCC_sinϕ_TOLERANCE && return 0.0
    return Vm * sqrt(6) / π * I_dc * sϕ -
           sqrt(6) / π * cos(ϕ) * sign(I_dc) * I_dc^2 * x_t /
           (sqrt(2) * t * sϕ)
end

"""
    _calculate_dQ_dα_lcc(t::Float64, I_dc::Float64, x_t::Float64, Vm::Float64, ϕ::Float64, α::Float64) -> Float64

Compute the derivative of reactive power Q with respect to firing/extinction angle α for LCC converter calculations.
"""
function _calculate_dQ_dα_lcc(
    t::Float64,
    I_dc::Float64,
    x_t::Float64,
    Vm::Float64,
    ϕ::Float64,
    α::Float64,
)::Float64
    sϕ = sin(ϕ)
    sϕ < LCC_sinϕ_TOLERANCE && return 0.0
    return Vm * t * sqrt(6) / π * I_dc * cos(ϕ) * sin(α) / sϕ
end

"""
    _update_ybus_lcc!(data, time_step)

Recompute `data.lcc.rectifier.phi`, `data.lcc.inverter.phi`, and
`data.lcc.branch_admittances` for each LCC at `time_step`. Reads `|V|` at each
AC terminal from `data.bus_magnitude` (the polar convention). The
`(e_state, f_state)` method below covers the rectangular CI case where
`|V_state| = sqrt(e² + f²)` must be used instead — at PV buses,
`data.bus_magnitude` holds `V_set` rather than the actual state magnitude.
"""
function _update_ybus_lcc!(data::PowerFlowData, time_step::Int64)
    for (i, (fb, tb)) in enumerate(data.lcc.bus_indices)
        Vm_fb = data.bus_magnitude[fb, time_step]
        Vm_tb = data.bus_magnitude[tb, time_step]
        data.lcc.rectifier.phi[i, time_step] = _calculate_ϕ_lcc(
            data.lcc.rectifier.tap[i, time_step],
            data.lcc.rectifier.thyristor_angle[i, time_step],
            data.lcc.i_dc[i, time_step],
            data.lcc.rectifier.transformer_reactance[i],
            Vm_fb,
        )
        data.lcc.inverter.phi[i, time_step] = _calculate_ϕ_lcc(
            data.lcc.inverter.tap[i, time_step],
            data.lcc.inverter.thyristor_angle[i, time_step],
            -data.lcc.i_dc[i, time_step],
            data.lcc.inverter.transformer_reactance[i],
            Vm_tb,
        )
        rectifier_admittance = _calculate_y_lcc(
            data.lcc.rectifier.tap[i, time_step],
            data.lcc.i_dc[i, time_step],
            Vm_fb,
            data.lcc.rectifier.phi[i, time_step],
        )
        inverter_admittance = _calculate_y_lcc(
            data.lcc.inverter.tap[i, time_step],
            data.lcc.i_dc[i, time_step],
            Vm_tb,
            data.lcc.inverter.phi[i, time_step],
        )
        data.lcc.branch_admittances[i] = (rectifier_admittance, inverter_admittance)
    end
    return
end

"""
    _update_ybus_lcc!(data, time_step, e_state, f_state)

Rectangular variant: reads `|V|` at each AC terminal from
`sqrt(e_state[i]^2 + f_state[i]^2)` so the LCC math stays consistent with the
rectangular CI residual / Jacobian (which operate on `(e, f)` instead of
`(|V|, θ)`).
"""
function _update_ybus_lcc!(
    data::PowerFlowData,
    time_step::Int64,
    e_state::Vector{Float64},
    f_state::Vector{Float64},
)
    for (i, (fb, tb)) in enumerate(data.lcc.bus_indices)
        Vm_fb = sqrt(e_state[fb]^2 + f_state[fb]^2)
        Vm_tb = sqrt(e_state[tb]^2 + f_state[tb]^2)
        data.lcc.rectifier.phi[i, time_step] = _calculate_ϕ_lcc(
            data.lcc.rectifier.tap[i, time_step],
            data.lcc.rectifier.thyristor_angle[i, time_step],
            data.lcc.i_dc[i, time_step],
            data.lcc.rectifier.transformer_reactance[i],
            Vm_fb,
        )
        data.lcc.inverter.phi[i, time_step] = _calculate_ϕ_lcc(
            data.lcc.inverter.tap[i, time_step],
            data.lcc.inverter.thyristor_angle[i, time_step],
            -data.lcc.i_dc[i, time_step],
            data.lcc.inverter.transformer_reactance[i],
            Vm_tb,
        )
        rectifier_admittance = _calculate_y_lcc(
            data.lcc.rectifier.tap[i, time_step],
            data.lcc.i_dc[i, time_step],
            Vm_fb,
            data.lcc.rectifier.phi[i, time_step],
        )
        inverter_admittance = _calculate_y_lcc(
            data.lcc.inverter.tap[i, time_step],
            data.lcc.i_dc[i, time_step],
            Vm_tb,
            data.lcc.inverter.phi[i, time_step],
        )
        data.lcc.branch_admittances[i] = (rectifier_admittance, inverter_admittance)
    end
    return
end

"""
    _set_lcc_tail_residuals!(F, data, base_offset, time_step) [polar]
    _set_lcc_tail_residuals!(F, data, base_offset, time_step, e_state, f_state) [rect]

Write the 4 LCC tail residual rows (P-setpoint, DC-line balance, two α
limit constraints) for each LCC into `F`, starting at slot
`base_offset + 1`. The i-th LCC occupies slots `base_offset + 4(i-1) + 1
.. base_offset + 4i`. The polar method reads `|V|` from
`data.bus_magnitude`; the rectangular method reads `sqrt(e² + f²)` from
the (e, f) state (since `data.bus_magnitude` holds `V_set` at PV buses,
not the actual state magnitude). Mirrors the two-method layout of
[`_update_ybus_lcc!`](@ref).
"""
function _set_lcc_tail_residuals!(
    F::AbstractVector{Float64},
    data::PowerFlowData,
    base_offset::Int,
    time_step::Int,
)
    @inbounds for i in 1:size(data.lcc.p_set, 1)
        (fb, tb) = data.lcc.bus_indices[i]
        _write_lcc_tail!(
            F, data, base_offset, time_step, i, fb, tb,
            data.bus_magnitude[fb, time_step],
            data.bus_magnitude[tb, time_step],
        )
    end
    return
end

function _set_lcc_tail_residuals!(
    F::AbstractVector{Float64},
    data::PowerFlowData,
    base_offset::Int,
    time_step::Int,
    e_state::Vector{Float64},
    f_state::Vector{Float64},
)
    @inbounds for i in 1:size(data.lcc.p_set, 1)
        (fb, tb) = data.lcc.bus_indices[i]
        Vm_fb = sqrt(e_state[fb]^2 + f_state[fb]^2)
        Vm_tb = sqrt(e_state[tb]^2 + f_state[tb]^2)
        _write_lcc_tail!(F, data, base_offset, time_step, i, fb, tb, Vm_fb, Vm_tb)
    end
    return
end

@inline function _write_lcc_tail!(
    F::AbstractVector{Float64},
    data::PowerFlowData,
    base_offset::Int,
    time_step::Int,
    i::Int,
    fb::Int,
    tb::Int,
    Vm_fb::Float64,
    Vm_tb::Float64,
)
    offset_lcc = base_offset + (i - 1) * 4
    tap_r = data.lcc.rectifier.tap[i, time_step]
    tap_i = data.lcc.inverter.tap[i, time_step]
    phi_r = data.lcc.rectifier.phi[i, time_step]
    phi_i = data.lcc.inverter.phi[i, time_step]
    i_dc = data.lcc.i_dc[i, time_step]
    P_lcc_from = Vm_fb * tap_r * SQRT6_DIV_PI * i_dc * cos(phi_r)
    P_lcc_to = Vm_tb * tap_i * SQRT6_DIV_PI * i_dc * cos(phi_i)
    F[offset_lcc + 1] = if data.lcc.setpoint_at_rectifier[i]
        P_lcc_from - data.lcc.p_set[i, time_step]
    else
        -P_lcc_to - data.lcc.p_set[i, time_step]
    end
    F[offset_lcc + 2] =
        P_lcc_from + P_lcc_to - data.lcc.dc_line_resistance[i] * i_dc^2
    F[offset_lcc + 3] =
        data.lcc.rectifier.thyristor_angle[i, time_step] -
        data.lcc.rectifier.min_thyristor_angle[i]
    F[offset_lcc + 4] =
        data.lcc.inverter.thyristor_angle[i, time_step] -
        data.lcc.inverter.min_thyristor_angle[i]
    return
end

"""
    _lcc_jacobian_scalars(data, i, time_step, Vm_fb, Vm_tb)

Precompute the scalar coefficients used by both the polar and the
rectangular LCC Jacobian assembly for LCC `i` at `time_step`. `Vm_fb` /
`Vm_tb` are the AC-side voltage magnitudes — polar reads them from
`data.bus_magnitude`; rectangular computes `sqrt(e² + f²)` from state.

The returned NamedTuple includes the six tail-row × tail-column entries
that are identical between formulations (the tail rows themselves are
identical, and so are the tail-state columns). These are computed via
the true-ϕ helpers (`_calculate_dP_dt_lcc`, `_calculate_dP_dα_lcc`),
which apply the `sin(ϕ) → 0` boundary guard: in the interior the
algebraic identity makes the result equal to the α-approximation form,
and at the clamp the guard correctly drops the chain term so the
Jacobian matches the residual (which sees `∂ϕ/∂x = 0` at the clamp).
"""
function _lcc_jacobian_scalars(
    data::PowerFlowData,
    i::Int,
    time_step::Int,
    Vm_fb::Float64,
    Vm_tb::Float64,
)
    i_dc = max(data.lcc.i_dc[i, time_step], 1e-9)
    tap_r = data.lcc.rectifier.tap[i, time_step]
    tap_i = data.lcc.inverter.tap[i, time_step]
    alpha_r = data.lcc.rectifier.thyristor_angle[i, time_step]
    alpha_i = data.lcc.inverter.thyristor_angle[i, time_step]
    phi_r = data.lcc.rectifier.phi[i, time_step]
    phi_i = data.lcc.inverter.phi[i, time_step]
    xtr_r = data.lcc.rectifier.transformer_reactance[i]
    xtr_i = data.lcc.inverter.transformer_reactance[i]
    cos_alpha_r = cos(alpha_r)
    sin_alpha_r = sin(alpha_r)
    cos_alpha_i = cos(alpha_i)
    sin_alpha_i = sin(alpha_i)
    common_fb = Vm_fb * SQRT6_DIV_PI * i_dc
    common_tb = Vm_tb * SQRT6_DIV_PI * (-i_dc)
    common_tap_r = tap_r * SQRT6_DIV_PI * i_dc * cos_alpha_r
    common_tap_i = tap_i * SQRT6_DIV_PI * (-i_dc) * cos_alpha_i
    common_alpha_r = -common_fb * tap_r * sin_alpha_r
    common_alpha_i = -common_tb * tap_i * sin_alpha_i
    # True-ϕ derivatives of P_lcc_{from, to} for the tail × tail block.
    # Inverter signs:
    #   ∂P_lcc_to/∂tap_i: the helper returns the rectifier-style formula;
    #     for the inverter `phi_i ≈ π − α_i` makes `cos(phi_i) < 0`, so the
    #     helper already returns the correct negative coefficient — no sign
    #     flip needed here.
    #   ∂P_lcc_to/∂α_i: ϕ_i convention flips `∂ϕ_i/∂α_i`, so negate the helper.
    dP_dV_fb = _calculate_dP_dV_lcc(tap_r, i_dc, xtr_r, Vm_fb, phi_r)
    dP_dV_tb = _calculate_dP_dV_lcc(tap_i, i_dc, xtr_i, Vm_tb, phi_i)
    dP_dt_fb = _calculate_dP_dt_lcc(tap_r, i_dc, xtr_r, Vm_fb, phi_r)
    dP_dt_tb = _calculate_dP_dt_lcc(tap_i, i_dc, xtr_i, Vm_tb, phi_i)
    dP_dα_fb = _calculate_dP_dα_lcc(tap_r, i_dc, Vm_fb, alpha_r, phi_r)
    dP_dα_tb = -_calculate_dP_dα_lcc(tap_i, i_dc, Vm_tb, alpha_i, phi_i)
    return (
        i_dc = i_dc,
        tap_r = tap_r,
        tap_i = tap_i,
        cos_alpha_r = cos_alpha_r,
        sin_alpha_r = sin_alpha_r,
        cos_alpha_i = cos_alpha_i,
        sin_alpha_i = sin_alpha_i,
        common_fb = common_fb,
        common_tb = common_tb,
        common_tap_r = common_tap_r,
        common_tap_i = common_tap_i,
        common_alpha_r = common_alpha_r,
        common_alpha_i = common_alpha_i,
        # Side-specific dP/dx helpers (true-ϕ, with sin(ϕ) → 0 clamp guard
        # where applicable). Exposed so polar's bus-row entries — and
        # rect's tail × bus chain rules — can read pre-computed values
        # instead of re-calling the helpers.
        dP_dV_fb = dP_dV_fb,
        dP_dV_tb = dP_dV_tb,
        dP_dt_fb = dP_dt_fb,
        dP_dt_tb = dP_dt_tb,
        dP_dα_fb = dP_dα_fb,
        dP_dα_tb = dP_dα_tb,
        # Tail-row × tail-column block (6 entries). F_t_fb has the
        # P_lcc_from contribution (in the setpoint_at_rectifier case);
        # F_t_tb has both P_lcc_from and P_lcc_to. d_Ft_fb_d_alpha_i is
        # zero because F_t_fb doesn't depend on α_i.
        d_Ft_fb_d_tap_r = dP_dt_fb,
        d_Ft_fb_d_alpha_r = dP_dα_fb,
        d_Ft_tb_d_tap_r = dP_dt_fb,
        d_Ft_tb_d_tap_i = dP_dt_tb,
        d_Ft_tb_d_alpha_r = dP_dα_fb,
        d_Ft_tb_d_alpha_i = dP_dα_tb,
    )
end

"""
Initialize the `arcs` and `bus_indices` fields of the LCCParameters structure in the PowerFlowData.
"""
function initialize_LCC_arcs_and_buses!(
    data::PowerFlowData,
    lccs::Vector{PSY.TwoTerminalLCCLine},
    bus_lookup::Dict{Int, Int},
    reverse_bus_search_map::Dict{Int, Int},
)
    lcc_arcs = PSY.get_arc.(lccs)
    nrd = get_network_reduction_data(data)
    for (i, arc) in enumerate(lcc_arcs)
        data.lcc.arcs[i] = PNM.get_arc_tuple(arc, nrd)
        data.lcc.bus_indices[i] = (
            _get_bus_ix(
                bus_lookup,
                reverse_bus_search_map,
                PSY.get_number(PSY.get_from(arc)),
            ),
            _get_bus_ix(
                bus_lookup,
                reverse_bus_search_map,
                PSY.get_number(PSY.get_to(arc)),
            ),
        )
    end
    return
end

function initialize_LCCParameters!(
    data::Union{PTDFPowerFlowData, vPTDFPowerFlowData, ABAPowerFlowData},
    sys::PSY.System,
    bus_lookup::Dict{Int, Int},
    reverse_bus_search_map::Dict{Int, Int},
    removed_buses::Set{Int},
)
    check_unit_setting(sys)
    lccs = collect(
        PSY.get_available_components(
            x -> x.arc.from.number ∉ removed_buses && x.arc.to.number ∉ removed_buses,
            PSY.TwoTerminalLCCLine,
            sys,
        ),
    )
    isempty(lccs) && return

    initialize_LCC_arcs_and_buses!(data, lccs, bus_lookup, reverse_bus_search_map)

    # for DC power flow calculations, LCC arc flows are known from quantities from setup.
    for (i, lcc_branch) in enumerate(lccs)
        # it's an LCC, so flow can't be reversed; rhs will error if it is.
        (P_from_to, P_to_from, _) = get_hvdc_power_loss(lcc_branch, sys)
        data.lcc.arc_active_power_flow_from_to[i, :] .= P_from_to
        data.lcc.arc_active_power_flow_to_from[i, :] .= P_to_from
    end
    return
end

function initialize_LCCParameters!(
    data::ACPowerFlowData,
    sys::PSY.System,
    bus_lookup::Dict{Int, Int},
    reverse_bus_search_map::Dict{Int, Int},
    removed_buses::Set{Int},
)
    check_unit_setting(sys)
    lccs = collect(
        PSY.get_available_components(
            x -> x.arc.from.number ∉ removed_buses && x.arc.to.number ∉ removed_buses,
            PSY.TwoTerminalLCCLine,
            sys,
        ),
    )
    isempty(lccs) && return

    lcc_setpoint_at_rectifier = get_lcc_setpoint_at_rectifier(data)
    @assert length(lcc_setpoint_at_rectifier) == length(lccs)
    lcc_p_set = get_lcc_p_set(data)
    lcc_i_dc = get_lcc_i_dc(data)
    lcc_dc_line_resistance = get_lcc_dc_line_resistance(data)
    lcc_rectifier_tap = get_lcc_rectifier_tap(data)
    lcc_inverter_tap = get_lcc_inverter_tap(data)
    lcc_rectifier_delay_angle = get_lcc_rectifier_thyristor_angle(data)
    lcc_inverter_extinction_angle = get_lcc_inverter_thyristor_angle(data)

    lcc_rectifier_bus = get_lcc_rectifier_bus(data)
    lcc_inverter_bus = get_lcc_inverter_bus(data)
    lcc_rectifier_transformer_reactance = get_lcc_rectifier_transformer_reactance(data)
    lcc_inverter_transformer_reactance = get_lcc_inverter_transformer_reactance(data)
    lcc_rectifier_min_alpha = get_lcc_rectifier_min_thyristor_angle(data)
    lcc_inverter_min_gamma = get_lcc_inverter_min_thyristor_angle(data)

    initialize_LCC_arcs_and_buses!(data, lccs, bus_lookup, reverse_bus_search_map)

    lcc_arcs = PSY.get_arc.(lccs)

    base_power = PSY.get_base_power(sys)
    # todo: if current set point, transform into p set point
    # lcc_p_set = I_dc_A * V_dc_V / system_base_MVA

    lcc_setpoint_at_rectifier .= (PSY.get_transfer_setpoint.(lccs) .>= 0.0)
    lcc_p_set .= abs.(PSY.get_transfer_setpoint.(lccs) ./ base_power) # only one direction is supported, no reverse flow possible
    lcc_rectifier_tap[:, 1] .= PSY.get_rectifier_tap_setting.(lccs)
    lcc_inverter_tap[:, 1] .= PSY.get_inverter_tap_setting.(lccs)
    lcc_dc_line_resistance .=
        PSY.get_r.(lccs) .+ PSY.get_rectifier_rc.(lccs) .+ PSY.get_inverter_rc.(lccs)
    lcc_i_dc .=
        (-1 .+ sqrt.(1 .+ 4 .* lcc_dc_line_resistance .* lcc_p_set)) ./
        (2 .* lcc_dc_line_resistance)
    lcc_rectifier_delay_angle[:, 1] .= PSY.get_rectifier_delay_angle.(lccs)
    lcc_inverter_extinction_angle[:, 1] .= PSY.get_inverter_extinction_angle.(lccs)
    lcc_rectifier_bus .= [
        _get_bus_ix(bus_lookup, reverse_bus_search_map, x) for
        x in PSY.get_number.(PSY.get_from.(lcc_arcs))
    ]
    lcc_inverter_bus .= [
        _get_bus_ix(bus_lookup, reverse_bus_search_map, x) for
        x in PSY.get_number.(PSY.get_to.(lcc_arcs))
    ]
    lcc_rectifier_transformer_reactance .= PSY.get_rectifier_xc.(lccs)
    lcc_inverter_transformer_reactance .= PSY.get_inverter_xc.(lccs)
    lcc_rectifier_min_alpha .=
        [x.min for x in PSY.get_rectifier_delay_angle_limits.(lccs)]
    lcc_inverter_min_gamma .=
        [x.min for x in PSY.get_inverter_extinction_angle_limits.(lccs)]
    return
end

"""
Adjust the power injections/withdrawal vectors to account for all HVDC lines of a given type,
modeling those HVDC lines as a simple fixed injection/withdrawal at each terminal.
"""
function hvdc_fixed_injections!(
    data::PowerFlowData,
    hvdc_type::Type{<:PSY.TwoTerminalHVDC},
    sys::PSY.System,
    bus_lookup::Dict{Int, Int},
    reverse_bus_search_map::Dict{Int, Int},
    removed_buses::Set{Int},
)
    for hvdc in PSY.get_available_components(hvdc_type, sys)
        arc = PSY.get_arc(hvdc)
        from_number = PSY.get_number(PSY.get_from(arc))
        to_number = PSY.get_number(PSY.get_to(arc))
        from_number in removed_buses && continue
        to_number in removed_buses && continue
        (P_net_from, P_net_to) = get_hvdc_injections(hvdc, sys)
        from_bus_ix = _get_bus_ix(bus_lookup, reverse_bus_search_map, from_number)
        to_bus_ix = _get_bus_ix(bus_lookup, reverse_bus_search_map, to_number)
        data.bus_hvdc_net_power[from_bus_ix, :] .+= P_net_from
        data.bus_hvdc_net_power[to_bus_ix, :] .+= P_net_to
    end
    return
end

lcc_vsc_fixed_injections!(
    ::ACPowerFlowData,
    ::PSY.System,
    ::Dict{Int, Int},
    ::Dict{Int, Int},
    ::Set{Int},
) = nothing

lcc_vsc_fixed_injections!(
    data::Union{PTDFPowerFlowData, vPTDFPowerFlowData, ABAPowerFlowData},
    sys::PSY.System,
    bus_lookup::Dict{Int, Int},
    reverse_bus_search_map::Dict{Int, Int},
    removed_buses::Set{Int},
) =
    hvdc_fixed_injections!.(
        (data,),
        (PSY.TwoTerminalLCCLine, PSY.TwoTerminalVSCLine),
        (sys,),
        (bus_lookup,),
        (reverse_bus_search_map,),
        (removed_buses,),
    )

function initialize_generic_hvdc_flows!(
    data::PowerFlowData,
    sys::PSY.System,
    reverse_bus_search_map::Dict{Int, Int},
)
    for comp in PSY.get_available_components(PSY.TwoTerminalGenericHVDCLine, sys)
        (P_dc, P_loss, flow_reversed) = get_hvdc_power_loss(comp, sys)
        arc = PSY.get_arc(comp)
        arc_tuple = get_arc_tuple(arc, reverse_bus_search_map)
        if !flow_reversed
            data.generic_hvdc_flows[arc_tuple] = (P_dc, P_loss - P_dc)
        else
            data.generic_hvdc_flows[arc_tuple] = (P_loss - P_dc, P_dc)
        end
    end
end
