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
