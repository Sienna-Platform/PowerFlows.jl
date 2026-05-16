"""
Return the reactive power limits that should be used in power flow calculations and PSS/E
exports. Redirects to `PSY.get_reactive_power_limits` in all but special cases.
"""
get_reactive_power_limits_for_power_flow(gen::PSY.Device) =
    PSY.get_reactive_power_limits(gen)

check_unit_setting(sys::PSY.System) = IS.@assert_op PSY.get_units_base(sys) == "SYSTEM_BASE"

function get_reactive_power_limits_for_power_flow(gen::PSY.RenewableNonDispatch)
    val = PSY.get_reactive_power(gen)
    return (min = val, max = val)
end

function get_reactive_power_limits_for_power_flow(gen::PSY.Storage)
    limits = PSY.get_reactive_power_limits(gen)
    isnothing(limits) && return (min = -Inf, max = Inf)  # TODO decide on proper behavior in this case
    return limits
end

"""
Return the active power limits that should be used in power flow calculations and PSS/E
exports. Redirects to `PSY.get_active_power_limits` in all but special cases.
"""
get_active_power_limits_for_power_flow(gen::PSY.Device) = PSY.get_active_power_limits(gen)

get_active_power_limits_for_power_flow(::PSY.Source) = (min = -Inf, max = Inf)

function get_active_power_limits_for_power_flow(gen::PSY.SynchronousCondenser)
    return (min = 0.0, max = 0.0)
end

function get_active_power_limits_for_power_flow(gen::PSY.RenewableNonDispatch)
    val = PSY.get_active_power(gen)
    return (min = val, max = val)
end

get_active_power_limits_for_power_flow(gen::PSY.RenewableDispatch) =
    (min = 0.0, max = PSY.get_rating(gen))

# TODO verify whether this is the correct behavior for Storage, (a) for redistribution and (b) for exporting
get_active_power_limits_for_power_flow(gen::PSY.Storage) =
    (min = 0.0, max = PSY.get_output_active_power_limits(gen).max)

"""
Return the active and reactive power generation from a generator component.
It's pg=0 as default for synchronous condensers since there's no field in the component for active power.
"""
function get_active_and_reactive_power_from_generator(gen::PSY.SynchronousCondenser)
    pg = 0.0
    qg = PSY.get_reactive_power(gen)
    return pg, qg
end

function get_active_and_reactive_power_from_generator(gen)
    pg = PSY.get_active_power(gen)
    qg = PSY.get_reactive_power(gen)
    return pg, qg
end

function set_power_flow!(br::PSY.ACTransmission, flow::Complex)
    PSY.set_active_power_flow!(br, real(flow))
    PSY.set_reactive_power_flow!(br, imag(flow))
    return
end

function set_power_flow!(br::PNM.BranchesParallel, flow::Complex)
    for segment in br
        weight = PNM.compute_parallel_multiplier(br, PSY.get_name(segment))
        set_power_flow!(segment, flow * weight)
    end
    return
end

function set_power_flow!(br::PSY.TwoTerminalLCCLine, flow::Complex)
    PSY.set_active_power_flow!(br, real(flow))
    # TwoTerminalLCCLine does not have reactive power flow attributes (even though PFD has Q results)
    return
end

function set_power_flow!(winding::PNM.ThreeWindingTransformerWinding, flow::Complex)
    (trf, num) = (PNM.get_transformer(winding), PNM.get_winding_number(winding))
    if num == 1
        PSY.set_active_power_flow_primary!(trf, real(flow))
        PSY.set_reactive_power_flow_primary!(trf, imag(flow))
    elseif num == 2
        PSY.set_active_power_flow_secondary!(trf, real(flow))
        PSY.set_reactive_power_flow_secondary!(trf, imag(flow))
    elseif num == 3
        PSY.set_active_power_flow_tertiary!(trf, real(flow))
        PSY.set_reactive_power_flow_tertiary!(trf, imag(flow))
    else
        error("Invalid winding number: $num")
    end
    return
end

function set_voltage!(bus::PSY.ACBus, V::Complex)
    PSY.set_magnitude!(bus, abs(V))
    PSY.set_angle!(bus, angle(V))
    return
end

"""Return set of all bus numbers that must be PV: i.e. have an available generator."""
function must_be_PV(sys::System)
    gen_buses = Set{Int}()
    for gen in PSY.get_available_components(PSY.Generator, sys)
        push!(gen_buses, PSY.get_number(PSY.get_bus(gen)))
    end
    # PSSe counts buses with switched shunts as PV, so we do the same here.
    for gen in PSY.get_available_components(PSY.SwitchedAdmittance, sys)
        push!(gen_buses, PSY.get_number(PSY.get_bus(gen)))
    end
    for gen in PSY.get_available_components(PSY.SynchronousCondenser, sys)
        push!(gen_buses, PSY.get_number(PSY.get_bus(gen)))
    end
    return gen_buses
end

"""Return set of all bus numbers that can be PV: i.e. have an available generator,
or certain voltage regulation devices."""
function can_be_PV(sys::System)
    pv_buses = must_be_PV(sys)
    for source in PSY.get_available_components(PSY.Source, sys)
        push!(pv_buses, PSY.get_number(PSY.get_bus(source)))
    end
    # Storage can also be considered PV (and are not Generator)
    for gen in PSY.get_available_components(PSY.Storage, sys)
        push!(pv_buses, PSY.get_number(PSY.get_bus(gen)))
    end
    return pv_buses
end

error_if_reversed(::PSY.TwoTerminalHVDC, ::Float64) = nothing

function error_if_reversed(hvdc::PSY.TwoTerminalLCCLine, P_dc::Float64)
    P_dc < 0 && throw(
        ArgumentError(
            "Power flow in $(PSY.summary(hvdc)) is reversed: active power flow " *
            "is $(PSY.get_active_power_flow(hvdc)), negative. Please check your inputs.",
        ),
    )
end

_eval_loss_function(curve::PSY.LinearCurve, x::Float64) = curve(x)

_eval_loss_function(pwl::PSY.PiecewiseIncrementalCurve, x::Float64) =
    IS.InputOutputCurve(pwl)(x)

# returns the tuple (P_dc, P_loss, flow_reversed), first two in natural units
function hvdc_power_loss_natural_units(hvdc::PSY.TwoTerminalHVDC)
    P_dc = with_units_base(hvdc, "NATURAL_UNITS") do
        PSY.get_active_power_flow(hvdc)
    end
    error_if_reversed(hvdc, P_dc)
    flow_reversed = P_dc < 0
    P_dc = abs(P_dc)
    loss_curve = PSY.get_loss(hvdc)
    P_loss = _eval_loss_function(loss_curve, P_dc)
    P_loss > P_dc && @warn "The loss curve of $(PSY.summary(hvdc)) " *
          "indicates the losses are greater than the transmitted power $P_dc. " *
          "Setting the loss equal to the transmitted power instead."
    P_loss < 0.0 && @warn "The loss curve of $(PSY.summary(hvdc)) " *
          "indicates negative losses for transmitted power $P_dc. " *
          "Setting the loss equal to zero instead."
    return (P_dc, clamp(P_loss, 0.0, P_dc), flow_reversed)
end

# VSC lines have separate converter losses on each end
function hvdc_power_loss_natural_units(hvdc::PSY.TwoTerminalVSCLine)
    P_dc = with_units_base(hvdc, "NATURAL_UNITS") do
        PSY.get_active_power_flow(hvdc)
    end
    flow_reversed = P_dc < 0
    P_dc = abs(P_dc)
    # Sum losses from both converters
    loss_from = _eval_loss_function(PSY.get_converter_loss_from(hvdc), P_dc)
    loss_to = _eval_loss_function(PSY.get_converter_loss_to(hvdc), P_dc)
    P_loss = loss_from + loss_to
    P_loss > P_dc && @warn "The converter losses of $(PSY.summary(hvdc)) " *
          "indicate losses greater than the transmitted power $P_dc. " *
          "Setting the loss equal to the transmitted power instead."
    P_loss < 0.0 && @warn "The converter losses of $(PSY.summary(hvdc)) " *
          "indicate negative losses for transmitted power $P_dc. " *
          "Setting the loss equal to zero instead."
    return (P_dc, clamp(P_loss, 0.0, P_dc), flow_reversed)
end

function get_hvdc_power_loss(
    hvdc::PSY.TwoTerminalHVDC,
    sys::PSY.System,
)
    base_power = PSY.get_base_power(sys)
    (P_dc, P_loss, flow_reversed) = hvdc_power_loss_natural_units(hvdc)
    return (P_dc / base_power, P_loss / base_power, flow_reversed)
end

# returns the tuple (P_net_from, P_net_to), both in natural units
function hvdc_injections_natural_units(hvdc::PSY.TwoTerminalHVDC)
    P_dc, P_loss, flow_reversed = hvdc_power_loss_natural_units(hvdc)
    P_received = P_dc - P_loss
    @assert P_received >= 0.0 - eps() && P_received <= P_dc + eps()
    # (from, to) net powers: reversed means from is receiving power.
    return flow_reversed ? (P_received, -P_dc) : (-P_dc, P_received)
end

function get_hvdc_injections(
    hvdc::PSY.TwoTerminalHVDC,
    sys::PSY.System,
)
    base_power = PSY.get_base_power(sys)
    (P_from, P_to) = hvdc_injections_natural_units(hvdc)
    return (P_from / base_power, P_to / base_power)
end

# somewhat duplicative of code in PNM. But currently we're passing around just the reverse
# bus search map, rather than the full network reduction data.
function get_arc_tuple(arc::PSY.Arc, reverse_bus_search_map::Dict{Int, Int})
    from_bus, to_bus = PSY.get_number(PSY.get_from(arc)), PSY.get_number(PSY.get_to(arc))
    return (
        get(reverse_bus_search_map, from_bus, from_bus),
        get(reverse_bus_search_map, to_bus, to_bus),
    )
end

function convert_zip_to_constant_power!(p_load::AbstractArray{T, N},
    i_load::AbstractArray{T, N},
    z_load::AbstractArray{T, N},
    voltage_magnitude::T,
) where {T <: Real, N}
    # faster with broadcast fusion via @. ?
    p_load .+= voltage_magnitude .* i_load .+ voltage_magnitude^2 .* z_load
    i_load .= zero(T)
    z_load .= zero(T)
end
