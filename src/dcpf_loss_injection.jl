# DCLF-style loss injection
# Pre-compute branch losses from the AC voltage profile and inject at sending-end buses.

# Helpers to extract tap ratio and phase shift from direct branches,
# following the same dispatch pattern as PNM._get_tap in Ybus.jl.
_branch_tap(::PSY.ACBranch) = 1.0
_branch_tap(br::PSY.TwoWindingTransformer) = PSY.get_tap(br)
_branch_tap(::PSY.Transformer2W) = 1.0

_branch_shift(::PSY.ACBranch) = 0.0
_branch_shift(br::PSY.PhaseShiftingTransformer) = PSY.get_α(br)

# Helpers for three-winding transformer tap and shift.
# Only PhaseShiftingTransformer3W windings carry tap/shift; regular Transformer3W defaults to 1.0/0.0.
_winding_tap(w::PNM.ThreeWindingTransformerWinding{PSY.PhaseShiftingTransformer3W}) =
    PNM.get_equivalent_tap(w)
_winding_tap(::PNM.ThreeWindingTransformerWinding) = 1.0

_winding_shift(w::PNM.ThreeWindingTransformerWinding{PSY.PhaseShiftingTransformer3W}) =
    PNM.get_equivalent_α(w)
_winding_shift(::PNM.ThreeWindingTransformerWinding) = 0.0

_added_impedance_rx(z::Number) = (real(z), imag(z))
_added_impedance_rx(z::PSY.GenericArcImpedance) = (z.r, z.x)

"""
    _get_arc_branch_params(data) -> (rs, xs, taps, shifts)

Return vectors of series resistance, reactance, tap ratio, and phase shift
for each arc, looking up every branch map in the `NetworkReductionData`.

This is the single source of truth for per-arc electrical parameters;
[`_get_arc_resistances`](@ref) delegates to it.
"""
function _get_arc_branch_params(
    data::Union{PTDFPowerFlowData, vPTDFPowerFlowData, ABAPowerFlowData},
)
    nrd = get_network_reduction_data(data)
    arc_ax = get_arc_axis(data)
    n_arcs = length(arc_ax)
    rs = zeros(n_arcs)
    xs = zeros(n_arcs)
    taps = ones(n_arcs)
    shifts = zeros(n_arcs)
    direct_map = PNM.get_direct_branch_map(nrd)
    parallel_map = PNM.get_parallel_branch_map(nrd)
    series_map = PNM.get_series_branch_map(nrd)
    transformer3W_map = PNM.get_transformer3W_map(nrd)
    added_map = PNM.get_added_arc_impedance_map(nrd)
    for (ix, arc) in enumerate(arc_ax)
        if haskey(direct_map, arc)
            branch = direct_map[arc]
            rs[ix] = PSY.get_r(branch)
            xs[ix] = PSY.get_x(branch)
            taps[ix] = _branch_tap(branch)
            shifts[ix] = _branch_shift(branch)
        elseif haskey(parallel_map, arc) || haskey(series_map, arc)
            reduction = haskey(parallel_map, arc) ? parallel_map[arc] : series_map[arc]
            eq = PNM.get_equivalent_physical_branch_parameters(reduction)
            rs[ix] = PNM.get_equivalent_r(eq)
            xs[ix] = PNM.get_equivalent_x(eq)
            taps[ix] = PNM.get_equivalent_tap(eq)
            shifts[ix] = PNM.get_equivalent_shift(eq)
        elseif haskey(transformer3W_map, arc)
            winding = transformer3W_map[arc]
            rs[ix] = PNM.get_equivalent_r(winding)
            xs[ix] = PNM.get_equivalent_x(winding)
            taps[ix] = _winding_tap(winding)
            shifts[ix] = _winding_shift(winding)
        elseif haskey(added_map, arc)
            z = added_map[arc]
            r, x = _added_impedance_rx(z)
            rs[ix] = r
            xs[ix] = x
        else
            error("Arc $arc not found in any branch map.")
        end
    end
    return rs, xs, taps, shifts
end

# No-op for non-ABA data types (AC, PTDF, vPTDF).
_populate_loss_injections!(::PowerFlowData, ::PSY.System) = nothing

"""
    _populate_loss_injections!(data::ABAPowerFlowData, sys::PSY.System)

Compute DCLF-style loss injections from the AC voltage profile stored
in `sys` and add them as withdrawals in `data.bus_active_power_withdrawals`.

For each in-service branch k with series admittance `y_k = g_k + j·b_k`,
the branch losses are computed using complex voltages:

    P_loss_k = g_k · |V_i / tap_k − V_j|²

Losses are withdrawn at the sending-end bus (determined by power flow direction).
This is a single-pass, non-iterative computation.

When the system is flat-start (V=1, θ=0 everywhere), all losses are zero and
the method degenerates to standard lossless DCLF.
"""
function _populate_loss_injections!(data::ABAPowerFlowData, sys::PSY.System)
    if data.loss_factors !== nothing
        @warn "Both loss_approximation_as_injection and calculate_loss_factors are enabled. " *
              "The pre-injected AC-derived losses use a different approximation than " *
              "the PTDF-based loss factors; results may be inconsistent."
    end

    arc_ax = get_arc_axis(data)
    bus_lookup = get_bus_lookup(data)

    # Build bus_number → complex voltage map from the system's current AC state.
    bus_voltage = Dict{Int, ComplexF64}()
    for bus in PSY.get_components(PSY.ACBus, sys)
        V = PSY.get_magnitude(bus)
        θ = PSY.get_angle(bus)
        bus_voltage[PSY.get_number(bus)] = V * exp(1im * θ)
    end

    rs, xs, taps, shifts = _get_arc_branch_params(data)

    for (ix, arc) in enumerate(arc_ax)
        from_bus_no, to_bus_no = arc
        r = rs[ix]
        x = xs[ix]
        tap = taps[ix]
        shift = shifts[ix]

        z = r + x * im
        abs2(z) < eps && continue
        g_k = real(1 / z)

        Vi = bus_voltage[from_bus_no]
        Vj = bus_voltage[to_bus_no]

        # Adjust sending-end voltage for tap and phase shift.
        Vi_shifted = Vi / (tap * exp(1im * shift))

        P_loss = g_k * abs2(Vi_shifted - Vj)

        # Sending end: the bus from which real power flows into the branch.
        from_ix = bus_lookup[from_bus_no]
        to_ix = bus_lookup[to_bus_no]
        sending_ix = imag(Vi * conj(Vj)) >= 0 ? from_ix : to_ix

        # Withdraw losses at sending end.
        @views data.bus_active_power_withdrawals[sending_ix, :] .+= P_loss
    end
    return
end
