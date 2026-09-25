# DCLF-style loss injection
# Pre-compute branch losses from the AC voltage profile and inject at sending-end buses.

"""
    _get_arc_branch_params(data) -> (rs, xs, taps, shifts)

PNM owns the reduction bookkeeping, so each arc resolves through
`PNM.arc_equivalent_branch`, which covers direct branches (including a
`ThreeWindingTransformerCircuit` on its star-point arc), parallel groups, series chains, and
added Ward-equivalent impedances uniformly. This is the single source of truth for per-arc
electrical parameters; [`_get_arc_resistances`](@ref) delegates to it.
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
    for (ix, arc) in enumerate(arc_ax)
        eq = PNM.arc_equivalent_branch(nrd, arc)
        rs[ix] = PNM.get_equivalent_r(eq)
        xs[ix] = PNM.get_equivalent_x(eq)
        taps[ix] = PNM.get_equivalent_tap(eq)
        shifts[ix] = PNM.get_equivalent_shift(eq)
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

"""
    _populate_phase_shift_terms!(data)

Precompute the DC phase-shifter terms: per-arc flow offsets `b_eq·α_eq` and the paired bus
injections (`+b·α` at from, `−b·α` at to). Zero on α-free systems. Computed once at data
construction from the stored circuit angles — mutating a circuit's α afterwards requires
rebuilding the `PowerFlowData` (same staleness contract as the cached arc resistances).
"""
function _populate_phase_shift_terms!(
    data::Union{PTDFPowerFlowData, vPTDFPowerFlowData, ABAPowerFlowData},
)
    nrd = get_network_reduction_data(data)
    bus_lookup = get_bus_lookup(data)
    for (ix, arc) in enumerate(get_arc_axis(data))
        injection = PNM.arc_dc_shift_injection(nrd, arc)
        iszero(injection) && continue
        data.arc_phase_shift_flow_offsets[ix] = injection
        data.bus_phase_shift_injections[bus_lookup[arc[1]]] += injection
        data.bus_phase_shift_injections[bus_lookup[arc[2]]] -= injection
    end
    return
end

_populate_phase_shift_terms!(::PowerFlowData) = nothing
