# Lowering of PSY DC components into a `DCNetwork`, plus the converter physics kernels.
#
# I0 implements the isolated-2-node lowering of `TwoTerminalVSCLine` (point-to-point VSC). The
# multi-terminal lowering (`InterconnectingConverter`/`DCBus`/`TModelHVDCLine`) is added in M2 and
# produces the same `DCNetwork`, so all downstream residual/Jacobian kernels are written once.

# Extract `(a, b, c)` from a converter-loss curve so `P_loss = a + b·I_c + c·I_c²`. Linear curves
# have `c = 0`. Dispatch on the function-data type — no `isa` branching.
function _loss_coefficients(curve::PSY.LinearCurve)
    fd = PSY.get_function_data(curve)
    return (PSY.get_constant_term(fd), PSY.get_proportional_term(fd), 0.0)
end

function _loss_coefficients(curve::PSY.QuadraticCurve)
    fd = PSY.get_function_data(curve)
    return (
        PSY.get_constant_term(fd),
        PSY.get_proportional_term(fd),
        PSY.get_quadratic_term(fd),
    )
end

# Map the PSY per-terminal DC-side / AC-side control enums to a `DCNetwork` control mode.
function _vsc_control_mode(dc_control, ac_control)
    dc_control == PSY.VSCDCControlModes.DC_VOLTAGE_DROOP && return ControlPVdcDroop
    is_dc_voltage = dc_control == PSY.VSCDCControlModes.DC_VOLTAGE
    is_ac_voltage = ac_control == PSY.VSCACControlModes.AC_VOLTAGE
    if is_dc_voltage && is_ac_voltage
        return ControlVdcQ
    elseif is_dc_voltage
        return ControlVdc
    elseif is_ac_voltage
        return ControlPVac
    end
    return ControlPQ
end

# Point-to-point VSC lines eligible for joint AC↔DC modeling: both AC endpoints survive network
# reduction, and the DC link has nonzero conductance. A `g == 0` line is an open DC link, which
# makes `_build_G_dc` singular (no DC path for a DC-power setpoint) and the joint solve infeasible;
# such a line is excluded with a warning and left out of the AC solve (as before VSC support).
function _available_vsc_lines(sys::PSY.System, removed_buses::Set{Int})
    endpoints_kept =
        l ->
            PSY.get_number(PSY.get_from(PSY.get_arc(l))) ∉ removed_buses &&
                PSY.get_number(PSY.get_to(PSY.get_arc(l))) ∉ removed_buses
    lines =
        collect(PSY.get_available_components(endpoints_kept, PSY.TwoTerminalVSCLine, sys))
    for line in lines
        if iszero(PSY.get_g(line))
            @warn "VSC line $(PSY.get_name(line)) has zero DC conductance (g = 0), an open DC " *
                  "link that cannot be solved as a joint AC-DC model; it is ignored in the AC " *
                  "power flow. Set a nonzero `g` to model it jointly."
        end
    end
    return filter(line -> !iszero(PSY.get_g(line)), lines)
end

# AC bus numbers that host a VSC / MTDC converter terminal. Passed to PNM as `irreducible_buses`
# so network reduction never removes a converter's AC bus — a reduced-away terminal would silently
# drop that converter from the joint AC↔DC model. Skips `g == 0` VSC lines (not modeled jointly).
function _dc_converter_ac_buses(sys::PSY.System)
    buses = Set{Int}()
    for l in PSY.get_available_components(PSY.TwoTerminalVSCLine, sys)
        iszero(PSY.get_g(l)) && continue
        arc = PSY.get_arc(l)
        push!(buses, PSY.get_number(PSY.get_from(arc)))
        push!(buses, PSY.get_number(PSY.get_to(arc)))
    end
    for ic in PSY.get_available_components(PSY.InterconnectingConverter, sys)
        push!(buses, PSY.get_number(PSY.get_bus(ic)))
    end
    return buses
end

# Union-find connected components over DC branches; validate each connected DC subnet has at least
# one slack node (or is wholly droop-controlled). Fails fast naming the offending subnet.
function _validate_dc_slacks(dcn::DCNetwork)
    nn = n_dc_nodes(dcn)
    iszero(nn) && return
    parent = collect(1:nn)
    function find(i)
        if parent[i] == i
            return i
        end
        parent[i] = find(parent[i])
        return parent[i]
    end
    for b in 1:n_dc_branches(dcn)
        parent[find(dcn.branch_from[b])] = find(dcn.branch_to[b])
    end
    node_has_droop = falses(nn)
    for c in 1:n_vsc_converters(dcn)
        dcn.converter_mode[c] == ControlPVdcDroop &&
            (node_has_droop[dcn.converter_dc_node_ix[c]] = true)
    end
    comps = Dict{Int, Bool}()  # root → anchored?
    for k in 1:nn
        r = find(k)
        anchored = get(comps, r, false) || dcn.node_is_slack[k] || node_has_droop[k]
        comps[r] = anchored
    end
    for (r, anchored) in comps
        if !anchored
            members = [dcn.node_number[k] for k in 1:nn if find(k) == r]
            error(
                "DC subnet containing DC node(s) $(members) has no V_dc-controlling (slack) or " *
                "droop converter; the DC voltages are undetermined. Set one converter to DC-voltage " *
                "control or droop.",
            )
        end
    end
    return
end

# Build the dense DC nodal conductance matrix from the branch list.
function _build_G_dc(n_node::Int, branch_from::Vector{Int}, branch_to::Vector{Int},
    branch_g::Vector{Float64})
    G = zeros(Float64, n_node, n_node)
    for b in eachindex(branch_from)
        f = branch_from[b]
        t = branch_to[b]
        g = branch_g[b]
        G[f, f] += g
        G[t, t] += g
        G[f, t] -= g
        G[t, f] -= g
    end
    return G
end

"""
    initialize_DCNetwork!(data, sys, bus_lookup, reverse_bus_search_map, removed_buses)

Scan `sys` for DC components, lower them into a single [`DCNetwork`](@ref), and store it on `data`.
Handles point-to-point `TwoTerminalVSCLine` (each → 2 converters + 2 DC nodes + 1 DC branch). A
system with no DC components leaves the empty `DCNetwork()` placeholder in place. The joint AC↔DC
tail model is AC-only; for DC power flow the VSC stays a fixed-power injection (see
`lcc_vsc_fixed_injections!`), so this is a no-op on non-AC data.
"""
function initialize_DCNetwork!(
    ::PowerFlowData,
    ::PSY.System,
    ::Dict{Int, Int},
    ::Dict{Int, Int},
    ::Set{Int},
)
    return
end

# Reactive-power limits: `InterconnectingConverter` carries them as `Union{Nothing, MinMax}`.
_q_limits(x) = (x.min, x.max)
_q_limits(::Nothing) = (-Inf, Inf)

# Mutable accumulator for the converters/nodes/branches discovered while scanning the system.
# Setpoints are scalar (time-invariant) here and expanded to (n_conv, n_time) matrices at the end.
Base.@kwdef struct _DCNetworkBuilder
    ac_bus_ix::Vector{Int} = Int[]
    dc_node_ix::Vector{Int} = Int[]
    mode::Vector{VSCControlMode} = VSCControlMode[]
    ac_bus_number::Vector{Int} = Int[]
    loss_a::Vector{Float64} = Float64[]
    loss_b::Vector{Float64} = Float64[]
    loss_c::Vector{Float64} = Float64[]
    s_max::Vector{Float64} = Float64[]
    q_min::Vector{Float64} = Float64[]
    q_max::Vector{Float64} = Float64[]
    p_min::Vector{Float64} = Float64[]
    p_max::Vector{Float64} = Float64[]
    droop_k::Vector{Float64} = Float64[]
    p_set::Vector{Float64} = Float64[]
    q_set::Vector{Float64} = Float64[]
    vac_set::Vector{Float64} = Float64[]
    vdc_set::Vector{Float64} = Float64[]
    node_is_slack::Vector{Bool} = Bool[]
    node_number::Vector{Int} = Int[]
    branch_from::Vector{Int} = Int[]
    branch_to::Vector{Int} = Int[]
    branch_g::Vector{Float64} = Float64[]
    dc_node_of_number::Dict{Int, Int} = Dict{Int, Int}()
end

function _new_dc_node!(b::_DCNetworkBuilder, number::Int)
    push!(b.node_is_slack, false)
    push!(b.node_number, number)
    return length(b.node_is_slack)
end

# Get (or register) the DC node index for a `DCBus`, keyed by its number so shared buses merge.
function _dc_node!(b::_DCNetworkBuilder, dc_bus)
    number = PSY.get_number(dc_bus)
    return get!(b.dc_node_of_number, number) do
        _new_dc_node!(b, number)
    end
end

# Append one converter; splits the overloaded `dc_set` into `vdc_set`/`p_set` by mode and marks its
# DC node a slack when it pins V_dc.
function _push_converter!(
    b::_DCNetworkBuilder,
    ac_bus_ix::Int,
    ac_bus_number::Int,
    dc_node_ix::Int,
    dc_control,
    ac_control,
    droop::Float64,
    loss_curve,
    s_max::Float64,
    p_lim,
    q_lim,
    dc_set::Float64,
    ac_set::Float64,
    q_set::Float64,
)
    mode = _vsc_control_mode(dc_control, ac_control)
    push!(b.ac_bus_ix, ac_bus_ix)
    push!(b.dc_node_ix, dc_node_ix)
    push!(b.mode, mode)
    push!(b.ac_bus_number, ac_bus_number)
    (la, lb, lc) = _loss_coefficients(loss_curve)
    push!(b.loss_a, la)
    push!(b.loss_b, lb)
    push!(b.loss_c, lc)
    push!(b.s_max, s_max)
    push!(b.p_min, p_lim.min)
    push!(b.p_max, p_lim.max)
    (qmin, qmax) = _q_limits(q_lim)
    push!(b.q_min, qmin)
    push!(b.q_max, qmax)
    push!(b.droop_k, droop)
    push!(b.q_set, q_set)
    push!(b.vac_set, ac_set)
    if uses_vdc_setpoint(mode)
        push!(b.vdc_set, dc_set)
        push!(b.p_set, 0.0)
    else
        push!(b.vdc_set, 1.0)
        push!(b.p_set, dc_set)
    end
    fixes_dc_voltage(mode) && (b.node_is_slack[dc_node_ix] = true)
    return length(b.ac_bus_ix)
end

# Lower point-to-point `TwoTerminalVSCLine`: 2 implicit DC nodes + 2 converters + 1 DC branch.
function _lower_vsc_lines!(b::_DCNetworkBuilder, lines, bus_lookup, reverse_bus_search_map)
    for line in lines
        arc = PSY.get_arc(line)
        from_number = PSY.get_number(PSY.get_from(arc))
        to_number = PSY.get_number(PSY.get_to(arc))
        from_ix = _get_bus_ix(bus_lookup, reverse_bus_search_map, from_number)
        to_ix = _get_bus_ix(bus_lookup, reverse_bus_search_map, to_number)
        nf = _new_dc_node!(b, -1)
        nt = _new_dc_node!(b, -1)
        _push_converter!(
            b, from_ix, from_number, nf,
            PSY.get_dc_control_from(line), PSY.get_ac_control_from(line),
            PSY.get_dc_voltage_droop_from(line), PSY.get_converter_loss_from(line),
            PSY.get_rating_from(line), PSY.get_active_power_limits_from(line),
            PSY.get_reactive_power_limits_from(line), PSY.get_dc_setpoint_from(line),
            PSY.get_ac_setpoint_from(line), PSY.get_reactive_power_from(line),
        )
        _push_converter!(
            b, to_ix, to_number, nt,
            PSY.get_dc_control_to(line), PSY.get_ac_control_to(line),
            PSY.get_dc_voltage_droop_to(line), PSY.get_converter_loss_to(line),
            PSY.get_rating_to(line), PSY.get_active_power_limits_to(line),
            PSY.get_reactive_power_limits_to(line), PSY.get_dc_setpoint_to(line),
            PSY.get_ac_setpoint_to(line), PSY.get_reactive_power_to(line),
        )
        push!(b.branch_from, nf)
        push!(b.branch_to, nt)
        push!(b.branch_g, PSY.get_g(line))
    end
    return
end

# Lower the multi-terminal DC model: `InterconnectingConverter` (AC↔DC) on `DCBus` nodes joined by
# `TModelHVDCLine` DC branches.
function _lower_mtdc!(
    b::_DCNetworkBuilder,
    sys,
    bus_lookup,
    reverse_bus_search_map,
    removed_buses,
)
    for ic in PSY.get_available_components(PSY.InterconnectingConverter, sys)
        bus_number = PSY.get_number(PSY.get_bus(ic))
        bus_number in removed_buses && continue
        ac_ix = _get_bus_ix(bus_lookup, reverse_bus_search_map, bus_number)
        node = _dc_node!(b, PSY.get_dc_bus(ic))
        _push_converter!(
            b, ac_ix, bus_number, node,
            PSY.get_dc_control(ic), PSY.get_ac_control(ic),
            PSY.get_dc_voltage_droop(ic), PSY.get_loss_function(ic),
            PSY.get_rating(ic), PSY.get_active_power_limits(ic),
            PSY.get_reactive_power_limits(ic), PSY.get_dc_setpoint(ic),
            PSY.get_ac_setpoint(ic), 0.0,
        )
    end
    for dcline in PSY.get_available_components(PSY.TModelHVDCLine, sys)
        arc = PSY.get_arc(dcline)
        nf = _dc_node!(b, PSY.get_from(arc))
        nt = _dc_node!(b, PSY.get_to(arc))
        r = PSY.get_r(dcline)
        # steady-state DC: only series resistance matters (l, c dropped). A zero-resistance line
        # is a hard short; fall back to a large conductance.
        if iszero(r)
            g = 1.0e6
        else
            g = 1.0 / r
        end
        push!(b.branch_from, nf)
        push!(b.branch_to, nt)
        push!(b.branch_g, g)
    end
    return
end

function initialize_DCNetwork!(
    data::ACPowerFlowData,
    sys::PSY.System,
    bus_lookup::Dict{Int, Int},
    reverse_bus_search_map::Dict{Int, Int},
    removed_buses::Set{Int},
)
    # A system that has a DC network should model it, so this is ON by default — the DC equipment is
    # solved as part of the power flow. A sequential decoupled DC warm-start (see
    # `_vsc_warm_start!`) seeds the joint AC↔DC Newton for robustness. The escape hatch
    # `solver_settings = Dict(:model_dc_network => false)` restores the historical behavior (DC
    # components ignored in the AC solve, kept as fixed injections only on the DC path).
    get(get_solver_kwargs(data.pf), :model_dc_network, true) || return

    vsc_lines = _available_vsc_lines(sys, removed_buses)
    # Count only converters whose AC bus survives network reduction: if reduction removed every
    # interconnecting converter, the multi-terminal DC grid has no AC interface left and is ignored
    # entirely (rather than lowering converter-less DC lines into an unanchored subnet).
    has_ic = any(
        ic -> PSY.get_number(PSY.get_bus(ic)) ∉ removed_buses,
        PSY.get_available_components(PSY.InterconnectingConverter, sys),
    )
    (isempty(vsc_lines) && !has_ic) && return

    b = _DCNetworkBuilder()
    _lower_vsc_lines!(b, vsc_lines, bus_lookup, reverse_bus_search_map)
    has_ic && _lower_mtdc!(b, sys, bus_lookup, reverse_bus_search_map, removed_buses)

    n_time = size(data.bus_active_power_injections, 2)
    n_conv = length(b.ac_bus_ix)
    n_node = length(b.node_is_slack)
    expand(v) = repeat(reshape(v, :, 1), 1, n_time)
    p_set = expand(b.p_set)
    q_set = expand(b.q_set)
    vac_set = expand(b.vac_set)
    vdc_set = expand(b.vdc_set)

    # Seed each DC node's V_dc at a Vdc/droop converter's target if one sits on it, else 1.0.
    # Gate on the mode (power-control converters carry a placeholder vdc_set = 1.0 and must not seed).
    node_vdc = ones(Float64, n_node, n_time)
    for c in 1:n_conv
        if uses_vdc_setpoint(b.mode[c])
            node_vdc[b.dc_node_ix[c], :] .= b.vdc_set[c]
        end
    end

    G_dc = _build_G_dc(n_node, b.branch_from, b.branch_to, b.branch_g)

    dcn = DCNetwork(;
        converter_ac_bus_ix = b.ac_bus_ix,
        converter_dc_node_ix = b.dc_node_ix,
        converter_mode = b.mode,
        converter_ac_bus_number = b.ac_bus_number,
        loss_a = b.loss_a,
        loss_b = b.loss_b,
        loss_c = b.loss_c,
        s_max = b.s_max,
        q_min = b.q_min,
        q_max = b.q_max,
        p_min = b.p_min,
        p_max = b.p_max,
        droop_k = b.droop_k,
        p_set,
        q_set,
        vac_set,
        vdc_set,
        p_c = copy(p_set),
        q_c = copy(q_set),
        node_is_slack = b.node_is_slack,
        node_number = b.node_number,
        node_vdc,
        branch_from = b.branch_from,
        branch_to = b.branch_to,
        branch_g = b.branch_g,
        G_dc,
    )
    _validate_dc_slacks(dcn)
    data.dc_network[] = dcn
    # Seed converter/DC-node states with a sequential decoupled DC solve (AC voltages held fixed) so
    # the joint AC↔DC Newton starts from a consistent DC operating point: the robustness of a
    # sequential AC–DC initializer, without giving up the single-Jacobian solve.
    for t in 1:n_time
        _vsc_warm_start!(dcn, view(data.bus_magnitude, :, t), t)
    end
    return
end

# DC-KCL residual block: writes I_dc = G_dc * V_dc into F[base+1 : base+nnode].
function _dc_kcl_residual!(
    F::AbstractVector{Float64},
    base::Int,
    dcn::DCNetwork,
    nnode::Int,
    time_step::Int,
)
    G = dcn.G_dc
    @inbounds for k in 1:nnode
        acc = 0.0
        for jnode in 1:nnode
            acc += G[k, jnode] * dcn.node_vdc[jnode, time_step]
        end
        F[base + k] = acc
    end
    return
end

# Warm-start residual for the VSC tail with AC voltages FIXED: per converter the active/Vdc control
# row `r1` and a reactive pin `Q_c − q_set` (the true AC-voltage control rows constrain AC voltage,
# which is fixed here, so they are replaced by the Q pin for the initializer); per DC node the DC-KCL
# row. `y`-ordering: [(P_c, Q_c) per converter; V_dc per node].
function _vsc_warm_residual!(
    F::Vector{Float64},
    dcn::DCNetwork,
    Vm::AbstractVector{Float64},
    time_step::Int,
)
    nconv = n_vsc_converters(dcn)
    nnode = n_dc_nodes(dcn)
    @inbounds for c in 1:nconv
        node = dcn.converter_dc_node_ix[c]
        mode = dcn.converter_mode[c]
        P = dcn.p_c[c, time_step]
        Vdc = dcn.node_vdc[node, time_step]
        F[2 * c - 1] = _vsc_r1(mode, dcn, c, P, Vdc, time_step)
        F[2 * c] = dcn.q_c[c, time_step] - dcn.q_set[c, time_step]
    end
    base = 2 * nconv
    _dc_kcl_residual!(F, base, dcn, nnode, time_step)
    @inbounds for c in 1:nconv
        node = dcn.converter_dc_node_ix[c]
        ix = dcn.converter_ac_bus_ix[c]
        Vdc = dcn.node_vdc[node, time_step]
        F[base + node] += _vsc_pdc(dcn, c, Vm[ix], time_step) / Vdc
    end
    return
end

# Dense tail×tail Jacobian of `_vsc_warm_residual!` (AC voltages fixed).
function _vsc_warm_jacobian!(
    J::Matrix{Float64},
    dcn::DCNetwork,
    Vm::AbstractVector{Float64},
    time_step::Int,
)
    nconv = n_vsc_converters(dcn)
    nnode = n_dc_nodes(dcn)
    base = 2 * nconv
    fill!(J, 0.0)
    G = dcn.G_dc
    @inbounds for k in 1:nnode, j in 1:nnode
        J[base + k, base + j] += G[k, j]
    end
    @inbounds for c in 1:nconv
        node = dcn.converter_dc_node_ix[c]
        ix = dcn.converter_ac_bus_ix[c]
        mode = dcn.converter_mode[c]
        pc = 2 * c - 1
        qc = 2 * c
        vk = base + node
        Vdc = dcn.node_vdc[node, time_step]
        J[pc, pc] = _vsc_dr1_dP(mode, dcn, c)
        J[pc, vk] = _vsc_dr1_dVdc(mode)
        J[qc, qc] = 1.0
        (Pdc, dP, dQ, _) = _vsc_pdc_derivatives(dcn, c, Vm[ix], time_step)
        J[vk, pc] += dP / Vdc
        J[vk, qc] += dQ / Vdc
        J[vk, vk] += -Pdc / (Vdc * Vdc)
    end
    return
end

# Sequential decoupled DC initializer: Newton-solve the VSC tail (converter P_c,Q_c + node V_dc) for
# fixed AC voltages, writing the result into the network's state mirrors. Used to seed the joint
# AC↔DC solve. Best-effort — a partial solve still improves the starting point.
function _vsc_warm_start!(
    dcn::DCNetwork,
    Vm::AbstractVector{Float64},
    time_step::Int;
    max_iter::Int = 30,
    tol::Float64 = 1.0e-10,
)
    nconv = n_vsc_converters(dcn)
    nnode = n_dc_nodes(dcn)
    n = 2 * nconv + nnode
    iszero(n) && return
    y = Vector{Float64}(undef, n)
    @inbounds for c in 1:nconv
        y[2 * c - 1] = dcn.p_c[c, time_step]
        y[2 * c] = dcn.q_c[c, time_step]
    end
    @inbounds for k in 1:nnode
        y[2 * nconv + k] = dcn.node_vdc[k, time_step]
    end
    F = Vector{Float64}(undef, n)
    J = Matrix{Float64}(undef, n, n)
    function write_back!(z)
        @inbounds for c in 1:nconv
            dcn.p_c[c, time_step] = z[2 * c - 1]
            dcn.q_c[c, time_step] = z[2 * c]
        end
        @inbounds for k in 1:nnode
            dcn.node_vdc[k, time_step] = z[2 * nconv + k]
        end
        return
    end
    for _ in 1:max_iter
        write_back!(y)
        _vsc_warm_residual!(F, dcn, Vm, time_step)
        norm(F) < tol && break
        _vsc_warm_jacobian!(J, dcn, Vm, time_step)
        # The warm-start is best-effort: a singular/ill-conditioned tail must never abort the solve,
        # so on any linear-solve failure we keep the best seed so far and let the joint Newton run.
        local Δ
        try
            Δ = J \ F
        catch e
            e isa LinearAlgebra.SingularException || rethrow()
            break
        end
        all(isfinite, Δ) || break
        y .-= Δ
    end
    write_back!(y)
    return
end

# ──────────────────────────────────────────────────────────────────────────────────────────────
# Converter physics kernels (shared across formulations; polar uses |V_ac| = bus_magnitude).
#
# Sign convention: P_c is the active power the converter INJECTS into its AC bus. The converter
# draws `P_dc = P_c + P_loss` from its DC node, so the DC current it injects into the node is
# −P_dc/V_dc. DC-KCL at node k: Σ_j G_dc[k,j]·V_dc[j] + Σ_{c on k} P_dc_c/V_dc[k] = 0.
# ──────────────────────────────────────────────────────────────────────────────────────────────

# DC power drawn from the node by converter `c` (= AC injection + converter losses).
function _vsc_pdc(dcn::DCNetwork, c::Int, Vm_ac::Float64, time_step::Int)
    P = dcn.p_c[c, time_step]
    Q = dcn.q_c[c, time_step]
    Ic = sqrt(max(P * P + Q * Q, V_FLOOR2)) / sqrt(max(Vm_ac * Vm_ac, V_FLOOR2))
    Ploss = dcn.loss_a[c] + dcn.loss_b[c] * Ic + dcn.loss_c[c] * Ic * Ic
    return P + Ploss
end

# First control row (active-power / V_dc), branching on the control mode.
function _vsc_r1(mode::VSCControlMode, dcn, c, P, Vdc, t)
    if mode == ControlVdc || mode == ControlVdcQ
        return Vdc - dcn.vdc_set[c, t]
    elseif mode == ControlPVdcDroop
        return (Vdc - dcn.vdc_set[c, t]) + dcn.droop_k[c] * P
    end
    return P - dcn.p_set[c, t]  # ControlPQ, ControlPVac
end

# Second control row (reactive power / |V_ac|²). The AC-voltage row uses raw |V_ac|² (floor-free,
# like the rectangular PV pin) for a consistent derivative.
function _vsc_r2(mode::VSCControlMode, dcn, c, Q, Vac, t)
    if controls_ac_voltage(mode)
        return Vac * Vac - dcn.vac_set[c, t]^2
    end
    return Q - dcn.q_set[c, t]
end

# Read the VSC tail of the state vector into the network's solved-state mirrors.
function _read_vsc_state!(dcn::DCNetwork, x::Vector{Float64}, vsc_off::Int, time_step::Int)
    nconv = n_vsc_converters(dcn)
    @inbounds for c in 1:nconv
        dcn.p_c[c, time_step] = x[vsc_off + 2 * c - 1]
        dcn.q_c[c, time_step] = x[vsc_off + 2 * c]
    end
    @inbounds for k in 1:n_dc_nodes(dcn)
        dcn.node_vdc[k, time_step] = x[vsc_off + 2 * nconv + k]
    end
    return
end

# Add each converter's (P_c, Q_c) AC injection to the polar bus power-balance rows.
function _apply_vsc_bus_injections_polar!(
    F::Vector{Float64},
    dcn::DCNetwork,
    time_step::Int,
)
    @inbounds for c in 1:n_vsc_converters(dcn)
        ix = dcn.converter_ac_bus_ix[c]
        F[2 * ix - 1] -= dcn.p_c[c, time_step]
        F[2 * ix] -= dcn.q_c[c, time_step]
    end
    return
end

# Write the VSC tail residual rows: 2 control rows per converter, then 1 DC-KCL row per DC node.
# `Vm` is the per-bus voltage magnitude (polar). `vsc_off` is the index just before the VSC tail.
function _set_vsc_tail_residuals!(
    F::Vector{Float64},
    dcn::DCNetwork,
    Vm::AbstractVector{Float64},
    vsc_off::Int,
    time_step::Int,
)
    nconv = n_vsc_converters(dcn)
    nnode = n_dc_nodes(dcn)
    @inbounds for c in 1:nconv
        ix = dcn.converter_ac_bus_ix[c]
        node = dcn.converter_dc_node_ix[c]
        mode = dcn.converter_mode[c]
        P = dcn.p_c[c, time_step]
        Q = dcn.q_c[c, time_step]
        Vdc = dcn.node_vdc[node, time_step]
        F[vsc_off + 2 * c - 1] = _vsc_r1(mode, dcn, c, P, Vdc, time_step)
        F[vsc_off + 2 * c] = _vsc_r2(mode, dcn, c, Q, Vm[ix], time_step)
    end
    base = vsc_off + 2 * nconv
    _dc_kcl_residual!(F, base, dcn, nnode, time_step)
    @inbounds for c in 1:nconv
        node = dcn.converter_dc_node_ix[c]
        ix = dcn.converter_ac_bus_ix[c]
        Vdc = dcn.node_vdc[node, time_step]
        F[base + node] += _vsc_pdc(dcn, c, Vm[ix], time_step) / Vdc
    end
    return
end

# ── Jacobian derivative helpers (branch on the control mode) ──────────────────────────────────
#
# Each converter contributes two control rows: r1 (active-power / V_dc, see `_vsc_r1`) and r2
# (reactive-power / |V_ac|², see `_vsc_r2`). The helpers below return the partials of r1/r2 (and of
# the DC-side loss term P_dc) w.r.t. the states (P_c, Q_c, V_dc, |V_ac|); `_set_entries_for_vsc`
# places them in the Jacobian. The values depend only on the control mode (plus |V_ac| for the
# AC-voltage and loss terms), so each helper is a small branch on the mode, not method dispatch.

# ∂r1/∂P_c: 1 for the P-control modes, the droop gain for droop, 0 for the V_dc-control modes.
function _vsc_dr1_dP(mode::VSCControlMode, dcn, c)
    if mode == ControlVdc || mode == ControlVdcQ
        return 0.0
    elseif mode == ControlPVdcDroop
        return dcn.droop_k[c]
    end
    return 1.0  # ControlPQ, ControlPVac
end

# ∂r1/∂V_dc: 1 for the V_dc-control and droop modes, 0 otherwise.
function _vsc_dr1_dVdc(mode::VSCControlMode)
    if fixes_dc_voltage(mode) || mode == ControlPVdcDroop
        return 1.0
    end
    return 0.0
end

# ∂r2/∂Q_c: 0 when the row pins AC voltage, else 1.
function _vsc_dr2_dQ(mode::VSCControlMode)
    if controls_ac_voltage(mode)
        return 0.0
    end
    return 1.0
end

# ∂r2/∂|V_ac|: 2·|V_ac| when the row pins AC voltage, else 0.
function _vsc_dr2_dVm(mode::VSCControlMode, Vm)
    if controls_ac_voltage(mode)
        return 2.0 * Vm
    end
    return 0.0
end

# P_dc and its derivatives w.r.t. (P_c, Q_c, |V_ac|). Returns `(P_dc, ∂/∂P_c, ∂/∂Q_c, ∂/∂|V_ac|)`;
# P_dc matches `_vsc_pdc`, computed here from the same `I_c` so the Jacobian path needs one call, not
# two. Off-P derivatives are nonzero only when the converter has losses.
function _vsc_pdc_derivatives(dcn::DCNetwork, c::Int, Vm_ac::Float64, time_step::Int)
    P = dcn.p_c[c, time_step]
    Q = dcn.q_c[c, time_step]
    Vmf = sqrt(max(Vm_ac * Vm_ac, V_FLOOR2))
    # Floor S² with V_FLOOR2 (mirroring |V_ac|²) so the apparent-power magnitude never reaches the
    # P/S, Q/S singularity at S = 0; at the floor the derivatives go smoothly to ~0.
    S = sqrt(max(P * P + Q * Q, V_FLOOR2))
    Ic = S / Vmf
    Ploss = dcn.loss_a[c] + dcn.loss_b[c] * Ic + dcn.loss_c[c] * Ic * Ic
    dloss_dIc = dcn.loss_b[c] + 2.0 * dcn.loss_c[c] * Ic
    dIc_dP = (P / S) / Vmf
    dIc_dQ = (Q / S) / Vmf
    dIc_dVm = -Ic / Vmf
    return (P + Ploss, 1.0 + dloss_dIc * dIc_dP, dloss_dIc * dIc_dQ, dloss_dIc * dIc_dVm)
end

# ── Rectangular / MCPB kernels (bus balance is current injection conj(S/V)) ───────────────────

# Add each converter's current injection to the bus current-balance rows (real, imag). Mirrors the
# per-bus I_spec term `(P·e + Q·f)/D, (P·f − Q·e)/D` used for ordinary injections.
function _apply_vsc_bus_injections_rect!(
    F::Vector{Float64},
    dcn::DCNetwork,
    e_state::Vector{Float64},
    f_state::Vector{Float64},
    bus_state_offset::AbstractVector,
    time_step::Int,
)
    @inbounds for c in 1:n_vsc_converters(dcn)
        ix = dcn.converter_ac_bus_ix[c]
        off = Int(bus_state_offset[ix])
        e = e_state[ix]
        f = f_state[ix]
        D = max(e * e + f * f, V_FLOOR2)
        P = dcn.p_c[c, time_step]
        Q = dcn.q_c[c, time_step]
        F[off] += (P * e + Q * f) / D
        F[off + 1] += (P * f - Q * e) / D
    end
    return
end

# VSC tail residual rows for rectangular/MCPB: identical layout to polar, but |V_ac| = √(e²+f²).
function _set_vsc_tail_residuals_rect!(
    F::Vector{Float64},
    dcn::DCNetwork,
    e_state::Vector{Float64},
    f_state::Vector{Float64},
    vsc_off::Int,
    time_step::Int,
)
    nconv = n_vsc_converters(dcn)
    nnode = n_dc_nodes(dcn)
    @inbounds for c in 1:nconv
        ix = dcn.converter_ac_bus_ix[c]
        node = dcn.converter_dc_node_ix[c]
        mode = dcn.converter_mode[c]
        Vm = sqrt(e_state[ix]^2 + f_state[ix]^2)
        P = dcn.p_c[c, time_step]
        Q = dcn.q_c[c, time_step]
        Vdc = dcn.node_vdc[node, time_step]
        F[vsc_off + 2 * c - 1] = _vsc_r1(mode, dcn, c, P, Vdc, time_step)
        F[vsc_off + 2 * c] = _vsc_r2(mode, dcn, c, Q, Vm, time_step)
    end
    base = vsc_off + 2 * nconv
    _dc_kcl_residual!(F, base, dcn, nnode, time_step)
    @inbounds for c in 1:nconv
        node = dcn.converter_dc_node_ix[c]
        ix = dcn.converter_ac_bus_ix[c]
        Vm = sqrt(e_state[ix]^2 + f_state[ix]^2)
        Vdc = dcn.node_vdc[node, time_step]
        F[base + node] += _vsc_pdc(dcn, c, Vm, time_step) / Vdc
    end
    return
end

# MCPB bus injection: like rect but PQ buses use imag-first row order (the two current slots
# swapped). REF keeps rect ordering. (VSC converters are expected on PQ buses.)
function _apply_vsc_bus_injections_mixed!(
    F::Vector{Float64},
    dcn::DCNetwork,
    e_state::Vector{Float64},
    f_state::Vector{Float64},
    bus_state_offset::AbstractVector,
    bus_types::AbstractVector,
    time_step::Int,
)
    @inbounds for c in 1:n_vsc_converters(dcn)
        ix = dcn.converter_ac_bus_ix[c]
        off = Int(bus_state_offset[ix])
        e = e_state[ix]
        f = f_state[ix]
        D = max(e * e + f * f, V_FLOOR2)
        P = dcn.p_c[c, time_step]
        Q = dcn.q_c[c, time_step]
        Ir = (P * e + Q * f) / D
        Ii = (P * f - Q * e) / D
        if bus_types[ix] == PSY.ACBusTypes.PQ
            F[off] += Ii
            F[off + 1] += Ir
        else
            F[off] += Ir
            F[off + 1] += Ii
        end
    end
    return
end

# Shared VSC tail Jacobian writer for the rectangular-CI and MCPB formulations. The two are
# identical except that MCPB uses the imag-first slot order for the bus current-injection rows at
# PQ buses (`imag_first_pq = true`); rectangular CI always uses real-first (`imag_first_pq = false`).
# The control + DC-KCL tail rows (and their e,f columns) are the same for both. All writes go through
# the pre-built `vsc_nz` nzval-index cache (and `diag_base_nz` for the current-injection coupling), so
# the hot path is `O(n_conv + n_node + n_branch)` with no `O(log nnz)` `Jv[r,c]` setindex.
function _set_entries_for_vsc_rect_mcpb!(
    Jvnz::Vector{Float64},
    diag_base_nz::Matrix{Int},
    vsc_nz::VSCJacobianNZCache,
    dcn::DCNetwork,
    e_state::Vector{Float64},
    f_state::Vector{Float64},
    bus_types::AbstractVector,
    time_step::Int,
    imag_first_pq::Bool,
)
    nconv = n_vsc_converters(dcn)
    nnode = n_dc_nodes(dcn)
    G = dcn.G_dc
    conv = vsc_nz.conv
    node = vsc_nz.node
    branch = vsc_nz.branch
    @inbounds for b in 1:n_dc_branches(dcn)
        f = dcn.branch_from[b]
        t = dcn.branch_to[b]
        Jvnz[branch[2 * b - 1]] = G[f, t]
        Jvnz[branch[2 * b]] = G[t, f]
    end
    @inbounds for k in 1:nnode
        Jvnz[node[k]] = G[k, k]
    end
    @inbounds for c in 1:nconv
        ix = dcn.converter_ac_bus_ix[c]
        k = dcn.converter_dc_node_ix[c]
        mode = dcn.converter_mode[c]
        e = e_state[ix]
        f = f_state[ix]
        D = max(e * e + f * f, V_FLOOR2)
        Vm = sqrt(D)
        P = dcn.p_c[c, time_step]
        Q = dcn.q_c[c, time_step]
        Vdc = dcn.node_vdc[k, time_step]
        num_r = P * e + Q * f
        num_i = P * f - Q * e
        D2 = D * D
        dIr_de = (P * D - num_r * 2.0 * e) / D2
        dIr_df = (Q * D - num_r * 2.0 * f) / D2
        dIi_de = (-Q * D - num_i * 2.0 * e) / D2
        dIi_df = (P * D - num_i * 2.0 * f) / D2
        if imag_first_pq && bus_types[ix] == PSY.ACBusTypes.PQ
            # imag-first (MCPB PQ): F[off] = Ii, F[off+1] = Ir
            Jvnz[diag_base_nz[1, ix]] += dIi_de
            Jvnz[diag_base_nz[2, ix]] += dIi_df
            Jvnz[diag_base_nz[3, ix]] += dIr_de
            Jvnz[diag_base_nz[4, ix]] += dIr_df
            Jvnz[conv[1, c]] = f / D
            Jvnz[conv[2, c]] = -e / D
            Jvnz[conv[3, c]] = e / D
            Jvnz[conv[4, c]] = f / D
        else
            # real-first: F[off] = Ir, F[off+1] = Ii
            Jvnz[diag_base_nz[1, ix]] += dIr_de
            Jvnz[diag_base_nz[2, ix]] += dIr_df
            Jvnz[diag_base_nz[3, ix]] += dIi_de
            Jvnz[diag_base_nz[4, ix]] += dIi_df
            Jvnz[conv[1, c]] = e / D
            Jvnz[conv[2, c]] = f / D
            Jvnz[conv[3, c]] = f / D
            Jvnz[conv[4, c]] = -e / D
        end
        # control + DC-KCL tail rows (columns are bus e,f states — no imag-first swap)
        Jvnz[conv[5, c]] = _vsc_dr1_dP(mode, dcn, c)
        Jvnz[conv[6, c]] = _vsc_dr1_dVdc(mode)
        Jvnz[conv[7, c]] = _vsc_dr2_dQ(mode)
        if controls_ac_voltage(mode)
            Jvnz[conv[8, c]] = 2.0 * e
            Jvnz[conv[9, c]] = 2.0 * f
        else
            Jvnz[conv[8, c]] = 0.0
            Jvnz[conv[9, c]] = 0.0
        end
        (Pdc, dP, dQ, dVm) = _vsc_pdc_derivatives(dcn, c, Vm, time_step)
        Jvnz[conv[10, c]] = dP / Vdc
        Jvnz[conv[11, c]] = dQ / Vdc
        Jvnz[conv[12, c]] = (dVm * e / Vm) / Vdc
        Jvnz[conv[13, c]] = (dVm * f / Vm) / Vdc
        Jvnz[node[k]] += -Pdc / (Vdc * Vdc)
    end
    return
end
