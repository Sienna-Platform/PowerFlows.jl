"""
    struct ACRectangularCIResidual

Residual functor for the rectangular AC power flow: bus power mismatch in Cartesian
coordinates (MATPOWER's `newtonpf_S_cart`). Mirrors [`ACPowerFlowResidual`](@ref) but with
`(e, f)` voltage states. Every bus has a 2-slot block: PQ `(e, f)` with rows
`(ΔP, ΔQ)`; PV `(e, f)` with rows `(ΔP, |V|² − V_set²)`; REF `(P_net, Q_net)` with rows
`(ΔP, ΔQ)` at its fixed voltage. A PV bus's reactive power is not a state: it is recovered
from the converged voltages ([`rect_finalize_bus_injections!`](@ref)).

`ΔS = V·conj(I) − S_spec`, with `I = Y_bus_eff·V` plus the LCC terminal currents. Power
mismatch is unchanged by a common rotation of all bus angles, so Newton converges from a flat
start like the polar form; a current mismatch is not, and fails once the solution's angle spread
passes about 0.6 rad.

# Fields
- `Rv::Vector{Float64}` — residual values, length `total_bus_state + tail`
- `Y_bus_eff::SparseMatrixCSC{ComplexF64, Int}` — Y_bus with ZIP constant-Z folded in
- `P_net_const::Vector{Float64}` — constant-power net injection (no |V| dependence)
- `Q_net_const::Vector{Float64}` — constant-power net reactive injection
- `const_I_P::Vector{Float64}` — constant-current P-withdrawal coefficient per bus
- `const_I_Q::Vector{Float64}` — constant-current Q-withdrawal coefficient per bus
- `P_net_set::Vector{Float64}` — initial P_net for distributed-slack delta computation
- `bus_slack_participation_factors::SparseVector{Float64, Int}`
- `subnetworks::Dict{Int64, Vector{Int64}}`
- `independent_ref::Set{Int}` — REF buses that share an island with another REF
  (multi-swing); precomputed once here (bus REF-status is fixed across a solve)
  so the hot per-iteration path never allocates a `Set`.
- `bus_state_offset::Vector{REC_INDEX_TYPE}`
- `bus_block_size::Vector{Int8}`
- `total_bus_state::Int`
- `validate_offsets::Vector{Int}` — precomputed `x`-offsets of PQ/PV buses for
  the per-iteration voltage-magnitude diagnostic
- `e_state`, `f_state` — per-bus `(e, f)` of the last evaluation (REF from its fixed
  voltage); `data.bus_magnitude` holds `V_set` at PV buses, not `|V_state|`.
- `P_eff_cache`, `Q_eff_cache` — per-bus specified net injection of the last evaluation
- `Ir_acc`, `Ii_acc` — per-bus network current `Re/Im(Y_bus_eff·V + Y_lcc·V)` of the last
  evaluation, read by the Jacobian and by [`rect_finalize_bus_injections!`](@ref)
"""
struct ACRectangularCIResidual
    Rv::Vector{Float64}
    Y_bus_eff::SparseMatrixCSC{ComplexF64, Int}
    P_net_const::Vector{Float64}
    Q_net_const::Vector{Float64}
    const_I_P::Vector{Float64}
    const_I_Q::Vector{Float64}
    P_net_set::Vector{Float64}
    bus_slack_participation_factors::SparseVector{Float64, Int}
    subnetworks::Dict{Int64, Vector{Int64}}
    independent_ref::Set{Int}
    bus_state_offset::Vector{REC_INDEX_TYPE}
    bus_block_size::Vector{Int8}
    total_bus_state::Int
    validate_offsets::Vector{Int}
    e_state::Vector{Float64}
    f_state::Vector{Float64}
    P_eff_cache::Vector{Float64}
    Q_eff_cache::Vector{Float64}
    Ir_acc::Vector{Float64}
    Ii_acc::Vector{Float64}
end

function ACRectangularCIResidual(data::ACPowerFlowData, time_step::Int64)
    n_buses = first(size(data.bus_type))
    n_lccs = size(data.lcc.p_set, 1)
    bus_type = view(data.bus_type, :, time_step)

    offsets, block_sizes, total_bus_state = compute_bus_state_offsets(bus_type)
    validate_offsets = _pqpv_validate_offsets(bus_type, offsets)
    total_state = total_bus_state + state_tail_length(data, get_dc_network(data))

    P_net_const = Vector{Float64}(undef, n_buses)
    Q_net_const = Vector{Float64}(undef, n_buses)
    const_I_P = Vector{Float64}(undef, n_buses)
    const_I_Q = Vector{Float64}(undef, n_buses)
    P_net_set = Vector{Float64}(undef, n_buses)

    subnetworks =
        _find_subnetworks_for_reference_buses(data.power_network_matrix.data, bus_type)
    # REF status is fixed for the life of a solve, so this is computed once here
    # rather than per-iteration (see the `independent_ref` field docstring).
    independent_ref = _multi_swing_ref_indices(data.bus_type, subnetworks, time_step)

    for ix in 1:n_buses
        # Constant-power net injection (no |V| dependence)
        P_net_const[ix] =
            data.bus_active_power_injections[ix, time_step] -
            data.bus_active_power_withdrawals[ix, time_step] +
            data.bus_hvdc_net_power[ix, time_step]
        Q_net_const[ix] =
            data.bus_reactive_power_injections[ix, time_step] -
            data.bus_reactive_power_withdrawals[ix, time_step]
        # ZIP constant-current coefficients (carried as withdrawals)
        const_I_P[ix] =
            data.bus_active_power_constant_current_withdrawals[ix, time_step]
        const_I_Q[ix] =
            data.bus_reactive_power_constant_current_withdrawals[ix, time_step]
        # P_net_set tracks the initial P injection at setup (for slack delta)
        P_net_set[ix] = P_net_const[ix] -
                        const_I_P[ix] * data.bus_magnitude[ix, time_step]
    end

    bus_slack_participation_factors =
        _build_bus_slack_participation_factors(data, bus_type, subnetworks, time_step)

    # Build Y_bus_eff: copy Y_bus + fold constant-Z ZIP loads
    Y = data.power_network_matrix.data
    Y_bus_eff = SparseArrays.sparse(ComplexF64.(Y))
    fold_zip_constant_z!(Y_bus_eff, data, time_step)

    return ACRectangularCIResidual(
        Vector{Float64}(undef, total_state),
        Y_bus_eff,
        P_net_const,
        Q_net_const,
        const_I_P,
        const_I_Q,
        P_net_set,
        bus_slack_participation_factors,
        subnetworks,
        independent_ref,
        offsets,
        block_sizes,
        total_bus_state,
        validate_offsets,
        Vector{Float64}(undef, n_buses),
        Vector{Float64}(undef, n_buses),
        Vector{Float64}(undef, n_buses),
        Vector{Float64}(undef, n_buses),
        Vector{Float64}(undef, n_buses),
        Vector{Float64}(undef, n_buses),
    )
end

function (R::ACRectangularCIResidual)(
    data::ACPowerFlowData,
    Rv::Vector{Float64},
    x::Vector{Float64},
    time_step::Int64,
)
    _update_rect_ci_residual_values!(R, x, data, time_step)
    copyto!(Rv, R.Rv)
    return
end

function (R::ACRectangularCIResidual)(
    data::ACPowerFlowData,
    x::Vector{Float64},
    time_step::Int64,
)
    _update_rect_ci_residual_values!(R, x, data, time_step)
    return
end

"""
Evaluate the rectangular power-mismatch residual `R.Rv` at `x`.

Walks `Y_bus_eff` once to accumulate the network current `Y·V` (plus the LCC terminal
currents) into `R.Ir_acc`/`R.Ii_acc`, then forms per bus `V·conj(I) − S_spec`; PV buses
replace the reactive row by `|V|² − V_set²`. ZIP constant-Z is folded into `Y_bus_eff`, so it
enters through `Y·V`; constant-current loads are subtracted from the specified injection
at `|V_state|`. The LCC and VSC tail rows follow the bus rows.
"""
function _update_rect_ci_residual_values!(
    R::ACRectangularCIResidual,
    x::Vector{Float64},
    data::ACPowerFlowData,
    time_step::Int64,
)
    F = R.Rv
    e_state = R.e_state
    f_state = R.f_state
    P_eff_cache = R.P_eff_cache
    Q_eff_cache = R.Q_eff_cache
    Ir_acc = R.Ir_acc
    Ii_acc = R.Ii_acc
    const_I_P = R.const_I_P
    const_I_Q = R.const_I_Q
    P_net_set = R.P_net_set
    spf = R.bus_slack_participation_factors
    bus_state_offset = R.bus_state_offset
    n_buses = first(size(data.bus_type))
    n_lccs = size(data.lcc.p_set, 1)
    bus_types = view(data.bus_type, :, time_step)

    # 1) Push state into data (only PQ updates bus_magnitude; PV preserves V_set).
    rect_update_data!(data, x, bus_state_offset, R.bus_block_size, time_step)
    # Populate state caches before LCC admittance refresh (LCC needs |V_state|).
    @inbounds for i in 1:n_buses
        off = Int(bus_state_offset[i])
        bt = bus_types[i]
        if bt == PSY.ACBusTypes.REF
            Vm = data.bus_magnitude[i, time_step]
            θ = data.bus_angles[i, time_step]
            e_state[i] = Vm * cos(θ)
            f_state[i] = Vm * sin(θ)
        else
            e_state[i] = x[off]
            f_state[i] = x[off + 1]
        end
    end
    if n_lccs > 0
        # PV buses store V_set in data.bus_magnitude; use the rect form so LCC math
        # sees |V_state| = sqrt(e² + f²), matching the rectangular Jacobian.
        _update_ybus_lcc!(data, time_step, e_state, f_state)
    end

    # 2) Compute P_eff / Q_eff (slack distribution + ZIP constant-current correction).
    # ZIP constant-Z is folded into `Y_bus_eff` at setup (see `fold_zip_constant_z!`
    # in `rectangular_ci_setup.jl`), so only constant-P and constant-I appear here.
    @inbounds for i in 1:n_buses
        # ZIP const-I uses |V_state|; V_FLOOR2 (1e-16) guards 1/|V|². The floor only
        # trips at degenerate |V| < 1e-8 pu (never near a solution), where the Jacobian
        # keeps the unfloored derivative: inexact but finite and |V|-restoring, and
        # harmless since the iteration never converges there.
        Vm = sqrt(max(e_state[i]^2 + f_state[i]^2, V_FLOOR2))
        P_eff_cache[i] = R.P_net_const[i] - const_I_P[i] * Vm
        Q_eff_cache[i] = R.Q_net_const[i] - const_I_Q[i] * Vm
    end
    for (ref_bus, subnetwork_buses) in R.subnetworks
        # An island with more than one swing (REF) bus holds each swing at its own
        # fixed complex voltage, so each swing carries its OWN slack (handled in the
        # REF branch below, using x[off] directly); no slack is distributed to any
        # other bus in that island. Single-swing islands keep the distributed path.
        ref_bus in R.independent_ref && continue
        ref_off = Int(bus_state_offset[ref_bus])
        P_slack_total = x[ref_off] - P_net_set[ref_bus]
        for bus_k in subnetwork_buses
            c_k = spf[bus_k]
            c_k == 0.0 && continue
            bus_k == ref_bus && continue
            P_eff_cache[bus_k] += c_k * P_slack_total
        end
    end

    # 3) Network current I = Y_bus_eff·V, plus the LCC terminal currents.
    fill!(Ir_acc, 0.0)
    fill!(Ii_acc, 0.0)
    Y = R.Y_bus_eff
    Yvals = SparseArrays.nonzeros(Y)
    Yrows = SparseArrays.rowvals(Y)
    @inbounds for col in 1:n_buses
        e_col = e_state[col]
        f_col = f_state[col]
        for j in SparseArrays.nzrange(Y, col)
            row = Yrows[j]
            g = real(Yvals[j])
            b = imag(Yvals[j])
            Ir_acc[row] += g * e_col - b * f_col
            Ii_acc[row] += g * f_col + b * e_col
        end
    end
    if n_lccs > 0
        for (bus_indices, self_admittances) in
            zip(data.lcc.bus_indices, data.lcc.branch_admittances)
            for (bus_ix, y_val) in zip(bus_indices, self_admittances)
                e_i = e_state[bus_ix]
                f_i = f_state[bus_ix]
                g = real(y_val)
                b = imag(y_val)
                Ir_acc[bus_ix] += g * e_i - b * f_i
                Ii_acc[bus_ix] += g * f_i + b * e_i
            end
        end
    end

    # 4) Per-bus rows: S_calc − S_spec, PV's reactive row replaced by the |V|² pin.
    fill!(F, 0.0)
    @inbounds for i in 1:n_buses
        off = Int(bus_state_offset[i])
        bt = bus_types[i]
        e_i = e_state[i]
        f_i = f_state[i]
        P_calc = e_i * Ir_acc[i] + f_i * Ii_acc[i]
        Q_calc = f_i * Ir_acc[i] - e_i * Ii_acc[i]
        if bt == PSY.ACBusTypes.REF
            # x[off] holds `P_net_set[ref] + total_slack` (polar convention: the state
            # carries the WHOLE island slack); REF's own share is `c_ref · total_slack`.
            if i in R.independent_ref
                P_net_cp = x[off]
            else
                P_net_cp = P_net_set[i] + spf[i] * (x[off] - P_net_set[i])
            end
            # |V| at REF is fixed at V_set; subtract the ZIP constant-current draw so the
            # recovered injection matches polar's `bus_active_power_injections`.
            Vm = sqrt(max(e_i^2 + f_i^2, V_FLOOR2))
            F[off] = P_calc - (P_net_cp - const_I_P[i] * Vm)
            F[off + 1] = Q_calc - (x[off + 1] - const_I_Q[i] * Vm)
        elseif bt == PSY.ACBusTypes.PV
            F[off] = P_calc - P_eff_cache[i]
            # V_set² from data.bus_magnitude (preserved by rect_update_data!).
            F[off + 1] = e_i^2 + f_i^2 - data.bus_magnitude[i, time_step]^2
        else
            F[off] = P_calc - P_eff_cache[i]
            F[off + 1] = Q_calc - Q_eff_cache[i]
        end
    end

    # 5) LCC tail residuals, with |V_state| instead of polar's bus_magnitude (V_set at PV).
    if n_lccs > 0
        _set_lcc_tail_residuals!(
            F, data, R.total_bus_state, time_step, e_state, f_state,
        )
    end

    # 6) VSC / DC-network tail: converter injections into the bus rows + control/DC-KCL rows.
    dcn = get_dc_network(data)
    if has_dc_network(dcn)
        vsc_off = R.total_bus_state + 4 * n_lccs
        _read_vsc_state!(dcn, x, vsc_off, time_step)
        _apply_vsc_bus_injections_rect!(F, dcn, bus_state_offset, bus_types, time_step)
        _set_vsc_tail_residuals_rect!(F, dcn, e_state, f_state, vsc_off, time_step)
    end
    return
end
