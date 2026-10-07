# VSC DC substep and final settle. AC-voltage converters get Q_c from the AC side, so the DC
# substep settles only P_c and V_dc. The iteration kernel calculates the Q of these converters,
# and this file writes it back to the DC network.

struct GANoDC end

struct GAVSCSubstep
    dcn::DCNetwork
    Vm::Vector{Float64}
    lpos::Vector{Tuple{Int, Int}}
end

const GADCContext = Union{GANoDC, GAVSCSubstep}

function _ga_dc_context(data::ACPowerFlowData, part::GAPartition, conv::GAConverterTerms,
    time_step::Int)
    dcn = get_dc_network(data)
    nconv = n_vsc_converters(dcn)
    if iszero(nconv)
        return GANoDC()
    end
    pos = Dict(ix => j for (j, ix) in enumerate(part.l_ix))
    lpos = [
        (c, pos[dcn.converter_ac_bus_ix[c]]) for c in 1:nconv
        if haskey(pos, dcn.converter_ac_bus_ix[c])
    ]
    for c in 1:nconv
        ix = dcn.converter_ac_bus_ix[c]
        conv.p_c[ix] += dcn.p_c[c, time_step]
        if !controls_ac_voltage(dcn.converter_mode[c])
            conv.q_c[ix] += dcn.q_c[c, time_step]
        end
    end
    return GAVSCSubstep(dcn, get_bus_magnitude(data)[:, time_step], lpos)
end

function _ga_dc_substep!(::GANoDC, ::GAWorkspace, ::GANodalPower, ::GAPartition, ::Int)
    return 0.0
end

function _ga_dc_substep!(ctx::GAVSCSubstep, ws::GAWorkspace, np::GANodalPower,
    part::GAPartition, time_step::Int)
    dcn = ctx.dcn
    nconv = n_vsc_converters(dcn)
    for (c, k) in ctx.lpos
        ctx.Vm[dcn.converter_ac_bus_ix[c]] = abs(ws.u[k])
        if controls_ac_voltage(dcn.converter_mode[c])
            dcn.q_c[c, time_step] = imag(_ga_s(np, k, part.Vset[k])) - ws.q_v[k]
        end
    end
    p_old = dcn.p_c[1:nconv, time_step]
    _vsc_warm_start!(dcn, ctx.Vm, time_step; max_iter = GA_DC_MAX_ITER)
    change = 0.0
    # REF-bus converters have no sP slot.
    for (c, k) in ctx.lpos
        np.sP[k] += p_old[c] - dcn.p_c[c, time_step]
    end
    for c in 1:nconv
        change = max(change, abs(dcn.p_c[c, time_step] - p_old[c]))
    end
    return change
end

function _ga_dc_finalize!(::GANoDC, ::ACPowerFlowData, ::Int)
    return
end

function _ga_eval_polar_residual(data::ACPowerFlowData, time_step::Int)
    x = calculate_x0(data, time_step)
    residual = ACPowerFlowResidual(data, time_step)
    residual(data, x, time_step)
    return residual, x
end

# At the final V: close each AC-voltage converter's own AC Q row (its F_Q falls by Q_c), then
# settle P_c and V_dc.
function _ga_dc_finalize!(ctx::GAVSCSubstep, data::ACPowerFlowData, time_step::Int)
    residual, _ = _ga_eval_polar_residual(data, time_step)
    dcn = ctx.dcn
    for c in 1:n_vsc_converters(dcn)
        if controls_ac_voltage(dcn.converter_mode[c])
            dcn.q_c[c, time_step] += residual.Rv[2 * dcn.converter_ac_bus_ix[c]]
        end
    end
    _vsc_warm_start!(dcn, view(get_bus_magnitude(data), :, time_step), time_step)
    return
end
