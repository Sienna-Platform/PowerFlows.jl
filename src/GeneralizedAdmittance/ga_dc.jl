# VSC DC substep and final settle (spec §3.8). AC-voltage converters own their Q_c on the AC side,
# so the DC substep only settles P_c and V_dc; the AC-voltage converters' Q is extracted by the
# iteration kernel and handed back to the DC network here.

struct GANoDC end

struct GAVSCSubstep
    dcn::DCNetwork
    Vm::Vector{Float64}
    lpos::Vector{Int}
end

function _ga_dc_context(data::ACPowerFlowData, part::GAPartition, conv::GAConverterTerms,
    time_step::Int)
    dcn = get_dc_network(data)
    nconv = n_vsc_converters(dcn)
    if iszero(nconv)
        return GANoDC()
    end
    pos = Dict(ix => j for (j, ix) in enumerate(part.l_ix))
    lpos = [get(pos, dcn.converter_ac_bus_ix[c], 0) for c in 1:nconv]
    for c in 1:nconv
        ix = dcn.converter_ac_bus_ix[c]
        conv.p_c[ix] = dcn.p_c[c, time_step]
        if controls_ac_voltage(dcn.converter_mode[c])
            conv.q_c[ix] = 0.0
        else
            conv.q_c[ix] = dcn.q_c[c, time_step]
        end
    end
    return GAVSCSubstep(dcn, data.bus_magnitude[:, time_step], lpos)
end

_ga_dc_substep!(::GANoDC, args...) = 0.0

function _ga_dc_substep!(ctx::GAVSCSubstep, ws::GAWorkspace, np::GANodalPower,
    part::GAPartition, conv::GAConverterTerms, time_step::Int)
    dcn = ctx.dcn
    nconv = n_vsc_converters(dcn)
    for c in 1:nconv
        k = ctx.lpos[c]
        if iszero(k)
            continue
        end
        ctx.Vm[dcn.converter_ac_bus_ix[c]] = abs(ws.u[k])
        if controls_ac_voltage(dcn.converter_mode[c])
            dcn.q_c[c, time_step] = imag(_ga_s(np, k, part.Vset[k])) - ws.q_v[k]
        end
    end
    _vsc_warm_start!(dcn, ctx.Vm, time_step; max_iter = GA_DC_MAX_ITER)
    change = 0.0
    for c in 1:nconv
        ix = dcn.converter_ac_bus_ix[c]
        p_new = dcn.p_c[c, time_step]
        change = max(change, abs(p_new - conv.p_c[ix]))
        if !iszero(ctx.lpos[c])
            np.sP[ctx.lpos[c]] += conv.p_c[ix] - p_new
        end
        conv.p_c[ix] = p_new
    end
    return change
end

_ga_dc_finalize!(::GANoDC, ::ACPowerFlowData, ::Int) = nothing

# At the final V: close each AC-voltage converter's own AC Q row (its F_Q falls by Q_c), then
# settle P_c and V_dc.
function _ga_dc_finalize!(ctx::GAVSCSubstep, data::ACPowerFlowData, time_step::Int)
    x = calculate_x0(data, time_step)
    residual = ACPowerFlowResidual(data, time_step)
    residual(data, x, time_step)
    dcn = ctx.dcn
    for c in 1:n_vsc_converters(dcn)
        if controls_ac_voltage(dcn.converter_mode[c])
            dcn.q_c[c, time_step] += residual.Rv[2 * dcn.converter_ac_bus_ix[c]]
        end
    end
    _vsc_warm_start!(dcn, view(data.bus_magnitude, :, time_step), time_step)
    return
end
