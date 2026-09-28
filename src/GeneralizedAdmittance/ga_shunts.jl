struct GAConverterTerms
    p_lcc::Vector{Float64}
    q_lcc::Vector{Float64}
    p_c::Vector{Float64}
    q_c::Vector{Float64}
end

GAConverterTerms(n::Int) = GAConverterTerms(zeros(n), zeros(n), zeros(n), zeros(n))

struct GANodalPower
    sP::Vector{ComplexF64}
    sI::Vector{ComplexF64}
    sZ::Vector{ComplexF64}
end

# Only the real part of s is read at v-buses; their reactive power is extracted (eq. 12).
function GANodalPower(data::ACPowerFlowData, part::GAPartition, conv::GAConverterTerms,
    time_step::Int)
    t = time_step
    sP = ComplexF64[
        complex(
            data.bus_active_power_withdrawals[ix, t] -
            data.bus_active_power_injections[ix, t] - data.bus_hvdc_net_power[ix, t] +
            conv.p_lcc[ix] - conv.p_c[ix],
            data.bus_reactive_power_withdrawals[ix, t] -
            data.bus_reactive_power_injections[ix, t] + conv.q_lcc[ix] - conv.q_c[ix],
        ) for ix in part.l_ix
    ]
    sI = ComplexF64[
        complex(data.bus_active_power_constant_current_withdrawals[ix, t],
            data.bus_reactive_power_constant_current_withdrawals[ix, t]) for
        ix in part.l_ix
    ]
    sZ = ComplexF64[
        complex(data.bus_active_power_constant_impedance_withdrawals[ix, t],
            data.bus_reactive_power_constant_impedance_withdrawals[ix, t]) for
        ix in part.l_ix
    ]
    return GANodalPower(sP, sI, sZ)
end

_ga_s(np::GANodalPower, k::Int, vm::Float64) = np.sP[k] + np.sI[k] * vm + np.sZ[k] * vm^2

function _ga_flat_start_q0(b::GABlocks, part::GAPartition, y::Vector{ComplexF64},
    Vm_s::Vector{Float64})
    nv = n_v(part)
    if iszero(nv)
        return Float64[]
    end
    s, v, q = part.s_ix, part.v_ix, part.q_ix
    B = imag.(b.Ynet)
    Bqq = B[q, q] + SparseArrays.spdiagm(0 => imag.(y[(nv + 1):end]))
    u_q = -(B[q, s] * Vm_s + B[q, v] * part.Vset)
    PNM.solve!(PNM.klu_factorize(Bqq), u_q)
    return part.Vset .* (B[v, s] * Vm_s + B[v, v] * part.Vset + B[v, q] * u_q)
end

function _ga_initial_shunts(b::GABlocks, np::GANodalPower, part::GAPartition,
    data::ACPowerFlowData, time_step::Int)
    nv = n_v(part)
    y = Vector{ComplexF64}(undef, n_l(part))
    for k in (nv + 1):n_l(part)
        ur = data.bus_magnitude[part.l_ix[k], time_step]
        y[k] = conj(np.sP[k] + np.sI[k] * ur) / ur^2 + conj(np.sZ[k])
    end
    q0 = _ga_flat_start_q0(b, part, y, data.bus_magnitude[part.s_ix, time_step])
    for k in 1:nv
        vs = part.Vset[k]
        y[k] = complex(real(_ga_s(np, k, vs)), -q0[k]) / vs^2
    end
    return y
end
