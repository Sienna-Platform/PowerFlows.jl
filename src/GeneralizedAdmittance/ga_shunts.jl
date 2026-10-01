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
            get_bus_active_power_withdrawals(data)[ix, t] -
            get_bus_active_power_injections(data)[ix, t] -
            get_bus_hvdc_net_power(data)[ix, t] + conv.p_lcc[ix] - conv.p_c[ix],
            get_bus_reactive_power_withdrawals(data)[ix, t] -
            get_bus_reactive_power_injections(data)[ix, t] + conv.q_lcc[ix] -
            conv.q_c[ix],
        ) for ix in part.l_ix
    ]
    sI = ComplexF64[
        complex(
            get_bus_active_power_constant_current_withdrawals(data)[ix, t],
            get_bus_reactive_power_constant_current_withdrawals(data)[ix, t],
        ) for ix in part.l_ix
    ]
    sZ = ComplexF64[
        complex(
            get_bus_active_power_constant_impedance_withdrawals(data)[ix, t],
            get_bus_reactive_power_constant_impedance_withdrawals(data)[ix, t],
        ) for ix in part.l_ix
    ]
    return GANodalPower(sP, sI, sZ)
end

_ga_s(np::GANodalPower, k::Int, vm::Float64) = np.sP[k] + np.sI[k] * vm + np.sZ[k] * vm^2

# Shunts that absorb the bus power at |u| = vm (PQ, eq. 7) or at Vset with net reactive
# consumption q (PV, eq. 8).
_ga_pq_shunt(np::GANodalPower, k::Int, vm::Float64) = conj(_ga_s(np, k, vm)) / vm^2
_ga_pv_shunt(np::GANodalPower, k::Int, vs::Float64, q::Float64) =
    complex(real(_ga_s(np, k, vs)), -q) / vs^2

function _ga_flat_start_q0(b::GABlocks, part::GAPartition, y::Vector{ComplexF64},
    Vm_s::Vector{Float64})
    nv = n_v(part)
    if iszero(nv)
        return Float64[]
    end
    nz = SparseArrays.nonzeros(b.Bqq)
    copyto!(nz, b.Bqq_net_nz)
    for j in eachindex(b.Bqq_diag)
        nz[b.Bqq_diag[j]] += imag(y[nv + j])
    end
    Yvs = view(b.Yls, 1:nv, :)
    Yqs = view(b.Yls, (nv + 1):size(b.Yls, 1), :)
    u_q = -imag.(Yqs * Vm_s .+ b.Yqv * part.Vset)
    PNM.solve!(PNM.klu_factorize(b.Bqq), u_q)
    return part.Vset .* imag.(Yvs * Vm_s .+ b.Yvv * part.Vset .+ b.Yvq * u_q)
end

function _ga_initial_shunts(b::GABlocks, np::GANodalPower, part::GAPartition,
    data::ACPowerFlowData, time_step::Int)
    nv = n_v(part)
    y = Vector{ComplexF64}(undef, n_l(part))
    for k in (nv + 1):n_l(part)
        y[k] = _ga_pq_shunt(np, k, get_bus_magnitude(data)[part.l_ix[k], time_step])
    end
    q0 = _ga_flat_start_q0(b, part, y, get_bus_magnitude(data)[part.s_ix, time_step])
    for k in 1:nv
        y[k] = _ga_pv_shunt(np, k, part.Vset[k], q0[k])
    end
    return y
end

# The shunt choice leaves the solution unchanged (the corrective current absorbs any
# error) but sets the contraction: a PV current j(q0 − q)/ū rotates with its bus angle,
# with loop gain ≈ |Z_th|·|q0 − q|/V². The inductive term lowers |Z_th| at PV buses.
function _ga_stiffen_pv!(y::Vector{ComplexF64}, b::GABlocks, nv::Int, κ::Float64)
    for k in 1:nv
        y[k] -= im * κ * abs(b.Yll_net_nz[b.Yll_diag[k]])
    end
    return
end
