# Equation numbers refer to Artoisenet & Verstraete, arXiv:2609.14132. The blocks
# Yvv/Yvq/Yqv/Yls are network-only, so the code adds the PV-shunt y·ũ term to iv_raw.
# `_ga_iterate!` must not allocate after warm-up: all buffers live in the GAWorkspace.

_ga_slack_voltages(data::ACPowerFlowData, part::GAPartition, time_step::Int) =
    get_bus_magnitude(data)[part.s_ix, time_step] .*
    cis.(get_bus_angles(data)[part.s_ix, time_step])

function _ga_u0!(ws::GAWorkspace, cache::GeneralizedAdmittanceCache,
    u_s::Vector{ComplexF64})
    mul!(ws.u0, cache.blocks.Yls, u_s)
    PNM.solve!(cache.Fl, ws.u0)
    ws.u0 .*= -1
    return
end

function _ga_iterate!(cache::GeneralizedAdmittanceCache, np::GANodalPower,
    y::Vector{ComplexF64}, part::GAPartition)
    ws = cache.ws
    Vset = part.Vset
    nv = n_v(part)
    nl = length(ws.u)
    R = ws.R
    # R = Y′ℓℓ⁻¹[i, [0; i_q]]: response to all currents, and to the PQ currents only
    @inbounds for k in 1:nl
        R[k, 1] = ws.i[k]
        R[k, 2] = ws.i[k]
    end
    @inbounds for k in 1:nv
        R[k, 2] = zero(ComplexF64)
    end
    PNM.solve!(cache.Fl, R)
    @inbounds for k in 1:nl
        ws.u[k] = ws.u0[k] + R[k, 1]
    end
    # Tentative u from all currents; pin |u_v| = Vset and keep the angle
    # ũ_v: the part of u_v that the v-bus currents must supply
    @inbounds for k in 1:nv
        a2 = abs2(ws.u[k])
        ws.u[k] *= Vset[k] / sqrt(a2)
        ws.ut[k] = ws.u[k] - ws.u0[k] - R[k, 2]
    end
    # Schur on q: the i_v that gives ũ_v with no added q current
    mul!(ws.w, cache.blocks.Yqv, ws.ut)
    PNM.solve!(cache.Fq, ws.w)
    mul!(ws.iv_raw, cache.blocks.Yvv, ws.ut)
    mul!(ws.iv_raw, cache.blocks.Yvq, ws.w, -1.0, 1.0)
    @inbounds for k in 1:nv
        ws.iv_raw[k] += y[k] * ws.ut[k]
    end
    # u_q from the slack, the PQ currents and the ũ_v response (eq. 21)
    @inbounds for j in 1:(nl - nv)
        k = nv + j
        ws.u[k] = ws.u0[k] + R[k, 2] - ws.w[j]
    end
    # u is final: mismatches and new currents from the power model at u
    gap = 0.0
    island = part.island_of_l
    fill!(ws.psum, 0.0)
    @inbounds for k in 1:nv                           # PV buses
        uk = ws.u[k]
        vm2 = abs2(uk)
        α = vm2 * real(y[k]) - real(_ga_s(np, k, Vset[k]))
        z = conj(uk) * ws.iv_raw[k]
        gap = max(gap, abs(real(z) - α))
        ws.psum[island[k]] += real(z) - α
        ws.q_v[k] = imag(z) - vm2 * imag(y[k])
        ws.i[k] = complex(α, imag(z)) * uk / vm2
    end
    @inbounds for k in (nv + 1):nl                    # PQ buses; gap uses the previous i_q
        uk = ws.u[k]
        a2 = abs2(uk)
        s = _ga_s(np, k, sqrt(a2))
        g = uk * conj(ws.i[k]) - a2 * conj(y[k]) + s
        gap = max(gap, abs(real(g)), abs(imag(g)))
        ws.psum[island[k]] += real(g)
        ws.i[k] = (a2 * y[k] - conj(s)) * uk / a2
    end
    return gap
end

# Shunts that make zero corrective current reproduce the current iterate (paper eq. 10).
function _ga_ideal_shunts!(y::Vector{ComplexF64}, ws::GAWorkspace, np::GANodalPower,
    part::GAPartition)
    nv = n_v(part)
    for k in 1:nv
        y[k] = _ga_pv_shunt(np, k, part.Vset[k], ws.q_v[k])
    end
    for k in (nv + 1):n_l(part)
        y[k] = _ga_pq_shunt(np, k, abs(ws.u[k]))
    end
    return
end
