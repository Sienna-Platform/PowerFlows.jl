# One step of the generalized-admittance fixed point (spec §3.5). Per iteration: one
# 2-column Yℓℓ solve (RHS i and [0; i_q]), one Yqq solve (Yqv·ũ), and four sparse mat-vecs.
# The blocks Yvv/Yvq/Yqv/Yls are network-only, so the PV-shunt y·ũ term is added back in
# step 6. Allocation-free after warm-up: all buffers live in the GAWorkspace.

_ga_slack_voltages(data::ACPowerFlowData, part::GAPartition, time_step::Int) =
    data.bus_magnitude[part.s_ix, time_step] .* cis.(data.bus_angles[part.s_ix, time_step])

function _ga_u0!(ws::GAWorkspace, cache::GeneralizedAdmittanceCache,
    u_s::Vector{ComplexF64})
    mul!(ws.u0, cache.blocks.Yls, u_s)
    PNM.solve!(cache.Fl, ws.u0)
    ws.u0 .*= -1
    return
end

function _ga_iterate!(ws::GAWorkspace, cache::GeneralizedAdmittanceCache,
    np::GANodalPower, y::Vector{ComplexF64}, Vset::Vector{Float64}, nv::Int)
    nl = length(ws.u)
    R = ws.R
    @inbounds for k in 1:nl                           # step 1: RHS [i, [0; i_q]]
        R[k, 1] = ws.i[k]
        R[k, 2] = ws.i[k]
    end
    @inbounds for k in 1:nv
        R[k, 2] = zero(ComplexF64)
    end
    PNM.solve!(cache.Fl, R)
    @inbounds for k in 1:nl                           # step 2
        ws.u[k] = ws.u0[k] + R[k, 1]
    end
    @inbounds for k in 1:nv                           # steps 3-4
        ws.u[k] = Vset[k] * ws.u[k] / abs(ws.u[k])
        ws.ut[k] = ws.u[k] - ws.u0[k] - R[k, 2]
    end
    mul!(ws.w, cache.blocks.Yqv, ws.ut)               # step 5
    PNM.solve!(cache.Fq, ws.w)
    mul!(ws.iv_raw, cache.blocks.Yvv, ws.ut)          # step 6 (Yvv is network-only)
    mul!(ws.iv_raw, cache.blocks.Yvq, ws.w, -1.0, 1.0)
    @inbounds for k in 1:nv
        ws.iv_raw[k] += y[k] * ws.ut[k]
    end
    @inbounds for j in 1:(nl - nv)                    # step 7, eq. (10)
        k = nv + j
        ws.u[k] = ws.u0[k] + R[k, 2] - ws.w[j]
    end
    gap = 0.0
    @inbounds for k in 1:nv                           # steps 8, 10, 11 (PV)
        uk = ws.u[k]
        vm2 = abs2(uk)
        α = vm2 * real(y[k]) - real(_ga_s(np, k, Vset[k]))
        z = conj(uk) * ws.iv_raw[k]
        gap = max(gap, abs(real(z) - α))
        ws.q_v[k] = imag(z) - vm2 * imag(y[k])
        ws.i[k] = complex(α, imag(z)) / conj(uk)
    end
    @inbounds for k in (nv + 1):nl                    # steps 8, 9 (PQ); gap uses i_q^(k-1)
        uk = ws.u[k]
        s = _ga_s(np, k, abs(uk))
        g = uk * conj(ws.i[k]) - abs2(uk) * conj(y[k]) + s
        gap = max(gap, abs(real(g)), abs(imag(g)))
        ws.i[k] = (abs2(uk) * y[k] - conj(s)) / conj(uk)
    end
    return gap
end
