function ga_dense_problem(data::PF.PowerFlowData, ts::Int = 1)
    Y = Matrix{ComplexF64}(PNM.get_data(data.power_network_matrix))
    s_ix, v_ix, q_ix = PF.bus_type_idx(data, ts)
    l_ix = vcat(v_ix, q_ix)
    s = ComplexF64[
        complex(
            data.bus_active_power_withdrawals[ix, ts] -
            data.bus_active_power_injections[ix, ts] - data.bus_hvdc_net_power[ix, ts],
            data.bus_reactive_power_withdrawals[ix, ts] -
            data.bus_reactive_power_injections[ix, ts],
        ) for ix in l_ix
    ]
    return (; Y, s_ix, v_ix, q_ix, l_ix,
        u_s = data.bus_magnitude[s_ix, ts] .* cis.(data.bus_angles[s_ix, ts]),
        Vset = data.bus_magnitude[v_ix, ts], s, u_ref = data.bus_magnitude[l_ix, ts])
end

function ga_dense_reference(p, y; tol = 1e-11, maxiter = 500)
    nv = length(p.v_ix)
    nl = length(p.l_ix)
    qr = (nv + 1):nl
    Yll = p.Y[p.l_ix, p.l_ix] + Diagonal(y)
    Zvv_inv = Yll[1:nv, 1:nv] - Yll[1:nv, qr] * (Yll[qr, qr] \ Yll[qr, 1:nv])   # (18)
    u0 = -(Yll \ (p.Y[p.l_ix, p.s_ix] * p.u_s))                               # (12)
    i = zeros(ComplexF64, nl)
    steps = []
    for _ in 1:maxiter
        u = u0 + Yll \ i                                                     # (14)
        u[1:nv] = p.Vset .* u[1:nv] ./ abs.(u[1:nv])                         # (15)
        du_q = Yll \ vcat(zeros(ComplexF64, nv), i[qr])                      # (16)
        ut = u[1:nv] - u0[1:nv] - du_q[1:nv]                                 # (17)
        iv_raw = Zvv_inv * ut                                                # (19)
        u[qr] = (u0 + Yll \ vcat(iv_raw, i[qr]))[qr]                         # (20)-(21)
        uv = u[1:nv]
        uq = u[qr]
        gq = uq .* conj.(i[qr]) .- abs2.(uq) .* conj.(y[qr]) .+ p.s[qr]
        gv = real.(conj.(uv) .* iv_raw) .- (abs2.(uv) .* real.(y[1:nv]) .- real.(p.s[1:nv]))
        gap = maximum(abs, vcat(real.(gq), imag.(gq), gv); init = 0.0)     # (26)
        iq = y[qr] .* (abs2.(uq) .- p.u_ref[qr] .^ 2) ./ conj.(uq)           # (22)
        iv = im .* imag.(conj.(uv) .* iv_raw) ./ conj.(uv)                   # (25)
        i = vcat(iv, iq)
        push!(steps, (; u = copy(u), iv_raw = copy(iv_raw), i_next = copy(i), gap))
        if gap <= tol
            return steps
        end
    end
    return steps
end

function ga_parity(sys::PSY.System; pf_kwargs = (;), ga_solve_kwargs = (;), tol = 1e-6)
    data_nr = PowerFlowData(ACPowerFlow{NewtonRaphsonACPowerFlow}(; pf_kwargs...), sys)
    data_ga = PowerFlowData(
        ACPowerFlow{GeneralizedAdmittanceACPowerFlow}(; pf_kwargs...), sys)
    @test solve_power_flow!(data_nr)
    @test solve_power_flow!(data_ga; ga_solve_kwargs...)
    for f in (:bus_magnitude, :bus_angles, :bus_active_power_injections,
        :bus_reactive_power_injections)
        @test maximum(abs.(getfield(data_nr, f) .- getfield(data_ga, f))) < tol
    end
    return data_nr, data_ga
end
