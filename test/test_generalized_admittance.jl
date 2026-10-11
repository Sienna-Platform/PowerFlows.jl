import Pardiso
const GA = GeneralizedAdmittanceACPowerFlow
const GA_SYS14 = PSB.build_system(PSB.PSITestSystems, "c_sys14"; add_forecasts = false)

# Fresh copy for tests that mutate the system; unmodified tests use GA_SYS14 directly.
ga_sys14() = deepcopy(GA_SYS14)

function ga_pq_buses(sys::PSY.System)
    return sort!(
        collect(
            PSY.get_components(
                b -> PSY.get_bustype(b) == PSY.ACBusTypes.PQ,
                PSY.ACBus,
                sys,
            ),
        );
        by = PSY.get_number,
    )
end

function ga_setup(sys::PSY.System)
    data = PowerFlowData(ACPowerFlow{GA}(), sys)
    part = PF.GAPartition(data, 1, Dict{Int, Float64}())
    np = PF.GANodalPower(
        data, part, PF.GAConverterTerms(size(PF.get_bus_type(data), 1)), 1)
    return data, part, np
end

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

# Equation numbers refer to Artoisenet & Verstraete, arXiv:2609.14132, 2026.
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
    data_ga = PowerFlowData(ACPowerFlow{GA}(; pf_kwargs...), sys)
    @test solve_power_flow!(data_nr)
    @test solve_power_flow!(data_ga; ga_solve_kwargs...)
    for f in (:bus_magnitude, :bus_angles, :bus_active_power_injections,
        :bus_reactive_power_injections)
        @test maximum(abs.(getfield(data_nr, f) .- getfield(data_ga, f))) < tol
    end
    return data_nr, data_ga
end

function ga_true_flat_start!(data)
    bus_type = PF.get_bus_type(data)
    for ix in axes(bus_type, 1)
        if bus_type[ix, 1] == PSY.ACBusTypes.PQ
            PF.get_bus_magnitude(data)[ix, 1] = 1.0
        end
        if bus_type[ix, 1] != PSY.ACBusTypes.REF
            PF.get_bus_angles(data)[ix, 1] = 0.0
        end
    end
    return
end

function ga_compare(data_a, data_b; tol = 1e-6)
    @test maximum(abs.(PF.get_bus_magnitude(data_a) .- PF.get_bus_magnitude(data_b))) < tol
    @test maximum(abs.(PF.get_bus_angles(data_a) .- PF.get_bus_angles(data_b))) < tol
    return
end

function ga_kernel_setup(sys)
    data, part, np = ga_setup(sys)
    cache = PF._get_or_build_ga_cache!(data, part, PNM.KLUSolver())
    y = PF._ga_initial_shunts(cache.blocks, np, part, data, 1)
    PF._ga_factor!(cache, y, part, PF.get_bus_lookup(data))
    PF._ga_u0!(cache.ws, cache, PF._ga_slack_voltages(data, part, 1))
    fill!(cache.ws.i, 0.0im)
    return data, part, np, cache, y
end

function ga_check_against_dense(sys, n_iter)
    data, part, np, cache, y = ga_kernel_setup(sys)
    steps = ga_dense_reference(ga_dense_problem(data), y; maxiter = n_iter, tol = 0.0)
    for k in 1:n_iter
        gap = PF._ga_iterate!(cache, np, y, part)
        @test maximum(abs.(cache.ws.u .- steps[k].u)) < 1e-9
        @test maximum(abs.(cache.ws.i .- steps[k].i_next)) < 1e-9
        @test isapprox(gap, steps[k].gap; rtol = 1e-7, atol = 1e-12)
    end
    return part, np, cache, y
end

function ga_add_vsc!(
    sys::PSY.System, name::String; to_bus = ga_pq_buses(sys)[2], kw...,
)
    arc = _get_or_make_arc(sys, ga_pq_buses(sys)[1], to_bus)
    PSY.add_component!(
        sys,
        PSY.TwoTerminalVSCLine(;
            name = name,
            available = true,
            arc = arc,
            active_power_flow = 0.3,
            rating = 2.0,
            g = 1 / 32.0, # 50 pu on the 400 kV DC base and 100 MVA
            rated_dc_voltage = 400.0,
            ac_control_from = PSY.VSCACControlModes.AC_REACTIVE_POWER,
            power_factor_setpoint_from = 1.0,
            ac_control_to = PSY.VSCACControlModes.AC_REACTIVE_POWER,
            power_factor_setpoint_to = 1.0,
            input_basis = u"CU",
            kw...,
        ),
    )
    return sys
end

function ga_vsc_droop_system()
    return ga_add_vsc!(
        ga_sys14(), "ga_vsc_droop";
        dc_control_from = PSY.VSCDCControlModes.DC_VOLTAGE_DROOP,
        dc_voltage_droop_from = 0.02,
        dc_voltage_setpoint_from = 1.05,
        reactive_power_from = 0.0,
        dc_control_to = PSY.VSCDCControlModes.DC_VOLTAGE_DROOP,
        dc_voltage_droop_to = 0.03,
        dc_voltage_setpoint_to = 1.03,
        reactive_power_to = 0.0,
        converter_loss_to = PSY.LossCurve(
            PSY.QuadraticCurve(0.01, 0.02, 0.005),
            PSY.NaturalUnit(),
        ),
    )
end

# Both terminals pin AC voltage (from = ControlVdcQ, to = ControlPVac): both Q rows must hold.
function ga_vsc_ac_voltage_system()
    return ga_add_vsc!(
        ga_sys14(), "ga_vsc_av";
        dc_control_from = PSY.VSCDCControlModes.DC_VOLTAGE,
        ac_control_from = PSY.VSCACControlModes.AC_VOLTAGE,
        dc_voltage_setpoint_from = 1.05,
        ac_voltage_setpoint_from = 1.01,
        dc_control_to = PSY.VSCDCControlModes.DC_POWER,
        ac_control_to = PSY.VSCACControlModes.AC_VOLTAGE,
        dc_power_setpoint_to = 0.25,
        ac_voltage_setpoint_to = 1.0,
    )
end

# The lossy `to` terminal is on the REF bus: the DC substep must settle P_c at the fixed REF |V|.
function ga_vsc_system_ref_terminal(; g = 9 / 320.0) # 45 pu on the 400 kV DC base
    sys = ga_sys14()
    pick(t) = first(
        sort!(
            collect(PSY.get_components(b -> PSY.get_bustype(b) == t, PSY.ACBus, sys));
            by = PSY.get_number,
        ),
    )
    arc = _get_or_make_arc(sys, pick(PSY.ACBusTypes.PQ), pick(PSY.ACBusTypes.REF))
    PSY.add_component!(
        sys,
        PSY.TwoTerminalVSCLine(;
            name = "ga_vsc_ref",
            available = true,
            arc = arc,
            active_power_flow = 0.3,
            rating = 2.0,
            g = g,
            dc_control_from = PSY.VSCDCControlModes.DC_VOLTAGE,
            ac_control_from = PSY.VSCACControlModes.AC_REACTIVE_POWER,
            power_factor_setpoint_from = 1.0,
            dc_voltage_setpoint_from = 1.03,
            rated_dc_voltage = 400.0,
            reactive_power_from = 0.0,
            dc_control_to = PSY.VSCDCControlModes.DC_POWER,
            ac_control_to = PSY.VSCACControlModes.AC_REACTIVE_POWER,
            power_factor_setpoint_to = 1.0,
            dc_power_setpoint_to = 0.35,
            reactive_power_to = 0.05,
            converter_loss_to = PSY.LossCurve(
                PSY.QuadraticCurve(0.01, 0.02, 0.005),
                PSY.NaturalUnit(),
            ),
            input_basis = u"CU",
        ),
    )
    return sys
end

function ga_lcc_system()
    return system_from_openapi(
        PFP.PowerModelsData(joinpath(TEST_DATA_DIR, "case5_2_lcc.raw"));
        runchecks = false,
    )
end

@testset "GeneralizedAdmittance" begin
    @testset "types and rejections" begin
        @test typeof(ACPowerFlow{GA}()) === ACPolarPowerFlow{GA}
        @test_throws ArgumentError ACPowerFlow{GA}(; check_reactive_power_limits = true)
        @test_throws ArgumentError ACPowerFlow{GA}(;
            distribute_slack_proportional_to_headroom = true)
        @test_throws ArgumentError ACPowerFlow{GA}(;
            generator_slack_participation_factors = Dict(
                (PSY.ThermalStandard, "gen") => 1.0))
        @test_throws ArgumentError ACPowerFlow{GA}(; control_discrete_devices = true)
        @test_throws ArgumentError ACPowerFlow{GA}(; area_interchange_control = true)
        @test_throws ArgumentError PF.ACRectangularPowerFlow{GA}()
        @test_throws ArgumentError PF.ACMixedPowerFlow{GA}()
        @test PF.get_solver_kwargs(ACPowerFlow{GA}()).maxIterations ==
              PF.DEFAULT_GA_MAX_ITER
        @test PF.get_solver_kwargs(
            ACPowerFlow{GA}(;
                solution_parameters = SolutionParameters(; maxIterations = 7),
            ),
        ).maxIterations == 7
    end

    @testset "partition and blocks on c_sys14" begin
        data, part, _ = ga_setup(GA_SYS14)
        ref, pv, pq = PF.bus_type_idx(data, 1)
        @test (part.s_ix, part.v_ix, part.q_ix) == (ref, pv, pq)
        @test part.l_ix == vcat(pv, pq)
        @test part.Vset == data.bus_magnitude[pv, 1]
        b = PF.GABlocks(data, part)
        Y = Matrix{ComplexF64}(PNM.get_data(data.power_network_matrix))
        nv = PF.n_v(part)
        @test Matrix(b.Yll) ≈ Y[part.l_ix, part.l_ix]
        @test Matrix(b.Yvq) ≈ Y[pv, pq]
        @test Matrix(b.Yqv) ≈ Y[pq, pv]
        @test Matrix(b.Yls) ≈ Y[part.l_ix, ref]
        y = ComplexF64.(1:PF.n_l(part)) .* (0.1 - 0.05im)
        for scale in (1.0, 2.0)
            PF._ga_set_shunts!(b, scale .* y, nv)
            @test Matrix(b.Yll) ≈ Y[part.l_ix, part.l_ix] + Diagonal(scale .* y)
            @test Matrix(b.Yqq) ≈ Y[pq, pq] + Diagonal(scale .* y[(nv + 1):end])
        end
        @test b.net_diag ≈ diag(Y[part.l_ix, part.l_ix])
    end

    @testset "VSC AC-voltage buses join set v, bus types unchanged" begin
        data = PowerFlowData(
            ACPowerFlow{GA}(; solution_parameters = VSC_SOLUTION_PARAMETERS),
            ga_vsc_ac_voltage_system())
        targets = PF._ga_vsc_ac_voltage_targets(data, 1)
        @test length(targets) == 2
        part = PF.GAPartition(data, 1, targets)
        npv = length(PF.bus_type_idx(data, 1)[2])
        extra = part.v_ix[(npv + 1):end]
        @test sort(extra) == sort(collect(keys(targets)))
        @test part.Vset[(npv + 1):end] == [targets[ix] for ix in extra]
        @test all(data.bus_type[ix, 1] === PSY.ACBusTypes.PQ for ix in extra)
    end

    @testset "dense reference reproduces NR on c_sys5" begin
        sys = PSB.build_system(PSB.PSITestSystems, "c_sys5"; add_forecasts = false)
        data_nr = PowerFlowData(ACPowerFlow{NewtonRaphsonACPowerFlow}(), sys)
        @test solve_power_flow!(data_nr)
        p = ga_dense_problem(PowerFlowData(ACPowerFlow{NewtonRaphsonACPowerFlow}(), sys))
        nv = length(p.v_ix)
        q_nr = [
            data_nr.bus_reactive_power_withdrawals[ix, 1] -
            data_nr.bus_reactive_power_injections[ix, 1] for ix in p.v_ix
        ]
        y = vcat(complex.(real.(p.s[1:nv]), -q_nr) ./ p.Vset .^ 2,
            conj.(p.s[(nv + 1):end]) ./ p.u_ref[(nv + 1):end] .^ 2)
        steps = ga_dense_reference(p, y)
        @test last(steps).gap <= 1e-11
        @test maximum(abs.(abs.(last(steps).u) .- data_nr.bus_magnitude[p.l_ix, 1])) < 1e-8
        @test maximum(abs.(angle.(last(steps).u) .- data_nr.bus_angles[p.l_ix, 1])) < 1e-8
    end

    @testset "nodal power, ZIP split, PQ shunts, flat-start q0" begin
        @testset "ZIP split and PQ shunts on c_sys14" begin
            sys = ga_sys14()
            _add_simple_zip_load!(
                sys,
                first(ga_pq_buses(sys));
                constant_current_active_power = 0.1,
                constant_current_reactive_power = 0.05,
                constant_impedance_active_power = 0.2,
                constant_impedance_reactive_power = 0.07,
            )
            data, part, np = ga_setup(sys)
            y = PF._ga_initial_shunts(PF.GABlocks(data, part), np, part, data, 1)
            @test any(!iszero, np.sI) && any(!iszero, np.sZ)
            for (j, ix) in enumerate(part.l_ix)
                sP = complex(
                    data.bus_active_power_withdrawals[ix, 1] -
                    data.bus_active_power_injections[ix, 1] -
                    data.bus_hvdc_net_power[ix, 1],
                    data.bus_reactive_power_withdrawals[ix, 1] -
                    data.bus_reactive_power_injections[ix, 1],
                )
                @test np.sP[j] ≈ sP
                if j > PF.n_v(part)
                    ur = data.bus_magnitude[ix, 1]
                    @test y[j] ≈ conj(sP + np.sI[j] * ur) / ur^2 + conj(np.sZ[j])
                end
            end
        end

        @testset "flat-start q0 on a hand-built 3-bus system" begin
            sys = PSY.System(100.0)
            b1 = _add_simple_bus!(sys, 1, PSY.ACBusTypes.REF, 230, 1.0)
            b2 = _add_simple_bus!(sys, 2, PSY.ACBusTypes.PV, 230, 1.05)
            b3 = _add_simple_bus!(sys, 3, PSY.ACBusTypes.PQ, 230, 1.0)
            _add_simple_source!(sys, b1)
            _add_simple_thermal_standard!(sys, b2, 0.5, 0.0)
            _add_simple_load!(sys, b3, 80.0, 30.0)
            _add_simple_line!(sys, b1, b2, 0.0, 0.1, 0.0)
            _add_simple_line!(sys, b1, b3, 0.0, 0.1, 0.0)
            _add_simple_line!(sys, b2, b3, 0.0, 0.2, 0.0)
            data, part, np = ga_setup(sys)
            b = PF.GABlocks(data, part)
            y = PF._ga_initial_shunts(b, np, part, data, 1)
            q0 = only(
                PF._ga_flat_start_q0(b, part, y, data.bus_magnitude[part.s_ix, 1],
                    PF.get_bus_lookup(data)),
            )
            q3 = data.bus_reactive_power_withdrawals[only(part.q_ix), 1]
            u3 = (10 * 1.0 + 5 * 1.05) / (15 + q3) # by hand: B33 = -15 - q3
            @test q0 ≈ 1.05 * (-15 * 1.05 + 10 * 1.0 + 5 * u3) atol = 1e-10
            @test q0 < 0.0
            @test y[1] ≈ complex(real(PF._ga_s(np, 1, 1.05)), -q0) / 1.05^2
        end
    end

    @testset "solver cache" begin
        @testset "KLU factor, solve, reuse and refactor on c_sys14" begin
            data, part, np = ga_setup(GA_SYS14)
            cache = PF._get_or_build_ga_cache!(data, part, PNM.KLUSolver())
            @test data.solver_cache[] === cache
            y = PF._ga_initial_shunts(cache.blocks, np, part, data, 1)
            rhs = ComplexF64.(randn(PF.n_l(part)) .+ im .* randn(PF.n_l(part)))
            for yk in (y, 1.1 .* y)
                PF._ga_factor!(cache, yk, part, PF.get_bus_lookup(data))
                x = copy(rhs)
                PNM.solve!(cache.Fl, x)
                @test Matrix(cache.blocks.Yll) * x ≈ rhs
                xq = rhs[1:PF.n_q(part)]
                PNM.solve!(cache.Fq, xq)
                @test Matrix(cache.blocks.Yqq) * xq ≈ rhs[1:PF.n_q(part)]
            end
            @test PF._get_or_build_ga_cache!(data, part, PNM.KLUSolver()) === cache
        end

        @testset "flat-start q0 with a purely real PQ diagonal" begin
            part = PF.GAPartition([1], [2], [3, 4], [2, 3, 4], [1.0], [1, 1, 1], 1)
            lookup = Dict(10 => 1, 20 => 2, 30 => 3, 40 => 4)
            function blocks(Y, part)
                Yll = Y[part.l_ix, part.l_ix]
                Yqq = Y[part.q_ix, part.q_ix]
                Yll_diag = PF._ga_diag_positions(Yll)
                return PF.GABlocks(Yll, Yll_diag, Yqq, PF._ga_diag_positions(Yqq),
                    SparseArrays.nonzeros(Yll)[Yll_diag], Y[part.v_ix, part.v_ix],
                    Y[part.v_ix, part.q_ix], Y[part.q_ix, part.v_ix],
                    Y[part.l_ix, part.s_ix])
            end
            Y = SparseArrays.sparse(
                ComplexF64[
                    1-5im -1+5im 0 0
                    -1+5im 3-15im -1+5im -1+5im
                    0 -1+5im 1.0+0im -0.0+3im
                    0 -1+5im -0.0+3im 1-8im],
            )
            b = blocks(Y, part)
            y = zeros(ComplexF64, 3)
            q0 = PF._ga_flat_start_q0(b, part, y, [1.0], lookup)
            @test length(q0) == 1 && isfinite(only(q0))

            Y1 = SparseArrays.sparse(
                ComplexF64[
                    1-5im -1+5im 0
                    -1+5im 2-10im -1+5im
                    0 -1+5im 1.0+0im],
            )
            part1 = PF.GAPartition([1], [2], [3], [2, 3], [1.0], [1, 1], 1)
            b1 = blocks(Y1, part1)
            @test_throws r"Yqq is singular at bus 30 \(index 3\)" PF._ga_flat_start_q0(
                b1, part1, zeros(ComplexF64, 2), [1.0], Dict(10 => 1, 20 => 2, 30 => 3))
        end

        @testset "singular factorization error names the bus" begin
            part = PF.GAPartition([1], [2], [3, 4], [2, 3, 4], [1.0], [1, 1, 1], 1)
            lookup = Dict(10 => 1, 20 => 2, 30 => 3, 40 => 4)
            @test_throws r"Yℓℓ is singular at bus 40 \(index 4\)" PF._ga_factor_error(
                LinearAlgebra.SingularException(3), part, lookup, PF.GABlockYll())
            @test_throws r"Yqq is singular at bus 40 \(index 4\)" PF._ga_factor_error(
                LinearAlgebra.SingularException(2), part, lookup, PF.GABlockYqq())
            @test_throws ArgumentError PF._ga_factor_error(
                ArgumentError("x"), part, lookup, PF.GABlockYll())
        end

        if PF.PNM._has_apple_accelerate_backend()
            @testset "AppleAccelerate singular block names the bus" begin
                part = PF.GAPartition([1], [2], [3, 4], [2, 3, 4], [1.0], [1, 1, 1], 1)
                lookup = Dict(10 => 1, 20 => 2, 30 => 3, 40 => 4)
                A = SparseArrays.sparse(
                    [1, 2], [1, 2], ComplexF64[1.0, 1.0], 3, 3)
                F = PF.make_linear_solver_cache(PNM.AppleAccelerateLUSolver(), A)
                @test_throws r"singular" PF._ga_factor_block!(
                    F, A, false, PF.GABlockYll(), part, lookup)
            end
        end

        @testset "non-singular factor error is rethrown" begin
            part = PF.GAPartition([1], [2], [3, 4], [2, 3, 4], [1.0], [1, 1, 1], 1)
            lookup = Dict(10 => 1, 20 => 2, 30 => 3, 40 => 4)
            A = SparseArrays.sparse(
                [1, 2, 3], [1, 2, 3], ComplexF64[1.0, 1.0, 1.0], 3, 3)
            @test_throws ArgumentError begin
                try
                    throw(ArgumentError("backend failure"))
                catch e
                    PF._ga_on_factor_error(e, :other, A, PF.GABlockYll(), part, lookup)
                end
            end
        end
    end

    @testset "iteration kernel matches the dense reference" begin
        @testset "c_sys14, allocation-free after warm-up" begin
            part, np, cache, y = ga_check_against_dense(GA_SYS14, 5)
            @test (@allocated PF._ga_iterate!(cache, np, y, part)) == 0
        end

        if PF.PNM._has_apple_accelerate_backend()
            @testset "c_sys14, AppleAccelerate allocation-free after warm-up" begin
                data, part, np = ga_setup(GA_SYS14)
                cache = PF._get_or_build_ga_cache!(
                    data, part, PF.resolve_linear_solver_backend("AppleAccelerateLU"))
                y = PF._ga_initial_shunts(cache.blocks, np, part, data, 1)
                PF._ga_factor!(cache, y, part, PF.get_bus_lookup(data))
                PF._ga_u0!(cache.ws, cache, PF._ga_slack_voltages(data, part, 1))
                fill!(cache.ws.i, 0.0im)
                for _ in 1:3
                    PF._ga_iterate!(cache, np, y, part)
                end
                @test (@allocated PF._ga_iterate!(cache, np, y, part)) == 0
            end
        end

        @testset "no PV buses" begin
            sys = ga_sys14()
            for b in PSY.get_components(
                b -> PSY.get_bustype(b) == PSY.ACBusTypes.PV, PSY.ACBus, sys)
                PSY.set_bustype!(b, PSY.ACBusTypes.PQ)
            end
            part, _, _, _ = ga_check_against_dense(sys, 3)
            @test iszero(PF.n_v(part))
        end

        @testset "island P sum is the synced REF P row" begin
            data, part, np, cache, y = ga_kernel_setup(GA_SYS14)
            for _ in 1:4
                PF._ga_iterate!(cache, np, y, part)
            end
            for (j, ix) in enumerate(part.l_ix)
                data.bus_magnitude[ix, 1] = abs(cache.ws.u[j])
                data.bus_angles[ix, 1] = angle(cache.ws.u[j])
            end
            residual, _ = PF._ga_polar_state(data, 1)
            ref = only(part.s_ix)
            @test abs(only(cache.ws.psum)) > 1e-4
            @test residual.Rv[2 * ref - 1] ≈ -only(cache.ws.psum) atol = 1e-12
        end
    end

    @testset "shunt schedule and Anderson mixing" begin
        @testset "PV stiffening touches only the PV shunts" begin
            data, part, np = ga_setup(GA_SYS14)
            b = PF.GABlocks(data, part)
            y = PF._ga_initial_shunts(b, np, part, data, 1)
            ys = copy(y)
            PF._ga_stiffen_pv!(ys, b, PF.n_v(part), 0.5)
            Y = PNM.get_data(data.power_network_matrix)
            for (k, ix) in enumerate(part.l_ix)
                expected = y[k]
                if k <= PF.n_v(part)
                    expected -= im * 0.5 * abs(ComplexF64(Y[ix, ix]))
                end
                @test ys[k] ≈ expected
            end
        end

        @testset "refresh moves the currents by Δy ⊙ u and resets the mixing" begin
            data, part, np, cache, y = ga_kernel_setup(GA_SYS14)
            for _ in 1:3
                PF._ga_iterate!(cache, np, y, part)
            end
            ws = cache.ws
            u = copy(ws.u)
            i = copy(ws.i)
            u_s = PF._ga_slack_voltages(data, part, 1)
            y_old = copy(y)
            PF._ga_refresh_shunts!(cache, np, y, part, 0.0, u_s, PF.get_bus_lookup(data))
            @test maximum(abs.(y .- y_old)) > 1e-3
            @test ws.i ≈ i .+ (y .- y_old) .* u
            @test Matrix(cache.blocks.Yll) ≈
                  Matrix(PNM.get_data(data.power_network_matrix))[part.l_ix, part.l_ix] +
                  Diagonal(y)
            @test cache.aa.x == ws.i
            nv = PF.n_v(part)
            for k in (nv + 1):PF.n_l(part)
                @test y[k] ≈ conj(PF._ga_s(np, k, abs(u[k]))) / abs2(u[k])
            end
        end

        @testset "Anderson step solves the real least-squares mixing problem" begin
            nl, m = 7, 3
            aa = PF.GAAnderson(nl, m)
            G(x) = 0.5 .* conj.(x) .+ (0.1 + 0.2im) .* x .+ (1.0 - 0.5im)
            PF._ga_anderson_reset!(aa, zeros(ComplexF64, nl))
            xs = Vector{ComplexF64}[]
            gs = Vector{ComplexF64}[]
            for _ in 1:5
                x = copy(aa.x)
                g = G(x)
                push!(xs, x)
                push!(gs, g)
                PF._ga_anderson_step!(aa, g)
            end
            stack(v) = vcat(real.(v), imag.(v))
            f = [stack(gs[k] .- xs[k]) for k in eachindex(xs)]
            DF = reduce(hcat, [f[k + 1] .- f[k] for k in 2:4])
            DG = reduce(hcat, [stack(gs[k + 1] .- gs[k]) for k in 2:4])
            γ = DF \ f[end]
            x_ref = stack(gs[end]) .- DG * γ
            @test stack(aa.x) ≈ x_ref
            x = copy(aa.x)
            g = G(x)
            @test (@allocated PF._ga_anderson_step!(aa, g)) == 0
        end
    end

    @testset "solve parity with NR" begin
        @testset "$name" for name in ("c_sys5", "c_sys14")
            ga_parity(PSB.build_system(PSB.PSITestSystems, name; add_forecasts = false))
        end

        @testset "solve report on c_sys14" begin
            pf = ACPowerFlow{GA}()
            report = PF._ga_solve(pf, PowerFlowData(pf, GA_SYS14), 1)
            @test report.converged
            @test report.stage_exit === PF.GAConverged
            @test iszero(report.handoff_iterations)
            @test report.stage_iterations <= 20
            @test report.refreshes > 0
        end

        @testset "starts from the voltages in data ($name)" for (name, build) in (
            ("c_sys14", ga_sys14), ("LCC", ga_lcc_system))
            sys = build()
            data_nr = PowerFlowData(ACPowerFlow{NewtonRaphsonACPowerFlow}(), sys)
            @test solve_power_flow!(data_nr)
            pf = ACPowerFlow{GA}()
            data = PowerFlowData(pf, sys)
            PF.get_bus_magnitude(data) .= PF.get_bus_magnitude(data_nr)
            PF.get_bus_angles(data) .= PF.get_bus_angles(data_nr)
            report = PF._ga_solve(pf, data, 1)
            @test report.converged
            @test report.stage_iterations == 1
            ga_compare(data_nr, data; tol = 1e-8)
        end

        @testset "non-zero REF angle" begin
            sys = ga_sys14()
            ref = only(
                collect(
                    PSY.get_components(
                        x -> PSY.get_bustype(x) == PSY.ACBusTypes.REF, PSY.ACBus, sys),
                ),
            )
            PSY.set_angle!(ref, 0.12345)
            ga_parity(sys)
        end

        @testset "loss factors match NR" begin
            data_nr, data_ga =
                ga_parity(GA_SYS14; pf_kwargs = (; calculate_loss_factors = true))
            @test maximum(abs.(data_nr.loss_factors .- data_ga.loss_factors)) < 1e-6
        end

        @testset "radial network reduction" begin
            ga_parity(
                GA_SYS14;
                pf_kwargs = (;
                    network_reductions = PNM.NetworkReduction[PNM.RadialReduction()]),
            )
        end

        @testset "large ACTIVSg2000: plain solve and NR polish" begin
            sys = PSB.build_system(PSB.MatpowerTestSystems, "matpower_ACTIVSg2000_sys")
            data_nr = PowerFlowData(
                ACPowerFlow{NewtonRaphsonACPowerFlow}(; correct_bustypes = true), sys)
            @test solve_power_flow!(data_nr)
            pf = ACPowerFlow{GA}(; correct_bustypes = true)
            data = PowerFlowData(pf, sys)
            report = PF._ga_solve(
                pf, data, 1; handoff_solver = NewtonRaphsonACPowerFlow,
                handoff_tol = 1e-3)
            @test report.converged
            ga_compare(data_nr, data)
            report_plain = PF._ga_solve(pf, PowerFlowData(pf, sys), 1)
            @test report_plain.converged
            @test report_plain.stage_exit === PF.GAConverged
            @test report_plain.stage_iterations <= 40
            @test iszero(report_plain.handoff_iterations)
            data_plain = PowerFlowData(pf, sys)
            @test solve_power_flow!(data_plain)
            ga_compare(data_nr, data_plain)
        end

        @testset "large ACTIVSg10k: plain solve from a true flat start" begin
            sys = PSB.build_system(PSB.MatpowerTestSystems, "matpower_ACTIVSg10k_sys")
            data_nr = PowerFlowData(
                ACPowerFlow{NewtonRaphsonACPowerFlow}(; correct_bustypes = true), sys)
            @test solve_power_flow!(data_nr)
            pf = ACPowerFlow{GA}(; correct_bustypes = true)
            data = PowerFlowData(pf, sys)
            ga_true_flat_start!(data)
            report = PF._ga_solve(pf, data, 1)
            @test report.converged
            @test report.stage_exit === PF.GAConverged
            @test report.stage_iterations <= 40
            ga_compare(data_nr, data)

            pf_nr = ACPowerFlow{NewtonRaphsonACPowerFlow}(;
                correct_bustypes = true, ga_flat_start = true)
            data_flat = PowerFlowData(pf_nr, sys)
            ga_true_flat_start!(data_flat)
            @test solve_power_flow!(data_flat; pf = pf_nr)
            ga_compare(data_nr, data_flat)
        end
    end

    @testset "components parity" begin
        @testset "ZIP loads" begin
            sys = ga_sys14()
            pq = ga_pq_buses(sys)
            # device base 10 MVA: 1.0 here = 0.1 p.u. on the 100 MVA system base
            _add_simple_zip_load!(
                sys,
                pq[1];
                constant_power_active_power = 0.5,
                constant_current_active_power = 1.0,
                constant_current_reactive_power = 0.4,
                constant_impedance_active_power = 2.0,
                constant_impedance_reactive_power = 0.8,
            )
            _add_simple_zip_load!(sys, pq[2]; constant_impedance_reactive_power = -1.0)
            ga_parity(sys)
        end

        @testset "three-winding transformer" begin
            sys = ga_sys14()
            pq = ga_pq_buses(sys)
            _add_simple_transformer_3w!(sys, pq[1], pq[2], pq[3], 1001)
            ga_parity(sys)
        end

        @testset "generic HVDC" begin
            ga_parity(PSB.build_system(PSB.MatpowerTestSystems, "matpower_case5_dc_sys"))
        end

        @testset "two islands with two REF buses" begin
            sys = PSB.build_system(PSB.PSITestSystems, "c_sys5"; add_forecasts = false)
            b101 = _add_simple_bus!(sys, 101, PSY.ACBusTypes.REF, 230, 1.0)
            b102 = _add_simple_bus!(sys, 102, PSY.ACBusTypes.PQ, 230, 1.0)
            b103 = _add_simple_bus!(sys, 103, PSY.ACBusTypes.PQ, 230, 1.0)
            _add_simple_source!(sys, b101)
            _add_simple_line!(sys, b101, b102, 0.01, 0.1, 0.02)
            _add_simple_line!(sys, b102, b103, 0.01, 0.1, 0.02)
            _add_simple_load!(sys, b102, 30.0, 10.0)
            _add_simple_load!(sys, b103, 20.0, 5.0)
            _, data_ga = ga_parity(sys)
            part = PF.GAPartition(data_ga, 1, Dict{Int, Float64}())
            @test length(part.s_ix) == 2
            @test part.n_islands == 2
            @test sort(unique(part.island_of_l)) == [1, 2]
        end
    end

    @testset "NR flat-start option" begin
        @test PF.get_ga_flat_start(ACPowerFlow(; ga_flat_start = true))
        @test !PF.get_ga_flat_start(ACPowerFlow())
        @test !PF.get_ga_flat_start(PF.ACRectangularPowerFlow())
        @test_throws MethodError PF.ACRectangularPowerFlow(; ga_flat_start = true)

        pf = ACPowerFlow{NewtonRaphsonACPowerFlow}(;
            ga_flat_start = true, solution_parameters = VSC_SOLUTION_PARAMETERS)
        data = PowerFlowData(pf, ga_vsc_ac_voltage_system())
        dcn = PF.get_dc_network(data)
        dc_before = (copy(dcn.p_c), copy(dcn.q_c), copy(dcn.node_vdc))
        slot = data.solver_cache[]
        x0 = PF.calculate_x0(data, 1)
        residual = PF.ACPowerFlowResidual(data, 1)
        residual(data, x0, 1)
        r0 = norm(residual.Rv, 1)
        x = PF._ga_flat_start(x0, data, residual, 1, 1e-3, PNM.KLUSolver())
        @test data.solver_cache[] === slot
        @test (dcn.p_c, dcn.q_c, dcn.node_vdc) == dc_before
        residual(data, x, 1)
        @test norm(residual.Rv, 1) < r0
        # The stage builds its cache with the backend it receives.
        @test_throws MethodError PF._ga_flat_start(x0, data, residual, 1, 1e-3, nothing)
        pf_klu = ACPowerFlow{NewtonRaphsonACPowerFlow}(;
            ga_flat_start = true,
            solution_parameters = SolutionParameters(; linear_solver = "KLU"))
        @test PF.resolve_linear_solver_backend(
            PF.get_solution_parameters(pf_klu).linear_solver) == PNM.KLUSolver()
        x_klu = PF.improve_x0(pf_klu, data, residual, 1)
        @test length(x_klu) == length(x0)
        @test solve_power_flow!(data; pf = pf)
        data_nr = PowerFlowData(
            ACPowerFlow{NewtonRaphsonACPowerFlow}(;
                solution_parameters = VSC_SOLUTION_PARAMETERS),
            ga_vsc_ac_voltage_system())
        @test solve_power_flow!(data_nr)
        ga_compare(data_nr, data)

        pf_fd =
            ACPowerFlow{FastDecoupledACPowerFlow}(; ga_flat_start = true, time_steps = 2)
        data_fd = PowerFlowData(pf_fd, GA_SYS14)
        data_fd.bus_active_power_withdrawals[:, 2] .*= 1.1
        data_fd.bus_angles .= 0.0
        @test solve_power_flow!(data_fd; pf = pf_fd)
        @test typeof(data_fd.solver_cache[]) !== PF.GeneralizedAdmittanceCache
    end

    @testset "handoff paths on c_sys14" begin
        pf = ACPowerFlow{GA}()
        data_nr = PowerFlowData(ACPowerFlow{NewtonRaphsonACPowerFlow}(), GA_SYS14)
        @test solve_power_flow!(data_nr)

        data = PowerFlowData(pf, GA_SYS14)
        polish = PF._ga_solve(
            pf, data, 1; handoff_solver = NewtonRaphsonACPowerFlow, handoff_tol = 1e-3)
        @test polish.converged
        @test polish.stage_exit === PF.GAConverged
        @test polish.handoff_iterations <= 4
        ga_compare(data_nr, data)

        rescue = PF._ga_solve(
            pf, PowerFlowData(pf, GA_SYS14), 1;
            maxIterations = 2, handoff_solver = NewtonRaphsonACPowerFlow,
            handoff_tol = 1e-12)
        @test rescue.stage_exit === PF.GAMaxIter
        @test rescue.handoff_iterations > 0
        @test rescue.converged

        data_fail = PowerFlowData(pf, GA_SYS14)
        fail = PF._ga_solve(pf, data_fail, 1; maxIterations = 2)
        @test !fail.converged
        @test all(isfinite, data_fail.bus_magnitude[:, 1])

        @test_throws ArgumentError PF._ga_solve(
            pf, PowerFlowData(pf, GA_SYS14), 1; handoff_solver = GradientDescentACPowerFlow,
        )
    end

    @testset "consistency check errors when the polar residual exceeds 10·tol" begin
        data = PowerFlowData(ACPowerFlow{GA}(), GA_SYS14)
        residual, _ = PF._ga_polar_state(data, 1)
        @test norm(residual.Rv, Inf) > 0.1
        @test_throws r"formulation bug" PF._ga_check_consistency(
            PF.GAConverged, PF.NoHandoff, residual, 1e-20)
        @test PF._ga_check_consistency(
            PF.GAMaxIter, PF.NoHandoff, residual, 1e-20) === nothing
        @test PF._ga_check_consistency(
            PF.GAConverged, NewtonRaphsonACPowerFlow, residual, 1e-20) ===
              nothing
        @test PF._ga_check_consistency(
            PF.GAConverged, PF.NoHandoff, residual, 1.0) === nothing
    end

    @testset "multi-period solve reuses the factorization cache" begin
        data_ga = PowerFlowData(ACPowerFlow{GA}(; time_steps = 3), GA_SYS14)
        data_nr = PowerFlowData(
            ACPowerFlow{NewtonRaphsonACPowerFlow}(; time_steps = 3), GA_SYS14)
        for (t, scale) in enumerate((1.0, 1.1, 0.9)), d in (data_ga, data_nr)
            d.bus_active_power_withdrawals[:, t] .*= scale
        end
        @test solve_power_flow!(data_nr)
        @test solve_power_flow!(data_ga)
        ga_compare(data_nr, data_ga)
        cache = data_ga.solver_cache[]
        @test typeof(cache) == PF.GeneralizedAdmittanceCache{typeof(cache.Fl)}
        @test typeof(cache.Fl) == typeof(cache.Fq)
        @test PF._get_or_build_ga_cache!(
            data_ga,
            PF.GAPartition(data_ga, 1, Dict{Int, Float64}()),
            PF.resolve_linear_solver_backend(nothing),
        ) === cache
        @test cache.factored
        part = PF.GAPartition(data_ga, 1, Dict{Int, Float64}())
        conv = PF.GAConverterTerms(size(PF.get_bus_type(data_ga), 1))
        y1 = PF._ga_initial_shunts(cache.blocks, PF.GANodalPower(data_ga, part, conv, 1),
            part, data_ga, 1)
        y2 = PF._ga_initial_shunts(cache.blocks, PF.GANodalPower(data_ga, part, conv, 2),
            part, data_ga, 2)
        @test maximum(abs.(y1 .- y2)) > 1e-3
        blocks = cache.blocks
        @test solve_power_flow!(data_ga)
        @test data_ga.solver_cache[].blocks === blocks
    end

    @testset "VSC DC substep" begin
        @testset "warm start holds AC-voltage Q and reproduces NR ($name)" for (
            name, build) in (
            ("droop", ga_vsc_droop_system), ("ac_voltage", ga_vsc_ac_voltage_system))
            sys = build()
            pf = ACPowerFlow{NewtonRaphsonACPowerFlow}(;
                solution_parameters = VSC_SOLUTION_PARAMETERS)
            data_nr = PowerFlowData(pf, sys)
            @test solve_power_flow!(data_nr)
            dn = PF.get_dc_network(data_nr)
            data = PowerFlowData(pf, sys)
            dcn = PF.get_dc_network(data)
            dcn.q_c[:, 1] .= dn.q_c[:, 1]
            PF._vsc_warm_start!(dcn, data_nr.bus_magnitude[:, 1], 1; tol = 1e-12)
            @test maximum(abs.(dcn.p_c[:, 1] .- dn.p_c[:, 1])) < 1e-8
            @test maximum(abs.(dcn.node_vdc[:, 1] .- dn.node_vdc[:, 1])) < 1e-8
            @test maximum(abs.(dcn.q_c[:, 1] .- dn.q_c[:, 1])) < 1e-8
        end

        @testset "parity with NR ($name)" for (name, build) in (
            ("pq",
                () -> _build_vsc_pq_system(;
                    g = 50.0, p_set = 0.4, q_set = 0.1, vdc = 1.05)),
            ("droop", ga_vsc_droop_system),
            ("ac_voltage", ga_vsc_ac_voltage_system),
            ("pv_terminal", _vsc_system_pv_terminal),
            ("ref_terminal", ga_vsc_system_ref_terminal),
            ("mtdc", _build_mtdc_system),
        )
            data_nr, data_ga = ga_parity(
                build();
                pf_kwargs = (; solution_parameters = VSC_SOLUTION_PARAMETERS),
            )
            dn, dg = PF.get_dc_network(data_nr), PF.get_dc_network(data_ga)
            @test maximum(abs.(dn.p_c .- dg.p_c)) < 1e-6
            @test maximum(abs.(dn.q_c .- dg.q_c)) < 1e-6
            @test maximum(abs.(dn.node_vdc .- dg.node_vdc)) < 1e-6
        end
    end

    @testset "LCC" begin
        @testset "closed form matches NR terminal powers" begin
            sys = ga_lcc_system()
            data_nr = PowerFlowData(ACPowerFlow{NewtonRaphsonACPowerFlow}(), sys)
            @test solve_power_flow!(data_nr)
            data = PowerFlowData(ACPowerFlow{NewtonRaphsonACPowerFlow}(), sys)
            n = size(PF.get_bus_type(data), 1)
            conv = PF.GAConverterTerms(n)
            PF._ga_add_lcc_terms!(conv, data, 1)
            p_nr = zeros(n)
            q_nr = zeros(n)
            for l in eachindex(data_nr.lcc.bus_indices)
                fb, tb = data_nr.lcc.bus_indices[l]
                Vf, Vt = data_nr.bus_magnitude[fb, 1], data_nr.bus_magnitude[tb, 1]
                Pf, Pt = PF._lcc_ac_active_powers(data_nr, l, 1, Vf, Vt)
                yf, yt = data_nr.lcc.branch_admittances[l]
                p_nr[fb] += Pf
                p_nr[tb] += Pt
                q_nr[fb] -= Vf^2 * imag(yf)
                q_nr[tb] -= Vt^2 * imag(yt)
            end
            @test maximum(abs.(conv.p_lcc .- p_nr)) < 1e-8
            @test maximum(abs.(conv.q_lcc .- q_nr)) < 1e-8
        end

        @testset "infeasible operating point errors" begin
            @test_throws r"LCC 1 rectifier is infeasible" PF._ga_lcc_q(
                1.0,
                1.0,
                5.0,
                1,
                "rectifier",
            )
            @test PF._ga_lcc_q(2.0, 1.0, 0.0, 1, "inverter") ≈ 2.0 * PF.SQRT6_DIV_PI
        end

        @testset "parity with NR, zero_setpoint = $zero_setpoint" for zero_setpoint in
                                                                      (false, true)
            sys = ga_lcc_system()
            if zero_setpoint
                PSY.set_power_transfer_setpoint!(
                    first(PSY.get_components(PSY.TwoTerminalLCCLine, sys)),
                    0.0 * u"CU",
                )
            end
            data_nr, data_ga = ga_parity(sys)
            @test maximum(abs.(data_nr.lcc.rectifier.tap .- data_ga.lcc.rectifier.tap)) <
                  1e-6
            @test maximum(abs.(data_nr.lcc.inverter.tap .- data_ga.lcc.inverter.tap)) < 1e-6
        end

        @testset "plain solve, and NR handoff rescues after the iteration cap" begin
            pf = ACPowerFlow{GA}()
            data_nr =
                PowerFlowData(ACPowerFlow{NewtonRaphsonACPowerFlow}(), ga_lcc_system())
            @test solve_power_flow!(data_nr)
            plain = PF._ga_solve(pf, PowerFlowData(pf, ga_lcc_system()), 1)
            @test plain.converged
            @test plain.stage_iterations <= 60
            data = PowerFlowData(pf, ga_lcc_system())
            report = PF._ga_solve(
                pf, data, 1; maxIterations = 3,
                handoff_solver = NewtonRaphsonACPowerFlow,
                handoff_tol = 1e-3)
            @test report.stage_exit === PF.GAMaxIter
            @test report.stage_iterations == 3
            @test report.handoff_iterations > 0
            @test report.converged
            ga_compare(data_nr, data)
        end
    end

    @testset "multiple VSC converters at one AC bus" begin
        function two_vsc_system(av_first::Bool)
            sys = ga_sys14()
            pq = ga_pq_buses(sys)
            for (name, to_bus, av, p_to) in (
                ("ga_vsc_shared_1", pq[2], av_first, 0.25),
                ("ga_vsc_shared_2", pq[3], false, 0.15),
            )
                ga_kw = (;
                    dc_control_from = PSY.VSCDCControlModes.DC_VOLTAGE,
                    dc_voltage_setpoint_from = 1.05,
                    dc_control_to = PSY.VSCDCControlModes.DC_POWER,
                    dc_power_setpoint_to = p_to,
                    reactive_power_from = 0.1,
                    reactive_power_to = 0.0,
                )
                if av
                    ga_kw = (; ga_kw...,
                        ac_control_from = PSY.VSCACControlModes.AC_VOLTAGE,
                        ac_voltage_setpoint_from = 1.01)
                end
                ga_add_vsc!(sys, name; to_bus = to_bus, ga_kw...)
            end
            return sys
        end

        @testset "$name" for (name, av_first) in
                             (("two Q-mode converters", false),
            ("AC-voltage and Q-mode converters", true))
            data_nr, data_ga = ga_parity(
                two_vsc_system(av_first);
                pf_kwargs = (; solution_parameters = VSC_SOLUTION_PARAMETERS),
            )
            dn, dg = PF.get_dc_network(data_nr), PF.get_dc_network(data_ga)
            @test count(==(first(dg.converter_ac_bus_ix)), dg.converter_ac_bus_ix) == 2
            @test maximum(abs.(dn.q_c .- dg.q_c)) < 1e-6
            @test maximum(abs.(dn.p_c .- dg.p_c)) < 1e-6
        end
    end

    @testset "every available backend matches KLU" begin
        sys = ga_sys14()
        backends = String[]
        if PF.PNM._has_apple_accelerate_backend()
            push!(backends, "AppleAccelerateLU")
        end
        if PF.PNM._has_mkl_pardiso_ext() && Pardiso.mkl_is_available()
            push!(backends, "MKLPardiso")
        end
        ga(name) = ACPowerFlow{GeneralizedAdmittanceACPowerFlow}(;
            solution_parameters = SolutionParameters(; linear_solver = name))
        res_klu = solve_power_flow(ga("KLU"), sys)
        for name in backends
            res = solve_power_flow(ga(name), sys)
            @test isapprox(
                res["bus_results"][!, :Vm], res_klu["bus_results"][!, :Vm]; atol = 1e-7)
            @test isapprox(
                res["bus_results"][!, :θ], res_klu["bus_results"][!, :θ]; atol = 1e-7)
        end
    end

    @testset "singular check catches a floating block" begin
        y = 1.0 - 10.0im
        L = SparseArrays.spzeros(ComplexF64, 4, 4)
        for k in 1:4
            j = mod1(k + 1, 4)
            L[k, k] += y
            L[j, j] += y
            L[k, j] -= y
            L[j, k] -= y
        end
        names = String[]
        if PF.PNM._has_apple_accelerate_backend()
            push!(names, "AppleAccelerateLU")
        end
        if PF.PNM._has_mkl_pardiso_ext() && Pardiso.mkl_is_available()
            push!(names, "MKLPardiso")
        end
        for name in names
            F = PF.make_linear_solver_cache(PF.resolve_linear_solver_backend(name), L)
            ok = try
                PF.full_factor!(F, L)
                PF._ga_factor_ok(F, L)
            catch e
                @test typeof(e) == LinearAlgebra.SingularException ||
                      occursin("Pardiso", string(typeof(e)))
                false
            end
            @test !ok
        end
    end
end
