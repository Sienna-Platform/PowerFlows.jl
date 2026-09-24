# Safety-net regression tests for a later cache-reuse refactor of the polar NR/TR
# path. They assert behavior that already holds on the current code: reusing the
# (eventual) cache across Q-limit retries and across time steps must not change
# the converged voltages versus a from-scratch solve.

@testset "NR cache reuse: PV→PQ flip across Q-limit retries" begin
    for ACSolver in (NewtonRaphsonACPowerFlow, TrustRegionACPowerFlow)
        @testset "AC Solver: $(ACSolver)" begin
            sys = PSB.build_system(PSB.PSITestSystems, "c_sys14"; add_forecasts = false)
            pf = ACPowerFlow{ACSolver}(;
                check_reactive_power_limits = true,
                correct_bustypes = true,
            )

            data = PowerFlows.PowerFlowData(pf, sys)
            # Capture the bus types before solving so we can prove the flip path ran.
            original_bus_types = deepcopy(data.bus_type[:, 1])

            converged = PowerFlows._ac_power_flow(data, pf, 1)
            @test converged
            x = _calc_x(data, 1)

            # A PV bus must have flipped to PQ during the Q-limit retry loop
            # (generator on "Bus8" violates Q-max — see test_solve_power_flow.jl).
            @test any(data.bus_type[:, 1] .!= original_bus_types)

            # Reference: replicate the retry loop on a fresh `data_ref`, forcing
            # `polar_nr_cache` to `nothing` before every retry so it never reuses.
            data_ref = PowerFlows.PowerFlowData(pf, sys)
            converged_ref = false
            for _ in 1:(PowerFlows.MAX_REACTIVE_POWER_ITERATIONS)
                data_ref.polar_nr_cache[] = nothing
                converged_ref = PowerFlows._newton_power_flow(pf, data_ref, 1)
                if !converged_ref || !PowerFlows.get_check_reactive_power_limits(pf) ||
                   PowerFlows._check_q_limit_bounds!(data_ref, 1)
                    break
                end
            end
            @test converged_ref
            x_ref = _calc_x(data_ref, 1)

            @test isapprox(x, x_ref; atol = 1e-8)
            @test isapprox(
                data.bus_magnitude[:, 1],
                data_ref.bus_magnitude[:, 1];
                atol = 1e-8,
            )
            @test isapprox(data.bus_angles[:, 1], data_ref.bus_angles[:, 1]; atol = 1e-8)
        end
    end
end

@testset "NR cache reuse: multi-period equals per-step fresh solves" begin
    for ACSolver in (NewtonRaphsonACPowerFlow, TrustRegionACPowerFlow)
        @testset "AC Solver: $(ACSolver)" begin
            time_steps = 24

            # Multi-period solve over all steps at once (varies injections per step
            # via the c_sys14 timeseries CSVs, exactly like test_multiperiod_ac_power_flow.jl).
            sys = PSB.build_system(PSB.PSITestSystems, "c_sys14"; add_forecasts = false)
            pf = ACPowerFlow{ACSolver}(; time_steps = time_steps)
            data = PowerFlowData(pf, sys)
            prepare_ts_data!(data, time_steps)
            @test solve_power_flow!(data)

            # For each step, build a fresh single-step data carrying that step's
            # injections/withdrawals and solve it independently.
            for t in 1:time_steps
                sys_t =
                    PSB.build_system(PSB.PSITestSystems, "c_sys14"; add_forecasts = false)
                pf_t = ACPowerFlow{ACSolver}()
                data_t = PowerFlowData(pf_t, sys_t)
                data_t.bus_active_power_injections[:, 1] .=
                    data.bus_active_power_injections[:, t]
                data_t.bus_active_power_withdrawals[:, 1] .=
                    data.bus_active_power_withdrawals[:, t]
                data_t.bus_reactive_power_injections[:, 1] .=
                    data.bus_reactive_power_injections[:, t]
                data_t.bus_reactive_power_withdrawals[:, 1] .=
                    data.bus_reactive_power_withdrawals[:, t]

                @test PowerFlows._ac_power_flow(data_t, pf_t, 1)

                @test isapprox(
                    data.bus_magnitude[:, t],
                    data_t.bus_magnitude[:, 1];
                    atol = 1e-8,
                )
                @test isapprox(data.bus_angles[:, t], data_t.bus_angles[:, 1]; atol = 1e-8)
            end
        end
    end
end

@testset "NR cache reuse: LCC systems reuse the polar workspace across time steps" begin
    # An LCC system reuses the polar workspace (J, linSolveCache) across time steps.
    sys, _ = simple_lcc_system()
    time_steps = 6
    pf = ACPowerFlow{NewtonRaphsonACPowerFlow}(;
        time_steps = time_steps,
        correct_bustypes = true,
    )
    data = PowerFlowData(pf, sys)
    # `prepare_ts_data!` is c_sys14-specific (hardcoded CSV shape); this system's per-step
    # data starts as every column equal to the snapshot, so perturb each step distinctly
    # so every step must actually iterate.
    for t in 1:time_steps
        data.bus_active_power_injections[:, t] .*= (1.0 + 0.001 * t)
    end
    @test solve_power_flow!(data)
    cache = data.polar_nr_cache[]
    @test !isnothing(cache)
    J_before, lsc_before = cache.J, cache.linSolveCache

    data.bus_active_power_injections[:, time_steps] .*= 1.0007
    @test PowerFlows._newton_power_flow(pf, data, time_steps)
    cache_after = data.polar_nr_cache[]
    @test cache_after.J === J_before
    @test cache_after.linSolveCache === lsc_before
end

@testset "NR cache reuse: VSC Jacobian ∂KCL/∂|V_ac| slot clears when the bus returns to PV" begin
    # A Jacobian refilled in place across a PV→PQ→PV flip matches a fresh PV build.
    sys = _vsc_system_pv_terminal()
    pf = ACPowerFlow{NewtonRaphsonACPowerFlow}(;
        solution_parameters = VSC_SOLUTION_PARAMETERS,
    )
    data = PowerFlowData(pf, sys)
    dcn = PowerFlows.get_dc_network(data)
    n = first(size(data.bus_type))
    nconv = PowerFlows.n_vsc_converters(dcn)
    base = 2n + 2nconv
    c = findfirst(
        cc -> data.bus_type[dcn.converter_ac_bus_ix[cc], 1] == PSY.ACBusTypes.PV,
        1:nconv)
    @test !isnothing(c)
    ix = dcn.converter_ac_bus_ix[c]
    vk = base + dcn.converter_dc_node_ix[c]
    col = 2 * ix - 1

    residual = PowerFlows.ACPowerFlowResidual(data, 1)
    jac = PowerFlows.ACPowerFlowJacobian(residual, 1)
    x = PowerFlows.calculate_x0(data, 1)
    residual(x, 1)
    jac(1)
    @test jac.Jv[vk, col] == 0.0   # fresh PV build: structural zero

    data.bus_type[ix, 1] = PSY.ACBusTypes.PQ
    jac(1)
    @test jac.Jv[vk, col] != 0.0   # PQ: the loss-coupling derivative now enters

    data.bus_type[ix, 1] = PSY.ACBusTypes.PV
    jac(1)   # refilled IN PLACE, same Jacobian object, as a reused cache would do
    @test jac.Jv[vk, col] == 0.0   # must match a fresh PV build, not the stale PQ value
end

@testset "Singular-Jacobian fallback matrix and factorization reuse" begin
    # The fallback matrix `M` shares its pattern with `Jᵀ*J`, so a later Jacobian with the
    # same structural pattern but different values refreshes `M` in place.
    sys = PSB.build_system(PSB.PSITestSystems, "c_sys14")
    pf = ACPowerFlow{NewtonRaphsonACPowerFlow}(; correct_bustypes = true)
    data = PowerFlowData(pf, sys)
    residual = PowerFlows.ACPowerFlowResidual(data, 1)
    jac = PowerFlows.ACPowerFlowJacobian(residual, 1)
    x = PowerFlows.calculate_x0(data, 1)
    residual(x, 1)
    jac(1)

    M = PowerFlows._build_singular_J_fallback(jac.Jv, x)
    F = jac.Jv' * jac.Jv
    @test SparseArrays.nnz(M) == SparseArrays.nnz(F)
    @test M.colptr == F.colptr && M.rowval == F.rowval

    # A later Newton iterate has the SAME structural pattern but different numeric values.
    Jv2 = copy(jac.Jv)
    SparseArrays.nonzeros(Jv2) .+= 1e-4 .* Random.randn(SparseArrays.nnz(Jv2))
    @test PowerFlows._refresh_singular_J_fallback!(copy(M), Jv2, x)
end

@testset "Discrete-control symbolic factorization count is honest" begin
    # Each real symbolic factorization in the continuation is counted exactly once.
    sys = _make_solvable_tap_shunt_system()
    pf = ACPolarPowerFlow(; control_discrete_devices = true)
    data = PowerFlowData(pf, sys)
    @test solve_power_flow!(data)
    @test PowerFlows.get_control_symbolic_factor_count(data) == 2
end

@testset "NR cache reuse: rectangular/mixed formulations reuse the linear-solver cache" begin
    for (FormulationT, label) in (
        (ACRectangularPowerFlow, "rectangular"), (ACMixedPowerFlow, "mixed"),
    )
        @testset "$label" begin
            sys = PSB.build_system(PSB.PSITestSystems, "c_sys14"; add_forecasts = false)
            pf = FormulationT{NewtonRaphsonACPowerFlow}(; correct_bustypes = true)
            data = PowerFlowData(pf, sys)
            @test solve_power_flow!(data)
            cache = data.solver_cache[]
            @test typeof(cache).name.wrapper === PowerFlows.RectMixedNRCache
            lsc_before = cache.linSolveCache
            sv_before = cache.stateVector

            # Bus types (hence the Jacobian pattern) are unchanged by an injection-only
            # perturbation, so the symbolic factorization and state-vector buffers are reused.
            data.bus_active_power_injections[:, 1] .*= 1.001
            @test solve_power_flow!(data)
            cache_after = data.solver_cache[]
            @test cache_after === cache
            @test cache_after.linSolveCache === lsc_before
            @test cache_after.stateVector === sv_before

            # A PV→PQ Q-limit flip changes the per-bus block size (rect PV is 3 slots vs PQ's 2),
            # which must invalidate the cache rather than reuse a mismatched pattern.
            pf_q = FormulationT{NewtonRaphsonACPowerFlow}(;
                correct_bustypes = true, check_reactive_power_limits = true)
            sys_q = PSB.build_system(PSB.PSITestSystems, "c_sys14"; add_forecasts = false)
            data_q = PowerFlowData(pf_q, sys_q)
            original_bus_types = deepcopy(data_q.bus_type[:, 1])
            @test PowerFlows._ac_power_flow(data_q, pf_q, 1)
            @test any(data_q.bus_type[:, 1] .!= original_bus_types)
            @test typeof(data_q.solver_cache[]).name.wrapper === PowerFlows.RectMixedNRCache
        end
    end
end

@testset "Polar residual and Jacobian hold a concretely typed data field" begin
    sys = PSB.build_system(PSB.PSITestSystems, "c_sys14")
    pf = ACPowerFlow{NewtonRaphsonACPowerFlow}(; correct_bustypes = true)
    data = PowerFlowData(pf, sys)
    residual = PowerFlows.ACPowerFlowResidual(data, 1)
    J = PowerFlows.ACPowerFlowJacobian(residual, 1)
    @test isconcretetype(fieldtype(typeof(residual), :data))
    @test isconcretetype(fieldtype(typeof(J), :data))
    x0 = PowerFlows.calculate_x0(data, 1)
    residual(x0, 1)
    J(1)
    @test (@allocated residual(x0, 1)) == 0
    @test (@allocated J(1)) == 0
end
