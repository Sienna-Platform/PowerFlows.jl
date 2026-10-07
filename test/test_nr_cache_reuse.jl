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
    qc = 2n + 2c
    col = 2 * ix - 1

    residual = PowerFlows.ACPowerFlowResidual(data, 1)
    jac = PowerFlows.ACPowerFlowJacobian(data, residual, 1)
    x = PowerFlows.calculate_x0(data, 1)
    residual(data, x, 1)
    jac(data, 1)
    @test jac.Jv[vk, col] == 0.0   # fresh PV build: structural zero
    @test jac.Jv[qc, col] == 0.0

    data.bus_type[ix, 1] = PSY.ACBusTypes.PQ
    jac(data, 1)
    @test jac.Jv[vk, col] != 0.0   # PQ: the loss-coupling derivative now enters

    data.bus_type[ix, 1] = PSY.ACBusTypes.PV
    jac(data, 1)   # refilled IN PLACE, same Jacobian object, as a reused cache would do
    @test jac.Jv[vk, col] == 0.0   # must match a fresh PV build, not the stale PQ value
    @test jac.Jv[qc, col] == 0.0
end

@testset "NR cache reuse: LCC Jacobian ∂F_tap/∂V slots clear when a terminal bus returns to PV" begin
    # A Jacobian refilled in place across a PV→PQ→PV flip matches a fresh PV build.
    sys, _ = simple_lcc_system()
    pf = ACPowerFlow{NewtonRaphsonACPowerFlow}(; correct_bustypes = true)
    data = PowerFlowData(pf, sys)
    terminals = collect(Iterators.flatten(data.lcc.bus_indices))
    function set_terminals!(bus_type)
        for ix in terminals
            data.bus_type[ix, 1] = bus_type
        end
        return
    end

    set_terminals!(PSY.ACBusTypes.PV)
    residual = PowerFlows.ACPowerFlowResidual(data, 1)
    jac = PowerFlows.ACPowerFlowJacobian(data, residual, 1)
    x = PowerFlows.calculate_x0(data, 1)
    residual(data, x, 1)
    jac(data, 1)
    J_pv = copy(SparseArrays.nonzeros(jac.Jv))

    set_terminals!(PSY.ACBusTypes.PQ)
    jac(data, 1)
    @test SparseArrays.nonzeros(jac.Jv) != J_pv   # PQ: the bus-V derivatives now enter

    set_terminals!(PSY.ACBusTypes.PV)
    jac(data, 1)   # refilled IN PLACE, same Jacobian object, as a reused cache would do
    @test SparseArrays.nonzeros(jac.Jv) ≈ J_pv
end

@testset "Singular-Jacobian fallback matrix and factorization reuse" begin
    # The fallback matrix `M` shares its pattern with `Jᵀ*J`, so a later Jacobian with the
    # same structural pattern but different values refreshes `M` in place.
    sys = PSB.build_system(PSB.PSITestSystems, "c_sys14")
    pf = ACPowerFlow{NewtonRaphsonACPowerFlow}(; correct_bustypes = true)
    data = PowerFlowData(pf, sys)
    residual = PowerFlows.ACPowerFlowResidual(data, 1)
    jac = PowerFlows.ACPowerFlowJacobian(data, residual, 1)
    x = PowerFlows.calculate_x0(data, 1)
    residual(data, x, 1)
    jac(data, 1)

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

@testset "solver objects do not store data" begin
    for T in (PF.ACPowerFlowResidual, PF.ACPowerFlowJacobian, PF.ACRectangularCIResidual,
        PF.ACRectangularCIJacobian, PF.ACMixedCPBResidual, PF.ACMixedCPBJacobian,
        PF.HomotopyHessian)
        @test !hasfield(T, :data)
    end
    sys = PSB.build_system(PSB.PSITestSystems, "c_sys14")
    data = PowerFlowData(ACPowerFlow{NewtonRaphsonACPowerFlow}(), sys)
    residual = PowerFlows.ACPowerFlowResidual(data, 1)
    J = PowerFlows.ACPowerFlowJacobian(data, residual, 1)
    x0 = PowerFlows.calculate_x0(data, 1)
    residual(data, x0, 1)
    J(data, 1)
    @test (@allocated residual(data, x0, 1)) == 0
    @test (@allocated J(data, 1)) == 0
end

function _solve_logged!(data)
    logs, converged = Test.collect_test_logs(; min_level = Logging.Debug) do
        solve_power_flow!(data)
    end
    return converged, logs
end

_count_logs(logs, level, pattern) =
    count(l -> l.level == level && occursin(pattern, string(l.message)), logs)

# A visible PQ→PV flip drops the KLU Numeric. `hide_flip` updates the cache's bus-type snapshot,
# so the solve refactors on the stale PQ pivot order and must use the re-pivot guard.
function _stale_pivot_flip(pf, sys, bus_number; hide_flip = false)
    data = PowerFlowData(pf, sys)
    i = PF.get_bus_lookup(data)[bus_number]
    setpoint = data.bus_magnitude[i, 1]
    data.bus_type[i, 1] = PSY.ACBusTypes.PQ
    converged_pq, _ = _solve_logged!(data)
    cache = data.polar_nr_cache[]
    data.bus_type[i, 1] = PSY.ACBusTypes.PV
    data.bus_magnitude[i, 1] = setpoint
    if hide_flip
        copyto!(cache.bus_type_snapshot, view(data.bus_type, :, 1))
    end
    converged, logs = _solve_logged!(data)
    return data, converged_pq, converged, logs, cache
end

@testset "stale KLU pivot after PQ→PV re-pivots" begin
    cases = (
        (
            "c_sys14",
            PSB.PSITestSystems,
            "c_sys14",
            (6, 3, 2),
            false,
            (; add_forecasts = false),
        ),
        (
            "ACTIVSg2000",
            PSB.MatpowerTestSystems,
            "matpower_ACTIVSg2000_sys",
            (5065,),
            true,
            (;),
        ),
    )
    for (label, set, name, buses, correct, kwargs) in cases
        sys = PSB.build_system(set, name; kwargs...)
        for ACSolver in (NewtonRaphsonACPowerFlow, TrustRegionACPowerFlow),
            bus_number in buses, hide_flip in (false, true)

            @testset "$label bus $bus_number $ACSolver hide_flip=$hide_flip" begin
                pf = ACPowerFlow{ACSolver}(;
                    check_reactive_power_limits = false,
                    correct_bustypes = correct,
                    solution_parameters = SolutionParameters(; linear_solver = "KLU"),
                )
                data, converged_pq, converged, logs, c0 =
                    _stale_pivot_flip(pf, sys, bus_number; hide_flip)
                @test converged_pq
                @test PF._repivots(c0.linSolveCache)
                @test data.polar_nr_cache[] === c0
                @test (_count_logs(logs, Logging.Debug, "stale KLU pivot order") > 0) ==
                      hide_flip
                @test converged
                @test iszero(_count_logs(logs, Logging.Warn, "Jacobian is singular"))

                fresh = PowerFlowData(pf, sys)
                @test solve_power_flow!(fresh)
                @test isapprox(data.bus_magnitude, fresh.bus_magnitude; atol = 1e-8)
                @test isapprox(data.bus_angles, fresh.bus_angles; atol = 1e-8)
            end
        end
    end
end

@testset "NR cache reuse: a bridge outage keeps J; a new slack slot rebuilds" begin
    sys = PSB.build_system(PSB.PSITestSystems, "c_sys14"; add_forecasts = false)
    sp = SolutionParameters(; linear_solver = "KLU")
    function solved(pf, edit!)
        data = PowerFlowData(pf, sys)
        @test solve_power_flow!(data)
        c0 = data.polar_nr_cache[]
        edit!(data)
        data.bus_active_power_withdrawals[:, 1] .*= 1.01
        @test solve_power_flow!(data)
        # Same two solves with the cache dropped in between: the first solve writes the slack
        # distribution back into the injections, so a never-solved `data` is not the reference.
        fresh = PowerFlowData(pf, sys)
        @test solve_power_flow!(fresh)
        fresh.polar_nr_cache[] = nothing
        fresh.ac_jacobian_structure_cache[] = nothing
        edit!(fresh)
        fresh.bus_active_power_withdrawals[:, 1] .*= 1.01
        @test solve_power_flow!(fresh)
        @test isapprox(data.bus_magnitude, fresh.bus_magnitude; atol = 1e-10)
        @test isapprox(data.bus_angles, fresh.bus_angles; atol = 1e-10)
        return data, c0
    end

    pf = ACPowerFlow{NewtonRaphsonACPowerFlow}(;
        correct_bustypes = true, solution_parameters = sp)
    # Trans4 (7-8) is bus 8's only connection: its outage islands bus 8, which becomes its
    # own REF.
    function outage_trans4!(d)
        ybus = d.power_network_matrix
        i, j = PF.get_bus_lookup(d)[7], PF.get_bus_lookup(d)[8]
        a = PNM.get_arc_lookup(ybus.arc_admittance_from_to)[(7, 8)]
        ybus.data[i, i] -= ybus.arc_admittance_from_to.data[a, i]
        ybus.data[j, j] -= ybus.arc_admittance_to_from.data[a, j]
        ybus.data[i, j] = 0
        ybus.data[j, i] = 0
        d.bus_type[j, 1] = PSY.ACBusTypes.REF
        d.bus_slack_participation_factors[j, 1] = 1.0
        PF._invalidate_partition!(d)
        return
    end
    data, c0 = solved(pf, outage_trans4!)
    @test data.polar_nr_cache[] === c0
    @test length(c0.residual.subnetworks) == 2
    @test c0.bus_type_snapshot == data.bus_type[:, 1]

    # A participant off the REF's neighbourhood that had no factor has no slot: rebuild.
    probe = PowerFlowData(pf, sys)
    ref = only(findall(==(PSY.ACBusTypes.REF), probe.bus_type[:, 1]))
    k = findfirst(
        i -> probe.bus_type[i, 1] == PSY.ACBusTypes.PV && !(ref in probe.neighbors[i]),
        axes(probe.bus_type, 1),
    )
    data_n, c1 = solved(pf, d -> (d.bus_slack_participation_factors[k, 1] = 1.0))
    @test data_n.polar_nr_cache[] !== c1

    # Slots span every bus with a factor, whatever its type: dropping a participant's factor
    # or flipping it to PQ keeps the pattern.
    factors = Dict{Tuple{DataType, String}, Float64}(
        (typeof(g), PSY.get_name(g)) => 1.0 for
        g in PSY.get_components(PSY.Generator, sys)
    )
    pf_d = ACPowerFlow{NewtonRaphsonACPowerFlow}(;
        correct_bustypes = true, solution_parameters = sp,
        generator_slack_participation_factors = factors)
    data_d, c2 = solved(pf_d, d -> (d.bus_slack_participation_factors[k, 1] = 0.0))
    @test data_d.polar_nr_cache[] === c2
    data_q, c3 = solved(pf_d, d -> (d.bus_type[k, 1] = PSY.ACBusTypes.PQ))
    @test data_q.polar_nr_cache[] === c3
    # The outage orphans bus 8's slot to the main REF, so the refresh must zero it.
    data_b, c4 = solved(pf_d, outage_trans4!)
    @test data_b.polar_nr_cache[] === c4
    fresh_res = PF.ACPowerFlowResidual(data_b, 1)
    fresh_J = PF.ACPowerFlowJacobian(data_b, fresh_res, 1)
    fresh_J(data_b, 1)
    c4.J(data_b, 1)
    @test isapprox(Matrix(c4.J.Jv), Matrix(fresh_J.Jv); atol = 1e-12)
end

@testset "_pick_better_x0 leaves data and residual at the returned x0" begin
    # It reads the residual at x0 from `residual.Rv` instead of re-evaluating, so on both the
    # accept and the reject path a fresh evaluation at the returned x0 must reproduce `Rv`.
    sys = PSB.build_system(PSB.PSITestSystems, "c_sys14"; add_forecasts = false)
    pf = ACPowerFlow{NewtonRaphsonACPowerFlow}()
    solved = PowerFlowData(pf, sys)
    @test solve_power_flow!(solved)
    x_solved = PF.calculate_x0(solved, 1)

    data = PowerFlowData(pf, sys)
    residual = PF.ACPowerFlowResidual(data, 1)
    x0 = PF.calculate_x0(data, 1)
    residual(data, x0, 1)
    @test PF._pick_better_x0(x0, copy(x_solved), 1, residual, data, "solved point")
    @test x0 == x_solved
    rv = copy(residual.Rv)
    residual(data, x0, 1)
    @test residual.Rv == rv

    worse = x0 .+ 0.3
    @test !PF._pick_better_x0(x0, worse, 1, residual, data, "worse point")
    @test x0 == x_solved
    rv = copy(residual.Rv)
    residual(data, x0, 1)
    @test residual.Rv == rv
end
