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

            # Fresh from-scratch reference solve of the same config must agree.
            data_ref = PowerFlows.PowerFlowData(pf, sys)
            converged_ref = PowerFlows._ac_power_flow(data_ref, pf, 1)
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
