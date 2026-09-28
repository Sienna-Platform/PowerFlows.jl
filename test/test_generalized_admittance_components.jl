# GA component parity (spec §3.7, §3.9): ZIP loads, three-winding transformers,
# generic two-terminal HVDC, and LCC must match the polar NR solver.

@testset "GA: ZIP loads parity" begin
    sys = PSB.build_system(PSB.PSITestSystems, "c_sys14"; add_forecasts = false)
    pq = sort!(
        collect(
            PSY.get_components(
                b -> PSY.get_bustype(b) == PSY.ACBusTypes.PQ, PSY.ACBus, sys),
        ); by = PSY.get_number)
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

@testset "GA: three-winding transformer parity" begin
    sys = PSB.build_system(PSB.PSITestSystems, "c_sys14"; add_forecasts = false)
    pq = sort!(
        collect(
            PSY.get_components(
                b -> PSY.get_bustype(b) == PSY.ACBusTypes.PQ, PSY.ACBus, sys),
        ); by = PSY.get_number)
    _add_simple_transformer_3w!(sys, pq[1], pq[2], pq[3], 1001)
    ga_parity(sys)
end

@testset "GA: generic HVDC parity" begin
    ga_parity(PSB.build_system(PSB.MatpowerTestSystems, "matpower_case5_dc_sys"))
end

@testset "GA: LCC parity, including a zero transfer setpoint" begin
    for zero_setpoint in (false, true)
        sys = system_from_openapi(
            PFP.PowerModelsData(joinpath(TEST_DATA_DIR, "case5_2_lcc.raw"));
            runchecks = false,
        )
        if zero_setpoint
            PSY.set_transfer_setpoint!(
                first(PSY.get_components(PSY.TwoTerminalLCCLine, sys)),
                0.0,
            )
        end
        # This network's GA fixed point contracts linearly at ~0.995/iter even with all
        # LCC terms zeroed (resistive near-short branches + large PV reactive
        # flow-through), so reaching the no-handoff stage tolerance needs ~4300
        # iterations — far past DEFAULT_GA_MAX_ITER. Same sanctioned pattern as the
        # plan's `maxIterations = 2000` ACTIVSg2000 cases; constants are not tuned.
        data_nr, data_ga = ga_parity(sys; ga_solve_kwargs = (; maxIterations = 6000))
        @test maximum(abs.(data_nr.lcc.rectifier.tap .- data_ga.lcc.rectifier.tap)) < 1e-6
        @test maximum(abs.(data_nr.lcc.inverter.tap .- data_ga.lcc.inverter.tap)) < 1e-6
    end
end
