@testset "DC distributed slack: constructor options" begin
    gspf = Dict((ThermalStandard, "gen") => 1.0)

    for T in (DCPowerFlow, PTDFDCPowerFlow, vPTDFDCPowerFlow)
        pf = T(; generator_slack_participation_factors = gspf)
        @test PF.get_slack_participation_factors(pf) === gspf
        @test PF.get_distribute_slack_proportional_to_headroom(pf) == false

        pf_hr = T(; distribute_slack_proportional_to_headroom = true)
        @test PF.get_distribute_slack_proportional_to_headroom(pf_hr) == true
        @test PF.get_slack_participation_factors(pf_hr) === nothing

        pf_default = T()
        @test PF.get_slack_participation_factors(pf_default) === nothing
        @test PF.get_distribute_slack_proportional_to_headroom(pf_default) == false

        @test_throws ErrorException T(;
            generator_slack_participation_factors = gspf,
            distribute_slack_proportional_to_headroom = true,
        )
    end
end

function _dc_subnetworks(data)
    bus_lookup = PF.get_bus_lookup(data)
    subnetworks = Dict{Int, Vector{Int}}()
    for (ref_bus, ax) in PF.subnetwork_axes(data)
        rows = [bus_lookup[ref_bus]]
        append!(rows, [bus_lookup[b] for b in first(ax)])
        subnetworks[bus_lookup[ref_bus]] = rows
    end
    return subnetworks
end

@testset "DC distributed slack: injection distribution ($T)" for T in
                                                                 (DCPowerFlow,
    PTDFDCPowerFlow, vPTDFDCPowerFlow)
    sys = PSB.build_system(PSB.PSITestSystems, "c_sys14"; add_forecasts = false)
    generators = collect(get_components(ThermalStandard, sys))

    Random.seed!(0)
    factor_values = abs.(randn(Float64, length(generators)) .+ 0.1)
    gspf = Dict(
        (ThermalStandard, get_name(g)) => v for (g, v) in zip(generators, factor_values)
    )

    pf = T(; generator_slack_participation_factors = gspf)
    data = PowerFlowData(pf, sys)
    init_p = copy(data.bus_active_power_injections)
    solve_power_flow!(data)

    subnetworks = _dc_subnetworks(data)
    _check_distributed_slack_consistency(
        subnetworks,
        data.bus_active_power_injections[:, 1],
        collect(data.bus_slack_participation_factors[:, 1]),
        init_p[:, 1],
    )

    for (_, buses) in subnetworks
        net = sum(
            data.bus_active_power_injections[buses, 1] .-
            data.bus_active_power_withdrawals[buses, 1] .+
            data.bus_hvdc_net_power[buses, 1],
        )
        @test isapprox(net, 0.0; atol = 1e-8)
    end
end

@testset "DC distributed slack: default run does not activate distribution" begin
    sys = PSB.build_system(PSB.PSITestSystems, "c_sys14"; add_forecasts = false)
    data = PowerFlowData(DCPowerFlow(), sys)
    # No distributed-slack configuration: the computed per-time-step factors stay empty,
    # so _distribute_dc_slack! is a no-op. The reference bus injection may still be
    # adjusted by the single-slack balance correction, so we only check the opt-in signal.
    @test isempty(PF.get_computed_gspf(data))
    solve_power_flow!(data)
    @test isempty(PF.get_computed_gspf(data))
end

@testset "DC distributed slack: solve_and_store write-back ($T)" for T in
                                                                     (DCPowerFlow,
    PTDFDCPowerFlow, vPTDFDCPowerFlow)
    sys = PSB.build_system(PSB.PSITestSystems, "c_sys14"; add_forecasts = false)
    generators = collect(get_components(ThermalStandard, sys))
    gspf = Dict((ThermalStandard, get_name(g)) => 1.0 for g in generators)

    pf = T(; generator_slack_participation_factors = gspf)
    res = solve_power_flow(pf, sys)

    solve_and_store_power_flow!(pf, sys)

    # Stored generator set-points now match the reported bus P_gen.
    for row in eachrow(res["1"]["bus_results"])
        bus_gen = 0.0
        for g in generators
            if get_number(get_bus(g)) == row.bus_number
                bus_gen += get_active_power(g) * get_base_power(sys)
            end
        end
        if !iszero(bus_gen)
            @test isapprox(bus_gen, row.P_gen; atol = 1e-4)
        end
    end
end

@testset "DC distributed slack: headroom mode ($T)" for T in
                                                        (DCPowerFlow, PTDFDCPowerFlow,
    vPTDFDCPowerFlow)
    sys = PSB.build_system(PSB.PSITestSystems, "c_sys14"; add_forecasts = false)
    pf = T(; distribute_slack_proportional_to_headroom = true)
    data = PowerFlowData(pf, sys)

    # Headroom mode populates the participation factors from (Pmax - Pset).
    @test !isempty(PF.get_computed_gspf(data))
    @test any(!iszero, data.bus_slack_participation_factors[:, 1])

    init_p = copy(data.bus_active_power_injections)
    solve_power_flow!(data)
    @test data.bus_active_power_injections != init_p
end

@testset "DC distributed slack: negative factor errors ($T)" for T in
                                                                 (DCPowerFlow,
    PTDFDCPowerFlow, vPTDFDCPowerFlow)
    sys = PSB.build_system(PSB.PSITestSystems, "c_sys14"; add_forecasts = false)
    gen = first(get_components(ThermalStandard, sys))
    gspf = Dict((ThermalStandard, get_name(gen)) => -1.0)
    pf = T(; generator_slack_participation_factors = gspf)
    data = PowerFlowData(pf, sys)
    @test_throws ArgumentError solve_power_flow!(data)
end
