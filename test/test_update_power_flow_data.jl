function _active_withdrawal_rows(data, ix::Int)
    return (
        data.bus_active_power_withdrawals[ix, :],
        data.bus_active_power_constant_current_withdrawals[ix, :],
        data.bus_active_power_constant_impedance_withdrawals[ix, :],
    )
end

function _reactive_withdrawal_rows(data, ix::Int)
    return (
        data.bus_reactive_power_withdrawals[ix, :],
        data.bus_reactive_power_constant_current_withdrawals[ix, :],
        data.bus_reactive_power_constant_impedance_withdrawals[ix, :],
    )
end

function _all_withdrawals_zero(data)
    arrays = (
        data.bus_active_power_withdrawals,
        data.bus_active_power_constant_current_withdrawals,
        data.bus_active_power_constant_impedance_withdrawals,
        data.bus_reactive_power_withdrawals,
        data.bus_reactive_power_constant_current_withdrawals,
        data.bus_reactive_power_constant_impedance_withdrawals,
    )
    return all(iszero, arrays)
end

function _off_area_fixture()
    sys = PSB.build_system(
        PSB.PSISystems,
        "two_area_pjm_DA_off_area_load";
        add_forecasts = false,
        add_reserves = false,
    )
    _set_slack!(sys, "Bus_nodeA_2")
    load = PSY.get_component(PSY.StandardLoad, sys, "OffAreaLoad")
    PSY.set_bus!(load, PSY.get_component(PSY.ACBus, sys, "Bus_nodeB_2"))
    _set_schedule!(sys, "Area1", 0.3)
    _set_schedule!(sys, "Area2", -0.3)
    return sys
end

@testset "a source adds to the injection array" begin
    sys = _make_tap_shunt_system()
    source = only(PSY.get_components(PSY.Source, sys))
    for pf in (ACPowerFlow(; time_steps = 2), DCPowerFlow(; time_steps = 2))
        data = PowerFlowData(pf, sys)
        ix = PF.get_bus_lookup(data)[PSY.get_number(PSY.get_bus(source))]
        PF.clear_injection_data!(data)
        PF.add_active_power!(data, source, [0.3, 0.4])
        @test data.bus_active_power_injections[ix, :] == [0.3, 0.4]
        @test _all_withdrawals_zero(data)
        PF.add_reactive_power!(data, source, [0.1, 0.2])
        @test data.bus_reactive_power_injections[ix, :] == [0.1, 0.2]
        @test _all_withdrawals_zero(data)
    end
end

@testset "a constant power load adds a positive withdrawal" begin
    sys = _make_tap_shunt_system()
    load = PSY.get_component(PSY.PowerLoad, sys, "load_2")
    data = PowerFlowData(ACPowerFlow(; time_steps = 2), sys)
    ix = PF.get_bus_lookup(data)[PSY.get_number(PSY.get_bus(load))]
    PF.clear_injection_data!(data)
    PF.add_active_power!(data, load, [-0.2, -0.1])
    @test data.bus_active_power_withdrawals[ix, :] ≈ [0.2, 0.1]
    @test iszero(data.bus_active_power_injections)
    PF.add_reactive_power!(data, load, [-0.05, -0.02])
    @test data.bus_reactive_power_withdrawals[ix, :] ≈ [0.05, 0.02]
    @test iszero(data.bus_reactive_power_injections)
end

@testset "AC data divides a ZIP load by its System parts" begin
    sys = _build_zip_2bus_system(;
        power_pq = (0.2, 0.1),
        current_pq = (0.1, 0.05),
        impedance_pq = (0.3, 0.15),
    )
    zip_load = only(PSY.get_components(PSY.StandardLoad, sys))
    data = PowerFlowData(ACPowerFlow(; time_steps = 2), sys)
    ix = PF.get_bus_lookup(data)[PSY.get_number(PSY.get_bus(zip_load))]
    PF.clear_injection_data!(data)
    PF.add_active_power!(data, zip_load, [-1.2, -0.6])
    (p, i, z) = _active_withdrawal_rows(data, ix)
    @test p ≈ [0.4, 0.2]
    @test i ≈ [0.2, 0.1]
    @test z ≈ [0.6, 0.3]
    @test iszero(data.bus_active_power_injections)
    PF.add_reactive_power!(data, zip_load, [-0.6, -0.3])
    (p, i, z) = _reactive_withdrawal_rows(data, ix)
    @test p ≈ [0.2, 0.1]
    @test i ≈ [0.1, 0.05]
    @test z ≈ [0.3, 0.15]
    @test iszero(data.bus_reactive_power_injections)
end

function _exponential_load_fixture(; α::Float64, β::Float64, P::Float64, Q::Float64)
    sys = _build_zip_2bus_system()
    bus = PSY.get_component(PSY.ACBus, sys, "bus_2")
    load = PSY.ExponentialLoad(;
        name = "exp_load",
        available = true,
        bus = bus,
        active_power = P,
        reactive_power = Q,
        α = α,
        β = β,
        base_power = 100.0,
        max_active_power = 1.0,
        max_reactive_power = 1.0,
        input_basis = u"CU",
    )
    PSY.add_component!(sys, load)
    data = PowerFlowData(ACPowerFlow(; time_steps = 2), sys)
    ix = PF.get_bus_lookup(data)[PSY.get_number(bus)]
    PF.clear_injection_data!(data)
    return (data, load, ix)
end

@testset "AC data puts an exponential load in the slot of its exponent" begin
    (data, load, ix) = _exponential_load_fixture(; α = 1.0, β = 2.0, P = 0.5, Q = 0.2)
    PF.add_active_power!(data, load, [-0.5, -0.25])
    (p, i, z) = _active_withdrawal_rows(data, ix)
    @test iszero(p)
    @test i ≈ [0.5, 0.25]
    @test iszero(z)
    PF.add_reactive_power!(data, load, [-0.2, -0.1])
    (p, i, z) = _reactive_withdrawal_rows(data, ix)
    @test iszero(p)
    @test iszero(i)
    @test z ≈ [0.2, 0.1]
end

@testset "an exponential load with zero reactive power keeps its slot" begin
    (data, load, ix) = _exponential_load_fixture(; α = 0.0, β = 2.0, P = 0.5, Q = 0.0)
    PF.add_reactive_power!(data, load, [-0.4, -0.2])
    (p, i, z) = _reactive_withdrawal_rows(data, ix)
    @test iszero(p)
    @test iszero(i)
    @test z ≈ [0.4, 0.2]
end

@testset "an exponential load with zero active power keeps its slot" begin
    (data, load, ix) = _exponential_load_fixture(; α = 1.0, β = 0.0, P = 0.0, Q = 0.2)
    PF.add_active_power!(data, load, [-0.4, -0.2])
    (p, i, z) = _active_withdrawal_rows(data, ix)
    @test iszero(p)
    @test i ≈ [0.4, 0.2]
    @test iszero(z)
end

@testset "DC data puts a ZIP load in constant power" begin
    sys = _build_zip_2bus_system(;
        power_pq = (0.2, 0.1),
        current_pq = (0.1, 0.05),
        impedance_pq = (0.3, 0.15),
    )
    zip_load = only(PSY.get_components(PSY.StandardLoad, sys))
    dc_data = PowerFlowData(DCPowerFlow(; time_steps = 2), sys)
    ix = PF.get_bus_lookup(dc_data)[PSY.get_number(PSY.get_bus(zip_load))]
    PF.clear_injection_data!(dc_data)
    PF.add_active_power!(dc_data, zip_load, [-1.2, -0.6])
    (p, i, z) = _active_withdrawal_rows(dc_data, ix)
    @test p ≈ [1.2, 0.6]
    @test iszero(i)
    @test iszero(z)
end

@testset "a ZIP load with zero parts is constant power" begin
    sys = _build_zip_2bus_system(;
        power_pq = (0.0, 0.0),
        current_pq = (0.0, 0.0),
        impedance_pq = (0.0, 0.0),
    )
    zip_load = only(PSY.get_components(PSY.StandardLoad, sys))
    data = PowerFlowData(ACPowerFlow(; time_steps = 2), sys)
    ix = PF.get_bus_lookup(data)[PSY.get_number(PSY.get_bus(zip_load))]
    PF.clear_injection_data!(data)
    PF.add_active_power!(data, zip_load, [-0.5, -0.5])
    (p, i, z) = _active_withdrawal_rows(data, ix)
    @test p ≈ [0.5, 0.5]
    @test iszero(i)
    @test iszero(z)
end

@testset "a wrong length throws" begin
    sys = _make_tap_shunt_system()
    source = only(PSY.get_components(PSY.Source, sys))
    data = PowerFlowData(ACPowerFlow(; time_steps = 2), sys)
    PF.clear_injection_data!(data)
    @test_throws DimensionMismatch PF.add_active_power!(data, source, [0.1])
    @test_throws DimensionMismatch PF.add_reactive_power!(data, source, [0.1, 0.2, 0.3])
end

@testset "an off-area load record gets the three parts" begin
    sys = _off_area_fixture()
    pf = ACPolarPowerFlow{NewtonRaphsonACPowerFlow}(;
        net_interchange_control = true,
        tie_definition = LinesAndLoads(),
        time_steps = 2,
    )
    data = PowerFlowData(pf, sys)
    load = PSY.get_component(PSY.StandardLoad, sys, "OffAreaLoad")
    @test PF.has_off_area_load(data, PSY.get_name(load))
    ix = PF.get_bus_lookup(data)[PSY.get_number(PSY.get_bus(load))]
    PF.clear_injection_data!(data)
    PF.add_active_power!(data, load, [-0.5, -0.3])
    (p, i, z) = _active_withdrawal_rows(data, ix)
    @test p[1] + i[1] + z[1] ≈ 0.5
    @test p[2] + i[2] + z[2] ≈ 0.3
    @test all(!iszero, i)
    @test all(!iszero, z)
    record = only(PF.get_area_interchange_data(data).off_area_loads)
    @test record.p_constant_power == p
    @test record.p_constant_current == i
    @test record.p_constant_impedance == z
end
