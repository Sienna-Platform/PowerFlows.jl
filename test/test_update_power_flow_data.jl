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

function _off_area_zip_load_fixture()
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
    sys = _off_area_zip_load_fixture()
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

function _constant_impedance_arrays(data)
    return (
        copy(data.bus_active_power_constant_impedance_withdrawals),
        copy(data.bus_reactive_power_constant_impedance_withdrawals),
    )
end

function _engaged_shunt_system()
    sys = _make_tap_shunt_system()
    sa = only(PSY.get_components(PSY.SwitchedAdmittance, sys))
    PSY.set_Y_increase!(sa, [0.02 + 0.05im])
    PSY.set_number_engaged!(sa, [3])
    return (sys, sa)
end

@testset "a switched admittance adds its System admittance to every time step" begin
    (sys, sa) = _engaged_shunt_system()
    data = PowerFlowData(ACPowerFlow(; time_steps = 2), sys)
    (p_expected, q_expected) = _constant_impedance_arrays(data)
    ix = PF.get_bus_lookup(data)[PSY.get_number(PSY.get_bus(sa))]
    @test p_expected[ix, :] ≈ [0.06, 0.06]
    @test q_expected[ix, :] ≈ [-0.15, -0.15]
    PF.clear_injection_data!(data)
    PF.add_switched_admittance!(data, sa)
    @test data.bus_active_power_constant_impedance_withdrawals == p_expected
    @test data.bus_reactive_power_constant_impedance_withdrawals == q_expected
end

function _constant_power_and_impedance_arrays(data)
    return (
        copy(data.bus_active_power_withdrawals),
        copy(data.bus_reactive_power_withdrawals),
        copy(data.bus_active_power_constant_impedance_withdrawals),
        copy(data.bus_reactive_power_constant_impedance_withdrawals),
    )
end

@testset "DC data puts the real part of a switched admittance in constant power" begin
    (sys, sa) = _engaged_shunt_system()
    (sys_off, sa_off) = _engaged_shunt_system()
    PSY.set_number_engaged!(sa_off, [0])
    data = PowerFlowData(DCPowerFlow(; time_steps = 2), sys)
    data_off = PowerFlowData(DCPowerFlow(; time_steps = 2), sys_off)
    expected = map(
        -,
        _constant_power_and_impedance_arrays(data),
        _constant_power_and_impedance_arrays(data_off),
    )
    ix = PF.get_bus_lookup(data)[PSY.get_number(PSY.get_bus(sa))]
    @test expected[1][ix, :] ≈ [0.06, 0.06]
    @test iszero(expected[3])
    PF.clear_injection_data!(data)
    PF.add_switched_admittance!(data, sa)
    for (actual, copied) in zip(_constant_power_and_impedance_arrays(data), expected)
        @test actual ≈ copied
    end
end

@testset "a switched admittance on a bus outside the power flow adds nothing" begin
    (sys, _) = _engaged_shunt_system()
    isolated = _add_simple_bus!(sys, 4, ACBusTypes.ISOLATED, 230, 1.0, 0.0)
    outside = PSY.SwitchedAdmittance(;
        name = "shunt_4",
        available = true,
        bus = isolated,
        number_engaged = [1],
        number_of_steps = [4],
        Y_increase = [0.0 + 0.05im],
        voltage_limits = (min = 0.9, max = 1.1),
        control_mode = PSY.SwitchedAdmittanceControlMode.DISCRETE_VOLTAGE,
    )
    PSY.add_component!(sys, outside)
    data = PowerFlowData(ACPowerFlow(; time_steps = 2), sys)
    @test !haskey(PF.get_bus_lookup(data), PSY.get_number(isolated))
    PF.clear_injection_data!(data)
    PF.add_switched_admittance!(data, outside)
    @test _all_withdrawals_zero(data)
end

_replay_component!(data, c::PSY.Source) = _replay_source!(data, c)

function _replay_component!(data, c::PSY.PowerLoad)
    _replay_load!(
        data,
        c,
        -PSY.get_active_power(c, u"SU"),
        -PSY.get_reactive_power(c, u"SU"),
    )
    return
end

function _replay_component!(data, c::PSY.StandardLoad)
    p = -(
        PSY.get_constant_active_power(c, u"SU") +
        PSY.get_current_active_power(c, u"SU") +
        PSY.get_impedance_active_power(c, u"SU")
    )
    q = -(
        PSY.get_constant_reactive_power(c, u"SU") +
        PSY.get_current_reactive_power(c, u"SU") +
        PSY.get_impedance_reactive_power(c, u"SU")
    )
    _replay_load!(data, c, p, q)
    return
end

_replay_component!(data, c::PSY.SwitchedAdmittance) = PF.add_switched_admittance!(data, c)

function _replay_source!(data, source)
    PF.add_active_power!(data, source, fill(PSY.get_active_power(source, u"SU"), 2))
    PF.add_reactive_power!(data, source, fill(PSY.get_reactive_power(source, u"SU"), 2))
    return
end

function _replay_load!(data, load, p::Float64, q::Float64)
    PF.add_active_power!(data, load, fill(p, 2))
    PF.add_reactive_power!(data, load, fill(q, 2))
    return
end

@testset "the update functions reproduce the arrays of the System" begin
    (sys, _) = _engaged_shunt_system()
    source = only(PSY.get_components(PSY.Source, sys))
    PSY.set_active_power!(source, 0.3 * u"SU")
    PSY.set_reactive_power!(source, 0.1 * u"SU")
    bus_2 = PSY.get_component(PSY.ACBus, sys, "bus_2")
    _add_simple_zip_load!(
        sys,
        bus_2;
        constant_power_active_power = 0.2,
        constant_power_reactive_power = 0.1,
        constant_current_active_power = 0.1,
        constant_current_reactive_power = 0.05,
        constant_impedance_active_power = 0.3,
        constant_impedance_reactive_power = 0.15,
    )
    data = PowerFlowData(ACPowerFlow(; time_steps = 2), sys)
    arrays(d) = (
        copy(d.bus_active_power_injections),
        copy(d.bus_reactive_power_injections),
        copy(d.bus_active_power_withdrawals),
        copy(d.bus_reactive_power_withdrawals),
        copy(d.bus_active_power_constant_current_withdrawals),
        copy(d.bus_reactive_power_constant_current_withdrawals),
        copy(d.bus_active_power_constant_impedance_withdrawals),
        copy(d.bus_reactive_power_constant_impedance_withdrawals),
    )
    expected = arrays(data)
    for copied in expected
        @test any(!iszero, copied)
    end
    PF.clear_injection_data!(data)
    for c in PSY.get_available_components(PSY.StaticInjection, sys)
        _replay_component!(data, c)
    end
    for (actual, copied) in zip(arrays(data), expected)
        @test isapprox(actual, copied; atol = 1e-12)
    end
end
