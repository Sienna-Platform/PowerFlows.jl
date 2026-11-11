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
        @test data.bus_reactive_power_injections[ix, :] ==
              [0.1, 0.2] .* PF.considers_reactive_power(pf)
        @test _all_withdrawals_zero(data)
    end
end

@testset "a constant power load adds a positive withdrawal" begin
    sys = _make_tap_shunt_system()
    load = PSY.get_component(PSY.PowerLoad, sys, "load_2")
    data = PowerFlowData(ACPowerFlow(; time_steps = 2), sys)
    ix = PF.get_bus_lookup(data)[PSY.get_number(PSY.get_bus(load))]
    PF.clear_injection_data!(data)
    zeros2 = [0.0, 0.0]
    PF.add_active_power!(data, load, [-0.2, -0.1], zeros2, zeros2)
    @test data.bus_active_power_withdrawals[ix, :] ≈ [0.2, 0.1]
    @test iszero(data.bus_active_power_injections)
    PF.add_reactive_power!(data, load, [-0.05, -0.02], zeros2, zeros2)
    @test data.bus_reactive_power_withdrawals[ix, :] ≈ [0.05, 0.02]
    @test iszero(data.bus_reactive_power_injections)
end

@testset "AC data writes the three ZIP arrays to their slots" begin
    sys = _build_zip_2bus_system(;
        power_pq = (0.2, 0.1),
        current_pq = (0.1, 0.05),
        impedance_pq = (0.3, 0.15),
    )
    zip_load = only(PSY.get_components(PSY.StandardLoad, sys))
    data = PowerFlowData(ACPowerFlow(; time_steps = 2), sys)
    ix = PF.get_bus_lookup(data)[PSY.get_number(PSY.get_bus(zip_load))]
    PF.clear_injection_data!(data)
    PF.add_active_power!(
        data, zip_load, [-0.4, -0.2], [-0.2, -0.1], [-0.6, -0.3],
    )
    (p, i, z) = _active_withdrawal_rows(data, ix)
    @test p ≈ [0.4, 0.2]
    @test i ≈ [0.2, 0.1]
    @test z ≈ [0.6, 0.3]
    @test iszero(data.bus_active_power_injections)
    PF.add_reactive_power!(
        data, zip_load, [-0.2, -0.1], [-0.1, -0.05], [-0.3, -0.15],
    )
    (p, i, z) = _reactive_withdrawal_rows(data, ix)
    @test p ≈ [0.2, 0.1]
    @test i ≈ [0.1, 0.05]
    @test z ≈ [0.3, 0.15]
    @test iszero(data.bus_reactive_power_injections)
end

@testset "the one-array form throws for a load" begin
    sys = _make_tap_shunt_system()
    load = PSY.get_component(PSY.PowerLoad, sys, "load_2")
    data = PowerFlowData(ACPowerFlow(; time_steps = 2), sys)
    PF.clear_injection_data!(data)
    @test_throws ErrorException PF.add_active_power!(data, load, [-0.2, -0.1])
    @test_throws ErrorException PF.add_reactive_power!(data, load, [-0.2, -0.1])
    @test _all_withdrawals_zero(data)
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
    return (sys, load)
end

@testset "DC data adds the sum of the three arrays to constant power" begin
    sys = _build_zip_2bus_system(;
        power_pq = (0.2, 0.1),
        current_pq = (0.1, 0.05),
        impedance_pq = (0.3, 0.15),
    )
    zip_load = only(PSY.get_components(PSY.StandardLoad, sys))
    dc_data = PowerFlowData(DCPowerFlow(; time_steps = 2), sys)
    ix = PF.get_bus_lookup(dc_data)[PSY.get_number(PSY.get_bus(zip_load))]
    PF.clear_injection_data!(dc_data)
    PF.add_active_power!(
        dc_data, zip_load, [-0.4, -0.2], [-0.2, -0.1], [-0.6, -0.3],
    )
    (p, i, z) = _active_withdrawal_rows(dc_data, ix)
    @test p ≈ [1.2, 0.6]
    @test iszero(i)
    @test iszero(z)
    PF.add_reactive_power!(
        dc_data, zip_load, [-0.2, -0.1], [-0.1, -0.05], [-0.3, -0.15],
    )
    @test iszero(dc_data.bus_reactive_power_withdrawals)
    @test iszero(dc_data.bus_reactive_power_constant_current_withdrawals)
    @test iszero(dc_data.bus_reactive_power_constant_impedance_withdrawals)
end

@testset "reactive power is a no-op when the power flow ignores it" begin
    sys = _make_tap_shunt_system()
    source = only(PSY.get_components(PSY.Source, sys))
    dc_data = PowerFlowData(DCPowerFlow(; time_steps = 2), sys)
    PF.clear_injection_data!(dc_data)
    PF.add_reactive_power!(dc_data, source, [0.1, 0.2])
    @test iszero(dc_data.bus_reactive_power_injections)
    @test_throws DimensionMismatch PF.add_reactive_power!(dc_data, source, [0.1])
end

@testset "a wrong length throws" begin
    sys = _make_tap_shunt_system()
    source = only(PSY.get_components(PSY.Source, sys))
    load = PSY.get_component(PSY.PowerLoad, sys, "load_2")
    data = PowerFlowData(ACPowerFlow(; time_steps = 2), sys)
    PF.clear_injection_data!(data)
    @test_throws DimensionMismatch PF.add_active_power!(data, source, [0.1])
    @test_throws DimensionMismatch PF.add_reactive_power!(data, source, [0.1, 0.2, 0.3])
    @test_throws DimensionMismatch PF.add_active_power!(
        data, load, [0.1, 0.1], [0.1], [0.1, 0.1],
    )
end

@testset "a component on a bus outside the power flow throws a named error" begin
    sys = _make_tap_shunt_system()
    isolated = _add_simple_bus!(sys, 4, ACBusTypes.ISOLATED, 230, 1.0, 0.0)
    source = _add_simple_source!(sys, isolated, 0.0, 0.0)
    data = PowerFlowData(ACPowerFlow(; time_steps = 2), sys)
    PF.clear_injection_data!(data)
    @test_throws ErrorException PF.add_active_power!(data, source, [0.1, 0.1])
    @test_throws string(typeof(source)) PF.add_active_power!(data, source, [0.1, 0.1])
    @test_throws PSY.get_name(source) PF.add_active_power!(data, source, [0.1, 0.1])
    @test_throws "bus 4" PF.add_active_power!(data, source, [0.1, 0.1])
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
    PF.add_active_power!(data, load, [-0.3, -0.2], [-0.1, -0.06], [-0.1, -0.04])
    (p, i, z) = _active_withdrawal_rows(data, ix)
    @test p ≈ [0.3, 0.2]
    @test i ≈ [0.1, 0.06]
    @test z ≈ [0.1, 0.04]
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

function _replay_load!(data, load, p::NTuple{3, Float64}, q::NTuple{3, Float64})
    PF.add_active_power!(data, load, (fill(-x, 2) for x in p)...)
    PF.add_reactive_power!(data, load, (fill(-x, 2) for x in q)...)
    return
end

_replay_component!(data, c::PSY.Source) = _replay_source!(data, c)

function _replay_source!(data, source)
    PF.add_active_power!(data, source, fill(PSY.get_active_power(source, u"SU"), 2))
    PF.add_reactive_power!(data, source, fill(PSY.get_reactive_power(source, u"SU"), 2))
    return
end

function _replay_component!(data, c::PSY.PowerLoad)
    _replay_load!(
        data,
        c,
        (PSY.get_active_power(c, u"SU"), 0.0, 0.0),
        (PSY.get_reactive_power(c, u"SU"), 0.0, 0.0),
    )
    return
end

function _replay_component!(data, c::PSY.StandardLoad)
    _replay_load!(
        data,
        c,
        (
            PSY.get_constant_active_power(c, u"SU"),
            PSY.get_current_active_power(c, u"SU"),
            PSY.get_impedance_active_power(c, u"SU"),
        ),
        (
            PSY.get_constant_reactive_power(c, u"SU"),
            PSY.get_current_reactive_power(c, u"SU"),
            PSY.get_impedance_reactive_power(c, u"SU"),
        ),
    )
    return
end

# The fixture sets α = 1 (constant current) and β = 2 (constant impedance).
function _replay_component!(data, c::PSY.ExponentialLoad)
    _replay_load!(
        data,
        c,
        (0.0, PSY.get_active_power(c, u"SU"), 0.0),
        (0.0, 0.0, PSY.get_reactive_power(c, u"SU")),
    )
    return
end

_replay_component!(data, c::PSY.SwitchedAdmittance) = PF.add_switched_admittance!(data, c)

function _replay_system()
    (sys, sa) = _engaged_shunt_system()
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
    exp_load = PSY.ExponentialLoad(;
        name = "exp_load",
        available = true,
        bus = bus_2,
        active_power = 0.15,
        reactive_power = 0.07,
        α = 1.0,
        β = 2.0,
        base_power = 100.0,
        max_active_power = 1.0,
        max_reactive_power = 1.0,
        input_basis = u"CU",
    )
    PSY.add_component!(sys, exp_load)
    return sys
end

function _ac_arrays(d)
    return (
        copy(d.bus_active_power_injections),
        copy(d.bus_reactive_power_injections),
        copy(d.bus_active_power_withdrawals),
        copy(d.bus_reactive_power_withdrawals),
        copy(d.bus_active_power_constant_current_withdrawals),
        copy(d.bus_reactive_power_constant_current_withdrawals),
        copy(d.bus_active_power_constant_impedance_withdrawals),
        copy(d.bus_reactive_power_constant_impedance_withdrawals),
    )
end

# A DC power flow has no reactive balance, so only the active arrays compare.
function _dc_arrays(d)
    return (
        copy(d.bus_active_power_injections),
        copy(d.bus_active_power_withdrawals),
        copy(d.bus_active_power_constant_current_withdrawals),
        copy(d.bus_active_power_constant_impedance_withdrawals),
    )
end

function _check_round_trip(pf, arrays::Function)
    sys = _replay_system()
    data = PowerFlowData(pf, sys)
    expected = arrays(data)
    @test any(!iszero, first(expected))
    PF.clear_injection_data!(data)
    for c in PSY.get_available_components(PSY.StaticInjection, sys)
        _replay_component!(data, c)
    end
    for (actual, copied) in zip(arrays(data), expected)
        @test isapprox(actual, copied; atol = 1e-12)
    end
    return
end

@testset "the update functions reproduce the arrays of the System, AC" begin
    _check_round_trip(ACPowerFlow(; time_steps = 2), _ac_arrays)
end

@testset "the update functions reproduce the arrays of the System, DC" begin
    _check_round_trip(DCPowerFlow(; time_steps = 2), _dc_arrays)
end

@testset "an exponential load in the replay fills its slots" begin
    (sys, load) = _exponential_load_fixture(; α = 1.0, β = 2.0, P = 0.5, Q = 0.2)
    data = PowerFlowData(ACPowerFlow(; time_steps = 2), sys)
    ix = PF.get_bus_lookup(data)[PSY.get_number(PSY.get_bus(load))]
    (p_system, _, _) = _active_withdrawal_rows(data, ix)
    PF.clear_injection_data!(data)
    _replay_component!(data, load)
    (p, i, z) = _active_withdrawal_rows(data, ix)
    @test iszero(p)
    @test i ≈ [0.5, 0.5]
    @test iszero(z)
    (p, i, z) = _reactive_withdrawal_rows(data, ix)
    @test iszero(p)
    @test iszero(i)
    @test z ≈ [0.2, 0.2]
end

# ---- shunt writer with a decided susceptance ----

@testset "the susceptance writer replaces the System susceptance" begin
    (sys, sa) = _engaged_shunt_system()
    data = PowerFlowData(ACPowerFlow(; time_steps = 2), sys)
    ix = PF.get_bus_lookup(data)[PSY.get_number(PSY.get_bus(sa))]
    PF.clear_injection_data!(data)
    PF.add_switched_admittance!(data, sa, [0.2, -0.1])
    @test data.bus_reactive_power_constant_impedance_withdrawals[ix, :] ≈ [-0.2, 0.1]
    # The System conductance 3 * 0.02 stays.
    @test data.bus_active_power_constant_impedance_withdrawals[ix, :] ≈ [0.06, 0.06]
    dc_data = PowerFlowData(DCPowerFlow(; time_steps = 2), sys)
    PF.clear_injection_data!(dc_data)
    PF.add_switched_admittance!(dc_data, sa, [0.2, -0.1])
    @test dc_data.bus_active_power_withdrawals[ix, :] ≈ [0.06, 0.06]
    @test_throws DimensionMismatch PF.add_switched_admittance!(data, sa, [0.2])
end

@testset "the susceptance writer starts the shunt controller from b" begin
    sys = _make_tap_shunt_system()
    sa = only(PSY.get_components(PSY.SwitchedAdmittance, sys))
    data =
        PowerFlowData(ACPowerFlow(; control_discrete_devices = true, time_steps = 2), sys)
    set = PF.get_controlled_devices(data)
    @test length(set.shunts) == 1
    @test set.shunt_current[1, :] == [0.0, 0.0]
    PF.clear_injection_data!(data)
    PF.add_switched_admittance!(data, sa, [0.1, 0.2])
    @test set.shunt_current[1, :] == [0.1, 0.2]
end

# ---- condenser losses ----

function _add_condenser!(sys, bus; losses = 0.02)
    sc = PSY.SynchronousCondenser(;
        name = "condenser_1",
        available = true,
        bus = bus,
        reactive_power = 0.0,
        rating = 0.5,
        reactive_power_limits = (min = -0.5, max = 0.5),
        base_power = 100.0,
        active_power_losses = losses,
        input_basis = u"CU",
    )
    PSY.add_component!(sys, sc)
    return sc
end

@testset "the condenser loss writer reproduces the System withdrawal" begin
    sys = _build_zip_2bus_system()
    sc = _add_condenser!(sys, PSY.get_component(PSY.ACBus, sys, "bus_2"))
    for pf in (ACPowerFlow(; time_steps = 2), DCPowerFlow(; time_steps = 2))
        data = PowerFlowData(pf, sys)
        ix = PF.get_bus_lookup(data)[PSY.get_number(PSY.get_bus(sc))]
        expected = copy(data.bus_active_power_withdrawals)
        @test expected[ix, :] ≈ [0.02, 0.02]
        PF.clear_injection_data!(data)
        @test iszero(data.bus_active_power_withdrawals)
        PF.add_active_power_losses!(data, sc)
        @test data.bus_active_power_withdrawals ≈ expected
    end
end

@testset "the reactive power of a FACTS device adds nothing" begin
    sys = _build_zip_2bus_system()
    facts = PSY.FACTSControlDevice(;
        name = "facts_1",
        available = true,
        bus = PSY.get_component(PSY.ACBus, sys, "bus_2"),
        control_mode = PSY.FACTSOperationModes.NML,
        input_basis = u"CU",
    )
    PSY.add_component!(sys, facts)
    data = PowerFlowData(ACPowerFlow(; time_steps = 2), sys)
    PF.clear_injection_data!(data)
    PF.add_reactive_power!(data, facts, [0.1, 0.2])
    @test iszero(data.bus_reactive_power_injections)
    @test_throws DimensionMismatch PF.add_reactive_power!(data, facts, [0.1])
end

# ---- pump withdrawal ----

function _add_pump!(sys, bus; pump = 0.05, available = true)
    pt = PSY.HydroPumpTurbine(;
        name = "pump_1",
        available = available,
        bus = bus,
        active_power = 0.08,
        reactive_power = 0.0,
        rating = 0.2,
        active_power_limits = (min = 0.0, max = 0.2),
        reactive_power_limits = (min = -0.1, max = 0.1),
        active_power_limits_pump = (min = 0.0, max = 0.2),
        outflow_limits = nothing,
        powerhouse_elevation = 0.0,
        ramp_limits = nothing,
        time_limits = nothing,
        base_power = 100.0,
        active_power_pump = pump,
        input_basis = u"CU",
    )
    PSY.add_component!(sys, pt)
    return pt
end

@testset "the System build withdraws the pump power of a pump turbine" begin
    for pf in (ACPowerFlow(; time_steps = 2), DCPowerFlow(; time_steps = 2))
        sys = _build_zip_2bus_system()
        bus_2 = PSY.get_component(PSY.ACBus, sys, "bus_2")
        base = PowerFlowData(pf, sys)
        ix = PF.get_bus_lookup(base)[PSY.get_number(bus_2)]
        _add_pump!(sys, bus_2)
        data = PowerFlowData(pf, sys)
        @test data.bus_active_power_withdrawals[ix, :] ≈
              base.bus_active_power_withdrawals[ix, :] .+ 0.05
        @test data.bus_active_power_injections[ix, :] ≈
              base.bus_active_power_injections[ix, :] .+ 0.08
    end
end

@testset "an unavailable pump turbine adds no pump withdrawal" begin
    sys = _build_zip_2bus_system()
    bus_2 = PSY.get_component(PSY.ACBus, sys, "bus_2")
    base = PowerFlowData(ACPowerFlow(; time_steps = 2), sys)
    _add_pump!(sys, bus_2; available = false)
    data = PowerFlowData(ACPowerFlow(; time_steps = 2), sys)
    @test data.bus_active_power_withdrawals == base.bus_active_power_withdrawals
end

@testset "the PSS/E export writes the pump as a constant power load" begin
    export_location = joinpath(mktempdir(), "pump_load")
    function exported_load_lines(sys)
        exporter = PSSEExporter(sys, :v35, export_location; overwrite = true)
        write_export(exporter, "pump"; overwrite = true)
        raw_path, _ = get_psse_export_paths(joinpath(export_location, "pump"))
        lines = readlines(raw_path)
        first_ix = findfirst(startswith("@!   I,'ID',STAT"), lines) + 1
        last_ix = findnext(==("0"), lines, first_ix) - 1
        return PF._split_record.(lines[first_ix:last_ix])
    end
    sys = _build_zip_2bus_system()
    bus_2 = PSY.get_component(PSY.ACBus, sys, "bus_2")
    _add_pump!(sys, bus_2)
    records = exported_load_lines(sys)
    @test length(records) == 2
    ids = [strip(r[2]) for r in records]
    @test allunique(ids)
    pump_record = only(filter(r -> parse(Float64, strip(r[6])) ≈ 5.0, records))
    @test parse(Int, strip(pump_record[1])) == 2
    @test parse(Int, strip(pump_record[3])) == 1
    @test exported_load_lines(sys) == records

    sys_off = _build_zip_2bus_system()
    _add_pump!(sys_off, PSY.get_component(PSY.ACBus, sys_off, "bus_2"); available = false)
    @test length(exported_load_lines(sys_off)) == 1
end

# ---- converter ----

function _converter_row(data, name::String)
    dcn = PF.get_dc_network(data)
    return only(findall(==(name), dcn.converter_name))
end

@testset "a power control converter gets its p_set when the DC network is modeled" begin
    sys = _build_mtdc_system()
    data = PowerFlowData(ACPowerFlow(; time_steps = 2), sys)
    dcn = PF.get_dc_network(data)
    ic2 = PSY.get_component(PSY.InterconnectingConverter, sys, "ic2")
    row = _converter_row(data, "ic2")
    @test dcn.p_set[row, :] ≈ [0.3, 0.3]
    PF.clear_injection_data!(data)
    PF.add_active_power!(data, ic2, [0.35, 0.45])
    @test dcn.p_set[row, :] == [0.35, 0.45]
    @test iszero(data.bus_active_power_injections)
    PF.add_reactive_power!(data, ic2, [0.1, 0.1])
    @test iszero(data.bus_reactive_power_injections)
end

@testset "the p_set of the writer matches the System build" begin
    sys = _build_mtdc_system()
    data = PowerFlowData(ACPowerFlow(; time_steps = 2), sys)
    dcn = PF.get_dc_network(data)
    expected = copy(dcn.p_set)
    dcn.p_set .= 0.0
    for name in ("ic2", "ic3")
        ic = PSY.get_component(PSY.InterconnectingConverter, sys, name)
        PF.add_active_power!(data, ic, fill(PSY.get_dc_power_setpoint(ic, u"SU"), 2))
    end
    @test dcn.p_set ≈ expected
end

@testset "a converter in a DC voltage mode writes nothing" begin
    sys = _build_mtdc_system()
    data = PowerFlowData(ACPowerFlow(; time_steps = 2), sys)
    dcn = PF.get_dc_network(data)
    ic1 = PSY.get_component(PSY.InterconnectingConverter, sys, "ic1")
    before = copy(dcn.p_set)
    PF.clear_injection_data!(data)
    PF.add_active_power!(data, ic1, [0.9, 0.9])
    @test dcn.p_set == before
    @test iszero(data.bus_active_power_injections)
end

@testset "a converter adds to the bus injection without a DC network" begin
    sys = _build_mtdc_system()
    ic2 = PSY.get_component(PSY.InterconnectingConverter, sys, "ic2")
    pf_no_dc = ACPowerFlow(;
        time_steps = 2,
        solution_parameters = SolutionParameters(; model_dc_network = false),
    )
    for pf in (pf_no_dc, DCPowerFlow(; time_steps = 2))
        data = PowerFlowData(pf, sys)
        ix = PF.get_bus_lookup(data)[PSY.get_number(PSY.get_bus(ic2))]
        PF.clear_injection_data!(data)
        PF.add_active_power!(data, ic2, [0.35, 0.45])
        @test data.bus_active_power_injections[ix, :] == [0.35, 0.45]
    end
end
