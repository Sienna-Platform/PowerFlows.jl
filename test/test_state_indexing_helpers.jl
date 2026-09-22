@testset "partition state" begin
    sys = PSB.build_system(PSB.PSITestSystems, "c_sys5")
    pf = ACPowerFlow()
    data = PowerFlowData(pf, sys)
    time_step = 1
    x0 = collect(1.0:1.0:10.0) # just need 10 distinct Float64's
    bus_types = @view PF.get_bus_type(data)[:, time_step]
    (Ps_1, Qs_1, Vms_1, Vas_1) = (Vector{Float64}() for _ in 1:4)
    i = 1
    for bt in bus_types
        if bt == PSY.ACBusTypes.REF
            push!(Ps_1, x0[i])
            i += 1
            push!(Qs_1, x0[i])
            i += 1
            push!(Vms_1, NaN)
            push!(Vas_1, NaN)
        elseif bt == PSY.ACBusTypes.PV
            push!(Qs_1, x0[i])
            i += 1
            push!(Vas_1, x0[i])
            i += 1
            push!(Ps_1, NaN)
            push!(Vms_1, NaN)
        elseif bt == PSY.ACBusTypes.PQ
            push!(Vms_1, x0[i])
            i += 1
            push!(Vas_1, x0[i])
            i += 1
            push!(Ps_1, NaN)
            push!(Qs_1, NaN)
        end
    end
    tp = PF.partition_state(x0, bus_types)
    @test isequal(tp[:P], Ps_1)
    @test isequal(tp[:Q], Qs_1)
    @test isequal(tp[:Vm], Vms_1)
    @test isequal(tp[:Va], Vas_1)
end

"""Three-bus system (REF / PV / PQ) carrying a ZIP load at every bus, with
off-nominal REF and PV voltage setpoints so the constant-current (`∝ |V|`) and
constant-impedance (`∝ |V|²`) terms are distinguishable from the constant-power
term. Used to pin `update_state!` / `update_data!` against issue #431."""
function _zip_state_indexing_system()
    sys = System(100.0)
    b1 = _add_simple_bus!(sys, 1, ACBusTypes.REF, 230.0, 1.02, 0.0)
    b2 = _add_simple_bus!(sys, 2, ACBusTypes.PV, 230.0, 1.03, 0.0)
    b3 = _add_simple_bus!(sys, 3, ACBusTypes.PQ, 230.0, 1.0, 0.0)
    _add_simple_line!(sys, b1, b2, 0.01, 0.08, 0.0)
    _add_simple_line!(sys, b2, b3, 0.01, 0.08, 0.0)
    _add_simple_source!(sys, b1, 0.0, 0.0)
    _add_simple_thermal_standard!(sys, b2, 0.5, 0.0)
    # ZIP loads on the REF and PV buses too: those are the buses whose power slots
    # `update_state!`/`update_data!` build from the withdrawal fields.
    for b in (b1, b2, b3)
        _add_simple_zip_load!(
            sys,
            b;
            constant_power_active_power = 1.0,
            constant_power_reactive_power = 0.5,
            constant_current_active_power = 2.0,
            constant_current_reactive_power = 1.0,
            constant_impedance_active_power = 3.0,
            constant_impedance_reactive_power = 1.5,
        )
    end
    return sys
end

"""Independent restatement of the ZIP withdrawal at bus `ix`: constant power,
plus constant current scaled by `|V|`, plus constant impedance scaled by `|V|²`."""
function _expected_total_withdrawals(data, ix, time_step)
    vm = data.bus_magnitude[ix, time_step]
    p =
        data.bus_active_power_withdrawals[ix, time_step] +
        data.bus_active_power_constant_current_withdrawals[ix, time_step] * vm +
        data.bus_active_power_constant_impedance_withdrawals[ix, time_step] * vm^2
    q =
        data.bus_reactive_power_withdrawals[ix, time_step] +
        data.bus_reactive_power_constant_current_withdrawals[ix, time_step] * vm +
        data.bus_reactive_power_constant_impedance_withdrawals[ix, time_step] * vm^2
    return (p, q)
end

@testset "update_state! includes ZIP withdrawals" begin
    sys = _zip_state_indexing_system()
    data = PowerFlowData(ACPowerFlow(), sys)
    time_step = 1
    n_buses = size(data.bus_type, 1)
    bus_types = data.bus_type[:, time_step]

    # Guard the fixture: without nonzero const-I/const-Z at REF and PV, and
    # off-nominal setpoints there, this testset cannot detect the bug.
    for ix in 1:n_buses
        @test data.bus_active_power_constant_current_withdrawals[ix, time_step] != 0.0
        @test data.bus_active_power_constant_impedance_withdrawals[ix, time_step] != 0.0
        @test data.bus_reactive_power_constant_current_withdrawals[ix, time_step] != 0.0
        @test data.bus_reactive_power_constant_impedance_withdrawals[ix, time_step] != 0.0
    end
    for ix in 1:n_buses
        if bus_types[ix] != PSY.ACBusTypes.PQ
            @test data.bus_magnitude[ix, time_step] != 1.0
        end
    end

    x = PF.calculate_x0(data, time_step)
    for ix in 1:n_buses
        (p_w, q_w) = _expected_total_withdrawals(data, ix, time_step)
        if bus_types[ix] == PSY.ACBusTypes.REF
            @test x[2 * ix - 1] ≈
                  data.bus_active_power_injections[ix, time_step] - p_w
            @test x[2 * ix] ≈
                  data.bus_reactive_power_injections[ix, time_step] - q_w
        elseif bus_types[ix] == PSY.ACBusTypes.PV
            @test x[2 * ix - 1] ≈
                  data.bus_reactive_power_injections[ix, time_step] - q_w
            @test x[2 * ix] == data.bus_angles[ix, time_step]
        else
            @test x[2 * ix - 1] == data.bus_magnitude[ix, time_step]
            @test x[2 * ix] == data.bus_angles[ix, time_step]
        end
    end
end

@testset "update_data! includes ZIP withdrawals" begin
    sys = _zip_state_indexing_system()
    data = PowerFlowData(ACPowerFlow(), sys)
    time_step = 1
    n_buses = size(data.bus_type, 1)
    bus_types = data.bus_type[:, time_step]

    # Write chosen NET injections into the power slots, then check `update_data!`
    # recovers gross injections by adding back the full ZIP withdrawal.
    x = PF.calculate_x0(data, time_step)
    net_p = Dict{Int, Float64}()
    net_q = Dict{Int, Float64}()
    for ix in 1:n_buses
        if bus_types[ix] == PSY.ACBusTypes.REF
            net_p[ix] = 0.25 * ix
            net_q[ix] = -0.15 * ix
            x[2 * ix - 1] = net_p[ix]
            x[2 * ix] = net_q[ix]
        elseif bus_types[ix] == PSY.ACBusTypes.PV
            net_q[ix] = -0.15 * ix
            x[2 * ix - 1] = net_q[ix]
        end
    end

    PF.update_data!(data, x, time_step)

    for ix in 1:n_buses
        (p_w, q_w) = _expected_total_withdrawals(data, ix, time_step)
        if bus_types[ix] == PSY.ACBusTypes.REF
            @test data.bus_active_power_injections[ix, time_step] ≈ net_p[ix] + p_w
            @test data.bus_reactive_power_injections[ix, time_step] ≈ net_q[ix] + q_w
        elseif bus_types[ix] == PSY.ACBusTypes.PV
            @test data.bus_reactive_power_injections[ix, time_step] ≈ net_q[ix] + q_w
        end
    end
end

@testset "update_state!/update_data! round trip with ZIP loads" begin
    sys = _zip_state_indexing_system()
    data = PowerFlowData(ACPowerFlow(), sys)
    time_step = 1
    solve_power_flow!(data)

    x = PF.calculate_x0(data, time_step)
    p_inj = copy(data.bus_active_power_injections)
    q_inj = copy(data.bus_reactive_power_injections)
    vm = copy(data.bus_magnitude)
    va = copy(data.bus_angles)

    PF.update_data!(data, x, time_step)

    @test data.bus_active_power_injections ≈ p_inj
    @test data.bus_reactive_power_injections ≈ q_inj
    @test data.bus_magnitude ≈ vm
    @test data.bus_angles ≈ va
end

@testset "update_state! at the solution is a residual zero with ZIP loads" begin
    sys = _zip_state_indexing_system()
    data = PowerFlowData(ACPowerFlow(), sys)
    time_step = 1
    solve_power_flow!(data)

    # The converged `data` must map back to a state vector at which the residual
    # vanishes. Dropping the const-I/const-Z terms from the REF/PV power slots
    # leaves an error of exactly those withdrawals, so this fails on unfixed code.
    x = PF.calculate_x0(data, time_step)
    residual = PF.ACPowerFlowResidual(data, time_step)
    residual(x, time_step)
    @test norm(residual.Rv, Inf) < 1e-8
end
