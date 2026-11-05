# Off-area loads in area interchange control (`tie_definition = LinesAndLoads()`, PSS/E
# AREAIN=2). Area1 holds REF and never enrolls, so Area2 is the one controlled area.
# `host = true`: the load sits at a PQ bus of Area2 and Area1 owns it (Area2 exports it,
# sign +1). `host = false`: the load sits at a PQ bus of Area1 and Area2 owns it (Area2
# imports it, sign -1). Nonzero schedules make the tests fail if the field were ignored.

const _OFF_AREA_PARTS = (0.2, 0.1, 0.08)
const _OFF_AREA_PDES2 = -0.3

function _off_area_fixture(; host::Bool = true, zip::Bool = true)
    sys = PSB.build_system(
        PSB.PSISystems,
        "two_area_pjm_DA_off_area_load";
        add_forecasts = false,
        add_reserves = false,
    )
    _set_slack!(sys, "Bus_nodeA_2")
    load = PSY.get_component(PSY.StandardLoad, sys, "OffAreaLoad")
    if host
        PSY.set_bus!(load, PSY.get_component(PSY.ACBus, sys, "Bus_nodeB_2"))
    else
        PSY.set_bus!(load, PSY.get_component(PSY.ACBus, sys, "Bus_nodeB_1"))
        PSY.set_out_of_area_loads!(
            PSY.get_component(PSY.Area, sys, "Area1"), PSY.ElectricLoad[])
        PSY.set_out_of_area_loads!(
            PSY.get_component(PSY.Area, sys, "Area2"), PSY.ElectricLoad[load])
    end
    if !zip
        PSY.set_current_active_power!(load, 0.0 * u"SU")
        PSY.set_impedance_active_power!(load, 0.0 * u"SU")
    end
    _set_schedule!(sys, "Area1", -_OFF_AREA_PDES2)
    _set_schedule!(sys, "Area2", _OFF_AREA_PDES2)
    return sys
end

function _pf_loads(solver = NewtonRaphsonACPowerFlow; kwargs...)
    return ACPolarPowerFlow{solver}(;
        net_interchange_control = true,
        tie_definition = LinesAndLoads(),
        kwargs...,
    )
end

_pf_lines(solver = NewtonRaphsonACPowerFlow) =
    ACPolarPowerFlow{solver}(; net_interchange_control = true)

# Active power of the off-area load at the solved voltage of its bus, pu system base.
function _off_area_load_p(data, load_bus_ix::Int)
    V = data.bus_magnitude[load_bus_ix, 1]
    (P, I, Z) = _OFF_AREA_PARTS
    return P + I * V + Z * V^2
end

function _solved_tie_export_area2(data)
    # The one tie is `inter_area_line`, from Bus_nodeC_1 (Area1) to Bus_nodeC_2 (Area2),
    # metered at `from`; the metered power out of Area1 is Area2's import.
    tie = only(PF.get_area_interchange_data(data).ties)
    Vm = data.bus_magnitude[:, 1]
    θ = data.bus_angles[:, 1]
    ybus_nzval = SparseArrays.nonzeros(data.power_network_matrix.data)
    f = tie.from_bus_ix
    t = tie.to_bus_ix
    return -PF._tie_metered_active_power(tie, Vm[f], θ[f], Vm[t], θ[t], ybus_nzval)
end

@testset "area interchange off-area load tie_definition accepted and records built" begin
    sys = _off_area_fixture()
    data = PowerFlowData(_pf_loads(), sys)
    aid = PF.get_area_interchange_data(data)
    @test only(aid.areas).name == "Area2"
    load = only(aid.off_area_loads)
    @test load.sign == 1.0
    @test load.area_tail == 1
    @test aid.pristine_off_area_loads == aid.off_area_loads
    @test_throws TypeError ACPolarPowerFlow(;
        net_interchange_control = true, tie_definition = :loads_only)
    data_lines = PowerFlowData(_pf_lines(), sys)
    @test isempty(PF.get_area_interchange_data(data_lines).off_area_loads)
end

@testset "area interchange off-area load analytic parity with a shifted lines-only PDES" for host in
                                                                                             (
    true, false,
)
    sys = _off_area_fixture(; host = host, zip = false)
    data = PowerFlowData(_pf_loads(), sys)
    @test solve_power_flow!(data)
    sign = only(PF.get_area_interchange_data(data).off_area_loads).sign
    P = _OFF_AREA_PARTS[1]

    # NI_2 = ties_2 + sign * P = PDES_2, so a lines-only solve targeting PDES_2 - sign * P
    # has the same tie flow and the same state.
    sys_lines = _off_area_fixture(; host = host, zip = false)
    _set_schedule!(sys_lines, "Area2", _OFF_AREA_PDES2 - sign * P)
    data_lines = PowerFlowData(_pf_lines(), sys_lines)
    @test solve_power_flow!(data_lines)

    @test isapprox(data.bus_magnitude[:, 1], data_lines.bus_magnitude[:, 1]; atol = 1e-8)
    @test isapprox(data.bus_angles[:, 1], data_lines.bus_angles[:, 1]; atol = 1e-8)
    @test isapprox(
        PF.get_area_interchange_data(data).delta_p[1, 1],
        PF.get_area_interchange_data(data_lines).delta_p[1, 1];
        atol = 1e-8,
    )
    @test isapprox(_solved_tie_export_area2(data), _OFF_AREA_PDES2 - sign * P; atol = 1e-8)
end

@testset "area interchange off-area load physical direction: the host area delivers it" begin
    # With PDES_2 = 0 and Area1 owning a load inside Area2, Area1 must send exactly the
    # load's power over the tie: Area2's tie export is -P.
    sys = _off_area_fixture(; host = true, zip = false)
    _set_schedule!(sys, "Area1", 0.0)
    _set_schedule!(sys, "Area2", 0.0)
    data = PowerFlowData(_pf_loads(), sys)
    @test solve_power_flow!(data)
    @test isapprox(_solved_tie_export_area2(data), -_OFF_AREA_PARTS[1]; atol = 1e-8)
end

@testset "area interchange off-area ZIP load Jacobian entries" for host in (true, false)
    sys = _off_area_fixture(; host = host)
    data = PowerFlowData(_pf_loads(), sys)
    residual = PF.ACPowerFlowResidual(data, 1)
    jac = PF.ACPowerFlowJacobian(data, residual, 1)
    x0 = PF.calculate_x0(data, 1)
    x = x0 .+ 0.02 .* sin.(1:length(x0))
    residual(data, x, 1)
    jac(data, 1)

    load = only(PF.get_area_interchange_data(data).off_area_loads)
    @test data.bus_type[load.bus_ix, 1] == PSY.ACBusTypes.PQ
    row = PF.area_tail_offset(data, PF.get_dc_network(data)) + load.area_tail
    V = data.bus_magnitude[load.bus_ix, 1]
    (_, I, Z) = _OFF_AREA_PARTS
    @test isapprox(
        jac.Jv[row, 2 * load.bus_ix - 1],
        load.sign * (I + 2 * Z * V);
        atol = 1e-12,
    )
    @test iszero(jac.Jv[row, 2 * load.bus_ix])

    verify_jacobian_asymptotic(
        residual, data, jac.Jv, x, 1; label = "off-area ZIP load (host = $host)")
end

@testset "area interchange off-area ZIP load: NR, TR, LM and FD agree" begin
    sys = _off_area_fixture()
    data_nr = PowerFlowData(_pf_loads(), sys)
    @test solve_power_flow!(data_nr)
    load_bus_ix = only(PF.get_area_interchange_data(data_nr).off_area_loads).bus_ix
    @test isapprox(
        _solved_tie_export_area2(data_nr) + _off_area_load_p(data_nr, load_bus_ix),
        _OFF_AREA_PDES2;
        atol = 1e-8,
    )

    for (label, pf, atol) in (
        ("TR", _pf_loads(TrustRegionACPowerFlow), 1e-8),
        ("LM", _pf_loads(LevenbergMarquardtACPowerFlow), 1e-6),
        (
            "FD fixed",
            _pf_loads(
                FastDecoupledACPowerFlow{FDFixedJacobian, FDSchemeXB};
                solution_parameters = FD_PARITY_PARAMETERS,
            ),
            1e-6,
        ),
        (
            "FD decoupled",
            _pf_loads(
                FastDecoupledACPowerFlow{PF.FDDecoupled, PF.FDSchemeXB};
                solution_parameters = FD_PARITY_PARAMETERS,
            ),
            1e-5,
        ),
    )
        data = PowerFlowData(pf, _off_area_fixture())
        @test solve_power_flow!(data)
        @test isapprox(data.bus_magnitude[:, 1], data_nr.bus_magnitude[:, 1]; atol = atol)
        @test isapprox(data.bus_angles[:, 1], data_nr.bus_angles[:, 1]; atol = atol)
        @test isapprox(
            PF.get_area_interchange_data(data).delta_p[1, 1],
            PF.get_area_interchange_data(data_nr).delta_p[1, 1];
            atol = atol,
        )
    end
end

@testset "area interchange off-area load hand check and results table" begin
    sys = _off_area_fixture()
    pf = _pf_loads()
    data = PowerFlowData(pf, sys)
    @test solve_power_flow!(data)
    aid = PF.get_area_interchange_data(data)
    load = only(aid.off_area_loads)
    ni_hand =
        _solved_tie_export_area2(data) + load.sign * _off_area_load_p(data, load.bus_ix)
    @test isapprox(ni_hand, _OFF_AREA_PDES2; atol = aid.tolerance)
    @test isapprox(ni_hand, _OFF_AREA_PDES2; atol = 1e-8)

    df = PF.area_interchange_results_dataframe(sys, data, 1)
    row = only(eachrow(df))
    @test row.area == "Area2"
    @test row.schedule_status == :enforced
    base = PSY.get_base_power(sys)
    @test isapprox(row.ni_solved, base * ni_hand; atol = 1e-6)
    @test isapprox(row.pdes, base * _OFF_AREA_PDES2; atol = 1e-12)

    results = solve_power_flow(pf, sys)
    @test isapprox(
        only(results["area_interchange_results"].ni_solved), base * _OFF_AREA_PDES2;
        atol = 1e-6)
end

@testset "area interchange off-area load bad references error" begin
    sys = _off_area_fixture()
    load = PSY.get_component(PSY.StandardLoad, sys, "OffAreaLoad")
    PSY.set_bus!(load, PSY.get_component(PSY.ACBus, sys, "Bus_nodeB_1"))
    @test_throws r"Area \"Area1\" lists load \"OffAreaLoad\".*same area" PowerFlowData(
        _pf_loads(), sys)

    sys = _off_area_fixture()
    load = PSY.get_component(PSY.StandardLoad, sys, "OffAreaLoad")
    PSY.set_area!(PSY.get_bus(load), nothing)
    @test_throws r"Area \"Area1\" lists load \"OffAreaLoad\".*has no area" PowerFlowData(
        _pf_loads(), sys)
end

@testset "area interchange off-area load records follow greedy relax de-enrollment" begin
    sys = _off_area_fixture()
    data = PowerFlowData(_pf_loads(), sys)
    aid = PF.get_area_interchange_data(data)
    @test length(aid.off_area_loads) == 1
    PF._deenroll_area!(data, 1)
    @test isempty(aid.areas)
    @test isempty(aid.off_area_loads)
    @test length(aid.pristine_off_area_loads) == 1
    PF._restore_pristine_area_set!(data, nothing)
    @test aid.off_area_loads == aid.pristine_off_area_loads
    @test solve_power_flow!(data)
    ni = PF._area_net_interchange(
        aid.pristine_ties, aid.pristine_dc_ties, aid.pristine_off_area_loads, 1, data, 1,
    )
    @test isapprox(ni, _OFF_AREA_PDES2; atol = 1e-8)
end

# Load "Bus6" sits at Bus 6 of Area2 (a PV bus under a tight Q limit). Area1 or Area3 owns it.
function _three_area_off_area_fixture(owners::Vector{String})
    sys = _three_area_transfer_fixture(; slack_area3 = true)
    load = PSY.get_component(PSY.PowerLoad, sys, "Bus6")
    for owner in owners
        PSY.set_out_of_area_loads!(
            PSY.get_component(PSY.Area, sys, owner), PSY.ElectricLoad[load])
    end
    return sys
end

# Net interchange of every enrolled area, from the pristine records, at `time_step`.
function _enrolled_net_interchange(data, time_step::Int = 1)
    aid = PF.get_area_interchange_data(data)
    return [
        PF._area_net_interchange(
            aid.pristine_ties, aid.pristine_dc_ties, aid.pristine_off_area_loads,
            area.tail_ix, data, time_step,
        ) for area in aid.pristine_areas
    ]
end

@testset "area interchange off-area load survives a PV -> PQ flip at its bus" begin
    sys = _three_area_off_area_fixture(["Area1"])
    gen6 = PSY.get_component(PSY.ThermalStandard, sys, "Bus6")
    PSY.set_reactive_power_limits!(gen6, (min = -0.001 * u"SU", max = 0.24 * u"SU"))
    pf = _pf_loads(; check_reactive_power_limits = true)
    data = PowerFlowData(pf, sys)
    load_bus_ix = only(PF.get_area_interchange_data(data).off_area_loads).bus_ix
    @test data.bus_type[load_bus_ix, 1] == PSY.ACBusTypes.PV
    @test solve_power_flow!(data)
    @test data.bus_type[load_bus_ix, 1] == PSY.ACBusTypes.PQ
    aid = PF.get_area_interchange_data(data)
    @test isapprox(
        _enrolled_net_interchange(data), [area.pdes[1] for area in aid.pristine_areas];
        atol = 1e-6,
    )
end

@testset "area interchange off-area load in area B owned by area C, both enrolled" begin
    sys = _three_area_off_area_fixture(["Area3"])
    data = PowerFlowData(_pf_loads(), sys)
    aid = PF.get_area_interchange_data(data)
    @test sort([area.name for area in aid.areas]) == ["Area2", "Area3"]
    @test sort([load.sign for load in aid.off_area_loads]) == [-1.0, 1.0]
    @test all(load.name == "Bus6" for load in aid.off_area_loads)
    @test solve_power_flow!(data)
    @test isapprox(
        _enrolled_net_interchange(data), [area.pdes[1] for area in aid.pristine_areas];
        atol = 1e-6,
    )
end

@testset "set_off_area_load_power! updates every record and the interchange follows" begin
    delta = 0.05
    function solve_second_step(use_setter::Bool)
        sys = _three_area_off_area_fixture(["Area3"])
        step_data = PowerFlowData(_pf_loads(; time_steps = 2), sys)
        step_aid = PF.get_area_interchange_data(step_data)
        @test PF.has_off_area_load(step_data, "Bus6")
        @test !PF.has_off_area_load(step_data, "no_such_load")
        @test solve_power_flow!(step_data; time_steps = [1])
        records = step_aid.pristine_off_area_loads
        (P, I, Z) = (
            records[1].p_constant_power[1],
            records[1].p_constant_current[1],
            records[1].p_constant_impedance[1],
        )
        step_data.bus_active_power_withdrawals[records[1].bus_ix, 2] += delta
        if use_setter
            PF.set_off_area_load_power!(step_data, "Bus6", 2, P + delta, I, Z)
            for load in records
                @test load.p_constant_power[2] == P + delta
                @test load.p_constant_power[1] == P
            end
            for (work, pristine) in zip(step_aid.off_area_loads, records)
                @test work.p_constant_power === pristine.p_constant_power
            end
        end
        @test solve_power_flow!(step_data; time_steps = [2])
        return (step_data, step_aid)
    end
    (data, aid) = solve_second_step(true)
    @test isapprox(
        _enrolled_net_interchange(data, 2), [area.pdes[2] for area in aid.pristine_areas];
        atol = 1e-6,
    )
    (data_stale, _) = solve_second_step(false)
    @test !isapprox(
        PF.get_area_interchange_data(data_stale).delta_p[:, 2],
        aid.delta_p[:, 2];
        atol = 1e-4,
    )
    @test_throws r"no_such_load" PF.set_off_area_load_power!(
        data, "no_such_load", 2, 0.0, 0.0, 0.0)
    @test_throws BoundsError PF.set_off_area_load_power!(data, "Bus6", 3, 0.0, 0.0, 0.0)
end

@testset "area interchange off-area load at an ISOLATED bus is skipped with a warning" begin
    sys = _off_area_fixture()
    bus = _add_simple_bus!(sys, 999, PSY.ACBusTypes.ISOLATED, 230)
    PSY.set_area!(bus, PSY.get_component(PSY.Area, sys, "Area2"))
    iso_load = _add_simple_load!(sys, bus, 0.05, 0.02)
    area1 = PSY.get_component(PSY.Area, sys, "Area1")
    PSY.set_out_of_area_loads!(
        area1,
        PSY.ElectricLoad[PSY.get_out_of_area_loads(area1)..., iso_load],
    )
    data =
        @test_logs (:warn, r"off-area load \"load_999\".*ISOLATED bus bus_999") match_mode =
            :any PowerFlowData(_pf_loads(), sys)
    aid = PF.get_area_interchange_data(data)
    @test [load.name for load in aid.off_area_loads] == ["OffAreaLoad"]
    @test !PF.has_off_area_load(data, "load_999")
    @test solve_power_flow!(data)
end

@testset "area interchange off-area load clear_injection_data! zeroes the records" begin
    sys = _off_area_fixture()
    data = PowerFlowData(_pf_loads(; time_steps = 2), sys)
    records = PF.get_area_interchange_data(data).pristine_off_area_loads
    @test any(!iszero, only(records).p_constant_power)
    PF.clear_injection_data!(data)
    for r in records
        @test all(iszero, r.p_constant_power)
        @test all(iszero, r.p_constant_current)
        @test all(iszero, r.p_constant_impedance)
    end
    PF.set_off_area_load_power!(data, "OffAreaLoad", 2, 0.2, 0.1, 0.08)
    r = only(records)
    @test r.p_constant_power == [0.0, 0.2]
    @test r.p_constant_current == [0.0, 0.1]
    @test r.p_constant_impedance == [0.0, 0.08]
end
