const SYSTEM_REIMPORT_COMPARISON_TOLERANCE = 1e-10
const POWERFLOW_COMPARISON_TOLERANCE = 3e-4  # TODO refine -- most comparisons can be made much tighter

power_flow_match_fn(
    a::T,
    b::T,
) where {T <: Union{AbstractFloat, AbstractArray{<:AbstractFloat}}} =
    isapprox(a, b; atol = POWERFLOW_COMPARISON_TOLERANCE) || IS.isequivalent(a, b)
power_flow_match_fn(a, b) = IS.isequivalent(a, b)

# TODO another temporary hack
"Create a version of the RTS_GMLC system that plays nice with the current implementation of AC power flow"
function create_pf_friendly_rts_gmlc()
    sys = build_system(PSISystems, "RTS_GMLC_DA_sys")
    remove_component!(sys, only(get_components(PSY.TwoTerminalHVDC, sys)))  # HVDC power flow not implemented yet
    # Modify some things so reactive power redistribution succeeds
    for (component_type, component_name, new_limits) in [
        (RenewableDispatch, "113_PV_1", (min = -30.0, max = 30.0))
        (ThermalStandard, "115_STEAM_3", (min = -50.0, max = 100.0))
        (ThermalStandard, "207_CT_1", (min = -70.0, max = 70.0))
        (RenewableDispatch, "215_PV_1", (min = -40.0, max = 40.0))
        (ThermalStandard, "307_CT_1", (min = -70.0, max = 70.0))
        (ThermalStandard, "315_CT_8", (min = 0.0, max = 80.0))
    ]
        set_reactive_power_limits!(
            get_component(component_type, sys, component_name),
            new_limits,
        )
    end
    return sys
end

"Take RTS_GMLC_DA_sys and make some changes to it that are fully captured in the PowerFlowData(ACPowerFlow(), ...)"
function modify_rts_system!(sys::System)
    # For REF bus, voltage and angle are fixed; update active and reactive
    ref_bus = get_bus(sys, 113)  # "Arne"
    @assert get_bustype(ref_bus) == ACBusTypes.REF
    # NOTE: we are not testing the correctness of _power_redistribution_ref here, it is used on both sides of the test
    PF._power_redistribution_ref(
        sys,
        2.4375,
        0.1875,
        ref_bus,
        PF.DEFAULT_MAX_REDISTRIBUTION_ITERATIONS,
    )

    # For PV bus, active and voltage are fixed; update reactive and angle
    pv_bus = get_bus(sys, 202)  # "Bacon"
    @assert get_bustype(pv_bus) == ACBusTypes.PV
    PF._reactive_power_redistribution_pv(
        sys,
        0.37267,
        pv_bus,
        PF.DEFAULT_MAX_REDISTRIBUTION_ITERATIONS,
    )
    set_angle!(pv_bus, -0.13778)

    # For PQ bus, active and reactive are fixed; update voltage and angle
    pq_bus = get_bus(sys, 117)  # "Aston"
    @assert get_bustype(pq_bus) == ACBusTypes.PQ
    set_magnitude!(pq_bus, 0.84783)
    set_angle!(pq_bus, 0.14956)
end

"Make the same changes to the PowerFlowData that modify_rts_system! makes to the System"
function modify_rts_power_flow!(data::PowerFlowData)
    bus_lookup = PF.get_bus_lookup(data)
    # For REF bus, voltage and angle are fixed; update active and reactive
    data.bus_active_power_injections[bus_lookup[113]] = 2.4375
    data.bus_reactive_power_injections[bus_lookup[113]] = 0.1875

    # For PV bus, active and voltage are fixed; update reactive and angle
    data.bus_reactive_power_injections[bus_lookup[202]] = 0.37267
    data.bus_angles[bus_lookup[202]] = -0.13778

    # For PQ bus, active and reactive are fixed; update voltage and angle
    data.bus_magnitude[bus_lookup[117]] = 0.84783
    data.bus_angles[bus_lookup[117]] = 0.14956
end

function _system_generation_power(
    sys::System,
    bus_numbers::Vector{Int},
)
    bus_power = zeros(Float64, length(bus_numbers))
    generators = collect(get_components(Union{Generator, Source}, sys))
    gen_power = zeros(Float64, length(generators))
    with_units_base(sys, UnitSystem.NATURAL_UNITS) do
        bus_power .= [
            isempty(g) ? 0 : sum([get_active_power(gg) for gg in g]) for g in [
                get_components(
                    x -> get_number(get_bus(x)) == i,
                    Union{Generator, Source},
                    sys,
                )
                for i in bus_numbers
            ]
        ]
        gen_power .= get_active_power.(generators)
    end
    return bus_power, gen_power
end

function _reset_gen_power!(
    sys::System,
    original_gen_power::Vector{Float64},
)
    with_units_base(sys, UnitSystem.NATURAL_UNITS) do
        for (g, og) in
            zip(get_components(Union{Generator, Source}, sys), original_gen_power)
            set_active_power!(g, og)
        end
    end
end

function _check_distributed_slack_consistency(
    subnetworks::Dict{Int, Vector{Int}},
    result_bus_power::Vector{Float64},
    slack_participation_factors::Vector{Float64},
    original_bus_power::Vector{Float64},
)
    for (_, subnetwork_buses) in subnetworks
        subnetwork_factors = slack_participation_factors[subnetwork_buses]
        slack_provided =
            result_bus_power[subnetwork_buses] .- original_bus_power[subnetwork_buses]
        nnz = subnetwork_factors .!= 0.0

        @test all(isapprox.(slack_provided[.!nnz], 0.0, atol = 1e-6, rtol = 0))
        @test !any(isapprox.(slack_provided[nnz], 0.0, atol = 1e-6, rtol = 0))
        @test isapprox(
            slack_provided[nnz] ./ sum(slack_provided),
            subnetwork_factors[nnz] ./ sum(subnetwork_factors);
            atol = 1e-6,
            rtol = 0,
        )
    end
    return
end

function _check_ds_pf(
    pf::ACPowerFlow,
    sys::System,
    bus_slack_participation_factors::Vector{Float64},
    bus_numbers::Vector{Int},
    original_bus_power::Vector{Float64},
    original_gen_power::Vector{Float64},
    data_original_bus_power::Vector{Float64},
)
    res = solve_power_flow(pf, sys)

    data = PowerFlowData(pf, sys)
    subnetworks = PowerFlows._find_subnetworks_for_reference_buses(
        data.power_network_matrix.data,
        data.bus_type[:, 1],
    )

    _check_distributed_slack_consistency(
        subnetworks,
        res["bus_results"][:, :P_gen],
        bus_slack_participation_factors,
        original_bus_power,
    )

    solve_and_store_power_flow!(pf, sys)
    p_solve, _ = _system_generation_power(sys, bus_numbers)

    @test isapprox(p_solve, res["bus_results"][:, :P_gen]; atol = 1e-6, rtol = 0)

    _reset_gen_power!(sys, original_gen_power)
    # to make sure the reset function is working properly:
    p_bus_reset, p_gen_reset = _system_generation_power(sys, bus_numbers)
    @test original_bus_power == p_bus_reset
    @test original_gen_power == p_gen_reset

    @test data.bus_slack_participation_factors[:, 1] == bus_slack_participation_factors
    solve_power_flow!(data)
    # now check the slack power distribution logic
    _check_distributed_slack_consistency(
        subnetworks,
        data.bus_active_power_injections[:, 1],
        bus_slack_participation_factors,
        data_original_bus_power,
    )
    return
end

"""These functions are used to create simple components for the tests to have more compact code"""

"""
    _check_name(sys::System, name::String, component_type::DataType)
    Check if the name is unique in the system. If not, append a number to the name.
"""
function _check_name(sys::System, name::String, component_type::DataType)
    # Check if the name is unique
    check = true
    i = 1
    while check
        if has_component(sys, component_type, name)
            i += 1
            name = name * "_$i"
        else
            check = false
        end
    end
    return name
end

"""
    _add_simple_bus!(sys::System, number::Int, bus_type::ACBusTypes, base_voltage::Number, voltage_magnitude::Float64=1.0, voltage_angle::Float64=0.0)
    Simplified function to create and add a bus to the system with the given parameters.
"""
function _add_simple_bus!(
    sys::System,
    number::Int,
    bus_type::ACBusTypes,
    base_voltage::Number,
    voltage_magnitude::Float64 = 1.0,
    voltage_angle::Float64 = 0.0,
)
    bus = ACBus(;
        number = number,
        name = _check_name(sys, "bus_$number", ACBus),
        available = true,
        bustype = bus_type,
        angle = voltage_angle,
        magnitude = voltage_magnitude,
        voltage_limits = (0.0, 2.0),
        base_voltage = Float64(base_voltage),
    )
    add_component!(sys, bus)
    return bus
end

"""
    _add_simple_load!(sys::System, bus::ACBus, active_power::Number, reactive_power::Number)
    Simplified function to create and add a load to the system with the given parameters.
"""
function _add_simple_load!(
    sys::System,
    bus::ACBus,
    active_power::Number,
    reactive_power::Number,
)
    load = PowerLoad(;
        name = _check_name(sys, "load_$(get_number(bus))", PowerLoad),
        available = true,
        bus = bus,
        active_power = Float64(active_power), # Per-unitized by device base_power
        reactive_power = Float64(reactive_power), # Per-unitized by device base_power
        base_power = 1.0, # MVA
        max_active_power = 100.0, # 10 MW per-unitized by device base_power
        max_reactive_power = 100.0,
    )

    add_component!(sys, load)
    return load
end

"""
    _add_simple_source!(sys::System, bus::ACBus, active_power::Number=0.0, reactive_power::Number=0.0)
    Simplified function to create and add a source to the system with the given parameters.
"""
function _add_simple_source!(
    sys::System,
    bus::ACBus,
    active_power::Number = 0.0,
    reactive_power::Number = 0.0,
)
    source = Source(;
        name = _check_name(sys, "source_$(get_number(bus))", Source),
        available = true,
        bus = bus,
        active_power = Float64(active_power),
        reactive_power = Float64(reactive_power),
        R_th = 1e-5,
        X_th = 1e-5,
    )
    add_component!(sys, source)
    return source
end

"""
    _add_simple_thermal_standard!(sys::System, bus::ACBus, active_power::Number=0.0, reactive_power::Number=0.0)
    Simplified function to create and add a thermal standard generator to the system with the given parameters.
"""
function _add_simple_thermal_standard!(
    sys::System,
    bus::ACBus,
    active_power::Number,
    reactive_power::Number,
)
    gen = ThermalStandard(;
        name = _check_name(sys, "thermal_standard_$(get_number(bus))", ThermalStandard),
        available = true,
        status = true,
        bus = bus,
        active_power = Float64(active_power),
        reactive_power = Float64(reactive_power),
        rating = 1.0,
        active_power_limits = (min = 0, max = 1),
        reactive_power_limits = (min = -1, max = 1),
        ramp_limits = nothing,
        operation_cost = ThermalGenerationCost(nothing),
        base_power = 100.0,
        time_limits = nothing,
        prime_mover_type = PrimeMovers.OT,
        fuel = ThermalFuels.OTHER,
        services = Device[],
        dynamic_injector = nothing,
        ext = Dict{String, Any}(),
    )
    add_component!(sys, gen)
    return gen
end

"""
    _add_simple_line!(sys::System, bus1::ACBus, bus2::ACBus, r::Float64=1e-3, x::Float64=1e-3, b::Float64=0.0)
    Simplified function to create and add a line to the system with the given parameters.
"""
function _add_simple_line!(
    sys::System,
    bus1::ACBus,
    bus2::ACBus,
    r::Float64 = 1e-3,
    x::Float64 = 1e-3,
    b::Float64 = 0.0,
)
    line = Line(;
        name = _check_name(sys, "line_$(get_number(bus1))_$(get_number(bus2))", Line),
        available = true,
        active_power_flow = 0.0,
        reactive_power_flow = 0.0,
        arc = Arc(; from = bus1, to = bus2),
        r = r,
        x = x,
        b = (from = b / 2, to = b / 2),
        rating = 1.0,
        angle_limits = (min = -pi / 2, max = pi / 2),
    )
    add_component!(sys, line)
    return line
end

"""
    Simplified function to create and add a standard load to the system with the given parameters.
"""
function _add_simple_zip_load!(
    sys::System,
    bus::ACBus;
    constant_power_active_power::Float64 = 0.0,
    constant_power_reactive_power::Float64 = 0.0,
    constant_current_active_power::Float64 = 0.0,
    constant_current_reactive_power::Float64 = 0.0,
    constant_impedance_active_power::Float64 = 0.0,
    constant_impedance_reactive_power::Float64 = 0.0,
)
    zip_load = StandardLoad(;
        name = _check_name(sys, "zip_load_$(get_number(bus))", StandardLoad),
        available = true,
        bus = bus,
        base_power = 10.0,
        constant_active_power = constant_power_active_power,
        constant_reactive_power = constant_power_reactive_power,
        current_active_power = constant_current_active_power,
        current_reactive_power = constant_current_reactive_power,
        impedance_active_power = constant_impedance_active_power,
        impedance_reactive_power = constant_impedance_reactive_power,
        max_constant_active_power = 0.0,
        max_constant_reactive_power = 0.0,
        max_impedance_active_power = 0.0,
        max_impedance_reactive_power = 0.0,
        max_current_active_power = 0.0,
        max_current_reactive_power = 0.0,
    )
    add_component!(sys, zip_load)
    return zip_load
end

function _add_simple_vsc!(
    sys,
    bus1::ACBus,
    bus2::ACBus;
    active_power_flow::Float64 = 0.5,
    loss_coefficient::Float64 = 0.01,
)
    vsc = TwoTerminalVSCLine(;
        name = _check_name(
            sys,
            "VSC_$(get_number(bus1))_$(get_number(bus2))",
            TwoTerminalVSCLine,
        ),
        available = true,
        arc = Arc(bus1, bus2),
        active_power_flow = active_power_flow,
        rating = 1.0,
        active_power_limits_from = (min = -1.0, max = 1.0),
        active_power_limits_to = (min = -1.0, max = 1.0),
        g = 0.0,
        dc_current = 0.0,
        reactive_power_from = 0.0,
        dc_control_from = PSY.VSCDCControlModes.DC_POWER,
        ac_control_from = PSY.VSCACControlModes.AC_REACTIVE_POWER,
        dc_setpoint_from = 0.0,
        ac_setpoint_from = 1.0,
        converter_loss_from = LinearCurve(loss_coefficient),
        max_dc_current_from = 1.0,
        rating_from = 1.0,
        reactive_power_limits_from = (min = -1.0, max = 1.0),
        power_factor_weighting_fraction_from = 0.0,
        voltage_limits_from = (min = 0.9, max = 1.1),
        reactive_power_to = 0.0,
        dc_control_to = PSY.VSCDCControlModes.DC_POWER,
        ac_control_to = PSY.VSCACControlModes.AC_REACTIVE_POWER,
        dc_setpoint_to = 0.0,
        ac_setpoint_to = 1.0,
        converter_loss_to = LinearCurve(loss_coefficient),
        max_dc_current_to = 1.0,
        rating_to = 1.0,
        reactive_power_limits_to = (min = -1.0, max = 1.0),
        power_factor_weighting_fraction_to = 0.0,
        voltage_limits_to = (min = 0.9, max = 1.1),
    )
    add_component!(sys, vsc)
    return vsc
end

function _add_simple_lcc!(
    sys,
    bus1::ACBus,
    bus2::ACBus,
    r::Float64,
    xr::Float64,
    xi::Float64,
)
    lcc = TwoTerminalLCCLine(;
        name = "LCC",
        available = true,
        arc = Arc(bus1, bus2),
        active_power_flow = 0.0,
        r = r,
        transfer_setpoint = 50,
        scheduled_dc_voltage = 800.0,
        rectifier_bridges = 1,
        rectifier_delay_angle_limits = (min = 0.0, max = π / 2),
        rectifier_rc = 0.0,
        rectifier_xc = xr,
        rectifier_base_voltage = 230.0,
        inverter_bridges = 1,
        # Keep operation conditions off the clamp.
        inverter_extinction_angle_limits = (min = deg2rad(17), max = π / 2),
        inverter_rc = 0.0,
        inverter_xc = xi,
        inverter_base_voltage = 230.0,
        power_mode = true,
        switch_mode_voltage = 0.0,
        compounding_resistance = 0.0,
        min_compounding_voltage = 0.0,
        rectifier_transformer_ratio = 1.0,
        rectifier_tap_setting = 1.0,
        rectifier_tap_limits = (min = 0.5, max = 1.5),
        rectifier_tap_step = 0.05,
        rectifier_delay_angle = 0.01,
        rectifier_capacitor_reactance = 0.0,
        inverter_transformer_ratio = 1.0,
        inverter_tap_setting = 1.0,
        inverter_tap_limits = (min = 0.5, max = 1.5),
        inverter_tap_step = 0.05,
        inverter_extinction_angle = deg2rad(17),
        inverter_capacitor_reactance = 0.0,
        active_power_limits_from = (min = 0.0, max = 0.0),
        active_power_limits_to = (min = 0.0, max = 0.0),
        reactive_power_limits_from = (min = 0.0, max = 0.0),
        reactive_power_limits_to = (min = 0.0, max = 0.0),
    )
    add_component!(sys, lcc)
    return lcc
end

function prepare_ts_data!(data::PowerFlowData, time_steps::Int64 = 24)
    injections = CSV.read(
        joinpath(TEST_DATA_DIR, "c_sys14_injections.csv"),
        DataFrame;
        header = 0,
    )
    withdrawals = CSV.read(
        joinpath(TEST_DATA_DIR, "c_sys14_withdrawals.csv"),
        DataFrame;
        header = 0,
    )
    # allocate data from csv
    injs = Matrix(injections)
    withs = Matrix(withdrawals)

    data.bus_active_power_injections .= deepcopy(injs[:, 1:time_steps])
    data.bus_active_power_withdrawals .= deepcopy(withs[:, 1:time_steps])
    # The CSVs carry ACTIVE power only. The scenario every consumer of this helper was
    # calibrated against has reactive power at the system snapshot for t=1 and ZERO for t>=2
    # (historically an artifact of column-1-only seeding; `initialize_power_flow_data!` now
    # seeds every column from the snapshot, so the zeros must be explicit). FDDecoupled's
    # Q-limit outer loop does not converge on all 24 steps with snapshot Q load everywhere.
    if time_steps > 1
        data.bus_reactive_power_injections[:, 2:end] .= 0.0
        data.bus_reactive_power_withdrawals[:, 2:end] .= 0.0
    end
    return
end

"""Build a minimal 3-bus system with one `TapTransformer` (VOLTAGE control) and one
`SwitchedAdmittance` for testing `build_controlled_device_set`."""
function _make_tap_shunt_system()
    sys = System(100.0)
    b1 = _add_simple_bus!(sys, 1, ACBusTypes.REF, 230, 1.0, 0.0)
    b2 = _add_simple_bus!(sys, 2, ACBusTypes.PQ, 230, 1.0, 0.0)
    b3 = _add_simple_bus!(sys, 3, ACBusTypes.PQ, 230, 1.0, 0.0)
    _add_simple_source!(sys, b1, 0.0, 0.0)
    _add_simple_load!(sys, b2, 0.1, 0.05)
    _add_simple_load!(sys, b3, 0.1, 0.05)
    # Line between buses 1 and 3 so the network is connected.
    _add_simple_line!(sys, b1, b3, 1e-2, 1e-2, 0.0)
    tap_arc = Arc(; from = b1, to = b2)
    tx = TapTransformer(;
        name = "tap_1_2",
        available = true,
        active_power_flow = 0.0,
        reactive_power_flow = 0.0,
        arc = tap_arc,
        r = 0.01,
        x = 0.10,
        primary_shunt = 0.0 + 0.0im,
        tap = 1.0,
        rating = 1.0,
        base_power = 100.0,
        control_objective = PSY.TransformerControlObjective.VOLTAGE,
    )
    add_component!(sys, tx)
    sa = SwitchedAdmittance(;
        name = "shunt_3",
        available = true,
        bus = b3,
        Y = 0.0 + 0.0im,
        initial_status = [0],
        number_of_steps = [4],
        Y_increase = [0.0 + 0.05im],
        admittance_limits = (min = 0.9, max = 1.1),
        control_mode = PSY.SwitchedAdmittanceControlMode.DISCRETE_VOLTAGE,
    )
    add_component!(sys, sa)
    return sys
end

"""Build a 3-bus system with one `TapTransformer` (VOLTAGE control) and one
`SwitchedAdmittance`, designed so the AC base case converges cleanly.

Bus 2 carries a significant load (0.5 pu on 100 MVA base) through a low-impedance
transformer from the REF bus, so that the tap is the sole voltage-support mechanism
for that bus.  Bus 3 is separately connected to the REF bus via a line and hosts the
shunt.  Buses 2 and 3 are decoupled (both see the REF bus but not each other), so
shunt adjustments do not perturb the tap-controlled bus and vice-versa.

The tap has full authority over bus 2 (dV/dp ≈ -1.04) and V₂ = vset = 1.0 is
reachable at tap ≈ 0.973, well inside [0.9, 1.1], so the damped steepness ramp
runs to completion and the continuation regulates the controlled bus into the
vset deadband before snapping.  Integration tests on this fixture assert both
convergence AND tight voltage proximity (within one discrete tap spacing)."""
function _make_solvable_tap_shunt_system()
    sys = System(100.0)
    b1 = _add_simple_bus!(sys, 1, ACBusTypes.REF, 230, 1.0, 0.0)
    b2 = _add_simple_bus!(sys, 2, ACBusTypes.PQ, 230, 1.0, 0.0)
    b3 = _add_simple_bus!(sys, 3, ACBusTypes.PQ, 230, 1.0, 0.0)
    _add_simple_source!(sys, b1, 0.0, 0.0)
    # Significant load so the base case converges with a non-trivial solution.
    load2 = PowerLoad(;
        name = "load_2",
        available = true,
        bus = b2,
        active_power = 0.5,
        reactive_power = 0.25,
        base_power = 100.0,
        max_active_power = 100.0,
        max_reactive_power = 100.0,
    )
    add_component!(sys, load2)
    load3 = PowerLoad(;
        name = "load_3",
        available = true,
        bus = b3,
        active_power = 0.05,
        reactive_power = 0.025,
        base_power = 100.0,
        max_active_power = 100.0,
        max_reactive_power = 100.0,
    )
    add_component!(sys, load3)
    # Bus 3 connected to REF bus; decoupled from bus 2.
    _add_simple_line!(sys, b1, b3, 1e-2, 1e-2, 0.0)
    tap_arc = Arc(; from = b1, to = b2)
    tx = TapTransformer(;
        name = "tap_1_2",
        available = true,
        active_power_flow = 0.0,
        reactive_power_flow = 0.0,
        arc = tap_arc,
        r = 0.01,
        x = 0.10,
        primary_shunt = 0.0 + 0.0im,
        tap = 1.0,
        rating = 1.0,
        base_power = 100.0,
        control_objective = PSY.TransformerControlObjective.VOLTAGE,
    )
    add_component!(sys, tx)
    sa = SwitchedAdmittance(;
        name = "shunt_3",
        available = true,
        bus = b3,
        Y = 0.0 + 0.0im,
        initial_status = [0],
        number_of_steps = [4],
        Y_increase = [0.0 + 0.05im],
        admittance_limits = (min = 0.9, max = 1.1),
        control_mode = PSY.SwitchedAdmittanceControlMode.DISCRETE_VOLTAGE,
    )
    add_component!(sys, sa)
    return sys
end

"""Build a 2-bus system with a weak PQ bus regulated by a `FACTSControlDevice` (SVC/STATCOM).

REF(1) ─line(x=0.10)─ PQ(2); bus 2 carries a reactive load that pulls its voltage below
1.0 p.u. The SVC at bus 2 (`voltage_setpoint=1.0`) injects reactive power to hold the bus.
`max_shunt_current` (MVA at unity voltage) sets the reactive capability; a small value forces
the SVC to saturate at its susceptance limit before reaching the setpoint."""
function _make_svc_system(;
    max_shunt_current::Float64 = 100.0,
    control_mode = PSY.FACTSOperationModes.NML,
    reactive_load::Float64 = 0.4,
)
    sys = System(100.0)
    b1 = _add_simple_bus!(sys, 1, ACBusTypes.REF, 230, 1.0, 0.0)
    b2 = _add_simple_bus!(sys, 2, ACBusTypes.PQ, 230, 1.0, 0.0)
    _add_simple_source!(sys, b1, 0.0, 0.0)
    _add_simple_line!(sys, b1, b2, 0.01, 0.10, 0.0)
    load2 = PowerLoad(;
        name = "load_2",
        available = true,
        bus = b2,
        active_power = 0.2,
        reactive_power = reactive_load,
        base_power = 100.0,
        max_active_power = 100.0,
        max_reactive_power = 100.0,
    )
    add_component!(sys, load2)
    svc = FACTSControlDevice(;
        name = "svc_2",
        available = true,
        bus = b2,
        control_mode = control_mode,
        voltage_setpoint = 1.0,
        max_shunt_current = max_shunt_current,
        reactive_power_required = 100.0,
    )
    add_component!(sys, svc)
    return sys
end

"""Add a `base_power = 100.0` `PowerLoad` (named `load_<busno>`) to the multiperiod fixtures."""
function _add_mp_load!(
    sys::System,
    bus::ACBus,
    active_power::Float64,
    reactive_power::Float64,
)
    load = PowerLoad(;
        name = "load_$(get_number(bus))",
        available = true,
        bus = bus,
        active_power = active_power,
        reactive_power = reactive_power,
        base_power = 100.0,
        max_active_power = 100.0,
        max_reactive_power = 100.0,
    )
    add_component!(sys, load)
    return load
end

"""Add a CONTINUOUS_VOLTAGE `SwitchedAdmittance` (named `shunt_<busno>`) regulating `bus`. The
narrow `admittance_limits` band (±5e-4 around 1.0) settles the continuous continuation tight to
the setpoint. `Y` is the FIXED susceptance (a nonzero value adds a constant-Z baseline, "b0")."""
function _add_cv_shunt!(sys::System, bus::ACBus; Y = 0.0 + 0.0im)
    sa = SwitchedAdmittance(;
        name = "shunt_$(get_number(bus))",
        available = true,
        bus = bus,
        Y = Y,
        initial_status = [0],
        number_of_steps = [12],
        Y_increase = [0.0 + 0.1im],
        admittance_limits = (min = 0.9995, max = 1.0005),
        control_mode = PSY.SwitchedAdmittanceControlMode.CONTINUOUS_VOLTAGE,
    )
    add_component!(sys, sa)
    return sa
end

"""Build a 2-bus system (REF—PQ) with one CONTINUOUS_VOLTAGE `SwitchedAdmittance` regulating the
PQ bus, for multiperiod discrete-control tests. The base-case load pulls bus 2 below the setpoint;
`_set_multiperiod_shunt_loads!` scales it per step so the required susceptance differs at each step.
`shunt_Y` sets the shunt's FIXED susceptance (nonzero = a constant-Z "b0" baseline)."""
function _make_multiperiod_shunt_system(; shunt_Y = 0.0 + 0.0im)
    sys = System(100.0)
    b1 = _add_simple_bus!(sys, 1, ACBusTypes.REF, 230, 1.0, 0.0)
    b2 = _add_simple_bus!(sys, 2, ACBusTypes.PQ, 230, 1.0, 0.0)
    _add_simple_source!(sys, b1, 0.0, 0.0)
    _add_simple_line!(sys, b1, b2, 0.01, 0.10, 0.0)
    _add_mp_load!(sys, b2, 0.2, 0.3)
    _add_cv_shunt!(sys, b2; Y = shunt_Y)
    return sys
end

"""Overwrite `data`'s per-time-step withdrawals for the 2-bus multiperiod fixtures.
`initialize_power_flow_data!` only populates column 1 from the system, so columns
`2:n` start at zero; this replicates the active-power withdrawal unchanged across
steps and scales the reactive withdrawal by `q_scale(t)` per step, so the controlled
device must settle at a DIFFERENT setting at each time step."""
function _set_multiperiod_loads!(data, n::Int, q_scale)
    base_p = copy(data.bus_active_power_withdrawals[:, 1])
    base_q = copy(data.bus_reactive_power_withdrawals[:, 1])
    for t in 1:n
        data.bus_active_power_withdrawals[:, t] .= base_p
        data.bus_reactive_power_withdrawals[:, t] .= base_q .* q_scale(t)
    end
    return
end

_shunt_step_q_scale(t::Int) = 0.8 + 0.2 * t
_facts_step_q_scale(t::Int) = 0.6 + 0.3 * t
_tap_step_q_scale(t::Int) = 0.8 + 0.2 * t

function _set_multiperiod_shunt_loads!(data, n::Int)
    return _set_multiperiod_loads!(data, n, _shunt_step_q_scale)
end

"""Network index of the regulated bus (PSY bus number 2) in the 2-bus
`_make_multiperiod_*_system` fixtures."""
function _regulated_bus_index(data)
    return PF.get_bus_lookup(data)[2]
end

"""`_make_multiperiod_shunt_system` with a nonzero FIXED shunt susceptance (`Y = 0.0 + 0.2im`), a
constant-Z "b0" baseline `_get_withdrawals!` folds into `bus_reactive_power_constant_impedance_withdrawals`.
Regression fixture for the bug where `initialize_power_flow_data!` seeded that baseline only into
column 1, so `ts≥2` silently lost it."""
function _make_multiperiod_shunt_system_with_baseline()
    return _make_multiperiod_shunt_system(; shunt_Y = 0.0 + 0.2im)
end

"""Replicate column 1's injections and withdrawals into every time-step column, producing an
IDENTICAL operating-point snapshot at each step. `initialize_power_flow_data!` seeds these (and the
reactive-power bounds) only into column 1; this fills the load/generation columns but deliberately
LEAVES the bounds untouched, so a multiperiod solve exercises the per-step bounds seeding."""
function _replicate_col1_to_all_steps!(data, n::Int)
    for t in 2:n
        data.bus_active_power_injections[:, t] .= data.bus_active_power_injections[:, 1]
        data.bus_reactive_power_injections[:, t] .= data.bus_reactive_power_injections[:, 1]
        data.bus_active_power_withdrawals[:, t] .= data.bus_active_power_withdrawals[:, 1]
        data.bus_reactive_power_withdrawals[:, t] .=
            data.bus_reactive_power_withdrawals[:, 1]
    end
    return
end

"""Independent single-time-step solve of `sys` (built with `pf`) at multiperiod step `t`'s load:
scale column 1's reactive withdrawal by `_shunt_step_q_scale(t)`. The gold-standard per-step parity
oracle — column 1 keeps its correctly-initialized baseline, only the load moves to step `t`."""
function _solve_single_ts_at_step(sys, pf, t::Int)
    data = PowerFlowData(pf, sys)
    data.bus_reactive_power_withdrawals[:, 1] .*= _shunt_step_q_scale(t)
    solve_power_flow!(data)
    return data
end

function _solve_shunt_single_ts_at_step(t::Int)
    return _solve_single_ts_at_step(
        _make_multiperiod_shunt_system_with_baseline(),
        ACPowerFlow{NewtonRaphsonACPowerFlow}(; control_discrete_devices = true),
        t,
    )
end

"""Build a 3-bus system (REF—PV—PQ) combining both control mechanisms: a PV-bus generator
with TIGHT reactive limits (so heavier steps force a PV→PQ Q-limit switch) and a
CONTINUOUS_VOLTAGE `SwitchedAdmittance` regulating the PQ bus. Exercises
`control_discrete_devices = true` together with `check_reactive_power_limits = true` in a
multiperiod solve."""
function _make_multiperiod_qlimit_shunt_system()
    sys = System(100.0)
    b1 = _add_simple_bus!(sys, 1, ACBusTypes.REF, 230, 1.0, 0.0)
    b2 = _add_simple_bus!(sys, 2, ACBusTypes.PV, 230, 1.0, 0.0)
    b3 = _add_simple_bus!(sys, 3, ACBusTypes.PQ, 230, 1.0, 0.0)
    _add_simple_source!(sys, b1, 0.0, 0.0)
    gen = _add_simple_thermal_standard!(sys, b2, 0.1, 0.0)
    set_reactive_power_limits!(gen, (min = -0.02, max = 0.02))
    _add_simple_line!(sys, b1, b2, 0.01, 0.10, 0.0)
    _add_simple_line!(sys, b2, b3, 0.01, 0.10, 0.0)
    _add_mp_load!(sys, b3, 0.2, 0.3)
    _add_cv_shunt!(sys, b3)
    return sys
end

"""`_solve_single_ts_at_step` for `_make_multiperiod_qlimit_shunt_system` with BOTH
`control_discrete_devices` and `check_reactive_power_limits` — the parity oracle for the
combined-mode multiperiod test."""
function _solve_qlimit_shunt_single_ts_at_step(t::Int)
    return _solve_single_ts_at_step(
        _make_multiperiod_qlimit_shunt_system(),
        ACPowerFlow{NewtonRaphsonACPowerFlow}(;
            control_discrete_devices = true, check_reactive_power_limits = true),
        t,
    )
end

"""Build a 2-bus system (REF—PQ) with a `FACTSControlDevice` in explicit SVC mode
(`shunt_control_type = SVC`) regulating the weak PQ bus, for multiperiod
discrete-control tests. `max_shunt_current = 100.0` gives the SVC ample susceptance
headroom so it never saturates identically across time steps; the base-case load
pulls bus 2 below `voltage_setpoint = 1.0` and `_set_multiperiod_facts_loads!` then
scales it per time step so the SVC settles at a DIFFERENT susceptance at each step."""
function _make_multiperiod_facts_system()
    sys = System(100.0)
    b1 = _add_simple_bus!(sys, 1, ACBusTypes.REF, 230, 1.0, 0.0)
    b2 = _add_simple_bus!(sys, 2, ACBusTypes.PQ, 230, 1.0, 0.0)
    _add_simple_source!(sys, b1, 0.0, 0.0)
    _add_simple_line!(sys, b1, b2, 0.01, 0.10, 0.0)
    _add_mp_load!(sys, b2, 0.2, 0.3)
    svc = FACTSControlDevice(;
        name = "svc_2",
        available = true,
        bus = b2,
        control_mode = PSY.FACTSOperationModes.NML,
        voltage_setpoint = 1.0,
        max_shunt_current = 100.0,
        shunt_control_type = PSY.FACTSShuntControlType.SVC,
        reactive_power_required = 100.0,
    )
    add_component!(sys, svc)
    return sys
end

function _set_multiperiod_facts_loads!(data, n::Int)
    return _set_multiperiod_loads!(data, n, _facts_step_q_scale)
end

"""Build a 2-bus system (REF—PQ) with one voltage-controlling `TapTransformer` regulating
the PQ bus, for multiperiod discrete-control tests (reset-to-baseline tap design). Mirrors
`_make_solvable_tap_shunt_system`'s impedance (r=0.01, x=0.10) and base load (0.5+j0.25) so
the tap has full authority over bus 2; the explicit control fields (`tap_limits`,
`number_of_tap_positions`, `regulated_bus_number`, `voltage_setpoint`) pin a fine tap grid
(31 positions over [0.85, 1.15]) so `_set_multiperiod_tap_loads!`'s per-step reactive-load
scaling drives the required tap to a DIFFERENT discrete position at each time step."""
function _make_multiperiod_tap_system()
    sys = System(100.0)
    b1 = _add_simple_bus!(sys, 1, ACBusTypes.REF, 230, 1.0, 0.0)
    b2 = _add_simple_bus!(sys, 2, ACBusTypes.PQ, 230, 1.0, 0.0)
    _add_simple_source!(sys, b1, 0.0, 0.0)
    _add_mp_load!(sys, b2, 0.5, 0.25)
    tap_arc = Arc(; from = b1, to = b2)
    tx = TapTransformer(;
        name = "tap_1_2",
        available = true,
        active_power_flow = 0.0,
        reactive_power_flow = 0.0,
        arc = tap_arc,
        r = 0.01,
        x = 0.10,
        primary_shunt = 0.0 + 0.0im,
        tap = 1.0,
        rating = 1.0,
        base_power = 100.0,
        tap_limits = (min = 0.85, max = 1.15),
        number_of_tap_positions = 31,
        regulated_bus_number = 2,
        voltage_setpoint = 1.0,
        control_objective = PSY.TransformerControlObjective.VOLTAGE,
    )
    add_component!(sys, tx)
    return sys
end

"""Overwrite `data`'s per-time-step withdrawals for `_make_multiperiod_tap_system`.
Column `t` equals `_set_single_tap_load!(data1, t)` exactly (both scale by
`_tap_step_q_scale`), so each multi-ts step can be checked against an independent
single-time-step solve at the same load (the gold-standard reset-to-baseline parity
check)."""
function _set_multiperiod_tap_loads!(data, n::Int)
    return _set_multiperiod_loads!(data, n, _tap_step_q_scale)
end

"""Set a SINGLE-time-step `data`'s (built with `time_steps=1`) column-1 withdrawal to the
SAME load `_set_multiperiod_tap_loads!` assigns to multi-ts column `t` — the load level an
independent single-time-step solve must reproduce for the gold-standard parity check."""
function _set_single_tap_load!(data, t::Int)
    data.bus_reactive_power_withdrawals[:, 1] .*= _tap_step_q_scale(t)
    return
end

"""Build a 2-bus system with an API-convention `SwitchedAdmittance` (empty `initial_status`,
not the parser's zeroed-status BINIT marker) at a weak PQ bus, tuned so the controlled solve
snaps the shunt to a NONZERO, non-saturated block count (`block_n = [3]` of 4 available steps)
while landing inside the [vswlo, vswhi] deadband — exercising the PRIMARY (realizable) branch
of `write_device_settings!`, not the never-snapped or unrealizable fallback."""
function _make_shunt_snap_system()
    sys = System(100.0)
    b1 = _add_simple_bus!(sys, 1, ACBusTypes.REF, 230, 1.0, 0.0)
    b2 = _add_simple_bus!(sys, 2, ACBusTypes.PQ, 230, 1.0, 0.0)
    _add_simple_source!(sys, b1, 0.0, 0.0)
    _add_simple_line!(sys, b1, b2, 0.01, 0.10, 0.0)
    load2 = PowerLoad(;
        name = "load_2",
        available = true,
        bus = b2,
        active_power = 0.2,
        reactive_power = 0.28,
        base_power = 100.0,
        max_active_power = 100.0,
        max_reactive_power = 100.0,
    )
    add_component!(sys, load2)
    sa = SwitchedAdmittance(;
        name = "shunt_2",
        available = true,
        bus = b2,
        Y = 0.0 + 0.0im,
        initial_status = Int[],
        number_of_steps = [4],
        Y_increase = [0.0 + 0.05im],
        admittance_limits = (min = 0.98, max = 1.02),
        control_mode = PSY.SwitchedAdmittanceControlMode.DISCRETE_VOLTAGE,
    )
    add_component!(sys, sa)
    return sys
end

"""Build a 3-bus system with one voltage-controlling `TapTransformer` whose controllability is set
through the FIRST-CLASS PSY fields (`tap_limits`, `number_of_tap_positions`, `regulated_bus_number`,
`voltage_setpoint`) — no `ext` scrape — to exercise the post-#1684 builder path. The tap (b1→b2)
remotely regulates b3."""
function _make_field_controlled_tap_system()
    sys = System(100.0)
    b1 = _add_simple_bus!(sys, 1, ACBusTypes.REF, 230, 1.0, 0.0)
    b2 = _add_simple_bus!(sys, 2, ACBusTypes.PQ, 230, 1.0, 0.0)
    b3 = _add_simple_bus!(sys, 3, ACBusTypes.PQ, 230, 1.0, 0.0)
    _add_simple_source!(sys, b1, 0.0, 0.0)
    _add_simple_line!(sys, b2, b3, 1e-2, 1e-2, 0.0)
    tx = TapTransformer(;
        name = "tap_1_2",
        available = true,
        active_power_flow = 0.0,
        reactive_power_flow = 0.0,
        arc = Arc(; from = b1, to = b2),
        r = 0.01,
        x = 0.10,
        primary_shunt = 0.0 + 0.0im,
        tap = 1.0,
        rating = 1.0,
        base_power = 100.0,
        tap_limits = (min = 0.85, max = 1.15),
        number_of_tap_positions = 17,
        regulated_bus_number = 3,
        voltage_setpoint = 1.02,
        control_objective = PSY.TransformerControlObjective.VOLTAGE,
    )
    add_component!(sys, tx)
    return sys
end

"""Build the IEEE 14-bus system (`PSB.PSITestSystems, "c_sys14"`) with every `PowerLoad`
scaled by `load_scale`. Used for testing reactive power control logic: switched shunt and
FACTS device adjustment."""
function _make_ieee14_scaled_load_system(load_scale::Float64 = 1.4)
    sys = PSB.build_system(PSB.PSITestSystems, "c_sys14"; add_forecasts = false)
    set_units_base_system!(sys, "SYSTEM_BASE")
    for load in get_components(PowerLoad, sys)
        set_active_power!(load, get_active_power(load) * load_scale)
        set_reactive_power!(load, get_reactive_power(load) * load_scale)
    end
    return sys
end

"""Add a continuous shunt `FACTSControlDevice` (SVC/STATCOM-style) at `bus_number` targeting
`voltage_setpoint`."""
function _add_facts_shunt!(
    sys,
    bus_number::Int;
    voltage_setpoint::Float64 = 1.0,
    max_shunt_current::Float64 = 100.0,
)
    b = get_bus(sys, bus_number)
    facts = FACTSControlDevice(;
        name = "facts_$bus_number",
        available = true,
        bus = b,
        control_mode = PSY.FACTSOperationModes.NML,
        voltage_setpoint = voltage_setpoint,
        max_shunt_current = max_shunt_current,
        reactive_power_required = 100.0,
    )
    add_component!(sys, facts)
    return facts
end

"""Add a discrete `SwitchedAdmittance` (PSS/E-style block-switched capacitor bank) at
`bus_number`, built from `n_steps` blocks of `mvar_per_step` MVar each so the total capacity
(`n_steps * mvar_per_step` MVar) doesn't saturate. `voltage_setpoint` becomes the midpoint of
a narrow `admittance_limits` deadband (`± deadband`, in p.u.) — narrow enough that the
discrete continuation drives the bus close to `voltage_setpoint` rather than stopping as soon
as voltage enters a wide PSS/E-style VSWLO/VSWHI band."""
function _add_switched_shunt!(
    sys,
    bus_number::Int;
    voltage_setpoint::Float64 = 1.0,
    n_steps::Int = 60,
    mvar_per_step::Float64 = 1.0,
    deadband::Float64 = 0.001,
)
    base_power = get_base_power(sys)
    b = get_bus(sys, bus_number)
    sa = SwitchedAdmittance(;
        name = "shunt_$bus_number",
        available = true,
        bus = b,
        Y = 0.0 + 0.0im,
        initial_status = [0],
        number_of_steps = [n_steps],
        Y_increase = [0.0 + (mvar_per_step / base_power) * im],
        admittance_limits = (
            min = voltage_setpoint - deadband,
            max = voltage_setpoint + deadband,
        ),
        control_mode = SwitchedAdmittanceControlMode.DISCRETE_VOLTAGE,
    )
    add_component!(sys, sa)
    return sa
end

"""Build a 4-bus system where the TapTransformer's FROM bus is the controlled bus,
exercising the from-side control orientation (the plant-sign probe must measure the
opposite dV/dp sign to the usual to-side wiring).

Topology: REF(1) ─line─ PQ(2) ─tap─ PQ(3); REF(1) ─line─ PQ(4).
Bus 2 is both the FROM bus of the tap and the controlled bus (set via `regulated_bus_number`).
The tap has real authority over bus 2 voltage through the impedance seen by bus 2."""
function _make_primary_controlled_tap_system()
    sys = System(100.0)
    b1 = _add_simple_bus!(sys, 1, ACBusTypes.REF, 230, 1.0, 0.0)
    b2 = _add_simple_bus!(sys, 2, ACBusTypes.PQ, 230, 1.0, 0.0)
    b3 = _add_simple_bus!(sys, 3, ACBusTypes.PQ, 230, 1.0, 0.0)
    b4 = _add_simple_bus!(sys, 4, ACBusTypes.PQ, 230, 1.0, 0.0)
    _add_simple_source!(sys, b1, 0.0, 0.0)
    _add_simple_load!(sys, b2, 0.3, 0.15)
    _add_simple_load!(sys, b3, 0.1, 0.05)
    _add_simple_load!(sys, b4, 0.05, 0.025)
    # Line feeds bus 2 from REF.
    _add_simple_line!(sys, b1, b2, 1e-2, 5e-2, 0.0)
    # Bus 4 connected to REF (keeps network connected after b3 has only the tap).
    _add_simple_line!(sys, b1, b4, 1e-2, 1e-2, 0.0)
    tap_arc = Arc(; from = b2, to = b3)
    tx = TapTransformer(;
        name = "tap_2_3",
        available = true,
        active_power_flow = 0.0,
        reactive_power_flow = 0.0,
        arc = tap_arc,
        r = 0.01,
        x = 0.10,
        primary_shunt = 0.0 + 0.0im,
        tap = 1.0,
        rating = 1.0,
        base_power = 100.0,
        control_objective = PSY.TransformerControlObjective.VOLTAGE,
        regulated_bus_number = 2,  # controlled bus = bus 2 (FROM) → primary
    )
    add_component!(sys, tx)
    return sys
end

function simple_lcc_system()
    sys = System(100.0)
    b1 = _add_simple_bus!(sys, 1, ACBusTypes.REF, 230, 1.1, 0.0)
    b2 = _add_simple_bus!(sys, 2, ACBusTypes.PQ, 230, 1.1, 0.0)
    b3 = _add_simple_bus!(sys, 3, ACBusTypes.PQ, 230, 1.1, 0.0)
    _add_simple_load!(sys, b2, 10, 5)
    _add_simple_load!(sys, b3, 60, 20)
    _add_simple_line!(sys, b1, b2, 5e-3, 5e-3, 1e-3)
    _add_simple_line!(sys, b1, b3, 5e-3, 5e-3, 1e-3)
    _add_simple_source!(sys, b1, 0.0, 0.0)
    lcc = _add_simple_lcc!(sys, b2, b3, 0.05, 0.05, 0.08)
    return sys, lcc
end

"""Validate that `data.arc_angle_differences` matches θ_from − θ_to for every arc
at each time step in `time_steps`."""
function validate_arc_angle_differences(data::PowerFlowData, time_steps::Vector{Int})
    arc_axis = PF.get_arc_axis(data)
    bus_lookup = PF.get_bus_lookup(data)
    for t in time_steps
        for (arc_ix, arc) in enumerate(arc_axis)
            from_ix = bus_lookup[first(arc)]
            to_ix = bus_lookup[last(arc)]
            expected = data.bus_angles[from_ix, t] - data.bus_angles[to_ix, t]
            @test isapprox(data.arc_angle_differences[arc_ix, t], expected; atol = 1e-12)
        end
    end
end

"""Validate that DC branch losses equal R * flow^2 for every row in the results DataFrame.
Computes expected losses directly from the DataFrame's P_from_to column, avoiding
arc ordering issues between internal data arrays and the sorted DataFrame."""
function validate_dc_branch_losses(
    data::PowerFlowData,
    results::Dict,
    base_power::Float64,
    time_steps::Vector{Int},
)
    # Build a lookup from (bus_from, bus_to) -> resistance.
    Rs = PF._get_arc_resistances(data)
    arc_axis = PF.get_arc_axis(data)
    r_lookup = Dict{Tuple{Int, Int}, Float64}()
    for (ix, arc) in enumerate(arc_axis)
        r_lookup[(first(arc), last(arc))] = Rs[ix]
    end

    for t in time_steps
        flow_df = results[string(t)]["flow_results"]
        for row in eachrow(flow_df)
            r = r_lookup[(row[:bus_from], row[:bus_to])]
            flow_pu = row[:P_from_to] / base_power
            expected_loss_mw = r * flow_pu^2 * base_power
            @test isapprox(row[:P_losses], expected_loss_mw; atol = 1e-6)
        end
    end
end

function power_flow_with_units(
    sys::PSY.System,
    T::Type{<:PF.ACPowerFlow},
    units::PSY.UnitSystem,
)
    with_units_base(sys, units) do
        results = solve_power_flow(T(; correct_bustypes = true), sys)
        if "1" in keys(results)
            first_line_flow = results["1"]["flow_results"][1, :]
        else
            first_line_flow = results["flow_results"][1, :]
        end
        return (first_line_flow[:flow_name], first_line_flow[:P_from_to])
    end
end

function power_flow_with_units(
    sys::PSY.System,
    T::Type{<:PF.AbstractDCPowerFlow},
    units::PSY.UnitSystem,
)
    with_units_base(sys, units) do
        results =
            solve_power_flow(T(; correct_bustypes = true), sys, PF.FlowReporting.ARC_FLOWS)
        if "1" in keys(results)
            first_line_flow = results["1"]["flow_results"][1, :]
        else
            first_line_flow = results["flow_results"][1, :]
        end
        return (first_line_flow[:flow_name], first_line_flow[:P_from_to])
    end
end

# Reconstruct the polar state vector from solved `data` (REF→(P,Q),
# PV→(Q,θ), PQ→(Vm,θ)). Restored from the removed `legacy_pf.jl`: PR #370
# deleted that file (the duplicate matrix-based NR implementation) but this is
# a pure data→state-format helper, unrelated to the legacy solver, and #370
# left its callers in `test_solve_power_flow.jl`.
function _calc_x(
    data::PowerFlows.ACPowerFlowData,
    time_step::Int64,
)
    n_buses = first(size(data.bus_type))
    x = zeros(Float64, 2 * n_buses)
    bus_types = view(data.bus_type, :, time_step)
    for (ix, bt) in enumerate(bus_types)
        if bt == PSY.ACBusTypes.REF
            x[2 * ix - 1] =
                data.bus_active_power_injections[ix, time_step] -
                data.bus_active_power_withdrawals[ix, time_step]
            x[2 * ix] =
                data.bus_reactive_power_injections[ix, time_step] -
                data.bus_reactive_power_withdrawals[ix, time_step]
        elseif bt == PSY.ACBusTypes.PV
            x[2 * ix - 1] =
                data.bus_reactive_power_injections[ix, time_step] -
                data.bus_reactive_power_withdrawals[ix, time_step]
            x[2 * ix] = data.bus_angles[ix, time_step]
        elseif bt == PSY.ACBusTypes.PQ
            x[2 * ix - 1] = data.bus_magnitude[ix, time_step]
            x[2 * ix] = data.bus_angles[ix, time_step]
        end
    end
    return x
end

# Reuse an existing same-orientation Arc between two buses if present, else make and add one.
# Orientation is significant and must match: a VSC/HVDC line's from/to terminals carry distinct
# controls, so an Arc oriented to->from must NOT be reused (it would swap the converter terminals).
# A correctly-oriented parallel Arc is created instead; sharing only applies to a same-oriented branch.
function _get_or_make_arc(sys, from_bus, to_bus)
    existing = PSY.get_components(
        a -> PSY.get_from(a) === from_bus && PSY.get_to(a) === to_bus,
        PSY.Arc,
        sys,
    )
    isempty(existing) || return first(existing)
    arc = PSY.Arc(; from = from_bus, to = to_bus)
    PSY.add_component!(sys, arc)
    return arc
end

# Add a voltage-controlling TapTransformer between two existing AC buses (mirrors the
# TapTransformer block in `_make_tap_shunt_system`), for fixtures that need a controlled
# device layered on top of an otherwise-fixed system (e.g. a VSC system).
function _add_control_tap!(sys, from_bus, to_bus; name = "tap_ctrl")
    tap_arc = _get_or_make_arc(sys, from_bus, to_bus)
    tx = PSY.TapTransformer(;
        name = name,
        available = true,
        active_power_flow = 0.0,
        reactive_power_flow = 0.0,
        arc = tap_arc,
        r = 0.01,
        x = 0.10,
        primary_shunt = 0.0 + 0.0im,
        tap = 1.0,
        rating = 1.0,
        base_power = 100.0,
        control_objective = PSY.TransformerControlObjective.VOLTAGE,
    )
    PSY.add_component!(sys, tx)
    return tx
end

# ── Shared VSC test builders ────────────────────────────────────────────────────────────────────

const VSC_SETTINGS = Dict{Symbol, Any}(:model_dc_network => true)

# One point-to-point VSC line between the first two PQ buses of c_sys14: from = DC-voltage control
# (DC slack), to = (P, Q) control. Extra `TwoTerminalVSCLine` fields pass through `vsc_kwargs...`
# (last-wins, so callers may override the defaults below, e.g. capability limits).
function _build_vsc_pq_system(;
    g = 50.0,
    p_set = 0.4,
    q_set = 0.1,
    vdc = 1.05,
    q_set_from = 0.05,
    name = "vsc_i1",
    active_power_flow = p_set,
    set_system_base = false,
    vsc_kwargs...,
)
    sys = deepcopy(PSB.build_system(PSB.PSITestSystems, "c_sys14"; add_forecasts = false))
    set_system_base && PSY.set_units_base_system!(sys, "SYSTEM_BASE")
    pq = sort!(
        collect(
            PSY.get_components(
                b -> PSY.get_bustype(b) == PSY.ACBusTypes.PQ,
                PSY.ACBus,
                sys,
            ),
        );
        by = PSY.get_number,
    )
    from_bus = pq[1]
    to_bus = pq[2]
    arc = _get_or_make_arc(sys, from_bus, to_bus)
    vsc = PSY.TwoTerminalVSCLine(;
        name = name,
        available = true,
        arc = arc,
        active_power_flow = active_power_flow,
        rating = 2.0,
        active_power_limits_from = (min = -2.0, max = 2.0),
        active_power_limits_to = (min = -2.0, max = 2.0),
        g = g,
        # from: DC-voltage control (slack), reactive setpoint
        dc_control_from = PSY.VSCDCControlModes.DC_VOLTAGE,
        ac_control_from = PSY.VSCACControlModes.AC_REACTIVE_POWER,
        dc_setpoint_from = vdc,
        reactive_power_from = q_set_from,
        # to: power control (P, Q)
        dc_control_to = PSY.VSCDCControlModes.DC_POWER,
        ac_control_to = PSY.VSCACControlModes.AC_REACTIVE_POWER,
        dc_setpoint_to = p_set,
        reactive_power_to = q_set,
        vsc_kwargs...,
    )
    PSY.add_component!(sys, vsc)
    return sys
end

# 3-terminal MTDC on c_sys14: DCBus nodes + InterconnectingConverter (AC↔DC) + TModelHVDCLine DC
# branches. ic1 = DC-voltage slack (1.05), ic2/ic3 = power orders; all ICs start with
# `active_power = 0.0`.
function _build_mtdc_system()
    sys = deepcopy(PSB.build_system(PSB.PSITestSystems, "c_sys14"; add_forecasts = false))
    PSY.set_units_base_system!(sys, "SYSTEM_BASE")
    pq = sort!(
        collect(
            PSY.get_components(
                b -> PSY.get_bustype(b) == PSY.ACBusTypes.PQ,
                PSY.ACBus,
                sys,
            ),
        );
        by = PSY.get_number,
    )
    ac = pq[1:3]
    dcbuses = PSY.DCBus[]
    for k in 1:3
        dcb = PSY.DCBus(;
            number = 100 + k,
            name = "dc$k",
            available = true,
            magnitude = 1.0,
            voltage_limits = (min = 0.8, max = 1.2),
            base_voltage = 230.0,
        )
        PSY.add_component!(sys, dcb)
        push!(dcbuses, dcb)
    end
    configs = (
        (dc_control = PSY.VSCDCControlModes.DC_VOLTAGE, dc_setpoint = 1.05),
        (dc_control = PSY.VSCDCControlModes.DC_POWER, dc_setpoint = 0.30),
        (dc_control = PSY.VSCDCControlModes.DC_POWER, dc_setpoint = 0.20),
    )
    for k in 1:3
        ic = PSY.InterconnectingConverter(;
            name = "ic$k",
            available = true,
            bus = ac[k],
            dc_bus = dcbuses[k],
            active_power = 0.0,
            rating = 3.0,
            active_power_limits = (min = -3.0, max = 3.0),
            base_power = 100.0,
            dc_control = configs[k].dc_control,
            ac_control = PSY.VSCACControlModes.AC_REACTIVE_POWER,
            dc_setpoint = configs[k].dc_setpoint,
        )
        PSY.add_component!(sys, ic)
    end
    # DC branches dc1–dc2 and dc2–dc3
    for (a, c) in ((1, 2), (2, 3))
        arc = PSY.Arc(; from = dcbuses[a], to = dcbuses[c])
        PSY.add_component!(sys, arc)
        dcl = PSY.TModelHVDCLine(;
            name = "dcline$(a)$(c)",
            available = true,
            active_power_flow = 0.0,
            arc = arc,
            r = 0.01,
            l = 0.0,
            c = 0.0,
            active_power_limits_from = (min = -5.0, max = 5.0),
            active_power_limits_to = (min = -5.0, max = 5.0),
        )
        PSY.add_component!(sys, dcl)
    end
    return sys
end
