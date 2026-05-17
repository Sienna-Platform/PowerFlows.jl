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
        dc_voltage_control_from = false,
        ac_voltage_control_from = false,
        dc_setpoint_from = 0.0,
        ac_setpoint_from = 1.0,
        converter_loss_from = LinearCurve(loss_coefficient),
        max_dc_current_from = 1.0,
        rating_from = 1.0,
        reactive_power_limits_from = (min = -1.0, max = 1.0),
        power_factor_weighting_fraction_from = 0.0,
        voltage_limits_from = (min = 0.9, max = 1.1),
        reactive_power_to = 0.0,
        dc_voltage_control_to = false,
        ac_voltage_control_to = false,
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
        inverter_extinction_angle_limits = (min = 0, max = π / 2),
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
        inverter_extinction_angle = 0.0,
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
    return
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
