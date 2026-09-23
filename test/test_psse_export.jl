test_psse_export_dir = joinpath(BASE_DIR, "test", "test_exports")
isdir(test_psse_export_dir) && rm(test_psse_export_dir; recursive = true)

function _log_assert(result, msg, comparison_name)
    result ||
        @error "Failed check: $(string(msg))$(isnothing(comparison_name) ? "" :  " ($comparison_name)")"
    return result
end
"""
If the expression is false, log an error; in any case, pass through the result of the
expression. Optionally accepts a name to include in the error log.
"""
macro log_assert(ex, comparison_name = nothing)
    return :(_log_assert($(esc(ex)), $(string(ex)), $(esc(comparison_name))))
end

"""
Compare the two dataframes by column. Specify tolerances using kwargs; tolerances default to
default_tol. If tolerance is `nothing`, skip that column. Otherwise, if the column is
floating point, compare element-wise with `isapprox(atol = tolerance)`; if not, test strict
equality element-wise. Optionally accepts a name to include in any failure logs.
"""
function compare_df_within_tolerance(
    comparison_name::String,
    df1::DataFrame,
    df2::DataFrame,
    default_tol = SYSTEM_REIMPORT_COMPARISON_TOLERANCE;
    kwargs...,
)
    result = true
    n_rows_match = (@log_assert size(df1, 1) == size(df2, 1) comparison_name)
    result &= n_rows_match
    result &= (@log_assert names(df1) == names(df2) comparison_name)
    result &= (@log_assert eltype.(eachcol(df1)) == eltype.(eachcol(df2)) comparison_name)
    n_rows_match || return result  # Can't compare the cols if number of rows doesn't match
    for (colname, my_eltype, col1, col2) in
        zip(names(df1), eltype.(eachcol(df1)), eachcol(df1), eachcol(df2))
        my_tol = (Symbol(colname) in keys(kwargs)) ? kwargs[Symbol(colname)] : default_tol
        isnothing(my_tol) && continue
        inner_result = if (my_eltype <: AbstractFloat)
            all(isapprox.(col1, col2; atol = my_tol))
        else
            all(IS.isequivalent.(col1, col2))
        end
        inner_result ||
            (@error "Mismatch on $colname$((my_eltype <: AbstractFloat) ? ", max discrepancy $(maximum(abs.(col2 - col1)))" : "") ($comparison_name)")
        result &= inner_result
    end
    return result
end

compare_df_within_tolerance(
    df1::DataFrame,
    df2::DataFrame,
    default_tol = SYSTEM_REIMPORT_COMPARISON_TOLERANCE;
    kwargs...,
) = compare_df_within_tolerance("unnamed", df1, df2, default_tol; kwargs...)

# If we have a name like "Bus1-Bus2-OtherInfo," reverse it to "Bus2-Bus1-OtherInfo"
function reverse_composite_name(name::String)
    parts = split(name, "-")
    (length(parts) > 2) || return name
    return join([parts[2], parts[1], parts[3:end]...], "-")
end

loose_system_match_fn(a::Float64, b::Float64) =
    isapprox(a, b; atol = SYSTEM_REIMPORT_COMPARISON_TOLERANCE,
        rtol = SYSTEM_REIMPORT_RELATIVE_TOLERANCE) || IS.isequivalent(a, b)
loose_system_match_fn(a, b) = IS.isequivalent(a, b)

"""PSS/E's COD field has no spelling for `UNDEFINED`, so an unset control objective exports
blank and re-parses as `FIXED`. Every other objective round-trips exactly."""
function expected_reimported_objective(objective::PSY.TransformerControlObjective.Value)
    if objective == PSY.TransformerControlObjective.UNDEFINED
        return PSY.TransformerControlObjective.FIXED
    end
    return objective
end

function _circuit_objective_round_trips(
    circuit1::PSY.TransformerCircuit,
    circuit2::PSY.TransformerCircuit,
)
    original = PSY.get_control_objective(circuit1)
    actual = PSY.get_control_objective(circuit2)
    # This helper also compares a system against itself, where nothing was exported and the
    # objective is untouched. So accept the original, or the one documented lossy mapping a
    # re-import applies. Any other objective is a genuine COD export bug.
    if actual == original || actual == expected_reimported_objective(original)
        return true
    end
    @error "control_objective did not round-trip: expected $original or \
        $(expected_reimported_objective(original)), got $actual"
    return false
end

# `compare_systems_loosely` excludes `:control_objective` because `IS.compare_values` matches
# exclusions by field name recursively, so it cannot express the asymmetric UNDEFINED→FIXED
# mapping. These methods restore the check as an explicit contract, so a genuine COD export
# bug on any of the other objectives still fails the round trip. Both arities are checked at
# the `TransformerCircuit` level, since that is where the objective lives.
_control_objectives_round_trip(comp1, comp2) = true

_control_objectives_round_trip(
    tx1::PSY.TwoWindingTransformer,
    tx2::PSY.TwoWindingTransformer,
) = _circuit_objective_round_trips(PSY.get_circuit(tx1), PSY.get_circuit(tx2))

function _control_objectives_round_trip(
    tx1::PSY.ThreeWindingTransformer,
    tx2::PSY.ThreeWindingTransformer,
)
    result = true
    for (circuit1, circuit2) in zip(PSY.get_circuits(tx1), PSY.get_circuits(tx2))
        result &= _circuit_objective_round_trips(circuit1, circuit2)
    end
    return result
end

function compare_systems_loosely(sys1::PSY.System, sys2::PSY.System;
    bus_name_mapping = Dict{String, String}(),
    include_types = [
        PSY.ACBus,
        PSY.Arc,
        PSY.Area,
        PSY.DiscreteControlledACBranch,
        PSY.FACTSControlDevice,
        PSY.FixedAdmittance,
        PSY.InterruptibleStandardLoad,
        PSY.Line,
        PSY.LoadZone,
        PSY.StandardLoad,
        PSY.SwitchedAdmittance,
        PSY.ThermalStandard,
        PSY.ThreeWindingTransformer,
        PSY.TwoTerminalLCCLine,
        PSY.TwoTerminalVSCLine,
        PSY.TwoWindingTransformer,
    ],
    # TODO when possible, don't exclude so many fields
    exclude_fields = Set([
        :ext,
        :ramp_limits,
        :time_limits,
        :services,
        :angle_limits,
        :control_objective_primary,
    ]),
    exclude_fields_for_type = Dict(
        PSY.ThermalStandard => Set([
            :prime_mover_type,
            :rating,
            :fuel,
            :dynamic_injector,
            :operation_cost,
        ]),
        PSY.LoadZone => Set([
            :peak_active_power,
            :peak_reactive_power,
        ]),
        PSY.Line => Set([
            :active_power_flow,
            :reactive_power_flow,
        ]),
        PSY.TwoWindingTransformer => Set([
            :active_power_flow,
            :reactive_power_flow,
            # PSS/E's COD field can't spell "no control block" (`UNDEFINED`), so it exports
            # blank and re-parses as `FIXED`. Excluded here only because IS matches
            # exclusions by name recursively; `_control_objectives_round_trip` asserts the
            # exact mapping instead, so the other objectives are still checked.
            :control_objective,
        ]),
        PSY.ThreeWindingTransformer => Set([
            :active_power_flow,
            :reactive_power_flow,
            :rating,  # TODO why don't ratings match?
            :control_objective,  # same UNDEFINED→FIXED mapping; see the 2W note above
        ]),
        # PSS/E's two-terminal DC records do not contain PSY's active/reactive power limit
        # tuples, so those fields cannot be recovered by a raw-file round trip.
        PSY.TwoTerminalLCCLine => Set([
            :active_power_limits_from,
            :active_power_limits_to,
            :reactive_power_limits_from,
            :reactive_power_limits_to,
        ]),
        # PowerFlowFileParser does not preserve the v33 FACTS SHMX/TRMX fields during
        # re-import; both PSY fields are reconstructed from the parser's 9999.0 default.
        PSY.FACTSControlDevice => Set([:max_shunt_current, :max_reactive_power]),
    ),
    generator_comparison_fns = [  # TODO rating
        PSY.get_name,
        PSY.get_bus,
        PSY.get_active_power,
        PSY.get_reactive_power,
        PSY.get_base_power,
    ],
    ignore_name_order = true,
    ignore_extra_of_type = Union{PSY.ThermalStandard, PSY.StaticLoad},
    exclude_reactive_power = false)
    result = true
    if exclude_reactive_power
        push!(exclude_fields, :reactive_power)
        generator_comparison_fns =
            filter(!=(PSY.get_reactive_power), generator_comparison_fns)
    end

    # Compare system-level metadata here; components are compared below.
    result &= IS.compare_values(sys1, sys2; exclude = [:data, :internal])

    # Compare the components by concrete type
    for my_type in include_types
        !isconcretetype(my_type) &&
            throw(ArgumentError("All `include_types` must be concrete, got $my_type"))

        names1 = collect(PSY.get_name.(PSY.get_components(my_type, sys1)))
        predicted_names2 = replace.(names1, bus_name_mapping...)
        actual_names2 = collect(PSY.get_name.(PSY.get_components(my_type, sys2)))

        if ignore_name_order
            for (i, predicted) in enumerate(predicted_names2)
                if !(predicted in actual_names2) &&
                   reverse_composite_name(predicted) in actual_names2
                    @info "Reversing name $predicted"
                    predicted_names2[i] = reverse_composite_name(predicted)
                end
            end
        end

        if my_type <: ignore_extra_of_type
            if !isempty(setdiff(predicted_names2, actual_names2))
                @error "Predicting generator names that do not exist for $my_type"
                result = false
            end
            (Set(predicted_names2) != Set(actual_names2)) &&
                @warn "Predicted $my_type names are a strict subset of actual $my_type names"
        else
            if Set(predicted_names2) != Set(actual_names2)
                @error "Predicted names do not match actual names for $my_type"
                @error "Predicted: $(sort(collect(Set(predicted_names2))))"
                @error "Actual: $(sort(collect(Set(actual_names2))))"
                result = false
            end
        end

        tr3w_starbuses =
            PSY.get_name.(
                PSY.get_star_bus.(
                    PSY.get_components(PSY.ThreeWindingTransformer, sys1)
                )
            )
        my_excludes =
            union(Set(exclude_fields), get(exclude_fields_for_type, my_type, Set()))
        for (name1, name2) in zip(names1, predicted_names2)
            (name2 in actual_names2) || continue
            # Do not compare starbuses of 3-winding transformers
            (name1 in tr3w_starbuses || name2 in tr3w_starbuses) && continue
            comp1 = PSY.get_component(my_type, sys1, name1)
            comp2 = PSY.get_component(my_type, sys2, name2)
            @assert !isnothing(comp2) comp2

            comparison = IS.compare_values(
                loose_system_match_fn,
                comp1,
                comp2;
                exclude = my_excludes,
            )
            objectives_ok = _control_objectives_round_trip(comp1, comp2)
            result &= comparison & objectives_ok
            if !comparison || !objectives_ok
                @error "Mismatched component LHS: $comp1"
                @error "Mismatched component RHS: $comp2"
            end
        end
    end

    # Extra checks for other types of generators
    GenLike = Union{Generator, Source, Storage}
    gen1_names = sort(PSY.get_name.(PSY.get_components(GenLike, sys1)))
    gen2_names = sort(PSY.get_name.(PSY.get_components(GenLike, sys2)))
    if gen1_names != gen2_names
        @error "Predicted Generator/Source/Storage names do not match actual generator names"
        @error "Predicted: $gen1_names"
        @error "Actual: $gen2_names"
        result = false
    end
    gen_common_names = intersect(gen1_names, gen2_names)
    for (gen1, gen2) in zip(
        PSY.get_component.(GenLike, [sys1], gen_common_names),
        PSY.get_component.(GenLike, [sys2], gen_common_names),
    )
        # Skip pairs we've already compared
        # e.g., if they're both ThermalStandards, we've already compared them
        any(Union{typeof(gen1), typeof(gen2)} .<: include_types) && continue
        for comp_fn in generator_comparison_fns
            comparison = IS.compare_values(
                loose_system_match_fn,
                comp_fn(gen1),
                comp_fn(gen2);
                exclude = exclude_fields,
            )
            result &= comparison
            if !comparison
                @error "Generator $(get_name(gen1)) mismatch on $comp_fn: $(comp_fn(gen1)) vs. $(comp_fn(gen2))"
            end
        end
    end
    return result
end

function test_power_flow(
    pf::ACPowerFlow{<:ACPowerFlowSolverType},
    sys1::System,
    sys2::System;
    exclude_reactive_flow = false,
)
    pf_with_bustypes = ACPowerFlow{typeof(pf).parameters[1]}(; correct_bustypes = true)
    result1 = solve_power_flow(pf_with_bustypes, sys1)
    result2 = solve_power_flow(pf_with_bustypes, sys2)
    reactive_power_tol = if exclude_reactive_flow
        nothing
    else
        POWERFLOW_COMPARISON_TOLERANCE
    end
    @test compare_df_within_tolerance("bus_results", result1["bus_results"],
        result2["bus_results"], POWERFLOW_COMPARISON_TOLERANCE;
        Q_gen = reactive_power_tol, Q_net = reactive_power_tol)
    @test compare_df_within_tolerance("flow_results",
        sort(result1["flow_results"], names(result1["flow_results"])[2:end]),
        sort(result2["flow_results"], names(result2["flow_results"])[2:end]),
        POWERFLOW_COMPARISON_TOLERANCE; line_name = nothing, Q_to_from = reactive_power_tol,
        Q_from_to = reactive_power_tol, Q_losses = reactive_power_tol)
end

function test_power_flow(
    pf::DCPowerFlow,
    sys1::System,
    sys2::System,
)
    pf_with_bustypes = DCPowerFlow(; correct_bustypes = true)
    result1 = solve_power_flow(pf_with_bustypes, sys1, PF.FlowReporting.ARC_FLOWS)
    result2 = solve_power_flow(pf_with_bustypes, sys2, PF.FlowReporting.ARC_FLOWS)
    @test compare_df_within_tolerance("bus_results", result1["1"]["bus_results"],
        result2["1"]["bus_results"], POWERFLOW_COMPARISON_TOLERANCE)
    @test compare_df_within_tolerance("flow_results",
        sort(result1["1"]["flow_results"], names(result1["1"]["flow_results"])[2:end]),
        sort(result2["1"]["flow_results"], names(result2["1"]["flow_results"])[2:end]),
        POWERFLOW_COMPARISON_TOLERANCE; line_name = nothing)
end

# Exercise PSCB's ability to parse a PSS/E System from a filename and a metadata dict.
# The `System(::AbstractString, ::Dict)` method is defined in PSCB's
# `parsers/psse_metadata_reimport.jl`.
function read_system_with_metadata(raw_path, metadata_path)
    # PSCB's re-import constructor requires a concrete Dict, while JSON3 can
    # return a JSON.Object when reading directly from a metadata file path.
    md = Dict(JSON3.read(metadata_path, Dict))
    sys = System(raw_path, md)
    return sys
end

# Exercise automatic export-path discovery while keeping metadata conversion in the helper
# above. PSCB's one-argument convenience method passes JSON3's JSON.Object directly to System.
function read_system_with_metadata(export_subdir)
    raw_path, metadata_path = get_psse_export_paths(export_subdir)
    return read_system_with_metadata(raw_path, metadata_path)
end

function test_psse_round_trip(
    pf::ACPowerFlow{<:ACPowerFlowSolverType},
    sys::System,
    exporter::PSSEExporter,
    scenario_name::AbstractString,
    export_location::AbstractString;
    do_power_flow_test = true,
    exclude_reactive_flow = false,
)
    raw_path, metadata_path =
        get_psse_export_paths(joinpath(export_location, scenario_name))

    write_export(exporter, scenario_name; overwrite = true)
    @test isfile(raw_path)
    @test isfile(metadata_path)

    sys2 = read_system_with_metadata(raw_path, metadata_path)
    @test compare_systems_loosely(sys, sys2)
    do_power_flow_test &&
        test_power_flow(pf, sys, sys2; exclude_reactive_flow = exclude_reactive_flow)
end

function test_psse_round_trip(
    pf::DCPowerFlow,
    sys::System,
    exporter::PSSEExporter,
    scenario_name::AbstractString,
    export_location::AbstractString;
    do_power_flow_test = true,
)
    raw_path, metadata_path =
        get_psse_export_paths(joinpath(export_location, scenario_name))

    write_export(exporter, scenario_name; overwrite = true)
    @test isfile(raw_path)
    @test isfile(metadata_path)

    sys2 = read_system_with_metadata(raw_path, metadata_path)
    @test compare_systems_loosely(sys, sys2)
    do_power_flow_test &&
        test_power_flow(pf, sys, sys2)
end

"Test that the two raw files are exactly identical and the two metadata files parse to identical JSON"
function test_psse_export_strict_equality(
    raw1,
    metadata1,
    raw2,
    metadata2;
    exclude_metadata_keys = ["case_name"],
    exclude_export_settings_keys = ["original_name"],
)
    parsed1 = JSON3.read(metadata1, Dict)
    parsed2 = JSON3.read(metadata2, Dict)
    case_name1 = parsed1["case_name"]
    case_name2 = parsed2["case_name"]

    open(raw1, "r") do handle1
        open(raw2, "r") do handle2
            # `countlines` reads a handle to EOF; comparing lengths from the same
            # `readlines` vectors (rather than `countlines` first) is what makes the
            # per-line comparison below see anything at all. The case-identification
            # record carries the export's own name and a `write_comments` header carries
            # a wall-clock timestamp; both legitimately differ between calls, mirroring
            # `exclude_metadata_keys` on the JSON side below.
            lines1 = readlines(handle1)
            lines2 = readlines(handle2)
            @test length(lines1) == length(lines2)
            timestamp_re = r"RAW via PowerFlows\.jl, \d{4}-\d{2}-\d{2}T"
            for (line1, line2) in zip(lines1, lines2)
                (line1 == case_name1 && line2 == case_name2) && continue
                (occursin(timestamp_re, line1) && occursin(timestamp_re, line2)) &&
                    continue
                @test line1 == line2
            end
        end
    end

    for key in exclude_metadata_keys
        parsed1[key] = nothing
        parsed2[key] = nothing
    end
    for key in exclude_export_settings_keys
        parsed1["export_settings"][key] = nothing
        parsed2["export_settings"][key] = nothing
    end
    @test parsed1 == parsed2
end

function load_test_system(sys_name::String)
    sys = with_logger(SimpleLogger(Error)) do
        build_system(PSSEParsingTestSystems, sys_name; force_build = true)
    end
    return sys
end

# I test so much, my tests have tests
@testset "Test system comparison utilities" begin
    sys = load_test_system("pti_case16_complete_sys")
    isnothing(sys) && return

    @test compare_systems_loosely(sys, sys)
    @test compare_systems_loosely(sys, deepcopy(sys))
end

function test_psse_exporter_version(sys_name::String, version::Symbol, folder_name::String)
    sys = load_test_system(sys_name)
    pf = DCPowerFlow()
    isnothing(sys) && return

    # PSS/E version must be one of the supported ones
    @test_throws ArgumentError PSSEExporter(sys, :vNonexistent, test_psse_export_dir)

    # Reimported export should be comparable to original system
    export_location = joinpath(test_psse_export_dir, string(version), folder_name)

    exporter = PSSEExporter(sys, version, export_location; write_comments = true)
    test_psse_round_trip(pf, sys, exporter, "basic", export_location)

    # Exporting the exact same thing again should result in the exact same files
    write_export(exporter, "basic2"; overwrite = true)
    test_psse_export_strict_equality(
        get_psse_export_paths(joinpath(export_location, "basic"))...,
        get_psse_export_paths(joinpath(export_location, "basic2"))...)
end

@testset "PSSE Exporter with case16_sys.raw, v33" begin
    test_psse_exporter_version("pti_case16_complete_sys", :v33, "case16_sys.raw")
end

@testset "PSSE Exporter with modified_case25_sys.raw, v35" begin
    test_psse_exporter_version("pti_modified_case25_v35_sys", :v35,
        "modified_case25_sys.raw")
end

@testset "PSSE Exporter round-trip preserves a VSC DC line, v33" begin
    # `pti_case16_complete_sys` contains a VSC DC line, so this round-trip explicitly exercises
    # VSC export → re-parse: `compare_systems_loosely` includes `PSY.TwoTerminalVSCLine`, so the
    # converter records (control TYPE/MODE, setpoints, losses, ratings) must survive the round-trip.
    sys = load_test_system("pti_case16_complete_sys")
    isnothing(sys) && return
    @test !isempty(PSY.get_components(PSY.TwoTerminalVSCLine, sys))

    export_location = joinpath(test_psse_export_dir, "v33", "case16_vsc_roundtrip")
    exporter = PSSEExporter(sys, :v33, export_location; write_comments = true)
    test_psse_round_trip(DCPowerFlow(), sys, exporter, "basic", export_location)

    # Focused check on the converter loss curve slope/intercept: BLOSS is normalized by
    # `rated_dc_voltage`, not `base_power`, so this pins the p.u.-per-p.u.-current inversion
    # `compare_systems_loosely` only exercises indirectly.
    sys2 = read_system_with_metadata(joinpath(export_location, "basic"))
    for vsc1 in PSY.get_components(PSY.TwoTerminalVSCLine, sys)
        vsc2 = PSY.get_component(PSY.TwoTerminalVSCLine, sys2, PSY.get_name(vsc1))
        for get_loss in (PSY.get_converter_loss_from, PSY.get_converter_loss_to)
            fd1 = PSY.get_function_data(get_loss(vsc1))
            fd2 = PSY.get_function_data(get_loss(vsc2))
            @test isapprox(
                PSY.get_proportional_term(fd1), PSY.get_proportional_term(fd2);
                rtol = 1e-9)
            @test isapprox(
                PSY.get_constant_term(fd1), PSY.get_constant_term(fd2);
                rtol = 1e-9)
        end
    end
end

# @testset "Parsed VSC lowers to p.u.-sane setpoints and the AC power flow solves (v33)" begin
#     # Regression: the parser stored DCSET raw (kV/MW), so lowered vdc_set was ~100s of "p.u." and
#     # the joint NR diverged on any parsed VSC line.
#     sys = load_test_system("pti_case16_complete_sys")
#     isnothing(sys) && return
#     # The fixture's MODE=1 records are ill-posed independent of this test: bus 103 is PV
#     # (rejected) and bus 501's ACSET is unreachable within its Q limits (PSS/E itself backed off
#     # to a Q limit). Use fixed-Q control within limits.
#     for vsc in PSY.get_components(PSY.TwoTerminalVSCLine, sys)
#         PSY.set_ac_control_from!(vsc, PSY.VSCACControlModes.AC_REACTIVE_POWER)
#         PSY.set_reactive_power_from!(vsc, -0.45 * PSY.SU)
#         PSY.set_ac_control_to!(vsc, PSY.VSCACControlModes.AC_REACTIVE_POWER)
#         PSY.set_reactive_power_to!(vsc, -0.4 * PSY.SU)
#     end
#     data = PowerFlowData(ACPowerFlow{NewtonRaphsonACPowerFlow}(), sys)
#     dcn = PF.get_dc_network(data)
#     @test PF.n_vsc_converters(dcn) > 0
#     @test all(0.5 .<= dcn.vdc_set .<= 1.5)
#     @test all(abs.(dcn.p_set) .< 10.0)
#     # the MW order must survive at full magnitude (guards against re-scaling, e.g. a double
#     # baseMVA division: 0.96 → 0.0096 would still pass the sanity bounds above)
#     @test maximum(abs, dcn.p_set) > 0.1
#     @test solve_power_flow!(data)
# end

@testset "PSSE Exporter: a VSC built without PSS/E ext metadata re-parses (v33)" begin
    # Regression: a VSC with no `ext` REMOT/RMPCT must still export valid numeric fields. Previously
    # REMOT defaulted to an empty field, so the exported .raw failed to re-parse (empty-Int error).
    sys = load_test_system("pti_case16_complete_sys")
    isnothing(sys) && return
    n0 = length(collect(PSY.get_components(PSY.TwoTerminalVSCLine, sys)))
    buses = sort!(collect(PSY.get_components(PSY.ACBus, sys)); by = PSY.get_number)
    arc = _get_or_make_arc(sys, buses[3], buses[4])
    PSY.add_component!(
        sys,
        PSY.TwoTerminalVSCLine(;
            name = "vsc_no_ext",
            available = true,
            arc = arc,
            active_power_flow = 0.2,
            rating = 1.5,
            active_power_limits_from = (min = -1.5, max = 1.5),
            active_power_limits_to = (min = -1.5, max = 1.5),
            g = 40.0,
            dc_control_from = PSY.VSCDCControlModes.DC_VOLTAGE,
            dc_setpoint_from = 1.0,
            dc_control_to = PSY.VSCDCControlModes.DC_POWER,
            dc_setpoint_to = 0.2,
        ),
    )
    export_location = joinpath(test_psse_export_dir, "v33", "case16_vsc_no_ext")
    exporter = PSSEExporter(sys, :v33, export_location; write_comments = true)
    write_export(exporter, "basic"; overwrite = true)
    raw_path, metadata_path = get_psse_export_paths(joinpath(export_location, "basic"))
    # the re-parse must succeed (previously threw on the empty REMOT field) and keep both VSCs
    sys2 = read_system_with_metadata(raw_path, metadata_path)
    @test length(collect(PSY.get_components(PSY.TwoTerminalVSCLine, sys2))) == n0 + 1
end

@testset "PSSE Exporter FACTS: RMPCT blank, FCREG/REMOT from regulated_bus_number (v33/v35)" begin
    # `reactive_power_required` is a solver OUTPUT, not the PSS/E RMPCT input, and PSY models no
    # RMPCT, so the field is written blank — the exporter must not fall back to `ext`. FCREG/REMOT
    # come from the first-class `regulated_bus_number` field.
    sys = System(100.0)
    b1 = _add_simple_bus!(sys, 1, ACBusTypes.REF, 230, 1.0, 0.0)
    b7 = _add_simple_bus!(sys, 7, ACBusTypes.PQ, 230, 1.0, 0.0)
    _add_simple_source!(sys, b1, 0.0, 0.0)
    _add_simple_line!(sys, b1, b7, 0.01, 0.10, 0.0)
    facts = PSY.FACTSControlDevice(;
        name = "facts_1",
        available = true,
        bus = b1,
        control_mode = PSY.FACTSOperationModes.NML,
        voltage_setpoint = 1.0,
        regulated_bus_number = 7,
        reactive_power_required = 42.0,  # solved output; must NOT be written as RMPCT
        ext = Dict{String, Any}("RMPCT" => 55.0),  # stale ext; the exporter must ignore it
    )
    # `max_shunt_current` is stored in device base; the constructor kwarg takes a raw CU
    # value, so set it through the units-aware setter to honor the MVA input.
    PSY.set_max_shunt_current!(facts, 100.0 * u"MVA")
    PSY.add_component!(sys, facts)

    export_location = joinpath(test_psse_export_dir, "v33", "facts_rmpct_fcreg")
    exporter = PSSEExporter(sys, :v33, export_location; write_comments = true)
    write_export(exporter, "basic"; overwrite = true)
    raw_path, metadata_path = get_psse_export_paths(joinpath(export_location, "basic"))

    raw_lines = readlines(raw_path)
    facts_line_idx = findfirst(l -> occursin("'facts_1'", l), raw_lines)
    @test !isnothing(facts_line_idx)
    isnothing(facts_line_idx) && return
    facts_line = raw_lines[facts_line_idx]
    fields = strip.(split(facts_line, ","))
    # v33 record: NAME, I, J, MODE, PDES, QDES, VSET, SHMX, TRMX, VTMN, VTMX, VSMX, IMX, LINX,
    # RMPCT, OWNER, SET1, SET2, VSREF, REMOT
    @test isempty(fields[15])
    @test fields[20] == "7"
    @test !occursin("42.0", facts_line)
    @test !occursin("55.0", facts_line)

    sys2 = read_system_with_metadata(raw_path, metadata_path)
    facts2 = only(collect(PSY.get_components(PSY.FACTSControlDevice, sys2)))
    @test PSY.get_regulated_bus_number(facts2) == 7
end

@testset "PSSE Exporter: switched shunt control_mode/regulated_bus_number round-trip (v33)" begin
    # MODSW/SWREM must survive export + reimport so parsed switched shunts keep
    # regulating instead of silently defaulting to FIXED (control_mode = 0).
    sys = System(100.0)
    b1 = _add_simple_bus!(sys, 1, ACBusTypes.REF, 230, 1.0, 0.0)
    b7 = _add_simple_bus!(sys, 7, ACBusTypes.PQ, 230, 1.0, 0.0)
    _add_simple_source!(sys, b1, 0.0, 0.0)
    _add_simple_line!(sys, b1, b7, 0.01, 0.10, 0.0)
    shunt = PSY.SwitchedAdmittance(;
        name = "shunt_1",
        available = true,
        bus = b1,
        number_engaged = [1],
        number_of_steps = [4],
        Y_increase = [0.0 + 0.05im],
        admittance_limits = (min = 0.9, max = 1.1),
        control_mode = PSY.SwitchedAdmittanceControlMode.DISCRETE_VOLTAGE,
        regulated_bus_number = 7,
    )
    PSY.add_component!(sys, shunt)

    export_location = joinpath(test_psse_export_dir, "v33", "switched_shunt_modsw_swrem")
    exporter = PSSEExporter(sys, :v33, export_location; write_comments = true)
    write_export(exporter, "basic"; overwrite = true)
    raw_path, metadata_path = get_psse_export_paths(joinpath(export_location, "basic"))

    sys2 = read_system_with_metadata(raw_path, metadata_path)
    shunt2 = only(collect(PSY.get_components(PSY.SwitchedAdmittance, sys2)))
    @test PSY.get_control_mode(shunt2) == PSY.SwitchedAdmittanceControlMode.DISCRETE_VOLTAGE
    @test PSY.get_regulated_bus_number(shunt2) == 7
end

@testset "PSSE Exporter: v35 switched shunt Si is a 0/1 block status" begin
    sys = System(100.0)
    b1 = _add_simple_bus!(sys, 1, ACBusTypes.REF, 230, 1.0, 0.0)
    shunt = PSY.SwitchedAdmittance(;
        name = "shunt_1",
        available = true,
        bus = b1,
        number_engaged = [3, 1, 0],
        number_of_steps = [3, 2, 4],
        Y_increase = [0.05im, 0.02im, 0.01im],
    )
    S, N, _ = PF._build_switched_shunt_steps_v35(
        shunt,
        PSY.get_number_of_steps(shunt),
        PSY.get_Y_increase(shunt),
        100.0,
    )
    @test S[1:3] == [1, 1, 0]
    @test N[1:3] == [3, 2, 4]
end

@testset "PSSE Exporter: phase-shift control_limits round-trip degrees/radians (v33)" begin
    # PSS/E RMA/RMI are degrees for phase-shift CODs (ACTIVE_POWER_FLOW here); PSY stores
    # `control_limits` in radians for those objectives. The exporter must write degrees, and
    # the raw file must NOT contain the stored radian values.
    sys = System(100.0)
    b1 = _add_simple_bus!(sys, 1, ACBusTypes.REF, 230, 1.0, 0.0)
    b2 = _add_simple_bus!(sys, 2, ACBusTypes.PQ, 230, 1.0, 0.0)
    _add_simple_source!(sys, b1, 0.0, 0.0)
    arc = Arc(; from = b1, to = b2)
    tx = TwoWindingTransformer(;
        name = "ps_tx1",
        circuit = TransformerCircuit(;
            available = true,
            arc = arc,
            r = 0.01,
            x = 0.10,
            tap = 1.0,
            rating = 1.0,
            base_power = 100.0,
            control_objective = PSY.TransformerControlObjective.ACTIVE_POWER_FLOW,
            control_limits = (min = deg2rad(-30), max = deg2rad(30)),
        ),
    )
    PSY.add_component!(sys, tx)

    export_location =
        joinpath(test_psse_export_dir, "v33", "transformer_phase_shift_limits")
    exporter = PSSEExporter(sys, :v33, export_location; write_comments = true)
    write_export(exporter, "basic"; overwrite = true)
    raw_path, metadata_path = get_psse_export_paths(joinpath(export_location, "basic"))

    raw_lines = readlines(raw_path)
    name_line_idx = findfirst(l -> occursin("'ps_tx1'", l), raw_lines)
    @test !isnothing(name_line_idx)
    isnothing(name_line_idx) && return
    # record1 (name/status), record2 (impedance), record3 (winding1: RMA/RMI at fields 9/10)
    record3 = raw_lines[name_line_idx + 2]
    fields = strip.(split(record3, ","))
    @test isapprox(parse(Float64, fields[9]), 30.0; atol = 1e-8)
    @test isapprox(parse(Float64, fields[10]), -30.0; atol = 1e-8)
    @test !occursin(string(deg2rad(30)), record3)

    sys2 = read_system_with_metadata(raw_path, metadata_path)
    tx2 = only(collect(PSY.get_components(PSY.TwoWindingTransformer, sys2)))
    control_limits2 = PSY.get_control_limits(PSY.get_circuit(tx2))
    @test isapprox(control_limits2.min, deg2rad(-30); atol = 1e-8)
    @test isapprox(control_limits2.max, deg2rad(30); atol = 1e-8)
end

@testset "PSSE Exporter RTS regression: non-unity tap circuit and v35 default ratings" begin
    sys = with_logger(SimpleLogger(Error)) do
        build_system(PSISystems, "modified_RTS_GMLC_DA_sys"; force_build = true)
    end
    isnothing(sys) && return

    two_winding_transformers = collect(PSY.get_components(PSY.TwoWindingTransformer, sys))
    target_tap_idx = findfirst(
        t -> !isapprox(PSY.get_tap(PSY.get_circuit(t)), 1.0),
        two_winding_transformers,
    )
    @test !isnothing(target_tap_idx)
    isnothing(target_tap_idx) && return
    target_tap = two_winding_transformers[target_tap_idx]

    lines = sort!(collect(PSY.get_components(PSY.Line, sys)); by = PSY.get_name)
    @test !isempty(lines)
    isempty(lines) && return
    target_line = first(lines)

    # In v35, unspecified extra rating fields are exported as explicit 0.0 values.
    export_location = joinpath(test_psse_export_dir, "v35", "rts_targeted_regressions")
    scenario_name = "nonunity_tap_circuit_and_v35_missing_ratings"
    exporter =
        PSSEExporter(sys, :v35, export_location; write_comments = true, overwrite = true)
    write_export(exporter, scenario_name; overwrite = true)

    raw_path, metadata_path =
        get_psse_export_paths(joinpath(export_location, scenario_name))
    @test isfile(raw_path)
    @test isfile(metadata_path)

    md = JSON3.read(metadata_path, Dict)
    transformer_ckt_mapping = md["transformer_ckt_mapping"]
    branch_name_mapping = md["branch_name_mapping"]

    tap_name = PSY.get_name(target_tap)
    two_winding_transformer_keys =
        filter(k -> endswith(k, "_" * tap_name), collect(keys(transformer_ckt_mapping)))
    tap_branch_keys =
        filter(k -> endswith(k, "_" * tap_name), collect(keys(branch_name_mapping)))
    @test length(two_winding_transformer_keys) == 1
    @test isempty(tap_branch_keys)

    line_name = PSY.get_name(target_line)
    line_keys =
        filter(k -> endswith(k, "_" * line_name), collect(keys(branch_name_mapping)))
    @test length(line_keys) == 1
    isempty(line_keys) && return

    bus_number_mapping = md["bus_number_mapping"]
    raw_lines = readlines(raw_path)

    line_key = line_keys[1]
    line_bus_pair = split(line_key, "_"; limit = 2)[1]
    line_from_orig, line_to_orig = split(line_bus_pair, "-")
    line_from = bus_number_mapping[line_from_orig]
    line_to = bus_number_mapping[line_to_orig]
    line_ckt = branch_name_mapping[line_key]
    line_record_idx = findfirst(
        l -> occursin("$line_from, $line_to, '$line_ckt'", l),
        raw_lines,
    )
    @test !isnothing(line_record_idx)
    isnothing(line_record_idx) && return
    @test occursin(", 0.0, 0.0, 0.0,", raw_lines[line_record_idx])

    tap_key = two_winding_transformer_keys[1]
    tap_bus_pair = split(tap_key, "_"; limit = 2)[1]
    tap_from_orig, tap_to_orig = split(tap_bus_pair, "-")
    tap_from = bus_number_mapping[tap_from_orig]
    tap_to = bus_number_mapping[tap_to_orig]
    tap_ckt = transformer_ckt_mapping[tap_key]
    tap_record1_idx = findfirst(
        l -> occursin("$tap_from, $tap_to, 0, '$tap_ckt'", l),
        raw_lines,
    )
    @test !isnothing(tap_record1_idx)
    isnothing(tap_record1_idx) && return
    @test tap_record1_idx + 2 <= length(raw_lines)
    tap_winding1_record = raw_lines[tap_record1_idx + 2]
    @test occursin(", 0.0, 0.0, 0.0,", tap_winding1_record)
end

# Regression for issue #361: a programmatically-built Line carries no RATE4..RATE12 data,
# which used to trigger a `MethodError: Cannot convert String to Float64` in the v35
# non-transformer branch writer. The absent extra ratings must export as numeric 0.0.
@testset "PSSE Exporter issue #361: v35 Line with no RATE4..RATE12 data" begin
    sys = System(100.0)
    b1 = ACBus(; number = 1, name = "b1", available = true, bustype = ACBusTypes.REF,
        angle = 0.0, magnitude = 1.0, voltage_limits = (0.0, 2.0), base_voltage = 138.0,
    )
    b2 = ACBus(; number = 2, name = "b2", available = true, bustype = ACBusTypes.PV,
        angle = 0.0, magnitude = 1.0, voltage_limits = (0.0, 2.0), base_voltage = 138.0,
    )
    add_component!(sys, b1)
    add_component!(sys, b2)
    line = Line(; name = "L", available = true, active_power_flow = 0.0,
        reactive_power_flow = 0.0, arc = Arc(; from = b1, to = b2), r = 0.01, x = 0.1,
        b = (from = 0.0, to = 0.0), rating = 1.0,
        angle_limits = (min = -pi / 2, max = pi / 2))
    add_component!(sys, line)

    export_location = joinpath(test_psse_export_dir, "v35", "issue361_missing_rate_keys")
    exporter = PSSEExporter(sys, :v35, export_location; overwrite = true)
    # The export must not throw (the regression was a MethodError during writing).
    write_export(exporter, "missing_rate_keys"; overwrite = true)

    raw_path, metadata_path =
        get_psse_export_paths(joinpath(export_location, "missing_rate_keys"))
    @test isfile(raw_path)
    @test isfile(metadata_path)

    md = JSON3.read(metadata_path, Dict)
    branch_name_mapping = md["branch_name_mapping"]
    bus_number_mapping = md["bus_number_mapping"]
    I = bus_number_mapping["1"]
    J = bus_number_mapping["2"]
    raw_lines = readlines(raw_path)
    line_record_idx = findfirst(l -> startswith(strip(l), "$I, $J,"), raw_lines)
    @test !isnothing(line_record_idx)
    isnothing(line_record_idx) && return
    # All twelve rating fields (RATEA..RATE12) present and the missing ones written as 0.0.
    @test occursin(
        "0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0",
        raw_lines[line_record_idx],
    )
end

# Regression for issue #361 (related): a system with no non-transformer branches yields an
# empty branch name-mapping whose key type inferred as `Tuple{Tuple{Int,Int,Vararg{Int}},
# String}`, which matched no `serialize_component_ids` method. The export must succeed and
# produce an empty branch mapping.
@testset "PSSE Exporter issue #361: v35 system with no non-transformer branches" begin
    sys = System(100.0)
    b1 = ACBus(; number = 1, name = "b1", available = true, bustype = ACBusTypes.REF,
        angle = 0.0, magnitude = 1.0, voltage_limits = (0.0, 2.0), base_voltage = 138.0,
    )
    add_component!(sys, b1)
    @test isempty(PSY.get_components(PSY.ACBranch, sys))

    export_location = joinpath(test_psse_export_dir, "v35", "issue361_no_branches")
    exporter = PSSEExporter(sys, :v35, export_location; overwrite = true)
    # The export must not throw (the regression was a serialize_component_ids MethodError).
    write_export(exporter, "no_branches"; overwrite = true)

    raw_path, metadata_path =
        get_psse_export_paths(joinpath(export_location, "no_branches"))
    @test isfile(raw_path)
    @test isfile(metadata_path)

    md = JSON3.read(metadata_path, Dict)
    @test isempty(md["branch_name_mapping"])
end

function test_psse_exporter_inner(
    ACSolver::Type{<:ACPowerFlowSolverType},
    folder_name::String,
)
    sys = load_test_system("pti_case24_sys")
    pf = ACPowerFlow{ACSolver}()
    isnothing(sys) && return

    # PSS/E version must be one of the supported ones
    @test_throws ArgumentError PSSEExporter(sys, :vNonexistent, test_psse_export_dir)

    # Reimported export should be comparable to original system
    export_location = joinpath(test_psse_export_dir, "v33", folder_name)
    exporter = PSSEExporter(sys, :v33, export_location)
    test_psse_round_trip(pf, sys, exporter, "basic", export_location;
        exclude_reactive_flow = true)

    # Exporting the exact same thing again should result in the exact same files
    write_export(exporter, "basic2"; overwrite = true)
    test_psse_export_strict_equality(
        get_psse_export_paths(joinpath(export_location, "basic"))...,
        get_psse_export_paths(joinpath(export_location, "basic2"))...)

    # Updating with a completely different system should fail
    different_system = load_test_system("pti_case5_alc_sys")
    @test_throws ArgumentError update_exporter!(exporter, different_system)

    # Updating with the exact same system should result in the exact same files
    update_exporter!(exporter, sys)
    write_export(exporter, "basic3"; overwrite = true)
    test_psse_export_strict_equality(
        get_psse_export_paths(joinpath(export_location, "basic"))...,
        get_psse_export_paths(joinpath(export_location, "basic3"))...)

    # Updating with changed value should result in a different reimport (System version)
    sys2 = deepcopy(sys)
    line_to_change = first(get_components(Line, sys2))
    set_rating!(line_to_change, get_rating(line_to_change, PSY.SU) * 123.4 * PSY.SU)  # careful not to exceed PF.INFINITE_BOUND
    update_exporter!(exporter, sys2)
    write_export(exporter, "basic4"; overwrite = true)
    reread_sys2 = read_system_with_metadata(joinpath(export_location, "basic4"))
    @test_logs((:error, r"values do not match"),
        match_mode = :any, min_level = Logging.Error,
        compare_systems_loosely(sys, reread_sys2))
    @test compare_systems_loosely(sys2, reread_sys2)
    test_power_flow(pf, sys2, reread_sys2; exclude_reactive_flow = true)
end

@testset "PSSE Exporter with case24_sys.raw, v33 - NewtonRaphsonACPowerFlow" begin
    test_psse_exporter_inner(NewtonRaphsonACPowerFlow, "case24_sys_NR")
end

@testset "update_exporter!(::PowerFlowData) writes solved discrete-control settings" begin
    # A controlled solve moves the tap; update_exporter! must write the solved tap into
    # the exporter's (deepcopied) system without touching the caller's system.
    sys = _make_solvable_tap_shunt_system()
    tx = first(PSY.get_components(PSY.TwoWindingTransformer, sys))
    tap_before = PSY.get_tap(PSY.get_circuit(tx))
    pf = ACPolarPowerFlow{NewtonRaphsonACPowerFlow}(; control_discrete_devices = true)
    data = PowerFlowData(pf, sys)
    @test PowerFlows.solve_power_flow!(data)
    t = data.controlled_devices.taps[1]
    @test t.current != tap_before

    export_location = joinpath(test_psse_export_dir, "v33", "controlled_tap_update")
    exporter = PSSEExporter(sys, :v33, export_location)
    update_exporter!(exporter, data)
    tx_exported = PSY.get_component(PSY.TwoWindingTransformer, exporter.system, t.name)
    @test PSY.get_tap(PSY.get_circuit(tx_exported)) == t.current
    @test PSY.get_tap(PSY.get_circuit(tx_exported)) != tap_before

    tx_user = PSY.get_component(PSY.TwoWindingTransformer, sys, t.name)
    @test PSY.get_tap(PSY.get_circuit(tx_user)) == tap_before
end

@testset "Test exporter helper functions" begin
    @test PF._psse_bus_numbers([2, 3, 999_997, 999_998, 1_000_001, 1]) ==
          Dict(
        2 => 2,
        3 => 3,
        999_997 => 999_997,
        999_998 => 899_998,
        1_000_001 => 4,
        1 => 1,
    )
    @test !PF._is_valid_psse_name("a pretty long name")
    @test !PF._is_valid_psse_name("-bad")
    @test PF._is_valid_psse_name(raw"¯\_(ツ)_/¯")
    @test PF._psse_bus_names(["-bad1", "compliant", "BUS_100", "-bad2", "ok just too long"],
        [10, 2, 3, 4, 5], Dict(10 => 100, 2 => 20, 3 => 30, 4 => 40, 5 => 50)) ==
          Dict("-bad1" => "BUS_100-", "compliant" => "compliant", "BUS_100" => "BUS_100",
        "-bad2" => "BUS_40", "ok just too long" => "ok just too ")
    @test PF.create_component_ids(
        ["generator-1234-AB", "123_CT_7", "load1234", "load1334"], [1, 1, 2, 2]) ==
          Dict((1, "generator-1234-AB") => "AB", (1, "123_CT_7") => "7",
        (2, "load1234") => "34", (2, "load1334") => "35")

    @test PowerFlows._map_psse_container_names(["1", "3", "2"]) ==
          OrderedDict("1" => 1, "3" => 3, "2" => 2)
    @test PowerFlows._map_psse_container_names(["1", "a", "2"]) ==
          OrderedDict("1" => 1, "a" => 2, "2" => 3)
    @test PowerFlows._map_psse_container_names(["2.0", "1.0"]) ==
          OrderedDict("2.0" => 2, "1.0" => 1)
end

@testset "PSSE export: non-ImpedanceCorrectionData supplemental attribute on transformer (issue #418)" begin
    sys = load_test_system("pti_case16_complete_sys")

    transformers = collect(PSY.get_components(PSY.TwoWindingTransformer, sys))
    @test !isempty(transformers)
    isempty(transformers) && return
    transformer = first(transformers)

    ffo = PSY.FixedForcedOutage(; outage_status = 1.0)
    @test_throws MethodError PSY.get_table_number(ffo)

    # Attach the non-impedance supplemental attribute and verify PSSE export succeeds.
    PSY.add_supplemental_attribute!(sys, transformer, ffo)
    @test !isempty(
        PSY.get_supplemental_attributes(PSY.FixedForcedOutage, transformer),
    )

    export_location = joinpath(test_psse_export_dir, "v33", "export_non_icd_supp_attr")
    exporter = PSSEExporter(sys, :v33, export_location; overwrite = true)
    # Must not throw MethodError: no method matching get_table_number(::FixedForcedOutage)
    @test (write_export(exporter, "regression"; overwrite = true); true)

    raw_path, md_path = get_psse_export_paths(joinpath(export_location, "regression"))
    @test isfile(raw_path)
    @test isfile(md_path)
end

# # TODO add tests for unit system agnosticism

@testset "Solve parameters reach the exported v35 solution records" begin
    sys = _make_solvable_tap_shunt_system()
    pf = ACPolarPowerFlow{NewtonRaphsonACPowerFlow}(;
        solution_parameters = SolutionParameters(;
            tol = 1e-6,
            maxIterations = 30,
            check_reactive_power_limits = true,
            control_discrete_devices = true,
        ),
    )
    data = PowerFlowData(pf, sys)
    @test PowerFlows.solve_power_flow!(data)

    export_location = joinpath(test_psse_export_dir, "v35", "solution_records")
    exporter = PSSEExporter(sys, :v35, export_location)

    # With no solve attached the block is written at the format defaults, so an export
    # that never saw a solve is unchanged.
    write_export(exporter, "before"; overwrite = true)
    raw_before, _ = get_psse_export_paths(joinpath(export_location, "before"))
    @test occursin("ACTAPS=0", read(raw_before, String))

    # `update_exporter!(::PowerFlowData)` is what marks a solve as having happened.
    update_exporter!(exporter, data)
    write_export(exporter, "after"; overwrite = true)
    raw_after, _ = get_psse_export_paths(joinpath(export_location, "after"))
    text = read(raw_after, String)
    @test occursin("ACTAPS=1", text)
    @test occursin("SWSHNT=1", text)
    @test occursin("VARLIM=0", text)
    @test occursin("ITMXN=30", text)
    @test occursin("FNSL", text)

    recovered = read_solution_parameters(raw_after)
    @test !isnothing(recovered)
    @test recovered.maxIterations == 30
    @test recovered.control_discrete_devices
    @test recovered.check_reactive_power_limits
    @test recovered.tol ≈ 1e-6

    write_export(exporter, "after2"; overwrite = true)
    test_psse_export_strict_equality(
        get_psse_export_paths(joinpath(export_location, "after"))...,
        get_psse_export_paths(joinpath(export_location, "after2"))...)
end

@testset "A v33 export warns that solve parameters are dropped" begin
    sys = _make_solvable_tap_shunt_system()
    pf = ACPolarPowerFlow{NewtonRaphsonACPowerFlow}(;
        solution_parameters = SolutionParameters(; control_discrete_devices = true),
    )
    data = PowerFlowData(pf, sys)
    @test PowerFlows.solve_power_flow!(data)

    export_location = joinpath(test_psse_export_dir, "v33", "solution_records_dropped")
    exporter = PSSEExporter(sys, :v33, export_location)
    # The v33 raw format has no system-wide solution data section at all.
    @test_logs (:warn, r"v33") match_mode = :any update_exporter!(exporter, data)
end

@testset "Call-site solve parameters can be attached explicitly" begin
    sys = _make_solvable_tap_shunt_system()
    pf = ACPolarPowerFlow{NewtonRaphsonACPowerFlow}()
    export_location = joinpath(test_psse_export_dir, "v35", "solution_records_kwargs")
    exporter = PSSEExporter(sys, :v35, export_location)

    # `tol` passed to the solve call is never stored on the model, so only the caller can
    # report it.
    update_exporter!(exporter, pf; solver_kwargs = (; tol = 1e-5, maxIterations = 17))
    write_export(exporter, "kwargs"; overwrite = true)
    raw, _ = get_psse_export_paths(joinpath(export_location, "kwargs"))
    text = read(raw, String)
    @test occursin("ITMXN=17", text)
    @test occursin("TOLN=0.001", text)
end

@testset "PSSE Exporter: LCC SETVL is written in stored MW, not scaled by SBASE" begin
    sys = System(100.0)
    b1 = _add_simple_bus!(sys, 1, ACBusTypes.REF, 230.0)
    b2 = _add_simple_bus!(sys, 2, ACBusTypes.PQ, 230.0)
    _add_simple_source!(sys, b1, 0.0, 0.0)
    _add_simple_load!(sys, b2, 0.1, 0.05)
    lcc = _add_simple_lcc!(sys, b1, b2, 0.01, 0.01, 0.01)  # transfer_setpoint = 50 (MW)

    export_location = joinpath(test_psse_export_dir, "v35", "lcc_setvl")
    exporter = PSSEExporter(sys, :v35, export_location; overwrite = true)
    write_export(exporter, "lcc_setvl"; overwrite = true)
    raw_path, _ = get_psse_export_paths(joinpath(export_location, "lcc_setvl"))
    lcc_line = only(filter(l -> occursin("LCC", l), readlines(raw_path)))
    fields = PF._split_record(lcc_line)
    @test parse(Float64, strip(fields[4])) ≈ 50.0  # SETVL, not 50 * SBASE = 5000

    sys2 = read_system_with_metadata(joinpath(export_location, "lcc_setvl"))
    lcc2 = only(PSY.get_components(PSY.TwoTerminalLCCLine, sys2))
    @test PSY.get_transfer_setpoint(lcc2) ≈ PSY.get_transfer_setpoint(lcc)
end

@testset "PSSE Exporter: 3W transformer built from star circuits writes no `nothing` field" begin
    sys = System(100.0)
    b1 = _add_simple_bus!(sys, 1, ACBusTypes.REF, 230.0)
    b2 = _add_simple_bus!(sys, 2, ACBusTypes.PQ, 230.0)
    b3 = _add_simple_bus!(sys, 3, ACBusTypes.PQ, 138.0)
    _add_simple_source!(sys, b1, 0.0, 0.0)
    _add_simple_load!(sys, b2, 0.1, 0.05)
    _add_simple_load!(sys, b3, 0.05, 0.02)
    xfmr = _add_simple_transformer_3w!(sys, b1, b2, b3, 90)
    # The pairwise fields are `nothing` when a 3W is built directly from star-leg circuits.
    @test isnothing(PSY.get_r_12(xfmr, PSY.SU))

    export_location = joinpath(test_psse_export_dir, "v35", "xfmr3w_no_pairwise")
    exporter = PSSEExporter(sys, :v35, export_location; overwrite = true)
    write_export(exporter, "xfmr3w"; overwrite = true)
    raw_path, _ = get_psse_export_paths(joinpath(export_location, "xfmr3w"))
    @test !occursin("nothing", read(raw_path, String))
end

@testset "PSSE Exporter: generic HVDC synthetic generators follow PF's own injection sign" begin
    sys = System(100.0)
    b1 = _add_simple_bus!(sys, 1, ACBusTypes.REF, 230.0)
    b2 = _add_simple_bus!(sys, 2, ACBusTypes.PQ, 230.0)
    _add_simple_source!(sys, b1, 0.0, 0.0)
    _add_simple_load!(sys, b2, 0.0, 0.0)
    hvdc = TwoTerminalGenericHVDCLine(;
        name = "hvdc_1_2",
        available = true,
        active_power_flow = 30.0,
        arc = Arc(b1, b2),
        active_power_limits_from = (min = -100.0, max = 100.0),
        active_power_limits_to = (min = -100.0, max = 100.0),
        reactive_power_limits_from = (min = 0.0, max = 0.0),
        reactive_power_limits_to = (min = 0.0, max = 0.0),
        base_power = 100.0,
    )
    add_component!(sys, hvdc)

    # PF's own convention: FROM withdraws from the AC network, TO supplies it.
    (P_from, P_to) = PF.get_hvdc_injections(hvdc, sys)
    @test P_from < 0.0 && P_to > 0.0

    export_location = joinpath(test_psse_export_dir, "v35", "generic_hvdc_sign")
    exporter = PSSEExporter(sys, :v35, export_location; overwrite = true)
    write_export(exporter, "generic_hvdc_sign"; overwrite = true)
    sys2 = read_system_with_metadata(joinpath(export_location, "generic_hvdc_sign"))

    gen_from =
        only(
            PSY.get_components(
                g -> endswith(PSY.get_name(g), "_FR"),
                ThermalStandard,
                sys2,
            ),
        )
    gen_to =
        only(
            PSY.get_components(
                g -> endswith(PSY.get_name(g), "_TO"),
                ThermalStandard,
                sys2,
            ),
        )
    @test PSY.get_active_power(gen_from, PSY.SU) ≈ P_from
    @test PSY.get_active_power(gen_to, PSY.SU) ≈ P_to
end

@testset "PSSE Exporter: droop VSC DCSET follows the AC-supply sign convention" begin
    sys = System(100.0)
    b1 = _add_simple_bus!(sys, 1, ACBusTypes.REF, 230.0)
    b2 = _add_simple_bus!(sys, 2, ACBusTypes.PQ, 230.0)
    _add_simple_source!(sys, b1, 0.0, 0.0)
    _add_simple_load!(sys, b2, 0.0, 0.0)
    vsc = _add_simple_vsc!(sys, b1, b2; active_power_flow = 0.2)
    # One converter must keep a real DC-voltage reference or the record has no TYPE-1
    # terminal and re-parsing rejects it; only the `to` side is droop, the case under test.
    PSY.set_dc_control_from!(vsc, PSY.VSCDCControlModes.DC_VOLTAGE)
    PSY.set_dc_control_to!(vsc, PSY.VSCDCControlModes.DC_VOLTAGE_DROOP)

    export_location = joinpath(test_psse_export_dir, "v35", "vsc_droop_dcset")
    exporter = PSSEExporter(sys, :v35, export_location; overwrite = true)
    write_export(exporter, "vsc_droop"; overwrite = true)
    sys2 = read_system_with_metadata(joinpath(export_location, "vsc_droop"))
    vsc2 = only(PSY.get_components(PSY.TwoTerminalVSCLine, sys2))
    # The format has no droop mode, so the `to` side reimports as DC_POWER, whose own
    # convention is positive == supplies the AC network at that bus. Flow is FROM -> TO
    # (positive), so `to` supplies its AC network: the setpoint must be positive.
    @test PSY.get_dc_control_to(vsc2) == PSY.VSCDCControlModes.DC_POWER
    @test PSY.get_dc_setpoint_to(vsc2) > 0.0
end

@testset "PSSE Exporter: switching-device RATE1 round-trips through SBASE" begin
    # PFFP's switch/breaker importer stores RATE1 unscaled, so a 12.06 CU rating must
    # export as 12.06, not 1206.0, to round-trip.
    sys = System(100.0)
    b1 = _add_simple_bus!(sys, 1, ACBusTypes.REF, 230.0)
    b2 = _add_simple_bus!(sys, 2, ACBusTypes.PQ, 230.0)
    _add_simple_source!(sys, b1, 0.0, 0.0)
    _add_simple_load!(sys, b2, 0.1, 0.05)
    arc = Arc(; from = b1, to = b2)
    add_component!(sys, arc)
    sw = DiscreteControlledACBranch(;
        name = "brk_1_2",
        available = true,
        active_power_flow = 0.0,
        reactive_power_flow = 0.0,
        arc = arc,
        r = 0.001,
        x = 0.01,
        rating = 12.06,  # 1206 MVA on a 100 MVA system base
        discrete_branch_type = DiscreteControlledBranchType.BREAKER,
        branch_status = DiscreteControlledBranchStatus.CLOSED,
    )
    add_component!(sys, sw)
    @test PSY.get_rating(sw, PSY.NU) ≈ 1206.0

    export_location = joinpath(test_psse_export_dir, "v35", "breaker_rate1")
    exporter = PSSEExporter(sys, :v35, export_location; overwrite = true)
    write_export(exporter, "breaker_rate1"; overwrite = true)
    sys2 = read_system_with_metadata(joinpath(export_location, "breaker_rate1"))
    sw2 = only(PSY.get_components(PSY.DiscreteControlledACBranch, sys2))
    @test PSY.get_rating(sw2, PSY.NU) ≈ PSY.get_rating(sw, PSY.NU)
end
