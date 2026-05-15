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
    isapprox(a, b; atol = SYSTEM_REIMPORT_COMPARISON_TOLERANCE) || IS.isequivalent(a, b)
loose_system_match_fn(a, b) = IS.isequivalent(a, b)

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
        PSY.PhaseShiftingTransformer,
        PSY.PhaseShiftingTransformer3W,
        PSY.StandardLoad,
        PSY.SwitchedAdmittance,
        PSY.TapTransformer,
        PSY.ThermalStandard,
        PSY.Transformer2W,
        PSY.Transformer3W,
        PSY.TwoTerminalLCCLine,
        PSY.TwoTerminalVSCLine,
    ],
    # TODO when possible, don't exclude so many fields
    exclude_fields = Set([
        :ext,
        :ramp_limits,
        :time_limits,
        :services,
        :angle_limits,
        :winding_group_number,
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
        PSY.TapTransformer => Set([
            :active_power_flow,
            :reactive_power_flow,
        ]),
        PSY.Transformer2W => Set([
            :active_power_flow,
            :reactive_power_flow,
        ]),
        PSY.Transformer3W => Set([
            :active_power_flow,
            :reactive_power_flow,
            :rating,  # TODO why don't ratings match?
            :rating_primary,
            :rating_secondary,
            :rating_tertiary,
        ]),
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

    # Compare everything about the systems except the actual components
    result &= IS.compare_values(sys1, sys2; exclude = [:data])

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
            result &= comparison
            if !comparison
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
    reactive_power_tol =
        exclude_reactive_flow ? nothing : POWERFLOW_COMPARISON_TOLERANCE
    @test compare_df_within_tolerance("bus_results", result1["bus_results"],
        result2["bus_results"], POWERFLOW_COMPARISON_TOLERANCE)
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

# Exercise PowerSystems' ability to parse a PSS/E System from a filename and a metadata dict
function read_system_with_metadata(raw_path, metadata_path)
    md = JSON3.read(metadata_path, Dict)
    sys = System(raw_path, md)
    return sys
end

# Exercise PowerSystems' ability to automatically find the export metadata file
read_system_with_metadata(export_subdir) =
    System(first(get_psse_export_paths(export_subdir)))

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
    open(raw1, "r") do handle1
        open(raw2, "r") do handle2
            @test countlines(handle1) == countlines(handle2)
            for (line1, line2) in zip(readlines(handle1), readlines(handle2))
                @test line1 == line2
            end
        end
    end

    parsed1 = JSON3.read(metadata1, Dict)
    parsed2 = JSON3.read(metadata2, Dict)
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
    set_units_base_system!(sys, UnitSystem.SYSTEM_BASE)
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

# Test configurations: (test_name, sys_name, version, folder_name)
# ReTest chokes on @testset over a loop.
#=
test_configs = [
    (
        "PSSE Exporter with case16_sys.raw, v33",
        "pti_case16_complete_sys",
        :v33,
        "case16_sys.raw",
    ),
    (
        "PSSE Exporter with modified_case25_sys.raw, v35",
        "pti_modified_case25_v35_sys",
        :v35,
        "modified_case25_sys.raw",
    ),
]=#

@testset "PSSE Exporter with case16_sys.raw, v33" begin
    test_psse_exporter_version("pti_case16_complete_sys", :v33, "case16_sys.raw")
end

@testset "PSSE Exporter with modified_case25_sys.raw, v35" begin
    test_psse_exporter_version("pti_modified_case25_v35_sys", :v35,
        "modified_case25_sys.raw")
end

@testset "PSSE Exporter RTS regression: TapTransformer and v35 default ratings" begin
    sys = with_logger(SimpleLogger(Error)) do
        build_system(PSISystems, "modified_RTS_GMLC_DA_sys"; force_build = true)
    end
    isnothing(sys) && return
    set_units_base_system!(sys, UnitSystem.SYSTEM_BASE)

    undefined_obj =
        PSY.TransformerControlObjectiveModule.TransformerControlObjective.UNDEFINED
    tap_transformers = collect(PSY.get_components(PSY.TapTransformer, sys))
    target_tap_idx = findfirst(
        t ->
            PSY.get_control_objective(t) == undefined_obj &&
                !isapprox(PSY.get_tap(t), 1.0),
        tap_transformers,
    )
    @test !isnothing(target_tap_idx)
    isnothing(target_tap_idx) && return
    target_tap = tap_transformers[target_tap_idx]

    lines = sort!(collect(PSY.get_components(PSY.Line, sys)); by = PSY.get_name)
    @test !isempty(lines)
    isempty(lines) && return
    target_line = first(lines)

    # In v35, unspecified extra rating fields are exported as explicit 0.0 values.
    export_location = joinpath(test_psse_export_dir, "v35", "rts_targeted_regressions")
    scenario_name = "taptransformer_nonunity_and_v35_missing_ratings"
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
    tap_transformer_keys =
        filter(k -> endswith(k, "_" * tap_name), collect(keys(transformer_ckt_mapping)))
    tap_branch_keys =
        filter(k -> endswith(k, "_" * tap_name), collect(keys(branch_name_mapping)))
    @test length(tap_transformer_keys) == 1
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

    tap_key = tap_transformer_keys[1]
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
    set_rating!(line_to_change, get_rating(line_to_change) * 123.4)  # careful not to exceed PF.INFINITE_BOUND
    update_exporter!(exporter, sys2)
    write_export(exporter, "basic4"; overwrite = true)
    reread_sys2 = read_system_with_metadata(joinpath(export_location, "basic4"))
    @test compare_systems_loosely(sys2, reread_sys2)
    @test_logs((:error, r"values do not match"),
        match_mode = :any, min_level = Logging.Error,
        compare_systems_loosely(sys, reread_sys2))
    test_power_flow(pf, sys2, reread_sys2; exclude_reactive_flow = true)
end

@testset "PSSE Exporter with case24_sys.raw, v33 - LUACPowerFlow" begin
    test_psse_exporter_inner(LUACPowerFlow, "case24_sys_LU")
end

@testset "PSSE Exporter with case24_sys.raw, v33 - NewtonRaphsonACPowerFlow" begin
    test_psse_exporter_inner(NewtonRaphsonACPowerFlow, "case24_sys_NR")
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

# # TODO add tests for unit system agnosticism
