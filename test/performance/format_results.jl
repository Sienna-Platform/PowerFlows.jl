"""
Reads per-version CSV results and writes a Markdown comparison table.

Usage:
    julia format_results.jl <main_csv> <branch_csv> <main_precompile_txt> <branch_precompile_txt> <output_md>
"""

function read_results(path::AbstractString)::Vector{Tuple{String, String}}
    results = Tuple{String, String}[]
    for line in eachline(path)
        isempty(strip(line)) && continue
        parts = split(line, ","; limit = 2)
        push!(results, (strip(parts[1]), strip(parts[2])))
    end
    return results
end

function format_time(val::AbstractString)::String
    (val == "FAILED" || val == "N/A") && return val
    t = parse(Float64, val)
    if t < 1.0
        return string(round(t * 1000; digits = 1), " ms")
    else
        return string(round(t; digits = 3), " s")
    end
end

function compute_delta(main_val::AbstractString, branch_val::AbstractString)::String
    (
        main_val == "FAILED" || branch_val == "FAILED" ||
        main_val == "N/A" || branch_val == "N/A"
    ) && return "N/A"
    main_t = parse(Float64, main_val)
    branch_t = parse(Float64, branch_val)
    main_t == 0.0 && return "N/A"
    pct = round((branch_t - main_t) / main_t * 100; digits = 1)
    sign = pct > 0 ? "+" : ""
    return "$(sign)$(pct)%"
end

function main()
    main_csv = ARGS[1]
    branch_csv = ARGS[2]
    main_precompile = ARGS[3]
    branch_precompile = ARGS[4]
    output_md = ARGS[5]

    main_results = read_results(main_csv)
    branch_results = read_results(branch_csv)

    main_dict = Dict(name => val for (name, val) in main_results)
    branch_dict = Dict(name => val for (name, val) in branch_results)

    # Preserve order from main results, then add any branch-only tests
    all_tests = String[name for (name, _) in main_results]
    for (name, _) in branch_results
        name in all_tests || push!(all_tests, name)
    end

    main_precompile_time = strip(read(main_precompile, String))
    branch_precompile_time = strip(read(branch_precompile, String))

    open(output_md, "w") do io
        write(io, "## Performance Results\n\n")

        # Precompile table
        precompile_delta =
            compute_delta(main_precompile_time, branch_precompile_time)
        write(io, "### Precompile Time\n\n")
        write(io, "| Main | This Branch | Delta |\n")
        write(io, "| :---: | :---: | :---: |\n")
        write(
            io,
            "| $(format_time(main_precompile_time)) | $(format_time(branch_precompile_time)) | $(precompile_delta) |\n",
        )

        # Solve time table
        write(io, "\n### Solve Time\n\n")
        write(io, "| Test | Main | This Branch | Delta |\n")
        write(io, "| :--- | :---: | :---: | :---: |\n")
        for test in all_tests
            main_val = get(main_dict, test, "N/A")
            branch_val = get(branch_dict, test, "N/A")
            delta = compute_delta(main_val, branch_val)
            write(
                io,
                "| $(test) | $(format_time(main_val)) | $(format_time(branch_val)) | $(delta) |\n",
            )
        end
    end
end

main()
