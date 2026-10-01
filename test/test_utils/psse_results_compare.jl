function psse_bus_results_compare(file_name::String, pf_results::Dict)
    pf_result_bus = CSV.read(file_name, DataFrame)

    v_diff = Float64[]
    angle_diff = Float64[]
    number = Int[]
    for (ix, n) in enumerate(eachrow(pf_result_bus))
        push!(v_diff, pf_results["bus_results"][ix, "Vm"] - n."Voltage (pu)")
        push!(
            angle_diff,
            pf_results["bus_results"][ix, "θ"] - (n."Angle (deg)" * (π / 180)),
        )
        push!(number, pf_results["bus_results"][ix, "bus_number"])
    end
    return v_diff, angle_diff, number
end

function psse_bus_results_compare(file_name::String, pf_results::Missing)
    throw(ArgumentError("pf_results are missing - calculation failed"))
end

function psse_gen_results_compare(file_name, system::PSY.System)
    base_power = get_base_power(system, PSY.NU)
    pf_result_gen = CSV.read(file_name, DataFrame)
    p_diff = Float64[]
    q_diff = Float64[]
    names = String[]
    for row in eachrow(pf_result_gen)
        if ismissing(row."Bus  Number")
            continue
        end
        name = "generator-" * "$(row."Bus  Number")" * "-" * row."Id"
        gen = get_component(ThermalStandard, system, name)
        ap = get_active_power(gen, PSY.SU)
        rp = get_reactive_power(gen, PSY.SU)
        push!(p_diff, ap - row."PGen (MW)" / base_power)
        push!(q_diff, rp - row."QGen (Mvar)" / base_power)
        push!(names, name)
    end
    return p_diff, q_diff, names
end
