function test_reduced_power_flow(
    pf::PF.PowerFlowEvaluationModel,
    sys::PSY.System,
    nrs::Vector{PNM.NetworkReduction},
)
    data = PF.PowerFlowData(pf, sys)
    nrd = PNM.get_network_reduction_data(data.power_network_matrix)
    @test issetequal(
        filter(
            !=(PNM.ZeroImpedanceBranchReduction),
            map(typeof, PNM.get_applied_reductions(nrd)),
        ),
        map(typeof, nrs),
    )
    if pf isa PF.ACPowerFlow
        PF.solve_power_flow!(data; pf = pf)
    else
        PF.solve_power_flow!(data)
    end
    return data
end

function get_bus_voltages(data::PF.PowerFlowData)
    bus_Vm = PF.get_bus_magnitude(data)
    bus_Va = PF.get_bus_angles(data)
    bus_voltages = Dict{Int, Complex{Float64}}()
    for (bus, i) in PF.get_bus_lookup(data)
        bus_voltages[bus] = bus_Vm[i] * exp(1im * bus_Va[i])
    end
    return bus_voltages
end

function get_branch_flows(data::PF.PowerFlowData)
    branch_flows = Dict{Tuple{Int, Int}, ComplexF64}()
    arc_ax = PF.get_arc_axis(data)
    for (i, arc) in enumerate(arc_ax)
        branch_flows[arc] =
            data.arc_active_power_flow_from_to[i] .+
            (im .* data.arc_reactive_power_flow_from_to[i])
        branch_flows[reverse(arc)] =
            data.arc_active_power_flow_to_from[i] .+
            (im .* data.arc_reactive_power_flow_to_from[i])
    end
    return branch_flows
end

function get_reduced_buses(nrd::PNM.NetworkReductionData)
    reduced_buses = Set()
    for (bus, reduced_set) in PNM.get_bus_reduction_map(nrd)
        if length(reduced_set) > 0
            push!(reduced_buses, bus)
        end
    end
    return reduced_buses
end

function validate_reduced_power_flow(
    pf::PF.PowerFlowEvaluationModel,
    sys::PSY.System,
    nrs::Vector{PNM.NetworkReduction},
    unreduced_solved_data::PF.PowerFlowData,
)
    data = PF.PowerFlowData(pf, sys)
    nrd = PNM.get_network_reduction_data(data.power_network_matrix)
    @test issetequal(
        filter(
            !=(PNM.ZeroImpedanceBranchReduction),
            map(typeof, PNM.get_applied_reductions(nrd)),
        ),
        map(typeof, nrs),
    )
    if pf isa PF.ACPowerFlow
        PF.solve_power_flow!(data; pf = pf)
    else
        PF.solve_power_flow!(data)
    end
    @test all(data.converged)
    reduced_bus_results = get_bus_voltages(data)
    unreduced_bus_results = get_bus_voltages(unreduced_solved_data)
    reduced_buses = get_reduced_buses(nrd)
    if all(data.converged)
        buses_match, branches_match = true, true
        for bus in PNM.get_bus_axis(data.power_network_matrix)
            if !(bus in reduced_buses)
                buses_match =
                    buses_match && isapprox(
                        reduced_bus_results[bus],
                        unreduced_bus_results[bus];
                        atol = 1e-6,
                    )
            end
        end
        @test buses_match
        reduced_branch_flows = get_branch_flows(data)
        unreduced_branch_results = get_branch_flows(unreduced_solved_data)
        for arc in keys(PNM.get_direct_branch_map(nrd))
            branches_match =
                branches_match && isapprox(
                    reduced_branch_flows[arc],
                    unreduced_branch_results[arc];
                    atol = 1e-4, # why is the accuracy much worse for branches?
                )
        end
        @test branches_match
    else
        @error "Reduced power flow did not converge; skipping result validation."
    end
end
