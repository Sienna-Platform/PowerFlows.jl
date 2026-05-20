"""
    solve_and_store_power_flow!(pf::AbstractACPowerFlow{<:ACPowerFlowSolverType}, system::PSY.System; kwargs...)

Solves the power flow in the system and writes the solution into the relevant structs.
Updates active and reactive power setpoints for generators and active and reactive
power flows for branches (calculated in the From - To direction and in the To - From direction).

Configuration options like `time_steps`, `time_step_names`, `network_reductions`, and
`correct_bustypes` should be set on the `ACPowerFlow` object.

The bus types can be changed from PV to PQ if the reactive power limits are violated.

# Arguments
- [`pf::AbstractACPowerFlow{<:ACPowerFlowSolverType}`](@ref AbstractACPowerFlow): the power flow struct,
    which contains configuration options.
- `system::PSY.System`: The power system model, a [`PowerSystems.System`](@extref) struct.
- `kwargs...`: Additional keyword arguments passed to the solver.

## Keyword Arguments
- `tol`: Infinite norm of residuals under which convergence is declared. Default is `1e-9`.
- `maxIterations`: Maximum number of Newton-Raphson iterations. Default is `30`.

# Returns
- `converged::Bool`: Indicates whether the power flow solution converged.
- The power flow results are written into the system struct.

# Examples

```julia
solve_and_store_power_flow!(pf, sys)

# With correct_bustypes enabled
pf = ACPowerFlow(; correct_bustypes = true)
solve_and_store_power_flow!(pf, sys)

# Passing solver keyword arguments
solve_and_store_power_flow!(pf, sys; maxIterations=100)
```
"""
function solve_and_store_power_flow!(
    pf::AbstractACPowerFlow{<:ACPowerFlowSolverType},
    system::PSY.System;
    kwargs...,
)
    # converged must be defined in the outer scope to be visible for return
    converged = false
    with_units_base(system, PSY.UnitSystem.SYSTEM_BASE) do
        data = PowerFlowData(pf, system)

        converged = solve_power_flow!(data; kwargs...)

        if converged
            write_power_flow_solution!(
                system,
                pf,
                data,
                get(kwargs, :maxIterations, DEFAULT_NR_MAX_ITER),
            )
            @info("PowerFlow solve converged, the results have been stored in the system")
        else
            @error("The power flow solver returned convergence = $converged")
        end
    end

    return converged
end

"""
Similar to [solve\\_and\\_store\\_power\\_flow!](@ref) but does not update the system struct with results.
Returns the results in a dictionary of dataframes.

## Examples

```julia
res = solve_power_flow(pf, sys)
res = solve_power_flow(pf, sys, FlowReporting.BRANCH_FLOWS)
```
"""
function solve_power_flow(
    pf::AbstractACPowerFlow{<:ACPowerFlowSolverType},
    system::PSY.System;
    kwargs...,
)
    return solve_power_flow(pf, system, FlowReporting.ARC_FLOWS; kwargs...)
end

function solve_power_flow(
    pf::AbstractACPowerFlow{<:ACPowerFlowSolverType},
    system::PSY.System,
    flow_reporting::FlowReporting;
    kwargs...,
)
    # df_results must be defined in the outer scope first to be visible for return
    df_results = Dict{String, DataFrames.DataFrame}()
    converged = false
    time_step = 1
    with_units_base(system, PSY.UnitSystem.SYSTEM_BASE) do
        data = PowerFlowData(pf, system)

        converged = solve_power_flow!(data; kwargs...)

        if converged
            @info("PowerFlow solve converged, the results are exported in DataFrames")
            df_results = write_results(pf, system, data, time_step, flow_reporting)
        else
            df_results = missing
            @error("The power flow solver returned convergence = $(converged)")
        end
    end

    return df_results
end

"""
    solve_power_flow!(data::ACPowerFlowData; kwargs...)

Solve the multiperiod AC power flow problem for the given power flow data.

The bus types can be changed from PV to PQ if the reactive power limits are violated.
The power flow solver settings are taken from the `ACPowerFlow` object stored in `data`.

# Arguments
- [`data::ACPowerFlowData`](@ref ACPowerFlowData): The power flow data containing the grid information and initial conditions.
- `kwargs...`: Additional keyword arguments. If these overlap with those in the 
    `solver_settings` of the `ACPowerFlow` object, the values in `kwargs` take precedence.

# Keyword Arguments
- `time_steps`: Specifies the time steps to solve. Defaults to sorting and collecting the keys of `get_time_step_map(data)`.

# Description
This function solves the AC power flow problem for each time step specified in `data`.
It preallocates memory for the results and iterates over the sorted time steps.
    For each time step, it calls the `_ac_power_flow` function to solve the power flow equations and updates the `data` object with the results.
    If the power flow converges, it updates the active and reactive power injections, as well as the voltage magnitudes and angles for different bus types (REF, PV, PQ).
    If the power flow does not converge, it sets the corresponding entries in `data` to `NaN`.
    Finally, it calculates the branch power flows and updates the `data` object.

# Notes
- If the grid topology changes (e.g., tap positions of transformers or in-service status of branches), the admittance matrices `Yft` and `Ytf` must be updated.
- If `Yft` and `Ytf` change between time steps, the branch flow calculations must be moved inside the loop.

# Examples
```julia
solve_power_flow!(data)
```
"""
function solve_power_flow!(
    data::ACPowerFlowData;
    kwargs...,
)
    pf = get_pf(data)
    # Merge solver_settings from pf with any explicitly passed kwargs (explicit kwargs take precedence)
    merged_kwargs = merge(get_solver_kwargs(pf), kwargs)
    sorted_time_steps =
        get(merged_kwargs, :time_steps, sort(collect(keys(get_time_step_map(data)))))
    # This can be done from PSI by directly writing to `data`'s fields; we just don't
    # do it here in PF alone.
    if length(sorted_time_steps) > 1
        @warn(
            "Multi-period AC power flow: each time step is solved independently " *
            "using the same network data. Time-varying generator setpoints or " *
            "limits are not updated between time steps.",
            maxlog = 1,
        )
    end
    # preallocate results
    ts_converged = fill(false, length(sorted_time_steps))

    Yft = data.power_network_matrix.arc_admittance_from_to
    Ytf = data.power_network_matrix.arc_admittance_to_from
    @assert PNM.get_bus_lookup(Yft) == get_bus_lookup(data)
    arcs = PNM.get_arc_axis(Yft)
    @assert arcs == PNM.get_arc_axis(Ytf)
    @assert length(PNM.get_bus_axis(Yft)) == length(data.bus_angles[:, 1])
    bus_lookup = get_bus_lookup(data)
    fb_ix = [bus_lookup[bus_no] for bus_no in first.(arcs)]  # from bus indices
    tb_ix = [bus_lookup[bus_no] for bus_no in last.(arcs)]   # to bus indices
    @assert length(fb_ix) == length(arcs)

    for time_step in sorted_time_steps
        converged = _ac_power_flow(data, pf, time_step; merged_kwargs...)
        ts_converged[time_step] = converged

        if OVERWRITE_NON_CONVERGED && !converged
            # set values to NaN for not converged time steps
            data.bus_active_power_injections[:, time_step] .= NaN
            data.bus_active_power_withdrawals[:, time_step] .= NaN
            data.bus_active_power_constant_current_withdrawals[:, time_step] .= NaN
            data.bus_active_power_constant_impedance_withdrawals[:, time_step] .= NaN
            data.bus_reactive_power_injections[:, time_step] .= NaN
            data.bus_reactive_power_withdrawals[:, time_step] .= NaN
            data.bus_reactive_power_constant_current_withdrawals[:, time_step] .= NaN
            data.bus_reactive_power_constant_impedance_withdrawals[:, time_step] .= NaN
            data.bus_magnitude[:, time_step] .= NaN
            data.bus_angles[:, time_step] .= NaN
        elseif get_lcc_count(data) > 0 && converged
            # calculate branch flows for LCCs: their self-admittances may change.
            V =
                data.bus_magnitude[:, time_step] .*
                exp.(1im .* data.bus_angles[:, time_step])
            for (i, (bus_indices, self_admittances)) in
                enumerate(zip(data.lcc.bus_indices, data.lcc.branch_admittances))
                (rectifier_ix, inverter_ix) = bus_indices
                (rectifier_y, inverter_y) = self_admittances
                S_inverter = V[inverter_ix] * conj(inverter_y * V[inverter_ix])
                S_rectifier = V[rectifier_ix] * conj(rectifier_y * V[rectifier_ix])
                data.lcc.arc_active_power_flow_from_to[i, time_step] =
                    real(S_rectifier)
                data.lcc.arc_reactive_power_flow_from_to[i, time_step] =
                    imag(S_rectifier)
                data.lcc.arc_active_power_flow_to_from[i, time_step] =
                    real(S_inverter)
                data.lcc.arc_reactive_power_flow_to_from[i, time_step] =
                    imag(S_inverter)
            end
        end
    end

    # write branch flows
    # NOTE PNM's structs use ComplexF32, while the system objects store Float64's.
    #      so if you set the system bus angles/voltages to match these fields, then repeat 
    #      this math using the system voltages, you'll see differences in the flows, ~1e-4.
    ts_V =
        data.bus_magnitude[:, sorted_time_steps] .*
        exp.(1im .* data.bus_angles[:, sorted_time_steps])

    Sft = ts_V[fb_ix, :] .* conj.(Yft.data * ts_V)
    Stf = ts_V[tb_ix, :] .* conj.(Ytf.data * ts_V)
    data.arc_active_power_flow_from_to .= real.(Sft)
    data.arc_reactive_power_flow_from_to .= imag.(Sft)
    data.arc_active_power_flow_to_from .= real.(Stf)
    data.arc_reactive_power_flow_to_from .= imag.(Stf)

    _compute_arc_angle_differences_from_indices!(data, fb_ix, tb_ix, sorted_time_steps)

    data.converged .= ts_converged

    return all(data.converged)
end

function _solve_with_q_limits!(
    pf::AbstractACPowerFlow{<:ACPowerFlowSolverType},
    data::ACPowerFlowData,
    time_step::Int64;
    kwargs...,
)
    check_reactive_power_limits = pf.check_reactive_power_limits
    converged = false

    for _ in 1:MAX_REACTIVE_POWER_ITERATIONS
        converged = _newton_power_flow(pf, data, time_step; kwargs...)
        if !converged || !check_reactive_power_limits ||
           _check_q_limit_bounds!(data, time_step)
            return converged
        end
    end

    @error("could not enforce reactive power limits after $MAX_REACTIVE_POWER_ITERATIONS")
    return converged
end

function _ac_power_flow(
    data::ACPowerFlowData,
    pf::AbstractACPowerFlow{<:ACPowerFlowSolverType},
    time_step::Int64;
    kwargs...,
)
    cd = data.controlled_devices
    if cd === nothing || isempty(cd)
        return _solve_with_q_limits!(pf, data, time_step; kwargs...)
    end
    return _control_continuation!(pf, data, time_step; kwargs...)
end

function _check_q_limit_bounds!(
    data::ACPowerFlowData,
    time_step::Int64,
)
    bus_names = data.power_network_matrix.axes[1]
    within_limits = true
    bus_types = view(data.bus_type, :, time_step)
    for (ix, bt) in enumerate(bus_types)
        bt != PSY.ACBusTypes.PV && continue
        Q_gen = data.bus_reactive_power_injections[ix, time_step]

        Q_max = data.bus_reactive_power_bounds[ix, time_step][2]
        Q_min = data.bus_reactive_power_bounds[ix, time_step][1]

        if !(Q_min - BOUNDS_TOLERANCE <= Q_gen <= Q_max + BOUNDS_TOLERANCE)
            @info "Bus $(bus_names[ix]) changed to PSY.ACBusTypes.PQ"
            within_limits = false
            data.bus_type[ix, time_step] = PSY.ACBusTypes.PQ
            data.bus_reactive_power_injections[ix, time_step] =
                clamp(Q_gen, Q_min, Q_max)
        else
            @debug "Within Limits"
        end
    end
    return within_limits
end

function bus_type_idx(
    data::ACPowerFlowData,
    time_step::Int64 = 1,
    bus_types::Tuple{Vararg{PSY.ACBusTypes}} = (
        PSY.ACBusTypes.REF,
        PSY.ACBusTypes.PV,
        PSY.ACBusTypes.PQ,
    ),
)
    # Find indices for each bus type
    return [
        findall(==(bus_type), data.bus_type[:, time_step]) for bus_type in bus_types
    ]
end
