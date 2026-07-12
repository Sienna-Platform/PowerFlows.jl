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

When the solve ran with `control_discrete_devices`, the solved tap ratios / shunt admittances /
phase-shifter angles are written back into the system (see [`write_device_settings!`](@ref)): the
stored branch flows are only self-consistent with the mutated device settings, so the input
system's controlled devices are updated to the solved values. With no controls active this is a
no-op. This write-back only happens for single-period solves; for multiperiod solves
(`time_steps > 1`) it is skipped with a warning, since a PSY component holds one scalar per
setting and cannot round-trip a per-time-step schedule — use
[`get_controlled_device_results`](@ref) for the per-time-step schedule instead.

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
            # Write moved device settings back BEFORE write_power_flow_solution! recomputes flows —
            # its consistency assertion compares against the stored (moved) flows. Self-guards to a
            # no-op when no discrete controls ran.
            write_device_settings!(system, data)
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
    write_device_settings!(system, data)

Write the solved discrete-control device settings back into the `PSY.System`:
tap ratios (`set_tap!`), switched-shunt admittances (`set_Y!`/`set_initial_status!`,
convention-aware — see below), and phase-shifter angles (`set_α!`). FACTS devices
carry no stored setting field in PSY and are skipped. A device no longer present
in `system` is skipped with a `@warn` (its solved setting is not written back).
Mutates the user's system — called by [`solve_and_store_power_flow!`](@ref) after a
converged solve; a no-op when no discrete controls ran.

Switched shunts write back per their sourcing convention (see
[Metadata sourcing](@ref discrete-control-metadata) for how `psse_convention` is
determined): PSS/E-parsed (BINIT) components take the solved total straight into
`Y`; PSY API-built components keep `Y` at the fixed base and write the last-snap
`block_n` into `initial_status`, since overwriting `Y` would double-count the
status on re-enrollment. A never-snapped API-built device (continuous, or held in
its deadband the whole solve) whose `block_n` cannot reconstruct `d.current` falls
back to the BINIT write (solved total into `Y`, `initial_status` zeroed).

No-op for `time_steps > 1`: a PSY component holds a single scalar setting, but a
multiperiod solve produces one setting per time step, so there is no single value to
write back without silently discarding all but the last-processed step. Per-time-step
results remain available via [`get_controlled_device_results`](@ref).
"""
function write_device_settings!(system::PSY.System, data)
    set = get_controlled_devices(data)
    isnothing(set) && return
    if get_time_steps(data) > 1
        @warn "write_device_settings!: skipped — a PSY component holds a single scalar and \
            cannot round-trip a per-time-step schedule. Use get_controlled_device_results \
            for per-time-step device settings." maxlog = 1
        return
    end
    for d in set.taps
        tx = PSY.get_component(PSY.TapTransformer, system, d.name)
        if isnothing(tx)
            @warn "write_device_settings!: TapTransformer \"$(d.name)\" not found in the \
                system; its solved tap ratio $(d.current) was NOT written back."
            continue
        end
        PSY.set_tap!(tx, d.current)
    end
    for d in set.shunts
        sa = PSY.get_component(PSY.SwitchedAdmittance, system, d.name)
        if isnothing(sa)
            @warn "write_device_settings!: SwitchedAdmittance \"$(d.name)\" not found in \
                the system; its solved susceptance $(d.current) was NOT written back."
            continue
        end
        if d.psse_convention
            PSY.set_Y!(sa, Complex(d.g0, d.current))
        else
            realizable = d.b0 + sum(d.block_n .* d.block_dB; init = 0.0)
            if abs(realizable - d.current) <= BOUNDS_TOLERANCE
                PSY.set_Y!(sa, Complex(d.g0, d.b0))
                PSY.set_initial_status!(sa, copy(d.block_n))
            else
                PSY.set_Y!(sa, Complex(d.g0, d.current))
                PSY.set_initial_status!(sa, zeros(Int, length(d.block_n)))
            end
        end
    end
    for d in set.facts
        fd = PSY.get_component(PSY.FACTSControlDevice, system, d.name)
        if isnothing(fd)
            @warn "write_device_settings!: FACTSControlDevice \"$(d.name)\" not found in \
                the system; its solved reactive output was NOT written back."
            continue
        end
        # Delivered reactive power Q = b·|V_local|² (MVA) at the device's own bus.
        PSY.set_reactive_power_required!(
            fd, delivered_q_mvar(d, data.bus_magnitude[d.bus_ix, 1]))
    end
    return
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
    If the power flow converges, it updates the active and reactive power injections, as well as the voltage magnitudes and angles for different bus types (REF, PV, PQ), and calculates that time step's branch power flows.
    If the power flow does not converge, it sets the corresponding entries in `data` to `NaN`.

# Notes
- If the grid topology changes (e.g., tap positions of transformers or in-service status of branches), the admittance matrices `Yft` and `Ytf` must be updated before that time step's branch flows are computed.

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

    cd = get_controlled_devices(data)
    validate_device_store_width(cd, get_time_steps(data))
    for (ts_pos, time_step) in enumerate(sorted_time_steps)
        load_device_state!(cd, data, time_step)
        converged = _ac_power_flow_with_area_relax!(data, pf, time_step; merged_kwargs...)
        save_device_state!(cd, data, time_step)
        ts_converged[ts_pos] = converged
        converged && _warn_vsc_limit_violations(data, time_step)

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

        # Per-step branch flows (not batched after the loop) so a future per-step Yft/Ytf
        # (e.g. varying tap positions) is used correctly.
        # NOTE PNM's structs use ComplexF32, while the system objects store Float64's.
        #      so if you set the system bus angles/voltages to match these fields, then repeat
        #      this math using the system voltages, you'll see differences in the flows, ~1e-4.
        step_V =
            data.bus_magnitude[:, time_step] .* exp.(1im .* data.bus_angles[:, time_step])
        Sft = step_V[fb_ix] .* conj.(Yft.data * step_V)
        Stf = step_V[tb_ix] .* conj.(Ytf.data * step_V)
        data.arc_active_power_flow_from_to[:, time_step] .= real.(Sft)
        data.arc_reactive_power_flow_from_to[:, time_step] .= imag.(Sft)
        data.arc_active_power_flow_to_from[:, time_step] .= real.(Stf)
        data.arc_reactive_power_flow_to_from[:, time_step] .= imag.(Stf)

        _compute_arc_angle_differences_from_indices!(data, fb_ix, tb_ix, time_step)
    end

    data.converged[sorted_time_steps] .= ts_converged

    return all(ts_converged)
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

    # Iteration cap reached: the last `_check_q_limit_bounds!` flipped one or more PV buses to
    # PQ (and clamped their Q) without a follow-up solve, so `data`'s voltages no longer match
    # its bus types. Pin that final PV/PQ assignment and solve once more so the returned state is
    # self-consistent, then return THAT solve's actual convergence (not a forced `true`) — the
    # classic "fix-as-PQ after N iterations" resolution, rather than discarding a solution that
    # does converge.
    @warn(
        "reactive power limits still oscillating after $MAX_REACTIVE_POWER_ITERATIONS \
        iterations; pinning the final PV/PQ assignment and solving once more"
    )
    return _newton_power_flow(pf, data, time_step; kwargs...)
end

function _ac_power_flow(
    data::ACPowerFlowData,
    pf::AbstractACPowerFlow{<:ACPowerFlowSolverType},
    time_step::Int64;
    kwargs...,
)
    cd = data.controlled_devices
    if isnothing(cd) || isempty(cd)
        return _solve_with_q_limits!(pf, data, time_step; kwargs...)
    end
    return _control_continuation!(pf, data, time_step; kwargs...)
end

"""
    _ac_power_flow_with_area_relax!(data, pf, time_step; kwargs...) -> Bool

Wraps `_ac_power_flow` (itself already wrapping the Q-limit retry loop) with the
greedy-relax handling for embedded area net-interchange control (design spec §6, USER
DIRECTIVE — supersedes the spec's original "surface as non-convergence" posture): on a
non-converged solve with at least one controlled area still enrolled, de-enroll the area
with the largest `|r_a|` at the failed iterate (`_area_residual_gaps`/`_deenroll_area!`,
`area_residual.jl`) and re-solve from scratch — a fresh Q-limit loop, warm-started off the
current bus state, with surviving areas' `ΔP_a` re-seeded from their previous tail values
(the `delta_p` mirror already holds them — see `AreaInterchangeData`'s docstring) — repeating
until convergence or the enrolled set is exhausted. Zero enrolled areas left and still
failing is genuine network non-convergence, reported exactly as today (attribution: the
schedule was NOT the culprit) plus a terminal diagnostic naming the worst-missed area
(`_report_area_interchange_failure`, `residual_condition_diagnostics.jl`).

Converging after a relax still returns `true` — relaxation is never silent: an `@error` is
raised at EACH de-enrollment (naming the area, its target, and its residual gap) and again
as a solve-end summary if the time step converged only after relaxing; the results table
(`post_processing.jl`) also carries a `:relaxed` status row per relaxed area. A `data` on
which area interchange control was never enrolled at all (`pristine_areas` empty)
short-circuits to a bare `_ac_power_flow` call — no relax bookkeeping, no pristine-reset
check; this is the only zero-cost path (no allocation, no extra work) and is genuinely
per-`data`-lifetime, unlike the check below.

Resets to the FULL pristine enrollment before this time step's own attempt
(`_ensure_pristine_area_set!`), so a PREVIOUS time step's relax on this same `data` never
carries over — relax decisions are per time step. The short-circuit above deliberately tests
the PRISTINE set, not the WORKING one: the WORKING set is exactly what a previous time
step's greedy relax may have emptied (exhaustion, or a relax-to-zero-then-converge), and
short-circuiting on IT would read that leftover empty state before
`_ensure_pristine_area_set!` ever ran, permanently disabling area control for the rest of
`data`'s lifetime instead of restoring it for this time step.
"""
function _ac_power_flow_with_area_relax!(
    data::ACPowerFlowData,
    pf::AbstractACPowerFlow{<:ACPowerFlowSolverType},
    time_step::Int64;
    kwargs...,
)
    aid = data.area_interchange
    isempty(aid.pristine_areas) &&
        return _ac_power_flow(data, pf, time_step; kwargs...)
    _ensure_pristine_area_set!(data, time_step)
    relaxed_this_step = RelaxedAreaRecord[]
    converged = false
    while true
        converged = _ac_power_flow(data, pf, time_step; kwargs...)
        converged && break
        iszero(n_controlled_areas(data)) && break
        gaps = _area_residual_gaps(data, time_step)
        gap, worst_ix = findmax(abs, gaps)
        area = data.area_interchange.areas[worst_ix]
        @error "Area interchange: Newton did not converge with area \"$(area.name)\" " *
               "controlled (target PDES = $(area.pdes), NI gap at the failed iterate = " *
               "$gap); de-enrolling it and re-solving with the remaining " *
               "$(n_controlled_areas(data) - 1) controlled area(s)."
        push!(relaxed_this_step, RelaxedAreaRecord(area.name, area.pdes))
        _deenroll_area!(data, worst_ix)
    end
    if !converged
        _report_area_interchange_failure(data, time_step)
        return converged
    end
    _sync_pristine_delta_p!(data, time_step)
    _warn_area_violations(data, time_step)
    isempty(relaxed_this_step) && return converged
    data.area_interchange.relaxed[time_step] = relaxed_this_step
    pristine_tail_of = Dict(a.name => a.tail_ix for a in aid.pristine_areas)
    relaxed_detail = join(
        (
            let tail_ix = pristine_tail_of[r.name],
                ni_solved =
                    _area_net_interchange(aid.pristine_ties, tail_ix, data, time_step)

                "$(r.name) (ni_solved=$(ni_solved), pdes=$(r.pdes), " *
                "gap=$(ni_solved - r.pdes))"
            end
            for r in relaxed_this_step
        ),
        ", ",
    )
    @error "Area interchange: time step $time_step converged only after relaxing " *
           "$(length(relaxed_this_step)) area(s): $relaxed_detail. Their schedules were " *
           "infeasible given network/tie capacity."
    return converged
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
