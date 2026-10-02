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
    end

    return converged
end

# Re-resolve a tap's owning transformer in `system` by name, rather than holding a reference,
# so the write lands in the caller's system even when it is not the one enrollment read.
# Looked up under the concrete arity types, never abstract `PSY.ACTransmission`: a `Line`
# sharing the transformer's name would otherwise make the lookup ambiguous.
function _lookup_tap_transformer(system::PSY.System, name::String)
    tx = PSY.get_component(PSY.TwoWindingTransformer, system, name)
    isnothing(tx) || return tx
    return PSY.get_component(PSY.ThreeWindingTransformer, system, name)
end

# The tap may sit on either arity, and `PSY.get_circuits` covers both (a 2W returns a
# 1-tuple). Bool predicate + accessor, not a `nothing`-returning resolver.
function _has_tap_circuit(system::PSY.System, d::ControlledTap)
    tx = _lookup_tap_transformer(system, d.device_name)
    isnothing(tx) && return false
    return d.circuit_index <= length(PSY.get_circuits(tx))
end

function _tap_circuit(system::PSY.System, d::ControlledTap)
    tx = _lookup_tap_transformer(system, d.device_name)
    return PSY.get_circuits(tx)[d.circuit_index]
end

"""
    write_device_settings!(system::PSY.System, data)

Write solved discrete-control device settings back onto the components of `system`: tap
ratios onto the owning `PSY.TransformerCircuit` (of either transformer arity), and switched
shunt and FACTS settings onto their devices. A no-op when `data` carries no controlled
devices.

Skips with a warning when `get_time_steps(data) > 1`: a PSY component holds a single scalar
setting and cannot represent a per-time-step schedule, so writing back would silently
discard every step but one. Use [`get_controlled_device_results`](@ref) for the full
per-step settings.
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
        if !_has_tap_circuit(system, d)
            @warn "write_device_settings!: transformer \"$(d.device_name)\" not found in \
                the system; the solved tap ratio $(d.current) for \"$(d.name)\" was NOT \
                written back."
            continue
        end
        PSY.set_tap!(_tap_circuit(system, d), d.current)
    end
    for d in set.shunts
        sa = PSY.get_component(PSY.SwitchedAdmittance, system, d.name)
        if isnothing(sa)
            @warn "write_device_settings!: SwitchedAdmittance \"$(d.name)\" not found in \
                the system; its solved susceptance $(d.current) was NOT written back."
            continue
        end
        if d.psse_convention
            PSY.set_solved_admittance!(sa, d.current)
        else
            realizable = sum(d.block_n .* d.block_dB; init = 0.0)
            if abs(realizable - d.current) <= BOUNDS_TOLERANCE
                PSY.set_number_engaged!(sa, copy(d.block_n))
                PSY.set_solved_admittance!(sa, nothing)
            else
                PSY.set_number_engaged!(sa, zeros(Int, length(d.block_n)))
                PSY.set_solved_admittance!(sa, d.current)
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
Similar to [`solve_and_store_power_flow!`](@ref) but does not update the system struct with results.
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
    flow_reporting::FlowReporting.Value;
    kwargs...,
)
    # df_results must be defined in the outer scope first to be visible for return
    df_results = Dict{String, DataFrames.DataFrame}()
    converged = false
    time_step = 1
    data = PowerFlowData(pf, system)

    converged = solve_power_flow!(data; kwargs...)

    if converged
        @info("PowerFlow solve converged, the results are exported in DataFrames")
        df_results = write_results(pf, system, data, time_step, flow_reporting)
    else
        df_results = missing
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
    `solution_parameters` of the `ACPowerFlow` object, the values in `kwargs` take precedence.

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
    merged_kwargs = merge(get_solver_kwargs(pf), NamedTuple(kwargs))
    merged_kwargs.maxIterations < 1 && error(
        "maxIterations must be >= 1, got $(merged_kwargs.maxIterations) for $(typeof(pf)).",
    )
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

    validate_device_store_width(get_controlled_devices(data), get_time_steps(data))
    n_tasks = min(get_n_threads(pf), length(sorted_time_steps))
    if n_tasks > 1
        _solve_time_steps_threaded!(
            data, pf, sorted_time_steps, ts_converged, n_tasks; merged_kwargs...)
    else
        flow_ws = _BranchFlowWorkspace(data)
        for (ts_pos, time_step) in enumerate(sorted_time_steps)
            ts_converged[ts_pos] =
                _solve_time_step!(data, pf, time_step, flow_ws; merged_kwargs...)
        end
    end

    data.converged[sorted_time_steps] .= ts_converged

    if !all(ts_converged)
        failed = sorted_time_steps[.!ts_converged]
        @error "AC power flow did not converge in $(length(failed)) of $(length(ts_converged)) time step(s): $failed"
    end

    return all(ts_converged)
end

"""Branch-flow inputs and buffers for [`_solve_time_step!`](@ref), reused across time steps.
One per task: the buffers are written in place."""
struct _BranchFlowWorkspace{Y}
    Yft::Y
    Ytf::Y
    fb_ix::Vector{Int}  # from bus indices
    tb_ix::Vector{Int}  # to bus indices
    step_V::Vector{ComplexF64}
    Sft::Vector{ComplexF64}
    Stf::Vector{ComplexF64}
end

function _BranchFlowWorkspace(data::ACPowerFlowData)
    Yft = data.power_network_matrix.arc_admittance_from_to
    Ytf = data.power_network_matrix.arc_admittance_to_from
    @assert PNM.get_bus_lookup(Yft) == get_bus_lookup(data)
    arcs = PNM.get_arc_axis(Yft)
    @assert arcs == PNM.get_arc_axis(Ytf)
    n_buses = size(data.bus_angles, 1)
    @assert length(PNM.get_bus_axis(Yft)) == n_buses
    bus_lookup = get_bus_lookup(data)
    fb_ix = [bus_lookup[bus_no] for bus_no in first.(arcs)]
    tb_ix = [bus_lookup[bus_no] for bus_no in last.(arcs)]
    @assert length(fb_ix) == length(arcs)
    return _BranchFlowWorkspace(
        Yft, Ytf, fb_ix, tb_ix,
        Vector{ComplexF64}(undef, n_buses),
        Vector{ComplexF64}(undef, length(arcs)),
        Vector{ComplexF64}(undef, length(arcs)),
    )
end

"""Solve one time step of the multiperiod AC power flow and write its results (bus state,
LCC and branch flows, angle differences) into column `time_step` of `data`. Returns whether
the step converged. Reads and writes only column `time_step` of the time-indexed fields."""
function _solve_time_step!(
    data::ACPowerFlowData,
    pf::AbstractACPowerFlow,
    time_step::Int,
    ws::_BranchFlowWorkspace;
    kwargs...,
)
    cd = get_controlled_devices(data)
    load_device_state!(cd, data, time_step)
    converged = _ac_power_flow_with_area_relax!(data, pf, time_step; kwargs...)
    save_device_state!(cd, data, time_step)
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

    # Per-step branch flows (not batched after the loop) so a per-step Yft/Ytf
    # (e.g. varying tap positions) is used correctly.
    # NOTE PNM's structs use ComplexF32, while the system objects store Float64's.
    #      so if you set the system bus angles/voltages to match these fields, then repeat
    #      this math using the system voltages, you'll see differences in the flows, ~1e-4.
    @views ws.step_V .=
        data.bus_magnitude[:, time_step] .* exp.(1im .* data.bus_angles[:, time_step])
    mul!(ws.Sft, ws.Yft.data, ws.step_V)
    mul!(ws.Stf, ws.Ytf.data, ws.step_V)
    ws.Sft .= view(ws.step_V, ws.fb_ix) .* conj.(ws.Sft)
    ws.Stf .= view(ws.step_V, ws.tb_ix) .* conj.(ws.Stf)
    data.arc_active_power_flow_from_to[:, time_step] .= real.(ws.Sft)
    data.arc_reactive_power_flow_from_to[:, time_step] .= imag.(ws.Sft)
    data.arc_active_power_flow_to_from[:, time_step] .= real.(ws.Stf)
    data.arc_reactive_power_flow_to_from[:, time_step] .= imag.(ws.Stf)

    _compute_arc_angle_differences_from_indices!(data, ws.fb_ix, ws.tb_ix, time_step)
    return converged
end

"""
    _make_time_step_worker(data::ACPowerFlowData) -> ACPowerFlowData

A `PowerFlowData` for one task of a threaded solve. It shares `data`'s time-indexed arrays
(each step touches only its own column) and read-only network data, and owns its own copy of
everything else a solve mutates:
- the solver caches;
- `converged`, all `false`, so no step warm-starts from another step's solution;
- `lcc.branch_admittances`;
- the area-interchange data: a relax renumbers it across all columns, so only each step's
  result column is copied back ([`_merge_worker_area_step!`](@ref));
- the controlled devices' scratch objects and counters (their per-step stores stay shared);
- the network matrix, when there are controlled taps (they write Y-bus, `Yft`, `Ytf`).

When adding state that a solve mutates, give the worker its own copy here.
"""
function _make_time_step_worker(data::ACPowerFlowData)
    cd = get_controlled_devices(data)
    has_taps = !isnothing(cd) && !isempty(cd.taps)
    worker_cd = if isnothing(cd)
        nothing
    else
        _override(cd;
            taps = deepcopy(cd.taps),
            shunts = deepcopy(cd.shunts),
            facts = deepcopy(cd.facts),
            inner_solves = Ref(0),
            symbolic_factors = Ref(0),
            numeric_refactors = Ref(0),
        )
    end
    return _override(data;
        power_network_matrix = if has_taps
            deepcopy(data.power_network_matrix)
        else
            data.power_network_matrix
        end,
        converged = fill(false, length(data.converged)),
        lcc = _override(data.lcc; branch_admittances = copy(data.lcc.branch_admittances)),
        area_interchange = deepcopy(data.area_interchange),
        controlled_devices = worker_cd,
        solver_cache = Base.RefValue{Union{Nothing, SolverCache}}(nothing),
        ac_jacobian_structure_cache = Base.RefValue{
            Union{Nothing, ACJacobianStructureCache},
        }(
            nothing,
        ),
        polar_nr_cache = Base.RefValue{Union{Nothing, AbstractNRCache}}(nothing),
    )
end

"""Copy column `time_step` of a worker's `pristine_delta_p` (what post-processing reads)
back into `data`. Safe concurrently, since workers write disjoint columns."""
function _merge_worker_area_step!(data::ACPowerFlowData, worker::ACPowerFlowData, time_step)
    isempty(data.area_interchange.pristine_areas) && return
    @views data.area_interchange.pristine_delta_p[:, time_step] .=
        worker.area_interchange.pristine_delta_p[:, time_step]
    return
end

"""Solve `time_steps` over `n_tasks` tasks, each with its own worker
([`_make_time_step_worker`](@ref)) and a contiguous block of steps. Fills `ts_converged`,
ordered like `time_steps`."""
function _solve_time_steps_threaded!(
    data::ACPowerFlowData,
    pf::AbstractACPowerFlow,
    time_steps::AbstractVector{Int},
    ts_converged::Vector{Bool},
    n_tasks::Int;
    kwargs...,
)
    if Threads.nthreads() < n_tasks
        @warn "n_threads = $n_tasks but Julia has $(Threads.nthreads()) thread(s); \
            start Julia with `--threads=$n_tasks` to solve the time steps in parallel." maxlog =
            1
    end
    blocks = collect(
        Iterators.partition(eachindex(time_steps), cld(length(time_steps), n_tasks)),
    )
    workers = [_make_time_step_worker(data) for _ in blocks]
    # `@sync` waits for every task before rethrowing a failure (as a `CompositeException`),
    # so no task is still writing into `data` when control returns to the caller.
    @sync for (block, worker) in zip(blocks, workers)
        Threads.@spawn begin
            ws = _BranchFlowWorkspace(worker)
            for pos in block
                time_step = time_steps[pos]
                ts_converged[pos] =
                    _solve_time_step!(worker, pf, time_step, ws; kwargs...)
                _merge_worker_area_step!(data, worker, time_step)
            end
        end
    end
    # `relaxed` is a `Dict`, so merge after the join. Each worker's started as a copy of
    # `data`'s: take only the steps it solved.
    for (block, worker) in zip(blocks, workers), pos in block
        time_step = time_steps[pos]
        records = get(worker.area_interchange.relaxed, time_step, nothing)
        isnothing(records) || (data.area_interchange.relaxed[time_step] = records)
    end
    return
end

function _solve_with_q_limits!(
    pf::AbstractACPowerFlow{<:ACPowerFlowSolverType},
    data::ACPowerFlowData,
    time_step::Int64;
    kwargs...,
)
    check_reactive_power_limits = get(
        kwargs, :check_reactive_power_limits, get_check_reactive_power_limits(pf))
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

"""Dispatch on `data.controlled_devices` so the discrete-control continuation is compiled only
for solves that carry a `ControlledDeviceSet`."""
function _ac_power_flow(
    data::ACPowerFlowData,
    pf::AbstractACPowerFlow{<:ACPowerFlowSolverType},
    time_step::Int64;
    kwargs...,
)
    return _ac_power_flow(data.controlled_devices, data, pf, time_step; kwargs...)
end

function _ac_power_flow(
    ::Nothing,
    data::ACPowerFlowData,
    pf::AbstractACPowerFlow{<:ACPowerFlowSolverType},
    time_step::Int64;
    kwargs...,
)
    return _solve_with_q_limits!(pf, data, time_step; kwargs...)
end

function _ac_power_flow(
    cd::ControlledDeviceSet,
    data::ACPowerFlowData,
    pf::AbstractACPowerFlow{<:ACPowerFlowSolverType},
    time_step::Int64;
    kwargs...,
)
    isempty(cd) && return _solve_with_q_limits!(pf, data, time_step; kwargs...)
    return _control_continuation!(pf, data, time_step; kwargs...)
end

"""
    _ac_power_flow_with_area_relax!(data, pf, time_step; kwargs...) -> Bool

Wraps `_ac_power_flow` with greedy-relax handling for embedded area net-interchange
control: on non-convergence with areas still enrolled, de-enroll the worst-`|r_a|` area
and re-solve, warm-started with surviving areas' `ΔP_a` re-seeded from the `delta_p`
mirror; repeat until convergence or exhaustion. Exhaustion while still failing is genuine
network non-convergence plus a terminal diagnostic (`_report_area_interchange_failure`).
Relaxation is never silent: an `@error` at each de-enrollment and a solve-end summary;
converging after a relax still returns `true`.

Resets to the full pristine enrollment before each time step's attempt
(`_ensure_pristine_area_set!`) so a previous step's relax never carries over. The
never-enrolled short-circuit deliberately tests the PRISTINE set, not the WORKING one —
a previous time step's relax may have emptied the working set, and short-circuiting on it
would permanently disable area control for the rest of `data`'s lifetime.
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
                ni_solved = _area_net_interchange(
                    aid.pristine_ties, aid.pristine_dc_ties, tail_ix, data,
                    time_step,
                )

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
    bus_types::Tuple{Vararg{PSY.ACBusTypes.Value}} = (
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
