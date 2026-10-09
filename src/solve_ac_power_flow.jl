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

The number of tasks solving contiguous chunks of `time_steps` concurrently is the stored
`n_threads` of the model's [`SolutionParameters`](@ref).

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
    # The solvers swallow unknown keywords, so a per-call thread count would run serially
    # unnoticed.
    for k in (:threads, :n_threads)
        haskey(kwargs, k) && throw(
            ArgumentError(
                "solve_power_flow! takes no `$k` keyword; set the worker count with " *
                "`SolutionParameters(; n_threads)` on the power flow model.",
            ),
        )
    end
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
    n_work = min(get_n_threads(pf), length(sorted_time_steps))
    if n_work > 1
        _solve_columns_threaded!(
            ts_converged, data, pf, sorted_time_steps, n_work, merged_kwargs)
    else
        _solve_columns!(
            ts_converged, data, pf, sorted_time_steps, 1:length(sorted_time_steps),
            merged_kwargs)
    end

    # Written after the solves: `improve_x0` reads `converged` as of entry.
    data.converged[sorted_time_steps] .= ts_converged

    if !all(ts_converged)
        failed = sorted_time_steps[.!ts_converged]
        @error "AC power flow did not converge in $(length(failed)) of $(length(ts_converged)) time step(s): $failed"
    end

    return all(ts_converged)
end

function _solve_columns!(
    ts_converged::Vector{Bool},
    data::ACPowerFlowData,
    pf::AbstractACPowerFlow{<:ACPowerFlowSolverType},
    steps::AbstractVector{Int},
    positions::UnitRange{Int},
    merged_kwargs::NamedTuple,
)
    flows = Base.RefValue{ArcFlowScratch}()
    cd = get_controlled_devices(data)
    for pos in positions
        ts_converged[pos] = _solve_column!(data, pf, steps[pos], flows, cd, merged_kwargs)
    end
    @debug "lean LU refactors on this cache so far" lean = _lean_counts(data)
    return ts_converged
end

# Fetched after the solve, so a first solve's fresh polar cache lends its scratch instead of a
# second one being built.
function _column_arc_flows!(slot::Base.RefValue, data::ACPowerFlowData)
    arcs = get_arc_axis(data)
    if !isassigned(slot) || slot[].arcs !== arcs
        slot[] = _arc_flow_scratch(data.polar_nr_cache[], data)
    end
    return slot[]
end

"""Solve one time step and write its column: voltages, injections, branch and LCC flows.
Touches no other column of `data`."""
function _solve_column!(
    data::ACPowerFlowData,
    pf::AbstractACPowerFlow{<:ACPowerFlowSolverType},
    time_step::Int,
    flows_slot::Base.RefValue,
    cd::Union{Nothing, ControlledDeviceSet},
    merged_kwargs::NamedTuple,
)
    Yft = data.power_network_matrix.arc_admittance_from_to
    Ytf = data.power_network_matrix.arc_admittance_to_from

    load_device_state!(cd, data, time_step)
    data.iterations[time_step] = 0
    # Before the solve, so serial and threaded runs build the lean plan at the same step.
    _prepare_lean_plan!(pf, data, time_step,
        resolve_linear_solver_backend(get(merged_kwargs, :linear_solver, nothing)))
    converged = _ac_power_flow_with_area_relax!(data, pf, time_step; merged_kwargs...)
    save_device_state!(cd, data, time_step)
    converged && _warn_vsc_limit_violations(data, time_step)

    if OVERWRITE_NON_CONVERGED && !converged
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

    flows = _column_arc_flows!(flows_slot, data)
    (; fb_ix, tb_ix, Sft, Stf) = flows
    step_V = flows.V
    # PNM's structs use ComplexF32 and the System stores Float64. Flows computed again from the
    # System voltages differ from these flows by approximately 1e-4.
    _fill_flow_voltages!(step_V, data.polar_nr_cache[], data, time_step)
    mul!(Sft, Yft.data, step_V)
    mul!(Stf, Ytf.data, step_V)
    θ = view(data.bus_angles, :, time_step)
    @inbounds for k in eachindex(fb_ix, tb_ix)
        f = fb_ix[k]
        t = tb_ix[k]
        s_ft = step_V[f] * conj(Sft[k])
        s_tf = step_V[t] * conj(Stf[k])
        data.arc_active_power_flow_from_to[k, time_step] = real(s_ft)
        data.arc_reactive_power_flow_from_to[k, time_step] = imag(s_ft)
        data.arc_active_power_flow_to_from[k, time_step] = real(s_tf)
        data.arc_reactive_power_flow_to_from[k, time_step] = imag(s_tf)
        data.arc_angle_differences[k, time_step] = θ[f] - θ[t]
    end
    return converged
end

"""Solve contiguous chunks of `steps` on `n_work` tasks. Each task gets a `_column_worker`
view of `data`, so its Newton workspace and KLU factorization are its own. The workers' caches
live in `data.worker_slots`, so a repeated solve reuses them."""
function _solve_columns_threaded!(
    ts_converged::Vector{Bool},
    data::ACPowerFlowData,
    pf::AbstractACPowerFlow{<:ACPowerFlowSolverType},
    steps::AbstractVector{Int},
    n_work::Int,
    merged_kwargs::NamedTuple,
)
    allunique(steps) || throw(
        ArgumentError("time_steps must be unique when n_threads > 1, got $steps."),
    )
    backend = _check_threadable(merged_kwargs)
    # Built here, before any task can race to build it, so every worker shares one pivot order.
    _prepare_lean_plan!(pf, data, first(steps), backend)
    chunks = collect(Iterators.partition(1:length(steps), cld(length(steps), n_work)))
    slots = _worker_slots!(data, length(chunks))
    workers = [_column_worker(data, steps, c, s) for (c, s) in zip(chunks, slots)]
    # With no stored cache, the first chunk's first step solves before any task starts, so its
    # cache can seed the other workers: they share its read-only maps instead of building their own.
    head = _head_steps(slots[1].polar_nr_cache[], chunks[1])
    _solve_slot!(ts_converged, workers[1], slots[1], pf, steps, head, merged_kwargs)
    _seed_workers!(workers, steps, chunks)
    @sync for (i, positions) in enumerate(chunks)
        worker = workers[i]
        slot = slots[i]
        rest = positions
        if isone(i)
            rest = (last(head) + 1):last(positions)
        end
        Threads.@spawn _solve_slot!(
            ts_converged, worker, slot, pf, steps, $rest, merged_kwargs)
    end
    for (worker, positions) in zip(workers, chunks)
        _merge_worker_area!(data, worker, steps, positions)
    end
    return ts_converged
end

function _worker_slots!(data::ACPowerFlowData, n::Int)
    slots = data.worker_slots
    while length(slots) < n
        push!(slots, WorkerSlot())
    end
    return view(slots, 1:n)
end

# A worker that raised an error leaves its caches in an unknown state.
# Drop them so that the next call rebuilds them.
function _solve_slot!(
    ts_converged::Vector{Bool},
    worker::ACPowerFlowData,
    slot::WorkerSlot,
    pf::AbstractACPowerFlow,
    steps::AbstractVector{Int},
    positions::UnitRange{Int},
    kwargs::NamedTuple,
)
    try
        _solve_columns!(ts_converged, worker, pf, steps, positions, kwargs)
    catch
        slot.polar_nr_cache[] = nothing
        slot.solver_cache[] = nothing
        rethrow()
    end
    return
end

_head_steps(::Nothing, positions::UnitRange{Int}) = first(positions):first(positions)
_head_steps(::AbstractNRCache, positions::UnitRange{Int}) =
    first(positions):(first(positions) - 1)

function _seed_workers!(
    workers::Vector{<:ACPowerFlowData},
    steps::AbstractVector{Int},
    chunks::Vector{UnitRange{Int}},
)
    seed = workers[1].polar_nr_cache[]
    memo = workers[1].ac_jacobian_structure_cache[]
    for i in 2:length(workers)
        worker = workers[i]
        _seed_worker!(worker, worker.polar_nr_cache[], seed, memo, steps[first(chunks[i])])
    end
    return
end

_prepare_lean_plan!(
    ::AbstractACPowerFlow,
    ::ACPowerFlowData,
    ::Int,
    ::PNM.LinearSolverType,
) =
    nothing

function _prepare_lean_plan!(
    ::ACPolarPowerFlow{<:Union{NewtonRaphsonACPowerFlow, TrustRegionACPowerFlow}},
    data::ACPowerFlowData,
    time_step::Int,
    ::PNM.KLUSolver,
)
    _USE_LEAN_LU[] || return
    _lean_plan_tried(data.ac_jacobian_structure_cache[], data) && return
    _lean_plan_slot!(data, time_step)
    return
end

# Skips the residual and Jacobian builds of `_lean_plan_slot!` while `data` keeps its memo. A memo
# whose slack slots no longer fit is replaced by the solve, which then builds its plan itself.
_lean_plan_tried(::Nothing, ::ACPowerFlowData) = false
function _lean_plan_tried(memo::ACJacobianStructureCache, data::ACPowerFlowData)
    return memo.lean.tried && memo.matrix === data.power_network_matrix &&
           memo.area_data === data.area_interchange
end

"""A `PowerFlowData` sharing every time-indexed array of `data` (each task writes only its own
columns) with the slot's solver caches, a private `converged` (steps owned by other tasks are
cleared, so `improve_x0` never warm-starts from them), and private copies of the state a solve
mutates outside its column: the controlled-device scratch and counters, the network matrix when
taps are controlled, the LCC branch admittances, and the area-interchange data when it is active.
The read-only Jacobian-structure memo, which carries the lean-LU plan, is shared; a worker with its
own network matrix or area data misses it and plans itself. See [`SolutionParameters`](@ref)'s
`n_threads` for how results compare to a serial solve."""
function _column_worker(
    data::D,
    steps::AbstractVector{Int},
    positions::UnitRange{Int},
    slot::WorkerSlot,
) where {D <: ACPowerFlowData}
    converged = copy(data.converged)
    for (pos, t) in enumerate(steps)
        if !(pos in positions)
            converged[t] = false
        end
    end
    cd = get_controlled_devices(data)
    # A worker with active areas gets a private area set that a relax shrinks. The slot's caches
    # may be sized for a previous call's shrunk set, so they cannot be reused.
    if !isempty(data.area_interchange.pristine_areas)
        slot.polar_nr_cache[] = nothing
        slot.solver_cache[] = nothing
    end
    fresh = (
        converged = converged,
        power_network_matrix = _worker_network_matrix(cd, data.power_network_matrix),
        lcc = _override(data.lcc; branch_admittances = copy(data.lcc.branch_admittances)),
        area_interchange = _worker_area_data(data.area_interchange),
        controlled_devices = _worker_devices(cd),
        solver_cache = slot.solver_cache,
        ac_jacobian_structure_cache = Base.RefValue{
            Union{Nothing, ACJacobianStructureCache},
        }(
            data.ac_jacobian_structure_cache[],
        ),
        polar_nr_cache = slot.polar_nr_cache,
        worker_slots = WorkerSlot[],
    )
    args = map(f -> get(fresh, f, getfield(data, f)), fieldnames(D))
    return D(args...)
end

_worker_devices(::Nothing) = nothing
# The per-step stores stay shared: each step reads and writes only its own column.
function _worker_devices(cd::ControlledDeviceSet)
    return _override(cd;
        taps = deepcopy(cd.taps),
        shunts = deepcopy(cd.shunts),
        facts = deepcopy(cd.facts),
        inner_solves = Ref(0),
        symbolic_factors = Ref(0),
        numeric_refactors = Ref(0),
    )
end

# Controlled taps write only the Y-bus, `Yft` and `Ytf` values. Axes, lookups and the branch
# catalog stay shared: the catalog reaches the System's components, so a deepcopy would copy it.
_worker_network_matrix(::Nothing, matrix) = matrix
function _worker_network_matrix(cd::ControlledDeviceSet, matrix)
    if isempty(cd.taps)
        return matrix
    end
    return _override(matrix;
        data = copy(matrix.data),
        arc_admittance_from_to = _copy_values(matrix.arc_admittance_from_to),
        arc_admittance_to_from = _copy_values(matrix.arc_admittance_to_from),
    )
end

_copy_values(::Nothing) = nothing
_copy_values(m::PNM.ArcAdmittanceMatrix) = _override(m; data = copy(m.data))

# A relax renumbers the working area set across all columns. Shared while inactive, so the
# workers keep hitting the memo, which keys on this object's identity.
function _worker_area_data(aid::AreaInterchangeData)
    if isempty(aid.pristine_areas)
        return aid
    end
    return deepcopy(aid)
end

"""Copy what post-processing reads for the worker's own steps back into `data`: the
`pristine_delta_p` columns and the `relaxed` records."""
function _merge_worker_area!(
    data::ACPowerFlowData,
    worker::ACPowerFlowData,
    steps::AbstractVector{Int},
    positions::UnitRange{Int},
)
    aid = data.area_interchange
    isempty(aid.pristine_areas) && return
    worker_aid = worker.area_interchange
    for pos in positions
        t = steps[pos]
        @views aid.pristine_delta_p[:, t] .= worker_aid.pristine_delta_p[:, t]
        if haskey(worker_aid.relaxed, t)
            aid.relaxed[t] = worker_aid.relaxed[t]
        end
    end
    return
end

# Resolved again here: a per-call `linear_solver` overrides the one checked at construction.
function _check_threadable(merged_kwargs::NamedTuple)
    backend = resolve_linear_solver_backend(get(merged_kwargs, :linear_solver, nothing))
    _check_concurrent_factorization(backend)
    return backend
end

_check_concurrent_factorization(::PNM.KLUSolver) = nothing
function _check_concurrent_factorization(backend::PNM.LinearSolverType)
    throw(
        ArgumentError(
            "n_threads > 1 requires the KLU linear solver; $(nameof(typeof(backend))) is not " *
            "verified safe under concurrent factorization. Pass linear_solver = \"KLU\" or " *
            "n_threads = 1.",
        ),
    )
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
