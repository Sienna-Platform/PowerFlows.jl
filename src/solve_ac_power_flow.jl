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
- `maxIterations`: Maximum number of Newton-Raphson iterations. Default is
  `$DEFAULT_NR_MAX_ITER`.

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
- `threads::Int = 1`: number of tasks solving contiguous chunks of `time_steps` concurrently, each
    with its own Newton workspace and KLU factorization. A first solve gives the same result for
    any `threads`; on a re-solve a chunk's first step may warm-start differently, because steps
    owned by other tasks are not used as warm starts. Errors for discrete device control, area
    interchange control, LCC HVDC lines, or a non-KLU linear solver. `solve_power_flow` and
    `solve_and_store_power_flow!` pass it through.

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
    threads::Int = 1,
    kwargs...,
)
    pf = get_pf(data)
    merged_kwargs = merge(get_solver_kwargs(pf), NamedTuple(kwargs))
    merged_kwargs.maxIterations < 1 && error(
        "maxIterations must be >= 1, got $(merged_kwargs.maxIterations) for $(typeof(pf)).",
    )
    threads < 1 && error("threads must be >= 1, got $threads.")
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
    ts_converged = fill(false, length(sorted_time_steps))
    validate_device_store_width(get_controlled_devices(data), get_time_steps(data))
    n_work = min(threads, length(sorted_time_steps))
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
    (; attempts, rejects, solve_failures, late_analyses) = _lean_counts(data)
    @debug "lean LU refactors on this cache so far" attempts rejects solve_failures late_analyses
    return ts_converged
end

# Fetched after the solve, so a first solve's fresh polar cache lends its scratch instead of a
# second one being built.
function _column_arc_flows!(slot::Base.RefValue, data::ACPowerFlowData)
    arcs = PNM.get_arc_axis(data.power_network_matrix.arc_admittance_from_to)
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
        for (i, (bus_indices, self_admittances)) in
            enumerate(zip(data.lcc.bus_indices, data.lcc.branch_admittances))
            (rectifier_ix, inverter_ix) = bus_indices
            (rectifier_y, inverter_y) = self_admittances
            V_inverter = _bus_voltage_phasor(data, inverter_ix, time_step)
            V_rectifier = _bus_voltage_phasor(data, rectifier_ix, time_step)
            S_inverter = V_inverter * conj(inverter_y * V_inverter)
            S_rectifier = V_rectifier * conj(rectifier_y * V_rectifier)
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
    # Per-step branch flows so a future per-step Yft/Ytf (e.g. varying tap positions) is used
    # correctly.
    # NOTE PNM's structs use ComplexF32, while the system objects store Float64's.
    #      so if you set the system bus angles/voltages to match these fields, then repeat
    #      this math using the system voltages, you'll see differences in the flows, ~1e-4.
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
view of `data`, so its Newton workspace and KLU factorization are its own."""
function _solve_columns_threaded!(
    ts_converged::Vector{Bool},
    data::ACPowerFlowData,
    pf::AbstractACPowerFlow{<:ACPowerFlowSolverType},
    steps::AbstractVector{Int},
    n_work::Int,
    merged_kwargs::NamedTuple,
)
    backend = _check_threadable(data, merged_kwargs)
    # Built here, before any task can race to build it, so every worker shares one pivot order.
    _prepare_lean_plan!(pf, data, first(steps), backend)
    chunks = Iterators.partition(1:length(steps), cld(length(steps), n_work))
    @sync for positions in chunks
        worker = _column_worker(data, steps, positions)
        Threads.@spawn _solve_columns!(
            ts_converged, worker, pf, steps, positions, merged_kwargs)
    end
    return ts_converged
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

"""A `PowerFlowData` sharing every array of `data` (each task writes only its own columns) with
fresh solver caches and a private `converged`. The Jacobian-structure memo is shared too: it is
read-only once built, and it carries the lean-LU plan. `improve_x0` warm-starts from the last step
converged at entry; steps owned by other tasks are cleared there, since their columns are being
rewritten concurrently. So a first solve matches the serial one exactly, while a re-solve may pick
a different warm start at a chunk's first step."""
function _column_worker(
    data::ACPowerFlowData,
    steps::AbstractVector{Int},
    positions::UnitRange{Int},
)
    converged = copy(data.converged)
    for (pos, t) in enumerate(steps)
        if !(pos in positions)
            converged[t] = false
        end
    end
    fresh = (
        converged = converged,
        solver_cache = Base.RefValue{Union{Nothing, SolverCache}}(nothing),
        ac_jacobian_structure_cache = Base.RefValue{
            Union{Nothing, ACJacobianStructureCache},
        }(
            data.ac_jacobian_structure_cache[],
        ),
        polar_nr_cache = Base.RefValue{Union{Nothing, AbstractNRCache}}(nothing),
    )
    args = map(f -> get(fresh, f, getfield(data, f)), fieldnames(typeof(data)))
    return typeof(data)(args...)
end

# Anything holding per-solve state outside the time-step columns cannot be split across tasks.
function _check_threadable(data::ACPowerFlowData, merged_kwargs::NamedTuple)
    _check_threadable(get_controlled_devices(data))
    isempty(data.area_interchange.pristine_areas) || error(
        "threads > 1 is not supported with area interchange control: the enrolled-area set " *
        "is shared across time steps. Solve with threads = 1.",
    )
    iszero(get_lcc_count(data)) || error(
        "threads > 1 is not supported with LCC HVDC lines: their branch admittances are " *
        "shared across time steps. Solve with threads = 1.",
    )
    backend = resolve_linear_solver_backend(
        get(merged_kwargs, :linear_solver, nothing))
    _concurrent_factorization_safe(backend) || error(
        "threads > 1 requires the KLU linear solver; $(nameof(typeof(backend))) is not " *
        "verified safe under concurrent factorization. Pass linear_solver = \"KLU\" or " *
        "threads = 1.",
    )
    return backend
end

_check_threadable(::Nothing) = nothing

function _check_threadable(cd::ControlledDeviceSet)
    isempty(cd) || error(
        "threads > 1 is not supported with discrete device control: tap, shunt and FACTS " *
        "settings carry across time steps. Solve with threads = 1.",
    )
    return
end

_concurrent_factorization_safe(::PNM.LinearSolverType) = false
_concurrent_factorization_safe(::PNM.KLUSolver) = true

_bus_voltage_phasor(data::ACPowerFlowData, ix::Int, time_step::Int) =
    data.bus_magnitude[ix, time_step] * exp(1im * data.bus_angles[ix, time_step])

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
            @debug "Bus $(bus_names[ix]) changed to PSY.ACBusTypes.PQ"
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
