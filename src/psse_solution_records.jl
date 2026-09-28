# Solution records for the PSS/E v35 raw format — the system-wide solution-parameter block
# between the case identification and bus records. The v33 format has no such block; PSS/E
# keeps solution parameters in the binary save case instead.

# Records PowerFlows has no counterpart for are emitted verbatim at the format defaults.
# GAUSS: PowerFlows implements no Gauss-Seidel solver.
# ADJUST: the continuation-based discrete control has no per-record analogue of these
#         acceleration factors and pass limits.
# TYSL:   PowerFlows implements no switching-time solution.
const SOLUTION_RECORD_GAUSS_DEFAULT = "GAUSS, ITMX=100, ACCP=1.6, ACCQ=1.6, ACCM=1.0, TOL=0.0001"
const SOLUTION_RECORD_ADJUST_DEFAULT = "ADJUST, ADJTHR=0.005, ACCTAP=1.0, TAPLIM=0.05, SWVBND=100.0, MXTPSS=99, MXSWIM=10"
const SOLUTION_RECORD_TYSL_DEFAULT = "TYSL, ITMXTY=20, ACCTY=1.0, TOLTY=0.00001"

# Rating-set labels. Not solve parameters; written at the defaults and parsed only so a
# round trip does not lose them.
const SOLUTION_RECORD_RATING_LABELS = [
    (1, "RATE1 ", "RATING SET 1                    "),
    (2, "RATE2 ", "RATING SET 2                    "),
    (3, "RATE3 ", "RATING SET 3                    "),
    (4, "RATE4 ", "RATING SET 4                    "),
    (5, "RATE5 ", "RATING SET 5                    "),
    (6, "RATE6 ", "RATING SET 6                    "),
    (7, "RATE7 ", "RATING SET 7                    "),
    (8, "RATE8 ", "RATING SET 8                    "),
    (9, "RATE9 ", "RATING SET 9                    "),
    (10, "RATE10", "RATING SET 10                   "),
    (11, "RATE11", "RATING SET 11                   "),
    (12, "RATE12", "RATING SET 12                   "),
]

# Format defaults for the fields PowerFlows does map, used when no solve is attached.
const SOLUTION_RECORD_DEFAULT_THRSHZ = 0.0001
const SOLUTION_RECORD_DEFAULT_PQBRAK = 0.7
const SOLUTION_RECORD_DEFAULT_BLOWUP = 5.0
const SOLUTION_RECORD_DEFAULT_ITMXN = 20
const SOLUTION_RECORD_DEFAULT_TOLN = 0.1
const SOLUTION_RECORD_DEFAULT_DVLIM = 0.99
const SOLUTION_RECORD_DEFAULT_NDVFCT = 0.99

# Both fast-decoupled variants report the same decoupled code: the distinction between
# B′/B″ half-steps and a frozen Jacobian has no counterpart in the format.
const SOLUTION_RECORD_SOLVER_FULL_NEWTON = "FNSL"
const SOLUTION_RECORD_SOLVER_DECOUPLED = "FDNS"

"""
Render a `Float64` in plain decimal, never scientific notation: `string(1e-5)` is `"1.0e-5"`,
which would reformat records otherwise passed through unchanged. Twelve places is clean for
every value the records carry; below 1e-12 prints as `0.0`.
"""
function _decimal_string(x::Float64)
    return replace(rstrip(@sprintf("%.12f", x), '0'), r"\.$" => ".0")
end

_decimal_string(x::Integer) = string(x)

"""
    SolutionRecordValues

The subset of the solution records PowerFlows maps, in record units. Sits between
[`SolutionParameters`](@ref) and the file so the writer and the reader share one
description of what is mapped.

`toln` is a mismatch in MW/MVAr, where the PowerFlows `tol` it comes from is an ∞-norm on
the per-unit mismatch; the two differ by the system base.
"""
Base.@kwdef struct SolutionRecordValues
    # PSS/E solver method code: FNSL (full Newton) or FDNS (fast-decoupled).
    solver::String = ""
    # PSS/E zero-impedance line threshold. Round-tripped only; no PowerFlows consumer.
    thrshz::Float64 = SOLUTION_RECORD_DEFAULT_THRSHZ
    # PSS/E fast-decoupled PQ-brake parameter. Round-tripped only; no PowerFlows consumer.
    pqbrak::Float64 = SOLUTION_RECORD_DEFAULT_PQBRAK
    # Divergence (blow-up) detection threshold; a fast-decoupled step-control parameter.
    blowup::Float64 = SOLUTION_RECORD_DEFAULT_BLOWUP
    # Maximum solver iteration count.
    itmxn::Int = SOLUTION_RECORD_DEFAULT_ITMXN
    # Convergence mismatch tolerance, in MW/MVAr (see the struct docstring).
    toln::Float64 = SOLUTION_RECORD_DEFAULT_TOLN
    # Per-iteration voltage-change limit; a Newton/fast-decoupled step-control parameter.
    dvlim::Float64 = SOLUTION_RECORD_DEFAULT_DVLIM
    # Non-divergent-solution voltage-change factor; a step-control parameter.
    ndvfct::Float64 = SOLUTION_RECORD_DEFAULT_NDVFCT
    # Automatic transformer tap adjustment flag (paired with swshnt for discrete control).
    actaps::Int = 0
    # Area interchange control mode: 0 off, 1 tie lines only, 2 tie lines and loads.
    areain::Int = 0
    # Automatic phase-shifter adjustment flag; no PowerFlows counterpart, format default only.
    phshft::Int = 0
    # DC tap adjustment flag; no PowerFlows counterpart, format default only.
    dctaps::Int = 0
    # Automatic switched-shunt adjustment flag (paired with actaps for discrete control).
    swshnt::Int = 0
    # Enhanced flat-start flag.
    flatst::Int = 0
    # Reactive power limit handling: 0 applies limits, -1 ignores them.
    varlim::Int = 0
    # Fast-decoupled non-divergent-solution flag.
    nondiv::Int = 0
end

# Solvers whose iteration budget and step-control parameters the fast-decoupled fields
# describe. For every other solver those fields keep the format defaults.
_is_fast_decoupled(::Type{<:FastDecoupledACPowerFlow}) = true
_is_fast_decoupled(::Type{<:ACPowerFlowSolverType}) = false

_solver_code(::Type{<:FastDecoupledACPowerFlow}) = SOLUTION_RECORD_SOLVER_DECOUPLED
_solver_code(::Type{<:ACPowerFlowSolverType}) = SOLUTION_RECORD_SOLVER_FULL_NEWTON

_solver_type(::AbstractACPowerFlow{S}) where {S} = S

# The fast-decoupled step-control fields; every other solver reports the format defaults.
_solution_record_step_control(
    ::Type{<:FastDecoupledACPowerFlow},
    params::SolutionParameters,
) =
    (;
        blowup = params.fd_blowup,
        dvlim = params.fd_dvlim,
        ndvfct = params.fd_ndvfct,
        nondiv = Int(params.fd_non_divergent),
    )
_solution_record_step_control(::Type{<:ACPowerFlowSolverType}, ::SolutionParameters) = (;
    blowup = SOLUTION_RECORD_DEFAULT_BLOWUP,
    dvlim = SOLUTION_RECORD_DEFAULT_DVLIM,
    ndvfct = SOLUTION_RECORD_DEFAULT_NDVFCT,
    nondiv = 0,
)

"""
    solution_record_values(pf, params, base_power) -> SolutionRecordValues

Map a solve onto the record fields. `pf` supplies the solver identity and `params` the
parameters it ran with — the two are separate because a parameter may be overridden at the
call site rather than stored on the model. `base_power` is the system base in MVA and
converts the per-unit convergence tolerance into the record's MW/MVAr mismatch.

DC models carry no solve parameters, so they map to the format defaults.
"""
function solution_record_values(
    pf::AbstractACPowerFlow,
    params::SolutionParameters,
    base_power::Float64,
)
    solver = _solver_type(pf)
    step_control = _solution_record_step_control(solver, params)

    # A discrete-control solve moves both tap changers and switched shunts; PowerFlows has
    # one flag where the format has two.
    if params.control_discrete_devices
        tap_and_shunt = 1
    else
        tap_and_shunt = 0
    end

    area = if !params.area_interchange_control
        0
    elseif params.tie_definition === :lines_and_loads
        2
    else
        1
    end

    # 0 applies the limits, -1 ignores them. PowerFlows enforces them between solves, so
    # there is no counterpart to the "apply after n iterations" form.
    if params.check_reactive_power_limits
        varlim = 0
    else
        varlim = -1
    end

    # `params.maxIterations` is already resolved to the solver's default by the model
    # constructor (see `_default_max_iterations`), so no branch is needed here.
    iterations = params.maxIterations

    if params.enhanced_flat_start
        flatst = 1
    else
        flatst = 0
    end

    return SolutionRecordValues(;
        solver = _solver_code(solver),
        itmxn = iterations,
        # Rounded: the per-unit-to-MW conversion leaves float noise (1e-7*100 =
        # 9.999999999999999e-6), and no solver tolerance is meaningful past twelve digits.
        toln = round(params.tol * base_power; sigdigits = 12),
        actaps = tap_and_shunt,
        swshnt = tap_and_shunt,
        areain = area,
        varlim = varlim,
        flatst = flatst,
        # No PowerFlows counterpart: there is no phase-shift or DC-tap control to report.
        phshft = 0,
        dctaps = 0,
        step_control...,
    )
end

solution_record_values(::PowerFlowEvaluationModel, ::SolutionParameters, ::Float64) =
    SolutionRecordValues()

"""
    write_solution_records(io, pf, params, base_power)

Write the v35 solution-record block for a solve, or the format defaults when `pf` is
`nothing` — an export with no solve attached must be byte-identical to one produced before
this block carried any solve information.
"""
function write_solution_records(io::IO, pf, params, base_power::Float64)
    v = if isnothing(pf) || isnothing(params)
        SolutionRecordValues()
    else
        solution_record_values(pf, params, base_power)
    end

    println(
        io,
        "GENERAL, THRSHZ=", _decimal_string(v.thrshz),
        ", PQBRAK=", _decimal_string(v.pqbrak),
        ", BLOWUP=", _decimal_string(v.blowup),
        ", MaxIsolLvls=4, CAMaxReptSln=20, ChkDupCntLbl=0",
    )
    println(io, SOLUTION_RECORD_GAUSS_DEFAULT)
    println(
        io,
        "NEWTON, ITMXN=", v.itmxn,
        ", ACCN=1.0, TOLN=", _decimal_string(v.toln),
        ", VCTOLQ=0.1, VCTOLV=0.00001",
        ", DVLIM=", _decimal_string(v.dvlim),
        ", NDVFCT=", _decimal_string(v.ndvfct),
    )
    println(io, SOLUTION_RECORD_ADJUST_DEFAULT)
    println(io, SOLUTION_RECORD_TYSL_DEFAULT)
    # The method name is positional and blank-able; right-aligning to five chars renders
    # an empty name as the five spaces the format defaults use.
    println(
        io,
        "SOLVER,", lpad(v.solver, 5),
        ", ACTAPS=", v.actaps,
        ", AREAIN=", v.areain,
        ", PHSHFT=", v.phshft,
        ", DCTAPS=", v.dctaps,
        ", SWSHNT=", v.swshnt,
        ", FLATST=", v.flatst,
        ", VARLIM=", v.varlim,
        ", NONDIV=", v.nondiv,
    )
    for (index, short_label, long_label) in SOLUTION_RECORD_RATING_LABELS
        println(
            io,
            "RATING,",
            lpad(index, 2),
            ", \"",
            short_label,
            "\", \"",
            long_label,
            "\"",
        )
    end
    return
end

# ---------------------------------------------------------------------------------------
# Reading solution records back out of a case file
# ---------------------------------------------------------------------------------------

"""
Split a record on commas that sit outside quotes. Rating labels are quoted and may contain
a comma, so a plain `split` would tear them apart.
"""
function _split_record(line::AbstractString)
    fields = String[]
    current = IOBuffer()
    in_quotes = false
    for c in line
        if c == '"' || c == '\''
            in_quotes = !in_quotes
            print(current, c)
        elseif c == ',' && !in_quotes
            push!(fields, String(take!(current)))
        else
            print(current, c)
        end
    end
    push!(fields, String(take!(current)))
    return fields
end

# `NAME=VALUE` assignments in one record, keyed case-insensitively: the format is not
# consistent about case (`MaxIsolLvls` beside `THRSHZ`).
function _record_assignments(fields)
    out = Dict{String, String}()
    for field in fields
        parts = split(field, '='; limit = 2)
        length(parts) == 2 || continue
        out[uppercase(strip(parts[1]))] = strip(parts[2])
    end
    return out
end

_record_int(d, key, fallback) = something(tryparse(Int, get(d, key, "")), fallback)
_record_float(d, key, fallback) = something(tryparse(Float64, get(d, key, "")), fallback)

# Section terminators customarily carry a trailing `/` comment, e.g. `0 / END OF
# SYSTEM-WIDE DATA, BEGIN BUS DATA` — stripped here.
function _leading_token(line::AbstractString)
    field = first(_split_record(line))
    return strip(first(split(field, '/')))
end

# Terminator `0` ends the block normally; any other leading integer means bus data started
# and the block was absent — treated as the end too, not read as solution records.
function _ends_solution_block(line::AbstractString)
    token = _leading_token(line)
    isempty(token) && return false
    return !isnothing(tryparse(Int, token))
end

# Records, with the `@!` column-header comments and blank lines dropped. Used only for the
# solution-record block: the two title records ahead of it are skipped by fixed position
# (see `_case_header_and_block`), not by this filter, so a non-blank second title line is
# never mistaken for the block's first record.
function _significant_records(lines)
    return [
        line for line in map(strip, lines)
        if !isempty(line) && !startswith(line, "@!")
    ]
end

# The case identification record and the significant records of the block that follows the
# two title records, or `nothing` when the file has no case identification record at all.
function _case_header_and_block(lines::Vector{<:AbstractString})
    header_ix = findfirst(
        line -> !isempty(strip(line)) && !startswith(strip(line), "@!"),
        lines,
    )
    isnothing(header_ix) && return nothing
    block_start = header_ix + 3
    block_start <= length(lines) + 1 || return nothing
    return (strip(lines[header_ix]), _significant_records(lines[block_start:end]))
end

"""
    read_solution_records(path) -> Union{Nothing, SolutionRecordValues}

The mapped solution-record fields of a case file, or `nothing` when the file carries none
— a format revision below 35, or a revision 35 file whose block is absent. `nothing`
rather than defaults: defaults would be indistinguishable from parsed values.

Unrecognized records and unrecognized field names are ignored, so a case written by a
newer PSS/E than this mapping knows about still reads.
"""
function read_solution_records(path::AbstractString)
    header_and_block = _case_header_and_block(readlines(path))
    isnothing(header_and_block) && return nothing
    return _read_solution_records(header_and_block...)
end

# Core of `read_solution_records`, taking the header record and the already-split
# significant block records so a caller that also needs another field of the file (e.g. the
# base power) can read it once.
function _read_solution_records(header::AbstractString, block::Vector{<:AbstractString})
    fields = _split_record(header)
    length(fields) >= 3 || return nothing
    revision = tryparse(Int, strip(first(split(strip(fields[3]), '/'))))
    (isnothing(revision) || revision < 35) && return nothing

    values = SolutionRecordValues()
    found = false
    for stripped in block
        _ends_solution_block(stripped) && break

        fields = _split_record(stripped)
        keyword = uppercase(strip(fields[1]))
        assignments = _record_assignments(fields[2:end])

        if keyword == "GENERAL"
            found = true
            values = SolutionRecordValues(
                values;
                thrshz = _record_float(assignments, "THRSHZ", values.thrshz),
                pqbrak = _record_float(assignments, "PQBRAK", values.pqbrak),
                blowup = _record_float(assignments, "BLOWUP", values.blowup),
            )
        elseif keyword == "NEWTON"
            found = true
            values = SolutionRecordValues(
                values;
                itmxn = _record_int(assignments, "ITMXN", values.itmxn),
                toln = _record_float(assignments, "TOLN", values.toln),
                dvlim = _record_float(assignments, "DVLIM", values.dvlim),
                ndvfct = _record_float(assignments, "NDVFCT", values.ndvfct),
            )
        elseif keyword == "SOLVER"
            found = true
            # The first field after the keyword is the positional method name.
            if length(fields) >= 2
                name = strip(fields[2])
            else
                name = ""
            end
            occursin('=', name) && (name = "")
            values = SolutionRecordValues(
                values;
                solver = String(name),
                actaps = _record_int(assignments, "ACTAPS", values.actaps),
                areain = _record_int(assignments, "AREAIN", values.areain),
                phshft = _record_int(assignments, "PHSHFT", values.phshft),
                dctaps = _record_int(assignments, "DCTAPS", values.dctaps),
                swshnt = _record_int(assignments, "SWSHNT", values.swshnt),
                flatst = _record_int(assignments, "FLATST", values.flatst),
                varlim = _record_int(assignments, "VARLIM", values.varlim),
                nondiv = _record_int(assignments, "NONDIV", values.nondiv),
            )
        elseif keyword in ("GAUSS", "ADJUST", "TYSL", "RATING")
            found = true  # recognized, but nothing here maps onto PowerFlows
        else
            @debug "Ignoring unrecognized solution record: $keyword"
        end
    end
    if found
        return values
    else
        return nothing
    end
end

# Copy-with-overrides, so each record's parse only has to name the fields it sets.
SolutionRecordValues(base::SolutionRecordValues; kwargs...) = _override(base; kwargs...)

"""
    solution_parameters(values::SolutionRecordValues, base_power) -> SolutionParameters

Map solution records back onto PowerFlows parameters. `base_power` is the system base in
MVA and converts the record's MW/MVAr mismatch back to a per-unit tolerance.

Record fields with no PowerFlows counterpart are dropped; PowerFlows parameters the format
cannot express keep their defaults. This includes `enhanced_flat_start`: PSS/E's FLATST
means "always flat start," while PowerFlows' flag is a fallback used only when the starting
residual is large (default `true`), so the two are not the same setting — `values.flatst`
is parsed but never applied, and `enhanced_flat_start` keeps `SolutionParameters`' own
default.
"""
function solution_parameters(values::SolutionRecordValues, base_power::Float64)
    fd = uppercase(values.solver) == SOLUTION_RECORD_SOLVER_DECOUPLED

    # A zero mismatch target is not a tolerance anyone can converge to; fall back rather
    # than hand a solver `tol = 0`.
    if values.toln > 0
        tol = values.toln / base_power
    else
        tol = DEFAULT_NR_TOL
    end

    if values.itmxn > 0
        maxIterations = values.itmxn
    else
        maxIterations = UNSET_MAX_ITERATIONS
    end

    # Tie-line-and-load interchange is not implemented and the model constructor rejects
    # it, so a case using it is read as the tie-line form rather than failing to load.
    tie_definition = :lines_only
    if values.areain == 2
        @warn(
            "Area interchange over tie lines and loads is not implemented; reading the " *
            "case as tie lines only.",
            maxlog = 1,
        )
    end

    if fd
        fd_step_control = (;
            fd_blowup = values.blowup,
            fd_dvlim = values.dvlim,
            fd_ndvfct = values.ndvfct,
            fd_non_divergent = values.nondiv == 1,
        )
    else
        fd_step_control = (;
            fd_blowup = DEFAULT_FD_BLOWUP,
            fd_dvlim = DEFAULT_FD_DVLIM,
            fd_ndvfct = DEFAULT_FD_NDVFCT,
            fd_non_divergent = DEFAULT_FD_NON_DIVERGENT,
        )
    end

    control_discrete_devices = values.actaps != 0 || values.swshnt != 0
    # Discrete control is a Newton/trust-region continuation; the model constructor
    # rejects it under fast-decoupled solving, so a case that requests both is read with
    # the control flag dropped rather than producing a `SolutionParameters` that fails to
    # build a model.
    if fd && control_discrete_devices
        @warn(
            "The solution records request discrete-device control (ACTAPS/SWSHNT) under " *
            "fast-decoupled solving (FDNS); PowerFlows does not support discrete control " *
            "as a fast-decoupled continuation, so it is being dropped.",
            maxlog = 1,
        )
        control_discrete_devices = false
    end

    return SolutionParameters(;
        tol = tol,
        maxIterations = maxIterations,
        check_reactive_power_limits = values.varlim >= 0,
        control_discrete_devices = control_discrete_devices,
        area_interchange_control = values.areain != 0,
        tie_definition = tie_definition,
        fd_step_control...,
    )
end

"""
    read_solution_parameters(path; base_power = nothing) -> Union{Nothing, SolutionParameters}

The solve parameters a PSS/E case file was distributed with, ready to hand to an
evaluation model:

```julia
params = read_solution_parameters("case.raw")
pf = ACPolarPowerFlow{NewtonRaphsonACPowerFlow}(; solution_parameters = params)
```

Returns `nothing` when the file carries no solution parameters — a v33 raw keeps them in
the binary save case instead, and a v35 raw may omit the block. Callers that want defaults
in that case can write `something(read_solution_parameters(path), SolutionParameters())`.

`base_power` is the system base in MVA, used to convert the file's MW/MVAr mismatch target
into a per-unit tolerance. It is read from the case identification record when not given.

Parameters the format cannot express — the formulation, the linear-solver backend, slack
distribution, refinement settings — keep their defaults.
"""
function read_solution_parameters(
    path::AbstractString;
    base_power::Union{Nothing, Real} = nothing,
)
    header_and_block = _case_header_and_block(readlines(path))
    isnothing(header_and_block) && return nothing
    header, block = header_and_block
    values = _read_solution_records(header, block)
    isnothing(values) && return nothing
    if isnothing(base_power)
        sbase = _case_base_power(header)
    else
        sbase = Float64(base_power)
    end
    return solution_parameters(values, sbase)
end

# Field 2 of the case identification record is the system base in MVA. Reached only once a
# solution-record block has been found, so the header is a case identification record by
# construction; a missing or unparseable SBASE field means the file is malformed, not that
# 100 MVA is a safe guess.
function _case_base_power(header::AbstractString)
    fields = _split_record(header)
    length(fields) >= 2 ||
        error("Case identification record has no SBASE field: \"$header\"")
    sbase = tryparse(Float64, strip(fields[2]))
    isnothing(sbase) &&
        error("Case identification record SBASE field is not a number: \"$header\"")
    return sbase
end
