# Solution records for the PSS/E v35 raw format: the block of system-wide solution
# parameters that sits between the case identification records and the bus records.
#
# The format is a vendor file format, not an interface: nothing outside this file speaks
# it. `SolutionParameters` is the PowerFlows-side type, and the two mappings here are the
# only place the record field names appear.
#
# The v33 raw format has no such block — PSS/E keeps solution parameters in the binary
# save case instead — so everything here is v35-only.

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

# Solver method codes. Both fast-decoupled variants report the decoupled code: the
# distinction between them (B′/B″ half-steps versus a frozen Jacobian) has no counterpart
# in the format, and claiming one would assert a mapping the format does not define.
const SOLUTION_RECORD_SOLVER_FULL_NEWTON = "FNSL"
const SOLUTION_RECORD_SOLVER_DECOUPLED = "FDNS"

"""
Render a `Float64` without scientific notation, so the emitted records stay in the plain
decimal form the format uses. `string(1e-5)` is `"1.0e-5"`, which would silently reformat
records that are otherwise passed through unchanged.

Falls back to `string` for exponents large enough that a decimal expansion would be
absurd; the format accepts scientific notation there.
"""
function _decimal_string(x::Float64)
    s = string(x)
    m = match(r"^(-?)(\d+)\.(\d+)e(-?\d+)$", s)
    isnothing(m) && return s
    sign, int_part, frac, expo = m[1], m[2], m[3], parse(Int, m[4])
    abs(expo) > 20 && return s
    frac = rstrip(frac, '0')
    digits = int_part * frac
    point = length(int_part) + expo  # index in `digits` the decimal point follows
    if point <= 0
        return sign * "0." * "0"^(-point) * digits
    elseif point >= length(digits)
        return sign * digits * "0"^(point - length(digits)) * ".0"
    else
        return sign * digits[1:point] * "." * digits[(point + 1):end]
    end
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
    solver::String = ""
    thrshz::Float64 = SOLUTION_RECORD_DEFAULT_THRSHZ
    pqbrak::Float64 = SOLUTION_RECORD_DEFAULT_PQBRAK
    blowup::Float64 = SOLUTION_RECORD_DEFAULT_BLOWUP
    itmxn::Int = SOLUTION_RECORD_DEFAULT_ITMXN
    toln::Float64 = SOLUTION_RECORD_DEFAULT_TOLN
    dvlim::Float64 = SOLUTION_RECORD_DEFAULT_DVLIM
    ndvfct::Float64 = SOLUTION_RECORD_DEFAULT_NDVFCT
    actaps::Int = 0
    areain::Int = 0
    phshft::Int = 0
    dctaps::Int = 0
    swshnt::Int = 0
    flatst::Int = 0
    varlim::Int = 0
    nondiv::Int = 0
end

# Solvers whose iteration budget and step-control parameters the fast-decoupled fields
# describe. For every other solver those fields keep the format defaults.
_is_fast_decoupled(::Type{<:FastDecoupledACPowerFlow}) = true
_is_fast_decoupled(::Type{<:ACPowerFlowSolverType}) = false

_solver_code(::Type{<:FastDecoupledACPowerFlow}) = SOLUTION_RECORD_SOLVER_DECOUPLED
_solver_code(::Type{<:ACPowerFlowSolverType}) = SOLUTION_RECORD_SOLVER_FULL_NEWTON

_solver_type(::AbstractACPowerFlow{S}) where {S} = S

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
    fd = _is_fast_decoupled(solver)

    # A discrete-control solve moves both tap changers and switched shunts; PowerFlows has
    # one flag where the format has two.
    tap_and_shunt = params.control_discrete_devices ? 1 : 0

    area = if !params.area_interchange_control
        0
    elseif params.tie_definition === :lines_and_loads
        2
    else
        1
    end

    # 0 applies the limits, -1 ignores them. PowerFlows enforces them between solves, so
    # there is no counterpart to the "apply after n iterations" form.
    varlim = params.check_reactive_power_limits ? 0 : -1

    iterations = something(
        params.maxIterations,
        fd ? DEFAULT_FD_MAX_ITER : DEFAULT_NR_MAX_ITER,
    )

    return SolutionRecordValues(;
        solver = _solver_code(solver),
        blowup = fd ? params.fd_blowup : SOLUTION_RECORD_DEFAULT_BLOWUP,
        itmxn = iterations,
        # Rounded: the per-unit-to-MW conversion leaves float noise that would otherwise
        # be written out in full (1e-7 * 100 is 9.999999999999999e-6), and no solver
        # tolerance is meaningful past twelve significant digits.
        toln = round(params.tol * base_power; sigdigits = 12),
        dvlim = fd ? params.fd_dvlim : SOLUTION_RECORD_DEFAULT_DVLIM,
        ndvfct = fd ? params.fd_ndvfct : SOLUTION_RECORD_DEFAULT_NDVFCT,
        actaps = tap_and_shunt,
        swshnt = tap_and_shunt,
        areain = area,
        varlim = varlim,
        flatst = params.enhanced_flat_start ? 1 : 0,
        nondiv = (fd && params.fd_non_divergent) ? 1 : 0,
        # No PowerFlows counterpart: there is no phase-shift or DC-tap control to report.
        phshft = 0,
        dctaps = 0,
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
    # The method name is positional and blank-able. The field is five characters wide, so
    # right-aligning it puts a space before a name and renders an empty name as the five
    # spaces the format defaults use.
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

_record_int(d, key, fallback) =
    haskey(d, key) ? something(tryparse(Int, d[key]), fallback) : fallback
_record_float(d, key, fallback) =
    haskey(d, key) ? something(tryparse(Float64, d[key]), fallback) : fallback

# The leading token of a record, with any trailing `/` comment removed. Section
# terminators customarily carry one — `0 / END OF SYSTEM-WIDE DATA, BEGIN BUS DATA`.
function _leading_token(line::AbstractString)
    field = first(_split_record(line))
    return strip(first(split(field, '/')))
end

# A line ends the block when its leading token is the terminator `0`. A leading token that
# parses as any other integer means bus data started and the block was absent — treat that
# as the end too, rather than reading bus records as solution records.
function _ends_solution_block(line::AbstractString)
    token = _leading_token(line)
    isempty(token) && return false
    return !isnothing(tryparse(Int, token))
end

# Records, in order, with the `@!` column-header comments and blank lines dropped. A v35
# export leads with a `@!IC,SBASE,REV,...` header, so the case identification record is not
# necessarily the first line of the file.
function _significant_records(lines)
    return [
        line for line in map(strip, lines)
        if !isempty(line) && !startswith(line, "@!")
    ]
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
    records = _significant_records(readlines(path))
    length(records) >= 3 || return nothing

    # Case identification record 1, field 3, is the format revision.
    header = _split_record(records[1])
    length(header) >= 3 || return nothing
    revision = tryparse(Int, strip(first(split(strip(header[3]), '/'))))
    (isnothing(revision) || revision < 35) && return nothing

    values = SolutionRecordValues()
    found = false
    # Record 2 is the case name; the block starts after it.
    for stripped in records[3:end]
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
            name = length(fields) >= 2 ? strip(fields[2]) : ""
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
    return found ? values : nothing
end

# Copy-with-overrides, so each record's parse only has to name the fields it sets.
function SolutionRecordValues(base::SolutionRecordValues; kwargs...)
    values = map(fieldnames(SolutionRecordValues)) do name
        haskey(kwargs, name) ? kwargs[name] : getfield(base, name)
    end
    return SolutionRecordValues(values...)
end

"""
    solution_parameters(values::SolutionRecordValues, base_power) -> SolutionParameters

Map solution records back onto PowerFlows parameters. `base_power` is the system base in
MVA and converts the record's MW/MVAr mismatch back to a per-unit tolerance.

Record fields with no PowerFlows counterpart are dropped; PowerFlows parameters the format
cannot express keep their defaults.
"""
function solution_parameters(values::SolutionRecordValues, base_power::Float64)
    fd = uppercase(values.solver) == SOLUTION_RECORD_SOLVER_DECOUPLED

    # A zero mismatch target is not a tolerance anyone can converge to; fall back rather
    # than hand a solver `tol = 0`.
    tol = values.toln > 0 ? values.toln / base_power : DEFAULT_NR_TOL

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

    return SolutionParameters(;
        tol = tol,
        maxIterations = values.itmxn > 0 ? values.itmxn : nothing,
        check_reactive_power_limits = values.varlim >= 0,
        enhanced_flat_start = values.flatst == 1,
        control_discrete_devices = values.actaps != 0 || values.swshnt != 0,
        area_interchange_control = values.areain != 0,
        tie_definition = tie_definition,
        fd_blowup = fd ? values.blowup : DEFAULT_FD_BLOWUP,
        fd_dvlim = fd ? values.dvlim : DEFAULT_FD_DVLIM,
        fd_ndvfct = fd ? values.ndvfct : DEFAULT_FD_NDVFCT,
        fd_non_divergent = fd ? values.nondiv == 1 : DEFAULT_FD_NON_DIVERGENT,
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
    values = read_solution_records(path)
    isnothing(values) && return nothing
    sbase = isnothing(base_power) ? _read_case_base_power(path) : Float64(base_power)
    return solution_parameters(values, sbase)
end

# Field 2 of the case identification record is the system base in MVA.
function _read_case_base_power(path::AbstractString)
    records = _significant_records(readlines(path))
    isempty(records) && return 100.0
    fields = _split_record(records[1])
    length(fields) >= 2 || return 100.0
    return something(tryparse(Float64, strip(fields[2])), 100.0)
end
