const PSSE_DEFAULT = ""  # Used below in cases where we want to insert an empty field to signify the PSSE default
const PSSE_INFINITY = 9999.0
const PSSE_BUS_TYPE_MAP = Dict(
    PSY.ACBusTypes.PQ => 1,
    PSY.ACBusTypes.PV => 2,
    PSY.ACBusTypes.REF => 3,
    PSY.ACBusTypes.SLACK => 3,
    PSY.ACBusTypes.ISOLATED => 4,
)
const PSSE_BRANCH_SPECIAL_CHARACTERS = ["&", "@", "*"]
const DISCRETE_BRANCH_MAP = Dict(
    PSY.DiscreteControlledBranchType.SWITCH => "*",
    PSY.DiscreteControlledBranchType.BREAKER => "@",
)

# Winding categories to map IC data to the transformer windings
const WINDING_CATEGORIES = [
    (PSY.WindingCategory.PRIMARY_WINDING, "1"),
    (PSY.WindingCategory.SECONDARY_WINDING, "2"),
    (PSY.WindingCategory.TERTIARY_WINDING, "3"),
]

# Accessor functions for each winding category
const WINDING_ACCESSORS = Dict(
    PSY.WindingCategory.PRIMARY_WINDING => (
        get_base_voltage = PSY.get_base_voltage_primary,
        get_turns_ratio = PSY.get_primary_turns_ratio,
        get_angle = PSY.get_α_primary,
        get_rating = PSY.get_rating_primary,
        get_group_number = PSY.get_primary_group_number,
    ),
    PSY.WindingCategory.SECONDARY_WINDING => (
        get_base_voltage = PSY.get_base_voltage_secondary,
        get_turns_ratio = PSY.get_secondary_turns_ratio,
        get_angle = PSY.get_α_secondary,
        get_rating = PSY.get_rating_secondary,
        get_group_number = PSY.get_secondary_group_number,
    ),
    PSY.WindingCategory.TERTIARY_WINDING => (
        get_base_voltage = PSY.get_base_voltage_tertiary,
        get_turns_ratio = PSY.get_tertiary_turns_ratio,
        get_angle = PSY.get_α_tertiary,
        get_rating = PSY.get_rating_tertiary,
        get_group_number = PSY.get_tertiary_group_number,
    ),
)

# Winding categories representing different types of transformer windings
const WINDING_GROUP_NUMBER_TO_DEGREES = Dict(
    -99 => PSSE_DEFAULT,   # UNDEFINED, export as default
    0 => 0,              # GROUP_0: 0 Degrees
    1 => -30,            # GROUP_1: -30 Degrees
    5 => -150,           # GROUP_5: -150 Degrees
    6 => 180,            # GROUP_6: 180 Degrees
    7 => 150,            # GROUP_7: 150 Degrees
    11 => 30,             # GROUP_11: 30 Degrees
)

# Each of the groups in the PSS/E v33 standard
const PSSE_GROUPS = [
    "Case Identification Data",
    "Bus Data",
    "Load Data",
    "Fixed Shunt Data",
    "Generator Data",
    "Non-Transformer Branch Data",
    "Transformer Data",
    "Area Interchange Data",
    "Two-Terminal DC Transmission Line Data",
    "Voltage Source Converter (VSC) DC Transmission Line Data",
    "Transformer Impedance Correction Tables",
    "Multi-Terminal DC Transmission Line Data",
    "Multi-Section Line Grouping Data",
    "Zone Data",
    "Interarea Transfer Data",
    "Owner Data",
    "FACTS Device Data",
    "Switched Shunt Data",
    "GNE Device Data",
    "Induction Machine Data",
    "Q Record",
]

# Extra groups for PSS/E v35 standard
const PSSE_V35_EXTRA_GROUPS = [
    "Switching Device Data",
    "Substation Data",
]

const PSSE_RAW_BUFFER_SIZEHINT = 1024
const PSSE_MD_BUFFER_SIZEHINT = 1024

# Header comments for v35 format (ordered by data section)
const PSSE_V35_HEADERS = Dict{String, String}(
    "Case Identification Data" => "@!IC,SBASE,REV,XFRRAT,NXFRAT,BASFRQ",
    "Bus Data" => "@!   I,'NAME        ', BASKV, IDE,AREA,ZONE,OWNER, VM,        VA,    NVHI,   NVLO,   EVHI,   EVLO",
    "Load Data" => "@!   I,'ID',STAT,AREA,ZONE,      PL,        QL,        IP,        IQ,        YP,        YQ, OWNER,SCALE,INTRPT,  DGENP,     DGENQ,DGENF,'  LOAD TYPE '",
    "Fixed Shunt Data" => "@!   I,'ID',STATUS,  GL,        BL",
    "Generator Data" => "@!   I,'ID',      PG,        QG,        QT,        QB,     VS,    IREG,NREG,     MBASE,     ZR,         ZX,         RT,         XT,     GTAP,STAT, RMPCT,      PT,        PB,BASLOD,O1,    F1,  O2,    F2,  O3,    F3,  O4,    F4,WMOD, WPF",
    "Non-Transformer Branch Data" => "@!   I,     J,'CKT',      R,           X,       B,                   'N A M E'                 ,  RATE1,  RATE2,  RATE3,  RATE4,
@!      RATE5,  RATE6,  RATE7,  RATE8,  RATE9, RATE10, RATE11, RATE12,   GI,      BI,      GJ,      BJ,STAT,MET, LEN,  O1,  F1,    O2,  F2,    O3,  F3,    O4,  F4",
    "Switching Device Data" => "@!   I,     J,'CKT',          X,  RATE1,  RATE2,  RATE3,  RATE4,  RATE5,  RATE6,  RATE7,  RATE8,  RATE9, RATE10, RATE11, RATE12, STAT,NSTAT,  MET,STYPE,'NAME'",
    "Transformer Data" => """
@!   I,     J,     K,'CKT',CW,CZ,CM,     MAG1,        MAG2,NMETR,               'N A M E',               STAT,O1,  F1,    O2,  F2,    O3,  F3,    O4,  F4,     'VECGRP', ZCOD
@!   R1-2,       X1-2, SBASE1-2,     R2-3,       X2-3, SBASE2-3,     R3-1,       X3-1, SBASE3-1, VMSTAR,   ANSTAR
@!WINDV1, NOMV1,   ANG1, RATE1-1, RATE1-2, RATE1-3, RATE1-4, RATE1-5, RATE1-6, RATE1-7, RATE1-8, RATE1-9,RATE1-10,RATE1-11,RATE1-12,COD1,CONT1,NOD1,  RMA1,   RMI1,   VMA1,   VMI1, NTP1,TAB1, CR1,    CX1,  CNXA1
@!WINDV2, NOMV2,   ANG2, RATE2-1, RATE2-2, RATE2-3, RATE2-4, RATE2-5, RATE2-6, RATE2-7, RATE2-8, RATE2-9,RATE2-10,RATE2-11,RATE2-12,COD2,CONT2,NOD2,  RMA2,   RMI2,   VMA2,   VMI2, NTP2,TAB2, CR2,    CX2,  CNXA2
@!WINDV3, NOMV3,   ANG3, RATE3-1, RATE3-2, RATE3-3, RATE3-4, RATE3-5, RATE3-6, RATE3-7, RATE3-8, RATE3-9,RATE3-10,RATE3-11,RATE3-12,COD3,CONT3,NOD3,  RMA3,   RMI3,   VMA3,   VMI3, NTP3,TAB3, CR3,    CX3,  CNXA3""",
    "Two-Terminal DC Transmission Line Data" => """
@!  'NAME',   MDC,    RDC,     SETVL,    VSCHD,    VCMOD,    RCOMP,   DELTI,METER,   DCVMIN,CCCITMX,CCCACC
@! IPR,NBR,  ANMXR,  ANMNR,   RCR,    XCR,   EBASR,  TRR,    TAPR,   TMXR,   TMNR,   STPR,    ICR,NDR,   IFR,   ITR,'IDR', XCAPR
@! IPI,NBI,  ANMXI,  ANMNI,   RCI,    XCI,   EBASI,  TRI,    TAPI,   TMXI,   TMNI,   STPI,    ICI,NDI,   IFI,   ITI,'IDI', XCAPI""",
    "Voltage Source Converter (VSC) DC Transmission Line Data" => """
@!  'NAME',   MDC,  RDC,   O1,  F1,    O2,  F2,    O3,  F3,    O4,  F4
@!IBUS,TYPE,MODE,  DCSET,  ACSET,  ALOSS,  BLOSS,MINLOSS,  SMAX,   IMAX,   PWF,     MAXQ,   MINQ, VSREG,NREG, RMPCT""",
    "Transformer Impedance Correction Tables" => """
@!I,  T1,   Re(F1), Im(F1),   T2,   Re(F2), Im(F2),   T3,   Re(F3), Im(F3),   T4,   Re(F4), Im(F4),   T5,   Re(F5), Im(F5),   T6,   Re(F6), Im(F6)
@!    T7,   Re(F7), Im(F7),   T8,   Re(F8), Im(F8),   T9,   Re(F9), Im(F9),   T10, Re(F10),Im(F10),   T11, Re(F11),Im(F11),   T12, Re(F12),Im(F12)
@!      ...""",
    "Zone Data" => "@! I,   'ZONAME'",
    "FACTS Device Data" => "@!  'NAME',         I,     J,MODE,PDES,   QDES,  VSET,   SHMX,   TRMX,   VTMN,   VTMX,   VSMX,    IMX,   LINX,   RMPCT,OWNER,  SET1,    SET2,VSREF, FCREG,NREG,   'MNAME'",
    "Switched Shunt Data" => "@!   I,'ID',MODSW,ADJM,ST, VSWHI,  VSWLO, SWREG,NREG, RMPCT,   'RMIDNT',     BINIT,S1,N1,    B1, S2,N2,    B2, S3,N3,    B3, S4,N4,    B4, S5,N5,    B5, S6,N6,    B6, S7,N7,    B7, S8,N8,    B8",
    "GNE Device Data" => """
@!  'NAME',        'MODEL',     NTERM,BUS1...BUSNTERM,NREAL,NINTG,NCHAR
@!ST,OWNER,NMETR
@! REAL1...REAL(MIN(10,NREAL))
@! INTG1...INTG(MIN(10,NINTG))
@! CHAR1...CHAR(MIN(10,NCHAR))""",
    "Induction Machine Data" => "@!   I,'ID',ST,SC,DC,AREA,ZONE,OWNER,TC,BC,  MBASE, RATEKV,PC,  PSET,      H,       A,       B,       D,       E,     RA,        XA,        XM,        R1,        X1,        R2,        X2,        X3,       E1,    SE1,   E2,    SE2,   IA1,   IA2, XAMULT",
)

@kwdef struct PSSEExportPowerFlow <: PowerFlowEvaluationModel
    psse_version::Symbol
    export_dir::String
    name::String = PSSE_DEFAULT_EXPORT_NAME
    write_comments::Bool = false
    overwrite::Bool = false
end

"""
    PSSEExportPowerFlow(psse_version::Symbol, export_dir::AbstractString; kwargs...)

An evaluation model for exporting power flow results to PSSE format.

Arguments:
- `psse_version::Symbol`: The version of PSSE to export to. Must be among `$PSSE_EXPORT_SUPPORTED_VERSIONS`.
- `export_dir::AbstractString`: The directory where the PSSE files will be exported.
Optional keyword arguments:
- `name::AbstractString`: The base name for the exported files. Defaults to `\"$PSSE_DEFAULT_EXPORT_NAME\"`.
- `write_comments::Bool`: Whether to write comments in the exported files. Defaults to `false`.
- `overwrite::Bool`: Whether to overwrite the file if it exists already. Defaults to `false`.
"""
PSSEExportPowerFlow(psse_version::Symbol, export_dir::AbstractString; kwargs...) =
    PSSEExportPowerFlow(; psse_version = psse_version, export_dir = export_dir, kwargs...)

get_exporter(pfem::PowerFlowEvaluationModel) = pfem.exporter
get_exporter(::PSSEExportPowerFlow) = nothing

"""
Expand a single `PowerFlowEvaluationModel` into its possibly multiple parts for separate
evaluation. Namely, if `pfem` contains a non-nothing `exporter`, return `[pfem, exporter]`,
else return `[pfem]`.
"""
function flatten_power_flow_evaluation_model(pfem::PowerFlowEvaluationModel)
    exporter = get_exporter(pfem)
    return if isnothing(exporter)
        PowerFlowEvaluationModel[pfem]
    else
        PowerFlowEvaluationModel[pfem, exporter]
    end
end

"""
Structure to perform an export from a Sienna System, plus optional updates from
`PowerFlowData`, to the PSS/E format.

Construct this object from a [`System`](@extref PowerSystems.System) and a PSS/E version,
update using `update_exporter` with any new data as relevant, and perform the export with
`write_export`. Writes a `<name>.raw` file and a `<name>_export_metadata.json` file with
transformations that had to be made to conform to PSS/E naming rules, which can be parsed by
PowerSystems.jl to perform a round trip with the names restored.

# Arguments:
  - `base_system::PSY.System`: the system to be exported. Later updates may change power
    flow-related values but may not fundamentally alter the system
  - `psse_version::Symbol`: the version of PSS/E to target, must be one of
    `$PSSE_EXPORT_SUPPORTED_VERSIONS`
  - `write_comments::Bool = false`: whether to add the customary-but-not-in-spec-annotations
    after a slash on the first line and at group boundaries
  - `name::AbstractString = "export"`: the base name of the export
  - `step::Any = nothing`: optional step data to append to the base export name. User is
    responsible for updating the step data. If the step data is `nothing`, it is not used;
    if it is a tuple or vector, it is joined with \\_ and concatted; else it is concatted
    after \\_.
  - `overwrite::Bool = false`: `true` to silently overwrite existing exports, `false` to
    throw an error if existing results are encountered
"""
mutable struct PSSEExporter <: SystemPowerFlowContainer
    system::PSY.System
    psse_version::Symbol
    export_dir::String
    name::String
    write_comments::Bool
    overwrite::Bool
    step::Any
    raw_buffer::IOBuffer  # Persist an IOBuffer to reduce allocations on repeated exports
    md_dict::OrderedDict{String, Any}  # Persist metadata to avoid unnecessary recomputation
    md_valid::Bool  # If this is true, the metadata need not be reserialized
    md_buffer::IOBuffer  # Cache a serialized version of the metadata
    components_cache::Dict{String, Any}  # Cache sorted lists of components to reduce allocations

    function PSSEExporter(
        base_system::PSY.System,
        psse_version::Symbol,
        export_dir::AbstractString;
        name::AbstractString = PSSE_DEFAULT_EXPORT_NAME,
        write_comments::Bool = false,
        overwrite::Bool = false,
        step::Any = nothing,
    )
        (psse_version in PSSE_EXPORT_SUPPORTED_VERSIONS) ||
            throw(
                ArgumentError(
                    "PSS/E version $psse_version is not supported, must be one of $PSSE_EXPORT_SUPPORTED_VERSIONS",
                ),
            )
        system = PSY.fast_deepcopy_system(base_system; skip_supplemental_attributes = false)
        mkpath(export_dir)
        new(
            system,
            psse_version,
            String(export_dir),
            String(name),
            write_comments,
            overwrite,
            step,
            IOBuffer(),
            OrderedDict{String, Any}(),
            false,
            IOBuffer(),
            Dict{String, Any}(),
        )
    end
end

supports_multi_period(::PSSEExporter) = false

_value_or_default(val, default) = isnothing(val) ? default : val

"""
Write v35 header comments for a given section if applicable.
"""
function write_v35_header(io::IO, exporter::PSSEExporter, section_name::String)
    if exporter.psse_version == :v35 && haskey(PSSE_V35_HEADERS, section_name)
        header = PSSE_V35_HEADERS[section_name]
        println(io, header)
    end
end

function update_version_group(psse_version::Symbol)
    groups = copy(PSSE_GROUPS)
    if psse_version == :v35
        # Insert v35-specific group at the correct position
        switching_idx = findfirst(==("Non-Transformer Branch Data"), groups) + 1
        insert!(groups, switching_idx, "Switching Device Data")
        substation_idx = findfirst(==("Induction Machine Data"), groups) + 1
        insert!(groups, substation_idx, "Substation Data")
    end
    return groups
end

function _validate_same_system(sys1::PSY.System, sys2::PSY.System)
    return IS.get_uuid(PSY.get_internal(sys1)) == IS.get_uuid(PSY.get_internal(sys2))
end

"""
Update the `PSSEExporter` with new `data`.

# Arguments:
  - `exporter::PSSEExporter`: the exporter to update
  - `data::PSY.PowerFlowData`: the new data. Must correspond to the
    [`System`](@extref PowerSystems.System) with which the exporter was constructed.
"""
function update_exporter!(exporter::PSSEExporter, data::PowerFlowData)
    # NOTE this relies on exporter.system being a deepcopy of the original system so we're not changing that one here
    update_system!(exporter.system, data)
end

"Force all cached information (serialized metadata, component lists, etc.) to be regenerated"
function reset_caches(exporter::PSSEExporter)
    take!(exporter.raw_buffer)
    empty!(exporter.components_cache)
    exporter.md_valid = false
    # We do not clear the md_buffer here, but !md_valid implies that its contents are not valid
end

"""
Update the `PSSEExporter` with new `data`.

# Arguments:
  - `exporter::PSSEExporter`: the exporter to update
  - `data::PSY.System`: system containing the new data. Must be fundamentally the same \
  [`System`](@extref PowerSystems.System) as the one with which the exporter was
    constructed, just with different values — this is the user's responsibility, we do not
    exhaustively verify it.
"""
function update_exporter!(exporter::PSSEExporter, data::PSY.System)
    _validate_same_system(exporter.system, data) || throw(
        ArgumentError(
            "System passed to update_exporter must be the same system as the one with which the exporter was constructed, just with different values",
        ),
    )
    exporter.system = PSY.fast_deepcopy_system(data)
    reset_caches(exporter)
    return
end

get_data_array(buf::Base.GenericIOBuffer{<:Array{UInt8}}) =  # < Julia 1.11
    buf.data

(@isdefined GenericMemory) && (  # >= Julia 1.11
    get_data_array(buf::Base.GenericIOBuffer{<:GenericMemory{:not_atomic, UInt8}}) =
        Base.wrap(Array, buf.data)
)

const _FloatToBufSupportedTypes = if (@isdefined GenericMemory)
    Union{
        Base.GenericIOBuffer{<:Array{UInt8}},
        Base.GenericIOBuffer{<:GenericMemory{:not_atomic, UInt8}},
    }
else
    Base.GenericIOBuffer{<:Array{UInt8}}
end

(IOBuffer <: _FloatToBufSupportedTypes) ||
    @warn "Fast Float64 to IOBuffer implementation is out of date, will not be used"

"Temporary, very specialized proof of concept patch for https://github.com/JuliaLang/julia/issues/55835"
function better_float_to_buf(buf::_FloatToBufSupportedTypes, n::Float64)
    Base.ensureroom(buf, Base.Ryu.neededdigits(Float64))
    # get_data_array incurs an allocation on Julia >= 1.11. I think writeshortest could work
    # with the underlying Memory with minimal modification, which would be nice because
    # other than this, better_float_to_buf is completely allocation free.
    data_array = get_data_array(buf)
    new_pos = Base.Ryu.writeshortest(data_array, buf.ptr, n, false, false, true, -1,
        UInt8('e'), false, UInt8('.'), true, false)
    buf.ptr = new_pos
    buf.size = new_pos - 1
    return
end

fastprint(io::IO, val) = print(io, val)
fastprint(io::_FloatToBufSupportedTypes, val::Float64) = better_float_to_buf(io, val)

function fastprintdelim(io, val, delim = ", ")
    fastprint(io, val)
    fastprint(io, delim)
end

function fastprintln(io, val, ln = "\n")
    fastprint(io, val)
    fastprint(io, ln)
end

# call fastprintdelim multiple times on val, unrolled
macro fastprintdelim_multi(io, val, newline::Bool, mult::Int)
    local exprs = [Expr(:call, :fastprintdelim, esc(io), esc(val)) for _ in 1:(mult - 1)]
    local lastCall = newline ? :fastprintln : :fastprintdelim
    local lastExpr = Expr(:call, lastCall, esc(io), esc(val))
    return Expr(:block, exprs..., lastExpr)
end
# call fastprintdelim on each item in vals, unrolled
macro fastprintdelim_unroll(io, newline::Bool, vals...)
    local allButLast = vals[begin:(end - 1)]
    local exprs = [Expr(:call, :fastprintdelim, esc(io), esc(item)) for item in allButLast]
    local lastCall = newline ? :fastprintln : :fastprintdelim
    local lastExpr = Expr(:call, lastCall, esc(io), esc(vals[end]))
    return Expr(:block, exprs..., lastExpr)
end

function fastprintdelim_psse_default_ownership(io::IO)
    @fastprintdelim_multi(io, PSSE_DEFAULT, false, 8)
end

function fastprintln_psse_default_ownership(io::IO)
    @fastprintdelim_multi(io, PSSE_DEFAULT, true, 8)
end

function end_group(
    io::IO,
    md::AbstractDict,
    exporter::PSSEExporter,
    group_name::String,
    written::Bool,
)
    groups = update_version_group(exporter.psse_version)
    current_index = findfirst(==(group_name), groups)

    end_msg = "0"
    if exporter.write_comments
        end_msg *= " / End of $group_name"
        if current_index < length(groups) && groups[current_index + 1] != "Q Record"
            next_group = groups[current_index + 1]
            end_msg *= ", Begin $next_group"
        end
    end
    println(io, end_msg)
    exporter.md_valid || (md["record_groups"][group_name] = written)
end

"""
Parse a value to Int64, handling strings, floats, and PSSE_DEFAULT.
Returns PSSE_DEFAULT for non-whole numbers or invalid values.
"""
_to_float(x::Number) = Float64(x)
_to_float(x::AbstractString) = tryparse(Float64, x)
_to_float(::Any) = nothing
function _permissive_parse_int(x)
    (x == PSSE_DEFAULT || x == "") && return PSSE_DEFAULT

    n = _to_float(x)
    isnothing(n) && return PSSE_DEFAULT
    (round(n) == n) || return PSSE_DEFAULT
    return Int64(n)
end

"""
If `val` is empty, returns `T()`; if not, asserts that `val isa T` and returns `val`. Has nice type checker semantics.

# Examples
```julia
convert_empty(Vector{String}, [])  # -> String[]
convert_empty(Vector{String}, ["a"])  # -> ["a"]
convert_empty(Vector{String}, [2])  # -> TypeError: in typeassert, expected Vector{String}, got a value of type Vector{Int64}
Base.return_types(Base.Fix1(convert_empty, Vector{String}))  # -> [Vector{String}]
```
"""
convert_empty(::Type{T}, val) where {T} = isempty(val) ? T() : val::T
convert_empty_stringvec = Base.Fix1(convert_empty, Vector{String})

# PERF could be improved by appending to the buffer rather than doing string interpolation, seems unnecessary
_psse_quote_string(s::String) = "'$s'"

# Rounds a value up to 4 decimals, to avoid large approximations in file
_psse_round_val(val::String) = val
_psse_round_val(val::Number) = round(val; digits = 4)

branch_to_bus_numbers(branch::PSY.Branch) =
    PSY.get_number.((PSY.get_from_bus(branch), PSY.get_to_bus(branch)))::Tuple{Int, Int}

function branch_to_bus_numbers(
    branch::PSY.ThreeWindingTransformer,
)
    p = PSY.get_number(PSY.get_from(PSY.get_primary_star_arc(branch)))
    s = PSY.get_number(PSY.get_from(PSY.get_secondary_star_arc(branch)))
    t = PSY.get_number(PSY.get_from(PSY.get_tertiary_star_arc(branch)))
    return (p, s, t)
end

"Throw a `NotImplementedError` if the `psse_version` is not supported"
check_supported_version(exporter::PSSEExporter) =
    check_supported_version(exporter.psse_version)
check_supported_version(psse_version::Symbol) =
    (psse_version in PSSE_EXPORT_SUPPORTED_VERSIONS) ||
    throw(
        IS.NotImplementedError(
            "Only implemented for psse_version $(PSSE_EXPORT_SUPPORTED_VERSIONS), got $psse_version",
        ),
    )

"Validate that the Sienna area/zone names parse as PSS/E-compatible area/zone numbers, output a mapping"
function _map_psse_container_names(container_names::Vector{String})
    (length(container_names) <= 9999) || throw(ArgumentError("Too many container_names"))
    mapping = OrderedDict{String, Int}()
    used = Set{Int}()
    for name in container_names
        try
            # Ideally it's a number and we just parse it
            parsed = _permissive_parse_int(name)
            if !isnothing(parsed) && (parsed in 1:9999) && !(parsed in used)
                mapping[name] = parsed
                push!(used, parsed)
                continue
            end
        catch e
            (e isa Union{InexactError, ArgumentError}) || rethrow(e)
        end

        # If parsing fails or the number doesn't work, assign it the lowest unused number
        i = 1
        while i in used  # PERF inefficient but unlikely to matter
            i += 1
        end
        @assert i <= 9999
        mapping[name] = i
        push!(used, i)
    end
    return mapping
end

"""
Setting a value of zero 0.0 when having a value greater than or equal to INFINITE_BOUND
reverses the operation done in the PSY parsing side, according to PSSE Manual.
"""
_fix_3w_transformer_rating(::AbstractString) = 0.0
_fix_3w_transformer_rating(rate::Number) = rate >= INFINITE_BOUND ? 0.0 : rate

"WRITTEN TO SPEC: PSS/E 33.3/35.4 POM 5.2.1 Case Identification Data"
function write_to_buffers!(
    exporter::PSSEExporter,
    ::Val{Symbol("Case Identification Data")},
)
    io = exporter.raw_buffer
    md = exporter.md_dict

    check_supported_version(exporter)
    write_v35_header(io, exporter, "Case Identification Data")
    now = Dates.now()

    # Update version-specific values
    version_number = exporter.psse_version == :v33 ? 33 : 35
    version_string = exporter.psse_version == :v33 ? "33.3" : "35"

    md_string = "PSS/E $version_string RAW via PowerFlows.jl, $now"

    # Record 1
    IC = 0
    SBASE = PSY.get_base_power(exporter.system)
    REV = version_number
    XFRRAT = 0
    NXFRAT = 1
    BASFRQ = PSY.get_frequency(exporter.system)
    exporter.write_comments && (BASFRQ = "$BASFRQ    / $md_string")

    @fastprintdelim_unroll(io, true, IC, SBASE, REV, XFRRAT, NXFRAT, BASFRQ)

    # Record 2
    case_name = md["case_name"]
    (length(case_name) <= 60) ||
        throw(ArgumentError("case_name may be up to 60 characters"))
    println(io, case_name)

    # Record 3
    line3 = md_string
    @assert length(line3) <= 60
    println(io, line3)
    exporter.md_valid || (md["record_groups"]["Case Identification Data"] = true)
end

"""
Given a vector of Sienna bus numbers, create a dictionary from Sienna bus number to
PSS/E-compatible bus number. Assumes that the Sienna bus numbers are positive and unique.
Guarantees determinism: if the input contains the same bus numbers in the same order, the
output will. Guarantees minimal changes: that if an existing bus number is compliant, it
will not be changed.

WRITTEN TO SPEC: PSS/E 33.3 POM 5.2.1 Bus Data
"""
function _psse_bus_numbers(buses::Vector{Int64})
    used_numbers = Set{Int64}()
    sizehint!(used_numbers, length(buses))
    mapping = OrderedDict{Int64, Int64}()
    sizehint!(mapping, length(buses))

    for original_number in buses
        if original_number in 1:999_997
            mapping[original_number] = original_number
            push!(used_numbers, original_number)
        end
    end
    for original_number in buses
        haskey(mapping, original_number) && continue
        new_number = original_number
        new_number %= 1_000_000
        (new_number in 999_997:1_000_000) && (new_number -= 100_000)
        while new_number in used_numbers
            new_number += 1
        end
        mapping[original_number] = new_number
        push!(used_numbers, new_number)
    end
    return mapping
end

function _is_valid_psse_name(name::String)
    (length(name) <= 12) || (return false)
    (length(name) >= 1) && (first(name) == '-') && (return false)
    return true  # Does the allowance for special characters cover *any* special characters?
end

function get_ext_key_or_default(
    component::PSY.Component,
    key::String,
    default = PSSE_DEFAULT,
)
    ext = PSY.get_ext(component)
    if isnothing(ext)
        return default
    end
    return get(ext, key, default)
end

"""
Given a vector of Sienna bus names, create a dictionary from Sienna bus name to
PSS/E-compatible bus name. Guarantees determinism and minimal changes.

WRITTEN TO SPEC: PSS/E 33.3 POM 5.2.1 Bus Data
"""
function _psse_bus_names(
    buses::Vector{String},
    bus_numbers::Vector{Int64},
    bus_number_mapping::AbstractDict{Int64, Int64},
)
    used_names = Set{String}()
    sizehint!(used_names, length(buses))
    mapping = OrderedDict{String, String}()
    sizehint!(mapping, length(buses))

    for original_name in buses
        if _is_valid_psse_name(original_name)
            mapping[original_name] = original_name
            push!(used_names, original_name)
        end
    end
    for (original_name, original_number) in zip(buses, bus_numbers)
        haskey(mapping, original_name) && continue
        new_name = first(original_name, 12)
        if !_is_valid_psse_name(new_name) || new_name in used_names
            new_name = "BUS_$(bus_number_mapping[original_number])"
            while new_name in used_names
                new_name *= "-"
            end
        end
        @assert _is_valid_psse_name(new_name) new_name
        mapping[original_name] = new_name
        push!(used_names, new_name)
    end
    return mapping
end

"""
WRITTEN TO SPEC: PSS/E 33.3/35.4 POM 5.2.1 Bus Data. Sienna voltage limits treated as PSS/E
normal voltage limits; PSSE emergency voltage limits left as default.
"""
function write_to_buffers!(
    exporter::PSSEExporter,
    ::Val{Symbol("Bus Data")},
)
    io = exporter.raw_buffer
    md = exporter.md_dict

    check_supported_version(exporter)
    write_v35_header(io, exporter, "Bus Data")

    tr3w_starbuses =
        PSY.get_name.(
            PSY.get_star_bus.(
                PSY.get_components(PSY.ThreeWindingTransformer, exporter.system)
            )
        )
    buses = get!(exporter.components_cache, "buses") do
        sort!(
            [
                bus for bus in collect(PSY.get_components(PSY.ACBus, exporter.system))
                if !(PSY.get_name(bus) in tr3w_starbuses)
            ];
            by = PSY.get_number,
        )
    end

    old_bus_numbers = PSY.get_number.(buses)

    if !exporter.md_valid
        md["bus_number_mapping"] = _psse_bus_numbers(old_bus_numbers)
        md["bus_name_mapping"] =
            _psse_bus_names(
                convert_empty_stringvec(PSY.get_name.(buses)),
                old_bus_numbers,
                md["bus_number_mapping"],
            )
    end
    bus_number_mapping = md["bus_number_mapping"]
    bus_name_mapping = md["bus_name_mapping"]

    for bus in buses
        bus_name = PSY.get_name(bus)
        I = bus_number_mapping[PSY.get_number(bus)]
        NAME = _psse_quote_string(bus_name_mapping[bus_name])
        BASKV = PSY.get_base_voltage(bus)
        IDE = PSSE_BUS_TYPE_MAP[PSY.get_bustype(bus)]
        AREA = if isnothing(PSY.get_area(bus))
            PSSE_DEFAULT
        else
            md["area_mapping"][PSY.get_name(PSY.get_area(bus))]
        end
        ZONE = if isnothing(PSY.get_load_zone(bus))
            PSSE_DEFAULT
        else
            md["zone_mapping"][PSY.get_name(PSY.get_load_zone(bus))]
        end
        OWNER = PSSE_DEFAULT
        VM = PSY.get_magnitude(bus)
        VA = rad2deg(PSY.get_angle(bus))
        NVHI = PSY.get_voltage_limits(bus).max
        NVLO = PSY.get_voltage_limits(bus).min
        EVHI = PSSE_DEFAULT
        EVLO = PSSE_DEFAULT

        @fastprintdelim_unroll(io, true, I, NAME, BASKV, IDE, AREA,
            ZONE, OWNER, VM, VA,
            NVHI, NVLO, EVHI, EVLO)
    end
    end_group(io, md, exporter, "Bus Data", true)
end

function _increment_component_char(component_char::Char)
    (component_char == '9') && return 'A'
    (component_char == 'Z') && return '0'
    return component_char + 1
end

function _increment_component_id(component_id::String)
    carry = (last(component_id) == 'Z')
    if length(component_id) == 1
        carry && return '0' * _increment_component_char(first(component_id))
        return string(_increment_component_char(first(component_id)))
    end
    return (carry ? _increment_component_char(first(component_id)) : first(component_id)) *
           _increment_component_char(last(component_id))
end

"""
Try to make an informative one or two character name for the load/generator/etc.

  - generator-1234-AB -> AB
  - 123\\_CT\\_7 -> 7
  - load1234 -> 34
"""
function _first_choice_gen_id(name::String)
    my_split = argmax(length, [split(name, "-"), split(name, "_")])
    return uppercase(last(last(my_split), 2))
end

"""
Given a vector of component names and a corresponding vector of container IDs (e.g., bus
numbers), create unique-per-container PSS/E-compatible IDs, output a dictionary from
(container ID, component name) to PSS/E-compatible component ID. The "singles_to_1" flag
detects components that are the only one on their bus and gives them the name "1".
"""
function create_component_ids(
    component_names::Vector{<:String},
    container_ids::Vector{T};
    singles_to_1 = false,
) where {T}
    id_mapping = Dict{Tuple{T, String}, String}()
    sizehint!(id_mapping, length(component_names))
    ids_by_container = Dict{T, Vector{String}}()

    for (name, container_id) in zip(component_names, container_ids)
        haskey(ids_by_container, container_id) ||
            (ids_by_container[container_id] = Vector{String}())
        my_blocked = ids_by_container[container_id]
        id = _first_choice_gen_id(name)
        while id in my_blocked
            id = _increment_component_id(id)
        end
        id_mapping[(container_id, name)] = id
        push!(my_blocked, id)
    end
    if singles_to_1
        for (name, container_id) in zip(component_names, container_ids)
            (length(ids_by_container[container_id]) == 1) &&
                (id_mapping[(container_id, name)] = "1")
        end
    end
    return id_mapping
end

"Take the output of `create_component_ids` and make it more suitable for JSON serialization"
serialize_component_ids(id_mapping::Dict{Tuple{Int64, String}, String}) =
    Dict("$(s_bus_n)_$(s_name)" => p_name for ((s_bus_n, s_name), p_name) in id_mapping)
serialize_component_ids(
    id_mapping::Dict{Tuple{Tuple{Int64, Int64, Int64}, String}, String},
) =
    Dict(
        "$(s_bus_1)-$(s_bus_2)-$(s_bus_3)_$(s_name)" => p_name
        for (((s_bus_1, s_bus_2, s_bus_3), s_name), p_name) in id_mapping
    )
serialize_component_ids(id_mapping::Dict{Tuple{Tuple{Int64, Int64}, String}, String}) =
    Dict(
        "$(s_bus_1)-$(s_bus_2)_$(s_name)" => p_name for
        (((s_bus_1, s_bus_2), s_name), p_name) in id_mapping
    )

# ============================================================================
# Helper functions for Transformer Data export
# ============================================================================

_get_mag_defaults(t::PSY.ThreeWindingTransformer) =
    (PSY.get_g(t), PSY.get_b(t))
_get_mag_defaults(t::PSY.TwoWindingTransformer) =
    (real(PSY.get_primary_shunt(t)), imag(PSY.get_primary_shunt(t)))

"""Write the first record line for a 2-winding transformer."""
function _write_2w_transformer_record1!(
    io::IO,
    exporter::PSSEExporter,
    I::Int,
    J::Int,
    K::Int,
    CKT::String,
    transformer::Union{PSY.TwoWindingTransformer, PSY.ThreeWindingTransformer},
    NAME::String,
    STAT::Int,
)
    default_CW, default_CZ, default_CM = _get_cw_cz_cm_defaults(transformer)
    CW = get_ext_key_or_default(transformer, "CW", default_CW)
    CZ = get_ext_key_or_default(transformer, "CZ", default_CZ)
    CM = get_ext_key_or_default(transformer, "CM", default_CM)
    mag1_default, mag2_default = _get_mag_defaults(transformer)
    MAG1 = get_ext_key_or_default(transformer, "MAG1", mag1_default)
    MAG2 = get_ext_key_or_default(transformer, "MAG2", mag2_default)
    NMETR = get_ext_key_or_default(transformer, "NMETR")
    VECGRP = _psse_quote_string(get_ext_key_or_default(transformer, "VECGRP"))
    ZCOD = get_ext_key_or_default(transformer, "ZCOD")

    if exporter.psse_version == :v35
        @fastprintdelim_unroll(io, false, I, J, K, CKT, CW, CZ, CM,
            MAG1, MAG2, NMETR, NAME, STAT)
        fastprintdelim_psse_default_ownership(io)
        @fastprintdelim_unroll(io, true, VECGRP, ZCOD)
    else
        @fastprintdelim_unroll(io, false, I, J, K, CKT, CW, CZ, CM,
            MAG1, MAG2, NMETR, NAME, STAT)
        fastprintdelim_psse_default_ownership(io)
        fastprintln(io, VECGRP)
    end
end

"""Write the second record line (impedance data) for a 2-winding transformer."""
function _write_2w_transformer_record2!(
    io::IO,
    transformer::PSY.TwoWindingTransformer,
)
    SBASE1_2 = get_ext_key_or_default(
        transformer,
        "SBASE1-2",
        PSY.get_base_power(transformer),
    )
    R1_2 = Float64(get_ext_key_or_default(
        transformer,
        "R1-2",
        PSY.get_r(transformer))
    )
    X1_2 = get_ext_key_or_default(
        transformer,
        "X1-2",
        PSY.get_x(transformer),
    )
    @fastprintdelim_unroll(io, true, R1_2, X1_2, SBASE1_2)
end

_get_2w_ang1(t::PSY.PhaseShiftingTransformer) = rad2deg(PSY.get_α(t))
_get_2w_ang1(t::Union{PSY.Transformer2W, PSY.TapTransformer}) =
    get(
        WINDING_GROUP_NUMBER_TO_DEGREES,
        PSY.get_winding_group_number(t).value,
        PSSE_DEFAULT,
    )
_get_2w_ang1(::PSY.TwoWindingTransformer) = 0.0

_resolve_cod1_val(
    cod1_val::PSY.TransformerControlObjectiveModule.TransformerControlObjective,
    transformer,
) =
    if cod1_val ==
       PSY.TransformerControlObjectiveModule.TransformerControlObjective.UNDEFINED
        get_ext_key_or_default(transformer, "COD1")
    else
        cod1_val.value
    end
_resolve_cod1_val(cod1_val, _) = cod1_val

function _get_2w_cod1(
    transformer::Union{PSY.TapTransformer, PSY.PhaseShiftingTransformer},
)
    cod1_val = get_ext_key_or_default(
        transformer,
        "COD1",
        PSY.get_control_objective(transformer),
    )
    return _resolve_cod1_val(cod1_val, transformer)
end
_get_2w_cod1(transformer::PSY.TwoWindingTransformer) =
    get_ext_key_or_default(transformer, "COD1")

_get_windv1_default(transformer::PSY.TapTransformer) = PSY.get_tap(transformer)
_get_windv1_default(transformer::PSY.TwoWindingTransformer) =
    PSY.get_base_voltage_primary(transformer)

_get_windv2_default(::PSY.TapTransformer) = 1.0
_get_windv2_default(transformer::PSY.TwoWindingTransformer) =
    PSY.get_base_voltage_secondary(transformer)

function _get_cw_cz_cm_defaults(transformer::PSY.Transformer2W)
    from_base_voltage = PSY.get_base_voltage(PSY.get_from(PSY.get_arc(transformer)))
    to_base_voltage = PSY.get_base_voltage(PSY.get_to(PSY.get_arc(transformer)))
    windv1 =
        get_ext_key_or_default(transformer, "WINDV1", _get_windv1_default(transformer))
    windv2 =
        get_ext_key_or_default(transformer, "WINDV2", _get_windv2_default(transformer))

    if windv1 isa Number && windv2 isa Number &&
       isapprox(windv1, from_base_voltage) &&
       isapprox(windv2, to_base_voltage)
        return (2, 2, 1)
    end

    return (PSSE_DEFAULT, PSSE_DEFAULT, PSSE_DEFAULT)
end

function _get_cw_cz_cm_defaults(transformer::PSY.Transformer3W)
    primary_base_voltage =
        PSY.get_base_voltage(PSY.get_from(PSY.get_primary_star_arc(transformer)))
    secondary_base_voltage =
        PSY.get_base_voltage(PSY.get_from(PSY.get_secondary_star_arc(transformer)))
    tertiary_base_voltage =
        PSY.get_base_voltage(PSY.get_from(PSY.get_tertiary_star_arc(transformer)))

    windv1 = get_ext_key_or_default(
        transformer,
        "WINDV1",
        PSY.get_primary_turns_ratio(transformer),
    )
    windv2 = get_ext_key_or_default(
        transformer,
        "WINDV2",
        PSY.get_secondary_turns_ratio(transformer),
    )
    windv3 = get_ext_key_or_default(
        transformer,
        "WINDV3",
        PSY.get_tertiary_turns_ratio(transformer),
    )

    if windv1 isa Number && windv2 isa Number && windv3 isa Number &&
       isapprox(windv1, primary_base_voltage) &&
       isapprox(windv2, secondary_base_voltage) &&
       isapprox(windv3, tertiary_base_voltage)
        return (2, 2, 1)
    end

    return (PSSE_DEFAULT, PSSE_DEFAULT, PSSE_DEFAULT)
end
_get_cw_cz_cm_defaults(::PSY.TwoWindingTransformer) =
    (PSSE_DEFAULT, PSSE_DEFAULT, PSSE_DEFAULT)
_get_cw_cz_cm_defaults(::PSY.ThreeWindingTransformer) =
    (PSSE_DEFAULT, PSSE_DEFAULT, PSSE_DEFAULT)

"""Write the third record line (winding 1 data) for a 2-winding transformer."""
function _write_2w_transformer_record3_winding1!(
    io::IO,
    exporter::PSSEExporter,
    transformer::PSY.TwoWindingTransformer,
)
    WINDV1 = get_ext_key_or_default(
        transformer,
        "WINDV1",
        _get_windv1_default(transformer),
    )
    NOMV1 = get_ext_key_or_default(
        transformer,
        "NOMV1",
        PSY.get_base_voltage_primary(transformer),
    )

    ANG1 = _get_2w_ang1(transformer)
    COD1 = _get_2w_cod1(transformer)

    RMA1 = get_ext_key_or_default(transformer, "RMA1")
    RMI1 = get_ext_key_or_default(transformer, "RMI1")
    VMA1 = get_ext_key_or_default(transformer, "VMA1")
    VMI1 = get_ext_key_or_default(transformer, "VMI1")
    # Set to default/actual value to integer, since by default is getting exported as float, causing an error when loaded into PSSE
    NTP1 = _permissive_parse_int(get_ext_key_or_default(transformer, "NTP1"))
    NOD1 = get_ext_key_or_default(transformer, "NOD1")
    CONT1 = get_ext_key_or_default(transformer, "CONT1")

    supp_attr = PSY.get_supplemental_attributes(transformer)
    TAB1 = !isempty(supp_attr) ? PSY.get_table_number(supp_attr[1]) : 0
    CR1 = get_ext_key_or_default(transformer, "CR1")
    CX1 = get_ext_key_or_default(transformer, "CX1")
    CNXA1 = get_ext_key_or_default(transformer, "CNXA1")

    if exporter.psse_version == :v35
        # Using 0.0 as default for rating exporter, since PSSEv35 does not allow blank values
        RATA1, RATB1, RATC1 =
            with_units_base(exporter.system, PSY.UnitSystem.NATURAL_UNITS) do
                _value_or_default(PSY.get_rating(transformer), 0.0),
                _value_or_default(PSY.get_rating_b(transformer), 0.0),
                _value_or_default(PSY.get_rating_c(transformer), 0.0)
            end

        rates_1 = [
            get_ext_key_or_default(transformer, "RATE11", RATA1),
            get_ext_key_or_default(transformer, "RATE12", RATB1),
            get_ext_key_or_default(transformer, "RATE13", RATC1),
        ]
        for i in 4:12
            push!(rates_1, get_ext_key_or_default(transformer, "RATE1$i", 0.0))
        end

        @fastprintdelim_unroll(io, false, WINDV1, NOMV1, ANG1)
        for rate in rates_1
            fastprintdelim(io, rate)
        end
        @fastprintdelim_unroll(
            io,
            true,
            COD1,
            CONT1,
            NOD1,
            RMA1,
            RMI1,
            VMA1,
            VMI1,
            NTP1,
            TAB1,
            CR1,
            CX1,
            CNXA1
        )
    else
        RATA1, RATB1, RATC1 =
            with_units_base(exporter.system, PSY.UnitSystem.NATURAL_UNITS) do
                _value_or_default(PSY.get_rating(transformer), PSSE_DEFAULT),
                _value_or_default(PSY.get_rating_b(transformer), PSSE_DEFAULT),
                _value_or_default(PSY.get_rating_c(transformer), PSSE_DEFAULT)
            end
        @fastprintdelim_unroll(io, true, WINDV1, NOMV1, ANG1, RATA1,
            RATB1, RATC1, COD1, CONT1, RMA1, RMI1,
            VMA1, VMI1, NTP1, TAB1, CR1, CX1, CNXA1)
    end
end

"""Write the fourth record line (winding 2 data) for a 2-winding transformer."""
function _write_2w_transformer_record4_winding2!(
    io::IO,
    transformer::PSY.TwoWindingTransformer,
)
    WINDV2 = get_ext_key_or_default(
        transformer,
        "WINDV2",
        _get_windv2_default(transformer),
    )
    NOMV2 = get_ext_key_or_default(
        transformer,
        "NOMV2",
        PSY.get_base_voltage_secondary(transformer),
    )
    @fastprintdelim_unroll(io, true, WINDV2, NOMV2)
end

"""Write the second record line (impedance data) for a 3-winding transformer."""
function _write_3w_transformer_record2!(
    io::IO,
    transformer::PSY.ThreeWindingTransformer,
)
    R1_2 = Float64(
        get_ext_key_or_default(transformer, "R1-2", PSY.get_r_12(transformer)),
    )
    X1_2 = get_ext_key_or_default(transformer, "X1-2", PSY.get_x_12(transformer))
    SBASE1_2 = PSY.get_base_power_12(transformer)
    R2_3 = get_ext_key_or_default(transformer, "R2-3", PSY.get_r_23(transformer))
    X2_3 = get_ext_key_or_default(transformer, "X2-3", PSY.get_x_23(transformer))
    SBAS2_3 = PSY.get_base_power_23(transformer)
    R3_1 = get_ext_key_or_default(transformer, "R3-1", PSY.get_r_13(transformer))
    X3_1 = get_ext_key_or_default(transformer, "X3-1", PSY.get_x_13(transformer))
    SBAS3_1 = PSY.get_base_power_13(transformer)
    VMSTAR = get_ext_key_or_default(transformer, "VMSTAR")
    ANSTAR = get_ext_key_or_default(transformer, "ANSTAR")

    @fastprintdelim_unroll(io, true, R1_2, X1_2, SBASE1_2, R2_3,
        X2_3, SBAS2_3, R3_1, X3_1, SBAS3_1, VMSTAR, ANSTAR
    )
end

_get_3w_ang(acc, t::PSY.PhaseShiftingTransformer3W) =
    _psse_round_val(rad2deg(acc.get_angle(t)))
_get_3w_ang(acc, t::PSY.Transformer3W) =
    get(WINDING_GROUP_NUMBER_TO_DEGREES, acc.get_group_number(t).value, PSSE_DEFAULT)
_get_3w_ang(_, ::PSY.ThreeWindingTransformer) = 0.0

"""Collect winding data for a 3-winding transformer."""
function _collect_3w_winding_data(
    exporter::PSSEExporter,
    transformer::PSY.ThreeWindingTransformer,
)
    winding_data = Tuple[]
    for (category, prefix) in WINDING_CATEGORIES
        acc = WINDING_ACCESSORS[category]
        NOMV = get_ext_key_or_default(
            transformer,
            "NOMV$prefix",
            acc.get_base_voltage(transformer),
        )
        WINDV = get_ext_key_or_default(
            transformer,
            "WINDV$prefix",
            acc.get_turns_ratio(transformer),
        )
        ANG = _get_3w_ang(acc, transformer)
        RAT = acc.get_rating(transformer)

        if exporter.psse_version == :v35
            rates = [
                get_ext_key_or_default(transformer, "RATE$(prefix)1", RAT),
                get_ext_key_or_default(transformer, "RATE$(prefix)2", RAT),
                get_ext_key_or_default(transformer, "RATE$(prefix)3", RAT),
            ]
            for i in 4:12
                push!(
                    rates,
                    get_ext_key_or_default(transformer, "RATE$(prefix)$i", 0.0),
                )
            end
            RATES = tuple(rates...)
        else
            RATA = get_ext_key_or_default(transformer, "RATA$prefix", RAT)
            RATB = get_ext_key_or_default(transformer, "RATB$prefix", RAT)
            RATC = get_ext_key_or_default(transformer, "RATC$prefix", RAT)
            RATES = (RATA, RATB, RATC)
        end

        COD = get_ext_key_or_default(transformer, "COD$prefix")
        CONT = get_ext_key_or_default(transformer, "CONT$prefix")
        NOD = get_ext_key_or_default(transformer, "NOD$prefix")
        RMA = get_ext_key_or_default(transformer, "RMA$prefix")
        RMI = get_ext_key_or_default(transformer, "RMI$prefix")
        VMA = get_ext_key_or_default(transformer, "VMA$prefix")
        VMI = get_ext_key_or_default(transformer, "VMI$prefix")
        # Set to default/actual value to integer, since by default is getting exported as float, causing an error when loaded into PSSE
        NTP = _permissive_parse_int(get_ext_key_or_default(transformer, "NTP$prefix"))
        TAB = 0
        supp_attr = PSY.get_supplemental_attributes(transformer)
        for icd_tr in supp_attr
            if PSY.get_transformer_winding(icd_tr) == category
                TAB = !isempty(supp_attr) ? PSY.get_table_number(icd_tr) : 0
            end
        end
        CR = get_ext_key_or_default(transformer, "CR$prefix")
        CX = get_ext_key_or_default(transformer, "CX$prefix")
        CNXA = get_ext_key_or_default(transformer, "CNXA$prefix")

        if exporter.psse_version == :v35
            push!(
                winding_data,
                (
                    WINDV, NOMV, ANG, RATES..., COD, CONT, NOD,
                    RMA, RMI, VMA, VMI, NTP, TAB, CR, CX, CNXA,
                ),
            )
        else
            push!(
                winding_data,
                (
                    WINDV, NOMV, ANG, RATES..., COD, CONT,
                    RMA, RMI, VMA, VMI, NTP, TAB, CR, CX, CNXA,
                ),
            )
        end
    end
    return winding_data
end

"""Write winding records for a 3-winding transformer."""
function _write_3w_winding_records!(
    io::IO,
    exporter::PSSEExporter,
    winding_data::Vector{<:Tuple},
)
    for wd in winding_data
        if exporter.psse_version == :v35
            @fastprintdelim_unroll(io, true,
                wd[1], wd[2], wd[3], wd[4], wd[5], wd[6], wd[7], wd[8], wd[9],
                wd[10], wd[11], wd[12], wd[13], wd[14], wd[15], wd[16], wd[17],
                wd[18],
                wd[19], wd[20], wd[21], wd[22], wd[23], wd[24], wd[25], wd[26],
                wd[27]
            )
        else
            @fastprintdelim_unroll(io, true,
                wd[1], wd[2], wd[3], wd[4], wd[5], wd[6], wd[7], wd[8], wd[9],
                wd[10], wd[11], wd[12], wd[13], wd[14], wd[15], wd[16], wd[17]
            )
        end
    end
end

# Fetch PL, QL, IP, IQ, YP, YQ
_psse_get_load_data(
    exporter::PSSEExporter,
    load::Union{PSY.StandardLoad, PSY.InterruptibleStandardLoad},
) =
    with_units_base(exporter.system, PSY.UnitSystem.NATURAL_UNITS) do
        _psse_round_val(PSY.get_constant_active_power(load)),
        _psse_round_val(PSY.get_constant_reactive_power(load)),
        _psse_round_val(PSY.get_current_active_power(load)),
        _psse_round_val(PSY.get_current_reactive_power(load)),
        _psse_round_val(PSY.get_impedance_active_power(load)),
        _psse_round_val(PSY.get_impedance_reactive_power(load))
    end

# Fallback if not all the data is available
# This mapping corresponds to `function make_power_load` in the parser
_psse_get_load_data(exporter::PSSEExporter, load::PSY.StaticLoad) =
    with_units_base(exporter.system, PSY.UnitSystem.NATURAL_UNITS) do
        PSY.get_active_power(load),
        PSY.get_reactive_power(load),
        PSSE_DEFAULT,
        PSSE_DEFAULT,
        PSSE_DEFAULT,
        PSSE_DEFAULT
    end

_psse_interruptible(::PSY.ControllableLoad) = 1
_psse_interruptible(::PSY.StaticLoad) = 0

"""
WRITTEN TO SPEC: PSS/E 33.3/35.4 POM 5.2.1 Load Data
"""
function write_to_buffers!(
    exporter::PSSEExporter,
    ::Val{Symbol("Load Data")},
)
    io = exporter.raw_buffer
    md = exporter.md_dict

    check_supported_version(exporter)

    write_v35_header(io, exporter, "Load Data")

    loads = get!(exporter.components_cache, "loads") do
        sort!(collect(PSY.get_components(PSY.StaticLoad, exporter.system)); by = PSY.get_name)
    end
    load_name_mapping = get!(exporter.components_cache, "load_name_mapping") do
        create_component_ids(
            convert_empty_stringvec(PSY.get_name.(loads)),
            PSY.get_number.(PSY.get_bus.(loads));
            singles_to_1 = true,
        )
    end
    for load in loads
        sienna_bus_number = PSY.get_number(PSY.get_bus(load))
        I = md["bus_number_mapping"][sienna_bus_number]
        ID = _psse_quote_string(load_name_mapping[(sienna_bus_number, PSY.get_name(load))])
        STATUS = PSY.get_available(load) ? 1 : 0
        if !isnothing(PSY.get_area(PSY.get_bus(load)))
            AREA = _permissive_parse_int(PSY.get_name(PSY.get_area(PSY.get_bus(load))))
        else
            AREA = PSSE_DEFAULT
        end
        if !isnothing(PSY.get_load_zone(PSY.get_bus(load)))
            ZONE = _permissive_parse_int(PSY.get_name(PSY.get_load_zone(PSY.get_bus(load))))
        else
            ZONE = PSSE_DEFAULT
        end
        PL, QL, IP, IQ, YP, YQ = _psse_get_load_data(exporter, load)
        OWNER = PSSE_DEFAULT  # defaults to bus's owner
        load_conformity = PSY.get_conformity(load)
        SCALE = load_conformity == PSY.LoadConformity.CONFORMING ? 1 : 0
        INTRPT = _psse_interruptible(load)

        if exporter.psse_version == :v35
            DGENP = get_ext_key_or_default(load, "DGENP")
            DGENQ = get_ext_key_or_default(load, "DGENQ")
            DGENF = get_ext_key_or_default(load, "DGENF")
            LOAD_TYPE = get_ext_key_or_default(load, "LOADTYPE")

            @fastprintdelim_unroll(io, true, I, ID, STATUS, AREA, ZONE,
                PL, QL, IP, IQ, YP, YQ, OWNER,
                SCALE, INTRPT, DGENP, DGENQ, DGENF, LOAD_TYPE)
        else
            @fastprintdelim_unroll(io, true, I, ID, STATUS, AREA, ZONE,
                PL, QL, IP, IQ, YP, YQ, OWNER,
                SCALE, INTRPT)
        end
    end
    end_group(io, md, exporter, "Load Data", true)
    exporter.md_valid ||
        (md["load_name_mapping"] = serialize_component_ids(load_name_mapping))
end

"""
WRITTEN TO SPEC: PSS/E 33.3/35.4 POM 5.2.1 Fixed Bus Shunt Data
"""
function write_to_buffers!(
    exporter::PSSEExporter,
    ::Val{Symbol("Fixed Shunt Data")},
)
    io = exporter.raw_buffer
    md = exporter.md_dict

    check_supported_version(exporter)
    write_v35_header(io, exporter, "Fixed Shunt Data")

    shunts = get!(exporter.components_cache, "shunts") do
        sort!(
            collect(PSY.get_components(PSY.FixedAdmittance, exporter.system));
            by = PSY.get_name,
        )
    end
    shunt_name_mapping = get!(exporter.components_cache, "shunt_name_mapping") do
        create_component_ids(
            convert_empty_stringvec(PSY.get_name.(shunts)),
            PSY.get_number.(PSY.get_bus.(shunts));
            singles_to_1 = true,
        )
    end
    for shunt in shunts
        sienna_bus_number = PSY.get_number(PSY.get_bus(shunt))
        I = md["bus_number_mapping"][sienna_bus_number]
        ID =
            _psse_quote_string(shunt_name_mapping[(sienna_bus_number, PSY.get_name(shunt))])
        STATUS = PSY.get_available(shunt) ? 1 : 0
        GL = real(PSY.get_Y(shunt)) * PSY.get_base_power(exporter.system)
        BL = imag(PSY.get_Y(shunt)) * PSY.get_base_power(exporter.system)

        @fastprintdelim_unroll(io, true, I, ID, STATUS, GL, BL)
    end
    end_group(io, md, exporter, "Fixed Shunt Data", true)
    exporter.md_valid ||
        (md["shunt_name_mapping"] = serialize_component_ids(shunt_name_mapping))
end

# ============================================================================
# Helper functions for Generator Data export
# ============================================================================

"""Compute generator active and reactive power considering HVDC scaling."""
function _compute_generator_powers(
    exporter::PSSEExporter,
    generator::PSY.StaticInjection,
    hvdc_end::Union{String, Nothing},
    base_power::Float64,
)
    return with_units_base(exporter.system, PSY.UnitSystem.NATURAL_UNITS) do
        pg, qg = get_active_and_reactive_power_from_generator(generator)
        gen_sign = hvdc_end == "TO" ? -1.0 : 1.0
        if hvdc_end !== nothing
            pg *= gen_sign * base_power
            qg *= base_power
        end
        pg, qg
    end
end

"""Compute reactive power limits considering HVDC scaling."""
function _compute_reactive_power_limits(
    exporter::PSSEExporter,
    generator::PSY.StaticInjection,
    hvdc_end::Union{String, Nothing},
    base_power::Float64,
)
    return with_units_base(
        () -> begin
            limits = get_reactive_power_limits_for_power_flow(generator)
            if hvdc_end !== nothing
                scaled_limits = (
                    min = limits.min * base_power,
                    max = limits.max * base_power,
                )
                return scaled_limits
            end
            return limits
        end,
        exporter.system,
        PSY.UnitSystem.NATURAL_UNITS,
    )
end

"""Compute active power limits considering HVDC scaling."""
function _compute_active_power_limits(
    exporter::PSSEExporter,
    generator::PSY.StaticInjection,
    hvdc_end::Union{String, Nothing},
    base_power::Float64,
)
    return with_units_base(
        () -> begin
            limits = get_active_power_limits_for_power_flow(generator)
            if hvdc_end !== nothing
                scaled_limits = (
                    min = limits.min * base_power,
                    max = limits.max * base_power,
                )
                return scaled_limits
            end
            return limits
        end,
        exporter.system,
        PSY.UnitSystem.NATURAL_UNITS,
    )
end

"""Write generator record for PSS/E v35 format."""
function _write_generator_v35_record!(
    io::IO,
    I::Int,
    ID::String,
    PG::Float64,
    QG::Float64,
    QT,
    QB,
    VS::Float64,
    IREG,
    NREG,
    MBASE::Float64,
    ZR,
    ZX,
    RT,
    XT,
    GTAP,
    STAT::Int,
    RMPCT,
    PT,
    PB,
    BASLOD,
    WMOD,
    WPF,
)
    @fastprintdelim_unroll(io, false, I, ID, PG, QG, QT, QB,
        VS, IREG, NREG, MBASE, ZR, ZX,
        RT, XT, GTAP, STAT, RMPCT,
        PT, PB, BASLOD)
    fastprintdelim_psse_default_ownership(io)
    @fastprintdelim_unroll(io, true, WMOD, WPF)
end

"""Write generator record for PSS/E v33 format."""
function _write_generator_v33_record!(
    io::IO,
    I::Int,
    ID::String,
    PG::Float64,
    QG::Float64,
    QT,
    QB,
    VS::Float64,
    IREG,
    MBASE::Float64,
    ZR,
    ZX,
    RT,
    XT,
    GTAP,
    STAT::Int,
    RMPCT,
    PT,
    PB,
    WMOD,
    WPF,
)
    @fastprintdelim_unroll(io, false, I, ID, PG, QG, QT, QB,
        VS, IREG, MBASE, ZR, ZX,
        RT, XT, GTAP, STAT, RMPCT,
        PT, PB)
    fastprintdelim_psse_default_ownership(io)
    @fastprintdelim_unroll(io, true, WMOD, WPF)
end

function _warn_finite_default(val::Number; field_name::String, component_name::String)
    isfinite(val) && return val
    if val == Inf
        newval = PSSE_INFINITY
    elseif val == -Inf
        newval = -PSSE_INFINITY
    elseif isnan(val)
        newval = PSSE_DEFAULT
    else
        error("Should be unreachable")
    end
    @warn "Detected non-finite value $field_name = $val for $component_name, using '$newval'"
    return newval
end

"""
Create a synthetic generator (`PSY.ThermalStandard`) representing one end of a TwoTerminalGenericHVDCLine
for export purposes. The generator is initialized with parameters reflecting the HVDC line's state.
# Notes
    - The generator's name is constructed as "<hvdc_line_name>_<suffix>".
    - The `ext` field includes `"HVDC_END"` to indicate the end ("FR"/"TO").
"""
function _make_gens_from_hvdc(
    hvdc_line::PSY.TwoTerminalGenericHVDCLine,
    suffix::String,
    bus::PSY.ACBus,
    active_power::Float64,
    rating::Float64,
    active_power_limits::NamedTuple{(:min, :max), Tuple{Float64, Float64}},
    reactive_power_limits::NamedTuple{(:min, :max), Tuple{Float64, Float64}},
    exporter::PSSEExporter,
)
    return PSY.ThermalStandard(;
        name = "$(PSY.get_name(hvdc_line))_$suffix",
        available = PSY.get_available(hvdc_line) ? 1 : 0,
        status = true,
        bus = bus,
        active_power = active_power,
        reactive_power = 0.0,
        rating = rating,
        active_power_limits = active_power_limits,
        reactive_power_limits = reactive_power_limits,
        ramp_limits = (up = 0.0, down = 0.0),
        operation_cost = PSY.ThermalGenerationCost(
            PSY.CostCurve(PSY.LinearCurve(0.0)),
            0.0, 0.0, 0.0,
        ),
        base_power = PSY.get_base_power(exporter.system),
        ext = Dict{String, Any}(
            "HVDC_END" => suffix,
        ),
    )
end

"""
Update the parameters of synthetic generators created from HVDC lines,
so they reflect the current setpoints and limits of the HVDC devices in the system.
"""
function _update_gens_from_hvdc!(
    synthetic_gens::Vector{PSY.ThermalStandard},
    gen_to_hvdc_map::Dict{
        PSY.ThermalStandard,
        Tuple{PSY.TwoTerminalGenericHVDCLine, String},
    },
    exporter,
)
    for gen in synthetic_gens
        hvdc_line, suffix = gen_to_hvdc_map[gen]
        bus = if suffix == "FR"
            PSY.get_from(PSY.get_arc(hvdc_line))
        else
            PSY.get_to(PSY.get_arc(hvdc_line))
        end
        gen.available = PSY.get_available(hvdc_line) ? 1 : 0
        gen.status = gen.available == 1
        gen.bus = bus
        gen.active_power = PSY.get_active_power_flow(hvdc_line)
        gen.rating = if suffix == "FR"
            PSY.get_active_power_limits_from(hvdc_line).max
        else
            PSY.get_active_power_limits_to(hvdc_line).max
        end
        gen.active_power_limits = if suffix == "FR"
            PSY.get_active_power_limits_from(hvdc_line)
        else
            PSY.get_active_power_limits_to(hvdc_line)
        end
        gen.reactive_power_limits = if suffix == "FR"
            PSY.get_reactive_power_limits_from(hvdc_line)
        else
            PSY.get_reactive_power_limits_to(hvdc_line)
        end
        gen.base_power = PSY.get_base_power(exporter.system)
    end
end

"""Build the full generator list including optional sources, storages, condensers, and HVDC synthetics."""
function _build_generator_list(exporter::PSSEExporter, md::OrderedDict{String, Any})
    temp_gens::Vector{PSY.StaticInjection} = sort!(
        collect(PSY.get_components(PSY.Generator, exporter.system));
        by = PSY.get_name,
    )
    get(md["export_settings"], "sources_as_generators", false) && append!(
        temp_gens,
        sort!(
            collect(PSY.get_components(PSY.Source, exporter.system));
            by = PSY.get_name,
        ),
    )
    get(md["export_settings"], "storages_as_generators", false) && append!(
        temp_gens,
        sort!(
            collect(PSY.get_components(PSY.Storage, exporter.system));
            by = PSY.get_name,
        ),
    )
    append!(
        temp_gens,
        sort!(
            collect(PSY.get_components(PSY.SynchronousCondenser, exporter.system));
            by = PSY.get_name,
        ),
    )

    hvdc_lines =
        collect(PSY.get_components(PSY.TwoTerminalGenericHVDCLine, exporter.system))
    synthetic_gens = Vector{PSY.ThermalStandard}()
    gen_to_hvdc_map =
        Dict{PSY.ThermalStandard, Tuple{PSY.TwoTerminalGenericHVDCLine, String}}()

    if !isempty(hvdc_lines)
        @warn "Found $(length(hvdc_lines)) TwoTerminalGenericHVDCLine components. These will be exported as generators at each end of the DC line."
        for hvdc_line in hvdc_lines
            from_bus = PSY.get_from(PSY.get_arc(hvdc_line))
            to_bus = PSY.get_to(PSY.get_arc(hvdc_line))

            gen_fr = _make_gens_from_hvdc(
                hvdc_line, "FR", from_bus,
                PSY.get_active_power_flow(hvdc_line),
                PSY.get_active_power_limits_from(hvdc_line).max,
                PSY.get_active_power_limits_from(hvdc_line),
                PSY.get_reactive_power_limits_from(hvdc_line),
                exporter,
            )
            gen_to = _make_gens_from_hvdc(
                hvdc_line, "TO", to_bus,
                PSY.get_active_power_flow(hvdc_line),
                PSY.get_active_power_limits_to(hvdc_line).max,
                PSY.get_active_power_limits_to(hvdc_line),
                PSY.get_reactive_power_limits_to(hvdc_line),
                exporter,
            )
            push!(synthetic_gens, gen_fr)
            push!(synthetic_gens, gen_to)
            gen_to_hvdc_map[gen_fr] = (hvdc_line, "FR")
            gen_to_hvdc_map[gen_to] = (hvdc_line, "TO")
        end
    end

    _update_gens_from_hvdc!(synthetic_gens, gen_to_hvdc_map, exporter)
    append!(temp_gens, synthetic_gens)
    return temp_gens
end

"""
WRITTEN TO SPEC: PSS/E 33.3/35.4 POM 5.2.1 Generator Data
"""
function write_to_buffers!(
    exporter::PSSEExporter,
    ::Val{Symbol("Generator Data")},
)
    io = exporter.raw_buffer
    md = exporter.md_dict

    check_supported_version(exporter)
    write_v35_header(io, exporter, "Generator Data")

    generators = get!(exporter.components_cache, "generators") do
        _build_generator_list(exporter, md)
    end

    generator_name_mapping = get!(exporter.components_cache, "generator_name_mapping") do
        generators_by_bus = Dict{Int, Vector{Tuple{String, Int}}}()
        for (i, generator) in enumerate(generators)
            bus_num = PSY.get_number(PSY.get_bus(generator))
            if !haskey(generators_by_bus, bus_num)
                generators_by_bus[bus_num] = []
            end
            push!(generators_by_bus[bus_num], (PSY.get_name(generator), i))
        end

        # Create mapping with sequential numbering per bus
        mapping = Dict{Tuple{Int, String}, String}()
        for (bus_num, gens_on_bus) in generators_by_bus
            if length(gens_on_bus) == 1
                # Single generator on bus gets ID "1".
                gen_name = gens_on_bus[1][1]
                mapping[(bus_num, gen_name)] = "1"
            else
                # The mapping ensures that each generator (with synthetic ones) on a bus gets a unique sequential ID
                for (idx, (gen_name, _)) in enumerate(gens_on_bus)
                    mapping[(bus_num, gen_name)] = string(idx)
                end
            end
        end
        mapping
    end

    # Sorting of generators after including synthetic gens from the generic HVDC.
    sort!(
        generators;
        by = x -> (
            PSY.get_number(PSY.get_bus(x)),
            generator_name_mapping[(PSY.get_number(PSY.get_bus(x)), PSY.get_name(x))],
        ),
    )

    base_power = PSY.get_base_power(exporter.system)
    for generator in generators
        sienna_bus_number = PSY.get_number(PSY.get_bus(generator))
        hvdc_end = get_ext_key_or_default(generator, "HVDC_END", nothing)
        I = md["bus_number_mapping"][sienna_bus_number]
        ID =
            _psse_quote_string(
                generator_name_mapping[(sienna_bus_number, PSY.get_name(generator))],
            )

        # Compute powers and limits
        PG, QG = _compute_generator_powers(exporter, generator, hvdc_end, base_power)
        reactive_power_limits =
            _compute_reactive_power_limits(exporter, generator, hvdc_end, base_power)
        active_power_limits =
            _compute_active_power_limits(exporter, generator, hvdc_end, base_power)

        QT = _warn_finite_default(
            reactive_power_limits.max;
            field_name = "QT",
            component_name = PSY.get_name(generator),
        )
        QB = _warn_finite_default(
            reactive_power_limits.min;
            field_name = "QB",
            component_name = PSY.get_name(generator),
        )
        PT = _warn_finite_default(
            active_power_limits.max;
            field_name = "PT",
            component_name = PSY.get_name(generator),
        )
        PB = _warn_finite_default(
            active_power_limits.min;
            field_name = "PB",
            component_name = PSY.get_name(generator),
        )

        # Get common fields
        VS = PSY.get_magnitude(PSY.get_bus(generator))
        IREG = get_ext_key_or_default(generator, "IREG")
        MBASE = PSY.get_base_power(generator)
        ZR = get_ext_key_or_default(generator, "r")
        ZX = get_ext_key_or_default(generator, "x")
        RT = get_ext_key_or_default(generator, "rt")
        XT = get_ext_key_or_default(generator, "xt")
        GTAP = get_ext_key_or_default(generator, "GTAP")
        STAT = PSY.get_available(generator) ? 1 : 0
        RMPCT = get_ext_key_or_default(generator, "RMPCT")
        WMOD = get_ext_key_or_default(generator, "WMOD")
        WPF = get_ext_key_or_default(generator, "WPF")

        if exporter.psse_version == :v35
            NREG = get_ext_key_or_default(generator, "NREG")
            BASLOD = get_ext_key_or_default(generator, "BASLOD")
            _write_generator_v35_record!(
                io, I, ID, PG, QG, QT, QB, VS, IREG, NREG, MBASE, ZR, ZX,
                RT, XT, GTAP, STAT, RMPCT, PT, PB, BASLOD, WMOD, WPF,
            )
        else
            _write_generator_v33_record!(
                io, I, ID, PG, QG, QT, QB, VS, IREG, MBASE, ZR, ZX,
                RT, XT, GTAP, STAT, RMPCT, PT, PB, WMOD, WPF,
            )
        end
    end
    end_group(io, md, exporter, "Generator Data", true)
    exporter.md_valid ||
        (md["generator_name_mapping"] = serialize_component_ids(generator_name_mapping))
end

# Helpers for branch writing will be added before the transformer section

"""
Collects all AC branches (Line, MonitoredLine, DiscreteControlledACBranch) from the system,
sorts them by their bus numbers, and returns a vector of tuples (branch, bus_numbers).

# Arguments
- `exporter::PSSEExporter`: The exporter containing the system.

# Returns
- `Vector{Tuple{<:PSY.Branch, Tuple}}`: Each tuple contains a branch and its associated bus numbers.
"""
function get_branches_with_numbers(exporter::PSSEExporter)
    lines = collect(PSY.get_components(PSY.Line, exporter.system))
    mon_lines = collect(PSY.get_components(PSY.MonitoredLine, exporter.system))
    discrete_ac_branches =
        collect(PSY.get_components(PSY.DiscreteControlledACBranch, exporter.system))

    # Merge all branch variables into a single vector
    branches = vcat(lines, mon_lines, discrete_ac_branches)
    # Sort branches by their bus numbers to order them at exporting
    sort!(branches; by = branch_to_bus_numbers)
    # Pair each branch with its bus numbers
    return [(branch, branch_to_bus_numbers(branch)) for branch in branches]
end

"""Calculate the STAT field for a 3-winding transformer based on per-winding availability."""
function _calculate_3w_transformer_stat(transformer::PSY.ThreeWindingTransformer)
    primary = PSY.get_available_primary(transformer)
    secondary = PSY.get_available_secondary(transformer)
    tertiary = PSY.get_available_tertiary(transformer)
    # The STAT value is determined based on the availability of the windings
    if (!primary && !secondary) || (!primary && !tertiary) || (!secondary && !tertiary)
        return 0
    elseif !primary
        return 4
    elseif !secondary
        return 2
    elseif !tertiary
        return 3
    else
        return PSY.get_available(transformer) ? 1 : 0
    end
end

"""Write a regular (Line/MonitoredLine) branch record to the buffer."""
function _write_regular_branch_record!(
    io::IO,
    exporter::PSSEExporter,
    I::Int,
    J::Int,
    CKT::String,
    branch::PSY.ACBranch,
)
    ST = PSY.get_available(branch) ? 1 : 0
    MET = get_ext_key_or_default(branch, "MET")
    LEN = get_ext_key_or_default(branch, "LEN")
    R = PSY.get_r(branch)
    X = PSY.get_x(branch)
    B = if branch isa PSY.TapTransformer
        imag(PSY.get_primary_shunt(branch)) * 2
    else
        PSY.get_b(branch).from + PSY.get_b(branch).to
    end
    GI = get_ext_key_or_default(branch, "GI")
    BI = get_ext_key_or_default(branch, "BI")
    GJ = get_ext_key_or_default(branch, "GJ")
    BJ = get_ext_key_or_default(branch, "BJ")

    RATEA, RATEB, RATEC =
        with_units_base(exporter.system, PSY.UnitSystem.NATURAL_UNITS) do
            _value_or_default(PSY.get_rating(branch), PSSE_DEFAULT),
            _value_or_default(PSY.get_rating_b(branch), PSSE_DEFAULT),
            _value_or_default(PSY.get_rating_c(branch), PSSE_DEFAULT)
        end
    (RATEA, RATEB, RATEC) =
        (_fix_3w_transformer_rating(x) for x in (RATEA, RATEB, RATEC))

    if exporter.psse_version == :v35
        NAME = _psse_quote_string(get_ext_key_or_default(branch, "NAME", ""))
        # Using 0.0 as default for rating exporter, since PSSEv35 does not allow blank values
        @fastprintdelim_unroll(io, false, I, J, CKT, R, X, B, NAME)
        fastprintdelim(io, RATEA)
        fastprintdelim(io, RATEB)
        fastprintdelim(io, RATEC)
        for i in 4:12
            fastprintdelim(io, get_ext_key_or_default(branch, "RATE$i", 0.0))
        end

        @fastprintdelim_unroll(io, false, GI, BI, GJ, BJ, ST, MET, LEN)
        fastprintln_psse_default_ownership(io)
    else
        @fastprintdelim_unroll(io, false, I, J, CKT, R, X, B,
            RATEA, RATEB, RATEC,
            GI, BI, GJ, BJ, ST, MET, LEN)
        fastprintln_psse_default_ownership(io)
    end
end

"""Write a DiscreteControlledACBranch record as a non-transformer branch (v33 path)."""
function _write_discrete_branch_record!(
    io::IO,
    exporter::PSSEExporter,
    I::Int,
    J::Int,
    CKT::String,
    branch::PSY.DiscreteControlledACBranch,
    branch_type::PSY.DiscreteControlledBranchType,
)
    ST = PSY.get_available(branch) ? 1 : 0
    MET = get_ext_key_or_default(branch, "MET")
    LEN = get_ext_key_or_default(branch, "LEN")
    R = PSY.get_r(branch)
    X = PSY.get_x(branch)
    B = 0.0
    GI = get_ext_key_or_default(branch, "GI")
    BI = get_ext_key_or_default(branch, "BI")
    GJ = get_ext_key_or_default(branch, "GJ")
    BJ = get_ext_key_or_default(branch, "BJ")

    RATEA, RATEB, RATEC =
        with_units_base(exporter.system, PSY.UnitSystem.NATURAL_UNITS) do
            _value_or_default(PSY.get_rating(branch), PSSE_DEFAULT),
            0.0,
            0.0
        end
    RATEA =
        RATEA >= INFINITE_BOUND ? 0.0 : RATEA / PSY.get_base_power(exporter.system)

    @fastprintdelim_unroll(io, false, I, J, CKT, R, X, B,
        RATEA, RATEB, RATEC, GI, BI,
        GJ, BJ, ST, MET, LEN)
    fastprintln_psse_default_ownership(io)
end

_is_discrete_controlled(::PSY.DiscreteControlledACBranch) = true
_is_discrete_controlled(::PSY.ACBranch) = false

function _export_tap_transformer_as_branch(transformer::PSY.TapTransformer)
    control_obj = PSY.get_control_objective(transformer)
    return (
        control_obj ==
        PSY.TransformerControlObjectiveModule.TransformerControlObjective.UNDEFINED
    ) && isapprox(PSY.get_tap(transformer), 1.0)
end

"""
WRITTEN TO SPEC: PSS/E 33.3/35.4 POM 5.2.1 Non-Transformer Branch Data
"""
function write_to_buffers!(
    exporter::PSSEExporter,
    ::Val{Symbol("Non-Transformer Branch Data")},
)
    io = exporter.raw_buffer
    md = exporter.md_dict

    check_supported_version(exporter)
    write_v35_header(io, exporter, "Non-Transformer Branch Data")

    branches_with_numbers = get!(exporter.components_cache, "branches") do
        regular_branches = get_branches_with_numbers(exporter)

        transformer_as_branches = Tuple{PSY.ACBranch, Tuple{Int, Int}}[]
        for transformer in PSY.get_components(PSY.TapTransformer, exporter.system)
            if _export_tap_transformer_as_branch(transformer)
                bus_nums = branch_to_bus_numbers(transformer)
                push!(transformer_as_branches, (transformer, bus_nums))
            end
        end

        all_branches = vcat(regular_branches, transformer_as_branches)
        sort!(all_branches; by = x -> last(x))

        all_branches
    end

    branch_name_mapping = get!(exporter.components_cache, "branch_name_mapping") do
        create_component_ids(
            convert_empty_stringvec(PSY.get_name.(first.(branches_with_numbers))),
            last.(branches_with_numbers);
            singles_to_1 = false,
        )
    end

    for (branch, (from_n, to_n)) in branches_with_numbers
        # Skip discrete controlled branches for v35 (switches/breakers go to SWITCHING DEVICE DATA section)
        if exporter.psse_version == :v35 && _is_discrete_controlled(branch)
            continue
        end

        I = md["bus_number_mapping"][from_n]
        J = md["bus_number_mapping"][to_n]
        BASE_CKT = branch_name_mapping[((from_n, to_n), PSY.get_name(branch))]

        if _is_discrete_controlled(branch)
            branch_type = PSY.get_discrete_branch_type(branch)
            unquoted_base = strip(BASE_CKT, ['\''])
            CKT = if haskey(DISCRETE_BRANCH_MAP, branch_type)
                char = DISCRETE_BRANCH_MAP[branch_type]
                if occursin("_", unquoted_base)
                    replace(unquoted_base, "_" => char)
                else
                    char * unquoted_base
                end
            else
                @warn "Unknown discrete branch type $branch_type for branch $branch"
                unquoted_base
            end
            _write_discrete_branch_record!(
                io,
                exporter,
                I,
                J,
                _psse_quote_string(CKT),
                branch,
                branch_type,
            )
        else
            _write_regular_branch_record!(
                io,
                exporter,
                I,
                J,
                _psse_quote_string(BASE_CKT),
                branch,
            )
        end
    end
    end_group(io, md, exporter, "Non-Transformer Branch Data", true)
    exporter.md_valid ||
        (md["branch_name_mapping"] = serialize_component_ids(branch_name_mapping))
end

"""
WRITTEN TO SPEC: PSS/E 35.4 POM 5.2.1 System Switching Device Data
"""
function write_to_buffers!(
    exporter::PSSEExporter,
    ::Val{Symbol("Switching Device Data")},
)
    io = exporter.raw_buffer
    md = exporter.md_dict

    check_supported_version(exporter)
    write_v35_header(io, exporter, "Switching Device Data")

    discrete_branches = get!(exporter.components_cache, "discrete_branches") do
        branches_with_numbers = get_branches_with_numbers(exporter)
        filter(
            ((branch, _),) -> _is_discrete_controlled(branch),
            branches_with_numbers,
        )
    end

    branch_name_mapping = get!(exporter.components_cache, "branch_name_mapping") do
        create_component_ids(
            convert_empty_stringvec(PSY.get_name.(first.(discrete_branches))),
            last.(discrete_branches);
            singles_to_1 = false,
        )
    end

    for (branch, (from_n, to_n)) in discrete_branches
        I = md["bus_number_mapping"][from_n]
        J = md["bus_number_mapping"][to_n]

        BASE_CKT = branch_name_mapping[((from_n, to_n), PSY.get_name(branch))]
        branch_type = PSY.get_discrete_branch_type(branch)

        if haskey(DISCRETE_BRANCH_MAP, branch_type)
            char = DISCRETE_BRANCH_MAP[branch_type]
            CKT = if occursin("_", BASE_CKT)
                replace(BASE_CKT, "_" => char)
            else
                char * BASE_CKT
            end
        else
            @warn "Unknown discrete branch type $branch_type for branch $(PSY.get_name(branch))"
            CKT = BASE_CKT
        end
        CKT = _psse_quote_string(CKT)

        X = PSY.get_x(branch)
        RATE1 = with_units_base(exporter.system, PSY.UnitSystem.NATURAL_UNITS) do
            _value_or_default(PSY.get_rating(branch), PSSE_DEFAULT)
        end
        RATE1 = RATE1 >= INFINITE_BOUND ? 0.0 : RATE1 / PSY.get_base_power(exporter.system)

        rates = [RATE1]
        # Using 0.0 as default for rating exporter, since PSSEv35 does not allow blank values
        for i in 2:12
            push!(rates, get_ext_key_or_default(branch, "RATE$i", 0.0))
        end

        STAT = PSY.get_available(branch) ? 1 : 0
        NSTAT = get_ext_key_or_default(branch, "NSTAT")
        MET = get_ext_key_or_default(branch, "MET")

        STYPE = if branch_type == PSY.DiscreteControlledBranchType.BREAKER
            2  # Circuit breaker
        elseif branch_type == PSY.DiscreteControlledBranchType.SWITCH
            3  # Disconnect switch
        else
            1  # Generic connector (default for OTHER)
        end

        NAME = _psse_quote_string(get_ext_key_or_default(branch, "NAME", ""))

        @fastprintdelim_unroll(io, false, I, J, CKT, X)
        for rate in rates
            fastprintdelim(io, rate)
        end
        @fastprintdelim_unroll(io, true, STAT, NSTAT, MET, STYPE, NAME)
    end

    end_group(io, md, exporter, "Switching Device Data", true)
    exporter.md_valid ||
        (md["switching_device_name_mapping"] = serialize_component_ids(branch_name_mapping))
end

"""
Given a vector of Sienna transformer names, create a dictionary from Sienna transformer name
to PSS/E-compatible transformer name. Guarantees determinism and minimal changes.

WRITTEN TO SPEC: PSS/E 33.3/35.4 POM 5.2.1 Transformer Data
"""
function _psse_transformer_names(
    transformers::Vector{String},
    bus_numbers::Vector,
    bus_number_mapping::AbstractDict{Int64, Int64},
    transformer_ckt_mapping,
)
    used_names = Set{String}()
    sizehint!(used_names, length(transformers))
    mapping = OrderedDict{String, String}()
    sizehint!(mapping, length(transformers))

    for original_name in transformers
        if _is_valid_psse_name(original_name)
            mapping[original_name] = original_name
            push!(used_names, original_name)
        end
    end
    for (original_name, bus_tuple) in zip(transformers, bus_numbers)
        haskey(mapping, original_name) && continue

        # Handle both 2-winding and 3-winding tuples
        if length(bus_tuple) == 2
            orig_from, orig_to = bus_tuple
            ckt = transformer_ckt_mapping[((orig_from, orig_to), original_name)]
            new_name = "B$(bus_number_mapping[orig_from])-$(bus_number_mapping[orig_to])_$ckt"
        elseif length(bus_tuple) == 3
            orig_p, orig_s, orig_t = bus_tuple
            ckt = transformer_ckt_mapping[((orig_p, orig_s, orig_t), original_name)]
            new_name = "B$(bus_number_mapping[orig_p])-$(bus_number_mapping[orig_s])-$(bus_number_mapping[orig_t])_$ckt"
        else
            error("Unsupported bus tuple length: $(length(bus_tuple))")
        end

        while new_name in used_names
            new_name *= "-"
        end
        if !_is_valid_psse_name(new_name)
            n = 0
            while !_is_valid_psse_name(new_name) || (new_name in used_names)
                new_name = "B$(bus_number_mapping[bus_tuple[1]])-N$n"
                n += 1
            end
        end
        @assert _is_valid_psse_name(new_name) new_name
        mapping[original_name] = new_name
        push!(used_names, new_name)
    end
    return mapping
end
function _collect_control_objective!(
    mapping::AbstractDict,
    name::String,
    t::Union{PSY.TapTransformer, PSY.PhaseShiftingTransformer},
)
    cod1 = PSY.get_control_objective(t)
    cod1 ==
    PSY.TransformerControlObjectiveModule.TransformerControlObjective.UNDEFINED &&
        (mapping[name] = cod1.value)
    return
end
_collect_control_objective!(::AbstractDict, ::String, ::PSY.TwoWindingTransformer) = nothing

function _collect_winding_group!(
    mapping::AbstractDict,
    name::String,
    t::Union{PSY.TapTransformer, PSY.Transformer2W},
)
    ang1 = PSY.get_winding_group_number(t)
    ang1 == PSY.WindingGroupNumber.UNDEFINED && (mapping[name] = ang1.value)
    return
end
_collect_winding_group!(::AbstractDict, ::String, ::PSY.TwoWindingTransformer) = nothing

"""Build all transformer-related metadata mappings (names, control objectives, winding groups, impedance, taps)."""
function _build_transformer_metadata!(
    md::OrderedDict{String, Any},
    transformers_with_numbers::Vector,
    transformers_3w_with_numbers::Vector,
    transformer_ckt_mapping::Dict,
    transformer_3w_ckt_mapping::Dict,
)
    # Handle 2W transformers
    if !isempty(transformers_with_numbers)
        md["transformer_name_mapping"] = _psse_transformer_names(
            convert_empty_stringvec(PSY.get_name.(first.(transformers_with_numbers))),
            last.(transformers_with_numbers),
            md["bus_number_mapping"],
            transformer_ckt_mapping,
        )
        control_objective_mapping = OrderedDict{String, Any}()
        winding_group_category_mapping = OrderedDict{String, Any}()
        for (transformer, _) in transformers_with_numbers
            name = PSY.get_name(transformer)
            _collect_control_objective!(control_objective_mapping, name, transformer)
            _collect_winding_group!(winding_group_category_mapping, name, transformer)
        end
        md["transformer_control_objective_mapping"] = control_objective_mapping
        md["transformer_winding_group_category_mapping"] = winding_group_category_mapping
    else
        md["transformer_name_mapping"] = OrderedDict{String, String}()
        md["transformer_control_objective_mapping"] = OrderedDict{String, Any}()
        md["transformer_winding_group_category_mapping"] = OrderedDict{String, Any}()
    end

    # Handle 3W transformers
    if !isempty(transformers_3w_with_numbers)
        md["transformer_3w_name_mapping"] = _psse_transformer_names(
            convert_empty_stringvec(
                PSY.get_name.(first.(transformers_3w_with_numbers)),
            ),
            last.(transformers_3w_with_numbers),
            md["bus_number_mapping"],
            transformer_3w_ckt_mapping,
        )
    else
        md["transformer_3w_name_mapping"] = OrderedDict{String, String}()
    end
end

"""
Load transformer components and create circuit ID mappings.

Returns a tuple of:
- transformers_with_numbers: 2-winding transformers with their bus numbers
- transformers_3w_with_numbers: 3-winding transformers with their bus numbers
- transformer_ckt_mapping: Circuit ID mapping for 2-winding transformers
- transformer_3w_ckt_mapping: Circuit ID mapping for 3-winding transformers
"""
function _load_transformer_components_and_mappings(exporter::PSSEExporter)
    transformers_with_numbers = get!(exporter.components_cache, "transformers") do
        transformers = sort!(
            collect(PSY.get_components(PSY.TwoWindingTransformer, exporter.system));
            by = branch_to_bus_numbers,
        )
        # TapTransformers with UNDEFINED control objective and unity tap are exported as branches.
        # Keep all others in transformer export.
        filtered_transformers = filter(transformers) do transformer
            if transformer isa PSY.TapTransformer
                return !_export_tap_transformer_as_branch(transformer)
            end
            return true
        end
        [
            (transformer, branch_to_bus_numbers(transformer)) for
            transformer in filtered_transformers
        ]
    end
    transformers_3w_with_numbers = get!(exporter.components_cache, "transformers_3w") do
        transformers = sort!(
            collect(PSY.get_components(PSY.ThreeWindingTransformer, exporter.system));
            by = branch_to_bus_numbers,
        )
        [(transformer, branch_to_bus_numbers(transformer)) for transformer in transformers]
    end
    transformer_ckt_mapping = get!(exporter.components_cache, "transformer_ckt_mapping") do
        create_component_ids(
            convert_empty_stringvec(PSY.get_name.(first.(transformers_with_numbers))),
            last.(transformers_with_numbers);
            singles_to_1 = false,
        )
    end
    transformer_3w_ckt_mapping =
        get!(exporter.components_cache, "transformer_3w_ckt_mapping") do
            create_component_ids(
                convert_empty_stringvec(
                    PSY.get_name.(first.(transformers_3w_with_numbers)),
                ),
                last.(transformers_3w_with_numbers);
                singles_to_1 = false,
            )
        end

    return (transformers_with_numbers, transformers_3w_with_numbers,
        transformer_ckt_mapping, transformer_3w_ckt_mapping)
end

"""
WRITTEN TO SPEC: PSS/E 33.3/35.4 POM 5.2.1 Transformer Data
"""
function write_to_buffers!(
    exporter::PSSEExporter,
    ::Val{Symbol("Transformer Data")},
)
    io = exporter.raw_buffer
    md = exporter.md_dict

    check_supported_version(exporter)
    write_v35_header(io, exporter, "Transformer Data")

    # Load transformer components and create circuit ID mappings
    (transformers_with_numbers, transformers_3w_with_numbers,
        transformer_ckt_mapping, transformer_3w_ckt_mapping) =
        _load_transformer_components_and_mappings(exporter)
    if !exporter.md_valid
        _build_transformer_metadata!(
            md,
            transformers_with_numbers,
            transformers_3w_with_numbers,
            transformer_ckt_mapping,
            transformer_3w_ckt_mapping,
        )
    end

    bus_number_mapping = md["bus_number_mapping"]
    transformer_name_mapping = md["transformer_name_mapping"]
    transformer_3w_name_mapping = md["transformer_3w_name_mapping"]

    for (transformer, bus_tuple) in
        vcat(transformers_with_numbers, transformers_3w_with_numbers)
        winding_number = length(bus_tuple)
        if winding_number == 2  # Handle 2-winding transformer fields
            from_n, to_n = bus_tuple
            I = bus_number_mapping[from_n]
            J = bus_number_mapping[to_n]
            K = 0
            CKT = transformer_ckt_mapping[((from_n, to_n), PSY.get_name(transformer))]
            if startswith(CKT, "_")
                CKT = CKT[2:end]
            end
            CKT = _psse_quote_string(CKT)
            NAME = _psse_quote_string(transformer_name_mapping[PSY.get_name(transformer)])
            STAT = PSY.get_available(transformer) ? 1 : 0

            # Write record 1 (bus numbers, circuit ID, status, etc.)
            _write_2w_transformer_record1!(
                io,
                exporter,
                I,
                J,
                K,
                CKT,
                transformer,
                NAME,
                STAT,
            )
            # Write record 2 (impedance data)
            _write_2w_transformer_record2!(io, transformer)
            # Write record 3 (winding 1 data)
            _write_2w_transformer_record3_winding1!(io, exporter, transformer)
            # Write record 4 (winding 2 data)
            _write_2w_transformer_record4_winding2!(io, transformer)

        elseif winding_number == 3 # Handle 3-winding transformer fields
            p, s, t = bus_tuple
            I = bus_number_mapping[p]
            J = bus_number_mapping[s]
            K = bus_number_mapping[t]
            CKT = transformer_3w_ckt_mapping[((p, s, t), PSY.get_name(transformer))]
            if startswith(CKT, "_")
                CKT = CKT[2:end]
            end
            CKT = _psse_quote_string(CKT)
            NAME = transformer_3w_name_mapping[PSY.get_name(transformer)]
            NAME = _psse_quote_string(NAME)
            STAT = _calculate_3w_transformer_stat(transformer)

            # Write record 1 (bus numbers, circuit ID, status, etc.)
            _write_2w_transformer_record1!(
                io,
                exporter,
                I,
                J,
                K,
                CKT,
                transformer,
                NAME,
                STAT,
            )
            # Write record 2 (impedance data for 3W)
            _write_3w_transformer_record2!(io, transformer)
            # Collect and write winding records
            winding_data = _collect_3w_winding_data(exporter, transformer)
            _write_3w_winding_records!(io, exporter, winding_data)
        else
            error("Unsupported transformer bus tuple length: $(length(bus_tuple))")
        end
    end

    end_group(io, md, exporter, "Transformer Data", true)
    if !exporter.md_valid
        md["transformer_ckt_mapping"] = serialize_component_ids(transformer_ckt_mapping)
        md["transformer_3w_ckt_mapping"] =
            serialize_component_ids(transformer_3w_ckt_mapping)
    end
end

"""Compute common DC line fields (record 1) for Two-Terminal DC export."""
function _compute_dcline_common_fields(
    exporter::PSSEExporter,
    dcline::PSY.TwoTerminalLCCLine,
    I::Int,
    J::Int,
)
    dcline_name = PSY.get_name(dcline)
    # Using last() since some DC lines share rectifier bus numbers
    NAME = _is_valid_psse_name(dcline_name) ? dcline_name : last(dcline_name, 12)
    NAME = _psse_quote_string(NAME)
    MDC = Int(PSY.get_power_mode(dcline))
    RDC =
        _psse_round_val(
            PSY.get_r(dcline) * PSY.get_rectifier_base_voltage(dcline)^2 /
            PSY.get_base_power(exporter.system),
        )
    SETVL = PSY.get_transfer_setpoint(dcline)
    VSCHD = PSY.get_scheduled_dc_voltage(dcline)
    VCMOD = PSY.get_switch_mode_voltage(dcline)
    RCOMP = PSY.get_compounding_resistance(dcline)
    DELTI = get_ext_key_or_default(dcline, "DELTI")
    METER = PSSE_DEFAULT
    DCVMIN = PSY.get_min_compounding_voltage(dcline)
    CCCITMX = get_ext_key_or_default(dcline, "CCCITMX")
    CCCACC = get_ext_key_or_default(dcline, "CCCACC")
    return (;
        NAME, MDC, RDC, SETVL, VSCHD, VCMOD, RCOMP,
        DELTI, METER, DCVMIN, CCCITMX, CCCACC,
    )
end

"""Compute rectifier-side fields for Two-Terminal DC export."""
function _compute_dcline_rectifier_fields(
    exporter::PSSEExporter,
    dcline::PSY.TwoTerminalLCCLine,
    I::Int,
)
    base_power = PSY.get_base_power(exporter.system)
    IPR = I
    NBR = PSY.get_rectifier_bridges(dcline)
    ANMXR = _psse_round_val(rad2deg(PSY.get_rectifier_delay_angle_limits(dcline).max))
    ANMNR = _psse_round_val(rad2deg(PSY.get_rectifier_delay_angle_limits(dcline).min))
    RCR = _psse_round_val(
        PSY.get_rectifier_rc(dcline) * PSY.get_rectifier_base_voltage(dcline)^2 /
        base_power)
    XCR = _psse_round_val(
        PSY.get_rectifier_xc(dcline) * PSY.get_rectifier_base_voltage(dcline)^2 /
        base_power)
    EBASR = PSY.get_rectifier_base_voltage(dcline)
    TRR = PSY.get_rectifier_transformer_ratio(dcline)
    TAPR = PSY.get_rectifier_tap_setting(dcline)
    TMXR = PSY.get_rectifier_tap_limits(dcline).max
    TMNR = PSY.get_rectifier_tap_limits(dcline).min
    STPR = PSY.get_rectifier_tap_step(dcline)
    ICR = get_ext_key_or_default(dcline, "ICR")
    NDR = get_ext_key_or_default(dcline, "NDR")
    IFR = get_ext_key_or_default(dcline, "IFR")
    ITR = get_ext_key_or_default(dcline, "ITR")
    IDR = get_ext_key_or_default(dcline, "IDR")
    XCAPR =
        PSY.get_rectifier_capacitor_reactance(dcline) *
        PSY.get_rectifier_base_voltage(dcline)^2 /
        base_power
    return (;
        IPR, NBR, ANMXR, ANMNR, RCR, XCR, EBASR,
        TRR, TAPR, TMXR, TMNR, STPR, ICR, NDR, IFR, ITR, IDR, XCAPR,
    )
end

"""Compute inverter-side fields for Two-Terminal DC export."""
function _compute_dcline_inverter_fields(
    exporter::PSSEExporter,
    dcline::PSY.TwoTerminalLCCLine,
    J::Int,
)
    base_power = PSY.get_base_power(exporter.system)
    IPI = J
    NBI = PSY.get_inverter_bridges(dcline)
    ANMXI =
        _psse_round_val(rad2deg(PSY.get_inverter_extinction_angle_limits(dcline).max))
    ANMNI =
        _psse_round_val(rad2deg(PSY.get_inverter_extinction_angle_limits(dcline).min))
    RCI = _psse_round_val(
        PSY.get_inverter_rc(dcline) * PSY.get_inverter_base_voltage(dcline)^2 /
        base_power)
    XCI = _psse_round_val(
        PSY.get_inverter_xc(dcline) * PSY.get_inverter_base_voltage(dcline)^2 /
        base_power)
    EBASI = PSY.get_inverter_base_voltage(dcline)
    TRI = PSY.get_inverter_transformer_ratio(dcline)
    TAPI = PSY.get_inverter_tap_setting(dcline)
    TMXI = PSY.get_inverter_tap_limits(dcline).max
    TMNI = PSY.get_inverter_tap_limits(dcline).min
    STPI = PSY.get_inverter_tap_step(dcline)
    ICI = get_ext_key_or_default(dcline, "ICI")
    NDI = get_ext_key_or_default(dcline, "NDI")
    IFI = get_ext_key_or_default(dcline, "IFI")
    ITI = get_ext_key_or_default(dcline, "ITI")
    IDI = get_ext_key_or_default(dcline, "IDI")
    XCAPI =
        PSY.get_inverter_capacitor_reactance(dcline) *
        PSY.get_inverter_base_voltage(dcline)^2 /
        base_power
    return (;
        IPI, NBI, ANMXI, ANMNI, RCI, XCI, EBASI,
        TRI, TAPI, TMXI, TMNI, STPI, ICI, NDI, IFI, ITI, IDI, XCAPI,
    )
end

"""
WRITTEN TO SPEC: PSS/E 33.3/35.4 POM 5.2.1 Two-Terminal DC Transmission Line Data
"""
function write_to_buffers!(
    exporter::PSSEExporter,
    ::Val{Symbol("Two-Terminal DC Transmission Line Data")},
)
    io = exporter.raw_buffer
    md = exporter.md_dict

    check_supported_version(exporter)
    write_v35_header(io, exporter, "Two-Terminal DC Transmission Line Data")

    dclines_with_numbers = get!(exporter.components_cache, "dclines") do
        dclines = sort!(
            collect(
                PSY.get_components(PSY.TwoTerminalLCCLine, exporter.system),
            );
            by = branch_to_bus_numbers,
        )
        [(dcline, branch_to_bus_numbers(dcline)) for dcline in dclines]
    end
    dcline_name_mapping = get!(exporter.components_cache, "dcline_name_mapping") do
        create_component_ids(
            convert_empty_stringvec(PSY.get_name.(first.(dclines_with_numbers))),
            last.(dclines_with_numbers);
            singles_to_1 = false,
        )
    end

    for (dcline, (from_n, to_n)) in dclines_with_numbers
        I = md["bus_number_mapping"][from_n]
        J = md["bus_number_mapping"][to_n]

        c = _compute_dcline_common_fields(exporter, dcline, I, J)
        r = _compute_dcline_rectifier_fields(exporter, dcline, I)
        inv = _compute_dcline_inverter_fields(exporter, dcline, J)

        @fastprintdelim_unroll(io, false,
            c.NAME, c.MDC, c.RDC, c.SETVL, c.VSCHD, c.VCMOD,
            c.RCOMP, c.DELTI, c.METER, c.DCVMIN, c.CCCITMX)
        fastprintln(io, c.CCCACC)

        if exporter.psse_version == :v35
            @fastprintdelim_unroll(io, false,
                r.IPR, r.NBR, r.ANMXR, r.ANMNR, r.RCR, r.XCR, r.EBASR,
                r.TRR, r.TAPR, r.TMXR, r.TMNR, r.STPR, r.ICR, r.NDR,
                r.IFR, r.ITR, r.IDR)
            fastprintln(io, r.XCAPR)

            @fastprintdelim_unroll(io, false,
                inv.IPI, inv.NBI, inv.ANMXI, inv.ANMNI, inv.RCI, inv.XCI, inv.EBASI,
                inv.TRI, inv.TAPI, inv.TMXI, inv.TMNI, inv.STPI, inv.ICI, inv.NDI,
                inv.IFI, inv.ITI, inv.IDI)
            fastprintln(io, inv.XCAPI)
        else
            @fastprintdelim_unroll(io, false,
                r.IPR, r.NBR, r.ANMXR, r.ANMNR, r.RCR, r.XCR, r.EBASR,
                r.TRR, r.TAPR, r.TMXR, r.TMNR, r.STPR, r.ICR,
                r.IFR, r.ITR, r.IDR)
            fastprintln(io, r.XCAPR)

            @fastprintdelim_unroll(io, false,
                inv.IPI, inv.NBI, inv.ANMXI, inv.ANMNI, inv.RCI, inv.XCI, inv.EBASI,
                inv.TRI, inv.TAPI, inv.TMXI, inv.TMNI, inv.STPI, inv.ICI,
                inv.IFI, inv.ITI, inv.IDI)
            fastprintln(io, inv.XCAPI)
        end
    end
    end_group(io, md, exporter, "Two-Terminal DC Transmission Line Data", true)
    exporter.md_valid ||
        (md["dcline_name_mapping"] = serialize_component_ids(dcline_name_mapping))
end

"""Compute VSC converter fields for one side (from or to) of a VSC DC line."""
function _compute_vsc_converter_fields(
    exporter::PSSEExporter,
    vscline::PSY.TwoTerminalVSCLine,
    bus_number::Int,
    type_org::Int,
    side::Symbol,
)
    base_power = PSY.get_base_power(exporter.system)
    suffix = side == :from ? "FROM" : "TO"

    IBUS = bus_number
    TYPE = get_ext_key_or_default(vscline, "TYPE_$suffix", type_org)

    if side == :from
        MODE = PSY.get_ac_voltage_control_from(vscline) ? 1 : 2
        DCSET = PSY.get_dc_setpoint_from(vscline)
        ACSET = PSY.get_ac_setpoint_from(vscline)
        converter_loss = PSY.get_converter_loss_from(vscline)
        get_rating = PSY.get_rating_from
        get_imax = PSY.get_max_dc_current_from
        PWF = PSY.get_power_factor_weighting_fraction_from(vscline)
        q_limits = PSY.get_reactive_power_limits_from(vscline)
    else
        MODE = PSY.get_ac_voltage_control_to(vscline) ? 1 : 2
        DCSET = PSY.get_dc_setpoint_to(vscline)
        ACSET = PSY.get_ac_setpoint_to(vscline)
        converter_loss = PSY.get_converter_loss_to(vscline)
        get_rating = PSY.get_rating_to
        get_imax = PSY.get_max_dc_current_to
        PWF = PSY.get_power_factor_weighting_fraction_to(vscline)
        q_limits = PSY.get_reactive_power_limits_to(vscline)
    end

    BLOSS = _psse_round_val(
        PSY.get_proportional_term(PSY.get_function_data(converter_loss)) *
        1e3 * base_power)
    psse_converter_loss = _psse_round_val(BLOSS * abs(PSY.get_dc_current(vscline)))
    ALOSS_org = _psse_round_val(
        abs(
            psse_converter_loss -
            PSY.get_constant_term(PSY.get_function_data(converter_loss)) *
            1e3 * base_power,
        ),
    )
    ALOSS = get_ext_key_or_default(vscline, "ALOSS_$suffix", ALOSS_org)
    MINLOSS = get_ext_key_or_default(vscline, "MINLOSS_$suffix", psse_converter_loss)

    SMAX = get_rating(vscline)
    # Revert parser transformation: SMAX == 0.0 ? PSSE_INFINITY : SMAX / baseMVA
    SMAX = if SMAX == PSSE_INFINITY
        0.0
    else
        _psse_round_val(SMAX * base_power)
    end
    IMAX = get_imax(vscline)
    IMAX = IMAX == PSSE_INFINITY ? 0.0 : IMAX

    MAXQ = q_limits.max * base_power
    MINQ = q_limits.min * base_power
    VSREG = get_ext_key_or_default(vscline, "VSREG_$suffix", 0)
    NREG = get_ext_key_or_default(vscline, "NREG_$suffix", 0)
    REMOT = get_ext_key_or_default(vscline, "REMOT_$suffix")
    RMPCT = get_ext_key_or_default(vscline, "RMPCT_$suffix")

    return (;
        IBUS, TYPE, MODE, DCSET, ACSET, ALOSS, BLOSS, MINLOSS,
        SMAX, IMAX, PWF, MAXQ, MINQ, VSREG, NREG, REMOT, RMPCT,
    )
end

"""
WRITTEN TO SPEC: PSS/E 33.3/35.4 POM 5.2.1 Voltage Source Converter (VSC) DC Transmission Line Data
"""
function write_to_buffers!(
    exporter::PSSEExporter,
    ::Val{Symbol("Voltage Source Converter (VSC) DC Transmission Line Data")},
)
    io = exporter.raw_buffer
    md = exporter.md_dict

    check_supported_version(exporter)
    write_v35_header(
        io,
        exporter,
        "Voltage Source Converter (VSC) DC Transmission Line Data",
    )

    vsc_lines_with_numbers = get!(exporter.components_cache, "vsc_lines") do
        vsc_lines = sort!(
            collect(
                PSY.get_components(PSY.TwoTerminalVSCLine, exporter.system),
            );
            by = branch_to_bus_numbers,
        )
        [(vscline, branch_to_bus_numbers(vscline)) for vscline in vsc_lines]
    end
    vsc_line_name_mapping = get!(exporter.components_cache, "vsc_line_name_mapping") do
        create_component_ids(
            convert_empty_stringvec(PSY.get_name.(first.(vsc_lines_with_numbers))),
            last.(vsc_lines_with_numbers);
            singles_to_1 = false,
        )
    end

    for (vscline, (from_n, to_n)) in vsc_lines_with_numbers
        I = md["bus_number_mapping"][from_n]
        J = md["bus_number_mapping"][to_n]
        vsc_line_name = string(split(PSY.get_name(vscline), "_")[end])
        NAME = _psse_quote_string(vsc_line_name)
        MDC = PSY.get_available(vscline) ? 1 : 0
        if PSY.get_dc_voltage_control_from(vscline) &&
           !PSY.get_dc_voltage_control_to(vscline)
            base_voltage = PSY.get_dc_setpoint_from(vscline)
        else
            base_voltage = PSY.get_dc_setpoint_to(vscline)
        end
        Zbase = base_voltage^2 / PSY.get_base_power(exporter.system)
        RDC_org = PSY.get_g(vscline) != 0.0 ? (1 / PSY.get_g(vscline)) * Zbase : 0.0
        RDC = get_ext_key_or_default(vscline, "RDC", RDC_org)

        # Determine converter types based on DC voltage control configuration
        if PSY.get_dc_voltage_control_from(vscline) &&
           !PSY.get_dc_voltage_control_to(vscline)
            type1_org = 1
            type2_org = 2
        elseif !PSY.get_dc_voltage_control_from(vscline) &&
               PSY.get_dc_voltage_control_to(vscline)
            type1_org = 2
            type2_org = 1
        else
            type1_org = 0
            type2_org = 0
        end

        c1 = _compute_vsc_converter_fields(exporter, vscline, I, type1_org, :from)
        c2 = _compute_vsc_converter_fields(exporter, vscline, J, type2_org, :to)

        @fastprintdelim_unroll(io, false, NAME, MDC, RDC)
        fastprintln_psse_default_ownership(io)

        if exporter.psse_version == :v35
            @fastprintdelim_unroll(io, false,
                c1.IBUS, c1.TYPE, c1.MODE, c1.DCSET, c1.ACSET, c1.ALOSS, c1.BLOSS,
                c1.MINLOSS, c1.SMAX, c1.IMAX, c1.PWF, c1.MAXQ, c1.MINQ,
                c1.VSREG, c1.NREG)
            fastprintln(io, c1.RMPCT)

            @fastprintdelim_unroll(io, false,
                c2.IBUS, c2.TYPE, c2.MODE, c2.DCSET, c2.ACSET, c2.ALOSS, c2.BLOSS,
                c2.MINLOSS, c2.SMAX, c2.IMAX, c2.PWF, c2.MAXQ, c2.MINQ,
                c2.VSREG, c2.NREG)
            fastprintln(io, c2.RMPCT)
        else
            @fastprintdelim_unroll(io, false,
                c1.IBUS, c1.TYPE, c1.MODE, c1.DCSET, c1.ACSET, c1.ALOSS, c1.BLOSS,
                c1.MINLOSS, c1.SMAX, c1.IMAX, c1.PWF, c1.MAXQ, c1.MINQ, c1.REMOT)
            fastprintln(io, c1.RMPCT)

            @fastprintdelim_unroll(io, false,
                c2.IBUS, c2.TYPE, c2.MODE, c2.DCSET, c2.ACSET, c2.ALOSS, c2.BLOSS,
                c2.MINLOSS, c2.SMAX, c2.IMAX, c2.PWF, c2.MAXQ, c2.MINQ, c2.REMOT)
            fastprintln(io, c2.RMPCT)
        end
    end
    end_group(
        io,
        md,
        exporter,
        "Voltage Source Converter (VSC) DC Transmission Line Data",
        true,
    )
    exporter.md_valid ||
        (md["vsc_line_name_mapping"] = serialize_component_ids(vsc_line_name_mapping))
end

function _write_icd_y_v35!(io::IO, y::Complex)
    fastprint(io, real(y))
    fastprint(io, ", ")
    fastprint(io, imag(y))
end
function _write_icd_y_v35!(io::IO, y)
    fastprint(io, y)
    fastprint(io, ", ")
    fastprint(io, 0.0)
end

_write_icd_y_v33!(io::IO, y::Complex) = fastprintdelim(io, real(y))
_write_icd_y_v33!(io::IO, y) = fastprintdelim(io, y)

"""Write impedance correction table points in v35 format (T, Re(F), Im(F) triplets)."""
function _write_icd_v35_points!(io::IO, I, points)
    fastprint(io, " ")
    fastprint(io, I)
    point_count = 0
    total_points = length(points)

    for p in points
        if point_count > 0 && point_count % 6 == 0
            fastprintln(io, "")
            fastprint(io, "   ")
        else
            fastprint(io, ", ")
        end
        fastprint(io, p.x)
        fastprint(io, ", ")
        _write_icd_y_v35!(io, p.y)
        point_count += 1
    end

    # Pad with zeros if more than 6 points and last line is incomplete
    if total_points > 6
        remaining_slots = 6 - (point_count % 6)
        if remaining_slots > 0 && remaining_slots < 6
            for _ in 1:remaining_slots
                fastprint(io, ", ")
                fastprint(io, 0.0)
                fastprint(io, ", ")
                fastprint(io, 0.0)
                fastprint(io, ", ")
                fastprint(io, 0.0)
            end
        end
    end
    fastprintln(io, "")
end

"""Write impedance correction table points in v33 format (T, F pairs)."""
function _write_icd_v33_points!(io::IO, I, points)
    fastprint(io, I)
    fastprint(io, ", ")
    for p in points
        fastprintdelim(io, p.x)
        _write_icd_y_v33!(io, p.y)
    end
    fastprintln(io, "")
end

"""
WRITTEN TO SPEC: PSS/E 33.3/35.4 POM 5.2.1 Transformer Impedance Correction Tables
"""
function write_to_buffers!(
    exporter::PSSEExporter,
    ::Val{Symbol("Transformer Impedance Correction Tables")},
)
    io = exporter.raw_buffer
    md = exporter.md_dict

    check_supported_version(exporter)
    write_v35_header(io, exporter, "Transformer Impedance Correction Tables")

    icd_entries = get!(exporter.components_cache, "icd_entries") do
        sort(
            collect(
                PSY.get_supplemental_attributes(
                    PSY.ImpedanceCorrectionData,
                    exporter.system,
                ),
            );
            by = tn -> PSY.get_table_number(tn),
        )
    end

    unique_icd_entries = OrderedDict()
    for icd in icd_entries
        I = PSY.get_table_number(icd)
        unique_icd_entries[I] = icd
    end

    for (I, icd) in unique_icd_entries
        points = PSY.get_points(PSY.get_impedance_correction_curve(icd))
        if exporter.psse_version == :v35
            _write_icd_v35_points!(io, I, points)
        else
            _write_icd_v33_points!(io, I, points)
        end
    end

    end_group(io, md, exporter, "Transformer Impedance Correction Tables", true)
end

"""
WRITTEN TO SPEC: PSS/E 33.3/35.4 POM 5.2.1 Zone Data
"""
function write_to_buffers!(
    exporter::PSSEExporter,
    ::Val{Symbol("Zone Data")},
)
    io = exporter.raw_buffer
    md = exporter.md_dict

    check_supported_version(exporter)
    write_v35_header(io, exporter, "Zone Data")

    zone_mapping = md["zone_mapping"]
    zones = get!(exporter.components_cache, "zones") do
        sort!(
            collect(PSY.get_components(PSY.LoadZone, exporter.system));
            by = x -> zone_mapping[PSY.get_name(x)],
        )
    end
    for zone in zones
        name = PSY.get_name(zone)
        I = zone_mapping[name]
        @assert _is_valid_psse_name(name) name
        ZONAME = _psse_quote_string(name)

        @fastprintdelim_unroll(io, true, I, ZONAME)
    end
    end_group(io, md, exporter, "Zone Data", true)
end

"""
WRITTEN TO SPEC: PSS/E 33.3/35.4 POM 5.2.1 FACTS Device Data
"""
function write_to_buffers!(
    exporter::PSSEExporter,
    ::Val{Symbol("FACTS Device Data")},
)
    io = exporter.raw_buffer
    md = exporter.md_dict

    check_supported_version(exporter)
    write_v35_header(io, exporter, "FACTS Device Data")

    facts_devices = get!(exporter.components_cache, "facts_devices") do
        sort!(
            collect(PSY.get_components(PSY.FACTSControlDevice, exporter.system));
            by = PSY.get_name,
        )
    end
    facts_name_mapping = get!(exporter.components_cache, "facts_name_mapping") do
        create_component_ids(
            convert_empty_stringvec(PSY.get_name.(facts_devices)),
            PSY.get_number.(PSY.get_bus.(facts_devices));
            singles_to_1 = true,
        )
    end

    for facts in facts_devices
        sienna_bus_number = PSY.get_number(PSY.get_bus(facts))
        I = md["bus_number_mapping"][sienna_bus_number]
        J = get_ext_key_or_default(facts, "J")
        name = PSY.get_name(facts)
        if startswith(name, string(sienna_bus_number) * "_")
            name = name[(length(string(sienna_bus_number)) + 2):end]
        end
        NAME = _psse_quote_string(name)
        MODE = get(FACTS_MODE_MAP, PSY.get_control_mode(facts), 2)
        PDES = get_ext_key_or_default(facts, "PDES")
        QDES = get_ext_key_or_default(facts, "QDES")
        VSET = PSY.get_voltage_setpoint(facts)
        SHMX = PSY.get_max_shunt_current(facts)
        TRMX = get_ext_key_or_default(facts, "TRMX", PSSE_INFINITY)
        VTMX = get_ext_key_or_default(facts, "VTMX")
        VTMN = get_ext_key_or_default(facts, "VTMN")
        VSMX = get_ext_key_or_default(facts, "VSMX")
        IMX = get_ext_key_or_default(facts, "IMX")
        LINX = get_ext_key_or_default(facts, "LINX")
        RMPCT = PSY.get_reactive_power_required(facts)
        OWNER = PSSE_DEFAULT
        SET1 = get_ext_key_or_default(facts, "SET1")
        SET2 = get_ext_key_or_default(facts, "SET2")
        VSREF = get_ext_key_or_default(facts, "VSREF")
        FCREG = get_ext_key_or_default(facts, "FCREG")
        NREG = get_ext_key_or_default(facts, "NREG")
        REMOT = get_ext_key_or_default(facts, "REMOT")
        MNAME = get_ext_key_or_default(facts, "MNAME", "")
        MNAME = _psse_quote_string(String(MNAME))

        if exporter.psse_version == :v35
            @fastprintdelim_unroll(io, false, NAME, I, J, MODE, PDES, QDES,
                VSET, SHMX, TRMX, VTMN, VTMX, VSMX, IMX, LINX, RMPCT, OWNER,
                SET1, SET2, VSREF, FCREG, NREG)
            fastprintln(io, MNAME)
        else
            @fastprintdelim_unroll(io, false, NAME, I, J, MODE, PDES, QDES,
                VSET, SHMX, TRMX, VTMN, VTMX, VSMX, IMX, LINX, RMPCT, OWNER,
                SET1, SET2, VSREF, REMOT)
            fastprintln(io, MNAME)
        end
    end
    end_group(io, md, exporter, "FACTS Device Data", true)
    exporter.md_valid ||
        (md["facts_name_mapping"] = serialize_component_ids(facts_name_mapping))
end

"""Build v35 switched shunt step data (S, N, B triplets padded to 8)."""
function _build_switched_shunt_steps_v35(
    shunt::PSY.SwitchedAdmittance,
    steps::Vector{Int},
    increases::Vector{Complex{Float64}},
    base_power::Float64,
)
    S_vals = []
    N_vals = []
    B_vals = []
    for (N, B) in zip(steps, increases)
        push!(S_vals, get_ext_key_or_default(shunt, "S$(length(S_vals)+1)", 1))
        push!(N_vals, N)
        push!(B_vals, _psse_round_val(imag(B) * base_power))
    end
    while length(S_vals) < 8
        push!(S_vals, PSSE_DEFAULT)
        push!(N_vals, PSSE_DEFAULT)
        push!(B_vals, PSSE_DEFAULT)
    end
    return (
        [get(S_vals, i, PSSE_DEFAULT) for i in 1:8],
        [get(N_vals, i, PSSE_DEFAULT) for i in 1:8],
        [get(B_vals, i, PSSE_DEFAULT) for i in 1:8],
    )
end

"""Build v33 switched shunt step data (N, B pairs padded to 8)."""
function _build_switched_shunt_steps_v33(
    steps::Vector{Int},
    increases::Vector{Complex{Float64}},
    base_power::Float64,
)
    N_vals = []
    B_vals = []
    for (N, B) in zip(steps, increases)
        push!(N_vals, N)
        push!(B_vals, _psse_round_val(imag(B) * base_power))
    end
    while length(N_vals) < 8
        push!(N_vals, PSSE_DEFAULT)
        push!(B_vals, PSSE_DEFAULT)
    end
    return (
        [get(N_vals, i, PSSE_DEFAULT) for i in 1:8],
        [get(B_vals, i, PSSE_DEFAULT) for i in 1:8],
    )
end

"""
WRITTEN TO SPEC: PSS/E 33.3/35.4 POM 5.2.1 Switched Shunt Data
"""
function write_to_buffers!(
    exporter::PSSEExporter,
    ::Val{Symbol("Switched Shunt Data")},
)
    io = exporter.raw_buffer
    md = exporter.md_dict

    check_supported_version(exporter)
    write_v35_header(io, exporter, "Switched Shunt Data")

    switched_shunts = get!(exporter.components_cache, "switched_shunts") do
        sort!(
            collect(PSY.get_components(PSY.SwitchedAdmittance, exporter.system));
            by = PSY.get_name,
        )
    end
    switched_shunt_name_mapping =
        get!(exporter.components_cache, "switched_shunt_name_mapping") do
            create_component_ids(
                convert_empty_stringvec(PSY.get_name.(switched_shunts)),
                PSY.get_number.(PSY.get_bus.(switched_shunts));
                singles_to_1 = true,
            )
        end

    bus_id_counters = Dict{Int, Int}()
    base_power = PSY.get_base_power(exporter.system)

    for shunt in switched_shunts
        sienna_bus_number = PSY.get_number(PSY.get_bus(shunt))
        I = md["bus_number_mapping"][sienna_bus_number]

        if exporter.psse_version == :v35
            bus_id_counters[I] = get(bus_id_counters, I, 0) + 1
            ID = _psse_quote_string(string(bus_id_counters[I]))
        end

        MODSW = get_ext_key_or_default(shunt, "MODSW")
        ADJM = get_ext_key_or_default(shunt, "ADJM")
        STAT = PSY.get_available(shunt) ? 1 : 0
        VSWHI = PSY.get_admittance_limits(shunt).max
        VSWLO = PSY.get_admittance_limits(shunt).min

        if exporter.psse_version == :v35
            SWREG = get_ext_key_or_default(
                shunt,
                "SWREG",
                get_ext_key_or_default(shunt, "SWREM"),
            )
            NREG = get_ext_key_or_default(shunt, "NREG")
        else
            SWREM = get_ext_key_or_default(shunt, "SWREM")
        end

        RMPCT = get_ext_key_or_default(shunt, "RMPCT")
        RMIDNT = _psse_quote_string(String(get_ext_key_or_default(shunt, "RMIDNT")))
        BINIT = _psse_round_val(imag(PSY.get_Y(shunt)) * base_power)

        steps = PSY.get_number_of_steps(shunt)
        increases = PSY.get_Y_increase(shunt)

        if exporter.psse_version == :v35
            S_vars, N_vars, B_vars =
                _build_switched_shunt_steps_v35(shunt, steps, increases, base_power)

            @fastprintdelim_unroll(io, true, I, ID, MODSW, ADJM, STAT,
                VSWHI, VSWLO, SWREG, NREG, RMPCT, RMIDNT, BINIT,
                S_vars[1], N_vars[1], B_vars[1], S_vars[2], N_vars[2], B_vars[2],
                S_vars[3], N_vars[3], B_vars[3], S_vars[4], N_vars[4], B_vars[4],
                S_vars[5], N_vars[5], B_vars[5], S_vars[6], N_vars[6], B_vars[6],
                S_vars[7], N_vars[7], B_vars[7], S_vars[8], N_vars[8], B_vars[8])
        else
            N_vars, B_vars =
                _build_switched_shunt_steps_v33(steps, increases, base_power)

            @fastprintdelim_unroll(io, true, I, MODSW, ADJM, STAT,
                VSWHI, VSWLO, SWREM, RMPCT, RMIDNT, BINIT,
                N_vars[1], B_vars[1], N_vars[2], B_vars[2], N_vars[3], B_vars[3],
                N_vars[4], B_vars[4], N_vars[5], B_vars[5], N_vars[6], B_vars[6],
                N_vars[7], B_vars[7], N_vars[8], B_vars[8])
        end
    end
    end_group(io, md, exporter, "Switched Shunt Data", true)
    exporter.md_valid ||
        (
            md["switched_shunt_name_mapping"] =
                serialize_component_ids(switched_shunt_name_mapping)
        )
end

"""
WRITTEN TO SPEC: PSS/E 33.3 POM 5.2.1 Q Record
"""
function write_to_buffers!(
    exporter::PSSEExporter,
    ::Val{Symbol("Q Record")},
)
    io = exporter.raw_buffer
    md = exporter.md_dict
    check_supported_version(exporter)
    println(io, "Q")  # End of file
    exporter.md_valid || (md["record_groups"]["Q Record"] = true)
end

function _write_skip_group(
    io::IO,
    md::AbstractDict,
    exporter::PSSEExporter,
    this_section_name::String,
)
    check_supported_version(exporter)
    end_group(io, md, exporter, this_section_name, false)
    exporter.md_valid || (md["record_groups"][this_section_name] = false)
end

# If a writer for a given group is not defined, write that we are skipping it
function write_to_buffers!(exporter::PSSEExporter, ::Val{T}) where {T}
    io = exporter.raw_buffer
    md = exporter.md_dict
    group_name = string(T)
    @debug "Export for group $group_name not implemented, skipping it"
    _write_skip_group(io, md, exporter, group_name)
end

_step_to_string(::Nothing) = ""
_step_to_string(iterable_step::Union{Tuple, AbstractArray}) = "_" * join(iterable_step, "_")
_step_to_string(scalar_step::Any) = "_$scalar_step"

"Perform an export from the data contained in a `PSSEExporter` to the PSS/E file format."
function write_export(
    exporter::PSSEExporter,
    name::AbstractString;
    overwrite = false,
)
    original_name = name
    name = name * _step_to_string(exporter.step)
    # Construct paths
    export_subdir = joinpath(exporter.export_dir, name)
    dir_exists = isdir(export_subdir)
    (dir_exists && !overwrite) && throw(
        ArgumentError(
            "Target export directory $(abspath(export_subdir)) already exists; specify `overwrite = true` if it should be overwritten",
        ),
    )
    dir_exists || mkdir(export_subdir)
    @info "Exporting $name to $export_subdir"
    raw_path, md_path = get_psse_export_paths(export_subdir)

    # Build export files in buffers
    md = exporter.md_dict
    if !exporter.md_valid
        md["case_name"] = name

        md["export_settings"] = OrderedDict{String, Any}()
        export_settings = md["export_settings"]
        export_settings["psse_version"] = string(exporter.psse_version)
        export_settings["export_dir"] = exporter.export_dir
        export_settings["original_name"] = original_name
        export_settings["write_comments"] = exporter.write_comments
        export_settings["overwrite"] = exporter.overwrite
        export_settings["step"] = _step_to_string(exporter.step)
        export_settings["sources_as_generators"] = true
        export_settings["storages_as_generators"] = true

        md["record_groups"] = OrderedDict{String, Bool}()  # Keep track of which record groups we actually write to and which we skip

        # These mappings are accessed in e.g. _write_bus_data via the metadata
        md["area_mapping"] = _map_psse_container_names(
            sort!(
                collect(
                    convert_empty_stringvec(
                        PSY.get_name.(PSY.get_components(PSY.Area, exporter.system)),
                    ),
                )),
        )
        md["zone_mapping"] = _map_psse_container_names(
            sort!(
                collect(
                    convert_empty_stringvec(
                        PSY.get_name.(PSY.get_components(PSY.LoadZone, exporter.system)),
                    ),
                )),
        )
    end

    with_units_base(exporter.system, PSY.UnitSystem.SYSTEM_BASE) do
        groups_to_process = update_version_group(exporter.psse_version)
        # Each of these corresponds to a group of records in the PSS/E spec
        for group_name in groups_to_process
            @debug "Writing export for group $group_name"
            write_to_buffers!(exporter, Val{Symbol(group_name)}())
        end
    end

    skipped_groups = [k for (k, v) in md["record_groups"] if !v]
    !isempty(skipped_groups) && @warn "Skipped groups: $(join(skipped_groups, ", "))"

    if !exporter.md_valid
        @debug "Serializing PSS/E export metadata to in-memory buffer"
        take!(exporter.md_buffer)
        JSON3.pretty(exporter.md_buffer, md)
    end
    exporter.md_valid = true

    # Write files
    open(file -> write(file, take!(exporter.raw_buffer)), raw_path; truncate = true)
    open(file -> write(file, seekstart(exporter.md_buffer)), md_path; truncate = true)
    return
end

write_export(exporter::PSSEExporter; kwargs...) =
    write_export(
        exporter,
        exporter.name;
        merge(Dict(:overwrite => exporter.overwrite), kwargs)...,
    )

"Calculate the paths of the (raw, metadata) files that would be written by a certain call to `write_export`"
function get_psse_export_paths(
    export_subdir::AbstractString,
)
    name = last(splitdir(export_subdir))
    raw_path = joinpath(export_subdir, "$name.raw")
    metadata_path = joinpath(export_subdir, "$(name)$(PSY.PSSE_EXPORT_METADATA_EXTENSION)")
    return (raw_path, metadata_path)
end

# COMMON INTERFACE
make_power_flow_container(pfem::PSSEExportPowerFlow, sys::PSY.System; kwargs...) =
    PSSEExporter(
        sys,
        pfem.psse_version,
        pfem.export_dir;
        name = pfem.name,
        write_comments = pfem.write_comments,
        overwrite = pfem.overwrite,
        step = (0, 0),
    )

solve_power_flow!(exporter::PSSEExporter) = write_export(exporter)
