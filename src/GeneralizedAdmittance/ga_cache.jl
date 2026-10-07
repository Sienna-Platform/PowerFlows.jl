struct GAWorkspace
    u0::Vector{ComplexF64}
    u::Vector{ComplexF64}
    u_best::Vector{ComplexF64}
    i::Vector{ComplexF64}
    R::Matrix{ComplexF64}
    ut::Vector{ComplexF64}
    iv_raw::Vector{ComplexF64}
    w::Vector{ComplexF64}
    q_v::Vector{Float64}
    psum::Vector{Float64}
    y_new::Vector{ComplexF64}
end

function GAWorkspace(nv::Int, nq::Int, n_islands::Int)
    nl = nv + nq
    c(n) = zeros(ComplexF64, n)
    return GAWorkspace(c(nl), c(nl), c(nl), c(nl), zeros(ComplexF64, nl, 2), c(nv), c(nv),
        c(nq), zeros(nv), zeros(n_islands), c(nl))
end

# Type-II Anderson mixing of the corrective currents, with a circular history of `m`
# differences. `x` is the next input current.
mutable struct GAAnderson
    DF::Matrix{ComplexF64}
    DG::Matrix{ComplexF64}
    f_prev::Vector{ComplexF64}
    g_prev::Vector{ComplexF64}
    x::Vector{ComplexF64}
    gram::Matrix{Float64}
    γ::Vector{Float64}
    n::Int
    head::Int
    have_prev::Bool
end

function GAAnderson(nl::Int, m::Int)
    c() = zeros(ComplexF64, nl)
    return GAAnderson(zeros(ComplexF64, nl, m), zeros(ComplexF64, nl, m), c(), c(), c(),
        zeros(m, m), zeros(m), 0, 0, false)
end

function _ga_anderson_reset!(aa::GAAnderson, i::Vector{ComplexF64})
    copyto!(aa.x, i)
    aa.n = 0
    aa.head = 0
    aa.have_prev = false
    return
end

# `g` is the map's output for input `aa.x`. The map conjugates u, so it is only ℝ-linear:
# the mixing coefficients must be real (least squares over the stacked [Re; Im] vectors).
function _ga_anderson_step!(aa::GAAnderson, g::Vector{ComplexF64})
    m = size(aa.DF, 2)
    if aa.have_prev
        aa.head = mod1(aa.head + 1, m)
        @inbounds for k in eachindex(g)
            f = g[k] - aa.x[k]
            aa.DF[k, aa.head] = f - aa.f_prev[k]
            aa.DG[k, aa.head] = g[k] - aa.g_prev[k]
        end
        aa.n = min(aa.n + 1, m)
    end
    @inbounds for k in eachindex(g)
        aa.f_prev[k] = g[k] - aa.x[k]
        aa.g_prev[k] = g[k]
    end
    aa.have_prev = true
    copyto!(aa.x, g)
    n = aa.n
    if iszero(n)
        return
    end
    G = view(aa.gram, 1:n, 1:n)
    γ = view(aa.γ, 1:n)
    for a in 1:n
        da = view(aa.DF, :, a)
        γ[a] = real(dot(da, aa.f_prev))
        for b in a:n
            G[a, b] = real(dot(da, view(aa.DF, :, b)))
        end
    end
    _, info = LinearAlgebra.LAPACK.potrf!('U', G)
    if !iszero(info)
        # Dependent history: drop it and keep the plain fixed-point step.
        aa.n = 0
        aa.head = 0
        return
    end
    LinearAlgebra.LAPACK.potrs!('U', G, γ)
    for a in 1:n
        c = γ[a]
        @inbounds for k in eachindex(g)
            aa.x[k] -= c * aa.DG[k, a]
        end
    end
    return
end

struct GACacheKey
    ybus_id::UInt
    s_ix::Vector{Int}
    v_ix::Vector{Int}
    q_ix::Vector{Int}
    backend::PNM.LinearSolverType
end

function Base.:(==)(a::GACacheKey, b::GACacheKey)
    return a.ybus_id == b.ybus_id && a.s_ix == b.s_ix && a.v_ix == b.v_ix &&
           a.q_ix == b.q_ix && typeof(a.backend) == typeof(b.backend)
end

mutable struct GeneralizedAdmittanceCache{F} <: SolverCache
    key::GACacheKey
    blocks::GABlocks
    Fl::F
    Fq::F
    factored::Bool
    ws::GAWorkspace
    aa::GAAnderson
end

_ga_cache_key(data::ACPowerFlowData, part::GAPartition, backend::PNM.LinearSolverType) =
    GACacheKey(
        objectid(get_power_network_matrix(data)), part.s_ix, part.v_ix, part.q_ix, backend)

# An empty slot rebuilds. A cache from a different solver raises a MethodError.
_ga_can_reuse(::Nothing, ::GACacheKey) = false
_ga_can_reuse(c::GeneralizedAdmittanceCache, key::GACacheKey) = c.key == key

function _build_ga_cache(
    data::ACPowerFlowData,
    part::GAPartition,
    backend::PNM.LinearSolverType,
)
    blocks = GABlocks(data, part)
    return GeneralizedAdmittanceCache(_ga_cache_key(data, part, backend), blocks,
        make_linear_solver_cache(backend, blocks.Yll),
        make_linear_solver_cache(backend, blocks.Yqq), false,
        GAWorkspace(n_v(part), n_q(part), part.n_islands),
        GAAnderson(n_l(part), GA_ANDERSON_DEPTH))
end

function _get_or_build_ga_cache!(
    data::ACPowerFlowData,
    part::GAPartition,
    backend::PNM.LinearSolverType,
)
    slot = data.solver_cache[]
    if _ga_can_reuse(slot, _ga_cache_key(data, part, backend))
        return slot::GeneralizedAdmittanceCache
    end
    cache = _build_ga_cache(data, part, backend)
    data.solver_cache[] = cache
    return cache
end

struct GABlockYll end
struct GABlockYqq end
_ga_block_name(::GABlockYll) = "Yℓℓ"
_ga_block_name(::GABlockYqq) = "Yqq"
_ga_block_rows(::GABlockYll, part::GAPartition) = part.l_ix
_ga_block_rows(::GABlockYqq, part::GAPartition) = part.q_ix

function _ga_bus_number(bus_lookup::Dict{Int, Int}, ix::Int)
    for (number, index) in bus_lookup
        if index == ix
            return number
        end
    end
    error("GeneralizedAdmittanceACPowerFlow: bus index $ix is not in the bus lookup.")
end

function _ga_factor_error(e::LinearAlgebra.SingularException, part::GAPartition,
    bus_lookup::Dict{Int, Int}, block)
    ix = _ga_block_rows(block, part)[e.info]
    error(
        "GeneralizedAdmittanceACPowerFlow: $(_ga_block_name(block)) is singular at bus " *
        "$(_ga_bus_number(bus_lookup, ix)) (index $ix). Is there an island without a REF bus?",
    )
end

_ga_factor_error(e, ::GAPartition, ::Dict{Int, Int}, block) = throw(e)

const GA_SINGULAR_PROBE_RTOL = 1e-6

# KLU reports a singular block with its column. AppleAccelerate and MKL Pardiso can factor a
# numerically singular block without an error, so one solve checks the forward error.
# The matrix type matches the generic method exactly to avoid an ambiguity for a KLU cache.
_ga_factor_ok(::PNM.KLULinSolveCache, ::SparseMatrixCSC{ComplexF64, Int64}) = true

function _ga_factor_ok(F, A::SparseMatrixCSC{ComplexF64, Int64})
    v = ComplexF64.(1:size(A, 1))
    x = A * v
    solve!(F, x)
    return all(isfinite, x) &&
           LinearAlgebra.norm(x - v) <= GA_SINGULAR_PROBE_RTOL * LinearAlgebra.norm(v)
end

# Runs only after a failure: a KLU factorization of `A` names the bus.
function _ga_singular_error(A::SparseMatrixCSC{ComplexF64, Int64}, block,
    part::GAPartition, bus_lookup::Dict{Int, Int})
    try
        PNM.klu_factorize(A)
    catch e
        _ga_factor_error(e, part, bus_lookup, block)
    end
    return error(
        "GeneralizedAdmittanceACPowerFlow: $(_ga_block_name(block)) is numerically " *
        "singular. Is there an island without a REF bus?",
    )
end

_ga_on_factor_error(e, ::PNM.KLULinSolveCache, ::SparseMatrixCSC, block,
    part::GAPartition, bus_lookup::Dict{Int, Int}) =
    _ga_factor_error(e, part, bus_lookup, block)
_ga_on_factor_error(e, ::Any, A::SparseMatrixCSC, block,
    part::GAPartition, bus_lookup::Dict{Int, Int}) =
    _ga_located_error(e, A, block, part, bus_lookup)
_ga_located_error(::LinearAlgebra.SingularException, A::SparseMatrixCSC, block,
    part::GAPartition, bus_lookup::Dict{Int, Int}) =
    _ga_singular_error(A, block, part, bus_lookup)
_ga_located_error(e, ::SparseMatrixCSC, block, ::GAPartition, ::Dict{Int, Int}) =
    throw(e)

function _ga_factor_block!(F, A::SparseMatrixCSC{ComplexF64, Int64}, factored::Bool,
    block, part::GAPartition, bus_lookup::Dict{Int, Int})
    try
        if factored
            numeric_refactor!(F, A)
        else
            full_factor!(F, A)
        end
    catch e
        _ga_on_factor_error(e, F, A, block, part, bus_lookup)
    end
    if !_ga_factor_ok(F, A)
        _ga_singular_error(A, block, part, bus_lookup)
    end
    return
end

function _ga_factor!(cache::GeneralizedAdmittanceCache, y::Vector{ComplexF64},
    part::GAPartition, bus_lookup::Dict{Int, Int})
    _ga_set_shunts!(cache.blocks, y, n_v(part))
    _ga_factor_block!(cache.Fl, cache.blocks.Yll, cache.factored, GABlockYll(), part,
        bus_lookup)
    _ga_factor_block!(cache.Fq, cache.blocks.Yqq, cache.factored, GABlockYqq(), part,
        bus_lookup)
    cache.factored = true
    return
end
