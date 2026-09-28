# Solver cache for the generalized-admittance loop, stored in `data.solver_cache[]`.
#
# Holds the fixed sparse blocks (built once per network/partition), two KLU factorizations
# over the complex Yℓℓ and Yqq blocks (refactored, not re-ordered, each iteration as the
# shunts change), and the preallocated workspace the iteration kernel reads/writes. Reuse
# mirrors FD's `_reuse_fd_cache`: an empty slot rebuilds, a matching key refactors, and a
# foreign cache is a loud MethodError.

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
end

function GAWorkspace(nv::Int, nq::Int)
    nl = nv + nq
    c(n) = zeros(ComplexF64, n)
    return GAWorkspace(c(nl), c(nl), c(nl), c(nl), zeros(ComplexF64, nl, 2), c(nv), c(nv),
        c(nq), zeros(nv))
end

mutable struct GeneralizedAdmittanceCache <: SolverCache
    key::UInt
    blocks::GABlocks
    Fl::PNM.KLULinSolveCache{ComplexF64, Int64}
    Fq::PNM.KLULinSolveCache{ComplexF64, Int64}
    factored::Bool
    ws::GAWorkspace
end

_ga_cache_key(data::ACPowerFlowData, part::GAPartition) =
    hash((objectid(get_power_network_matrix(data)), part.s_ix, part.v_ix, part.q_ix))

# Mirrors `_reuse_fd_cache`: an empty slot rebuilds; a foreign cache is a loud MethodError.
_ga_can_reuse(::Nothing, ::UInt) = false
_ga_can_reuse(c::GeneralizedAdmittanceCache, key::UInt) = c.key == key

function _get_or_build_ga_cache!(data::ACPowerFlowData, part::GAPartition)
    key = _ga_cache_key(data, part)
    slot = data.solver_cache[]
    if _ga_can_reuse(slot, key)
        return slot::GeneralizedAdmittanceCache
    end
    blocks = GABlocks(data, part)
    cache = GeneralizedAdmittanceCache(key, blocks, PNM.KLULinSolveCache(blocks.Yll),
        PNM.KLULinSolveCache(blocks.Yqq), false, GAWorkspace(n_v(part), n_q(part)))
    data.solver_cache[] = cache
    return cache
end

function _ga_factor_error(e::LinearAlgebra.SingularException, part::GAPartition,
    which::String)
    if which == "Yqq"
        bus = part.q_ix[e.info]
    else
        bus = part.l_ix[e.info]
    end
    error(
        "GeneralizedAdmittanceACPowerFlow: $which is singular at bus index $bus. " *
        "Is there an island without a REF bus?",
    )
end

_ga_factor_error(e, ::GAPartition, ::String) = throw(e)

function _ga_factor!(cache::GeneralizedAdmittanceCache, y::Vector{ComplexF64},
    part::GAPartition)
    _ga_set_shunts!(cache.blocks, y, n_v(part))
    which = "Yℓℓ"
    try
        if cache.factored
            PNM.numeric_refactor!(cache.Fl, cache.blocks.Yll)
            which = "Yqq"
            PNM.numeric_refactor!(cache.Fq, cache.blocks.Yqq)
        else
            PNM.full_factor!(cache.Fl, cache.blocks.Yll)
            which = "Yqq"
            PNM.full_factor!(cache.Fq, cache.blocks.Yqq)
            cache.factored = true
        end
    catch e
        _ga_factor_error(e, part, which)
    end
    return
end
