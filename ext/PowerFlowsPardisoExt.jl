module PowerFlowsPardisoExt

# MKLPardiso linear-solver backend for PowerFlows, loaded when `Pardiso.jl` is
# available. Implements the PowerFlows-local solver-cache operations for
# `PowerFlows.PardisoLinSolveCache` (declared in `src/linear_solver_backend.jl`).
#
# MKL Pardiso requires a functional MKL (x86_64 Linux/Windows); it is unavailable
# on Apple Silicon. The factory below errors clearly when MKL is not available, so
# this extension can load on any platform while only solving where MKL works.

import PowerFlows
import PowerFlows: PardisoLinSolveCache
import PowerNetworkMatrices as PNM
using Pardiso
import SparseArrays: SparseMatrixCSC

# Configure a fresh MKL Pardiso solver for a real unsymmetric system solved from a
# Julia CSC matrix. Order matters: matrix type, then init (sets defaults from the
# type), then iparm tweaks, then the CSC transpose fix.
function _init_pardiso!(ps)
    Pardiso.set_matrixtype!(ps, Pardiso.REAL_NONSYM)
    Pardiso.pardisoinit(ps)
    Pardiso.set_iparm!(ps, 8, 2)   # up to 2 iterative-refinement steps in the solve phase
    # Pardiso expects CSR; Julia is CSC. fix_iparm!(:N) sets the transpose flag so we
    # solve A·x = b (not Aᵀ·x = b) from the CSC arrays.
    Pardiso.fix_iparm!(ps, :N)
    return ps
end

function PowerFlows.make_linear_solver_cache(
    ::PNM.MKLPardisoSolver,
    A::SparseMatrixCSC{Float64},
)
    Pardiso.mkl_is_available() || error(
        "MKLPardiso backend selected but MKL is not available on this platform. " *
        "MKL Pardiso requires x86_64 Linux/Windows; it is unavailable on Apple Silicon.",
    )
    ps = Pardiso.MKLPardisoSolver()
    _init_pardiso!(ps)
    cache =
        PardisoLinSolveCache(ps, A, false, Float64[], Matrix{Float64}(undef, 0, 0), false)
    finalizer(_finalize_pardiso_cache, cache)
    return cache
end

# Free the native MKL Pardiso handle, idempotently. Throws on a failed RELEASE_ALL
# (an MKL memory leak). Safe to call from a normal task (it acquires locks); NOT safe
# to call directly from a finalizer — see `_finalize_pardiso_cache`.
function _release!(c::PardisoLinSolveCache)
    c.released && return c
    c.released = true   # set first: a failed RELEASE_ALL leaks once but never double-frees
    Pardiso.set_phase!(c.ps, Pardiso.RELEASE_ALL)
    Pardiso.pardiso(c.ps)
    return c
end

# Finalizers run inside the GC and must not block. `RELEASE_ALL` (and any logging on failure)
# can deadlock there, so we defer the task via `@async`. Need `errormonitor` here
# because exceptions from detached tasks aren't logged to stderr by default
function _finalize_pardiso_cache(c::PardisoLinSolveCache)
    c.released && return
    errormonitor(@async _release!(c))
    return
end

function PowerFlows.symbolic_factor!(
    cache::PardisoLinSolveCache,
    A::SparseMatrixCSC{Float64},
)
    cache.A = A
    Pardiso.set_phase!(cache.ps, Pardiso.ANALYSIS)
    Pardiso.pardiso(cache.ps, cache.A, Float64[])
    cache.is_factored = false
    return cache
end

function PowerFlows.numeric_refactor!(
    cache::PardisoLinSolveCache,
    A::SparseMatrixCSC{Float64},
)
    cache.A = A
    Pardiso.set_phase!(cache.ps, Pardiso.NUM_FACT)
    Pardiso.pardiso(cache.ps, cache.A, Float64[])
    cache.is_factored = true
    return cache
end

function PowerFlows.full_factor!(
    cache::PardisoLinSolveCache,
    A::SparseMatrixCSC{Float64},
)
    cache.A = A
    Pardiso.set_phase!(cache.ps, Pardiso.ANALYSIS_NUM_FACT)
    Pardiso.pardiso(cache.ps, cache.A, Float64[])
    cache.is_factored = true
    return cache
end

# In-place solve A·x = b. Pardiso solves out-of-place, so it writes into a scratch buffer
# and copies back. Both paths reuse a persistent buffer to stay allocation-free, matching
# the KLU/AA caches: the vector path (Newton hot loop) reuses `scratch`, and the multi-RHS
# matrix path (e.g. multi-period / PCM DC, reusing a cached factorization) reuses
# `scratch_mat`.
function PowerFlows.solve!(cache::PardisoLinSolveCache, b::StridedVector{Float64})
    Pardiso.set_phase!(cache.ps, Pardiso.SOLVE_ITERATIVE_REFINE)
    length(cache.scratch) == length(b) || resize!(cache.scratch, length(b))
    Pardiso.pardiso(cache.ps, cache.scratch, cache.A, b)
    copyto!(b, cache.scratch)
    return b
end

function PowerFlows.solve!(cache::PardisoLinSolveCache, b::StridedMatrix{Float64})
    Pardiso.set_phase!(cache.ps, Pardiso.SOLVE_ITERATIVE_REFINE)
    # Reuse a persistent buffer so a loop of multi-RHS solves on a cached factorization
    # (e.g. multi-period / PCM DC) does not allocate an n×nrhs matrix per call. The shape
    # is stable across such a loop, so the resize happens at most once.
    size(cache.scratch_mat) == size(b) ||
        (cache.scratch_mat = Matrix{Float64}(undef, size(b)))
    Pardiso.pardiso(cache.ps, cache.scratch_mat, cache.A, b)
    copyto!(b, cache.scratch_mat)
    return b
end

"""No-op wrapper: MKLPardiso refines internally in its `SOLVE_ITERATIVE_REFINE` phase, so `A`
and `refinement_eps` are unused (kept only to match the Newton loop's call signature). The refined
solve equals the plain solve; the singular guard in `_set_Δx_nr!` handles MKL's silent pivot
perturbation on (near-)singular matrices."""
function PowerFlows.solve_w_refinement(
    cache::PardisoLinSolveCache,
    A::SparseMatrixCSC{Float64},
    b::Vector{Float64},
    refinement_eps::Float64,
)
    x = copy(b)
    PowerFlows.solve!(cache, x)
    return x
end

end # module
