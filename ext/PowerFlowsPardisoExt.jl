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
    cache = PardisoLinSolveCache(ps, A, false)
    finalizer(cache) do c
        try
            Pardiso.set_phase!(c.ps, Pardiso.RELEASE_ALL)
            Pardiso.pardiso(c.ps)
        catch
        end
    end
    return cache
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

# In-place solve A·x = b. Pardiso solves out-of-place, so use scratch and copy back.
function PowerFlows.solve!(cache::PardisoLinSolveCache, b::StridedVecOrMat{Float64})
    Pardiso.set_phase!(cache.ps, Pardiso.SOLVE_ITERATIVE_REFINE)
    x = similar(b)
    Pardiso.pardiso(cache.ps, x, cache.A, b)
    copyto!(b, x)
    return b
end

# Pardiso's SOLVE_ITERATIVE_REFINE phase already refines; preserve the (cache, A, b,
# eps) call shape used by PowerFlows' Newton loop. The backend-agnostic singular
# guard in `_set_Δx_nr!` covers MKL's silent pivot perturbation on (near-)singular
# matrices.
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
