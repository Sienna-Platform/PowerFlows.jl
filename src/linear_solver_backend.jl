# Linear-solver backend selection for PowerFlows.
#
# PowerFlows consumes PowerNetworkMatrices' (PNM) cached linear solvers instead
# of depending on KLU.jl directly. Three backends are available:
#   - KLU (SuiteSparse, all platforms)
#   - AppleAccelerate (libSparse, macOS only; Int64-indexed matrices only)
#   - MKLPardiso (Intel MKL, x86_64 only; lives in the `PowerFlowsPardisoExt`
#     extension, loaded on `import Pardiso`)
#
# PNM exposes KLU ops as PNM.solve!/full_factor!/... and AppleAccelerate ops as
# PNM.AccelerateWrapper.solve!/full_factor!/...; the MKLPardiso ops live in the
# extension. PowerFlows unifies them below via dispatch over the
# PFLinearSolverCache Union. A future PNM-side abstract supertype will let us
# drop this Union.

"""Cache for the MKLPardiso backend. `ps` (the `Pardiso.MKLPardisoSolver` handle) is held as
`Any`: its type is only available once the `Pardiso.jl` extension loads, and keeping it untyped
also keeps this struct concrete so it stays a splittable member of `PFLinearSolverCache` (the cost
is confined to the Pardiso solve path). `A` is snapshotted because Pardiso reads it at solve time;
`Ti` is left abstract since Pardiso converts indices to `Int32` internally."""
mutable struct PardisoLinSolveCache
    ps::Any                       # Pardiso.MKLPardisoSolver
    A::SparseMatrixCSC{Float64}
    is_factored::Bool
    scratch::Vector{Float64}      # persistent vector solve buffer (resized lazily) → non-alloc vector solve!
    scratch_mat::Matrix{Float64}  # persistent multi-RHS solve buffer (resized on shape change) → non-alloc matrix solve!
    released::Bool                # set once RELEASE_ALL has freed the native MKL handle (guards double-free)
end

"""Union of the KLU, AppleAccelerate, and MKLPardiso solver caches. Every member is concrete so
the 4-way union stays within Julia's small-union splitting. Both KLU index types are listed: the
AC Newton cache and its fallback are built from `J.Jv::SparseMatrixCSC{Float64, J_INDEX_TYPE}`
(`Int32` off Apple, `Int64` on Apple), while PNM's DC ABA factorization is always
`KLULinSolveCache{Float64, Int64}` regardless of platform — so the DC solve path needs the `Int64`
member even where `J_INDEX_TYPE === Int32`."""
const PFLinearSolverCache =
    Union{
        PNM.KLULinSolveCache{Float64, Int32},
        PNM.KLULinSolveCache{Float64, Int64},
        PNM.AAFactorCache,
        PardisoLinSolveCache,
    }

"""Supertype for the polar NR/TR reuse cache (`PolarNRCache`, `power_flow_method.jl`).
Exists so `PowerFlowData` can type its `polar_nr_cache` slot as a two-member union:
the concrete type cannot be referenced there because of the construction cycle
`PolarNRCache → ACPowerFlowResidual → PowerFlowData`."""
abstract type AbstractNRCache end

# --- Backend-agnostic operations (forward to the owning PNM namespace) ---

symbolic_factor!(c::PNM.KLULinSolveCache, A::SparseMatrixCSC{Float64}) =
    PNM.symbolic_factor!(c, A)
symbolic_factor!(c::PNM.AAFactorCache, A::SparseMatrixCSC{Float64}) =
    PNM.AccelerateWrapper.symbolic_factor!(c, A)

# A lean reject is mostly a pivot the frozen order puts on an exact zero (a bus type differing
# from the plan's), which every later iterate of the solve repeats. Finish the solve on KLU's own
# pivot order; `_resume_lean!` re-enables the lean path at the next solve.
function numeric_refactor!(c::PNM.KLULinSolveCache, A::SparseMatrixCSC{Float64})
    PNM.numeric_refactor!(c, A)
    if PNM.KLUWrapper.has_lean_plan(c) && !PNM.KLUWrapper.lean_active(c)
        PNM.KLUWrapper.pause_lean!(c, true)
    end
    return c
end
numeric_refactor!(c::PNM.AAFactorCache, A::SparseMatrixCSC{Float64}) =
    PNM.AccelerateWrapper.numeric_refactor!(c, A)

# A full factorization always pivots afresh, bypassing a lean plan, and keeps the rest of that
# solve on the fresh KLU order.
function full_factor!(c::PNM.KLULinSolveCache, A::SparseMatrixCSC{Float64})
    PNM.symbolic_factor!(c, A)
    PNM.KLUWrapper.pivoted_factor!(c, A)
    PNM.KLUWrapper.has_lean_plan(c) && PNM.KLUWrapper.pause_lean!(c, true)
    return c
end
full_factor!(c::PNM.AAFactorCache, A::SparseMatrixCSC{Float64}) =
    PNM.AccelerateWrapper.full_factor!(c, A)

# The re-pivot guard in `_set_Δx_nr!`: a fresh pivot order on the kept symbolic analysis (the
# pattern has not changed), with the rest of that solve kept off the lean path.
_repivot!(c::PNM.KLULinSolveCache, A::SparseMatrixCSC{Float64}) =
    PNM.KLUWrapper.repivot!(c, A)

# KLU reuses the first factorization's pivot order; AA and Pardiso refactor from scratch.
_repivots(::PNM.KLULinSolveCache) = true
_repivots(::PNM.AAFactorCache) = false
_repivots(::PardisoLinSolveCache) = false

solve!(c::PNM.KLULinSolveCache, b::StridedVecOrMat{Float64}) = PNM.solve!(c, b)
solve!(c::PNM.AAFactorCache, b::StridedVecOrMat{Float64}) =
    PNM.AccelerateWrapper.solve!(c, b)

"""Transpose solve `Aᵀ x = b` in place. KLU-only (AppleAccelerate has no
transpose solve)."""
tsolve!(c::PNM.KLULinSolveCache, b::StridedVecOrMat{Float64}) = PNM.tsolve!(c, b)

"""1-norm condition-number estimate of the cached factorization of `A`, which must be the
matrix of the last `numeric_refactor!`. KLU-only (libklu's `klu_condest`); AppleAccelerate
exposes no condition estimate. Used by the per-iteration solver diagnostics
([`run_solver_diagnostics!`](@ref))."""
condest!(c::PNM.KLULinSolveCache, A::SparseMatrixCSC{Float64}) = PNM.condest!(c, A)

# --- Backend resolution and construction ---

"""Resolve the active linear-solver backend tag.

Returns a PNM backend singleton: `PNM.KLUSolver()`, `PNM.AppleAccelerateLUSolver()`,
or `PNM.MKLPardisoSolver()`. When `override === nothing`, the platform default from
PNM's preference logic is used. Throws if AppleAccelerate is requested off an Apple
platform, if MKLPardiso is requested on a non-x86_64 architecture or without the
`PowerFlowsPardisoExt` extension loaded (`import Pardiso`), or if `"Dense"` is
requested (PNM resolves it to a real backend tag, but PowerFlows has no DC linear
solver cache for it)."""
function resolve_linear_solver_backend(override::Union{Nothing, AbstractString})
    name = if isnothing(override)
        PNM._default_linear_solver()
    else
        String(override)
    end
    return _validate_linear_solver_backend(PNM.resolve_linear_solver(name))
end

_validate_linear_solver_backend(tag::PNM.KLUSolver) = tag

function _validate_linear_solver_backend(tag::PNM.AppleAccelerateLUSolver)
    Sys.isapple() ||
        error("AppleAccelerate backend requested but not on an Apple platform.")
    return tag
end

# Intel MKL is x86_64-only. On other architectures (notably Apple Silicon) it can never
# load, so give a definitive message rather than suggesting `import Pardiso`, which would
# not help. macOS on x86_64 (incl. CI under Rosetta) reports `:x86_64` and is allowed through.
function _validate_linear_solver_backend(tag::PNM.MKLPardisoSolver)
    if Sys.ARCH !== :x86_64
        error(
            "MKLPardiso backend requires an x86_64 platform with Intel MKL; it is " *
            "unavailable on $(Sys.ARCH) architectures (e.g. Apple Silicon). " *
            "Use the \"KLU\" or \"AppleAccelerateLU\" backend instead.",
        )
    elseif !PNM._has_mkl_pardiso_ext()
        error(
            "MKLPardiso backend requested but Pardiso.jl is not loaded. " *
            "Run `import Pardiso` to load the PowerFlowsPardisoExt extension.",
        )
    end
    return tag
end

function _validate_linear_solver_backend(::PNM.DenseSolver)
    throw(
        ArgumentError(
            "linear_solver=\"Dense\" is not supported: PowerFlows has no DC linear " *
            "solver cache for it. Accepted backends are \"KLU\", \"AppleAccelerateLU\" " *
            "(macOS only), and \"MKLPardiso\" (x86_64 only, needs `import Pardiso`).",
        ),
    )
end

# No values snapshot: PowerFlows never uses the snapshot-reading KLU paths (`condest!` gets the
# matrix explicitly), so the per-refactor copy of `nonzeros(A)` is pure overhead.
"""Construct (without factorizing) the cache for backend `tag` over matrix `A`."""
make_linear_solver_cache(::PNM.KLUSolver, A::SparseMatrixCSC{Float64}) =
    PNM.KLULinSolveCache(A; snapshot_values = false)
make_linear_solver_cache(::PNM.AppleAccelerateLUSolver, A::SparseMatrixCSC{Float64}) =
    PNM.AAFactorCache(A)

"""Adapter: PowerFlows historically calls `solve_w_refinement(cache, A, b, eps)`
with a step-tolerance `eps`. Map onto PNM's residual-based refined solve."""
function solve_w_refinement(
    cache::PFLinearSolverCache,
    A::SparseMatrixCSC{Float64},
    b::Vector{Float64},
    refinement_eps::Float64,
)
    return PNM.solve_w_refinement(cache, A, b; tol = refinement_eps)
end
