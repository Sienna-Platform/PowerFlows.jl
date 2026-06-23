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
end

"""Union of the KLU, AppleAccelerate, and MKLPardiso solver caches. Every member is concrete so
the 3-way union stays within Julia's small-union splitting; the KLU member is pinned to
`J_INDEX_TYPE` because that is the only instantiation flowing through these methods (the AC Newton
cache and its fallback are built from `J.Jv::SparseMatrixCSC{Float64, J_INDEX_TYPE}`)."""
const PFLinearSolverCache =
    Union{
        PNM.KLULinSolveCache{Float64, J_INDEX_TYPE},
        PNM.AAFactorCache,
        PardisoLinSolveCache,
    }

"""Supertype for the polar NR/TR reuse cache (`PolarNRCache`, `power_flow_method.jl`).
Exists so `PowerFlowData` can type its `polar_nr_cache` slot as a two-member union:
the concrete type cannot be referenced there because of the construction cycle
`PolarNRCache → ACPowerFlowResidual → PowerFlowData`."""
abstract type AbstractNRCache end

"""An abstract supertype for the persistent per-solve caches stored in
`PowerFlowData.solver_cache[]`. Concrete subtypes ([`DCSolverCache`](@ref) for the DC/PTDF path,
`FastDecoupledCache` for the polar fast-decoupled solver) are type-disjoint, so the slot's
type discriminates which path populated it — no sentinel tag is needed and a cross-use is a plain
`MethodError` rather than a silent reuse."""
abstract type SolverCache end

"""Lazily-built cache for the DC solves, stored in `data.solver_cache`. Reused across
repeated solves on the same data (e.g. a PCM loop: fixed network, changing injections)
so the network matrix is factored once and the solve/scratch buffers are not
reallocated. `matrix` and `backend` exist only to invalidate the reuse — rebuild when
the network-matrix object identity or the backend changes (see
`_get_or_build_solver_cache!`). The remaining fields are the per-solve scratch:
`power_injections`/`p_inj` solve buffers, arc resistances `rs`, and the signed arc-bus
incidence (built once at `PowerFlowData` construction; `nothing` for vPTDF)."""
struct DCSolverCache <: SolverCache
    matrix::SparseMatrixCSC{Float64, J_INDEX_TYPE}
    backend::PNM.LinearSolverType
    cache::PFLinearSolverCache
    power_injections::Matrix{Float64}
    p_inj::Matrix{Float64}
    rs::Vector{Float64}
    arc_bus_incidence::Union{SparseMatrixCSC{Int8, Int}, Nothing}
end

# --- Backend-agnostic operations (forward to the owning PNM namespace) ---

symbolic_factor!(c::PNM.KLULinSolveCache, A::SparseMatrixCSC{Float64}) =
    PNM.symbolic_factor!(c, A)
symbolic_factor!(c::PNM.AAFactorCache, A::SparseMatrixCSC{Float64}) =
    PNM.AccelerateWrapper.symbolic_factor!(c, A)

numeric_refactor!(c::PNM.KLULinSolveCache, A::SparseMatrixCSC{Float64}) =
    PNM.numeric_refactor!(c, A)
numeric_refactor!(c::PNM.AAFactorCache, A::SparseMatrixCSC{Float64}) =
    PNM.AccelerateWrapper.numeric_refactor!(c, A)

full_factor!(c::PNM.KLULinSolveCache, A::SparseMatrixCSC{Float64}) =
    PNM.full_factor!(c, A)
full_factor!(c::PNM.AAFactorCache, A::SparseMatrixCSC{Float64}) =
    PNM.AccelerateWrapper.full_factor!(c, A)

solve!(c::PNM.KLULinSolveCache, b::StridedVecOrMat{Float64}) = PNM.solve!(c, b)
solve!(c::PNM.AAFactorCache, b::StridedVecOrMat{Float64}) =
    PNM.AccelerateWrapper.solve!(c, b)

"""Transpose solve `Aᵀ x = b` in place. KLU-only (AppleAccelerate has no
transpose solve)."""
tsolve!(c::PNM.KLULinSolveCache, b::StridedVecOrMat{Float64}) = PNM.tsolve!(c, b)

"""1-norm condition-number estimate of the cached factorization. KLU-only
(libklu's `klu_condest`); AppleAccelerate exposes no condition estimate. Used by
the per-iteration solver diagnostics ([`run_solver_diagnostics!`](@ref))."""
condest!(c::PNM.KLULinSolveCache) = PNM.condest!(c)

# --- Backend resolution and construction ---

"""Resolve the active linear-solver backend tag.

Returns a PNM backend singleton: `PNM.KLUSolver()`, `PNM.AppleAccelerateLUSolver()`,
or `PNM.MKLPardisoSolver()`. When `override === nothing`, the platform default from
PNM's preference logic is used. Throws if AppleAccelerate is requested off an Apple
platform, or if MKLPardiso is requested on a non-x86_64 architecture or without the
`PowerFlowsPardisoExt` extension loaded (`import Pardiso`)."""
function resolve_linear_solver_backend(override::Union{Nothing, AbstractString})
    name = isnothing(override) ? PNM._default_linear_solver() : String(override)
    tag = PNM.resolve_linear_solver(name)
    if tag isa PNM.AppleAccelerateLUSolver && !Sys.isapple()
        error("AppleAccelerate backend requested but not on an Apple platform.")
    elseif tag isa PNM.MKLPardisoSolver
        # Intel MKL is x86_64-only. On other architectures (notably Apple Silicon)
        # it can never load, so give a definitive message rather than suggesting
        # `import Pardiso`, which would not help. macOS on x86_64 (incl. CI under
        # Rosetta) reports `:x86_64` and is allowed through.
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
    end
    return tag
end

"""Construct (without factorizing) the cache for backend `tag` over matrix `A`."""
make_linear_solver_cache(::PNM.KLUSolver, A::SparseMatrixCSC{Float64}) =
    PNM.KLULinSolveCache(A)
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
