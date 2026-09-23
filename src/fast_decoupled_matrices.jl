# Fast/Fixed Decoupled Newton-Raphson (FDNR) — B′/B″ matrix machinery (WP1).
#
# Builds the constant fast-decoupled Jacobian approximations B′ (active-power/angle) and
# B″ (reactive-power/voltage) from the PowerFlowData network matrices. Everything here is a
# pure function of (Ybus, network reduction data, bus types); nothing reads or mutates the
# per-iteration state, so the matrices can be assembled and factored once and reused across all
# iterations and time steps.
#
# Conventions:
#
#   * Per-branch π-model parameters come straight from PowerNetworkMatrices as
#     `PNM.EquivalentBranch` (`r`, `x`, from/to shunts, `tap`, `shift`), resolved per retained arc
#     by `PNM.arc_equivalent_branches`. PNM owns the reduction bookkeeping, so a direct branch, a
#     parallel group, a series chain and a Ward-added impedance all resolve through that one
#     accessor. An arc yields more than one π branch only for a parallel group mixing phase-shift
#     angles with impedance angles; B′/B″ are linear stamps, so each is stamped in turn.
#
#   * Ybus stamp for one π branch (tap on the FROM side, matching `PNM._pi_to_ybus`):
#       yff = ys / |τ|² + y_fr
#       yft = −ys / conj(τ)
#       ytf = −ys / τ
#       ytt = ys + y_to
#     with `ys = 1/(r + j·x)` and `τ = tap·e^{j·shift}`. The from/to shunts are independent and
#     complex (they carry the real conductance), and the from shunt sits OUTSIDE the `1/|τ|²`.
#
#   * Per-bus shunt is taken as the residual against the reconstructed arc self-terms
#     (`ysh_i = Ybus[i,i] − Σ self-terms`). The self-terms use PNM's own stamp, so the residual is
#     the true bus shunt — fixed admittances plus anything a reduction folded onto the diagonal —
#     and `_restamp_ybus` is exact by construction.
#
#   * Sign convention for B′/B″: they approximate the codebase's OWN Jacobian sub-blocks. On a
#     lossless, shunt-free, nominal-tap network at flat start, B′ = (P-θ block)/V over pvpq and
#     B″ = (Q-V block)/V over pq EXACTLY, which equals −imag(Ybus) restricted to those rows/cols.
#     T1 (`test/test_fast_decoupled.jl`, "FastDecoupled WP1: B′/B″ vs exact Jacobian") is the
#     arbiter — it compares against the real `ACPowerFlowJacobian.Jv`, never against this code.
#
# See also `src/fast_decoupled_method.jl` (WP2/WP3 drivers) which consume these matrices.

# 1/x cap for the resistance-neglecting B′/B″ stamp (sign preserved for series capacitors), locked
# to PowerNetworkMatrices' reactance floor: PNM substitutes x = ZERO_IMPEDANCE_X_EPSILON for an
# r=x=0 branch when building the Ybus this code reads, so 1/ZERO_IMPEDANCE_X_EPSILON is the
# largest series susceptance that Ybus can contain. Deriving it here (rather than hard-coding)
# keeps the FD near-zero-reactance threshold from drifting away from PNM's definition. The cap is
# applied ONLY in `_fd_series` (the `1/x` resistance-drop path, where a true x→0 would otherwise
# blow up to Inf/NaN); the π parameters and the restamp stay at their true values so the restamp
# invariant holds exactly for every branch (incl. mostly-resistive near-zero-x ones).
const FD_INV_X_CAP = 1 / PNM.ZERO_IMPEDANCE_X_EPSILON  # = 1e6

"""
    FDArcParams

Per-branch π-model parameters read from PowerNetworkMatrices' `EquivalentBranch`, plus per-bus
shunt admittances. One entry per π branch: an arc contributes more than one only when a parallel
group has no single-π equivalent.

# Fields
- `nbus::Int`: number of buses (Ybus dimension).
- `from::Vector{Int}` / `to::Vector{Int}`: from/to bus row indices (Ybus order).
- `tau::Vector{ComplexF64}`: complex tap ratio `τ = tap·e^{j·shift}` (tap on the from side).
- `ys::Vector{ComplexF64}`: series admittance `1/(r + j·x)`.
- `x::Vector{Float64}`: series reactance, kept alongside `ys` for the resistance-drop stamp.
- `y_fr::Vector{ComplexF64}` / `y_to::Vector{ComplexF64}`: from/to π shunt admittances.
- `shunt::Vector{ComplexF64}`: per-bus shunt admittance (residual of `Ybus[i,i]` minus the
  reconstructed incident arc self-terms).
"""
struct FDArcParams
    nbus::Int
    from::Vector{Int}
    to::Vector{Int}
    tau::Vector{ComplexF64}
    ys::Vector{ComplexF64}
    x::Vector{Float64}
    y_fr::Vector{ComplexF64}
    y_to::Vector{ComplexF64}
    shunt::Vector{ComplexF64}
end

"""
    FDMatrices{S <: FDScheme}

Container for the constant fast-decoupled matrices for one `(data, scheme)`. Parametrized on the
scheme type `S` so `scheme` is a concretely-typed field.

# Fields
- `scheme::S`: the B′/B″ scheme instance, [`FDSchemeXB`](@ref) or [`FDSchemeBX`](@ref).
- `arc_params::FDArcParams`: cached arc π params + bus shunts (shared by B′ and B″_full).
- `pvpq::Vector{Int}`: non-REF bus indices (rows/cols of B′), sorted.
- `bp::SparseMatrixCSC{Float64, J_INDEX_TYPE}`: B′ over `pvpq` (assembled; symmetric except with
  phase shifters).
- `bp_cache::PFLinearSolverCache`: B′ factorization (built once, reused across iterations/steps).
- `bpp_full::SparseMatrixCSC{Float64, J_INDEX_TYPE}`: B″ assembled over ALL buses; the `[pq, pq]`
  submatrix is extracted per driver invocation via [`extract_bpp`](@ref).
"""
struct FDMatrices{S <: FDScheme}
    scheme::S
    arc_params::FDArcParams
    pvpq::Vector{Int}
    bp::SparseMatrixCSC{Float64, J_INDEX_TYPE}
    bp_cache::PFLinearSolverCache
    bpp_full::SparseMatrixCSC{Float64, J_INDEX_TYPE}
end

"""
    FDBppCache

A factored B″ over a specific PQ set. Produced by [`extract_bpp`](@ref); the cache is reusable
across iterations and time steps that return to the same PQ set (Q-limit / multi-period reuse).

# Fields
- `pq::Vector{Int}`: PQ bus indices defining the submatrix (sorted).
- `bpp::SparseMatrixCSC{Float64, J_INDEX_TYPE}`: the `[pq, pq]` submatrix of `bpp_full`.
- `bpp_cache::PFLinearSolverCache`: its factorization.
"""
struct FDBppCache
    pq::Vector{Int}
    bpp::SparseMatrixCSC{Float64, J_INDEX_TYPE}
    bpp_cache::PFLinearSolverCache
end

"""Accessor for the assembled (unfactored) B′ matrix. Used by tests and diagnostics."""
get_bp_matrix(fd::FDMatrices) = fd.bp

"""Accessor for the assembled (unfactored) B″ submatrix. Used by tests and diagnostics."""
get_bpp_matrix(c::FDBppCache) = c.bpp

# -------------------------------------------------------------------------------------------
# Per-arc parameters
# -------------------------------------------------------------------------------------------

"""
    _arc_params(data::ACPowerFlowData) -> FDArcParams

Read per-branch π-model parameters from PowerNetworkMatrices and take the per-bus shunts as the
residual against the reconstructed arc self-terms. See the file header for the stamp convention.
The near-zero-reactance cap lives in `_fd_series`, applied only on the resistance-drop stamp
path, so these parameters and the restamp stay at their true values.

An arc with no single-π equivalent (`|Yft| ≠ |Ytf|`, e.g. a degree-two chain over a mixed
phase-shift/impedance parallel group) gets a symmetrized fallback branch (`ys = -(Yft+Ytf)/2`,
unit tap) instead of throwing. B′/B″ only accelerate the Newton step, so this approximation
cannot corrupt the converged solution, only (rarely) the FD convergence rate on that arc.
"""
function _arc_params(data::ACPowerFlowData)
    ybus = get_power_network_matrix(data)
    Yb = ybus.data
    nrd = PNM.get_network_reduction_data(ybus)
    bus_lookup = get_bus_lookup(data)
    arcs = PNM.get_arc_axis(nrd)
    nbus = size(Yb, 1)

    Yft = ybus.arc_admittance_from_to
    Ytf = ybus.arc_admittance_to_from
    Yft_d = Yft.data
    Ytf_d = Ytf.data
    yft_arc_lookup = PNM.get_arc_lookup(Yft)

    # One π branch per arc is the rule; only a parallel group that mixes phase-shift angles with
    # impedance angles emits more, so `length(arcs)` sizes these exactly on every ordinary
    # network and is a lower bound otherwise.
    narc = length(arcs)
    from = Int[]
    to = Int[]
    tau = ComplexF64[]
    ys = ComplexF64[]
    xs = Float64[]
    y_fr = ComplexF64[]
    y_to = ComplexF64[]
    for v in (from, to, tau, ys, xs, y_fr, y_to)
        sizehint!(v, narc)
    end

    # Self-terms, used to back out the per-bus shunt; must match `_restamp_ybus` exactly.
    self_acc = zeros(ComplexF64, nbus)

    for arc in arcs
        f = bus_lookup[first(arc)]
        t = bus_lookup[last(arc)]
        if PNM.has_single_pi_equivalent(nrd, arc)
            for eb in PNM.arc_equivalent_branches(nrd, arc)
                x_b = PNM.get_equivalent_x(eb)
                ys_b = 1 / complex(PNM.get_equivalent_r(eb), x_b)
                τ = PNM.get_equivalent_tap(eb) * cis(PNM.get_equivalent_shift(eb))
                yfr_b =
                    complex(PNM.get_equivalent_g_from(eb), PNM.get_equivalent_b_from(eb))
                yto_b = complex(PNM.get_equivalent_g_to(eb), PNM.get_equivalent_b_to(eb))

                push!(from, f)
                push!(to, t)
                push!(tau, τ)
                push!(ys, ys_b)
                push!(xs, x_b)
                push!(y_fr, yfr_b)
                push!(y_to, yto_b)

                self_acc[f] += ys_b / abs2(τ) + yfr_b
                self_acc[t] += ys_b + yto_b
            end
        else
            a = yft_arc_lookup[arc]
            yft = ComplexF64(Yft_d[a, t])
            ytf = ComplexF64(Ytf_d[a, f])
            yff = ComplexF64(Yft_d[a, f])
            ytt = ComplexF64(Ytf_d[a, t])
            ys_b = -(yft + ytf) / 2
            x_b = imag(1 / ys_b)
            yfr_b = yff - ys_b
            yto_b = ytt - ys_b

            push!(from, f)
            push!(to, t)
            push!(tau, one(ComplexF64))
            push!(ys, ys_b)
            push!(xs, x_b)
            push!(y_fr, yfr_b)
            push!(y_to, yto_b)

            self_acc[f] += ys_b + yfr_b
            self_acc[t] += ys_b + yto_b
        end
    end

    shunt = Vector{ComplexF64}(undef, nbus)
    for i in 1:nbus
        shunt[i] = ComplexF64(Yb[i, i]) - self_acc[i]
    end

    return FDArcParams(nbus, from, to, tau, ys, xs, y_fr, y_to, shunt)
end

# -------------------------------------------------------------------------------------------
# Restamp validation hook
# -------------------------------------------------------------------------------------------

"""
    _restamp_ybus(p::FDArcParams) -> SparseMatrixCSC{ComplexF64, Int}

Rebuild the full Ybus from the π-model parameters plus per-bus shunts. Used by the WP1
restamp-reconstruction tests; should match the original Ybus within ComplexF32 noise.
"""
function _restamp_ybus(p::FDArcParams)
    I = Int[]
    J = Int[]
    V = ComplexF64[]
    for a in eachindex(p.from)
        f = p.from[a]
        t = p.to[a]
        τ = p.tau[a]
        ys = p.ys[a]
        push!(I, f)
        push!(J, f)
        push!(V, ys / abs2(τ) + p.y_fr[a])
        push!(I, t)
        push!(J, t)
        push!(V, ys + p.y_to[a])
        push!(I, f)
        push!(J, t)
        push!(V, -ys / conj(τ))
        push!(I, t)
        push!(J, f)
        push!(V, -ys / τ)
    end
    for i in 1:(p.nbus)
        push!(I, i)
        push!(J, i)
        push!(V, p.shunt[i])
    end
    return SparseArrays.sparse(I, J, V, p.nbus, p.nbus)
end

# -------------------------------------------------------------------------------------------
# B′ / B″ assembly (MATPOWER makeB semantics)
# -------------------------------------------------------------------------------------------

# Series admittance for the B′/B″ stamp. When `drop_resistance` is set the branch resistance is
# neglected (ys → 1/(j·x)), else the full ys is kept. The resistance-neglect side differs by
# scheme: B′ drops it under XB, B″ drops it under BX (MATPOWER makeB).
#
# The cap lives here (not in `_arc_params`) so it touches ONLY the resistance-drop path: a branch
# with |x| below PNM's reactance floor (incl. a true x=0, where `1/(j·x)` would be Inf/NaN) has
# its `1/x` clamped to `FD_INV_X_CAP`, sign preserved for series capacitors. The full-`ys` branch
# and the π params/restamp are left untouched.
@inline function _fd_series(ys::ComplexF64, x::Float64, drop_resistance::Bool)
    drop_resistance || return ys
    if abs(x) < 1 / FD_INV_X_CAP
        x = ifelse(x == 0, one(x), sign(x)) / FD_INV_X_CAP
    end
    return 1 / (im * x)
end

# Which side neglects branch resistance, by scheme (MATPOWER makeB): B′ drops it under XB, B″
# under BX. Dispatched on the scheme type so the assembly carries no `:XB`/`:BX` value comparison.
_bp_drops_resistance(::FDSchemeXB) = true
_bp_drops_resistance(::FDSchemeBX) = false
_bpp_drops_resistance(::FDSchemeXB) = false
_bpp_drops_resistance(::FDSchemeBX) = true

"""
    _assemble_bp_full(p::FDArcParams, scheme::FDScheme)
        -> SparseMatrixCSC{Float64, J_INDEX_TYPE}

Assemble the full-bus B′ matrix `−imag(Ybus_temp)`, where `Ybus_temp` is stamped with branch and
bus shunts = 0 and `|τ| = 1` (phase shift retained → mildly unsymmetric only with phase
shifters). The REF rows/cols are removed later by the `pvpq` restriction.
"""
function _assemble_bp_full(p::FDArcParams, scheme::FDScheme)
    n = p.nbus
    I = J_INDEX_TYPE[]
    Jc = J_INDEX_TYPE[]
    V = Float64[]
    diag = zeros(Float64, n)
    for a in eachindex(p.from)
        f = p.from[a]
        t = p.to[a]
        # B′: XB neglects resistance
        ys = _fd_series(p.ys[a], p.x[a], _bp_drops_resistance(scheme))
        phase = cis(angle(p.tau[a]))   # |τ| = 1, retain phase shift
        # shunts = 0:  yff = ys, ytt = ys, yft = −ys/conj(phase), ytf = −ys/phase.
        b_ff = -imag(ys)
        b_tt = -imag(ys)
        b_ft = -imag(-ys / conj(phase))
        b_tf = -imag(-ys / phase)
        diag[f] += b_ff
        diag[t] += b_tt
        push!(I, f)
        push!(Jc, t)
        push!(V, b_ft)
        push!(I, t)
        push!(Jc, f)
        push!(V, b_tf)
    end
    for i in 1:n
        push!(I, i)
        push!(Jc, i)
        push!(V, diag[i])
    end
    return SparseArrays.sparse(I, Jc, V, n, n)
end

"""
    _assemble_bpp_full(p::FDArcParams, scheme::FDScheme)
        -> SparseMatrixCSC{Float64, J_INDEX_TYPE}

Assemble the full-bus B″ matrix `−imag(Ybus_temp)`, where `Ybus_temp` is stamped with phase
shift = 0 (`|τ|` retained), and branch + bus shunts INCLUDED. The `[pq, pq]` submatrix is
extracted per driver invocation by [`extract_bpp`](@ref).
"""
function _assemble_bpp_full(p::FDArcParams, scheme::FDScheme)
    n = p.nbus
    I = J_INDEX_TYPE[]
    Jc = J_INDEX_TYPE[]
    V = Float64[]
    diag = zeros(Float64, n)
    for a in eachindex(p.from)
        f = p.from[a]
        t = p.to[a]
        # B″: BX neglects resistance
        ys = _fd_series(p.ys[a], p.x[a], _bpp_drops_resistance(scheme))
        τmag = abs(p.tau[a])      # retain magnitude, drop phase shift
        # yff = ys/|τ|² + y_fr, ytt = ys + y_to, yft = ytf = −ys/|τ| (real tap, no phase).
        yff = ys / τmag^2 + p.y_fr[a]
        ytt = ys + p.y_to[a]
        yoff = -ys / τmag
        diag[f] += -imag(yff)
        diag[t] += -imag(ytt)
        push!(I, f)
        push!(Jc, t)
        push!(V, -imag(yoff))
        push!(I, t)
        push!(Jc, f)
        push!(V, -imag(yoff))
    end
    # Bus shunts included in B″.
    for i in 1:n
        diag[i] += -imag(p.shunt[i])
    end
    for i in 1:n
        push!(I, i)
        push!(Jc, i)
        push!(V, diag[i])
    end
    return SparseArrays.sparse(I, Jc, V, n, n)
end

# -------------------------------------------------------------------------------------------
# Public API
# -------------------------------------------------------------------------------------------

"""
    _warn_low_reactance(p::FDArcParams)

Warn (once) if any branch reactance `|x|` is below [`FD_LOW_REACTANCE_WARNING`](@ref): such
super-low reactances make the B′/B″ decoupling ill-conditioned, so the `:decoupled` variant
converges only at a slow linear rate. The `:fixed_jacobian` variant and the Newton family are
unaffected.
"""
function _warn_low_reactance(p::FDArcParams)
    min_x = Inf
    @inbounds for x_b in p.x
        x = abs(x_b)
        x < min_x && (min_x = x)
    end
    if min_x < FD_LOW_REACTANCE_WARNING
        @warn "FastDecoupled: smallest branch reactance |x| = $(min_x) pu is below " *
              "$(FD_LOW_REACTANCE_WARNING) pu. Super-low reactances make the B′/B″ decoupling " *
              "ill-conditioned, so the :decoupled variant may converge slowly or hit the " *
              "iteration cap. Use a handoff_solver (NewtonRaphsonACPowerFlow / " *
              "TrustRegionACPowerFlow / LevenbergMarquardtACPowerFlow) or " *
              "FastDecoupledACPowerFlow{FDFixedJacobian}." maxlog = 1
    end
    return
end

"""
    build_fd_matrices(data::ACPowerFlowData, time_step::Int64, scheme::FDScheme) -> FDMatrices

Build the constant fast-decoupled matrices for the given `scheme` ([`FDSchemeXB`](@ref) or
[`FDSchemeBX`](@ref)):

  * read per-arc π params + per-bus shunts (cached on the result),
  * assemble B′ over the non-REF (`pvpq`) buses for `time_step`'s bus types and factor it once,
  * assemble the full-bus B″ (the `[pq, pq]` submatrix is extracted later via `extract_bpp`).

`pvpq`/`pq` are the bus-type index sets at `time_step` (frozen within a driver invocation; the
Q-limit outer loop re-invokes the driver after switching). The B′ factorization is reusable
across all iterations and time steps with the same `pvpq`.
"""
function build_fd_matrices(
    data::ACPowerFlowData,
    time_step::Int64,
    scheme::FDScheme;
    linear_solver = nothing,
)
    arc_params = _arc_params(data)
    _warn_low_reactance(arc_params)
    ref, pv, pq = bus_type_idx(data, time_step)
    pvpq = sort(vcat(pv, pq))

    bp_full = _assemble_bp_full(arc_params, scheme)
    bp = bp_full[pvpq, pvpq]
    backend = resolve_linear_solver_backend(linear_solver)
    bp_cache = make_linear_solver_cache(backend, bp)
    # An empty non-REF set (e.g. a lone-REF-bus island) yields a 0×0 B′; skip factoring it (the
    # decoupled loop skips the P half-step when there are no non-REF buses). Factoring a 0×0
    # system errors in the sparse backends (AppleAccelerate: "columnCount must be > 0").
    isempty(pvpq) || full_factor!(bp_cache, bp)

    bpp_full = _assemble_bpp_full(arc_params, scheme)
    return FDMatrices(scheme, arc_params, pvpq, bp, bp_cache, bpp_full)
end

"""
    extract_bpp(fd::FDMatrices, pq_set::AbstractVector{<:Integer};
                linear_solver = nothing) -> FDBppCache

Extract and factor the `[pq, pq]` submatrix of the full B″. Called once per distinct PQ set; the
result is cached by the driver so Q-limit retries and multi-period steps that return to the same
PQ set reuse the factorization (no refactorization).
"""
function extract_bpp(
    fd::FDMatrices,
    pq_set::AbstractVector{<:Integer};
    linear_solver = nothing,
)
    pq = sort(collect(Int, pq_set))
    bpp = fd.bpp_full[pq, pq]
    backend = resolve_linear_solver_backend(linear_solver)
    bpp_cache = make_linear_solver_cache(backend, bpp)
    # An all-PV/REF network has no PQ buses ⇒ 0×0 B″; skip factoring it (the decoupled loop skips
    # the Q half-step). Same 0×0-factorization guard as `build_fd_matrices` above.
    isempty(pq) || full_factor!(bpp_cache, bpp)
    return FDBppCache(pq, bpp, bpp_cache)
end
