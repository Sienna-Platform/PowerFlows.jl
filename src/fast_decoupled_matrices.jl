# Fast/Fixed Decoupled Newton-Raphson (FDNR) — B′/B″ matrix machinery (WP1).
#
# Builds the constant fast-decoupled Jacobian approximations B′ (active-power/angle) and
# B″ (reactive-power/voltage) from the PowerFlowData network matrices. Everything here is a
# pure function of (Ybus, arc-admittance matrices, bus types); nothing reads or mutates the
# per-iteration state, so the matrices can be assembled and factored once and reused across all
# iterations and time steps.
#
# Conventions (verified against PowerNetworkMatrices ^0.23 on c_sys14 / WECC240):
#
#   * Per-arc π-model stamp (tap on the FROM side):
#       yff = (ys + j·b_c/2) / |τ|²
#       yft = −ys / conj(τ)
#       ytf = −ys / τ
#       ytt =  ys + j·b_c/2
#     with `Yft[a, f] = yff`, `Yft[a, t] = yft`, `Ytf[a, f] = ytf`, `Ytf[a, t] = ytt`
#     (`Yft = arc_admittance_from_to`, `Ytf = arc_admittance_to_from`; row `a` is nonzero only
#     at the from/to bus columns).
#
#   * Recovery (algebraically exact, independent of any sign choice):
#       |τ| = sqrt(real(ytt / yff)),  θτ = −angle(ytf / yft) / 2,  τ = |τ|·e^{jθτ},
#       ys  = −yft · conj(τ),         b_c = 2·imag(ytt − ys).
#     Per-bus shunt is taken as the residual against the *reconstructed* arc self-terms
#     (`ysh_i = Ybus[i,i] − Σ reconstructed self-terms`), so `_restamp_ybus` is exact by
#     construction even when a transformer carries a magnetizing admittance that PNM does not
#     fold into the π-model `b_c` (validated on WECC240, rel err ≈ 1e-8 vs the ComplexF32 Ybus).
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
# blow up to Inf/NaN); the recovered `ys`/`b_c`/shunt and the restamp stay at their true values so
# the restamp invariant holds exactly for every branch (incl. mostly-resistive near-zero-x ones).
const FD_INV_X_CAP = 1 / PNM.ZERO_IMPEDANCE_X_EPSILON  # = 1e6
const FD_TAU_FLOOR = 1e-8       # |τ| floor guarding the ytt/yff ratio

"""
    FDRecoveredParams

Per-arc π-model parameters recovered from the PowerNetworkMatrices arc-admittance matrices,
plus per-bus shunt admittances. Promoted to `ComplexF64` from the stored `ComplexF32`.

# Fields
- `nbus::Int`: number of buses (Ybus dimension).
- `from::Vector{Int}` / `to::Vector{Int}`: per-arc from/to bus row indices (Ybus order).
- `tau::Vector{ComplexF64}`: per-arc complex tap ratio `τ = |τ|·e^{jθτ}` (tap on the from side).
- `ys::Vector{ComplexF64}`: per-arc series admittance.
- `bc::Vector{Float64}`: per-arc total line charging `b_c` (sum of both π half-shunts).
- `shunt::Vector{ComplexF64}`: per-bus shunt admittance (residual of `Ybus[i,i]` minus the
  reconstructed incident arc self-terms).
"""
struct FDRecoveredParams
    nbus::Int
    from::Vector{Int}
    to::Vector{Int}
    tau::Vector{ComplexF64}
    ys::Vector{ComplexF64}
    bc::Vector{Float64}
    shunt::Vector{ComplexF64}
end

"""
    FDMatrices

Container for the constant fast-decoupled matrices for one `(data, scheme)`.

# Fields
- `scheme::Symbol`: `:XB` (Stott–Alsac) or `:BX` (van Amerongen).
- `recovered::FDRecoveredParams`: cached arc/shunt recovery (shared by B′ and B″_full).
- `pvpq::Vector{Int}`: non-REF bus indices (rows/cols of B′), sorted.
- `bp::SparseMatrixCSC{Float64, J_INDEX_TYPE}`: B′ over `pvpq` (assembled; symmetric except with
  phase shifters).
- `bp_cache::PFLinearSolverCache`: B′ factorization (built once, reused across iterations/steps).
- `bpp_full::SparseMatrixCSC{Float64, J_INDEX_TYPE}`: B″ assembled over ALL buses; the `[pq, pq]`
  submatrix is extracted per driver invocation via [`extract_bpp`](@ref).
"""
struct FDMatrices
    scheme::Symbol
    recovered::FDRecoveredParams
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
# Per-arc parameter recovery
# -------------------------------------------------------------------------------------------

"""
    _recover_arc_params(data::ACPowerFlowData) -> FDRecoveredParams

Recover per-arc π-model parameters (`τ`, `ys`, `b_c`) and per-bus shunt admittances from the
PowerNetworkMatrices arc-admittance matrices and the Ybus diagonal. See the file header for the
stamp/recovery conventions. Guards `|τ| ≥ FD_TAU_FLOOR` and NaN/Inf. Recovered `ys`/`b_c`/shunt
are left at their true values (the near-zero-reactance cap lives in `_fd_series`, applied only on
the resistance-drop stamp path), so the restamp invariant holds exactly for every branch.
"""
function _recover_arc_params(data::ACPowerFlowData)
    Yb = data.power_network_matrix.data
    Yft = data.power_network_matrix.arc_admittance_from_to
    Ytf = data.power_network_matrix.arc_admittance_to_from
    bus_lookup = get_bus_lookup(data)
    arcs = PNM.get_arc_axis(Yft)
    Yft_d = Yft.data
    Ytf_d = Ytf.data
    nbus = size(Yb, 1)
    narc = length(arcs)

    from = Vector{Int}(undef, narc)
    to = Vector{Int}(undef, narc)
    tau = Vector{ComplexF64}(undef, narc)
    ys = Vector{ComplexF64}(undef, narc)
    bc = Vector{Float64}(undef, narc)

    # Self-terms reconstructed from the recovered params, used to back out the per-bus shunt.
    self_acc = zeros(ComplexF64, nbus)

    for (a, arc) in enumerate(arcs)
        f = bus_lookup[first(arc)]
        t = bus_lookup[last(arc)]
        yff = ComplexF64(Yft_d[a, f])
        yft = ComplexF64(Yft_d[a, t])
        ytf = ComplexF64(Ytf_d[a, f])
        ytt = ComplexF64(Ytf_d[a, t])

        τmag = sqrt(max(real(ytt / yff), FD_TAU_FLOOR^2))
        (isfinite(τmag) && τmag >= FD_TAU_FLOOR) || (τmag = 1.0)
        θτ = -angle(ytf / yft) / 2
        isfinite(θτ) || (θτ = 0.0)
        τ = τmag * cis(θτ)
        ys_a = -yft * conj(τ)
        if !isfinite(ys_a)
            @debug "FDNR arc recovery: non-finite ys on arc $a ($(first(arc))→$(last(arc)))"
            ys_a = ComplexF64(yff)
        end
        bc_a = 2 * imag(ytt - ys_a)
        isfinite(bc_a) || (bc_a = 0.0)

        from[a] = f
        to[a] = t
        tau[a] = τ
        ys[a] = ys_a
        bc[a] = bc_a

        # Reconstructed self terms (must match the restamp formula exactly so the shunt residual
        # absorbs everything the π-model can't represent).
        self_acc[f] += (ys_a + im * bc_a / 2) / abs2(τ)
        self_acc[t] += ys_a + im * bc_a / 2
    end

    shunt = Vector{ComplexF64}(undef, nbus)
    for i in 1:nbus
        shunt[i] = ComplexF64(Yb[i, i]) - self_acc[i]
    end

    return FDRecoveredParams(nbus, from, to, tau, ys, bc, shunt)
end

# -------------------------------------------------------------------------------------------
# Restamp validation hook
# -------------------------------------------------------------------------------------------

"""
    _restamp_ybus(p::FDRecoveredParams) -> SparseMatrixCSC{ComplexF64, Int}

Rebuild the full Ybus from recovered π-model parameters plus per-bus shunts. Used by the WP1
restamp-reconstruction tests; should match the original Ybus within ComplexF32 noise.
"""
function _restamp_ybus(p::FDRecoveredParams)
    I = Int[]
    J = Int[]
    V = ComplexF64[]
    for a in eachindex(p.from)
        f = p.from[a]
        t = p.to[a]
        τ = p.tau[a]
        ys = p.ys[a]
        half = im * p.bc[a] / 2
        push!(I, f)
        push!(J, f)
        push!(V, (ys + half) / abs2(τ))
        push!(I, t)
        push!(J, t)
        push!(V, ys + half)
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
# neglected (ys → 1/(j·x), x = imag(1/ys)), else the full ys is kept. The resistance-neglect side
# differs by scheme: B′ drops it under XB, B″ drops it under BX (MATPOWER makeB).
#
# The cap lives here (not in `_recover_arc_params`) so it touches ONLY the resistance-drop path: a
# branch with |x| below PNM's reactance floor (incl. a true x=0, where `1/(j·x)` would be Inf/NaN)
# has its `1/x` clamped to `FD_INV_X_CAP`, sign preserved for series capacitors. The full-`ys`
# branch and the recovered params/restamp are left untouched.
@inline function _fd_series(ys::ComplexF64, drop_resistance::Bool)
    drop_resistance || return ys
    x = imag(1 / ys)
    if abs(x) < 1 / FD_INV_X_CAP
        x = ifelse(x == 0, one(x), sign(x)) / FD_INV_X_CAP
    end
    return 1 / (im * x)
end

"""
    _assemble_bp_full(p::FDRecoveredParams, scheme::Symbol)
        -> SparseMatrixCSC{Float64, J_INDEX_TYPE}

Assemble the full-bus B′ matrix `−imag(Ybus_temp)`, where `Ybus_temp` is stamped with `b_c = 0`,
bus shunts = 0, `|τ| = 1` (phase shift retained → mildly unsymmetric only with phase shifters).
The REF rows/cols are removed later by the `pvpq` restriction.
"""
function _assemble_bp_full(p::FDRecoveredParams, scheme::Symbol)
    n = p.nbus
    I = J_INDEX_TYPE[]
    Jc = J_INDEX_TYPE[]
    V = Float64[]
    diag = zeros(Float64, n)
    for a in eachindex(p.from)
        f = p.from[a]
        t = p.to[a]
        ys = _fd_series(p.ys[a], scheme === :XB)   # B′: XB neglects resistance
        phase = cis(angle(p.tau[a]))   # |τ| = 1, retain phase shift
        # b_c = 0, shunt = 0:  yff = ys, ytt = ys, yft = −ys/conj(phase), ytf = −ys/phase.
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
    _assemble_bpp_full(p::FDRecoveredParams, scheme::Symbol)
        -> SparseMatrixCSC{Float64, J_INDEX_TYPE}

Assemble the full-bus B″ matrix `−imag(Ybus_temp)`, where `Ybus_temp` is stamped with phase
shift = 0 (`|τ|` retained), and `b_c` + bus shunts INCLUDED. The `[pq, pq]` submatrix is
extracted per driver invocation by [`extract_bpp`](@ref).
"""
function _assemble_bpp_full(p::FDRecoveredParams, scheme::Symbol)
    n = p.nbus
    I = J_INDEX_TYPE[]
    Jc = J_INDEX_TYPE[]
    V = Float64[]
    diag = zeros(Float64, n)
    for a in eachindex(p.from)
        f = p.from[a]
        t = p.to[a]
        ys = _fd_series(p.ys[a], scheme === :BX)   # B″: BX neglects resistance
        τmag = abs(p.tau[a])      # retain magnitude, drop phase shift
        half = im * p.bc[a] / 2
        # yff = (ys + j b_c/2)/|τ|², ytt = ys + j b_c/2, yft = ytf = −ys/|τ| (real tap, no phase).
        yff = (ys + half) / τmag^2
        ytt = ys + half
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
    build_fd_matrices(data::ACPowerFlowData, time_step::Int64, scheme::Symbol) -> FDMatrices

Build the constant fast-decoupled matrices for the given `scheme` (`:XB` or `:BX`):

  * recover per-arc params + per-bus shunts (cached on the result),
  * assemble B′ over the non-REF (`pvpq`) buses for `time_step`'s bus types and factor it once,
  * assemble the full-bus B″ (the `[pq, pq]` submatrix is extracted later via `extract_bpp`).

`pvpq`/`pq` are the bus-type index sets at `time_step` (frozen within a driver invocation; the
Q-limit outer loop re-invokes the driver after switching). The B′ factorization is reusable
across all iterations and time steps with the same `pvpq`.
"""
function build_fd_matrices(
    data::ACPowerFlowData,
    time_step::Int64,
    scheme::Symbol;
    linear_solver = nothing,
)
    scheme in (:XB, :BX) ||
        throw(ArgumentError("FDNR scheme must be :XB or :BX, got $(scheme)."))
    recovered = _recover_arc_params(data)
    ref, pv, pq = bus_type_idx(data, time_step)
    pvpq = sort(vcat(pv, pq))

    bp_full = _assemble_bp_full(recovered, scheme)
    bp = bp_full[pvpq, pvpq]
    backend = resolve_linear_solver_backend(linear_solver)
    bp_cache = make_linear_solver_cache(backend, bp)
    # An empty non-REF set (e.g. a lone-REF-bus island) yields a 0×0 B′; skip factoring it (the
    # decoupled loop skips the P half-step when there are no non-REF buses). Factoring a 0×0
    # system errors in the sparse backends (AppleAccelerate: "columnCount must be > 0").
    isempty(pvpq) || full_factor!(bp_cache, bp)

    bpp_full = _assemble_bpp_full(recovered, scheme)
    return FDMatrices(scheme, recovered, pvpq, bp, bp_cache, bpp_full)
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
