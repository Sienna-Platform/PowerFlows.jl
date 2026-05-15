"""
    verify_jacobian_asymptotic(residual, Jv, x0, time_step; Δx_mags, rtol, label)

Verify the analytic Jacobian `Jv` against `residual` by checking that the
first-order Taylor remainder

    e(Δx) = ||F(x0 + Δx·u_j) - F(x0) - Δx · J·u_j||

scales as O(Δx²) along each canonical direction `u_j = e_j` as `Δx → 0`.
For a correct Jacobian, `e(Δx) / Δx²` is constant across a geometric
`Δx`-sweep (the second-derivative contribution along `u_j`); this routine
asserts those normalized ratios agree within `rtol`.

Directions where the residual is locally linear in `x_j` within FP noise
are skipped (cannot distinguish a correct from an incorrect entry there).

`residual` must be callable as `residual(x, time_step)` and expose the
current residual vector as `residual.Rv`. Both `ACPowerFlowResidual` and
`ACRectangularCIResidual` satisfy this. `Jv` can be any object that
supports `Jv * u` for `u::Vector{Float64}` (sparse or dense matrix).

Methodology mirrors `test_homotopy_hessian.jl`'s asymptotic checks for the
Hessian/gradient.
"""
function verify_jacobian_asymptotic(
    residual,
    Jv,
    x0::Vector{Float64},
    time_step::Int;
    Δx_mags::Vector{Float64} = [1e-3, 1e-4, 1e-5, 1e-6],
    rtol::Float64 = 0.3,
    label::String = "",
)
    n = length(x0)
    residual(x0, time_step)
    F0 = copy(residual.Rv)
    F0_scale = max(LinearAlgebra.norm(F0), 1.0)
    # Raw-remainder noise floor — below this, the remainder is dominated by
    # floating-point roundoff from cancellation in `F(x+Δx) - F(x) - Δx·J·u`
    # and the asymptotic ratio test is meaningless. The factor 1e4 accounts
    # for typical accumulated relative roundoff in residuals built from many
    # trig / sqrt / division operations (e.g. LCC residual).
    raw_noise_floor = 1e4 * eps(Float64) * F0_scale

    for j in 1:n
        u = zeros(n)
        u[j] = 1.0
        Ju = Jv * u
        # errors[k] = ||remainder(Δx_k)|| / Δx_k (matches test_homotopy_hessian
        # convention; a correct Jacobian makes this O(Δx_k))
        errors = Vector{Float64}(undef, length(Δx_mags))
        for (k, Δx) in enumerate(Δx_mags)
            x1 = x0 .+ Δx .* u
            residual(x1, time_step)
            errors[k] =
                LinearAlgebra.norm(residual.Rv .- F0 .- Δx .* Ju) / Δx
        end
        residual(x0, time_step)  # restore in case caller relies on it

        # Filter Δx points whose raw remainder is at FP noise — those points
        # are dominated by roundoff and the ratio test breaks down there.
        # If every point is at noise, F is locally linear in x_j and the
        # symbolic J·u is correct to machine precision (nothing to check).
        keep = (errors .* Δx_mags) .> raw_noise_floor
        if count(keep) < 2
            @test true
            continue
        end
        # ratios[k] ≈ (1/2)||F''_jj||, should be Δx-independent in the
        # asymptotic regime.
        ratios = (errors ./ Δx_mags)[keep]
        ok = all(isapprox(r, ratios[1]; rtol = rtol) for r in ratios)
        if !ok
            # Locate the largest-residual row: most likely the row whose
            # symbolic J[row, j] entry is wrong.
            x_probe = x0 .+ Δx_mags[end - 1] .* u
            residual(x_probe, time_step)
            remainder = residual.Rv .- F0 .- Δx_mags[end - 1] .* Ju
            residual(x0, time_step)
            row_worst = argmax(abs.(remainder))
            symbolic_entry = Jv[row_worst, j]
            # Estimate ∂F[row_worst]/∂x[j] via central difference
            x_plus = x0 .+ Δx_mags[end - 1] .* u
            x_minus = x0 .- Δx_mags[end - 1] .* u
            residual(x_plus, time_step)
            Fp = residual.Rv[row_worst]
            residual(x_minus, time_step)
            Fm = residual.Rv[row_worst]
            residual(x0, time_step)
            fd_estimate = (Fp - Fm) / (2 * Δx_mags[end - 1])
            observed_orders =
                [
                    log(errors[k + 1] * Δx_mags[k + 1] /
                        (errors[k] * Δx_mags[k])) /
                    log(Δx_mags[k + 1] / Δx_mags[k])
                    for k in 1:(length(Δx_mags) - 1)
                ]
            @warn """
            verify_jacobian_asymptotic: non-quadratic remainder along column j.
              Test label : $label
              Column j   : $j   (sweeping x[$j], all other x fixed at x0)
              Δx_mags    : $Δx_mags
              ||remainder||  : $(errors .* Δx_mags)
              ||rem||/Δx     : $errors  (should decay linearly if J·u correct)
              ||rem||/Δx²    : $ratios  (should be constant if J·u correct)
              Observed order : $observed_orders   (≈ 2 if correct, ≈ 1 if wrong)

              Worst-row diagnostics (residual row most off the linear model):
              row = $row_worst
              analytic Jv[$row_worst, $j]      = $symbolic_entry
              central-difference ∂F[$row_worst]/∂x[$j] = $fd_estimate
              relative diff                            = $((symbolic_entry - fd_estimate) /
                                                          max(abs(fd_estimate), eps(Float64)))
            """ maxlog = 10
        end
        @test ok
    end
    return
end
