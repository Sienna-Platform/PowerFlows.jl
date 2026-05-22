@testset "LCC second-derivative helpers vs central FD" begin
    K = PF.SQRT6_DIV_PI

    # Universal side-aware P_s and Q_s for the FD reference. σ = +1 rectifier,
    # σ = -1 inverter. β = x_t · I_dc / √2.
    function P_lcc(V, t, α, x_t, I_dc, σ)
        β = x_t * I_dc / sqrt(2)
        u = σ * cos(α) - β / (V * t)
        return V * t * K * I_dc * u
    end
    function Q_lcc(V, t, α, x_t, I_dc, σ)
        β = x_t * I_dc / sqrt(2)
        u = σ * cos(α) - β / (V * t)
        return V * t * K * I_dc * sqrt(1 - u^2)
    end
    function ϕ_lcc(V, t, α, x_t, I_dc, σ)
        β = x_t * I_dc / sqrt(2)
        return acos(σ * cos(α) - β / (V * t))
    end

    # Central FD 2nd derivative (5-point for diagonal, 4-point for cross).
    function fd_d2(f, V, t, α, ix, iy; h = 1e-4)
        coords = [V, t, α]
        function eval_at(δx, δy)
            c = copy(coords)
            c[ix] += δx
            c[iy] += δy
            return f(c[1], c[2], c[3])
        end
        if ix == iy
            return (eval_at(h, 0.0) - 2 * eval_at(0.0, 0.0) + eval_at(-h, 0.0)) / h^2
        else
            return (
                eval_at(h, h) - eval_at(-h, h) -
                eval_at(h, -h) + eval_at(-h, -h)
            ) / (4 * h^2)
        end
    end

    rtol = 5e-3
    atol = 1e-6

    @testset "Rectifier ($(V), $(t), $(α))" for (V, t, α) in (
        (1.00, 1.00, deg2rad(15)),
        (0.97, 1.02, deg2rad(8)),
        (1.05, 0.95, deg2rad(25)),
    )
        x_t = 0.05
        I_dc = 1.5
        σ = +1
        ϕ = ϕ_lcc(V, t, α, x_t, I_dc, σ)

        d2P = PF._d2P_lcc(V, t, α, I_dc, σ)
        d2Q = PF._d2Q_lcc(V, t, α, x_t, I_dc, ϕ, σ)

        P_at = (a, b, c) -> P_lcc(a, b, c, x_t, I_dc, σ)
        Q_at = (a, b, c) -> Q_lcc(a, b, c, x_t, I_dc, σ)

        for (name, ix, iy, analytic) in (
            ("VV", 1, 1, d2P.VV), ("tt", 2, 2, d2P.tt),
            ("Vt", 1, 2, d2P.Vt), ("Vα", 1, 3, d2P.Vα),
            ("tα", 2, 3, d2P.tα), ("αα", 3, 3, d2P.αα),
        )
            fd = fd_d2(P_at, V, t, α, ix, iy)
            @test isapprox(analytic, fd; atol = atol, rtol = rtol)
        end
        for (name, ix, iy, analytic) in (
            ("VV", 1, 1, d2Q.VV), ("tt", 2, 2, d2Q.tt),
            ("Vt", 1, 2, d2Q.Vt), ("Vα", 1, 3, d2Q.Vα),
            ("tα", 2, 3, d2Q.tα), ("αα", 3, 3, d2Q.αα),
        )
            fd = fd_d2(Q_at, V, t, α, ix, iy)
            @test isapprox(analytic, fd; atol = atol, rtol = rtol)
        end
    end

    # Inverter points: need cos α_i + β/(Vt) < 1 (else u_i < -1 → past clamp).
    # With x_t=0.06, I_dc=1.2 (β≈0.051) and Vt≈1, that requires α ≳ 20°.
    @testset "Inverter ($(V), $(t), $(α))" for (V, t, α) in (
        (1.00, 1.00, deg2rad(30)),
        (1.02, 0.98, deg2rad(35)),
        (0.95, 1.05, deg2rad(25)),
    )
        x_t = 0.06
        I_dc = 1.2
        σ = -1
        ϕ = ϕ_lcc(V, t, α, x_t, I_dc, σ)

        d2P = PF._d2P_lcc(V, t, α, I_dc, σ)
        d2Q = PF._d2Q_lcc(V, t, α, x_t, I_dc, ϕ, σ)

        P_at = (a, b, c) -> P_lcc(a, b, c, x_t, I_dc, σ)
        Q_at = (a, b, c) -> Q_lcc(a, b, c, x_t, I_dc, σ)

        for (ix, iy, analytic) in (
            (1, 1, d2P.VV), (2, 2, d2P.tt), (1, 2, d2P.Vt),
            (1, 3, d2P.Vα), (2, 3, d2P.tα), (3, 3, d2P.αα),
        )
            fd = fd_d2(P_at, V, t, α, ix, iy)
            @test isapprox(analytic, fd; atol = atol, rtol = rtol)
        end
        for (ix, iy, analytic) in (
            (1, 1, d2Q.VV), (2, 2, d2Q.tt), (1, 2, d2Q.Vt),
            (1, 3, d2Q.Vα), (2, 3, d2Q.tα), (3, 3, d2Q.αα),
        )
            fd = fd_d2(Q_at, V, t, α, ix, iy)
            @test isapprox(analytic, fd; atol = atol, rtol = rtol)
        end
    end

    @testset "Clamp guard returns zeros" begin
        # Pass a ϕ with sin(ϕ) below the tolerance directly; mirrors how the
        # existing Jacobian helpers behave on the clamp branch.
        V, t, α = 1.0, 1.0, deg2rad(10)
        x_t, I_dc = 0.05, 1.5
        ϕ = 1e-10                # sin(ϕ) ≈ 1e-10 < LCC_sinϕ_TOLERANCE
        @test sin(ϕ) < PF.LCC_sinϕ_TOLERANCE
        d2Q = PF._d2Q_lcc(V, t, α, x_t, I_dc, ϕ, +1)
        @test d2Q == (VV = 0.0, tt = 0.0, Vt = 0.0, Vα = 0.0, tα = 0.0, αα = 0.0)
    end
end
