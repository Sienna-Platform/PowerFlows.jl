# An LCC's two terminals can resolve to one bus, and several LCCs can share one arc. The
# Jacobian must add the from and to terms when the terminals alias, and results must not
# identify an LCC by its arc.

const _ALIASING_FORMULATIONS = (
    ("polar", ACPowerFlow{NewtonRaphsonACPowerFlow}),
    ("rect-CI", PF.ACRectangularPowerFlow{NewtonRaphsonACPowerFlow}),
    ("mixed-CPB", PF.ACMixedPowerFlow{NewtonRaphsonACPowerFlow}),
)

# Triangle 1-2-3 with a REF at bus 1. Bus 4 has no connection yet.
function _lcc_aliasing_core()
    sys = System(100.0)
    b1 = _add_simple_bus!(sys, 1, ACBusTypes.REF, 230, 1.05, 0.0)
    b2 = _add_simple_bus!(sys, 2, ACBusTypes.PQ, 230, 1.05, 0.0)
    b3 = _add_simple_bus!(sys, 3, ACBusTypes.PQ, 230, 1.05, 0.0)
    b4 = _add_simple_bus!(sys, 4, ACBusTypes.PQ, 230, 1.05, 0.0)
    _add_simple_load!(sys, b2, 10, 5)
    _add_simple_load!(sys, b3, 60, 20)
    _add_simple_line!(sys, b1, b2, 5e-3, 5e-3, 1e-3)
    _add_simple_line!(sys, b1, b3, 5e-3, 5e-3, 1e-3)
    _add_simple_line!(sys, b2, b3, 5e-3, 5e-3, 1e-3)
    _add_simple_source!(sys, b1, 0.0, 0.0)
    return sys, b2, b3, b4
end

# Bus 4 is a leaf on bus 2 and the LCC is parallel to that line, so RadialReduction merges
# the two LCC terminals.
function _lcc_parallel_to_radial_line()
    sys, b2, _, b4 = _lcc_aliasing_core()
    _add_simple_line!(sys, b2, b4, 5e-3, 5e-3, 1e-3)
    _add_simple_lcc!(sys, b2, b4, 0.05, 0.05, 0.08)
    return sys
end

# Two LCCs with different set points on the arc 2 → 3.
function _parallel_lccs()
    sys, b2, b3, b4 = _lcc_aliasing_core()
    _add_simple_line!(sys, b3, b4, 5e-3, 5e-3, 1e-3)
    lcc_a = _add_simple_lcc!(sys, b2, b3, 0.05, 0.05, 0.08)
    set_name!(sys, lcc_a, "LCC_a")
    lcc_b = _add_simple_lcc!(sys, b2, b3, 0.05, 0.05, 0.08)
    set_name!(sys, lcc_b, "LCC_b")
    set_power_transfer_setpoint!(lcc_b, 0.2 * u"CU")
    return sys
end

@testset "LCC with merged terminals: Jacobian matches finite differences ($name)" for (
    name,
    PFType,
) in _ALIASING_FORMULATIONS
    pf = PFType(; network_reductions = PNM.NetworkReduction[PNM.RadialReduction()])
    data = PowerFlowData(pf, _lcc_parallel_to_radial_line())
    @test data.lcc.bus_indices[1][1] == data.lcc.bus_indices[1][2]

    residual, jac, x0 = PF.initialize_power_flow_variables(pf, data, 1)
    Random.seed!(1)
    x = x0 .+ 0.01 .* randn(length(x0))
    residual(data, x, 1)
    jac(data, 1)
    J = copy(Matrix(jac.Jv))

    fd = zeros(size(J))
    ε = 1e-6
    for k in axes(J, 2)
        xk = copy(x)
        xk[k] += ε
        residual(data, xk, 1)
        Fp = copy(residual.Rv)
        xk[k] -= 2ε
        residual(data, xk, 1)
        fd[:, k] = (Fp .- residual.Rv) ./ (2ε)
    end
    @test maximum(abs.(J .- fd)) / maximum(abs.(fd)) < 1e-8
end

@testset "Parallel LCCs keep their own names and solution" begin
    pf = ACPowerFlow{NewtonRaphsonACPowerFlow}()
    sys = _parallel_lccs()
    data = PowerFlowData(pf, sys)
    @test solve_power_flow!(data)
    @test sort(PF.get_lcc_names(data)) == ["LCC_a", "LCC_b"]
    # The set points make the flows differ, so a name swap or a lost name fails here.
    for (name, flow) in zip(PF.get_lcc_names(data), data.lcc.arc_active_power_flow_from_to)
        @test flow ≈ (name == "LCC_a" ? 0.5 : 0.2) rtol = 1e-6
    end

    results = solve_power_flow(pf, sys)
    lcc_df = results["lcc_results"]
    @test lcc_df.P_from_to[findfirst(==("LCC_a"), lcc_df.line_name)] ≈ 50.0 rtol = 1e-6
    @test lcc_df.P_from_to[findfirst(==("LCC_b"), lcc_df.line_name)] ≈ 20.0 rtol = 1e-6

    @test solve_and_store_power_flow!(pf, sys)
    lcc_a = get_component(TwoTerminalLCCLine, sys, "LCC_a")
    lcc_b = get_component(TwoTerminalLCCLine, sys, "LCC_b")
    @test get_active_power_flow(lcc_a, u"SU") ≈ 0.5 rtol = 1e-6
    @test get_active_power_flow(lcc_b, u"SU") ≈ 0.2 rtol = 1e-6
end
