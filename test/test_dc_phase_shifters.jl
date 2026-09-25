# DC phase-shifter injection modeling: α as an injection in the three DC solve methods
# (ABA, PTDF, vPTDF), circulating flow in per-member distribution, and loss-factor
# consistency. See .claude/plans/2026-07-28-dc-phase-shift-injections.md (PowerNetworkMatrices.jl
# checkout) for the full design and worked closed forms.

# 2-bus system, base 100 MVA: bus 1 REF with a generator at `load_p` output, bus 2 PV with a
# load of `load_p`; branches L1 (r=0, x=0.1) and PST (TwoWindingTransformer, tap=1, α=0.15,
# r=pst_r, x=0.2) both on arc (1, 2). Worked closed form (Line x=0.1 ∥ PST x=0.2,α=0.15):
# b_eq=15, α_eq=0.05, b_eq·α_eq=0.75.
function _dc_line_pst_parallel_sys(; load_p::Float64 = 0.0, pst_r::Float64 = 0.0)
    sys = PSY.System(100.0)
    b1 = _add_simple_bus!(sys, 1, PSY.ACBusTypes.REF, 230.0)
    b2 = _add_simple_bus!(sys, 2, PSY.ACBusTypes.PV, 230.0)
    _add_simple_source!(sys, b1, load_p, 0.0)
    # _add_simple_load! per-unitizes active_power by a fixed 1 MVA device base, so scale by
    # the system base to make `load_p` mean per-unit-at-system-base (matching the source and
    # the closed-form constants above).
    _add_simple_load!(sys, b2, load_p * PSY.get_base_power(sys), 0.0)
    line = PSY.Line(;
        name = "L1",
        available = true,
        active_power_flow = 0.0,
        reactive_power_flow = 0.0,
        arc = PSY.Arc(; from = b1, to = b2),
        r = 0.0,
        x = 0.1,
        b = (from = 0.0, to = 0.0),
        rating = 2.0,
        angle_limits = (min = -pi / 2, max = pi / 2),
    )
    add_component!(sys, line)
    pst = PSY.TwoWindingTransformer(;
        name = "PST",
        circuit = PSY.TransformerCircuit(;
            available = true,
            arc = PSY.Arc(; from = b1, to = b2),
            r = pst_r,
            x = 0.2,
            tap = 1.0,
            α = 0.15,
            rating = 2.0,
            base_power = 100.0,
            control_limits = (min = -0.7, max = 0.7),
        ),
    )
    add_component!(sys, pst)
    return sys
end

@testset "DC PST: phase-shift terms populated at init" begin
    sys = _dc_line_pst_parallel_sys()
    data = PowerFlowData(DCPowerFlow(), sys)
    bus_lookup = PF.get_bus_lookup(data)
    arc_lookup = PF.get_arc_lookup(data)
    @test data.arc_phase_shift_flow_offsets[arc_lookup[(1, 2)]] ≈ 0.75
    @test data.bus_phase_shift_injections[bus_lookup[1]] ≈ 0.75
    @test data.bus_phase_shift_injections[bus_lookup[2]] ≈ -0.75
    @test sum(data.bus_phase_shift_injections) ≈ 0.0 atol = 1e-12

    # α-free system: strictly zero everywhere (no-op guarantee).
    sys14 = PSB.build_system(PSB.PSITestSystems, "c_sys14")
    data14 = PowerFlowData(DCPowerFlow(), sys14)
    @test all(iszero, data14.bus_phase_shift_injections)
    @test all(iszero, data14.arc_phase_shift_flow_offsets)
end

# 3-bus loop, zero net injections: L12 (x=0.1), L23 (x=0.1), PST13 on arc (1, 3)
# (tap=1, α=0.15, x=0.2). Closed form: loop flow f = α / (x12+x23+x13·tap) = 0.15/0.4 =
# 0.375 around 1→2→3; the (1,3) arc carries −0.375 in its own from→to orientation.
function _dc_pst_loop_sys()
    sys = PSY.System(100.0)
    b1 = _add_simple_bus!(sys, 1, PSY.ACBusTypes.REF, 230.0)
    b2 = _add_simple_bus!(sys, 2, PSY.ACBusTypes.PV, 230.0)
    b3 = _add_simple_bus!(sys, 3, PSY.ACBusTypes.PV, 230.0)
    _add_simple_source!(sys, b1, 0.0, 0.0)
    _add_simple_load!(sys, b2, 0.0, 0.0)
    _add_simple_load!(sys, b3, 0.0, 0.0)
    line12 = PSY.Line(;
        name = "L12",
        available = true,
        active_power_flow = 0.0,
        reactive_power_flow = 0.0,
        arc = PSY.Arc(; from = b1, to = b2),
        r = 0.0,
        x = 0.1,
        b = (from = 0.0, to = 0.0),
        rating = 2.0,
        angle_limits = (min = -pi / 2, max = pi / 2),
    )
    add_component!(sys, line12)
    line23 = PSY.Line(;
        name = "L23",
        available = true,
        active_power_flow = 0.0,
        reactive_power_flow = 0.0,
        arc = PSY.Arc(; from = b2, to = b3),
        r = 0.0,
        x = 0.1,
        b = (from = 0.0, to = 0.0),
        rating = 2.0,
        angle_limits = (min = -pi / 2, max = pi / 2),
    )
    add_component!(sys, line23)
    pst = PSY.TwoWindingTransformer(;
        name = "PST13",
        circuit = PSY.TransformerCircuit(;
            available = true,
            arc = PSY.Arc(; from = b1, to = b3),
            r = 0.0,
            x = 0.2,
            tap = 1.0,
            α = 0.15,
            rating = 2.0,
            base_power = 100.0,
            control_limits = (min = -0.7, max = 0.7),
        ),
    )
    add_component!(sys, pst)
    return sys
end

@testset "DC PST: parallel Line∥PST closed form, all methods" begin
    for load_p in (0.0, 0.9)
        sys = _dc_line_pst_parallel_sys(; load_p = load_p)
        for pf_model in (DCPowerFlow(), PTDFDCPowerFlow(), vPTDFDCPowerFlow())
            data = PowerFlowData(pf_model, sys)
            solve_power_flow!(data)
            bus_lookup = PF.get_bus_lookup(data)
            arc_lookup = PF.get_arc_lookup(data)
            ix = arc_lookup[(1, 2)]
            dθ_expected = load_p / 15 + 0.05
            @test data.bus_angles[bus_lookup[1], 1] -
                  data.bus_angles[bus_lookup[2], 1] ≈ dθ_expected atol = 1e-9
            # Arc flow carries the −b·α offset: f = 15·Δθ − 0.75 = load_p.
            @test data.arc_active_power_flow_from_to[ix, 1] ≈ load_p atol = 1e-9
            @test data.arc_active_power_flow_to_from[ix, 1] ≈ -load_p atol = 1e-9
        end
    end
end

@testset "DC PST: pure circulating loop flow, all methods" begin
    sys = _dc_pst_loop_sys()
    for pf_model in (DCPowerFlow(), PTDFDCPowerFlow(), vPTDFDCPowerFlow())
        data = PowerFlowData(pf_model, sys)
        solve_power_flow!(data)
        arc_lookup = PF.get_arc_lookup(data)
        @test data.arc_active_power_flow_from_to[arc_lookup[(1, 2)], 1] ≈ 0.375 atol = 1e-9
        @test data.arc_active_power_flow_from_to[arc_lookup[(2, 3)], 1] ≈ 0.375 atol = 1e-9
        @test data.arc_active_power_flow_from_to[arc_lookup[(1, 3)], 1] ≈ -0.375 atol = 1e-9
    end
end

@testset "DC PST: per-member flows include circulating component" begin
    sys = _dc_line_pst_parallel_sys(; load_p = 0.9)
    data = PowerFlowData(DCPowerFlow(), sys)
    solve_power_flow!(data)
    results = write_results(data, sys, PF.FlowReporting.BRANCH_FLOWS)
    df = results["1"]["flow_results"]
    base = PSY.get_base_power(sys)
    p_line = only(df[df.flow_name .== "L1", :P_from_to])
    p_pst = only(df[df.flow_name .== "PST", :P_from_to])
    @test p_line ≈ ((2 / 3) * 0.9 + 0.5) * base atol = 1e-6
    @test p_pst ≈ (0.9 / 3 - 0.5) * base atol = 1e-6
    @test p_line + p_pst ≈ 0.9 * base atol = 1e-6          # members sum to the arc flow
    p_line_tf = only(df[df.flow_name .== "L1", :P_to_from])
    @test p_line_tf ≈ -p_line atol = 1e-6                   # lossless antisymmetry per member
end

@testset "DC PST: lossy shifted parallel group solves without throwing" begin
    sys = _dc_line_pst_parallel_sys(; load_p = 0.9, pst_r = 0.05)
    for pf_model in (DCPowerFlow(), PTDFDCPowerFlow(), vPTDFDCPowerFlow())
        data = PowerFlowData(pf_model, sys)   # today: throws on first solve (_make_dc_scratch)
        solve_power_flow!(data)
        @test all(isfinite, data.arc_active_power_flow_from_to)
    end
end

@testset "DC PST: loss factors use actual flows" begin
    sys = _dc_line_pst_parallel_sys(; load_p = 0.9, pst_r = 0.05)
    data = PowerFlowData(PTDFDCPowerFlow(; calculate_loss_factors = true), sys)
    solve_power_flow!(data)
    Rs = PF._get_arc_resistances(data)
    ptdf_t = data.power_network_matrix.data
    expected = 2 .* (ptdf_t * (Rs .* data.arc_active_power_flow_from_to))
    @test data.loss_factors ≈ expected atol = 1e-10
end

# --- Task 9: integration --------------------------------------------------------------

@testset "DC PST: multi-period broadcasts alpha, superposes with load" begin
    sys = _dc_line_pst_parallel_sys(; load_p = 0.9)
    for pf_model in (DCPowerFlow(; time_steps = 2), PTDFDCPowerFlow(; time_steps = 2))
        data = PowerFlowData(pf_model, sys)
        bus_lookup = PF.get_bus_lookup(data)
        arc_lookup = PF.get_arc_lookup(data)
        # Column 1: pure-α response (zero load); column 2: superposed with the 0.9 pu load
        # already broadcast at construction from the fixture.
        data.bus_active_power_withdrawals[bus_lookup[2], :] .= [0.0, 0.9]
        solve_power_flow!(data)
        ix = arc_lookup[(1, 2)]
        for (t, load_p) in enumerate((0.0, 0.9))
            dθ_expected = load_p / 15 + 0.05
            @test data.bus_angles[bus_lookup[1], t] -
                  data.bus_angles[bus_lookup[2], t] ≈ dθ_expected atol = 1e-9
            @test data.arc_active_power_flow_from_to[ix, t] ≈ load_p atol = 1e-9
        end
    end
end

# Extends the loop fixture with a degree-two bus (5) so `DegreeTwoReduction` merges it into
# a series chain containing a PST: bus 5 connects only line (1, 5, x=0.1) and PST (5, 3,
# α=0.1, x=0.2), and carries no injection.
function _dc_pst_loop_with_series_pst_sys()
    sys = _dc_pst_loop_sys()
    b1 = PSY.get_component(PSY.ACBus, sys, "bus_1")
    b3 = PSY.get_component(PSY.ACBus, sys, "bus_3")
    b5 = _add_simple_bus!(sys, 5, PSY.ACBusTypes.PQ, 230.0)
    line15 = PSY.Line(;
        name = "L15",
        available = true,
        active_power_flow = 0.0,
        reactive_power_flow = 0.0,
        arc = PSY.Arc(; from = b1, to = b5),
        r = 0.0,
        x = 0.1,
        b = (from = 0.0, to = 0.0),
        rating = 2.0,
        angle_limits = (min = -pi / 2, max = pi / 2),
    )
    add_component!(sys, line15)
    pst53 = PSY.TwoWindingTransformer(;
        name = "PST53",
        circuit = PSY.TransformerCircuit(;
            available = true,
            arc = PSY.Arc(; from = b5, to = b3),
            r = 0.0,
            x = 0.2,
            tap = 1.0,
            α = 0.1,
            rating = 2.0,
            base_power = 100.0,
            control_limits = (min = -0.7, max = 0.7),
        ),
    )
    add_component!(sys, pst53)
    return sys
end

@testset "DC PST: reduced vs unreduced series chain containing a PST" begin
    sys = _dc_pst_loop_with_series_pst_sys()
    unreduced = PowerFlowData(DCPowerFlow(), sys)
    solve_power_flow!(unreduced)
    @test all(unreduced.converged)
    reductions = PNM.NetworkReduction[PNM.DegreeTwoReduction()]
    reduced_pf = DCPowerFlow(; network_reductions = reductions)
    # The unreduced side exercises direct-PST α (PST13, PST53); the reduced side exercises
    # the series-chain α path end-to-end (bus 5 merged into a series chain with PST53).
    validate_reduced_power_flow(reduced_pf, sys, reductions, unreduced)
end

# The lossy path is trig-exact (P_ft = Σₘ bₘ·sin(Δθ−αₘ)); it matches the linear closed form
# (b·(Δθ−α)) only as Δθ, α → 0, so only the shared/exact θ-solve is pinned tightly here.
@testset "DC PST: lossy_flows path is theta-consistent with the lossless closed form" begin
    sys = _dc_line_pst_parallel_sys(; load_p = 0.9)
    data = PowerFlowData(DCPowerFlow(; lossy_flows = true), sys)
    solve_power_flow!(data)
    bus_lookup = PF.get_bus_lookup(data)
    dθ_expected = 0.9 / 15 + 0.05
    @test data.bus_angles[bus_lookup[1], 1] - data.bus_angles[bus_lookup[2], 1] ≈
          dθ_expected atol = 1e-9
    arc_lookup = PF.get_arc_lookup(data)
    ix = arc_lookup[(1, 2)]
    @test all(isfinite, data.arc_active_power_flow_from_to[:, 1])
    # Loose sanity bound, not an exact match: the trig-based flow differs from the linear
    # closed form by Θ(Δθ³, α³) per member, which is small but not 1e-6-small here.
    @test isapprox(data.arc_active_power_flow_from_to[ix, 1], 0.9; atol = 0.01)
end

@testset "DC PST: lossy_flows with lossy shifted group completes finitely" begin
    sys = _dc_line_pst_parallel_sys(; load_p = 0.9, pst_r = 0.05)
    data = PowerFlowData(DCPowerFlow(; lossy_flows = true), sys)
    solve_power_flow!(data)
    @test all(x -> x >= 0.0, data.arc_active_power_losses)
    @test all(isfinite, data.arc_active_power_flow_from_to)
    @test all(isfinite, data.arc_active_power_flow_to_from)
end
