# End-to-end NR solve for the Mixed Current-Power Balance (MCPB) formulation,
# validated against polar NR and rectangular CI NR within a tight 1e-7 parity
# tolerance (Vm/θ/P_gen/Q_gen).
#
# The shared parity helpers (`_mixed_polar_parity`,
# `_mixed_polar_parity_data`), the `MIXED_PARITY_ATOL` constant and
# `_mixed_pf_settings` live in `test_mixed_cpb_polar_parity.jl` (single
# definition site — the ReTest runner auto-includes every `test_*.jl` into one
# module). This file only exercises the NR-specific scenario coverage.

@testset "Mixed CPB Power Flow: NR parity with polar and rectangular" begin
    fixtures = [
        ("c_sys5", true),
        ("c_sys14", false),
    ]
    for (name, with_forecasts) in fixtures
        @testset "$name" begin
            sys_p = if with_forecasts
                PSB.build_system(PSB.PSITestSystems, name)
            else
                PSB.build_system(PSB.PSITestSystems, name; add_forecasts = false)
            end
            sys_h = deepcopy(sys_p)
            sys_r = deepcopy(sys_p)
            _mixed_polar_parity(sys_p, sys_h; sys_r = sys_r)
        end
    end
end

@testset "Mixed CPB Power Flow: distributed slack (participation factors)" begin
    sys_p = PSB.build_system(PSB.PSITestSystems, "c_sys14"; add_forecasts = false)
    sys_h = deepcopy(sys_p)
    spf = Dict{Tuple{DataType, String}, Float64}(
        (ThermalStandard, get_name(g)) => 1.0
        for g in get_components(ThermalStandard, sys_p)
    )
    _mixed_polar_parity(
        sys_p,
        sys_h;
        pf_kwargs = (; generator_slack_participation_factors = spf),
    )
end

@testset "Mixed CPB Power Flow: headroom-proportional distributed slack" begin
    sys_p = PSB.build_system(PSB.PSITestSystems, "c_sys14"; add_forecasts = false)
    sys_h = deepcopy(sys_p)
    _mixed_polar_parity(
        sys_p,
        sys_h;
        pf_kwargs = (; distribute_slack_proportional_to_headroom = true),
    )
end

# Q-limit enforcement triggers PV → PQ bus-type switching in the `_ac_power_flow`
# outer loop. The MCPB residual/Jacobian must be REBUILT against the mutated
# `data.bus_type` on each outer-loop pass (the constructors re-read
# `data.bus_type[:, ts]` via `compute_mixed_bus_state_offsets`). Parity vs polar
# and rectangular NR (all four bus quantities) validates this rebuild.
@testset "Mixed CPB Power Flow: Q-limit enforcement (PV → PQ switching, c_sys14)" begin
    sys_p = PSB.build_system(PSB.PSITestSystems, "c_sys14"; add_forecasts = false)
    sys_h = deepcopy(sys_p)
    sys_r = deepcopy(sys_p)
    _mixed_polar_parity(
        sys_p,
        sys_h;
        sys_r = sys_r,
        pf_kwargs = (; check_reactive_power_limits = true),
    )
end

@testset "Mixed CPB Power Flow: Q-limit enforcement (PV → PQ switching, c_sys5)" begin
    sys_p = PSB.build_system(PSB.PSITestSystems, "c_sys5")
    sys_h = deepcopy(sys_p)
    sys_r = deepcopy(sys_p)
    _mixed_polar_parity(
        sys_p,
        sys_h;
        sys_r = sys_r,
        pf_kwargs = (; check_reactive_power_limits = true),
    )
end

# Radial network reduction changes the Y-bus and the retained bus set. The MCPB
# residual (effective Y-bus build), finalize (raw-Y walk) and bus-state offsets
# must all use the reduced `data.power_network_matrix` consistently.
@testset "Mixed CPB Power Flow: radial network reduction" begin
    sys_p = PSB.build_system(
        PSB.PSSEParsingTestSystems, "psse_14_network_reduction_test_system")
    sys_h = deepcopy(sys_p)
    sys_r = deepcopy(sys_p)
    _mixed_polar_parity(
        sys_p,
        sys_h;
        sys_r = sys_r,
        pf_kwargs = (;
            network_reductions = PNM.NetworkReduction[PNM.RadialReduction()]),
    )
end

# 2k flat-start sanity: ACTIVSg2000 is large enough that an enhanced flat start
# is a meaningfully different (and harder) initial condition than the System
# setpoints. Asserts the MCPB NR solver converges from `enhanced_flat_start =
# true` and matches polar NR within MIXED_PARITY_ATOL_2K on Vm/θ. One solve
# per formulation (no loop) — the suite already builds 2k elsewhere.
@testset "Mixed CPB Power Flow: 2k flat-start parity (ACTIVSg2000)" begin
    sys_p = PSB.build_system(PSB.MatpowerTestSystems, "matpower_ACTIVSg2000_sys")
    sys_h = PSB.build_system(PSB.MatpowerTestSystems, "matpower_ACTIVSg2000_sys")
    pf_p = ACPowerFlow{NewtonRaphsonACPowerFlow}(;
        correct_bustypes = true,
        enhanced_flat_start = true,
    )
    pf_h = ACMixedPowerFlow{NewtonRaphsonACPowerFlow}(;
        correct_bustypes = true,
        enhanced_flat_start = true,
        solver_settings = _mixed_pf_settings(),
    )
    res_p = solve_power_flow(pf_p, sys_p)
    res_h = solve_power_flow(pf_h, sys_h)
    @test res_p !== missing
    @test res_h !== missing
    bus_p = res_p["bus_results"]
    bus_h = res_h["bus_results"]
    @test maximum(abs.(bus_p.Vm .- bus_h.Vm)) < MIXED_PARITY_ATOL_2K
    @test maximum(abs.(bus_p.θ .- bus_h.θ)) < MIXED_PARITY_ATOL_2K
end

@testset "Mixed CPB Power Flow: multi-period (same network, no time-varying loads)" begin
    sys_p = PSB.build_system(PSB.PSITestSystems, "c_sys14"; add_forecasts = false)
    sys_h = deepcopy(sys_p)
    pf_p = ACPowerFlow{NewtonRaphsonACPowerFlow}(; time_steps = 3)
    pf_h = ACMixedPowerFlow{NewtonRaphsonACPowerFlow}(;
        time_steps = 3,
        solver_settings = _mixed_pf_settings(),
    )
    _mixed_polar_parity_data(pf_p, pf_h, sys_p, sys_h)
end

@testset "Mixed CPB Power Flow: multi-period time-varying distributed slack" begin
    sys_p = PSB.build_system(PSB.PSITestSystems, "c_sys14"; add_forecasts = false)
    sys_h = deepcopy(sys_p)
    gens = collect(get_components(ThermalStandard, sys_p))
    # One participation dict per time step makes the slack split time-varying.
    spf = [
        Dict{Tuple{DataType, String}, Float64}(
            (ThermalStandard, get_name(g)) => (g === gens[k] ? 2.0 : 1.0)
            for g in gens)
        for k in (1, length(gens))
    ]
    pf_p = ACPowerFlow{NewtonRaphsonACPowerFlow}(;
        time_steps = 2, generator_slack_participation_factors = spf)
    pf_h = ACMixedPowerFlow{NewtonRaphsonACPowerFlow}(;
        time_steps = 2, generator_slack_participation_factors = spf,
        solver_settings = _mixed_pf_settings())
    _mixed_polar_parity_data(pf_p, pf_h, sys_p, sys_h)
end

# Same topology as polar's `_two_swing_system()` (test_jacobian.jl), named distinctly
# since ReTest merges every test_*.jl into one module. The nonzero swing-2 angle
# exercises real off-diagonal ∂P/∂θ terms. Shared by test_mixed_cpb_jacobian.jl and
# test_mixed_cpb_polar_parity.jl.
function _two_swing_mixed_system()
    sys = System(100.0)
    b1 = _add_simple_bus!(sys, 1, ACBusTypes.REF, 230, 1.06, 0.0)
    b2 = _add_simple_bus!(sys, 2, ACBusTypes.REF, 230, 1.05, 0.05)
    b3 = _add_simple_bus!(sys, 3, ACBusTypes.PQ, 230, 1.0, 0.0)
    _add_simple_source!(sys, b1, 0.0, 0.0)
    _add_simple_source!(sys, b2, 0.0, 0.0)
    _add_simple_load!(sys, b3, 40, 15)
    _add_simple_line!(sys, b1, b3, 5e-3, 5e-3, 1e-3)
    _add_simple_line!(sys, b2, b3, 5e-3, 5e-3, 1e-3)
    return sys
end

@testset "Mixed CPB Power Flow: multi-swing (two swings in one island each self-balance)" begin
    @testset "$(V)" for V in (NewtonRaphsonACPowerFlow, TrustRegionACPowerFlow)
        sys = _two_swing_mixed_system()
        pf = ACMixedPowerFlow{V}(; correct_bustypes = false)
        res = solve_power_flow(pf, sys)
        @test res !== missing
        bus_res = res["bus_results"]
        vm1 = bus_res[bus_res.bus_number .== 1, :Vm][1]
        va1 = bus_res[bus_res.bus_number .== 1, :θ][1]
        vm2 = bus_res[bus_res.bus_number .== 2, :Vm][1]
        va2 = bus_res[bus_res.bus_number .== 2, :θ][1]
        @test vm1 ≈ 1.06 atol = 1e-6
        @test va1 ≈ 0.0 atol = 1e-6
        @test vm2 ≈ 1.05 atol = 1e-6
        @test va2 ≈ 0.05 atol = 1e-6
        # The fixture is asymmetric (swing 2 leads by 0.05 rad, driving circulating
        # flow), so assert net P > load rather than a per-swing split.
        p1 = bus_res[bus_res.bus_number .== 1, :P_gen][1]
        p2 = bus_res[bus_res.bus_number .== 2, :P_gen][1]
        @test p1 + p2 > 40.0
        polar_res = solve_power_flow(
            ACPowerFlow{V}(; correct_bustypes = false), _two_swing_mixed_system())
        polar_bus = polar_res["bus_results"]
        for col in (:Vm, :θ, :P_gen, :Q_gen)
            @test isapprox(bus_res[!, col], polar_bus[!, col]; atol = 1e-6)
        end
    end
end
