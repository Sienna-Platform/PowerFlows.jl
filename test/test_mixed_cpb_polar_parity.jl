# Solve representative scenarios with the Mixed Current-Power Balance (MCPB)
# formulation and assert Vm/θ/P_gen/Q_gen parity vs polar (and, when requested,
# rectangular CI) within a tight tolerance. Mirrors
# `test_rectangular_ci_polar_parity.jl`'s fixture matrix, extended to validate
# BOTH the Newton-Raphson and Trust-Region solvers on the MCPB Jacobian.
#
# This file is the single home of the `_mixed_polar_parity` /
# `_mixed_polar_parity_data` helpers and the `MIXED_PARITY_ATOL` /
# `_mixed_pf_settings` definitions: the ReTest runner auto-includes every
# `test_*.jl` into one module, so these must be defined exactly once.

const MIXED_PARITY_ATOL = 1e-7
# ACTIVSg2000: zero-injection buses with G_ii ≈ 0 make the MCPB Jacobian more
# ill-conditioned than the small synthetic systems. The imag-first column
# ordering + KLU partial pivoting keep it solvable, but the converged-state
# round-off floor is looser than 1e-7; 1e-5 still pins formulation parity.
const MIXED_PARITY_ATOL_2K = 1e-5

_mixed_pf_settings() = Dict{Symbol, Any}(:validate_voltage_magnitudes => false)

# Assert MCPB matches polar (and, when requested, rectangular CI) on the four
# reported bus quantities, for an arbitrary AC solver. Mirrors
# `_rect_polar_parity`, parametrized over `solver` so the same fixture matrix
# validates Newton-Raphson and Trust-Region against the MCPB Jacobian.
function _mixed_polar_parity(
    sys_p::PSY.System,
    sys_h::PSY.System;
    sys_r::Union{Nothing, PSY.System} = nothing,
    pf_kwargs::NamedTuple = NamedTuple(),
    atol::Float64 = MIXED_PARITY_ATOL,
    solver = NewtonRaphsonACPowerFlow,
)
    pf_p = ACPowerFlow{solver}(; pf_kwargs...)
    pf_h = ACMixedPowerFlow{solver}(;
        pf_kwargs...,
        solver_settings = _mixed_pf_settings(),
    )
    res_p = solve_power_flow(pf_p, sys_p)
    res_h = solve_power_flow(pf_h, sys_h)
    @test res_p !== missing
    @test res_h !== missing
    bus_p = res_p["bus_results"]
    bus_h = res_h["bus_results"]
    @test maximum(abs.(bus_p.Vm .- bus_h.Vm)) < atol
    @test maximum(abs.(bus_p.θ .- bus_h.θ)) < atol
    # P_gen / Q_gen parity catches slack-recovery and Q-writeback bugs that
    # Vm/θ parity alone cannot — the internal residual math can converge to the
    # correct voltages while the reported generator outputs disagree.
    @test maximum(abs.(bus_p.P_gen .- bus_h.P_gen)) < atol
    @test maximum(abs.(bus_p.Q_gen .- bus_h.Q_gen)) < atol
    if sys_r !== nothing
        pf_r = ACRectangularPowerFlow{solver}(;
            pf_kwargs...,
            solver_settings = _mixed_pf_settings(),
        )
        res_r = solve_power_flow(pf_r, sys_r)
        @test res_r !== missing
        bus_r = res_r["bus_results"]
        @test maximum(abs.(bus_r.Vm .- bus_h.Vm)) < atol
        @test maximum(abs.(bus_r.θ .- bus_h.θ)) < atol
        @test maximum(abs.(bus_r.P_gen .- bus_h.P_gen)) < atol
        @test maximum(abs.(bus_r.Q_gen .- bus_h.Q_gen)) < atol
    end
    return
end

# Multi-period analogue of `_mixed_polar_parity`: solve both formulations in
# place and assert full per-time-step state-array parity. Exercises the minimal
# per-step `improve_x0` / per-ts offsets & caches (`time_step` threaded
# correctly). This checks per-step correctness, not warm-start efficiency.
function _mixed_polar_parity_data(
    pf_p::ACPowerFlow,
    pf_h::PF.ACMixedPowerFlow,
    sys_p::PSY.System,
    sys_h::PSY.System;
    atol::Float64 = MIXED_PARITY_ATOL,
)
    data_p = PowerFlowData(pf_p, sys_p)
    data_h = PowerFlowData(pf_h, sys_h)
    @test PowerFlows.solve_power_flow!(data_p)
    @test PowerFlows.solve_power_flow!(data_h)
    n_ts = size(data_p.bus_magnitude, 2)
    for ts in 1:n_ts
        @testset "time step $ts" begin
            @test maximum(
                abs.(data_p.bus_magnitude[:, ts] - data_h.bus_magnitude[:, ts]),
            ) < atol
            @test maximum(
                abs.(data_p.bus_angles[:, ts] - data_h.bus_angles[:, ts]),
            ) < atol
            @test maximum(
                abs.(
                    data_p.bus_active_power_injections[:, ts] -
                    data_h.bus_active_power_injections[:, ts]
                ),
            ) < atol
            @test maximum(
                abs.(
                    data_p.bus_reactive_power_injections[:, ts] -
                    data_h.bus_reactive_power_injections[:, ts]
                ),
            ) < atol
        end
    end
    return
end

# Solvers validated against the MCPB Jacobian: Newton-Raphson and Trust-Region
# (the shared `_trust_region_step` dogleg runs on the mixed sparse J unchanged).
const MIXED_PARITY_SOLVERS =
    (NewtonRaphsonACPowerFlow, TrustRegionACPowerFlow)

# Per-solver, parametrized rerun of the rectangular polar-parity fixture matrix.
# NR additionally asserts rectangular-CI parity (the rect file's reference
# formulation); TR asserts polar parity only (mirrors the rect LCC file, where
# TR is validated against polar). Tolerance is shared with the rect file.
@testset "Mixed CPB polar parity ($(solver))" for solver in
                                                  MIXED_PARITY_SOLVERS
    # NR is the only solver the rect parity file cross-checks against the
    # rectangular CI formulation; reuse that as the third leg for NR only.
    use_rect = solver === NewtonRaphsonACPowerFlow

    @testset "small systems: c_sys5 / c_sys14 / psse_3bus_gen_cls_sys" begin
        # (catalog, name, build_kwargs). `psse_3bus_gen_cls_sys` lives in
        # PSYTestSystems, which does not accept `add_forecasts`.
        fixtures = [
            (PSB.PSITestSystems, "c_sys5", (;)),
            (PSB.PSITestSystems, "c_sys14", (; add_forecasts = false)),
            (PSB.PSYTestSystems, "psse_3bus_gen_cls_sys", (;)),
        ]
        for (cat, name, build_kw) in fixtures
            @testset "$name" begin
                sys_p = PSB.build_system(cat, name; build_kw...)
                sys_h = deepcopy(sys_p)
                sys_r = use_rect ? deepcopy(sys_p) : nothing
                _mixed_polar_parity(
                    sys_p, sys_h;
                    sys_r = sys_r, solver = solver,
                )
            end
        end
    end

    @testset "ZIP loads (constant current)" begin
        sys_p = _build_zip_2bus_system(; current_pq = (2.0, 1.0))
        sys_h = _build_zip_2bus_system(; current_pq = (2.0, 1.0))
        sys_r = use_rect ?
                _build_zip_2bus_system(; current_pq = (2.0, 1.0)) : nothing
        _mixed_polar_parity(
            sys_p, sys_h;
            sys_r = sys_r,
            pf_kwargs = (; correct_bustypes = true),
            solver = solver,
        )
    end

    @testset "ZIP loads (full P+I+Z combination)" begin
        kw = (; power_pq = (0.5, 0.2), current_pq = (2.0, 1.0),
            impedance_pq = (1.5, 0.8))
        sys_p = _build_zip_2bus_system(; kw...)
        sys_h = _build_zip_2bus_system(; kw...)
        sys_r = use_rect ? _build_zip_2bus_system(; kw...) : nothing
        _mixed_polar_parity(
            sys_p, sys_h;
            sys_r = sys_r,
            pf_kwargs = (; correct_bustypes = true),
            solver = solver,
        )
    end

    @testset "LCC HVDC (case5_2_lcc, PQ terminals)" begin
        raw_path = joinpath(TEST_DATA_DIR, "case5_2_lcc.raw")
        sys = make_system(PFP.PowerModelsData(raw_path); runchecks = false)
        sys_p = deepcopy(sys)
        sys_h = deepcopy(sys)
        sys_r = use_rect ? deepcopy(sys) : nothing
        _mixed_polar_parity(
            sys_p, sys_h;
            sys_r = sys_r, solver = solver,
        )
    end

    @testset "headroom-proportional distributed slack (c_sys14)" begin
        sys_p =
            PSB.build_system(PSB.PSITestSystems, "c_sys14"; add_forecasts = false)
        sys_h = deepcopy(sys_p)
        sys_r = use_rect ? deepcopy(sys_p) : nothing
        _mixed_polar_parity(
            sys_p, sys_h;
            sys_r = sys_r,
            pf_kwargs = (; distribute_slack_proportional_to_headroom = true),
            solver = solver,
        )
    end

    @testset "explicit generator participation factors (c_sys14)" begin
        sys_p =
            PSB.build_system(PSB.PSITestSystems, "c_sys14"; add_forecasts = false)
        sys_h = deepcopy(sys_p)
        sys_r = use_rect ? deepcopy(sys_p) : nothing
        spf = Dict{Tuple{DataType, String}, Float64}(
            (ThermalStandard, get_name(g)) => 1.0
            for g in get_components(ThermalStandard, sys_p)
        )
        _mixed_polar_parity(
            sys_p, sys_h;
            sys_r = sys_r,
            pf_kwargs = (; generator_slack_participation_factors = spf),
            solver = solver,
        )
    end
end

# Levenberg-Marquardt on the MCPB Jacobian. The LM driver dispatches on
# `AbstractACPowerFlow{LevenbergMarquardtACPowerFlow}` and the LM workspace /
# step functions are Union-widened to the mixed functors (Task L1), so LM runs
# on the mixed sparse J unchanged. Mirrors the rectangular suite's
# "Rectangular CI: LM matches polar LM" testset (c_sys5 / c_sys14, 1e-7):
# mixed LM must match BOTH polar LM and rectangular LM at the same tolerance.
@testset "Mixed CPB: LM matches polar LM" begin
    for name in ("c_sys5", "c_sys14")
        @testset "$name" begin
            sys = PSB.build_system(PSB.PSITestSystems, name; add_forecasts = false)
            pf_polar = ACPolarPowerFlow{LevenbergMarquardtACPowerFlow}()
            res_polar = solve_power_flow(pf_polar, deepcopy(sys))
            pf_rect = ACRectangularPowerFlow{LevenbergMarquardtACPowerFlow}(;
                solver_settings = _mixed_pf_settings())
            res_rect = solve_power_flow(pf_rect, deepcopy(sys))
            pf_mixed = ACMixedPowerFlow{LevenbergMarquardtACPowerFlow}(;
                solver_settings = _mixed_pf_settings())
            res_mixed = solve_power_flow(pf_mixed, deepcopy(sys))
            @test res_polar !== missing
            @test res_rect !== missing
            @test res_mixed !== missing
            bus_p = res_polar["bus_results"]
            bus_r = res_rect["bus_results"]
            bus_h = res_mixed["bus_results"]
            # Mixed LM vs polar LM.
            @test maximum(abs.(bus_p.Vm .- bus_h.Vm)) < MIXED_PARITY_ATOL
            @test maximum(abs.(bus_p.θ .- bus_h.θ)) < MIXED_PARITY_ATOL
            @test maximum(abs.(bus_p.P_gen .- bus_h.P_gen)) < MIXED_PARITY_ATOL
            @test maximum(abs.(bus_p.Q_gen .- bus_h.Q_gen)) < MIXED_PARITY_ATOL
            # Mixed LM vs rectangular LM.
            @test maximum(abs.(bus_r.Vm .- bus_h.Vm)) < MIXED_PARITY_ATOL
            @test maximum(abs.(bus_r.θ .- bus_h.θ)) < MIXED_PARITY_ATOL
            @test maximum(abs.(bus_r.P_gen .- bus_h.P_gen)) < MIXED_PARITY_ATOL
            @test maximum(abs.(bus_r.Q_gen .- bus_h.Q_gen)) < MIXED_PARITY_ATOL
        end
    end
end

# ACTIVSg2000 is large enough that the converged-state round-off floor exceeds
# 1e-7; 1e-5 still pins formulation parity (see MIXED_PARITY_ATOL_2K). NR
# additionally cross-checks rectangular CI; TR is polar-only.
@testset "Mixed CPB polar parity: ACTIVSg2000 ($(solver))" for solver in
                                                               MIXED_PARITY_SOLVERS
    use_rect = solver === NewtonRaphsonACPowerFlow
    sys_p = PSB.build_system(PSB.MatpowerTestSystems, "matpower_ACTIVSg2000_sys")
    sys_h = PSB.build_system(PSB.MatpowerTestSystems, "matpower_ACTIVSg2000_sys")
    sys_r = if use_rect
        PSB.build_system(PSB.MatpowerTestSystems, "matpower_ACTIVSg2000_sys")
    else
        nothing
    end
    _mixed_polar_parity(
        sys_p, sys_h;
        sys_r = sys_r,
        pf_kwargs = (; correct_bustypes = true),
        atol = MIXED_PARITY_ATOL_2K,
        solver = solver,
    )
end
