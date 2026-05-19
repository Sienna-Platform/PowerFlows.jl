# MCPB modified flat start, initial-condition guards, and multi-period warm
# start, validated on the mixed (e, f) 2-slot layout.

const MIXED_FS_PARITY_ATOL = 1e-7
_mixed_fs_settings() = Dict{Symbol, Any}(:validate_voltage_magnitudes => false)

# Perturb the stored bus voltages of `sys` far from a flat 1∠0 start. Only the
# *initial guess* changes: PQ |V|/θ and PV θ are not physical unknowns'
# set-points, so the converged solution is identical to the unperturbed system.
# This drives the base flat-start residual well past the LARGE_RESIDUAL gate so
# the enhanced flat start path is exercised, mirroring the polar/rect approach.
function _mixed_perturb!(sys::PSY.System; vm = 0.7, apq = -0.7, apv = 0.6)
    for b in PSY.get_components(PSY.ACBus, sys)
        bt = PSY.get_bustype(b)
        if bt == PSY.ACBusTypes.PQ
            PSY.set_magnitude!(b, vm)
            PSY.set_angle!(b, apq)
        elseif bt == PSY.ACBusTypes.PV
            PSY.set_angle!(b, apv)
        end
    end
    return sys
end

@testset "Mixed CPB flat start: hard fixture converges and matches polar" begin
    sys_h = _mixed_perturb!(PSB.build_system(PSB.PSITestSystems, "c_sys5"))
    # Deterministic reference: polar on the *unperturbed* system (same physical
    # solution; only the hard fixture's initial guess differs).
    sys_ref = PSB.build_system(PSB.PSITestSystems, "c_sys5")

    pf_h = ACMixedPowerFlow{NewtonRaphsonACPowerFlow}(;
        enhanced_flat_start = true,
        solver_settings = _mixed_fs_settings(),
    )
    pf_ref = ACPowerFlow{NewtonRaphsonACPowerFlow}()

    res_h = solve_power_flow(pf_h, sys_h)
    res_ref = solve_power_flow(pf_ref, sys_ref)
    @test res_h !== missing
    @test res_ref !== missing

    bus_h = res_h["bus_results"]
    bus_ref = res_ref["bus_results"]
    @test all(isfinite, bus_h.Vm)
    @test all(isfinite, bus_h.θ)
    @test maximum(abs.(bus_ref.Vm .- bus_h.Vm)) < MIXED_FS_PARITY_ATOL
    @test maximum(abs.(bus_ref.θ .- bus_h.θ)) < MIXED_FS_PARITY_ATOL
    @test maximum(abs.(bus_ref.P_gen .- bus_h.P_gen)) < MIXED_FS_PARITY_ATOL
    @test maximum(abs.(bus_ref.Q_gen .- bus_h.Q_gen)) < MIXED_FS_PARITY_ATOL
end

@testset "Mixed CPB flat start: modified-flat-start construction" begin
    sys = _mixed_perturb!(PSB.build_system(PSB.PSITestSystems, "c_sys5"))
    pf = ACMixedPowerFlow{NewtonRaphsonACPowerFlow}(;
        enhanced_flat_start = true, solver_settings = _mixed_fs_settings())
    data = PowerFlowData(pf, sys)
    residual = PF.ACMixedCPBResidual(data, 1)

    x0 = Vector{Float64}(undef, length(residual.Rv))
    PF.mixed_initial_state!(
        x0, data, residual.bus_state_offset, residual.bus_block_size, 1,
    )
    residual(x0, 1)
    # The hard fixture must actually trip the LARGE_RESIDUAL gate so the
    # enhanced flat start path is the one under test.
    @test norm(residual.Rv, 1) > PF.LARGE_RESIDUAL * length(residual.Rv)

    newx0 = PF._enhanced_flat_start(x0, data, residual, 1)

    bt = data.bus_type[:, 1]
    # c_sys5 is a single subnetwork: one REF angle for all PV/PQ.
    ref = findall(==(PSY.ACBusTypes.REF), bt)
    pv = findall(==(PSY.ACBusTypes.PV), bt)
    pq = findall(==(PSY.ACBusTypes.PQ), bt)
    @test !isempty(ref) && !isempty(pv) && !isempty(pq)
    ref_angle = sum(data.bus_angles[r, 1] for r in ref) / length(ref)
    target_vm = sum(data.bus_magnitude[p, 1] for p in pv) / length(pv)

    # Mirror the rect rule exactly: when the subnetwork REF angle is identically
    # 0.0 the bus keeps its own angle (rect/polar heuristic — a flat REF angle
    # means the modified-flat-start angle is already 0-equivalent); otherwise
    # every PV/PQ angle is forced to the REF angle (the paper's statement).
    expected_θ(i) = ref_angle != 0.0 ? ref_angle : data.bus_angles[i, 1]
    for i in pv
        off = Int(residual.bus_state_offset[i])
        @test sqrt(newx0[off]^2 + newx0[off + 1]^2) ≈
              data.bus_magnitude[i, 1] atol = 1e-10
        @test atan(newx0[off + 1], newx0[off]) ≈ expected_θ(i) atol = 1e-10
    end
    for i in pq
        off = Int(residual.bus_state_offset[i])
        @test sqrt(newx0[off]^2 + newx0[off + 1]^2) ≈ target_vm atol = 1e-10
        @test atan(newx0[off + 1], newx0[off]) ≈ expected_θ(i) atol = 1e-10
    end
    # REF blocks (P_gen0, Q_gen0) untouched.
    for i in ref
        off = Int(residual.bus_state_offset[i])
        @test newx0[off] == x0[off]
        @test newx0[off + 1] == x0[off + 1]
    end

    # Paper's literal branch: with a non-zero subnetwork REF angle, EVERY PV/PQ
    # angle is forced to that REF angle (no per-bus fallback).
    θref = 0.15
    for r in ref
        data.bus_angles[r, 1] = θref
    end
    nz = PF._enhanced_flat_start(x0, data, residual, 1)
    for i in vcat(pv, pq)
        off = Int(residual.bus_state_offset[i])
        @test atan(nz[off + 1], nz[off]) ≈ θref atol = 1e-10
    end
end

@testset "Mixed CPB flat start: multi-period warm start" begin
    sys = PSB.build_system(PSB.PSITestSystems, "c_sys14"; add_forecasts = false)
    pf = ACMixedPowerFlow{NewtonRaphsonACPowerFlow}(;
        time_steps = 2, solver_settings = _mixed_fs_settings())
    data = PowerFlowData(pf, sys)

    # Converge step 1 in a single-step copy and inject its solution into the
    # 2-step data's column 1, marking it converged (mirrors the rect
    # warm-start test's "solve step 1, mark converged" setup). Step 2 has the
    # same loads, so step-1's converged state IS step-2's solution.
    d1 = PowerFlowData(
        ACMixedPowerFlow{NewtonRaphsonACPowerFlow}(;
            solver_settings = _mixed_fs_settings()), sys)
    @test PowerFlows.solve_power_flow!(d1)
    data.bus_magnitude[:, 1] .= d1.bus_magnitude[:, 1]
    data.bus_angles[:, 1] .= d1.bus_angles[:, 1]
    data.bus_active_power_injections[:, 1] .= d1.bus_active_power_injections[:, 1]
    data.bus_reactive_power_injections[:, 1] .=
        d1.bus_reactive_power_injections[:, 1]
    data.converged[1] = true

    # Make step-2's stored cold start poor so the warm start has something to
    # beat.
    for i in 1:size(data.bus_magnitude, 1)
        bt = data.bus_type[i, 2]
        if bt == PSY.ACBusTypes.PQ
            data.bus_magnitude[i, 2] = 0.6
            data.bus_angles[i, 2] = -0.8
        elseif bt == PSY.ACBusTypes.PV
            data.bus_angles[i, 2] = 0.7
        end
    end

    residual = PF.ACMixedCPBResidual(data, 2)

    # Cold flat start for step 2 (reads the perturbed step-2 data).
    cold = Vector{Float64}(undef, length(residual.Rv))
    PF.mixed_initial_state!(
        cold, data, residual.bus_state_offset, residual.bus_block_size, 2,
    )
    residual(cold, 2)
    cold_norm = norm(residual.Rv, 1)

    # Warm start: step-1 converged mixed state via the type/value split
    # (_mixed_fill_state! with type_ts=2, value_ts=1).
    warm = copy(cold)
    PF._mixed_fill_state!(warm, data, residual.bus_state_offset, 2, 1)
    residual(warm, 2)
    warm_norm = norm(residual.Rv, 1)

    @test warm_norm < 0.1 * cold_norm
end

@testset "Mixed CPB flat start: degenerate (e,f) guard holds" begin
    sys = PSB.build_system(PSB.PSITestSystems, "c_sys14"; add_forecasts = false)
    pf = ACMixedPowerFlow{NewtonRaphsonACPowerFlow}()
    data = PowerFlowData(pf, sys)
    residual = PF.ACMixedCPBResidual(data, 1)

    x0 = Vector{Float64}(undef, length(residual.Rv))
    PF.mixed_initial_state!(
        x0, data, residual.bus_state_offset, residual.bus_block_size, 1,
    )
    # Drive a non-REF bus to a degenerate (e,f). Two sub-cases: a near-zero
    # (e²+f² ≈ 1e-20) and an exactly-zero voltage — the latter makes the
    # unguarded 1/|V|² / 1/|V| sites produce Inf/NaN without the V_FLOOR2 floor.
    bt = data.bus_type[:, 1]
    i = findfirst(!=(PSY.ACBusTypes.REF), bt)
    @test i !== nothing
    off = Int(residual.bus_state_offset[i])
    x0[off] = 0.0
    x0[off + 1] = 0.0

    residual(x0, 1)
    @test all(isfinite, residual.Rv)

    J = PF.ACMixedCPBJacobian(residual, 1)
    J(1)
    @test all(isfinite, SparseArrays.nonzeros(J.Jv))
end
