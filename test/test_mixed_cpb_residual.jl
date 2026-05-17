@testset "Mixed CPB Residual: zero at polar solution" begin
    # Mirrors `test_rectangular_ci_residual.jl`'s "residual zero at polar-
    # converged state": converge with polar NR (writes Vm/θ and injections
    # back into sys), build a FRESH mixed PowerFlowData, map the converged
    # polar solution into the mixed x via `mixed_initial_state!`, and assert
    # the MCPB residual is ~0 there.
    @testset "c_sys5" begin
        sys = PSB.build_system(PSB.PSITestSystems, "c_sys5")
        pf_polar = ACPowerFlow{NewtonRaphsonACPowerFlow}()
        @test PF.solve_and_store_power_flow!(pf_polar, sys)
        pf_mixed = ACMixedPowerFlow{NewtonRaphsonACPowerFlow}()
        data = PF.PowerFlowData(pf_mixed, sys)
        R = PF.ACMixedCPBResidual(data, 1)
        x = Vector{Float64}(undef, length(R.Rv))
        PF.mixed_initial_state!(x, data, R.bus_state_offset, R.bus_block_size, 1)
        Rv = similar(x)
        R(Rv, x, 1)
        @test LinearAlgebra.norm(Rv, Inf) < 1e-6
    end

    @testset "c_sys14" begin
        sys = PSB.build_system(PSB.PSITestSystems, "c_sys14"; add_forecasts = false)
        pf_polar = ACPowerFlow{NewtonRaphsonACPowerFlow}()
        @test PF.solve_and_store_power_flow!(pf_polar, sys)
        pf_mixed = ACMixedPowerFlow{NewtonRaphsonACPowerFlow}()
        data = PF.PowerFlowData(pf_mixed, sys)
        R = PF.ACMixedCPBResidual(data, 1)
        x = Vector{Float64}(undef, length(R.Rv))
        PF.mixed_initial_state!(x, data, R.bus_state_offset, R.bus_block_size, 1)
        Rv = similar(x)
        R(Rv, x, 1)
        @test LinearAlgebra.norm(Rv, Inf) < 1e-6
    end

    @testset "ZIP load (P+I+Z combination)" begin
        # `_build_zip_2bus_system` is defined in test_rectangular_ci_polar_parity.jl
        # (auto-included in the same test session). Constant-current AND
        # constant-impedance components exercise the Y_bus_eff fold + const-I
        # correction paths of the MCPB residual.
        sys = _build_zip_2bus_system(;
            power_pq = (0.5, 0.2),
            current_pq = (2.0, 1.0),
            impedance_pq = (1.5, 0.8),
        )
        pf_polar = ACPowerFlow{NewtonRaphsonACPowerFlow}(; correct_bustypes = true)
        @test PF.solve_and_store_power_flow!(pf_polar, sys)
        pf_mixed = ACMixedPowerFlow{NewtonRaphsonACPowerFlow}(;
            correct_bustypes = true)
        data = PF.PowerFlowData(pf_mixed, sys)
        R = PF.ACMixedCPBResidual(data, 1)
        x = Vector{Float64}(undef, length(R.Rv))
        PF.mixed_initial_state!(x, data, R.bus_state_offset, R.bus_block_size, 1)
        Rv = similar(x)
        R(Rv, x, 1)
        @test LinearAlgebra.norm(Rv, Inf) < 1e-6
    end
end

@testset "Mixed CPB squared voltage-magnitude validation" begin
    range = (min = 0.5, max = 1.5)
    sys = PSB.build_system(PSB.PSITestSystems, "c_sys5")
    pf_mixed = ACMixedPowerFlow{NewtonRaphsonACPowerFlow}()
    data = PF.PowerFlowData(pf_mixed, sys)
    R = PF.ACMixedCPBResidual(data, 1)
    x = Vector{Float64}(undef, length(R.Rv))
    PF.mixed_initial_state!(x, data, R.bus_state_offset, R.bus_block_size, 1)
    bus_types = PF.get_bus_type(data)

    # Initial mixed state is ~1 p.u. ⇒ in range ⇒ silent.
    @test_logs min_level = Logging.Warn PF._validate_state_magnitudes(
        R, x, bus_types, range, 1)

    # PQ buses in mixed use (e,f) slots like rectangular ⇒ out-of-range warns.
    b = findfirst(bt -> bt == PSY.ACBusTypes.PQ, bus_types)
    off = Int(R.bus_state_offset[b])
    x[off] = 3.0
    x[off + 1] = 0.0
    @test_logs (:warn, r"voltage magnitudes outside of range") match_mode = :any PF._validate_state_magnitudes(
        R, x, bus_types, range, 1)
end
