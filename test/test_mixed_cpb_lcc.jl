function _mixed_lcc_settings()
    return Dict{Symbol, Any}(:validate_voltage_magnitudes => false)
end

# Map the polar-converged solution stored in `sys` into the MCPB state vector
# and assert the residual (bus current rows + 4 LCC tail rows) is ~0. Mirrors
# `test_mixed_cpb_residual.jl`'s "zero at polar solution" pattern, but on an
# LCC fixture so the LCC current accumulation (step 4) and tail residuals
# (step 6) are exercised. Returns norm(Rv, Inf) for reporting.
function _mixed_lcc_residual_norm(
    sys::System;
    correct_bustypes::Bool = false,
)
    pf_polar = ACPowerFlow{NewtonRaphsonACPowerFlow}(;
        correct_bustypes = correct_bustypes,
        solver_settings = _mixed_lcc_settings(),
    )
    @test PF.solve_and_store_power_flow!(pf_polar, sys)
    pf_mixed = ACMixedPowerFlow{NewtonRaphsonACPowerFlow}(;
        correct_bustypes = correct_bustypes,
        solver_settings = _mixed_lcc_settings(),
    )
    data = PF.PowerFlowData(pf_mixed, sys)
    R = PF.ACMixedCPBResidual(data, 1)
    x = Vector{Float64}(undef, length(R.Rv))
    PF.mixed_initial_state!(x, data, R.bus_state_offset, R.bus_block_size, 1)
    Rv = similar(x)
    R(Rv, x, 1)
    return LinearAlgebra.norm(Rv, Inf)
end

@testset "Mixed CPB LCC: residual zero at polar solution" begin
    # Same LCC fixture `test_rectangular_ci_lcc.jl` uses (case5_2_lcc.raw):
    # both LCC terminal buses are PQ. Confirms the bus current rows AND the
    # 4 LCC tail residual rows are satisfied at the polar-converged state.
    @testset "case5_2_lcc (PQ terminals)" begin
        raw_path = joinpath(TEST_DATA_DIR, "case5_2_lcc.raw")
        sys = make_system(PFP.PowerModelsData(raw_path); runchecks = false)
        nrm = _mixed_lcc_residual_norm(sys)
        @test nrm < 1e-6
    end

    # `simple_lcc_system()` (defined in test_utils/common.jl): 3-bus REF/PQ/PQ
    # with the LCC across the two PQ terminal buses. Smaller, fully synthetic.
    @testset "simple_lcc_system (PQ terminals)" begin
        sys, _ = simple_lcc_system()
        nrm = _mixed_lcc_residual_norm(sys)
        @test nrm < 1e-6
    end
end

@testset "Mixed CPB LCC: residual zero with PV LCC terminal" begin
    # `test_rectangular_ci_lcc.jl` has no PV-terminal LCC variant, so we build
    # one from `simple_lcc_system()`: bus_2 is one of the two LCC converter
    # terminal buses (the LCC arc is b2->b3). We promote bus_2 to PV by adding
    # a ThermalStandard generator there and setting the bustype to PV. This
    # validates that the PV power-balance row `e·Ir_acc + f·Ii_acc − P`
    # correctly absorbs the LCC current injected at a PV terminal: step 4 of
    # the residual accumulates LCC current into Ir_acc/Ii_acc for ALL terminal
    # bus types, and the PV branch in step 5 reads those accumulators.
    sys, _ = simple_lcc_system()
    b2 = PSY.get_component(ACBus, sys, "bus_2")
    PSY.set_bustype!(b2, ACBusTypes.PV)
    PSY.set_magnitude!(b2, 1.05)
    _add_simple_thermal_standard!(sys, b2, 0.3, 0.0)
    nrm = _mixed_lcc_residual_norm(sys)
    @test nrm < 1e-6
end
