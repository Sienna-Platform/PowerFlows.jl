# A 0-MW-scheduled LCC has i_dc = 0, which makes P_lcc ≡ 0: its P-setpoint and
# DC-line-balance equations become vacuous and its tap states unconstrained, so
# the Jacobian goes rank-deficient (tap_r/tap_i columns and those two rows zero
# out). The fix pins the taps to their scheduled setting (tap − tap_setpoint
# identity rows) when i_dc = 0, keeping the block nonsingular in every
# formulation while staying consistent with the residual.

const _ZERO_SP_FORMULATIONS = (
    ("polar", ACPowerFlow{NewtonRaphsonACPowerFlow}),
    ("rect-CI", PF.ACRectangularPowerFlow{NewtonRaphsonACPowerFlow}),
    ("mixed-CPB", PF.ACMixedPowerFlow{NewtonRaphsonACPowerFlow}),
)

# Build case5_2_lcc with the first LCC forced to a 0-MW transfer setpoint.
function _zero_setpoint_lcc_system()
    raw_path = joinpath(TEST_DATA_DIR, "case5_2_lcc.raw")
    sys = make_system(PFP.PowerModelsData(raw_path); runchecks = false)
    set_transfer_setpoint!(first(get_components(TwoTerminalLCCLine, sys)), 0.0)
    return sys
end

@testset "0-MW LCC: Jacobian stays nonsingular and solves ($name)" for (name, PFType) in
                                                                       _ZERO_SP_FORMULATIONS
    data = PowerFlowData(PFType(), _zero_setpoint_lcc_system())
    residual = PF.ACPowerFlowResidual(data, 1)
    jac = PF.ACPowerFlowJacobian(residual, 1)
    x0 = PF.calculate_x0(data, 1)
    residual(x0, 1)
    jac(1)

    # i_dc of the degenerate converter is exactly 0.
    @test iszero(data.lcc.i_dc[1, 1])
    # Without the tap-pin this σ_min collapses to ~0 (κ̂ → ~1e27); with it the
    # Jacobian is as well-conditioned as the rest of the network.
    @test minimum(svdvals(Matrix(jac.Jv))) > 1e-3
    # And the solve converges.
    @test solve_power_flow!(data)
end

@testset "0-MW LCC: analytic Jacobian matches finite differences ($name)" for (
    name,
    PFType,
) in _ZERO_SP_FORMULATIONS
    data = PowerFlowData(PFType(), _zero_setpoint_lcc_system())
    residual = PF.ACPowerFlowResidual(data, 1)
    jac = PF.ACPowerFlowJacobian(residual, 1)
    x0 = PF.calculate_x0(data, 1)
    Random.seed!(1)
    x = x0 .+ 0.01 .* randn(length(x0))
    residual(x, 1)
    jac(1)
    J = copy(Matrix(jac.Jv))

    v = randn(length(x))
    ε = 1e-6
    residual(x .+ ε .* v, 1)
    Fp = copy(residual.Rv)
    residual(x .- ε .* v, 1)
    Fm = copy(residual.Rv)
    fd = (Fp .- Fm) ./ (2ε)
    # The pinned tap rows (∂/∂tap = 1) must agree with FD just like every other row.
    @test norm(J * v .- fd, Inf) / norm(fd, Inf) < 1e-6
end

@testset "0-MW LCC: dimensions consistent across time steps" begin
    # The pin must not change the matrix size/structure between periods.
    sys = _zero_setpoint_lcc_system()
    data = PowerFlowData(ACPowerFlow{NewtonRaphsonACPowerFlow}(; time_steps = 3), sys)
    residual = PF.ACPowerFlowResidual(data, 1)
    jac = PF.ACPowerFlowJacobian(residual, 1)
    residual(PF.calculate_x0(data, 1), 1)
    jac(1)
    sz = size(jac.Jv)
    nnz1 = SparseArrays.nnz(jac.Jv)
    for t in 2:3
        residual(PF.calculate_x0(data, t), t)
        jac(t)
        @test size(jac.Jv) == sz
        @test SparseArrays.nnz(jac.Jv) == nnz1
    end
end
