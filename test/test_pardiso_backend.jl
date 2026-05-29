# MKLPardiso backend tests.
#
# Loading Pardiso triggers PowerFlows' PowerFlowsPardisoExt (and PNM's MKLPardisoExt).
# Pardiso.jl installs on every platform, but MKL is only functional on x86_64
# Linux/Windows; the numeric tests are gated on `Pardiso.mkl_is_available()` so they
# RUN on the Linux/Windows CI runners and SKIP cleanly on Apple Silicon. The
# availability-error path is asserted on every platform.
import Pardiso

@testset "MKLPardiso backend" begin
    if Pardiso.mkl_is_available()
        @testset "DC parity: KLU vs MKLPardiso" begin
            sys = build_system(PSITestSystems, "c_sys5")
            res_klu = solve_power_flow(
                DCPowerFlow(), sys, FlowReporting.ARC_FLOWS; linear_solver = "KLU")
            res_par = solve_power_flow(
                DCPowerFlow(), sys, FlowReporting.ARC_FLOWS;
                linear_solver = "MKLPardiso")
            @test isapprox(
                res_klu["1"]["bus_results"].θ,
                res_par["1"]["bus_results"].θ;
                atol = 1e-8,
            )
        end

        @testset "AC parity: KLU vs MKLPardiso (NR and TR)" begin
            sys = build_system(PSITestSystems, "c_sys14")
            res_klu = solve_power_flow(
                ACPowerFlow{NewtonRaphsonACPowerFlow}(;
                    solver_settings = Dict{Symbol, Any}(:linear_solver => "KLU")),
                sys,
            )
            for solver in (NewtonRaphsonACPowerFlow, TrustRegionACPowerFlow)
                res_par = solve_power_flow(
                    ACPowerFlow{solver}(;
                        solver_settings = Dict{Symbol, Any}(
                            :linear_solver => "MKLPardiso")),
                    sys,
                )
                @test isapprox(
                    res_klu["bus_results"][!, :Vm],
                    res_par["bus_results"][!, :Vm];
                    atol = 1e-7,
                )
                @test isapprox(
                    res_klu["bus_results"][!, :θ],
                    res_par["bus_results"][!, :θ];
                    atol = 1e-7,
                )
            end
        end

        @testset "MKLPardiso in-place solve correctness" begin
            sys = build_system(PSITestSystems, "c_sys14")
            pf = ACPowerFlow{NewtonRaphsonACPowerFlow}()
            data = PF.PowerFlowData(pf, sys)
            residual = PF.ACPowerFlowResidual(data, 1)
            J = PF.ACPowerFlowJacobian(residual, 1)
            J(1)
            cache = PF.make_linear_solver_cache(PF.PNM.MKLPardisoSolver(), J.Jv)
            PF.full_factor!(cache, J.Jv)
            b = randn(size(J.Jv, 1))
            x = copy(b)
            PF.solve!(cache, x)              # in place: x overwritten with J.Jv \ b
            @test isapprox(J.Jv * x, b; atol = 1e-7)

            # numeric_refactor! reuses the symbolic analysis on refreshed values.
            PF.symbolic_factor!(cache, J.Jv)
            PF.numeric_refactor!(cache, J.Jv)
            x2 = copy(b)
            PF.solve!(cache, x2)
            @test isapprox(J.Jv * x2, b; atol = 1e-7)
        end
    else
        @info "MKL Pardiso not functional on this platform; asserting graceful errors only."
        sys = build_system(PSITestSystems, "c_sys14")
        pf = ACPowerFlow{NewtonRaphsonACPowerFlow}()
        data = PF.PowerFlowData(pf, sys)
        residual = PF.ACPowerFlowResidual(data, 1)
        J = PF.ACPowerFlowJacobian(residual, 1)
        J(1)
        # Constructing the cache must fail with a clear error (never a crash/segfault)
        # when MKL is unusable — the functional guard runs before any MKL ccall.
        @test_throws ErrorException PF.make_linear_solver_cache(
            PF.PNM.MKLPardisoSolver(), J.Jv)
        if Sys.ARCH !== :x86_64
            # On Apple Silicon / other non-x86_64, the resolver rejects MKLPardiso
            # outright with a definitive message (does not suggest `import Pardiso`).
            @test_throws ErrorException PF.resolve_linear_solver_backend("MKLPardiso")
        end
    end
end
