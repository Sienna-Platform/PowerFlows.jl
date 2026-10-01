# `linear_solver = "LeanKLU"` must reproduce the KLU backend's solutions on every path.

_with_solver(pf_type, solver; kwargs...) = pf_type(;
    solution_parameters = SolutionParameters(; linear_solver = solver), kwargs...)

function _solve_and_state(pf, sys)
    data = PowerFlowData(pf, sys)
    @test solve_power_flow!(data)
    return data, data.bus_magnitude[:, 1], data.bus_angles[:, 1]
end

@testset "LeanKLU backend resolves" begin
    @test PF.resolve_linear_solver_backend("LeanKLU") === PF.PNM.LeanKLUSolver()
end

@testset "LeanKLU matches KLU: polar NR and TR" begin
    systems = (
        PSB.build_system(PSB.PSITestSystems, "c_sys14"; add_forecasts = false),
        PSB.build_system(PSB.PSISystems, "RTS_GMLC_DA_sys"),
    )
    kw = (; correct_bustypes = true, check_reactive_power_limits = true)
    for ACSolver in (NewtonRaphsonACPowerFlow, TrustRegionACPowerFlow), sys in systems
        pf_type = ACPowerFlow{ACSolver}
        _, vm_klu, va_klu = _solve_and_state(_with_solver(pf_type, "KLU"; kw...), sys)
        data, vm, va = _solve_and_state(_with_solver(pf_type, "LeanKLU"; kw...), sys)
        @test vm ≈ vm_klu rtol = 1e-8
        @test va ≈ va_klu rtol = 1e-8 atol = 1e-10

        lin = data.polar_nr_cache[].linSolveCache
        @test lin isa PF.PNM.LeanLUCache
        @test lin.plan !== nothing   # a plan was built, so lean refactors ran
    end
end

@testset "LeanKLU matches KLU: fast decoupled and DC" begin
    sys = PSB.build_system(PSB.PSITestSystems, "c_sys14"; add_forecasts = false)
    for V in (PF.FDDecoupled, PF.FDFixedJacobian)
        pf_type = ACPowerFlow{PF.FastDecoupledACPowerFlow{V, PF.FDSchemeXB}}
        _, vm_klu, va_klu =
            _solve_and_state(_with_solver(pf_type, "KLU"; correct_bustypes = true), sys)
        _, vm, va =
            _solve_and_state(_with_solver(pf_type, "LeanKLU"; correct_bustypes = true), sys)
        @test vm ≈ vm_klu rtol = 1e-8
        @test va ≈ va_klu rtol = 1e-8 atol = 1e-10
    end
    for pf in (DCPowerFlow(), PTDFDCPowerFlow(), vPTDFDCPowerFlow())
        data_klu = PowerFlowData(pf, sys)
        solve_power_flow!(data_klu; linear_solver = "KLU")
        data = PowerFlowData(pf, sys)
        solve_power_flow!(data; linear_solver = "LeanKLU")
        @test data.bus_angles ≈ data_klu.bus_angles rtol = 1e-10 atol = 1e-12
    end
end
