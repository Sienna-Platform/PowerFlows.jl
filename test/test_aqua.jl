@testset "Aqua: unbound type parameters" begin
    Aqua.test_unbound_args(PowerFlows)
end

@testset "Aqua: undefined exports" begin
    Aqua.test_undefined_exports(PowerFlows)
end

@testset "Aqua: method ambiguities" begin
    Aqua.test_ambiguities(PowerFlows)
end

@testset "Aqua: stale dependencies" begin
    # PowerMatricesCore is unregistered, so it is a direct dependency only to give its
    # `[sources]` entry effect for PowerNetworkMatrices.
    Aqua.test_stale_deps(PowerFlows; ignore = [:PowerMatricesCore])
end

@testset "Aqua: deps compat" begin
    Aqua.test_deps_compat(PowerFlows)
end
