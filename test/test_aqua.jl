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
    Aqua.test_stale_deps(PowerFlows)
end

@testset "Aqua: deps compat" begin
    Aqua.test_deps_compat(PowerFlows)
end
