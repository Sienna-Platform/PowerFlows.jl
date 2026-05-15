const RTOL = 10^-7
function testSolve(
    k::PF.KLULinSolveCache{T},
    A::SparseMatrixCSC{Float64, T},
) where {T <: Union{Int32, Int64}}
    xb = randn(size(A, 1))
    b = deepcopy(xb)
    PF.solve!(k, xb)
    isapprox(A * xb, b; rtol = RTOL)
    XB = randn(size(A, 1), 10)
    B = deepcopy(XB)
    PF.solve!(k, XB)
    isapprox(A * XB, B; rtol = RTOL)
end

function factorization_tests(dType::Type{T}) where {T <: Union{Int32, Int64}}
    N = 50
    A = SparseMatrixCSC{Float64, dType}(sprandn(Float64, N, N, 0.1))
    while abs(det(A)) < eps(Float64)
        A = SparseMatrixCSC{Float64, dType}(sprandn(Float64, N, N, 0.1))
    end
    @assert abs(det(A)) > eps(Float64)

    k = PF.KLULinSolveCache(A)
    PF.full_factor!(k, A)

    testSolve(k, A) # solve is solution.

    # solve! is non-allocating.
    b = randn(N)
    temp = @allocated PF.solve!(k, b)
    @test iszero(temp)

    # numeric_refactor! is non allocating, and solves correctly.
    A2 = deepcopy(A)
    copyto!(A2.nzval, rand(length(A2.nzval)))
    temp = @allocated PF.numeric_refactor!(k, A2)
    @test iszero(temp)
    testSolve(k, A2)

    # with reuse_symbolic and check_pattern both true,
    # refactor with a different structure throws.
    B = deepcopy(A)
    n, m = rand(1:N), rand(1:N)
    while (B[n, m] != 0.0)
        n, m = rand(1:N), rand(1:N)
    end
    B[n, m] = 0.1
    @test_throws ArgumentError PF.numeric_refactor!(k, B)
    @test_throws ArgumentError PF.symbolic_refactor!(k, B)

    # symbolic (re)factor and solve. this does allocate.
    PF.symbolic_factor!(k, B)
    PF.numeric_refactor!(k, B)
    testSolve(k, B)

    # automatically refactor if reuse_symbolic is false.
    autoRefactor = PF.KLULinSolveCache(A, false, false)
    @test PF.symbolic_factor!(autoRefactor, B) isa Any # shouldn't throw.

    # test iterative refinement. bigA here is purposefully almost singular.
    # keep the diagonal perturbation positive and bounded away from zero so this
    # near-singular construction remains stable enough for deterministic testing.
    N, d = 1000, 0.01
    epsilon = 0.004
    v1, v2 = sprandn(N, d), sprandn(N, d)
    bigA =
        sparse(v2 * transpose(v1)) + sparse(diagm(rand(N) .+ 0.5) * epsilon) +
        epsilon * sprand(N, N, 10 * d^2)
    bigK = PF.KLULinSolveCache(bigA)
    PF.full_factor!(bigK, bigA)
    b = rand(N)
    x = PF.solve_w_refinement(bigK, bigA, b)
    @test isapprox(bigA * x, b, rtol = RTOL)

    # error handling: singular.
    sing = SparseMatrixCSC{Float64, dType}(sparse([1], [1], [0.1], 2, 2))
    # If I add KLU.kluerror(cache.K.common) to the constructor and symbolic_factor!
    # in cache, it still doesn't throw until you call numeric_refactor!. Strange.
    singCache = PF.KLULinSolveCache(sing)
    PF.symbolic_factor!(singCache, sing)
    @test_throws SingularException PF.numeric_refactor!(singCache, sing)

    # error handling: non-square.
    nonSquare = SparseMatrixCSC{Float64, dType}(sprand(10, 11, 0.5))
    @test_throws ArgumentError PF.KLULinSolveCache(nonSquare)

    # error handling: mismtached dimensions.
    b = rand(N - 1)
    @test_throws DimensionMismatch PF.solve!(k, b)
end

@testset "KLU Linear Solver Cache: Int32" begin
    factorization_tests(Int32)
end

@testset "KLU Linear Solver Cache: Int64" begin
    factorization_tests(Int64)
end
