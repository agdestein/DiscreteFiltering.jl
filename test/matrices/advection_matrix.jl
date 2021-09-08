@testset "advection_matrix.jl" begin
    a = 0.0
    b = 2π
    N = 100

    # Test with ClosedIntervalDomain
    domain = ClosedIntervalDomain(a, b)
    C = advection_matrix(domain, N)
    @test C isa SparseMatrixCSC
    @test size(C) == (N + 1, N + 1)
    @test nnz(C) == 2(N + 1)
    d = diag(C)
    @test all(d[2:end-1] .== 0)
    @test d[1] < 0
    @test d[end] > 0

    # Test with PeriodicIntervalDomain
    domain = PeriodicIntervalDomain(a, b)
    C = advection_matrix(domain, N)
    @test C isa SparseMatrixCSC
    @test size(C) == (N, N)
    @test nnz(C) == 2N
    @test all(diag(C) .== 0)
    @test all(diag(C, -1) .< 0)
    @test all(diag(C, +1) .> 0)
    @test C[1, end] < 0
    @test C[end, 1] > 0

    # Concrete unknown domain type
    domain = UnknownDomain()
    @test_throws Exception advection_matrix(domain, N)
end
