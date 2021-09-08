@testset "diffusion_matrix.jl" begin
    a = 0.0
    b = 2π
    N = 100

    # Test with ClosedIntervalDomain
    domain = ClosedIntervalDomain(a, b)
    D = diffusion_matrix(domain, N)
    @test D isa SparseMatrixCSC
    @test size(D) == (N + 1, N + 1)
    @test nnz(D) == 3(N + 1) - 2
    @test all(diag(D) .< 0)

    # Test with PeriodicIntervalDomain
    domain = PeriodicIntervalDomain(a, b)
    D = diffusion_matrix(domain, N)
    @test D isa SparseMatrixCSC
    @test size(D) == (N, N)
    @test nnz(D) == 3N
    @test all(diag(D, -1) .> 0)
    @test all(diag(D) .< 0)
    @test all(diag(D, +1) .> 0)
    @test D[1, end] > 0
    @test D[end, 1] > 0

    # Concrete unknown domain type
    domain = UnknownDomain()
    @test_throws Exception diffusion_matrix(domain, N)
end
