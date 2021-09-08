@testset "reconstruction_matrix.jl" begin
    a = 0.0
    b = 2π
    N = 100
    M = N
    Δx = (b - a) / N

    # Domains
    closed_interval = ClosedIntervalDomain(a, b)
    periodic_interval = PeriodicIntervalDomain(a, b)
    unknown_domain = UnknownDomain()


    ## Identity filter
    f = IdentityFilter()

    # ClosedIntervalDomain
    R = reconstruction_matrix(f, closed_interval, M, N)
    @test R isa SparseMatrixCSC
    @test size(R) == (N + 1, M + 1)
    @test all(sum(R, dims = 2) .≈ 1)

    # PeriodicIntervalDomain
    R = reconstruction_matrix(f, periodic_interval, M, N)
    @test R isa SparseMatrixCSC
    @test size(R) == (N, M)
    @test all(sum(R, dims = 2) .≈ 1)

    # Unknown domain
    @test_throws Exception reconstruction_matrix(f, unknown_domain, M, N)


    ## Top hat filter
    h₀ = 0.05
    h = x -> h₀ * (1 - 1 / 2 * cos(x))
    f = TopHatFilter(h)

    # ClosedIntervalDomain
    R = reconstruction_matrix(f, closed_interval, M, N)
    @test R isa SparseMatrixCSC
    @test size(R) == (N + 1, M + 1)
    @test all(sum(R, dims = 2) .≈ 1)

    # PeriodicIntervalDomain
    R = reconstruction_matrix(f, periodic_interval, M, N)
    @test R isa SparseMatrixCSC
    @test size(R) == (N, M)
    @test all(sum(R, dims = 2) .≈ 1)

    # Unknown domain
    @test_throws Exception reconstruction_matrix(f, unknown_domain, M, N)


    ## Convolutional filter
    σ = Δx
    g = GaussianFilter(σ)

    # ClosedIntervalDomain
    R = reconstruction_matrix(g, closed_interval, M, N)
    @test R isa SparseMatrixCSC
    @test size(R) == (N + 1, M + 1)
    @test all(sum(R, dims = 2) .≈ 1)

    # PeriodicIntervalDomain
    R = reconstruction_matrix(g, periodic_interval, M, N)
    @test R isa SparseMatrixCSC
    @test size(R) == (N, M)
    @test all(sum(R, dims = 2) .≈ 1)

    # Unknown domain
    @test_throws Exception reconstruction_matrix(f, unknown_domain, M, N)


    ## Constant mesh-wide top hat filter
    h = x -> Δx / 2
    f = TopHatFilter(h)

    # ClosedIntervalDomain
    R = reconstruction_matrix_meshwidth(f, closed_interval, N)
    @test R isa SparseMatrixCSC
    @test size(R) == (N + 1, N + 1)
    @test all(sum(R, dims = 2) .≈ 1)

    # PeriodicIntervalDomain
    R = reconstruction_matrix_meshwidth(f, periodic_interval, N)
    @test R isa SparseMatrixCSC
    @test size(R) == (N, N)
    @test all(sum(R, dims = 2) .≈ 1)

    # Unknown domain
    @test_throws Exception reconstruction_matrix_meshwidth(f, unknown_domain, N)
end
