@testset "Filter matrix" begin
    a = 0.0
    b = 2π
    n = 500
    Δx = (b - a) / n

    # Domains
    closed_interval = ClosedIntervalDomain(a, b)
    periodic_interval = PeriodicIntervalDomain(a, b)
    struct UnknownDomain <: DiscreteFiltering.Domain end
    unknown_domain = UnknownDomain()


    ## Top hat filter
    f = IdentityFilter()

    # ClosedIntervalDomain
    W = filter_matrix(f, closed_interval, n)
    @test W isa SparseMatrixCSC
    @test size(W) == (n + 1, n + 1)
    @test all(sum(W, dims = 2) .≈ 1)

    # PeriodicIntervalDomain
    W = filter_matrix(f, periodic_interval, n)
    @test W isa SparseMatrixCSC
    @test size(W) == (n, n)
    @test all(sum(W, dims = 2) .≈ 1)

    # Unknown domain
    @test_throws Exception filter_matrix(f, unknown_domain, n)


    ## Top hat filter
    h₀ = 0.05
    h = x -> h₀ * (1 - 1 / 2 * cos(x))
    f = TopHatFilter(h)

    # ClosedIntervalDomain
    W = filter_matrix(f, closed_interval, n)
    @test W isa SparseMatrixCSC
    @test size(W) == (n + 1, n + 1)
    @test all(sum(W, dims = 2) .≈ 1)

    # PeriodicIntervalDomain
    W = filter_matrix(f, periodic_interval, n)
    @test W isa SparseMatrixCSC
    @test size(W) == (n, n)
    @test_broken all(sum(W, dims = 2) .≈ 1)

    # Unknown domain
    @test_throws Exception filter_matrix(f, unknown_domain, n)


    ## Convolutional filter
    g = Gaussian(h₀)

    # ClosedIntervalDomain
    @test_throws Exception filter_matrix(g, closed_interval, n)
    # W = filter_matrix(g, closed_interval, n)
    # @test W isa SparseMatrixCSC
    # @test size(W) == (n, n)
    # @test all(sum(W, dims = 2) .≈ 1)

    # PeriodicIntervalDomain
    @test_throws Exception filter_matrix(g, periodic_interval, n)
    # W = filter_matrix(g, periodic_interval, n)
    # @test W isa SparseMatrixCSC
    # @test size(W) == (n + 1, n + 1)
    # @test all(sum(W, dims = 2) .≈ 1)

    # Unknown domain
    @test_throws Exception filter_matrix(g, unknown_domain, n)


    ## Constant mesh-wide top hat filter
    h = x -> Δx / 2
    f = TopHatFilter(h)

    # ClosedIntervalDomain
    W = filter_matrix_meshwidth(f, closed_interval, n)
    @test W isa SparseMatrixCSC
    @test size(W) == (n + 1, n + 1)
    @test all(sum(W, dims = 2) .≈ 1)

    # PeriodicIntervalDomain
    W = filter_matrix_meshwidth(f, periodic_interval, n)
    @test W isa SparseMatrixCSC
    @test size(W) == (n, n)
    @test all(sum(W, dims = 2) .≈ 1)

    # Unknown domain
    @test_throws Exception filter_matrix_meshwidth(f, unknown_domain, n)
end
