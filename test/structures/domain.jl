@testset "domain.jl" begin
    # Closed interval domain
    a, b = 0.0, 1.0
    domain = ClosedIntervalDomain(a, b)
    @test domain.left == a
    @test domain.right == b

    N = 10
    x = discretize(domain, N)
    @test length(x) == N + 1
    @test minimum(x) == a
    @test maximum(x) == b

    # Periodic interval domain
    a, b = -1.0, 1π
    domain = PeriodicIntervalDomain(a, b)
    @test domain.left == a
    @test domain.right == b

    N = 15
    x = discretize(domain, N)
    @test length(x) == N
    @test minimum(x) == a + (b - a) / N
    @test maximum(x) == b

    # Concrete unknown domain type
    domain = UnknownDomain()
    @test_throws Exception discretize(domain, 10)
end
