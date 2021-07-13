@testset "Domain" begin
    # Closed interval domain
    a, b = 0.0, 1.0
    domain = ClosedIntervalDomain(a, b)
    @test domain.left == a
    @test domain.right == b

    n = 10
    x = discretize(domain, n)
    @test length(x) == n + 1
    @test minimum(x) == a
    @test maximum(x) == b

    # Periodic interval domain
    a, b = -1.0, 1π
    domain = PeriodicIntervalDomain(a, b)
    @test domain.left == a
    @test domain.right == b

    n = 15
    x = discretize(domain, n)
    @test length(x) == n
    @test minimum(x) == a + (b - a) / n
    @test maximum(x) == b

    # Concrete unknown domain type
    struct UnknownDomain <: DiscreteFiltering.Domain end
    domain = UnknownDomain()
    @test_throws Exception discretize(domain, 10)
end
