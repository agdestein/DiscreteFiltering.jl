@testset "equations.jl" begin
    # Domains
    a = 0.0
    b = 1.0
    closed = ClosedIntervalDomain(a, b)
    periodic = PeriodicIntervalDomain(a, b)

    # Filters
    identity_filter = IdentityFilter()
    top_hat_filter = TopHatFilter(x -> 0.1)
    convolutional_filter = ConvolutionalFilter(gaussian(0.1^2))

    # AdvectionEquation
    eq = AdvectionEquation(closed, identity_filter)
    @test eq isa AdvectionEquation{ClosedIntervalDomain{typeof(a)},IdentityFilter}
    eq = AdvectionEquation(closed, top_hat_filter)
    @test eq isa AdvectionEquation{ClosedIntervalDomain{typeof(a)},TopHatFilter}
    eq = AdvectionEquation(closed, convolutional_filter)
    @test eq isa AdvectionEquation{ClosedIntervalDomain{typeof(a)},ConvolutionalFilter}
    eq = AdvectionEquation(periodic, identity_filter)
    @test eq isa AdvectionEquation{PeriodicIntervalDomain{typeof(a)},IdentityFilter}

    # DiffusionEquation
    f = (x, t) -> x + t
    g_a = t -> t
    g_b = t -> 0.0
    eq = DiffusionEquation(closed, identity_filter; f, g_a, g_b)
    @test eq isa DiffusionEquation{ClosedIntervalDomain{typeof(a)},IdentityFilter}
    eq = DiffusionEquation(closed, top_hat_filter; f, g_a, g_b)
    @test eq isa DiffusionEquation{ClosedIntervalDomain{typeof(a)},TopHatFilter}
    eq = DiffusionEquation(closed, convolutional_filter; f, g_a, g_b)
    @test eq isa DiffusionEquation{ClosedIntervalDomain{typeof(a)},ConvolutionalFilter}
    eq = DiffusionEquation(periodic, identity_filter; f, g_a, g_b)
    @test eq isa DiffusionEquation{PeriodicIntervalDomain{typeof(a)},IdentityFilter}

    # BurgersEquation
    eq = BurgersEquation(closed, identity_filter)
    @test eq isa BurgersEquation{ClosedIntervalDomain{typeof(a)},IdentityFilter}
    eq = BurgersEquation(closed, top_hat_filter)
    @test eq isa BurgersEquation{ClosedIntervalDomain{typeof(a)},TopHatFilter}
    eq = BurgersEquation(closed, convolutional_filter)
    @test eq isa BurgersEquation{ClosedIntervalDomain{typeof(a)},ConvolutionalFilter}
    eq = BurgersEquation(periodic, identity_filter)
    @test eq isa BurgersEquation{PeriodicIntervalDomain{typeof(a)},IdentityFilter}
end
