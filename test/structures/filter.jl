@testset "IdentityFilter" begin
    f = IdentityFilter()
    g = IdentityFilter()
    @test f === g
end

@testset "TopHatFilter" begin
    h(x) = 1 - 1 / 2 * cos(x)
    f = TopHatFilter(h)
    @test f.width == h
end

@testset "ConvolutionalFilter" begin
    h = x -> 1
    G = x -> x^2
    f = ConvolutionalFilter(h, G)
    @test f.width == h
    @test f.kernel == G
end

@testset "GaussianFilter" begin
    σ = 0.1
    G = GaussianFilter(σ)
    @test G.kernel(0.0) ≈ 1 / √(2π * σ^2)
end
