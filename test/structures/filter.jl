@testset "Top hat filter" begin
    h(x) = 1 - 1 / 2 * cos(x)
    f = TopHatFilter(h)
    @test f.width == h
end

@testset "Convolutional filter" begin
    G(σ², x) = 1 / √(2πσ²) * exp(-x^2 / 2σ²)
    G(x) = G(0.1^2, x)
    f = ConvolutionalFilter(G)
    @test f.kernel == G
end