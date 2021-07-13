@testset "gaussian.jl" begin
    σ² = 0.1^2
    G = gaussian(σ²)
    @test G(0.0) ≈ 1 / √(2π * σ²)
end
