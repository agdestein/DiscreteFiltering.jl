@testset "ridge.jl" begin
    A = rand(3, 3)
    b = rand(3)
    x = ridge(A, b)
    @assert x ≈ A \ b
end
