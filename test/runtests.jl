using DiscreteFiltering
using Test
using SafeTestsets

@time @safetestset "Filters" begin
    include("test_filter.jl")
end

@time @safetestset "Matrix assembly" begin
    include("test_advection_matrix.jl")
    include("test_diffusion_matrix.jl")
end
