using DiscreteFiltering
using Test
using SafeTestsets

@time @safetestset "Domains" begin
    include("test_Domain.jl")
end

@time @safetestset "Filters" begin
    include("test_TopHatFilter.jl")
    include("test_ConvolutionalFilter.jl")
end

@time @safetestset "Matrix assembly" begin
    include("test_advection_matrix.jl")
    include("test_diffusion_matrix.jl")
    include("test_filter_matrix.jl")
    include("test_inverse_filter_matrix.jl")
end
