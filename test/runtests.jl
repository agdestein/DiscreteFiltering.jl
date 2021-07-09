using DiscreteFiltering
using Test

@time @testset "Domains" begin
    include("test_Domain.jl")
end

@time @testset "Filters" begin
    include("test_TopHatFilter.jl")
    include("test_ConvolutionalFilter.jl")
end

@time @testset "Matrix assembly" begin
    include("test_advection_matrix.jl")
    include("test_diffusion_matrix.jl")
    include("test_filter_matrix.jl")
    include("test_inverse_filter_matrix.jl")
end
