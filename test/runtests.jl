using DiscreteFiltering
using LinearAlgebra
using SparseArrays
using Test

@time @testset "Domains" begin
    include("domain.jl")
end

@time @testset "Filters" begin
    include("TopHatFilter.jl")
    include("ConvolutionalFilter.jl")
end

@time @testset "Matrix assembly" begin
    include("advection_matrix.jl")
    include("diffusion_matrix.jl")
    include("filter_matrix.jl")
    include("inverse_filter_matrix.jl")
end
