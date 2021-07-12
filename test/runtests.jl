using DiscreteFiltering
using LinearAlgebra
using SparseArrays
using Test

@testset "Domains" begin
    include("domain.jl")
end

@testset "Filters" begin
    include("filter.jl")
end

@testset "Matrix assembly" begin
    include("advection_matrix.jl")
    include("diffusion_matrix.jl")
    include("filter_matrix.jl")
    include("inverse_filter_matrix.jl")
end
