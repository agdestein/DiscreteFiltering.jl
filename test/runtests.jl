using DiscreteFiltering
using LinearAlgebra
using SparseArrays
using Test

@testset "Domains" begin
    include("structures/domain.jl")
end

@testset "Filters" begin
    include("structures/filter.jl")
end

@testset "Matrix assembly" begin
    include("matrices/advection_matrix.jl")
    include("matrices/diffusion_matrix.jl")
    include("matrices/filter_matrix.jl")
    include("matrices/inverse_filter_matrix.jl")
end
