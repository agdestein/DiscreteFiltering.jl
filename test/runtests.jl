using DiscreteFiltering
using LinearAlgebra
using SparseArrays
using Test

@testset "Domains" begin
    include("structures/domain.jl")
end

@testset "Filters" begin
    include("structures/gaussian.jl")
    include("structures/filter.jl")
end

@testset "Matrix assembly" begin
    include("matrices/advection_matrix.jl")
    include("matrices/diffusion_matrix.jl")
    include("matrices/filter_matrix.jl")
    include("matrices/inverse_filter_matrix.jl")
end

@testset "Equations" begin
    include("equations/equations.jl")
    include("equations/solve_diffusion.jl")
    include("equations/solve_advection.jl")
    include("equations/solve_burgers.jl")
end
