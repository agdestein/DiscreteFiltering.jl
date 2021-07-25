using DiscreteFiltering
using LinearAlgebra
using SparseArrays
using Test

@testset "Discrete filtering" begin
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
        include("matrices/reconstruction_matrix.jl")
    end

    @testset "Equations" begin
        include("equations/equations.jl")
        include("equations/solve.jl")
        include("equations/solve_diffusion.jl")
        include("equations/solve_advection.jl")
        include("equations/solve_burgers.jl")
        include("equations/solve_adbc.jl")
    end
end
