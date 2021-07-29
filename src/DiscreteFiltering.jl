"Discrete filtering toolbox"
module DiscreteFiltering

using ApproxFun: Fun, chebyshevt, integrate
using IntervalSets
using LinearAlgebra
using OrdinaryDiffEq:
    ODEFunction, ODEProblem, QNDF, OrdinaryDiffEq, DiffEqArrayOperator, LinearExponential
using Parameters
using SparseArrays
using NonNegLeastSquares
using Zygote

# Domain
include("structures/domain.jl")

# Filter
include("structures/filter.jl")

# Matrix assembly
include("matrices/advection_matrix.jl")
include("matrices/diffusion_matrix.jl")
include("matrices/filter_matrix.jl")
include("matrices/reconstruction_matrix.jl")

# Equations
include("equations/equations.jl")
include("equations/solve.jl")
include("equations/solve_adbc.jl")

export ClosedIntervalDomain, PeriodicIntervalDomain, discretize
export IdentityFilter,
    TopHatFilter,
    ConvolutionalFilter,
    GaussianFilter,
    apply_filter,
    apply_filter_extend
export advection_matrix,
    diffusion_matrix,
    filter_matrix,
    filter_matrix_meshwidth,
    reconstruction_matrix,
    reconstruction_matrix_meshwidth
export AdvectionEquation, DiffusionEquation, BurgersEquation, solve, solve_adbc

end
