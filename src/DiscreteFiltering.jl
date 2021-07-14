"Discrete filtering toolbox"
module DiscreteFiltering

using Intervals
using LinearAlgebra
using OrdinaryDiffEq:
    ODEFunction, ODEProblem, QNDF, OrdinaryDiffEq, DiffEqArrayOperator, LinearExponential
using Parameters
using Polynomials
using SparseArrays
using NonNegLeastSquares
using Zygote

# Domain
include("structures/domain.jl")

# Filter
include("structures/filter.jl")
include("structures/gaussian.jl")

# Matrix assembly
include("matrices/advection_matrix.jl")
include("matrices/diffusion_matrix.jl")
include("matrices/filter_matrix.jl")
include("matrices/inverse_filter_matrix.jl")

# Equations
include("equations/equations.jl")
include("equations/solve.jl")

export ClosedIntervalDomain, discretize, PeriodicIntervalDomain
export IdentityFilter, TopHatFilter, ConvolutionalFilter, gaussian
export advection_matrix,
    diffusion_matrix,
    filter_matrix,
    filter_matrix_meshwidth,
    inverse_filter_matrix,
    inverse_filter_matrix_meshwidth
export AdvectionEquation, DiffusionEquation, BurgersEquation, solve

end
