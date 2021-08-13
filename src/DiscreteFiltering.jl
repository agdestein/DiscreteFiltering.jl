"Discrete filtering toolbox"
module DiscreteFiltering

using ApproxFun: Fun, chebyshevt, integrate
using ForwardDiff
using IntervalSets
using LinearAlgebra
using OrdinaryDiffEq:
    ODEFunction, ODEProblem, QNDF, OrdinaryDiffEq, DiffEqArrayOperator, LinearExponential
using Parameters
using SparseArrays
using NonNegLeastSquares

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

# Domain
export ClosedIntervalDomain, PeriodicIntervalDomain, discretize

# Filter
export IdentityFilter
export TopHatFilter
export ConvolutionalFilter
export GaussianFilter
export apply_filter
export apply_filter_int
export apply_filter_extend

# Matrix assembly
export advection_matrix
export diffusion_matrix
export filter_matrix
export filter_matrix_meshwidth
export reconstruction_matrix
export reconstruction_matrix_meshwidth

# Equations
export AdvectionEquation, DiffusionEquation, BurgersEquation, solve, solve_adbc

end
