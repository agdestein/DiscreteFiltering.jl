"Discrete filtering toolbox"
module DiscreteFiltering

using ApproxFun: Fun, chebyshevt, integrate, Interval, (..)
using ForwardDiff: derivative
using IntervalSets: ±
using LinearAlgebra: Diagonal, I, mul!, factorize, ldiv!, lu
using OrdinaryDiffEq: OrdinaryDiffEq
using OrdinaryDiffEq: ODEFunction, ODEProblem, QNDF, DiffEqArrayOperator, LinearExponential
using Parameters: @unpack
using SparseArrays: dropzeros!, sparse, spdiagm, spzeros
using NonNegLeastSquares: nonneg_lsq

# Domain
include("domain/domain.jl")
include("domain/get_npoint.jl")
include("domain/discretize.jl")

# Filter
include("filter/filter.jl")
include("filter/apply_filter.jl")
include("filter/apply_filter_int.jl")
include("filter/apply_filter_extend.jl")

# Matrix assembly
include("matrices/advection_matrix.jl")
include("matrices/diffusion_matrix.jl")
include("matrices/interpolation_matrix.jl")
include("matrices/filter_matrix.jl")
include("matrices/filter_matrix_meshwidth.jl")
include("matrices/reconstruction_matrix.jl")
include("matrices/reconstruction_matrix_meshwidth.jl")
include("matrices/get_W_R.jl")

# Equations
include("equations/equations.jl")
include("equations/solve.jl")
include("equations/solve_adbc.jl")

# Utils
include("utils/ridge.jl")

# Domain
export ClosedIntervalDomain, PeriodicIntervalDomain, discretize, get_npoint

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
export get_W_R

# Equations
export AdvectionEquation, DiffusionEquation, BurgersEquation, solve, solve_adbc

end
