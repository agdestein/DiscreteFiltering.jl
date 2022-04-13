"Discrete filtering toolbox"
module DiscreteFiltering

using ApproxFun: Fun, chebyshevt, integrate, Interval, (..)
using ForwardDiff
using IntervalSets: ±
using LinearAlgebra
using Makie: Makie
using MLJLinearModels
using OrdinaryDiffEq: OrdinaryDiffEq
using OrdinaryDiffEq: ODEFunction, ODEProblem, QNDF, RK4, DiffEqArrayOperator, LinearExponential
using Parameters
using Plots: Plots
using SparseArrays
using NonNegLeastSquares

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
include("matrices/fit_Cbar.jl")

# Equations
include("equations/equations.jl")
include("equations/solve.jl")
include("equations/solve_adbc.jl")

# Utils
include("utils/circulant.jl")
include("utils/ridge.jl")
include("utils/sum_of_sines.jl")
include("utils/plotmat.jl")

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
export fit_Cbar, fit_Cbar_approx

# Equations
export AdvectionEquation, DiffusionEquation, BurgersEquation, solve, solve_adbc

# Utils
export circulant
export sum_of_sines
export pplotmat, mplotmat

end
