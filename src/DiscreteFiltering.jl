"Discrete filtering toolbox"
module DiscreteFiltering

using Intervals
using LinearAlgebra
using Polynomials
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
include("matrices/inverse_filter_matrix.jl")

# Equations
include("equations/solve_advection.jl")
include("equations/solve_diffusion.jl")
include("equations/solve_burgers.jl")

export ClosedIntervalDomain, discretize, PeriodicIntervalDomain
export TopHatFilter, ConvolutionalFilter
export advection_matrix,
    diffusion_matrix,
    filter_matrix,
    filter_matrix_meshwidth,
    inverse_filter_matrix,
    inverse_filter_matrix_meshwidth
export solve_advection, solve_diffusion, solve_burgers

end
