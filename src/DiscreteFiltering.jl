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

export advection_matrix,
    ClosedIntervalDomain,
    diffusion_matrix,
    discretize,
    filter_matrix,
    filter_matrix_meshwidth,
    inverse_filter_matrix,
    inverse_filter_matrix_meshwidth,
    PeriodicIntervalDomain,
    TopHatFilter,
    ConvolutionalFilter

end
