"Discrete filtering toolbox"
module DiscreteFiltering

using Intervals
using LinearAlgebra
using Polynomials
using SparseArrays
using NonNegLeastSquares

# Domain
include("Domain.jl")

# Filter
include("Filter.jl")

# Matrix assembly
include("advection_matrix.jl")
include("diffusion_matrix.jl")
include("filter_matrix.jl")
include("inverse_filter_matrix.jl")

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
