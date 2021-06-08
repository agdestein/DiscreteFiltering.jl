"Discrete filtering toolbox"
module DiscreteFiltering

using Intervals
using LinearAlgebra
using Polynomials
using SparseArrays

# Filter
include("Filter.jl")

# Matrix assembly
include("advection_matrix.jl")
include("diffusion_matrix.jl")
include("filter_matrix.jl")
include("inverse_filter_matrix.jl")

export advection_matrix,
    diffusion_matrix,
    filter_matrix,
    inverse_filter_matrix,
    TopHatFilter,
    ConvolutionalFilter

end
