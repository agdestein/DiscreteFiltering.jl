"Discrete filtering toolbox"
module DiscreteFiltering

using LinearAlgebra
using SparseArrays

# Filter
include("Filter.jl")

# Matrix assembly
include("advection_matrix.jl")
include("diffusion_matrix.jl")

export advection_matrix, diffusion_matrix, TopHatFilter

end
