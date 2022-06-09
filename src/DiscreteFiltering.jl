"""
Discrete filtering toolbox
"""
module DiscreteFiltering

using DiffEqFlux
using FFTW
using LinearAlgebra
using OrdinaryDiffEq
using Plots
using Printf
using Random
using SparseArrays

# ODE right hand side
f(u, A, t) = A * u
f!(du, u, A, t) = mul!(du, A, u)

"""
    S(A, u₀, t; kwargs...)

Solve ODE for given operator and IC. This form is differentiable.
"""
function S(A, u₀, t; kwargs...)
    problem = ODEProblem(ODEFunction(f), u₀, extrema(t), A)
    solve(problem, Tsit5(); saveat = t, kwargs...)
end

"""
    S!(A, u, t; kwargs...)

Solve ODE for given operator and IC (mutating form, not differentiable).
"""
function S!(A, u₀, t; kwargs...)
    problem = ODEProblem(ODEFunction(f!), u₀, extrema(t), A)
    solve(problem, Tsit5(); saveat = t, kwargs...)
end

include("filters.jl")
include("intrusive.jl")
include("create_data.jl")
include("errors.jl")

# Utils
include("utils/circulant.jl")
include("utils/plotmat.jl")
include("utils/figsave.jl")

export create_tophat, create_gaussian, filter_matrix
export S, S!
export create_loss, fit_intrusive
export relerrs, relerr, spectral_relerr
export create_data_exact, create_data_dns, create_data_filtered
export u, dudt, ū, dūdt

# Utils
export circulant
export sum_of_sines
export plotmat, figsave

end
