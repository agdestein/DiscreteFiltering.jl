"""
Discrete filtering toolbox
"""
module DiscreteFiltering

using DiffEqFlux
using LinearAlgebra
using OrdinaryDiffEq
using Plots
using SparseArrays

# ODE right hand side
f(u, A, t) = A * u
f!(du, u, A, t) = mul!(du, A, u)

"""
    S(A, u, t; kwargs...)

ODE solver for given operator and IC.
"""
function S(A, u, t; kwargs...)
    problem = ODEProblem(ODEFunction(f), u, (0.0, t[end]), A)
    solve(problem, Tsit5(); saveat = t, kwargs...)
end

"""
    S!(A, u, t; kwargs...)

ODE solver for given operator and IC (mutating form).
"""
function S!(A, u, t; kwargs...)
    problem = ODEProblem(ODEFunction(f!), u, (0.0, t[end]), A)
    solve(problem, Tsit5(); saveat = t, kwargs...)
end

include("intrusive.jl")
include("create_data.jl")
include("errors.jl")

# Utils
include("utils/circulant.jl")
include("utils/plotmat.jl")
include("utils/figsave.jl")

export S, S!
export create_loss, fit_intrusive
export relerrs, relerr
export create_data_exact, create_data_dns, create_data_filtered

# Utils
export circulant
export sum_of_sines
export plotmat, figsave

end
