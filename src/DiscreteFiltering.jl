"""
Discrete filtering toolbox
"""
module DiscreteFiltering

using FFTW
using LinearAlgebra
using OrdinaryDiffEq
using Plots
using Printf
using Random
using SciMLSensitivity
using SparseArrays
using Zygote

# ODE right hand side
f(u, A, t) = A * u
f!(du, u, A, t) = mul!(du, A, u)
function f_stencil!(du, u::AbstractVector, s, t)
    N = length(u)
    n = length(s) ÷ 2
    for i = 1:N
        du[i] = 0
        for j = -n:n
            du[i] += s[n+1+j] * u[mod(i + j, 1:N)]
        end
    end
    du
end
function f_stencil!(du, u::AbstractMatrix, s, t)
    N, K = size(u)
    n = length(s) ÷ 2
    @inbounds for k = 1:K
        for i = 1:N
            du[i, k] = 0
            for j = -n:n
                du[i, k] += s[n+1+j] * u[mod(i + j, 1:N), k]
            end
        end
    end
    du
end

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
    # problem = ODEProblem(ODEFunction(f!), u₀, extrema(t), A)
    # problem = ODEProblem(ODEFunction(f!; jac = (J, u, p, t) -> (J .= A)), u₀, extrema(t), A)
    problem = ODEProblem(DiffEqArrayOperator(A), u₀, extrema(t), nothing)
    solve(
        problem,
        # Tsit5();
        LinearExponential();
        # saveat = t,
        tstops = t,
        kwargs...,
    )
end

"""
    S_stencil!(A, u, t; kwargs...)

Solve ODE for given operator and IC (mutating form, not differentiable).
"""
function S_stencil!(stencil, u₀, t; kwargs...)
    @assert length(stencil) ≤ size(u₀, 1)
    problem = ODEProblem(ODEFunction(f_stencil!), u₀, extrema(t), stencil)
    solve(problem, Tsit5(); saveat = t, kwargs...)
end

include("filters.jl")
include("embedded.jl")
include("create_data.jl")
include("errors.jl")

# Utils
include("utils/circulant.jl")
include("utils/plotmat.jl")
include("utils/figsave.jl")

export create_tophat, create_gaussian, filter_matrix, interpolation_matrix
export S, S!, S_stencil!
export create_loss_fit,
    create_loss_prior, create_initial_state, create_loss_mixed, fit_embedded
export relerrs, relerr, spectral_relerr
export create_data_exact, create_data_dns, create_data_dns_stencil, create_data_filtered
export u, dudt, ū, dūdt

# Utils
export circulant
export plotmat, figsave

end
