using DiscreteFiltering
using DifferentialEquations
using LinearAlgebra
using Plots
using SparseArrays
using Zygote


## Filter
h₀ = 0.03
h(x) = h₀ * (1 - 1 / 2 * cos(x))
filter = TopHatFilter(h)


## Discretization
n = 500
x = LinRange(2π / n, 2π, n)
Δx = x[2] - x[1]

Δx^2 / maximum(abs.(α.(x)))


## Time
T = 1.0
t = T


## Plot filter
plot(x, h)
ylims!((0, ylims()[2]))


## Get matrices
C = advection_matrix(Δx, n)
D = diffusion_matrix(Δx, n)
W = filter_matrix(filter, x)
R = inverse_filter_matrix(filter, x)


## Exact solutions
u₀(x) = sin(x) + 0.6cos(5x) + 0.04sin(20(x - 1))

## Discrete initial conditions
uₕ = u₀.(x, 0.0)

plot(x, uₕ, label = "Discretized")

## Extension to a non-linear case: Burgers equation
ν = 0.03
burgers_∂uₕ∂t(uₕ, p, t) = -uₕ .* (C * uₕ) + ν * D * uₕ
burgers_prob = ODEProblem(burgers_∂uₕ∂t, uₕ, (0, T))
burgers_sol = solve(burgers_prob, abstol = 1e-6, reltol = 1e-4)

function burgers_∂uₕ_allbar∂t(uₕ_allbar, p, t)
    uₕ = R * uₕ_allbar
    -W * (uₕ .* (C * uₕ)) + ν * W * D * uₕ
end
burgers_prob_allbar = ODEProblem(burgers_∂uₕ_allbar∂t, W * uₕ, (0, T))
burgers_sol_allbar = solve(burgers_prob_allbar, abstol = 1e-6, reltol = 1e-4)

plot(x, uₕ, label = "Initial")
plot!(x, W * uₕ, label = "Initial discretized-then-filtered")
#plot!(x, burgers_sol(t), label = "Discretized")
plot!(x, W * burgers_sol(t), label = "W * Discretized")
plot!(x, burgers_sol_allbar(t), label = "Discretized-then-filtered")
ylims!(minimum(uₕ), maximum(uₕ))
