using DiscreteFiltering
using DifferentialEquations
using LinearAlgebra
using Plots
using SparseArrays
using Zygote


## Filter
h₀ = 0.03
h(x) = h₀ * (1 - 1 / 2 * cos(x))
# dh = h'
dh(x) = 0.0 h₀ / 2 * sin(x)
α(x) = 1 / 3 * dh(x) * h(x)
filter = TopHatFilter(h)


## Domain
a = 0.0
b = 2π
domain = PeriodicIntervalDomain(a, b)
# domain = ClosedIntervalDomain(a, b)


## Discretization
n = 500
x = discretize_uniform(domain, n)
Δx = (b - a) / n

Δx^2 / maximum(abs.(α.(x)))


## Time
T = 1.0
t = T


## Plot filter
plot(x, h)
ylims!((0, ylims()[2]))


## Get matrices
C = advection_matrix(domain, n)
D = diffusion_matrix(domain, n)
W = filter_matrix(filter, domain, n)
R = inverse_filter_matrix(filter, domain, n)
A = spdiagm(α.(x))


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
