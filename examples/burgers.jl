using DiscreteFiltering
using OrdinaryDiffEq: ODEProblem, solve, QNDF
using LinearAlgebra
using Plots
using SparseArrays


## Filter
h₀ = 0.03
h(x) = h₀ * (1 - 1 / 2 * cos(x))
dh(x) = 0.0 # h₀ / 2 * sin(x)
α(x) = 1 / 3 * dh(x) * h(x)
filter = TopHatFilter(h)


## Domain
a = 0.0
b = 2π
domain = PeriodicIntervalDomain(a, b)
# domain = ClosedIntervalDomain(a, b)


## Discretization
N = 500
M = N
x = discretize(domain, M)
ξ = discretize(domain, N)
Δx = (b - a) / N

Δx^2 / maximum(abs.(α.(x)))


## Time
T = 1.0
t = T


## Plot filter
plot(x, h)
ylims!((0, ylims()[2]))


## Get matrices
C = advection_matrix(domain, N)
D = diffusion_matrix(domain, N)
W = filter_matrix(filter, domain, M, N)
R = reconstruction_matrix(filter, domain, M, N)
A = spdiagm(α.(ξ))


## Exact solutions
u₀(ξ) = sin(ξ) + 0.6cos(5ξ) + 0.04sin(20(ξ - 1))

## Discrete initial conditions
uₕ = u₀.(ξ, 0.0)
plot(ξ, uₕ, label = "Discretized")

## Extension to a non-linear case: Burgers equation
ν = 0.03
duₕ(uₕ, p, t) = -uₕ .* (C * uₕ) + ν * D * uₕ
prob = ODEProblem(duₕ, uₕ, (0, T))
sol = solve(prob, QNDF(), abstol = 1e-6, reltol = 1e-4)

function duₕ_allbar(uₕ_allbar, p, t)
    uₕ = R * uₕ_allbar
    -W * (uₕ .* (C * uₕ)) + ν * W * D * uₕ
end
prob_allbar = ODEProblem(duₕ_allbar, W * uₕ, (0, T))
sol_allbar = solve(prob_allbar, QNDF(), abstol = 1e-6, reltol = 1e-4)

plot(ξ, uₕ, label = "Initial")
plot!(x, W * uₕ, label = "Initial discretized-then-filtered")
#plot!(ξ, sol(t), label = "Discretized")
plot!(x, W * sol(t), label = "W * Discretized")
plot!(x, sol_allbar(t), label = "Discretized-then-filtered")
ylims!(minimum(uₕ), maximum(uₕ))
