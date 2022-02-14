using OrdinaryDiffEq: ODEProblem, solve, Tsit5
using LinearAlgebra
using Plots
using SparseArrays

# LSP indexing solution from
# https://github.com/julia-vscode/julia-vscode/issues/800#issuecomment-650085983
if isdefined(@__MODULE__, :LanguageServer)
    include("../src/DiscreteFiltering.jl")
    using .DiscreteFiltering
else
    using DiscreteFiltering
end


## Domain
a = 0.0
b = 2π
domain = PeriodicIntervalDomain(a, b)
# domain = ClosedIntervalDomain(a, b)


## Discretization
N = 100
M = N
x = discretize(domain, M)
ξ = discretize(domain, N)
Δx = (b - a) / N


## Filter
h₀ = 2.1Δx
h(x) = h₀ * (1 - 1 / 2 * cos(x))
dh(x) = h₀ / 2 * sin(x)
α(x) = 1 / 3 * dh(x) * h(x)
f = TopHatFilter(h)

# h₀ = 5.1Δx
# h(x) = h₀ # * (1 - 1 / 2 * cos(x))
# dh(x) = 0.0 # h₀ / 2 * sin(x)
# α(x) = 1 / 3 * dh(x) * h(x)
# f = GaussianFilter(h, Δx / 2)


## Time
T = 1.5
t = T


## Plot filter
plot(x, h)
ylims!((0, ylims()[2]))


## Get matrices
C = advection_matrix(domain, N)
D = diffusion_matrix(domain, N)
W = filter_matrix(f, domain, M, N)
R = reconstruction_matrix(f, domain, M, N)
A = spdiagm(α.(ξ))

## Inspect matrices
spy(W)
spy(R)


## Exact solutions
u(ξ, t) = sin(ξ - t) + 3 / 5 * cos(5(ξ - t)) + 1 / 25 * sin(20(ξ - 1 - t))
u_int(ξ, t) = -cos(ξ - t) + 3 / 25 * sin(5(ξ - t)) - 1 / 25 / 20 * cos(20(ξ - 1 - t))
ū(x, t) = 1 / 2h(x) * (u_int(x + h(x), t) - u_int(x - h(x), t))


## Discrete initial conditions
uₕ = u.(ξ, 0.0)
ūₕ = ū.(x, 0.0)
uₕ_allbar = W * uₕ

plot(x, uₕ, label = "Discretized")
plot!(x, ūₕ, label = "Filtered-then-discretized")
plot!(x, uₕ_allbar, label = "Discretized-then-filtered")


## Solve discretized problem
duₕ(uₕ, p, t) = -C * uₕ
prob = ODEProblem(duₕ, uₕ, (0, T))
sol = solve(prob, Tsit5(), abstol = 1e-6, reltol = 1e-4)

plot(x, uₕ, label = "Initial conditions")
plot!(x, sol(t), label = "Discretized")
plot!(x, u.(x, t), label = "Exact")


## Solve filtered-and-then-discretized problem
dūₕ(ūₕ, p, t) = (-C + A * D) * ūₕ
prob_bar = ODEProblem(dūₕ, ūₕ, (0, T))
sol_bar = solve(prob_bar, Tsit5(), abstol = 1e-6, reltol = 1e-4)

plot(x, uₕ, label = "Initial")
plot!(x, ūₕ, label = "Initial filtered")
plot!(x, sol_bar(t), label = "Filtered-then-discretized")
plot!(x, ū.(x, t), label = "Filtered exact")
# plot!(x, 500*α.(ξ))


## Solve discretized-and-then-filtered problem
duₕ_allbar(uₕ_allbar, p, t) = -W * (C * (R * uₕ_allbar))
# duₕ_allbar(uₕ_allbar, p, t) = -W * (C * (W \ uₕ_allbar))
prob_allbar = ODEProblem(duₕ_allbar, W * uₕ, (0, T))
sol_allbar = solve(prob_allbar, Tsit5(), abstol = 1e-6, reltol = 1e-4)

plot(x, uₕ, label = "Initial")
plot!(x, uₕ_allbar, label = "Initial discretized-then-filtered")
plot!(x, sol_allbar(t), label = "Discretized-then-filtered")
plot!(x, W * u.(ξ, t), label = "Exact")

## PGFPlotsX
pgfplotsx()

## Comparison
p = plot(
    size = (400, 300),
    xlabel = "\$x\$",
    legend = :topright,
)
plot!(p, x, uₕ, label = "\$u(x, t = 0.0)\$")
# plot!(p, x, ūₕ, label = "Initial filtered")
# plot!(p, x, sol(t), label = "Discretized")
# plot!(p, x, sol_bar(t), label = "Filtered-then-discretized")
# plot!(p, x, sol_allbar(t), label = "Discretized-then-filtered")
# plot!(p, x, [u.(x, t), ū.(x, t)], label = "Exact")
plot!(p, x, u.(x, t), label = "\$u(x, t = $t)\$")
plot!(p, x, ū.(x, t), label = "\$u(x, t = $t)\$")
# ylims!(p, minimum(uₕ), maximum(uₕ))
display(p)
savefig(p, "output/advection/solution.tikz")


## Relative error
u_exact = u.(ξ, t)
ū_exact = ū.(x, t)
err = abs.(sol(t) - u_exact) ./ maximum(abs.(u_exact))
err_bar = abs.(sol_bar(t) - u_exact) ./ maximum(abs.(u_exact))
err_allbar = abs.(sol_allbar(t) - u_exact) ./ maximum(abs.(u_exact))

plot()
plot!(ξ, err, label = "Unfiltered discretized")
plot!(x, err_bar, label = "Filtered-then-discretized")
plot!(x, err_allbar, label = "Discretized-then-filtered")
