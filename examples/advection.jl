using DiscreteFiltering
using DifferentialEquations
using LinearAlgebra
using Plots
using SparseArrays
# using Zygote


## Filter
h₀ = 0.03
h(x) = h₀ # * (1 - 1 / 2 * cos(x))
# dh = h'
dh(x) = 0.0 # h₀ / 2 * sin(x)
α(x) = 1 / 3 * dh(x) * h(x)
filter = TopHatFilter(h)


## Domain
a = 0.0
b = 2π
domain = PeriodicIntervalDomain(a, b)
# domain = ClosedIntervalDomain(a, b)


## Discretization
n = 1000
x = discretize_uniform(domain, n)
Δx = (b - a) / n

Δx^2 / maximum(abs.(α.(x)))


## Time
T = 1.0
t = T


## Plot filter
plot(x, h)
ylims!((0, ylims()[2]))


## Plot α and step size
plot(x, abs.(α.(x)), label = "|α(x)|")
plot!([x[1], x[end]], [Δx / 2, Δx / 2], label = "Δx/2")


## Get matrices
C = advection_matrix(domain, n)
D = diffusion_matrix(domain, n)
W = filter_matrix(filter, domain, n)
R = inverse_filter_matrix(filter, domain, n)
A = spdiagm(α.(x))


## Exact solutions
u(x, t) = sin(x - t) + 0.6cos(5(x - t)) + 0.04sin(20(x - 1 - t))
u_int(x, t) = -cos(x - t) + 0.6 / 5 * sin(5(x - t)) - 0.04 / 20 * cos(20(x - 1 - t))
ū(x, t) = 1 / 2h(x) * (u_int(x + h(x), t) - u_int(x - h(x), t))


## Discrete initial conditions
uₕ = u.(x, 0.0)
ūₕ = ū.(x, 0.0)
uₕ_allbar = W * uₕ

plot(x, uₕ, label = "Discretized")
plot!(x, ūₕ, label = "Filtered-then-discretized")
plot!(x, uₕ_allbar, label = "Discretized-then-filtered")


## Solve discretized problem
∂uₕ∂t(uₕ, p, t) = -C * uₕ
prob = ODEProblem(∂uₕ∂t, uₕ, (0, T))
sol = solve(prob, abstol = 1e-6, reltol = 1e-4)

plot(x, uₕ, label = "Initial conditions")
plot!(x, sol(t), label = "Discretized")
plot!(x, u.(x, t), label = "Exact")


## Solve filtered-and-then-discretized problem
∂ūₕ∂t(ūₕ, p, t) = (-C + A * D) * ūₕ
prob_bar = ODEProblem(∂ūₕ∂t, ūₕ, (0, T))
sol_bar = solve(prob_bar, abstol = 1e-6, reltol = 1e-4)

plot(x, uₕ, label = "Initial")
plot!(x, ūₕ, label = "Initial filtered")
plot!(x, sol_bar(t), label = "Filtered-then-discretized")
plot!(x, ū.(x, t), label = "Filtered exact")
#plot!(x, 500*α.(x))


## Solve discretized-and-then-filtered problem
∂uₕ_allbar∂t(uₕ_allbar, p, t) = -W * (C * (R * uₕ_allbar))
# ∂uₕ_allbar∂t(uₕ_allbar, p, t) = -W * (C * (W \ uₕ_allbar))
prob_allbar = ODEProblem(∂uₕ_allbar∂t, W * uₕ, (0, T))
sol_allbar = solve(prob_allbar, abstol = 1e-6, reltol = 1e-4)

plot(x, uₕ, label = "Initial")
plot!(x, uₕ_allbar, label = "Initial discretized-then-filtered")
plot!(x, sol_allbar(t), label = "Discretized-then-filtered")
plot!(x, W * u.(x, t), label = "Exact")


## Comparison
plot(x, uₕ, label = "Initial")
plot!(x, ūₕ, label = "Initial filtered")
plot!(x, sol(t), label = "Discretized")
plot!(x, sol_bar(t), label = "Filtered-then-discretized")
plot!(x, sol_allbar(t), label = "Discretized-then-filtered")
# plot!(x, [u.(x, t), ū.(x, t)], label = "Exact")
ylims!(minimum(uₕ), maximum(uₕ))


## Relative error
u_exact = u.(x, t)
ū_exact = ū.(x, t)
err = abs.(sol(t) - u_exact) ./ maximum(abs.(u_exact))
err_bar = abs.(sol_bar(t) - u_exact) ./ maximum(abs.(u_exact))
err_allbar = abs.(sol_allbar(t) - u_exact) ./ maximum(abs.(u_exact))

plot()
plot!(x, err, label = "Unfiltered discretized")
plot!(x, err_bar, label = "Filtered-then-discretized")
plot!(x, err_allbar, label = "Discretized-then-filtered")
