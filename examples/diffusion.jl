using DiscreteFiltering
using DifferentialEquations
using LinearAlgebra
using Plots
using SparseArrays
using Zygote


## Filter
h₀ = 0.03
h(x) = h₀ * (1 - 1 / 2 * cos(x))
α(x) = 1 / 3 * h'(x) * h(x)
filter = TopHatFilter(h)


## Discretization
n = 500
x = LinRange(2π / n, 2π, n)
Δx = x[2] - x[1]

Δx^2 / maximum(abs.(α.(x)))

## Time
T = 0.04
t = T


## Plot filter
plot(x, h)
ylims!((0, ylims()[2]))


## Plot α and step size
plot(x, abs.(α.(x)), label = "|α(x)|")
plot!([x[1], x[end]], [Δx / 2, Δx / 2], label = "Δx/2")


## Get matrices
C = advection_matrix(Δx, n)
D = diffusion_matrix(Δx, n)
W = filter_matrix(filter, x, 0.001)
R = inverse_filter_matrix(filter, x, 0.001)
A = spdiagm(α.(x))


## Exact solutions
u₀(x) = sin(x) + 0.6cos(5x) + 0.04sin(20(x - 1))
u₀_int(x) = -cos(x) + 0.6 / 5 * sin(5x) - 0.04 / 20 * cos(20(x - 1))
ū₀(x) = 1 / 2h(x) * (u₀_int(x + h(x)) - u₀_int(x - h(x)))


## Discrete initial conditions
uₕ = u₀.(x)
ūₕ = ū₀.(x)
uₕ_allbar = W * uₕ

plot(x, uₕ, label = "Discretized")
plot!(x, ūₕ, label = "Filtered-then-discretized")
plot!(x, uₕ_allbar, label = "Discretized-then-filtered")


## Solve discretized problem
∂uₕ∂t(uₕ, p, t) = D * uₕ
prob = ODEProblem(∂uₕ∂t, uₕ, (0, T))
sol = solve(prob, abstol = 1e-6, reltol = 1e-4)

plot(x, uₕ, label = "Initial conditions")
plot!(x, sol(t), label = "Discretized")


## Solve filtered-and-then-discretized problem
∂ūₕ∂t(ūₕ, p, t) = (D + A * D) * ūₕ
prob_bar = ODEProblem(∂ūₕ∂t, ūₕ, (0, T))
sol_bar = solve(prob_bar, abstol = 1e-6, reltol = 1e-4)

plot(x, uₕ, label = "Initial")
plot!(x, ūₕ, label = "Initial filtered")
plot!(x, sol_bar(t), label = "Filtered-then-discretized")


## Solve discretized-and-then-filtered problem
∂uₕ_allbar∂t(uₕ_allbar, p, t) = W * (D * (R * uₕ_allbar))
# ∂uₕ_allbar∂t(uₕ_allbar, p, t) = W * (D * (W \ uₕ_allbar))
prob_allbar = ODEProblem(∂uₕ_allbar∂t, W * uₕ, (0, T))
sol_allbar = solve(prob_allbar, abstol = 1e-6, reltol = 1e-4)

plot(x, uₕ, label = "Initial")
plot!(x, uₕ_allbar, label = "Initial discretized-then-filtered")
plot!(x, sol_allbar(t), label = "Discretized-then-filtered")


## Comparison
plot(x, uₕ, label = "Initial")
plot!(x, ūₕ, label = "Initial filtered")
plot!(x, sol(t), label = "Discretized")
plot!(x, sol_bar(t), label = "Filtered-then-discretized")
plot!(x, sol_allbar(t), label = "Discretized-then-filtered")
# plot!(x, [u.(x, t), ū.(x, t)], label = "Exact")
ylims!(minimum(uₕ), maximum(uₕ))


## Relative error
u_exact = sol(t)
ū_exact = W * sol(t)
err = abs.(sol(t) - u_exact) ./ maximum(abs.(u_exact))
err_bar = abs.(sol_bar(t) - u_exact) ./ maximum(abs.(u_exact))
err_allbar = abs.(sol_allbar(t) - u_exact) ./ maximum(abs.(u_exact))

plot()
plot!(x, err, label = "Unfiltered discretized")
plot!(x, err_bar, label = "Filtered-then-discretized")
plot!(x, err_allbar, label = "Discretized-then-filtered")
