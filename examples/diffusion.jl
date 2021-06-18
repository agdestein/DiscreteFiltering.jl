using DiscreteFiltering
using DifferentialEquations
using LinearAlgebra
using Plots
using SparseArrays
# using Zygote


## Filter
h₀ = 0.07
h(x) = h₀# * (1 - 1 / 2 * cos(x))
# dh = h'
dh(x) = 0.0#h₀ / 2 * sin(x)
α(x) = 1 / 3 * dh(x) * h(x)
filter = TopHatFilter(h)


## Domain
a = 0.0
b = 2π
# domain = PeriodicIntervalDomain(a, b)
domain = ClosedIntervalDomain(a, b)

## Discretization
n = 200
x = discretize_uniform(domain, n)
Δx = (b - a) / n

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
C = advection_matrix(domain, n)
D = diffusion_matrix(domain, n)
W = filter_matrix(filter, domain, n)
R = inverse_filter_matrix(filter, domain, n)
A = spdiagm(α.(x))


## Inspect matrices
spy(W)
spy(R)

## Plot weights at different thicknesses
pl = plot()
for i in (n ÷ 10) * [0, 1, 2, 3, 4, 5] .+ 1
    a = W[i, :]
    inds = a.nzind
    vals = a.nzval
    ishift = inds .- i
    inds1 = ishift .≤ n ÷ 2
    inds2 = .!inds1
    ishift[inds2] .-= n
    plot!(pl, [ishift[inds2]; ishift[inds1]], [vals[inds2]; vals[inds1]], label = "i = $i")
    # scatter!(
    #     pl,
    #     [ishift[inds2]; ishift[inds1]],
    #     [vals[inds2]; vals[inds1]],
    #     label = "i = $i",
    # )
end
pl


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
title!("Initial conditions")


## Solve discretized problem
∂uₕ∂t(uₕ, p, t) = D * uₕ
prob = ODEProblem(∂uₕ∂t, uₕ, (0, T))
sol = solve(prob, abstol = 1e-6, reltol = 1e-4)

plot(x, uₕ, label = "Initial conditions")
plot!(x, sol(t), label = "Discretized")
title!("Solution")


## Solve filtered-and-then-discretized problem
∂ūₕ∂t(ūₕ, p, t) = (D + A * D) * ūₕ
prob_bar = ODEProblem(∂ūₕ∂t, ūₕ, (0, T))
sol_bar = solve(prob_bar, abstol = 1e-6, reltol = 1e-4)

plot(x, uₕ, label = "Initial")
plot!(x, ūₕ, label = "Initial filtered")
plot!(x, sol_bar(t), label = "Filtered-then-discretized")
title!("Solution")


## Solve discretized-and-then-filtered problem
∂uₕ_allbar∂t(uₕ_allbar, p, t) = W * (D * (R * uₕ_allbar))
# ∂uₕ_allbar∂t(uₕ_allbar, p, t) = W * (D * (W \ uₕ_allbar))
prob_allbar = ODEProblem(∂uₕ_allbar∂t, W * uₕ, (0, T))
sol_allbar = solve(prob_allbar, abstol = 1e-6, reltol = 1e-4)

plot(x, uₕ, label = "Initial")
plot!(x, uₕ_allbar, label = "Initial discretized-then-filtered")
plot!(x, sol_allbar(t), label = "Discretized-then-filtered")
title!("Solution")


## Comparison
plot(x, uₕ, label = "Initial")
plot!(x, ūₕ, label = "Initial filtered")
plot!(x, sol(t), label = "Discretized")
plot!(x, sol_bar(t), label = "Filtered-then-discretized")
plot!(x, sol_allbar(t), label = "Discretized-then-filtered")
# plot!(x, [u.(x, t), ū.(x, t)], label = "Exact")
ylims!(minimum(uₕ), maximum(uₕ))
title!("Solution")


## Relative error
u_exact = sol(t)
ū_exact = W * sol(t)
err = abs.(sol(t) - u_exact) ./ maximum(abs.(u_exact))
err_bar = abs.(sol_bar(t) - u_exact) ./ maximum(abs.(u_exact))
err_allbar = abs.(sol_allbar(t) - u_exact) ./ maximum(abs.(u_exact))

##
plot()
plot!(x, err, label = "Unfiltered discretized")
plot!(x, err_bar, label = "Filtered-then-discretized")
plot!(x, err_allbar, label = "Discretized-then-filtered")
title!("Relative error")
