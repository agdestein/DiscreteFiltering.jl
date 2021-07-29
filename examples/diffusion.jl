using DiscreteFiltering
using LinearAlgebra
using Plots
using SparseArrays


## Domain
a = 0.0
b = 2π
# domain = PeriodicIntervalDomain(a, b)
domain = ClosedIntervalDomain(a, b)


## Discretization
n = 200
x = discretize(domain, n)
Δx = (b - a) / n


## Filter
# h₀ = Δx
# h(x) = h₀ # * (1 - 1 / 2 * cos(x))
# dh(x) = 0.0 # h₀ / 2 * sin(x)
# filter = TopHatFilter(h)

h₀ = 5.1Δx
h(x) = h₀ # * (1 - 1 / 2 * cos(x))
dh(x) = 0.0 # h₀ / 2 * sin(x)
filter = GaussianFilter(h, Δx / 2)
# filter = ConvolutionalFilter(h, x -> (-Δx / 2 ≤ x ≤ Δx / 2) / Δx)

# Filter matrix
W = filter_matrix(filter, domain, n)

## Equations
f = (x, t) -> 0.0
g_a = t -> 0.0
g_b = t -> 0.0
equation = DiffusionEquation(domain, IdentityFilter(), f, g_a, g_b)
equation_filtered = DiffusionEquation(domain, filter, f, g_a, g_b)


## Time
T = 0.04
t = T


# ODE solver tolerances
tols = (; abstol = 1e-6, reltol = 1e-4)


## Plot filter
plot(x, h)
ylims!((0, ylims()[2]))


## Exact solutions (force zero BC)
_u₀(x) = sin(x) + 0.6cos(5x) + 0.04sin(20(x - 1))
u₀(x) = _u₀(x) - (_u₀(a) + (_u₀(b) - _u₀(a)) * (x - a) / (b - a))
u₀_int(x) =
    -cos(x) + 0.6 / 5 * sin(5x) - 0.04 / 20 * cos(20(x - 1)) -
    (_u₀(a)x + (_u₀(b) - _u₀(a)) * (x - a)^2 / 2(b - a))
if domain isa PeriodicIntervalDomain
    ū₀(x) = 1 / 2h(x) * (u₀_int(x + h(x)) - u₀_int(x - h(x)))
else
    ū₀(x) = begin
        α = max(a, x - h(x))
        β = min(b, x + h(x))
        1 / (β - α) * (u₀_int(β) - u₀_int(α))
    end
end


## Discrete initial conditions
uₕ = u₀.(x)
ūₕ = ū₀.(x)

plot(x, uₕ, label = "Unfiltered")
plot!(x, ūₕ, label = "Filtered")
title!("Initial conditions")


## Solve discretized problem
sol = solve(
    equation,
    u₀,
    (0.0, T),
    n;
    method = "discretizefirst",
    boundary_conditions = "derivative",
    tols...,
)
plot(x, uₕ, label = "Initial conditions")
plot!(x, sol(t), label = "Discretized")
title!("Solution")


## Solve filtered-and-then-discretized problem with ADBC
ū_adbc = solve_adbc(equation_filtered, u₀, (0.0, T), n, T / 100_000)
plot(x, uₕ, label = "Initial")
# plot!(x, ūₕ, label = "Initial filtered")
plot!(x, ū_adbc, label = "Filtered-then-discretized (ADBC)")
title!("Solution")


## Solve discretized-and-then-filtered problem
sol_allbar = solve(
    equation_filtered,
    u₀,
    (0.0, T),
    n;
    method = "discretizefirst",
    boundary_conditions = "derivative",
    tols...,
)

plot(x, uₕ, label = "Initial")
# plot!(x, ūₕ, label = "Initial discretized-then-filtered")
plot!(x, sol_allbar(t), label = "Discretized-then-filtered")
title!("Solution")


## Comparison
plot(x, uₕ, label = "Initial")
# plot!(x, ūₕ, label = "Initial filtered")
plot!(x, sol(t), label = "Discretized")
# plot!(x, sol_bar(t), label = "Filtered-then-discretized")
plot!(x, ū_adbc, label = "Filtered-then-discretized")
plot!(x, sol_allbar(t), label = "Discretized-then-filtered")
# plot!(x, [u.(x, t), ū.(x, t)], label = "Exact")
ylims!(minimum(uₕ), maximum(uₕ))
title!("Solution")


## Relative error
u_exact = sol(t)
ū_exact = W * sol(t)
# err = abs.(sol(t) - u_exact) ./ maximum(abs.(u_exact))
err_bar = abs.(ū_adbc - ū_exact) ./ maximum(abs.(ū_exact))
err_allbar = abs.(sol_allbar(t) - ū_exact) ./ maximum(abs.(ū_exact))

##
plot(yaxis = :log)
# plot!(x, err, label = "Unfiltered discretized")
plot!(x, err_bar, label = "Filtered-then-discretized (ADBC)")
plot!(x, err_allbar, label = "Discretized-then-filtered")
title!("Relative error")
