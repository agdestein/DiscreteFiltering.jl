using DiscreteFiltering
using LinearAlgebra: norm
using Plots


## Parameters

# Domain
a = 0.0
b = 2π
domain = PeriodicIntervalDomain(a, b)

# Time
T = 1.0

# Exact solutions (advection equation)
u(x, t) = sin(x - t) + 0.6cos(5(x - t)) + 0.04sin(20(x - 1 - t))
u_int(x, t) = -cos(x - t) + 0.6 / 5 * sin(5(x - t)) - 0.04 / 20 * cos(20(x - 1 - t))

# Ode solver tolerances
# tols = (;)
# tols = (; abstol = 1e-6, reltol = 1e-4)
# tols = (; abstol = 1e-7, reltol = 1e-5)
subspacedim = 50

# Number of mesh points
N = floor.(Int, 10 .^ LinRange(1, 5, 20))
# N = [100]

# Errors
err = zeros(length(N))
err_bar = zeros(length(N))
err_allbar = zeros(length(N))

## Solve
@time for (i, n) ∈ enumerate(N)

    println("Solving for n = $n")

    # Discretization
    x = discretize(domain, n)
    Δx = (b - a) / n

    # Filter
    h(x) = Δx / 2
    # h(x) = h₀ * (1 - 1 / 2 * cos(x))
    f = TopHatFilter(h)

    # Exact filtered solution
    ū(x, t) = 1 / 2h(x) * (u_int(x + h(x), t) - u_int(x - h(x), t))

    # Equations
    equation = AdvectionEquation(domain, IdentityFilter())
    equation_filtered = AdvectionEquation(domain, TopHatFilter(h))

    # Solve discretized problem
    sol = solve(
        equation,
        x -> u(x, 0.0),
        (0.0, T),
        n;
        method = "discretizefirst",
        subspacedim,
    )

    # Solve filtered-then-discretized problem
    sol_bar = solve(
        equation_filtered,
        x -> u(x, 0.0),
        (0.0, T),
        n;
        method = "filterfirst",
        subspacedim,
    )

    # Solve discretized-then-filtered problem
    sol_allbar = solve(
        equation_filtered,
        x -> u(x, 0.0),
        (0.0, T),
        n;
        method = "discretizefirst",
        subspacedim,
    )

    ## Relative error
    u_exact = u.(x, T)
    ū_exact = ū.(x, T)
    err[i] = norm(sol(T) - u_exact) / norm(u_exact)
    err_bar[i] = norm(sol_bar(T) - ū_exact) / norm(ū_exact)
    err_allbar[i] = norm(sol_allbar(T) - ū_exact) / norm(ū_exact)
end


## Set PGFPlotsX plotting backend to export to tikz
pgfplotsx()

##
p = plot(xaxis = :log, yaxis = :log, size = (400, 300), legend = :topright)
plot!(p, N, err, label = "Discretized")
plot!(p, N, err_bar, label = "Filtered-then-discretized")
plot!(p, N, err_allbar, label = "Discretized-then-filtered")
plot!(p, N, 1000N .^ -2, label = "\$100 / n^2\$")
xlabel!(p, "n")
title!(p, "Advection equation, \$h(x) = \\Delta x / 2\$")
display(p)

savefig(p, "output/advection_convergence.tikz")
