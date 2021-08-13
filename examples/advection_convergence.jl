using DiscreteFiltering
using LinearAlgebra: norm
using Plots
using ProgressLogging


## Parameters

# Domain
a = 0.0
b = 2π
domain = PeriodicIntervalDomain(a, b)

# Time
T = 1.0

# Exact solutions (advection equation)
u(x, t) = sin(x - t) + 3 / 5 * cos(5(x - t)) + 1 / 25 * sin(20(x - 1 - t))
u_int(x, t) = -cos(x - t) + 3 / 25 * sin(5(x - t)) - 1 / 25 / 20 * cos(20(x - 1 - t))

# Ode solver tolerances
# tols = (;)
# tols = (; abstol = 1e-6, reltol = 1e-4)
# tols = (; abstol = 1e-7, reltol = 1e-5)
subspacedim = 500

# Number of mesh points
N = floor.(Int, 10 .^ LinRange(2, 4, 20))
# N = [100]

# Errors
err = zeros(length(N))
err_bar = zeros(length(N))
err_allbar = zeros(length(N))

## Solve
@time @progress for (i, n) ∈ enumerate(N)
    enumerate(N)

    println("Solving for n = $n")

    # Discretization
    x = discretize(domain, n)
    Δx = (b - a) / n

    # Filter
    # h(x) = Δx / 2
    h₀ = 2.1Δx
    h(x) = h₀ * (1 - 1 / 2 * cos(x))
    f = TopHatFilter(h)
    # f = ConvolutionalFilter()

    # Exact filtered solution
    ū(x, t) = 1 / 2h(x) * (u_int(x + h(x), t) - u_int(x - h(x), t))

    # Equations
    equation = AdvectionEquation(domain, IdentityFilter())
    equation_filtered = AdvectionEquation(domain, TopHatFilter(h))

    # Solve discretized problem
    # sol = solve(
    #     equation,
    #     x -> u(x, 0.0),
    #     (0.0, T),
    #     n;
    #     method = "discretizefirst",
    #     subspacedim,
    # )

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
        reltol = 1e-8,
        abstol = 1e-10,
    )

    W = filter_matrix(f, domain, n)

    # Relative error
    u_exact = u.(x, T)
    ū_exact = ū.(x, T)
    u_allbar_exact = W * u_exact
    # err[i] = norm(sol(T) - u_exact) / norm(u_exact)
    err_bar[i] = norm(sol_bar(T) - ū_exact) / norm(ū_exact)
    err_allbar[i] = norm(sol_allbar(T) - ū_exact) / norm(ū_exact)
end


## Set PGFPlotsX plotting backend to export to tikz
pgfplotsx()

##
gr()

##
p = plot(
    xaxis = :log10,
    yaxis = :log10,
    minorgrid = true,
    # minorgridstyle = :dash,
    size = (400, 300),
    legend = :topright,
)
# p = plot()
# plot!(p, N, err, label = "Discretized")
plot!(p, N, err_bar, label = "Filtered-then-discretized")
plot!(p, N, err_allbar, label = "Discretized-then-filtered")
plot!(p, N, 1000N .^ -2, label = "\$1000 / N^2\$")
xlabel!(p, "N")
# title!(p, "Advection equation")
display(p)

savefig(p, "output/advection/varwidth_convergence.tikz")
