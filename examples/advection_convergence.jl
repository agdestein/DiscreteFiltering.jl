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
λ = 1e-4

# Number of mesh points
NN = floor.(Int, 10 .^ LinRange(2, 4, 20))

# Errors
err = zeros(length(NN))
err_bar = zeros(length(NN))
err_allbar = zeros(length(NN))

## Solve
@time @progress for (i, N) ∈ enumerate(NN)
    M = N

    println("Solving for M = $M and N = $N")

    # Discretization
    x = discretize(domain, M)
    ξ = discretize(domain, N)
    Δx = (b - a) / N

    # Filter
    # h(x) = Δx / 2
    # h₀ = 2.1Δx
    h₀ = (b - a) / 200
    h(x) = h₀ * (1 - 1 / 2 * cos(x))
    filter = TopHatFilter(h)
    # filter = ConvolutionalFilter(x -> 2h(x), x -> abs(x) ≤ h(x) ? 1/ 2h(x) : zero(x))

    # Equations
    equation = AdvectionEquation(domain, IdentityFilter())
    equation_filtered = AdvectionEquation(domain, filter)

    # Solve discretized problem
    # sol = solve(
    #     equation,
    #     ξ -> u(ξ, 0.0),
    #     (0.0, T),
    #     M,
    #     N;
    #     method = "discretizefirst",
    #     subspacedim,
    #     λ,
    # )

    # Solve filtered-then-discretized problem
    # sol_bar = solve(
    #     equation_filtered,
    #     ξ -> u(ξ, 0.0),
    #     (0.0, T),
    #     M,
    #     N;
    #     method = "filterfirst",
    #     subspacedim,
    #     λ,
    # )

    # Solve discretized-then-filtered problem
    sol_allbar = solve(
        equation_filtered,
        ξ -> u(ξ, 0.0),
        (0.0, T),
        M,
        N;
        method = "discretizefirst",
        reltol = 1e-8,
        abstol = 1e-10,
        λ,
    )

    # Exact filtered solution
    ū = apply_filter_int(ξ -> u_int(ξ, T), filter, domain)

    # Relative error
    u_exact = u.(ξ, T)
    ū_exact = ū.(x)
    # err[i] = norm(sol(T) - u_exact) / norm(u_exact)
    # err_bar[i] = norm(sol_bar(T) - ū_exact) / norm(ū_exact)
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
    # size = (400, 300),
    legend = :topright,
)
# p = plot()
# plot!(p, NN, err, label = "Discretized")
plot!(p, NN, err_bar, label = "Filtered-then-discretized")
plot!(p, NN, err_allbar, label = "Discretized-then-filtered")
plot!(p, NN, 1000N .^ -2, label = "\$1000 / NN^2\$")
xlabel!(p, "NN")
# title!(p, "Advection equation")
display(p)

# savefig(p, "output/advection/varwidth_convergence.tikz")
