using DiscreteFiltering
using LinearAlgebra
using Plots
using ProgressLogging


## Parameters

# Domain
a = 0.0
b = 2π
domain = PeriodicIntervalDomain(a, b)

# Exact solutions (advection equation)
u(x, t) = sin(x - t) + 3 / 5 * cos(5(x - t)) # + 1 / 25 * sin(20(x - 1 - t))
u_int(x, t) = -cos(x - t) + 3 / 25 * sin(5(x - t)) # - 1 / 25 / 20 * cos(20(x - 1 - t))

## Time
T = 1.0

# Ode solver tolerances
# tols = (;)
# tols = (; abstol = 1e-6, reltol = 1e-4)
# tols = (; abstol = 1e-7, reltol = 1e-5)
tols = (; abstol = 1e-10, reltol = 1e-8)
degmax = 100
subspacedim = 1000
λ = 1e-7

# Number of mesh points
nrefine = 5
NN = floor.(Int, 10 .^ LinRange(2, 3, nrefine))
MM = floor.(Int, 4 // 5 .* NN)
λλ = [0, 1.37e-7, 6.7e-9, 2.58e-6, 1.6e-8]

# Errors
err = zeros(nrefine)
err_bar = zeros(nrefine)
err_allbar = zeros(nrefine)

## Solve
@time @progress for i = 1:nrefine
    # for λ ∈ LinRange(5e-8, 6e-7, 20)
    #for λ ∈ 10 .^ LinRange(-10, -6, 20)
    # λ = λλ[i]
    λ = 1e-8
    M = 100
    # M = MM[i]
    N = NN[i]

    # println("Solving for M = $M and N = $N")

    # Discretization
    x = discretize(domain, M)
    ξ = discretize(domain, N)
    Δx = (b - a) / M

    # Filter
    h₀ = 1.0Δx
    # h₀ = Δx / 2
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
    #     tols...,
    #     degmax,
    #     λ,
    # )

    # Solve filtered-then-discretized problem
    sol_bar = solve(
        equation_filtered,
        ξ -> u(ξ, 0.0),
        (0.0, T),
        M,
        N;
        method = "filterfirst",
        subspacedim,
        tols...,
        degmax,
        λ,
    )

    # Solve discretized-then-filtered problem
    # @enter sol_allbar = solve(
    sol_allbar = solve(
        equation_filtered,
        ξ -> u(ξ, 0.0),
        (0.0, T),
        M,
        N;
        method = "discretizefirst",
        subspacedim,
        tols...,
        degmax,
        λ,
    )

    # Exact filtered solution
    ū = apply_filter_int(ξ -> u_int(ξ, T), filter, domain)

    # Relative error
    u_exact = u.(ξ, T)
    ū_exact = ū.(x)
    # err[i] = norm(sol(T) - u_exact) / norm(u_exact)
    err_bar[i] = norm(sol_bar(T) - ū.(ξ)) / norm(ū.(ξ))
    err_allbar[i] = norm(sol_allbar(T) - ū_exact) / norm(ū_exact)
    println("λ = $λ, err = $(err_allbar[i])")
    # end
end

##
err_bar
err_allbar

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
    legend = :bottomleft,
    ylims = (1e-3, 1e-1),
    # xticks = 10 .^ (1:4),
)
# p = plot()
# plot!(p, NN, err, label = "Discretized")
plot!(p, NN, err_bar, marker = :c, label = "Filtered-then-discretized")
plot!(p, NN, err_allbar, marker = :d, label = "Discretized-then-filtered")
# plot!(p, NN, 1000N .^ -2, label = "\$1000 / N^2\$")
xlabel!(p, "N")
title!(p, "M = $M, \$h = \\Delta x_M(1 - 1 / 2 \\cos(x))\$")
display(p)
# savefig(p, "output/advection/varwidth_convergence.tikz")
# savefig(p, "output/advection/M_constant_N_varying.tikz")
