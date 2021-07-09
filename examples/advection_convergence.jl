using DiscreteFiltering
using OrdinaryDiffEq
using LinearAlgebra
using Plots
using SparseArrays


## Parameters
# Domain
a = 0.0
b = 2π
domain = PeriodicIntervalDomain(a, b)
# domain = ClosedIntervalDomain(a, b)

# Time
T = 1.0

# Exact solutions (advection equation)
u(x, t) = sin(x - t) + 0.6cos(5(x - t)) + 0.04sin(20(x - 1 - t))
u_int(x, t) = -cos(x - t) + 0.6 / 5 * sin(5(x - t)) - 0.04 / 20 * cos(20(x - 1 - t))

# Ode solver tolerances
# tols = (;)
# tols = (; abstol = 1e-6, reltol = 1e-4)
# tols = (; abstol = 1e-7, reltol = 1e-5)
subspacedim = 200

# Number of mesh points
N = floor.(Int, 10 .^ LinRange(1, 5, 50))
# N = [100]

# Errors
err = zeros(length(N))
err_bar = zeros(length(N))
err_allbar = zeros(length(N))

## Solve
@time for (i, n) ∈ enumerate(N)

    println("Solving for n = $n")

    ## Discretization
    x = discretize_uniform(domain, n)
    Δx = (b - a) / n

    ## Filter
    h₀ = Δx
    h(x) = h₀ / 2
    dh(x) = 0.0
    # h(x) = h₀ * (1 - 1 / 2 * cos(x))
    # dh(x) = h₀ / 2 * sin(x)
    α(x) = 1 / 3 * dh(x) * h(x)
    f = TopHatFilter(h)

    ## Get matrices
    C = advection_matrix(domain, n)
    D = diffusion_matrix(domain, n)
    # W = filter_matrix(f, domain, n)
    # R = inverse_filter_matrix(f, domain, n)
    W = filter_matrix_meshwidth(f, domain, n)
    R = inverse_filter_matrix_meshwidth(f, domain, n)
    A = spdiagm(α.(x))

    ## Exact filtered solution
    ū(x, t) = 1 / 2h(x) * (u_int(x + h(x), t) - u_int(x - h(x), t))

    ## Discrete initial conditions
    uₕ = u.(x, 0.0)
    ūₕ = ū.(x, 0.0)
    uₕ_allbar = W * uₕ

    ## Solve discretized problem
    J = DiffEqArrayOperator(-C)
    prob = ODEProblem(J, uₕ, (0, T))
    sol = solve(prob, LinearExponential(krylov = :simple, m = subspacedim))

    ## Solve filtered-and-then-discretized problem
    J_bar = DiffEqArrayOperator(-C + A * D)
    prob_bar = ODEProblem(J_bar, ūₕ, (0, T))
    sol_bar = solve(prob_bar, LinearExponential(krylov = :simple, m = subspacedim))

    ## Solve discretized-and-then-filtered problem
    J_allbar = DiffEqArrayOperator(-W * C * R)
    prob_allbar = ODEProblem(J_allbar, W * uₕ, (0, T))
    sol_allbar = solve(prob_allbar, LinearExponential(krylov = :simple, m = subspacedim))

    ## Relative error
    u_exact = u.(x, T)
    err[i] = norm(sol(T) - u_exact) / norm(u_exact)
    err_bar[i] = norm(sol_bar(T) - u_exact) / norm(u_exact)
    err_allbar[i] = norm(sol_allbar(T) - u_exact) / norm(u_exact)
end


## Set PGFPlotsX plotting backend to export to tikz
pgfplotsx()

##
p = plot(xaxis = :log, yaxis = :log, size = (400, 300), legend = :topright)
plot!(p, N, err, label = "Discretized")
plot!(p, N, err_bar, label = "Filtered-then-discretized")
plot!(p, N, err_allbar, label = "Discretized-then-filtered")
xlabel!(p, "n")
title!(p, "Advection equation, \$h(x) = \\Delta x / 2\$")
display(p)

savefig(p, "output/advection_convergence.tikz")
