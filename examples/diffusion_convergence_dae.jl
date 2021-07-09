using DiscreteFiltering
using OrdinaryDiffEq
using LinearAlgebra
using Plots
using SparseArrays


## Parameters
# Domain
a = 0.0
b = 1.0
# domain = PeriodicIntervalDomain(a, b)
domain = ClosedIntervalDomain(a, b)

# Time
T = 1.0

# Exact solutions (heat equation)
u(x, t) = t + sin(2π * x) + sin(8π * x)
u_int(x, t) = t * x - 1 / 2π * cos(2π * x) + -1 / 8π * cos(8π * x)

# Forcing term
f(x) = 1 + 4π^2 * sin(2π * x) + 64π^2 * sin(8π * x)

# Ode solver tolerances
# tols = (;)
# tols = (; abstol = 1e-6, reltol = 1e-4)
tols = (; abstol = 1e-6, reltol = 1e-5)

# Number of mesh points
# N = floor.(Int, 10 .^ LinRange(1, 4, 20))
N = floor.(Int, 10 .^ LinRange(1, 4, 10))

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
    fil = TopHatFilter(h)

    ## Get matrices
    C = advection_matrix(domain, n)
    D = diffusion_matrix(domain, n)
    # W = filter_matrix(fil, domain, n)
    # R = inverse_filter_matrix(fil, domain, n)
    W = filter_matrix_meshwidth(fil, domain, n)
    R = inverse_filter_matrix_meshwidth(fil, domain, n)
    A = spdiagm(α.(x))

    ## Exact filtered solution
    function ū(x, t)
        α = max(a, x - h(x))
        β = min(b, x + h(x))
        1 / (β - α) * (u_int(β, t) - u_int(α, t))
    end
    ## Discrete initial conditions
    uₕ = u.(x, 0.0)

    ## Solve discretized problem
    M̃ = [
        spzeros(1, n + 1)
        spzeros(n - 1, 1) sparse(I, n - 1, n - 1) spzeros(n - 1, 1)
        spzeros(1, n + 1)
    ]
    J̃ = [
        1 spzeros(1, n)
        D[2:end-1, :]
        spzeros(1, n) 1
    ]
    f̃ = [0; f.(x[2:end-1]); 0]
    γ̃ = zeros(n + 1)
    γ̃[[1, end]] .= 1
    pₕ = (copy(uₕ), J̃, f̃, γ̃)
    function duₕ!(du, u, p, t)
        y, J, f, γ = p
        mul!(y, J, u)
        @. du = y + f - t * γ
    end
    Jacₕ!(J, uₕ, p, t) = (J .= J̃)
    prob = ODEProblem(
        ODEFunction(duₕ!, jac = Jacₕ!, jac_prototype = J̃, mass_matrix = M̃),
        uₕ,
        (0, T),
        pₕ,
    )
    sol = solve(prob, Rodas5(autodiff = false); tols...)

    ## Solve discretized-and-then-filtered problem
    J̃_allbar = W * J̃ * R
    f̃_allbar = W * f̃
    γ̃_allbar = W * γ̃
    pₕ_allbar = (copy(uₕ), J̃_allbar, f̃_allbar, γ̃_allbar)
    function duₕ_allbar!(du, u, p, t)
        y, J, f, γ = p
        mul!(y, J, u)
        @. du = y + f - t * γ
    end
    Jac_allbar!(J, uₕ, p, t) = (J .= J̃_allbar)
    prob_allbar = ODEProblem(
        ODEFunction(
            duₕ_allbar!,
            jac = Jac_allbar!,
            jac_prototype = J̃_allbar,
            mass_matrix = W * M̃ * R,
        ),
        W * uₕ,
        (0, T),
        pₕ_allbar,
    )
    sol_allbar = solve(prob_allbar, Rodas5(autodiff = false); tols...)

    ## Relative error
    u_exact = u.(x, T)
    ū_exact = ū.(x, T)
    err[i] = norm(sol(T) - u_exact) / norm(u_exact)
    err_allbar[i] = norm(sol_allbar(T) - ū_exact) / norm(ū_exact)
end


## Set GR plotting backend
gr()

## Set PGFPlotsX plotting backend to export to tikz
pgfplotsx()

## Plot exact solution
x = LinRange(0, 1, 500)
p = plot(xlabel = raw"$x$", size = (400, 300), legend = :topright)
plot!(p, x, x -> u(x, 0), label = raw"$u_0(x)$")
plot!(p, x, x -> u(x, T), label = raw"$u(x, T)$")
display(p)
savefig(p, "output/heat_solution.tikz")

## Plot convergence
p = plot(xaxis = :log, yaxis = :log, size = (400, 300), legend = :topright)
plot!(p, N, err, label = "Discretized")
# plot!(p, N, err_bar, label = "Filtered-then-discretized")
plot!(p, N, err_allbar, label = "Discretized-then-filtered")
# plot!(p, N, 1 ./ N .^ 2)
# plot!(p, N, 1 ./ N .^ 1)
xlabel!(p, raw"$n$")
title!(p, raw"Heat equation, $h(x) = \Delta x / 2$")
display(p)
savefig(p, "output/heat_convergence_dae.tikz")
