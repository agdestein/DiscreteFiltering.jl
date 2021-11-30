using DiscreteFiltering
using LinearAlgebra
using Plots
using ProgressLogging
using OrdinaryDiffEq: RK4


## Parameters

# Domain
a = 0.0
b = 2π
domain = PeriodicIntervalDomain(a, b)

# Exact solutions (advection equation)
u(x, t) = sin(x - t) + 3 / 5 * cos(5(x - t)) + 7 / 25 * sin(20(x - 1 - t))
u_int(x, t) = -cos(x - t) + 3 / 25 * sin(5(x - t)) - 7 / 25 / 20 * cos(20(x - 1 - t))

## Time
T = 1.0

# Ode solver tolerances
# tols = (;)
# tols = (; abstol = 1e-6, reltol = 1e-4)
# tols = (; abstol = 1e-7, reltol = 1e-5)
tols = (; abstol = 1e-10, reltol = 1e-8)
degmax = 1000
subspacedim = 1000
λ = 1e-8

# Amplitudes
c = [
    [1],
    [1, 3],
    [1, 3 / 5, 1 / 25],
    rand(5),
]

# Frequencies
ω = [
    [0],
    [2, 3],
    [1, 5, 20],
    (rand(1:10, 5)),
]

# Phase-shifts
ϕ = [
    [π / 2],
    [0, 0],
    [0, 0, 20],
    2π * rand(5),
]

# Number of time steps for fitting C̄
nₜ = 500
t = LinRange(0, T, nₜ)
UU = sum_of_sines.([domain], c, ω, ϕ) 
u₀_list = [UU[1] for UU ∈ UU] 
U₀_list = [UU[2] for UU ∈ UU] 

# Number of mesh points
nrefine = 10
MM = floor.(Int, 10 .^ LinRange(1, 4, nrefine))
NN = MM
# NN = floor.(Int, 5 // 4 .* MM)
# λλ = [0, 0, 0, 0, 1e-1]

# Errors
err = zeros(nrefine)
err_bar = zeros(nrefine)
err_allbar = zeros(nrefine)

## Solve
@time for i = 1:nrefine
    # λ = λλ[i]
    λ = 1e-8
    λ_ridge = 1e-8
    # M = 100
    M = MM[i]
    # N = NN[i]
    N = 1000 # NN[i]
    # M = N

    println("Solving for M = $M and N = $N")
    # Discretization
    x = discretize(domain, M)
    ξ = discretize(domain, N)
    Δx = (b - a) / M

    # Filter
    h₀ = 1.0Δx
    # h₀ = (b - a) / 200
    # h₀ = Δx / 2
    h(x) = h₀ * (1 - 1 / 2 * cos(x))
    # h(x) = 0.49999Δx
    filter = TopHatFilter(h)
    # filter = ConvolutionalFilter(x -> 2h(x), x -> abs(x) ≤ h(x) ? 1/ 2h(x) : zero(x))
    # h(x) = 2h₀
    # σ = h₀/2
    # filter = GaussianFilter(h, σ)

    C̄ = fit_Cbar(domain, filter, u₀_list, U₀_list, M, t; λ, method = :ridge)

    # Equations
    equation = AdvectionEquation(domain, IdentityFilter())
    equation_filtered = AdvectionEquation(domain, filter)

    # Solve discretized problem
    sol = solve(
        equation,
        ξ -> u(ξ, 0.0),
        (0.0, T),
        M,
        M;
        method = "discretizefirst",
        subspacedim,
        tols...,
        degmax,
        λ,
    )

    # Solve filtered-then-discretized problem
    # sol_bar = solve(
    #     equation_filtered,
    #     ξ -> u(ξ, 0.0),
    #     (0.0, T),
    #     M,
    #     M;
    #     method = "filterfirst",
    #     subspacedim,
    #     tols...,
    #     degmax,
    #     λ,
    # )

    # Solve discretized-then-filtered problem
    sol_allbar = solve(
        equation_filtered,
        ξ -> u(ξ, 0.0),
        (0.0, T),
        M,
        N;
        # method = "discretizefirst",
        # method = "discretizefirst-without-R",
        method = "discretizefirst-fit-Cbar",
        # solver = RK4(),
        subspacedim,
        tols...,
        degmax,
        λ,
        λ_ridge,
        C̄,
    )

    # Exact filtered solution
    ū = apply_filter_int(ξ -> u_int(ξ, T), filter, domain)
    # ū = apply_filter(ξ -> u(ξ, T), filter, domain)

    # Relative error
    # u_exact = u.(ξ, T)
    u_exact = u.(x, T)
    ū_exact = ū.(x)
    # err[i] = norm(sol(T) - u_exact) / norm(u_exact)
    err[i] = norm(sol(T) - ū_exact) / norm(ū_exact)
    # err_bar[i] = norm(sol_bar(T) - ū.(ξ)) / norm(ū.(ξ))
    # err_bar[i] = norm(sol_bar(T) - ū.(x)) / norm(ū.(x))
    err_allbar[i] = norm(sol_allbar(T) - ū_exact) / norm(ū_exact)
    # println("λ = $λ, err = $(err_allbar[i])")
    println("λ = $λ, err_bar = $(err_bar[i]), err_allbar = $(err_allbar[i])")
end

## Set PGFPlotsX plotting backend to export to tikz
pgfplotsx()

## In terminal plots
unicodeplots()

## Default backend
gr()

##
p = plot(
    xaxis = :log10,
    yaxis = :log10,
    minorgrid = true,
    # size = (400, 300),
    legend = :bottomleft,
    # xlims = (10, 10^4),
    ylims = (1e-6, 1e-0),
    xticks = 10 .^ (1:4),
    xlabel = "M",
    # title = "N = 1000"
);
plot!(p, MM, err_allbar; label = "Data driven Cbar, HF", marker = :d, color = 2);
# plot!(p, MM, err_allbar_lf; label = "Data driven Cbar, LF", marker = :d, color = 1);
plot!(p, MM, err; label = "Without filter, HF", marker = :c, linestyle = :dash, color = 2);
# plot!(p, MM, err_lf; label = "Without filter, LF", marker = :c, linestyle = :dash, color = 1);
# display(p)
# output = "output/advection/Cbar.tikz"
output = "output/advection/Cbar.pdf"
savefig(p, output)

