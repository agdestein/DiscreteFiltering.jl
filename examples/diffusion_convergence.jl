using DiscreteFiltering
using LinearAlgebra: norm
using Plots
using Symbolics
using Latexify

## Parameters
# Domain
a = 0.0
b = 1.0
# domain = PeriodicIntervalDomain(a, b)
domain = ClosedIntervalDomain(a, b)

# Time
T = 1.00

## Symbolics
@variables x t

# Exact solution (heat equation, Borggaard test case)
u = t + sin(2π * x) + sin(8π * x)
u_int = t * x - 1 / 2π * cos(2π * x) - 1 / 8π * cos(8π * x)

# Exact solution (heat equation, more complicated test case)
# u = 1 + sin(t) * (1 - 8 / 10 * x^2) + exp(-t) / 15 * sin(20π * x) + 1 / 5 * sin(10x)
# u_int =
#     x + sin(t) * (x - 8 / 30 * x^3) - exp(-t) / 15 / 20π * cos(20π * x) - 1 / 50 * cos(10x)

# Compute deduced quantities
dₜ = Differential(t)
dₓₓ = Differential(x)^2
f = expand_derivatives(dₜ(u) - dₓₓ(u))
g_b = substitute(u, Dict(x => b))
g_a = substitute(u, Dict(x => a))

# for sym ∈ [:u, :u_int, :f, :g_b, :g_a]
#     open("output/$sym.tex", "w") do io
#         @eval write($io, latexify($sym))
#     end
# end

u = eval(build_function(u, x, t))
u_int = eval(build_function(u_int, x, t))
f = eval(build_function(f, x, t))
g_a = eval(build_function(g_a, t))
g_b = eval(build_function(g_b, t))

## Ode solver tolerances
# tols = (;)
# tols = (; abstol = 1e-6, reltol = 1e-4)
tols = (; abstol = 1e-9, reltol = 1e-8)

# Number of mesh points
NN = floor.(Int, 10 .^ LinRange(1, 4, 7))

# Errors
err = zeros(length(NN))
err_bar = zeros(length(NN))
err_adbc = zeros(length(NN))

## Solve
@time for (i, N) ∈ enumerate(NN)
    M = N

    println("Solving for M = $M and N = $N")

    # Discretization
    x = discretize(domain, M)
    ξ = discretize(domain, N)
    Δx = (b - a) / N

    # h(x) = Δx / 2
    # filter = TopHatFilter(h)

    h₀ = 3.1Δx
    h(x) = h₀ # * (1 - 1 / 2 * cos(x))
    σ = Δx / 2
    filter = GaussianFilter(h, σ)

    # Equations
    equation = DiffusionEquation(domain, IdentityFilter(), f, g_a, g_b)
    equation_filtered = DiffusionEquation(domain, filter, f, g_a, g_b)

    u_use = ξ -> u(ξ, T)

    # Exact filtered solution
    ū = apply_filter(u_use, filter, domain)

    # Exact extended-filtered solution (for ADBC)
    ū_ext = apply_filter_extend(u_use, filter, domain)

    # Solve discretized problem
    # sol = solve(
    #     equation,
    #     ξ -> u(ξ, 0.0),
    #     (0.0, T),
    #     M,
    #     N;
    #     method = "discretizefirst",
    #     boundary_conditions = "derivative",
    #     tols...,
    # )

    # Solve discretized-then-filtered problem
    # sol_bar = solve(
    #     equation_filtered,
    #     ξ -> u(ξ, 0.0),
    #     (0.0, T),
    #     M,
    #     N;
    #     method = "discretizefirst",
    #     boundary_conditions = "derivative",
    #     tols...,
    # )

    # Solve filtered-then-discretized problem with ADBC
    ū_adbc = solve_adbc(equation_filtered, ξ -> u(ξ, 0.0), (0.0, T), M, T / 20_000_000)

    # Relative error
    u_exact = u.(ξ, T)
    ū_exact = ū.(x)
    ū_ext_exact = ū_ext.(x)
    # err[i] = norm(sol(T) - u_exact) / norm(u_ exact)
    # err_bar[i] = norm(sol_bar.u[end] - ū_exact) / norm(ū_exact)
    err_adbc[i] = norm(ū_adbc - ū_ext_exact) / norm(ū_ext_exact)
end

## Set GR backend for fast plotting
gr()

## Set PGFPlotsX plotting backend to export to tikz
pgfplotsx()

## Plot exact solution
ξ = LinRange(a, b, 101)
p = plot(xlabel = raw"$x$", size = (400, 300), legend = :topright)
for t ∈ LinRange(0.0, T, 5)
    plot!(p, ξ, u.(ξ, t), label = "\$t = $t\$")
end
display(p)
# savefig(p, "output/solution.tikz")

## Plot convergence
p = plot(
    xaxis = :log10,
    yaxis = :log10,
    # size = (400, 300),
    minorgrid = true,
    legend = :topright,
    # xlims = (NN[1], NN[end]),
    ylims = (1e-7, 1e0),
    xticks = 10 .^ (1:4)
)
# plot!(p, NN, err, label = "Discretized")
plot!(p, NN, err_adbc, marker = :c, label = "Filtered-then-discretized with ADBC")
plot!(p, NN, err_bar, marker = :d, label = "Discretized-then-filtered")
plot!(p, NN, 10 ./ NN .^ 2, linestyle = :dash, label = "\$10 / NN^2\$")
# plot!(p, NN, 20 ./ NN .^ 2, linestyle = :dash, label = raw"$20 N^{-2}$")
# plot!(p, NN, 10 ./ NN .^ 1.5, linestyle = :dash, label = raw"$10 N^{-3/2}}$")
xlabel!(p, raw"$NN$")
# title!(p, raw"Heat equation, $h(x) = \Delta x / 2$")
display(p)

# savefig(p, "output/diffusion/convergence.tikz")
