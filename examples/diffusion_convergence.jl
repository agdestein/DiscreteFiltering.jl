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
T = 0.05

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
N = floor.(Int, 10 .^ LinRange(2, 4, 5))

# Errors
err = zeros(length(N))
err_bar = zeros(length(N))
err_adbc = zeros(length(N))

## Solve
@time for (i, n) ∈ enumerate(N)

    println("Solving for n = $n")

    # Discretization
    x = discretize(domain, n)
    Δx = (b - a) / n

    # h(x) = Δx / 2
    # filter = TopHatFilter(h)

    h₀ = 3.1Δx
    h(x) = h₀ # * (1 - 1 / 2 * cos(x))
    σ = Δx / 2
    filter = GaussianFilter(h, σ)

    # Filter matrix
    W = filter_matrix(filter, domain, n)

    # Equations
    equation = DiffusionEquation(domain, IdentityFilter(), f, g_a, g_b)
    equation_filtered = DiffusionEquation(domain, filter, f, g_a, g_b)

    if filter isa TopHatFilter
        u_use = x -> u_int(x, T)
    else
        u_use = x -> u(x, T)
    end

    # Exact filtered solution
    ū = apply_filter(u_use, filter, domain)

    # Exact extended-filtered solution (for ADBC)
    ū_ext = apply_filter_extend(u_use, filter, domain)

    # Solve discretized problem
    # sol = solve(equation, x -> u(x, 0.0), (0.0, T), n; method = "discretizefirst", tols...)

    # Solve discretized-then-filtered problem
    sol_bar = solve(
        equation_filtered,
        x -> u(x, 0.0),
        (0.0, T),
        n;
        method = "discretizefirst",
        boundary_conditions = "derivative",
        tols...,
    )

    # Solve filtered-then-discretized problem with ADBC
    ū_adbc = solve_adbc(equation_filtered, x -> u(x, 0.0), (0.0, T), n, T / 100_000)

    # Relative error
    u_exact = u.(x, T)
    ū_exact = ū.(x)
    ū_ext_exact = ū_ext.(x)
    # err[i] = norm(sol(T) - u_exact) / norm(u_exact)
    err_bar[i] = norm(sol_bar(T) - ū_exact) / norm(ū_exact)
    err_adbc[i] = norm(ū_adbc - ū_ext_exact) / norm(ū_ext_exact)
end

## Set GR backend for fast plotting
gr()

## Set PGFPlotsX plotting backend to export to tikz
pgfplotsx()

## Plot exact solution
x = LinRange(0, 1, 101)
p = plot(xlabel = raw"$x$", size = (400, 300), legend = :topright)
for t ∈ LinRange(0.0, T, 5)
    plot!(p, x, u.(x, t), label = "\$t = $t\$")
end
display(p)
savefig(p, "output/solution.tikz")

## Plot convergence
p = plot(xaxis = :log, yaxis = :log, size = (400, 300), legend = :topright)
# plot!(p, N, err, label = "Discretized")
plot!(p, N, err_bar, label = "Discretized-then-filtered")
plot!(p, N, err_adbc, label = "Filtered-then-discretized with ADBC")
# plot!(p, N, 1 ./ N .^ 2)
# plot!(p, N, 20 ./ N .^ 2, linestyle = :dash, label = raw"$20 n^{-2}$")
# plot!(p, N, 10 ./ N .^ 1.5, linestyle = :dash, label = raw"$10 n^{-3/2}}$")
xlabel!(p, raw"$n$")
# title!(p, raw"Heat equation, $h(x) = \Delta x / 2$")
display(p)

# savefig(p, "output/convergence_adbc.tikz")
