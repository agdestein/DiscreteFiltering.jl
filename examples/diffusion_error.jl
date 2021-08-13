using DiscreteFiltering
using LinearAlgebra: norm
using Plots
using Symbolics
using Latexify

## Parameters
# Domain
a = 0.0
b = 1.0
domain = ClosedIntervalDomain(a, b)

# Time
T = 1#0.05

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
n = 500
x = discretize(domain, n)
Δx = (b - a) / n

# Discretization
# h(x) = Δx / 2
# filter = TopHatFilter(h)

h₀ = 3.1Δx
h(x) = h₀ # * (1 - 1 / 2 * cos(x))
σ = Δx / 2
filter = GaussianFilter(h, σ)
if filter isa TopHatFilter
    u_use = u_int
else
    u_use = u
end

# Filter matrix
W = filter_matrix(filter, domain, n)

# Equations
equation = DiffusionEquation(domain, IdentityFilter(), f, g_a, g_b)
equation_filtered = DiffusionEquation(domain, filter, f, g_a, g_b)

# Exact filtered solution
ū = (x, t) -> apply_filter(x -> u_use(x, t), filter, domain)(x)
ū_ext = (x, t) -> apply_filter_extend(x -> u_use(x, t), filter, domain)(x)

## Solve discretized problem
# sol = solve(equation, x -> u(x, 0.0), (0.0, T), n; method = "discretizefirst", tols...)

## Solve discretized-then-filtered problem
sol_bar = solve(
    equation_filtered,
    x -> u(x, 0.0),
    (0.0, T),
    n;
    method = "discretizefirst",
    boundary_conditions = "derivative",
    tols...,
)

## Solve filtered-then-discretized problem with ADBC
ū_adbc = solve_adbc(equation_filtered, x -> u(x, 0.0), (0.0, T), n, T / 100_000)

## Relative error
u_exact = u.(x, T)
ū_ext_exact = ū_ext.(x, T)
# err = t -> norm(sol(t) - u.(x, t)) /maximum(abs.(u.(x, t)))
err_bar = t -> abs.(sol_bar(t) - ū.(x, t)) / maximum(abs.(ū.(x, t)))
# err_adbc = norm(ū_adbc - ū_ext_exact) / maximum(abs.(ū_ext_exact))


## Set GR backend for fast plotting
gr()

## Set PGFPlotsX plotting backend to export to tikz
pgfplotsx()

## Plot exact solution
p = plot(xlabel = raw"$x$", size = (400, 300), legend = :topright)
for t ∈ LinRange(0.0, T, 2)
    plot!(p, LinRange(0, 1, 101), x -> u.(x, t), label = "\$t = $t\$")
end
display(p)
savefig(p, "output/diffusion/solution.tikz")

## Plot error
p = plot(xlabel = raw"$x$", size = (400, 300), legend = :topright)
plot!(p, x, err_bar(T), label = "Discretized-then-filtered")
display(p)
# savefig(p, "output/diffusion/error.tikz")

## Initial error
p = plot(xlabel = raw"$x$", size = (400, 300), legend = :topright)
plot!(
    p,
    x,
    abs.(W * u.(x, 0) .- ū.(x, 0)) / maximum(abs.(ū.(x, 0))),
    label = "Discretized-then-filtered",
)
display(p)
# savefig(p, "output/solution.tikz")

## Time plot
tspace = LinRange(0, T, 100)
# p = plot(xlabel = raw"$x$", size = (400, 300), legend = :topright)
plot(
    tspace,
    x,
    mapreduce(err_bar, hcat, tspace),
    st = :surface,
    label = "Discretized-then-filtered",
)
# display(p)
# savefig(p, "output/solution.tikz")
