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


## Symbolics
@variables x t

# Exact solution (heat equation, Borggaard test case)
u = t + sin(2π * x) + sin(8π * x)
u_int = t * x - 1 / 2π * cos(2π * x) - 1 / 8π * cos(8π * x)

# Exact solution (heat equation, more complicated test case)
# u = 1 + sin(t) * (1 - 8 / 10 * x^2) + exp(-t) / 15 * sin(20π * x) + 1 / 5 * sin(10x)

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
f = eval(build_function(f, x, t))
g_a = eval(build_function(g_a, t))
g_b = eval(build_function(g_b, t))

## Ode solver tolerances
# tols = (;)
# tols = (; abstol = 1e-6, reltol = 1e-4)
tols = (; abstol = 1e-9, reltol = 1e-8)

n = 20

## Solve

# Time
T = 0.05
nₜ = 500
tspace = LinRange(0, T, nₜ + 1)
tlist = (0.0, T)

# Discretization
x = discretize(domain, n)
Δx = (b - a) / n

# h(x) = Δx / 2
# filter = TopHatFilter(h)

σ = 2 / √3 * Δx
h₀ = 5σ
h(x) = h₀ # * (1 - 1 / 2 * cos(x))
filter = GaussianFilter(h, σ)

# Equations
equation = DiffusionEquation(domain, IdentityFilter(), f, g_a, g_b)
equation_filtered = DiffusionEquation(domain, filter, f, g_a, g_b)

u₀ = x -> u(x, 0.0)
uₜ = x -> u(x, T)

# Exact filtered solution
ū = apply_filter(uₜ, filter, domain)

# Exact extended-filtered solution (for ADBC)
ū_ext = t -> apply_filter_extend(x -> u(x, t), filter, domain)

# Solve discretized problem
sol = solve(
    equation,
    u₀,
    tlist,
    n;
    method = "discretizefirst",
    boundary_conditions = "derivative",
    tols...,
)

# Solve discretized-then-filtered problem
sol_bar = solve(
    equation_filtered,
    u₀,
    tlist,
    n;
    method = "discretizefirst",
    boundary_conditions = "derivative",
    tols...,
)

# Solve filtered-then-discretized problem with ADBC
ū_adbc = solve_adbc(equation_filtered, u₀, tlist, n, T / nₜ)

# Relative error
u_exact = u.(x, T)
ū_exact = ū.(x)
ū_ext_exact = t -> ū_ext(t).(x)
err = (sol.u[end] .- u_exact)
err_bar = (sol_bar.u[end] .- ū_exact)
err_adbc = (ū_adbc .- ū_ext_exact)

## Set GR backend for fast plotting
gr()

## Set PGFPlotsX plotting backend to export to tikz
pgfplotsx()

##
plotly()

## Plot error
p = plot(
    xlabel = raw"$x$",
    # size = (400, 300),
    legend = :topright,
)
plot!(p, x, err, label = "Discretized")
plot!(p, x, err_bar, label = "Discretized-then-filtered")
plot!(p, x, err_adbc, label = "ADBC")
display(p)
# savefig(p, "output/diffusion/error$n.tikz")

## Plot exact solution
p = plot(xlabel = raw"$x$", size = (400, 300), legend = :topright)
for t ∈ LinRange(0.0, T, 2)
    plot!(p, LinRange(0, 1, 101), x -> u.(x, t), label = "\$t = $t\$")
end
display(p)
# savefig(p, "output/diffusion/solution.tikz")

## Time plot
plot(
    tspace,
    x,
    err_adbc,
    st = :surface,
    # label = "Discretized-then-filtered",
    label = "ADBC",
)
