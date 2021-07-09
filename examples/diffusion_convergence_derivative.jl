using DiscreteFiltering
using OrdinaryDiffEq: ODEProblem, ODEFunction, solve, QNDF
using LinearAlgebra
using Plots
using SparseArrays
using Symbolics
using Latexify

## Parameters
# Domain
a = 0.0
b = 1.0
# domain = PeriodicIntervalDomain(a, b)
domain = ClosedIntervalDomain(a, b)

# Time
T = 1.0

## Symbolics
@variables x t

# Exact solution (heat equation, Borggaard test case)
# u = t + sin(2π * x) + sin(8π * x)
# u_int = t * x - 1 / 2π * cos(2π * x) - 1 / 8π * cos(8π * x)

# Exact solution (heat equation, more complicated test case)
u = 1 + sin(t) * (1 - 8 / 10 * x^2) + exp(-t) / 15 * sin(20π * x) + 1 / 5 * sin(10x)
u_int =
    x + sin(t) * (x - 8 / 30 * x^3) - exp(-t) / 15 / 20π * cos(20π * x) - 1 / 50 * cos(10x)

# Compute deduced quantities
dₜ = Differential(t)
dₓₓ = Differential(x)^2
f = expand_derivatives(dₜ(u) - dₓₓ(u))
g_b = substitute(u, Dict(x => b))
g_a = substitute(u, Dict(x => a))
dg_a = expand_derivatives(dₜ(g_a))
dg_b = expand_derivatives(dₜ(g_b))

for sym ∈ [:u, :u_int, :f, :g_b, :g_a, :dg_a, :dg_b]
    open("output/$sym.tex", "w") do io
        @eval write($io, latexify($sym))
    end
end

u = eval(build_function(u, x, t))
u_int = eval(build_function(u_int, x, t))
f = eval(build_function(f, x, t))
g_a = eval(build_function(g_a, t))
g_b = eval(build_function(g_b, t))
dg_a = eval(build_function(dg_a, t))
dg_b = eval(build_function(dg_b, t))

## Ode solver tolerances
# tols = (;)
# tols = (; abstol = 1e-6, reltol = 1e-4)
tols = (; abstol = 1e-9, reltol = 1e-8)

# Number of mesh points
N = floor.(Int, 10 .^ LinRange(1, 4, 20))

# Errors
err = zeros(length(N))
err_allbar = zeros(length(N))

## Solve
@time for (i, n) ∈ enumerate(N)

    println("Solving for n = $n")

    ## Discretization
    x = discretize(domain, n)
    Δx = (b - a) / n

    ## Filter
    h(x) = Δx / 2
    fil = TopHatFilter(h)

    ## Get matrices
    D = diffusion_matrix(domain, n)
    # W = filter_matrix(fil, domain, n)
    # R = inverse_filter_matrix(fil, domain, n)
    W = filter_matrix_meshwidth(fil, domain, n)
    R = inverse_filter_matrix_meshwidth(fil, domain, n)
    W₀ = W[:, 1]
    Wₙ = W[:, end]

    # Zero out boundary u
    D[[1, end], :] .= 0
    dropzeros!(D)

    ## Exact filtered solution
    ū(x, t) = begin
        α = max(a, x - h(x))
        β = min(b, x + h(x))
        1 / (β - α) * (u_int(β, t) - u_int(α, t))
    end

    ## Discrete initial conditions
    uₕ = u.(x, 0.0)
    ūₕ = ū.(x, 0.0)

    ## Solve discretized problem
    function fₕ!(fₕ, t)
        fₕ[2:end-1] .= f.(x[2:end-1], t)
    end
    fₕ = zeros(n + 1)
    p = (copy(uₕ), D, fₕ)
    function du!(du, u, p, t)
        # @show t
        y, D, f = p
        fₕ!(f, t)
        mul!(y, D, u)
        @. du = y + f
        du[1] += dg_a(t)
        du[end] += dg_b(t)
    end
    prob = ODEProblem(
        ODEFunction(du!, jac = (J, u, p, t) -> (J .= D), jac_prototype = D),
        uₕ,
        (0, T),
        p,
    )
    sol = solve(prob, QNDF(); tols...)

    ## Solve discretized-and-then-filtered problem
    D̄ = W * D * R
    fₕ_bar = W * fₕ
    p̄ = (copy(uₕ), D̄, fₕ, fₕ_bar)
    function du_bar!(du, u, p, t)
        # @show t
        y, D, f, Wf = p
        fₕ!(f, t)
        mul!(Wf, W, f)
        mul!(y, D, u)
        @. du = y + Wf + dg_a(t) * W₀ + dg_b(t) * Wₙ
    end
    prob_allbar = ODEProblem(
        ODEFunction(
            du_bar!,
            jac = (J, u, p, t) -> (J .= D̄),
            jac_prototype = D̄,
            mass_matrix = W * R,
        ),
        W * uₕ,
        (0, T),
        p̄,
    )
    sol_allbar = solve(prob_allbar, QNDF(); tols...)

    ## Relative error
    u_exact = u.(x, T)
    ū_exact = ū.(x, T)
    err[i] = norm(sol(T) - u_exact) / norm(u_exact)
    err_allbar[i] = norm(sol_allbar(T) - ū_exact) / norm(ū_exact)
end


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
plot!(p, N, err, label = "Discretized")
plot!(p, N, err_allbar, label = "Discretized-then-filtered")
# plot!(p, N, 1 ./ N .^ 2)
plot!(p, N, 20 ./ N .^ 2, linestyle = :dash, label = raw"$20 n^{-2}$")
# plot!(p, N, 10 ./ N .^ 1.5, linestyle = :dash, label = raw"$10 n^{-3/2}}$")
xlabel!(p, raw"$n$")
# title!(p, raw"Heat equation, $h(x) = \Delta x / 2$")

savefig(p, "output/convergence_derivative.tikz")
