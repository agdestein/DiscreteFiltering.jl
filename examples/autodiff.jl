# LSP indexing solution from
# https://github.com/julia-vscode/julia-vscode/issues/800#issuecomment-650085983
if isdefined(@__MODULE__, :LanguageServer)
    include("../src/DiscreteFiltering.jl")
    using .DiscreteFiltering
end

using DiscreteFiltering
using LinearAlgebra
using Plots
using OrdinaryDiffEq: OrdinaryDiffEq, ODEFunction, ODEProblem, Tsit5
using DiffEqFlux: DiffEqFlux, ADAM

## Domain
a = 0.0
b = 2π
domain = PeriodicIntervalDomain(a, b)

## Time
T = 1.0

# Discretization
M = 100
N = 1000
x = discretize(domain, M)
ξ = discretize(domain, N)
Δx = (b - a) / M

# Filter
# h₀ = 1.0Δx
h₀ = 0.2
# h₀ = (b - a) / 100
# h₀ = Δx / 2
h(x) = h₀ * (1 - 1 / 2 * cos(x))
filter = TopHatFilter(h)

# DNS operator
C = advection_matrix(domain, N)

# Initial guess for LES operator: Unfiltered
C̄ = Matrix(advection_matrix(domain, M))

# ODE function for given operator and IC
# f!(du, u, p, t) = mul!(du, p, u)
f(u, p, t) = -p * u
odefunction = ODEFunction(f)
function 𝓢(C, u)
    problem = ODEProblem(odefunction, u, (0.0, T), C)
    solution = OrdinaryDiffEq.solve(problem, Tsit5(); save_everystep = false)
    solution.u[end]
end

# Create signal
function create_data(K, nsine = 5)
    c = [rand(nsine) for _ = 1:K]
    ω = [rand(1:10, nsine) for _ = 1:K]
    ϕ = [2π * rand(nsine) for _ = 1:K]
    s = sum_of_sines.([domain], c, ω, ϕ)
    u₀ = [s.u for s ∈ s]
    u = [ξ -> s.u(ξ - T) for s ∈ s]
    ū₀ = [apply_filter_int(s.U, filter, domain) for s ∈ s]
    ū = [apply_filter_int(x -> s.U(x - T), filter, domain) for s ∈ s]
    ūₕ₀ = mapreduce(u -> u.(x), hcat, ū₀)
    ūₕ = mapreduce(u -> u.(x), hcat, ū)
    (; u₀, ū₀, u, ū, ūₕ₀, ūₕ)
end
train = create_data(50);
test = create_data(10);

pl = plot(size = (500,300));
# pl = plot(size = (1000,600));
j = 0
for i = 1:2
    j += 1
    plot!(ξ, train.u₀[i]; label = "u₀", color = j, linestyle = :dash)
    plot!(ξ, train.ū₀[i]; label = "ū₀", color = j)
    # j += 1
    # plot!(ξ, u₀[i].(ξ .- T), label = "u(T)", color = j, linestyle = :dash)
    # plot!(ξ, ū[i], label = "ū(T)", color = j)
end
pl

close(u) = C -> 𝓢(C, u)
𝓢ᵤ = close(train.ūₕ₀)

function loss(C)
    ū = 𝓢ᵤ(C)
    loss = sum(abs2, ū - train.ūₕ) / size(ū, 2)
    loss, ū
end
loss(C̄)[1]

function loss(C, u₀, u_exact)
    ū = 𝓢(C, u₀)
    loss = sum(abs2, ū - u_exact) / size(ū, 2)
    loss, ū
end

relerr(C, u₀, u_exact) = norm(𝓢(C, u₀) - u_exact) / norm(u_exact)


# callback = (p, loss) -> (display(loss); false)
y⁻ = -1.5 # minimum(train.ūₕ)
y⁺ = 1.5 # maximum(train.ūₕ)
Δy = y⁺ - y⁻
callback = function (p, loss, ū)
    display(loss)
    pl = plot(size = (800,500), ylims = (y⁻ - 0.1Δy, y⁺ + 0.1Δy))
    # pl = plot(size = (1000,625))
    j = 0
    for i = 10:12
        j += 1
        scatter!(x, train.ū[i].(x); label = "ū(T) exact", color = j)
        plot!(x, ū[:, i]; label = "ū(T) fit", color = j)
    end
    display(pl)
    # sleep(0.1)
    false
end
callback(C̄, loss(C̄)...)

# result_ode = DiffEqFlux.sciml_train(loss, C̄, cb = callback, maxiters = 1)
result_ode = DiffEqFlux.sciml_train(loss, C̄, ADAM(0.05); cb = callback, maxiters = 500)
result_ode = DiffEqFlux.sciml_train(loss, result_ode.u, ADAM(0.01); cb = callback, maxiters = 500)

Cfit = result_ode.u
loss(Cfit)[1]
callback(Cfit, loss(Cfit)...)

Cfit[50, 45:55]

heatmap(reverse(C̄', dims = 2))
heatmap(reverse(Cfit', dims = 2))

##
d = test;
p1 = plot(; xlims = (a,b), xlabel = "x", title = "Solution (test data)");
# p1 = plot();
j = 0
for i = 5:7
    j += 1
    # plot!(ξ, d.u₀[i].(ξ .- T); label = "u(T)", color = j, linestyle = :dot);
    # plot!(x, 𝓢(C̄, ū₀[i].(ξ)); label = "ū₀(x - T)", color = :white)
    scatter!(x, d.ū[i].(x); label = "ū(T) exact", color = j)
    plot!(x, 𝓢(Cfit, d.ū₀[i].(x)); label = "ū(T) fit", color = j)
end
p1

p2 = plot(; xlims = (a, b), xlabel = "x", title = "Filter width");
# p2 = plot();
xline(x, y) = plot!(x, [y, y]; label = nothing, color = 1);
for (i, x) ∈ enumerate(x)
    hᵢ = h(x)
    x⁻ = x - hᵢ
    x⁺ = x + hᵢ
    if x⁻ < a
        xline([a, x⁺], i)
        xline([b - (a - x⁻), b], i)
    elseif x⁺ > b
        xline([x⁻, b], i)
        xline([a, a + (x⁺ - b)], i)
    else
        xline([x⁻, x⁺], i)
    end
    scatter!([x], [i]; label = nothing, color = 1)
end
p2

plot(p1, p2; layout = (2,1), size = (800, 800))

loss(Cfit, train.ūₕ₀, train.ūₕ)[1]
loss(Cfit, test.ūₕ₀, test.ūₕ)[1]

relerr(Cfit, train.ūₕ₀, train.ūₕ)
relerr(Cfit, test.ūₕ₀, test.ūₕ)
