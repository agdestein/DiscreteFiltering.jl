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
L = b - a
domain = PeriodicIntervalDomain(a, b)

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
Aᴺ = -advection_matrix(domain, N)

# Initial guess for LES operator: Unfiltered
Aᴹ = -Matrix(advection_matrix(domain, M))

## Time
T = 2.0
tstops = LinRange(0, T, 10)[2:end]
nₜ = length(tstops)

# ODE function for given operator and IC
f(u, A, t) = A * u
odefunction = ODEFunction(f)
function S(A, u, t)
    problem = ODEProblem(odefunction, u, (0.0, t), A)
    sol = OrdinaryDiffEq.solve(problem, Tsit5(); save_everystep = false)
    sol.u[end]
end

# Create signal
function create_data(K, tstops; ωmax = 20)
    # ω = map(i -> rand() * cospi(i / ωmax), 0:ωmax)
    c = map(ω -> rand() * exp(-5/6π * ω), 0:ωmax)
    s = [sum_of_sines(domain, c, 0:ωmax, 2π * rand(ωmax + 1)) for _ = 1:K]
    u₀ = [s.u for s ∈ s]
    u = t -> [x -> s.u(x - t) for s ∈ s]
    ū₀ = [apply_filter_int(s.U, filter, domain) for s ∈ s]
    ūₕ₀ = mapreduce(u -> u.(x), hcat, ū₀)
    ū = t -> [apply_filter_int(x -> s.U(x - t), filter, domain) for s ∈ s]
    ūₕ = [mapreduce(u -> u.(x), hcat, ū(t)) for t ∈ tstops]
    (; u₀, u, ū₀, ū, ūₕ₀, ūₕ)
end
train = create_data(50, tstops; ωmax = 40)
test = create_data(50, tstops; ωmax = 40)

pl = plot(; title = "u₀(x), ū₀(x)", legend = false);
j = 0
for i = [1, 8]
    j += 1
    plot!(ξ, train.u₀[i]; label = "u₀", color = j, linestyle = :dash)
    plot!(ξ, train.ū₀[i]; label = "ū₀", color = j)
end
# xline(x, y) = plot!(x, [y, y]; label = nothing, color = 1);
# xline([a - h(a), a + h(a)], 0.0)
# xline([b - h(b), b + h(b)], 0.0)
# xm = (a + b) / 2
# xline([xm - h(xm), xm + h(xm)], 0.0)
pl

savefig("output/data.pdf")

##
j = 0
i = 3
pl = plot(; xlims = (a, b), xlabel = "x", title = "ū(x+t, t)");
j += 1
plot!(ξ, x -> train.u₀[i](x); label = "u₀", color = j, linestyle = :dash);
for t ∈ LinRange(0, T, 4)
    j += 1
    plot!(ξ, x -> train.ū(t)[i](x + t); label = "t = $t", color = j)
    # scatter!(x, train.ū(t)[i]; label = "t = $t", color = j, markeralpha = 0.5)
end
pl

inds = [mapreduce(ℓ -> abs(x[i] + ℓ - x[j]) ≤ 3h(x[i]), |, [-L, 0, L]) for i = 1:M, j = 1:M]
outside = .!inds
heatmap(reverse(inds', dims = 2); aspect_ratio = :equal, xlims = (1, M))

# callback = (p, loss) -> (display(loss); false)
iplot = 1:5
y⁻ = minimum(train.ūₕ₀[:, iplot])
y⁺ = maximum(train.ūₕ₀[:, iplot])
Δy = y⁺ - y⁻
# anim = Animation("output/loss", String[])
callback = function (Ā, loss)
    display(loss)
    p1 = plot(; ylims = (y⁻ - 0.2Δy, y⁺ + 0.2Δy), xlabel = "x",
              # legend = :topleft,
              legend = false,
              title = "ūᵢ(T)",
              # title = "ū(T) initial guess",
              # title = "ū(T) after training",
             )
    j = 0
    for i ∈ iplot
        j += 1
        scatter!(p1, x, train.ū(T)[i].(x);
                 # label = "i = $i, exact",
                 label = nothing,
                 color = j, markeralpha = 0.5)
        plot!(p1, x, S(Ā, train.ūₕ₀[:, i], T); label = "i = $i, fit", color = j)
    end
    # sleep(0.1)
    # frame(anim)
    p2 = heatmap(reverse((Ā - Aᴹ)', dims = 2); aspect_ratio = :equal, xlims = (1, M), title = "Ā - Aᴹ")
    pl = plot(p1, p2; layout = (2, 1), size = (800, 885))
    display(p1)
    # savefig(p1, "output/initial.pdf")
    # savefig(p1, "output/final.pdf")
    false
end

callback(Aᴹ, loss(Aᴹ))
callback(Ā, loss(Ā))

loss(A, u₀, u_exact) = sum(sum(abs2, S(A, u₀, t) .- u_exact) for (t, u_exact) ∈ zip(tstops, u_exact)) / size(u_exact[1], 2) / nₜ
# loss(Ā) = loss(Ā, train.ūₕ₀, train.ūₕ) + 1e-2 * sum(abs, Ā[outside])
# loss(Ā) = loss(Ā, train.ūₕ₀, train.ūₕ) + 1e-4 * sum(abs, Ā - Aᴹ)
# loss(Ā) = loss(Ā, train.ūₕ₀, train.ūₕ) + 1e-4 * sum(abs2, Ā - Aᴹ) + 1e-2 * sum(abs, Ā[outside])
# loss(Ā) = loss(Ā, train.ūₕ₀, train.ūₕ) + 1e-3 * sum(abs, Ā[outside])
loss(Ā) = loss(Ā, train.ūₕ₀, train.ūₕ)
loss(Aᴹ)

callback(Aᴹ, loss(Aᴹ))
savefig("output/initial.pdf")

relerr(A, u₀, u_exact) = sum(norm(S(A, u₀, t) - u_exact) / norm(u_exact) for (t, u_exact) ∈ zip(tstops, u_exact)) / nₜ
relerr(Aᴹ, train.ūₕ₀, train.ūₕ)

Ā = Aᴹ
result_ode = DiffEqFlux.sciml_train(loss, Aᴹ, ADAM(0.001); cb = callback, maxiters = 100)
result_ode =
    DiffEqFlux.sciml_train(loss, result_ode.u, ADAM(0.0005); cb = callback, maxiters = 500)

# gif(anim, "output/loss.gif"; fps = 10)

Ā = result_ode.u
loss(Ā)
callback(Ā, loss(Ā))
savefig("output/final.pdf")

Aᴹ[50, 45:55]
Ā[50, 45:55]

heatmap(-reverse(Aᴹ', dims = 2); aspect_ratio = :equal, xlims = (1, M))
heatmap(-reverse(Ā', dims = 2); aspect_ratio = :equal, xlims = (1, M))
heatmap(-reverse((Ā - Aᴹ)', dims = 2); aspect_ratio = :equal, xlims = (1, M))
# heatmap(reverse((Ā - Aᴹ)', dims = 2); aspect_ratio = :equal, xlims = (1, M), color = :viridis)
# heatmap(reverse(Ā[150:200, 150:200]', dims = 2); aspect_ratio = :equal, xlims = (1, 51))
bar(Ā[45, :])
savefig("output/C.pdf")
savefig("output/Cfit.pdf")

##
# d = train;
d = test;
# p1 = plot();
p1 = plot(; xlims = (a, b), xlabel = "x", title = "Solution (test data)");
j = 0
k = 9
i = 6
for k = [1, 3, 6, 9]
# for i = 1:3
    j += 1
    # plot!(ξ, d.u₀[i].(ξ .- T); label = "u(T)", color = j, linestyle = :dot);
    # plot!(x, 𝓢(C̄, ū₀[i].(ξ)); label = "ū₀(x - T)", color = :white)
    # lab = "ū(T)"
    lab = "ū($(tstops[k]))"
    scatter!(x, d.ūₕ[k][:, i];
             # label = "$lab exact",
             label = nothing,
             color = j, markeralpha = 0.5)
    # plot!(x, S(Aᴹ, d.ūₕ₀[:, i], tstops[k]); linestyle = :dot, label = "$lab", color = j)
    plot!(x, S(Aᴹ, d.ūₕ₀[:, i], tstops[k]); linestyle = :dot, label = nothing, color = j)
    plot!(x, S(Ā, d.ūₕ₀[:, i], tstops[k]); label = "$lab", color = j)
    # plot!(x, (𝓢(Cfit, d.ū₀[i].(x)) .- d.ū[i].(x)) ./ √(sum(d.ū[i].(x) .^ 2) / length(x)); label = "error $i", color = j)
end
p1
savefig("output/evolution.pdf")

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

plot(p1; layout = (1, 1), size = (800, 885))
plot(p1, p2; layout = (2, 1), size = (800, 885))

loss(Aᴹ, train.ūₕ₀, train.ūₕ)
loss(Aᴹ, test.ūₕ₀, test.ūₕ)
loss(Ā, train.ūₕ₀, train.ūₕ)
loss(Ā, test.ūₕ₀, test.ūₕ)

relerr(Aᴹ, train.ūₕ₀, train.ūₕ)
relerr(Aᴹ, test.ūₕ₀, test.ūₕ)
relerr(Ā, train.ūₕ₀, train.ūₕ)
relerr(Ā, test.ūₕ₀, test.ūₕ)

relerr(
    Aᴹ,
    mapreduce(u -> u.(x), hcat, test.ū₀),
    [mapreduce(u -> u.(x .- t), hcat, test.ū₀) for t ∈ tstops],
)
