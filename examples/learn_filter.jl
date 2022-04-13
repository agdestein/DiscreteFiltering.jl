# LSP indexing solution from
# https://github.com/julia-vscode/julia-vscode/issues/800#issuecomment-650085983
if isdefined(@__MODULE__, :LanguageServer)
    include("../src/DiscreteFiltering.jl")
    using .DiscreteFiltering
end

using DiscreteFiltering
using LinearAlgebra
using GLMakie
using FFTW
using OrdinaryDiffEq: OrdinaryDiffEq, ODEFunction, ODEProblem, Tsit5
using DiffEqFlux: DiffEqFlux, ADAM, LBFGS

skew(A) = (A - A') / 2
symm(A) = (A + A') / 2

A = randn(200, 200)
A = [sum(@view A[1:i, 1:j]) for i = 1:200, j = 1:200]
plotmat(A)

## Domain
a = 0.0
b = 1.0
L = b - a
domain = PeriodicIntervalDomain(a, b)

# Discretization
M = 100
N = 2000
x = discretize(domain, M)
ξ = discretize(domain, N)
Δx = (b - a) / M

# Filter
# h₀ = 1.0Δx
h₀ = L / 30
# h₀ = (b - a) / 100
# h₀ = Δx / 2
# h(x) = h₀ / 2 * (2 - cos(2π * x)),
h(x) = h₀ / 2 * (1 + 3 * sin(π * x) * exp(-2x^2))
filter = TopHatFilter(h)
lines(ξ, h; axis = (; xlabel = "x"))
ylims!(0.0, 0.06)

# DNS operator
Aᴺ = -advection_matrix(domain, N)

# Initial guess for LES operator: Unfiltered
Aᴹ = -Matrix(advection_matrix(domain, M))

## Time (one period)
T = 1.0
tstops = LinRange(0, T, 21)
nₜ = length(tstops)

# ODE solver for given operator and IC
f(u, A, t) = A * u
odefunction = ODEFunction(f)
function S(A, u, t)
    problem = ODEProblem(odefunction, u, (0.0, t), A)
    sol = OrdinaryDiffEq.solve(problem, Tsit5(); save_everystep = false)
    sol.u[end]
end
function S_mem(A, u, saveat)
    problem = ODEProblem(odefunction, u, (0.0, saveat[end]), A)
    OrdinaryDiffEq.solve(problem, Tsit5(); saveat)
end

# Create signal
function create_data(nsample, tstops; kmax = 20)
    # ω = map(i -> rand() * cospi(i / kmax), 0:kmax)
    # c = [map(ω -> (1 + 0.0 * rand()) / 2, 0:kmax) for _ = 1:nsample]
    c = [map(k -> (1 + 0.2 * randn()) * exp(-5 / 6 * max(0, k - 5)), 0:kmax) for _ = 1:nsample]
    s = [sum_of_sines(domain, c[i], 2π * (0:kmax), 2π * rand(kmax + 1)) for i = 1:nsample]
    u₀ = [s.u for s ∈ s]
    U₀ = [s.U for s ∈ s]
    u = t -> [x -> s.u(x - t) for s ∈ s]
    ū₀ = [apply_filter_int(s.U, filter, domain) for s ∈ s]
    ūₕ₀ = mapreduce(u -> u.(x), hcat, ū₀)
    ū = t -> [apply_filter_int(x -> s.U(x - t), filter, domain) for s ∈ s]
    ūₕ = [mapreduce(u -> u.(x), hcat, ū(t)) for t ∈ tstops]
    uₕ = [mapreduce(u -> u.(ξ), hcat, u(t)) for t ∈ tstops]
    (; c, u₀, U₀, u, ū₀, ū, ūₕ₀, ūₕ, uₕ)
end
kmax = 25
train = create_data(200, tstops; kmax);
test = create_data(200, tstops; kmax);

k = 0:kmax
e = abs2.(fft(train.u₀[1].(ξ)))[1:kmax+1]; e /= e[1]
ē = abs2.(fft(train.ūₕ₀[:, 1]))[1:kmax+1]; ē /= ē[1]
fig = Figure();
ax = Axis(fig[1,1]; xlabel = "k", yscale = log10);
lines!(ax, k, e; label = "u");
lines!(ax, k, ē; label = "ū");
lines!(ax, k[5:end], exp.(-5/3 .* (k[5:end] .- 5)); linestyle = :dash, label = "-5/3 k");
axislegend(ax);
fig

fig = Figure();
ax = Axis(fig[1,1]; xlabel = "k", yscale = log10, title = "e(k)")
for i = 1:5
    lines!(ax, 0:kmax, train.c[i]; label = "e(k)", linestyle = :dash)
end
fig

fig = Figure();
ax = Axis(fig[1, 1]; xlabel = "x", legend = false);
for i = 1:3
    lines!(ax, ξ, train.u₀[i]; linestyle = :dash)
    lines!(ax, ξ, train.ū₀[i]; color = Cycled(i))
end
fig

##
i = 3
fig = Figure();
ax = Axis(
    fig[1, 1];
    xlims = (a, b),
    xlabel = "x",
    # title = "ū(x+t, t)",
);
lines!(ξ, train.u₀[i].(ξ); label = "u(x, 0)", linestyle = :dash);
for t ∈ LinRange(0, T/2, 4)
    lines!(ξ, train.ū(t)[i].(ξ .+ t); label = "t = $t")
end
fig

Wpat = [mapreduce(ℓ -> abs(x[m] + ℓ - ξ[n]) ≤ h(x[m]), |, (-L, 0, L)) for m = 1:M, n = 1:N]
Rpat = Wpat'
Wpat = mapreduce(i -> circshift(Wpat, (0, i)), .|, -0:0)
Rpat = mapreduce(i -> circshift(Rpat, (0, i)), .|, -0:0)
Apat = Aᴺ .≠ 0
inds = Wpat * Apat * Rpat .≠ 0
outside = .!inds

plotmat(Wpat)
plotmat(Rpat)
plotmat(inds)

# callback(p, loss) = (display(loss); false)

iplot = 1:5
y⁻ = minimum(train.ūₕ₀[:, iplot])
y⁺ = maximum(train.ūₕ₀[:, iplot])
Δy = y⁺ - y⁻
rtp = Figure();
ax1 = Axis(
    rtp[1, 1:2];
    xlabel = "x",
);
xlims!(ax1, (a, b));
ylims!(ax1, (y⁻ - 0.2Δy, y⁺ + 0.2Δy));
fit = Observable[]
for (ii, i) ∈ enumerate(iplot)
    lines!(ax1, ξ, train.ū(T)[i].(ξ); color = Cycled(ii))
    push!(fit, Observable(S(Aᴹ, train.ūₕ₀[:, i], T)))
    scatter!(ax1, x, fit[ii]; label = "i = $i, fit", color = Cycled(ii))
end
Adiff = Observable(reverse(Aᴹ'; dims = 2));
# axislegend(ax1)
# ax2, hm = heatmap(rtp[2, 1], Adiff, axis = (; aspect = DataAspect(), title = "Ā - Aᴹ"))
# Colorbar(rtp[2, 2], hm)
rtp

function callback(Ā)
    for (ii, i) ∈ enumerate(iplot)
        fit[ii][] = S(Ā, train.ūₕ₀[:, i], T)
    end
    # Adiff[] = reverse((Ā .- Aᴹ .+ 1e-14 .* rand.())'; dims = 2)
    Adiff[] = reverse((Ā .+ 1e-14 .* rand.())'; dims = 2)
end

# Unfiltered operator
callback(Aᴹ)

# Relative errors
relerr(A, u₀, uₜ) = sum(norm(S(A, u₀, t) - uₜ) / norm(uₜ) for (t, uₜ) ∈ zip(tstops, uₜ)) / nₜ
relerr(A, u₀, uₜ, t) = norm(S(A, u₀, t) - uₜ) / norm(uₜ)
relerr(Aᴹ, train.ūₕ₀, train.ūₕ)
relerr(Aᴹ, train.ūₕ₀, mapreduce(u -> u.(x), hcat, train.ū(0.5T)), 0.5T)

# Fit non-intrusively using least squares on snapshots
# train_large = create_data(200, tstops; kmax = 50);
# Ā_ls = -fit_Cbar(domain, filter, train_large.u₀, train_large.U₀, M, N, LinRange(1, T/2, 500);
#     method = :ridge, λ = 1e-1)
Ā_ls = -fit_Cbar(
    domain, filter, train.u₀, train.U₀, M, N,
    # LinRange(1, T/2, 100);
    tstops;
    method = :ridge,
    λ = 1e-1
)
callback(Ā_ls)

# Fit filtering operator
# trainfive = create_data(1000, tstops; kmax);
trainfive = train
U = reduce(hcat, trainfive.uₕ)
Ū = reduce(hcat, trainfive.ūₕ)

# min ||WU - Ū||₂² + λ ||W||₂²
# W = ((U * U' + 1e-6 * I) \ (U * Ū'))'
W = (Ū * U') / (U * U' + 1e-6 * I) 
plotmat(W)
sum(W; dims = 2)

# min ||RŪ - U||₂² + λ ||R||₂²
# R = inv(W)
# R = (W'W + 1e-3 * I) \ W'
# R = ((Ū * Ū' + 1e-4 * I) \ (Ū * U'))'
R = (U * Ū') / (Ū * Ū' + 1e-4 * I)
plotmat(R)
sum(R; dims = 2)

# sinc
lines(LinRange(-10, 10, 1000), x -> sinpi(x) / π / x)

plotmat(W * R)
sum(W * R; dims = 2)

plotmat(R * W)
sum(R * W; dims = 2)

Ā = W * Aᴺ * R
plotmat(Ā_ls)
plotmat(Ā)
plotmat(Aᴹ)

rtp
callback(Aᴹ)
callback(Ā)
callback(Ā_ls)

# Compare resulting operators
plotmat(Aᴹ)
plotmat(Ā_ls)
plotmat(Ā)
plotmat(Ā_ls - Aᴹ)
plotmat(Ā - Aᴹ)
plotmat(inds)

# eigenvalues
plotmat(symm(Ā))
plotmat(symm(Ā_ls))
plotmat(skew(Ā))
plotmat(skew(Ā_ls))

eig = eigen(Ā)
eig = eigen(symm(Ā))
eig = eigen(skew(Ā))
scatter(real.(eig.values), imag.(eig.values))

lines(x, real(eig.vectors[:, 1]))
lines!(x, real(eig.vectors[:, 2]))
lines!(x, real(eig.vectors[:, 3]))
lines!(x, real(eig.vectors[:, 4]))

# Deviation from unfiltered operator
norm(Ā - Aᴹ) / norm(Aᴹ)
norm(Ā_ls - Aᴹ) / norm(Aᴹ)

# Performance on training time steps
relerr(Aᴹ, train.ūₕ₀, train.ūₕ)
relerr(Aᴹ, test.ūₕ₀, test.ūₕ)
relerr(Ā, train.ūₕ₀, train.ūₕ)
relerr(Ā, test.ūₕ₀, test.ūₕ)
relerr(Ā_ls, train.ūₕ₀, train.ūₕ)
relerr(Ā_ls, test.ūₕ₀, test.ūₕ)

# Performance outside of training time interval
tnew = 10.0T
train_exact = mapreduce(u -> u.(x), hcat, train.ū(tnew))
test_exact = mapreduce(u -> u.(x), hcat, test.ū(tnew))
relerr(Aᴹ, train.ūₕ₀, train_exact, tnew)
relerr(Aᴹ, test.ūₕ₀, test_exact, tnew)
relerr(Ā, train.ūₕ₀, train_exact, tnew)
relerr(Ā, test.ūₕ₀, test_exact, tnew)
relerr(Ā_ls, train.ūₕ₀, train_exact, tnew)
relerr(Ā_ls, test.ūₕ₀, test_exact, tnew)

# Energy evolution
tnew = 6T
i = 7
tplot = LinRange(0, tnew, 1000)
saveat = LinRange(0, tnew, 2000)
# sol = S_mem(Aᴹ, train.ūₕ₀[:, i], saveat)
sol = S_mem(Ā, train.ūₕ₀[:, i], saveat)
# sol = S_mem(Ā_ls, train.ūₕ₀[:, i], saveat)
E(t) = 1 / 2M * sum(abs2, train.u(t)[i].(x))
Ē(t) = 1 / 2M * sum(apply_filter(x -> train.u(t)[i](x)^2, filter, domain).(x))
Eu = E.(tplot)
Eū = [1 / 2M * sum(abs2, sol(t)) for t ∈ tplot]
Eup = [1 / 2M * sum(abs2, train.u(t)[i].(x) .- sol(t)) for t ∈ tplot]
fig = Figure();
ax = Axis(fig[1, 1], xlabel = "t",
    title = "Total kinetic energy",
);
lines!(ax, tplot, E, label = "E(u)")
# lines!(ax, tplot, Ē, label = "Ē(u)")
lines!(ax, tplot, Eū, label = "E(ū)")
# lines!(ax, tplot, Eup, label = "E(u')")
lines!(ax, tplot, Eū + Eup, label = "E(ū) + E(u')")
# axislegend(ax)
Legend(fig[1, 2], ax)
# save("filtered_kinetic_energy.pdf", fig)
fig
