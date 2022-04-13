# LSP indexing solution from
# https://github.com/julia-vscode/julia-vscode/issues/800#issuecomment-650085983
if isdefined(@__MODULE__, :LanguageServer)
    include("../src/DiscreteFiltering.jl")
    using .DiscreteFiltering
end

cd("examples")

using DiscreteFiltering
using JLD2
using LinearAlgebra
using SparseArrays
using GLMakie
using FFTW
using OrdinaryDiffEq: OrdinaryDiffEq, ODEFunction, ODEProblem, Tsit5
using DiffEqFlux

skew(A) = (A - A') / 2
symm(A) = (A + A') / 2

plotmat = mplotmat
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
x = LinRange(a, b, M + 1)[2:end]
ξ = LinRange(a, b, N + 1)[2:end]
Δx = (b - a) / M
Δξ = (b - a) / N

# Filter
h₀ = L / 30
h(x) = h₀ / 2 * (1 + 3 * sin(π * x) * exp(-2x^2))
ℱ = TopHatFilter(h)

# DNS operator
Aᴺ = circulant(N, [-1, 1], [1.0, -1.0] / 2Δξ)

# Initial guess for LES operator: Unfiltered
Aᴹ = circulant(M, [-1, 1], [1.0, -1.0] / 2Δx)
Aᴹ_left = circulant(M, [-1, 0], [1.0, -1.0] / Δx)

## Time (one period)
T = 1.0
tstops = LinRange(0, T, 51)[2:end]
nₜ = length(tstops)

# ODE solver for given operator and IC
f(u, A, t) = A * u
f!(du, u, A, t) = mul!(du, A, u)
function S(A, u, t)
    problem = ODEProblem(ODEFunction(f), u, (0.0, t), A)
    sol = OrdinaryDiffEq.solve(problem, Tsit5(); save_everystep = false)
    sol.u[end]
end
function S_mem(A, u, saveat; kwargs...)
    problem = ODEProblem(ODEFunction(f), u, (0.0, saveat[end]), A)
    OrdinaryDiffEq.solve(problem, Tsit5(); saveat, kwargs...)
end
function S_mem!(A, u, saveat; kwargs...)
    problem = ODEProblem(ODEFunction(f!), u, (0.0, saveat[end]), A)
    OrdinaryDiffEq.solve(problem, Tsit5(); saveat, kwargs...)
end

# Create signal
create_signal(nsample, kmax) = (;
    # c = [map(k -> (1 + 0.2 * randn()) * exp(-5 / 6 * max(0, k - 5)), 0:kmax) for _ = 1:nsample],
    c = [map(k -> (1 + 0.2 * randn()), 0:kmax) for _ = 1:nsample],
    ω = [2π * (0:kmax) for _ = 1:nsample],
    ϕ = [2π * rand(kmax + 1) for _ = 1:nsample]
)

# Create data from coefficients
function create_data(c, ω, ϕ, tstops)
    @assert length(c) == length(ω) == length(ϕ)
    nsample = length(c)
    s = [sum_of_sines(domain, c[i], ω[i], ϕ[i]) for i = 1:nsample]
    u₀ = [s.u for s ∈ s]
    U₀ = [s.U for s ∈ s]
    u = t -> [x -> s.u(x - t) for s ∈ s]
    ū₀ = [apply_filter_int(s.U, ℱ, domain) for s ∈ s]
    ūₕ₀ = mapreduce(u -> u.(x), hcat, ū₀)
    ū = t -> [apply_filter_int(x -> s.U(x - t), ℱ, domain) for s ∈ s]
    ūₕ = [mapreduce(u -> u.(x), hcat, ū(t)) for t ∈ tstops]
    # uₕ = [mapreduce(u -> u.(ξ), hcat, u(t)) for t ∈ tstops]
    (; u₀, U₀, u, ū₀, ū, ūₕ₀, ūₕ)
end

kmax = 25
ntrain = 500
ntrain_large = 500
ntest = 50
coeffs_train = create_signal(ntrain, kmax);
# coeffs_train_large = create_signal(ntrain_large, kmax);
coeffs_train_large = coeffs_train
coeffs_test = create_signal(ntest, kmax);

jldsave("output/data.jld2"; coeffs_train, coeffs_train_large, coeffs_test)
coeffs_train, coeffs_train_large, coeffs_test =
    load("output/data.jld2", "coeffs_train", "coeffs_train_large", "coeffs_test");

train = create_data(coeffs_train..., tstops);
# train_large = create_data(coeffs_train_large..., tstops);
train_large = train
test = create_data(coeffs_test..., tstops);

Wpat = [mapreduce(ℓ -> abs(x[m] + ℓ - ξ[n]) ≤ h(x[m]), |, (-L, 0, L)) for m = 1:M, n = 1:N]
Rpat = Wpat'
Wpat = mapreduce(i -> circshift(Wpat, (0, i)), .|, -2:2)
Rpat = mapreduce(i -> circshift(Rpat, (0, i)), .|, -2:2)
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
    xlabel = "x"
);
xlims!(ax1, (a, b));
fit = Observable[]
for (ii, i) ∈ enumerate(iplot)
    lines!(ax1, ξ, train.ū(T)[i].(ξ); color = Cycled(ii))
    push!(fit, Observable(S(Aᴹ, train.ūₕ₀[:, i], T)))
    scatter!(ax1, x, fit[ii]; label = "i = $i, fit", color = Cycled(ii))
end
Adiff = Observable(reverse(Aᴹ'; dims = 2);)
# axislegend(ax1)
ax2, hm = heatmap(rtp[1, 1], Adiff, axis = (; aspect = DataAspect(), title = "Ā - Aᴹ"))
Colorbar(rtp[1, 2], hm)
rtp

function callback(Ā, loss)
    display(loss)
    # for (ii, i) ∈ enumerate(iplot)
    #     fit[ii][] = S(Ā, train.ūₕ₀[:, i], T)
    # end
    Adiff[] = reverse((Ā .- Aᴹ .+ 1e-8 .* rand.())'; dims = 2)
    # Adiff[] = reverse((Ā .+ 1e-8 .* rand.())'; dims = 2)
    false
end

callback(A, l) = (println(l); false)

losses = zeros(0)
callback(A, l) = (println(l); push!(losses, l); false)

sol = S_mem(Aᴹ, train.ūₕ₀, tstops)

uex = zeros(M, ntrain, length(tstops));
for i ∈ eachindex(tstops)
    uex[:, :, i] = train.ūₕ[i]
end

loss(A, u₀, uₜ, tstops) = sum(abs2, S_mem(A, u₀, tstops) - uₜ)

nor = sum(abs2, uex)
nAm = sum(abs2, Aᴹ)
# loss(Ā) = loss(Ā, train.ūₕ₀, uex, tstops) / nor + 1e-2 * sum(abs, Ā[outside])
loss(Ā) = loss(Ā, train.ūₕ₀, uex, tstops) / nor + 5e0 * sum(abs2, Ā - Aᴹ) / nAm
# loss(Ā) = loss(Ā, train.ūₕ₀, uex, tstops) / nor + 1e-4 * sum(abs2, Ā - Aᴹ) + 1e-2 * sum(abs, Ā[outside])
# loss(Ā) = loss(Ā, train.ūₕ₀, uex, tstops) / nor + 1e-3 * sum(abs, Ā[outside])
# loss(Ā) = loss(Ā, train.ūₕ₀, uex, tstops) / nor
loss(Aᴹ)
plotmat(first(Zygote.gradient(loss, Aᴹ)))
using BenchmarkTools
@benchmark Zygote.gradient(loss, Aᴹ)

callback(Aᴹ, loss(Aᴹ))

function relerrs(A, u₀, uₜ, tstops; kwargs...)
    dim, nsample = size(u₀)
    sol = S_mem!(A, u₀, tstops; kwargs...)
    errs = zeros(length(tstops))
    for i ∈ eachindex(tstops)
        for j = 1:nsample
            errs[i] += @views norm(sol[:, j, i] - uₜ[i][:, j]) / norm(uₜ[i][:, j])
        end
        errs[i] /= nsample
    end
    errs
end

relerr(A, u₀, uₜ) = sum(relerrs(A, u₀, uₜ, tstops)) / nₜ
relerr(A, u₀, uₜ, t) = norm(S(A, u₀, t) - uₜ) / norm(uₜ)
relerr(Aᴹ, train.ūₕ₀, train.ūₕ)
relerr(Aᴹ, train.ūₕ₀, mapreduce(u -> u.(x), hcat, train.ū(0.5T)), 0.5T)

# Fit non-intrusively using least squares on snapshots
# train_large = create_data(200, tstops; kmax = 50);
# Ā_ls = -fit_Cbar(domain, ℱ, train_large.u₀, train_large.U₀, M, N, LinRange(1, T/2, 500);
#     method = :ridge, λ = 1e-1)
Ā_ls = -fit_Cbar(
    # domain, ℱ, train.u₀, train.U₀, M, N,
    domain, ℱ, train_large.u₀, train_large.U₀, M, N,
    # LinRange(1, T/2, 100);
    tstops;
    method = :ridge,
    λ = 1e-1
)
callback(Ā_ls, loss(Ā_ls))



# Fit intrusively
rtp
Ā = Aᴹ
result_ode = DiffEqFlux.sciml_train(loss, Ā, LBFGS(); cb = callback, maxiters = 50)
result_ode = DiffEqFlux.sciml_train(loss, Ā, ADAM(0.01); cb = callback, maxiters = 500)
result_ode =
    DiffEqFlux.sciml_train(loss, Ā, ADAM(0.001); cb = callback, maxiters = 500)
Ā = result_ode.u

lines(losses ./ loss(Aᴹ); axis = (; yscale = log10))

jldsave("output/Afit.jld2"; Abar = Ā)
Ā = load("output/Afit.jld2", "Abar")

loss(Ā)
relerr(Ā, train.ūₕ₀, train.ūₕ)
callback(Ā, loss(Ā))

# Fit filtering operator
# U = reduce(hcat, train_large.uₕ)
U = reduce(hcat, [mapreduce(u -> u.(ξ), hcat, train_large.u(t)) for t ∈ tstops])
Ū = reduce(hcat, train_large.ūₕ)

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

plotmat(W * R)
sum(W * R; dims = 2)

plotmat(R * W)
sum(R * W; dims = 2)

# Build explicit matrix
WAR = W * Aᴺ * R

plotmat(Aᴹ)
plotmat(WAR)
plotmat(Ā_ls)
plotmat(Ā)

# Compare resulting operators
plotmat(Aᴹ)
plotmat(Ā_ls)
plotmat(Ā)
plotmat(Ā_ls - Aᴹ)
plotmat(Ā - Aᴹ)
plotmat(inds)

# eigenvalues
plotmat(symm(Ā); title = "(Ā + Ā') / 2")
plotmat(symm(Ā_ls); title = "(Ā + Ā') / 2")
plotmat(skew(Ā - Aᴹ); title = "(Ā - Ā') / 2")
plotmat(skew(Ā_ls - Aᴹ); title = "(Ā - Ā') / 2")

bar((Ā-Aᴹ)[45, :])

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


# Propagate symbols into plotting scripts
if isdefined(@__MODULE__, :LanguageServer)
    include("plots.jl")
    include("makie.jl")
end
