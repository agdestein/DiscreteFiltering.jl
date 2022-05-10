if isdefined(@__MODULE__, :LanguageServer)
    include("../src/DiscreteFiltering.jl")
    using .DiscreteFiltering
end

cd("examples")

using DiscreteFiltering
using JLD2
using LinearAlgebra
using SparseArrays
using FFTW
using OrdinaryDiffEq
using DiffEqFlux
using Plots

skew(A) = (A - A') / 2
symm(A) = (A + A') / 2

plotmat = pplotmat
A = randn(200, 200)
A = [sum(@view A[1:i, 1:j]) for i = 1:200, j = 1:200]
plotmat(A)

## Domain
a = 0.0
b = 1.0
L = b - a

# Discretization
M = 100
N = 1000
x = LinRange(a, b, M + 1)[2:end]
ξ = LinRange(a, b, N + 1)[2:end]
Δx = (b - a) / M
Δξ = (b - a) / N

# Filter
h₀ = L / 30
# h(x) = h₀ / 2 * (1 + 3 * sin(π * x) * exp(-2x^2))
h(x) = h₀ * (1 + 1 / 3 * sin(2π * x))

# DNS operator
Aᴺ = circulant(N, [-1, 1], [1.0, -1.0] / 2Δξ)

# Initial guess for LES operator: Unfiltered
Aᴹ = Matrix(circulant(M, [-1, 1], [1.0, -1.0] / 2Δx))
Aᴹ_left = Matrix(circulant(M, [-1, 0], [1.0, -1.0] / Δx))

## Time (one period)
T = 1.0
tstops = LinRange(0, 1T, 51)[2:end]
nₜ = length(tstops)

# Create signal
create_signal(nsample, kmax) = (;
    # c = [map(k -> (1 + 0.2 * randn()) * exp(-5 / 6 * max(0, k - 5)), 0:kmax) for _ = 1:nsample],
    c = [map(k -> (1 + 0.2 * randn()) / (k + 5)^1.5, 0:kmax) for _ = 1:nsample],
    ϕ = [2π * rand(kmax + 1) for _ = 1:nsample],
)

u(c, ϕ) = (x, t) -> sum((c[k] * sin(2π * (k - 1) * (x - t) - ϕ[k]) for k ∈ eachindex(c)))
∂u∂t(c, ϕ) =
    (x, t) ->
        -sum((
            c[k] * 2π * (k - 1) * cos(2π * (k - 1) * (x - t) - ϕ[k]) for k ∈ eachindex(c)
        ))
ū(c, ϕ) =
    (x, t) ->
        -c[1] * sin(ϕ[1]) - sum((
            c[k] / (2π * (k - 1)) * (
                cos(2π * (k - 1) * (x + h(x) - t) - ϕ[k]) -
                cos(2π * (k - 1) * (x - h(x) - t) - ϕ[k])
            ) / 2h(x) for k ∈ eachindex(c)[2:end]
        ))
∂ū∂t(c, ϕ) =
    (x, t) ->
        -sum((
            c[k] * (
                sin(2π * (k - 1) * (x + h(x) - t) - ϕ[k]) -
                sin(2π * (k - 1) * (x - h(x) - t) - ϕ[k])
            ) / 2h(x) for k ∈ eachindex(c)
        ))

# Create data from coefficients
function create_data(c, ϕ, x, ξ, tstops)
    @assert length(c) == length(ϕ)
    nsample = length(c)
    us = u.(c, ϕ)
    ∂u∂ts = ∂u∂t.(c, ϕ)
    ūs = ū.(c, ϕ)
    ∂ū∂ts = ∂ū∂t.(c, ϕ)
    ū₀_data = [ū(x, 0.0) for x ∈ x, ū ∈ ūs]
    ū_data = [ū(x, t) for x ∈ x, ū ∈ ūs, t ∈ tstops]
    ∂ū∂t_data = [∂ū∂t(x, t) for x ∈ x, ∂ū∂t ∈ ∂ū∂ts, t ∈ tstops]
    # u_data = [u.(ξ, t) for ξ ∈ ξ, u ∈ us, t ∈ tstops]
    # ∂u∂t_data = [∂u∂t(ξ, t) for ξ ∈ ξ, ∂u∂t ∈ ∂u∂ts, t ∈ tstops]
    u_data = [0.0 for ξ ∈ ξ, u ∈ us, t ∈ tstops]
    ∂u∂t_data = [0.0 for ξ ∈ ξ, ∂u∂t ∈ ∂u∂ts, t ∈ tstops]
    (; u = us, ū = ūs, ∂ū∂t = ∂ū∂ts, ū₀_data, ū_data, ∂ū∂t_data, u_data, ∂u∂t_data)
end

# Create data from coefficients
function create_data(c, ϕ, x, ξ, tstops)
    K = length(c[1])
    M = length(x)
    N = length(ξ)
    nt = length(tstops)
    @assert length(c) == length(ϕ)
    nsample = length(c)

    ū₀_data = zeros(M, nsample)
    @inbounds for is = 1:nsample, (m, x) ∈ enumerate(x), k = 1:K
        ū₀_data[m, is] +=
            k == 1 ? -c[is][1] * sin(ϕ[is][1]) :
            -c[is][k] / (2π * (k - 1)) * (
                cos(2π * (k - 1) * (x + h(x)) - ϕ[is][k]) -
                cos(2π * (k - 1) * (x - h(x)) - ϕ[is][k])
            ) / 2h(x)
    end

    ū_data = zeros(M, nsample, nt)
    ∂ū∂t_data = zeros(M, nsample, nt)
    @inbounds for (it, t) ∈ enumerate(tstops),
        is = 1:nsample,
        (m, x) ∈ enumerate(x),
        k = 1:K

        ū_data[m, is, it] +=
            k == 1 ? -c[is][1] * sin(ϕ[is][1]) :
            -c[is][k] / (2π * (k - 1)) * (
                cos(2π * (k - 1) * (x + h(x) - t) - ϕ[is][k]) -
                cos(2π * (k - 1) * (x - h(x) - t) - ϕ[is][k])
            ) / 2h(x)
        ∂ū∂t_data[m, is, it] -=
            c[is][k] * (
                sin(2π * (k - 1) * (x + h(x) - t) - ϕ[is][k]) -
                sin(2π * (k - 1) * (x - h(x) - t) - ϕ[is][k])
            ) / 2h(x)
    end

    # u_data = zeros(N, nsample, nt)
    # ∂u∂t_data = zeros(N, nsample, nt)
    # @inbounds for (it, t) ∈ enumerate(tstops), is = 1:nsample, (n, ξ) ∈ enumerate(ξ), k = 1:K
    #     u_data[n, is, it] += c[is][k] * sin(2π * (k - 1) * (ξ - t) - ϕ[is][k])
    #     ∂u∂t_data[n, is, it] -= c[is][k] * 2π * (k - 1) * cos(2π * (k - 1) * (ξ - t) - ϕ[is][k])
    # end

    (; ū₀_data, ū_data, ∂ū∂t_data)
    # (; ū₀_data, ū_data, ∂ū∂t_data, u_data, ∂u∂t_data)
end

kmax = 100
# kmax = M ÷ 2
ntrain = 500
ntest = 20
coeffs_train = create_signal(ntrain, kmax);
coeffs_test = create_signal(ntest, kmax);
plot(coeffs_train.c[1:3] ./ norm.(coeffs_train.c[1:3]); yscale = :log10)

jldsave("output/data.jld2"; coeffs_train, coeffs_train_large, coeffs_test)
coeffs_train, coeffs_train_large, coeffs_test =
    load("output/data.jld2", "coeffs_train", "coeffs_train_large", "coeffs_test");

train = create_data(coeffs_train..., x, ξ, tstops);
@time test = create_data(coeffs_test..., x, ξ, tstops);

train30 = train;
test30 = test;

train50 = train;
test50 = test;

iplot = 1:5
y⁻ = minimum(test.ū₀_data[:, iplot])
y⁺ = maximum(test.ū₀_data[:, iplot])
Δy = y⁺ - y⁻
function callback(Ā, loss)
    gr()
    display(loss)
    p1 = plot(; xlabel = "x", xlims = (a, b), size = (1000, 700))
    for (ii, i) ∈ enumerate(iplot)
        plot!(p1, ξ, test.ū[i].(ξ, 1T); label = nothing, color = ii)
        scatter!(
            p1,
            x,
            S!(Ā, test.ū₀_data[:, i], [1T])[end];
            markeralpha = 0.5,
            label = "i = $i",
            color = ii,
        )
        # plot!(p1, x, S!(Ā, test.ū₀_data[:, i], [50T]; abstol = 1e-10, reltol = 1e-8)[end]; markeralpha = 0.5, label = "i = $i", color = ii, linestyle = :dash)
    end
    p2 = pplotmat(Ā .- Aᴹ)
    display(p1)
    # display(p2)
    # display(plot(p1, p2; layout = (1, 2)))
    false
end

callback(Aᴹ, loss(Aᴹ))
callback(WAR, loss(WAR))
callback(Ā_ls, loss(Ā_ls))
callback(Ā, loss(Ā))

callback(A, l) = (println(l); false)

losses = zeros(0)
callback(A, l) = (println(l); push!(losses, l); false)

callback(Aᴹ, loss(Aᴹ))

function momentumloss(A, u₀, uₜ, ∂uₜ∂t, tstops)
    u = S(A, u₀, tstops)
    # sum(abs2, u - uₜ) / sum(abs2, uₜ) + 1e-1 * sum(abs2, A * u - ∂uₜ∂t) / sum(abs2, ∂uₜ∂t)
    sum(abs2, u - uₜ) / sum(abs2, uₜ) +
    1e-1 * mapreduce(
        (u, du) -> sum(abs2, A * u - du),
        +,
        eachslice(u; dims = 3),
        eachslice(∂uₜ∂t; dims = 3),
    ) / sum(abs2, ∂uₜ∂t)
end
loss(A, u₀, uₜ, tstops) = sum(abs2, S(A, u₀, tstops) - uₜ)
# reg(Ā) = sum(abs, Ā[outside])
reg(Ā) = sum(abs2, Ā - Aᴹ) / sum(abs2, Aᴹ)
# reg(Ā) = 1e-4 * sum(abs2, Ā - Aᴹ) + 1e-2 * sum(abs, Ā[outside])
# reg(Ā) = 1e-3 * sum(abs, Ā[outside])
# reg(Ā) = 0.0
loss(Ā) =
    momentumloss(Ā, train.ū₀_data, train.ū_data, train.∂ū∂t_data, tstops) +
    1e-3 * reg(Ā)
loss(Aᴹ)

plotmat(first(Zygote.gradient(loss, Aᴹ)))

using BenchmarkTools
@benchmark Zygote.gradient(loss, Aᴹ)

function relerrs(A, u₀, uₜ, tstops; kwargs...)
    nsample = size(u₀, 2)
    sol = S!(A, u₀, tstops; kwargs...)
    errs = zeros(length(tstops))
    for i ∈ eachindex(tstops)
        for j = 1:nsample
            errs[i] += @views norm(sol[:, j, i] - uₜ[:, j, i]) / norm(uₜ[:, j, i])
        end
        errs[i] /= nsample
    end
    errs
end

relerr(A, u₀, uₜ) = sum(relerrs(A, u₀, uₜ, tstops)) / nₜ
relerr(A, u₀, uₜ, t) = norm(S(A, u₀, t) - uₜ) / norm(uₜ)
relerr(Aᴹ, train.ū₀, train.ū)

# Fit intrusively
Ā = Aᴹ
result_ode = DiffEqFlux.sciml_train(loss, Ā, LBFGS(); cb = callback, maxiters = 50)
result_ode = DiffEqFlux.sciml_train(loss, Ā, ADAM(0.01); cb = callback, maxiters = 500)
result_ode = DiffEqFlux.sciml_train(loss, Ā, ADAM(0.001); cb = callback, maxiters = 500)
Ā = result_ode.u

plot(losses ./ loss(Aᴹ); yscale = :log10)

jldsave("output/Afit.jld2"; Abar = Ā)
Ā = load("output/Afit.jld2", "Abar")

loss(Ā)
relerr(Ā, train.ū₀_data, train.ū_data)
callback(Ā, loss(Ā))

pred = S!(Aᴹ, test.ū₀_data, tstops)
p = plot(; size = (1000, 1000), legend = false);
for i = 1:50
    scatter!(p, pred[:, :, i], test.ū_data[:, :, i])#; markeralpha = 0.1)
    # scatter!(p, pred[:, :, i], test.ū_data[:, :, i])#; markeralpha = 0.1)
end
p

train = train30;
train = train50;

# Snapshot matrices
ϵ = 1e-8
# ϵ = 0.0
U = reshape(train.u_data, N, :) .+ ϵ .* randn.()
∂U∂t = reshape(train.∂u∂t_data, N, :) .+ ϵ .* randn.()
Ū = reshape(train.ū_data, M, :) .+ ϵ .* randn.()
∂Ū∂t = reshape(train.∂ū∂t_data, M, :) .+ ϵ .* randn.()
# Ū = reshape(train.ū_data, M, :)
# ∂Ū∂t = reshape(train.∂ū∂t_data, M, :)

# Fit non-intrusively using least squares on snapshots
λ = 1e-8 * size(Ū, 2)
Ā_ls = (∂Ū∂t * Ū' + λ * Aᴹ) / (Ū * Ū' + λ * I)
# callback(Ā_ls, loss(Ā_ls))
plotmat(Ā_ls)

# W solution to: min ||WU - Ū||₂² + λ ||W||₂²
λ = 1e-8 * size(U, 2)
W = (Ū * U') / (U * U' + λ * I)
# Wpat = [mapreduce(ℓ -> abs(x + ℓ - ξ) ≤ h(x), |, (-L, 0, L)) for x ∈ x, ξ ∈ ξ]
# W = Wpat ./ sum(Wpat, dims = 2)
plotmat(W)
sum(W; dims = 2)

# min ||RŪ - U||₂² + λ ||R||₂²
# R = inv(W)
λ = 1e-10 * M
R = (W'W + λ * I) \ W'
# λ = 1e-4 * size(U, 2)
# R = (U * Ū') / (Ū * Ū' + λ * I)
plotmat(R)
# sum(R; dims = 2)

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

# eigenvalues
plotmat(symm(Ā); title = "(Ā + Ā') / 2")
plotmat(symm(Ā_ls); title = "(Ā + Ā') / 2")
plotmat(skew(Ā - Aᴹ); title = "(Ā - Ā') / 2")
plotmat(skew(Ā_ls - Aᴹ); title = "(Ā - Ā') / 2")

# Deviation from unfiltered operator
norm(Ā - Aᴹ) / norm(Aᴹ)
norm(Ā_ls - Aᴹ) / norm(Aᴹ)

# Performance on training time steps
relerr(Aᴹ, train.ū₀_data, train.ū_data)
relerr(Aᴹ, test.ū₀_data, test.ū_data)
relerr(Ā, train.ū₀_data, train.ū_data)
relerr(Ā, test.ū₀_data, test.ū_data)
relerr(Ā_ls, train.ū₀_data, train.ū_data)
relerr(Ā_ls, test.ū₀_data, test.ū_data)

# Performance outside of training time interval
tnew = 10.0T
train_exact = mapreduce(u -> u.(x), hcat, train.ū(tnew))
test_exact = mapreduce(u -> u.(x), hcat, test.ū(tnew))
relerr(Aᴹ, train.ū₀_data, train_exact, tnew)
relerr(Aᴹ, test.ū₀_data, test_exact, tnew)
relerr(Ā, train.ū₀_data, train_exact, tnew)
relerr(Ā, test.ū₀_data, test_exact, tnew)
relerr(Ā_ls, train.ū₀_data, train_exact, tnew)
relerr(Ā_ls, test.ū₀_data, test_exact, tnew)
