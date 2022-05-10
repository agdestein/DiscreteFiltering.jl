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
using FFTW
using OrdinaryDiffEq
using DiffEqFlux
using Plots


u(c, x, t) = real(sum(c * exp(2π * im * k * (x - t)) for (k, c) ∈ zip(-K:K, c)))
∂u∂t(c, x, t) =
    real(-2π * im * sum(c * k * exp(2π * im * k * (x - t)) for (k, c) ∈ zip(-K:K, c)))
ū(Ĝ, c, x, t) = real(sum(
    c * Ĝ(k, x) * exp(2π * im * k * (x - t))
    for (k, c) ∈ zip(-K:K, c))
)
∂ū∂t(Ĝ, c, x, t) = real(sum(
    -2π * im * k * c * Ĝ(k, x) * exp(2π * im * k * (x - t))
    for (k, c) ∈ zip(-K:K, c))
)

# Discretization
M = 100
N = 1000
Nfine = 10000
x = LinRange(0, 1, M + 1)[2:end]
ξ = LinRange(0, 1, N + 1)[2:end]
ξfine = LinRange(0, 1, Nfine + 1)[2:end]
Δx = 1 / M
Δξ = 1 / N
Δξfine = 1 / Nfine

# Filter width
h₀ = 1 / 30
# h(x) = h₀ / 2 * (1 + 3 * sin(π * x) * exp(-2x^2))
h(x) = h₀ * (1 + 1 / 3 * sin(2π * x))

# Top-hat filter
G(x, ξ) = (abs(x - ξ) ≤ h(x)) / 2h(x)
Ĝ(k, x) = k == 0 ? 1.0 : sin(2π * k * h(x)) / (2π * k * h(x))

# Gaussian filter
G(x, ξ) = √(6 / π) / h(x) * exp(-6(x - ξ)^2 / h(x)^2)
Ĝ(k, x) = exp(-4π^2 / 3 * k^2 * h(x)^2)

# DNS operator
Aᴺ = circulant(N, [-1, 1], [1.0, -1.0] / 2Δξ)

# Initial guess for LES operator: Unfiltered
Aᴹ = Matrix(circulant(M, [-1, 1], [1.0, -1.0] / 2Δx))

# Maximum frequency
K = 50

# e(k, x) = exp(2π * im * k * x)
e = [exp(2π * im * k * ξ) for ξ ∈ ξ, k ∈ -K:K]
ē = [Ĝ(k, x) * exp(2π * im * k * x) for x ∈ x, k ∈ -K:K]
ēquad = W * e

eobs = [real.(e) imag.(e)]
ēobs = [real.(ē) imag.(ē)]

Wspec = ēobs * eobs' / (eobs * eobs' + 1e-10I)
plotmat(Wspec)

plotmat((Rspec = eobs * ēobs' / (ēobs * ēobs' + 1e-10I);); clims = (-20,25))
plotmat((Rspec = (Wspec'Wspec + 1e-10I) \ Wspec';); clims = (-20,25))
plotmat(Rspec)

plotmat(Wspec * Rspec)
plotmat(log.(abs.(Wspec * Rspec)))

p = plot()
for (color, i) ∈ enumerate(K+1:K+5)
    plot!(p, ξ, real.(e[:, i]); color, linestyle = :dash, label = "e$(color-1)")
    plot!(p, x, real.(ē[:, i]); color, label = "ē$(color-1)")
    plot!(p, x, real.(ēquad[:, i]); color, linestyle = :dot, label = "ēquad$(color-1)")
    # plot!(p, ξ, real.(efine[:, i]); color, linestyle = :dash, label = "e$(color-1)")
    # plot!(p, ξ, real.(ēfine[:, i]); color, label = "ē$(color-1)")
end
p

e = [exp(2π * im * k * x) for x ∈ x, k ∈ -K:K]
ē = [Ĝ(k, x) * exp(2π * im * k * x) for x ∈ x, k ∈ -K:K]
efine = [exp(2π * im * k * x) for x ∈ ξfine, k ∈ -K:K]
ēfine = [Ĝ(k, x) * exp(2π * im * k * x) for x ∈ ξfine, k ∈ -K:K]
# Φ = e'ē .* Δx
Φ = efine'ēfine .* Δξfine

Φtophat = copy(Φ)
Φgauss = copy(Φ)

plotmat(real.(Φ))
plotmat(imag.(Φ))
plotmat(abs.(Φ))
plotmat(log.(abs.(Φ)))

Φinv = inv(Φ)
Φinv = (Φ'Φ + 1e-4I) \ Φ'
plotmat(real.(Φinv))
plotmat(imag.(Φinv))
plotmat(abs.(Φinv))
plotmat(log.(abs.(Φinv)))

Â = -2π * im * Φ * Diagonal(-K:K) * Φinv
# Â = -2π * im * Φ * Diagonal(-K:K) / Φ
Âshift = Â
Âshift = circshift(Â, (K, K))
plotmat(real.(Âshift))
plotmat(imag.(Âshift))
plotmat(abs.(Âshift))
plotmat(log.(abs.(Âshift)))

Ā_fourier = real.(e * Â * e') .* Δx
plotmat(Aᴹ)
plotmat(Ā_fourier)
plotmat(Ā_ls)
plotmat(real.(e * Â * e') .* Δx)
# plotmat(real.(efine * Â * efine') .* Δξ)
plotmat(abs.(e * Â * e') .* Δx)

scatter(eigen(Ā_ls).values)
scatter!(eigen(Aᴹ).values)
scatter!(eigen(Ā_fourier).values)

p = plot()
for (color, i) ∈ enumerate(K+1:K+5)
    plot!(p, x, real.(e[:, i]); color, linestyle = :dash, label = "e$(color-1)")
    plot!(p, x, real.(ē[:, i]); color, label = "ē$(color-1)")
    # plot!(p, ξ, real.(efine[:, i]); color, linestyle = :dash, label = "e$(color-1)")
    # plot!(p, ξ, real.(ēfine[:, i]); color, label = "ē$(color-1)")
end
p

# Create signal
create_signal(K) = map(k -> (1 + 0.2 * randn()) * exp(2π * im * rand()) / (abs(k) + 5)^1.5, -K:K)
# create_signal(K) = map(k -> (1 + 0.2 * randn()) * exp(2π * im * rand()) * exp(-5 / 6 * max(abs(k), 5), -K:K)
# create_signal(K) = map(k -> (1 + 0.2 * randn()) * exp(2π * im * rand()) * exp(-0.5 * max(abs(k), 10)), -K:K)

function create_data(ē, c, tstops)
    nfreq, nsample = size(c)
    K = nfreq ÷ 2
    nt = length(tstops)
    ū₀ = real.(ē * c)
    ū = zeros(M, nsample, nt)
    ∂ū∂t = zeros(M, nsample, nt)
    for (i, t) ∈ enumerate(tstops)
        Et = [exp(-2π * im * k * t) for k ∈ -K:K]
        ū[:, :, i] = real.(ē * (Et .* c))
        ∂ū∂t[:, :, i] = real.(ē * (-2π * im .* (-K:K) .* Et .* c))
    end
    (; ū₀, ū, ∂ū∂t)
end

## Time (one period)
T = 1.0
tstops = LinRange(0, 1T, 51)[2:end]
nₜ = length(tstops)

ntrain = 10000
ntest = 20
c_train = reduce(hcat, (create_signal(K) for _ = 1:ntrain))
c_test = reduce(hcat, (create_signal(K) for _ = 1:ntest))

scatter(c_test[:, 1:3])
scatter(abs.(c_test[:, 1:3]); yscale = :log10)

train = create_data(ē, c_train, tstops);
test = create_data(ē, c_test, tstops);

plot(train.ū[:, 1, [1, 11, 21]])

p = plot()
for i = 1:3
    # plot!(p, x, real.(e * c_train[:, i]); color = i, linestyle = :dash)
    # plot!(p, x, real.(ē * c_train[:, i]); color = i)
    plot!(p, ξ, real.(efine * c_train[:, i]); color = i, linestyle = :dash)
    plot!(p, ξ, real.(ēfine * c_train[:, i]); color = i)
end
p

# Snapshot matrices
ϵ = 0 # 1e-8
# ϵ = 1e-2
Ū = reshape(train.ū, M, :) .+ ϵ .* randn.()
∂Ū∂t = reshape(train.∂ū∂t, M, :) .+ ϵ .* randn.()
# Ū = reshape(train.ū_data, M, :)
# ∂Ū∂t = reshape(train.∂ū∂t_data, M, :)

# Fit non-intrusively using least squares on snapshots
# λ = 1e-15 * size(Ū, 2)
# λ = 1e-10
# λ = 0
Ā_ls = (∂Ū∂t * Ū') / (Ū * Ū')
# Ā_ls = (∂Ū∂t * Ū' + λ * Aᴹ) / (Ū * Ū' + λ * I)
# Ā_ls = ∂Ū∂t * Ū' / (Ū * Ū' + λ * I)
# callback(Ā_ls, loss(Ā_ls))
plotmat(Ā_ls)

plotmat(Ā_fourier)
plotmat(Ā_ls)

