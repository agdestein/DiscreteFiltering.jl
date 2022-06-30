if isdefined(@__MODULE__, :LanguageServer)
    include("../src/DiscreteFiltering.jl")
    using .DiscreteFiltering
end

using Pkg
Pkg.activate(".")
cd("examples")

using DiscreteFiltering
using JLD2
using LinearAlgebra
using SciMLSensitivity
using SparseArrays
using Plots
using LaTeXStrings
using Random
using FFTW
using Printf

# create_magnitude(k) = (1 + 0.2 * randn()) * (k == 0 ? 1 : abs(k)^(-3/2))
# create_magnitude(k) = (1 + 0.2 * randn()) * (5 + abs(k))^(-3 / 2)
create_magnitude(k) = (1 + 0.2 * randn()) * (5 + abs(k))^(-2)
# create_magnitude(k) = (1 - 0.9 * rand()) / (5 + abs(k))^1.5
function create_signal(K)
    c₀ = create_magnitude(0)
    c₊ = map(k -> create_magnitude(k) * exp(2π * im * rand()), 1:K)
    c₋ = reverse(conj.(c₊))
    [c₋; c₀; c₊]
end

# Continuous data
K = 250
n_train = 1000
n_valid = 20
n_test = 100
n_long = 50
Random.seed!(0)
c_train = reduce(hcat, (create_signal(K) for _ = 1:n_train))
c_valid = reduce(hcat, (create_signal(K) for _ = 1:n_valid))
c_test = reduce(hcat, (create_signal(K) for _ = 1:n_test))
c_long = reduce(hcat, (create_signal(K) for _ = 1:n_long))

# jldsave("output/K$K/coefficients.jld2"; c_train, c_valid, c_test)
c_train, c_valid, c_test =
    load("output/K$K/coefficients.jld2", "c_train", "c_valid", "c_test")

# DNS discretization
N = 1000
ξ = LinRange(0, 1, N + 1)[2:end]
Δξ = 1 / N

# Stencils
s₂ = [1, 0, -1] / 2Δξ # 2nd order
s₄ = [-1, 8, 0, -8, 1] / 12Δξ # 4th order
s₆ = [1, -9, 45, 0, -45, 9, -1] / 60Δξ # 6th order
s₈ = [-3, 32, -168, 672, 0, -672, 168, -32, 3] / 840Δξ # 8th order 
s₁₀ =
    [
        5334336,
        -66679200,
        400075200,
        -1600300800,
        5601052800,
        0,
        -5601052800,
        1600300800,
        -400075200,
        66679200,
        -5334336,
    ] / 6721263360Δξ # 10th order 

# DNS operators
Aᴺ₂ = circulant(N, -1:1, s₂)
Aᴺ₄ = circulant(N, -2:2, s₄)
Aᴺ₆ = circulant(N, -3:3, s₆)
Aᴺ₈ = circulant(N, -4:4, s₈)
Aᴺ₁₀ = circulant(N, -5:5, s₁₀)

# Aᴺ = Aᴺ₂
# Aᴺ = Aᴺ₄
Aᴺ = Aᴺ₆
# Aᴺ = Aᴺ₈
# Aᴺ = Aᴺ₁₀

# stencil = s₂
# stencil = s₄
stencil = s₆
# stencil = s₈
# stencil = s₁₀

# Time (T = one period)
T = 1.0
t_train = LinRange(0, 0.1T, 51)
t_valid = LinRange(0, 1T, 11)
t_test = LinRange(0, 1T, 21)
t_long = [0; 10 .^ LinRange(-2, 2, 500)]

# DNS solution: Solve discrete DNS equations
dns_train = create_data_dns(Aᴺ, c_train, ξ, t_train)
dns_valid = create_data_dns(Aᴺ, c_valid, ξ, t_valid)
dns_test = create_data_dns(Aᴺ, c_test, ξ, t_test)
dns_long = create_data_dns(Aᴺ, c_long, ξ, t_long)

filename = "output/K$(K)/dns.jld2"
# @info("Saving DNS to \"$(pwd())/$filename\"")
# jldsave(filename; dns_train, dns_valid, dns_test, dns_long)

@info("Loading DNS from \"$(pwd())/$filename\"")
dns_train, dns_valid, dns_test, dns_long =
    load(filename, "dns_train", "dns_valid", "dns_test", "dns_long")

# # DNS solution: approximate
# dns_train = create_data_dns_stencil(stencil, c_train, ξ, t_train)
# dns_valid = create_data_dns_stencil(stencil, c_valid, ξ, t_valid)
# dns_test = create_data_dns_stencil(stencil, c_test, ξ, t_test)
# dns_long = create_data_dns_stencil(stencil, c_long, ξ, t_long)

# DNS solution: From continuous exact solutions
dns_train = create_data_exact(c_train, ξ, t_train)
dns_valid = create_data_exact(c_valid, ξ, t_valid)
dns_test = create_data_exact(c_test, ξ, t_test)
dns_long = create_data_exact(c_long, ξ, t_long)

float(Base.summarysize(dns_train))
float(Base.summarysize(dns_long))

# Filter width
h₀ = 1 / 50
# h(x) = (1 + 3 * sin(π * x) * exp(-2x^2)) * h₀ / 2
h(x) = (1 + 1 / 3 * sin(2π * x)) * h₀
dh(x) = 2π / 3 * cos(2π * x) * h₀
# h(x) = h₀
# dh(x) = zero(x)

# Top-hat filter
tophat = create_tophat(h)
F = tophat

# Gaussian filter
gaussian = create_gaussian(h)
F = gaussian

# Fine discretization
Nfine = 10000
ξfine = LinRange(0, 1, Nfine + 1)[2:end]
Δξfine = 1 / Nfine
efine = [exp(2π * im * k * x) for x ∈ ξfine, k ∈ -K:K]
ēfine = [F.Ĝ(k, x) * exp(2π * im * k * x) for x ∈ ξfine, k ∈ -K:K]
Φ = efine'ēfine .* Δξfine
# Φinv = inv(Φ)
Φinv = (Φ'Φ + 1e-8I) \ Φ'
# Â = -2π * im * Φ * Diagonal(-K:K) * Φinv
Â = -2π * im * Φ * Diagonal(-K:K) / Φ

ēfine_tophat = [tophat.Ĝ(k, x) * exp(2π * im * k * x) for x ∈ ξfine, k ∈ -K:K]
Φ_tophat = efine'ēfine_tophat .* Δξfine
ēfine_gaussian = [gaussian.Ĝ(k, x) * exp(2π * im * k * x) for x ∈ ξfine, k ∈ -K:K]
Φ_gaussian = efine'ēfine_gaussian .* Δξfine
