# LSP indexing solution from
# https://github.com/julia-vscode/julia-vscode/issues/800#issuecomment-650085983
if isdefined(@__MODULE__, :LanguageServer)
    include("../src/DiscreteFiltering.jl")
    using .DiscreteFiltering
end

using DiscreteFiltering
using LinearAlgebra
using FFTW
using Plots

## Parameters

# Domain
a = 0.0
b = 2π
domain = PeriodicIntervalDomain(a, b)

N = 1000
ξ = discretize(domain, N)
k = 0:N
u = map(x -> sum(n -> n * cos(5n * x), 1:5), ξ)
û = fft(u)

scatter(û)
scatter(û[1:30])
bar(eachindex(û).-1, real(û))
bar(0:29, real(û)[1:30])
plot(ξ, real(ifft(real(û))))
plot(ξ, u)


# Amplitudes
N = 30
# c = map(n -> max(0.0, 3 + randn()) * exp(-3(n / N)^2) / N, 0:N-1)
c = map(n -> max(0.0, 3 + randn()), 0:N-1)
bar(c)

# Frequencies
ω = 0:N-1

# Phase-shifts
ϕ = rand(N) * 2π

s = sum_of_sines(domain, c, ω, ϕ) 
plot(ξ, s.u)
