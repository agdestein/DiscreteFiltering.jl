using DiscreteFiltering
using Test

L = 10
n = 100
x = LinRange(L / n, L, n)
Δx = x[2] - x[1]

G(σ², x) = 1 / √(2πσ²) * exp(-x^2 / 2σ²)
G(x) = G(0.1^2, x)
f = ConvolutionalFilter(G)
