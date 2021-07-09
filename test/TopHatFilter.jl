using DiscreteFiltering
using Test

L = 10
n = 100
x = LinRange(L / n, L, n)
Δx = x[2] - x[1]

h(x) = 1 - 1 / 2 * cos(x)
f = TopHatFilter(h)
