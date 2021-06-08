using DiscreteFiltering
using LinearAlgebra
using SparseArrays
using Test

L = 2π
n = 100
x = LinRange(L / n, L, n)

# Test TopHatFilter matrix
h(x) = 1 - 1 / 2 * cos(x)
filter = TopHatFilter(h)
W = filter_matrix(filter, x)
@test W isa SparseMatrixCSC
@test size(W) == (n, n)
@test all(sum(W, dims = 2) .== 1)

# Test ConvolutionalFilter matrix
G(σ², x) = 1 / √(2πσ²) * exp(-x^2 / 2σ²)
G(x) = G(0.1^2, x)
filter = ConvolutionalFilter(G)
W = filter_matrix(filter, x)
@test W isa SparseMatrixCSC
@test size(W) == (n, n)
@test all(sum(W, dims = 2) .== 1)
