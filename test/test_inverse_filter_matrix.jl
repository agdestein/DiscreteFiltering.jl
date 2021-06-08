using DiscreteFiltering
using LinearAlgebra
using SparseArrays
using Test

L = 2π
n = 100
x = LinRange(L / n, L, n)
h(x) = 1 - 1 / 2 * cos(x)
filter = TopHatFilter(h)

R = inverse_filter_matrix(filter, x)
@test R isa SparseMatrixCSC
@test size(R) == (n, n)
