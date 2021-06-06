using DiscreteFiltering
using LinearAlgebra
using SparseArrays
using Test

L = 10
n = 100
x = LinRange(L / n, L, n)
Δx = x[2] - x[1]

D = diffusion_matrix(Δx, n)
@test D isa SparseMatrixCSC
@test size(D) == (n, n)
@test nnz(D) == 3n
