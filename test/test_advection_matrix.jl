using DiscreteFiltering
using LinearAlgebra
using SparseArrays
using Test


L = 10
n = 100
x = LinRange(L / n, L, n)
Δx = x[2] - x[1]

C = advection_matrix(Δx, n)
@test C isa SparseMatrixCSC
@test size(C) == (n, n)
@test nnz(C) == 2n
@test all(diag(C) .== 0)
