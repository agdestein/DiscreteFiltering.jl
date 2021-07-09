using DiscreteFiltering
using LinearAlgebra
using SparseArrays
using Test

a = 0.0
b = 2π
n = 100

# Domains
closed_interval = ClosedIntervalDomain(a, b)
periodic_interval = PeriodicIntervalDomain(a, b)

# Test with ClosedIntervalDomain
D = diffusion_matrix(closed_interval, n)
@test D isa SparseMatrixCSC
@test size(D) == (n + 1, n + 1)
@test nnz(D) == 3(n + 1) - 2
@test all(diag(D) .< 0)

# Test with PeriodicIntervalDomain
D = diffusion_matrix(periodic_interval, n)
@test D isa SparseMatrixCSC
@test size(D) == (n, n)
@test nnz(D) == 3n
@test all(diag(D, -1) .> 0)
@test all(diag(D) .< 0)
@test all(diag(D, +1) .> 0)
@test D[1, end] > 0
@test D[end, 1] > 0
