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
C = advection_matrix(closed_interval, n)
@test C isa SparseMatrixCSC
@test size(C) == (n + 1, n + 1)
@test nnz(C) == 2(n + 1)
d = diag(C)
@test all(d[2:end-1] .== 0)
@test d[1] < 0
@test d[end] > 0

# Test with PeriodicIntervalDomain
C = advection_matrix(periodic_interval, n)
@test C isa SparseMatrixCSC
@test size(C) == (n, n)
@test nnz(C) == 2n
@test all(diag(C) .== 0)
@test all(diag(C, -1) .< 0)
@test all(diag(C, +1) .> 0)
@test C[1, end] < 0
@test C[end, 1] > 0
