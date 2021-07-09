using DiscreteFiltering
using LinearAlgebra
using SparseArrays
using Test

a = 0.0
b = 2π
n = 100

# Filters
h(x) = 1 - 1 / 2 * cos(x)
G(σ², x) = 1 / √(2πσ²) * exp(-x^2 / 2σ²)
G(x) = G(0.1^2, x)
f = TopHatFilter(h)
g = ConvolutionalFilter(G)

# Domains
closed_interval = ClosedIntervalDomain(a, b)
periodic_interval = PeriodicIntervalDomain(a, b)

# Test TopHatFilter with ClosedIntervalDomain
R = inverse_filter_matrix(f, closed_interval, n)
@test R isa SparseMatrixCSC
@test size(R) == (n + 1, n + 1)
@test all(sum(R, dims = 2) .≈ 1)

# Test TopHatFilter with PeriodicIntervalDomain
R = inverse_filter_matrix(f, periodic_interval, n)
@test R isa SparseMatrixCSC
@test size(R) == (n, n)
@test all(sum(R, dims = 2) .≈ 1)

# Test ConvolutionalFilter with ClosedIntervalDomain
@test_throws Exception inverse_filter_matrix(g, closed_interval, n)
# R = inverse_filter_matrix(g, closed_interval, n)
# @test R isa SparseMatrixCSC
# @test size(R) == (n, n)
# @test all(sum(R, dims = 2) .≈ 1)

# Test ConvolutionalFilter with PeriodicIntervalDomain
@test_throws Exception inverse_filter_matrix(g, periodic_interval, n)
# R = inverse_filter_matrix(g, periodic_interval, n)
# @test R isa SparseMatrixCSC
# @test size(R) == (n + 1, n + 1)
# @test all(sum(R, dims = 2) .≈ 1)
