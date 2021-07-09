using DiscreteFiltering
using LinearAlgebra
using SparseArrays
using Test

a = 0.0
b = 2π
n = 500

# Filters
h₀ = 0.05
h(x) = h₀ * (1 - 1 / 2 * cos(x))
G(σ², x) = 1 / √(2πσ²) * exp(-x^2 / 2σ²)
G(x) = G(h₀^2, x)
f = TopHatFilter(h)
g = ConvolutionalFilter(G)

# Domains
closed_interval = ClosedIntervalDomain(a, b)
periodic_interval = PeriodicIntervalDomain(a, b)

# Test TopHatFilter with ClosedIntervalDomain
W = filter_matrix(f, closed_interval, n)
@test W isa SparseMatrixCSC
@test size(W) == (n + 1, n + 1)
@test all(sum(W, dims = 2) .≈ 1)

# Test TopHatFilter with PeriodicIntervalDomain
W = filter_matrix(f, periodic_interval, n)
@test W isa SparseMatrixCSC
@test size(W) == (n, n)
@test_broken all(sum(W, dims = 2) .≈ 1)

# Test ConvolutionalFilter with ClosedIntervalDomain
@test_throws Exception filter_matrix(g, closed_interval, n)
# W = filter_matrix(g, closed_interval, n)
# @test W isa SparseMatrixCSC
# @test size(W) == (n, n)
# @test all(sum(W, dims = 2) .≈ 1)

# Test ConvolutionalFilter with PeriodicIntervalDomain
@test_throws Exception filter_matrix(g, periodic_interval, n)
# W = filter_matrix(g, periodic_interval, n)
# @test W isa SparseMatrixCSC
# @test size(W) == (n + 1, n + 1)
# @test all(sum(W, dims = 2) .≈ 1)
