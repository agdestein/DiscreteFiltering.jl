"""
    Domain

Abstract type for different domains.
"""
abstract type Domain end


"""
    ClosedIntervalDomain

Abstract type for interval domains.
"""
abstract type AbstractIntervalDomain <: Domain end


"""
    ClosedIntervalDomain(left, right)

Interval domain.
"""
struct ClosedIntervalDomain{T} <: AbstractIntervalDomain
    left::T
    right::T
end


"""
    PeriodicIntervalDomain(left, right)

Periodic interval domain.
"""
struct PeriodicIntervalDomain{T} <: AbstractIntervalDomain
    left::T
    right::T
end


"""
    discretize(domain, n)

Discretize domain with `n` points.
"""
function discretize end

discretize(domain::ClosedIntervalDomain, n) = LinRange(domain.left, domain.right, n + 1)

function discretize(domain::PeriodicIntervalDomain, n)
    LinRange(domain.left + (domain.right - domain.left) / n, domain.right, n)
end
