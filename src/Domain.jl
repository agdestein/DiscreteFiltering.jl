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
struct ClosedIntervalDomain{T<:Number} <: AbstractIntervalDomain
    left::T
    right::T
end


"""
    PeriodicIntervalDomain(left, right)

Periodic interval domain.
"""
struct PeriodicIntervalDomain{T<:Number} <: AbstractIntervalDomain
    left::T
    right::T
end


"""
    discretize(domain)

Discretize domain.
"""
function discretize_uniform(::Domain)
    error("Not implemented")
end


function discretize_uniform(domain::ClosedIntervalDomain, n)
    LinRange(domain.left, domain.right, n + 1)
end


function discretize_uniform(domain::PeriodicIntervalDomain, n)
    LinRange(domain.left + (domain.right - domain.left) / n, domain.right, n)
end
