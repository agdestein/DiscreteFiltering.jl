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
PeriodicIntervalDomain(a, b) = PeriodicIntervalDomain(promote(a, b)...)
