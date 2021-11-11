"""
    get_npoint(domain, n)

Get number of points used to discretize `domain`.
"""
get_npoint(::ClosedIntervalDomain, n) = n + 1 
get_npoint(::PeriodicIntervalDomain, n) = n 

