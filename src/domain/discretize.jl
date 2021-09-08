"""
    discretize(domain, N)

Discretize domain with `N` points.
"""
function discretize end

discretize(domain::ClosedIntervalDomain, N) = LinRange(domain.left, domain.right, N + 1)

function discretize(domain::PeriodicIntervalDomain, N)
    LinRange(domain.left + (domain.right - domain.left) / N, domain.right, N)
end
