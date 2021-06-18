"""
    advection_matrix(domain, n)

Assemble discrete advection matrix.
"""
function advection_matrix(::Domain, n)
    error("Not implemented")
end

function advection_matrix(domain::ClosedIntervalDomain, n)
    Δx = (domain.right - domain.left) / n
    C = spdiagm(-1 => fill(-1 / 2, n), 1 => fill(1 / 2, n))
    C[1, 1] = -1.0
    C[1, 2] = 1.0
    C[end, end-1] = -1.0
    C[end, end] = 1.0
    C ./= Δx
    C
end


function advection_matrix(domain::PeriodicIntervalDomain, n)
    Δx = (domain.right - domain.left) / n
    C = spdiagm(-1 => fill(-1.0, n - 1), 1 => fill(1.0, n - 1))
    C[1, end] = -1.0
    C[end, 1] = 1.0
    C ./= 2Δx
    C
end
