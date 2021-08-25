"""
    diffusion_matrix(domain, n)

Assemble discrete diffusion matrix.
"""
function diffusion_matrix end

function diffusion_matrix(domain::ClosedIntervalDomain, n)
    Δx = (domain.right - domain.left) / n
    D = spdiagm(-1 => fill(1.0, n), 0 => fill(-2.0, n + 1), 1 => fill(1.0, n))
    D[1, 1] = -1.0
    D[end, end] = -1.0
    D ./= Δx^2
    D
end

function diffusion_matrix(domain::PeriodicIntervalDomain, n)
    Δx = (domain.right - domain.left) / n
    D = spdiagm(-1 => fill(1.0, n - 1), 0 => fill(-2.0, n), 1 => fill(1.0, n - 1))
    D[1, end] = 1.0
    D[end, 1] = 1.0
    D ./= Δx^2
    D
end
