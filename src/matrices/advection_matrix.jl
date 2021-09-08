"""
    advection_matrix(domain, n)

Assemble discrete advection matrix.
"""
function advection_matrix end

function advection_matrix(domain::ClosedIntervalDomain, n)
    Δx = 1 // n * (domain.right - domain.left)
    inds = [-1, 1]
    stencil = [-1 // 2, 1 // 2] / Δx
    diags = [i => fill(s, n + 1 - abs(i)) for (i, s) ∈ zip(inds, stencil)]
    C = spdiagm(diags...)
    C[1, 1] = 2stencil[1]
    C[1, 2] = 2stencil[2]
    C[end, end-1] = 2stencil[1]
    C[end, end] = 2stencil[2]
    C
end

function advection_matrix(domain::PeriodicIntervalDomain, n)
    Δx = 1 // n * (domain.right - domain.left)
    inds = [-1, 1]
    stencil = [-1 // 2, 1 // 2] / Δx
    diags = [i => fill(s, n - abs(i)) for (i, s) ∈ zip(inds, stencil)]
    C = spdiagm(diags...)
    C[1, end] = stencil[1]
    C[end, 1] = stencil[2]
    C
end
