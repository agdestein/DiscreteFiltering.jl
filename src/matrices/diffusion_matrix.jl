"""
    diffusion_matrix(domain, n)

Assemble discrete diffusion matrix.
"""
function diffusion_matrix end

function diffusion_matrix(domain::ClosedIntervalDomain, n)
    Δx = 1 // n * (domain.right - domain.left)
    inds = [-1, 0, 1]
    stencil = [1, -2, 1] / Δx^2
    diags = [i => fill(s, n + 1 - abs(i)) for (i, s) ∈ zip(inds, stencil)]
    D = spdiagm(diags...)
    D[1, 1] = stencil[1] + stencil[2]
    D[end, end] = stencil[end-1] + stencil[end]
    D
end

function diffusion_matrix(domain::PeriodicIntervalDomain, n)
    Δx = 1 // n * (domain.right - domain.left)
    inds = [-1, 0, 1]
    stencil = [1, -2, 1] / Δx^2
    diags = [i => fill(s, n - abs(i)) for (i, s) ∈ zip(inds, stencil)]
    D = spdiagm(diags...)
    D[1, end] = stencil[1]
    D[end, 1] = stencil[end]
    D
end
