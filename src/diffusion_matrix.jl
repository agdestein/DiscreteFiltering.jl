"""
    diffusion_matrix(n)

Assemble discrete diffusion matrix.
"""
function diffusion_matrix(Δx, n)
    D = spdiagm(-1 => fill(1.0, n - 1), 0 => fill(-2.0, n), 1 => fill(1.0, n - 1))
    D[1, end] = 1.0
    D[end, 1] = 1.0
    D ./= Δx^2
    D
end
