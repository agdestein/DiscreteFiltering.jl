"""
    advection_matrix(n)

Assemble discrete advection matrix.
"""
function advection_matrix(Δx, n)
    C = spdiagm(-1 => fill(-1.0, n - 1), 1 => fill(1.0, n - 1))
    C[1, end] = -1.0
    C[end, 1] = 1.0
    C ./= 2Δx
    C
end
