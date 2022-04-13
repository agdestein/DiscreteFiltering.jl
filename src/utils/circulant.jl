"""
    circulant(n, inds, stencil)

Create circulant `SparseMatrixCSC`.
"""
circulant(n, inds, stencil) = spdiagm(
    (i => fill(s, n - abs(i)) for (i, s) ∈ zip(inds, stencil))...,
    (i - sign(i) * n => fill(s, abs(i)) for (i, s) ∈ zip(inds, stencil))...,
)
