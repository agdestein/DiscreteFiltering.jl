
"""
filter_matrix_meshwidth(f, domain, n)

Assemble discrete filtering matrix from a continuous filter `f` width constant width
\$h(x) = \\Delta x / 2\$.
"""
function filter_matrix_meshwidth(f::TopHatFilter, domain::PeriodicIntervalDomain, n)
    L = (domain.right - domain.left)
    mid = (domain.left + domain.right) / 2
    x = discretize(domain, n)
    Δx = x[2] - x[1]
    h = f.width
    h₀ = h(x[1])

    all(≈(h(x[1])), h.(x)) || error("Filter width must be constant")
    Δx .≈ 2h₀ || error("Filter width must be equal to mesh width")

    # Three point stencil
    inds = [-1, 0, 1]
    stencil = [1 / 24, 11 / 12, 1 / 24]

    # Five point stencil
    # inds = [-2, -1, 0, 1, 2]
    # stencil = [-6 / 2033, 77 / 1440, 863 / 960, 77 / 1440, -6 / 2033]

    # Construct banded matrix
    diags = [i => fill(s, n - abs(i)) for (i, s) ∈ zip(inds, stencil)]
    W = spdiagm(diags...)

    # Periodic extension of three point stencil
    W[1, end] = 1 / 24
    W[end, 1] = 1 / 24

    # Periodic extension of five point stencil
    # W[1, [end - 1, end]] = [-6 / 2033, 77 / 1440]
    # W[2, end] = -6 / 2033
    # W[end - 1, 1] = -6 / 2033
    # W[end, [1, 2]] = [77 / 1440, -6 / 2033]

    W
end


function filter_matrix_meshwidth(f::TopHatFilter, domain::ClosedIntervalDomain, n)
    L = (domain.right - domain.left)
    mid = (domain.left + domain.right) / 2
    x = discretize(domain, n)
    Δx = x[2] - x[1]
    h = f.width
    h₀ = h(x[1])

    all(≈(h(x[1])), h.(x)) || error("Filter width must be constant")
    Δx ≈ 2h₀ || error("Filter width must be equal to mesh width")

    # Three point stencil
    inds = [-1, 0, 1]
    stencil = [1 / 24, 11 / 12, 1 / 24]

    # Five point stencil
    # inds = [-2, -1, 0, 1, 2]
    # stencil = [-6 / 2033, 77 / 1440, 863 / 960, 77 / 1440, -6 / 2033]

    # Construct banded matrix
    diags = [i => fill(s, n + 1 - abs(i)) for (i, s) ∈ zip(inds, stencil)]
    W = spdiagm(diags...)

    # Boundary weights for three point stencil
    W[1, [1, 2]] = [3 / 4, 1 / 4]
    W[end, [end, end - 1]] = [3 / 4, 1 / 4]

    # Boundary weights for five point stencil
    # W[1, [1, 2, 3]] = [2 / 3, 5 / 12, -1 / 12]
    # W[2, [1, 2, 3, 4]] = [1/24, 11/12, 1/24, 0]
    # W[end - 1, [end, end - 1, end - 2, end - 3]] = [1/24, 11/12, 1/24, 0]
    # W[end, [end, end - 1, end - 2]] = [2 / 3, 5 / 12, -1 / 12]

    W
end
