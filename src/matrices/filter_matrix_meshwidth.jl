
"""
filter_matrix_meshwidth(f, domain, N)

Assemble discrete filtering matrix from a continuous filter `f` width constant width
\$h(x) = \\Delta x / 2\$.
"""
function filter_matrix_meshwidth(f::TopHatFilter, domain::PeriodicIntervalDomain, N)
    x = discretize(domain, N)
    Δx = 1 // N * (domain.right - domain.left)
    h = f.width
    h₀ = h(x[1])

    all(≈(h(x[1])), h.(x)) || error("Filter width must be constant")
    if Δx .≈ 2h₀
        # Three point stencil
        inds = [-1, 0, 1]
        stencil = [1 // 24, 11 // 12, 1 // 24]
    elseif Δx .≈ h₀
        # Three point stencil
        inds = [-1, 0, 1]
        stencil = [1 // 6, 4 // 6, 1 // 6]
    else
        error("Filter width must be equal to mesh width")
    end

    # Five point stencil
    # inds = [-2, -1, 0, 1, 2]
    # stencil = [-6 // 2033, 77 // 1440, 863 // 960, 77 // 1440, -6 // 2033]

    # Construct banded matrix
    diags = [i => fill(s, N - abs(i)) for (i, s) ∈ zip(inds, stencil)]
    W = spdiagm(diags...)

    # Periodic extension of three point stencil
    W[1, end] = stencil[1]
    W[end, 1] = stencil[end]

    # Periodic extension of five point stencil
    # W[1, [end - 1, end]] = stencil[[1, 2]]
    # W[2, end] = stencil[1]
    # W[end - 1, 1] = stencil[end]
    # W[end, [1, 2]] = stencil[[end-1, end]]

    W
end


function filter_matrix_meshwidth(f::TopHatFilter, domain::ClosedIntervalDomain, N)
    x = discretize(domain, N)
    Δx = 1 // N * (domain.right - domain.left)
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
    diags = [i => fill(s, N + 1 - abs(i)) for (i, s) ∈ zip(inds, stencil)]
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
