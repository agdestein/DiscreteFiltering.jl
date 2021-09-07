
"""
reconstruction_matrix_meshwidth(f, domain, n)

Assemble inverse discrete filtering matrix from a continuous filter `f` width constant width
\$h(x) = \\Delta x / 2\$.
"""
function reconstruction_matrix_meshwidth(f::TopHatFilter, domain::PeriodicIntervalDomain, n)
    x = discretize(domain, n)
    Δx = 1 // n * (domain.right - domain.left)
    h = f.width
    h₀ = h(x[1])

    all(h.(x) .≈ h(x[1])) || error("Filter width must be constant")

    if Δx .≈ 2h₀
        # Three point stencil
        inds = [-1, 0, 1]
        stencil = [-1 // 24, 13 // 12, -1 // 24]
    elseif Δx .≈ h₀
        # Three point stencil
        inds = [-1, 0, 1]
        stencil = [-1 // 6, 8 // 6, -1 // 6]
    else
        error("Filter width must be equal to mesh width")
    end

    # Five point stencil
    # inds = [-2, -1, 0, 1, 2]
    # stencil = [3//640, -29//480, 1067//960, -29//480, 3//640]

    # Construct banded matrix
    diags = [i => fill(s, n - abs(i)) for (i, s) ∈ zip(inds, stencil)]
    R = spdiagm(diags...)

    # Periodic extension of three point stencil
    R[1, end] = stencil[1]
    R[end, 1] = stencil[end]

    # Periodic extension of five point stencil
    # R[1, [end - 1, end]] = stencil[[1, 2]]
    # R[2, end] = stencil[1]
    # R[end - 1, 1] = stencil[end]
    # R[end, [1, 2]] = stencil[[end-1, end]]

    R
end


function reconstruction_matrix_meshwidth(f::TopHatFilter, domain::ClosedIntervalDomain, n)
    x = discretize(domain, n)
    Δx = 1 // n * (domain.right - domain.left)
    h = f.width
    h₀ = h(x[1])

    all(h.(x) .≈ h(x[1])) || error("Filter width must be constant")
    Δx .≈ 2h₀ || error("Filter width must be equal to mesh width")

    # Three point stencil
    inds = [-1, 0, 1]
    stencil = [-1 // 24, 13 // 12, -1 // 24]

    # Five point stencil
    # inds = [-2, -1, 0, 1, 2]
    # stencil = [3 // 640, -29 // 480, 1067 // 960, -29 // 480, 3 // 640]

    # Construct banded matrix
    diags = [i => fill(s, n + 1 - abs(i)) for (i, s) ∈ zip(inds, stencil)]
    R = spdiagm(diags...)

    # Boundary weights for three point stencil
    R[1, [1, 2]] = [4 // 3, -1 // 3]
    R[end, [end, end - 1]] = [4 // 3, -1 // 3]

    # Boundary weights for five point stencil
    # R[1, [1, 2, 3]] = [23 // 15, -41 // 60, 3 // 20]
    # R[2, [1, 2, 3, 4]] = [-8//105, 319//280, -29//420, 1//168]
    # R[end - 1, [end, end - 1, end - 2, end - 3]] = [-8//105, 319//280, -29//420, 1//168]
    # R[end, [end, end - 1, end - 2]] = [23 // 15, -41 // 60, 3 // 20]

    R
end
