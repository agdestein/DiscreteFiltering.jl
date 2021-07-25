"""
    filter_matrix(f, domain, n)

Assemble discrete filtering matrix from a continuous filter `f`.
"""
function filter_matrix(::Filter, ::Domain, n)
    error("Not implemented")
end


function filter_matrix(::IdentityFilter, ::ClosedIntervalDomain, n)
    sparse(I, n + 1, n + 1)
end


function filter_matrix(::IdentityFilter, ::PeriodicIntervalDomain, n)
    sparse(I, n, n)
end


function filter_matrix(f::TopHatFilter, domain::ClosedIntervalDomain, n)

    degmax = 100

    L = (domain.right - domain.left)
    mid = (domain.left + domain.right) / 2

    x = discretize(domain, n)
    h = f.width

    if all(h.(x) .≈ h(x[1])) && x[2] - x[1] ≈ 2h(x[1])
        return filter_matrix_meshwidth(f, domain, n)
    end

    τ(x) = (x - mid) / L
    τ(x, a, b) = (x - (a + b) / 2) / (b - a)
    ϕ = chebyshevt.(0:degmax)
    ϕ_int = integrate.(ϕ)

    W = spzeros(n + 1, n + 1)
    for i = 1:n+1
        # Point
        xᵢ = x[i]

        # Filter width at point
        hᵢ = h(xᵢ)

        # Indices of integration points inside domaain
        Ival = Interval(xᵢ - hᵢ, xᵢ + hᵢ) ∩ Interval(domain.left, domain.right)
        Ival_length = Ival.last - Ival.first
        inds = x .∈ Ival
        deg = min(degmax, max(1, floor(Int, √(sum(inds) - 1))))

        # Polynomials evaluated at integration points
        Vᵢ = zeros(deg + 1, length(x[inds]))

        # Polynomial moments around point
        μᵢ = zeros(deg + 1)

        # Fill in
        for d = 1:deg+1
            Vᵢ[d, :] = ϕ[d].(τ.(x[inds]))
            μᵢ[d] = ϕ_int[d](τ(Ival.last)) - ϕ_int[d](τ(Ival.first))
        end
        μᵢ .*= L / Ival_length

        # Fit weights
        wᵢ = nonneg_lsq(Vᵢ, μᵢ)

        # Store weights
        W[i, inds] .= wᵢ[:]
    end

    W
end


function filter_matrix(f::TopHatFilter, domain::PeriodicIntervalDomain, n)

    L = (domain.right - domain.left)
    mid = (domain.left + domain.right) / 2

    x = discretize(domain, n)
    h = f.width

    if all(h.(x) .≈ h(x[1])) && x[2] - x[1] .≈ 2h(x[1])
        return filter_matrix_meshwidth(f, domain, n)
    end

    τ(x) = (x - mid) / L
    τ(x, a, b) = (x - (a + b) / 2) / (b - a)
    degmax = 100
    ϕ = chebyshevt.(0:degmax)
    ϕ_int = integrate.(ϕ)

    W = spzeros(n, n)
    for i = 1:n
        # Point
        xᵢ = x[i]

        # Filter width at point
        hᵢ = h(xᵢ)

        # Indices of integration points in circular reference
        Ival = Interval(xᵢ - hᵢ, xᵢ + hᵢ)
        inds_left = x .∈ Ival + L
        inds_mid = x .∈ Ival
        inds_right = x .∈ Ival - L
        inds = inds_left .| inds_mid .| inds_right
        deg = min(degmax, max(1, floor(Int, √(sum(inds) - 1))))

        # Polynomials evaluated at integration points
        Vᵢ = zeros(deg + 1, length(x[inds]))

        # Polynomial moments around point
        Domain = Interval(0, 2π)
        I_left = (Ival + 2π) ∩ Domain
        I_mid = Ival ∩ Domain
        I_right = (Ival - 2π) ∩ Domain

        # Fill in
        μᵢ = zeros(deg + 1)
        for d = 1:deg+1
            Vᵢ[d, :] = ϕ[d].(τ.(x[inds]))
            μᵢ[d] += ϕ_int[d](τ(I_left.last)) - ϕ_int[d](τ(I_left.first))
            μᵢ[d] += ϕ_int[d](τ(I_mid.last)) - ϕ_int[d](τ(I_mid.first))
            μᵢ[d] += ϕ_int[d](τ(I_right.last)) - ϕ_int[d](τ(I_right.first))
        end
        μᵢ .*= L / 2hᵢ

        # Fit weights
        wᵢ = nonneg_lsq(Vᵢ, μᵢ)
        # wᵢ = Vᵢ \ μᵢ

        # Store weights
        W[i, inds] .= wᵢ[:]
    end

    W
end


function filter_matrix(f::ConvolutionalFilter, domain::ClosedIntervalDomain, n)
    error("Not implemented")
    # G = f.kernel
    # W = spzeros(n + 1, n + 1)
    # W
end


function filter_matrix(f::ConvolutionalFilter, domain::PeriodicIntervalDomain, n)
    error("Not implemented")
    # G = f.kernel
    # W = spzeros(n, n)
    # W
end


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

    all(h.(x) .≈ h(x[1])) || error("Filter width must be constant")
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

    all(h.(x) .≈ h(x[1])) || error("Filter width must be constant")
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
