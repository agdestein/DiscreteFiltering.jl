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

    ϕ = chebyshevt.(0:degmax, [domain])
    ϕ_int = integrate.(ϕ)

    W = spzeros(n + 1, n + 1)
    for i = 1:n+1
        # Point
        xᵢ = x[i]

        # Filter width at point
        hᵢ = h(xᵢ)

        # Indices of integration points inside domaain
        Ival = (xᵢ ± hᵢ) ∩ (domain.left..domain.right)
        Ival_length = Ival.right - Ival.left
        inds = x .∈ Ival
        deg = min(degmax, max(1, floor(Int, √(sum(inds) - 1))))

        # Polynomials evaluated at integration points
        Vᵢ = zeros(deg + 1, length(x[inds]))

        # Polynomial moments around point
        μᵢ = zeros(deg + 1)

        # Fill in
        for d = 1:deg+1
            Vᵢ[d, :] = ϕ[d].(x[inds])
            μᵢ[d] = ϕ_int[d](Ival.right) - ϕ_int[d](Ival.left)
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

    degmax = 100
    ϕ = chebyshevt.(0:degmax, domain)
    ϕ_int = integrate.(ϕ)

    W = spzeros(n, n)
    for i = 1:n
        # Point
        xᵢ = x[i]

        # Filter width at point
        hᵢ = h(xᵢ)

        # Indices of integration points in circular reference
        Ival = xᵢ ± hᵢ
        inds_left = x .- L .∈ Ival
        inds_mid = x .∈ Ival
        inds_right = x .+ L .∈ Ival
        inds = inds_left .| inds_mid .| inds_right
        deg = min(degmax, max(1, floor(Int, √(sum(inds) - 1))))

        # Polynomials evaluated at integration points
        Vᵢ = zeros(deg + 1, length(x[inds]))

        # Polynomial moments around point
        D = (domain.left..domain.right)
        I_left = Interval((endpoints(Ival) .+ L)...) ∩ D
        I_mid = Ival ∩ D
        I_right = Interval((endpoints(Ival) .- L)...) ∩ D

        # Fill in
        μᵢ = zeros(deg + 1)
        for d = 1:deg+1
            Vᵢ[d, :] = ϕ[d].(x[inds])
            μᵢ[d] += ϕ_int[d](I_left.right) - ϕ_int[d](I_left.left)
            μᵢ[d] += ϕ_int[d](I_mid.right) - ϕ_int[d](I_mid.left)
            μᵢ[d] += ϕ_int[d](I_right.right) - ϕ_int[d](I_right.left)
        end
        μᵢ .*= L / 2hᵢ

        # Fit weights
        wᵢ = nonneg_lsq(Vᵢ, μᵢ)
        # wᵢ = Vᵢ \ μᵢ

        # Store weights
        W[i, inds] .= wᵢ[:] ./ sum(wᵢ)
    end

    W
end


function filter_matrix(f::ConvolutionalFilter, domain::ClosedIntervalDomain, n)

    degmax = 100

    L = (domain.right - domain.left)
    mid = (domain.left + domain.right) / 2

    h = f.width
    G = f.kernel
    x = discretize(domain, n)
    W = spzeros(n + 1, n + 1)

    for i = 1:n+1
        # Point
        xᵢ = x[i]

        # Filter width at point
        hᵢ = h(xᵢ)

        # Indices of integration points inside domaain
        Ival = (xᵢ ± hᵢ)# ∩ (domain.left..domain.right)
        j = x .∈ [Ival]
        xⱼ = x[j]
        n = length(xⱼ)

        # deg = min(degmax, max(1, floor(Int, √(n - 1))))
        deg = min(degmax, n)
        ϕ = chebyshevt(0:deg, Ival)

        # Normalized kernel Fun
        kern = Fun(x -> G(x - xᵢ), Ival)
        kern /= sum(kern)

        # Polynomials evaluated at integration points
        Vᵢ = zeros(deg + 1, n)

        # Polynomial moments around point
        μᵢ = zeros(deg + 1)

        # Fill in
        for d = 1:deg+1
            Vᵢ[d, :] = ϕ[d].(xⱼ)
            μᵢ[d] = sum(kern * ϕ[d])
        end

        # Fit weights
        wᵢ = Vᵢ \ μᵢ
        # wᵢ = nonneg_lsq(Vᵢ, μᵢ)
        # wᵢ = (Vᵢ'Vᵢ + sparse(1e-8I, n, n)) \ Vᵢ'μᵢ

        # Store weights
        W[i, j] .= wᵢ[:] ./ sum(wᵢ)
    end

    W
end


function filter_matrix(f::ConvolutionalFilter, domain::PeriodicIntervalDomain, n)
    degmax = 100

    L = (domain.right - domain.left)
    mid = (domain.left + domain.right) / 2

    h = f.width
    G = f.kernel
    x = discretize(domain, n)
    W = spzeros(n, n)

    for i = 1:n
        # Point
        xᵢ = x[i]

        # Filter width at point
        hᵢ = h(xᵢ)

        # Indices of integration points inside domaain
        Ival = (xᵢ ± hᵢ)
        indsleft = x .+ L .∈ [Ival]
        indsmid = x .∈ [Ival]
        indsright = x .- L .∈ [Ival]
        j = indsleft .| indsmid .| indsright
        xⱼ = (x + L * indsleft - L * indsright)[j]
        n = length(xⱼ)

        # deg = min(degmax, max(1, floor(Int, √(n - 1))))
        deg = min(degmax, sum(inds))
        ϕ = chebyshevt(0:deg, Ival)

        # Normalized kernel Fun
        kern = Fun(x -> G(x - xᵢ), Ival)
        kern /= sum(kern)

        # Polynomials evaluated at integration points
        Vᵢ = zeros(deg + 1, n)

        # Polynomial moments around point
        μᵢ = zeros(deg + 1)

        # Fill in
        for d = 1:deg+1
            Vᵢ[d, :] = ϕ[d].(xⱼ)
            μᵢ[d] = sum(kern * ϕ[d])
        end

        # Fit weights
        # wᵢ = Vᵢ \ μᵢ
        # wᵢ = nonneg_lsq(Vᵢ, μᵢ)
        wᵢ = (Vᵢ'Vᵢ + sparse(1e-8I, n, n)) \ Vᵢ'μᵢ

        # Store weights
        W[i, j] .= wᵢ[:]
    end

    W
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
