"""
    inverse_filter_matrix(f, domain, n)

Approximate inverse of discrete filtering matrix, given filter `f`.
"""
function inverse_filter_matrix(::Filter, ::Domain, n)
    error("Not implemented")
end


function inverse_filter_matrix(f::TopHatFilter, domain::ClosedIntervalDomain, n)

    degmax = 30

    x = discretize_uniform(domain, n)

    h = f.width
    R = spzeros(n + 1, n + 1)

    # Get reconstruction weights for each point
    for i = 1:n+1
        # Point
        xᵢ = x[i]

        dists = abs.(x .- xᵢ)

        # Find j such that xᵢ is reachable from xⱼ
        j = dists .< h.(x)

        # Polynomial degree (Taylor series order)
        deg = min(degmax, max(1, floor(Int, √sum(j))))

        # Vandermonde matrix
        d = 1:deg
        xⱼ = x[j]'
        hⱼ = h.(xⱼ)
        aⱼ = max.(xⱼ - hⱼ, domain.left)
        bⱼ = min.(xⱼ + hⱼ, domain.right)
        Δhⱼ = bⱼ - aⱼ

        Vᵢ = @. 1 / (Δhⱼ * factorial(d)) * ((bⱼ - xᵢ)^d - (aⱼ - xᵢ)^d)

        # Right-hand side
        μᵢ = fill(0.0, deg)
        μᵢ[1] = 1.0

        # Fit weights
        rᵢ = Vᵢ \ μᵢ

        # Store weights
        R[i, j] .= rᵢ[:]
    end

    R
end


function inverse_filter_matrix(f::TopHatFilter, domain::PeriodicIntervalDomain, n)

    degmax = 30

    x = discretize_uniform(domain, n)

    h = f.width
    R = spzeros(n, n)

    # Get reconstruction weights for each point
    for i = 1:n
        # Point
        xᵢ = x[i]

        dists = @. abs(xᵢ - x - [-2π 0 2π])

        # Move x by 2π * (shifts - 2) to get closer to xᵢ
        mininds = argmin(dists, dims = 2)
        shifts = [mininds[j].I[2] for j in eachindex(mininds)]

        # Find j such that xᵢ is reachable from xⱼ
        j = dists[mininds][:] .< h.(x)

        # Polynomial degree (Taylor series order)
        deg = min(degmax, max(1, floor(Int, √sum(j))))

        # Vandermonde matrix
        d = 1:deg
        xⱼ = x[j]'
        hⱼ = h.(xⱼ)
        sⱼ = 2π * (shifts[j]' .- 2)

        Vᵢ = @. 1 / (2hⱼ * factorial(d)) * ((xⱼ + sⱼ + hⱼ - xᵢ)^d - (xⱼ + sⱼ - hⱼ - xᵢ)^d)

        # Right-hand side
        μᵢ = fill(0.0, deg)
        μᵢ[1] = 1.0

        # Fit weights
        rᵢ = Vᵢ \ μᵢ

        # Store weights
        R[i, j] .= rᵢ[:]
    end

    R
end


function inverse_filter_matrix(f::ConvolutionalFilter, domain::ClosedIntervalDomain, n)
    error("Not implemented")
    # G = f.kernel
    # R = spzeros(n + 1, n + 1)
    # R
end


function inverse_filter_matrix(f::ConvolutionalFilter, domain::PeriodicIntervalDomain, n)
    error("Not implemented")
    # G = f.kernel
    # R = spzeros(n, n)
    # R
end


function inverse_filter_matrix_meshwidth(f::TopHatFilter, domain::PeriodicIntervalDomain, n)
    L = (domain.right - domain.left)
    mid = (domain.left + domain.right) / 2
    x = discretize_uniform(domain, n)
    Δx = x[2] - x[1]
    h = f.width
    h₀ = h(x[1])

    all(isequal(h(x[1])), h.(x)) || error("Filter width must be constant")
    Δx .≈ 2h₀ || error("Filter width must be equal to mesh width")

    # Three point stencil
    inds = [-1, 0, 1]
    stencil = [-1 / 24, 13 / 12, -1 / 24]

    # Five point stencil
    # inds = [-2, -1, 0, 1, 2]
    # stencil = [3 / 640, -29 / 480, 1067 / 960, -29 / 480, 3 / 640]

    # Construct banded matrix
    diags = [i => fill(s, n + 1 - abs(i)) for (i, s) ∈ zip(inds, stencil)]
    R = spdiagm(diags...)

    # Periodic extension of three point stencil
    R[1, end] = -1 / 24
    R[end, 1] = -1 / 24

    # Periodic extension of five point stencil
    # R[1, [end - 1, end]] = [3 / 640, -29 / 480]
    # R[2, end] = 3 / 640
    # R[end - 1, 1] = 3 / 640
    # R[end, [1, 2]] = [-29 / 480, 3 / 640]

    R
end


function inverse_filter_matrix_meshwidth(f::TopHatFilter, domain::ClosedIntervalDomain, n)
    L = (domain.right - domain.left)
    mid = (domain.left + domain.right) / 2
    x = discretize_uniform(domain, n)
    Δx = x[2] - x[1]
    h = f.width
    h₀ = h(x[1])

    all(isequal(h(x[1])), h.(x)) || error("Filter width must be constant")
    Δx .≈ 2h₀ || error("Filter width must be equal to mesh width")

    # Three point stencil
    inds = [-1, 0, 1]
    stencil = [-1 / 24, 13 / 12, -1 / 24]

    # Five point stencil
    # inds = [-2, -1, 0, 1, 2]
    # stencil = [3 / 640, -29 / 480, 1067 / 960, -29 / 480, 3 / 640]

    # Construct banded matrix
    diags = [i => fill(s, n + 1 - abs(i)) for (i, s) ∈ zip(inds, stencil)]
    R = spdiagm(diags...)

    # Boundary weights for three point stencil
    R[1, [1, 2]] = [4 / 3, -1 / 3]
    R[end, [end, end - 1]] = [4 / 3, -1 / 3]

    # Boundary weights for five point stencil
    # R[1, [1, 2, 3]] = [23 / 15, -41 / 60, 3 / 20]
    # R[2, [1, 2, 3, 4]] = [-8/105, 319/280, -29/420, 1/168]
    # R[end - 1, [end, end - 1, end - 2, end - 3]] = [-8/105, 319/280, -29/420, 1/168]
    # R[end, [end, end - 1, end - 2]] = [23 / 15, -41 / 60, 3 / 20]

    R
end
