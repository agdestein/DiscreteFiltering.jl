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
