"""
    filter_matrix(f, x, n)

Assemble discrete filtering matrix from a continuous filter `f`.
"""
function filter_matrix(::Filter, ::Domain, n)
    error("Not implemented")
end


function filter_matrix(f::TopHatFilter, domain::ClosedIntervalDomain, n)

    degmax = 100

    L = (domain.right - domain.left)
    mid = (domain.left + domain.right) / 2

    x = discretize_uniform(domain, n)

    h = f.width
    τ(x) = (x - mid) / L
    τ(x, a, b) = (x - (a + b) / 2) / (b - a)
    ϕ = [ChebyshevT([fill(0, i); 1]) for i = 0:degmax]

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
        Vᵢ = vander(ChebyshevT, τ.(x[inds]), deg)'

        # Polynomial moments around point
        μᵢ = integrate.(ϕ[1:deg+1], τ(Ival.first), τ(Ival.last))
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

    x = discretize_uniform(domain, n)

    h = f.width
    τ(x) = (x - mid) / L
    τ(x, a, b) = (x - (a + b) / 2) / (b - a)
    degmax = 100
    ϕ = [ChebyshevT([fill(0, i); 1]) for i = 0:degmax]

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
        Vᵢ = vander(ChebyshevT, τ.(x[inds]), deg)'

        # Polynomial moments around point
        Domain = Interval(0, 2π)
        I_left = (Ival + 2π) ∩ Domain
        I_mid = Ival ∩ Domain
        I_right = (Ival - 2π) ∩ Domain
        μᵢ = integrate.(ϕ[1:deg+1], τ(I_left.first), τ(I_left.last))
        μᵢ .+= integrate.(ϕ[1:deg+1], τ(I_mid.first), τ(I_mid.last))
        μᵢ .+= integrate.(ϕ[1:deg+1], τ(I_right.first), τ(I_right.last))
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
