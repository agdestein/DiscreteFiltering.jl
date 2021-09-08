"""
    filter_matrix(f, domain, M, N)

Assemble discrete filtering matrix from a continuous filter `f`.
"""
function filter_matrix end


function filter_matrix(::IdentityFilter, ::AbstractIntervalDomain, M, N)
    x = discretize(domain, M)
    ξ = discretize(domain, N)
    interpolation_matrix(x, ξ)
end

function filter_matrix(
    f::TophatFilter,
    domain::ClosedIntervalDomain,
    M,
    N,
    degmax = 10,
    λ = 0,
)
    x = discretize(domain, M)
    ξ = discretize(domain, N)
    Δx = x[2] - x[1]
    ival_domain = (domain.left - Δx / 1000)..(domain.right + Δx / 1000)
    h = f.width

    W = spzeros(M + 1, N + 1)
    for m = 1:(M+1)
        # Point
        xₘ = x[m]

        # Filter width at point
        hₘ = h(xₘ)

        # Indices of integration points inside domaain
        ival = (xₘ ± hₘ) ∩ ival_domain
        ival_length = ival.right - ival.left
        n = ξ .∈ [ival]
        P = min(degmax, max(1, floor(Int, √(sum(n) - 1))))

        ϕ = chebyshevt(0:P, ival)
        ϕ_int = integrate.(ϕ)

        # Polynomials evaluated at integration points
        Vₘ = spzeros(P + 1, length(ξ[n]))

        # Polynomial moments around point
        μₘ = zeros(P + 1)

        # Fill in
        for p = 1:(P+1)
            Vₘ[p, :] = ϕ[p].(ξ[n])
            μₘ[p] = (ϕ_int[p](ival.right) - ϕ_int[p](ival.left)) / ival_length
        end

        # Fit weights
        wₘ = ridge(Vₘ, μₘ, λ)
        # wₘ = Vₘ \ μₘ
        # wₘ = nonneg_lsq(Vₘ, μₘ)

        # Store weights
        W[m, n] .= wₘ[:] ./ sum(wₘ)
    end
    dropzeros!(W)

    W
end

function filter_matrix(
    f::TopHatFilter,
    domain::PeriodicIntervalDomain,
    M,
    N,
    degmax = 10,
    λ = 0,
)
    x = discretize(domain, M)
    ξ = discretize(domain, N)
    Δx = x[2] - x[1]
    L = (domain.right - domain.left)
    h = f.width

    W = spzeros(M, N)
    for m = 1:M
        # Point
        xₘ = x[m]

        # Filter width at point
        hₘ = h(xₘ)

        # Indices of integration points inside domain
        ival = xₘ ± hₘ
        n_left = ξ .+ L .∈ [ival]
        n_mid = ξ .∈ [ival]
        n_right = ξ .- L .∈ [ival]
        n = n_left .| n_mid .| n_right
        n = mapreduce(s -> circshift(n, s), .|, [-1, 0, 1])
        ξₙ = (ξ.+(xₘ.-ξ.>L/2)L.-(ξ.-xₘ.>L/2)L)[n]
        Nₘ = length(ξₙ) - 1

        # P = min(degmax, max(1, floor(Int, √(Nₘ - 1))))
        P = min(degmax, Nₘ)
        # ϕ = chebyshevt(0:P, ival)
        ϕ = chebyshevt(0:P, xₘ ± 3hₘ)
        ϕ_int = integrate.(ϕ)

        # Polynomials evaluated at integration points
        Vₘ = spzeros(P + 1, Nₘ + 1)

        # Polynomial moments around point
        zₘ = zeros(P + 1)

        # Fill in
        for p = 1:(P+1)
            Vₘ[p, :] = ϕ[p].(ξₙ)
            zₘ[p] = (ϕ_int[p](ival.right) - ϕ_int[p](ival.left)) / 2hₘ
        end

        # Fit weights
        wₘ = ridge(Vₘ, zₘ, λ)
        # wₘ = Vₘ \ zₘ
        # wₘ = nonneg_lsq(Vₘ, zₘ)

        # Store weights
        W[m, n] .= wₘ[:] ./ sum(wₘ)
    end
    dropzeros!(W)

    W
end

function filter_matrix(
    f::ConvolutionalFilter,
    domain::ClosedIntervalDomain,
    M,
    N,
    degmax = 10,
    λ = 0,
)
    x = discretize(domain, M)
    ξ = discretize(domain, N)
    Δx = x[2] - x[1]
    ival_domain = (domain.left - Δx / 1000)..(domain.right + Δx / 1000)
    h = f.width
    G = f.kernel
    W = spzeros(M + 1, N + 1)
    for m = 1:(M+1)
        # Point
        xₘ = x[m]

        # Filter width at point
        hₘ = h(xₘ)

        # Indices of integration points inside domaain
        ival = (xₘ ± hₘ) ∩ ival_domain
        n = ξ .∈ [ival]
        ξₙ = ξ[n]
        Nₘ = length(ξₙ)

        P = min(degmax, Nₘ - 1)
        ϕ = chebyshevt(0:P, ival)

        # Normalized kernel Fun
        kern = Fun(ξ -> G(ξ - xₘ), ival)
        kern /= sum(kern)

        # Polynomial moments around point
        zₘ = sum.(kern .* ϕ)

        # Polynomials evaluated at integration points
        Vₘ = spzeros(P + 1, Nₘ)
        for p = 1:(P+1)
            Vₘ[p, :] = ϕ[p].(ξₙ)
        end

        # Fit weights
        wₘ = ridge(Vₘ, zₘ, λ)
        # wₘ = Vₘ \ zₘ
        # wₘ = nonneg_lsq(Vₘ, zₘ)

        # Store weights
        W[m, n] .= wₘ[:] ./ sum(wₘ)
    end
    dropzeros!(W)

    W
end

function filter_matrix(
    f::ConvolutionalFilter,
    domain::PeriodicIntervalDomain,
    M,
    N,
    degmax = 10,
    λ = 0,
)
    x = discretize(domain, M)
    ξ = discretize(domain, N)
    L = (domain.right - domain.left)
    h = f.width
    G = f.kernel
    W = spzeros(M, N)
    for m = 1:M
        # Point
        xₘ = x[m]

        # Filter width at point
        hₘ = h(xₘ)

        # Indices of integration points inside domain
        ival = (xₘ ± hₘ)
        n_left = ξ .+ L .∈ [ival]
        n_mid = ξ .∈ [ival]
        n_right = ξ .- L .∈ [ival]
        n = n_left .| n_mid .| n_right
        ξₙ = (ξ+L*indsleft-L*indsright)[n]
        Nₘ = length(ξₙ)

        # P = min(degmax, max(1, floor(Int, √(Nₘ - 1))))
        P = min(degmax, sum(n))
        ϕ = chebyshevt(0:P, ival)

        # Normalized kernel Fun
        kern = Fun(ξ -> G(ξ - xₘ), ival)
        kern /= sum(kern)

        # Polynomials evaluated at integration points
        Vₘ = spzeros(P + 1, Nₘ)

        # Polynomial moments around point
        zₘ = zeros(P + 1)

        # Fill in
        for p = 1:(P+1)
            Vₘ[p, :] = ϕ[p].(ξₙ)
            zₘ[p] = sum(kern * ϕ[p])
        end

        # Fit weights
        wₘ = ridge(Vₘ, zₘ, λ)
        # wₘ = Vₘ \ zₘ
        # wₘ = nonneg_lsq(Vₘ, zₘ)

        # Store weights
        W[m, n] .= wₘ[:] ./ sum(wₘ)
    end
    dropzeros!(W)

    W
end
