"""
    filter_matrix(f, domain, N)

Assemble discrete filtering matrix from a continuous filter `f`.
"""
function filter_matrix end

filter_matrix(::IdentityFilter, ::ClosedIntervalDomain, N) = sparse(I, N + 1, N + 1)
filter_matrix(::IdentityFilter, ::PeriodicIntervalDomain, N) = sparse(I, N, N)

function filter_matrix(f::TopHatFilter, domain::ClosedIntervalDomain, N, degmax = 100)
    x = discretize(domain, N)
    Δx = x[2] - x[1]
    L = (domain.right - domain.left)
    mid = (domain.left + domain.right) / 2
    ival_domain = (domain.left - Δx / 1000)..(domain.right + Δx / 1000)
    h = f.width

    W = spzeros(N + 1, N + 1)
    for i = 1:N+1
        # Point
        xᵢ = x[i]

        # Filter width at point
        hᵢ = h(xᵢ)

        # Indices of integration points inside domaain
        ival = (xᵢ ± hᵢ) ∩ ival_domain
        ival_length = ival.right - ival.left
        inds = x .∈ [ival]
        deg = min(degmax, max(1, floor(Int, √(sum(inds) - 1))))

        ϕ = chebyshevt(0:deg, ival)
        ϕ_int = integrate.(ϕ)

        # Polynomials evaluated at integration points
        Vᵢ = spzeros(deg + 1, length(x[inds]))

        # Polynomial moments around point
        μᵢ = zeros(deg + 1)

        # Fill in
        for d = 1:deg+1
            Vᵢ[d, :] = ϕ[d].(x[inds])
            μᵢ[d] = (ϕ_int[d](ival.right) - ϕ_int[d](ival.left)) / ival_length
        end

        # Fit weights
        wᵢ = Vᵢ \ μᵢ
        # wᵢ = nonneg_lsq(Vᵢ, μᵢ)

        # Store weights
        W[i, inds] .= wᵢ[:] ./ sum(wᵢ)
    end
    dropzeros!(W)

    W
end

function filter_matrix(f::TopHatFilter, domain::PeriodicIntervalDomain, N, degmax = 10)
    x = discretize(domain, N)
    Δx = x[2] - x[1]
    L = (domain.right - domain.left)
    mid = (domain.left + domain.right) / 2
    ival_domain = (domain.left - Δx / 1000)..(domain.right + Δx / 1000)
    h = f.width

    W = spzeros(N, N)
    for i = 1:N
        # Point
        xᵢ = x[i]

        # Filter width at point
        hᵢ = h(xᵢ)

        # Indices of integration points inside domain
        ival = xᵢ ± hᵢ
        indsleft = x .+ L .∈ [ival]
        indsmid = x .∈ [ival]
        indsright = x .- L .∈ [ival]
        j = indsleft .| indsmid .| indsright
        j = mapreduce(s -> circshift(j, s), .|, [-1, 0, 1])
        xⱼ = (x.+(xᵢ.-x.>L/2)L.-(x.-xᵢ.>L/2)L)[j]
        Nₘ = length(xⱼ) - 1

        # deg = min(degmax, max(1, floor(Int, √(N - 1))))
        deg = min(degmax, Nₘ)
        # ϕ = chebyshevt(0:deg, ival)
        ϕ = chebyshevt(0:deg, xᵢ ± 3hᵢ)
        ϕ_int = integrate.(ϕ)

        # Polynomials evaluated at integration points
        Vᵢ = spzeros(deg + 1, Nₘ + 1)

        # Polynomial moments around point
        μᵢ = zeros(deg + 1)

        # Fill in
        for d = 1:deg+1
            Vᵢ[d, :] = ϕ[d].(xⱼ)
            μᵢ[d] = (ϕ_int[d](ival.right) - ϕ_int[d](ival.left)) / 2hᵢ
        end

        # Fit weights
        wᵢ = Vᵢ \ μᵢ
        # wᵢ = nonneg_lsq(Vᵢ, μᵢ)
        # wᵢ = (Vᵢ'Vᵢ + sparse(1e-8I, Nₘ + 1, Nₘ + 1)) \ Vᵢ'μᵢ

        # Store weights
        W[i, j] .= wᵢ[:] #./ sum(wᵢ)
    end
    dropzeros!(W)

    W
end

function filter_matrix(
    f::ConvolutionalFilter,
    domain::ClosedIntervalDomain,
    N,
    degmax = 100,
)
    x = discretize(domain, N)
    Δx = x[2] - x[1]
    L = (domain.right - domain.left)
    mid = (domain.left + domain.right) / 2
    ival_domain = (domain.left - Δx / 1000)..(domain.right + Δx / 1000)
    h = f.width
    G = f.kernel
    W = spzeros(N + 1, N + 1)
    for i = 1:N+1
        # Point
        xᵢ = x[i]

        # Filter width at point
        hᵢ = h(xᵢ)

        # Indices of integration points inside domaain
        ival = (xᵢ ± hᵢ) ∩ ival_domain
        j = x .∈ [ival]
        xⱼ = x[j]
        Nₘ = length(xⱼ)

        deg = min(degmax, Nₘ - 1)
        ϕ = chebyshevt(0:deg, ival)

        # Normalized kernel Fun
        kern = Fun(x -> G(x - xᵢ), ival)
        kern /= sum(kern)

        # Polynomial moments around point
        μᵢ = sum.(kern .* ϕ)

        # Polynomials evaluated at integration points
        Vᵢ = spzeros(deg + 1, Nₘ)
        for d = 1:deg+1
            Vᵢ[d, :] = ϕ[d].(xⱼ)
        end

        # Fit weights
        wᵢ = Vᵢ \ μᵢ
        # wᵢ = nonneg_lsq(Vᵢ, μᵢ)
        # wᵢ = (Vᵢ'Vᵢ + sparse(1e-8I, Nₘ, Nₘ)) \ Vᵢ'μᵢ

        # Store weights
        W[i, j] .= wᵢ[:] ./ sum(wᵢ)
    end
    dropzeros!(W)

    W
end

function filter_matrix(
    f::ConvolutionalFilter,
    domain::PeriodicIntervalDomain,
    N,
    degmax = 10,
)
    L = (domain.right - domain.left)
    mid = (domain.left + domain.right) / 2
    h = f.width
    G = f.kernel
    x = discretize(domain, N)
    W = spzeros(N, N)
    for i = 1:N
        # Point
        xᵢ = x[i]

        # Filter width at point
        hᵢ = h(xᵢ)

        # Indices of integration points inside domain
        ival = (xᵢ ± hᵢ)
        indsleft = x .+ L .∈ [ival]
        indsmid = x .∈ [ival]
        indsright = x .- L .∈ [ival]
        j = indsleft .| indsmid .| indsright
        xⱼ = (x+L*indsleft-L*indsright)[j]
        Nₘ = length(xⱼ)

        # deg = min(degmax, max(1, floor(Int, √(Nₘ - 1))))
        deg = min(degmax, sum(j))
        ϕ = chebyshevt(0:deg, ival)

        # Normalized kernel Fun
        kern = Fun(x -> G(x - xᵢ), ival)
        kern /= sum(kern)

        # Polynomials evaluated at integration points
        Vᵢ = spzeros(deg + 1, Nₘ)

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
        wᵢ = (Vᵢ'Vᵢ + sparse(1e-8I, Nₘ, Nₘ)) \ Vᵢ'μᵢ

        # Store weights
        W[i, j] .= wᵢ[:] ./ sum(wᵢ)
    end
    dropzeros!(W)

    W
end
