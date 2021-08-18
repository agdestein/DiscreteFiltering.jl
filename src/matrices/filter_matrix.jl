"""
    filter_matrix(f, domain, n)

Assemble discrete filtering matrix from a continuous filter `f`.
"""
function filter_matrix end

filter_matrix(::IdentityFilter, ::ClosedIntervalDomain, n) = sparse(I, n + 1, n + 1)
filter_matrix(::IdentityFilter, ::PeriodicIntervalDomain, n) = sparse(I, n, n)

function filter_matrix(f::TopHatFilter, domain::ClosedIntervalDomain, n, degmax = 100)
    x = discretize(domain, n)
    Δx = x[2] - x[1]
    L = (domain.right - domain.left)
    mid = (domain.left + domain.right) / 2
    ival_domain = (domain.left - Δx / 1000)..(domain.right + Δx / 1000)
    h = f.width

    if all(≈(h(x[1])), h.(x)) && x[2] - x[1] ≈ 2h(x[1])
        return filter_matrix_meshwidth(f, domain, n)
    end

    W = spzeros(n + 1, n + 1)
    for i = 1:n+1
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
        Vᵢ = zeros(deg + 1, length(x[inds]))

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

function filter_matrix(f::TopHatFilter, domain::PeriodicIntervalDomain, n, degmax = 100)
    x = discretize(domain, n)
    Δx = x[2] - x[1]
    L = (domain.right - domain.left)
    mid = (domain.left + domain.right) / 2
    ival_domain = (domain.left - Δx / 1000)..(domain.right + Δx / 1000)
    h = f.width

    if all(≈(h(x[1])), h.(x)) && x[2] - x[1] .≈ 2h(x[1])
        return filter_matrix_meshwidth(f, domain, n)
    end

    W = spzeros(n, n)
    for i = 1:n
        # Point
        xᵢ = x[i]

        # Filter width at point
        hᵢ = h(xᵢ)

        # Indices of integration points inside domain
        ival = (xᵢ ± hᵢ)
        # indsleft = x .+ L .∈ [ival]
        # indsmid = x .∈ [ival]
        # indsright = x .- L .∈ [ival]
        indsleft = x .+ L .∈ [ival]
        indsmid = x .∈ [ival]
        indsright = x .- L .∈ [ival]
        j = indsleft .| indsmid .| indsright
        j = mapreduce(s -> circshift(j, s), .|, [-1, 0, 1])
        xⱼ = (x.+(xᵢ.-x.>L/2)L.-(x.-xᵢ.>L/2)L)[j]
        N = length(xⱼ)

        # deg = min(degmax, max(1, floor(Int, √(n - 1))))
        deg = min(degmax, N - 1)
        # ϕ = chebyshevt(0:deg, ival)
        ϕ = chebyshevt(0:deg, xᵢ ± 3hᵢ)
        ϕ_int = integrate.(ϕ)

        # Polynomials evaluated at integration points
        Vᵢ = zeros(deg + 1, N)

        # Polynomial moments around point
        μᵢ = zeros(deg + 1)

        # Fill in
        for d = 1:deg+1
            Vᵢ[d, :] = ϕ[d].(xⱼ)
            μᵢ[d] = (ϕ_int[d](ival.right) - ϕ_int[d](ival.left)) / 2hᵢ
        end

        # Fit weights
        # wᵢ = Vᵢ \ μᵢ
        wᵢ = nonneg_lsq(Vᵢ, μᵢ)
        # wᵢ = (Vᵢ'Vᵢ + sparse(1e-8I, n, n)) \ Vᵢ'μᵢ

        # Store weights
        W[i, j] .= wᵢ[:] ./ sum(wᵢ)
    end
    dropzeros!(W)

    W
end

function filter_matrix(
    f::ConvolutionalFilter,
    domain::ClosedIntervalDomain,
    n,
    degmax = 100,
)
    x = discretize(domain, n)
    Δx = x[2] - x[1]
    L = (domain.right - domain.left)
    mid = (domain.left + domain.right) / 2
    ival_domain = (domain.left - Δx / 1000)..(domain.right + Δx / 1000)
    h = f.width
    G = f.kernel
    W = spzeros(n + 1, n + 1)
    for i = 1:n+1
        # Point
        xᵢ = x[i]

        # Filter width at point
        hᵢ = h(xᵢ)

        # Indices of integration points inside domaain
        ival = (xᵢ ± hᵢ) ∩ ival_domain
        j = x .∈ [ival]
        xⱼ = x[j]
        N = length(xⱼ)

        deg = min(degmax, N - 1)
        ϕ = chebyshevt(0:deg, ival)

        # Normalized kernel Fun
        kern = Fun(x -> G(x - xᵢ), ival)
        kern /= sum(kern)

        # Polynomial moments around point
        μᵢ = sum.(kern .* ϕ)

        # Polynomials evaluated at integration points
        Vᵢ = zeros(deg + 1, N)
        for d = 1:deg+1
            Vᵢ[d, :] = ϕ[d].(xⱼ)
        end

        # Fit weights
        wᵢ = Vᵢ \ μᵢ
        # wᵢ = nonneg_lsq(Vᵢ, μᵢ)
        # wᵢ = (Vᵢ'Vᵢ + sparse(1e-8I, N, N)) \ Vᵢ'μᵢ

        # Store weights
        W[i, j] .= wᵢ[:] ./ sum(wᵢ)
    end
    dropzeros!(W)

    W
end

function filter_matrix(
    f::ConvolutionalFilter,
    domain::PeriodicIntervalDomain,
    n,
    degmax = 100,
)
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
        ival = (xᵢ ± hᵢ)
        indsleft = x .+ L .∈ [ival]
        indsmid = x .∈ [ival]
        indsright = x .- L .∈ [ival]
        j = indsleft .| indsmid .| indsright
        xⱼ = (x+L*indsleft-L*indsright)[j]
        N = length(xⱼ)

        # deg = min(degmax, max(1, floor(Int, √(N - 1))))
        deg = min(degmax, sum(j))
        ϕ = chebyshevt(0:deg, ival)

        # Normalized kernel Fun
        kern = Fun(x -> G(x - xᵢ), ival)
        kern /= sum(kern)

        # Polynomials evaluated at integration points
        Vᵢ = zeros(deg + 1, N)

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
        wᵢ = (Vᵢ'Vᵢ + sparse(1e-8I, N, N)) \ Vᵢ'μᵢ

        # Store weights
        W[i, j] .= wᵢ[:] ./ sum(wᵢ)
    end
    dropzeros!(W)

    W
end
