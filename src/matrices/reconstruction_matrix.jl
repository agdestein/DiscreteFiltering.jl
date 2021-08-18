"""
    reconstruction_matrix(filter, domain, n)

Approximate inverse of discrete filtering matrix, given filter `filter`.
"""
function reconstruction_matrix end

reconstruction_matrix(::IdentityFilter, ::ClosedIntervalDomain, n) = sparse(I, n + 1, n + 1)
reconstruction_matrix(::IdentityFilter, ::PeriodicIntervalDomain, n) = sparse(I, n, n)

function reconstruction_matrix(
    filter::TopHatFilter,
    domain::ClosedIntervalDomain,
    n,
    degmax = 100,
)
    x = discretize(domain, n)
    h = filter.width

    if all(h.(x) .≈ h(x[1])) && x[2] - x[1] .≈ 2h(x[1])
        return reconstruction_matrix_meshwidth(filter, domain, n)
    end

    # Get reconstruction weights for each point
    R = spzeros(n + 1, n + 1)
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
        R[i, j] .= rᵢ[:] ./ sum(rᵢ)
    end
    dropzeros!(R)

    R
end

# function reconstruction_matrix(
#     filter::TopHatFilter,
#     domain::PeriodicIntervalDomain,
#     n,
#     degmax = 100,
# )
#     L = (domain.right - domain.left)
#     mid = (domain.left + domain.right) / 2
#     x = discretize(domain, n)
#     h = filter.width

#     if all(h.(x) .≈ h(x[1])) && x[2] - x[1] .≈ 2h(x[1])
#         return reconstruction_matrix_meshwidth(filter, domain, n)
#     end

#     # Get reconstruction weights for each point
#     R = spzeros(n, n)
#     for i = 1:n
#         # Point
#         xᵢ = x[i]

#         # Move x by L * (shifts - 2) to get closer to xᵢ
#         dists = @. abs(xᵢ - x - [-L 0 L])
#         mininds = argmin(dists, dims = 2)
#         shifts = [mininds[j].I[2] for j in eachindex(mininds)]

#         # Find j such that xᵢ is reachable from xⱼ
#         j = dists[mininds][:] .< h.(x)

#         # Polynomial degree (Taylor series order)
#         deg = min(degmax, max(1, floor(Int, √sum(j))))

#         # Vandermonde matrix
#         d = 1:deg
#         xⱼ = x[j]'
#         hⱼ = h.(xⱼ)
#         sⱼ = L * (shifts[j]' .- 2)

#         Vᵢ = @. 1 / (2hⱼ * factorial(d)) * ((xⱼ + sⱼ + hⱼ - xᵢ)^d - (xⱼ + sⱼ - hⱼ - xᵢ)^d)

#         # Right-hand side
#         μᵢ = fill(0.0, deg)
#         μᵢ[1] = 1.0

#         # Fit weights
#         rᵢ = Vᵢ \ μᵢ

#         # Store weights
#         R[i, j] .= rᵢ[:] ./ sum(rᵢ)
#     end
#     dropzeros!(R)

#     R
# end

function reconstruction_matrix(
    filter::TopHatFilter,
    domain::PeriodicIntervalDomain,
    n,
    λ = 1e-6,
    degmax = 100,
)
    L = (domain.right - domain.left)
    h = filter.width
    x = discretize(domain, n)

    # Get reconstruction weights for each point
    R = spzeros(n, n)
    for i = 1:n
        # Point
        xᵢ = x[i]

        dists = @. abs(xᵢ - x - [-L 0 L])

        # Move x by L * (shifts - 2) to get closer to xᵢ
        mininds = argmin(dists, dims = 2)
        shifts = [mininds[j].I[2] for j in eachindex(mininds)]

        # Find j such that xᵢ is reachable from xⱼ and include one point outside
        j = dists[mininds][:] .< 1.5h.(x)
        # j = mapreduce(s -> circshift(j, s), .|, [-1, 0, 1])

        # Vandermonde matrix
        hⱼ = h.(x[j]')
        sⱼ = L * (shifts[j]' .- 2)
        xⱼ = x[j]' + sⱼ

        # Polynomial degree (Taylor series order)
        N = length(xⱼ)
        deg = min(degmax, N - 1)
        # deg = min(degmax, max(1, floor(Int, √sum(j))))

        # Vandermonde matrix
        # aⱼ = max.(xⱼ - hⱼ, domain.left - 0.001L)
        # bⱼ = min.(xⱼ + hⱼ, domain.right + 0.001L)
        # Δhⱼ = bⱼ - aⱼ
        ivals = xⱼ .± hⱼ
        # ivals = Interval.(aⱼ, bⱼ)


        ϕ = chebyshevt.(0:deg, [minimum(xⱼ - 3hⱼ)..maximum(xⱼ + 3hⱼ)])
        # ϕ = chebyshevt.(0:deg, [domain.left..domain.right])
        ϕ_int = integrate.(ϕ)

        Z = sparse(deg + 1, N)
        v = sparse(deg + 1)
        for k = 1:(deg+1)
            for ival ∈ ivals
                Z[k, :] =
                    (ϕ_int[k](ival.right) - ϕ_int[k](ival.left)) / (ival.right - ival.left)
            end
            v[k] = ϕ[k](xᵢ)
        end

        rᵢ = Z \ v
        # rᵢ = (Z'Z + λ*I) \ (Z'v)
        # rᵢ = fit(LassoRegression(λ; fit_intercept = false), Z, v)

        # Store weights
        R[i, j] .= rᵢ[:] ./ sum(rᵢ)
    end
    dropzeros!(R)

    R
end

function reconstruction_matrix(
    filter::ConvolutionalFilter,
    domain::ClosedIntervalDomain,
    n,
    degmax = 100,
)
    x = discretize(domain, n)
    Δx = x[2] - x[1]
    L = (domain.right - domain.left)
    mid = (domain.left + domain.right) / 2
    Ival_domain = (domain.left - Δx / 1000)..(domain.right + Δx / 1000)
    h = filter.width
    G = filter.kernel

    # Get reconstruction weights for each point
    R = spzeros(n + 1, n + 1)
    for i = 1:n+1
        # Point
        xᵢ = x[i]

        # Find j such that xᵢ is reachable from xⱼ
        dists = abs.(x .- xᵢ)
        j = dists .< h.(x)

        # Polynomial degree (Taylor series order)
        N = sum(j)
        deg = sum(j) - 1

        xⱼ = x[j]
        hⱼ = h.(xⱼ)
        aⱼ = max.(xⱼ - hⱼ, Ival_domain.left)
        bⱼ = min.(xⱼ + hⱼ, Ival_domain.right)
        # Δhⱼ = bⱼ - aⱼ
        # ivals = xⱼ .± hⱼ
        ivals = Interval.(aⱼ, bⱼ)

        # ϕ = chebyshevt.(0:deg, [minimum(xⱼ - hⱼ)..maximum(xⱼ + hⱼ)])
        ϕ = chebyshevt.(0:deg, [minimum(aⱼ)..maximum(bⱼ)])
        # ϕ = chebyshevt.(0:deg, [Ival_domain])

        # Polynomials evaluated at xᵢ
        v = [ϕ(xᵢ) for ϕ ∈ ϕ]

        # Moment matrix
        Z = zeros(deg + 1, N)
        for k = 1:N
            ival = ivals[k]
            ϕₖ = Fun.(ϕ, [ival])
            kern = Fun(x -> G(x - xⱼ[k]), ival)
            Z[:, k] = sum.(kern .* ϕₖ) / sum(kern)
        end

        rᵢ = Z \ v
        # rᵢ = (Z'Z + 1e-8I) \ (Z'v)

        # Store weights
        R[i, j] .= rᵢ[:] ./ sum(rᵢ)
    end
    dropzeros!(R)

    R
end

function reconstruction_matrix(
    filter::ConvolutionalFilter,
    domain::PeriodicIntervalDomain,
    n,
    degmax = 100,
)
    x = discretize(domain, n)
    Δx = x[2] - x[1]
    L = (domain.right - domain.left)
    mid = (domain.left + domain.right) / 2
    h = filter.width
    G = filter.kernel

    # Get reconstruction weights for each point
    R = spzeros(n, n)
    for i = 1:n
        # Point
        xᵢ = x[i]

        dists = @. abs(xᵢ - x - [-L 0 L])

        # Move x by L * (shifts - 2) to get closer to xᵢ
        mininds = argmin(dists, dims = 2)
        shifts = [mininds[j].I[2] for j in eachindex(mininds)]

        # Find j such that xᵢ is reachable from xⱼ
        j = dists[mininds][:] .< h.(x)

        # Vandermonde matrix
        sⱼ = L * (shifts[j]' .- 2)
        xⱼ = x[j]' + sⱼ
        hⱼ = h.(xⱼ)

        # Polynomial degree (Taylor series order)
        deg = sum(j) - 1
        N = sum(j)
        # deg = min(degmax, max(1, floor(Int, √sum(j))))

        # Vandermonde matrix
        d = 0:deg-1
        aⱼ = max.(xⱼ - hⱼ, domain.left - Δx / 1000)
        bⱼ = min.(xⱼ + hⱼ, domain.right + Δx / 1000)
        # Δhⱼ = bⱼ - aⱼ
        ivals = xⱼ .± hⱼ
        # ivals = Interval.(aⱼ, bⱼ)

        ϕ = chebyshevt.(d, [minimum(xⱼ - hⱼ)..maximum(xⱼ + hⱼ)])
        # ϕ = chebyshevt.(d, [domain.left..domain.right])

        Z = zeros(deg + 1, N)
        v = zeros(deg + 1)
        for k = 1:N-1
            ival = ivals[k]
            ϕⱼ = Fun.(ϕ, [ival])
            kern = Fun(x -> G(x - xⱼ[k]), ival)
            kern /= sum(kern)
            Z[:, k] = sum.(kern .* ϕⱼ)
            v[k] = ϕ[k](xᵢ)
        end

        rᵢ = Z \ v
        # rᵢ = (Z'Z + 1e-8I) \ (Z'v)

        # Store weights
        R[i, j] .= rᵢ[:] ./ sum(rᵢ)
    end
    dropzeros!(R)

    R
end
