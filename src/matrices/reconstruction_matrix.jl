"""
    reconstruction_matrix(f, domain, n)

Approximate inverse of discrete filtering matrix, given filter `f`.
"""
function reconstruction_matrix end

reconstruction_matrix(::IdentityFilter, ::ClosedIntervalDomain, n) = sparse(I, n + 1, n + 1)
reconstruction_matrix(::IdentityFilter, ::PeriodicIntervalDomain, n) = sparse(I, n, n)

function reconstruction_matrix(
    f::TopHatFilter,
    domain::ClosedIntervalDomain,
    n,
    degmax = 100,
)
    x = discretize(domain, n)
    h = f.width

    if all(h.(x) .≈ h(x[1])) && x[2] - x[1] .≈ 2h(x[1])
        return reconstruction_matrix_meshwidth(f, domain, n)
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

    R
end


# function reconstruction_matrix(
#     f::TopHatFilter,
#     domain::PeriodicIntervalDomain,
#     n,
#     degmax = 100,
# )
#     L = (domain.right - domain.left)
#     mid = (domain.left + domain.right) / 2
#     x = discretize(domain, n)
#     h = f.width

#     if all(h.(x) .≈ h(x[1])) && x[2] - x[1] .≈ 2h(x[1])
#         return reconstruction_matrix_meshwidth(f, domain, n)
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

#     R
# end


function reconstruction_matrix(
    f::TopHatFilter,
    domain::PeriodicIntervalDomain,
    n,
    λ = 1e-6,
    degmax = 100,
)
    L = (domain.right - domain.left)
    h = f.width
    x = discretize(domain, n)
    R = spzeros(n, n)

    # Get reconstruction weights for each point
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

        V = zeros(deg + 1, N)
        f = zeros(deg + 1)
        for k = 1:(deg+1)
            V[k, :] = [
                (ϕ_int[k](ival.right) - ϕ_int[k](ival.left)) / (ival.right - ival.left)
                for ival ∈ ivals
            ]
            f[k] = ϕ[k](xᵢ)
        end

        rᵢ = V \ f
        # rᵢ = (V'V + λ*I) \ (V'f)
        # rᵢ = fit(LassoRegression(λ; fit_intercept = false), V, f)

        # Store weights
        R[i, j] .= rᵢ[:] ./ sum(rᵢ)
    end

    R
end


function reconstruction_matrix(
    f::ConvolutionalFilter,
    domain::ClosedIntervalDomain,
    n,
    degmax = 100,
)
    L = (domain.right - domain.left)
    mid = (domain.left + domain.right) / 2
    Ival_domain = (domain.left - 0.001L)..(domain.right + 0.001L)
    h = f.width
    G = f.kernel
    x = discretize(domain, n)
    R = spzeros(n + 1, n + 1)

    # Get reconstruction weights for each point
    for i = 1:n+1
        # Point
        xᵢ = x[i]

        dists = abs.(x .- xᵢ)

        # Find j such that xᵢ is reachable from xⱼ
        j = dists .< h.(x)

        # Polynomial degree (Taylor series order)
        deg = sum(j)
        # deg = min(degmax, max(1, floor(Int, √sum(j))))

        # Vandermonde matrix
        d = 0:deg-1
        xⱼ = x[j]
        hⱼ = h.(xⱼ)
        aⱼ = max.(xⱼ - hⱼ, domain.left - 0.001L)
        bⱼ = min.(xⱼ + hⱼ, domain.right + 0.001L)
        # Δhⱼ = bⱼ - aⱼ
        ivals = xⱼ .± hⱼ
        # ivals = Interval.(aⱼ, bⱼ)

        N = length(xⱼ) + 1

        ϕ = chebyshevt.(d, [minimum(xⱼ - hⱼ)..maximum(xⱼ + hⱼ)])
        # ϕ = chebyshevt.(d, [minimum(aⱼ)..maximum(bⱼ)])
        # ϕ = chebyshevt.(d, [Ival_domain])

        V = zeros(N - 1, N - 1)
        f = zeros(N - 1)
        for j = 1:N-1
            ival = ivals[j]
            ϕⱼ = Fun.(ϕ, [ival])
            kern = Fun(x -> G(x - xⱼ[j]), ival)
            V[:, j] = sum.(kern .* ϕⱼ) / sum(kern)
            f[j] = ϕ[j](xᵢ)
        end

        rᵢ = V \ f
        # rᵢ = (V'V + 1e-8I) \ (V'f)

        # Store weights
        R[i, j] .= rᵢ[:] ./ sum(rᵢ)
    end

    R
end

function reconstruction_matrix(
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
    R = spzeros(n, n)

    # Get reconstruction weights for each point
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
        deg = sum(j)
        # deg = min(degmax, max(1, floor(Int, √sum(j))))

        # Vandermonde matrix
        d = 0:deg-1
        aⱼ = max.(xⱼ - hⱼ, domain.left - 0.001L)
        bⱼ = min.(xⱼ + hⱼ, domain.right + 0.001L)
        # Δhⱼ = bⱼ - aⱼ
        ivals = xⱼ .± hⱼ
        # ivals = Interval.(aⱼ, bⱼ)

        N = length(xⱼ) + 1

        ϕ = chebyshevt.(d, [minimum(xⱼ - hⱼ)..maximum(xⱼ + hⱼ)])
        # ϕ = chebyshevt.(d, [domain.left..domain.right])

        V = zeros(N - 1, N - 1)
        f = zeros(N - 1)
        for k = 1:N-1
            ival = ivals[k]
            ϕⱼ = Fun.(ϕ, [ival])
            kern = Fun(x -> G(x - xⱼ[k]), ival)
            kern /= sum(kern)
            V[:, k] = sum.(kern .* ϕⱼ)
            f[k] = ϕ[k](xᵢ)
        end

        rᵢ = V \ f
        # rᵢ = (V'V + 1e-8I) \ (V'f)

        # Store weights
        R[i, j] .= rᵢ[:] ./ sum(rᵢ)
    end

    R
end
