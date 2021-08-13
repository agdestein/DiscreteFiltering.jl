"""
    reconstruction_matrix(f, domain, n)

Approximate inverse of discrete filtering matrix, given filter `f`.
"""
function reconstruction_matrix(::Filter, ::Domain, n)
    error("Not implemented")
end


function reconstruction_matrix(::IdentityFilter, ::ClosedIntervalDomain, n)
    sparse(I, n + 1, n + 1)
end


function reconstruction_matrix(::IdentityFilter, ::PeriodicIntervalDomain, n)
    sparse(I, n, n)
end


function reconstruction_matrix(f::TopHatFilter, domain::ClosedIntervalDomain, n)

    degmax = 30

    x = discretize(domain, n)
    h = f.width

    if all(h.(x) .≈ h(x[1])) && x[2] - x[1] .≈ 2h(x[1])
        return reconstruction_matrix_meshwidth(f, domain, n)
    end

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
        R[i, j] .= rᵢ[:] ./ sum(rᵢ)
    end

    R
end


# function reconstruction_matrix(f::TopHatFilter, domain::PeriodicIntervalDomain, n)

#     degmax = 30

#     L = (domain.right - domain.left)
#     mid = (domain.left + domain.right) / 2

#     x = discretize(domain, n)
#     h = f.width

#     if all(h.(x) .≈ h(x[1])) && x[2] - x[1] .≈ 2h(x[1])
#         return reconstruction_matrix_meshwidth(f, domain, n)
#     end

#     R = spzeros(n, n)

#     # Get reconstruction weights for each point
#     for i = 1:n
#         # Point
#         xᵢ = x[i]

#         dists = @. abs(xᵢ - x - [-L 0 L])

#         # Move x by L * (shifts - 2) to get closer to xᵢ
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


function reconstruction_matrix(f::TopHatFilter, domain::PeriodicIntervalDomain, n, λ = 1e-6)
    degmax = 100

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
        for k = 1:(deg + 1)
            V[k, :] = [(ϕ_int[k](ival.right) - ϕ_int[k](ival.left)) / (ival.right - ival.left)  for ival ∈ ivals]
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


function reconstruction_matrix(f::ConvolutionalFilter, domain::ClosedIntervalDomain, n)
    degmax = 100

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


function reconstruction_matrix(f::ConvolutionalFilter, domain::PeriodicIntervalDomain, n)
    degmax = 100

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


"""
    reconstruction_matrix_meshwidth(f, domain, n)

Assemble inverse discrete filtering matrix from a continuous filter `f` width constant width
\$h(x) = \\Delta x / 2\$.
"""
function reconstruction_matrix_meshwidth(f::TopHatFilter, domain::PeriodicIntervalDomain, n)
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
    stencil = [-1 / 24, 13 / 12, -1 / 24]

    # Five point stencil
    # inds = [-2, -1, 0, 1, 2]
    # stencil = [3 / 640, -29 / 480, 1067 / 960, -29 / 480, 3 / 640]

    # Construct banded matrix
    diags = [i => fill(s, n - abs(i)) for (i, s) ∈ zip(inds, stencil)]
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


function reconstruction_matrix_meshwidth(f::TopHatFilter, domain::ClosedIntervalDomain, n)
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
