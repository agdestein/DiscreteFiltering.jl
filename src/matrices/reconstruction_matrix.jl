"""
    reconstruction_matrix(filter, domain, N)

Approximate inverse of discrete filtering matrix, given filter `filter`.
"""
function reconstruction_matrix end

function reconstruction_matrix(::IdentityFilter, ::AbstractIntervalDomain, M, N)
    x = discretize(domain, M)
    ξ = discretize(domain, N)
    interpolation_matrix(ξ, x)
end

function reconstruction_matrix(
    filter::TopHatFilter,
    domain::ClosedIntervalDomain,
    M,
    N,
    degmax = 10,
    λ = 0,
)
    x = discretize(domain, M)
    ξ = discretize(domain, N)
    h = filter.width

    # Get reconstruction weights for each point
    R = spzeros(N + 1, M + 1)
    for n = 1:(N+1)
        # Point
        ξₙ = ξ[n]

        dists = abs.(x .- ξₙ)

        # Find m such that ξₙ is reachable from xₘ
        m = dists .< h.(x)

        # Polynomial degree (Taylor series order)
        Q = min(degmax, max(1, floor(Int, √sum(m))))

        # Vandermonde matrix
        d = 1:Q
        xₘ = x[m]'
        hₘ = h.(xₘ)
        aₘ = max.(xₘ - hₘ, domain.left)
        bₘ = min.(xₘ + hₘ, domain.right)
        Δhₘ = bₘ - aₘ

        Zₙ = @. 1 / (Δhₘ * factorial(d)) * ((bₘ - ξₙ)^d - (aₘ - ξₙ)^d)

        # Right-hand side
        vₙ = fill(0.0, Q)
        vₙ[1] = 1.0

        # Fit weights
        rₙ = ridge(Zₙ, vₙ)
        # rₙ = Zₙ \ vₙ

        # Store weights
        R[n, m] .= rₙ[:] ./ sum(rₙ)
    end
    dropzeros!(R)

    R
end

# function reconstruction_matrix(
#     filter::TopHatFilter,
#     domain::PeriodicIntervalDomain,
#     M,
#     N,
#     degmax = 10,
#     λ = 0,
# )
#     x = discretize(domain, M)
#     ξ = discretize(domain, N)
#     L = (domain.right - domain.left)
#     h = filter.width
#
#     # Get reconstruction weights for each point
#     R = spzeros(N, M)
#     for n = 1:N
#         # Point
#         ξₙ = ξ[n]
#
#         # Move x by L * (shifts - 2) to get closer to ξₙ
#         dists = @. abs(ξₙ - x - [-L 0 L])
#         mininds = argmin(dists, dims = 2)
#         shifts = [mininds[m].I[2] for m in eachindex(mininds)]
#
#         # Find m such that ξₙ is reachable from xₘ
#         m = dists[mininds][:] .< h.(x)
#
#         # Polynomial degree (Taylor series order)
#         Q = min(degmax, max(1, floor(Int, √sum(m))))
#
#         # Vandermonde matrix
#         d = 1:Q
#         xₘ = x[m]'
#         hₘ = h.(xₘ)
#         sₘ = L * (shifts[m]' .- 2)
#
#         Zₙ = @. 1 / (2hₘ * factorial(d)) * ((xₘ + sₘ + hₘ - ξₙ)^d - (xₘ + sₘ - hₘ - ξₙ)^d)
#
#         # Right-hand side
#         vₙ = fill(0.0, Q)
#         vₙ[1] = 1.0
#
#         # Fit weights
#         rₙ = ridge(Zₙ, vₙ)
#         # rₙ = Zₙ \ vₙ
#
#         # Store weights
#         R[n, m] .= rₙ[:] ./ sum(rₙ)
#     end
#     dropzeros!(R)
#
#     R
# end

function reconstruction_matrix(
    filter::TopHatFilter,
    domain::PeriodicIntervalDomain,
    M,
    N,
    degmax = 10,
    λ = 0,
)
    x = discretize(domain, M)
    ξ = discretize(domain, N)
    L = (domain.right - domain.left)
    h = filter.width

    # Get reconstruction weights for each point
    R = spzeros(N, M)
    for n = 1:N
        # Point
        ξₙ = ξ[n]

        dists = @. abs(ξₙ - x - [-L 0 L])

        # Move x by L * (shifts - 2) to get closer to ξₙ
        mininds = argmin(dists, dims = 2)
        shifts = [mininds[m].I[2] for m in eachindex(mininds)]

        # Find m such that ξₙ is reachable from xₘ and include one point outside
        m = dists[mininds][:] .< 1.5h.(x)
        # m = mapreduce(s -> circshift(m, s), .|, [-1, 0, 1])

        # Vandermonde matrix
        hₘ = h.(x[m]')
        sₘ = L * (shifts[m]' .- 2)
        xₘ = x[m]' + sₘ

        # Polynomial degree (Taylor series order)
        Mₙ = length(xₘ)
        Q = min(degmax, Mₙ - 1)
        # Q = min(degmax, max(1, floor(Int, √Mₙ)))

        # Vandermonde matrix
        # aₘ = max.(xₘ - hₘ, domain.left - 0.001L)
        # bₘ = min.(xₘ + hₘ, domain.right + 0.001L)
        # Δhₘ = bₘ - aₘ
        ivals = xₘ .± hₘ
        # ivals = Interval.(aₘ, bₘ)

        ϕ = chebyshevt.(0:Q, [minimum(xₘ - 3hₘ)..maximum(xₘ + 3hₘ)])
        # ϕ = chebyshevt.(0:Q, [domain.left..domain.right])
        ϕ_int = integrate.(ϕ)

        Zₙ = spzeros(Q + 1, Mₙ)
        vₙ = zeros(Q + 1)
        for q = 1:Q+1
            for (i, ival) ∈ enumerate(ivals)
                Zₙ[q, i] =
                    (ϕ_int[q](ival.right) - ϕ_int[q](ival.left)) / (ival.right - ival.left)
            end
            vₙ[q] = ϕ[q](ξₙ)
        end

        rₙ = ridge(Zₙ, vₙ, λ)
        # rₙ = Zₙ \ vₙ
        # rₙ = fit(LassoRegression(λ; fit_intercept = false), Zₙ, vₙ)

        # Store weights
        R[n, m] .= rₙ[:] ./ sum(rₙ)
    end
    dropzeros!(R)

    R
end

function reconstruction_matrix(
    filter::ConvolutionalFilter,
    domain::ClosedIntervalDomain,
    M,
    N,
    degmax = 10,
    λ = 0,
)
    x = discretize(domain, M)
    ξ = discretize(domain, N)
    Δx = x[2] - x[1]
    Ival_domain = (domain.left - Δx / 1000)..(domain.right + Δx / 1000)
    h = filter.width
    G = filter.kernel

    # Get reconstruction weights for each point
    R = spzeros(N + 1, M + 1)
    for n = 1:(N+1)
        # Point
        ξₙ = ξ[n]

        # Find m such that ξₙ is reachable from xₘ
        dists = abs.(x .- ξₙ)
        m = dists .< h.(x)

        # Polynomial degree (Taylor series order)
        Mₙ = sum(m)
        Q = sum(m) - 1

        xₘ = x[m]
        hₘ = h.(xₘ)
        aₘ = max.(xₘ - hₘ, Ival_domain.left)
        bₘ = min.(xₘ + hₘ, Ival_domain.right)
        # Δhₘ = bₘ - aₘ
        # ivals = xₘ .± hₘ
        ivals = Interval.(aₘ, bₘ)

        # ϕ = chebyshevt.(0:Q, [minimum(xₘ - hₘ)..maximum(xₘ + hₘ)])
        ϕ = chebyshevt.(0:Q, [minimum(aₘ)..maximum(bₘ)])
        # ϕ = chebyshevt.(0:Q, [Ival_domain])

        # Polynomials evaluated at ξₙ
        vₙ = [ϕ(ξₙ) for ϕ ∈ ϕ]

        # Moment matrix
        Zₙ = spzeros(Q + 1, Mₙ)
        for mm = 1:Mₙ
            ival = ivals[mm]
            ϕₘₘ = Fun.(ϕ, [ival])
            kern = Fun(x -> G(x - xₘ[mm]), ival)
            Zₙ[:, mm] = sum.(kern .* ϕₘₘ) / sum(kern)
        end

        rₙ = ridge(Zₙ, vₙ, λ)
        # rₙ = Zₙ \ vₙ

        # Store weights
        R[n, m] .= rₙ[:] ./ sum(rₙ)
    end
    dropzeros!(R)

    R
end

function reconstruction_matrix(
    filter::ConvolutionalFilter,
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
    h = filter.width
    G = filter.kernel

    # Get reconstruction weights for each point
    R = spzeros(N, M)
    for n = 1:N
        # Point
        ξₙ = ξ[n]

        dists = @. abs(ξₙ - x - [-L 0 L])

        # Move x by L * (shifts - 2) to get closer to ξₙ
        mininds = argmin(dists, dims = 2)
        shifts = [mininds[m].I[2] for m in eachindex(mininds)]

        # Find m such that ξₙ is reachable from xₘ
        m = dists[mininds][:] .< h.(x)

        # Vandermonde matrix
        sₘ = L * (shifts[m]' .- 2)
        xₘ = x[m]' + sₘ
        hₘ = h.(xₘ)

        # Polynomial degree (Taylor series order)
        Q = sum(m) - 1
        Mₙ = sum(m)
        # Q = min(degmax, max(1, floor(Int, √Mₙ)))

        # Vandermonde matrix
        aₘ = max.(xₘ - hₘ, domain.left - Δx / 1000)
        bₘ = min.(xₘ + hₘ, domain.right + Δx / 1000)
        # Δhₘ = bₘ - aₘ
        ivals = xₘ .± hₘ
        # ivals = Interval.(aₘ, bₘ)

        ϕ = chebyshevt.(0:Q, [minimum(xₘ - hₘ)..maximum(xₘ + hₘ)])
        # ϕ = chebyshevt.(0:Q, [domain.left..domain.right])

        # Polynomials evaluated at ξₙ
        vₙ = [ϕ(ξₙ) for ϕ ∈ ϕ]

        # Polynomial moments
        Zₙ = spzeros(Q + 1, Mₙ)
        for mm = 1:Mₙ
            ival = ivals[mm]
            ϕₘₘ = Fun.(ϕ, [ival])
            kern = Fun(x -> G(x - xₘ[mm]), ival)
            kern /= sum(kern)
            Zₙ[:, mm] = sum.(kern .* ϕₘₘ)
        end

        rₙ = ridge(Zₙ, vₙ, λ)
        # rₙ = Zₙ \ vₙ

        # Store weights
        R[n, m] .= rₙ[:] ./ sum(rₙ)
    end
    dropzeros!(R)

    R
end
