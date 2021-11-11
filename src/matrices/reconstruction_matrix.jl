"""
    reconstruction_matrix(filter, domain, M, N)

Approximate `R` of size `N × M`, the inverse of the discrete filtering matrix `W` of size `M × N`.
"""
function reconstruction_matrix end

function reconstruction_matrix(::IdentityFilter, domain::AbstractIntervalDomain, M, N; kwargs...)
    x = discretize(domain, M)
    ξ = discretize(domain, N)
    interpolation_matrix(ξ, x)
end

function reconstruction_matrix(
    filter::TopHatFilter,
    domain::ClosedIntervalDomain,
    M,
    N;
    degmax = 10,
    λ = 0,
)
    x = discretize(domain, M)
    ξ = discretize(domain, N)
    h = filter.width
    R = spzeros(N + 1, M + 1)
    for n = 1:(N+1)
        # Reconstruction point
        ξₙ = ξ[n]

        # Find m such that ξₙ is reachable from xₘ
        dists = abs.(x .- ξₙ)
        m = dists .< h.(x)
        Mₙ = sum(m)
        xₘ = x[m]'
        hₘ = h.(xₘ)
        aₘ = max.(xₘ - hₘ, domain.left)
        bₘ = min.(xₘ + hₘ, domain.right)
        Δhₘ = bₘ - aₘ

        # Polynomial degree (Taylor series order)
        Q = min(degmax + 1, max(1, Mₙ == 2 ? 2 : 3 ≤ Mₙ < 9 ? 3 : floor(Int, √Mₙ)))
        q = 1:Q

        # Vandermonde matrix
        Zₙ = @. 1 / (Δhₘ * factorial(q)) * ((bₘ - ξₙ)^q - (aₘ - ξₙ)^q)

        # Moments
        vₙ = fill(0.0, Q)
        vₙ[1] = 1.0

        # Fit weights
        rₙ = ridge(Zₙ, vₙ)
        # rₙ = Zₙ \ vₙ

        # Store row of weights
        R[n, m] .= rₙ[:] ./ sum(rₙ)
    end
    dropzeros!(R)

    R
end

# function reconstruction_matrix(
#     filter::TopHatFilter,
#     domain::PeriodicIntervalDomain,
#     M,
#     N;
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
#         # Reconstruction point
#         ξₙ = ξ[n]
#
#         # Move x by L * (shifts - 2) to get closer to ξₙ
#         dists = @. abs(ξₙ - x - [-L 0 L])
#         mininds = argmin(dists, dims = 2)
#         shifts = [mininds[m].I[2] for m in eachindex(mininds)]
#
#         # Find m such that ξₙ is reachable from xₘ
#         m = dists[mininds][:] .< h.(x)
#         Mₙ = sum(m)
#
#         # Vandermonde matrix
#         xₘ = x[m]'
#         hₘ = h.(xₘ)
#         sₘ = L * (shifts[m]' .- 2)
#
#         # Polynomial degree (Taylor series order)
#         Q = min(degmax + 1, max(1, Mₙ == 2 ? 2 : 3 ≤ Mₙ < 9 ? 3 : floor(Int, √Mₙ)))
#         q = 1:Q
#
#         # Vandermonde matrix
#         Zₙ = @. 1 / (2hₘ * factorial(q)) * ((xₘ + sₘ + hₘ - ξₙ)^q - (xₘ + sₘ - hₘ - ξₙ)^q)
#
#         # Moments
#         vₙ = fill(0.0, Q)
#         vₙ[1] = 1.0
#
#         # Fit weights
#         rₙ = ridge(Zₙ, vₙ)
#         # rₙ = Zₙ \ vₙ
#
#         # Store row of weights
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
    N;
    degmax = 10,
    λ = 0,
)
    x = discretize(domain, M)
    ξ = discretize(domain, N)
    L = (domain.right - domain.left)
    h = filter.width
    R = spzeros(N, M)
    for n = 1:N
        # Reconstruction point
        ξₙ = ξ[n]

        # Move x by L * (shifts - 2) to get closer to ξₙ
        dists = @. abs(ξₙ - x - [-L 0 L])
        mininds = argmin(dists, dims = 2)
        shifts = [mininds[m].I[2] for m in eachindex(mininds)]

        # Find m such that ξₙ is reachable from xₘ and include one point outside
        m = dists[mininds][:] .< 1.5h.(x)
        if !any(m)
            # No coarse point is inside the interval. Set closest weight to one
            m = argmin(dists; dims = (1,2))[1].I[1]
            R[n, m] = 1
            continue
        end
        # m = mapreduce(s -> circshift(m, s), .|, [-1, 0, 1])
        hₘ = h.(x[m]')
        sₘ = L * (shifts[m]' .- 2)
        xₘ = x[m]' + sₘ
        Mₙ = length(xₘ)
        # bₘ = min.(xₘ + hₘ, domain.right + 0.001L)
        # Δhₘ = bₘ - aₘ
        ivals = xₘ .± hₘ
        # ivals = Interval.(aₘ, bₘ)
        # aₘ = max.(xₘ - hₘ, domain.left - 0.001L)

        # Polynomial basis
        Q = min(degmax + 1, Mₙ)
        # Q = min(degmax + 1, max(1, Mₙ == 2 ? 2 : 3 ≤ Mₙ < 9 ? 3 : floor(Int, √Mₙ)))
        ϕ = chebyshevt.(0:Q-1, [minimum(xₘ - 1.5hₘ)..maximum(xₘ + 1.5hₘ)])
        # ϕ = chebyshevt.(0:Q-1, [domain.left..domain.right])
        ϕ_int = integrate.(ϕ)
        # Build reconstruction system
        Zₙ = spzeros(Q, Mₙ)
        vₙ = zeros(Q)
        for q = 1:Q
            for (i, ival) ∈ enumerate(ivals)
                Zₙ[q, i] =
                    (ϕ_int[q](ival.right) - ϕ_int[q](ival.left)) / (ival.right - ival.left)
            end
            vₙ[q] = ϕ[q](ξₙ)
        end

        # Fit reconstruction weights
        rₙ = ridge(Zₙ, vₙ, λ)
        # rₙ = Zₙ \ vₙ
        # rₙ = fit(LassoRegression(λ; fit_intercept = false), Zₙ, vₙ)

        # Store row of weights
        R[n, m] .= rₙ[:] ./ sum(rₙ)
    end
    dropzeros!(R)

    R
end

function reconstruction_matrix(
    filter::ConvolutionalFilter,
    domain::ClosedIntervalDomain,
    M,
    N;
    degmax = 10,
    λ = 0,
)
    x = discretize(domain, M)
    ξ = discretize(domain, N)
    Δx = x[2] - x[1]
    Ival_domain = (domain.left - Δx / 1000)..(domain.right + Δx / 1000)
    h = filter.width
    G = filter.kernel
    R = spzeros(N + 1, M + 1)
    for n = 1:(N+1)
        # Reconstruction point
        ξₙ = ξ[n]

        # Find m such that ξₙ is reachable from xₘ
        dists = abs.(x .- ξₙ)
        m = dists .< h.(x)
        xₘ = x[m]
        hₘ = h.(xₘ)
        aₘ = max.(xₘ - hₘ, Ival_domain.left)
        bₘ = min.(xₘ + hₘ, Ival_domain.right)
        # Δhₘ = bₘ - aₘ
        # ivals = xₘ .± hₘ
        ivals = Interval.(aₘ, bₘ)

        # Polynomial basis
        Mₙ = sum(m)
        Q = min(degmax + 1, Mₙ)
        # Q = min(degmax + 1, max(1, Mₙ == 2 ? 2 : 3 ≤ Mₙ < 9 ? 3 : floor(Int, √Mₙ)))
        # ϕ = chebyshevt.(0:Q-1, [minimum(xₘ - hₘ)..maximum(xₘ + hₘ)])
        ϕ = chebyshevt.(0:Q-1, [minimum(aₘ)..maximum(bₘ)])
        # ϕ = chebyshevt.(0:Q-1, [Ival_domain])

        # Polynomials evaluated at ξₙ
        vₙ = [ϕ(ξₙ) for ϕ ∈ ϕ]

        # Moment matrix
        Zₙ = spzeros(Q, Mₙ)
        for mm = 1:Mₙ
            ival = ivals[mm]
            ϕₘₘ = Fun.(ϕ, [ival])
            kern = Fun(x -> G(x - xₘ[mm]), ival)
            Zₙ[:, mm] = sum.(kern .* ϕₘₘ) / sum(kern)
        end

        # Fit weights
        rₙ = ridge(Zₙ, vₙ, λ)
        # rₙ = Zₙ \ vₙ

        # Store row of weights
        R[n, m] .= rₙ[:] ./ sum(rₙ)
    end
    dropzeros!(R)

    R
end

function reconstruction_matrix(
    filter::ConvolutionalFilter,
    domain::PeriodicIntervalDomain,
    M,
    N;
    degmax = 10,
    λ = 0,
)
    x = discretize(domain, M)
    ξ = discretize(domain, N)
    Δx = x[2] - x[1]
    L = (domain.right - domain.left)
    h = filter.width
    G = filter.kernel
    R = spzeros(N, M)
    for n = 1:N
        # Reconstruction point
        ξₙ = ξ[n]

        # Move x by L * (shifts - 2) to get closer to ξₙ
        dists = @. abs(ξₙ - x - [-L 0 L])
        mininds = argmin(dists, dims = 2)
        shifts = [mininds[m].I[2] for m in eachindex(mininds)]

        # Find m such that ξₙ is reachable from xₘ
        m = dists[mininds][:] .< h.(x)
        sₘ = L * (shifts[m]' .- 2)
        xₘ = x[m]' + sₘ
        hₘ = h.(xₘ)
        aₘ = max.(xₘ - hₘ, domain.left - Δx / 1000)
        bₘ = min.(xₘ + hₘ, domain.right + Δx / 1000)
        # Δhₘ = bₘ - aₘ
        ivals = xₘ .± hₘ
        # ivals = Interval.(aₘ, bₘ)
        Mₙ = sum(m)

        # Polynomial degree
        Q = min(degmax + 1, Mₙ)
        # Q = min(degmax + 1, max(1, Mₙ == 2 ? 2 : 3 ≤ Mₙ < 9 ? 3 : floor(Int, √Mₙ)))
        ϕ = chebyshevt.(0:Q-1, [minimum(xₘ - hₘ)..maximum(xₘ + hₘ)])
        # ϕ = chebyshevt.(0:Q-1, [domain.left..domain.right])

        # Polynomials evaluated at ξₙ
        vₙ = [ϕ(ξₙ) for ϕ ∈ ϕ]

        # Polynomial moments
        Zₙ = spzeros(Q, Mₙ)
        for mm = 1:Mₙ
            ival = ivals[mm]
            ϕₘₘ = Fun.(ϕ, [ival])
            kern = Fun(x -> G(x - xₘ[mm]), ival)
            kern /= sum(kern)
            Zₙ[:, mm] = sum.(kern .* ϕₘₘ)
        end

        # Fit weights
        rₙ = ridge(Zₙ, vₙ, λ)
        # rₙ = Zₙ \ vₙ

        # Store row of weights
        R[n, m] .= rₙ[:] ./ sum(rₙ)
    end
    dropzeros!(R)

    R
end
