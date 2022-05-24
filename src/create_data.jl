function u(c, x, t)
    K = size(c, 1) ÷ 2
    real(sum(c * exp(2π * im * k * (x - t)) for (k, c) ∈ zip(-K:K, c)))
end
function ∂u∂t(c, x, t)
    K = size(c, 1) ÷ 2
    real(-2π * im * sum(c * k * exp(2π * im * k * (x - t)) for (k, c) ∈ zip(-K:K, c)))
end
function ū(Ĝ, c, x, t)
    K = size(c, 1) ÷ 2
    real(sum(c * Ĝ(k, x) * exp(2π * im * k * (x - t)) for (k, c) ∈ zip(-K:K, c)))
end
function ∂ū∂t(Ĝ, c, x, t)
    K = size(c, 1) ÷ 2
    real(
        sum(
            -2π * im * k * c * Ĝ(k, x) * exp(2π * im * k * (x - t)) for
            (k, c) ∈ zip(-K:K, c)
        ),
    )
end

"""
    create_data_exact(A, c, ξ, t)

Create exact unfiltered solutions.
"""
function create_data_exact(A, c, ξ, t)
    N = length(ξ)
    nfreq, nsample = size(c)
    K = nfreq ÷ 2
    nt = length(t)
    e = [exp(2π * im * k * ξ) for ξ ∈ ξ, k ∈ (-K):K]
    u₀ = real.(e * c)
    u = zeros(N, nsample, nt)
    ∂u∂t = zeros(N, nsample, nt)
    for (i, t) ∈ enumerate(t)
        Et = [exp(-2π * im * k * t) for k ∈ (-K):K]
        u[:, :, i] = real.(e * (Et .* c))
        # ∂u∂t[:, :, i] = real.(e * (-2π * im .* (-K:K) .* Et .* c))
        # ∂u∂t[:, :, i] = A * real.(e * (Et .* c))
    end
    (; u)
    # (; u, ∂u∂t)
end

"""
    create_data_dns(A, c, ξ, t)

Create DNS solution (numerical approximation).
"""
function create_data_dns(A, c, ξ, t; abstol = 1e-10, reltol = 1e-8, kwargs...)
    N = length(ξ)
    nfreq, nsample = size(c)
    K = nfreq ÷ 2
    nt = length(t)
    e = [exp(2π * im * k * ξ) for ξ ∈ ξ, k ∈ -K:K]
    u₀ = real.(e * c)
    sol = S!(A, u₀, t; abstol, reltol, kwargs...)
    u = Array(sol)
    # ∂u∂t = zeros(N, nsample, nt)
    # for (i, t) ∈ enumerate(t)
    #     @views mul!(∂u∂t[:, :, i], A, u[:, :, i])
    # end
    (; u)
    # (; u, ∂u∂t)
end

"""
    create_data_filtered(W, A, dns)

Create discrete filtered solution from DNS
"""
function create_data_filtered(W, A, dns)
    (; u) = dns
    # (; u, ∂u∂t) = dns
    M = size(W, 1)
    _, nsample, nt = size(u)
    ū = zeros(M, nsample, nt)
    ∂ū∂t = zeros(M, nsample, nt)
    WA = W * A
    for i = 1:nt
        # ū[:, :, i] = W * u[:, :, i]
        # # ∂ū∂t[:, :, i] = W * ∂u∂t[:, :, i]
        # ∂ū∂t[:, :, i] = WA * u[:, :, i]
        @views mul!(ū[:, :, i], W, u[:, :, i])
        # @views mul!(∂ū∂t[:, :, i], W, ∂u∂t[:, :, i])
        @views mul!(∂ū∂t[:, :, i], WA, u[:, :, i])
    end
    (; ū, ∂ū∂t)
end
