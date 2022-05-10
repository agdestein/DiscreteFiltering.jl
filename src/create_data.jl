"""
Exact unfiltered solution
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
    (; u₀, u)
    # (; u₀, u, ∂u∂t)
end

"""
Create DNS solution (numerical approximation)
"""
function create_data_dns(A, c, ξ, t)
    N = length(ξ)
    nfreq, nsample = size(c)
    K = nfreq ÷ 2
    nt = length(t)
    e = [exp(2π * im * k * ξ) for ξ ∈ ξ, k ∈ (-K):K]
    u₀ = real.(e * c)
    sol = S!(A, u₀, t) #; reltol = 1e-8, abstol = 1e-10)
    u = Array(sol)
    # ∂u∂t = zeros(N, nsample, nt)
    # for (i, t) ∈ enumerate(t)
    #     @views mul!(∂u∂t[:, :, i], A, u[:, :, i])
    # end
    (; u₀, u)
    # (; u₀, u, ∂u∂t)
end

"""
Create discrete filtered solution from DNS
"""
function create_data_filtered(W, A, dns)
    (; u₀, u) = dns
    # (; u₀, u, ∂u∂t) = dns
    M = size(W, 1)
    _, nsample, nt = size(u)
    ū₀ = W * u₀
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
    (; ū₀, ū, ∂ū∂t)
end
