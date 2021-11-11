"""
    get_W_R(f::TopHatFilter, domain, M, N; kwargs...)

Construct `W` and `R` using a simplified counter rule. 

    get_W_R(f, domain, M, N; kwargs...)

Fallback function for the general case. Calls `filter_matrix` and
`reconstruction_matrix` separately.
"""
function get_W_R end

function get_W_R(f, domain, M, N; kwargs...)
    W = filter_matrix(f, domain, M, N; kwargs...)
    R = reconstruction_matrix(f, domain, M, N; kwargs...)

    W, R
end

function get_W_R(f::TopHatFilter, domain::AbstractIntervalDomain, M, N; kwargs...)
    h = f.width
    L = domain.right - domain.left
    ℒ = domain isa PeriodicIntervalDomain ? [-L, 0, L] : [0]
    x = discretize(domain, M)
    ξ = discretize(domain, N)
    W = spzeros(eltype(x), length(x), length(ξ))
    for m = 1:length(x)
        xₘ = x[m]
        # nₘ = @. (-h₀ < (ξ - x[m]) ≤ h₀)
        nₘ = mapreduce(ℓ -> -h(xₘ) .< (ξ .+ ℓ .- xₘ) .≤ h(xₘ), .|, ℒ)
        Nₘ = sum(nₘ)
        W[m, nₘ] .= 1 // Nₘ
    end

    hx = h.(x)
    R = spzeros(eltype(x), length(ξ), length(x))
    for n = 1:length(ξ)
        ξₙ = ξ[n]
        mₙ = mapreduce(ℓ -> -hx .< (ξₙ .+ ℓ .- x) .≤ hx, .|, ℒ)
        Mₙ = sum(mₙ)
        R[n, mₙ] .= 1 // Mₙ
    end

    W, R
end

