"""
    inverse_filter_matrix(f, x[, ε])

Approximate inverse of discrete filtering matrix, given filter `f`.
"""
function inverse_filter_matrix(::Filter, x) end


function inverse_filter_matrix(f::TopHatFilter, x, ε = 1e-6)

    n = length(x)
    h = f.width
    R = spzeros(n, n)

    # Get reconstruction weights for each point
    for i = 1:n
        # Point
        xᵢ = x[i]

        dists = @. abs(xᵢ - x - [-2π 0 2π])

        # Move x by 2π * (shifts - 2) to get closer to xᵢ
        mininds = argmin(dists, dims = 2)
        shifts = [mininds[j].I[2] for j in eachindex(mininds)]

        # Find j such that xᵢ is reachable from xⱼ
        j = dists[mininds][:] .< h.(x)

        # Polynomial degree (Taylor series order)
        deg = sum(j)

        # Vandermonde matrix
        d = 1:deg
        xⱼ = x[j]'
        hⱼ = h.(xⱼ)
        sⱼ = 2π * (shifts[j]' .- 2)

        Vᵢ = @. 1 / (2hⱼ * factorial(d)) * ((xⱼ + sⱼ + hⱼ - xᵢ)^d - (xⱼ + sⱼ - hⱼ - xᵢ)^d)

        # Right-hand side
        μᵢ = fill(0, deg)
        μᵢ[1] = 1

        # Fit weights
        # rᵢ = Vᵢ \  μᵢ
        rᵢ = (Vᵢ' * Vᵢ + ε * sparse(1.0I, deg, deg)) \ (Vᵢ' * μᵢ)

        # Store weights
        R[i, j] .= rᵢ
    end

    R
end


function inverse_filter_matrix(f::ConvolutionalFilter, x)
    error("Not implemented")
    n = length(x)
    G = f.kernel
    R = spzeros(n, n)
    R
end
