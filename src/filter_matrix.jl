"""
    filter_matrix(f, x[, ε])

Assemble discrete filtering matrix from a continuous filter `f`.
"""
function filter_matrix(::Filter, x)
    error("Not implemented")
end


function filter_matrix(f::TopHatFilter, x, ε = 1e-6)

    n = length(x)
    h = f.width
    τ(x) = (x - π) / 2π
    τ(x, a, b) = (x - (a + b) / 2) / (b - a)
    degmax = 1000
    ϕ = [ChebyshevT([fill(0, i); 1]) for i = 0:degmax]

    W = spzeros(n, n)
    for i = 1:n
        # Point
        xᵢ = x[i]

        # Filter width at point
        hᵢ = h(xᵢ)

        # Indices of integration points in circular reference
        Ival = Interval(xᵢ - hᵢ, xᵢ + hᵢ)
        inds_left = x .∈ Ival + 2π
        inds_mid = x .∈ Ival
        inds_right = x .∈ Ival - 2π
        inds = inds_left .| inds_mid .| inds_right
        deg = min(sum(inds) - 1, degmax)

        # Polynomials evaluated at integration points
        Vᵢ = vander(ChebyshevT, τ.(x[inds]), deg)'
        # Vᵢ = vander(ChebyshevT, τ.(x[inds], xᵢ - hᵢ, xᵢ + hᵢ), deg)'

        # Polynomial moments around point
        Domain = Interval(0, 2π)
        I_left = (Ival + 2π) ∩ Domain
        I_mid = Ival ∩ Domain
        I_right = (Ival - 2π) ∩ Domain
        μᵢ = integrate.(ϕ[1:deg+1], τ(I_left.first), τ(I_left.last))
        μᵢ .+= integrate.(ϕ[1:deg+1], τ(I_mid.first), τ(I_mid.last))
        μᵢ .+= integrate.(ϕ[1:deg+1], τ(I_right.first), τ(I_right.last))
        # μᵢ = integrate.(ϕ[1:deg+1], τ(Ival.first), τ(Ival.last))
        # μᵢ = integrate.(ϕ[1:deg+1], -1, 1)
        μᵢ .*= 2π / 2hᵢ
        # μᵢ .*= 2hᵢ / 2π
        # μᵢ ./= 2

        # Fit weights
        npoint = size(Vᵢ, 2)
        wᵢ = fill(1 / npoint, npoint)
        try
            # wᵢ = Vᵢ \ μᵢ
            wᵢ = (Vᵢ' * Vᵢ + ε * sparse(1.0I, npoint, npoint)) \ (Vᵢ' * μᵢ)
        catch e
            display(e)
            println(i)
        end

        # Store weights
        W[i, inds] .= wᵢ
    end

    W
end


function filter_matrix(f::ConvolutionalFilter, x)
    error("Not implemented")
    n = length(x)
    G = f.kernel
    W = spzeros(n, n)
    W
end
