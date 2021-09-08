

"""
apply_filter(u, filter, domain)

Apply `filter` to a spatial function `u` defined on `domain`.
"""
function apply_filter end

apply_filter(u, ::IdentityFilter, ::Domain) = u

function apply_filter(u, filter::TopHatFilter, domain::ClosedIntervalDomain)
    h = filter.width
    a, b = domain.left, domain.right

    ufun = Fun(u, domain.left..domain.right)
    u_int = integrate(ufun)

    # Exact filtered solution
    function ū(x)
        α = max(a, x - h(x))
        β = min(b, x + h(x))
        1 / (β - α) * (u_int(β) - u_int(α))
    end

    ū
end

function apply_filter(u, filter::TopHatFilter, domain::PeriodicIntervalDomain)
    h = filter.width
    a, b = domain.left, domain.right
    L = b - a

    ufun = Fun(u, domain.left..domain.right)
    u_int = integrate(ufun)
    i(α, β) = u_int(β) - u_int(α)

    # Exact filtered solution
    function ū(x)
        α = x - h(x)
        β = x + h(x)

        xₗ = max(a, α)
        xᵣ = min(β, b)

        I = i(xₗ, xᵣ)
        if α < a
            I += i(α + L, b)
        end
        if b < β
            I += i(a, β - L)
        end

        I / (β - α)
    end

    ū
end

function apply_filter(u, filter::ConvolutionalFilter, domain::ClosedIntervalDomain)
    h = filter.width
    G = filter.kernel
    a, b = domain.left, domain.right

    # Exact filtered solution
    function ū(x)
        α = max(a, x - h(x))
        β = min(b, x + h(x))

        ufun = Fun(u, α..β)
        Gfun = Fun(ξ -> G(ξ - x), α..β)

        sum(Gfun * ufun) / sum(Gfun)
    end

    ū
end

function apply_filter(u, filter::ConvolutionalFilter, domain::PeriodicIntervalDomain)
    h = filter.width
    G = filter.kernel
    a, b = domain.left, domain.right
    L = b - a

    # Exact filtered solution
    function ū(x)
        α = x - h(x)
        β = x + h(x)

        u_ext(x) = x < a ? u(x + L) : (x < b ? u(x) : u(x - L))

        ufun = Fun(u, α..β)
        Gfun = Fun(ξ -> G(ξ - x), α..β)

        sum(Gfun * ufun) / sum(Gfun)
    end

    ū
end
