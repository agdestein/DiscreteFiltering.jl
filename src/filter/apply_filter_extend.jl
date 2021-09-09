"""
    apply_filter_extend(u, filter, domain)

Extend function beyond beyond boundaries before filtering.
"""
function apply_filter_extend(u, filter, domain)
    error("Not implemented")
end

function apply_filter_extend(u_int, filter::TopHatFilter, domain::ClosedIntervalDomain)
    h = filter.width
    a, b = domain.left, domain.right

    function ū_ext(x, t)
        xₗ, xᵣ = x - h(x), x + h(x)
        1 / 2h(x) * (
            u_int(min(b, xᵣ), t) - u_int(max(a, xₗ), t) +
            g_a(t) * max(0.0, a - xₗ) +
            g_b(t) * max(0.0, xᵣ - b)
        )
    end
end


"""
    apply_filter_extend(u, filter, domain)

Extend function beyond beyond boundaries before filtering.
"""
function apply_filter_extend(u, filter::ConvolutionalFilter, domain::ClosedIntervalDomain)
    h = filter.width
    G = filter.kernel
    a, b = domain.left, domain.right

    # Exact filtered solution
    function ū_ext(x)
        x₋, x₊ = x - h(x), x + h(x)
        xₗ = max(a, x₋)
        xᵣ = min(b, x₊)

        Gfun = Fun(ξ -> G(ξ - x), x₋..x₊)
        Gfun /= sum(Gfun)

        S = sum(Fun(Gfun, xₗ..xᵣ) * Fun(u, xₗ..xᵣ))
        if x₋ < a
            S += u(a) * sum(Fun(Gfun, x₋..a))
        end
        if b < x₊
            S += u(b) * sum(Fun(Gfun, b..x₊))
        end

        S
    end
end
