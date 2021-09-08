function apply_filter_int(u_int, filter::TopHatFilter, domain::ClosedIntervalDomain)
    h = filter.width
    a, b = domain.left, domain.right

    # Exact filtered solution
    function ū(x)
        α = max(a, x - h(x))
        β = min(b, x + h(x))
        1 / (β - α) * (u_int(β, t) - u_int(α))
    end

    ū
end

function apply_filter_int(u_int, filter::TopHatFilter, domain::PeriodicIntervalDomain)
    h = filter.width
    a, b = domain.left, domain.right

    # Exact filtered solution
    function ū(x)
        α = x - h(x)
        β = x + h(x)
        1 / (β - α) * (u_int(β) - u_int(α))
    end

    ū
end
