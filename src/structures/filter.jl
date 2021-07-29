"""
Abstract continuous filter.
"""
abstract type Filter end


"""
    IdentityFilter()

Identity filter, which does not filter.
"""
struct IdentityFilter <: Filter end


"""
    TopHatFilter(width)

Top hat filter, parameterized by a variable filter width.
"""
struct TopHatFilter <: Filter
    width::Function
end


"""
    ConvolutionalFilter(kernel)

Convolutional filter, parameterized by a filter kernel.
"""
struct ConvolutionalFilter <: Filter
    width::Function
    kernel::Function
end

"""
GaussianFilter(h, σ) -> ConvolutionalFilter
GaussianFilter(σ) -> ConvolutionalFilter

Create Gaussian ConvolutionalFilter with domain width `2h` and variance `σ^2`.
"""
GaussianFilter(h, σ) = ConvolutionalFilter(h, x -> 1 / √(2π * σ^2) * exp(-x^2 / 2σ^2))
GaussianFilter(σ) = GaussianFilter(x -> 10σ, σ)



"""
    apply_filter(u, filter, domain)

Apply `filter` to a spatial function `u` defined on `domain`.
"""
function apply_filter(u, filter::Filter, domain::Domain)
    error("Not implemented")
end

function apply_filter(u_int, filter::TopHatFilter, domain::ClosedIntervalDomain)
    h = filter.width
    a, b = domain.left, domain.right

    # Exact filtered solution
    ū(x) = begin
        α = max(a, x - h(x))
        β = min(b, x + h(x))
        1 / (β - α) * (u_int(β, t) - u_int(α))
    end

    ū
end

function apply_filter(u_int, filter::TopHatFilter, domain::PeriodicIntervalDomain)
    h = filter.width
    a, b = domain.left, domain.right

    # Exact filtered solution
    ū(x) = begin
        α = x - h(x)
        β = x + h(x)
        1 / (β - α) * (u_int(β) - u_int(α))
    end

    ū
end

function apply_filter(u, filter::ConvolutionalFilter, domain::ClosedIntervalDomain)
    h = filter.width
    G = filter.kernel
    a, b = domain.left, domain.right

    # Exact filtered solution
    ū(x) = begin
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
    ū(x) = begin
        α = x - h(x)
        β = x + h(x)

        u_ext(x) = x < a ? u(x + L) : (x < b ? u(x) : u(x - L))

        ufun = Fun(u, α..β)
        Gfun = Fun(ξ -> G(ξ - x), α..β)

        sum(Gfun * ufun) / sum(Gfun)
    end

    ū
end


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

    ū_ext(x, t) = begin
        xₗ, xᵣ = x - h(x), x + h(x)
        1 / 2h(x) * (
            u_int(min(b, xᵣ), t) - u_int(max(a, xₗ), t) +
            g_a(t) * max(0.0, a - xₗ) +
            g_b(t) * max(0.0, xᵣ - b)
        )
    end

    ū_ext
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
    ū_ext(x) = begin
        xₗ, xᵣ = x - h(x), x + h(x)

        u_ext(x) = x < a ? u(a) : (x < b ? u(x) : u(b))

        ufun = Fun(u, xₗ..xᵣ)
        Gfun = Fun(ξ -> G(ξ - x), xₗ..xᵣ)

        sum(Gfun * ufun) / sum(Gfun)
    end

    ū_ext
end
