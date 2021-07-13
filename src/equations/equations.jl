"""
Abstract equation.
"""
abstract type Equation end


"""
    AdvectionEquation(domain, filter = IdentityFilter)

Filtered advection equation.
"""
struct AdvectionEquation{D<:Domain,F<:Filter} <: Equation
    domain::D
    filter::F
end
AdvectionEquation(domain::D, filter::F = IdentityFilter()) where {D<:Domain,F<:Filter} =
    AdvectionEquation{D,F}(domain, filter)


"""
    DiffusionEquation(
        domain,
        filter = IdentityFilter();
        f = nothing,
        g_a = nothing,
        g_b = nothing
    )

Filtered diffusion equation.
"""
struct DiffusionEquation{D<:Domain,F<:Filter} <: Equation
    domain::D
    filter::F
    f::Union{Nothing,Function}
    g_a::Union{Nothing,Function}
    g_b::Union{Nothing,Function}
end
DiffusionEquation(
    domain::D,
    filter::F = IdentityFilter();
    f = nothing,
    g_a = nothing,
    g_b = nothing,
) where {D<:Domain,F<:Filter} = DiffusionEquation{D,F}(domain, filter, f, g_a, g_b)

"""
    BurgersEquation(domain, filter = IdentityFilter())

Filtered Burgers equation.
"""
struct BurgersEquation{D<:Domain,F<:Filter} <: Equation
    domain::D
    filter::F
end
BurgersEquation(domain::D, filter::F = IdentityFilter()) where {D<:Domain,F<:Filter} =
    BurgersEquation{D,F}(domain, filter)
