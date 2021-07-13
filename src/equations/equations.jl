"""
Abstract equation.
"""
abstract type Equation end


"""
    AdvectionEquation(domain, filter = IdentityFilter)

Filtered advection equation.
"""
@with_kw struct AdvectionEquation{D<:Domain,F<:Filter} <: Equation
    domain::D
    filter::F = IdentityFilter()
end


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
@with_kw struct DiffusionEquation{D<:Domain,F<:Filter} <: Equation
    domain::D
    filter::F = IdentityFilter()
    f::Union{Nothing,Function} = nothing
    g_a::Union{Nothing,Function} = nothing
    g_b::Union{Nothing,Function} = nothing
end


"""
    BurgersEquation(domain, filter = IdentityFilter())

Filtered Burgers equation.
"""
@with_kw struct BurgersEquation{D<:Domain,F<:Filter} <: Equation
    domain::D
    filter::F = IdentityFilter()
end
