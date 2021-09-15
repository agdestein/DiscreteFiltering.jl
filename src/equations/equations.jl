"""
Abstract equation.
"""
abstract type Equation end

"""
    AdvectionEquation(domain, filter = IdentityFilter())

Filtered advection equation.
"""
Base.@kwdef struct AdvectionEquation{D<:Domain,F<:Filter} <: Equation
    domain::D
    filter::F = IdentityFilter()
end

"""
    DiffusionEquation(
        domain,
        filter = IdentityFilter(),
        f = nothing,
        g_a = nothing,
        g_b = nothing
    )

Filtered diffusion equation.
"""
Base.@kwdef struct DiffusionEquation{D<:Domain,F<:Filter} <: Equation
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
Base.@kwdef struct BurgersEquation{D<:Domain,F<:Filter} <: Equation
    domain::D
    filter::F = IdentityFilter()
end
