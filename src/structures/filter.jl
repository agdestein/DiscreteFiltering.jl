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
struct TopHatFilter{F<:Function} <: Filter
    width::F
end


"""
    ConvolutionalFilter(kernel)

Convolutional filter, parameterized by a filter kernel.
"""
struct ConvolutionalFilter{F<:Function} <: Filter
    kernel::F
end
