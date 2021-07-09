"""
Abstract continuous filter.
"""
abstract type Filter end


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
    kernel::Function
end
