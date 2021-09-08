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
