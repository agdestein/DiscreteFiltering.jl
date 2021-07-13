"""
gaussian(σ²) -> Function

Create Gaussian function with variance `σ²`.
"""
gaussian(σ²) = x -> 1 / √(2π * σ²) * exp(-x^2 / 2σ²)
