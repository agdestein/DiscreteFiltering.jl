"""
    ridge(A, b, λ = 0)

Compute the solution to min ||Ax - b||² + λ||x||².
"""
ridge(A, b, λ = 0) = (A'A + λ * I) \ A'b
