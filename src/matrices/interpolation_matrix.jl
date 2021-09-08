"""
    interpolation_matrix(x, ξ)

Create interpolation matrix from grid `ξ` to grid `x`.
"""
function interpolation_matrix(x, ξ)
    A = spzeros(length(x), length(ξ))
    ξᵀ = ξ'
    n = ξᵀ[[1], 1:end-1] .≤ x .< ξᵀ[[1], 2:end]
    for m = 1:length(x)-1
        nₘ = findfirst(n[m, :])
        A[m, nₘ] = (ξᵀ[nₘ+1] - x[m]) / (ξᵀ[nₘ+1] - ξᵀ[nₘ])
        A[m, nₘ+1] = (x[m] - ξᵀ[nₘ]) / (ξᵀ[nₘ+1] - ξᵀ[nₘ])
    end
    A[x.<ξ[1], 1] .= 1
    A[ξ[end].≤x, end] .= 1
end
