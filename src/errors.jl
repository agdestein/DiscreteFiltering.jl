"""
Relative errors
"""
function relerrs(A, u₀, uₜ, t; kwargs...)
    nsample = size(u₀, 2)
    sol = S!(A, u₀, t; kwargs...)
    errs = zeros(length(t))
    for i ∈ eachindex(t)
        for j = 1:nsample
            errs[i] += @views norm(sol[:, j, i] - uₜ[:, j, i]) / norm(uₜ[:, j, i])
        end
        errs[i] /= nsample
    end
    errs
end

"""
Relative errors (time averaged)
"""
relerr(A, u₀, uₜ, t; kwargs...) = sum(relerrs(A, u₀, uₜ, t; kwargs...)) / length(t)
