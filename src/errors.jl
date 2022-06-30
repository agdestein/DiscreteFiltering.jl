"""
    relerrs(A, uₜ, t; kwargs...)

Time dependent relative errors, averaged over data samples.
"""
function relerrs(A, uₜ, t; kwargs...)
    nsample = size(uₜ, 2)
    u₀ = uₜ[:, :, 1]
    sol = S!(A, u₀, t; kwargs...)
    sol.retcode == :Success || return fill(NaN, length(t))
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
    relerr(A, uₜ, t; kwargs...)

Relative error, averaged over time and data samples.
"""
relerr(A, uₜ, t; kwargs...) = sum(relerrs(A, uₜ, t; kwargs...)) / length(t)

"""
    spectral_relerr(A, uₜ, t; kwargs...)

Time dependent spectral relative errors, averaged over data samples.
"""
function spectral_relerr(A, uₜ, t; kwargs...)
    M, nsample = size(uₜ)
    u₀ = uₜ[:, :, 1]
    sol = S!(A, u₀, t; kwargs...)
    sol.retcode == :Success || return fill(NaN, length(t))
    errs = zeros(M, length(t))
    for i ∈ eachindex(t)
        for j = 1:nsample
            errs[:, i] += @views(
                abs.(fft(sol[:, j, i]) - fft(uₜ[:, j, i])) ./ sum(abs.(fft(uₜ[:, j, i])))
            )
        end
        errs[i] /= nsample
    end
    errs
end
