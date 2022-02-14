"""
    fit_Dbar(domain, filter::TopHatFilter, u₀, U₀, M, t; λ = 1e-4, method = :ridge)

Fit the filtered convection matrix `C̄` for a `TopHatFilter`, given a list of initial
conditions `u₀` and associated antiderivatives `U₀`.

The unfiltered equation is given by

``\\frac{d u_h}{d t} = D u_h.``

The associated filtered equation is

``\\frac{d \\bar{u}_h}{d t} = \\bar{D} \\bar{u}_h.``

This function fits `C̄` using a regularized least-squares regression based on
the above equation evaluated at different time steps and spatial points.
The regularizatin method is either `:ridge` or `:lasso`, the latter being
able to identify sparsity.

The sparsity pattern of `C̄` is also enforced as a constraint.
"""
function fit_Dbar(domain, filter::TopHatFilter, u₀, U₀, M, t; λ = 1e-4, method = :ridge)
    L = domain.right - domain.left
    x = discretize(domain, M)
    h = filter.width

    # Observation matrices
    U = zeros(M, 0)
    Uₜ = zeros(M, 0)
    for (u₀, U₀) ∈ zip(u₀, U₀)
        ū(x, t) = (U₀(x + h(x) - t) - U₀(x - h(x) - t)) / 2h(x)
        ūₜ(x, t) = -(u₀(x + h(x) - t) - u₀(x - h(x) - t)) / 2h(x)
        U = [U ū.(x, t')]
        Uₜ = [Uₜ ūₜ.(x, t')]
    end

    ℒ = domain isa PeriodicIntervalDomain ? [-L, 0, L] : [0]
    C̄ = spzeros(M, M)
    if method == :ridge
        reg = RidgeRegression(λ; fit_intercept = false)
    elseif method == :lasso
        reg = LassoRegression(λ; fit_intercept = false)
    else
        error("Unsupported method")
    end
    for m = 1:M
        xₘ = x[m]
        nₘ = mapreduce(ℓ -> -h(xₘ) .< (x .+ ℓ .- xₘ) .≤ h(xₘ), .|, ℒ)
        nₘ = mapreduce(i -> circshift(nₘ, i), .|, -1:1)
        Uₘ = U[nₘ, :]
        c̄ = fit(reg, Uₘ', Uₜ[m, :])
        # c̄ = (Uₘ * Uₘ' + λ * I) \ (Uₘ * Uₜ[m, :])
        C̄[m, nₘ] = -c̄
        # c̄ = fit(reg, U', Uₜ[m, :])
        # C̄[m, :] = -c̄
    end

    C̄
end
