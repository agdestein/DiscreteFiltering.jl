"""
    fit_Cbar(domain, filter::TopHatFilter, u₀, U₀, M, t; λ = 1e-4, method = :ridge)

Fit the filtered convection matrix `C̄` for a `TopHatFilter`, given a list of initial
conditions `u₀` and associated antiderivatives `U₀`.

The unfiltered equation is given by

``\\frac{d u_h}{d t} + C u_h = 0.``

The associated filtered equation is

``\\frac{d \\bar{u}_h}{d t} + \\bar{C} \\bar{u}_h = 0.``

This function fits `C̄` using a regularized least-squares regression based on
the above equation evaluated at different time steps and spatial points.
The regularizatin method is either `:ridge` or `:lasso`, the latter being
able to identify sparsity.

The sparsity pattern of `C̄` is also enforced as a constraint.
"""
function fit_Cbar(domain, filter::TopHatFilter, u₀, U₀, M, N, t; λ = 1e-4, method = :ridge)
    L = domain.right - domain.left
    x = discretize(domain, M)
    ξ = discretize(domain, N)
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

    ℒ = domain isa PeriodicIntervalDomain ? (-L, 0, L) : (0,)
    # C̄ = spzeros(M, M)
    C̄ = zeros(M, M)
    Cᴹ = advection_matrix(domain, M)
    Cᴺ = advection_matrix(domain, N)
    Wpat = [mapreduce(ℓ -> -h(x[m]) .< x[m] + ℓ - ξ[n] ≤ h(x[m]), |, ℒ) for m = 1:M, n = 1:N]
    # Wpat = [mapreduce(ℓ -> abs(x[m] + ℓ - ξ[n]) ≤ h(x[m]), |, (-L, 0, L)) for m = 1:M, n = 1:N]
    Wpat = mapreduce(i -> circshift(Wpat, (0, i)), .|, -2:2)
    Cpat = Cᴺ .≠ 0
    Rpat = Wpat'
    inds = Wpat * Cpat * Rpat .≠ 0
    
    if method == :ridge
        reg = RidgeRegression(λ; fit_intercept = false)
    elseif method == :lasso
        reg = LassoRegression(λ; fit_intercept = false)
    else
        error("Unsupported method")
    end
    # for m = 1:M
    #     # nₘ = inds[m, :]
    #     nₘ = 1:M
    #     Uₘ = U[nₘ, :]
    #     # Uₘ = U
    #     # c̄ = fit(reg, Uₘ', Uₜ[m, :])
    #     # c̄ = (Uₘ * Uₘ' + λ * I) \ (Uₘ * Uₜ[m, :] - λ * Cᴹ[m, nₘ])
    #     c̄ = (Uₘ * Uₘ' + λ * I) \ (Uₘ * Uₜ[m, :])
    #     C̄[m, nₘ] = -c̄
    #     # C̄[m, :] = -c̄
    #     # c̄ = fit(reg, U', Uₜ[m, :])
    #     # C̄[m, :] = -c̄
    # end
    C̄ = -(Uₜ * U' - λ * Cᴹ) / (U * U' + λ * I)
    # C̄ = -Uₜ * U' / (U * U' + λ * I)

    C̄
end

function fit_Cbar_approx(
    domain,
    filter::TopHatFilter,
    uₓ,
    u,
    M,
    N,
    t;
    λ = 1e-4,
    method = :ridge,
)
    L = domain.right - domain.left
    x = discretize(domain, M)
    ξ = discretize(domain, N)
    h = filter.width

    # Observation matrices
    Uₜ = zeros(M, 0)
    U = zeros(M, 0)
    for (uₓ, u) ∈ zip(uₓ, u)
        Uₜ = [Uₜ uₓ.(x .- t')]
        U = [U u.(x .- t')]
    end

    W = filter_matrix_meshwidth(filter, domain, M)
    # W = I

    Ūₜ = W * Uₜ
    Ū = W * U

    ℒ = domain isa PeriodicIntervalDomain ? [-L, 0, L] : [0]
    C̄ = spzeros(M, M)
    Cᴺ = advection_matrix(domain, N)
    Wpat = [mapreduce(ℓ -> -h(x[m]) .< x[m] + ℓ - ξ[n] ≤ h(x[m]), |, ℒ) for m = 1:M, n = 1:N]
    Wpat = mapreduce(i -> circshift(Wpat, (0, i)), .|, -1:1)
    Cpat = Cᴺ .≠ 0
    Rpat = Wpat'
    inds = Wpat * Cpat * Rpat .≠ 0
    if method == :ridge
        reg = RidgeRegression(λ; fit_intercept = false)
    elseif method == :lasso
        reg = LassoRegression(λ; fit_intercept = false)
    else
        error("Unsupported method")
    end
    for m = 1:M
        nₘ = inds[m, :]
        Uₘ = Ū[nₘ, :]
        # c̄ = fit(reg, Uₘ', Ūₜ[m, :])
        c̄ = (Uₘ * Uₘ' + λ * I) \ (Uₘ * Ūₜ[m, :])
        C̄[m, nₘ] = c̄
        # c̄ = fit(reg, U', Ūₜ[m, :])
        # C̄[m, :] = -c̄
    end

    C̄
end
