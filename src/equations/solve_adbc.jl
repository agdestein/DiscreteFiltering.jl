"""
    solve_adbc(equation, u, tlist, n, dt = (tlist[2] - tlist[1]) / 200)

Solve filtered equation using approximate deconvolution boundary conditions (ADBC). This
approach was proposed in [Borggaard 2006].

J. Borggaard, T. Iliescu, Approximate deconvolution boundary conditions for large eddy
simulation, Applied Mathematics Letters 19 (8) (2006) 735–740.
doi: https://doi.org/10.1016/j.aml.2005.08.022.
URL: https://www.sciencedirect.com/science/article/pii/S0893965905003319
"""
function solve_adbc(
    equation::DiffusionEquation{ClosedIntervalDomain{T},TopHatFilter},
    u,
    tlist,
    n,
    Δt = (tlist[2] - tlist[1]) / 200,
) where {T}

    @unpack domain, filter, f, g_a, g_b = equation
    x = discretize(domain, n)
    Δx = x[2] - x[1]
    h = filter.width
    all(h.(x) .≈ h(x[1])) || error("ADBC requires constant filter width")
    h₀ = h(x[1])
    a, b = domain.left, domain.right

    # Get matrices
    D = diffusion_matrix(domain, n)
    W = filter_matrix(filter, domain, n)
    R = inverse_filter_matrix(filter, domain, n)

    W₀ = W[1, :]'
    Wₙ = W[end, :]'

    ūᵏ = W * u.(x)
    ūᵏ⁺¹ = copy(ūᵏ)
    uᵏ = copy(ūᵏ)
    ũᵏ⁺¹ = copy(ūᵏ)
    tᵏ = tlist[1]
    while tᵏ + Δt ≤ tlist[2]
        # Current approximate unfiltered solution
        mul!(uᵏ, R, ūᵏ)

        # Nex approximate unfiltered solution
        ũᵏ⁺¹ .= uᵏ
        ũᵏ⁺¹[1] = g_a(tᵏ + Δt)
        ũᵏ⁺¹[end] = g_b(tᵏ + Δt)

        # Boundary conditions
        ūᵏ⁺¹[1] = W₀ * ũᵏ⁺¹
        ūᵏ⁺¹[end] = Wₙ * ũᵏ⁺¹

        # Next inner points
        ūᵏ⁺¹[2:end-1] .= (
            ūᵏ[2:end-1] .+
            Δt .* (
                D[2:end-1, :]ūᵏ .+ W[2:end-1, :]f.(x, tᵏ) .+
                (abs.(x[2:end-1] .- b) .< h₀) ./ 2h₀ .* (g_b(tᵏ) - uᵏ[2]) / Δx .-
                (abs.(x[2:end-1] .- a) .< h₀) ./ 2h₀ .* (uᵏ[end-1] - g_a(tᵏ)) / Δx
            )
        )

        # Advance by Δt
        tᵏ += Δt
        ūᵏ .= ūᵏ⁺¹
    end

    ūᵏ
end
