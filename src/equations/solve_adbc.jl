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
    Δt = (tlist[2] - tlist[1]) / 1000,
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

    # Filter matrix
    inds = [-1, 0, 1]
    stencil = [1 / 24, 11 / 12, 1 / 24]
    diags = [i => fill(s, n + 1 - abs(i)) for (i, s) ∈ zip(inds, stencil)]
    W = spdiagm(diags...)
    W[1, [1, 2]] = [23 / 24, 1 / 24]
    W[end, [end, end - 1]] = [23 / 24, 1 / 24]
    w₀ = W[1, :]
    wₙ = W[end, :]

    # Reconstruction matrix
    inds = [-1, 0, 1]
    stencil = [-1 / 24, 13 / 12, -1 / 24]
    diags = [i => fill(s, n + 1 - abs(i)) for (i, s) ∈ zip(inds, stencil)]
    R = spdiagm(diags...)
    R[1, [1, 2]] = [25 / 24, -1 / 24]
    R[end, [end, end - 1]] = [25 / 24, -1 / 24]

    ūᵏ = W * u.(x)
    ūᵏ⁺¹ = copy(ūᵏ)
    uᵏ = copy(ūᵏ)
    ũᵏ⁺¹ = copy(ūᵏ)
    tᵏ = tlist[1]
    while tᵏ + Δt < tlist[2]
        # Current approximate unfiltered solution
        mul!(uᵏ, R, ūᵏ)

        # Next approximate unfiltered solution
        ũᵏ⁺¹ .= uᵏ
        ũᵏ⁺¹[1] = g_a(tᵏ + Δt)
        ũᵏ⁺¹[end] = g_b(tᵏ + Δt)

        # Filtered boundary conditions
        ūᵏ⁺¹[1] = w₀'ũᵏ⁺¹
        ūᵏ⁺¹[end] = wₙ'ũᵏ⁺¹

        # Next inner points for filtered solution
        ūᵏ⁺¹[2:end-1] .+= (
            Δt .* (
                D[2:end-1, :]ūᵏ .+ W[2:end-1, :]f.(x, tᵏ) .+
                (abs.(x[2:end-1] .- b) .≤ h₀) ./ 2h₀ .* (g_b(tᵏ) - uᵏ[2]) / Δx .-
                (abs.(x[2:end-1] .- a) .≤ h₀) ./ 2h₀ .* (uᵏ[end-1] - g_a(tᵏ)) / Δx
            )
        )

        # Advance by Δt
        tᵏ += Δt
        ūᵏ .= ūᵏ⁺¹
    end

    # Perform last time step (with adapted step)
    Δt_last = tlist[2] - tᵏ

    # Current approximate unfiltered solution
    mul!(uᵏ, R, ūᵏ)

    # Next approximate unfiltered solution
    ũᵏ⁺¹ .= uᵏ
    ũᵏ⁺¹[1] = g_a(tᵏ + Δt_last)
    ũᵏ⁺¹[end] = g_b(tᵏ + Δt_last)

    # Filtered boundary conditions
    ūᵏ⁺¹[1] = w₀'ũᵏ⁺¹
    ūᵏ⁺¹[end] = wₙ'ũᵏ⁺¹

    # Next inner points for filtered solution
    ūᵏ⁺¹[2:end-1] .+= (
        Δt_last .* (
            D[2:end-1, :]ūᵏ .+ W[2:end-1, :]f.(x, tᵏ) .+
            (abs.(x[2:end-1] .- b) .≤ h₀) ./ 2h₀ .* (g_b(tᵏ) - uᵏ[2]) / Δx .-
            (abs.(x[2:end-1] .- a) .≤ h₀) ./ 2h₀ .* (uᵏ[end-1] - g_a(tᵏ)) / Δx
        )
    )

    # Advance by Δt_last
    tᵏ += Δt_last
    ūᵏ .= ūᵏ⁺¹

    ūᵏ
end
