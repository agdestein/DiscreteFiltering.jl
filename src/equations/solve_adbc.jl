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

    function perform_step!(ūᵏ, tᵏ, Δt, p)
        uᵏ, uᵏ⁺¹, ūᵏ⁺¹ = p

        # Current approximate unfiltered solution
        mul!(uᵏ, R, ūᵏ)

        # Next approximate unfiltered solution
        uᵏ⁺¹ .= uᵏ
        uᵏ⁺¹[1] = g_a(tᵏ + Δt)
        uᵏ⁺¹[end] = g_b(tᵏ + Δt)

        # Filtered boundary conditions
        ūᵏ⁺¹[1] = w₀'uᵏ⁺¹
        ūᵏ⁺¹[end] = wₙ'uᵏ⁺¹

        # Next inner points for filtered solution
        ūᵏ⁺¹[2:end-1] .+=
            Δt .* D[2:end-1, :]ūᵏ .+ W[2:end-1, :]f.(x, tᵏ) .+
            Δt .* (abs.(x[2:end-1] .- b) .≤ h₀) ./ 2h₀ .* (g_b(tᵏ) - uᵏ[2]) / Δx .-
            Δt .* (abs.(x[2:end-1] .- a) .≤ h₀) ./ 2h₀ .* (uᵏ[end-1] - g_a(tᵏ)) / Δx

        # Advance by Δt
        ūᵏ .= ūᵏ⁺¹
    end

    ūᵏ = W * u.(x)
    p = (copy(ūᵏ), copy(ūᵏ), copy(ūᵏ))
    tᵏ = tlist[1]
    while tᵏ + Δt < tlist[2]
        perform_step!(ūᵏ, tᵏ, Δt, p)
        tᵏ += Δt
    end

    # Perform last time step (with adapted step size)
    Δt_last = tlist[2] - tᵏ
    perform_step!(ūᵏ, tᵏ, Δt_last, p)
    tᵏ += Δt_last

    ūᵏ
end


function solve_adbc(
    equation::DiffusionEquation{ClosedIntervalDomain{T},ConvolutionalFilter},
    u,
    tlist,
    n,
    Δt = (tlist[2] - tlist[1]) / 1000,
) where {T}

    @unpack domain, filter, f, g_a, g_b = equation
    x = discretize(domain, n)
    Δx = x[2] - x[1]
    h = filter.width
    G = filter.kernel
    a, b = domain.left, domain.right

    # Get matrices
    D = diffusion_matrix(domain, n)

    # Filter matrix
    W = filter_matrix(filter, domain, n)
    R = reconstruction_matrix(filter, domain, n)
    w₀ = W[1, :]
    wₙ = W[end, :]

    function perform_step!(ūᵏ, tᵏ, Δt, p)
        uᵏ, uᵏ⁺¹, ūᵏ⁺¹ = p

        # Current approximate unfiltered solution
        mul!(uᵏ, R, ūᵏ)

        # Next approximate unfiltered solution
        uᵏ⁺¹ .= uᵏ
        uᵏ⁺¹[1] = g_a(tᵏ + Δt)
        uᵏ⁺¹[end] = g_b(tᵏ + Δt)

        # Filtered boundary conditions
        ūᵏ⁺¹[1] = w₀'uᵏ⁺¹
        ūᵏ⁺¹[end] = wₙ'uᵏ⁺¹

        # Next inner points for filtered solution
        ūᵏ⁺¹[2:end-1] .+=
            Δt .* D[2:end-1, :]ūᵏ .+ W[2:end-1, :]f.(x, tᵏ) .+
            Δt .* G.(b .- x[2:end-1]) .* (g_b(tᵏ) - uᵏ[2]) / Δx .-
            Δt .* G.(x[2:end-1] .- a) .* (uᵏ[end-1] - g_a(tᵏ)) / Δx

        # Advance by Δt
        ūᵏ .= ūᵏ⁺¹
    end

    ūᵏ = W * u.(x)
    p = (copy(ūᵏ), copy(ūᵏ), copy(ūᵏ))
    tᵏ = tlist[1]
    while tᵏ + Δt < tlist[2]
        perform_step!(ūᵏ, tᵏ, Δt, p)
        tᵏ += Δt
    end

    # Perform last time step (with adapted step size)
    Δt_last = tlist[2] - tᵏ
    perform_step!(ūᵏ, tᵏ, Δt_last, p)
    tᵏ += Δt_last

    ūᵏ
end