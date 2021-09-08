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
            Δt .* D[2:end-1, :]ūᵏ .+ Δt .* W[2:end-1, :]f.(x, tᵏ) .+
            Δt .* (abs.(x[2:end-1] .- b) .≤ h₀) ./ 2h₀ .* (g_b(tᵏ) - uᵏ[end-1]) / Δx .-
            Δt .* (abs.(x[2:end-1] .- a) .≤ h₀) ./ 2h₀ .* (uᵏ[2] - g_a(tᵏ)) / Δx

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
    Δt = (tlist[2] - tlist[1]) / 1000;
    solver = :euler,
    δ = √(6 / π) / equation.filter.kernel(0),
    ū_ext = nothing,
    uₓ = nothing,
) where {T}
    @unpack domain, filter, f, g_a, g_b = equation
    x = discretize(domain, n)
    Δx = x[2] - x[1]
    h₀ = filter.width(0)
    G = filter.kernel
    a, b = domain.left, domain.right

    # Get matrices
    D = diffusion_matrix(domain, n)

    # Filter stencil
    normalize(w) = w / sum(w)
    nw = floor(Int, h₀ / 2Δx)
    inds = -nw:nw
    w = normalize(G.(Δx .* inds))

    # Filter matrix
    diags = [i => fill(w, n + 1 - abs(i)) for (i, w) ∈ zip(inds, w)]
    W = spdiagm(diags...)

    # Move weights of outside points to endpoint
    for i = 1:nw
        W[i, 1] += sum(w[1:(nw+1-i)])
        W[end+1-i, end] += sum(w[(end-nw+i):end])
    end

    # Boundary stencils
    w₀ = W[1, :]
    wₙ = W[end, :]

    Ga = G.(x[2:end-1] .- a)
    Gb = G.(b .- x[2:end-1])

    f̄ᵏ_constant = apply_filter_extend(x -> f(x, tlist[1]), filter, domain).(x[2:end-1])

    function du!(du, ūᵏ, tᵏ, Δt, p)
        @unpack uᵏ, uᵏ⁺¹, Dūᵏ, fᵏ, f̄ᵏ, δ = p

        # Diffusion term
        mul!(Dūᵏ, D, ūᵏ)

        # Source term
        # fᵏ .= f.(x, tᵏ)
        # mul!(f̄ᵏ, W[2:end-1, :], fᵏ)
        # f̄ᵏ .= apply_filter_extend(x -> f(x, tᵏ), filter, domain).(x[2:end-1])
        f̄ᵏ = f̄ᵏ_constant

        # Current approximate unfiltered solution
        @. uᵏ = ūᵏ - δ^2 / 24 * Dūᵏ

        # Next approximate unfiltered solution
        uᵏ⁺¹ .= uᵏ
        uᵏ⁺¹[1] = g_a(tᵏ + Δt)
        uᵏ⁺¹[end] = g_b(tᵏ + Δt)

        # Filtered boundary conditions
        if isnothing(ū_ext)
            # Approximate deconvolution boundary conditions
            du[1] = (w₀'uᵏ⁺¹ - ūᵏ[1]) / Δt
            du[end] = (wₙ'uᵏ⁺¹ - ūᵏ[end]) / Δt
        else
            # Exact filtered boundary conditions
            du[1] = (ū_ext(tᵏ + Δt)(a) - ūᵏ[1]) / Δt
            du[end] = (ū_ext(tᵏ + Δt)(b) - ūᵏ[end]) / Δt
        end

        if isnothing(uₓ)
            # Approximate boundary terms
            uₓ_a = (uᵏ[2] - g_a(tᵏ)) / Δx
            uₓ_b = (g_b(tᵏ) - uᵏ[end-1]) / Δx
        else
            # Exact boundary terms
            uₓ_a = uₓ(a, tᵏ)
            uₓ_b = uₓ(b, tᵏ)
        end

        # Next inner points for filtered solution
        du[2:end-1] .= Dūᵏ[2:end-1] .+ f̄ᵏ .+ Gb .* uₓ_b .- Ga .* uₓ_a
    end

    function perform_step_euler!(ūᵏ, tᵏ, Δt, p)
        @unpack du = p
        du!(du, ūᵏ, tᵏ, Δt, p)
        ūᵏ .+= Δt .* du
    end

    function perform_step_RK4!(ūᵏ, tᵏ, Δt, p)
        @unpack k₁, k₂, k₃, k₄ = p

        # Compute change
        du!(k₁, ūᵏ, tᵏ, Δt / 2, p)
        du!(k₂, ūᵏ + Δt / 2 * k₁, tᵏ + Δt / 2, Δt / 2, p)
        du!(k₃, ūᵏ + Δt / 2 * k₂, tᵏ + Δt / 2, Δt / 2, p)
        du!(k₄, ūᵏ + Δt * k₃, tᵏ + Δt, Δt / 2, p)

        # Advance by Δt
        @. ūᵏ += Δt * (k₁ / 6 + k₂ / 3 + k₃ / 3 + k₄ / 6)
    end

    if solver == :euler
        perform_step! = perform_step_euler!
    elseif solver == :RK4
        perform_step! = perform_step_RK4!
    else
        error("Solver is not supported")
    end

    # ūᵏ = W * u.(x)
    ūᵏ = apply_filter_extend(u, filter, domain).(x)
    p = (;
        du = copy(ūᵏ),
        k₁ = copy(ūᵏ),
        k₂ = copy(ūᵏ),
        k₃ = copy(ūᵏ),
        k₄ = copy(ūᵏ),
        uᵏ = copy(ūᵏ),
        uᵏ⁺¹ = copy(ūᵏ),
        Dūᵏ = copy(ūᵏ),
        fᵏ = copy(ūᵏ),
        f̄ᵏ = copy(ūᵏ[2:end-1]),
        δ,
    )
    tᵏ = tlist[1]
    while tᵏ + Δt < tlist[2]
        # @show tᵏ
        perform_step!(ūᵏ, tᵏ, Δt, p)
        tᵏ += Δt
    end

    # Perform last time step (with adapted step size)
    Δt_last = tlist[2] - tᵏ
    perform_step!(ūᵏ, tᵏ, Δt_last, p)
    tᵏ += Δt_last

    # nT = round(Int, (tlist[2] - tlist[1]) / Δt)
    # uu = zeros(n + 1, nT + 1)
    # uu[:, 1] = ūᵏ
    # for i = 1:nT
    #     perform_step!(ūᵏ, tᵏ, Δt, p)
    #     uu[:, i+1] = ūᵏ
    #     tᵏ += Δt
    # end

    ūᵏ
    # uu
end
