"""
    solve(
        equation::DiffusionEquation{ClosedIntervalDomain{T},F},
        u,
        tlist,
        M,
        N;
        method = "filterfirst",
        boundary_conditions = "exact",
        solver = QNDF(),
        abstol = 1e-4,
        reltol = 1e-3,
    ) where {T,F}

Solve `equation` from `tlist[1]` to `tlist[2]` with initial conditions `u` and a
discretization of `M` and `N` points. If `method` is `"filterfirst"`, the equation is filtered then
discretized. If `method` is `"discretizefirst"`, the equation is discretized then filtered.
"""
function solve(
    equation::DiffusionEquation{ClosedIntervalDomain{T},F},
    u,
    tlist,
    M,
    N;
    method = "filterfirst",
    boundary_conditions = "exact",
    solver = QNDF(),
    abstol = 1e-4,
    reltol = 1e-3,
) where {T,F}

    @unpack domain, filter, f, g_a, g_b = equation
    x = discretize(domain, M)
    ξ = discretize(domain, N)

    # Get matrices
    D = diffusion_matrix(domain, N)
    W = filter_matrix(filter, domain, M, N)
    R = reconstruction_matrix(filter, domain, M, N)

    W₀ = W[:, 1]
    Wₙ = W[:, end]

    # Initial conditions
    ū = apply_filter(u, filter, domain)
    uₕ = u.(ξ)
    # ūₕ = ū.(x)
    ūₕ = filter_matrix(filter, domain, M, N) * uₕ

    function f!(fₕ, t)
        fₕ[1] = 0
        fₕ[2:end-1] .= f.(ξ[2:end-1], t)
        fₕ[end] = 0
    end

    if boundary_conditions == "exact"
        Mass = [
            spzeros(1, N + 1)
            spzeros(N - 1, 1) sparse(I, N - 1, N - 1) spzeros(N - 1, 1)
            spzeros(1, N + 1)
        ]
        J = [
            -1 spzeros(1, N)
            D[2:end-1, :]
            spzeros(1, N) -1
        ]
        γ_a = g_a
        γ_b = g_b
    elseif boundary_conditions == "derivative"
        Mass = sparse(I, N + 1, N + 1)
        # Zero out boundary u
        J = D
        J[[1, end], :] .= 0
        dropzeros!(J)
        γ_a = t -> ForwardDiff.derivative(g_a, t)
        γ_b = t -> ForwardDiff.derivative(g_b, t)
    else
        error("Unknown boundary conditions")
    end

    if method == "filterfirst"
        error("Not implemented")
    elseif method == "discretizefirst"
        # Solve discretized-then-filtered problem
        p = (; Ju = zero(x), J = W * J * R, fₕ = zero(ξ), Wf = zero(x))
        function Mdu!(du, u, p, t)
            # @show t
            @unpack Ju, J, fₕ, Wf = p
            f!(fₕ, t)
            mul!(Wf, W, fₕ)
            mul!(Ju, J, u)
            @. du = Ju + Wf + γ_a(t) * W₀ + γ_b(t) * Wₙ
        end
        odefunction = ODEFunction(
            Mdu!,
            jac = (J, u, p, t) -> (J .= p.J),
            jac_prototype = p.J,
            mass_matrix = W * Mass * R,
        )
        problem = ODEProblem(odefunction, ūₕ, tlist, p)
        solution = OrdinaryDiffEq.solve(problem, solver; abstol, reltol)
    else
        error("Unknown method")
    end

    solution
end

"""
    solve(
        equation::DiffusionEquation{PeriodicIntervalDomain{T},F},
        u,
        tlist,
        M,
        N;
        method = "filterfirst",
        solver = QNDF(),
        abstol = 1e-4,
        reltol = 1e-3,
    ) where {T,F}

Solve `equation` from `tlist[1]` to `tlist[2]` with initial conditions `u` and a
discretization of `M` and `N` points. If `method` is `"filterfirst"`, the equation is filtered then
discretized. If `method` is `"discretizefirst"`, the equation is discretized then filtered.
"""
function solve(
    equation::DiffusionEquation{PeriodicIntervalDomain{T},F},
    u,
    tlist,
    M,
    N;
    method = "filterfirst",
    solver = QNDF(),
    abstol = 1e-4,
    reltol = 1e-3,
) where {T,F}

    @unpack domain, filter = equation

    # Domain
    x = discretize(domain, M)
    ξ = discretize(domain, N)

    # Get matrices
    D = diffusion_matrix(domain, N)
    W = filter_matrix(filter, domain, M, N)
    R = reconstruction_matrix(filter, domain, M, N)
    A = spdiagm(α.(ξ))

    # Initial conditions
    uₕ = u.(ξ)
    ūₕ = W * uₕ

    if method == "filterfirst"
        error("Not implemented")
        # Filter
        h = filter.width
        dh(ξ) = 0.0
        α(ξ) = 1 / 3 * dh(ξ) * h(ξ)
    elseif method == "discretizefirst"
        p = (; J = W * D * R)
        du!(du, u, p, t) = mul!(du, p.J, u)
        odefunction = ODEFunction(
            du!,
            jac = (J, u, p, t) -> (J .= p.J),
            jac_prototype = p.J,
            mass_matrix = W * R,
        )
        problem = ODEProblem(odefunction, ūₕ, tlist, p)
        solution = OrdinaryDiffEq.solve(problem, solver; abstol, reltol)
    else
        error("Unknown method")
    end

    solution
end
