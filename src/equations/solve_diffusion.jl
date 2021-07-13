
function solve(
    equation::DiffusionEquation{ClosedIntervalDomain,F},
    u,
    tlist,
    n;
    method = "filterfirst",
    boundary_conditions = "exact",
) where {F<:Filter}
    @unpack domain, filter, f, g_a, g_b = equation
    x = discretize(domain, n)
    Δx = x[2] - x[1]
    uₕ = u.(x)
    ūₕ = W * uₕ
    h = filter.width
    dh = h'
    α(x) = 1 / 3 * dh(x) * h(x)

    # Get matrices
    D = diffusion_matrix(domain, n)
    # W = filter_matrix(filter, domain, n)
    # R = inverse_filter_matrix(filter, domain, n)
    W = filter_matrix_meshwidth(filter, domain, n)
    R = inverse_filter_matrix_meshwidth(filter, domain, n)
    A = spdiagm(α.(x))

    W₀ = W[:, 1]
    Wₙ = W[:, end]

    function fₕ!(fₕ, t)
        fₕ[1] = 0
        fₕ[2:end-1] .= f.(x[2:end-1], t)
        fₕ[end] = 0
    end

    if boundary_conditions == "exact"
        M = [
            spzeros(1, n + 1)
            spzeros(n - 1, 1) sparse(I, n - 1, n - 1) spzeros(n - 1, 1)
            spzeros(1, n + 1)
        ]
        J = [
            -1 spzeros(1, n)
            D[2:end-1, :]
            spzeros(1, n) -1
        ]
        γ_a = g_a
        γ_b = g_b
    elseif boundary_conditions == "exact_derivative"
        M = sparse(I, n + 1, n + 1)
        # Zero out boundary u
        J = D
        J[[1, end], :] .= 0
        dropzeros!(D)
        γ_a = g_a'
        γ_b = g_b'
    elseif boundary_conditions == "ADBC"
        error("Not implemented")
    else
        error("Unknown boundary conditions")
    end

    if method == "filterfirst"
        error("Not implemented")
    elseif method == "discretizefirst"
        # Solve discretized-then-filtered problem
        p = (; Ju = zero(uₕ), J = W * J * R, f = zero(uₕ), Wf = zero(uₕ))
        function du!(du, u, p, t)
            # @show t
            @unpack Ju, J, f, Wf = p
            fₕ!(f, t)
            mul!(Wf, W, f)
            mul!(Ju, J, u)
            @. du = Ju + Wf + dg_a(t) * W₀ + dg_b(t) * Wₙ
        end
        odefunction = ODEFunction(
            du!,
            jac = (J, u, p, t) -> (J .= p.J),
            jac_prototype = p.J,
            mass_matrix = W * M * R,
        )
        problem = ODEProblem(odefunction, W * uₕ, tlist, p)
        solution = OrdinaryDiffEq.solve(problem, solver; abstol, reltol)
    else
        error("Unknown method")
    end

    solution
end


function solve(
    equation::DiffusionEquation{PeriodicIntervalDomain,F},
    u,
    tlist,
    n;
    method = "filterfirst",
) where {F<:Filter}
    @unpack domain, filter = equation
    x = discretize(domain, n)
    Δx = x[2] - x[1]
    uₕ = u.(x)
    ūₕ = W * uₕ
    h = filter.width
    dh(x) = 0.0
    α(x) = 1 / 3 * dh(x) * h(x)

    # Get matrices
    D = diffusion_matrix(domain, n)
    # W = filter_matrix(filter, domain, n)
    # R = inverse_filter_matrix(filter, domain, n)
    W = filter_matrix_meshwidth(filter, domain, n)
    R = inverse_filter_matrix_meshwidth(filter, domain, n)
    A = spdiagm(α.(x))

    if method == "filterfirst"
        error("Not implemented")
    elseif method == "discretizefirst"
        p = (; J = W * D * R)
        du!(du, u, p, t) = mul!(du, p.J, u)
        odefunction = ODEFunction(
            du!,
            jac = (J, u, p, t) -> (J .= p.J),
            jac_prototype = p.J,
            mass_matrix = W * R,
        )
        problem = ODEProblem(odefunction, W * uₕ, tlist, p)
        solution = OrdinaryDiffEq.solve(problem, solver; abstol, reltol)
    else
        error("Unknown method")
    end

    solution
end
