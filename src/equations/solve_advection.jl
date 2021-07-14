function solve(
    equation::AdvectionEquation{PeriodicIntervalDomain{T},F},
    u,
    tlist,
    n;
    method = "filterfirst",
    subspacedim = 10,
) where {T,F}

    @unpack domain, filter = equation

    x = discretize(domain, n)
    Δx = x[2] - x[1]

    # Get matrices
    C = advection_matrix(domain, n)
    D = diffusion_matrix(domain, n)
    W = filter_matrix(filter, domain, n)
    R = inverse_filter_matrix(filter, domain, n)

    uₕ = u.(x)
    ūₕ = W * uₕ

    du!(du, u, p, t) = mul!(du, p.J, u)
    if method == "filterfirst"
        F === TopHatFilter ||
            error("Method \"filterfirst\" is only implemented for TopHatFilter")
        h = filter.width
        dh(x) = 0.0
        α(x) = 1 / 3 * dh(x) * h(x)
        A = spdiagm(α.(x))
        J = DiffEqArrayOperator(-C + A * D)
    elseif method == "discretizefirst"
        J = DiffEqArrayOperator(-W * C * R)
    else
        error("Unknown method")
    end

    problem = ODEProblem(J, W * uₕ, tlist)
    solution = OrdinaryDiffEq.solve(problem, LinearExponential(krylov = :simple, m = subspacedim))

    solution
end
