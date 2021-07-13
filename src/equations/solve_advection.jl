function solve(
    equation::AdvectionEquation{PeriodicIntervalDomain,TopHatFilter},
    u,
    tlist,
    n;
    method = "filterfirst",
    solver = QNDF(),
    abstol = 1e-4,
    reltol = 1e-3,
)

    @unpack domain, filter = equation

    x = discretize(domain, n)
    Δx = x[2] - x[1]

    h = filter.width
    dh(x) = 0.0
    α(x) = 1 / 3 * dh(x) * h(x)

    ## Get matrices
    C = advection_matrix(domain, n)
    # W = filter_matrix(filter, domain, n)
    # R = inverse_filter_matrix(filter, domain, n)
    W = filter_matrix_meshwidth(filter, domain, n)
    R = inverse_filter_matrix_meshwidth(filter, domain, n)
    A = spdiagm(α.(x))

    uₕ = u.(x)
    ūₕ = W * uₕ

    du!(du, u, p, t) = mul!(du, p.J, u)
    if method == "filterfirst"
        p = (; J = -C + A * D)
        odefunction =
            ODEFunction(du!, jac = (J, u, p, t) -> (J .= p.J), jac_prototype = p.J)
        problem = ODEProblem(odefunction, ūₕ, tlist, p)
        sol = OrdinaryDiffEq.solve(problem, solver; abstol, reltol)
    elseif method == "discretizefirst"
        p = (; J = -W * C * R)
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
