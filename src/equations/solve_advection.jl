"""
    solve(
        equation::AdvectionEquation{PeriodicIntervalDomain{T},F},
        u,
        tlist,
        n;
        method = "filterfirst",
        subspacedim = 10,
    ) where {T,F}

Solve `equation` from `tlist[1]` to `tlist[2]` with initial conditions `u` and a
discretization of `n` points. If `method` is `"filterfirst"`, the equation is filtered then
discretized. If `method` is `"discretizefirst"`, the equation is discretized then filtered.
The parameter `subspacedim` controls the accuracy of the linear exponential timestepping.
"""
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
    R = reconstruction_matrix(filter, domain, n)

    ū = apply_filter(u, filter, domain)

    uₕ = u.(x)
    ūₕ = ū.(x)
    # ūₕ = W * uₕ

    if method == "filterfirst"
        F === TopHatFilter ||
            error("Method \"filterfirst\" is only implemented for TopHatFilter")
        function fix_derivative(y)
            isnothing(y) ? 0.0 : y
        end
        h = filter.width
        dh(x) = ForwardDiff.derivative(h, x)
        α(x) = 1 / 3 * dh(x) * h(x)
        A = spdiagm(α.(x))
        J = DiffEqArrayOperator(-C + A * D)


        problem = ODEProblem(J, ūₕ, tlist)
        solution = OrdinaryDiffEq.solve(
            problem,
            LinearExponential(krylov = :simple, m = subspacedim),
        )
    elseif method == "discretizefirst"
        p = (; J = -W * C * R)
        Mdu!(du, u, p, t) = mul!(du, p.J, u)
        odefunction = ODEFunction(
            Mdu!,
            jac = (J, u, p, t) -> (J .= p.J),
            jac_prototype = p.J,
            mass_matrix = W * R,
        )
        problem = ODEProblem(odefunction, ūₕ, tlist, p)
        solution = OrdinaryDiffEq.solve(problem, QNDF(); abstol = 1e-10, reltol = 1e-8)
    else
        error("Unknown method")
    end
    solution
end
