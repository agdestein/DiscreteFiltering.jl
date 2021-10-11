"""
    solve(
        equation::AdvectionEquation{PeriodicIntervalDomain{T},F},
        u,
        tlist,
        M,
        N;
        method = "filterfirst",
        subspacedim = 10,
        solver = QNDF(),
        reltol = 1e-4,
        abstol = 1e-6,
    ) where {T,F}

Solve `equation` from `tlist[1]` to `tlist[2]` with initial conditions `u` and a
discretization of `M` and `N` points. If `method` is `"filterfirst"`, the equation is filtered then
discretized. If `method` is `"discretizefirst"`, the equation is discretized then filtered.
The parameter `subspacedim` controls the accuracy of the linear exponential timestepping.
"""
function solve(
    equation::AdvectionEquation{PeriodicIntervalDomain{T},F},
    u,
    tlist,
    M,
    N;
    method = "filterfirst",
    subspacedim = 10,
    solver = QNDF(),
    reltol = 1e-4,
    abstol = 1e-6,
    degmax = 10,
    λ = 1e-6,
) where {T,F}
    @unpack domain, filter = equation

    x = discretize(domain, M)
    ξ = discretize(domain, N)

    ū = apply_filter(u, filter, domain)

    if method == "filterfirst"
        # Get matrices
        C = advection_matrix(domain, N)
        D = diffusion_matrix(domain, N)
        F === TopHatFilter ||
            error("Method \"filterfirst\" is only implemented for TopHatFilter")
        # M == N || @warn "Method \"filterfirst\" will only use M"
        h = filter.width
        dh(x) = derivative(h, x)
        α(x) = 1 / 3 * dh(x) * h(x)
        A = spdiagm(α.(ξ))
        J = DiffEqArrayOperator(-C + A * D)
        ūₕ = ū.(ξ)
        problem = ODEProblem(J, ūₕ, tlist)
        solution = OrdinaryDiffEq.solve(
            problem,
            LinearExponential(krylov = :simple, m = subspacedim),
        )
    elseif method == "discretizefirst"
        # Get matrices
        C = advection_matrix(domain, N)
        D = diffusion_matrix(domain, N)
        W = filter_matrix(filter, domain, M, N; degmax, λ)
        R = reconstruction_matrix(filter, domain, M, N; degmax, λ)
        ūₕ = ū.(x)
        # ūₕ = W * u.(ξ)
        p = (; J = -W * C * R)
        Mdu!(du, u, p, t) = mul!(du, p.J, u)
        odefunction = ODEFunction(
            Mdu!,
            jac = (J, u, p, t) -> (J .= p.J),
            jac_prototype = p.J,
            # mass_matrix = W * R,
        )
        problem = ODEProblem(odefunction, ūₕ, tlist, p)
        solution = OrdinaryDiffEq.solve(problem, solver; reltol, abstol)
    elseif method == "discretizefirst-without-R"
        # Get matrices
        C = advection_matrix(domain, N)
        D = diffusion_matrix(domain, N)
        W = filter_matrix(filter, domain, M, N; degmax, λ)
        # A = factorize(W'W + λ * I)
        A = lu(W'W + λ * I)
        J = -W * C
        ūₕ = ū.(x)
        # ūₕ = W * u.(ξ)
        p = (; u☆ = zeros(N), utmp = zeros(N), J, A)
        function Mdu!_without_R(dū, ū, p, t)
            @unpack u☆, utmp, A = p
            mul!(utmp, W', ū)
            ldiv!(u☆, A, utmp)
            mul!(dū, p.J, u☆)
        end
        odefunction = ODEFunction(
            Mdu!_without_R,
            # jac = (J, u, p, t) -> (J .= p.J),
            # jac_prototype = p.J,
            # mass_matrix = W * R,
        )
        problem = ODEProblem(odefunction, ūₕ, tlist, p)
        solution = OrdinaryDiffEq.solve(problem, solver; reltol, abstol)
    else
        error("Unknown method")
    end

    solution
end
