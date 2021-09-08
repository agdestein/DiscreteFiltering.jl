@testset "solve_advection.jl" begin
    domain = PeriodicIntervalDomain(0.0, 2π)

    T = 1.0

    # Exact solutions
    u(ξ, t) = sin(ξ - t) + 3 / 5 * cos(5(ξ - t)) + 1 / 25 * sin(20(ξ - 1 - t))
    u_int(ξ, t) = -cos(ξ - t) + 3 / 25 * sin(5(ξ - t)) - 1 / 25 / 20 * cos(20(ξ - 1 - t))

    params = (; subspacedim = 50, reltol = 1e-7, abstol = 1e-9)

    # Number of mesh points
    NN = floor.(Int, 10 .^ LinRange(2.5, 4.0, 4))

    err_ref = 0
    err_bar_ref = 0
    err_allbar_ref = 0
    n_ref = 0
    for (i, N) ∈ enumerate(NN)
        M = N

        # Discretization
        x = discretize(domain, M)
        ξ = discretize(domain, N)
        Δx = (domain.right - domain.left) / N

        # Filter
        h(x) = Δx / 2

        # Exact filtered solution
        ū(x, t) = 1 / 2h(x) * (u_int(x + h(x), t) - u_int(x - h(x), t))

        # Equations
        equation = AdvectionEquation(domain, IdentityFilter())
        equation_filtered = AdvectionEquation(domain, TopHatFilter(h))

        # Solve discretized problem
        sol = solve(
            equation,
            ξ -> u(ξ, 0.0),
            (0.0, T),
            M,
            N;
            method = "discretizefirst",
            params...,
        )

        # Solve filtered-then-discretized problem
        sol_bar = solve(
            equation_filtered,
            ξ -> u(ξ, 0.0),
            (0.0, T),
            M,
            N;
            method = "filterfirst",
            params...,
        )

        # Solve discretized-then-filtered problem
        sol_allbar = solve(
            equation_filtered,
            ξ -> u(ξ, 0.0),
            (0.0, T),
            M,
            N;
            method = "discretizefirst",
            params...,
        )

        ## Relative error
        u_exact = u.(ξ, T)
        ū_exact = ū.(x, T)
        err = norm(sol(T) - u_exact) / norm(u_exact)
        err_bar = norm(sol_bar(T) - ū_exact) / norm(ū_exact)
        err_allbar = norm(sol_allbar(T) - ū_exact) / norm(ū_exact)

        if i == 1
            err_ref = err
            err_bar_ref = err_bar
            err_allbar_ref = err_allbar
            n_ref = N
        end

        # Test for quadratic convergence
        @test err < 2 * err_ref * (n_ref / N)^2
        @test err_bar < 2 * err_bar_ref * (n_ref / N)^2
        @test err_allbar < 2 * err_allbar_ref * (n_ref / N)^2
    end
end
