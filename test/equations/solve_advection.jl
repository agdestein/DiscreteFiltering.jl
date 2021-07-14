@testset "solve_advection.jl" begin

    domain = PeriodicIntervalDomain(0.0, 2π)

    T = 1.0

    # Exact solutions
    u(x, t) = sin(x - t) + 0.6cos(5(x - t)) + 0.04sin(20(x - 1 - t))
    u_int(x, t) = -cos(x - t) + 0.6 / 5 * sin(5(x - t)) - 0.04 / 20 * cos(20(x - 1 - t))

    params = (; subspacedim = 20)

    # Number of mesh points
    N = floor.(Int, 10 .^ LinRange(2.5, 4.0, 4))

    err_ref = 0
    err_bar_ref = 0
    err_allbar_ref = 0
    n_ref = 0
    for (i, n) ∈ enumerate(N)

        # Discretization
        x = discretize(domain, n)
        Δx = (domain.right - domain.left) / n

        # Filter
        h(x) = Δx / 2
        f = TopHatFilter(h)

        # Exact filtered solution
        ū(x, t) = 1 / 2h(x) * (u_int(x + h(x), t) - u_int(x - h(x), t))

        # Equations
        equation = AdvectionEquation(domain, IdentityFilter())
        equation_filtered = AdvectionEquation(domain, TopHatFilter(h))

        # Solve discretized problem
        sol = solve(
            equation,
            x -> u(x, 0.0),
            (0.0, T),
            n;
            method = "discretizefirst",
            params...,
        )

        # Solve filtered-then-discretized problem
        sol_bar = solve(
            equation_filtered,
            x -> u(x, 0.0),
            (0.0, T),
            n;
            method = "filterfirst",
            params...,
        )

        # Solve discretized-then-filtered problem
        sol_allbar = solve(
            equation_filtered,
            x -> u(x, 0.0),
            (0.0, T),
            n;
            method = "discretizefirst",
            params...,
        )

        ## Relative error
        u_exact = u.(x, T)
        ū_exact = ū.(x, T)
        err = norm(sol(T) - u_exact) / norm(u_exact)
        err_bar = norm(sol_bar(T) - ū_exact) / norm(ū_exact)
        err_allbar = norm(sol_allbar(T) - ū_exact) / norm(ū_exact)

        if i == 1
            err_ref = err
            err_bar_ref = err_bar
            err_allbar_ref = err_allbar
            n_ref = n
        end

        # Test for quadratic convergence
        @test err < 2 * err_ref * (n_ref / n)^2
        @test err_bar < 2 * err_bar_ref * (n_ref / n)^2
        @test err_allbar < 2 * err_allbar_ref * (n_ref / n)^2
    end
end
