@testset "solve_diffusion.jl" begin

    domain = ClosedIntervalDomain(0.0, 1.0)
    a, b = domain.left, domain.right

    T = 1.0

    # Exact solutions (from Borgaard 2006)
    u(x, t) = t + sin(2π * x) + sin(8π * x)
    u_int(x, t) = t * x - 1 / 2π * cos(2π * x) - 1 / 8π * cos(8π * x)
    f(x, t) = 1 + 4π^2 * sin(2π * x) + 64π^2 * sin(8π * x)
    g_a(t) = t
    g_b(t) = t

    params = (; abstol = 1e-7, reltol = 1e-5)

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

        # Exact filtered solution
        ū(x, t) = begin
            α = max(a, x - h(x))
            β = min(b, x + h(x))
            1 / (β - α) * (u_int(β, t) - u_int(α, t))
        end

        # Equations
        equation = DiffusionEquation(domain, IdentityFilter(), f, g_a, g_b)
        equation_filtered = DiffusionEquation(domain, TopHatFilter(h), f, g_a, g_b)

        # Solve discretized problem
        sol = solve(
            equation,
            x -> u(x, 0.0),
            (0.0, T),
            n;
            method = "discretizefirst",
            boundary_conditions = "exact",
            params...,
        )

        # Solve filtered-then-discretized problem
        @test_broken sol_bar = solve(
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
            boundary_conditions = "derivative",
            params...,
        )

        ## Relative error
        u_exact = u.(x, T)
        ū_exact = ū.(x, T)
        err = norm(sol(T) - u_exact) / norm(u_exact)
        # err_bar = norm(sol_bar(T) - ū_exact) / norm(ū_exact)
        err_allbar = norm(sol_allbar(T) - ū_exact) / norm(ū_exact)

        if i == 1
            err_ref = err
            # err_bar_ref = err_bar
            err_allbar_ref = err_allbar
            n_ref = n
        end

        # Test for quadratic convergence
        @test err < 1.6 * err_ref * (n_ref / n)^2
        # @test err_bar < 1.6 * err_bar_ref * (n_ref / n)^2
        @test err_allbar < 1.6 * err_allbar_ref * (n_ref / n)^2
    end

end
