@testset "solve_diffusion.jl" begin
    domain = ClosedIntervalDomain(0.0, 1.0)
    a, b = domain.left, domain.right

    T = 1.0

    # Exact solutions (from Borgaard 2006)
    u(ξ, t) = t + sin(2π * ξ) + sin(8π * ξ)
    u_int(ξ, t) = t * ξ - 1 / 2π * cos(2π * ξ) - 1 / 8π * cos(8π * ξ)
    f(ξ, t) = 1 + 4π^2 * sin(2π * ξ) + 64π^2 * sin(8π * ξ)
    g_a(t) = t
    g_b(t) = t

    tols = (; abstol = 1e-6, reltol = 1e-4)

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
        function ū(x, t)
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
            ξ -> u(ξ, 0.0),
            (0.0, T),
            M,
            N;
            method = "discretizefirst",
            boundary_conditions = "derivative",
            tols...,
        )

        # Solve filtered-then-discretized problem
        @test_broken sol_bar = solve(
            equation_filtered,
            ξ -> u(ξ, 0.0),
            (0.0, T),
            M,
            N;
            method = "filterfirst",
            boundary_conditions = "derivative",
            tols...,
        )

        # Solve discretized-then-filtered problem
        sol_allbar = solve(
            equation_filtered,
            ξ -> u(ξ, 0.0),
            (0.0, T),
            M,
            N;
            method = "discretizefirst",
            boundary_conditions = "derivative",
            tols...,
        )

        ## Relative error
        u_exact = u.(ξ, T)
        ū_exact = ū.(x, T)
        err = norm(sol(T) - u_exact) / norm(u_exact)
        # err_bar = norm(sol_bar(T) - ū_exact) / norm(ū_exact)
        err_allbar = norm(sol_allbar(T) - ū_exact) / norm(ū_exact)

        if i == 1
            err_ref = err
            # err_bar_ref = err_bar
            err_allbar_ref = err_allbar
            n_ref = N
        end

        # Test for quadratic convergence
        @test err < 1.6 * err_ref * (n_ref / N)^2
        # @test err_bar < 1.6 * err_bar_ref * (n_ref / N)^2
        @test err_allbar < 1.6 * err_allbar_ref * (n_ref / N)^2
    end


    # Test exact boundary conditions
    N = 100
    M = N
    equation = DiffusionEquation(domain, IdentityFilter(), f, g_a, g_b)

    # Solve discretized problem with DAE formulation
    sol_exact = solve(
        equation,
        ξ -> u(ξ, 0.0),
        (0.0, T),
        M,
        N;
        method = "discretizefirst",
        boundary_conditions = "exact",
        tols...,
    )

    # Solve discretized problem with derivative BC
    sol_derivative = solve(
        equation,
        ξ -> u(ξ, 0.0),
        (0.0, T),
        M,
        N;
        method = "discretizefirst",
        boundary_conditions = "derivative",
        tols...,
    )

    @test sol_exact(T) ≈ sol_derivative(T)
end
