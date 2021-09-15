@testset "solve_adbc.jl" begin
    a = 0.0
    b = 1.0
    domain = ClosedIntervalDomain(a, b)
    g_a(t) = 0.0
    g_b(t) = 0.0
    f(ξ, t) = (1.0 - ξ)ξ
    u₀(ξ) = 0.0
    T = 1.0
    N = 100
    Δx = (b - a) / N
    h(x) = 3.1Δx
    σ = Δx / 2
    filter = GaussianFilter(h, σ)
    equation = DiffusionEquation(domain, filter, f, g_a, g_b)
    ū_adbc = solve_adbc(equation, u₀, (0.0, T), N, T / 1000)
end
