"""
    sum_of_sines(domain, c, ω, ϕ)

Create initial condition function `u₀` and an antiderivative `U₀`.
The function is a sum of sines of amplitudes `c`, frequencies `ω` and phase-shifts `ϕ`.
"""
function sum_of_sines(domain, c, ω, ϕ)
    φ(ω, ϕ, x) = sin(ω * x - ϕ) 
    φₓ(ω, ϕ, x) = ω * cos(ω * x - ϕ) 
    Φ(ω, ϕ, x) = ω ≈ 0 ? -sin(ϕ) * x : -cos(ω * x - ϕ) / ω 
    u(x) = sum(c .* φ.(ω, ϕ, x))
    uₓ(x) = sum(c .* φₓ.(ω, ϕ, x))
    U(x) = sum(c .* Φ.(ω, ϕ, x))
    (; u, uₓ, U)
end
