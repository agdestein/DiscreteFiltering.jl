using LinearAlgebra
using SparseArrays
using GLMakie
using OrdinaryDiffEq

# ν = 0.0005
# ν = 5e-5
ν = 0.0

N = 1000
x = LinRange(0, 1, N + 1)
Δx = 1 / N

# C = 1 / 2Δx * spdiagm(N + 1, N + 1, -1 => fill(-1.0, N), 1 => fill(1.0, N))
C = 1 / Δx * spdiagm(N + 1, N + 1, -1 => fill(-1.0, N), 0 => fill(1.0, N + 1))
C = C[2:N, :]
plotmat(C)

D =
    1 / Δx^2 *
    spdiagm(N + 1, N + 1, -1 => fill(1.0, N), 0 => fill(-2.0, N + 1), 1 => fill(1.0, N))
D = D[2:N, :]
plotmat(D)

ga(t) = 0.30 * (1 - 0.6 * cos(0.7 * π * t)) / 2
gb(t) = 0.0
# u₀(x) = sin(2π * x)
u₀(x) = 0.0

function f!(du, u, p, t)
    u[1] = ga(t)
    u[N+1] = u[N]
    du[1] = 0.0
    du[2:N] = -1 / 2 * C * (u .^ 2) + ν * D * u
    du[N+1] = 0.0
end

T = 20.0
prob = ODEProblem(ODEFunction(f!), u₀.(x), (0.0, T), nothing)
sol = solve(prob, Tsit5())

fig = Figure()
ax = Axis(fig[1, 1])
ylims!(0, 0.4)
for t ∈ LinRange(0.0, T, 5)
    lines!(x, sol(t); label = "t = $t")
end
axislegend(ax)

nframe = 200
fig = Figure()
ax = Axis(fig[1, 1])
ylims!(0, 0.4)
u = Observable(u₀.(x))
lines!(x, u)
record(fig, "burgers.mp4", 1:nframe) do frame
    t = frame / nframe * T
    u[] = sol(t)
end
