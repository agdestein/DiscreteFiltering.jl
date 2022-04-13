# LSP indexing solution from
# https://github.com/julia-vscode/julia-vscode/issues/800#issuecomment-650085983
if isdefined(@__MODULE__, :LanguageServer)
    include("../src/DiscreteFiltering.jl")
    using .DiscreteFiltering
end

using DiscreteFiltering
using LinearAlgebra
using Plots

## Parameters

# Domain
a = 0.0
b = 2π
domain = PeriodicIntervalDomain(a, b)

# Exact solutions (advection equation)
u(x, t) = sin(x - t) + 3 / 5 * cos(5(x - t)) + 7 / 25 * sin(20(x - 1 - t))
# u̇(x, t) = -cos(x - t) + 3 * sin(5(x - t)) - 7 * 20 / 25 * cos(20(x - 1 - t))
u_int(x, t) = -cos(x - t) + 3 / 25 * sin(5(x - t)) - 7 / 25 / 20 * cos(20(x - 1 - t))

## Time
T = 1.0

# Discretization
N = 500
M = 500
x = discretize(domain, M)
ξ = discretize(domain, N)
Δx = (b - a) / M

# Filter
h(x) = Δx / 2
filter = TopHatFilter(h)

# Exact filtered operator
W = filter_matrix_meshwidth(filter, domain, N)
R = reconstruction_matrix_meshwidth(filter, domain, N)
C = advection_matrix(domain, N)
W = float(W)
R = float(R)
C = float(C)
# C̄_exact = W * C * R
# C̄_exact = Matrix(W * C) / Matrix(W)
C̄_exact = W * C * inv(Matrix(W))
C̄_exact[abs.(C̄_exact) .< 1e-14] .= 0

WW = inv(Matrix(W))
WW[abs.(WW) .< 1e-6] .= 0
WW

C̄_exact[50, :]

# Amplitudes
c = [
    [1],
    [1, 3],
    [1, 3 / 5, 1 / 25],
    rand(5),
]

# Frequencies
ω = [
    [0],
    [2, 3],
    [1, 5, 20],
    rand(1:10, 5),
]

# Phase-shifts
ϕ = [
    [π / 2],
    [0, 0],
    [0, 0, 20],
    2π * rand(5),
]

# Number of time steps for fitting C̄
K = [2:10; 100; 1000; 10000]
# K = map(x -> round(Int, 10 ^ x), LinRange(0.2, 4, 5))
err = zeros(length(K))
# for (i, K) ∈ enumerate(K)
#     t = LinRange(0, T, K)
#     s = sum_of_sines.([domain], c, ω, ϕ) 
#     u₀_list = [s.u for s ∈ s] 
#     U₀_list = [s.U for s ∈ s] 
#     C̄ = fit_Cbar(domain, filter, u₀_list, U₀_list, M, N, t; λ, method = :ridge)
#     err[i] = norm(C̄ - C̄_exact) / norm(C̄_exact)
# end
s = nothing
C̄ = nothing
for (i, K) ∈ enumerate(K)
    t = LinRange(0, T, K)
    s = sum_of_sines.([domain], c, ω, ϕ) 
    uₓ_list = [s.uₓ for s ∈ s]
    u_list = [s.u for s ∈ s]
    C̄ = fit_Cbar_approx(domain, filter, uₓ_list, u_list, M, t; λ = 1e-12, method = :ridge)
    err[i] = norm(C̄ - C̄_exact) / norm(C̄_exact)
end

C̄[50, :]

##
p = plot(
    xaxis = :log10,
    yaxis = :log10,
    minorgrid = true,
    # size = (400, 300),
    legend = :bottomright,
    # xlims = (10, 10^4),
    ylims = (1e-5, 1e-0),
    # xticks = 10 .^ (1:4),
    # xlabel = "M",
    xlabel = "K",
    # title = "N = 1000"
);
plot!(K, err; label = "N = $N");
# display(p)
# output = "output/advection/Cbar.tikz"
# output = "output/advection/Cbar.pdf"
output = "/tmp/toto.pdf"
savefig(p, output)

run(`zathura $output`; wait = false)

##
p = plot(LinRange(a, b, 500), [s.u for s ∈ s]);
output = "/tmp/tata.pdf"
savefig(p, output)


## Set PGFPlotsX plotting backend to export to tikz
using PGFPlotsX
pgfplotsx()

## In terminal plots
unicodeplots()

## Default backend
gr()
