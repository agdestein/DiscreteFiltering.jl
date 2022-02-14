using LinearAlgebra
using Plots
using SparseArrays

# LSP indexing solution from
# https://github.com/julia-vscode/julia-vscode/issues/800#issuecomment-650085983
if isdefined(@__MODULE__, :LanguageServer)
    include("../src/DiscreteFiltering.jl")
    using .DiscreteFiltering
else
    using DiscreteFiltering
end

# Filter
f = IdentityFilter()

# Domain
a = 0.0
b = 2π
domain = PeriodicIntervalDomain(a, b)
# domain = ClosedIntervalDomain(a, b)

ν = 0.01
equation = BurgersEquation(domain, f, ν)

# Initial conditions
u₀(x) = sin(x) # + 3 / 5 * cos(5x) + 1 / 25 * sin(20(x - 1))

# Discretization
N = 500
M = N
x = discretize(domain, M)
ξ = discretize(domain, N)

T = 1.5

sol = solve(equation, u₀, (0.0, T), M, N)

u = sol.u[end]

##
p = plot(minorgrid = true);
# plot!(p, ξ, u₀, label = "Initial contitions")
# plot!(p, ξ, u, label = "Approximation")
for t ∈ LinRange(0, T, 5)
    plot!(p, ξ, sol(t), label = "t = $t");
end;
xlabel!(p, "ξ");
display(p);
savefig(p, "output/burgers/solution.png")
