"""
    solve(equation, u, tlist, n; method = "filterfirst")

Solve `equation` from `tlist[1]` to `tlist[2]` with initial conditions `u` and a
discretization of `n` points. If `method` is `"filterfirst"`, the equation is filtered then
discretized. If `method` is `"discretizefirst"`, the equation is discretized then filtered.
"""
function solve(
    equation::Equation,
    u,
    tlist,
    n;
    method = "filterfirst",
    solver = QNDF(),
    abstol = 1e-4,
    reltol = 1e-3,
)
    error("Not implemented")
end

include("solve_advection.jl")
include("solve_diffusion.jl")
include("solve_burgers.jl")
