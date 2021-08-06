"""
    solve

Solve equation.
"""
function solve end

include("solve_advection.jl")
include("solve_diffusion.jl")
include("solve_burgers.jl")
