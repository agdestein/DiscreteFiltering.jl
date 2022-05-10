"""
    plotmat(A; kwargs...)

Plot matrix.
"""
function plotmat end

plotmat(A; kwargs...) = heatmap(
    reverse(A; dims = 1);
    # aspect_ratio = :equal,
    xlims = (1 / 2, size(A, 2) + 1 / 2),
    ylims = (1 / 2, size(A, 1) + 1 / 2),
    # xticks = nothing,
    # yticks = nothing,
    kwargs...,
)

plotmat(A::AbstractSparseMatrix; kwargs...) = plotmat(Matrix(A); kwargs...)
