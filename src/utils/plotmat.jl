function mplotmat(A; kwargs...)
    fig = Makie.Figure()
    ax, hm = Makie.heatmap(
        fig[1, 1],
        reverse(A', dims = 2);
        axis = (;
            # aspect = DataAspect(),
            # xlims = (1, size(A, 2)),
            # ylims = (1, size(A, 1)),
            kwargs...,
        ),
    )
    cb = Makie.Colorbar(fig[1, 2], hm)
    # Makie.colsize!(fig.layout, 1, Aspect(1, 1.0))
    fig
end
mplotmat(A::AbstractSparseMatrix; kwargs...) = mplotmat(Matrix(A); kwargs...)

pplotmat(A; kwargs...) = Plots.heatmap(
    reverse(A; dims = 1);
    # aspect_ratio = :equal,
    xlims = (1 / 2, size(A, 2) + 1 / 2),
    ylims = (1 / 2, size(A, 1) + 1 / 2),
    xticks = nothing,
    yticks = nothing,
    kwargs...,
)
pplotmat(A::AbstractSparseMatrix; kwargs...) = pplotmat(Matrix(A); kwargs...)
