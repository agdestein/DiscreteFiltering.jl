function figsave(
    p,
    name;
    savedir = "figures/",
    suffices = ("pdf", "tikz"),
    kwargs...,
)
    for suffix ∈ suffices
        path = joinpath(savedir, "$name.$suffix")
        @info "Saving figure to $path"
        savefig(plot(p; kwargs...), path)
    end
end
