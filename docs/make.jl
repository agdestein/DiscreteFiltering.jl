using DiscreteFiltering
using Documenter

DocMeta.setdocmeta!(
    DiscreteFiltering,
    :DocTestSetup,
    :(using DiscreteFiltering);
    recursive = true,
)

makedocs(;
    modules = [DiscreteFiltering],
    authors = "Syver Døving Agdestein and contributors",
    repo = "https://github.com/agdestein/DiscreteFiltering.jl/blob/{commit}{path}#{line}",
    sitename = "DiscreteFiltering.jl",
    format = Documenter.HTML(;
        prettyurls = get(ENV, "CI", "false") == "true",
        canonical = "https://agdestein.github.io/DiscreteFiltering.jl",
        assets = String[],
    ),
    pages = ["Home" => "index.md"],
)

deploydocs(; repo = "github.com/agdestein/DiscreteFiltering.jl", devbranch = "main")
