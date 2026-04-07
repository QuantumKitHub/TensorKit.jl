using Pkg

# if examples is not the current active environment, switch to it
if Base.active_project() != joinpath(@__DIR__, "Project.toml")
    Pkg.activate(@__DIR__)
    Pkg.develop(PackageSpec(; path = joinpath(@__DIR__, "..")))
    Pkg.resolve()
    Pkg.instantiate()
end

using Documenter
using DocumenterTypst
using Random
using TensorKit
using TensorKit: FusionTreePair, Index2Tuple
using TensorKit.TensorKitSectors
using TensorKit.MatrixAlgebraKit
using DocumenterInterLinks

links = InterLinks(
    "MatrixAlgebraKit" => "https://quantumkithub.github.io/MatrixAlgebraKit.jl/stable/",
    "TensorOperations" => "https://quantumkithub.github.io/TensorOperations.jl/stable/",
    "TensorKitSectors" => "https://quantumkithub.github.io/TensorKitSectors.jl/dev/"
)

pages = [
    "Home" => "index.md",
    "Manual" => [
        "man/intro.md", "man/tutorial.md",
        "man/spaces.md", "man/symmetries.md",
        "man/sectors.md", "man/gradedspaces.md",
        "man/fusiontrees.md", "man/tensors.md",
        "man/tensormanipulations.md",
    ],
    "Library" => [
        "lib/sectors.md", "lib/fusiontrees.md",
        "lib/spaces.md", "lib/tensors.md",
    ],
    "Index" => ["index/index.md"],
    "Appendix" => ["appendix/symmetric_tutorial.md", "appendix/categories.md"],
    "Changelog" => "Changelog.md",
]

mathengine = MathJax3(
    Dict(
        :loader => Dict("load" => ["[tex]/physics"]),
        :tex => Dict(
            "inlineMath" => [["\$", "\$"], ["\\(", "\\)"]],
            "tags" => "ams",
            "packages" => ["base", "ams", "autoload", "physics"]
        )
    )
)

# docstrings don't need `using TensorKit`
DocMeta.setdocmeta!(TensorKit, :DocTestSetup, :(using TensorKit); recursive = true)

makedocs(;
    modules = [TensorKit, TensorKitSectors],
    sitename = "TensorKit.jl",
    authors = "Jutho Haegeman",
    warnonly = [:missing_docs, :cross_references],
    format = Documenter.HTML(;
        prettyurls = true, mathengine, assets = ["assets/custom.css"]
    ),
    # format = DocumenterTypst.Typst(),
    pages = pages,
    pagesonly = true,
    plugins = [links]
)

deploydocs(; repo = "github.com/QuantumKitHub/TensorKit.jl.git", push_preview = true)
