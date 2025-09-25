using Documenter
using Random
using TensorKit
using TensorKit.TensorKitSectors
using TensorKit.MatrixAlgebraKit
using DocumenterInterLinks

links = InterLinks("MatrixAlgebraKit" => "https://quantumkithub.github.io/MatrixAlgebraKit.jl/stable/",
                   "TensorOperations" => "https://quantumkithub.github.io/TensorOperations.jl/stable/")

pages = ["Home" => "index.md",
         "Manual" => ["man/intro.md", "man/tutorial.md", "man/categories.md",
                      "man/spaces.md", "man/sectors.md", "man/tensors.md"],
         "Library" => ["lib/sectors.md", "lib/spaces.md", "lib/tensors.md"],
         "Index" => ["index/index.md"]]

makedocs(; modules=[TensorKit, TensorKitSectors],
         sitename="TensorKit.jl",
         authors="Jutho Haegeman",
         warnonly=[:missing_docs, :cross_references],
         format=Documenter.HTML(; prettyurls=true, mathengine=MathJax(),
                                assets=["assets/custom.css"]),
         pages=pages,
         pagesonly=true,
         plugins=[links])

deploydocs(; repo="github.com/QuantumKitHub/TensorKit.jl.git", push_preview=true)
