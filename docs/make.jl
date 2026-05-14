using Documenter, PowerSystems, DocStringExtensions, PowerFlows, DataStructures
using DocumenterInterLinks

links = InterLinks(
    "DocumenterInterLinks" => "http://juliadocs.org/DocumenterInterLinks.jl/stable/",
    "PowerSystems" => "https://sienna-platform.github.io/PowerSystems.jl/stable/",
    "PowerNetworkMatrices" => "https://sienna-platform.github.io/PowerNetworkMatrices.jl/stable/",
)

include(joinpath(@__DIR__, "make_tutorials.jl"))
make_tutorials()

pages = OrderedDict(
    "Welcome Page" => "index.md",
    "How-to-Guides" => "how-tos/stub.md",
    "Tutorials" => Any[
        "Solving a Power Flow" => "tutorials/generated_solving_a_power_flow.md",
    ],
    "Explanation" => Any[
        "LCC Model Implementation" => "explanation/lcc_model.md",
    ],
    "Reference" => Any[
        "Public API Reference" => "reference/api/public.md",
        "Internal API Reference - Core" => "reference/api/internal.md",
        "Internal API Reference - Solvers & Utilities" => "reference/api/internal_solvers.md",
        "Code Base Developer Guide" => "reference/developers/developer.md",
    ],
)

makedocs(;
    modules = [PowerFlows],
    format = Documenter.HTML(;
        prettyurls = haskey(ENV, "GITHUB_ACTIONS"),
        mathengine = Documenter.MathJax(),
    ),
    sitename = "PowerFlows.jl",
    pages = Any[p for p in pages],
    plugins = [links],
)

deploydocs(;
    repo = "github.com/Sienna-Platform/PowerFlows.jl.git",
    target = "build",
    branch = "gh-pages",
    devbranch = "main",
    devurl = "dev",
    push_preview = true,
    versions = ["stable" => "v^", "v#.#"],
)
