using Documenter, PowerSystems, DocStringExtensions, PowerFlows, DataStructures
using PowerSimulations, HiGHS
using DocumenterInterLinks

links = InterLinks(
    "DocumenterInterLinks" => "http://juliadocs.org/DocumenterInterLinks.jl/stable/",
    "PowerSystems" => "https://sienna-platform.github.io/PowerSystems.jl/stable/",
    "PowerNetworkMatrices" => "https://sienna-platform.github.io/PowerNetworkMatrices.jl/stable/",
)

pages = OrderedDict(
    "Welcome Page" => "index.md",
    "Tutorials" => Any[
        "Solving a Power Flow" => "tutorials/solving-a-power-flow.md",
        "Validating a UC Dispatch" => "tutorials/power-flow-after-unit-commitment.md",
        "Power Flow In The Loop with UC" => "tutorials/uc-power-flow-in-the-loop.md",
    ],
    "How-to-Guides" => "how-tos/stub.md",
    "Explanation" => Any[
        "Evaluation Models vs. Solver Algorithms" => "explanation/models-and-solvers.md",
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
