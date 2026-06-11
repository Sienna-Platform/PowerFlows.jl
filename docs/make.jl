using Documenter, PowerSystems, DocStringExtensions, PowerFlows, DataStructures
using DocumenterInterLinks

links = InterLinks(
    "DocumenterInterLinks" => "http://juliadocs.org/DocumenterInterLinks.jl/stable/",
    "PowerSystems" => "https://sienna-platform.github.io/PowerSystems.jl/stable/",
    "PowerNetworkMatrices" => "https://sienna-platform.github.io/PowerNetworkMatrices.jl/stable/",
    "PowerSimulations" => "https://sienna-platform.github.io/PowerSimulations.jl/stable/",
    "PowerSystemCaseBuilder" => "https://sienna-platform.github.io/PowerSystemCaseBuilder.jl/stable/",
    "Julia" => "https://docs.julialang.org/en/v1/",
)

include(joinpath(@__DIR__, "make_tutorials.jl"))
make_tutorials()

pages = OrderedDict(
    "Welcome Page" => "index.md",
    "Tutorials" => Any[
        "Solving a Power Flow" => "tutorials/generated_solving_a_power_flow.md",
    ],
    "How-to-Guides" => Any[
        "How to choose an AC formulation and solver" => "how-tos/choose_ac_formulation_and_solver.md",
    ],
    "Explanation" => Any[
        "Evaluation Models vs. Solver Algorithms" => "explanation/models-and-solvers.md",
        "Mixed Current-Power Balance Formulation" => "explanation/mixed_cpb_formulation.md",
        "Levenberg-Marquardt vs Gauss-Seidel" => "explanation/lm_vs_gauss_seidel.md",
        "Folds, Voltage Collapse, and Solver Diagnostics" => "explanation/folds_and_diagnostics.md",
        "LCC Model Implementation" => "explanation/lcc_model.md",
    ],
    "Reference" => Any[
        "Public API Reference" => "reference/api/public.md",
        "Internal API Reference - Core" => "reference/api/internal.md",
        "Internal API Reference - Solvers & Utilities" => "reference/api/internal_solvers.md",
        "Code Base Developer Guide" => Any[
            "Guidelines" => "reference/developers/developer.md",
            "Power Flow Overview for Developers" => "reference/developers/power_flow.md",
        ],
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
    # Only process .md files referenced in `pages` so that excluded tutorials
    # (see note above) are not executed during the docs build.
    pagesonly = true,
    warnonly = get(ENV, "POWERFLOWS_DOCS_WARNONLY", "false") == "true",
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
