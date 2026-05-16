# copied from InfrastructureSystems.jl test/load_tests.jl
using Revise

"""
    recursive_includet(filename)

Load tests for interactive use with `ReTest.jl`. Usage:

```julia
using TestEnv
TestEnv.activate()
include("test/load_tests.jl")
using .PowerFlowsTests
run_tests()
```

See the InfrastructureSystems.jl documentation page
["Running Tests"](https://sienna-platform.github.io/InfrastructureSystems.jl/stable/dev_guide/tests/)
for more details.

Copied from https://juliatesting.github.io/ReTest.jl/stable/#Working-with-Revise
"""
function recursive_includet(filename)
    already_included = copy(Revise.included_files)
    includet(filename)
    newly_included = setdiff(Revise.included_files, already_included)
    for (mod, file) in newly_included
        Revise.track(mod, file)
    end
end

recursive_includet("PowerFlowsTests.jl")
