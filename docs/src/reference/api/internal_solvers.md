# Internal - Solvers and Utilities

```@meta
CurrentModule = PowerFlows
DocTestSetup  = quote
    using PowerFlows
end
```

## Iterative Methods

```@autodocs
Modules = [PowerFlows]
Public = false
Private = true
Pages = [
    "power_flow_method.jl",
    "gradient_descent_ac_power_flow.jl",
]
```

## Robust Homotopy Method

```@autodocs
Modules = [PowerFlows]
Public = false
Private = true
Pages = [
    "RobustHomotopy/homotopy_hessian.jl",
    "RobustHomotopy/robust_homotopy_method.jl",
]
```

## Levenberg-Marquardt Method

```@autodocs
Modules = [PowerFlows]
Public = true
Private = true
Pages = [
    "levenberg-marquardt.jl",
]
```

# Linear Algebra Backends

## Robust Homotopy
```@autodocs
Modules = [PowerFlows]
Public = true
Private = true
Pages = [
    "RobustHomotopy/HessianSolver/cholesky_solver.jl",
    "RobustHomotopy/HessianSolver/fixed_structure_CHOLMOD.jl",
    "RobustHomotopy/HessianSolver/hessian_solver.jl",
    "RobustHomotopy/HessianSolver/KLU_hessian_solver.jl",
]
```

## Newton-Raphson
```@autodocs
Modules = [PowerFlows]
Public = true
Private = true
Pages = [
    "LinearSolverCache/klu_linear_solver.jl",
    "LinearSolverCache/linear_solver_cache.jl",
]
```

# Misc.

## PSSE Export

```@autodocs
Modules = [PowerFlows]
Public = false
Private = true
Pages = [
    "psse_export.jl",
]
```

## Post-Processing

```@autodocs
Modules = [PowerFlows]
Public = false
Private = true
Pages = [
    "post_processing.jl",
]
```

## Power Systems Utilities

```@autodocs
Modules = [PowerFlows]
Public = false
Private = true
Pages = [
    "powersystems_utils.jl",
]
```

## Common Utilities and Definitions

```@autodocs
Modules = [PowerFlows]
Public = false
Private = true
Pages = [
    "common.jl",
    "definitions.jl",
    "psi_utils.jl",
]
```
