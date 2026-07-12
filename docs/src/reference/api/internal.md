# Internal - Core

```@meta
CurrentModule = PowerFlows
DocTestSetup  = quote
    using PowerFlows
end
```

## Power Flow Types

```@autodocs
Modules = [PowerFlows]
Public = false
Private = true
Pages = [
    "power_flow_types.jl",
]
```

# Power Flow Data

## Struct and Type Definitions

```@autodocs
Modules = [PowerFlows]
Public = true
Private = true
Pages = [
    "PowerFlowData.jl",
]
```

## Solving a PowerFlowData instance

```@autodocs
Modules = [PowerFlows]
Public = false
Private = true
Pages = [
    "solve_ac_power_flow.jl",
    "solve_dc_power_flow.jl",
    "dcpf_loss_injection.jl",
]
```

## Manipulating a PowerFlowData instance

```@autodocs
Modules = [PowerFlows]
Public = true
Private = true
Pages = [
    "state_indexing_helpers.jl",
    "initialize_power_flow_data.jl",
    "power_flow_setup.jl",
]
```

# LCC HVDC Parameters and Utilities

```@autodocs
Modules = [PowerFlows]
Public = false
Private = true
Pages = [
    "lcc_parameters.jl",
    "lcc_utils.jl",
]
```

# VSC HVDC Parameters and Utilities

```@autodocs
Modules = [PowerFlows]
Public = false
Private = true
Pages = [
    "vsc_parameters.jl",
    "vsc_utils.jl",
]
```

# Area Interchange Control

## Types

```@autodocs
Modules = [PowerFlows]
Public = false
Private = true
Pages = [
    "area_interchange/area_types.jl",
]
```

## Tie Detection and Enrollment

```@autodocs
Modules = [PowerFlows]
Public = false
Private = true
Pages = [
    "area_interchange/tie_set.jl",
    "area_interchange/enrollment.jl",
]
```

## Residuals and Jacobian

```@autodocs
Modules = [PowerFlows]
Public = false
Private = true
Pages = [
    "area_interchange/area_residual.jl",
    "area_interchange/area_jacobian.jl",
]
```

# AC Power Flow

## Residuals

```@autodocs
Modules = [PowerFlows]
Public = false
Private = true
Pages = [
    "ac_power_flow_residual.jl",
]
```

## Jacobian

```@autodocs
Modules = [PowerFlows]
Public = false
Private = true
Pages = [
    "ac_power_flow_jacobian.jl",
]
```

# Rectangular Current-Injection AC Power Flow

## Setup

```@autodocs
Modules = [PowerFlows]
Public = false
Private = true
Pages = [
    "rectangular_ci_setup.jl",
]
```

## Residuals

```@autodocs
Modules = [PowerFlows]
Public = false
Private = true
Pages = [
    "rectangular_ci_power_flow_residual.jl",
]
```

## Jacobian

```@autodocs
Modules = [PowerFlows]
Public = false
Private = true
Pages = [
    "rectangular_ci_power_flow_jacobian.jl",
]
```

# Mixed Current-Power Balance AC Power Flow

## Setup

```@autodocs
Modules = [PowerFlows]
Public = false
Private = true
Pages = [
    "mixed_cpb_setup.jl",
]
```

## Residuals

```@autodocs
Modules = [PowerFlows]
Public = false
Private = true
Pages = [
    "mixed_cpb_power_flow_residual.jl",
]
```

## Jacobian

```@autodocs
Modules = [PowerFlows]
Public = false
Private = true
Pages = [
    "mixed_cpb_power_flow_jacobian.jl",
]
```
