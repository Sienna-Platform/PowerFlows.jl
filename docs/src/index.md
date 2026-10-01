# PowerFlows.jl

```@meta
CurrentModule = PowerFlows
```

## Overview

`PowerFlows.jl` provides a uniform interface to multiple power-flow formulations and solvers
for [`PowerSystems.jl`](https://sienna-platform.github.io/PowerSystems.jl/stable/) data models.
Main capabilities include:

  - **DC power flow** — bus-angle formulation and PTDF-based (dense and virtual) methods, with optional multi-period solves.
  - **AC power flow** — polar, rectangular current-injection, and mixed current–power balance formulations.
  - **Iterative AC solvers** — Newton–Raphson, trust region, Levenberg–Marquardt, and robust homotopy options.
  - **Multi-period DC workflows** — batch validation of time-coupled dispatches (for example, post–unit commitment checks).
  - **Post-processing and export** — structured `DataFrame` results, optional PSS/E export, loss and voltage-stability factors.

The package builds on network matrices from
[`PowerNetworkMatrices.jl`](https://sienna-platform.github.io/PowerNetworkMatrices.jl/stable/)
and is commonly used with operations simulations in
[`PowerSimulations.jl`](https://sienna-platform.github.io/PowerSimulations.jl/stable/)
(both power-flow-in-the-loop and post-solve validation). Test systems for the tutorials come from
[`PowerSystemCaseBuilder.jl`](https://sienna-platform.github.io/PowerSystemCaseBuilder.jl/stable/).

`PowerFlows.jl` is under active development; we welcome feedback, suggestions, and bug reports.

## About Sienna

`PowerFlows.jl` is part of the National Laboratory of the Rockies (formerly known as NREL)'s
[Sienna ecosystem](https://sienna-platform.github.io/Sienna/), an open source framework for
scheduling problems and dynamic simulations for power systems. The Sienna ecosystem can be
[found on GitHub](https://github.com/Sienna-Platform). It contains three applications:

  - [Sienna\Data](https://sienna-platform.github.io/Sienna/pages/applications/sienna_data.html) enables
    efficient data input, analysis, and transformation
  - [Sienna\Ops](https://sienna-platform.github.io/Sienna/pages/applications/sienna_ops.html) enables
    system scheduling simulations by formulating and solving optimization problems
  - [Sienna\Dyn](https://sienna-platform.github.io/Sienna/pages/applications/sienna_dyn.html) enables
    system transient analysis including small signal stability and full system dynamic
    simulations

Each application uses multiple packages in the [`Julia`](http://www.julialang.org)
programming language. `PowerFlows.jl` supports Sienna\Data and Sienna\Ops with power-flow
formulations and solvers on `PowerSystems.jl` data models.

## How To Use This Documentation

There are four main sections containing different information:

  - **Tutorials** — Detailed walk-throughs to help you *learn* how to use `PowerFlows.jl`
  - **How-to-Guides** — Directions to help *guide* your work for a particular task
  - **Explanation** — Additional details and background information to help you *understand*
    `PowerFlows.jl`, its formulations, and solver trade-offs
  - **Reference** — Technical references and API for a quick *look-up* during your work

`PowerFlows.jl` strives to follow the [Diátaxis](https://diataxis.fr/) documentation framework.

## Installation and Quick Links

  - [Sienna installation page](https://sienna-platform.github.io/Sienna/SiennaDocs/docs/build/how-to/install/):
    Instructions to install `PowerFlows.jl` and other Sienna packages
  - [Central Sienna documentation](https://sienna-platform.github.io/Sienna/SiennaDocs/docs/build/index.html):
    Cross-linked documentation website for the core user-facing Sienna packages
