# PowerFlows.jl

```@meta
CurrentModule = PowerFlows
```

## Overview

`PowerFlows.jl` is part of the National Laboratory of the Rockies'
[Sienna platform](https://sienna-platform.github.io/Sienna/), an open source framework for
scheduling problems and dynamic simulations for power systems. The Sienna platform can be
[found on github](https://github.com/Sienna-Platform/).

`PowerFlows.jl` provides a uniform interface to multiple power-flow formulations and solvers
for [`PowerSystems.jl`](https://sienna-platform.github.io/PowerSystems.jl/stable/) data models.
Main capabilities include:

  - **DC power flow** — bus-angle formulation and PTDF-based (dense and virtual) methods, with optional multi-period solves.
  - **AC power flow** — polar, rectangular current-injection, and mixed current–power balance formulations.
  - **Iterative AC solvers** — Newton–Raphson, trust region, Levenberg–Marquardt, and robust homotopy options.
  - **Multi-period DC workflows** — batch validation of time-coupled dispatches (for example, post–unit commitment checks).
  - **Post-processing and export** — structured `DataFrame` results, optional PSS/e export, loss and voltage-stability factors.

The package builds on network matrices from
[`PowerNetworkMatrices.jl`](https://sienna-platform.github.io/PowerNetworkMatrices.jl/stable/)
and is commonly used with operations simulations in
[`PowerSimulations.jl`](https://sienna-platform.github.io/PowerSimulations.jl/stable/)
(both power-flow-in-the-loop and post-solve validation). Test systems for the tutorials come from
[`PowerSystemCaseBuilder.jl`](https://sienna-platform.github.io/PowerSystemCaseBuilder.jl/stable/).

`PowerFlows.jl` is under active development; we welcome feedback, suggestions, and bug reports.

## Installation and Quick Links

  - [Sienna installation page](https://sienna-platform.github.io/Sienna/SiennaDocs/docs/build/how-to/install/):
    Instructions to install `PowerFlows.jl` and other Sienna packages
  - [Sienna Documentation Hub](https://sienna-platform.github.io/Sienna/SiennaDocs/docs/build/index.html):
    Links to other Sienna packages' documentation

## How To Use This Documentation

There are four main sections containing different information:

  - **Tutorials** — Detailed walk-throughs to help you *learn* how to use `PowerFlows.jl`
  - **How-to-Guides** — Directions to help *guide* your work for a particular task
  - **Explanation** — Additional details and background information to help you *understand*
    `PowerFlows.jl`, its formulations, and solver trade-offs
  - **Reference** — Technical references and API for a quick *look-up* during your work

`PowerFlows.jl` strives to follow the [Diataxis](https://diataxis.fr/) documentation framework.
