# PowerFlows.jl Repository Guide

> **Development Guidelines:** Always load [Sienna.md](./Sienna.md) development preferences, style conventions, and best practices for projects using Sienna. Before running tests confirm that the [Sienna.md](./Sienna.md) file has been read.

## Overview

PowerFlows.jl is a Julia package designed for high-performance power flow analysis at scale. It provides an interface to multiple solution methods for the Power Flow problem, along with utilities commonly found in commercial software like Siemens PSS/e and GE PSLF. The package is architected to handle large-scale power systems (tens of thousands of buses) through extensive use of sparse matrix operations and specialized linear solvers. This package is part of the Sienna-Platform ecosystem focused on scalable power systems modeling and simulation.

## Main Objectives

1. **Multiple Solution Methods**: Provide a unified interface to various power flow solution algorithms:
   - DC Power Flow (linear approximation)
   - AC Power Flow using Newton-Raphson
   - AC Power Flow using Trust Region methods
   - AC Power Flow using Levenberg-Marquardt
   - Robust Homotopy methods for difficult-to-converge systems
   - PTDF-based DC Power Flow methods

2. **Commercial Software Compatibility**: Export power flow results in formats compatible with commercial software PSS/e. PSLF not supported yet.

3. **Integration with Sienna**: Seamless integration with PowerSystems.jl and PowerSimulations.jl (PSI) for comprehensive power systems analysis

4. **High-Performance at Scale**: Designed for large-scale power systems with thousands to tens of thousands of buses through:
   - Extensive use of sparse matrix operations (SparseArrays.jl)
   - Specialized sparse direct solvers (KLU from SuiteSparse)
   - Optimized Jacobian and Hessian construction leveraging sparsity patterns
   - In-place operations to minimize memory allocations
   - Efficient caching of factorizations and matrix structures
   - Support for parallel factorizations (CHOLMOD) in homotopy methods

5. **Flexibility**: Support for various power system features including:
   - Multi-period power flow analysis
   - HVDC systems and LCC converters
   - Distributed slack bus models
   - Arc types and network reductions
   - Loss factor calculations
   - Stabiltiy Factor Calculations

## Repository Structure

### Core Source Code (`src/`)

#### Main Module
- **`PowerFlows.jl`**: Main module file containing exports and include statements

#### Power Flow Data and Setup
- **`PowerFlowData.jl`**: Core data structure holding power flow problem state
- **`power_flow_types.jl`**: Type definitions for different power flow methods
- **`initialize_power_flow_data.jl`**: Initialize power flow data structures from PowerSystems
- **`power_flow_setup.jl`**: Setup utilities for power flow problems
- **`power_flow_method.jl`**: Entry point for solving power flow

#### DC Power Flow
- **`solve_dc_power_flow.jl`**: DC power flow implementation and PTDF methods

#### AC Power Flow
- **`solve_ac_power_flow.jl`**: AC power flow implementations (Newton-Raphson, Trust Region)
- **`ac_power_flow_residual.jl`**: Residual (mismatch) calculations for AC power flow
- **`ac_power_flow_jacobian.jl`**: Jacobian matrix construction for Newton methods
- **`levenberg-marquardt.jl`**: Levenberg-Marquardt algorithm implementation

#### Robust Homotopy (`RobustHomotopy/`)
- **`robust_homotopy_method.jl`**: Robust homotopy continuation method for difficult cases
- **`homotopy_hessian.jl`**: Hessian calculations for homotopy methods
- **`HessianSolver/`**: Specialized solvers for Hessian systems (KLU, CHOLMOD, Cholesky)

#### Linear Algebra (`LinearSolverCache/`)
- **`linear_solver_cache.jl`**: Caching infrastructure for linear solvers
- **`klu_linear_solver.jl`**: KLU sparse direct solver interface

#### Utilities
- **`common.jl`**: Shared utility functions
- **`definitions.jl`**: Constants and type definitions
- **`state_indexing_helpers.jl`**: Helpers for indexing state variables
- **`lcc_parameters.jl`** & **`lcc_utils.jl`**: Line-commutated converter (HVDC) utilities
- **`powersystems_utils.jl`**: Integration utilities with PowerSystems.jl
- **`psi_utils.jl`**: Integration utilities with PowerSimulations.jl
- **`post_processing.jl`**: Result processing and calculations
- **`psse_export.jl`**: PSS/e format export functionality

### Tests (`test/`)

Comprehensive test suite organized by functionality:
- **`runtests.jl`**: Main test entry point
- **DC Power Flow**: `test_dc_power_flow.jl`, `test_multiperiod_dc_power_flow.jl`, `test_reduced_dc_power_flow.jl`
- **AC Power Flow**: `test_solve_power_flow.jl`, `test_multiperiod_ac_power_flow.jl`, `test_reduced_ac_power_flow.jl`
- **Specialized Features**:
  - `test_hvdc.jl`: HVDC systems
  - `test_distributed_slack.jl`: Distributed slack bus
  - `test_robust_power_flow.jl`: Robust homotopy methods
  - `test_iterative_methods.jl`: Iterative solution methods
- **Components**: `test_jacobian.jl`, `test_homotopy_hessian.jl`, `test_klu_linear_solver_cache.jl`
- **Utilities**: `test_psse_export.jl`, `test_post_processing.jl`, `test_power_flow_data.jl`, `test_loss_factors.jl`
- **Integration**: `test_psi_utils.jl`
- **Performance**: `performance/performance_test.jl`
- **Test Data**: `test_data/` contains validation datasets and reference results
- **Test Utilities**: `test_utils/` contains helper functions for testing

### Documentation (`docs/`)

- **`src/`**: Documentation source files (Markdown)
- **`build/`**: Generated documentation (not tracked in git)
- **`make.jl`**: Documentation build script
- Structure follows Diataxis framework:
  - **`tutorials/`**: Learning-oriented guides
  - **`how-tos/`**: Task-oriented guides
  - **`reference/`**: Information-oriented technical reference
    - `api/`: API documentation (public and internal)
    - `developers/`: Developer documentation
  - **`explanation/`**: Understanding-oriented explanations

### Scripts (`scripts/`)

- **`formatter/`**: Code formatting utilities using JuliaFormatter

### Configuration Files

- **`Project.toml`**: Package dependencies and metadata
- **`Manifest.toml`**: Exact versions of all dependencies (locked)
- **`codecov.yml`**: Code coverage configuration
- **`CONTRIBUTING.md`**: Contribution guidelines
- **`LICENSE`**: BSD license

## Performance and Scalability Architecture

### Design Philosophy

PowerFlows.jl is explicitly designed for large-scale power system analysis, with performance as a first-class concern:

**Sparse-First Design**: All matrix operations use sparse representations. Power system networks have naturally sparse structure (each bus connects to only a few neighbors), and the package exploits this throughout:
- Admittance matrices (Y-bus)
- Jacobian matrices for Newton methods
- Hessian matrices for second-order methods
- PTDF matrices for sensitivity analysis

**Specialized Linear Solvers**: Different solvers optimized for different problem structures:
- **KLU**: Primary sparse direct solver for general AC power flow, optimized for circuit matrices
- **CHOLMOD**: Cholesky factorization for symmetric positive-definite systems in homotopy methods
- **Custom Hessian Solvers**: Specialized solvers for the unique structure of power flow Hessians

**Solver Caching and Reuse**: Linear solver caches maintain factorizations across iterations:
- Symbolic factorization computed once, reused across Newton iterations
- Numeric factorization updated only when needed
- Reduces overhead in iterative methods

**Memory Efficiency**:
- In-place operations (functions with `!`) to avoid unnecessary allocations
- Pre-allocated working arrays in `PowerFlowData`
- Views instead of copies where possible
- Careful attention to type stability (see [Sienna.md](./Sienna.md))

**Scalability Testing**: Package is validated on large-scale test systems:
- WECC system (Western Electricity Coordinating Council)
- EI system (Eastern Interconnect)
- ACTIVSg2000 (synthetic 2000-bus system)

### Performance-Critical Components

1. **Jacobian Construction** ([ac_power_flow_jacobian.jl](../src/ac_power_flow_jacobian.jl)): Sparse Jacobian assembly exploiting network topology
2. **Linear Solver Cache** ([LinearSolverCache/](../src/LinearSolverCache/)): Factorization management and reuse
3. **Residual Evaluation** ([ac_power_flow_residual.jl](../src/ac_power_flow_residual.jl)): Efficient mismatch calculations
4. **Homotopy Hessian** ([RobustHomotopy/homotopy_hessian.jl](../src/RobustHomotopy/homotopy_hessian.jl)): Second-order information for robust methods

### Performance Considerations for Contributors

When working on PowerFlows.jl, maintain performance discipline:
- Profile before optimizing (use `@time`, `@allocated`, `@code_warntype`)
- Preserve sparsity patterns in matrix operations
- Add in-place variants for hot-path operations
- Document computational complexity for new algorithms
- Test performance impact on large-scale systems
- See [Sienna.md](./Sienna.md) for detailed performance guidelines

## Key Concepts and Workflows

### Power Flow Problem

The power flow (load flow) problem solves for steady-state voltages (magnitude and angle) at all buses in a power network given:
- Generation levels
- Load demands
- Network topology and parameters

### Typical Usage Pattern

1. **Create System**: Use PowerSystems.jl to create or load a `System` object
2. **Choose Method**: Select appropriate power flow method (DC, AC with specific algorithm)
3. **Solve**: Call `solve_power_flow` or `solve_power_flow!` (in-place)
4. **Analyze Results**: Extract voltages, flows, losses, etc. from the results
5. **Export** (optional): Export results in PSS/e format using `PSSEExporter`

### Solution Methods Selection Guide

- **DCPowerFlow**: Fast linear approximation, good for screening studies
- **NewtonRaphsonACPowerFlow**: Standard method, fast convergence for well-conditioned systems
- **TrustRegionACPowerFlow**: More robust than Newton-Raphson, handles ill-conditioned cases better
- **LevenbergMarquardtACPowerFlow**: Robust nonlinear solver, good for difficult cases
- **RobustHomotopyPowerFlow**: Most robust method for hard-to-converge or non-convergent cases
- **PTDFDCPowerFlow** / **vPTDFDCPowerFlow**: DC power flow with pre-computed power transfer distribution factors

### Integration with PowerSimulations.jl (PSI)

PowerFlows is used within PowerSimulations.jl for:
- Initialization of dyanmic simulation problems
- Network model validation
- Post-processing of optimal power flow results

The `psi_utils.jl` file provides the integration layer.

## Development Workflow

1. **Setup**: Clone repository, instantiate environment with `]instantiate` in Julia REPL
2. **Code**: Follow Sienna conventions (see [Sienna.md](./Sienna.md))
3. **Format**: Run formatter from `scripts/formatter/` before committing
4. **Test**: Run tests with `]test PowerFlows` or specific test files
5. **Document**: Add docstrings following Diataxis framework
6. **Submit**: Create pull request following CONTRIBUTING.md

## Active Development Areas

Based on the current PR context (Refactor: Updates on PSS/e Exporter, testing DCPF validation on EI/WECC):
- **PSS/e Export Functionality**: Ongoing refactoring and enhancement of power system export capabilities
- **DC Power Flow Validation**: Testing and validation on large-scale systems (Eastern Interconnect, WECC)
- **Exporter Architecture**: Improving the design and maintainability of export functionality

## Dependencies

### Key External Packages
- **PowerSystems.jl**: Power system data models and network representation
- **PowerNetworkMatrices.jl**: Network matrix formulations (admittance, incidence, PTDF)
- **InfrastructureSystems.jl**: Core infrastructure shared across Sienna packages
- **KLU.jl**: Sparse direct solver (from SuiteSparse)
- **DataFrames.jl**: Tabular data handling for results

## Resources

- **Documentation**: https://sienna-platform.github.io/PowerFlows.jl/dev/
- **Issues**: GitHub Issues for bug reports and feature requests
- **Style Guide**: https://sienna-platform.github.io/InfrastructureSystems.jl/stable/style/

## Version Information

- **Julia Compatibility**: 1.10+
- **Status**: Active development (API subject to change)

---

*This document is maintained to help developers (human and AI) understand and contribute to PowerFlows.jl effectively.*
