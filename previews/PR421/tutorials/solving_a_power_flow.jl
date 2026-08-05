# # Solving a Power Flow
# In this tutorial, you'll solve power flows on a 5-bus test system using three different
# solvers and compare their results.

# ## Building a System
# To get started, load the needed packages.

using PowerSystemCaseBuilder
using PowerFlows
using PowerSystems

# Create a [`PowerSystems.System`](@extref) from [`PowerSystemCaseBuilder.build_system`](@extref).
# We build the test system with `runchecks = false` to reduce REPL output:

sys = build_system(MatpowerTestSystems, "matpower_case5_sys"; runchecks = false)

# ## DC Power Flow
# [`DCPowerFlow`](@ref) solves for bus voltage angles using the bus admittance matrix,
# then computes branch flows from the angle differences. Create a [`DCPowerFlow`](@ref) solver:

pf_dc = DCPowerFlow()

# Solve the power flow with [`solve_power_flow`](@ref). For DC methods, pass a
# [`FlowReporting`](@ref) mode so flows are reported on the
# same arc basis as elsewhere in the package:

dc_results = solve_power_flow(pf_dc, sys, FlowReporting.ARC_FLOWS)

# The result is a `Dict{String, Dict{String, DataFrame}}`. The outer key is the time step
# name: `"1"`. The inner dictionary stores the power flow results at that time step:
# `"bus_results"` for bus data and `"flow_results"` for AC line data.
# (There is also a third key, `"lcc_results"`, for HVDC lines, but this system
# contains no such components, so the matching dataframe will be empty.) Inspect `"bus_results"`:

dc_results["1"]["bus_results"]

# Notice that `Vm` (voltage magnitude) is 1.0 for all buses, and `Q_gen` and `Q_load` are 0.
# This is expected for DC power flow, which assumes flat voltage magnitudes and ignores reactive power.

dc_results["1"]["flow_results"]

# Likewise, `Q_from_to` and `Q_to_from` (reactive power flow on the line) are zero, for all lines.

# ## PTDF DC Power Flow
# [`PTDFDCPowerFlow`](@ref) computes branch flows directly from bus power injections using
# the Power Transfer Distribution Factor matrix, without solving for voltage angles as an
# intermediate step. (This means we can omit the angle computation in contexts where we only
# care about line flows, though we don't have that option implemented here.) Create a [`PTDFDCPowerFlow`](@ref)
# solver:

pf_ptdf = PTDFDCPowerFlow()

# As before, solve the power flow with [`solve_power_flow`](@ref):

ptdf_results = solve_power_flow(pf_ptdf, sys, FlowReporting.ARC_FLOWS)

# Look at the bus results:

ptdf_results["1"]["bus_results"]

# The results match `DCPowerFlow`, as they should: the two are mathematically equivalent.
# For very large systems where forming the full PTDF matrix would be too expensive,
# consider [`vPTDFDCPowerFlow`](@ref), which computes the same results without
# storing the dense matrix.

# ## AC Power Flow
# Create an [`ACPowerFlow`](@ref) solver:

pf_ac = ACPowerFlow()

# Solve the power flow:

ac_results = solve_power_flow(pf_ac, sys)

# AC results are returned as a flat `Dict{String, DataFrame}`, with the same keys as
# before: `"bus_results"`, `"flow_results"` (AC lines), and `"lcc_results"` (HVDC lines).
# Look at the bus results:

ac_results["bus_results"]

# Notice that `Vm` now varies across buses (not all 1.0), and `Q_gen` has non-zero values.
#
# Look at the line flows:

ac_results["flow_results"]

# `Q_from_to` and `Q_to_from` now show reactive power flows, and `P_from_to` differs from
# `P_to_from` due to losses.

# ## Fast Decoupled AC Power Flow
# The solver is an independent type parameter of the AC power flow. [`ACPowerFlow`](@ref)`()`
# above used the default [`NewtonRaphsonACPowerFlow`](@ref) solver, which refactorizes the
# Jacobian every iteration. [`FastDecoupledACPowerFlow`](@ref) instead builds constant
# approximate Jacobian matrices (``B'``/``B''``) **once** and reuses them across all iterations
# and time steps — the fast decoupled power flow of Stott & Alsac, equivalent to PSS/E's `FDNS`.
# It converges at a linear rate but at a fraction of the per-iteration cost, which makes it well
# suited to repeated solves (multi-period dispatch, contingency screening). Select it as a
# solver type parameter:

pf_fd = ACPowerFlow{FastDecoupledACPowerFlow}()

# Solve it the same way as any other AC solver:

fd_results = solve_power_flow(pf_fd, sys)

# The converged solution is identical to the Newton-Raphson result to tolerance: fast decoupled
# evaluates the *exact* mismatches every iteration, so only the convergence *rate* differs, never
# the solution. Compare the bus results:

fd_results["bus_results"]

# ### Handoff to Newton-Raphson
# Because the fast decoupled state is a valid iterate of the exact residual, it is also a good
# warm start for an exact-Newton solver. You can optionally run the cheap fast decoupled stage to
# a looser tolerance, then hand off to [`NewtonRaphsonACPowerFlow`](@ref) for final refinement,
# via `solver_settings`:

pf_handoff = ACPowerFlow{FastDecoupledACPowerFlow}(;
    solver_settings = Dict(:handoff_solver => NewtonRaphsonACPowerFlow),
)
handoff_results = solve_power_flow(pf_handoff, sys)

# This staging — fast decoupled as a cheap conditioner, Newton-Raphson as the closer — mirrors
# commercial practice. For the underlying math (``B'``/``B''``, the XB/BX schemes, the
# fixed-Jacobian variant, and a solver guidance table), see
# [Fast/Fixed Decoupled Power Flow](@ref).

# ## When AC Power Flow Fails
# Unlike DC power flow, AC power flow is iterative and not guaranteed to converge. Systems
# with high impedance lines, poor initial voltage profiles, or insufficient reactive power
# support can cause the solver to fail. When this happens, `solve_power_flow` returns
# `missing`: you'll also see a logged error. If you encounter convergence failures, consider
# using a more robust solver such as [`TrustRegionACPowerFlow`](@ref) or [`RobustHomotopyPowerFlow`](@ref).

# ## Next Steps
#
# - Compare AC formulations and solvers at scale in [How to choose an AC formulation and solver](@ref choose-ac-formulation-and-solver).
# - Read [Evaluation Models vs. Solver Algorithms](@ref) for the distinction between
#   evaluation models and iterative solvers (polar, rectangular, and mixed).
# - For the mixed current–power balance formulation, see
#   [Mixed Current-Power Balance Formulation](@ref).
# - For the fast/fixed decoupled solver, see
#   [Fast/Fixed Decoupled Power Flow](@ref).
