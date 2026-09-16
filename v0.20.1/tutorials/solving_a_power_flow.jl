# # Solving a Power Flow
# In this tutorial, you'll solve power flows on a 5-bus test system using three different
# solvers and compare their results.

# ## Building a System
# To get started, load the needed packages. We're using a standard test system and want to
# keep output clean, so we adjust the logging settings to filter out a few precautionary warnings.

using PowerSystemCaseBuilder
using PowerFlows
using PowerSystems
using Logging

disable_logging(Logging.Warn)

# Create a [`System`](@extref PowerSystems.System) from [PowerSystemCaseBuilder.jl](https://github.com/sienna-platform/PowerSystemCaseBuilder.jl):

sys = build_system(MatpowerTestSystems, "matpower_case5_sys")

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
# (We don't support multi-period AC power flows yet.) Look at the bus results:

ac_results["bus_results"]

# Notice that `Vm` now varies across buses (not all 1.0), and `Q_gen` has non-zero values.
#
# Look at the line flows:

ac_results["flow_results"]

# `Q_from_to` and `Q_to_from` now show reactive power flows, and `P_from_to` differs from
# `P_to_from` due to losses.

# ## When AC Power Flow Fails
# Unlike DC power flow, AC power flow is iterative and not guaranteed to converge. Systems
# with high impedance lines, poor initial voltage profiles, or insufficient reactive power
# support can cause the solver to fail. When this happens, `solve_power_flow` returns
# `missing`: you'll also see a logged error. If you encounter convergence failures, consider
# using a more robust solver such as [`TrustRegionACPowerFlow`](@ref) or [`RobustHomotopyPowerFlow`](@ref).

# ## Choosing a Formulation and Solver
# AC power flow has two independent choices: the **formulation** (how the
# network equations are written) and the **solver** (the iterative algorithm).
# Each is a type parameter: `ACPolarPowerFlow{S}` (power balance, polar state),
# [`ACRectangularPowerFlow`](@ref)`{S}` (Da Costa current injection), and
# [`ACMixedPowerFlow`](@ref)`{S}` (mixed current/power balance, the most compact
# state) — each combined with [`NewtonRaphsonACPowerFlow`](@ref) (`NR`),
# [`TrustRegionACPowerFlow`](@ref) (`TR`), or
# [`LevenbergMarquardtACPowerFlow`](@ref) (`LM`). `ACPowerFlow()` is
# `ACPolarPowerFlow{NewtonRaphsonACPowerFlow}` — a good default.
#
# Warm-solve timings: median of 10 runs after warm-up, with the `[min, max]`
# range. Hardware-dependent — compare medians across cells, not absolutes.
#
# 2000-bus (`ACTIVSg2000`, tol `1e-9`):
#
# | Formulation \ Solver | NR | TR | LM |
# |---|---|---|---|
# | Polar       | 4 it, 0.032 s `[0.031, 0.045]` | 3 it, 0.032 s `[0.031, 0.250]` | 4 it, 0.104 s `[0.095, 0.348]` |
# | Rectangular | 4 it, 0.028 s `[0.028, 0.044]` | 3 it, **0.029 s** `[0.028, 0.042]` | 7 it, 0.150 s `[0.136, 0.398]` |
# | Mixed       | 5 it, 0.030 s `[0.029, 0.039]` | 4 it, 0.029 s `[0.029, 0.041]` | 3 it, 0.083 s `[0.075, 0.341]` |
#
# 10000-bus (`ACTIVSg10k`, tol `1e-9`):
#
# | Formulation \ Solver | NR | TR | LM |
# |---|---|---|---|
# | Polar       | 5 it, 0.214 s `[0.208, 0.447]` | 4 it, 0.223 s `[0.210, 0.433]` | 5 it, 0.614 s `[0.492, 0.792]` |
# | Rectangular | 4 it, **0.183 s** `[0.178, 0.402]` | 3 it, 0.189 s `[0.180, 0.399]` | 56 it, 4.315 s `[4.15, 4.42]` |
# | Mixed       | 5 it, 0.186 s `[0.182, 0.209]` | 4 it, 0.194 s `[0.182, 0.414]` | 5 it, 0.590 s `[0.451, 0.755]` |
#
# All nine combinations converge to the same solution. On small systems the
# differences are negligible (every combination solves a 14-bus case in
# 2–3 iterations in well under 1 ms median); the choice matters at scale.
# The wide LM ranges reflect its per-iteration refactorization variance — the
# median is the representative figure.
#
# Recommendations (based on the median):
#
# - **Default / general use:** `ACPowerFlow()` (Polar + NR). Well-trodden and
#   the reference all other formulations are validated against.
# - **Fastest at scale:** [`ACRectangularPowerFlow`](@ref)`{NewtonRaphsonACPowerFlow}`
#   or `{TrustRegionACPowerFlow}`. The rectangular formulation's off-diagonal
#   Jacobian blocks are the constant admittance matrix, so the sparse
#   factorization is reused across iterations — consistently ~15% faster than
#   Polar/NR by median (2000 and 10k bus alike); `ACMixedPowerFlow` with NR/TR
#   is within a few percent of it.
# - **Most robust (poor start, ill-conditioned, high-impedance):**
#   [`TrustRegionACPowerFlow`](@ref) on any formulation — its median is on par
#   with NR at every size while globalizing the step. For cases that still will
#   not converge, [`RobustHomotopyPowerFlow`](@ref) (Polar only).
# - **Smallest state / most predictable:** [`ACMixedPowerFlow`](@ref) — `2n`
#   unknowns, the most compact of the three, and the tightest timing spread at
#   every scale (Mixed/NR 10k `[0.182, 0.209]`).
# - **Levenberg-Marquardt:** a robust least-squares fallback, the slowest per
#   iteration (it refactorizes a sparse augmented system every step) and the
#   highest variance — prefer NR/TR and reach for LM only when they stall. If
#   you need LM, use **Mixed/LM** (the only LM that scales: 3–5 iterations and
#   the lowest LM median at every size) or Polar/LM. **Avoid Rectangular/LM at
#   scale**: its Marquardt scaling helps at ≤2000 buses (7 iters) but does not
#   scale — 56 iterations / 4.3 s at 10k, ~20× Rectangular/NR. See
#   [Levenberg-Marquardt in Place of Gauss-Seidel](@ref).
#
# See also the explanation pages
# [Mixed Current-Power Balance Formulation](@ref) and
# [Levenberg-Marquardt in Place of Gauss-Seidel](@ref) for the underlying
# trade-offs.
