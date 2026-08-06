# Precompilation workload: runs the core solve paths on a tiny, fully in-memory system at
# precompile time so first-call latency moves from every user session into the cached
# pkgimage build.
#
# Design constraints:
#   - No PowerSystemCaseBuilder, no file parsing, no deserialization: the system is
#     hand-built from PSY constructors (the same pattern as `test/test_utils/common.jl`),
#     so precompile pays no I/O or parsing cost.
#   - Nothing leaks out of the workload: every object is local to the block, so KLU
#     factorization handles inside the locally-built `PowerFlowData` are never serialized
#     into the image.
#   - Solver logging is silenced so the precompile output stays clean.
#
# To skip the workload during development iterations (rebuilds run it on every
# precompile), set the standard PrecompileTools preference for this package:
#     using PrecompileTools, Preferences
#     Preferences.set_preferences!(PowerFlows, "precompile_workload" => false; force = true)

# Minimal in-memory system the workload solves: 4 buses covering the REF/PV/PQ bus-type
# partitions, a Source and a ThermalStandard injector, two PowerLoads, a ZIP StandardLoad,
# and four Lines forming a mesh.
function _precompilation_workload_system()
    sys = PSY.System(100.0; time_series_in_memory = true)
    bus_types = (
        PSY.ACBusTypes.REF,
        PSY.ACBusTypes.PV,
        PSY.ACBusTypes.PQ,
        PSY.ACBusTypes.PQ,
    )
    buses = [
        PSY.ACBus(;
            number = i,
            name = "bus_$i",
            available = true,
            bustype = t,
            angle = 0.0,
            magnitude = 1.0,
            voltage_limits = (min = 0.0, max = 2.0),
            base_voltage = 230.0,
        ) for (i, t) in enumerate(bus_types)
    ]
    for b in buses
        PSY.add_component!(sys, b)
    end
    PSY.add_component!(
        sys,
        PSY.Source(;
            name = "source_1",
            available = true,
            bus = buses[1],
            active_power = 0.0,
            reactive_power = 0.0,
            R_th = 1e-5,
            X_th = 1e-5,
        ),
    )
    PSY.add_component!(
        sys,
        PSY.ThermalStandard(;
            name = "thermal_2",
            available = true,
            status = true,
            bus = buses[2],
            active_power = 0.2,
            reactive_power = 0.0,
            rating = 1.0,
            active_power_limits = (min = 0.0, max = 1.0),
            reactive_power_limits = (min = -1.0, max = 1.0),
            ramp_limits = nothing,
            operation_cost = PSY.ThermalGenerationCost(nothing),
            base_power = 100.0,
            time_limits = nothing,
            prime_mover_type = PSY.PrimeMovers.OT,
            fuel = PSY.ThermalFuels.OTHER,
        ),
    )
    for (i, p, q) in ((3, 10.0, 5.0), (4, 8.0, 3.0))
        PSY.add_component!(
            sys,
            PSY.PowerLoad(;
                name = "load_$i",
                available = true,
                bus = buses[i],
                active_power = p,
                reactive_power = q,
                base_power = 1.0,
                max_active_power = 100.0,
                max_reactive_power = 100.0,
            ),
        )
    end
    PSY.add_component!(
        sys,
        PSY.StandardLoad(;
            name = "zip_3",
            available = true,
            bus = buses[3],
            base_power = 10.0,
            constant_active_power = 0.1,
            constant_reactive_power = 0.05,
            current_active_power = 0.05,
            current_reactive_power = 0.02,
            impedance_active_power = 0.05,
            impedance_reactive_power = 0.02,
            max_constant_active_power = 0.0,
            max_constant_reactive_power = 0.0,
            max_impedance_active_power = 0.0,
            max_impedance_reactive_power = 0.0,
            max_current_active_power = 0.0,
            max_current_reactive_power = 0.0,
        ),
    )
    line_params = ((1, 2, 0.10), (2, 3, 0.20), (3, 4, 0.05), (1, 4, 0.25))
    for (f, t, x) in line_params
        PSY.add_component!(
            sys,
            PSY.Line(;
                name = "line_$(f)_$(t)",
                available = true,
                active_power_flow = 0.0,
                reactive_power_flow = 0.0,
                arc = PSY.Arc(; from = buses[f], to = buses[t]),
                r = 1e-3,
                x = x,
                b = (from = 0.01, to = 0.01),
                rating = 1.0,
                angle_limits = (min = -pi / 2, max = pi / 2),
            ),
        )
    end
    return sys
end

PrecompileTools.@setup_workload begin
    # The component constructors log informational range checks; keep precompile quiet.
    sys = Logging.with_logger(Logging.NullLogger()) do
        _precompilation_workload_system()
    end
    PrecompileTools.@compile_workload begin
        Logging.with_logger(Logging.NullLogger()) do
            # First-call latency is dominated by the one-shot public entry
            # `solve_power_flow(pf, sys)` (setup + solve-to-convergence + DataFrames
            # results), so every solver we care about gets that call. NR and TR — the
            # coupled solvers POM drives through the in-place PSI-stable surface — ALSO get
            # the `solve_power_flow!` path, capped at two iterations: enough to compile the
            # step/refinement/Jacobian-refresh machinery, the OVERWRITE_NON_CONVERGED
            # writeback, and (after a reset, so it factors real numbers not NaNs) the
            # PolarNRCache refresh/reuse hot path used across Q-limit retries and time steps.

            # Polar Newton-Raphson — the default, by far the most common solver.
            pf = ACPowerFlow()
            data = PowerFlowData(pf, sys)
            solve_power_flow!(data; maxIterations = 2)
            clear_injection_data!(data)
            solve_power_flow!(data; maxIterations = 2)
            solve_power_flow(pf, sys)

            # Trust region — POM's robust fallback. Same two entries as NR.
            pf_tr = ACPowerFlow{TrustRegionACPowerFlow}()
            data_tr = PowerFlowData(pf_tr, sys)
            solve_power_flow!(data_tr; maxIterations = 2)
            clear_injection_data!(data_tr)
            solve_power_flow!(data_tr; maxIterations = 2)
            solve_power_flow(pf_tr, sys)

            # Fast-decoupled — the variant axis (two distinct hot paths): FDDecoupled/XB, the
            # bare polar default (B′/B″ half-iterations), and FDFixedJacobian (frozen full
            # Jacobian). The one-shot converged solve warms the public entry AND, by running
            # to convergence, both the factor-once FastDecoupledCache build and its
            # per-iteration reuse. The FDSchemeBX permutation is left out on purpose: rarely
            # used, and its separate assembly path would add build cost for little gain.
            for fd_solver in (FastDecoupledACPowerFlow, FastDecoupledFixed)
                solve_power_flow(ACPowerFlow{fd_solver}(), sys)
            end

            # Rectangular current-injection formulation (Da Costa), Newton-Raphson. Its
            # residual/Jacobian/setup kernels are rectangular-specific — separate compiled
            # code from polar — so a capped 2-iteration in-place solve is what warms them.
            pf_rect = ACRectangularPowerFlow{NewtonRaphsonACPowerFlow}()
            data_rect = PowerFlowData(pf_rect, sys)
            solve_power_flow!(data_rect; maxIterations = 2)

            # DC ABA direct solve (non-iterative, so no maxIterations and no NaN-writeback
            # reset). Two in-place solves on the same data warm the DCSolverCache build AND
            # its reuse branch — the multi-period / PCM-loop hot path that reuses the cached
            # factorization; the one-shot then warms the DC DataFrames results path. PTDF is
            # excluded on purpose — uncommon, and its factorization would add build cost
            # without meaningfully lowering first-call latency.
            dc = DCPowerFlow()
            data_dc = PowerFlowData(dc, sys)
            solve_power_flow!(data_dc)
            solve_power_flow!(data_dc)
            solve_power_flow(dc, sys)
        end
    end
end
