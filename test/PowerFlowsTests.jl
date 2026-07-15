module PowerFlowsTests

using ReTest
import Test  # for Test.TestLogger (ReTest re-exports macros but not the module)
using PowerFlows
using Logging
using Dates
using PowerSystems
using PowerSystemCaseBuilder
using PowerNetworkMatrices
using InfrastructureSystems
using LinearAlgebra
using CSV
using DataFrames
using JSON3
using InteractiveUtils
using DataStructures
import SparseArrays
import SparseArrays: SparseMatrixCSC, sparse, sprandn, sprand
import Random
import PROPACK

import Aqua
Aqua.test_unbound_args(PowerFlows)
Aqua.test_undefined_exports(PowerFlows)
Aqua.test_ambiguities(PowerFlows)
Aqua.test_stale_deps(PowerFlows)
Aqua.test_deps_compat(PowerFlows)

import InfrastructureSystems as IS
import PowerSystemCaseBuilder as PSB
import PowerSystems as PSY
import PowerNetworkMatrices as PNM
import PowerFlows as PF

# used to be public, no longer: import here so we can use in tests
import PowerFlows: PowerFlowData
import PowerFlows: ACPowerFlowData, PTDFPowerFlowData, vPTDFPowerFlowData, ABAPowerFlowData
import PowerFlows: solve_power_flow!, write_results

const BASE_DIR = dirname(dirname(Base.find_package("PowerFlows")))
const TEST_DATA_DIR = joinpath(
    dirname(dirname(Base.find_package("PowerFlows"))),
    "test",
    "test_data",
)
const DIFF_INF_TOLERANCE = 1e-4
const DIFF_L2_TOLERANCE = 1e-3
const TIGHT_TOLERANCE = 1e-7

const LOG_FILE = "power-flows.log"

# [include test utils here]
include("test_utils/common.jl")
include("test_utils/psse_results_compare.jl")
include("test_utils/penalty_factors_brute_force.jl")
include("test_utils/validate_reduced_power_flow.jl")
include("test_utils/jacobian_verification.jl")

const AC_SOLVERS_TO_TEST = (
    NewtonRaphsonACPowerFlow,
    TrustRegionACPowerFlow,
    LevenbergMarquardtACPowerFlow,
    RobustHomotopyPowerFlow,
    FastDecoupledACPowerFlow,
)

for filename in readdir(joinpath(BASE_DIR, "test"))
    if startswith(filename, "test_") && endswith(filename, ".jl")
        include(filename)
    end
end

function get_logging_level_from_env(env_name::String, default)
    level = get(ENV, env_name, default)
    return IS.get_logging_level(level)
end

# Expected-@error allowlist for the stray-error gate: the area-interchange greedy-relax path
# logs an infeasible-schedule Error BY DESIGN (_ac_power_flow_with_area_relax!), and those
# already-@test_logs-asserted events still reach this global tracker under the full-suite
# ReTest schedule, so the gate must exclude exactly them.
const _AREA_RELAX_ERROR_MARKER = "Area interchange:"
const _CONVERGENCE_FAILURE_MARKER = "solver failed to converge after"

_is_area_relax_error(event) = occursin(_AREA_RELAX_ERROR_MARKER, event.message)
_is_convergence_failure_error(event) =
    occursin(_CONVERGENCE_FAILURE_MARKER, event.message)

"""Error-level log events the stray-error gate should fail on: everything except the
area-interchange greedy-relax sequence (and the pre-relax convergence failure it causes, which
is excused only when a relax actually happened)."""
function unexpected_error_events(tracker)
    events = IS.get_log_events(tracker, Logging.Error)
    saw_relax = any(_is_area_relax_error, events)
    unexpected = Vector{eltype(events)}()
    for event in events
        if _is_area_relax_error(event)
            continue
        end
        if saw_relax && _is_convergence_failure_error(event)
            continue
        end
        push!(unexpected, event)
    end
    return unexpected
end

# See also `load_tests.jl` for running tests interactively with ReTest.jl
function run_tests(args...; kwargs...)
    logger = global_logger()
    try
        logging_config_filename = get(ENV, "SIIP_LOGGING_CONFIG", nothing)
        if logging_config_filename !== nothing
            config = IS.LoggingConfiguration(logging_config_filename)
        else
            config = IS.LoggingConfiguration(;
                filename = LOG_FILE,
                file_level = get_logging_level_from_env("SIENNA_FILE_LOG_LEVEL", "Info"),
                console_level = get_logging_level_from_env(
                    "SIENNA_CONSOLE_LOG_LEVEL",
                    "Error",
                ),
            )
        end
        console_logger = Logging.ConsoleLogger(config.console_stream, config.console_level)

        IS.open_file_logger(config.filename, config.file_level) do file_logger
            levels = (Logging.Info, Logging.Warn, Logging.Error)
            multi_logger =
                IS.MultiLogger([console_logger, file_logger], IS.LogEventTracker(levels))
            Logging.global_logger(multi_logger)

            if !isempty(config.group_levels)
                IS.set_group_levels!(multi_logger, config.group_levels)
            end

            @time retest(args...; kwargs...)
            unexpected = unexpected_error_events(multi_logger.tracker)
            # Name the offenders: a bare count gives no way to find which site tripped the gate.
            for event in unexpected
                @warn "Unexpected error-level log event" event.file event.line event.count event.message
            end
            @test isempty(unexpected)
            @info IS.report_log_summary(multi_logger)
        end
    finally
        # Guarantee that the global logger is reset.
        global_logger(logger)
        nothing
    end
end

export run_tests

end
