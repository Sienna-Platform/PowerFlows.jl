module PowerFlowsTests

using ReTest
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
include("test_utils/legacy_pf.jl")
include("test_utils/validate_reduced_power_flow.jl")

const AC_SOLVERS_TO_TEST = (
    LUACPowerFlow,
    NewtonRaphsonACPowerFlow,
    TrustRegionACPowerFlow,
    LevenbergMarquardtACPowerFlow,
    RobustHomotopyPowerFlow,
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
            @test length(IS.get_log_events(multi_logger.tracker, Logging.Error)) == 0
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
