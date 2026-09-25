# Shared preamble evaluated into every test worker's sandbox module before the test file
# body runs.

using Test
using Logging
using Dates
using Random
using LinearAlgebra
using PowerFlows
using PowerSystems
using PowerSystemCaseBuilder
import PowerSystemCaseBuilder: system_from_openapi
const PFP = PowerSystemCaseBuilder.PowerFlowFileParser
using PowerNetworkMatrices
using InfrastructureSystems
using CSV
using DataFrames
using JSON3
using InteractiveUtils
using DataStructures
import SparseArrays
import SparseArrays: SparseMatrixCSC, sparse, sprandn, sprand
import Aqua

import InfrastructureSystems as IS
import PowerSystemCaseBuilder as PSB
import PowerSystems as PSY
import PowerNetworkMatrices as PNM
import PowerFlows as PF

# used to be public, no longer: import here so tests can use them
import PowerFlows: PowerFlowData
import PowerFlows: ACPowerFlowData, PTDFPowerFlowData, vPTDFPowerFlowData, ABAPowerFlowData
import PowerFlows: solve_power_flow!, write_results

const BASE_DIR = dirname(dirname(Base.find_package("PowerFlows")))
const TEST_DATA_DIR = joinpath(BASE_DIR, "test", "test_data")
const DIFF_INF_TOLERANCE = 1e-4
const DIFF_L2_TOLERANCE = 1e-3
const TIGHT_TOLERANCE = 1e-7

# Keep each worker's captured console output to real problems; the runner echoes it back.
Logging.global_logger(Logging.ConsoleLogger(stderr, Logging.Error))

include("test_utils/common.jl")
include("test_utils/psse_results_compare.jl")
include("test_utils/penalty_factors_brute_force.jl")
include("test_utils/validate_reduced_power_flow.jl")
include("test_utils/jacobian_verification.jl")
include("test_utils/cross_file_fixtures.jl")

const AC_SOLVERS_TO_TEST = (
    NewtonRaphsonACPowerFlow,
    TrustRegionACPowerFlow,
    LevenbergMarquardtACPowerFlow,
    RobustHomotopyPowerFlow,
    FastDecoupledACPowerFlow,
)

# Expected-@error allowlist for the stray-error gate: the area-interchange greedy-relax path
# logs an infeasible-schedule Error BY DESIGN (_ac_power_flow_with_area_relax!).
const _AREA_RELAX_ERROR_MARKER = "Area interchange:"

_is_area_relax_error(event) = occursin(_AREA_RELAX_ERROR_MARKER, event.message)

"Error-level log events the stray-error gate should fail on: everything except the
area-interchange greedy-relax sequence."
function unexpected_error_events(tracker)
    events = IS.get_log_events(tracker, Logging.Error)
    return [event for event in events if !_is_area_relax_error(event)]
end

"Run `f` (one test file's `include`) under a log-event tracker and fail the enclosing
testset if it logs an unexpected Error-level event. Each worker runs exactly one test
file, so this is that file's stray-error gate; restores the previous global logger after."
function with_stray_error_gate(f::Function)
    previous_logger = global_logger()
    tracker = IS.LogEventTracker((Logging.Info, Logging.Warn, Logging.Error))
    console_logger = Logging.ConsoleLogger(stderr, Logging.Error)
    multi_logger = IS.MultiLogger([console_logger], tracker)
    Logging.global_logger(multi_logger)
    try
        f()
        unexpected = unexpected_error_events(tracker)
        for event in unexpected
            @warn "Unexpected error-level log event" event.file event.line event.count event.message
        end
        @test isempty(unexpected)
    finally
        Logging.global_logger(previous_logger)
    end
    return
end
