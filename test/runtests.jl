# Each test_*.jl runs in its own worker (ParallelTestRunner); files share nothing but
# `includes.jl`'s preamble.
#
#   julia --project=test test/runtests.jl                     # full suite, all jobs
#   julia --project=test test/runtests.jl test_dc_power_flow  # filter by FILE name (startswith)
#   julia --project=test test/runtests.jl --jobs=4            # cap parallelism
#   julia --project=test test/runtests.jl --list               # list discoverable tests

using PowerFlows
using ParallelTestRunner
import PowerSystemCaseBuilder as PSB

const TEST_DIR = @__DIR__

const DISABLED_TESTS = Set(String[])

# Each file's expression installs its own stray-error gate (see `with_stray_error_gate`
# in includes.jl) around the `include`, since ParallelTestRunner gives each test file its
# own worker rather than a suite-wide logger to share.
testsuite = Dict{String, Expr}(
    splitext(f)[1] => :(with_stray_error_gate(() -> include($(joinpath(TEST_DIR, f)))))
    for f in readdir(TEST_DIR) if
    startswith(f, "test_") && endswith(f, ".jl") && splitext(f)[1] ∉ DISABLED_TESTS
)

const INIT_CODE = :(include($(joinpath(TEST_DIR, "includes.jl"))))

# Worker-process env: PowerSystemCaseBuilder reads a shared serialized-system HDF5 store
# concurrently across workers — disable HDF5 file locking to avoid cross-process contention.
const WORKER_ENV = [
    "HDF5_USE_FILE_LOCKING" => "FALSE",
    "RUNNING_SIENNA_TESTS" => "true",
    "VECLIB_MAXIMUM_THREADS" => "1",
]

# A cold PSB cache means every worker misses `is_serialized` and they race to write the same
# bundle directory, which `PSY.to_file` does not do atomically. Running serially populates it
# safely; once warm this costs nothing. An explicit `--jobs` wins, because a second `--jobs`
# in ARGS would survive `extract_flag!` and then be read as a test-name filter.
function _psb_cache_is_cold()
    if !isdir(PSB.SERIALIZED_DIR)
        return true
    end
    return isempty(readdir(PSB.SERIALIZED_DIR))
end

if _psb_cache_is_cold() && !any(startswith("--jobs"), ARGS)
    @info "PowerSystemCaseBuilder cache is empty; building it serially before testing."
    push!(ARGS, "--jobs=1")
end

runtests(PowerFlows, ARGS; testsuite, init_code = INIT_CODE, env = WORKER_ENV)
