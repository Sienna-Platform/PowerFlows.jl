const MAX_INIT_RESIDUAL = 10.0
const BOUNDS_TOLERANCE = 1e-6
const INFINITE_BOUND = 1e6 # used as default when a branch has rating 0.0, as implied by the PSSE Manual
const MAX_REACTIVE_POWER_ITERATIONS = 10

# Discrete control device λ-continuation outer loop.
# Pass budget PER STEEPNESS STAGE (each of the ~7 stages settles independently; one
# global budget let a slow early stage starve the stiff later ones).
const MAX_CONTROL_PASSES_PER_STAGE = 20
const CONTROL_PARAM_TOL = 1e-5
# Relative floor of the settle tolerance: tol_d = max(CONTROL_PARAM_TOL, RTOL·range).
const CONTROL_PARAM_RTOL = 1e-4
# A device whose full parameter range moves the regulated quantity by less than this
# (|dy/dp|·(hi−lo), e.g. p.u. voltage) is ineffective: freeze it instead of letting the
# steep sigmoid slam it to a rail with no feedback (PV-pinned controlled buses probe 0).
const CONTROL_GAIN_FLOOR = 1e-4
const MIN_LAMBDA_STEP = 1e-3
const MAX_LAMBDA_STEP = 1.0
const CONTROL_STEP_GROWTH = 1.5
const INITIAL_CONTROL_STEEPNESS = 1.0e2   # paper eq.10: start (S−λ_S)≈100
const MAX_CONTROL_STEEPNESS = 5.0e3       # paper: full S≈5000
const CONTROL_STEEPNESS_GROWTH = 2.0
const CONTROL_OSCILLATION_LIMIT = 3
# Tap-control defaults used on PSY 5.x, where `TapTransformer` has no first-class control
# fields (they arrive with PSY #1705 / the psy6 branch, whose field defaults these match).
const DEFAULT_TAP_RATIO_MIN = 0.9
const DEFAULT_TAP_RATIO_MAX = 1.1
const DEFAULT_TAP_POSITIONS = 33
const DEFAULT_TAP_VSET = 1.0
const DEFAULT_SHUNT_MODSW = 1   # discrete voltage control when ext lacks "MODSW"
# Voltage setpoints outside this band are treated as data errors and lock the device
# (PSY's `admittance_limits` carries the PSS/E VSWLO/VSWHI *voltage* band only by parser
# convention; an API-built component holding actual admittance bounds there would
# otherwise silently drive |V| toward a garbage setpoint).
const CONTROL_VSET_MIN = 0.5
const CONTROL_VSET_MAX = 1.5
const DEFAULT_MAX_REDISTRIBUTION_ITERATIONS = 10
const LARGE_RESIDUAL = 10 # threshold for "bad initial guess": default
# norm(residual, 1)/length(residual) > 10.

const ISAPPROX_ZERO_TOLERANCE = 1e-6

const V_FLOOR2 = 1e-16 # lower bound on |V|² (e²+f²) to guard 1/D in rectangular/MCPB current balance

const LCC_sinϕ_TOLERANCE = 1e-8 # if sin(ϕ) < this, treat dQ/dV as zero to avoid singularity in Jacobian

const DEFAULT_NR_MAX_ITER = 50 # default maxIterations for the NR power flow
const DEFAULT_NR_TOL = 1e-9 # default tolerance for the NR power flow
const DEFAULT_REFINEMENT_THRESHOLD = 5e-2 # do refinement if relative error > 5%.
const DEFAULT_REFINEMENT_MAX_ITER = 10 # how many times to try iterative refinement
const DEFAULT_REFINEMENT_EPS = 1e-6 # when to stop iterative refinement.
const NR_SINGULAR_SCALING = 1e-6 # scaling factor in fallback method for singular Jacobian

# Iwamoto step control constants
const IWAMOTO_MU_MIN = 0.0 # minimum step multiplier (zero permits "no step" when all damped steps worsen the residual)
const IWAMOTO_MU_MAX = 1.0 # maximum step multiplier
const IWAMOTO_DEGENERACY_TOL = 1e-30 # near-zero tolerance for degenerate cubic/quadratic
const IWAMOTO_MAX_REVERTS = 3 # consecutive reverted steps before early termination
# only used for trust region.
const DEFAULT_TRUST_REGION_ETA = 1e-4 # if actual improvement/predicted improvement
# is < eta, then reject the step Δx, shrink the trust region, and try again.
const DEFAULT_TRUST_REGION_FACTOR = 1.0 # controls starting size of trust region
# improvement factor cutoffs for updating size of trust region.
const HALVE_TRUST_REGION = 0.1
const MAX_DOUBLE_TRUST_REGION = 0.5
const DOUBLE_TRUST_REGION = 0.9
const DEFAULT_TRUST_REGION_DELTA_MAX_FACTOR = 10.0 # δ_max = factor * δ_0 (Nocedal & Wright §4.1)
const DEFAULT_AUTOSCALE = false # correct for scaling of the system
# typically converges in fewer iteration without autoscaling.
const DEFAULT_IWAMOTO_FALLBACK = true # when a trust region step is rejected, try Iwamoto damping

const PF_MAX_LOG = 10
# only used for Levenberg-Maquardt
const DEFAULT_λ_0 = 1e-5
# Upper bound on the LM damping factor μ. μ only grows on rejected steps, so
# hitting this cap is a divergence signal — the solver aborts with an error.
const DEFAULT_μ_MAX = 1e8

# Fast/Fixed Decoupled Newton-Raphson (FDNR) defaults.
const DEFAULT_FD_MAX_ITER = 150 # FD-stage iteration cap (linear rate needs more, cheaper iterations than NR's 50)
const DEFAULT_FD_HANDOFF_TOL = 1e-3 # FD-stage exit ∞-norm when handing off (≈0.1 MW/MVAr on 100 MVA base)
const DEFAULT_FD_SCHEME = :XB # B′/B″ scheme: :XB (Stott–Alsac) default; :BX (van Amerongen)
const DEFAULT_FD_REFREEZE_ON_STALL = true # :fixed_jacobian only: refactor frozen J once on stall, then continue
const DEFAULT_FD_NON_DIVERGENT = true # non-divergent backtracking; on by default (pure-FD default mode)
const DEFAULT_FD_NDVFCT = 0.99 # non-divergent improvement factor (accept a half-step only if it reduces the mismatch)
const DEFAULT_FD_MAX_STEP_HALVINGS = 10 # ≤10 inner mismatch calculations (step factor down to ~0.002)
const DEFAULT_FD_BLOWUP = 5.0 # largest unscaled per-half-step |Δθ| (rad) / |ΔV/V| abort threshold
const DEFAULT_FD_DVLIM = 0.99 # uniform ΔV-vector scale-down so largest applied |ΔV| ≤ this, + ΔV/V ≤ −1 guard
const DEFAULT_FD_VM_ABORT = 0.01 # abort if any bus |V| driven to ≈0
"""
Warn-once threshold for branch reactance `|x|` in the fast/fixed-decoupled methods.
"""
const FD_LOW_REACTANCE_WARNING = 1e-3 # warn: a branch |x| (pu) below this makes B′/B″ ill-conditioned ⇒ slow :decoupled convergence

const DEFAULT_Δt_k = 0.2

const AC_PF_KW = []

const BUS_VOLTAGE_MAGNITUDE_CUTOFF_MIN = 0.8
const BUS_VOLTAGE_MAGNITUDE_CUTOFF_MAX = 1.2

const TIs = Union{Int32, Int64}
# Int64 on Apple so AppleAccelerate's libSparse (Int64-only `columnStarts` ABI)
# can factor the AC Jacobian natively; Int32 elsewhere (KLU only).
# `@static` resolves the platform branch at lowering time so `INDEX_TYPE` const-folds
# to a single concrete type per build (verifiable with `code_lowered`).
const INDEX_TYPE = @static if Sys.isapple()
    Int64
else
    Int32
end
const J_INDEX_TYPE = INDEX_TYPE
const REC_INDEX_TYPE = INDEX_TYPE

# LCC line-commutated-converter scaling factor: the fundamental component of the
# AC-side current per unit DC current is `(√6/π)·t·I_dc`. Used in `lcc_utils.jl`,
# `ac_power_flow_residual.jl`, `ac_power_flow_jacobian.jl`, and the rectangular
# CI counterparts.
const SQRT6_DIV_PI = sqrt(6) / π

# voltage validation
const DEFAULT_VALIDATE_VOLTAGES = true
const MinMax = NamedTuple{(:min, :max), Tuple{Float64, Float64}}
const DEFAULT_VALIDATION_RANGE = (min = 0.5, max = 1.5)
# const MAX_INDS_TO_PRINT = 10

const FACTS_MODE_MAP = Dict(
    PSY.FACTSOperationModes.OOS => 0,
    PSY.FACTSOperationModes.NML => 1,
    PSY.FACTSOperationModes.BYP => 2,
)

const OVERWRITE_NON_CONVERGED = true # overwrite non-converged time steps with NaN values

# Adam / gradient descent power flow
const ADAM_BACKTRACK_FACTOR = 0.5
const ADAM_MAX_BACKTRACKS = 10

# robust homotopy method constants
const β = 10.0^-3
const INSUFFICIENT_CHANGE_IN_X = 10^(-11)
const GRAD_ZERO = 2 * eps()
# cholesky solver specific
const VTypes = SparseArrays.CHOLMOD.VRealTypes
const ITypes = SparseArrays.CHOLMOD.ITypes

# force arc names to be unique when reporting power flow results.
const FORCE_UNIQUE_NAMES = true

const BUS_TYPE_PRIORITIES = Dict{PSY.ACBusTypes, Int}(
    PSY.ACBusTypes.REF => 3,
    PSY.ACBusTypes.PV => 2,
    PSY.ACBusTypes.PQ => 1,
)

const PSSE_DEFAULT_EXPORT_NAME = "export"
const PSSE_EXPORT_SUPPORTED_VERSIONS = [:v33, :v35]

IS.@scoped_enum(
    FlowReporting,
    ARC_FLOWS = 0,
    BRANCH_FLOWS = 1,
)
@doc "
  FlowReporting

  Enumeration describing the type of flows reported in power flow results.

  Values
  - ARC_FLOWS = 0: Report total flows corresponding to arcs.
  - BRANCH_FLOWS = 1: Report flows for individual branches.
 " FlowReporting
