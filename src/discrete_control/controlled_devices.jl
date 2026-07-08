abstract type AbstractControlledDevice end
abstract type AbstractBranchControl <: AbstractControlledDevice end
abstract type AbstractShuntControl <: AbstractControlledDevice end

"""Voltage-controlling tap transformer. `nz_offsets` are the 4 cached
`nzval` linear indices of the (from,to)×(from,to) Y-bus block. The control
orientation is NOT stored: it comes from the measured plant sensitivity dV/dp
(see `_control_target`), which is correct for any wiring of the controlled bus."""
mutable struct ControlledTap <: AbstractBranchControl
    name::String
    from_ix::Int
    to_ix::Int
    controlled_ix::Int
    vset::Float64
    yt::ComplexF64                   # 1/(r+jx)
    alpha::Float64                   # winding-group phase shift (PSY.get_α)
    p_min::Float64
    p_max::Float64
    levels::Vector{Float64}          # discrete tap ratios
    nz_offsets::NTuple{4, Int}       # nzval idx for Y11,Y12,Y21,Y22
    initial::Float64                 # enrollment-time tap (reporting)
    synced::Float64                  # tap reflected in the arc-admittance rows
    current::Float64
end

"""Voltage-controlling switched shunt, snapped onto the PSS/E cumulative
block-activation chain (blocks switch on in listed order, off in reverse)."""
mutable struct ControlledSwitchedShunt <: AbstractShuntControl
    name::String
    bus_ix::Int
    controlled_ix::Int
    vset::Float64
    vset_lo::Float64                 # VSWLO/VSWHI deadband: held anywhere inside
    vset_hi::Float64
    g0::Float64                      # real(get_Y)
    b0::Float64                      # fixed (non-switchable) susceptance base
    block_steps::Vector{Int}         # number_of_steps per block
    block_dB::Vector{Float64}        # imag(Y_increase) per block
    b_min::Float64
    b_max::Float64
    block_n::Vector{Int}             # per-block step counts of the last snap (reporting)
    continuous::Bool                 # MODSW==2 ⇒ continuous regulation (no discrete snap)
    initial::Float64                 # enrollment-time susceptance (reporting)
    current::Float64                 # current total susceptance b
    psse_convention::Bool            # true ⇒ PSS/E parser convention (Y=BINIT total,
    # initial_status zeroed); false ⇒ PSY API convention (Y is the fixed base,
    # initial_status meaningful). Determines how write_device_settings! sources Y/status.
end

"""Continuous phase-angle regulator (PAR) on a `PhaseShiftingTransformer` (ACTIVE_POWER_FLOW
control). Drives the branch's own active-power flow (from→to) toward `p_target` by varying the
phase-shift angle `α ∈ [angle_min, angle_max]` at fixed tap ratio. `apply_parameter!` mutates the
Y-bus off-diagonals (Y11 = yt/|t|² is invariant under a pure phase change), so — like a tap — a
step invalidates a fast-decoupled B″ factorization."""
mutable struct ControlledPhaseShifter <: AbstractBranchControl
    name::String
    from_ix::Int
    to_ix::Int
    p_target::Float64                # active-power flow setpoint (p.u., from→to)
    yt::ComplexF64                   # series admittance 1/(r+jx)
    tap::Float64                     # fixed tap ratio (magnitude)
    angle_min::Float64               # phase-angle band (radians)
    angle_max::Float64
    nz_offsets::NTuple{4, Int}       # nzval idx for Y11,Y12,Y21,Y22
    initial::Float64                 # enrollment-time angle (reporting)
    synced::Float64                  # angle reflected in the arc-admittance rows
    current::Float64                 # current phase angle α (radians)
end

"""Continuous shunt SVC/STATCOM (`FACTSControlDevice`). Holds `vset` at its bus by varying a
shunt susceptance `b ∈ [b_min, b_max]` (negative = inductive, positive = capacitive); the
injected reactive power is `b·|V|²`. Applied through the constant-Z reactive-withdrawal slot
(never the Y-bus), so a step does not invalidate a fast-decoupled B″ factorization. At a
susceptance limit the clamp holds it there — the homotopy equivalent of the PV→PQ Q-limit
release."""
mutable struct ControlledFACTS <: AbstractShuntControl
    name::String
    bus_ix::Int
    controlled_ix::Int
    vset::Float64
    b_min::Float64                   # max inductive susceptance (≤ 0)
    b_max::Float64                   # max capacitive susceptance (≥ 0)
    initial::Float64                 # enrollment-time susceptance (reporting)
    current::Float64                 # current susceptance b
end

struct ControlledDeviceSet
    taps::Vector{ControlledTap}
    shunts::Vector{ControlledSwitchedShunt}
    facts::Vector{ControlledFACTS}
    phase_shifters::Vector{ControlledPhaseShifter}
    # Perf counters for the last `_control_continuation!` (counts, not wall-clock; the regression
    # harness). symbolic_factors stays O(1) per continuation with PolarNRCache reuse. See getters.
    inner_solves::Base.RefValue{Int}
    symbolic_factors::Base.RefValue{Int}
    numeric_refactors::Base.RefValue{Int}
end
function ControlledDeviceSet(
    taps::Vector{ControlledTap},
    shunts::Vector{ControlledSwitchedShunt},
    facts::Vector{ControlledFACTS},
    phase_shifters::Vector{ControlledPhaseShifter},
)
    return ControlledDeviceSet(
        taps, shunts, facts, phase_shifters, Ref(0), Ref(0), Ref(0))
end
function Base.isempty(s::ControlledDeviceSet)
    return isempty(s.taps) && isempty(s.shunts) && isempty(s.facts) &&
           isempty(s.phase_shifters)
end

"""Number of inner `_solve_with_q_limits!` calls the last discrete-control continuation
performed (0 when the data was built without discrete control)."""
function get_control_inner_solve_count(data)
    isnothing(data.controlled_devices) && return 0
    return data.controlled_devices.inner_solves[]
end

"""Number of KLU/AA SYMBOLIC factorizations performed inside the last discrete-control
continuation. With `PolarNRCache` symbolic reuse this stays O(1) per continuation even as
`inner_solves` grows; without it, it tracks `inner_solves`. 0 when built without control."""
function get_control_symbolic_factor_count(data)
    isnothing(data.controlled_devices) && return 0
    return data.controlled_devices.symbolic_factors[]
end

# Hot-path instrumentation. The `nothing` (non-control) case — every ordinary AC solve — is a
# single branch and early return, mirroring `_ctrl_solve!`. Only the continuation path counts.
@inline function _count_symbolic_factor!(data)
    cd = data.controlled_devices
    isnothing(cd) || (cd.symbolic_factors[] += 1)
    return
end
@inline function _count_numeric_refactor!(data)
    cd = data.controlled_devices
    isnothing(cd) || (cd.numeric_refactors[] += 1)
    return
end

"""Number of per-NR-iteration NUMERIC refactorizations performed inside the last
discrete-control continuation. 0 when the data was built without discrete control."""
function get_control_numeric_refactor_count(data)
    isnothing(data.controlled_devices) && return 0
    return data.controlled_devices.numeric_refactors[]
end

controlled_bus_ix(d::ControlledTap) = d.controlled_ix
controlled_bus_ix(d::ControlledSwitchedShunt) = d.controlled_ix
controlled_bus_ix(d::ControlledFACTS) = d.controlled_ix
voltage_setpoint(d::ControlledTap) = d.vset
voltage_setpoint(d::ControlledSwitchedShunt) = d.vset
voltage_setpoint(d::ControlledFACTS) = d.vset
parameter_limits(d::ControlledTap) = (d.p_min, d.p_max)
parameter_limits(d::ControlledSwitchedShunt) = (d.b_min, d.b_max)
parameter_limits(d::ControlledFACTS) = (d.b_min, d.b_max)
parameter_limits(d::ControlledPhaseShifter) = (d.angle_min, d.angle_max)
current_parameter(d::ControlledTap) = d.current
current_parameter(d::ControlledSwitchedShunt) = d.current
current_parameter(d::ControlledFACTS) = d.current
current_parameter(d::ControlledPhaseShifter) = d.current

# The continuation drives `measured_value(d, data, ts)` toward `control_setpoint(d)`. Voltage
# devices read the controlled-bus magnitude and target their `vset`; the phase shifter reads its
# own active-power flow and targets `p_target`. This dispatched pair is the only quantity-specific
# seam — the rest of the continuation engine is agnostic to what is being regulated.
measured_value(d::ControlledTap, data, ts::Int) =
    data.bus_magnitude[controlled_bus_ix(d), ts]
measured_value(d::ControlledSwitchedShunt, data, ts::Int) =
    data.bus_magnitude[controlled_bus_ix(d), ts]
measured_value(d::ControlledFACTS, data, ts::Int) =
    data.bus_magnitude[controlled_bus_ix(d), ts]
control_setpoint(d::ControlledTap) = voltage_setpoint(d)
control_setpoint(d::ControlledSwitchedShunt) = voltage_setpoint(d)
control_setpoint(d::ControlledFACTS) = voltage_setpoint(d)
control_setpoint(d::ControlledPhaseShifter) = d.p_target

# Active-power flow from→to on the PST from the converged bus state and the complex tap
# t = ratio·e^{iα}: P = Re(V_f·conj(I_f)), I_f = (yt/|t|²)·V_f − (yt/conj(t))·V_t.
function measured_value(d::ControlledPhaseShifter, data, ts::Int)
    Vf = data.bus_magnitude[d.from_ix, ts] * cis(data.bus_angles[d.from_ix, ts])
    Vt = data.bus_magnitude[d.to_ix, ts] * cis(data.bus_angles[d.to_ix, ts])
    t = d.tap * cis(d.current)
    If = (d.yt / abs2(t)) * Vf - (d.yt / conj(t)) * Vt
    return real(Vf * conj(If))
end

# Seam: future implicit embedding dispatches here. Never called by the outer loop.
stamp_control!(d::AbstractControlledDevice, args...) =
    error("implicit embedding not implemented for $(typeof(d))")

# PSS/E deadband semantics: a switched shunt is held while the controlled voltage is
# anywhere INSIDE [VSWLO, VSWHI]; only excursions outside the band trigger switching.
# Other device families carry a point setpoint (no parsed band) and always regulate.
_in_deadband(::AbstractControlledDevice, ::Float64) = false
_in_deadband(d::ControlledSwitchedShunt, y::Float64) = d.vset_lo <= y <= d.vset_hi

function _nz_index(A::SparseArrays.SparseMatrixCSC, row::Int, col::Int)
    @inbounds for k in SparseArrays.nzrange(A, col)
        A.rowval[k] == row && return k
    end
    error("Ybus has no stored entry at ($row,$col); structural zero")
end

function _ybus_block_offsets(ybus, i::Int, j::Int)
    A = ybus.data
    return (
        _nz_index(A, i, i),
        _nz_index(A, i, j),
        _nz_index(A, j, i),
        _nz_index(A, j, j),
    )
end

# Y11/Y12/Y21 are delta-updated (new−old) so parallel branches' contributions
# in the shared nzval slots are preserved; Y22=Yt is tap-independent (zero delta,
# even with parallels) so it is skipped. The nzval is single-precision ComplexF32;
# correctness of the running sum relies on `old_tap` coming from `d.current`
# (the last applied value), never read back from the lossy nzval.
function apply_parameter!(d::ControlledTap, data, p::Float64, ::Int)
    A = data.power_network_matrix.data
    old_tap = d.current * cis(d.alpha)
    new_tap = p * cis(d.alpha)
    o = d.nz_offsets
    @inbounds begin
        A.nzval[o[1]] += d.yt / abs2(new_tap) - d.yt / abs2(old_tap)
        A.nzval[o[2]] += -d.yt / conj(new_tap) - (-d.yt / conj(old_tap))
        A.nzval[o[3]] += -d.yt / new_tap - (-d.yt / old_tap)
        # Y22 = Yt is tap-independent; no update needed.
    end
    d.current = p
    return
end

# Pure phase change at fixed |t| = tap: Y11 = yt/|t|² and Y22 = yt are invariant, so only the
# off-diagonals rotate. Delta-update (`+=`) preserves any parallel branch sharing the nzval slot;
# `old` comes from `d.current` (the last applied angle), never read back from the lossy nzval.
function apply_parameter!(d::ControlledPhaseShifter, data, alpha::Float64, ::Int)
    A = data.power_network_matrix.data
    old_t = d.tap * cis(d.current)
    new_t = d.tap * cis(alpha)
    o = d.nz_offsets
    @inbounds begin
        A.nzval[o[2]] += -d.yt / conj(new_t) - (-d.yt / conj(old_t))
        A.nzval[o[3]] += -d.yt / new_t - (-d.yt / old_t)
    end
    d.current = alpha
    return
end

# Delta-update (`+=`, not `=`): `_get_withdrawals!` accumulates all constant-Z devices on
# this bus into one slot, so overwriting would drop co-located contributions. Only
# susceptance is controlled; g0 is constant and stays in the baseline.
function apply_parameter!(d::ControlledSwitchedShunt, data, b::Float64, ts::Int)
    data.bus_reactive_power_constant_impedance_withdrawals[d.bus_ix, ts] +=
        d.current - b
    d.current = b
    return
end

# Same constant-Z reactive-withdrawal delta as the switched shunt: raising the (capacitive)
# susceptance lowers the bus's reactive withdrawal, injecting Q and raising the voltage.
function apply_parameter!(d::ControlledFACTS, data, b::Float64, ts::Int)
    data.bus_reactive_power_constant_impedance_withdrawals[d.bus_ix, ts] +=
        d.current - b
    d.current = b
    return
end

@inline function _sigmoid(lo::Float64, hi::Float64, S::Float64,
    x::Float64, xset::Float64)
    return (hi - lo) / (1.0 + exp(S * (x - xset))) + lo
end

function snap_to_discrete(d::ControlledTap, p::Float64)
    pc = clamp(p, d.p_min, d.p_max)
    best = d.levels[1]
    @inbounds for lv in d.levels
        abs(lv - pc) < abs(best - pc) && (best = lv)
    end
    return best
end

# PSS/E mixed banks: capacitor blocks (dB>0) switch on cumulatively in listed order,
# reactor blocks (dB<0) likewise — two independent chains stepping away from the
# all-off base, NOT one serial chain, so a mixed bank reaches both signs. Realizable
# totals = b0 ∪ {b0 + capacitor prefixes} ∪ {b0 + reactor prefixes}. Same-sign banks
# reduce to the previous single-chain walk. O(Σ steps), allocation-free.
function snap_to_discrete(d::ControlledSwitchedShunt, b::Float64)
    d.continuous && return clamp(b, d.b_min, d.b_max)
    target = clamp(b, d.b_min, d.b_max)
    best = d.b0
    best_steps = 0
    best_positive = true
    @inbounds for positive in (true, false)
        total = d.b0
        steps_taken = 0
        for k in eachindex(d.block_steps, d.block_dB)
            dB = d.block_dB[k]
            on_side = if positive
                dB > 0.0
            else
                dB < 0.0
            end
            on_side || continue
            for _ in 1:d.block_steps[k]
                total += dB
                steps_taken += 1
                if abs(total - target) < abs(best - target)
                    best = total
                    best_steps = steps_taken
                    best_positive = positive
                end
            end
        end
    end
    # Record the winning side's prefix in block_n; the other side's blocks are 0.
    fill!(d.block_n, 0)
    remaining = best_steps
    @inbounds for k in eachindex(d.block_steps)
        dB = d.block_dB[k]
        on_winning_side = if best_positive
            dB > 0.0
        else
            dB < 0.0
        end
        if on_winning_side
            n = min(remaining, d.block_steps[k])
            d.block_n[k] = n
            remaining -= n
        end
    end
    return best
end

# Continuous devices: no discrete grid, just clamp into the band.
snap_to_discrete(d::ControlledFACTS, b::Float64) = clamp(b, d.b_min, d.b_max)
snap_to_discrete(d::ControlledPhaseShifter, alpha::Float64) =
    clamp(alpha, d.angle_min, d.angle_max)

# The parameter-dependent from-side arc-admittance terms (ff, ft, tf) of a branch device;
# the tt term (= yt) is parameter-independent for both device types.
@inline function _branch_terms(d::ControlledTap, p::Float64)
    t = p * cis(d.alpha)
    return (d.yt / abs2(t), -d.yt / conj(t), -d.yt / t)
end
@inline function _branch_terms(d::ControlledPhaseShifter, alpha::Float64)
    t = d.tap * cis(alpha)
    return (d.yt / abs2(t), -d.yt / conj(t), -d.yt / t)
end

# Delta-update the from/to arc-admittance rows to `d.current` (parallel branches sharing the arc
# row are preserved; `d.synced` tracks the reflected value). Reported flows are computed from
# these matrices post-loop, so an unsynced branch would report flows at its original tap/angle.
function _sync_branch_arc_rows!(
    Yft::SparseArrays.SparseMatrixCSC,
    Ytf::SparseArrays.SparseMatrixCSC,
    d::Union{ControlledTap, ControlledPhaseShifter},
    arc_row::Dict{Tuple{Int, Int}, Int},
    ix_to_number::Dict{Int, Int},
)
    d.current == d.synced && return
    fb = ix_to_number[d.from_ix]
    tb = ix_to_number[d.to_ix]
    ff_new, ft_new, tf_new = _branch_terms(d, d.current)
    ff_old, ft_old, tf_old = _branch_terms(d, d.synced)
    r = get(arc_row, (fb, tb), 0)
    if !iszero(r)
        @inbounds begin
            Yft.nzval[_nz_index(Yft, r, d.from_ix)] += ff_new - ff_old
            Yft.nzval[_nz_index(Yft, r, d.to_ix)] += ft_new - ft_old
            Ytf.nzval[_nz_index(Ytf, r, d.from_ix)] += tf_new - tf_old
            # Ytf(r, to_ix) holds the tt term (= yt): parameter-independent, skipped.
        end
    else
        # The branch may be stored under the reversed arc orientation; the from/to roles
        # of the four terms swap accordingly (its "from side" is our to bus).
        r = get(arc_row, (tb, fb), 0)
        if iszero(r)
            @warn "discrete control: arc ($fb, $tb) of device \"$(d.name)\" not found \
                in the arc-admittance axes; reported flows on that branch reflect its \
                original parameter." maxlog = 1
            return
        end
        @inbounds begin
            Yft.nzval[_nz_index(Yft, r, d.from_ix)] += tf_new - tf_old
            Ytf.nzval[_nz_index(Ytf, r, d.to_ix)] += ft_new - ft_old
            Ytf.nzval[_nz_index(Ytf, r, d.from_ix)] += ff_new - ff_old
            # Yft(r, to_ix) is the reversed arc's from-side self term (= yt): skipped.
        end
    end
    d.synced = d.current
    return
end

"""One-shot post-continuation sync: bring the arc-admittance rows of every moved branch
device (taps, PARs) in line with its final parameter so the branch flows reported by
`solve_power_flow!` match the network the voltages were solved on. Shunt-side devices
never touch the arc matrices. No-op when the arc admittance matrices were not built."""
function _sync_arc_admittances!(data, set::ControlledDeviceSet)
    Yft = data.power_network_matrix.arc_admittance_from_to
    Ytf = data.power_network_matrix.arc_admittance_to_from
    (isnothing(Yft) || isnothing(Ytf)) && return
    isempty(set.taps) && isempty(set.phase_shifters) && return
    bus_lookup = PNM.get_bus_lookup(Yft)
    ix_to_number = Dict{Int, Int}(v => k for (k, v) in bus_lookup)
    arc_row = get_arc_lookup(data)   # (from_no, to_no) => arc row index
    for d in set.taps
        _sync_branch_arc_rows!(Yft.data, Ytf.data, d, arc_row, ix_to_number)
    end
    for d in set.phase_shifters
        _sync_branch_arc_rows!(Yft.data, Ytf.data, d, arc_row, ix_to_number)
    end
    return
end
