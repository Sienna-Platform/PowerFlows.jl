abstract type AbstractControlledDevice end
abstract type AbstractBranchControl <: AbstractControlledDevice end
abstract type AbstractShuntControl <: AbstractControlledDevice end

"""Voltage-controlling tap transformer. `nz_offsets` are the 4 cached
`nzval` linear indices of the (from,to)×(from,to) Y-bus block."""
mutable struct ControlledTap <: AbstractBranchControl
    name::String
    from_ix::Int
    to_ix::Int
    controlled_ix::Int
    controlled_on_primary::Bool      # true → eq.46, false → eq.47
    vset::Float64
    yt::ComplexF64                   # 1/(r+jx)
    y_shunt::ComplexF64              # primary shunt
    alpha::Float64                   # phase-shift angle (0 for TapTransformer)
    p_min::Float64
    p_max::Float64
    levels::Vector{Float64}          # discrete tap ratios
    nz_offsets::NTuple{4, Int}       # nzval idx for Y11,Y12,Y21,Y22
    current::Float64
end

"""Voltage-controlling switched shunt, snapped block-greedily."""
mutable struct ControlledSwitchedShunt <: AbstractShuntControl
    name::String
    bus_ix::Int
    controlled_ix::Int
    vset::Float64                    # current point target; set to the nearest violated
    # band edge (or held at the base voltage, frozen) once
    # per solve by `_apply_shunt_deadband!` — see PSS/E's
    # VSWLO/VSWHI semantics: switch the minimum number of
    # steps to re-enter the band, not to a fixed setpoint.
    vset_lo::Float64                 # PSS/E VSWLO
    vset_hi::Float64                 # PSS/E VSWHI
    g0::Float64                      # real(get_Y)
    b0::Float64                      # imag(get_Y)
    block_steps::Vector{Int}         # number_of_steps per block
    block_dB::Vector{Float64}        # imag(Y_increase) per block
    b_min::Float64
    b_max::Float64
    block_order::Vector{Int}         # sortperm(block_dB; rev=true), cached at construction
    block_n::Vector{Int}             # per-block chosen step counts, reused in-place each snap
    continuous::Bool                 # MODSW==2 ⇒ continuous regulation (no discrete snap)
    current::Float64                 # current total susceptance b
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
    current::Float64                 # current susceptance b
end

struct ControlledDeviceSet
    taps::Vector{ControlledTap}
    shunts::Vector{ControlledSwitchedShunt}
    facts::Vector{ControlledFACTS}
    phase_shifters::Vector{ControlledPhaseShifter}
end
Base.isempty(s::ControlledDeviceSet) =
    isempty(s.taps) && isempty(s.shunts) && isempty(s.facts) &&
    isempty(s.phase_shifters)

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
set_current_parameter!(d::ControlledTap, p::Float64) = (d.current = p; nothing)
set_current_parameter!(d::ControlledSwitchedShunt, p::Float64) =
    (d.current = p; nothing)
set_current_parameter!(d::ControlledFACTS, p::Float64) = (d.current = p; nothing)
set_current_parameter!(d::ControlledPhaseShifter, p::Float64) =
    (d.current = p; nothing)

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
    return nothing
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
    return nothing
end

# Delta-update (`+=`, not `=`): `_get_withdrawals!` accumulates all constant-Z devices on
# this bus into one slot, so overwriting would drop co-located contributions. Only
# susceptance is controlled; g0 is constant and stays in the baseline.
function apply_parameter!(d::ControlledSwitchedShunt, data, b::Float64, ts::Int)
    data.bus_reactive_power_constant_impedance_withdrawals[d.bus_ix, ts] +=
        d.current - b
    d.current = b
    return nothing
end

# Same constant-Z reactive-withdrawal delta as the switched shunt: raising the (capacitive)
# susceptance lowers the bus's reactive withdrawal, injecting Q and raising the voltage.
function apply_parameter!(d::ControlledFACTS, data, b::Float64, ts::Int)
    data.bus_reactive_power_constant_impedance_withdrawals[d.bus_ix, ts] +=
        d.current - b
    d.current = b
    return nothing
end

apply_parameter!(d::ControlledPhaseShifter, args...) = _seam_err(d)

@inline function _sigmoid(lo::Float64, hi::Float64, S::Float64,
    x::Float64, xset::Float64)
    return (hi - lo) / (1.0 + exp(S * (x - xset))) + lo
end

# Branch (tap): controlled-on-primary uses eq.46 (lo=tr_min,hi=tr_max);
# controlled-on-secondary uses eq.47 (limits swapped).
function target_from_voltage(d::ControlledTap, vmag::Float64, S::Float64)
    lo, hi = d.controlled_on_primary ? (d.p_min, d.p_max) : (d.p_max, d.p_min)
    return clamp(_sigmoid(lo, hi, S, vmag, d.vset), d.p_min, d.p_max)
end

# Shunt: eq.9, low V → high B. x→-∞ gives hi=b_max; x→+∞ gives lo=b_min.
function target_from_voltage(d::ControlledSwitchedShunt, vmag::Float64,
    S::Float64)
    b = _sigmoid(d.b_min, d.b_max, S, vmag, d.vset)
    return clamp(b, d.b_min, d.b_max)
end

function snap_to_discrete(d::ControlledTap, p::Float64)
    pc = clamp(p, d.p_min, d.p_max)
    best = d.levels[1]
    @inbounds for lv in d.levels
        abs(lv - pc) < abs(best - pc) && (best = lv)
    end
    return best
end

# Block-greedy (floor): largest blocks first, take as many steps as fit without
# overshooting; ±1 bounded refinement then corrects any under-committed block.
# block_order and block_n are pre-allocated fields — no per-call heap allocation.
function snap_to_discrete(d::ControlledSwitchedShunt, b::Float64)
    d.continuous && return clamp(b, d.b_min, d.b_max)   # continuous: no grid snap
    target_clamped = clamp(b, d.b_min, d.b_max)
    target = target_clamped - d.b0
    total = d.b0
    # Greedy pass: floor to avoid overshooting.
    @inbounds for k in d.block_order
        dB = d.block_dB[k]
        dB == 0.0 && (d.block_n[k] = 0; continue)
        n = clamp(floor(Int, (target - (total - d.b0)) / dB),
            0, d.block_steps[k])
        d.block_n[k] = n
        total += n * dB
    end
    # ±1 bounded refinement: single pass, in-place update of total and block_n.
    @inbounds for k in d.block_order
        dB = d.block_dB[k]
        dB == 0.0 && continue
        n = d.block_n[k]
        for δ in (-1, 1)
            n_new = n + δ
            (n_new < 0 || n_new > d.block_steps[k]) && continue
            candidate = total + δ * dB
            if abs(candidate - target_clamped) < abs(total - target_clamped)
                d.block_n[k] = n_new
                total = candidate
                n = n_new
            end
        end
    end
    return clamp(total, d.b_min, d.b_max)
end

# Continuous devices: no discrete grid, just clamp into the band.
snap_to_discrete(d::ControlledFACTS, b::Float64) = clamp(b, d.b_min, d.b_max)
snap_to_discrete(d::ControlledPhaseShifter, alpha::Float64) =
    clamp(alpha, d.angle_min, d.angle_max)
