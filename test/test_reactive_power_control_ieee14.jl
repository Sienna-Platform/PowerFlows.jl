# Reference values below are transcribed from a university power-systems assignment
# (Cañizares, ECE 662 Project 1: Power Flow & Short Circuits — IEEE 14-bus system), part 1.c:
# all P/Q loads increased 40%, solved with generator reactive-power limits enforced. See
# scratchwork/prototyping/reactive-power-tests/reactive-power-devices-solution.md for the
# full transcription (part c, "Table 1: Bus voltage results for different placements of the
# capacitor bank").
#
# The reference solver enforces Q-limits by hand-converting each candidate bus to a PV bus at
# 1.0 pu (their "treat as PV" hint); here that maps onto our `FACTSControlDevice`. It reports
# the required capacitor sizes as 24 MVar (Bus 14) and 18.35 MVar (Bus 13, "clamps to 18 or
# 19" once discretized into blocks) -- our own network reproduces the base voltage profile
# very closely (see the "baseline" testset), but the required-MVar figure is a more sensitive
# derived quantity, so its match is looser than the voltage match.

const IEEE14_NO_CAP_V = Dict(  # Table 1, "No Capacitor" column
    4 => 0.9344,
    7 => 0.9480,
    8 => 0.9907,
    9 => 0.9206,
    10 => 0.9167,
    14 => 0.8959,
    6 => 0.9657,
    11 => 0.9353,
    12 => 0.9404,
    13 => 0.9311,
)
const IEEE14_CAP14_V = Dict(  # Table 1, "Capacitor Placement in Bus 14" column
    4 => 0.9631,
    7 => 0.9983,
    8 => 1.039,
    9 => 0.9827,
    10 => 0.9787,
    14 => 1.00,
    6 => 1.0230,
    11 => 0.9953,
    12 => 1.0044,
    13 => 1.00,
)
const IEEE14_CAP13_V = Dict(  # Table 1, "Capacitor Placement in Bus 13" column
    4 => 0.9559,
    7 => 0.9816,
    8 => 1.023,
    9 => 0.9606,
    10 => 0.9595,
    14 => 0.9502,
    6 => 1.0187,
    11 => 0.9834,
    12 => 1.0025,
    13 => 1.00,
)
const IEEE14_CAP_BUS14_MVAR = 24.0
const IEEE14_CAP_BUS13_MVAR = 18.35

# Generators/synchronous condensers the reference reports as pinned at Q_max ("(H)") once
# Q-limits are enforced under the 40% load increase -- Buses 6 and 8 are the synchronous
# condensers (P_gen = 0), Buses 2 and 3 are the two PV generators. The slack (Bus 1) is never
# Q-limit-checked at all (`_check_q_limit_bounds!` only inspects `ACBusTypes.PV` buses), so it's
# intentionally excluded here.
const IEEE14_PV_QMAX_MVAR = Dict(2 => 50.0, 3 => 40.0, 6 => 24.0, 8 => 24.0)

# Tolerances shared across multiple testsets below (each used in >1 spot; genuine one-offs are
# left as inline literals).
const IEEE14_BASELINE_TABLE_ATOL = 0.01  # system-wide voltage match, no device present
const IEEE14_ELSEWHERE_TABLE_ATOL = 0.02  # system-wide voltage match, with a device present
const FACTS_BUS_VOLTAGE_ATOL = 5e-3  # tight: continuous FACTS device, at its own bus
const SHUNT_BUS_VOLTAGE_ATOL = 0.01  # one discrete step looser: switched shunt, at its own bus
const BUS14_MVAR_ATOL = 3.0  # delivered MVar vs reference's 24 MVar (Bus 14 matches closely)
const BUS13_MVAR_RTOL = 0.25  # coarse plain-MVar comparison (Bus 13 is sensitivity-amplified)
const BLOCK_INTEGER_ATOL = 1e-6  # switched-shunt susceptance snaps to an integer block count

_ieee14_solve(sys, pf) = begin
    data = PowerFlowData(pf, sys)
    converged = solve_power_flow!(data)
    (data, converged)
end

_bus_v(data, bus_lookup, bus_number) = data.bus_magnitude[bus_lookup[bus_number], 1]

# reactive power from a control device, in NU: b * |V|^2 * sys_base_power.
_shunt_mvar(dev, data, bus_lookup, bus_number, base_power) =
    dev.current * _bus_v(data, bus_lookup, bus_number)^2 * base_power

"""Voltage at `bus_number` with NO control device installed (Q-limits still enforced), used
below as the baseline half of a local dV/dQ sensitivity estimate."""
function _ieee14_uncontrolled_voltage(bus_number::Int)
    sys = _make_ieee14_scaled_load_system(1.4)
    pf = ACPolarPowerFlow(; check_reactive_power_limits = true)
    data, converged = _ieee14_solve(sys, pf)
    @assert converged
    return _bus_v(data, PF.get_bus_lookup(data), bus_number)
end

"""Predict the MVar the reference solution would need at `bus_number`, using OUR solver's own
local dV/dQ sensitivity there (measured from `v_baseline_ours -> v_final_ours` for `q_mvar_ours`
MVar delivered), applied to the REFERENCE's reported (larger) voltage deficit
`1.0 - v_baseline_ref`.

This exists because Bus 13's required-MVar figure is a much more sensitive derived quantity
than voltage: our own uncontrolled baseline voltage there differs from the reference's
reported baseline by only ~0.003 pu, but Bus 13 has a steep local dV/dQ, so that small
baseline gap alone accounts for most of the difference between our measured MVar and the
reference's reported MVar. This way, we can keep a tighter tolerance."""
function _predict_reference_mvar(
    bus_number::Int,
    v_baseline_ours,
    v_final_ours,
    q_mvar_ours,
)
    dVdQ = (v_final_ours - v_baseline_ours) / q_mvar_ours
    v_baseline_ref = IEEE14_NO_CAP_V[bus_number]
    return (1.0 - v_baseline_ref) / dVdQ
end

@testset "reactive power control: IEEE-14 heavy-load baseline (no device)" begin
    sys = _make_ieee14_scaled_load_system(1.4)
    pf = ACPolarPowerFlow(; check_reactive_power_limits = true)
    data, converged = _ieee14_solve(sys, pf)
    @test converged
    bus_lookup = PF.get_bus_lookup(data)

    # without any control device, Bus 14 sags below 0.9 pu.
    @test _bus_v(data, bus_lookup, 14) < 0.9

    # check our voltage numbers match the reference.
    for (bus, vref) in IEEE14_NO_CAP_V
        @test isapprox(
            _bus_v(data, bus_lookup, bus),
            vref;
            atol = IEEE14_BASELINE_TABLE_ATOL,
        )
    end

    # check that we get the same buses PV -> PQ switched as the reference.
    base = get_base_power(sys)
    for (bus, qmax) in IEEE14_PV_QMAX_MVAR
        ix = bus_lookup[bus]
        @test data.bus_type[ix, 1] == ACBusTypes.PQ
        @test isapprox(data.bus_reactive_power_injections[ix, 1] * base, qmax; atol = 1e-3)
    end
    @test data.bus_type[bus_lookup[1], 1] == ACBusTypes.REF
end

@testset "reactive power control: FACTS device at Bus 14 -> V=1.0" begin
    sys = _make_ieee14_scaled_load_system(1.4)
    _add_facts_shunt!(sys, 14; voltage_setpoint = 1.0)
    pf = ACPolarPowerFlow(;
        check_reactive_power_limits = true,
        control_discrete_devices = true,
    )
    data, converged = _ieee14_solve(sys, pf)
    @test converged
    bus_lookup = PF.get_bus_lookup(data)

    # very tight voltage tolerance at the controlled bus; loose tolerance on MVar.
    @test isapprox(_bus_v(data, bus_lookup, 14), 1.0; atol = FACTS_BUS_VOLTAGE_ATOL)
    facts = only(data.controlled_devices.facts)
    q_mvar = _shunt_mvar(facts, data, bus_lookup, 14, get_base_power(sys))
    @test isapprox(q_mvar, IEEE14_CAP_BUS14_MVAR; atol = BUS14_MVAR_ATOL)

    # moderate voltage tolerance elsewhere
    for (bus, vref) in IEEE14_CAP14_V
        @test isapprox(
            _bus_v(data, bus_lookup, bus),
            vref;
            atol = IEEE14_ELSEWHERE_TABLE_ATOL,
        )
    end
end

@testset "reactive power control: switched shunt at Bus 14 -> V=1.0, clamped MVar" begin
    sys = _make_ieee14_scaled_load_system(1.4)
    _add_switched_shunt!(sys, 14; voltage_setpoint = 1.0)
    pf = ACPolarPowerFlow(;
        check_reactive_power_limits = true,
        control_discrete_devices = true,
    )
    data, converged = _ieee14_solve(sys, pf)
    @test converged
    bus_lookup = PF.get_bus_lookup(data)

    # One discrete (1 MVar) step of tolerance on top of the FACTS device's continuous
    # tolerance.
    @test isapprox(_bus_v(data, bus_lookup, 14), 1.0; atol = SHUNT_BUS_VOLTAGE_ATOL)

    shunt = only(data.controlled_devices.shunts)
    # The switched susceptance itself lands on an integer number of 1-MVar blocks (the
    # "clamps to a discrete value" behavior) -- note this is `b * base_power`, not the
    # delivered Q = b * |V|^2 * base_power, since V isn't exactly 1.0 p.u.
    blocks_mvar = shunt.current * get_base_power(sys)
    @test isapprox(blocks_mvar, round(blocks_mvar); atol = BLOCK_INTEGER_ATOL)
    q_mvar = _shunt_mvar(shunt, data, bus_lookup, 14, get_base_power(sys))
    # Delivered MVar should still land close to the reference's continuous 24 MVar requirement.
    @test isapprox(q_mvar, IEEE14_CAP_BUS14_MVAR; atol = BUS14_MVAR_ATOL)
end

@testset "reactive power control: FACTS device at Bus 13 -> V=1.0" begin
    sys = _make_ieee14_scaled_load_system(1.4)
    _add_facts_shunt!(sys, 13; voltage_setpoint = 1.0)
    pf = ACPolarPowerFlow(;
        check_reactive_power_limits = true,
        control_discrete_devices = true,
    )
    data, converged = _ieee14_solve(sys, pf)
    @test converged
    bus_lookup = PF.get_bus_lookup(data)

    @test isapprox(_bus_v(data, bus_lookup, 13), 1.0; atol = FACTS_BUS_VOLTAGE_ATOL)
    facts = only(data.controlled_devices.facts)
    q_mvar = _shunt_mvar(facts, data, bus_lookup, 13, get_base_power(sys))

    # Coarse comparison on plain MVar.
    @test isapprox(q_mvar, IEEE14_CAP_BUS13_MVAR; rtol = BUS13_MVAR_RTOL)

    # Finer comparison on sensitivity-corrected value (one-off tolerance: this check only
    # appears once, since the switched-shunt version below rounds to a block instead).
    v_baseline_ours = _ieee14_uncontrolled_voltage(13)
    predicted_ref_mvar =
        _predict_reference_mvar(13, v_baseline_ours, _bus_v(data, bus_lookup, 13), q_mvar)
    @test isapprox(predicted_ref_mvar, IEEE14_CAP_BUS13_MVAR; atol = 1.0)

    for (bus, vref) in IEEE14_CAP13_V
        @test isapprox(
            _bus_v(data, bus_lookup, bus),
            vref;
            atol = IEEE14_ELSEWHERE_TABLE_ATOL,
        )
    end
end

@testset "reactive power control: switched shunt at Bus 13 -> clamps near 18-19 MVar" begin
    sys = _make_ieee14_scaled_load_system(1.4)
    _add_switched_shunt!(sys, 13; voltage_setpoint = 1.0)
    pf = ACPolarPowerFlow(;
        check_reactive_power_limits = true,
        control_discrete_devices = true,
    )
    data, converged = _ieee14_solve(sys, pf)
    @test converged
    bus_lookup = PF.get_bus_lookup(data)

    @test isapprox(_bus_v(data, bus_lookup, 13), 1.0; atol = SHUNT_BUS_VOLTAGE_ATOL)

    shunt = only(data.controlled_devices.shunts)
    blocks_mvar = shunt.current * get_base_power(sys)
    @test isapprox(blocks_mvar, round(blocks_mvar); atol = BLOCK_INTEGER_ATOL)
    q_mvar = _shunt_mvar(shunt, data, bus_lookup, 13, get_base_power(sys))
    # Coarse comparison on plain MVar.
    @test isapprox(q_mvar, IEEE14_CAP_BUS13_MVAR; rtol = BUS13_MVAR_RTOL)

    # Same sensitivity correction here can't be as tight because shunt clamps to nearest block.
    v_baseline_ours = _ieee14_uncontrolled_voltage(13)
    predicted_ref_mvar =
        _predict_reference_mvar(13, v_baseline_ours, _bus_v(data, bus_lookup, 13), q_mvar)
    @test round(predicted_ref_mvar) in (18.0, 19.0)
end
