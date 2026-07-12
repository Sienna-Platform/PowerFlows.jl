# Task 6: polar residual for embedded area net-interchange control -- the tie-flow kernel
# (`_tie_metered_active_power`) and the tail writer (`_set_area_tail_residuals!`), plus the
# ΔP coupling into the controlled area's slack-bus P-balance row in
# `_update_residual_values!`. Residual-evaluation-only: the Jacobian (Task 7) doesn't exist
# yet, so no solver runs here.

# Locate the CSC offset of entry (i, j) in a sparse matrix -- mirrors the technique already
# used by the tie-enumeration tests to check `nz_offsets` against `A[f, t]`.
function _find_nz_offset(A::SparseMatrixCSC, i::Int, j::Int)
    for k in SparseArrays.nzrange(A, j)
        SparseArrays.rowvals(A)[k] == i && return k
    end
    error("no stored entry at ($i, $j)")
end

@testset "area interchange tie kernel matches hand-computed P_m" begin
    g11, b11 = 5.0, -12.0
    g12, b12 = -5.0, 12.0
    g21, b21 = -5.0, 12.0
    g22, b22 = 5.0, -12.0
    Y = sparse(
        [1, 2, 1, 2],
        [1, 1, 2, 2],
        PF.YBUS_ELTYPE[g11 + im * b11, g21 + im * b21, g12 + im * b12, g22 + im * b22],
    )
    ybus_nzval = SparseArrays.nonzeros(Y)
    o = (
        _find_nz_offset(Y, 1, 1),
        _find_nz_offset(Y, 1, 2),
        _find_nz_offset(Y, 2, 1),
        _find_nz_offset(Y, 2, 2),
    )

    Vf, θf, Vt, θt = 1.02, 0.05, 0.98, -0.03
    no_pollution = (0.0 + 0.0im, 0.0 + 0.0im)

    tie_from = PF.AreaTie(1, 2, o, true, 1, 2, no_pollution)
    expected_from = Vf^2 * g11 + Vf * Vt * (g12 * cos(θf - θt) + b12 * sin(θf - θt))
    @test PF._tie_metered_active_power(tie_from, Vf, θf, Vt, θt, ybus_nzval) ≈
          expected_from

    tie_to = PF.AreaTie(1, 2, o, false, 1, 2, no_pollution)
    expected_to = Vt^2 * g22 + Vt * Vf * (g21 * cos(θt - θf) + b21 * sin(θt - θf))
    @test PF._tie_metered_active_power(tie_to, Vf, θf, Vt, θt, ybus_nzval) ≈ expected_to
end

# Task 6b: the degree-1 fixture above cannot catch the diagonal-pollution bug (its Y-bus
# diagonal IS the tie's own primitive, with nothing else incident). This fixture puts an
# EXTRA branch and a shunt on bus 1 and an extra branch on bus 2, so the aggregate Y-bus
# diagonal at each tie endpoint is polluted with non-corridor contributions -- exactly the
# degree>1 shape that made the pre-fix kernel (which read `ybus_nzval[o[1]]`/`o[4]]` as if
# they WERE the tie's own self-admittance) wrong. `diag_pollution` is supplied by hand as
# `aggregate_diag - tie_own_primitive_diag`, mirroring what `_dedup_ties` computes at
# tie-build time; the expected P_m is computed from the TIE's OWN primitive only.
@testset "area interchange tie kernel recovers own primitive from a polluted diagonal" begin
    g11, b11 = 5.0, -12.0     # tie's own primitive (1<->2), reused from the test above
    g12, b12 = -5.0, 12.0
    g21, b21 = -5.0, 12.0
    g22, b22 = 5.0, -12.0
    extra_branch_bus1 = 2.0 - 4.0im    # non-member branch (1-3), own contribution at bus 1
    shunt_bus1 = 0.3 - 0.1im           # non-member shunt at bus 1
    extra_branch_bus2 = 1.5 - 3.0im    # non-member branch (2-4), own contribution at bus 2
    agg11 = (g11 + im * b11) + extra_branch_bus1 + shunt_bus1
    agg22 = (g22 + im * b22) + extra_branch_bus2
    Y = sparse(
        [1, 2, 1, 2],
        [1, 1, 2, 2],
        PF.YBUS_ELTYPE[agg11, g21 + im * b21, g12 + im * b12, agg22],
    )
    ybus_nzval = SparseArrays.nonzeros(Y)
    o = (
        _find_nz_offset(Y, 1, 1),
        _find_nz_offset(Y, 1, 2),
        _find_nz_offset(Y, 2, 1),
        _find_nz_offset(Y, 2, 2),
    )
    pollution_from = ComplexF64(extra_branch_bus1 + shunt_bus1)
    pollution_to = ComplexF64(extra_branch_bus2)

    Vf, θf, Vt, θt = 1.02, 0.05, 0.98, -0.03

    # atol (not the tight default rtol): `agg11`/`agg22` are stored into the hand-built
    # `Y::Vector{PF.YBUS_ELTYPE}` (ComplexF32), so `ybus_nzval[o[1]] - diag_pollution[1]`
    # carries the same ComplexF32 rounding production code does; the comparison is against
    # a Float64-exact `expected_from`/`expected_to`.
    tie_from = PF.AreaTie(1, 2, o, true, 1, 2, (pollution_from, pollution_to))
    expected_from = Vf^2 * g11 + Vf * Vt * (g12 * cos(θf - θt) + b12 * sin(θf - θt))
    @test PF._tie_metered_active_power(tie_from, Vf, θf, Vt, θt, ybus_nzval) ≈
          expected_from atol = 1e-6

    tie_to = PF.AreaTie(1, 2, o, false, 1, 2, (pollution_from, pollution_to))
    expected_to = Vt^2 * g22 + Vt * Vf * (g21 * cos(θt - θf) + b21 * sin(θt - θf))
    @test PF._tie_metered_active_power(tie_to, Vf, θf, Vt, θt, ybus_nzval) ≈
          expected_to atol = 1e-6
end

# Shared fixture: Area1 owns REF and is never enrolled (rule 3); Area2 (tail 1, pdes 0.3)
# and Area3 (tail 2, pdes 0.2) enroll. Ties: Trans1/Trans2/Trans3 touch the uncontrolled
# Area1 (one tail is 0); Line11/Line12/Line16 run entirely between the two controlled areas
# (both tails nonzero).
function _two_controlled_area_data()
    sys = _three_area_transfer_fixture(; slack_area3 = true)
    pf = ACPolarPowerFlow(; area_interchange_control = true)
    return PowerFlowData(pf, sys)
end

# (arc, primitive-Y-source) pairs to fold into the oracle sum for one branch. Dispatched
# (never `isa`/`<:`), independently re-deriving -- not calling -- tie_set.jl's `_tie_arcs`:
# `PSY.ACTransmission` contributes its own single arc/branch; `PSY.ThreeWindingTransformer`
# decomposes into up to 3 star-node windings (mirrors `_tie_arcs(::ThreeWindingTransformer)`).
_oracle_branch_windings(branch::PSY.ACTransmission) = ((PSY.get_arc(branch), branch),)

function _oracle_branch_windings(branch::PSY.ThreeWindingTransformer)
    windings = Tuple{PSY.Arc, PNM.ThreeWindingTransformerWinding}[]
    PSY.get_available_primary(branch) && push!(
        windings,
        (PSY.get_primary_star_arc(branch), PNM.ThreeWindingTransformerWinding(branch, 1)),
    )
    PSY.get_available_secondary(branch) && push!(
        windings,
        (PSY.get_secondary_star_arc(branch), PNM.ThreeWindingTransformerWinding(branch, 2)),
    )
    PSY.get_available_tertiary(branch) && push!(
        windings,
        (PSY.get_tertiary_star_arc(branch), PNM.ThreeWindingTransformerWinding(branch, 3)),
    )
    return windings
end

# Fold one (arc, primitive) pair's own Y11/Y12/Y21/Y22 into `sums` (a `(y11,y12,y21,y22)`
# tuple accumulator) if `arc` spans `tie`'s bus pair, oriented against the tie's own
# (from,to) via the same reversed-orientation check `_dedup_ties` uses. Not a tie member
# (neither orientation matches) -> `sums` unchanged.
function _oracle_accumulate(
    sums,
    bus_lookup,
    reverse_bus_search_map,
    tie::PF.AreaTie,
    arc,
    primitive_entry,
)
    fix = PF._resolve_bus_ix(
        bus_lookup, reverse_bus_search_map, PSY.get_number(PSY.get_from(arc)))
    tix = PF._resolve_bus_ix(
        bus_lookup, reverse_bus_search_map, PSY.get_number(PSY.get_to(arc)))
    (isnothing(fix) || isnothing(tix)) && return sums
    (y11, y12, y21, y22) = PNM.ybus_branch_entries(primitive_entry)
    (s11, s12, s21, s22) = sums
    fix == tie.from_bus_ix && tix == tie.to_bus_ix &&
        return (s11 + y11, s12 + y12, s21 + y21, s22 + y22)
    fix == tie.to_bus_ix && tix == tie.from_bus_ix &&
        return (s11 + y22, s12 + y21, s21 + y12, s22 + y11)
    return sums
end

# Independent oracle for one tie's metered-end active power: sums `PNM.ybus_branch_entries`
# over the ACTUAL PSY branches (and, for a `ThreeWindingTransformer`, its individual star-node
# windings) between the tie's bus pair (the corridor's own primitives), NOT the kernel's
# `ybus_nzval[o[1]]`/`o[4]` diagonal reads -- so it cannot share the diagonal-pollution bug
# (aggregate diagonal sums EVERY incident branch/shunt, not just this tie's own members).
# Mirrors `_segment_flow_entry`'s `S = V_from * conj(y11*V_from + y12*V_to)` power-flow
# formula, the same primitive source that post-processing's branch-flow reporting uses.
# Handles multiple corridor members (parallel branches, or multiple windings both touching the
# same bus pair) by summing their primitives via `_oracle_accumulate`. A SINGLE loop over
# `PSY.ACTransmission` (mirrors `build_area_ties`): `ThreeWindingTransformer <: ACTransmission`,
# so a `Transformer3W` is already enumerated here -- `_oracle_branch_windings` dispatches on
# its CONCRETE type to the 3-winding decomposition; a second, separate
# `get_available_components(ThreeWindingTransformer, sys)` loop would double-count it.
function _oracle_tie_metered_power(sys, data, tie::PF.AreaTie, time_step::Int)
    bus_lookup = PF.get_bus_lookup(data)
    nrd = PF.get_network_reduction_data(data)
    reverse_bus_search_map = PNM.get_reverse_bus_search_map(nrd)
    sums = (zero(ComplexF64), zero(ComplexF64), zero(ComplexF64), zero(ComplexF64))
    for branch in PSY.get_available_components(PSY.ACTransmission, sys)
        for (arc, primitive_entry) in _oracle_branch_windings(branch)
            sums = _oracle_accumulate(
                sums,
                bus_lookup,
                reverse_bus_search_map,
                tie,
                arc,
                primitive_entry,
            )
        end
    end
    (sum_y11, sum_y12, sum_y21, sum_y22) = sums
    Vm = view(data.bus_magnitude, :, time_step)
    θ = view(data.bus_angles, :, time_step)
    Vf = Vm[tie.from_bus_ix] * exp(im * θ[tie.from_bus_ix])
    Vt = Vm[tie.to_bus_ix] * exp(im * θ[tie.to_bus_ix])
    if tie.metered_from
        return real(Vf * conj(sum_y11 * Vf + sum_y12 * Vt))
    end
    return real(Vt * conj(sum_y21 * Vf + sum_y22 * Vt))
end

# Task 8: per-tail net interchange (tail_ix -> NI) computed from `ties` via the SAME
# independent branch-primitive oracle above, generalized to any tie vector (not just
# `data.area_interchange.ties`) -- reused by the E2E tests to verify post-solve targets and,
# for the 3-area case, to recompute the de-enrolled area's implicit NI from a fresh
# all-areas tie set built directly against `PF.build_area_ties`.
function _oracle_ni_by_tail(
    sys::PSY.System,
    data::PF.ACPowerFlowData,
    ties::Vector{PF.AreaTie},
    time_step::Int = 1,
)
    ni = Dict{Int, Float64}()
    for tie in ties
        P_m = _oracle_tie_metered_power(sys, data, tie, time_step)
        metered_tail = tie.from_area_tail
        other_tail = tie.to_area_tail
        if !tie.metered_from
            metered_tail = tie.to_area_tail
            other_tail = tie.from_area_tail
        end
        iszero(metered_tail) || (ni[metered_tail] = get(ni, metered_tail, 0.0) + P_m)
        iszero(other_tail) || (ni[other_tail] = get(ni, other_tail, 0.0) - P_m)
    end
    return ni
end

@testset "area interchange residual matches independent branch-primitive oracle NI" begin
    sys = _three_area_transfer_fixture(; slack_area3 = true)
    data = PowerFlowData(ACPolarPowerFlow(; area_interchange_control = true), sys)
    @test PF.n_controlled_areas(data) == 2

    residual = PF.ACPowerFlowResidual(data, 1)
    x0 = PF.calculate_x0(data, 1)
    residual(x0, 1)
    F = residual.Rv
    dcn = PF.get_dc_network(data)
    area_off = PF.area_tail_offset(data, dcn)

    ni = zeros(PF.n_controlled_areas(data))
    for tie in data.area_interchange.ties
        P_m = _oracle_tie_metered_power(sys, data, tie, 1)
        metered_tail = tie.from_area_tail
        other_tail = tie.to_area_tail
        if !tie.metered_from
            metered_tail = tie.to_area_tail
            other_tail = tie.from_area_tail
        end
        iszero(metered_tail) || (ni[metered_tail] += P_m)
        iszero(other_tail) || (ni[other_tail] -= P_m)
    end
    for area in data.area_interchange.areas
        @test F[area_off + area.tail_ix] ≈ ni[area.tail_ix] - area.pdes atol = 1e-6
    end
end

@testset "area interchange NI antisymmetry at an arbitrary state" begin
    data = _two_controlled_area_data()
    # Keep only ties whose BOTH endpoints are controlled areas (Line11/Line12/Line16),
    # so the tie-cancellation identity Σ_a NI_a = 0 holds exactly with no uncontrolled-side
    # leakage.
    filter!(
        t -> !iszero(t.from_area_tail) && !iszero(t.to_area_tail),
        data.area_interchange.ties,
    )
    @test !isempty(data.area_interchange.ties)

    residual = PF.ACPowerFlowResidual(data, 1)
    x0 = PF.calculate_x0(data, 1)
    # Arbitrary (non-solution, non-flat-start) state: the identity is structural (every
    # tie contributes +P_m to one tracked area and -P_m to the other), so it must hold at
    # ANY state, not only at a converged solution.
    x1 = x0 .+ 0.05 .* sin.(1:length(x0))
    residual(x1, 1)
    F = residual.Rv
    area_off = PF.area_tail_offset(data, PF.get_dc_network(data))

    total = sum(F[area_off + a.tail_ix] + a.pdes for a in data.area_interchange.areas)
    @test isapprox(total, 0.0; atol = 1e-10)
end

@testset "area interchange ΔP coupling shifts the slack bus P-mismatch row" begin
    data = _two_controlled_area_data()
    residual = PF.ACPowerFlowResidual(data, 1)
    x0 = PF.calculate_x0(data, 1)
    dcn = PF.get_dc_network(data)
    area_off = PF.area_tail_offset(data, dcn)
    area = first(data.area_interchange.areas)
    slack_ix = area.slack_bus_ix

    residual(x0, 1)
    F_base = copy(residual.Rv)

    ΔP = 0.037
    x1 = copy(x0)
    x1[area_off + area.tail_ix] = ΔP
    residual(x1, 1)
    F_pert = residual.Rv

    # Derived convention: ΔP_a is added to P_net[slack_bus_ix] at the same seam as the
    # distributed-slack P_slack term, and F[2ix-1] accumulates the Ybus-calculated
    # injection MINUS P_net[ix]. So increasing P_net by ΔP DECREASES (shifts by -ΔP) the
    # active-power mismatch row at the area's slack bus.
    @test F_pert[2 * slack_ix - 1] - F_base[2 * slack_ix - 1] ≈ -ΔP atol = 1e-10
    # Vm/θ are identical between x0 and x1 (only the tail entry changed), so every OTHER
    # row -- including this area's own NI residual row, which depends on tie-endpoint
    # Vm/θ, not on ΔP -- is unaffected.
    @test F_pert[area_off + area.tail_ix] ≈ F_base[area_off + area.tail_ix] atol = 1e-10
    for ix in eachindex(F_base)
        ix == 2 * slack_ix - 1 && continue
        @test F_pert[ix] ≈ F_base[ix] atol = 1e-10
    end
end

@testset "area interchange tail writer allocation-free" begin
    data = _two_controlled_area_data()
    residual = PF.ACPowerFlowResidual(data, 1)
    x0 = PF.calculate_x0(data, 1)
    residual(x0, 1)   # warm: populate data.bus_magnitude/bus_angles, JIT compile
    dcn = PF.get_dc_network(data)
    area_off = PF.area_tail_offset(data, dcn)
    F = copy(residual.Rv)

    PF._set_area_tail_residuals!(F, x0, data, area_off, 1)   # warm
    @test (@allocated PF._set_area_tail_residuals!(F, x0, data, area_off, 1)) == 0
end

@testset "area interchange control-off polar solve is unaffected" begin
    sys = PSB.build_system(PSB.PSITestSystems, "c_sys14"; add_forecasts = false)
    data = PowerFlowData(ACPowerFlow(), sys)
    @test PF.n_controlled_areas(data) == 0
    @test solve_power_flow!(data)
end

# Task 6b guard: `AreaTie.diag_pollution` is a constant cached at tie-build time, so a
# SwitchedAdmittance sitting at a tie endpoint (not a corridor member -- shunts attach at one
# bus, never between a tie's bus pair) can silently invalidate it if switched post-
# enrollment. Bus 6 is the endpoint of exactly one tie (Trans2, 5-6), so exactly one warning
# is expected -- picked over the degree-4 Bus 9 to keep this an exact `@test_logs` match.
@testset "area interchange diag pollution guard warns for endpoint SwitchedAdmittance" begin
    sys = _three_area_transfer_fixture(; slack_area3 = true)
    _add_switched_shunt!(sys, 6)
    pf = ACPolarPowerFlow(; area_interchange_control = true)
    data = @test_logs(
        (:warn, r"SwitchedAdmittance \"shunt_6\": sits at a tie endpoint"),
        min_level = Logging.Warn,
        PowerFlowData(pf, sys)
    )
    @test PF.n_controlled_areas(data) == 2
end

# Task 7: `ACJacobianStructureCache`'s key now includes `area_data` BY IDENTITY (spec §5.4).
# A Q-limit flip / repeated solve on the SAME `PowerFlowData` keeps the same
# `AreaInterchangeData` object, so the structure is built once and reused; a freshly
# constructed `PowerFlowData` enrolls a NEW `AreaInterchangeData` object even against the
# identical system, forcing a rebuild.
@testset "area interchange Jacobian structure cache reuse" begin
    data1 = _two_controlled_area_data()
    residual1a = PF.ACPowerFlowResidual(data1, 1)
    PF.ACPowerFlowJacobian(residual1a, 1)
    cache1 = data1.ac_jacobian_structure_cache[]
    @test !isnothing(cache1)
    @test cache1.area_data === data1.area_interchange

    # Second construction on the SAME data -> reused (identical cache object, not rebuilt).
    residual1b = PF.ACPowerFlowResidual(data1, 1)
    PF.ACPowerFlowJacobian(residual1b, 1)
    @test data1.ac_jacobian_structure_cache[] === cache1

    # A freshly constructed PowerFlowData (even from the same fixture) enrolls a fresh
    # AreaInterchangeData object -> cache key misses -> rebuilt.
    data2 = _two_controlled_area_data()
    @test data2.area_interchange !== data1.area_interchange
    residual2 = PF.ACPowerFlowResidual(data2, 1)
    PF.ACPowerFlowJacobian(residual2, 1)
    cache2 = data2.ac_jacobian_structure_cache[]
    @test cache2.area_data === data2.area_interchange
    @test cache2.area_data !== cache1.area_data
end

# Task 7 (carried Important from 6b): a boundary-crossing THREE-WINDING transformer winding,
# whose star bus's Y-bus diagonal is polluted by BOTH a sibling winding of the SAME
# transformer (the primary, sharing the star bus) and an unrelated extra Line -- neither is a
# member of the boundary-crossing winding's own corridor. Topology (single island, REF at
# Bus1):
#   Bus1 (REF, AreaA) -- Line -- Bus3 (AreaA, primary terminal)
#   Bus3 --[primary winding]-- StarBus (AreaA)  (same area as Bus3: NOT a tie)
#   StarBus --[secondary winding]-- Bus4 (AreaB)  (crosses the boundary: THE tie)
#   StarBus -- Line -- Bus5 (AreaA)  (extra branch, pollutes the star bus's diagonal further)
#   Bus2 (SLACK, AreaB) -- Line -- Bus4
# Tertiary winding disabled (`available_tertiary = false`) -- not needed to exercise the
# boundary-crossing/polluted-diagonal path.
function _make_3w_boundary_fixture()
    sys = System(100.0)
    area_a = PSY.Area(; name = "AreaA")
    area_b = PSY.Area(; name = "AreaB")
    PSY.add_component!(sys, area_a)
    PSY.add_component!(sys, area_b)

    bus1 = _add_simple_bus!(sys, 1, ACBusTypes.REF, 230)
    bus2 = _add_simple_bus!(sys, 2, ACBusTypes.PV, 230)
    bus3 = _add_simple_bus!(sys, 3, ACBusTypes.PQ, 230)
    bus4 = _add_simple_bus!(sys, 4, ACBusTypes.PQ, 230)
    bus5 = _add_simple_bus!(sys, 5, ACBusTypes.PQ, 230)
    PSY.set_area!(bus1, area_a)
    PSY.set_area!(bus2, area_b)
    PSY.set_area!(bus3, area_a)
    PSY.set_area!(bus4, area_b)
    PSY.set_area!(bus5, area_a)

    _add_simple_source!(sys, bus1, 0.0, 0.0)
    _add_simple_thermal_standard!(sys, bus2, 0.1, 0.0)
    _add_simple_load!(sys, bus3, 5.0, 2.0)
    _add_simple_load!(sys, bus4, 5.0, 2.0)
    _add_simple_load!(sys, bus5, 2.0, 1.0)

    _add_simple_line!(sys, bus1, bus3)
    _add_simple_line!(sys, bus2, bus4)

    xfmr = _add_simple_transformer_3w!(sys, bus3, bus4, bus3, 99)
    star_bus = PSY.get_star_bus(xfmr)
    PSY.set_area!(star_bus, area_a)
    _add_simple_line!(sys, star_bus, bus5)

    PSY.set_bustype!(bus2, ACBusTypes.SLACK)
    return sys
end

@testset "area interchange 3W winding NI matches independent oracle (polluted star-bus diagonal)" begin
    sys = _make_3w_boundary_fixture()
    pf = PF.ACPowerFlow{NewtonRaphsonACPowerFlow}(;
        correct_bustypes = true,
        area_interchange_control = true,
    )
    data = PF.PowerFlowData(pf, sys)
    @test PF.n_controlled_areas(data) == 1

    # Exactly one tie: the 3W transformer's SECONDARY winding (star bus <-> Bus4) -- the only
    # winding whose terminal area differs from the star bus's area.
    @test length(data.area_interchange.ties) == 1
    tie = only(data.area_interchange.ties)

    residual = PF.ACPowerFlowResidual(data, 1)
    x0 = PF.calculate_x0(data, 1)
    Random.seed!(7)
    x0 .+= 0.02 .* randn(length(x0))
    residual(x0, 1)   # updates data.bus_magnitude/bus_angles in place

    expected = _oracle_tie_metered_power(sys, data, tie, 1)
    actual = PF._tie_metered_active_power(
        tie,
        data.bus_magnitude[tie.from_bus_ix, 1], data.bus_angles[tie.from_bus_ix, 1],
        data.bus_magnitude[tie.to_bus_ix, 1], data.bus_angles[tie.to_bus_ix, 1],
        SparseArrays.nonzeros(data.power_network_matrix.data),
    )
    @test actual ≈ expected atol = 1e-6
end

# Task 8: end-to-end NR/TR solves + Q-limit interplay + warm start (design spec §5.5-§5.6).
# `_newton_power_flow` runs UNCHANGED on the augmented system per the spec; these tests are
# the first to exercise it under real iteration and are what flushed out the two src fixes
# documented at each call site below (`AreaInterchangeData.delta_p` warm-start mirror in
# `state_indexing_helpers.jl`/`power_flow_setup.jl`, and the ΔP-into-`F`-not-`P_net` seam in
# `ac_power_flow_residual.jl`).
#
# All oracle-vs-target comparisons use atol = 1e-6, not the solver's own ~1e-12 residual
# tolerance: `_oracle_tie_metered_power` reads `data.power_network_matrix.data`'s `nzval`,
# stored as `YBUS_ELTYPE == ComplexF32`, so the independent oracle carries an inherent
# ComplexF32 rounding floor -- the same precedent already used by the tie kernel and 6b
# diagonal-pollution tests above (`atol = 1e-6`).

@testset "area interchange NR converges and meets interchange targets; control off unaffected" begin
    sys = _three_area_transfer_fixture(; slack_area3 = true)
    pf = ACPolarPowerFlow{NewtonRaphsonACPowerFlow}(; area_interchange_control = true)
    data = PowerFlowData(pf, sys)
    @test PF.n_controlled_areas(data) == 2
    @test solve_power_flow!(data)

    ni = _oracle_ni_by_tail(sys, data, data.area_interchange.ties, 1)
    for area in data.area_interchange.areas
        @test isapprox(ni[area.tail_ix], area.pdes; atol = 1e-6)
    end

    # Same system, control off: enrollment is a no-op and the plain AC solve is unaffected.
    pf_off = ACPolarPowerFlow{NewtonRaphsonACPowerFlow}()
    data_off = PowerFlowData(pf_off, sys)
    @test PF.n_controlled_areas(data_off) == 0
    @test solve_power_flow!(data_off)
end

@testset "area interchange TR solution matches NR" begin
    sys_nr = _three_area_transfer_fixture(; slack_area3 = true)
    pf_nr = ACPolarPowerFlow{NewtonRaphsonACPowerFlow}(; area_interchange_control = true)
    data_nr = PowerFlowData(pf_nr, sys_nr)
    @test solve_power_flow!(data_nr)

    sys_tr = _three_area_transfer_fixture(; slack_area3 = true)
    pf_tr = ACPolarPowerFlow{TrustRegionACPowerFlow}(; area_interchange_control = true)
    data_tr = PowerFlowData(pf_tr, sys_tr)
    @test solve_power_flow!(data_tr)

    @test isapprox(data_nr.bus_magnitude[:, 1], data_tr.bus_magnitude[:, 1]; atol = 1e-8)
    @test isapprox(data_nr.bus_angles[:, 1], data_tr.bus_angles[:, 1]; atol = 1e-8)
    for (area_nr, area_tr) in
        zip(data_nr.area_interchange.areas, data_tr.area_interchange.areas)
        @test isapprox(
            data_nr.area_interchange.delta_p[area_nr.tail_ix, 1],
            data_tr.area_interchange.delta_p[area_tr.tail_ix, 1];
            atol = 1e-8,
        )
    end
end

@testset "area interchange distributed slack across areas meets targets" begin
    sys = _three_area_transfer_fixture(; slack_area3 = true)
    # Spread across generators in both the uncontrolled area (Area1: Bus1/Bus2/Bus3) and
    # both controlled areas (Area2: Bus6/Bus8; Area3: Bus9Gen), so each enrolled area's raw
    # `w_a` (guard 6, sum of its OWN buses' participation) stays well under the 0.9 limit.
    gspf = Dict(
        (PSY.ThermalStandard, "Bus1") => 0.4,
        (PSY.ThermalStandard, "Bus2") => 0.3,
        (PSY.ThermalStandard, "Bus3") => 0.1,
        (PSY.ThermalStandard, "Bus6") => 0.1,
        (PSY.ThermalStandard, "Bus8") => 0.1,
        (PSY.ThermalStandard, "Bus9Gen") => 0.1,
    )
    pf = ACPolarPowerFlow{NewtonRaphsonACPowerFlow}(;
        area_interchange_control = true,
        generator_slack_participation_factors = gspf,
    )
    data = PowerFlowData(pf, sys)
    @test PF.n_controlled_areas(data) == 2
    @test solve_power_flow!(data)

    ni = _oracle_ni_by_tail(sys, data, data.area_interchange.ties, 1)
    for area in data.area_interchange.areas
        @test isapprox(ni[area.tail_ix], area.pdes; atol = 1e-6)
    end

    # Fix wave 8b, Finding 2: prove guard 6 (slack-absorption weight cap, `w_a > 0.9`
    # de-enrolls the area) did NOT fire for either enrolled area -- both areas being
    # enrolled (`n_controlled_areas == 2`, already checked above) is necessary but not
    # sufficient, since an area could be de-enrolled for an UNRELATED reason while a
    # comment merely asserts guard 6 specifically stayed quiet. Recompute each enrolled
    # area's raw `w_a` the same way guard 6 does (`_area_slack_candidate` in
    # `enrollment.jl`) and assert it explicitly.
    bus_lookup = PF.get_bus_lookup(data)
    nrd = PF.get_network_reduction_data(data)
    reverse_bus_search_map = PNM.get_reverse_bus_search_map(nrd)
    buses_by_area = PF._buses_by_area(sys)
    spf = PF.get_bus_slack_participation_factors(data)
    for area in data.area_interchange.areas
        resolved =
            PF._resolved_bus_ixs(
                buses_by_area[area.name],
                bus_lookup,
                reverse_bus_search_map,
            )
        w_a = sum(spf[ix, 1] for ix in resolved)
        @test w_a < PF.AREA_SLACK_ABSORPTION_LIMIT
    end
end

@testset "area interchange Q-limit flip retains ΔP coupling through the bus-type change" begin
    sys = _three_area_transfer_fixture(; slack_area3 = true)
    gen6 = PSY.get_component(PSY.ThermalStandard, sys, "Bus6")
    # Natural (unconstrained) Q at Bus6 solves to about -0.0156 pu; this tightened min just
    # barely excludes it so `_check_q_limit_bounds!` flips the bus PV -> PQ mid-solve.
    PSY.set_reactive_power_limits!(gen6, (min = -0.012, max = 0.24))
    pf = ACPolarPowerFlow{NewtonRaphsonACPowerFlow}(;
        area_interchange_control = true,
        check_reactive_power_limits = true,
    )
    data = PowerFlowData(pf, sys)
    bus6_ix = PF.get_bus_lookup(data)[6]
    @test data.bus_type[bus6_ix, 1] == PSY.ACBusTypes.PV
    @test solve_power_flow!(data)
    @test data.bus_type[bus6_ix, 1] == PSY.ACBusTypes.PQ

    ni = _oracle_ni_by_tail(sys, data, data.area_interchange.ties, 1)
    for area in data.area_interchange.areas
        @test isapprox(ni[area.tail_ix], area.pdes; atol = 1e-6)
    end
end

@testset "area interchange warm re-solve converges in 0-1 iterations with targets retained" begin
    sys = _three_area_transfer_fixture(; slack_area3 = true)
    pf = ACPolarPowerFlow{NewtonRaphsonACPowerFlow}(; area_interchange_control = true)
    data = PowerFlowData(pf, sys)
    @test solve_power_flow!(data)
    delta_p_first = copy(data.area_interchange.delta_p)

    @test_logs(
        (:info, r"converged after [01] iterations"),
        match_mode = :any,
        solve_power_flow!(data)
    )

    # Spec §2: warm starts RETAIN the previous ΔP_a (the `delta_p` mirror seam). Without
    # retention the tail resets to 0 and the re-solve must re-derive ΔP from a 0.89-pu-off
    # slack-bus P-balance row -- it happens to recover in 1 iteration on this fixture (ΔP
    # enters the system linearly), so the iteration band alone cannot catch a dropped tail.
    @test isapprox(data.area_interchange.delta_p, delta_p_first; atol = 1e-8)

    ni = _oracle_ni_by_tail(sys, data, data.area_interchange.ties, 1)
    for area in data.area_interchange.areas
        @test isapprox(ni[area.tail_ix], area.pdes; atol = 1e-6)
    end
end

# Fix wave 8b, Finding 1: `AreaInterchangeData.delta_p` gained a time-step dimension so a
# multi-period `data` keeps each time step's converged ΔP_a independent. Before the fix,
# `delta_p` was a single Vector shared by every time step -- solving ts=2 clobbered ts=1's
# already-converged mirror. RED (pre-fix): this test errors/fails because `delta_p` has no
# second dimension to index. GREEN (post-fix): each column survives the other's solve.
@testset "area interchange multi-period warm start does not contaminate delta_p across time steps" begin
    sys = _three_area_transfer_fixture(; slack_area3 = true)
    pf = ACPolarPowerFlow{NewtonRaphsonACPowerFlow}(;
        area_interchange_control = true, time_steps = 2)
    data = PowerFlowData(pf, sys)
    @test PF.n_controlled_areas(data) == 2

    # Perturb ts=2's loads so its converged ΔP_a differs from ts=1's: same network, same
    # fixed PDES targets, but a different load level requires a different area-slack
    # redistribution to hit those targets.
    data.bus_active_power_withdrawals[:, 2] .+= 0.05

    @test solve_power_flow!(data; time_steps = [1])
    delta_p_ts1 = copy(data.area_interchange.delta_p[:, 1])

    # Solve ts=2 in isolation -- the direct test of "solving ts=2 must not touch ts=1".
    @test solve_power_flow!(data; time_steps = [2])
    # Sanity: ts=2's converged ΔP must actually differ from ts=1's, else this test cannot
    # distinguish contamination from coincidence.
    @test !isapprox(data.area_interchange.delta_p[:, 2], delta_p_ts1; atol = 1e-4)

    # The real assertion: solving ts=2 must not clobber ts=1's already-converged mirror.
    @test isapprox(data.area_interchange.delta_p[:, 1], delta_p_ts1; atol = 1e-8)
end

@testset "area interchange 3-area: enrolled targets met, de-enrolled area is the implicit complement" begin
    sys = _three_area_transfer_fixture(; slack_area3 = true)
    pf = ACPolarPowerFlow{NewtonRaphsonACPowerFlow}(; area_interchange_control = true)
    data = PowerFlowData(pf, sys)
    @test PF.n_controlled_areas(data) == 2
    @test solve_power_flow!(data)

    ni = _oracle_ni_by_tail(sys, data, data.area_interchange.ties, 1)
    for area in data.area_interchange.areas
        @test isapprox(ni[area.tail_ix], area.pdes; atol = 1e-6)
    end

    # Area1 (REF-owning, de-enrolled by guard 3): recompute its NI via a FRESH all-three-area
    # tie set (`data.area_interchange.ties` only tags Area2/Area3's tails) built directly
    # against `PF.build_area_ties`, then check the tie-cancellation identity -- Area1's
    # implicit NI is the negative complement of the two enrolled areas' solved NI.
    bus_lookup = PF.get_bus_lookup(data)
    ybus = PF.get_power_network_matrix(data)
    nrd = PF.get_network_reduction_data(data)
    tail_of_area = Dict("Area1" => 1, "Area2" => 2, "Area3" => 3)
    bus_area_map = Dict{Int, Int}()
    for bus in PSY.get_components(PSY.ACBus, sys)
        area = PSY.get_area(bus)
        isnothing(area) && continue
        bus_area_map[bus_lookup[PSY.get_number(bus)]] = tail_of_area[PSY.get_name(area)]
    end
    all_ties = PF.build_area_ties(sys, bus_lookup, ybus, nrd, bus_area_map)
    ni_all = _oracle_ni_by_tail(sys, data, all_ties, 1)
    @test isapprox(ni_all[1], -(ni_all[2] + ni_all[3]); atol = 1e-6)
end

# Task 9: results table, diagnostics, and greedy-relax infeasibility handling.

@testset "area interchange results happy path" begin
    sys = _three_area_transfer_fixture(; slack_area3 = true)
    pf = ACPolarPowerFlow{NewtonRaphsonACPowerFlow}(; area_interchange_control = true)
    df_results = solve_power_flow(pf, sys)
    @test haskey(df_results, "area_interchange_results")
    df = df_results["area_interchange_results"]
    @test nrow(df) == 2
    @test Set(df.area) == Set(["Area2", "Area3"])
    for row in eachrow(df)
        @test row.schedule_status == :enforced
        @test row.beyond_limits == false
        @test isapprox(row.ni_solved, row.pdes; atol = 1e-3)
    end
    # `delta_p` matches the converged ΔP_a mirror (scaled to MW like every other column).
    data = PowerFlowData(pf, sys)
    @test solve_power_flow!(data)
    sys_basepower = PSY.get_base_power(sys)
    for area in data.area_interchange.areas
        row = only(filter(:area => ==(area.name), df))
        @test isapprox(
            row.delta_p,
            sys_basepower * data.area_interchange.delta_p[area.tail_ix, 1];
            atol = 1e-6,
        )
    end
end

@testset "area interchange results beyond_limits flag" begin
    # Shrink Area2's slack machine (Bus6Gen, native active_power = 0.0) so the ΔP_a needed
    # to hit PDES = 0.3 pu cannot be absorbed within its active_power_limits.
    sys = _three_area_transfer_fixture(; slack_area3 = true)
    gen6 = PSY.get_component(PSY.ThermalStandard, sys, "Bus6")
    PSY.set_active_power_limits!(gen6, (min = 0.0, max = 0.001))
    pf = ACPolarPowerFlow{NewtonRaphsonACPowerFlow}(; area_interchange_control = true)
    df_results = solve_power_flow(pf, sys)
    df = df_results["area_interchange_results"]
    area2_row = only(filter(:area => ==("Area2"), df))
    area3_row = only(filter(:area => ==("Area3"), df))
    @test area2_row.beyond_limits == true
    @test area3_row.beyond_limits == false
    # Flag only: values stay un-clamped (still tracking the true, achieved target).
    @test isapprox(area2_row.ni_solved, area2_row.pdes; atol = 1e-3)
    @test area2_row.schedule_status == :enforced
end

# Custom 3-bus, 3-area fixture for the greedy-relax tests: a strong Bus1-Bus2 tie (Area2,
# an easily achievable target) and a deliberately WEAK, high-reactance Bus1-Bus3 tie
# (Area3), so a modest Area3 PDES is already far beyond that corridor's transfer capability.
# Built from primitives (not `_three_area_transfer_fixture`/c_sys14) because c_sys14's
# transfer-capability nose is a sharp voltage-collapse bifurcation: pushing a single area's
# PDES toward it produces genuinely NONDETERMINISTIC Newton iterates run-to-run (verified
# empirically -- same input, different failed iterate, sometimes different area picked as
# worst-offender). This fixture's weak tie is so far from feasible for ANY of the tested
# margins that the failure is clean and deterministic every run.
function _weak_tie_three_area_fixture(; x_weak::Float64 = 2.0, pdes2::Float64 = 0.1,
    pdes3::Float64 = 2.0)
    sys = System(100.0)
    area1 = PSY.Area(; name = "Area1")
    area2 = PSY.Area(; name = "Area2")
    area3 = PSY.Area(; name = "Area3")
    PSY.add_component!(sys, area1)
    PSY.add_component!(sys, area2)
    PSY.add_component!(sys, area3)

    bus1 = _add_simple_bus!(sys, 1, ACBusTypes.REF, 230)
    bus2 = _add_simple_bus!(sys, 2, ACBusTypes.PV, 230)
    bus3 = _add_simple_bus!(sys, 3, ACBusTypes.PV, 230)
    PSY.set_area!(bus1, area1)
    PSY.set_area!(bus2, area2)
    PSY.set_area!(bus3, area3)

    _add_simple_source!(sys, bus1, 0.0, 0.0)
    _add_simple_thermal_standard!(sys, bus2, 0.1, 0.0)
    _add_simple_thermal_standard!(sys, bus3, 0.1, 0.0)
    _add_simple_load!(sys, bus1, 0.05, 0.02)

    _add_simple_line!(sys, bus1, bus2, 1e-3, 1e-3)      # strong: Area2's tie
    _add_simple_line!(sys, bus1, bus3, 0.02, x_weak)    # weak: Area3's tie

    PSY.set_bustype!(bus2, ACBusTypes.SLACK)
    PSY.set_bustype!(bus3, ACBusTypes.SLACK)

    _add_area_interchange!(sys, "Area2", "Area1", pdes2; name = "A2_A1")
    _add_area_interchange!(sys, "Area3", "Area1", pdes3; name = "A3_A1")
    return sys
end

@testset "area interchange infeasible schedule greedy relax" begin
    sys = _weak_tie_three_area_fixture()
    pf = ACPolarPowerFlow{NewtonRaphsonACPowerFlow}(; area_interchange_control = true)
    data = PowerFlowData(pf, sys)
    @test PF.n_controlled_areas(data) == 2

    converged = @test_logs(
        (:error, r"solver failed to converge"),
        (
            :error,
            r"Area interchange:.*Area3.*de-enrolling it and re-solving with the remaining 1",
        ),
        (
            :error,
            r"Area interchange:.*converged only after relaxing 1 area.*Area3 " *
            r"\(ni_solved=.*, pdes=.*, gap=.*\)",
        ),
        match_mode = :any,
        min_level = Logging.Warn,
        solve_power_flow!(data)
    )
    @test converged
    # Greedy keeps the enforceable Area2 schedule; only Area3 (the infeasible one) relaxes.
    @test length(data.area_interchange.areas) == 1
    @test only(data.area_interchange.areas).name == "Area2"
    @test haskey(data.area_interchange.relaxed, 1)
    @test only(data.area_interchange.relaxed[1]).name == "Area3"

    df_results = @test_logs(
        (:error, r"solver failed to converge"),
        (:error, r"Area interchange:.*Area3.*de-enrolling"),
        (
            :error,
            r"Area interchange:.*converged only after relaxing 1 area.*Area3 " *
            r"\(ni_solved=.*, pdes=.*, gap=.*\)",
        ),
        match_mode = :any,
        min_level = Logging.Warn,
        solve_power_flow(pf, sys)
    )
    df = df_results["area_interchange_results"]
    @test nrow(df) == 2
    area2_row = only(filter(:area => ==("Area2"), df))
    area3_row = only(filter(:area => ==("Area3"), df))
    @test area2_row.schedule_status == :enforced
    @test isapprox(area2_row.ni_solved, area2_row.pdes; atol = 1e-3)
    @test area3_row.schedule_status == :relaxed
    @test area3_row.delta_p == 0.0
    # Infeasibility certificate: achieved NI is far short of the 200 MW (2.0 pu) target.
    @test area3_row.ni_solved < 0.5 * area3_row.pdes
end

# Modest, individually-achievable targets on the SAME strong-tie topology as
# `_weak_tie_three_area_fixture` -- the schedule is not the problem here, `maxIterations = 1`
# from a flat start is. Mirrors the project's existing non-convergence fixture idiom
# (`test_rectangular_ci_power_flow.jl`'s "maxIterations = 1 from flat start cannot converge").
function _normal_two_area_fixture(; pdes2::Float64 = 0.05, pdes3::Float64 = 0.02)
    return _weak_tie_three_area_fixture(; x_weak = 1e-3, pdes2 = pdes2, pdes3 = pdes3)
end

@testset "area interchange network non-convergence after relax exhausts" begin
    sys_baseline = _normal_two_area_fixture()
    pf_baseline =
        ACPolarPowerFlow{NewtonRaphsonACPowerFlow}(; area_interchange_control = true)
    data_baseline = PowerFlowData(pf_baseline, sys_baseline)
    @test solve_power_flow!(data_baseline)   # sanity: fully solvable given enough iterations

    sys = _normal_two_area_fixture()
    pf = ACPolarPowerFlow{NewtonRaphsonACPowerFlow}(; area_interchange_control = true)
    data = PowerFlowData(pf, sys)
    converged = @test_logs(
        (:error, r"solver failed to converge"),
        (:error, r"Area interchange:.*Newton did not converge with area"),
        (
            :warn,
            r"Area interchange: Newton did not converge after the greedy relax loop de-enrolled every controlled area",
        ),
        match_mode = :any,
        min_level = Logging.Warn,
        solve_power_flow!(data; maxIterations = 1)
    )
    @test !converged
    @test PF.n_controlled_areas(data) == 0
end

# Fix wave 9b, Finding 1 (CRITICAL): the short-circuit at the top of
# `_ac_power_flow_with_area_relax!` used to read the WORKING area set
# (`n_controlled_areas(data)`), not the PRISTINE one. A time step whose greedy relax
# exhausts the WORKING set to zero (as above) leaves that empty state on `data` for good --
# a LATER time step's own call would then short-circuit straight to a bare `_ac_power_flow`
# BEFORE `_ensure_pristine_area_set!` ever ran, permanently disabling area control for the
# rest of `data`'s lifetime, regardless of that later time step's own schedule being
# perfectly feasible. RED (pre-fix, verified against the code before this fix): ts=2
# "converges" with area control silently OFF (`n_controlled_areas(data) == 0`) and
# `area_interchange_results_dataframe` throws `KeyError: key "Area2" not found` (the
# `:enforced` branch's `working_tail_of` lookup against the now-empty working set). GREEN
# (post-fix): ts=1's exhaustion stays contained to ts=1; ts=2 restores the full pristine
# enrollment and meets its own (feasible) targets.
@testset "area interchange multi-period relax-to-zero does not permanently disable area control" begin
    sys = _normal_two_area_fixture()
    pf = ACPolarPowerFlow{NewtonRaphsonACPowerFlow}(;
        area_interchange_control = true, time_steps = 2)
    data = PowerFlowData(pf, sys)
    @test PF.n_controlled_areas(data) == 2

    # ts=1: force exhaustion via maxIterations = 1 (same deterministic technique as the
    # single-period exhaustion test above) -- greedy relax drops both areas one at a time,
    # and even the final 0-area plain solve still fails to converge in a single iteration
    # from a flat start.
    converged1 = @test_logs(
        (:error, r"solver failed to converge"),
        (:error, r"Area interchange:.*Newton did not converge with area"),
        (
            :warn,
            r"Area interchange: Newton did not converge after the greedy relax loop de-enrolled every controlled area",
        ),
        match_mode = :any,
        min_level = Logging.Warn,
        solve_power_flow!(data; time_steps = [1], maxIterations = 1)
    )
    @test !converged1
    @test PF.n_controlled_areas(data) == 0
    @test length(data.area_interchange.pristine_areas) == 2

    # ts=2: an ISOLATED non-leading `time_steps` subset through the top-level
    # `solve_power_flow!` -- exercises Finding 1's guard exactly as a real caller would.
    # Normal iteration budget: this schedule is fully feasible (proved by the exhaustion
    # test's own baseline sanity check above).
    @test solve_power_flow!(data; time_steps = [2])
    @test data.converged[2]
    @test PF.n_controlled_areas(data) == 2

    # atol = 1e-5, not the file-standard 1e-6: this fixture's ComplexF32 Y-bus rounding
    # floor (see the Task 8 header note) lands at ~1.1e-6 for the weak tie, just above it.
    ni = _oracle_ni_by_tail(sys, data, data.area_interchange.ties, 2)
    for area in data.area_interchange.areas
        @test isapprox(ni[area.tail_ix], area.pdes; atol = 1e-5)
    end

    # The area dataframe must build without error and show BOTH areas :enforced for ts=2 --
    # this is exactly the KeyError path Finding 1's guard fix makes unreachable.
    df2 = PF.area_interchange_results_dataframe(sys, data, 2)
    @test nrow(df2) == 2
    @test all(==(:enforced), df2.schedule_status)
    for row in eachrow(df2)
        @test isapprox(row.ni_solved, row.pdes; atol = 1e-3)
    end
end

# Fix wave 11b, Finding 1 (CRITICAL): `area_interchange_results_dataframe` used to build
# `working_tail_of` from the GLOBAL working `aid.areas`/`aid.delta_p` -- state that
# `_deenroll_area!` mutates permanently for `data`'s lifetime, not scoped to the time step
# that triggered the relax. A LATER time step's relax therefore corrupts a read-back of an
# EARLIER, cleanly-enforced time step's row: `working_tail_of` no longer contains the name of
# whatever area a later relax dropped, so the `:enforced` branch's
# `aid.delta_p[working_tail_of[name], time_step]` throws `KeyError`. RED (pre-fix, verified
# against the code before this fix): querying ts=1 (which enforced BOTH areas cleanly) after
# simulating a ts=2 relax raises `KeyError("Area2")`. GREEN (post-fix): the `:enforced` branch
# reads `aid.pristine_delta_p[area.tail_ix, time_step]` -- the persistent, PRISTINE-tail_ix,
# per-time-step mirror `_sync_pristine_delta_p!` maintains -- so ts=1's row is immune to
# whatever ts=2 did to the working set.
#
# The relax itself is driven directly through the same internal primitives
# `_ac_power_flow_with_area_relax!` uses (`_deenroll_area!`, `relaxed`, `_sync_pristine_delta_p!`)
# rather than through a genuinely-infeasible schedule: `ControlledArea.pdes` and the network
# topology are fixed for `data`'s whole lifetime, so a schedule infeasible enough to force a
# real relax at ts=2 would be equally infeasible at ts=1, defeating the "ts=1 solves clean"
# premise this regression needs. Driving the exact mutating seam directly is deterministic and
# targets precisely the read-back bug, mirroring the existing multi-period tests' idiom of
# calling `_ac_power_flow_with_area_relax!` (and friends) directly to reach a specific seam.
@testset "area interchange multi-period results read-back is immune to a later time step's relax" begin
    sys = _three_area_transfer_fixture(; slack_area3 = true)
    pf = ACPolarPowerFlow{NewtonRaphsonACPowerFlow}(;
        area_interchange_control = true, time_steps = 2)
    data = PowerFlowData(pf, sys)
    aid = data.area_interchange
    @test PF.n_controlled_areas(data) == 2

    # ts=1: both areas enforced cleanly, no relax.
    @test solve_power_flow!(data; time_steps = [1])
    @test !haskey(aid.relaxed, 1)
    ni_ts1 = _oracle_ni_by_tail(sys, data, aid.ties, 1)
    delta_p_ts1_by_name =
        Dict(area.name => aid.delta_p[area.tail_ix, 1] for area in aid.areas)
    ni_ts1_by_name = Dict(area.name => ni_ts1[area.tail_ix] for area in aid.areas)

    # ts=2: simulate a genuine greedy relax of the FIRST working area (whichever pristine
    # area currently sits at tail_ix == 1) via the exact mutating primitives
    # `_ac_power_flow_with_area_relax!` itself calls on a real relax.
    dropped = PF._deenroll_area!(data, 1)
    aid.relaxed[2] = [PF.RelaxedAreaRecord(dropped.name, dropped.pdes)]
    survivor_name = only(a.name for a in aid.pristine_areas if a.name != dropped.name)
    @test PF.n_controlled_areas(data) == 1
    @test only(aid.areas).name == survivor_name

    # Converge the reduced (1-area) working set for ts=2 for real, then persist it exactly as
    # `_ac_power_flow_with_area_relax!` would on a successful post-relax retry.
    @test PF._ac_power_flow(data, pf, 2; PF.get_solver_kwargs(pf)...)
    PF._sync_pristine_delta_p!(data, 2)

    # The bug: querying ts=1 AFTER ts=2's relax mutated the working set. Must not KeyError,
    # must still report BOTH areas :enforced with ts=1's OWN (pre-relax) values.
    df1 = PF.area_interchange_results_dataframe(sys, data, 1)
    @test nrow(df1) == 2
    @test all(==(:enforced), df1.schedule_status)
    sys_basepower = PSY.get_base_power(sys)
    for row in eachrow(df1)
        @test isapprox(
            row.delta_p,
            sys_basepower * delta_p_ts1_by_name[row.area];
            atol = 1e-8,
        )
        # atol = 1e-5, not the file-standard 1e-6: this fixture's ComplexF32 Y-bus rounding
        # floor (see the Task 8 header note) lands at ~1.1e-6.
        @test isapprox(row.ni_solved, sys_basepower * ni_ts1_by_name[row.area]; atol = 1e-5)
    end

    # ts=2 correctly shows the relax: dropped area's row is :relaxed (delta_p = 0.0), the
    # survivor is :enforced and met its own target.
    df2 = PF.area_interchange_results_dataframe(sys, data, 2)
    @test nrow(df2) == 2
    dropped_row = only(filter(:area => ==(dropped.name), df2))
    survivor_row = only(filter(:area => ==(survivor_name), df2))
    @test dropped_row.schedule_status == :relaxed
    @test dropped_row.delta_p == 0.0
    @test survivor_row.schedule_status == :enforced
    @test isapprox(survivor_row.ni_solved, survivor_row.pdes; atol = 1e-3)
end

@testset "area interchange residual diagnostics VSC bus-block partition regression" begin
    # Carried debt (3b): regression coverage for the Schur bus-block partition fix
    # (`n_bus = n_state - state_tail_length(data, dcn)`, `residual_condition_diagnostics.jl`).
    # A VSC-only system (n_lcc == 0) is exactly the shape the OLD formula
    # (`n_state - 4 * n_lcc`) mispartitioned: with zero LCCs it reduced to `n_bus == n_state`,
    # folding the entire VSC tail into the "bus" block instead of excluding it.
    sys = _build_vsc_system(; g = 50.0)
    settings = merge(VSC_SETTINGS, Dict{Symbol, Any}(:linear_solver => "KLU"))
    pf = ACPowerFlow{NewtonRaphsonACPowerFlow}(;
        log_solver_diagnostics = true,
        solver_settings = settings,
    )
    data = PowerFlowData(pf, sys)
    residual = PF.ACPowerFlowResidual(data, 1)
    jac = PF.ACPowerFlowJacobian(residual, 1)
    x0 = PF.calculate_x0(data, 1)
    residual(x0, 1)
    jac(1)

    n_state = size(jac.Jv, 1)
    dcn = PF.get_dc_network(data)
    n_bus_correct = n_state - PF.state_tail_length(data, dcn)
    n_lcc = size(data.lcc.p_set, 1)
    @test n_lcc == 0
    @test n_bus_correct != n_state - 4 * n_lcc   # the OLD (buggy) formula's value

    # The smallest DIRECTLY observable regression: the Schur matvec at the CORRECT partition
    # must match the dense ground truth (same technique as
    # "Schur min-eigenvalue matches dense ground truth (LCC)",
    # test_residual_condition_diagnostics.jl) -- a wrong partition size would either error
    # (dimension mismatch) or silently diverge from this dense cross-check.
    backend = PNM.KLUSolver()
    cache = PF.make_linear_solver_cache(backend, jac.Jv)
    PF.symbolic_factor!(cache, jac.Jv)
    PF.numeric_refactor!(cache, jac.Jv)
    op = PF.SchurInverseOperator(cache, n_bus_correct, Vector{Float64}(undef, n_state))
    λ, schur_converged = PF._schur_min_eigenvalue(op)
    @test schur_converged
    Jinv = inv(Matrix(jac.Jv))
    S = inv(Jinv[1:n_bus_correct, 1:n_bus_correct])
    ev = eigvals(S)
    λ_true = ev[argmin(abs.(ev))]
    @test abs(λ - λ_true) / abs(λ_true) < 1e-6

    # End-to-end: `log_solver_diagnostics` must run through the full solve with no crash and
    # a converged (never "not-converged"/NaN) λ_min(S) on every logged iteration.
    tl = Test.TestLogger(; min_level = Logging.Info)
    converged = Logging.with_logger(tl) do
        solve_power_flow!(data)
    end
    @test converged
    lines = [r.message for r in tl.logs if occursin(r"iter \d+", r.message)]
    @test !isempty(lines)
    for line in lines
        @test occursin("λ_min(S) = ", line)
        @test !occursin("λ_min(S) = not-converged", line)
    end
end

# Task 10: cross-cutting identities, multi-period, and cache-reuse regressions (spec §7).

# Fix wave 10b, Finding 2: the previous version of this test built `ni_all` by having the
# oracle add ±P_m per tie to two tails -- that sum cancels by construction (bookkeeping),
# for ANY state, and never exercises whether NI/metering/pollution are computed correctly.
# Replaced with the genuinely independent per-area power-balance identity below, checked at
# the SOLVED state for EVERY area (enrolled + uncontrolled).
#
# `_oracle_corridor_loss` reuses the ALREADY-independent `_oracle_tie_metered_power` (never
# the kernel) for both internal-branch loss and the metering correction below, by handing
# it throwaway `AreaTie`s: that function only ever reads `.from_bus_ix`/`.to_bus_ix`/
# `.metered_from` -- never the kernel-only `nz_offsets`/`diag_pollution`/tail fields -- so
# this is a legitimate reuse of an existing oracle, not a new low-level Y-bus
# re-derivation, and the dummy tail/nz_offsets/diag_pollution values are inert filler.
_oracle_dummy_tie(fix::Int, tix::Int, metered_from::Bool) =
    PF.AreaTie(fix, tix, (1, 1, 1, 1), metered_from, 0, 0, (0.0 + 0.0im, 0.0 + 0.0im))

function _oracle_corridor_loss(sys, data, fix::Int, tix::Int, time_step::Int)
    P_f = _oracle_tie_metered_power(sys, data, _oracle_dummy_tie(fix, tix, true), time_step)
    P_t =
        _oracle_tie_metered_power(sys, data, _oracle_dummy_tie(fix, tix, false), time_step)
    return P_f + P_t
end

# Spec §7 row 2: independent per-area power-balance identity, checked over EVERY area with
# any incident tie -- including the uncontrolled REF-owning Area1 -- not just the enrolled
# Area2/Area3 subset carried on `data.area_interchange.ties`. For area `a`:
#
#   NI_a (tie-flow oracle) ≈ generation_a - load_a - internal_losses_a - metering_correction_a
#
# Accounting for each term, all derived from bus-level solved data + branch primitives
# (never `_set_area_tail_residuals!`/`ni_scratch`, the production NI accumulator):
#   * generation_a = Σ_{buses in a} `data.bus_active_power_injections` (0 for a bus with no
#     generator; the solved dispatch for a REF/PV bus, ALREADY including distributed-slack
#     `p_bus_slack` -- see `_setpq`/`_set_state_variables_at_bus!`) PLUS, for a's own
#     controlled slack bus only, the converged `ΔP_a`
#     (`data.area_interchange.delta_p[tail_ix, ts]`). `ΔP_a` is deliberately NOT folded
#     into `bus_active_power_injections` (`_update_residual_values!` applies it straight to
#     `F` at the area slack row, same seam as -- but never merged into -- the
#     distributed-slack `P_slack` term), so the bus's TRUE physical net injection is
#     `injections[slack_bus] + ΔP_a`, not `injections[slack_bus]` alone; omitting this term
#     was verified (below) to leave an ~0.89 pu residual on Area2 in this fixture, where
#     virtually all of the area's real supply comes through the interchange-control ΔP
#     rather than a local generator.
#   * load_a = Σ_{buses in a} `PF.get_bus_active_power_total_withdrawals` (ZIP-evaluated at
#     the solved voltage).
#   * internal_losses_a = Σ over unique bus pairs with BOTH resolved endpoints in area a of
#     `_oracle_corridor_loss` (P_f_own + P_t_own from the corridor's own primitives).
#   * metering_correction_a: NI_a is a METERED-END quantity (PSS/E convention), not a
#     pure-KCL export quantity. As the "metered-end flip" test above derives, for a
#     boundary tie between areas X (metered) and Y (unmetered), NI_Y's contribution
#     UNDERSTATES Y's true own-end KCL flow by exactly that tie's own loss
#     (`true_Y = NI_contribution_Y + loss`). So the naive identity with no metering term is
#     only exact when EVERY tie touching a happens to be metered at a's own end --
#     empirically false here: omitting this term left an ~8.6e-4 residual on Area2 (two
#     orders of magnitude above the ~5e-7 max residual with it included) -- verified NOT to
#     be a `src` bug (it is the same metered-end mechanism Finding 1 exercises directly,
#     confirmed by reproducing the residual, tracing it to metering asymmetry, and
#     confirming it vanishes once this term is added -- see fix-wave-10b report). For every
#     tie where a is the NON-metered side, subtract that tie's own loss.
@testset "area interchange tie identity: per-area power balance against independent bus physics" begin
    sys = _three_area_transfer_fixture(; slack_area3 = true)
    pf = ACPolarPowerFlow{NewtonRaphsonACPowerFlow}(; area_interchange_control = true)
    data = PowerFlowData(pf, sys)
    @test PF.n_controlled_areas(data) == 2
    @test solve_power_flow!(data)

    bus_lookup = PF.get_bus_lookup(data)
    ybus = PF.get_power_network_matrix(data)
    nrd = PF.get_network_reduction_data(data)
    reverse_bus_search_map = PNM.get_reverse_bus_search_map(nrd)
    tail_of_area = Dict("Area1" => 1, "Area2" => 2, "Area3" => 3)
    area_name_of = Dict(tail => name for (name, tail) in tail_of_area)
    bus_area_map = Dict{Int, Int}()
    for bus in PSY.get_components(PSY.ACBus, sys)
        area = PSY.get_area(bus)
        isnothing(area) && continue
        bus_area_map[bus_lookup[PSY.get_number(bus)]] = tail_of_area[PSY.get_name(area)]
    end
    all_ties = PF.build_area_ties(sys, bus_lookup, ybus, nrd, bus_area_map)
    ni_all = _oracle_ni_by_tail(sys, data, all_ties, 1)
    @test length(ni_all) == 3

    # internal_losses_a: dedup to unique (fix,tix) bus pairs before pricing -- a parallel
    # branch pair would otherwise be double-counted (`_oracle_corridor_loss` already sums
    # every matching PSY branch for a given pair internally).
    internal_pairs = Dict{Tuple{Int, Int}, Int}()
    for branch in PSY.get_available_components(PSY.ACTransmission, sys)
        for (arc, _) in _oracle_branch_windings(branch)
            fix = PF._resolve_bus_ix(
                bus_lookup, reverse_bus_search_map, PSY.get_number(PSY.get_from(arc)))
            tix = PF._resolve_bus_ix(
                bus_lookup, reverse_bus_search_map, PSY.get_number(PSY.get_to(arc)))
            (isnothing(fix) || isnothing(tix)) && continue
            area_f = get(bus_area_map, fix, 0)
            area_t = get(bus_area_map, tix, 0)
            (area_f == area_t && !iszero(area_f)) || continue
            key = (fix, tix)
            fix > tix && (key = (tix, fix))
            internal_pairs[key] = area_f
        end
    end
    internal_losses = Dict(a => 0.0 for a in values(tail_of_area))
    for ((fix, tix), area) in internal_pairs
        internal_losses[area] += _oracle_corridor_loss(sys, data, fix, tix, 1)
    end

    # metering_correction_a: every tie contributes -loss to whichever area is the
    # NON-metered side (see derivation above); flip technique mirrors the "metered-end
    # flip" test above.
    metering_correction = Dict(a => 0.0 for a in values(tail_of_area))
    for tie in all_ties
        P_metered = _oracle_tie_metered_power(sys, data, tie, 1)
        flipped = PF.AreaTie(
            tie.from_bus_ix, tie.to_bus_ix, tie.nz_offsets, !tie.metered_from,
            tie.from_area_tail, tie.to_area_tail, tie.diag_pollution)
        P_other = _oracle_tie_metered_power(sys, data, flipped, 1)
        loss_tie = P_metered + P_other
        # NON-metered side gets the correction (mirrors `_oracle_ni_by_tail`'s own
        # metered_tail/other_tail branch, keyed here off `bus_area_map` directly since
        # `tie.from_area_tail`/`to_area_tail` are 0 for an uncontrolled area and can't be
        # used to identify Area1).
        other_area = bus_area_map[tie.to_bus_ix]
        if !tie.metered_from
            other_area = bus_area_map[tie.from_bus_ix]
        end
        metering_correction[other_area] -= loss_tie
    end

    name_to_controlled_area = Dict(a.name => a for a in data.area_interchange.areas)
    # Tolerance floor: ComplexF32 Y-bus storage (`PF.YBUS_ELTYPE`) rounds every primitive to
    # ~1e-7 relative; this identity sums ~15 branch/tie primitives plus bus-level
    # injections/withdrawals, so a few 1e-6 of accumulated rounding is expected -- the NR
    # solver itself converges to ~1e-14 (well below), so it never dominates. Empirically
    # observed max |diff| across all 3 areas in this fixture is ~5e-7 (see fix-wave-10b
    # report); atol picks a generous, non-tuned margin above that.
    for area_tail in 1:3
        buses = [ix for (ix, a) in bus_area_map if a == area_tail]
        gen = sum(
            data.bus_active_power_injections[ix, 1] + data.bus_hvdc_net_power[ix, 1]
            for ix in buses)
        load = sum(PF.get_bus_active_power_total_withdrawals(data, ix, 1) for ix in buses)
        area_name = area_name_of[area_tail]
        extra_dp = 0.0
        if haskey(name_to_controlled_area, area_name)
            ca = name_to_controlled_area[area_name]
            extra_dp = data.area_interchange.delta_p[ca.tail_ix, 1]
        end
        expected_ni =
            (gen + extra_dp) - load - internal_losses[area_tail] +
            metering_correction[area_tail]
        @test isapprox(get(ni_all, area_tail, 0.0), expected_ni; atol = 1e-5)
    end
end

# Fix wave 10b, Finding 1 (CRITICAL): the previous version of this test hand-assigned
# `contribution_from_metering = P_from; contribution_to_metering = -P_to` and asserted
# their difference equals `-(P_from+P_to)` -- true for ANY reals by construction, never
# exercising the kernel's metered-end branch at all. Replaced below with the PRODUCTION
# kernel `_area_net_interchange` (the exact accumulation `_set_area_tail_residuals!` runs
# inside the Newton residual -- chosen over the test-only oracle precisely because the
# oracle is itself test code and cannot catch a kernel-side regression), run against the
# SAME solved state under both metered-end orientations, and compared to a loss computed
# independently (`_oracle_tie_metered_power`, PNM primitives, never the kernel).
#
# Spec §7 row 2 (metered-end half): flipping a tie's `ext["metered_end"]` changes which
# end's flow is used to compute that tie's NI contribution. The physically meaningful
# invariant is that the two orientations differ by exactly the corridor's own active-power
# loss (P_from + P_to, both defined as power flowing OUT of their respective bus into the
# line) -- NOT that a rebuild+resolve under the flipped convention reproduces the SAME
# converged state (it can't: hitting the identical PDES target under a different loss
# allocation requires a genuinely different ΔP_a). So the identity itself is checked at a
# SINGLE fixed converged state (both orientations evaluated via the independent
# branch-primitive oracle against the SAME data.bus_magnitude/bus_angles) -- no second solve
# involved in that assertion, hence no solver-drift contamination. The rebuild+resolve is
# still exercised separately, as an end-to-end sanity check that flipping metered_end for
# real still converges and still meets both areas' targets.
@testset "area interchange tie identity: metered-end flip shifts NI by the tie's own loss" begin
    sys = _three_area_transfer_fixture(; slack_area3 = true)
    pf = ACPolarPowerFlow{NewtonRaphsonACPowerFlow}(; area_interchange_control = true)
    data = PowerFlowData(pf, sys)
    @test solve_power_flow!(data)

    bus_lookup = PF.get_bus_lookup(data)
    fix, tix = bus_lookup[9], bus_lookup[10]   # Line11: Area3 (Bus9) <-> Area2 (Bus10)
    tie = _find_tie(data.area_interchange.ties, fix, tix)
    @test tie.metered_from == true   # no ext set yet: defaults "from"

    # Same tie, both metered-end orientations, evaluated at the SAME fixed converged state.
    tie_from = PF.AreaTie(
        tie.from_bus_ix, tie.to_bus_ix, tie.nz_offsets, true,
        tie.from_area_tail, tie.to_area_tail, tie.diag_pollution)
    tie_to = PF.AreaTie(
        tie.from_bus_ix, tie.to_bus_ix, tie.nz_offsets, false,
        tie.from_area_tail, tie.to_area_tail, tie.diag_pollution)

    # Independent oracle loss (PNM.ybus_branch_entries primitives, NEVER the kernel):
    # P_from/P_to are both "power flowing OUT of that bus into the line"; their sum is the
    # corridor's ohmic loss.
    P_from_oracle = _oracle_tie_metered_power(sys, data, tie_from, 1)
    P_to_oracle = _oracle_tie_metered_power(sys, data, tie_to, 1)
    loss = P_from_oracle + P_to_oracle
    @test loss > 0   # sanity: a real R>0 line has strictly positive I²R loss

    # THE PRODUCTION KERNEL under test: `_area_net_interchange` runs the exact same
    # `_tie_metered_active_power` accumulation `_set_area_tail_residuals!` uses inside the
    # Newton residual. Handing it a ONE-tie vector containing only `tie_from`/`tie_to`
    # isolates the flip's effect: the only thing that differs between the two calls is
    # `metered_from`, evaluated at the identical solved `data.bus_magnitude`/`bus_angles`.
    area3_tail = tie.from_area_tail   # Area3, the from-side tail
    area2_tail = tie.to_area_tail     # Area2, the to-side tail
    NI3_from = PF._area_net_interchange([tie_from], area3_tail, data, 1)
    NI3_to = PF._area_net_interchange([tie_to], area3_tail, data, 1)
    NI2_from = PF._area_net_interchange([tie_from], area2_tail, data, 1)
    NI2_to = PF._area_net_interchange([tie_to], area2_tail, data, 1)

    # Derivation (`_area_net_interchange`'s own metered_tail/other_tail branch, matching
    # `_set_area_tail_residuals!`): with tie.from_area_tail = Area3, tie.to_area_tail =
    # Area2 --
    #   tie_from (metered_from=true):  metered_tail=Area3 -> ni[Area3] += P_m;
    #                                   other_tail=Area2   -> ni[Area2] -= P_m.
    #     So NI3_from = +P_m(tie_from), NI2_from = -P_m(tie_from).
    #   tie_to (metered_from=false):   metered_tail=Area2 -> ni[Area2] += P_m;
    #                                   other_tail=Area3   -> ni[Area3] -= P_m.
    #     So NI3_to = -P_m(tie_to), NI2_to = +P_m(tie_to).
    # At the kernel's OWN correct behavior, P_m(tie_from) ≈ P_from_oracle and
    # P_m(tie_to) ≈ P_to_oracle (both metering the SAME real quantities the oracle
    # independently derives), giving:
    #   NI3_to - NI3_from = -P_m(tie_to) - P_m(tie_from) ≈ -(P_from_oracle+P_to_oracle) = -loss
    #   NI2_to - NI2_from = +P_m(tie_to) + P_m(tie_from) ≈ +(P_from_oracle+P_to_oracle) = +loss
    # A kernel bug where the metered_to path wrongly reused the from-side Y-bus block (i.e.
    # `_tie_metered_active_power`'s `else` branch ignored `tie.metered_from` and always
    # evaluated the "from" formula) would make P_m(tie_to) ≈ P_m(tie_from) instead of
    # ≈ P_to_oracle, so NI3_to - NI3_from would come out ≈ -2*P_m(tie_from) instead of
    # -loss -- these assertions would then fail. Verified directly: injecting exactly that
    # bug (a local copy of `_tie_metered_active_power`'s `else` branch hard-wired to the
    # "from" formula) against this same fixture changes `NI3_to - NI3_from` from
    # -2.1e-4 (≈ -loss, passes) to +5.9e-2 (fails by >200x atol) -- see fix-wave-10b report.
    @test isapprox(NI3_to - NI3_from, -loss; atol = 1e-6)
    @test isapprox(NI2_to - NI2_from, loss; atol = 1e-6)

    # End-to-end sanity: flip Line11's metered_end for real, rebuild, resolve -- still
    # converges and both areas still meet PDES under the new metering convention.
    line11 = PSY.get_component(PSY.Line, sys, "Line11")
    PSY.get_ext(line11)["metered_end"] = "to"
    data2 = PowerFlowData(pf, sys)
    @test solve_power_flow!(data2)
    tie2 = _find_tie(data2.area_interchange.ties, fix, tix)
    @test tie2.metered_from == false
    ni2 = _oracle_ni_by_tail(sys, data2, data2.area_interchange.ties, 1)
    for area in data2.area_interchange.areas
        @test isapprox(ni2[area.tail_ix], area.pdes; atol = 1e-6)
    end
end

# Spec §7 row 4 (second half): guard 6 (`AREA_SLACK_ABSORPTION_LIMIT`, `w_a > 0.9`) fires
# when an enrolled area's OWN buses hold almost the entire network's raw slack-participation
# weight. Concentrate the distributed-slack input (established via the pf kwarg, same
# machinery as the Task 8 distributed-slack testset) almost entirely on Area2's Bus6, with
# small remainders on Area1/Area3 buses so the network-wide (and per-subnetwork) sum stays
# positive/well-posed -- "near-singular" in the guard's own terms (Area2's raw share ~0.93),
# not in the linear-algebra sense.
@testset "area interchange w_a guard 6 de-enrolls the concentrated area; solve converges without its row" begin
    sys = _three_area_transfer_fixture(; slack_area3 = true)
    gspf = Dict(
        (PSY.ThermalStandard, "Bus1") => 0.02,
        (PSY.ThermalStandard, "Bus2") => 0.02,
        (PSY.ThermalStandard, "Bus3") => 0.01,
        (PSY.ThermalStandard, "Bus6") => 0.92,    # Area2's own bus: raw share > 0.9
        (PSY.ThermalStandard, "Bus8") => 0.01,    # also Area2 (w_a(Area2) = 0.93)
        (PSY.ThermalStandard, "Bus9Gen") => 0.02, # Area3: raw share 0.02, well under limit
    )
    pf = ACPolarPowerFlow{NewtonRaphsonACPowerFlow}(;
        area_interchange_control = true,
        generator_slack_participation_factors = gspf,
    )
    data = @test_logs(
        (:warn, r"Area \"Area2\": area buses hold a slack-participation weight of 0\.93"),
        min_level = Logging.Warn,
        match_mode = :any,
        PowerFlowData(pf, sys)
    )

    # Guard 6 fired at CONSTRUCTION time: Area2 never makes it into the enrolled set.
    @test PF.n_controlled_areas(data) == 1
    @test only(data.area_interchange.areas).name == "Area3"

    # The solve still converges -- WITHOUT Area2's NI-tail row -- and Area3 (unaffected by
    # the guard) still meets its own target.
    @test solve_power_flow!(data)
    ni = _oracle_ni_by_tail(sys, data, data.area_interchange.ties, 1)
    for area in data.area_interchange.areas
        @test isapprox(ni[area.tail_ix], area.pdes; atol = 1e-6)
    end
end

# Spec §7 row 9: multi-period. 3-step fixture; per-step load perturbation (established
# multi-period idiom, same as the delta_p cross-contamination test above) so each step's
# converged ΔP_a genuinely differs. The Jacobian sparse structure depends only on network
# topology/REF layout/slack-participation PATTERN -- none of which change across these 3
# steps (no bus-type flip, no Q-limit event, same gspf pattern every step) -- so it must be
# memoized ONCE (spec §5.4) and reused identically across every time step: same
# `data.ac_jacobian_structure_cache[]` object, not rebuilt at any step transition. This is
# the same object-identity regression device as the "area interchange Jacobian structure
# cache reuse" testset above.
@testset "area interchange multi-period: per-step convergence, delta_p, results, cache built once" begin
    sys = _three_area_transfer_fixture(; slack_area3 = true)
    pf = ACPolarPowerFlow{NewtonRaphsonACPowerFlow}(;
        area_interchange_control = true, time_steps = 3)
    data = PowerFlowData(pf, sys)
    @test PF.n_controlled_areas(data) == 2
    @test isnothing(data.ac_jacobian_structure_cache[])   # nothing built yet

    # Different load perturbation per step -> different required ΔP_a redistribution.
    data.bus_active_power_withdrawals[:, 2] .+= 0.05
    data.bus_active_power_withdrawals[:, 3] .+= 0.10

    # Solve ts=1 alone first: builds the Jacobian structure for the first time.
    @test solve_power_flow!(data; time_steps = [1])
    cache1 = data.ac_jacobian_structure_cache[]
    @test !isnothing(cache1)

    # Solve the full 1:3 range (ts=1 re-solves warm/harmless; ts=2, ts=3 solve for the first
    # time) -- exercises genuine multi-period cache reuse across a single call.
    @test solve_power_flow!(data; time_steps = [1, 2, 3])

    # Cache identity preserved across all 3 steps -> built once, never rebuilt.
    @test data.ac_jacobian_structure_cache[] === cache1

    # Per-step convergence + targets met.
    for t in 1:3
        ni = _oracle_ni_by_tail(sys, data, data.area_interchange.ties, t)
        for area in data.area_interchange.areas
            @test isapprox(ni[area.tail_ix], area.pdes; atol = 1e-6)
        end
    end

    # Per-step delta_p columns genuinely differ (not coincidentally equal).
    dp1 = copy(data.area_interchange.delta_p[:, 1])
    dp2 = copy(data.area_interchange.delta_p[:, 2])
    dp3 = copy(data.area_interchange.delta_p[:, 3])
    @test !isapprox(dp1, dp2; atol = 1e-4)
    @test !isapprox(dp2, dp3; atol = 1e-4)
    @test !isapprox(dp1, dp3; atol = 1e-4)

    # Results table: `write_results`/`solve_power_flow` only support single-period
    # evaluation, so build the multi-period table by calling
    # `area_interchange_results_dataframe` once per time step directly (same technique as
    # the "multi-period relax-to-zero" test above) and stacking. 3 rows per enrolled area.
    dfs = DataFrame[]
    for t in 1:3
        df_t = PF.area_interchange_results_dataframe(sys, data, t)
        df_t[!, :time_step] .= t
        push!(dfs, df_t)
    end
    df = vcat(dfs...)
    @test nrow(df) == 6
    for area_name in ("Area2", "Area3")
        @test nrow(filter(:area => ==(area_name), df)) == 3
    end
end
