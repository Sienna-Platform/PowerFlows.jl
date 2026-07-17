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

# The degree-1 fixture above can't catch the diagonal-pollution bug (its own primitive IS
# the whole diagonal). This fixture adds an extra branch/shunt at each tie endpoint so the
# aggregate diagonal is polluted with non-corridor contributions; `diag_pollution` is
# supplied by hand and the expected P_m uses the tie's own primitive only.
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

    # atol, not rtol: `agg11`/`agg22` are stored as ComplexF32 (`PF.YBUS_ELTYPE`), so the
    # kernel's diagonal-pollution subtraction carries that rounding, while
    # `expected_from`/`expected_to` are exact Float64.
    tie_from = PF.AreaTie(1, 2, o, true, 1, 2, (pollution_from, pollution_to))
    expected_from = Vf^2 * g11 + Vf * Vt * (g12 * cos(θf - θt) + b12 * sin(θf - θt))
    @test PF._tie_metered_active_power(tie_from, Vf, θf, Vt, θt, ybus_nzval) ≈
          expected_from atol = 1e-6

    tie_to = PF.AreaTie(1, 2, o, false, 1, 2, (pollution_from, pollution_to))
    expected_to = Vt^2 * g22 + Vt * Vf * (g21 * cos(θt - θf) + b21 * sin(θt - θf))
    @test PF._tie_metered_active_power(tie_to, Vf, θf, Vt, θt, ybus_nzval) ≈
          expected_to atol = 1e-6
end

function _two_controlled_area_data()
    sys = _three_area_transfer_fixture(; slack_area3 = true)
    pf = ACPolarPowerFlow(; area_interchange_control = true)
    return PowerFlowData(pf, sys)
end

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

# Independent oracle for a tie's metered-end power: sums the ACTUAL PSY branch primitives
# between the tie's bus pair (never the kernel's diagonal reads), so it can't share the
# diagonal-pollution bug. A single loop over `ACTransmission` handles
# `ThreeWindingTransformer` via dispatch, avoiding double-count.
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

    # ΔP_a is added to P_net[slack_bus_ix] at the same seam as the distributed-slack
    # P_slack term; F accumulates injection MINUS P_net, so increasing P_net by ΔP shifts
    # the mismatch row by -ΔP.
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

# `AreaTie.diag_pollution` is cached at tie-build time, so a SwitchedAdmittance at a tie
# endpoint can silently invalidate it if switched post-enrollment. Bus 6 is the endpoint of
# exactly one tie, chosen over the degree-4 Bus 9 for an exact `@test_logs` match.
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

    residual1b = PF.ACPowerFlowResidual(data1, 1)
    PF.ACPowerFlowJacobian(residual1b, 1)
    @test data1.ac_jacobian_structure_cache[] === cache1

    data2 = _two_controlled_area_data()
    @test data2.area_interchange !== data1.area_interchange
    residual2 = PF.ACPowerFlowResidual(data2, 1)
    PF.ACPowerFlowJacobian(residual2, 1)
    cache2 = data2.ac_jacobian_structure_cache[]
    @test cache2.area_data === data2.area_interchange
    @test cache2.area_data !== cache1.area_data
end

# A boundary-crossing 3W transformer winding whose star bus's Y-bus diagonal is polluted by
# BOTH a sibling winding of the same transformer and an unrelated extra line -- neither is a
# member of the boundary-crossing winding's own corridor. Tertiary winding disabled: not
# needed here.
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

# All oracle-vs-target comparisons use atol = 1e-6, not the solver's own ~1e-12 residual
# tolerance: `_oracle_tie_metered_power` reads the Y-bus `nzval` stored as ComplexF32, so
# the independent oracle carries an inherent rounding floor at that level.

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

@testset "area interchange LM parity" begin
    sys_nr = _three_area_transfer_fixture(; slack_area3 = true)
    pf_nr = ACPolarPowerFlow{NewtonRaphsonACPowerFlow}(; area_interchange_control = true)
    data_nr = PowerFlowData(pf_nr, sys_nr)
    @test solve_power_flow!(data_nr)

    sys_lm = _three_area_transfer_fixture(; slack_area3 = true)
    pf_lm =
        ACPolarPowerFlow{LevenbergMarquardtACPowerFlow}(; area_interchange_control = true)
    data_lm = PowerFlowData(pf_lm, sys_lm)
    @test solve_power_flow!(data_lm)

    @test isapprox(data_nr.bus_magnitude[:, 1], data_lm.bus_magnitude[:, 1]; atol = 1e-6)
    @test isapprox(data_nr.bus_angles[:, 1], data_lm.bus_angles[:, 1]; atol = 1e-6)

    ni_nr = _oracle_ni_by_tail(sys_nr, data_nr, data_nr.area_interchange.ties, 1)
    ni_lm = _oracle_ni_by_tail(sys_lm, data_lm, data_lm.area_interchange.ties, 1)
    for area in data_nr.area_interchange.areas
        @test isapprox(
            ni_nr[area.tail_ix],
            area.pdes;
            atol = data_nr.area_interchange.tolerance,
        )
    end
    for area in data_lm.area_interchange.areas
        @test isapprox(
            ni_lm[area.tail_ix],
            area.pdes;
            atol = data_lm.area_interchange.tolerance,
        )
    end

    for (area_nr, area_lm) in
        zip(data_nr.area_interchange.areas, data_lm.area_interchange.areas)
        @test isapprox(
            data_nr.area_interchange.delta_p[area_nr.tail_ix, 1],
            data_lm.area_interchange.delta_p[area_lm.tail_ix, 1];
            atol = 1e-6,
        )
    end
end

@testset "area interchange FD fixed-jacobian parity" begin
    sys_nr = _three_area_transfer_fixture(; slack_area3 = true)
    pf_nr = ACPolarPowerFlow{NewtonRaphsonACPowerFlow}(; area_interchange_control = true)
    data_nr = PowerFlowData(pf_nr, sys_nr)
    @test solve_power_flow!(data_nr)

    sys_fdfj = _three_area_transfer_fixture(; slack_area3 = true)
    pf_fdfj = ACPolarPowerFlow{FastDecoupledACPowerFlow{FDFixedJacobian, FDSchemeXB}}(;
        area_interchange_control = true,
    )
    data_fdfj = PowerFlowData(pf_fdfj, sys_fdfj)
    @test solve_power_flow!(data_fdfj)

    @test isapprox(
        data_nr.bus_magnitude[:, 1],
        data_fdfj.bus_magnitude[:, 1];
        atol = 1e-6,
    )
    @test isapprox(data_nr.bus_angles[:, 1], data_fdfj.bus_angles[:, 1]; atol = 1e-6)

    ni_nr = _oracle_ni_by_tail(sys_nr, data_nr, data_nr.area_interchange.ties, 1)
    ni_fdfj = _oracle_ni_by_tail(sys_fdfj, data_fdfj, data_fdfj.area_interchange.ties, 1)
    for area in data_nr.area_interchange.areas
        @test isapprox(
            ni_nr[area.tail_ix],
            area.pdes;
            atol = data_nr.area_interchange.tolerance,
        )
    end
    for area in data_fdfj.area_interchange.areas
        @test isapprox(
            ni_fdfj[area.tail_ix],
            area.pdes;
            atol = data_fdfj.area_interchange.tolerance,
        )
    end

    for (area_nr, area_fdfj) in
        zip(data_nr.area_interchange.areas, data_fdfj.area_interchange.areas)
        @test isapprox(
            data_nr.area_interchange.delta_p[area_nr.tail_ix, 1],
            data_fdfj.area_interchange.delta_p[area_fdfj.tail_ix, 1];
            atol = 1e-6,
        )
    end
end

@testset "area interchange FD decoupled parity" begin
    sys_nr = _three_area_transfer_fixture(; slack_area3 = true)
    pf_nr = ACPolarPowerFlow{NewtonRaphsonACPowerFlow}(; area_interchange_control = true)
    data_nr = PowerFlowData(pf_nr, sys_nr)
    @test solve_power_flow!(data_nr)

    sys_fd = _three_area_transfer_fixture(; slack_area3 = true)
    pf_fd = ACPolarPowerFlow{FastDecoupledACPowerFlow{PF.FDDecoupled, PF.FDSchemeXB}}(;
        area_interchange_control = true,
    )
    data_fd = PowerFlowData(pf_fd, sys_fd)
    @test solve_power_flow!(data_fd)

    # Looser than the fixed-Jacobian parity test: the B′/B″ half-steps drop the |V| coupling
    # in the area border row (the FD approximation), so the STEP direction is inexact even
    # though the RESIDUAL — and hence the converged target — is exact.
    @test isapprox(data_nr.bus_magnitude[:, 1], data_fd.bus_magnitude[:, 1]; atol = 1e-5)
    @test isapprox(data_nr.bus_angles[:, 1], data_fd.bus_angles[:, 1]; atol = 1e-5)

    ni_nr = _oracle_ni_by_tail(sys_nr, data_nr, data_nr.area_interchange.ties, 1)
    ni_fd = _oracle_ni_by_tail(sys_fd, data_fd, data_fd.area_interchange.ties, 1)
    for area in data_nr.area_interchange.areas
        @test isapprox(
            ni_nr[area.tail_ix],
            area.pdes;
            atol = data_nr.area_interchange.tolerance,
        )
    end
    for area in data_fd.area_interchange.areas
        @test isapprox(
            ni_fd[area.tail_ix],
            area.pdes;
            atol = data_fd.area_interchange.tolerance,
        )
    end
end

@testset "area interchange FD no refactor" begin
    sys_fd = _three_area_transfer_fixture(; slack_area3 = true)
    pf_fd = ACPolarPowerFlow{FastDecoupledACPowerFlow{PF.FDDecoupled, PF.FDSchemeXB}}(;
        area_interchange_control = true,
    )
    data_fd = PowerFlowData(pf_fd, sys_fd)
    @test solve_power_flow!(data_fd)
    cache = data_fd.solver_cache[]
    @test cache isa PF.FastDecoupledCache
    # The bordered area-interchange correction must reuse the once-factored B′ (extra
    # back-solves only) — never a `full_factor!`/refactor of it.
    @test cache.bp_factor_count == 1

    # Regression: a pure-AC FD solve (no controlled areas) is unaffected by the area-border
    # scratch/substep — mirrors the WP5b factor-once pattern in test_fast_decoupled.jl.
    sys_ac = PSB.build_system(PSB.PSITestSystems, "c_sys14"; add_forecasts = false)
    pf_ac = ACPolarPowerFlow{FastDecoupledACPowerFlow{PF.FDDecoupled, PF.FDSchemeXB}}()
    data_ac = PowerFlowData(pf_ac, sys_ac)
    @test PF.n_controlled_areas(data_ac) == 0
    @test solve_power_flow!(data_ac)

    pf_ac_nr = ACPolarPowerFlow{NewtonRaphsonACPowerFlow}()
    data_ac_nr = PowerFlowData(pf_ac_nr, sys_ac)
    @test solve_power_flow!(data_ac_nr)
    @test isapprox(data_ac.bus_magnitude, data_ac_nr.bus_magnitude; atol = 1e-8)
    @test isapprox(data_ac.bus_angles, data_ac_nr.bus_angles; atol = 1e-8)
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

    # Warm starts must retain the previous ΔP_a. This fixture happens to recover in 1
    # iteration even without retention (ΔP enters linearly), so the iteration-count check
    # alone can't catch a dropped tail -- the delta_p equality below is the real assertion.
    @test isapprox(data.area_interchange.delta_p, delta_p_first; atol = 1e-8)

    ni = _oracle_ni_by_tail(sys, data, data.area_interchange.ties, 1)
    for area in data.area_interchange.areas
        @test isapprox(ni[area.tail_ix], area.pdes; atol = 1e-6)
    end
end

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

    @test solve_power_flow!(data; time_steps = [2])
    # Sanity: ts=2's converged ΔP must actually differ from ts=1's, else this test cannot
    # distinguish contamination from coincidence.
    @test !isapprox(data.area_interchange.delta_p[:, 2], delta_p_ts1; atol = 1e-4)

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

    # Area1 (de-enrolled, guard 3): recompute its NI via a FRESH all-three-area tie set
    # built directly against `PF.build_area_ties`, then check the tie-cancellation identity.
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
    (all_ties, _dc_ties) = PF.build_area_ties(sys, bus_lookup, ybus, nrd, bus_area_map)
    ni_all = _oracle_ni_by_tail(sys, data, all_ties, 1)
    @test isapprox(ni_all[1], -(ni_all[2] + ni_all[3]); atol = 1e-6)
end

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

# Strong Bus1-Bus2 tie (Area2) + deliberately weak, high-reactance Bus1-Bus3 tie (Area3).
# Built from primitives, not c_sys14: c_sys14's transfer-capability nose is a
# voltage-collapse bifurcation that makes Newton iterates nondeterministic near
# infeasibility; this weak tie is far enough from feasible to fail deterministically.
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

# SAME strong-tie topology as `_weak_tie_three_area_fixture` -- the schedule is achievable,
# `maxIterations = 1` from a flat start is what fails. Mirrors the project's existing
# non-convergence fixture idiom in test_rectangular_ci_power_flow.jl.
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

@testset "area interchange multi-period relax-to-zero does not permanently disable area control" begin
    sys = _normal_two_area_fixture()
    pf = ACPolarPowerFlow{NewtonRaphsonACPowerFlow}(;
        area_interchange_control = true, time_steps = 2)
    data = PowerFlowData(pf, sys)
    @test PF.n_controlled_areas(data) == 2

    # ts=1: force exhaustion via maxIterations = 1 -- greedy relax drops both areas one at
    # a time, and the final 0-area plain solve still fails in a single iteration from flat.
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

    # ts=2: an isolated, non-leading `time_steps` subset through the top-level
    # `solve_power_flow!`, with a normal iteration budget (fully feasible schedule).
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
        @test isapprox(row.ni_solved, sys_basepower * ni_ts1_by_name[row.area]; atol = 1e-5)
    end

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
    # Regression coverage for the Schur bus-block partition fix: a VSC-only system
    # (n_lcc == 0) is exactly the shape the OLD formula (`n_state - 4*n_lcc`) mispartitioned,
    # folding the whole VSC tail into the "bus" block.
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

    # The Schur matvec at the CORRECT partition must match the dense ground truth (same
    # technique as "Schur min-eigenvalue matches dense ground truth (LCC)") -- a wrong
    # partition size would error or silently diverge.
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

# `_oracle_corridor_loss` reuses the already-independent `_oracle_tie_metered_power` (never
# the kernel) for both internal-branch loss and the metering correction, via throwaway
# `AreaTie`s that only read `.from_bus_ix`/`.to_bus_ix`/`.metered_from` -- a legitimate
# reuse, not a new derivation.
_oracle_dummy_tie(fix::Int, tix::Int, metered_from::Bool) =
    PF.AreaTie(fix, tix, (1, 1, 1, 1), metered_from, 0, 0, (0.0 + 0.0im, 0.0 + 0.0im))

function _oracle_corridor_loss(sys, data, fix::Int, tix::Int, time_step::Int)
    P_f = _oracle_tie_metered_power(sys, data, _oracle_dummy_tie(fix, tix, true), time_step)
    P_t =
        _oracle_tie_metered_power(sys, data, _oracle_dummy_tie(fix, tix, false), time_step)
    return P_f + P_t
end

# Independent per-area power-balance identity, checked for EVERY area: NI_a ≈ generation_a -
# load_a - internal_losses_a - metering_correction_a, derived from bus-level data + branch
# primitives. metering_correction_a is needed because NI_a is a METERED-END quantity: a
# boundary tie's non-metered side understates its true KCL flow by that tie's own loss.
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
    (all_ties, _dc_ties) = PF.build_area_ties(sys, bus_lookup, ybus, nrd, bus_area_map)
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
        # NON-metered side gets the correction, keyed off `bus_area_map` directly since
        # `tie.from_area_tail`/`to_area_tail` are 0 for the uncontrolled Area1.
        other_area = bus_area_map[tie.to_bus_ix]
        if !tie.metered_from
            other_area = bus_area_map[tie.from_bus_ix]
        end
        metering_correction[other_area] -= loss_tie
    end

    name_to_controlled_area = Dict(a.name => a for a in data.area_interchange.areas)
    # ComplexF32 Y-bus storage rounds every primitive to ~1e-7 relative; this identity sums
    # ~15 primitives plus bus-level injections/withdrawals, so a few 1e-6 of rounding is
    # expected. atol picks a margin above the observed ~5e-7 max diff.
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

# Flipping a tie's `ext["metered_end"]` shifts its NI contribution by exactly the corridor's
# own active-power loss (P_from + P_to). Checked at a SINGLE fixed converged state via the
# independent oracle -- not by rebuild+resolve, since a different loss allocation needs a
# different ΔP_a.
@testset "area interchange tie identity: metered-end flip shifts NI by the tie's own loss" begin
    sys = _three_area_transfer_fixture(; slack_area3 = true)
    pf = ACPolarPowerFlow{NewtonRaphsonACPowerFlow}(; area_interchange_control = true)
    data = PowerFlowData(pf, sys)
    @test solve_power_flow!(data)

    bus_lookup = PF.get_bus_lookup(data)
    fix, tix = bus_lookup[9], bus_lookup[10]   # Line11: Area3 (Bus9) <-> Area2 (Bus10)
    tie = _find_tie(data.area_interchange.ties, fix, tix)
    @test tie.metered_from == true   # no ext set yet: defaults "from"

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

    # `_area_net_interchange` runs the exact accumulation `_set_area_tail_residuals!` uses.
    # A one-tie vector containing only `tie_from`/`tie_to` isolates the flip's effect: the
    # only thing differing is `metered_from`.
    area3_tail = tie.from_area_tail   # Area3, the from-side tail
    area2_tail = tie.to_area_tail     # Area2, the to-side tail
    NI3_from = PF._area_net_interchange([tie_from], PF.DCTie[], area3_tail, data, 1)
    NI3_to = PF._area_net_interchange([tie_to], PF.DCTie[], area3_tail, data, 1)
    NI2_from = PF._area_net_interchange([tie_from], PF.DCTie[], area2_tail, data, 1)
    NI2_to = PF._area_net_interchange([tie_to], PF.DCTie[], area2_tail, data, 1)

    @test isapprox(NI3_to - NI3_from, -loss; atol = 1e-6)
    @test isapprox(NI2_to - NI2_from, loss; atol = 1e-6)

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

# Guard 6 (`w_a > 0.9`) fires when an area's own buses hold nearly the whole network's raw
# slack-participation weight. Concentrate the gspf almost entirely on Area2's Bus6 (raw
# share ~0.93) with small remainders elsewhere.
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

    @test solve_power_flow!(data)
    ni = _oracle_ni_by_tail(sys, data, data.area_interchange.ties, 1)
    for area in data.area_interchange.areas
        @test isapprox(ni[area.tail_ix], area.pdes; atol = 1e-6)
    end
end

# The Jacobian sparse structure depends only on topology/REF layout/slack-participation
# PATTERN, none of which change across these 3 steps, so it must be memoized once and
# reused -- same object-identity check as the cache-reuse testset above.
@testset "area interchange multi-period: per-step convergence, delta_p, results, cache built once" begin
    sys = _three_area_transfer_fixture(; slack_area3 = true)
    pf = ACPolarPowerFlow{NewtonRaphsonACPowerFlow}(;
        area_interchange_control = true, time_steps = 3)
    data = PowerFlowData(pf, sys)
    @test PF.n_controlled_areas(data) == 2
    @test isnothing(data.ac_jacobian_structure_cache[])   # nothing built yet

    data.bus_active_power_withdrawals[:, 2] .+= 0.05
    data.bus_active_power_withdrawals[:, 3] .+= 0.10

    @test solve_power_flow!(data; time_steps = [1])
    cache1 = data.ac_jacobian_structure_cache[]
    @test !isnothing(cache1)

    # Solve the full 1:3 range (ts=1 re-solves warm/harmless; ts=2, ts=3 solve for the first
    # time) -- exercises genuine multi-period cache reuse across a single call.
    @test solve_power_flow!(data; time_steps = [1, 2, 3])

    @test data.ac_jacobian_structure_cache[] === cache1

    for t in 1:3
        ni = _oracle_ni_by_tail(sys, data, data.area_interchange.ties, t)
        for area in data.area_interchange.areas
            @test isapprox(ni[area.tail_ix], area.pdes; atol = 1e-6)
        end
    end

    dp1 = copy(data.area_interchange.delta_p[:, 1])
    dp2 = copy(data.area_interchange.delta_p[:, 2])
    dp3 = copy(data.area_interchange.delta_p[:, 3])
    @test !isapprox(dp1, dp2; atol = 1e-4)
    @test !isapprox(dp2, dp3; atol = 1e-4)
    @test !isapprox(dp1, dp3; atol = 1e-4)

    # `solve_power_flow` only supports single-period evaluation, so build the multi-period
    # table by calling `area_interchange_results_dataframe` once per time step and stacking.
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

@testset "area interchange LM greedy relax" begin
    sys = _weak_tie_three_area_fixture()
    pf_lm =
        ACPolarPowerFlow{LevenbergMarquardtACPowerFlow}(; area_interchange_control = true)
    data = PowerFlowData(pf_lm, sys)
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
        solve_power_flow(pf_lm, sys)
    )
    df = df_results["area_interchange_results"]
    @test nrow(df) == 2
    area2_row = only(filter(:area => ==("Area2"), df))
    area3_row = only(filter(:area => ==("Area3"), df))
    @test area2_row.schedule_status == :enforced
    @test isapprox(area2_row.ni_solved, area2_row.pdes; atol = 1e-3)
    @test area3_row.schedule_status == :relaxed
    @test area3_row.delta_p == 0.0
    @test area3_row.ni_solved < 0.5 * area3_row.pdes
end

@testset "area interchange LM multi period" begin
    sys = _three_area_transfer_fixture(; slack_area3 = true)
    pf_lm = ACPolarPowerFlow{LevenbergMarquardtACPowerFlow}(;
        area_interchange_control = true, time_steps = 3)
    data = PowerFlowData(pf_lm, sys)
    @test PF.n_controlled_areas(data) == 2

    # Different load perturbation per step -> different required ΔP_a redistribution
    # (same idiom as the NR multi-period tests above).
    data.bus_active_power_withdrawals[:, 2] .+= 0.05
    data.bus_active_power_withdrawals[:, 3] .+= 0.10

    for t in 1:3
        @test solve_power_flow!(data; time_steps = [t])
        ni = _oracle_ni_by_tail(sys, data, data.area_interchange.ties, t)
        for area in data.area_interchange.areas
            @test isapprox(ni[area.tail_ix], area.pdes; atol = 1e-6)
        end
    end

    dp1 = copy(data.area_interchange.delta_p[:, 1])
    dp2 = copy(data.area_interchange.delta_p[:, 2])
    dp3 = copy(data.area_interchange.delta_p[:, 3])
    @test !isapprox(dp1, dp2; atol = 1e-4)
    @test !isapprox(dp2, dp3; atol = 1e-4)
    @test !isapprox(dp1, dp3; atol = 1e-4)

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

@testset "area interchange LM diagnostics border" begin
    # `_normal_two_area_fixture` keeps this end-to-end LM solve fast while still exercising
    # the bordered square Jacobian's area-interchange tail. `log_solver_diagnostics`
    # activates the same Schur-complement λ_min diagnostic under LM as under NR/TR.
    sys = _normal_two_area_fixture()
    pf_lm = ACPolarPowerFlow{LevenbergMarquardtACPowerFlow}(;
        area_interchange_control = true,
        log_solver_diagnostics = true,
    )
    data = PowerFlowData(pf_lm, sys)
    @test PF.n_controlled_areas(data) == 2

    # Regression, solver-agnostic: `_describe_residual_entry` used to assume every residual
    # row past the bus block belonged to the LCC tail, throwing `BoundsError` whenever
    # ‖F‖∞ landed on an area-interchange row. Fixed by adding an area-tail branch.
    tl = Test.TestLogger(; min_level = Logging.Info)
    converged = Logging.with_logger(tl) do
        solve_power_flow!(data)
    end
    @test converged
    lines = [r.message for r in tl.logs if occursin(r"LM iter \d+", r.message)]
    @test !isempty(lines)
    for line in lines
        @test occursin("λ_min(S) = ", line)
        @test !occursin("λ_min(S) = not-converged", line)
    end
    # Stronger: at least one monitor line must land on an area-interchange row and label it
    # as such (not as an LCC row), proving the area tail is no longer mislabeled/overrun.
    @test any(occursin("area ", line) && occursin("NI−PDES", line) for line in lines)
end

@testset "area interchange FD greedy relax" begin
    sys_fd = _weak_tie_three_area_fixture()
    pf_fd = ACPolarPowerFlow{FastDecoupledACPowerFlow{FDDecoupled, FDSchemeXB}}(;
        area_interchange_control = true,
    )
    data_fd = PowerFlowData(pf_fd, sys_fd)
    @test PF.n_controlled_areas(data_fd) == 2

    converged_fd = @test_logs(
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
        solve_power_flow!(data_fd)
    )
    @test converged_fd
    @test length(data_fd.area_interchange.areas) == 1
    @test only(data_fd.area_interchange.areas).name == "Area2"
    @test haskey(data_fd.area_interchange.relaxed, 1)
    @test only(data_fd.area_interchange.relaxed[1]).name == "Area3"

    df_results_fd = @test_logs(
        (:error, r"solver failed to converge"),
        (:error, r"Area interchange:.*Area3.*de-enrolling"),
        (
            :error,
            r"Area interchange:.*converged only after relaxing 1 area.*Area3 " *
            r"\(ni_solved=.*, pdes=.*, gap=.*\)",
        ),
        match_mode = :any,
        min_level = Logging.Warn,
        solve_power_flow(pf_fd, sys_fd)
    )
    df_fd = df_results_fd["area_interchange_results"]
    @test nrow(df_fd) == 2
    area2_row_fd = only(filter(:area => ==("Area2"), df_fd))
    area3_row_fd = only(filter(:area => ==("Area3"), df_fd))
    @test area2_row_fd.schedule_status == :enforced
    @test isapprox(area2_row_fd.ni_solved, area2_row_fd.pdes; atol = 1e-3)
    @test area3_row_fd.schedule_status == :relaxed
    @test area3_row_fd.delta_p == 0.0
    @test area3_row_fd.ni_solved < 0.5 * area3_row_fd.pdes

    sys_fdfj = _weak_tie_three_area_fixture()
    pf_fdfj = ACPolarPowerFlow{FastDecoupledACPowerFlow{FDFixedJacobian, FDSchemeXB}}(;
        area_interchange_control = true,
    )
    data_fdfj = PowerFlowData(pf_fdfj, sys_fdfj)
    @test PF.n_controlled_areas(data_fdfj) == 2

    converged_fdfj = @test_logs(
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
        solve_power_flow!(data_fdfj)
    )
    @test converged_fdfj
    @test length(data_fdfj.area_interchange.areas) == 1
    @test only(data_fdfj.area_interchange.areas).name == "Area2"
    @test haskey(data_fdfj.area_interchange.relaxed, 1)
    @test only(data_fdfj.area_interchange.relaxed[1]).name == "Area3"

    df_results_fdfj = @test_logs(
        (:error, r"solver failed to converge"),
        (:error, r"Area interchange:.*Area3.*de-enrolling"),
        (
            :error,
            r"Area interchange:.*converged only after relaxing 1 area.*Area3 " *
            r"\(ni_solved=.*, pdes=.*, gap=.*\)",
        ),
        match_mode = :any,
        min_level = Logging.Warn,
        solve_power_flow(pf_fdfj, sys_fdfj)
    )
    df_fdfj = df_results_fdfj["area_interchange_results"]
    @test nrow(df_fdfj) == 2
    area2_row_fdfj = only(filter(:area => ==("Area2"), df_fdfj))
    area3_row_fdfj = only(filter(:area => ==("Area3"), df_fdfj))
    @test area2_row_fdfj.schedule_status == :enforced
    @test isapprox(area2_row_fdfj.ni_solved, area2_row_fdfj.pdes; atol = 1e-3)
    @test area3_row_fdfj.schedule_status == :relaxed
    @test area3_row_fdfj.delta_p == 0.0
    @test area3_row_fdfj.ni_solved < 0.5 * area3_row_fdfj.pdes
end

# `_deenroll_area!` nulls `ac_jacobian_structure_cache`/`polar_nr_cache`, but the FD
# bordered-Schur scratch lives in `data.solver_cache[]`, sized to `n_controlled_areas(data)`
# -- not part of `FDCacheKey` -- so de-enrollment doesn't naturally invalidate it. This is
# the regression proof the resize path works.
@testset "area interchange FD cache invariant after relax" begin
    sys = _weak_tie_three_area_fixture()
    pf_fd = ACPolarPowerFlow{FastDecoupledACPowerFlow{FDDecoupled, FDSchemeXB}}(;
        area_interchange_control = true,
    )
    data = PowerFlowData(pf_fd, sys)
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
    n_areas_post_relax = PF.n_controlled_areas(data)
    @test n_areas_post_relax == 1

    cache = data.solver_cache[]
    @test cache isa PF.FastDecoupledCache
    @test size(cache.area_S) == (n_areas_post_relax, n_areas_post_relax)
    @test size(cache.area_W, 2) == n_areas_post_relax
    @test size(cache.area_W, 1) == length(cache.fd.pvpq)
    @test length(cache.area_g) == n_areas_post_relax

    @test solve_power_flow!(data)
end

@testset "area interchange FD multi period" begin
    sys_fd = _three_area_transfer_fixture(; slack_area3 = true)
    pf_fd = ACPolarPowerFlow{FastDecoupledACPowerFlow{FDDecoupled, FDSchemeXB}}(;
        area_interchange_control = true, time_steps = 3)
    data_fd = PowerFlowData(pf_fd, sys_fd)
    @test PF.n_controlled_areas(data_fd) == 2

    data_fd.bus_active_power_withdrawals[:, 2] .+= 0.05
    data_fd.bus_active_power_withdrawals[:, 3] .+= 0.10

    for t in 1:3
        @test solve_power_flow!(data_fd; time_steps = [t])
        ni = _oracle_ni_by_tail(sys_fd, data_fd, data_fd.area_interchange.ties, t)
        for area in data_fd.area_interchange.areas
            @test isapprox(
                ni[area.tail_ix], area.pdes; atol = data_fd.area_interchange.tolerance)
        end
    end

    cache_fd = data_fd.solver_cache[]
    @test cache_fd isa PF.FastDecoupledCache
    @test cache_fd.bp_factor_count == 1

    dp1_fd = copy(data_fd.area_interchange.delta_p[:, 1])
    dp2_fd = copy(data_fd.area_interchange.delta_p[:, 2])
    dp3_fd = copy(data_fd.area_interchange.delta_p[:, 3])
    @test !isapprox(dp1_fd, dp2_fd; atol = 1e-4)
    @test !isapprox(dp2_fd, dp3_fd; atol = 1e-4)
    @test !isapprox(dp1_fd, dp3_fd; atol = 1e-4)

    dfs_fd = DataFrame[]
    for t in 1:3
        df_t = PF.area_interchange_results_dataframe(sys_fd, data_fd, t)
        df_t[!, :time_step] .= t
        push!(dfs_fd, df_t)
    end
    df_fd = vcat(dfs_fd...)
    @test nrow(df_fd) == 6
    for area_name in ("Area2", "Area3")
        @test nrow(filter(:area => ==(area_name), df_fd)) == 3
    end

    sys_fdfj = _three_area_transfer_fixture(; slack_area3 = true)
    pf_fdfj = ACPolarPowerFlow{FastDecoupledACPowerFlow{FDFixedJacobian, FDSchemeXB}}(;
        area_interchange_control = true, time_steps = 3)
    data_fdfj = PowerFlowData(pf_fdfj, sys_fdfj)
    @test PF.n_controlled_areas(data_fdfj) == 2

    data_fdfj.bus_active_power_withdrawals[:, 2] .+= 0.05
    data_fdfj.bus_active_power_withdrawals[:, 3] .+= 0.10

    for t in 1:3
        @test solve_power_flow!(data_fdfj; time_steps = [t])
        ni = _oracle_ni_by_tail(sys_fdfj, data_fdfj, data_fdfj.area_interchange.ties, t)
        for area in data_fdfj.area_interchange.areas
            @test isapprox(
                ni[area.tail_ix], area.pdes; atol = data_fdfj.area_interchange.tolerance)
        end
    end

    dp1_fdfj = copy(data_fdfj.area_interchange.delta_p[:, 1])
    dp2_fdfj = copy(data_fdfj.area_interchange.delta_p[:, 2])
    dp3_fdfj = copy(data_fdfj.area_interchange.delta_p[:, 3])
    @test !isapprox(dp1_fdfj, dp2_fdfj; atol = 1e-4)
    @test !isapprox(dp2_fdfj, dp3_fdfj; atol = 1e-4)
    @test !isapprox(dp1_fdfj, dp3_fdfj; atol = 1e-4)

    dfs_fdfj = DataFrame[]
    for t in 1:3
        df_t = PF.area_interchange_results_dataframe(sys_fdfj, data_fdfj, t)
        df_t[!, :time_step] .= t
        push!(dfs_fdfj, df_t)
    end
    df_fdfj = vcat(dfs_fdfj...)
    @test nrow(df_fdfj) == 6
    for area_name in ("Area2", "Area3")
        @test nrow(filter(:area => ==(area_name), df_fdfj)) == 3
    end
end

@testset "area interchange DC comprehensive fixture" begin
    sys = _comprehensive_area_dc_fixture()

    @test length(collect(PSY.get_components(PSY.Area, sys))) >= 3
    @test !isempty(PSY.get_components(PSY.TwoTerminalLCCLine, sys))
    @test !isempty(PSY.get_components(PSY.TwoTerminalVSCLine, sys))
    @test !isempty(PSY.get_components(PSY.ThreeWindingTransformer, sys))
    @test !isempty(PSY.get_components(PSY.SwitchedAdmittance, sys))
    @test !isempty(PSY.get_components(PSY.DiscreteControlledACBranch, sys))
    @test !isempty(PSY.get_components(PSY.TapTransformer, sys))

    lcc = only(PSY.get_components(PSY.TwoTerminalLCCLine, sys))
    vsc = only(PSY.get_components(PSY.TwoTerminalVSCLine, sys))

    pf_on =
        ACPolarPowerFlow{NewtonRaphsonACPowerFlow}(; area_interchange_control = true)
    data_on = PowerFlowData(pf_on, sys)
    @test PF.n_controlled_areas(data_on) == 2

    # Check on the REDUCED network, not the raw `PSY.Arc`: VSC AC terminals are protected as
    # `irreducible_buses`, but LCC terminals are NOT, so a raw-arc check would keep passing
    # even if a future reduction merged an LCC terminal across the boundary.
    bus_lookup = PF.get_bus_lookup(data_on)
    nrd = PF.get_network_reduction_data(data_on)
    reverse_bus_search_map = PNM.get_reverse_bus_search_map(nrd)
    for arc in (PSY.get_arc(lcc), PSY.get_arc(vsc))
        from_num = PSY.get_number(PSY.get_from(arc))
        to_num = PSY.get_number(PSY.get_to(arc))
        from_ix = PF._resolve_bus_ix(bus_lookup, reverse_bus_search_map, from_num)
        to_ix = PF._resolve_bus_ix(bus_lookup, reverse_bus_search_map, to_num)
        @test !isnothing(from_ix)
        @test !isnothing(to_ix)
        @test from_ix != to_ix
        from_parent = get(reverse_bus_search_map, from_num, from_num)
        to_parent = get(reverse_bus_search_map, to_num, to_num)
        from_area = PSY.get_area(PSY.get_bus(sys, from_parent))
        to_area = PSY.get_area(PSY.get_bus(sys, to_parent))
        @test from_area !== to_area
    end

    @test solve_power_flow!(data_on)
    @test all(data_on.converged)

    # The LCC tail residual pins alpha_r/alpha_i to the LCC's configured `.min` angle
    # unconditionally, so this fixture's angle limits were retuned off the defaults (which
    # sit AT the arccos clamp boundary) to a realistic operating point with margin.
    alpha_r = PF.get_lcc_rectifier_thyristor_angle(data_on)[1, 1]
    alpha_i = PF.get_lcc_inverter_thyristor_angle(data_on)[1, 1]
    r_limits = PSY.get_rectifier_delay_angle_limits(lcc)
    i_limits = PSY.get_inverter_extinction_angle_limits(lcc)
    @test r_limits.min <= alpha_r <= r_limits.max
    @test i_limits.min <= alpha_i <= i_limits.max
    margin = deg2rad(5.0)
    lo = PF.LCC_SMALL_ANGLE_THRESHOLD + margin
    hi = π / 2 - PF.LCC_SMALL_ANGLE_THRESHOLD - margin
    @test lo < alpha_r < hi
    @test lo < alpha_i < hi

    pf_off = ACPolarPowerFlow{NewtonRaphsonACPowerFlow}()
    data_off = PowerFlowData(pf_off, sys)
    @test PF.n_controlled_areas(data_off) == 0
    @test solve_power_flow!(data_off)
    @test all(data_off.converged)
end

@testset "area interchange DC tie enumeration" begin
    sys = _comprehensive_area_dc_fixture()
    lcc = only(PSY.get_components(PSY.TwoTerminalLCCLine, sys))
    vsc = only(PSY.get_components(PSY.TwoTerminalVSCLine, sys))

    pf_on = ACPolarPowerFlow{NewtonRaphsonACPowerFlow}(; area_interchange_control = true)
    data_on = PowerFlowData(pf_on, sys)
    @test PF.n_controlled_areas(data_on) == 2
    area_tail = Dict(a.name => a.tail_ix for a in data_on.area_interchange.areas)
    @test area_tail == Dict("Area2" => 1, "Area3" => 2)

    bus_lookup = PF.get_bus_lookup(data_on)
    nrd = PF.get_network_reduction_data(data_on)
    reverse_bus_search_map = PNM.get_reverse_bus_search_map(nrd)
    lcc_from_ix = PF._resolve_bus_ix(
        bus_lookup, reverse_bus_search_map,
        PSY.get_number(PSY.get_from(PSY.get_arc(lcc))))
    lcc_to_ix = PF._resolve_bus_ix(
        bus_lookup, reverse_bus_search_map, PSY.get_number(PSY.get_to(PSY.get_arc(lcc))))
    vsc_from_ix = PF._resolve_bus_ix(
        bus_lookup, reverse_bus_search_map,
        PSY.get_number(PSY.get_from(PSY.get_arc(vsc))))
    vsc_to_ix = PF._resolve_bus_ix(
        bus_lookup, reverse_bus_search_map, PSY.get_number(PSY.get_to(PSY.get_arc(vsc))))

    dc_ties = data_on.area_interchange.dc_ties
    @test length(dc_ties) == 2

    lcc_tie = only(filter(t -> t.kind == PF.DC_TIE_LCC, dc_ties))
    @test lcc_tie.lcc_ix == 1
    @test (lcc_tie.from_conv_ix, lcc_tie.to_conv_ix) == (0, 0)
    @test (lcc_tie.from_bus_ix, lcc_tie.to_bus_ix) == (lcc_from_ix, lcc_to_ix)
    @test lcc_tie.metered_from == true   # no ext["metered_end"] set -> defaults from
    @test lcc_tie.from_area_tail == 0    # Area1 (rectifier side): uncontrolled
    @test lcc_tie.to_area_tail == 2      # Area3 (inverter side): tail 2

    vsc_tie = only(filter(t -> t.kind == PF.DC_TIE_VSC, dc_ties))
    @test vsc_tie.lcc_ix == 0
    @test (vsc_tie.from_conv_ix, vsc_tie.to_conv_ix) == (1, 2)
    @test (vsc_tie.from_bus_ix, vsc_tie.to_bus_ix) == (vsc_from_ix, vsc_to_ix)
    @test vsc_tie.metered_from == true
    @test vsc_tie.from_area_tail == 1    # Area2
    @test vsc_tie.to_area_tail == 2      # Area3

    ac_ties = data_on.area_interchange.ties
    @test length(ac_ties) == 4
    ac_tail_pairs = sort([(t.from_area_tail, t.to_area_tail) for t in ac_ties])
    @test ac_tail_pairs == [(0, 1), (0, 1), (1, 0), (1, 2)]

    # A DC tie's in-area terminal coinciding with that area's own slack bus can't
    # double-count: `_lcc_dc_ties`/`_vsc_dc_ties` are driven purely by `bus_area_map`, never
    # by `ControlledArea.slack_bus_ix`.
    area2_slack_ix = only(
        a.slack_bus_ix for a in data_on.area_interchange.areas if a.name == "Area2")
    @test ac_ties[1].to_bus_ix == area2_slack_ix
    @test ac_ties[1].to_area_tail == 1

    # Interior DC link (same-area both terminals) contributes NO tie. Force both terminals
    # into the SAME tail via a synthetic `bus_area_map` rather than a second fixture (user
    # directive: one fixture).
    removed_buses = PNM.get_removed_buses(nrd)
    same_area_map = Dict(lcc_from_ix => 1, lcc_to_ix => 1, vsc_from_ix => 1, vsc_to_ix => 1)
    @test isempty(PF._lcc_dc_ties(sys, data_on.lcc, removed_buses, same_area_map))
    @test isempty(
        PF._vsc_dc_ties(sys, PF.get_dc_network(data_on), removed_buses, same_area_map))
    @test isempty(
        PF.build_dc_ties(
            sys, data_on.lcc, PF.get_dc_network(data_on), nrd, same_area_map),
    )

    sys_ac = _three_area_transfer_fixture(; slack_area3 = true)
    pf_ac = ACPolarPowerFlow{NewtonRaphsonACPowerFlow}(; area_interchange_control = true)
    data_ac = PowerFlowData(pf_ac, sys_ac)
    @test PF.n_controlled_areas(data_ac) == 2
    @test isempty(data_ac.area_interchange.dc_ties)
    @test isempty(data_ac.area_interchange.pristine_dc_ties)
    @test !isempty(data_ac.area_interchange.ties)  # sanity: AC ties still enumerate
end

# Independent oracle for one DC tie's metered-end active power (the DC analogue of
# `_oracle_tie_metered_power`): LCC recomputes P_lcc_from/to from `data.lcc` state; VSC negates
# the metered converter's bus-injection-signed `P_c`. A SEPARATE code path from the residual
# kernel, so a sign / tail-routing / metered-end / kind-dispatch wiring bug surfaces here.
function _oracle_dc_tie_metered_power(data, dcn, tie::PF.DCTie, time_step::Int)
    if tie.kind == PF.DC_TIE_LCC
        i = tie.lcc_ix
        Vm = view(data.bus_magnitude, :, time_step)
        i_dc = data.lcc.i_dc[i, time_step]
        P_from =
            Vm[tie.from_bus_ix] * data.lcc.rectifier.tap[i, time_step] *
            PF.SQRT6_DIV_PI * i_dc * cos(data.lcc.rectifier.phi[i, time_step])
        P_to =
            Vm[tie.to_bus_ix] * data.lcc.inverter.tap[i, time_step] *
            PF.SQRT6_DIV_PI * i_dc * cos(data.lcc.inverter.phi[i, time_step])
        if tie.metered_from
            return P_from
        end
        return P_to
    end
    if tie.metered_from
        return -dcn.p_c[tie.from_conv_ix, time_step]
    end
    return -dcn.p_c[tie.to_conv_ix, time_step]
end

@testset "area interchange DC residual matches oracle NI" begin
    # Both metered-end orientations of the LCC tie -- "to" flips the DC tie to
    # inverter-metered (the oracle must then pick the inverter terminal's power).
    for lcc_metered_end in ("from", "to")
        sys = _comprehensive_area_dc_fixture(; lcc_metered_end = lcc_metered_end)
        data = PowerFlowData(
            ACPolarPowerFlow{NewtonRaphsonACPowerFlow}(; area_interchange_control = true),
            sys)
        @test PF.n_controlled_areas(data) == 2

        lcc_tie = only(filter(t -> t.kind == PF.DC_TIE_LCC, data.area_interchange.dc_ties))
        expected_metered_from = lcc_metered_end == "from"
        @test lcc_tie.metered_from == expected_metered_from

        residual = PF.ACPowerFlowResidual(data, 1)
        x0 = PF.calculate_x0(data, 1)
        residual(x0, 1)
        F = residual.Rv
        dcn = PF.get_dc_network(data)
        area_off = PF.area_tail_offset(data, dcn)

        n = PF.n_controlled_areas(data)
        ni = zeros(n)
        dc_only = zeros(n)
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
        @test !isempty(data.area_interchange.dc_ties)
        for tie in data.area_interchange.dc_ties
            P_conv = _oracle_dc_tie_metered_power(data, dcn, tie, 1)
            metered_tail = tie.from_area_tail
            other_tail = tie.to_area_tail
            if !tie.metered_from
                metered_tail = tie.to_area_tail
                other_tail = tie.from_area_tail
            end
            if !iszero(metered_tail)
                ni[metered_tail] += P_conv
                dc_only[metered_tail] += P_conv
            end
            if !iszero(other_tail)
                ni[other_tail] -= P_conv
                dc_only[other_tail] -= P_conv
            end
        end

        # Non-vacuity: the DC ties actually move a tracked area's NI, so the equality below
        # genuinely exercises the new DC-tie residual term (RED without it: F omits the term
        # while the oracle includes it).
        @test maximum(abs, dc_only) > 1e-6

        for area in data.area_interchange.areas
            @test F[area_off + area.tail_ix] ≈ ni[area.tail_ix] - area.pdes atol = 1e-6
        end
    end
end

@testset "area interchange DC jacobian matches FD" begin
    # The analytic area rows must carry the DC-tie cross-derivatives (LCC ∂P/∂(V,t,α), VSC
    # ∂P_conv/∂P_c). Both metered-end orientations are tested: "to" flips to the
    # inverter-metered branch, previously untested by any FD gate.
    for lcc_metered_end in ("from", "to")
        sys = _comprehensive_area_dc_fixture(; lcc_metered_end = lcc_metered_end)
        data = PowerFlowData(
            ACPolarPowerFlow{NewtonRaphsonACPowerFlow}(; area_interchange_control = true),
            sys)
        residual = PF.ACPowerFlowResidual(data, 1)
        jac = PF.ACPowerFlowJacobian(residual, 1)
        x0 = PF.calculate_x0(data, 1)
        # Perturbed, non-solution state so bus Vm/θ AND the DC-tail columns (LCC tap/α, VSC
        # P_c) all carry nonzero sensitivity through the area-interchange NI rows.
        x = x0 .+ 0.02 .* sin.(1:length(x0))
        residual(x, 1)
        jac(1)
        verify_jacobian_asymptotic(
            residual, jac.Jv, x, 1;
            label = "area interchange DC ($lcc_metered_end-metered)")
    end
end

@testset "area interchange DC LM parity" begin
    # Both metered-end orientations of the LCC tie -- see the jacobian-FD testset above for
    # why the "to" (inverter-metered) case matters.
    for lcc_metered_end in ("from", "to")
        sys_nr = _comprehensive_area_dc_fixture(; lcc_metered_end = lcc_metered_end)
        pf_nr =
            ACPolarPowerFlow{NewtonRaphsonACPowerFlow}(; area_interchange_control = true)
        data_nr = PowerFlowData(pf_nr, sys_nr)
        @test solve_power_flow!(data_nr)

        sys_lm = _comprehensive_area_dc_fixture(; lcc_metered_end = lcc_metered_end)
        pf_lm =
            ACPolarPowerFlow{LevenbergMarquardtACPowerFlow}(;
                area_interchange_control = true)
        data_lm = PowerFlowData(pf_lm, sys_lm)
        @test solve_power_flow!(data_lm)

        @test !isempty(data_lm.area_interchange.dc_ties)

        @test isapprox(
            data_nr.bus_magnitude[:, 1],
            data_lm.bus_magnitude[:, 1];
            atol = 1e-6,
        )
        @test isapprox(data_nr.bus_angles[:, 1], data_lm.bus_angles[:, 1]; atol = 1e-6)

        @test isapprox(
            data_nr.lcc.rectifier.tap[:, 1],
            data_lm.lcc.rectifier.tap[:, 1];
            atol = 1e-6,
        )
        @test isapprox(
            data_nr.lcc.inverter.tap[:, 1],
            data_lm.lcc.inverter.tap[:, 1];
            atol = 1e-6,
        )
        @test isapprox(
            data_nr.lcc.rectifier.thyristor_angle[:, 1],
            data_lm.lcc.rectifier.thyristor_angle[:, 1];
            atol = 1e-6,
        )
        @test isapprox(
            data_nr.lcc.inverter.thyristor_angle[:, 1],
            data_lm.lcc.inverter.thyristor_angle[:, 1];
            atol = 1e-6,
        )

        dcn_nr = PF.get_dc_network(data_nr)
        dcn_lm = PF.get_dc_network(data_lm)
        @test isapprox(dcn_nr.p_c[:, 1], dcn_lm.p_c[:, 1]; atol = 1e-6)

        for (area_nr, area_lm) in
            zip(data_nr.area_interchange.areas, data_lm.area_interchange.areas)
            @test isapprox(
                data_nr.area_interchange.delta_p[area_nr.tail_ix, 1],
                data_lm.area_interchange.delta_p[area_lm.tail_ix, 1];
                atol = 1e-6,
            )
        end

        # Same oracle as "area interchange DC residual matches oracle NI": AC-tie metered
        # power plus the DC-tie (LCC/VSC) metered converter power, summed by controlled-area
        # tail.
        n = PF.n_controlled_areas(data_lm)
        ni = zeros(n)
        for tie in data_lm.area_interchange.ties
            P_m = _oracle_tie_metered_power(sys_lm, data_lm, tie, 1)
            metered_tail = tie.from_area_tail
            other_tail = tie.to_area_tail
            if !tie.metered_from
                metered_tail = tie.to_area_tail
                other_tail = tie.from_area_tail
            end
            iszero(metered_tail) || (ni[metered_tail] += P_m)
            iszero(other_tail) || (ni[other_tail] -= P_m)
        end
        for tie in data_lm.area_interchange.dc_ties
            P_conv = _oracle_dc_tie_metered_power(data_lm, dcn_lm, tie, 1)
            metered_tail = tie.from_area_tail
            other_tail = tie.to_area_tail
            if !tie.metered_from
                metered_tail = tie.to_area_tail
                other_tail = tie.from_area_tail
            end
            iszero(metered_tail) || (ni[metered_tail] += P_conv)
            iszero(other_tail) || (ni[other_tail] -= P_conv)
        end

        for area in data_lm.area_interchange.areas
            @test isapprox(ni[area.tail_ix], area.pdes; atol = 1e-5)
        end
    end
end

@testset "area interchange DC FD fixed-jacobian parity" begin
    sys_nr = _comprehensive_area_dc_fixture()
    pf_nr = ACPolarPowerFlow{NewtonRaphsonACPowerFlow}(; area_interchange_control = true)
    data_nr = PowerFlowData(pf_nr, sys_nr)
    @test solve_power_flow!(data_nr)

    sys_fdfj = _comprehensive_area_dc_fixture()
    pf_fdfj = ACPolarPowerFlow{FastDecoupledACPowerFlow{FDFixedJacobian, FDSchemeXB}}(;
        area_interchange_control = true,
    )
    data_fdfj = PowerFlowData(pf_fdfj, sys_fdfj)
    @test solve_power_flow!(data_fdfj)

    @test !isempty(data_fdfj.area_interchange.dc_ties)

    @test isapprox(data_nr.bus_magnitude[:, 1], data_fdfj.bus_magnitude[:, 1]; atol = 1e-5)
    @test isapprox(data_nr.bus_angles[:, 1], data_fdfj.bus_angles[:, 1]; atol = 1e-5)

    @test isapprox(
        data_nr.lcc.rectifier.tap[:, 1],
        data_fdfj.lcc.rectifier.tap[:, 1];
        atol = 1e-5,
    )
    @test isapprox(
        data_nr.lcc.inverter.tap[:, 1],
        data_fdfj.lcc.inverter.tap[:, 1];
        atol = 1e-5,
    )
    @test isapprox(
        data_nr.lcc.rectifier.thyristor_angle[:, 1],
        data_fdfj.lcc.rectifier.thyristor_angle[:, 1];
        atol = 1e-5,
    )
    @test isapprox(
        data_nr.lcc.inverter.thyristor_angle[:, 1],
        data_fdfj.lcc.inverter.thyristor_angle[:, 1];
        atol = 1e-5,
    )

    dcn_nr = PF.get_dc_network(data_nr)
    dcn_fdfj = PF.get_dc_network(data_fdfj)
    @test isapprox(dcn_nr.p_c[:, 1], dcn_fdfj.p_c[:, 1]; atol = 1e-5)

    for (area_nr, area_fdfj) in
        zip(data_nr.area_interchange.areas, data_fdfj.area_interchange.areas)
        @test isapprox(
            data_nr.area_interchange.delta_p[area_nr.tail_ix, 1],
            data_fdfj.area_interchange.delta_p[area_fdfj.tail_ix, 1];
            atol = 1e-5,
        )
    end

    n = PF.n_controlled_areas(data_fdfj)
    ni = zeros(n)
    for tie in data_fdfj.area_interchange.ties
        P_m = _oracle_tie_metered_power(sys_fdfj, data_fdfj, tie, 1)
        metered_tail = tie.from_area_tail
        other_tail = tie.to_area_tail
        if !tie.metered_from
            metered_tail = tie.to_area_tail
            other_tail = tie.from_area_tail
        end
        iszero(metered_tail) || (ni[metered_tail] += P_m)
        iszero(other_tail) || (ni[other_tail] -= P_m)
    end
    for tie in data_fdfj.area_interchange.dc_ties
        P_conv = _oracle_dc_tie_metered_power(data_fdfj, dcn_fdfj, tie, 1)
        metered_tail = tie.from_area_tail
        other_tail = tie.to_area_tail
        if !tie.metered_from
            metered_tail = tie.to_area_tail
            other_tail = tie.from_area_tail
        end
        iszero(metered_tail) || (ni[metered_tail] += P_conv)
        iszero(other_tail) || (ni[other_tail] -= P_conv)
    end

    for area in data_fdfj.area_interchange.areas
        @test isapprox(ni[area.tail_ix], area.pdes; atol = 1e-5)
    end
end

@testset "area interchange DC FD decoupled parity" begin
    sys_nr = _comprehensive_area_dc_fixture()
    pf_nr = ACPolarPowerFlow{NewtonRaphsonACPowerFlow}(; area_interchange_control = true)
    data_nr = PowerFlowData(pf_nr, sys_nr)
    @test solve_power_flow!(data_nr)

    sys_fd = _comprehensive_area_dc_fixture()
    pf_fd = ACPolarPowerFlow{FastDecoupledACPowerFlow{PF.FDDecoupled, PF.FDSchemeXB}}(;
        area_interchange_control = true,
    )
    data_fd = PowerFlowData(pf_fd, sys_fd)
    @test solve_power_flow!(data_fd)

    @test !isempty(data_fd.area_interchange.dc_ties)

    @test isapprox(data_nr.bus_magnitude[:, 1], data_fd.bus_magnitude[:, 1]; atol = 1e-5)
    @test isapprox(data_nr.bus_angles[:, 1], data_fd.bus_angles[:, 1]; atol = 1e-5)

    @test isapprox(
        data_nr.lcc.rectifier.tap[:, 1],
        data_fd.lcc.rectifier.tap[:, 1];
        atol = 1e-5,
    )
    @test isapprox(
        data_nr.lcc.inverter.tap[:, 1],
        data_fd.lcc.inverter.tap[:, 1];
        atol = 1e-5,
    )
    @test isapprox(
        data_nr.lcc.rectifier.thyristor_angle[:, 1],
        data_fd.lcc.rectifier.thyristor_angle[:, 1];
        atol = 1e-5,
    )
    @test isapprox(
        data_nr.lcc.inverter.thyristor_angle[:, 1],
        data_fd.lcc.inverter.thyristor_angle[:, 1];
        atol = 1e-5,
    )

    dcn_nr = PF.get_dc_network(data_nr)
    dcn_fd = PF.get_dc_network(data_fd)
    @test isapprox(dcn_nr.p_c[:, 1], dcn_fd.p_c[:, 1]; atol = 1e-5)

    for (area_nr, area_fd) in
        zip(data_nr.area_interchange.areas, data_fd.area_interchange.areas)
        @test isapprox(
            data_nr.area_interchange.delta_p[area_nr.tail_ix, 1],
            data_fd.area_interchange.delta_p[area_fd.tail_ix, 1];
            atol = 1e-5,
        )
    end

    cache = data_fd.solver_cache[]
    @test cache.bp_factor_count == 1

    n = PF.n_controlled_areas(data_fd)
    ni = zeros(n)
    for tie in data_fd.area_interchange.ties
        P_m = _oracle_tie_metered_power(sys_fd, data_fd, tie, 1)
        metered_tail = tie.from_area_tail
        other_tail = tie.to_area_tail
        if !tie.metered_from
            metered_tail = tie.to_area_tail
            other_tail = tie.from_area_tail
        end
        iszero(metered_tail) || (ni[metered_tail] += P_m)
        iszero(other_tail) || (ni[other_tail] -= P_m)
    end
    for tie in data_fd.area_interchange.dc_ties
        P_conv = _oracle_dc_tie_metered_power(data_fd, dcn_fd, tie, 1)
        metered_tail = tie.from_area_tail
        other_tail = tie.to_area_tail
        if !tie.metered_from
            metered_tail = tie.to_area_tail
            other_tail = tie.from_area_tail
        end
        iszero(metered_tail) || (ni[metered_tail] += P_conv)
        iszero(other_tail) || (ni[other_tail] -= P_conv)
    end

    for area in data_fd.area_interchange.areas
        @test isapprox(ni[area.tail_ix], area.pdes; atol = 1e-5)
    end
end

"""Zero-impedance branch between an LCC's rectifier/inverter buses so PNM's reduction
genuinely merges them (`fix == tix`) -- the self-tie merge guard case, distinct from the
"interior DC link" test (same-tail on two DIFFERENT buses, not a merge)."""
function _lcc_self_merge_fixture()
    sys = System(100.0)
    area_a = PSY.Area(; name = "AreaA")
    area_b = PSY.Area(; name = "AreaB")
    PSY.add_component!(sys, area_a)
    PSY.add_component!(sys, area_b)

    b1 = _add_simple_bus!(sys, 1, ACBusTypes.REF, 230)
    b2 = _add_simple_bus!(sys, 2, ACBusTypes.PQ, 230)
    b3 = _add_simple_bus!(sys, 3, ACBusTypes.PQ, 230)
    PSY.set_area!(b1, area_a)
    PSY.set_area!(b2, area_a)
    PSY.set_area!(b3, area_b)

    _add_simple_source!(sys, b1, 0.0, 0.0)
    _add_simple_load!(sys, b2, 10.0, 5.0)
    _add_simple_load!(sys, b3, 60.0, 20.0)
    _add_simple_line!(sys, b1, b2, 5e-3, 5e-3, 1e-3)
    _add_simple_line!(sys, b1, b3, 5e-3, 5e-3, 1e-3)
    _add_simple_line!(sys, b2, b3, 0.0, 0.0)  # zero-impedance -- merges the LCC's own terminals

    _add_simple_lcc!(sys, b2, b3, 0.05, 0.05, 0.08)
    return sys
end

@testset "area interchange DC tie self-merge guard: zero-impedance reduction merges an LCC's own terminals" begin
    sys = _lcc_self_merge_fixture()
    pf = ACPolarPowerFlow{NewtonRaphsonACPowerFlow}(; area_interchange_control = true)
    data = PowerFlowData(pf, sys)

    # Confirm this is a REAL merge (genuinely produced by PNM's zero-impedance reduction),
    # not a hand-poked `bus_indices` -- `initialize_LCCParameters!` already resolved the
    # rectifier/inverter pair onto the SAME reduced bus.
    (fix, tix) = only(data.lcc.bus_indices)
    @test fix == tix

    nrd = PF.get_network_reduction_data(data)
    removed_buses = PNM.get_removed_buses(nrd)
    dc_ties = @test_logs(
        (:warn, r"a zero-impedance reduction merged its rectifier and inverter buses"),
        min_level = Logging.Warn,
        PF._lcc_dc_ties(sys, data.lcc, removed_buses, Dict{Int, Int}())
    )
    @test isempty(dc_ties)   # no bogus self-tie

    # VSC has no analogous test: it's structurally unreachable. VSC AC terminals are always
    # protected as `irreducible_buses`, so PNM's reduction can never merge one; faking
    # `fix == tix` would test a state PNM can never produce.
end

@testset "area interchange DC greedy relax" begin
    # Area3 borders BOTH DC ties. A 20.0 pu target is unenforceable (confirmed empirically):
    # Newton fails with Area3 controlled, greedy relax de-enrolls it, Area2 survives alone.
    sys = _comprehensive_area_dc_fixture()
    ai3 = PSY.get_component(PSY.AreaInterchange, sys, "A3_A1")
    PSY.set_active_power_flow!(ai3, 20.0)
    pf_nr = ACPolarPowerFlow{NewtonRaphsonACPowerFlow}(; area_interchange_control = true)
    data = PowerFlowData(pf_nr, sys)
    @test PF.n_controlled_areas(data) == 2
    @test !isempty(data.area_interchange.dc_ties)

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
    @test length(data.area_interchange.areas) == 1
    survivor = only(data.area_interchange.areas)
    @test survivor.name == "Area2"
    @test haskey(data.area_interchange.relaxed, 1)
    @test only(data.area_interchange.relaxed[1]).name == "Area3"

    dcn = PF.get_dc_network(data)
    n = PF.n_controlled_areas(data)
    ni = zeros(n)
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
    for tie in data.area_interchange.dc_ties
        P_conv = _oracle_dc_tie_metered_power(data, dcn, tie, 1)
        metered_tail = tie.from_area_tail
        other_tail = tie.to_area_tail
        if !tie.metered_from
            metered_tail = tie.to_area_tail
            other_tail = tie.from_area_tail
        end
        iszero(metered_tail) || (ni[metered_tail] += P_conv)
        iszero(other_tail) || (ni[other_tail] -= P_conv)
    end
    @test isapprox(ni[survivor.tail_ix], survivor.pdes; atol = 1e-5)

    # Defect-specific bite: `_deenroll_area!` must translate `dc_ties` tails exactly like it
    # translates `ties` tails -- a stale tail pointing past the (now smaller) WORKING area
    # set is the bug (`_set_area_tail_residuals!`/`ni_scratch` indexing would be broken).
    @test all(
        t ->
            t.from_area_tail <= length(data.area_interchange.areas) &&
                t.to_area_tail <= length(data.area_interchange.areas),
        data.area_interchange.dc_ties,
    )
    # The VSC tie borders the survivor (Area2) directly: its tail must read the survivor's
    # NEW (post-renumbering) tail index, not a value keyed to the old (2-area) numbering.
    vsc_tie = only(filter(t -> t.kind == PF.DC_TIE_VSC, data.area_interchange.dc_ties))
    @test vsc_tie.from_area_tail == survivor.tail_ix
end

@testset "area interchange DC results dataframe" begin
    sys = _comprehensive_area_dc_fixture()
    pf_nr = ACPolarPowerFlow{NewtonRaphsonACPowerFlow}(; area_interchange_control = true)
    data = PowerFlowData(pf_nr, sys)
    @test solve_power_flow!(data)

    df = PF.area_interchange_results_dataframe(sys, data, 1)
    @test nrow(df) == 2

    dcn = PF.get_dc_network(data)
    sys_basepower = PSY.get_base_power(sys)
    n = length(data.area_interchange.pristine_areas)
    ni = zeros(n)
    dc_only = zeros(n)
    for tie in data.area_interchange.pristine_ties
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
    for tie in data.area_interchange.pristine_dc_ties
        P_conv = _oracle_dc_tie_metered_power(data, dcn, tie, 1)
        metered_tail = tie.from_area_tail
        other_tail = tie.to_area_tail
        if !tie.metered_from
            metered_tail = tie.to_area_tail
            other_tail = tie.from_area_tail
        end
        if !iszero(metered_tail)
            ni[metered_tail] += P_conv
            dc_only[metered_tail] += P_conv
        end
        if !iszero(other_tail)
            ni[other_tail] -= P_conv
            dc_only[other_tail] -= P_conv
        end
    end

    for area in data.area_interchange.pristine_areas
        # Non-vacuity: every controlled area in this fixture borders a DC tie, so the report
        # can never pass vacuously -- if it changes to be false the fixture, not the test, is
        # wrong (RED without the fix: `ni_solved` is off from the oracle by exactly this).
        @test abs(dc_only[area.tail_ix]) > 1e-3
        row = only(filter(:area => ==(area.name), df))
        @test isapprox(row.ni_solved, sys_basepower * ni[area.tail_ix]; atol = 1e-5)
        @test row.schedule_status == :enforced
    end
end

@testset "area interchange DC multi period" begin
    sys = _comprehensive_area_dc_fixture()
    pf_nr = ACPolarPowerFlow{NewtonRaphsonACPowerFlow}(;
        area_interchange_control = true, time_steps = 3)
    data = PowerFlowData(pf_nr, sys)
    @test PF.n_controlled_areas(data) == 2
    @test !isempty(data.area_interchange.dc_ties)

    data.bus_active_power_withdrawals[:, 2] .+= 0.05
    data.bus_active_power_withdrawals[:, 3] .+= 0.10

    dcn = PF.get_dc_network(data)
    for t in 1:3
        @test solve_power_flow!(data; time_steps = [t])
        n = PF.n_controlled_areas(data)
        ni = zeros(n)
        for tie in data.area_interchange.ties
            P_m = _oracle_tie_metered_power(sys, data, tie, t)
            metered_tail = tie.from_area_tail
            other_tail = tie.to_area_tail
            if !tie.metered_from
                metered_tail = tie.to_area_tail
                other_tail = tie.from_area_tail
            end
            iszero(metered_tail) || (ni[metered_tail] += P_m)
            iszero(other_tail) || (ni[other_tail] -= P_m)
        end
        for tie in data.area_interchange.dc_ties
            P_conv = _oracle_dc_tie_metered_power(data, dcn, tie, t)
            metered_tail = tie.from_area_tail
            other_tail = tie.to_area_tail
            if !tie.metered_from
                metered_tail = tie.to_area_tail
                other_tail = tie.from_area_tail
            end
            iszero(metered_tail) || (ni[metered_tail] += P_conv)
            iszero(other_tail) || (ni[other_tail] -= P_conv)
        end
        for area in data.area_interchange.areas
            @test isapprox(ni[area.tail_ix], area.pdes; atol = 1e-5)
        end
    end

    dp1 = copy(data.area_interchange.delta_p[:, 1])
    dp2 = copy(data.area_interchange.delta_p[:, 2])
    dp3 = copy(data.area_interchange.delta_p[:, 3])
    @test !isapprox(dp1, dp2; atol = 1e-4)
    @test !isapprox(dp2, dp3; atol = 1e-4)
    @test !isapprox(dp1, dp3; atol = 1e-4)

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

# A greedy relax de-enrolls a DC-tie-bordering area; `data` is then re-solved with that
# area's target restored to feasible, forcing the working set to re-grow 1->2 areas --
# exercising stale-cache reuse. Restoring feasibility (not re-solving the still-infeasible
# schedule) matters: the infeasible re-solve's relax order is start-point-dependent.
@testset "area interchange DC cache invariant after relax" begin
    for ACSolver in (
        NewtonRaphsonACPowerFlow,
        FastDecoupledACPowerFlow{FDFixedJacobian, FDSchemeXB},
    )
        sys = _comprehensive_area_dc_fixture()
        pf = ACPolarPowerFlow{ACSolver}(; area_interchange_control = true)
        # Area3's feasible derived NI target, read BEFORE its schedule is made infeasible -- the
        # value restored below so the warm re-solve must re-grow to BOTH areas.
        feasible_a3 =
            only(
                filter(
                    a -> a.name == "Area3",
                    PowerFlowData(pf, sys).area_interchange.pristine_areas,
                ),
            ).pdes
        ai3 = PSY.get_component(PSY.AreaInterchange, sys, "A3_A1")
        PSY.set_active_power_flow!(ai3, 20.0)
        data = PowerFlowData(pf, sys)
        @test PF.n_controlled_areas(data) == 2
        @test !isempty(data.area_interchange.dc_ties)

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
        @test PF.n_controlled_areas(data) == 1
        survivor = only(data.area_interchange.areas)
        @test survivor.name == "Area2"

        @test all(
            t ->
                t.from_area_tail <= length(data.area_interchange.areas) &&
                    t.to_area_tail <= length(data.area_interchange.areas),
            data.area_interchange.dc_ties,
        )

        for (i, a) in enumerate(data.area_interchange.pristine_areas)
            a.name == "Area3" || continue
            data.area_interchange.pristine_areas[i] =
                PF.ControlledArea(a.name, a.slack_bus_ix, feasible_a3, a.tail_ix)
        end
        @test solve_power_flow!(data)
        @test PF.n_controlled_areas(data) == 2
        @test sort([a.name for a in data.area_interchange.areas]) == ["Area2", "Area3"]

        @test all(
            t ->
                t.from_area_tail <= length(data.area_interchange.areas) &&
                    t.to_area_tail <= length(data.area_interchange.areas),
            data.area_interchange.dc_ties,
        )

        dcn = PF.get_dc_network(data)
        n = PF.n_controlled_areas(data)
        ni = zeros(n)
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
        for tie in data.area_interchange.dc_ties
            P_conv = _oracle_dc_tie_metered_power(data, dcn, tie, 1)
            metered_tail = tie.from_area_tail
            other_tail = tie.to_area_tail
            if !tie.metered_from
                metered_tail = tie.to_area_tail
                other_tail = tie.from_area_tail
            end
            iszero(metered_tail) || (ni[metered_tail] += P_conv)
            iszero(other_tail) || (ni[other_tail] -= P_conv)
        end
        for area in data.area_interchange.areas
            @test isapprox(ni[area.tail_ix], area.pdes; atol = 1e-5)
        end
    end
end

@testset "area interchange DC homotopy rejected" begin
    @test_throws r"RobustHomotopyPowerFlow" ACPolarPowerFlow{RobustHomotopyPowerFlow}(;
        area_interchange_control = true)
end
