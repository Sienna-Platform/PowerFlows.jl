@testset "Jacobian verification" begin
    sys = PSB.build_system(PSITestSystems, "c_sys14")
    verify_jacobian(sys; label = "polar c_sys14")
end

@testset "Jacobian verification with LCC" begin
    sys = System(100.0)
    b1 = _add_simple_bus!(sys, 1, ACBusTypes.REF, 230, 1.1, 0.0)
    b2 = _add_simple_bus!(sys, 2, ACBusTypes.PQ, 230, 1.1, 0.0)
    b3 = _add_simple_bus!(sys, 3, ACBusTypes.PQ, 230, 1.1, 0.0)
    ld2 = _add_simple_load!(sys, b2, 10, 5)
    ld3 = _add_simple_load!(sys, b3, 60, 20)
    l12 = _add_simple_line!(sys, b1, b2, 5e-3, 5e-3, 1e-3)
    l13 = _add_simple_line!(sys, b1, b3, 5e-3, 5e-3, 1e-3)
    s1 = _add_simple_source!(sys, b1, 0.0, 0.0)
    lcc = _add_simple_lcc!(sys, b2, b3, 0.05, 0.05, 0.08)
    # `_add_simple_lcc!` already sets rectifier_delay_angle = 0.01 > 0; the
    # default inverter_extinction_angle is 0.0 which makes ϕ_i hit the
    # acos clamp boundary (sin(ϕ_i) = 0) at x0 — that's a separate boundary
    # test (see "Jacobian verification with LCC at inverter ϕ clamp"
    # below). Bump α_i here to verify the interior regime.
    PSY.set_inverter_extinction_angle!(lcc, 1.0)
    # Smaller perturbation here so the LCC α tail entries (α_r ≈ 0.087,
    # α_i = 1.0) stay clear of the min-thyristor-angle clamp.
    verify_jacobian(sys; label = "polar 3-bus LCC", perturbation = 0.01)
end

@testset "Jacobian verification with LCC, inverter-side setpoint" begin
    # A negative transfer setpoint puts the P-setpoint constraint on the
    # inverter: F_t_fb = −P_lcc_to − P_set, so the F_t_fb Jacobian row must
    # carry ∂/∂(V_tb, tap_i, α_i) instead of ∂/∂(V_fb, tap_r, α_r). Before
    # the fix the row still held the rectifier-side derivatives — the
    # asymptotic verifier catches that as order-1 decay on those columns.
    sys = System(100.0)
    b1 = _add_simple_bus!(sys, 1, ACBusTypes.REF, 230, 1.1, 0.0)
    b2 = _add_simple_bus!(sys, 2, ACBusTypes.PQ, 230, 1.1, 0.0)
    b3 = _add_simple_bus!(sys, 3, ACBusTypes.PQ, 230, 1.1, 0.0)
    _add_simple_load!(sys, b2, 10, 5)
    _add_simple_load!(sys, b3, 60, 20)
    _add_simple_line!(sys, b1, b2, 5e-3, 5e-3, 1e-3)
    _add_simple_line!(sys, b1, b3, 5e-3, 5e-3, 1e-3)
    _add_simple_source!(sys, b1, 0.0, 0.0)
    lcc = _add_simple_lcc!(sys, b2, b3, 0.05, 0.05, 0.08)
    PSY.set_inverter_extinction_angle!(lcc, 1.0)   # interior, off the ϕ clamp
    PSY.set_transfer_setpoint!(lcc, -0.5)          # setpoint at inverter
    verify_jacobian(sys; label = "polar 3-bus LCC, inverter-side setpoint",
        perturbation = 0.01)
end

@testset "Jacobian verification with LCC at a PV terminal" begin
    # An LCC terminal at a PV bus: state is (Q_gen, θ), with V fixed at V_set.
    # The bus Q-balance still depends on tap_r/α_r through the LCC's Q
    # contribution, so ∂Q/∂tap and ∂Q/∂α must be filled in the Jacobian
    # for the PV terminal — they previously weren't, leaving these entries
    # stuck at 0 from the sparsity pattern.
    sys = System(100.0)
    b1 = _add_simple_bus!(sys, 1, ACBusTypes.REF, 230, 1.1, 0.0)
    b2 = _add_simple_bus!(sys, 2, ACBusTypes.PV, 230, 1.05, 0.0)   # PV terminal
    b3 = _add_simple_bus!(sys, 3, ACBusTypes.PQ, 230, 1.1, 0.0)
    _add_simple_thermal_standard!(sys, b2, 0.2, 0.1)  # generator at PV bus
    _add_simple_load!(sys, b2, 10, 5)
    _add_simple_load!(sys, b3, 60, 20)
    _add_simple_line!(sys, b1, b2, 5e-3, 5e-3, 1e-3)
    _add_simple_line!(sys, b1, b3, 5e-3, 5e-3, 1e-3)
    _add_simple_source!(sys, b1, 0.0, 0.0)
    lcc = _add_simple_lcc!(sys, b2, b3, 0.05, 0.05, 0.08)
    PSY.set_inverter_extinction_angle!(lcc, 1.0)
    verify_jacobian(sys; label = "polar 3-bus LCC, PV rectifier terminal",
        perturbation = 0.01)
end

@testset "Jacobian verification with LCC at inverter ϕ clamp" begin
    # Drive the inverter into the `raw < -1` clamp of `_calculate_ϕ_lcc`:
    # need cos(α_i) + x_t·I_dc/(√2·V·tap) > 1. Large inverter x_t plus a
    # small extinction angle does it. This exercises the `sin(ϕ) → 0`
    # boundary guards in the dP/dV, dP/dt helpers — without those guards
    # the analytic Jacobian disagrees with the residual at the inverter,
    # and the asymptotic verifier would catch it as order-1 decay.
    sys = System(100.0)
    b1 = _add_simple_bus!(sys, 1, ACBusTypes.REF, 230, 1.1, 0.0)
    b2 = _add_simple_bus!(sys, 2, ACBusTypes.PQ, 230, 1.1, 0.0)
    b3 = _add_simple_bus!(sys, 3, ACBusTypes.PQ, 230, 1.1, 0.0)
    _add_simple_load!(sys, b2, 10, 5)
    _add_simple_load!(sys, b3, 60, 20)
    _add_simple_line!(sys, b1, b2, 5e-3, 5e-3, 1e-3)
    _add_simple_line!(sys, b1, b3, 5e-3, 5e-3, 1e-3)
    _add_simple_source!(sys, b1, 0.0, 0.0)
    # xc_i = 0.20 (large), α_i = 0.1 rad (small) → inverter raw < -1, ϕ = π.
    lcc = _add_simple_lcc!(sys, b2, b3, 0.05, 0.05, 0.20)
    PSY.set_inverter_extinction_angle!(lcc, 0.1)
    PSY.set_rectifier_delay_angle!(lcc, 0.1)
    verify_jacobian(
        sys; label = "polar 3-bus LCC, inverter ϕ-clamp", perturbation = 0.01,
    )
end

@testset "Jacobian verification with LCC, realistic inverter (interior, tap≠1, NBR>1)" begin
    # The regime large interconnection-scale planning cases hit and the tap=1 / α≈0 tests
    # above never exercised:
    # extinction/delay angles ~15-18°, transformer taps off nominal, 2 bridges per side.
    # With the corrected inverter commutation (drop SUBTRACTS), ϕ_i stays interior
    # (sin ϕ_i > 0) so the converter carries reactive power, and the −xtr_i sign on the
    # inverter's commutation-chain derivatives must make the analytic Jacobian match the
    # residual. A wrong inverter commutation sign shows up as order-1 decay here.
    sys = System(100.0)
    b1 = _add_simple_bus!(sys, 1, ACBusTypes.REF, 230, 1.05, 0.0)
    b2 = _add_simple_bus!(sys, 2, ACBusTypes.PQ, 230, 1.02, 0.0)
    b3 = _add_simple_bus!(sys, 3, ACBusTypes.PQ, 230, 1.0, 0.0)
    _add_simple_load!(sys, b2, 10, 5)
    _add_simple_load!(sys, b3, 60, 20)
    _add_simple_line!(sys, b1, b2, 5e-3, 5e-3, 1e-3)
    _add_simple_line!(sys, b1, b3, 5e-3, 5e-3, 1e-3)
    _add_simple_source!(sys, b1, 0.0, 0.0)
    lcc = _add_simple_lcc!(sys, b2, b3, 0.02, 0.03, 0.04)
    PSY.set_rectifier_delay_angle!(lcc, deg2rad(15))
    PSY.set_inverter_extinction_angle!(lcc, deg2rad(18))
    PSY.set_rectifier_tap_setting!(lcc, 0.9)
    PSY.set_inverter_tap_setting!(lcc, 0.95)
    PSY.set_rectifier_bridges!(lcc, 2)
    PSY.set_inverter_bridges!(lcc, 2)
    verify_jacobian(sys; label = "polar 3-bus LCC, realistic interior inverter",
        perturbation = 0.005)
end

@testset "LCC inverter reactive power is nonzero at the solution (regression)" begin
    # Regression for the inverter ϕ-commutation-sign defect: before the fix a realistic
    # small-γ inverter had its commutation drop ADDED, driving raw = −(cos γ + comm) < −1,
    # so ϕ_i clamped to π (sin ϕ_i = 0). That zeroed the inverter's reactive draw and let
    # the terminal voltage run away. With the fix the inverter stays interior and consumes
    # reactive power. Assert the solved inverter is well off the clamp.
    sys = System(100.0)
    b1 = _add_simple_bus!(sys, 1, ACBusTypes.REF, 230, 1.0, 0.0)
    b2 = _add_simple_bus!(sys, 2, ACBusTypes.PQ, 230, 1.0, 0.0)
    b3 = _add_simple_bus!(sys, 3, ACBusTypes.PQ, 230, 1.0, 0.0)
    _add_simple_load!(sys, b2, 10, 5)
    _add_simple_load!(sys, b3, 60, 20)
    _add_simple_line!(sys, b1, b2, 5e-3, 5e-3, 1e-3)
    _add_simple_line!(sys, b1, b3, 5e-3, 5e-3, 1e-3)
    _add_simple_source!(sys, b1, 0.0, 0.0)
    lcc = _add_simple_lcc!(sys, b2, b3, 0.02, 0.04, 0.08)
    PSY.set_rectifier_delay_angle!(lcc, deg2rad(15))
    PSY.set_inverter_extinction_angle!(lcc, deg2rad(17))
    PSY.set_inverter_tap_setting!(lcc, 0.95)
    pf = PF.ACPowerFlow{NewtonRaphsonACPowerFlow}(; correct_bustypes = true)
    data = PF.PowerFlowData(pf, sys)
    @test PF.solve_power_flow!(data)
    # Off the acos clamp: sin(ϕ_i) = 0 would mean zero reactive contribution.
    @test sin(data.lcc.inverter.phi[1, 1]) > 0.1
    # Nonzero DC current carrying the transfer.
    @test data.lcc.i_dc[1, 1] > 0.0
end

@testset "Jacobian verification with ZIP load" begin
    sys = System(100.0)
    b1 = _add_simple_bus!(sys, 1, ACBusTypes.REF, 230, 1.0, 0.0)
    b2 = _add_simple_bus!(sys, 2, ACBusTypes.PQ, 230, 1.0, 0.0)
    _add_simple_line!(sys, b1, b2, 5e-3, 5e-3, 1e-3)
    _add_simple_source!(sys, b1, 0.0, 0.0)
    _add_simple_zip_load!(
        sys, b2;
        constant_power_active_power = 0.5,
        constant_power_reactive_power = 0.2,
        constant_current_active_power = 2.0,
        constant_current_reactive_power = 1.0,
        constant_impedance_active_power = 1.5,
        constant_impedance_reactive_power = 0.8,
    )
    verify_jacobian(sys; label = "polar ZIP")
end

@testset "Jacobian verification with distributed slack" begin
    sys = PSB.build_system(PSITestSystems, "c_sys14")
    generators = collect(get_components(ThermalStandard, sys))
    # Assign distinct nonzero participation factors to all generators (REF and PV buses).
    # This exercises the cross-terms ∂F_P_k/∂x[2*ref-1] = -c_k for PV buses
    # and the corrected REF diagonal ∂F_P_ref/∂x[2*ref-1] = -c_ref.
    gspf = Dict(
        (ThermalStandard, get_name(g)) => Float64(i)
        for (i, g) in enumerate(generators)
    )
    pf = PF.ACPowerFlow{NewtonRaphsonACPowerFlow}(;
        correct_bustypes = true,
        generator_slack_participation_factors = gspf,
    )
    verify_jacobian(sys; pf = pf, label = "polar c_sys14 distributed-slack")
end

@testset "Multi-swing: two swings in one island each self-balance (solve)" begin
    sys = _two_swing_system()
    pf = PF.ACPowerFlow{NewtonRaphsonACPowerFlow}(; correct_bustypes = false)
    data = PF.PowerFlowData(pf, sys)
    @test PF.solve_power_flow!(data; tol = 1e-9)
    lookup = PF.get_bus_lookup(data)
    vm = PF.get_bus_magnitude(data)
    va = PF.get_bus_angles(data)
    @test vm[lookup[1], 1] ≈ 1.06 atol = 1e-9
    @test va[lookup[1], 1] ≈ 0.0 atol = 1e-9
    @test vm[lookup[2], 1] ≈ 1.05 atol = 1e-9
    @test va[lookup[2], 1] ≈ 0.05 atol = 1e-9
end

@testset "Multi-swing: RobustHomotopy is rejected, NR still solves" begin
    # HomotopyHessian has no independent-per-swing slack handling, so its curvature would
    # disagree with the residual's self-balancing rows. Gate until that is implemented.
    sys = _two_swing_system()
    pf_rh = PF.ACPowerFlow{RobustHomotopyPowerFlow}(; correct_bustypes = false)
    data_rh = PF.PowerFlowData(pf_rh, sys)
    @test_throws ArgumentError PF.solve_power_flow!(data_rh)
    # The gate rejects only RobustHomotopy: NR on the same system still solves.
    pf_nr = PF.ACPowerFlow{NewtonRaphsonACPowerFlow}(; correct_bustypes = false)
    @test PF.solve_power_flow!(PF.PowerFlowData(pf_nr, sys); tol = 1e-9)
end

@testset "Jacobian verification with two swings (multi-swing)" begin
    # Each swing's ∂F_P/∂x[2i−1] = −1 with no cross-terms; a wrong diagonal or stray
    # cross-term shows as order-1 decay in the asymptotic check.
    verify_jacobian(
        _two_swing_system();
        pf = PF.ACPowerFlow{NewtonRaphsonACPowerFlow}(; correct_bustypes = false),
        label = "polar two-swing", perturbation = 0.01,
    )
end

@testset "Multi-swing under Fast Decoupled ($(V))" for V in
                                                       (FDFixedJacobian, FDDecoupled)
    # fdfixed reuses the polar Jacobian (independent-swing diagonal); :decoupled uses the
    # explicit-state sync, which must close each swing's own P/Q rows (not a rank-1 island sum).
    pf = PF.ACPowerFlow{FastDecoupledACPowerFlow{V, FDSchemeXB}}(; correct_bustypes = false)
    data = PF.PowerFlowData(pf, _two_swing_system())
    @test PF.solve_power_flow!(data)
    lookup = PF.get_bus_lookup(data)
    vm = PF.get_bus_magnitude(data)
    va = PF.get_bus_angles(data)
    @test vm[lookup[1], 1] ≈ 1.06 atol = 1e-6
    @test va[lookup[1], 1] ≈ 0.0 atol = 1e-6
    @test vm[lookup[2], 1] ≈ 1.05 atol = 1e-6
    @test va[lookup[2], 1] ≈ 0.05 atol = 1e-6
end

# `true` iff the sparse structure has a STORED entry at (i, j) -- unlike `A[i, j] == 0.0`,
# this distinguishes "structurally absent" from "present but numerically zero" (e.g. a PV
# endpoint's Vm column, spec's union pattern).
function _has_stored_entry(A::SparseMatrixCSC, i::Int, j::Int)
    for k in SparseArrays.nzrange(A, j)
        SparseArrays.rowvals(A)[k] == i && return true
    end
    return false
end

# Runs the shared area-interchange Jacobian assertions against a system enrolled with
# `area_interchange_control = true`: (1) no (area_row, area_col) diagonal entry in the
# structure (the border's zero diagonal — KLU full pivoting handles it); (2) the -1.0
# column entry at each area's slack-bus P-mismatch row; (3) the
# full asymptotic FD sweep over EVERY state entry, including the area tail.
function _verify_area_jacobian(sys::PSY.System, label::String)
    pf = PF.ACPowerFlow{NewtonRaphsonACPowerFlow}(;
        correct_bustypes = true,
        area_interchange_control = true,
    )
    data = PF.PowerFlowData(pf, sys)
    @test PF.n_controlled_areas(data) >= 1
    time_step = 1
    residual = PF.ACPowerFlowResidual(data, time_step)
    J = PF.ACPowerFlowJacobian(data, residual, time_step)
    x0 = PF.calculate_x0(data, time_step)
    Random.seed!(42)
    x0 .+= 0.02 .* randn(length(x0))
    residual(data, x0, time_step)
    J(data, time_step)

    dcn = PF.get_dc_network(data)
    area_off = PF.area_tail_offset(data, dcn)
    Jv = J.Jv
    for area in data.area_interchange.areas
        row = area_off + area.tail_ix
        @test !_has_stored_entry(Jv, row, row)
        @test Jv[2 * area.slack_bus_ix - 1, area_off + area.tail_ix] == -1.0
    end

    verify_jacobian_asymptotic(residual, data, deepcopy(Jv), x0, time_step; label = label)
    return
end

@testset "Jacobian verification with area interchange control (2-area, 1 controlled)" begin
    sys = _make_two_area_system()
    _set_slack!(sys, "Bus 6")
    _add_area_interchange!(sys, "Area2", "Area1", 0.3; name = "A2_A1")
    _verify_area_jacobian(sys, "polar area interchange (1 controlled area)")
end

@testset "Jacobian verification with area interchange control (3-area, degree-4 boundary)" begin
    sys = _three_area_transfer_fixture(; slack_area3 = true)
    _verify_area_jacobian(sys, "polar area interchange (2 controlled areas, degree-4 bus)")
end

@testset "Jacobian verification with area interchange control (3W transformer tie, polluted star-bus diagonal)" begin
    sys = _make_3w_boundary_fixture()
    _verify_area_jacobian(sys, "polar area interchange (3W winding tie)")
end

# The fused kernel (the NR step's residual + Jacobian in one Ybus sweep) must equal the
# separate F-only and J-only evaluations at the same iterate.
function _check_fused_kernel(data::PF.ACPowerFlowData, label::String)
    residual = PF.ACPowerFlowResidual(data, 1)
    J = PF.ACPowerFlowJacobian(data, residual, 1)
    x = PF.calculate_x0(data, 1)
    Random.seed!(11)
    x .+= 0.02 .* randn(length(x))
    residual(data, x, 1)
    J(data, 1)
    F_sep = copy(residual.Rv)
    J_sep = copy(SparseArrays.nonzeros(J.Jv))
    fill!(residual.Rv, NaN)
    fill!(SparseArrays.nonzeros(J.Jv), 0.0)
    PF._update_residual_and_jacobian!(residual, J, x, data, 1)
    @testset "$label" begin
        @test isapprox(residual.Rv, F_sep; rtol = 1e-12, atol = 1e-12)
        @test isapprox(SparseArrays.nonzeros(J.Jv), J_sep; rtol = 1e-12, atol = 1e-12)
    end
    return
end

@testset "Fused residual + Jacobian kernel matches the separate evaluations" begin
    nr(; kw...) = PF.ACPowerFlow{NewtonRaphsonACPowerFlow}(; kw...)
    c14 = PSB.build_system(PSITestSystems, "c_sys14")
    _check_fused_kernel(PF.PowerFlowData(nr(; correct_bustypes = true), c14), "c_sys14")
    gspf = Dict(
        (ThermalStandard, get_name(g)) => Float64(i)
        for (i, g) in enumerate(get_components(ThermalStandard, c14))
    )
    _check_fused_kernel(
        PF.PowerFlowData(
            nr(; correct_bustypes = true, generator_slack_participation_factors = gspf),
            c14,
        ),
        "c_sys14 distributed slack",
    )
    _check_fused_kernel(
        PF.PowerFlowData(nr(; correct_bustypes = false), _two_swing_system()),
        "two swings",
    )
    _check_fused_kernel(
        PF.PowerFlowData(nr(; correct_bustypes = true), build_lcc_control_system()),
        "two LCCs",
    )
    area_sys = _make_two_area_system()
    _set_slack!(area_sys, "Bus 6")
    _add_area_interchange!(area_sys, "Area2", "Area1", 0.3; name = "A2_A1")
    _check_fused_kernel(
        PF.PowerFlowData(
            nr(; correct_bustypes = true, area_interchange_control = true), area_sys),
        "area interchange",
    )
end

# The fused kernel defers `data`'s voltages and injections to `_write_back_bus_state!`, which must
# leave `data` exactly as a write-through evaluation at the same iterate does.
function _check_deferred_write_back(make_data, label::String)
    eager = make_data()
    deferred = make_data()
    R_eager = PF.ACPowerFlowResidual(eager, 1)
    R = PF.ACPowerFlowResidual(deferred, 1)
    J = PF.ACPowerFlowJacobian(deferred, R, 1)
    x = PF.calculate_x0(deferred, 1)
    Random.seed!(7)
    x .+= 0.02 .* randn(length(x))
    R_eager(eager, x, 1)
    q_before = deferred.bus_reactive_power_injections[:, 1]
    PF._update_residual_and_jacobian!(R, J, x, deferred, 1)
    PF._update_residual_and_jacobian!(R, J, x, deferred, 1)
    @testset "$label" begin
        @test R.bus_state.data_stale
        @test deferred.bus_reactive_power_injections[:, 1] == q_before
        PF._write_back_bus_state!(R, deferred, 1)
        @test !R.bus_state.data_stale
        @test isapprox(R.Rv, R_eager.Rv; rtol = 1e-12, atol = 1e-12)
        for f in (:bus_magnitude, :bus_angles, :bus_active_power_injections,
            :bus_reactive_power_injections)
            @test getfield(deferred, f)[:, 1] == getfield(eager, f)[:, 1]
        end
    end
    return
end

@testset "Fused kernel write-back matches a write-through evaluation" begin
    nr(; kw...) = PF.ACPowerFlow{NewtonRaphsonACPowerFlow}(; kw...)
    c14 = PSB.build_system(PSITestSystems, "c_sys14")
    _check_deferred_write_back(
        () -> PF.PowerFlowData(nr(; correct_bustypes = true), c14), "c_sys14")
    lcc = build_lcc_control_system()
    _check_deferred_write_back(
        () -> PF.PowerFlowData(nr(; correct_bustypes = true), lcc), "two LCCs")
end

# The pre-direct-CSC builder: COO triplets for every `neighbors` pair plus the tails, through
# `sparse`. `_create_jacobian_matrix_structure` must reproduce it byte for byte.
function _reference_jacobian_structure(data::PF.ACPowerFlowData, slots)
    rows = PF.J_INDEX_TYPE[]
    columns = PF.J_INDEX_TYPE[]
    values = Float64[]
    num_buses = first(size(data.bus_type))
    for bus_from in 1:num_buses, bus_to in data.neighbors[bus_from]
        PF._create_jacobian_matrix_structure_bus!(rows, columns, values, bus_from, bus_to,
            2 * bus_from - 1, 2 * bus_from, 2 * bus_to - 1, 2 * bus_to)
    end
    for (bus_k, ref_bus) in slots
        push!(rows, 2 * bus_k - 1)
        push!(columns, 2 * ref_bus - 1)
        push!(values, 0.0)
    end
    PF._create_jacobian_matrix_structure_lcc(data, rows, columns, values, num_buses)
    PF._create_jacobian_matrix_structure_vsc(data, rows, columns, values, num_buses)
    PF._create_jacobian_matrix_structure_area(data, rows, columns, values)
    return SparseArrays.sparse(rows, columns, values)
end

@testset "Direct-CSC Jacobian structure matches the COO reference" begin
    nr(; kw...) = PF.ACPowerFlow{NewtonRaphsonACPowerFlow}(; kw...)
    c14 = PSB.build_system(PSITestSystems, "c_sys14")
    gspf = Dict(
        (ThermalStandard, get_name(g)) => Float64(i)
        for (i, g) in enumerate(get_components(ThermalStandard, c14))
    )
    area_sys = _make_two_area_system()
    _set_slack!(area_sys, "Bus 6")
    _add_area_interchange!(area_sys, "Area2", "Area1", 0.3; name = "A2_A1")
    cases = [
        ("c_sys14", nr(; correct_bustypes = true), c14),
        ("c_sys14 distributed slack",
            nr(; correct_bustypes = true, generator_slack_participation_factors = gspf),
            c14),
        (
            "c_sys5",
            nr(; correct_bustypes = true),
            PSB.build_system(PSITestSystems, "c_sys5"),
        ),
        ("ACTIVSg2000", nr(; correct_bustypes = true),
            PSB.build_system(PSB.MatpowerTestSystems, "matpower_ACTIVSg2000_sys")),
        ("two LCCs", nr(; correct_bustypes = true), build_lcc_control_system()),
        ("VSC", nr(; correct_bustypes = true), _build_vsc_system()),
        ("area interchange",
            nr(; correct_bustypes = true, area_interchange_control = true),
            area_sys),
    ]
    n_slotted = 0
    n_tailed = 0
    for (label, pf, sys) in cases
        data = PF.PowerFlowData(pf, sys)
        residual = PF.ACPowerFlowResidual(data, 1)
        slots = PF._extra_slack_slots(data, residual.subnetworks, 1)
        n_slotted += !isempty(slots)
        J = PF._create_jacobian_matrix_structure(data, slots)
        R = _reference_jacobian_structure(data, slots)
        n_tailed += size(J, 1) > 2 * first(size(data.bus_type))
        @testset "$label" begin
            @test size(J) == size(R)
            @test J.colptr == R.colptr
            @test J.rowval == R.rowval
            @test SparseArrays.nonzeros(J) == SparseArrays.nonzeros(R)
        end
    end
    @test n_slotted >= 1
    @test n_tailed == 3
end
