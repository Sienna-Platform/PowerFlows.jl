@testset "Mixed CPB: setup offsets / fill / update roundtrip" begin
    sys = PSB.build_system(PSB.PSITestSystems, "c_sys14"; add_forecasts = false)
    pf = PF.ACMixedPowerFlow{NewtonRaphsonACPowerFlow}()
    data = PF.PowerFlowData(pf, sys)

    offs, bs, total =
        PF.compute_mixed_bus_state_offsets(view(data.bus_type, :, 1))
    n_buses = size(data.bus_type, 1)
    @test all(bs .== 2)
    @test total == 2 * n_buses
    @test offs[1] == 1 && offs[end] == total + 1

    n_lccs = size(data.lcc.p_set, 1)
    x = Vector{Float64}(undef, total + 4 * n_lccs)
    PF.mixed_initial_state!(x, data, offs, bs, 1)
    @test all(isfinite, x)

    bt = view(data.bus_type, :, 1)
    Vm_before = copy(view(data.bus_magnitude, :, 1))
    θ_before = copy(view(data.bus_angles, :, 1))

    # Perturb data, round-trip through mixed_update_data!. PV bus_magnitude
    # must be preserved (V_set); PQ bus_magnitude/bus_angles recovered.
    for i in 1:n_buses
        if bt[i] == PSY.ACBusTypes.PQ
            data.bus_magnitude[i, 1] = NaN
        end
        bt[i] != PSY.ACBusTypes.REF && (data.bus_angles[i, 1] = NaN)
    end
    PF.mixed_update_data!(data, x, offs, bs, 1)

    for i in 1:n_buses
        if bt[i] == PSY.ACBusTypes.PV
            @test data.bus_magnitude[i, 1] ≈ Vm_before[i] atol = 1e-12
        end
        if bt[i] == PSY.ACBusTypes.PQ
            @test data.bus_magnitude[i, 1] ≈ Vm_before[i] atol = 1e-12
        end
        if bt[i] != PSY.ACBusTypes.REF
            @test data.bus_angles[i, 1] ≈ θ_before[i] atol = 1e-12
        end
    end
end

@testset "Mixed CPB: PV Q_gen recovery excludes const-Z double-count (C1)" begin
    # Regression for C1: mixed_finalize_bus_injections! must recover PV Q_gen
    # using the RAW Y-bus, not Y_bus_eff (which has const-Z ZIP folded into its
    # diagonal). If Y_bus_eff were used, a PV bus with nonzero β_Q would get
    # Q_gen off by exactly β_Q·|V|².
    sys = PSB.build_system(PSB.PSITestSystems, "c_sys5")
    pf = PF.ACMixedPowerFlow{NewtonRaphsonACPowerFlow}()
    data = PF.PowerFlowData(pf, sys)

    bt = view(data.bus_type, :, 1)
    n_buses = size(data.bus_type, 1)
    pv_bus = findfirst(==(PSY.ACBusTypes.PV), bt)
    ref_bus = findfirst(==(PSY.ACBusTypes.REF), bt)

    # Attach a nonzero constant-impedance reactive load on the PV bus.
    β_Q = 0.37
    β_P = 0.21
    data.bus_active_power_constant_impedance_withdrawals[pv_bus, 1] = β_P
    data.bus_reactive_power_constant_impedance_withdrawals[pv_bus, 1] = β_Q

    # Known converged voltages (arbitrary but fixed) for every bus.
    e_state = collect(range(1.00; step = 0.01, length = n_buses))
    f_state = collect(range(0.02; step = 0.005, length = n_buses))
    V = e_state .+ im .* f_state
    for i in 1:n_buses
        data.bus_magnitude[i, 1] = abs(V[i])
        data.bus_angles[i, 1] = atan(f_state[i], e_state[i])
    end

    offs, bs, total = PF.compute_mixed_bus_state_offsets(bt)
    n_lccs = size(data.lcc.p_set, 1)
    x = zeros(total + 4 * n_lccs)
    # REF P/Q slots; set so P_slack_total == 0 (Q_gen recovery is independent).
    P_net_set = zeros(n_buses)
    x[Int(offs[ref_bus])] = P_net_set[ref_bus]
    x[Int(offs[ref_bus]) + 1] = 0.0

    subnetworks = Dict{Int64, Vector{Int64}}(ref_bus => collect(1:n_buses))
    spf = SparseArrays.spzeros(Float64, n_buses)
    for i in 1:n_buses
        spf[i] = 1.0 / n_buses
    end

    PF.mixed_finalize_bus_injections!(
        data, x, offs, spf, subnetworks,
        e_state, f_state, 1,
    )

    # Analytic expected Q_gen from the RAW network + full load total.
    Y_raw = SparseArrays.sparse(ComplexF64.(data.power_network_matrix.data))
    S_net_raw = V[pv_bus] * conj((Y_raw * V)[pv_bus])
    Q_load_total =
        PF.get_bus_reactive_power_total_withdrawals(data, pv_bus, 1)
    Q_gen_expected = imag(S_net_raw) + Q_load_total
    inj_expected =
        Q_gen_expected + data.bus_reactive_power_withdrawals[pv_bus, 1]

    @test data.bus_reactive_power_injections[pv_bus, 1] ≈ inj_expected atol =
        1e-10

    # The buggy (Y_bus_eff) path would land here — assert we are NOT off by
    # β_Q·|V|² (the const-Z double-count signature). Build the const-Z-folded
    # Y_bus_eff locally only to compute the buggy reference; the finalize
    # itself must NOT consume it (it walks the raw Y-bus internally).
    Y_bus_eff =
        SparseArrays.sparse(ComplexF64.(data.power_network_matrix.data))
    PF.fold_zip_constant_z!(Y_bus_eff, data, 1)
    S_net_eff = V[pv_bus] * conj((Y_bus_eff * V)[pv_bus])
    Q_gen_buggy = imag(S_net_eff) + Q_load_total
    inj_buggy = Q_gen_buggy + data.bus_reactive_power_withdrawals[pv_bus, 1]
    @test !isapprox(
        data.bus_reactive_power_injections[pv_bus, 1],
        inj_buggy;
        atol = 1e-8,
    )
    # fold_zip_constant_z! adds complex(β_P, -β_Q) to the diagonal, so the
    # buggy Y_bus_eff path inflates imag(S) by exactly β_Q·|V|².
    @test isapprox(
        inj_buggy - inj_expected,
        β_Q * abs2(V[pv_bus]);
        atol = 1e-10,
    )
end
