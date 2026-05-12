@testset "Rectangular CI: compute_bus_state_offsets" begin
    @testset "all PQ" begin
        bt = fill(PSY.ACBusTypes.PQ, 5)
        off, bs, total = PF.compute_bus_state_offsets(bt)
        @test off == Int32[1, 3, 5, 7, 9, 11]
        @test bs == fill(Int8(2), 5)
        @test total == 10
    end

    @testset "mixed REF/PV/PQ" begin
        bt = [
            PSY.ACBusTypes.REF,
            PSY.ACBusTypes.PV,
            PSY.ACBusTypes.PQ,
            PSY.ACBusTypes.PV,
            PSY.ACBusTypes.PQ,
        ]
        off, bs, total = PF.compute_bus_state_offsets(bt)
        # REF=2, PV=3, PQ=2, PV=3, PQ=2 ⇒ 12 total
        @test off == Int32[1, 3, 6, 8, 11, 13]
        @test bs == Int8[2, 3, 2, 3, 2]
        @test total == 12
    end

    @testset "all PV" begin
        bt = fill(PSY.ACBusTypes.PV, 4)
        off, bs, total = PF.compute_bus_state_offsets(bt)
        @test off == Int32[1, 4, 7, 10, 13]
        @test bs == fill(Int8(3), 4)
        @test total == 12
    end
end

@testset "Rectangular CI: fold_zip_constant_z!" begin
    @testset "sign convention" begin
        sys = PSB.build_system(PSB.PSITestSystems, "c_sys5")
        pf = ACPowerFlow{RectangularCurrentInjectionACPowerFlow}()
        data = PF.PowerFlowData(pf, sys)
        Y = data.power_network_matrix.data
        Y_eff = SparseArrays.sparse(ComplexF64.(Y))
        # Synthetic const-Z load at bus 1: β_P = 0.5, β_Q = 0.3 at V₀ = 1.0
        data.bus_active_power_constant_impedance_withdrawals[1, 1] = 0.5
        data.bus_reactive_power_constant_impedance_withdrawals[1, 1] = 0.3
        original = data.bus_magnitude[1, 1]
        data.bus_magnitude[1, 1] = 1.0
        Y_before = copy(Y_eff)
        PF.fold_zip_constant_z!(Y_eff, data, 1)
        @test Y_eff[1, 1] - Y_before[1, 1] ≈ complex(0.5, -0.3) atol = 1e-12
        # Other diagonals untouched
        for i in 2:size(Y_eff, 1)
            @test Y_eff[i, i] == Y_before[i, i]
        end
        data.bus_magnitude[1, 1] = original
    end

    @testset "zero β skipped" begin
        sys = PSB.build_system(PSB.PSITestSystems, "c_sys5")
        pf = ACPowerFlow{RectangularCurrentInjectionACPowerFlow}()
        data = PF.PowerFlowData(pf, sys)
        Y_eff = SparseArrays.sparse(ComplexF64.(data.power_network_matrix.data))
        Y_before = copy(Y_eff)
        # Default ZIP-Z withdrawals are 0 in c_sys5
        PF.fold_zip_constant_z!(Y_eff, data, 1)
        @test Y_eff == Y_before
    end
end

@testset "Rectangular CI: rect_initial_state! / rect_update_data! roundtrip" begin
    sys = PSB.build_system(PSB.PSITestSystems, "c_sys5")
    pf = ACPowerFlow{RectangularCurrentInjectionACPowerFlow}()
    data = PF.PowerFlowData(pf, sys)
    bt = view(data.bus_type, :, 1)
    off, bs, total = PF.compute_bus_state_offsets(bt)
    n_lccs = size(data.lcc.p_set, 1)
    total_state = total + 4 * n_lccs

    x = Vector{Float64}(undef, total_state)
    PF.rect_initial_state!(x, data, off, bs, 1)

    # Bus 1 in c_sys5 is REF: x[off[1]] = P_gen_net, x[off[1]+1] = Q_gen_net
    # The (e,f) representation must hold for PQ/PV buses
    Vm_before = copy(view(data.bus_magnitude, :, 1))
    θ_before = copy(view(data.bus_angles, :, 1))
    Q_inj_before = copy(view(data.bus_reactive_power_injections, :, 1))

    # Perturb data; round-trip via rect_update_data!. PV-bus bus_magnitude is
    # preserved (not overwritten) since V_set must be retained for the ΔV² row.
    for i in 1:length(bt)
        if bt[i] == PSY.ACBusTypes.PQ
            data.bus_magnitude[i, 1] = NaN
        end
        bt[i] != PSY.ACBusTypes.REF && (data.bus_angles[i, 1] = NaN)
    end
    PF.rect_update_data!(data, x, off, bs, 1)

    for i in 1:length(bt)
        if bt[i] == PSY.ACBusTypes.PQ
            @test data.bus_magnitude[i, 1] ≈ Vm_before[i] atol = 1e-12
        end
        if bt[i] != PSY.ACBusTypes.REF
            @test data.bus_angles[i, 1] ≈ θ_before[i] atol = 1e-12
        end
    end
end
