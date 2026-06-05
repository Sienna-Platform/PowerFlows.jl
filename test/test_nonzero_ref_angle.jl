# Regression test for DC power flows on systems whose reference bus has a nonzero
# stored angle (e.g. a case exported from a solved AC/PSS\e case). The reduced DC
# solve computes angles relative to 0 at the ref bus, but `bus_angles` is initialized
# from the system data — previously the ref bus row kept its stored angle while all
# other rows held solved (relative-to-zero) values, so flow extraction `BA' * θ` put a
# spurious, time-constant flow `θ_stored / x` on every arc incident to the ref bus.
#
# The required behavior: flows are invariant to the stored ref bus angle, KCL holds at
# every bus (including the ref bus), and reported angles are re-referenced so the ref
# bus honors its stored angle.

# Net flow out of each bus index implied by the arc flows.
function _net_flow_out(data::PF.PowerFlowData, t::Int)
    bus_lookup = PF.get_bus_lookup(data)
    ft = PF.get_arc_active_power_flow_from_to(data)
    net_out = zeros(length(bus_lookup))
    for (arc, ix) in PF.get_arc_lookup(data)
        net_out[bus_lookup[arc[1]]] += ft[ix, t]
        net_out[bus_lookup[arc[2]]] -= ft[ix, t]
    end
    return net_out
end

@testset "DC power flow with nonzero stored ref bus angle" begin
    REF_ANGLE = 0.12345  # rad; anything nonzero exercises the bug

    for pf_type in (DCPowerFlow, PTDFDCPowerFlow, vPTDFDCPowerFlow)
        # Identical systems except for the stored angle at the reference bus.
        sys_zero = PSB.build_system(PSB.PSITestSystems, "c_sys14"; add_forecasts = false)
        sys_shifted =
            PSB.build_system(PSB.PSITestSystems, "c_sys14"; add_forecasts = false)
        get_ref_bus(sys) = only(
            collect(
                PSY.get_components(
                    x -> PSY.get_bustype(x) == PSY.ACBusTypes.REF,
                    PSY.ACBus,
                    sys,
                ),
            ),
        )
        PSY.set_angle!(get_ref_bus(sys_zero), 0.0)
        PSY.set_angle!(get_ref_bus(sys_shifted), REF_ANGLE)
        ref_bus_number = PSY.get_number(get_ref_bus(sys_shifted))

        data_zero = PowerFlowData(pf_type(; correct_bustypes = true), sys_zero)
        data_shifted = PowerFlowData(pf_type(; correct_bustypes = true), sys_shifted)
        solve_power_flow!(data_zero)
        solve_power_flow!(data_shifted)
        @test all(PF.get_converged(data_zero))
        @test all(PF.get_converged(data_shifted))

        # 1. Flows must be invariant to the stored ref bus angle.
        @test isapprox(
            PF.get_arc_active_power_flow_from_to(data_shifted),
            PF.get_arc_active_power_flow_from_to(data_zero);
            atol = 1e-9,
        )
        @test isapprox(
            PF.get_arc_active_power_flow_to_from(data_shifted),
            PF.get_arc_active_power_flow_to_from(data_zero);
            atol = 1e-9,
        )

        # 2. KCL: the net flow out of every non-ref bus must equal that bus's net
        # injection. (The ref bus is excluded: its balance equation is the redundant
        # one, so any system-wide injection imbalance — e.g. the AC-case losses baked
        # into c_sys14's device setpoints — lands there by design.) Before the fix this
        # failed at the ref bus's neighbors by O(θ_stored / x).
        bus_lookup = PF.get_bus_lookup(data_shifted)
        ref_ix = bus_lookup[ref_bus_number]
        t = 1
        net_out = _net_flow_out(data_shifted, t)
        net_injection = [
            data_shifted.bus_active_power_injections[ix, t] -
            PF.get_bus_active_power_total_withdrawals(data_shifted, ix, t)
            for ix in eachindex(net_out)
        ]
        non_ref = setdiff(eachindex(net_out), ref_ix)
        @test isapprox(net_out[non_ref], net_injection[non_ref]; atol = 1e-8)

        # 3. Reported angles honor the stored reference: the ref bus sits at its stored
        # angle and every other bus is shifted by exactly that constant, so angle
        # differences (and hence flows) are unchanged.
        θ_zero = PF.get_bus_angles(data_zero)
        θ_shifted = PF.get_bus_angles(data_shifted)
        @test isapprox(θ_shifted[ref_ix, t], REF_ANGLE; atol = 1e-12)
        @test isapprox(θ_shifted, θ_zero .+ REF_ANGLE; atol = 1e-9)
    end
end
