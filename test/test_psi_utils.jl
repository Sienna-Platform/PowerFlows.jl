@testset "contributes active/reactive power" begin
    device_types = Type[PSY.StaticInjection]
    while !isempty(device_types)
        T = pop!(device_types)
        if !isabstracttype(T)
            instance = T(nothing)
            if PF.contributes_active_power(instance)
                @test hasmethod(PF.active_power_contribution_type, Tuple{T})
                if T == PSY.StandardLoad
                    @test hasmethod(
                        PSY.get_constant_active_power,
                        Tuple{T, IS.AbstractUnitSystem},
                    )
                elseif T == PSY.SynchronousCondenser
                    @test hasmethod(PSY.get_active_power_losses, Tuple{T})
                else
                    @test hasmethod(PSY.get_active_power, Tuple{T, IS.AbstractUnitSystem})
                end
            end
            # for FACTS, reactive_power_required is the equivalent of get_reactive_power,
            # but it depends on control mode, and PSY isn't updated for those modes yet.
            if PF.contributes_reactive_power(instance) && T != PSY.FACTSControlDevice
                @test hasmethod(PF.reactive_power_contribution_type, Tuple{T})
                if T == PSY.StandardLoad
                    @test hasmethod(
                        PSY.get_constant_reactive_power,
                        Tuple{T, IS.AbstractUnitSystem},
                    )
                else
                    @test hasmethod(PSY.get_reactive_power, Tuple{T, IS.AbstractUnitSystem})
                end
            end
        end
        append!(device_types, InteractiveUtils.subtypes(T))
    end
end

@testset "SwitchedAdmittance: empty `number_engaged` is 0, a mismatched length errors" begin
    y_increase = ComplexF64[0.01 + 0.02im, 0.03 + 0.04im]
    @test PF._switched_admittance(nothing, Int[], y_increase) == 0.0 + 0.0im
    @test PF._switched_admittance(nothing, [1, 2], y_increase) ==
          (0.01 + 0.02im) + 2 * (0.03 + 0.04im)
    @test_throws DimensionMismatch PF._switched_admittance(nothing, [1], y_increase)
end

@testset "ExponentialLoad: exponents 0/1/2 map onto the ZIP withdrawals, others error" begin
    function exponential_system(α, β)
        sys = PSB.build_system(PSB.PSITestSystems, "c_sys5"; add_forecasts = false)
        load = first(PSY.get_components(PSY.PowerLoad, sys))
        bus = PSY.get_bus(load)
        P0 = PSY.get_active_power(load, u"SU")
        Q0 = PSY.get_reactive_power(load, u"SU")
        PSY.set_available!(load, false)
        PSY.add_component!(
            sys,
            PSY.ExponentialLoad(;
                name = "exp_load",
                available = true,
                bus = bus,
                active_power = PSY.get_active_power(load, u"NU"),
                reactive_power = PSY.get_reactive_power(load, u"NU"),
                α = α,
                β = β,
                base_power = PSY.get_base_power(load, u"NU"),
                max_active_power = PSY.get_max_active_power(load, u"NU"),
                max_reactive_power = PSY.get_max_reactive_power(load, u"NU"),
                input_basis = u"NU",
            ),
        )
        return sys, PSY.get_number(bus), P0, Q0
    end
    sys, bus_no, P0, Q0 = exponential_system(1.0, 2.0)
    @test P0 > 0.0
    data = PowerFlowData(ACPowerFlow(), sys)
    ix = PF.get_bus_lookup(data)[bus_no]
    @test iszero(data.bus_active_power_withdrawals[ix, 1])
    @test iszero(data.bus_reactive_power_withdrawals[ix, 1])
    @test data.bus_active_power_constant_current_withdrawals[ix, 1] ≈ P0
    @test data.bus_reactive_power_constant_impedance_withdrawals[ix, 1] ≈ Q0

    # DC reads only constant power withdrawals: at V = 1 p.u. every ZIP term equals P0.
    data_dc = PowerFlowData(DCPowerFlow(), sys)
    @test data_dc.bus_active_power_withdrawals[ix, 1] ≈ P0

    # PSS/E export puts P0 and Q0 in the slot of their exponent.
    @test PF._psse_zip_field(2, 2, P0) == P0
    @test PF._psse_zip_field(2, 1, P0) == PF.PSSE_DEFAULT

    sys0, _, _, _ = exponential_system(0.0, 0.0)
    data0 = PowerFlowData(ACPowerFlow(), sys0)
    @test data0.bus_active_power_withdrawals[ix, 1] ≈ P0
    @test data0.bus_reactive_power_withdrawals[ix, 1] ≈ Q0

    sys_bad, _, _, _ = exponential_system(1.5, 0.0)
    @test_throws r"ExponentialLoad exp_load has voltage exponent 1.5" PowerFlowData(
        ACPowerFlow(),
        sys_bad,
    )
end
