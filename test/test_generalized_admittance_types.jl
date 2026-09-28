@testset "GA: type construction and rejections" begin
    GA = GeneralizedAdmittanceACPowerFlow
    @test typeof(ACPowerFlow{GA}()) === ACPolarPowerFlow{GA}
    @test_throws ArgumentError ACPowerFlow{GA}(; check_reactive_power_limits = true)
    @test_throws ArgumentError ACPowerFlow{GA}(;
        distribute_slack_proportional_to_headroom = true)
    @test_throws ArgumentError ACPowerFlow{GA}(;
        generator_slack_participation_factors = Dict((PSY.ThermalStandard, "gen") => 1.0))
    @test_throws ArgumentError ACPowerFlow{GA}(; control_discrete_devices = true)
    @test_throws ArgumentError ACPowerFlow{GA}(; area_interchange_control = true)
    @test_throws ArgumentError PF.ACRectangularPowerFlow{GA}()
    @test_throws ArgumentError PF.ACMixedPowerFlow{GA}()
end
