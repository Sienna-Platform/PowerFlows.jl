@testset "ACMixedPowerFlow type" begin
    pf = ACMixedPowerFlow{NewtonRaphsonACPowerFlow}()
    @test PowerFlows.get_calculate_loss_factors(pf) == false
    @test PowerFlows.get_calculate_voltage_stability_factors(pf) == false
    @test PowerFlows.get_robust_power_flow(pf) == false
    @test PowerFlows.get_enhanced_flat_start(pf) == true
    @test_throws ArgumentError ACMixedPowerFlow{RobustHomotopyPowerFlow}()
    @test_throws ArgumentError ACMixedPowerFlow{GradientDescentACPowerFlow}()
end
