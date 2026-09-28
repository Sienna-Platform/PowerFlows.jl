@testset "GA: dense reference reproduces NR on c_sys5" begin
    sys = PSB.build_system(PSB.PSITestSystems, "c_sys5"; add_forecasts = false)
    data_nr = PowerFlowData(ACPowerFlow{NewtonRaphsonACPowerFlow}(), sys)
    @test solve_power_flow!(data_nr)
    p = ga_dense_problem(PowerFlowData(ACPowerFlow{NewtonRaphsonACPowerFlow}(), sys))
    nv = length(p.v_ix)
    q_nr = [
        data_nr.bus_reactive_power_withdrawals[ix, 1] -
        data_nr.bus_reactive_power_injections[ix, 1] for ix in p.v_ix
    ]
    y = vcat(complex.(real.(p.s[1:nv]), -q_nr) ./ p.Vset .^ 2,
        conj.(p.s[(nv + 1):end]) ./ p.u_ref[(nv + 1):end] .^ 2)
    steps = ga_dense_reference(p, y)
    @test last(steps).gap <= 1e-11
    @test maximum(abs.(abs.(last(steps).u) .- data_nr.bus_magnitude[p.l_ix, 1])) < 1e-8
    @test maximum(abs.(angle.(last(steps).u) .- data_nr.bus_angles[p.l_ix, 1])) < 1e-8
end
