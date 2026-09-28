# GA LCC closed form (spec §3.7): constant terminal P+jQ withdrawals must equal the NR
# converged terminal powers, and infeasible operating points must error.

@testset "GA: LCC closed form matches NR terminal powers" begin
    sys = system_from_openapi(
        PFP.PowerModelsData(joinpath(TEST_DATA_DIR, "case5_2_lcc.raw"));
        runchecks = false,
    )
    data_nr = PowerFlowData(ACPowerFlow{NewtonRaphsonACPowerFlow}(), sys)
    @test solve_power_flow!(data_nr)
    data = PowerFlowData(ACPowerFlow{NewtonRaphsonACPowerFlow}(), sys)
    n = size(data.bus_type, 1)
    conv = PF.GAConverterTerms(n)
    PF._ga_add_lcc_terms!(conv, data, 1)
    p_nr = zeros(n)
    q_nr = zeros(n)
    for l in eachindex(data_nr.lcc.bus_indices)
        fb, tb = data_nr.lcc.bus_indices[l]
        Vf, Vt = data_nr.bus_magnitude[fb, 1], data_nr.bus_magnitude[tb, 1]
        Pf, Pt = PF._lcc_ac_active_powers(data_nr, l, 1, Vf, Vt)
        yf, yt = data_nr.lcc.branch_admittances[l]
        p_nr[fb] += Pf
        p_nr[tb] += Pt
        q_nr[fb] -= Vf^2 * imag(yf)
        q_nr[tb] -= Vt^2 * imag(yt)
    end
    @test maximum(abs.(conv.p_lcc .- p_nr)) < 1e-8
    @test maximum(abs.(conv.q_lcc .- q_nr)) < 1e-8
end

@testset "GA: LCC infeasible operating point errors" begin
    @test_throws r"LCC 1 rectifier is infeasible" PF._ga_lcc_q(
        1.0,
        1.0,
        5.0,
        1,
        "rectifier",
    )
    @test PF._ga_lcc_q(2.0, 1.0, 0.0, 1, "inverter") ≈ 2.0 * PF.SQRT6_DIV_PI
end
