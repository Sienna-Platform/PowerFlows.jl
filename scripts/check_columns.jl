using PowerFlows, PowerSystemCaseBuilder
sys = build_system(MatpowerTestSystems, "matpower_case5_sys")
dc = solve_power_flow(DCPowerFlow(), sys)
println(names(dc["1"]["flow_results"]))
