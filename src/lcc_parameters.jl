struct LCCConverterParameters
    bus::Vector{Int} # bus indices, not bus numbers.
    tap::Matrix{Float64}
    thyristor_angle::Matrix{Float64}
    phi::Matrix{Float64}
    transformer_reactance::Vector{Float64}
    min_thyristor_angle::Vector{Float64}
    # Scheduled tap setting (time-invariant). Used to pin the tap state when the
    # converter carries no DC current (i_dc = 0, e.g. a 0-MW transfer setpoint):
    # P_lcc ≡ 0 makes the P-setpoint / DC-line-balance equations vacuous and the
    # tap states unconstrained, so they are replaced by `tap - tap_setpoint`.
    tap_setpoint::Vector{Float64}
end

LCCConverterParameters(n_time_steps::Int, n_lccs::Int) = LCCConverterParameters(
    zeros(Int, n_lccs),
    zeros(Float64, n_lccs, n_time_steps),
    zeros(Float64, n_lccs, n_time_steps),
    zeros(Float64, n_lccs, n_time_steps),
    zeros(Float64, n_lccs),
    zeros(Float64, n_lccs),
    zeros(Float64, n_lccs),
)

# TODO could add an lcc arc axis/arc lookup pair here and switch to matrices instead of vectors of dicts.
# or better yet, use an ArcAdmittanceMatrix from PNM. But this is fine for now: should only be a few LCCs.
struct LCCParameters
    # bus number pairs
    arcs::Vector{Tuple{Int, Int}}
    # all the following fields are ordered to match arcs:
    # bus index pairs. Slightly redundant: matches zip(rectifier.bus, inverter.bus)
    bus_indices::Vector{Tuple{Int, Int}}
    # (y_ff, y_tt) admittance pairs
    branch_admittances::Vector{Tuple{ComplexF64, ComplexF64}}
    arc_active_power_flow_from_to::Matrix{Float64}
    arc_active_power_flow_to_from::Matrix{Float64}
    arc_reactive_power_flow_from_to::Matrix{Float64}
    arc_reactive_power_flow_to_from::Matrix{Float64}
    setpoint_at_rectifier::Vector{Bool}
    p_set::Matrix{Float64}
    i_dc::Matrix{Float64}
    dc_line_resistance::Vector{Float64}
    rectifier::LCCConverterParameters
    inverter::LCCConverterParameters
end

LCCParameters(n_time_steps::Int, n_lccs::Int) = LCCParameters(
    fill((-1, -1), n_lccs),
    fill((-1, -1), n_lccs),
    fill((0, 0), n_lccs),
    zeros(Float64, n_lccs, n_time_steps),
    zeros(Float64, n_lccs, n_time_steps),
    zeros(Float64, n_lccs, n_time_steps),
    zeros(Float64, n_lccs, n_time_steps),
    falses(n_lccs),
    zeros(Float64, n_lccs, n_time_steps),
    zeros(Float64, n_lccs, n_time_steps),
    zeros(Float64, n_lccs),
    LCCConverterParameters(n_time_steps, n_lccs),
    LCCConverterParameters(n_time_steps, n_lccs),
)
