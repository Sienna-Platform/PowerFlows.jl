# LCC closed form (spec §3.7). With the tail pinned at α = α_min and γ = γ_min, each
# terminal's complex power is constant: P from the setpoint + DC-line balance, Q from
# Q = √((V·t)·K·I)² − P². The products (V·t) are |V|-independent, so the P+jQ withdrawals
# are fixed per time step and fold into the constant sP terms. Terminal powers are
# withdrawals (positive = absorbed).

function _ga_lcc_powers(setpoint_at_rectifier::Bool, p_set::Float64, R::Float64,
    I::Float64)
    if setpoint_at_rectifier
        return (p_set, R * I^2 - p_set)
    end
    return (p_set + R * I^2, -p_set)
end

function _ga_lcc_q(vt::Float64, I::Float64, P::Float64, l::Int, side::String)
    S = vt * SQRT6_DIV_PI * I
    d = S^2 - P^2
    if vt <= 0.0 || d < -GA_LCC_FEASIBILITY_TOL
        error(
            "GeneralizedAdmittanceACPowerFlow: LCC $l $side is infeasible: V·t = $vt, " *
            "|S| = $(abs(S)) < |P| = $(abs(P)). Check the commutation reactance, DC " *
            "current, and minimum firing/extinction angle.",
        )
    end
    return sqrt(max(d, 0.0))
end

function _ga_add_lcc_terms!(conv::GAConverterTerms, data::ACPowerFlowData, time_step::Int)
    for l in eachindex(data.lcc.bus_indices)
        I = get_lcc_i_dc(data)[l, time_step]
        if iszero(I)
            continue
        end
        P_r, P_i = _ga_lcc_powers(
            get_lcc_setpoint_at_rectifier(data)[l],
            get_lcc_p_set(data)[l, time_step],
            get_lcc_dc_line_resistance(data)[l],
            I,
        )
        KI = SQRT6_DIV_PI * I
        vt_r =
            (P_r / KI + get_lcc_rectifier_transformer_reactance(data)[l] * I / sqrt(2)) /
            cos(get_lcc_rectifier_min_thyristor_angle(data)[l])
        vt_i =
            (-P_i / KI + get_lcc_inverter_transformer_reactance(data)[l] * I / sqrt(2)) /
            cos(get_lcc_inverter_min_thyristor_angle(data)[l])
        fb, tb = data.lcc.bus_indices[l]
        conv.p_lcc[fb] += P_r
        conv.q_lcc[fb] += _ga_lcc_q(vt_r, I, P_r, l, "rectifier")
        conv.p_lcc[tb] += P_i
        conv.q_lcc[tb] += _ga_lcc_q(vt_i, I, P_i, l, "inverter")
    end
    return
end
