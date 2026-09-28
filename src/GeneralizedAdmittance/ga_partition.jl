struct GAPartition
    s_ix::Vector{Int}
    v_ix::Vector{Int}
    q_ix::Vector{Int}
    l_ix::Vector{Int}
    Vset::Vector{Float64}
    n_pv::Int
end

n_v(p::GAPartition) = length(p.v_ix)
n_q(p::GAPartition) = length(p.q_ix)
n_l(p::GAPartition) = length(p.l_ix)

function _ga_vsc_ac_voltage_targets(data::ACPowerFlowData, time_step::Int)
    dcn = get_dc_network(data)
    targets = Dict{Int, Float64}()
    for c in 1:n_vsc_converters(dcn)
        if controls_ac_voltage(dcn.converter_mode[c])
            targets[dcn.converter_ac_bus_ix[c]] = dcn.vac_set[c, time_step]
        end
    end
    return targets
end

function GAPartition(data::ACPowerFlowData, time_step::Int, ac_vset::Dict{Int, Float64})
    ref, pv, pq = bus_type_idx(data, time_step)
    vsc_ac = filter(ix -> haskey(ac_vset, ix), pq)
    q = filter(ix -> !haskey(ac_vset, ix), pq)
    if isempty(ref)
        error("GeneralizedAdmittanceACPowerFlow: no REF bus at time step $time_step.")
    end
    if isempty(q)
        error(
            "GeneralizedAdmittanceACPowerFlow needs at least one PQ bus (time step $time_step).",
        )
    end
    v = vcat(pv, vsc_ac)
    Vset = vcat(data.bus_magnitude[pv, time_step], [ac_vset[ix] for ix in vsc_ac])
    return GAPartition(ref, v, q, vcat(v, q), Vset, length(pv))
end

struct GABlocks
    Ynet::SparseMatrixCSC{ComplexF64, Int64}
    Yll::SparseMatrixCSC{ComplexF64, Int64}
    Yll_net_nz::Vector{ComplexF64}
    Yll_diag::Vector{Int}
    Yqq::SparseMatrixCSC{ComplexF64, Int64}
    Yqq_net_nz::Vector{ComplexF64}
    Yqq_diag::Vector{Int}
    Yvv::SparseMatrixCSC{ComplexF64, Int64}
    Yvq::SparseMatrixCSC{ComplexF64, Int64}
    Yqv::SparseMatrixCSC{ComplexF64, Int64}
    Yls::SparseMatrixCSC{ComplexF64, Int64}
end

_ga_diag_positions(A::SparseMatrixCSC) = [_nz_index(A, k, k) for k in axes(A, 2)]

function GABlocks(data::ACPowerFlowData, part::GAPartition)
    Ynet = SparseMatrixCSC{ComplexF64, Int64}(PNM.get_data(get_power_network_matrix(data)))
    Yll = Ynet[part.l_ix, part.l_ix]
    Yqq = Ynet[part.q_ix, part.q_ix]
    return GABlocks(Ynet, Yll, copy(SparseArrays.nonzeros(Yll)), _ga_diag_positions(Yll),
        Yqq, copy(SparseArrays.nonzeros(Yqq)), _ga_diag_positions(Yqq),
        Ynet[part.v_ix, part.v_ix], Ynet[part.v_ix, part.q_ix], Ynet[part.q_ix, part.v_ix],
        Ynet[part.l_ix, part.s_ix])
end

# Shunts go into stored diagonal slots so the pattern never changes (KLU refactor needs that).
function _ga_set_shunts!(b::GABlocks, y::Vector{ComplexF64}, nv::Int)
    nz_ll = SparseArrays.nonzeros(b.Yll)
    copyto!(nz_ll, b.Yll_net_nz)
    for k in eachindex(b.Yll_diag)
        nz_ll[b.Yll_diag[k]] += y[k]
    end
    nz_qq = SparseArrays.nonzeros(b.Yqq)
    copyto!(nz_qq, b.Yqq_net_nz)
    for j in eachindex(b.Yqq_diag)
        nz_qq[b.Yqq_diag[j]] += y[nv + j]
    end
    return
end
