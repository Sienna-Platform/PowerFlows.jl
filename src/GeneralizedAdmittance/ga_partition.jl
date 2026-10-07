struct GAPartition
    s_ix::Vector{Int}
    v_ix::Vector{Int}
    q_ix::Vector{Int}
    l_ix::Vector{Int}
    Vset::Vector{Float64}
    island_of_l::Vector{Int}
    n_islands::Int
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
    Vset = vcat(get_bus_magnitude(data)[pv, time_step], [ac_vset[ix] for ix in vsc_ac])
    l = vcat(v, q)
    island_of_l, n_islands = _ga_islands(data, l, time_step)
    return GAPartition(ref, v, q, l, Vset, island_of_l, n_islands)
end

# Island id per ℓ-bus: the REF P row of the polar residual sums its island's P rows.
function _ga_islands(data::ACPowerFlowData, l::Vector{Int}, time_step::Int)
    groups = _find_subnetworks_for_reference_buses(
        PNM.get_data(get_power_network_matrix(data)),
        view(get_bus_type(data), :, time_step),
    )
    pos = Dict(ix => j for (j, ix) in enumerate(l))
    island_of_l = zeros(Int, length(l))
    for (island, members) in enumerate(values(groups))
        for ix in members
            if haskey(pos, ix)
                island_of_l[pos[ix]] = island
            end
        end
    end
    return island_of_l, length(groups)
end

# l = vcat(v, q), so net_diag[nv + j] is the Yqq diagonal.
struct GABlocks
    Yll::SparseMatrixCSC{ComplexF64, Int64}
    Yll_diag::Vector{Int}
    Yqq::SparseMatrixCSC{ComplexF64, Int64}
    Yqq_diag::Vector{Int}
    net_diag::Vector{ComplexF64}
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
    Yll_diag = _ga_diag_positions(Yll)
    return GABlocks(Yll, Yll_diag, Yqq, _ga_diag_positions(Yqq),
        SparseArrays.nonzeros(Yll)[Yll_diag],
        Ynet[part.v_ix, part.v_ix], Ynet[part.v_ix, part.q_ix], Ynet[part.q_ix, part.v_ix],
        Ynet[part.l_ix, part.s_ix])
end

# Shunts go into stored diagonal slots so the pattern never changes (KLU refactor needs that).
function _ga_set_shunts!(b::GABlocks, y::Vector{ComplexF64}, nv::Int)
    nz_ll = SparseArrays.nonzeros(b.Yll)
    for k in eachindex(b.Yll_diag)
        nz_ll[b.Yll_diag[k]] = b.net_diag[k] + y[k]
    end
    nz_qq = SparseArrays.nonzeros(b.Yqq)
    for j in eachindex(b.Yqq_diag)
        nz_qq[b.Yqq_diag[j]] = b.net_diag[nv + j] + y[nv + j]
    end
    return
end
