# VSC converter + DC-network data model.
#
# Every PSY DC component (point-to-point `TwoTerminalVSCLine`, and — for multi-terminal grids —
# `InterconnectingConverter`/`DCBus`/`TModelHVDCLine`) lowers into ONE internal `DCNetwork`. The
# Newton solver only ever sees `DCNetwork`: converter active/reactive injections `(P_c, Q_c)` and
# free DC-node voltages `V_dc` are appended as tail unknowns after the bus and LCC blocks, mirroring
# the LCC tail-unknown pattern. M1 (point-to-point VSC) is the isolated-2-node special case; M2
# (multi-terminal DC) is the same struct with more nodes/branches and the same kernels.

# Converter control modes. An `Int8`-backed `@enum` stored directly on `DCNetwork`: reading
# `dcn.converter_mode[c]` is type-stable (concrete enum), so the residual/Jacobian hot path calls
# the mode-specific kernels (`_vsc_r1`/`_vsc_r2`/…) with no dynamic dispatch. The kernels branch on the
# enum value — cheap and branch-predictable for the handful of converters.
#   ControlPQ        r1: P_c − P_set            r2: Q_c − Q_set
#   ControlPVac      r1: P_c − P_set            r2: |V_ac|² − V_set²
#   ControlVdc       r1: V_dc − V_set           r2: Q_c − Q_set            (DC-slack converter)
#   ControlVdcQ      r1: V_dc − V_set           r2: |V_ac|² − V_set²
#   ControlPVdcDroop r1: (V_dc − V_set) + k·P_c r2: Q_c − Q_set
@enum VSCControlMode::Int8 ControlPQ = 1 ControlPVac = 2 ControlVdc = 3 ControlVdcQ = 4 ControlPVdcDroop =
    5

# A mode fixes the DC-node voltage (makes its node a DC slack) iff it is a V_dc-control mode.
fixes_dc_voltage(m::VSCControlMode) = m == ControlVdc || m == ControlVdcQ

# Whether the PSY `dc_setpoint` field is a V_dc target (vs an active-power order) for a given mode.
# Drives how the overloaded `dc_setpoint` is split into `vdc_set` vs `p_set` at lowering.
uses_vdc_setpoint(m::VSCControlMode) =
    m == ControlVdc || m == ControlVdcQ || m == ControlPVdcDroop

# Whether the second control row pins AC voltage (|V_ac|² − V_set²) rather than reactive power.
controls_ac_voltage(m::VSCControlMode) = m == ControlPVac || m == ControlVdcQ

# Keyword-constructed (`@kwdef`) so the 27-field build sites name every field — many fields share
# `Vector{Float64}`/`Matrix{Float64}` types, so positional construction would silently corrupt on a
# field reorder. All fields default to empty, so a bare `DCNetwork()` is the empty network used for
# pure-AC systems (regression-safe — pure AC is untouched).
"""
Internal lowering of all PSY DC components into a single DC network solved jointly with the AC
buses. Fields ordered: converters (length `n_conv`), DC nodes (length `n_node`), DC branches
(length `n_branch`). Built once from the `System` in [`initialize_DCNetwork!`](@ref) and stored on
`PowerFlowData` behind a `Ref` (the struct is immutable; the network is assigned after the system
is scanned, exactly like `solver_cache`).
"""
Base.@kwdef struct DCNetwork
    # converters
    converter_ac_bus_ix::Vector{Int} = Int[]        # AC bus index (post network reduction)
    converter_dc_node_ix::Vector{Int} = Int[]       # DC node index this converter feeds
    converter_mode::Vector{VSCControlMode} = VSCControlMode[]  # per-converter control mode
    converter_ac_bus_number::Vector{Int} = Int[]    # AC bus number (results / diagnostics)
    loss_a::Vector{Float64} = Float64[]             # P_loss = a + b·I_c + c·I_c²
    loss_b::Vector{Float64} = Float64[]
    loss_c::Vector{Float64} = Float64[]
    s_max::Vector{Float64} = Float64[]              # capability-circle radius (MVA, device base)
    q_min::Vector{Float64} = Float64[]
    q_max::Vector{Float64} = Float64[]
    p_min::Vector{Float64} = Float64[]
    p_max::Vector{Float64} = Float64[]
    droop_k::Vector{Float64} = Float64[]            # DC-voltage droop gain (0 unless ControlPVdcDroop)
    p_set::Matrix{Float64} = zeros(Float64, 0, 0)   # (n_conv, n_time) control setpoints
    q_set::Matrix{Float64} = zeros(Float64, 0, 0)
    vac_set::Matrix{Float64} = zeros(Float64, 0, 0)
    vdc_set::Matrix{Float64} = zeros(Float64, 0, 0)
    p_c::Matrix{Float64} = zeros(Float64, 0, 0)     # (n_conv, n_time) solved state mirror
    q_c::Matrix{Float64} = zeros(Float64, 0, 0)
    # DC nodes. A node is a "slack" iff a V_dc-controlling converter pins its voltage. Every node
    # (slack or not) carries a V_dc state and a DC-KCL row; `node_is_slack` is used only to validate
    # that each DC subnet has an anchor (slack or droop).
    node_is_slack::Vector{Bool} = Bool[]
    node_number::Vector{Int} = Int[]                # DC-bus number, or -1 for a point-to-point node
    node_vdc::Matrix{Float64} = zeros(Float64, 0, 0)  # (n_node, n_time) solved V_dc state mirror
    # DC branches
    branch_from::Vector{Int} = Int[]                # DC node index
    branch_to::Vector{Int} = Int[]                  # DC node index
    branch_g::Vector{Float64} = Float64[]           # DC conductance (1/r)
    # dense DC nodal conductance (DC grids are tiny — dense is simplest and fastest)
    G_dc::Matrix{Float64} = zeros(Float64, 0, 0)    # (n_node, n_node)
end

n_vsc_converters(dcn::DCNetwork) = length(dcn.converter_ac_bus_ix)
n_dc_nodes(dcn::DCNetwork) = length(dcn.node_is_slack)
n_vsc_free_nodes(dcn::DCNetwork) = count(!, dcn.node_is_slack)
n_dc_branches(dcn::DCNetwork) = length(dcn.branch_from)

# Length of the tail this network appends to the Newton state: 2 per converter (P_c, Q_c) plus one
# V_dc per DC node. Every DC node carries a V_dc state and a DC-KCL row — a V_dc-controlling
# (slack) converter pins its node with a real `V_dc − V_set` control row rather than the node being
# removed from the state, which keeps the Jacobian uniform and non-degenerate. Non-singularity of
# the DC block is guaranteed by requiring ≥1 V_dc-controlling or droop converter per DC subnet
# (`_validate_dc_slacks`).
vsc_tail_length(dcn::DCNetwork) = 2 * n_vsc_converters(dcn) + n_dc_nodes(dcn)

has_dc_network(dcn::DCNetwork) = n_vsc_converters(dcn) > 0

# Pre-computed `nonzeros(Jv)` indices for the VSC tail entries of the rectangular-CI / MCPB Jacobian,
# so the per-iteration writer hits `Jvnz[...]` directly instead of `O(log nnz)` `Jv[r,c]` setindex
# (mirrors the `lcc_nz` cache). Built once at construction by `_build_vsc_nz_cache`.
struct VSCJacobianNZCache
    conv::Matrix{Int}    # 13 × n_conv; per-converter tail slots, order set by `_build_vsc_nz_cache`
    node::Vector{Int}    # n_node; DC-KCL node-diagonal slot
    branch::Vector{Int}  # 2·n_branch; DC-KCL off-diagonals, interleaved (from→to, to→from)
end
