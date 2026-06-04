# Mixed Current-Power Balance Formulation

`ACMixedPowerFlow` implements the mixed current-power balance (MCPB) AC
power flow formulation in rectangular coordinates. This MCPB formulation
adapts the *Hybrid Current-Power Balance* method of Abhyankar, Cui & Flueck,
*"Fast Power Flow Analysis using a Hybrid Current-Power Balance Formulation in
Rectangular Coordinates"*, with implementation differences from that paper
(the residual sign convention is aligned with
[`ACRectangularPowerFlow`](@ref), and the PQ rows use an
imaginary-current-balance-first ordering), which is why this implementation is
named "Mixed" rather than "Hybrid". It is the third AC formulation in
PowerFlows.jl, alongside [`ACPolarPowerFlow`](@ref) (power balance, polar
state) and [`ACRectangularPowerFlow`](@ref) (augmented Da Costa current
injection).

The voltage at each bus is expressed in rectangular coordinates,
$V_i = e_i + j f_i$, and the state vector groups the two unknowns per bus as
$[e_i, f_i]$. The system has exactly $2n$ equations and unknowns — the
smallest state vector of the three AC formulations (the Da Costa rectangular
form adds a third variable per PV bus).

## Motivation

The computational bottleneck of the polar power-balance form is the Jacobian:
every entry is a nonlinear function of voltage magnitudes and angles, so the
entire matrix must be rebuilt and refactored each Newton iteration. The MCPB
formulation chooses, per bus type, the equation whose Jacobian rows are
*cheapest*: the off-diagonal blocks of the PQ rows become the constant
admittance-matrix entries and are assembled once. This reduces per-iteration
Jacobian work on large systems.

## Equations

Let $Y_{bus} = G + jB$. The network current injected at bus $i$ has real and
imaginary parts

```math
\begin{aligned}
I^{r}_i &= \sum_k \left(G_{ik} e_k - B_{ik} f_k\right) \\
I^{i}_i &= \sum_k \left(B_{ik} e_k + G_{ik} f_k\right)
\end{aligned}
```

``P_i`` and ``Q_i`` denote the effective constant-power net injection (generation minus
constant-power load). Constant-impedance ZIP load is folded into the
admittance matrix diagonal; constant-current ZIP load is subtracted as
$-\,\mathrm{const\_I}\cdot|V_i|$ — identical to the
[`ACRectangularPowerFlow`](@ref) treatment, which gives full ZIP and LCC
feature parity for free.

### PQ buses — current balance (divided form)

```math
\begin{aligned}
I^{i}_i - \frac{P_i f_i - Q_i e_i}{|V_i|^2} &= 0 \\
I^{r}_i - \frac{P_i e_i + Q_i f_i}{|V_i|^2} &= 0
\end{aligned}
```

The two rows are stored *imaginary-balance first*. With this ordering the
nonzero $B_{ii}$ lands on the block diagonal instead of $G_{ii}$, which is
$\approx 0$ for zero-injection buses; this avoids a zero pivot for buses with
no load or generation. (KLU's BTF + AMD ordering with partial pivoting
already mitigates this, but the ordering is kept as a robustness measure for
very large systems with many zero-injection buses.)

### PV buses — power balance and voltage magnitude constraint

```math
\begin{aligned}
e_i I^{r}_i + f_i I^{i}_i - P_i &= 0 \\
e_i^2 + f_i^2 - |V^{\text{set}}_i|^2 &= 0
\end{aligned}
```

PV buses carry **only** $[e_i, f_i]$ — no auxiliary reactive-power variable.
The generator reactive output at PV buses is recovered after convergence from
the network current at the solved voltage.

### Reference bus

The slack-bus voltage is fixed; its two state slots carry $(P_{gen},
Q_{gen})$ and the residual is the divided current balance, identical to the
[`ACRectangularPowerFlow`](@ref) reference treatment, including the
distributed-slack convention in which the reference state variable carries the
whole subnetwork slack.

## Jacobian structure

| Block                    | PQ rows                                                              | PV rows                                                                                   |
|:------------------------ |:-------------------------------------------------------------------- |:----------------------------------------------------------------------------------------- |
| Off-diagonal $(k\neq i)$ | constant $\equiv$ $Y_{bus}$ real 2×2 block (assembled once)          | nonlinear $e_i G_{ik}+f_i B_{ik}$, $-e_i B_{ik}+f_i G_{ik}$ (refreshed each iteration)    |
| Diagonal                 | nonlinear divided-current partials + ZIP constant-current chain term | nonlinear power-balance partials; $\partial(\lvert V \rvert^2)/\partial(e,f)=(2e_i,2f_i)$ |

Only the PQ off-diagonal blocks are constant. They are written once at
construction; every other state-dependent entry is refreshed in place each
iteration through pre-computed `nonzeros` indices, and the KLU symbolic
factorization is reused across iterations. The Jacobian update kernel is
allocation-free.

## Solvers, parity, and robustness

`ACMixedPowerFlow{S}` supports the [`NewtonRaphsonACPowerFlow`](@ref),
[`TrustRegionACPowerFlow`](@ref) and [`LevenbergMarquardtACPowerFlow`](@ref)
solvers. Robust Homotopy and Gradient Descent operate on the polar
formulation only and are rejected at construction. Voltage stability factors,
loss factors, and the DC robust fallback are not provided for this
formulation.

The formulation is validated to machine precision against both the polar and
the rectangular formulations across ZIP loads, LCC HVDC (including converter
terminals at PV buses), distributed slack, reactive-power-limit switching,
network reductions, and multi-period studies, for all three solvers. From a
flat start (true flat and the paper's modified flat start with PV magnitudes
at setpoint and all angles at the reference angle), Newton-Raphson and Trust
Region converge on systems up to 10 000 buses with the same iteration count
as the polar formulation and a near-degenerate-voltage guard
(`V_FLOOR2 = 1e-16`, floored $|V_i|^2$) preventing blow-up.

Benchmarking on the 2 000- and 10 000-bus ACTIVSg systems shows the mixed
form is roughly **1.4–1.8× faster than the polar power-balance form** (and
~1.7–1.8× per Newton iteration) under flat starts — consistent with, though
at the lower end of, the original paper's range (the gap narrows because KLU
already gives the polar formulation a reused symbolic factorization). It does
**not** outperform the existing [`ACRectangularPowerFlow`](@ref), which is
also a current-balance form; the practical value of `ACMixedPowerFlow` is its
minimal $2n$ state vector for memory-bound very-large-scale studies and as a
validated reference implementation of the paper.

## Usage

```julia
using PowerFlows

# Newton-Raphson (default solver)
pf = ACMixedPowerFlow()
results = solve_power_flow(pf, system)

# Trust Region from a flat start
pf = ACMixedPowerFlow{TrustRegionACPowerFlow}(; enhanced_flat_start = true)
results = solve_power_flow(pf, system)

# Levenberg-Marquardt
pf = ACMixedPowerFlow{LevenbergMarquardtACPowerFlow}()
results = solve_power_flow(pf, system)
```

`ACMixedPowerFlow` accepts the same keyword arguments as
[`ACRectangularPowerFlow`](@ref) (`check_reactive_power_limits`,
`generator_slack_participation_factors`, `enhanced_flat_start`,
`distribute_slack_proportional_to_headroom`, `network_reductions`,
`time_steps`, `correct_bustypes`, `solver_settings`, …).
