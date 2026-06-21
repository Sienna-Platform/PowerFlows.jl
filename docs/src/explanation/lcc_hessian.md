# LCC Second Derivatives (Hessian Blocks)

This page derives the second derivatives needed to extend the
[`RobustHomotopyPowerFlow`](@ref) Hessian assembly (and the spectral-radius
diagnostic) to systems with LCC HVDC lines. It is a companion to
[lcc_model.md](lcc_model.md), which lays out the residual and Jacobian rows.

The homotopy Hessian computes

```math
H(x) = t_k \left( J(x)^\top J(x) + \sum_k F_k(x)\, \nabla^2 F_k(x) \right) + (1-t_k)\, \text{diag}(\mathbb{1}_{\text{PQ }V}).
```

The `J^\top J` term and the network-only `∑ F_k ∇² F_k` block are already
handled by `_update_hessian_matrix_values!` in
[homotopy_hessian.jl](../../../src/RobustHomotopy/homotopy_hessian.jl). The
missing piece is the LCC contribution to `∑ F_k ∇² F_k`. This page enumerates
those contributions.

## Notation and setup

For one LCC, write the rectifier-side and inverter-side AC variables as
`(V_r, t_r, α_r)` and `(V_i, t_i, α_i)`, and let

```math
K := \frac{\sqrt{6}}{\pi}, \qquad
\beta_s := \frac{x_s\, I_{dc}}{\sqrt{2}} \quad (s \in \{r, i\}),
```

with `I_dc > 0` the (constant during the solve) DC-line current magnitude and
`x_s` the side-`s` transformer reactance. Define the `arccos` arguments

```math
u_r(V_r, t_r, \alpha_r) = \cos\alpha_r - \frac{\beta_r}{V_r\, t_r},
\qquad
u_i(V_i, t_i, \alpha_i) = -\cos\alpha_i - \frac{\beta_i}{V_i\, t_i}.
```

(`u_i` differs from `u_r` only by the sign convention — the helper
`_calculate_ϕ_lcc` is called with `-I_dc` on the inverter, which flips the
overall sign of the bracket.) Then `cos\phi_s = u_s` and
`\sin\phi_s = \sqrt{1 - u_s^2} \ge 0` strictly in the interior.

All formulas below are the **interior** formulas (`\sin\phi_s` bounded away
from 0). The clamp regime is handled in [a separate section](#clamp-regime).

## First derivatives of `u_s`

Useful intermediate quantities (rectifier shown; inverter differs only in
sign of `α` terms):

```math
\begin{aligned}
\frac{\partial u_r}{\partial V_r} &= \frac{\beta_r}{V_r^2 t_r}, &
\frac{\partial u_r}{\partial t_r} &= \frac{\beta_r}{V_r\, t_r^2}, &
\frac{\partial u_r}{\partial \alpha_r} &= -\sin\alpha_r, \\
\frac{\partial u_i}{\partial V_i} &= \frac{\beta_i}{V_i^2 t_i}, &
\frac{\partial u_i}{\partial t_i} &= \frac{\beta_i}{V_i\, t_i^2}, &
\frac{\partial u_i}{\partial \alpha_i} &= +\sin\alpha_i.
\end{aligned}
```

Second derivatives of `u_s`:

```math
\begin{aligned}
\frac{\partial^2 u_s}{\partial V_s^2} &= -\frac{2\beta_s}{V_s^3 t_s}, &
\frac{\partial^2 u_s}{\partial t_s^2} &= -\frac{2\beta_s}{V_s\, t_s^3}, &
\frac{\partial^2 u_s}{\partial V_s\,\partial t_s} &= -\frac{\beta_s}{V_s^2 t_s^2}, \\
\frac{\partial^2 u_r}{\partial \alpha_r^2} &= -\cos\alpha_r, &
\frac{\partial^2 u_i}{\partial \alpha_i^2} &= +\cos\alpha_i, &
\frac{\partial^2 u_s}{\partial V_s\,\partial \alpha_s} = \frac{\partial^2 u_s}{\partial t_s\,\partial \alpha_s} &= 0.
\end{aligned}
```

(Sign of `\partial^2 u/\partial V \partial t`: `\partial u/\partial V =
\beta/(V^2 t)`, so `\partial^2 u/\partial V \partial t = \beta/V^2 \cdot
\partial(1/t)/\partial t = -\beta/(V^2 t^2)`. Easy to miss.)

## Active power: closed form and Hessian

In the interior, the chain term in the `dP/d·` Jacobians cancels exactly
against the leading factor, and `P_s` reduces to a particularly clean form:

```math
P_r = V_r t_r\, K I_{dc} \cos\phi_r = K I_{dc}\,(V_r t_r \cos\alpha_r - \beta_r),
\qquad
P_i = K I_{dc}\,(-V_i t_i \cos\alpha_i - \beta_i).
```

That is, `P_s` is **linear in the bilinear coordinate `V_s t_s`** with an
`α_s`-dependent coefficient, plus a constant. So its Hessian has only three
non-zero entries (one mixed `V·t` term and the pure `α·α`, `V·α`, `t·α`
terms):

**Rectifier (`P_r`):**

```math
\begin{aligned}
\frac{\partial^2 P_r}{\partial V_r^2} = \frac{\partial^2 P_r}{\partial t_r^2} &= 0, \\
\frac{\partial^2 P_r}{\partial V_r\,\partial t_r} &= K I_{dc} \cos\alpha_r, \\
\frac{\partial^2 P_r}{\partial V_r\,\partial \alpha_r} &= -K I_{dc}\, t_r \sin\alpha_r, \\
\frac{\partial^2 P_r}{\partial t_r\,\partial \alpha_r} &= -K I_{dc}\, V_r \sin\alpha_r, \\
\frac{\partial^2 P_r}{\partial \alpha_r^2} &= -K I_{dc}\, V_r t_r \cos\alpha_r.
\end{aligned}
```

**Inverter (`P_i`):** same shape, opposite sign on every entry (since
`P_i = -K I_{dc} (V_i t_i \cos\alpha_i + \beta_i)`):

```math
\begin{aligned}
\frac{\partial^2 P_i}{\partial V_i^2} = \frac{\partial^2 P_i}{\partial t_i^2} &= 0, \\
\frac{\partial^2 P_i}{\partial V_i\,\partial t_i} &= -K I_{dc} \cos\alpha_i, \\
\frac{\partial^2 P_i}{\partial V_i\,\partial \alpha_i} &= +K I_{dc}\, t_i \sin\alpha_i, \\
\frac{\partial^2 P_i}{\partial t_i\,\partial \alpha_i} &= +K I_{dc}\, V_i \sin\alpha_i, \\
\frac{\partial^2 P_i}{\partial \alpha_i^2} &= +K I_{dc}\, V_i t_i \cos\alpha_i.
\end{aligned}
```

## Reactive power: derivation and Hessian

`Q_s = V_s t_s\, K I_{dc} \sin\phi_s`. With `C := \cos\phi_s = u_s`,
`S := \sin\phi_s = \sqrt{1-u_s^2}`, and using `\partial S/\partial x =
-(C/S)\,\partial u_s/\partial x`, write `Q_s = K I_{dc}\, A\, S` where
`A := V_s t_s` (so `\partial A/\partial V_s = t_s`, `\partial A/\partial t_s
= V_s`, `\partial A/\partial \alpha_s = 0`, `\partial^2 A/\partial V_s\partial
t_s = 1`, all other second partials of `A` are zero).

Differentiating once, then again, gives the product/chain expansion

```math
\frac{\partial^2 Q_s}{\partial x\,\partial y}
= K I_{dc}\Bigl[\,\frac{\partial^2 A}{\partial x \partial y}\, S
  + \frac{\partial A}{\partial x}\, \frac{\partial S}{\partial y}
  + \frac{\partial A}{\partial y}\, \frac{\partial S}{\partial x}
  + A\, \frac{\partial^2 S}{\partial x \partial y} \Bigr],
```

with

```math
\frac{\partial^2 S}{\partial x\,\partial y}
= -\frac{1}{S^3}\,\frac{\partial u_s}{\partial x}\,\frac{\partial u_s}{\partial y}
  - \frac{C}{S}\,\frac{\partial^2 u_s}{\partial x \partial y}.
```

(The `1/S^3` form comes from `u^2 + S^2 = 1`, which collapses the
`(1/S + u^2/S^3)` algebra to `1/S^3`.) Carrying out the substitution and
collecting yields the closed-form entries below. Let `\sigma_s = +1` for
the rectifier, `\sigma_s = -1` for the inverter — this encodes
`\partial u_s/\partial \alpha_s = -\sigma_s \sin\alpha_s` and
`\partial^2 u_s/\partial \alpha_s^2 = -\sigma_s \cos\alpha_s`. (The `V`/`t`
derivatives of `u_s` are sign-invariant in `\sigma_s`, so all entries
without an `\alpha`-derivative come out side-symmetric.)

**Pure `V_s`, `t_s` block (no `\alpha`).** Two of the three entries
collapse to single-term expressions because the `(\partial A/\partial x) \cdot
(\partial S/\partial x)` term cancels the leading piece of `A \cdot
(\partial^2 S/\partial x^2)`:

```math
\begin{aligned}
\frac{\partial^2 Q_s}{\partial V_s^2}
&= -\frac{K I_{dc}\, \beta_s^2}{V_s^3 t_s\, S^3}, \\[4pt]
\frac{\partial^2 Q_s}{\partial t_s^2}
&= -\frac{K I_{dc}\, \beta_s^2}{V_s t_s^3\, S^3}, \\[4pt]
\frac{\partial^2 Q_s}{\partial V_s\,\partial t_s}
&= K I_{dc}\, S
  - \frac{K I_{dc}\, \cos\phi_s\, \beta_s}{V_s\, t_s\, S}
  - \frac{K I_{dc}\, \beta_s^2}{V_s^2 t_s^2\, S^3}.
\end{aligned}
```

(The `\partial^2 Q/\partial V^2` and `\partial^2 Q/\partial t^2`
single-term forms come from `2(\partial A/\partial V)(\partial S/\partial
V) + A \cdot \partial^2 S/\partial V^2 = -2u\beta/(V^2 S) +
(-\beta^2/(V^3 t S^3) + 2u\beta/(V^2 S)) = -\beta^2/(V^3 t S^3)`. The
collapse is unconditional in `\sigma_s`.)

**Mixed with `\alpha_s`:**

```math
\begin{aligned}
\frac{\partial^2 Q_s}{\partial V_s\,\partial \alpha_s}
&= \sigma_s\, K I_{dc}\, \sin\alpha_s \left[
   \frac{t_s\, \cos\phi_s}{S}
   + \frac{\beta_s}{V_s\, S^3}
\right], \\[4pt]
\frac{\partial^2 Q_s}{\partial t_s\,\partial \alpha_s}
&= \sigma_s\, K I_{dc}\, \sin\alpha_s \left[
   \frac{V_s\, \cos\phi_s}{S}
   + \frac{\beta_s}{t_s\, S^3}
\right].
\end{aligned}
```

**Pure `\alpha_s \alpha_s`:**

```math
\frac{\partial^2 Q_s}{\partial \alpha_s^2}
= -\frac{K I_{dc}\, V_s\, t_s\, \sin^2\alpha_s}{S^3}
  + \sigma_s\,\frac{K I_{dc}\, V_s\, t_s\, \cos\phi_s\, \cos\alpha_s}{S}.
```

(The first term, `-V t K I \sin^2\alpha / S^3`, is sign-invariant in
`\sigma`: it picks up `(\partial u/\partial \alpha)^2 = \sin^2\alpha`
either way. The second term carries `\partial^2 u/\partial \alpha^2 =
-\sigma_s \cos\alpha_s`, which through the outer `-\cos\phi_s/S` factor
becomes `+\sigma_s \cos\phi_s \cos\alpha_s / S`.)

### Sanity check against the Jacobian

The existing Jacobian entry from `_calculate_dQ_dV_lcc` is

```math
\frac{\partial Q_s}{\partial V_s}
= t_s K I_{dc}\, S - \frac{K I_{dc}\, \beta_s\, \cos\phi_s}{V_s\, S}.
```

Differentiating once more w.r.t. `V_s` (treating `u_s = \cos\phi_s` and
`S = \sin\phi_s` as functions of `V_s`) reproduces the
`\partial^2 Q_s/\partial V_s^2 = -K I_{dc} \beta_s^2 / (V_s^3 t_s S^3)`
formula above. Specifically:

```math
\begin{aligned}
\frac{\partial}{\partial V_s}\Bigl(\frac{\beta_s \cos\phi_s}{V_s S}\Bigr)
&= \frac{\beta_s}{V_s} \cdot \frac{\partial(u_s/S)}{\partial V_s}
  - \frac{\beta_s u_s}{V_s^2 S} \\
&= \frac{\beta_s}{V_s} \cdot \frac{\beta_s}{V_s^2 t_s S^3}
  - \frac{\beta_s u_s}{V_s^2 S}
\end{aligned}
```

(using `\partial(u/S)/\partial V = \beta/(V^2 t S^3)` from `u^2 + S^2 =1`),
and `\partial(t_s S)/\partial V_s = -t_s u_s \beta/(V_s^2 t_s S) =-u_s \beta/(V_s^2 S)`.
Combining and simplifying gives the
single-term result. The recommended numerical check is finite differences;
this hand check is shown to demonstrate the cancellation that produces
the clean `-K I_{dc} \beta^2/(V^3 t S^3)` form.

## Hessian contributions per residual row

Per LCC, the residual rows that depend on LCC state and contribute to
`∑_k F_k \nabla^2 F_k` are:

| Row | What it contains | `\nabla^2 F_k` (LCC block) |
|-----|------------------|-----------------------------|
| `F_{P, f_b}` (bus-`f_b` active power balance) | adds `+P_r(V_r, t_r, \alpha_r)` to the network sum | `\nabla^2 P_r` |
| `F_{Q, f_b}` (bus-`f_b` reactive balance) | adds `+Q_r(V_r, t_r, \alpha_r)` | `\nabla^2 Q_r` |
| `F_{P, t_b}` (bus-`t_b` active power balance) | adds `+P_i(V_i, t_i, \alpha_i)` | `\nabla^2 P_i` |
| `F_{Q, t_b}` (bus-`t_b` reactive balance) | adds `+Q_i(V_i, t_i, \alpha_i)` | `\nabla^2 Q_i` |
| `F_{t_r}` (setpoint constraint) | `\pm P_{r\text{ or }i} - P_{\text{set}}` | `\pm \nabla^2 P_{r\text{ or }i}` (sign per `setpoint_at_rectifier`) |
| `F_{t_i}` (DC line balance) | `P_r + P_i - R_{dc} I_{dc}^2` | `\nabla^2 P_r + \nabla^2 P_i` |
| `F_{\alpha_r}`, `F_{\alpha_i}` | `\alpha_s - \alpha_{s,\min}` (linear) | `0` |

So the additive contribution to the Hessian from LCC `\ell` is

```math
\begin{aligned}
H_\ell &= F_{P, f_b}\, \nabla^2 P_r \;+\; F_{Q, f_b}\, \nabla^2 Q_r \\
       &\quad + F_{P, t_b}\, \nabla^2 P_i \;+\; F_{Q, t_b}\, \nabla^2 Q_i \\
       &\quad + F_{t_r}\, \nabla^2 P_{r\text{ or }i}^{\;(\pm)} \;+\; F_{t_i}\,(\nabla^2 P_r + \nabla^2 P_i).
\end{aligned}
```

Each `\nabla^2 P_s` and `\nabla^2 Q_s` has support only on its own
side's `(V_s, t_s, \alpha_s)` columns/rows — the rectifier block and the
inverter block of a single LCC do not share any second-derivative entries
(they share **no** state variables, only constants like `I_{dc}` and
`R_{dc}`, which are not differentiated). The two sides are therefore
combined as a block-diagonal pair within each LCC's 6×6 LCC-state block.

## Sparsity pattern of the LCC Hessian block

For LCC `\ell` with rectifier-side bus `f_b` and inverter-side bus `t_b`,
the state coordinates touched are

```math
(V_{f_b}, \theta_{f_b}, V_{t_b}, \theta_{t_b}, t_r, t_i, \alpha_r, \alpha_i).
```

The Hessian contributions above touch only `V_{f_b}, t_r, \alpha_r` and
`V_{t_b}, t_i, \alpha_i` (the `\theta` coordinates do not appear in any
LCC residual). The added sparse structure is therefore two 3×3 dense
blocks (one per side), embedded into the global Hessian at the
appropriate row/column indices:

- **Rectifier block:** rows/cols `\{V_{f_b}, t_r, \alpha_r\}` →
  `\partial^2 (\text{linear comb of } P_r, Q_r)`.
- **Inverter block:** rows/cols `\{V_{t_b}, t_i, \alpha_i\}` →
  `\partial^2 (\text{linear comb of } P_i, Q_i)`.

The new structural entries needed for the LCC Hessian are
`(V_{f_b}, t_r)`, `(V_{f_b}, \alpha_r)`, `(t_r, t_r)`, `(t_r, \alpha_r)`,
`(\alpha_r, \alpha_r)` (and symmetric), plus the analogous inverter
block on `(V_{t_b}, t_i, \alpha_i)`. None of these come from
*network-only* rows of `J` (which have no `t` or `\alpha` columns at all).
But the Jacobian has two other categories of rows that do:

1. **LCC bus rows** (`P_{f_b}`, `Q_{f_b}`, `P_{t_b}`, `Q_{t_b}`): these
   carry the LCC self-admittance contributions and thus have nonzero
   entries in all of `V`, `t`, and `\alpha` for the relevant side.
2. **LCC tail rows** (`F_{t_r}`, `F_{t_i}`, `F_{\alpha_r}`,
   `F_{\alpha_i}`): the `F_t` rows depend on `V` (via the chain rule
   through `P_{lcc,from/to}`) and on `t`, `\alpha`; the `F_\alpha` rows
   contain only the unit entry `\partial F_{\alpha_s}/\partial \alpha_s
   = 1`.

So when `J^\top J` is formed, the columns `V_{f_b}`, `t_r`, `\alpha_r`
all share the LCC bus rows and the `F_{t_r}` / `F_{t_i}` tail rows as
common nonzero positions, which gives `J^\top J` structural entries at
every pairwise combination of those three columns — exactly the
slots the LCC Hessian needs. Same on the inverter side. The
**sparsity pattern of `J^\top J` therefore already covers the LCC
Hessian block** without needing additional structural fill, provided
`A_plus_eq_BT_B!`'s structural-zero preservation continues to hold (it
asserts `colptr` and `rowval` equality on each call). The numeric
updates from `_update_hessian_lcc_contributions!` go into existing
slots.

## Clamp regime

When `\sin\phi_s < `[LCC_sinϕ_TOLERANCE](../../../src/definitions.jl), the
residual treats `\phi_s` as locally pinned and the Jacobian helpers drop
the chain term (see [lcc_utils.jl](../../../src/lcc_utils.jl)). The
corresponding Hessian behavior:

- `P_s` becomes linear in `(V_s, t_s)` with no `α`-dependence (since
  `\cos\phi_s = \pm 1` constant): the *only* non-zero second partial is
  `\partial^2 P_s / \partial V_s \partial t_s = \pm K I_{dc}`. All
  `α`-mixed and `α^2` partials vanish.
- `Q_s = 0` at the clamp boundary (since `\sin\phi_s = 0`) and the
  derivative is pinned to 0 in the residual: drop **all** second
  partials of `Q_s` on the clamp branch.

Both clamp branches are degenerate one-sided regimes — the analytic
formulas above have `1/S` and `1/S^3` factors that blow up. Matching the
Jacobian's clamp guard, the implementation should test `\sin\phi_s` and
return zero (or the simple `\pm K I_{dc}` mixed-`Vt` entry for `P_s`)
when below the tolerance.

## Side comparison

A compact side-by-side reference for the most error-prone entries:

| Entry | Rectifier (`s = r`) | Inverter (`s = i`) |
|-------|---------------------|--------------------|
| `\partial u_s/\partial \alpha_s` | `-\sin\alpha_r` | `+\sin\alpha_i` |
| `\partial^2 u_s/\partial \alpha_s^2` | `-\cos\alpha_r` | `+\cos\alpha_i` |
| `\partial^2 P_s/\partial V_s \partial t_s` | `+K I_{dc} \cos\alpha_r` | `-K I_{dc} \cos\alpha_i` |
| `\partial^2 P_s/\partial V_s \partial \alpha_s` | `-K I_{dc} t_r \sin\alpha_r` | `+K I_{dc} t_i \sin\alpha_i` |
| `\partial^2 P_s/\partial t_s \partial \alpha_s` | `-K I_{dc} V_r \sin\alpha_r` | `+K I_{dc} V_i \sin\alpha_i` |
| `\partial^2 P_s/\partial \alpha_s^2` | `-K I_{dc} V_r t_r \cos\alpha_r` | `+K I_{dc} V_i t_i \cos\alpha_i` |
| `\partial^2 Q_s/\partial V_s^2`, `\partial t_s^2`, `\partial V_s \partial t_s` | same formula both sides | same formula both sides |
| `\partial^2 Q_s/\partial V_s \partial \alpha_s` | sign of `+\sin\alpha_r` (overall `\sigma_r = +1`) | sign of `-\sin\alpha_i` (overall `\sigma_i = -1`) |
| `\partial^2 Q_s/\partial \alpha_s^2` | use `\sigma_r = +1` | use `\sigma_i = -1` |

The `V_s, t_s`-only second partials of `Q_s` are identical on both sides
because they depend only on `(V_s, t_s, \cos\phi_s, \sin\phi_s, \beta_s)`
— quantities whose **dependence** on side enters only through the value
of `\cos\phi_s = u_s` (whose sign already lives in `\cos\phi_s` itself
and does not need an extra `\sigma_s`).
