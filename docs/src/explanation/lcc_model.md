# Line-Commutated Converter (LCC) Implementations

## Implementations in PSSE

In real-world operation, on-load tap changers (OLTCs) adjust transformer taps so that the converter reaches its minimum firing or extinction angle limits (α_min or γ_min). The objective is to reduce the firing/extinction angle to minimize reactive power (Q) demand.

As OLTCs operate typically over several seconds, the thyristor angles act as fast control. First, the thyristor angles respond to fast power flow changes. Afterwards, the OLTCs are adjusted to keep maintaining the power set point while also minimizing the thyristor angles.

In PSSE, the control using OLTCs is ignored in power flow calculations. Instead, PSSE adjusts the thyristor angles to represent the fast control. 
PSSE solves for the rectifier firing angle (α) such that `Idc = Iset`, and for the inverter extinction angle (γ) such that `Udc = Uset`.
The solver adjusts α and γ automatically to satisfy these setpoints and only switches control mode if a limit is violated.

OLTC actions are not represented in the load-flow process because they operate on a slower time scale (seconds). The rationale is to provide a starting point for dynamic simulations. The tap positions can be set manually.

For safe operation and to avoid commutation failure, the rectifier should normally operate close to its minimum firing angle (α_min) with a small safety margin. Overly broad firing angle settings should be avoided. For example, setting max firing angle to 90° is too permissive. A narrower range encourages PSSE to adjust taps appropriately.

In PSSE, the variable `VSched` (scheduled DC voltage) represents the DC voltage reference, not the AC-side voltage. If `VSched` is set too low, the inverter will use unnecessarily large extinction angles (γ) to transfer power.

Power flow calculation in PSSE often fails to converge with LCC. In this case, it can help to first run the Gauss-Seidel algorithm, then Newton will also converge in PSSE.

## Implementation in Sienna

We implement the control logic of LCC based on the principle of maintaining the thyristor angles at their respective minimum limits, and using transformer tap adjustments for active power control. This control principle is better suited for representing realistic steady-state grid conditions.

The AC power flow calculation in Sienna is modified to directly solve for tap steps and thyristor angles of the LCC system. To this end, the state vector, the Jacobian matrix, the residuals are modified. The state vector is extended by 4 additional variables for each LCC system, namely 2 for tap positions and 2 for thyristor angles of the rectifier and inverter sides of the LCC system. The Jacobian matrix is extended by additional terms that represent the relevant partial derivatives. The residuals are extended by 4 terms per LCC system to match the additional state variables. The first two account for the active power set point and active power balance in the LCC system using the tap steps. The other two residual terms control for keeping the thyristor angles at their respective minimum limits. This approach follows the method in Panosyan, A. (2010). Modeling of Advanced Power Transmission System Controllers (PhD dissertation).

The complex apparent power for a rectifier or inverter is calculated as $S = V t \frac{\sqrt{6}}{\pi} I_{dc} e^{j \phi}$, where $V$ is the magnitude of AC voltage at the terminal, $I_{dc}$ is the DC current in the LCC system (positive for flow direction rectifier to inverter, and vice versa), and $\phi$ is the angle between AC voltage and current.

To allow for the Jacobian implementation, a simplified calculation of the angle between the AC current and voltage at the LCC terminals was used. The equation below represents the calculation used for the angle:
```math
\phi = \arccos\left(\cos(\alpha) \text{sign}(I_{dc}) - \frac{x I_{dc}}{\sqrt{2} t V}\right)
```

In the equation above, the variable ``\alpha`` represents the thyristor angle. The active and reactive powers for a converter station are 

```math
\begin{aligned}
P &= V t \frac{\sqrt{6}}{\pi} I_{dc} \cos(\phi) = V t \frac{\sqrt{6}}{\pi} I_{dc} \cos(\alpha) - \frac{\sqrt{6}}{\pi} x I_{dc}^2 \\
Q &= V t \frac{\sqrt{6}}{\pi} I_{dc} \sin(\phi) = V t \frac{\sqrt{6}}{\pi} I_{dc} \sin\left(\arccos\left(\cos(\alpha) \text{sign}(I_{dc}) - \frac{x I_{dc}}{\sqrt{2} t V}\right)\right)
\end{aligned}
```

The relevant non-zero entries in the Jacobian matrix for the rectifier (r) and inverter (i) sides are:

```math
\begin{aligned}
\frac{\partial P_r}{\partial V_r} &= t_r \frac{\sqrt{6}}{\pi} I_{dc} \cos(\alpha_r) \\
\frac{\partial Q_r}{\partial V_r} &= V_r t_r \frac{\sqrt{6}}{\pi} I_{dc} \left(\sin(\phi_r) - \cos(\phi_r) \frac{x_r I_{dc}}{\sqrt{2} V_r t_r \sin^2(\phi_r)}\right) \\
\frac{\partial Q_r}{\partial t_r} &= V_r t_r \frac{\sqrt{6}}{\pi} I_{dc} \left(\frac{\sin(\phi_r)}{t_r} - \cos(\phi_r) \frac{x_r I_{dc}}{\sqrt{2} V_r t_r^2 \sin^2(\phi_r)}\right) \\
\frac{\partial Q_r}{\partial \alpha_r} &= V_r t_r \frac{\sqrt{6}}{\pi} I_{dc} \frac{\cos(\phi_r) \sin(\alpha_r)}{\sin(\phi_r)} \\
\frac{\partial P_r}{\partial t_r} &= V_r \frac{\sqrt{6}}{\pi} I_{dc} \cos(\alpha_r) \\
\frac{\partial P_r}{\partial \alpha_r} &= -V_r \frac{\sqrt{6}}{\pi} I_{dc} t_r \sin(\alpha_r) \\
\frac{\partial F_{t_i}}{\partial V_i} &= t_i \frac{\sqrt{6}}{\pi} (-I_{dc}) \cos(\alpha_i) \\
\frac{\partial F_{t_r}}{\partial t_r} &= V_r \frac{\sqrt{6}}{\pi} I_{dc} \cos(\alpha_r) \\
\frac{\partial F_{t_r}}{\partial \alpha_r} &= -V_r \frac{\sqrt{6}}{\pi} I_{dc} t_r \sin(\alpha_r) \\
\frac{\partial F_{t_i}}{\partial t_i} &= V_i \frac{\sqrt{6}}{\pi} (-I_{dc}) \cos(\alpha_i) \\
\frac{\partial F_{t_r}}{\partial V_r} &= t_r \frac{\sqrt{6}}{\pi} I_{dc} \cos(\alpha_r) \\
\frac{\partial F_{t_i}}{\partial V_r} &= t_r \frac{\sqrt{6}}{\pi} I_{dc} \cos(\alpha_r) \\
\frac{\partial F_{t_i}}{\partial t_r} &= V_r \frac{\sqrt{6}}{\pi} I_{dc} \cos(\alpha_r) \\
\frac{\partial F_{t_i}}{\partial \alpha_i} &= -V_i \frac{\sqrt{6}}{\pi} (-I_{dc}) t_i \sin(\alpha_i) \\
\frac{\partial F_{t_i}}{\partial \alpha_r} &= -V_r \frac{\sqrt{6}}{\pi} I_{dc} t_r \sin(\alpha_r) \\
\frac{\partial F_{\alpha_r}}{\partial \alpha_r} &= 1 \\
\frac{\partial F_{\alpha_i}}{\partial \alpha_i} &= 1
\end{aligned}
```

The residuals are defined as follows:

```math
\begin{aligned}
F_{t_r} &= P_r - P_{\text{set}, r} \\
F_{t_i} &= P_r + P_i - I_{dc} R_{dc}^2 \\
F_{\alpha_r} &= \alpha_r - \alpha_{r, \min} \\
F_{\alpha_i} &= \alpha_i - \alpha_{i, \min}
\end{aligned}
```
