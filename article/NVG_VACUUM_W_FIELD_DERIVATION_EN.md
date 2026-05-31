# Analytical Derivation and Dynamics of the Real Vacuum Condensate Amplitude $\mathcal{W}(x)$ from QFT First Principles

Within the theoretical framework of Null-Vector Gravity (NVG) and the Vacuum Mass Fraction (VMF), the vacuum condensate is described by a complex scalar field $\Phi(x)$. In polar representation, it is parameterized as $\Phi(x) = \frac{1}{\sqrt{2}} \mathcal{W}(x) e^{i\theta(x)}$, where the real amplitude $\mathcal{W}(x)$ determines the local energy density of the condensate, and the Goldstone phase $\theta(x)$ defines the emergent timelike coordinate.

This document provides a rigorous mathematical derivation of the equations of motion for the real amplitude field $\mathcal{W}(x)$, its gauge-invariant coupling to a vector field $A_\mu$, and the conditions for the local phase transition (melting) of the vacuum condensate ($\mathcal{W} \to 0$).

---

## 1. Action and Gauge Invariance of the Scalar Sector

We start with a complex scalar field $\Phi(x)$ minimally coupled to an external Abelian gauge field $A_\mu$ (associated with $U(1)$ symmetry). Under a metric signature of $(+, -, -, -)$, the action of the scalar sector is given by:

$$ S = \int d^4x \sqrt{-g} \left[ (D_\mu \Phi)^\dagger (D^\mu \Phi) - V(\Phi) \right] $$

where:
* $D_\mu = \partial_\mu - i q A_\mu$ is the covariant derivative,
* $q$ is the effective coupling constant (charge) of the scalar condensate to the vector field,
* $V(\Phi)$ is the Ginzburg-Landau (or Higgs-Goldstone) self-interaction potential that triggers spontaneous symmetry breaking when $\mu^2 > 0$:

$$ V(\Phi) = -\mu^2 \Phi^\dagger \Phi + \lambda (\Phi^\dagger \Phi)^2 $$

Here, $\lambda > 0$ is the dimensionless self-coupling constant, and $\mu^2$ is the mass parameter of the field.

---

## 2. Polar (Amplitude-Phase) Decomposition

To isolate the physical degrees of freedom of the vacuum condensate, we perform a polar decomposition of the complex field $\Phi(x)$:

$$ \Phi(x) = \frac{1}{\sqrt{2}} \mathcal{W}(x) e^{i \theta(x)} $$

where:
* $\mathcal{W}(x) \ge 0$ is the real radial mode (amplitude of the vacuum condensate),
* $\theta(x)$ is the real angular mode (Goldstone phase).

### 2.1. Covariant Derivative Expansion

Applying the covariant derivative to the polar form of $\Phi(x)$ yields:

$$ D_\mu \Phi = (\partial_\mu - i q A_\mu) \left[ \frac{1}{\sqrt{2}} \mathcal{W} e^{i\theta} \right] = \frac{1}{\sqrt{2}} \left[ \partial_\mu \mathcal{W} \cdot e^{i\theta} + i \mathcal{W} \partial_\mu \theta \cdot e^{i\theta} - i q A_\mu \mathcal{W} e^{i\theta} \right] $$

Factoring out the phase $e^{i\theta}$:

$$ D_\mu \Phi = \frac{1}{\sqrt{2}} e^{i\theta} \left[ \partial_\mu \mathcal{W} + i \mathcal{W} (\partial_\mu \theta - q A_\mu) \right] $$

The complex conjugate is:

$$ (D_\mu \Phi)^\dagger = \frac{1}{\sqrt{2}} e^{-i\theta} \left[ \partial_\mu \mathcal{W} - i \mathcal{W} (\partial_\mu \theta - q A_\mu) \right] $$

### 2.2. Lagrangian Density of the Scalar Sector

We substitute these terms back into the Lagrangian density $\mathcal{L} = (D_\mu \Phi)^\dagger (D^\mu \Phi) - V(\Phi)$:

$$ (D_\mu \Phi)^\dagger (D^\mu \Phi) = \frac{1}{2} \left[ \partial_\mu \mathcal{W} + i \mathcal{W} (\partial_\mu \theta - q A_\mu) \right] \left[ \partial^\mu \mathcal{W} - i \mathcal{W} (\partial^\mu \theta - q A^\mu) \right] $$

Since the cross terms cancel out ($i \dots - i \dots = 0$), the kinetic term simplifies to:

$$ (D_\mu \Phi)^\dagger (D^\mu \Phi) = \frac{1}{2} \partial_\mu \mathcal{W} \partial^\mu \mathcal{W} + \frac{1}{2} \mathcal{W}^2 (\partial_\mu \theta - q A_\mu)(\partial^\mu \theta - q A^\mu) $$

Using the identity $\Phi^\dagger \Phi = \frac{1}{2} \mathcal{W}^2$, the potential term becomes:

$$ V(\mathcal{W}) = -\frac{1}{2} \mu^2 \mathcal{W}^2 + \frac{1}{4} \lambda \mathcal{W}^4 $$

The full gauge-invariant Lagrangian density in terms of the physical fields $\mathcal{W}(x)$ and $\theta(x)$ is:

$$ \mathcal{L} = \frac{1}{2} \partial_\mu \mathcal{W} \partial^\mu \mathcal{W} + \frac{1}{2} \mathcal{W}^2 (\partial_\mu \theta - q A_\mu)(\partial^\mu \theta - q A^\mu) + \frac{1}{2} \mu^2 \mathcal{W}^2 - \frac{1}{4} \lambda \mathcal{W}^4 $$

---

## 3. Derivation of the Equations of Motion

Applying the Euler-Lagrange equation to the real field $\mathcal{W}(x)$:

$$ \partial_\mu \left( \frac{\partial \mathcal{L}}{\partial (\partial_\mu \mathcal{W})} \right) - \frac{\partial \mathcal{L}}{\partial \mathcal{W}} = 0 $$

### 3.1. Evaluating the Variational Derivatives

1. Derivative with respect to the field gradient:
   $$ \frac{\partial \mathcal{L}}{\partial (\partial_\mu \mathcal{W})} = \partial^\mu \mathcal{W} \implies \partial_\mu \left( \frac{\partial \mathcal{L}}{\partial (\partial_\mu \mathcal{W})} \right) = \partial_\mu \partial^\mu \mathcal{W} = \square \mathcal{W} $$
   where $\square \equiv g^{\mu\nu} \nabla_\mu \nabla_\nu$ is the Laplace-Beltrami operator (d'Alembertian) on the curved spacetime background.

2. Derivative with respect to the field amplitude:
   $$ \frac{\partial \mathcal{L}}{\partial \mathcal{W}} = \mathcal{W} (\partial_\mu \theta - q A_\mu)(\partial^\mu \theta - q A^\mu) + \mu^2 \mathcal{W} - \lambda \mathcal{W}^3 $$

### 3.2. Amplitude Field Equation of Motion

Combining the derivatives gives the non-linear equation of motion for the real vacuum amplitude $\mathcal{W}(x)$:

$$ \square \mathcal{W} - \mathcal{W} (\partial_\mu \theta - q A_\mu)(\partial^\mu \theta - q A^\mu) - \mu^2 \mathcal{W} + \lambda \mathcal{W}^3 = 0 $$

Rearranging the terms linear in $\mathcal{W}$:

$$ \square \mathcal{W} + \left[ \lambda \mathcal{W}^2 - \left( \mu^2 + (\partial_\mu \theta - q A_\mu)(\partial^\mu \theta - q A^\mu) \right) \right] \mathcal{W} = 0 $$

---

## 4. Effective Potential and Phase Transition (Vacuum Melting)

Let us define the gauge-invariant 4-velocity/gradient vector $g_\mu \equiv \partial_\mu \theta - q A_\mu$. The equation of motion becomes:

$$ \square \mathcal{W} + \left[ \lambda \mathcal{W}^2 - (\mu^2 + g_\mu g^\mu) \right] \mathcal{W} = 0 $$

In a static or space-like dominated regime, where the spatial components of the phase gradient and gauge fields are dominant over time variations ($g_\mu g^\mu = g_0^2 - \vec{g}^2 \approx - \vec{g}^2 < 0$), we define the effective potential $V_{\text{eff}}(\mathcal{W})$ for the static field:

$$ V_{\text{eff}}(\mathcal{W}) = \frac{1}{2} (|\vec{g}|^2 - \mu^2) \mathcal{W}^2 + \frac{1}{4} \lambda \mathcal{W}^4 $$

### 4.1. Spontaneous Symmetry Breaking and the VEV

In the weak-field limit ($|\vec{g}|^2 < \mu^2$), the quadratic coefficient of the potential is negative, giving rise to the characteristic "Mexican hat" shape. The vacuum expectation value (VEV) at the stable minimum is:

$$ \mathcal{W}_0 = \sqrt{\frac{\mu^2 - |\vec{g}|^2}{\lambda}} $$

Here, the condensate is stable with a non-zero amplitude, and phase fluctuations correspond to massless Goldstone modes.

### 4.2. Critical Threshold and Vacuum Melting ($\mathcal{W} \to 0$)

When the energy density of the external gauge field increases such that:

$$ |\vec{g}|^2 \ge \mu^2 $$

the effective mass parameter changes sign. The potential $V_{\text{eff}}(\mathcal{W})$ undergoes a second-order phase transition, transitioning from a degenerate multi-minimum state to a single-well potential with its unique stable minimum at:

$$ \mathcal{W}_0 = 0 $$

This represents the **complete local melting of the vacuum condensate** under a high-intensity external gauge field. The local $U(1)$ symmetry is restored, the condensate density drops to zero, and the local dielectric properties of the vacuum space transition to their bare, unperturbed state.
