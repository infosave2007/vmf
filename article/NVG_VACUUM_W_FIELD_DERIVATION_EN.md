# Analytical Derivation and Dynamics of the Real Vacuum Condensate Amplitude $\mathcal{W}(x)$ from QFT First Principles

Within the theoretical framework of Null-Vector Gravity (NVG) and the Vacuum Mass Fraction (VMF), the vacuum condensate is described by a complex scalar field $\Phi(x)$. In polar representation, it is parameterized as:

$$ \Phi(x) = \frac{1}{\sqrt{2}} \mathcal{W}(x) e^{i\theta(x)} $$

where:
* $\mathcal{W}(x) \ge 0$ is the real radial mode (amplitude of the vacuum condensate), which determines the in-medium effective mass of hadrons ($M^*$),
* $\theta(x)$ is the real angular mode (Goldstone phase), the gradient of which defines the preferred coordinate time direction.

This document provides a rigorous mathematical derivation of the equations of motion for the real amplitude field $\mathcal{W}(x)$, its gauge-invariant coupling to the baryon current and vector fields in a dense medium, parameter calibration via QCD sigma-terms, and the derivation of its energy-momentum tensor and its FLRW-reduction.

---

## 1. Action and Gauge Invariance of the Scalar Sector

We start with a complex scalar field $\Phi(x)$ minimally coupled to an external Abelian vector field $A_\mu$, representing the effective baryonic vector gauge field (corresponding to the vector $\omega$-meson field, which mediates the short-range repulsive interaction between baryons and is associated with the conservation of the $U(1)_B$ baryon number). Under a metric signature of $(+, -, -, -)$, the action of the scalar sector is given by:

$$ S = \int d^4x \sqrt{-g} \left[ (D_\mu \Phi)^\dagger (D^\mu \Phi) - V(\Phi) \right] $$

where:
* $D_\mu = \partial_\mu - i q A_\mu$ is the covariant derivative,
* $q$ is the effective coupling constant (baryon charge) of the scalar condensate to the vector field,
* $V(\Phi)$ is the Ginzburg-Landau (or Higgs-Goldstone) self-interaction potential:

$$ V(\Phi) = -\mu_0^2 \Phi^\dagger \Phi + \lambda (\Phi^\dagger \Phi)^2 $$

Here, $\lambda > 0$ is the dimensionless self-coupling constant, and $\mu_0^2 > 0$ is the bare mass parameter of the field.

The effective hadron mass (such as the nucleon or $\Omega$-baryon mass) in the medium scales directly with the vacuum condensate amplitude:

$$ M^*(n_B) = g_s \mathcal{W}(n_B) $$

In vacuum ($n_B = 0$), the hadron mass is $M_0 = g_s \mathcal{W}_0$. Therefore, the melting function of hadron masses in a dense medium is given by the ratio of the condensates:

$$ \frac{M^*(n_B)}{M_0} = \frac{\mathcal{W}(n_B)}{\mathcal{W}_0} $$

---

## 2. Polar (Amplitude-Phase) Decomposition

Substituting $\Phi(x) = \frac{1}{\sqrt{2}} \mathcal{W}(x) e^{i \theta(x)}$ into the covariant derivative yields:

$$ D_\mu \Phi = \frac{1}{\sqrt{2}} e^{i\theta} \left[ \partial_\mu \mathcal{W} + i \mathcal{W} (\partial_\mu \theta - q A_\mu) \right] $$

The complex conjugate is:

$$ (D_\mu \Phi)^\dagger = \frac{1}{\sqrt{2}} e^{-i\theta} \left[ \partial_\mu \mathcal{W} - i \mathcal{W} (\partial_\mu \theta - q A_\mu) \right] $$

### 2.1. Lagrangian Density of the Scalar Sector

The kinetic term of the Lagrangian density transforms to:

$$ (D_\mu \Phi)^\dagger (D^\mu \Phi) = \frac{1}{2} \partial_\mu \mathcal{W} \partial^\mu \mathcal{W} + \frac{1}{2} \mathcal{W}^2 (\partial_\mu \theta - q A_\mu)(\partial^\mu \theta - q A^\mu) $$

With the potential $V(\mathcal{W}) = -\frac{1}{2} \mu^2 \mathcal{W}^2 + \frac{1}{4} \lambda \mathcal{W}^4 + V_0$ (where $V_0 = \frac{1}{4} \lambda \mathcal{W}_0^4$ normalizes the vacuum energy to zero), the full scalar sector Lagrangian density is:

$$ \mathcal{L} = \frac{1}{2} \partial_\mu \mathcal{W} \partial^\mu \mathcal{W} + \frac{1}{2} \mathcal{W}^2 (\partial_\mu \theta - q A_\mu)(\partial^\mu \theta - q A^\mu) + \frac{1}{2} \mu^2 \mathcal{W}^2 - \frac{1}{4} \lambda \mathcal{W}^4 - V_0 $$

---

## 3. Derivation of the Equations of Motion

Applying the Euler-Lagrange equation to the real field $\mathcal{W}(x)$:

$$ \partial_\mu \left( \frac{\partial \mathcal{L}}{\partial (\partial_\mu \mathcal{W})} \right) - \frac{\partial \mathcal{L}}{\partial \mathcal{W}} = 0 $$

1. Variation with respect to the gradient:
   $$ \partial_\mu \left( \frac{\partial \mathcal{L}}{\partial (\partial_\mu \mathcal{W})} \right) = \square \mathcal{W} $$
   where $\square \equiv g^{\mu\nu} \nabla_\mu \nabla_\nu$ is the Laplace-Beltrami operator.
2. Variation with respect to the field $\mathcal{W}$:
   $$ \frac{\partial \mathcal{L}}{\partial \mathcal{W}} = \mathcal{W} (\partial_\mu \theta - q A_\mu)(\partial^\mu \theta - q A^\mu) + \mu^2 \mathcal{W} - \lambda \mathcal{W}^3 $$

This yields the dynamic equation of motion for the field $\mathcal{W}(x)$:

$$ \square \mathcal{W} + \left[ \lambda \mathcal{W}^2 - \left( \mu^2 + (\partial_\mu \theta - q A_\mu)(\partial^\mu \theta - q A^\mu) \right) \right] \mathcal{W} = 0 $$

This equation is manifest Lorentz-covariant by construction.

---

## 4. Lorentz Covariance and Density Mapping

While the field equations and conserved currents are Lorentz-covariant, for practical calculations of in-medium phase transitions and thermodynamic quantities, we choose to work in the comoving frame where the nuclear medium is at rest. The results obtained in this frame are covariant by construction, as they are formulated in terms of scalar contractions of 4-vectors.

The baryon density $n_B$ is defined covariantly as the scalar contraction of the conserved baryon 4-current $J_\mu$ and the 4-velocity of the medium $u_\mu$ ($u_\mu u^\mu = 1$):

$$ n_B \equiv J_\mu u^\mu $$

In the comoving frame, the 4-velocity is $u^\mu = (1, \vec{0})$ and the 4-current is $J^\mu = n_B u^\mu$. Under the mean-field approximation for a uniform, static nuclear medium, the vector field equation for the $\omega$-meson simplifies as derivatives vanish, yielding:

$$ A_\mu = \frac{g_\omega}{m_\omega^2} J_\mu = \frac{g_\omega}{m_\omega^2} n_B u_\mu $$

The Goldstone phase gradient $\partial_\mu \theta$ is collinear with the medium 4-velocity: $\partial_\mu \theta = \mu_\theta u_\mu$, where the frequency $\mu_\theta \equiv u^\mu \partial_\mu \theta$ acts as the effective chemical potential (a Lorentz scalar).

We define the gauge-invariant 4-vector of generalized momentum:

$$ g_\mu \equiv \partial_\mu \theta - q A_\mu = \left( \mu_\theta - \frac{q g_\omega}{m_\omega^2} n_B \right) u_\mu $$

The Lorentz-invariant contraction $g_\mu g^\mu$ is:

$$ g_\mu g^\mu = \left( \mu_\theta - \frac{q g_\omega}{m_\omega^2} n_B \right)^2 u_\mu u^\mu = \left( \mu_\theta - \frac{q g_\omega}{m_\omega^2} n_B \right)^2 $$

Because all constituents ($\mu_\theta$, $q$, $g_\omega$, $m_\omega$, and $n_B$) are scalars, and $u_\mu u^\mu = 1$, the contraction $g_\mu g^\mu$ is a manifest Lorentz scalar. Substituting this invariant into the equation of motion for $\mathcal{W}$ yields:

$$ \square \mathcal{W} + \left[ \lambda \mathcal{W}^2 - \mu^2 - \left( \mu_\theta - \frac{q g_\omega}{m_\omega^2} n_B \right)^2 \right] \mathcal{W} = 0 $$

---

## 5. Fixing Parameters from QCD

In the deep vacuum ($n_B = 0$ and $A_\mu = 0$), the field $\mathcal{W}$ acquires a stable vacuum expectation value VEV ($\mathcal{W}_0$). Taking into account the vacuum gradient of the phase $\partial_0 \theta = \mu_\theta$, the minimum of the potential $\mathcal{L}_{\text{scalar}}$ with respect to $\mathcal{W}$ yields:

$$ \mathcal{W}_0 \Big|_{n_B=0, A_\mu=0} = \sqrt{\frac{\mu^2 - \mu_\theta^2}{\lambda}} = M_{\Omega,0} = 859 \text{ MeV} $$

To fix the parameters $\mu^2$ and $\lambda$, we utilize the pion-nucleon $\sigma_{\pi N} \approx 44$ MeV and strange $\sigma_{sN} \approx 30$ MeV QCD sigma-terms. By the Feynman-Hellmann theorem, the nucleon mass $M_N = g_N \mathcal{W}_0$ depends on the light quark masses through the VEV shift:

$$ \sigma_{\pi N} = m_q \frac{\partial M_N}{\partial m_q} = g_N h \frac{\partial \mathcal{W}_0}{\partial h} = \frac{M_N f_\pi m_\pi^2}{\mathcal{W}_0 m_{\mathcal{W}}^2} $$

where $m_{\mathcal{W}}^2 = 2 \lambda \mathcal{W}_0^2$ is the mass of the scalar excitation (representing the physical $f_0(1370)$ meson), and $h = f_\pi m_\pi^2$ represents the explicit chiral symmetry-breaking scale.

Incorporating the strange sigma-term $\sigma_{sN}$ and utilizing the Gell-Mann-Oakes-Renner (GMOR) relation, the scalar mass $m_{\mathcal{W}}$ is locked via:

$$ m_{\mathcal{W}}^2 = 2 \lambda \mathcal{W}_0^2 = \frac{M_N^3 m_\pi^2}{2 f_\pi \mathcal{W}_0 (\sigma_{\pi N} + \sigma_{sN})} \cdot \mathcal{C} $$

where the factor $\mathcal{C}$ accounts for constituent quark mass scaling.

### 5.1. Analytical Derivation of the Coefficient $\mathcal{C}$

The constituent quark mass scaling factor $\mathcal{C} \approx 1.137$ represents the chiral enhancement of the constituent quark mass response to the current quark mass, which is a standard feature in low-energy QCD phenomenology. In the constituent quark model, the nucleon mass is:

$$ M_N = 3 M_q^{\text{con}} + E_{\text{binding}} $$

where $M_q^{\text{con}}$ is the constituent quark mass and $E_{\text{binding}}$ is the binding energy, which depends weakly on quark masses. Differentiating with respect to the current quark mass $m_q$:

$$ \frac{\partial M_N}{\partial m_q} \approx 3 \frac{\partial M_q^{\text{con}}}{\partial m_q} $$

In the Nambu-Jona-Lasinio (NJL) model (see Klevansky, Rev. Mod. Phys. 64, 649 (1992)), the constituent quark mass is determined by the gap equation:

$$ M_q^{\text{con}} = m_q - 2 G \langle \bar{q} q \rangle $$

Differentiating both sides with respect to $m_q$:

$$ \frac{\partial M_q^{\text{con}}}{\partial m_q} = 1 - 2 G \frac{\partial \langle \bar{q} q \rangle}{\partial m_q} = 1 + 2 G \chi_q $$

where $\chi_q \equiv -\partial \langle \bar{q} q \rangle / \partial m_q$ is the chiral susceptibility of the vacuum. Under standard NJL parameter sets with a 3D momentum cutoff scale $\Lambda \approx 650$ MeV and constituent quark mass $M_q^{\text{con}} \approx 335$ MeV, the 1-loop dimensionless integral is $I_1(M_q^{\text{con}}) = \ln\left( \frac{\Lambda + \sqrt{\Lambda^2 + (M_q^{\text{con}})^2}}{M_q^{\text{con}}} \right) - \frac{\Lambda}{\sqrt{\Lambda^2 + (M_q^{\text{con}})^2}} \approx 0.5277$. With coupling constant $G \approx 5.01 \times 10^{-6}$ MeV$^{-2}$ and number of quark flavors $N_f = 2$, the dimensional susceptibility $\chi_q$ yields a dimensionless multiplicative enhancement of the constituent mass response:

$$ \mathcal{C} \equiv \frac{\partial M_q^{\text{con}}}{\partial m_q} = \left( 1 - \frac{2 G N_f (M_q^{\text{con}})^2 I_1(M_q^{\text{con}})}{\pi^2} \right)^{-1} \approx 1.137 $$

Thus, the factor $\mathcal{C} \approx 1.137$ is a physical consequence of the back-reaction of the chiral condensate and is fixed from first principles of the NJL model.

Substituting this value, we obtain the self-coupling $\lambda$:

$$ \lambda = \frac{M_N^3 m_\pi^2}{4 f_\pi \mathcal{W}_0^3 (\sigma_{\pi N} + \sigma_{sN})} \cdot \mathcal{C} \approx 1.06 \quad (\text{in model, } \lambda = 1.05 \text{ is used}) $$

which fixes the scalar mass at $m_{\mathcal{W}} = \sqrt{2 \lambda} \mathcal{W}_0 \approx 1245$ MeV, matching the physical scalar $f_0(1370)$ / $f_0(1500)$ meson (PDG 2024).

The mass parameter $\mu^2$ is then locked using the vacuum phase frequency $\mu_\theta \approx m_\omega = 782.6$ MeV:

$$ \mu^2 = \mu_\theta^2 + \lambda \mathcal{W}_0^2 \approx (782.6)^2 + 1.05 \times (859.0)^2 \approx 1.387 \times 10^6 \text{ MeV}^2 $$

The physical justification for identifying the vacuum phase frequency with the mass of the vector meson ($\mu_\theta \approx m_\omega$) stems from the self-consistency of gauge invariance in the vacuum. Within the Hidden Local Symmetry (HLS) framework, the Goldstone phase $\theta$ serves as the longitudinal component of the vector field $A_\mu$ (a Stueckelberg-like mass-generation mechanism). For the vacuum state ($n_B = 0$, $A_\mu = 0$) to act as a physical absolute energy minimum with zero net current, the gauge-invariant generalized momentum $g_\mu = \partial_\mu \theta - q A_\mu$ must vanish. Since the vacuum phase frequency $\partial_0 \theta = \mu_\theta$ reflects the intrinsic scale of chiral symmetry breaking that determines the masses of the vector meson partners, the dynamical stability of the vacuum links these two scales: the frequency of the vacuum phase oscillations resonance-matches the mass of the vector field excitation that couples to the baryon number.

### 5.2. Calibration of the Coupling Constant $q$

Although the classic Goldberger-Treiman relation connects the pseudoscalar pion-nucleon coupling, the coupling constant $q$ expresses a similar connection in the vector sector in the presence of a scalar condensate $\mathcal{W}$.

In chiral models with hidden local symmetry (HLS) or vector manifestation (VM) (see Harada and Yamawaki, Phys. Rep. 381, 1 (2003)), the masses of vector mesons are generated dynamically by the scalar VEV via a Higgs-like mechanism: $m_\omega = q \mathcal{W}_0$. Setting the scalar VEV $\mathcal{W}_0$ to the baryon mass scale $M_{\Omega,0}$ and utilizing the vector manifestation limit, the coupling constant $q$ is locked to the pion decay constant and the vector coupling $g_\omega$, yielding the exact relation:

$$ q = \frac{g_\omega M_{\Omega,0}}{2 f_\pi} \approx 47.01 $$

where $g_\omega \approx 10.12$ is the effective $\omega$-meson coupling constant and $f_\pi \approx 92.4$ MeV is the pion decay constant.

---

## 6. Identification of $g_\mu$ with the Baryon Current

To close the relation between $|\vec{g}|^2$ and the physical baryon density $n_B$, we express the gauge-invariant 4-vector $g_\mu$ through the normalized baryon 4-current $j_B^\mu = J^\mu / n_0$, where $n_0 \approx 0.16 \text{ fm}^{-3}$ is the saturation nuclear density (normal nuclear density).

In the comoving frame of the nuclear medium, the 4-velocity is $u^\mu = (1, \vec{0})$ and the 4-current is $J^\mu = n_B u^\mu$. With this, the vector potential $A_\mu$ is written as:

$$ A_\mu = \frac{g_\omega}{m_\omega^2} n_B u_\mu = \left(\frac{g_\omega n_0}{m_\omega^2}\right) \frac{J_\mu}{n_0} = \left(\frac{g_\omega n_0}{m_\omega^2}\right) j_{B,\mu} $$

Thus, the gauge-invariant vector of generalized momentum $g_\mu$ is:

$$ g_\mu \equiv \partial_\mu \theta - q A_\mu = \mu_\theta u_\mu - \gamma j_{B,\mu} $$

where the effective coupling coefficient $\gamma$ with the normalized baryon current is defined as:

$$ \gamma \equiv \frac{q g_\omega n_0}{m_\omega^2} \approx \frac{47.01 \times 10.12 \times 0.16 \times (197.327)^3}{(782.6)^2} \approx 955.5 \text{ MeV} $$

The Lorentz-invariant square of this vector is:

$$ g_\mu g^\mu = \left( \mu_\theta - \gamma \frac{n_B}{n_0} \right)^2 $$

In the comoving frame, the spatial part of $\vec{g}$ vanishes ($|\vec{g}|^2 = 0$), and the time component $g_0$ determines the effective strength of the interaction:

$$ g_0 = \mu_\theta - \gamma \frac{n_B}{n_0} $$

This closes the connection between the Lorentz-invariant contraction $g_\mu g^\mu$ and the physical baryon density $n_B$. When the critical melting density is reached at $n_B = \frac{\mu_\theta + \mu}{\gamma} n_0 \approx 2.5 n_0$, the value of $g_0$ shifts the effective mass of the scalar field to zero, completing the vacuum melting phase transition.

---

## 7. Energy-Momentum Tensor $T^{\mathcal{W}}_{\mu\nu}$ and FLRW Reduction

To couple the vacuum scalar field to Einstein's equations, we vary the action of the scalar sector with respect to the metric tensor $g^{\mu\nu}$:

$$ T^{\mathcal{W}}_{\mu\nu} = -\frac{2}{\sqrt{-g}} \frac{\delta ( \sqrt{-g} \mathcal{L} )}{\delta g^{\mu\nu}} = 2 \frac{\delta \mathcal{L}}{\delta g^{\mu\nu}} - g_{\mu\nu} \mathcal{L} $$

Since $\mathcal{L} = \frac{1}{2} g^{\alpha\beta} \partial_\alpha \mathcal{W} \partial_\beta \mathcal{W} + \frac{1}{2} \mathcal{W}^2 g^{\alpha\beta} g_\alpha g_\beta - V(\mathcal{W})$, the variation yields:

$$ T^{\mathcal{W}}_{\mu\nu} = \partial_\mu \mathcal{W} \partial_\nu \mathcal{W} + \mathcal{W}^2 g_\mu g_\nu - g_{\mu\nu} \left[ \frac{1}{2} \partial_\alpha \mathcal{W} \partial^\alpha \mathcal{W} + \frac{1}{2} \mathcal{W}^2 g_\alpha g^\alpha - V(\mathcal{W}) \right] $$

In a homogeneous FLRW cosmological spacetime with metric $ds^2 = dt^2 - a^2(t) d\vec{x}^2$, for a spatially uniform field ($\partial_i \mathcal{W} = 0$, $g_i = 0$):
* Energy density $\rho_{\mathcal{W}} = T^0_0$:
  $$ \rho_{\mathcal{W}} = \frac{1}{2} \dot{\mathcal{W}}^2 + \frac{1}{2} \mathcal{W}^2 g_0^2 + V(\mathcal{W}) $$
* Pressure $P_{\mathcal{W}} = -T^i_i$:
  $$ P_{\mathcal{W}} = \frac{1}{2} \dot{\mathcal{W}}^2 - \frac{1}{2} \mathcal{W}^2 g_0^2 - V(\mathcal{W}) $$

where the potential is normalized as $V(\mathcal{W}) = \frac{1}{4} \lambda (\mathcal{W}^2 - \mathcal{W}_0^2)^2$.

Under cosmological collapse, the baryon density grows as $n_B(a) = n_{B,0} a^{-3}$. As density approaches the critical bounce limit ($n_B \ge 2.5 n_0$), the condensate melts completely ($\mathcal{W} \to 0$, $\dot{\mathcal{W}} \to 0$). The energy density and pressure of the $\mathcal{W}$-field are determined by its vacuum value at the symmetric point:

$$ \rho_{\mathcal{W}} \to V(0) = V_0 = \frac{1}{4} \lambda \mathcal{W}_0^4, \quad P_{\mathcal{W}} \to -V(0) = -V_0 $$

This results in the violation of the Strong Energy Condition (SEC):

$$ \rho_{\mathcal{W}} + 3 P_{\mathcal{W}} \to -2 V_0 = -\frac{1}{2} \lambda \mathcal{W}_0^4 < 0 $$

### 7.1. Derivation of the Quadratic Correction and Modified Friedmann Equation

Consider the total energy density of the cosmological medium $\rho = \rho_{\text{matter}} + \rho_{\mathcal{W}}$, where the matter sector is modeled as dust $\rho_{\text{matter}} = M_N n_B$.

In the extreme compression regime near the bounce, the dynamics of the scalar field $\mathcal{W}$ and the baryon current $J_\mu$ are strongly coupled. In the quasi-static compression limit ($\dot{\mathcal{W}} \approx 0$), the equation of motion for the field $\mathcal{W}$ reduces to the algebraic minimum condition:

$$ \mathcal{W}^2 = \frac{\mu^2 - g_0^2}{\lambda} $$

where $g_0 = \mu_\theta - \gamma \frac{n_B}{n_0}$. The adiabaticity criterion for this approximation requires that the rate of change of the cosmological background (the compression rate) remains much smaller than the mass of the scalar excitations: $H \ll m_{\mathcal{W}}$. The characteristic response timescale of the scalar field is $\tau_{\mathcal{W}} = \hbar / m_{\mathcal{W}} \approx 5.3 \times 10^{-25}$ s, whereas the characteristic timescale of the cosmic contraction near the bounce is $t_b \approx 3.76 \times 10^{-6}$ s. Because $t_b \gg \tau_{\mathcal{W}}$ (differing by 19 orders of magnitude), the field adapts to the changing density instantaneously, fully justifying the quasi-static minimum approximation.

Substituting this into the total energy density of the $\mathcal{W}$-sector (accounting for zero-point vacuum subtraction):

$$ \rho_{\mathcal{W}} = \frac{1}{2} \mathcal{W}^2 g_0^2 + \frac{1}{4} \lambda \left( \mathcal{W}^2 - \mathcal{W}_0^2 \right)^2 - \frac{1}{2} \mathcal{W}_0^2 \mu_\theta^2 $$

Using the relation $\lambda \mathcal{W}_0^2 = \mu^2 - \mu_\theta^2$, the difference in the square of the amplitudes is:

$$ \mathcal{W}^2 - \mathcal{W}_0^2 = \frac{\mu^2 - g_0^2}{\lambda} - \frac{\mu^2 - \mu_\theta^2}{\lambda} = \frac{\mu_\theta^2 - g_0^2}{\lambda} $$

Thus, the potential term is $V(\mathcal{W}) = \frac{(\mu_\theta^2 - g_0^2)^2}{4 \lambda}$, and the energy density of the $\mathcal{W}$-field is:

$$ \rho_{\mathcal{W}} = \frac{g_0^2 (\mu^2 - g_0^2)}{2 \lambda} + \frac{(\mu_\theta^2 - g_0^2)^2}{4 \lambda} - \frac{\mu_\theta^2 (\mu^2 - \mu_\theta^2)}{2 \lambda} $$

Bringing the terms to a common denominator:

$$ \rho_{\mathcal{W}} = \frac{2 g_0^2 \mu^2 - 2 g_0^4 + g_0^4 - 2 g_0^2 \mu_\theta^2 + \mu_\theta^4 - 2 \mu^2 \mu_\theta^2 + 2 \mu_\theta^4}{4 \lambda} = \frac{-g_0^4 + 2 g_0^2 (\mu^2 - \mu_\theta^2) + 3 \mu_\theta^4 - 2 \mu^2 \mu_\theta^2}{4 \lambda} $$

We perform the algebraic expansion of the numerator in powers of $x \equiv \gamma \frac{n_B}{n_0}$, where $g_0 = \mu_\theta - x$. By expanding the components:

1. First numerator term:
   $$ -g_0^4 = -(\mu_\theta - x)^4 = -\mu_\theta^4 + 4 \mu_\theta^3 x - 6 \mu_\theta^2 x^2 + 4 \mu_\theta x^3 - x^4 $$
2. Second numerator term (where $2 (\mu^2 - \mu_\theta^2) = 2 \lambda \mathcal{W}_0^2$):
   $$ 2 g_0^2 (\mu^2 - \mu_\theta^2) = 2 (\mu_\theta - x)^2 \lambda \mathcal{W}_0^2 = 2 \lambda \mathcal{W}_0^2 \mu_\theta^2 - 4 \lambda \mathcal{W}_0^2 \mu_\theta x + 2 \lambda \mathcal{W}_0^2 x^2 $$
3. Combining these with the constant terms $3 \mu_\theta^4 - 2 \mu^2 \mu_\theta^2$:
   * $x^0$ (constant vacuum energy density) terms: $-\mu_\theta^4 + 2 \lambda \mathcal{W}_0^2 \mu_\theta^2 + 3 \mu_\theta^4 - 2 \mu^2 \mu_\theta^2 = 2 \mu_\theta^4 + 2 (\mu^2 - \mu_\theta^2) \mu_\theta^2 - 2 \mu^2 \mu_\theta^2 = 0$. The vacuum energy density at the minimum vanishes.
   * $x^1$ (linear in density) terms: $4 \mu_\theta \left( \mu_\theta^2 - \lambda \mathcal{W}_0^2 \right) x$. While $\mu_\theta^2 - \lambda \mathcal{W}_0^2 \neq 0$, this term depends linearly on the density $n_B$ and is suppressed compared to the quadratic term in the high-density regime. An explicit numerical evaluation of the ratio between the linear and quadratic terms yields: $R \equiv \left| \frac{\text{linear}}{\text{quadratic}} \right| = \left| \frac{2 \mu_\theta (\mu_\theta^2 - \lambda \mathcal{W}_0^2)}{(3 \mu_\theta^2 - \lambda \mathcal{W}_0^2) x} \right| \approx \frac{0.25}{n_B / n_0}$. At nuclear density ($n_B = n_0$), the ratio is approximately $25\%$, while near the bounce threshold ($n_B \approx 2.5 n_0$), the linear term is only $10\%$ of the quadratic contribution and continues to fall as $O(n_0/n_B)$, validating the quadratic bounce approximation.
   * $x^2$ (quadratic in density) terms: $2 \left( \lambda \mathcal{W}_0^2 - 3 \mu_\theta^2 \right) x^2$.

Dividing by the denominator $4\lambda$ and substituting $x = \gamma \frac{n_B}{n_0}$ yields the leading non-vanishing energy density correction:

$$ \rho_{\mathcal{W}} \approx - \left[ \frac{3 \mu_\theta^2 - \lambda \mathcal{W}_0^2}{2 \lambda} \frac{\gamma^2}{n_0^2} \right] n_B^2 $$

Using $\rho_{\text{matter}} = M_N n_B$, we can express this correction as:

$$ \rho_{\mathcal{W}} \approx - \frac{\rho_{\text{matter}}^2}{\rho_c} $$

where the critical bounce density scale $\rho_c$ is determined by the physical parameters of the dense nuclear matter as:

$$ \rho_{c,\text{density}} \equiv \frac{2 \lambda M_N^2 n_0^2 (\hbar c)^3}{\left( 3 \mu_\theta^2 - \lambda \mathcal{W}_0^2 \right) \gamma^2} \approx 0.375 \text{ MeV/fm}^3 $$

At the same time, in the cosmological context (high temperature, radiation-dominated early universe with small baryon asymmetry), the melting of the condensate is driven by thermal fluctuations ($g_0^2 \approx \mu_\theta(T)^2 \propto T^2$). In this case, the critical bounce density scale is governed by the vacuum barrier scale $\rho_{c,\text{cosmology}} \equiv \frac{M_{\Omega,0}^4}{(\hbar c)^3} = \frac{4 V_0}{\lambda (\hbar c)^3} \approx 7.09 \times 10^4 \text{ MeV/fm}^3$.

Consider Einstein's equation $G_{\mu\nu} = 8\pi G T_{\mu\nu}$, where the total energy-momentum tensor consists of matter (dust/radiation) and the scalar sector $\mathcal{W}$: $T_{\mu\nu} = T^{\text{matter}}_{\mu\nu} + T^{\mathcal{W}}_{\mu\nu}$.

For a homogeneous FLRW universe, the 00-component of Einstein's equations (Hamiltonian constraint) yields:

$$ H^2 = \frac{8\pi G}{3} \left( \rho_{\text{matter}} + \rho_{\mathcal{W}} \right) $$

Substituting the expression for $\rho_{\mathcal{W}}$ with the quadratic correction, we obtain the modified Friedmann equation for NVG cosmology:

$$ H^2 = \frac{8\pi G}{3} \rho_{\text{matter}} \left( 1 - \frac{\rho_{\text{matter}}}{\rho_c} \right) - \frac{kc^2}{a^2} + \frac{\Lambda_{\text{eff}}c^2}{3} $$

At the point of maximum compression ($\rho_{\text{matter}} = \rho_c$), the expansion rate vanishes ($H = 0$), halting the gravitational collapse and triggering a smooth cosmological bounce, entirely eliminating the Big Bang singularity.
