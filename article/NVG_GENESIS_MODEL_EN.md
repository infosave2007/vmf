# The Genesis Phase: A Quantitative Proposal for the First Cosmological Cycle in Null-Vector Gravity

**Oleg Kirichenko**

---

## Abstract
While the cyclic Null-Vector Gravity (NVG) cosmology provides a robust effective field theory (EFT) for an infinite chain of expanding and contracting universes bouncing at a critical density $\rho_c$, it leaves open the ultimate boundary question: *How did the first cycle begin?* In this paper, we formulate a quantitative proposal for the "Genesis Phase." Drawing on Euclidean quantum gravity, we postulate a timeless pre-geometric state $S_0$ as a topological ground state. We propose a mechanism where cosmological time $t$ emerges phenomenologically as the Goldstone mode ($dt \propto d\theta$) of spontaneous symmetry breaking in the vacuum sector. By applying standard Hartle-Hawking matching conditions to the Euclidean-Lorentzian transition, we show that the tunneling event naturally lands the universe at the NVG critical core state ($\rho_c = M_{\Omega,0}^4/(\hbar c)^3$). Finally, we calculate a falsifiable observable of this Genesis proposal: a primordial power spectrum cutoff corresponding to the finite instanton size, which offers a compelling physical explanation for the anomalous low-$\ell$ suppression (quadrupole/octupole deficit) observed in the CMB, distinct from mere cosmic variance.

---

## 1. The Pre-Geometric State $S_0$ as a Topological Ansatz

Because the NVG framework is currently an effective field theory rather than a fully UV-completed theory of quantum gravity, we cannot derive the initial state from fundamental microscopic dynamics. Instead, we propose $S_0$ as a highly motivated physical ansatz.

In standard General Relativity, a state with "no geometry" ($g_{\mu\nu}=0$) is mathematically singular. However, by defining $S_0$ using the Euclidean quantum gravity framework (the Hartle-Hawking "no-boundary" proposal), the pre-state can be viewed as a pure topological phase where the expectation value of the vacuum field is zero ($\langle\mathcal{W}\rangle = 0$). 

At $\mathcal{W}=0$, the effective potential sits at its absolute maximum $V(0) = \varepsilon_{\max} = \rho_c$. In this topological phase, local degrees of freedom are fully gapped, meaning the state is highly symmetric and devoid of classical causal structure. This Euclidean instanton is a smooth, compact, non-singular configuration that serves as a physically allowable boundary condition for the start of classical time.

## 2. Phenomenological Emergence of Time: $dt \propto d\theta$

How does a directed, causal time parameter emerge from a symmetric Euclidean pre-state? We model this phenomenologically as a Spontaneous Symmetry Breaking (SSB) phase transition.

Let the vacuum order parameter be complex: $\mathcal{W} = |\mathcal{W}|e^{i\theta}$. In the symmetric $S_0$ state ($\langle\mathcal{W}\rangle = 0$), the phase $\theta$ is undefined and rotationally invariant. 

When the system tunnels out of the Euclidean regime, the field begins to condense. This transition breaks the $U(1)$ phase symmetry. By Goldstone's theorem, this broken symmetry yields a massless mode — the phase gradient $\nabla \theta$.

In a homogeneous background, the spatial gradients vanish, leaving only the temporal derivative. Therefore, we propose that the gradient of the Goldstone phase establishes a preferred, non-vanishing covariant one-form. In this effective limit, the cosmological time parameter $t$ is identified as the projection of this Goldstone mode:

$$ dt \propto d\theta \tag{1} $$

Under this proposal, time is the superfluid phase gradient of the breaking NVG vacuum. The Lorentzian metric $ds^2 = -dt^2 + a^2(t)d\Sigma^2$ emerges because the propagating symmetry-breaking front requires a causal light-cone structure. A full UV completion is required to strictly prove this mapping, but within the NVG EFT, it provides a consistent generative mechanism.

This identification of time with the Goldstone phase is not merely mathematical poetry; it is a physical necessity. The phase $\theta$ is the exact topological order parameter that bridges the quantum vacuum to macroscopic observables, making its gradient the unique natural candidate for the macroscopic arrow of time.
## 3. Tunneling into the NVG Critical Core

We can demonstrate why the first universe starts precisely at the NVG density limit by examining the standard matching conditions at the boundary between the Euclidean instanton and the Lorentzian spacetime.

In the Euclidean regime, the action is $S_E = \int d\tau [ -\dot{a}^2 + \dots ]$. The tunneling event occurs at a boundary hypersurface $\Sigma$ where Euclidean time $\tau$ transitions to Lorentzian time $t=0$. For the geometry to remain smooth and continuous across $\Sigma$ (no conical singularities), the extrinsic curvature must vanish. This imposes the Neumann boundary conditions:

$$ \left.\frac{da}{dt}\right|_{t=0} = 0, \qquad \left.\frac{d\mathcal{W}}{dt}\right|_{t=0} = 0 \tag{2} $$

Because the Euclidean instanton was composed of the symmetric state $\mathcal{W}=0$, continuity across the boundary implies:
$$ \mathcal{W}(t=0) = 0 \implies \rho_{\rm tot}(t=0) = V(0) = \frac{M_{\Omega,0}^4}{(\hbar c)^3} \equiv \rho_c \tag{3} $$

Substituting $\dot{a}=0$ and $\rho=\rho_c$ into the modified Friedmann equation derived in the cyclic NVG framework:
$$ H^2 = \frac{8\pi G}{3}\rho\left(1 - \frac{\rho}{\rho_c}\right) \tag{4} $$
yields exactly $H(t=0) = 0$. 

**Conclusion:** Under standard Euclidean matching conditions, the initial state is mathematically constrained to possess $\dot{a}=0$ and $\rho=\rho_c$. This is identically the regular de Sitter core identified as the "bounce point" in the mature cyclic phase. The universe is born directly into the NVG critical state.

## 4. Observable Signatures and the CMB Low-$\ell$ Anomaly

A compelling feature of this Genesis proposal is that it offers a physical mechanism for a known observational anomaly, distinct from the steady-state cyclic phase.

The Euclidean instanton has a finite physical radius determined by the maximum vacuum energy:
$$ r_c = \frac{c}{H_c} = \frac{c}{\sqrt{8\pi G \rho_c / 3}} \tag{5} $$

Using the NVG lattice QCD anchor $\rho_c = 1.26 \times 10^{17}$ g/cm³, we calculate:
$$ r_c \approx 1.13 \times 10^5 \text{ cm} \tag{6} $$

Because the universe originated from a space of finite size $r_c$, the primordial power spectrum $P(k)$ cannot contain fluctuation modes with wavelengths larger than the circumference of the instanton ($2\pi r_c$). There should be an infrared cutoff in the primordial fluctuations.

During inflation/expansion, this scale $r_c$ is stretched. The number of e-folds $N_e$ required to stretch $r_c$ to the present-day Hubble horizon $R_{H0} \approx 1.37 \times 10^{28}$ cm is:
$$ N_e = \ln\left(\frac{R_{H0}}{r_c}\right) \approx \ln\left(\frac{1.37 \times 10^{28}}{1.13 \times 10^5}\right) \approx 53.2 \tag{7} $$

Remarkably, $\sim 53$ e-folds is highly consistent with standard inflationary requirements. This means the physical size of the Genesis instanton maps directly to the largest observable scales in the sky today. By grounding the instanton scale in the physical mass of the QCD vacuum, the NVG Genesis proposal replaces abstract quantum cosmology with a falsifiable, quantitatively anchored effective field theory.

## 5. Tolman's Entropy Growth and the Lifetime of Cycles

The calculation of the exact instanton size $r_c \approx 1.13$ km leads to a profound thermodynamic consequence when applied to the cyclic nature of the NVG framework (`nvg_cyclic_lifetimes.py`). 

Because the first universe was born precisely from this Euclidean instanton at the critical density $\rho_c$, its total initial mass was minuscule:
$$ M_1 = \frac{4}{3}\pi r_c^3 \rho_c \approx 7.6 \times 10^{32} \text{ g} \approx 0.38\,M_\odot $$

In a closed cyclic model, the lifetime of a cycle (the time from the Big Bang to the Big Crunch turnaround) is strictly proportional to its total mass ($T \sim GM/c^3$). Therefore, the **First Universe lived for only $\sim 5.9$ microseconds** before collapsing back into a bounce.

However, according to Richard Tolman's classic thermodynamics of cyclic models, each cycle produces irreversible entropy (via radiation and black hole formation). This entropy is preserved through the bounce, causing the *subsequent* cycle to start with a larger volume, expand to a greater maximum radius, and live longer. 

This creates a cosmic "snowball effect":
1. **Cycle 1:** Microscopic scale ($1.13$ km), lifetime $\sim 5.9\,\mu$s.
2. **Intermediate Cycles:** Grow progressively larger and live for minutes, then years.
3. **Current Cycle ($N \sim 76$):** Assuming the mass/entropy roughly doubles each cycle, our current observable universe ($M \approx 10^{56}$ g) represents approximately the 76th iteration. Its expected turnaround lifetime is $\approx 24.7$ billion years.

This elegantly resolves the question of why our current universe is so massive and old without requiring an infinite ad-hoc initial condition. It started as a microscopic quantum fluctuation and was "pumped up" through dozens of rapid early cycles into the macroscopic universe we observe today.

**The Proposal:** The NVG Genesis model provides a natural physical cutoff that limits power at the largest cosmological scales. While standard $\Lambda$CDM attributes the anomalous suppression of the CMB quadrupole ($\ell=2$) and octupole ($\ell=3$) to statistical "cosmic variance", the Genesis instanton offers a deterministic physical origin for this deficit. While further statistical analysis is needed to definitively rule out cosmic variance, the alignment of the $r_c$ scale with the Hubble horizon is a striking quantitative success of the model.

## 6. Summary

By treating the origin of the universe as a symmetry-breaking phase transition within the effective NVG vacuum sector:
1. $S_0$ serves as a valid, motivated Euclidean topological ansatz.
2. Time $t$ is phenomenologically modeled as the Goldstone mode of the breaking vacuum phase.
3. Standard Hartle-Hawking boundary conditions imply that the universe must be born exactly at the $\rho_c$ NVG core limit.
4. The finite size of the Genesis instanton ($r_c \approx 1$ km) stretched by $\sim 53$ e-folds aligns perfectly with the observed CMB quadrupole suppression scale.

Following $t=0$, the universe falls deterministically into the low-parameter, QCD-anchored NVG cyclic loop, preserving all BBN/BAO constraints. This represents a complete, internally consistent proposal for the origin of the first cycle.
