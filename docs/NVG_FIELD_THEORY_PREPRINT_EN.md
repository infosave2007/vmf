# Vacuum-Deviation Variables in Metric Gravity: A Foundational Note on GR Plus a Canonical Scalar Sector

## Abstract

We formulate a metric field-theoretic description in which physically realized states are localized excitations above a vacuum configuration `\Omega_0` of an effective scalar sector `\mathcal{W}`. In its minimal form, the construction is mathematically equivalent to general relativity supplemented by a canonical scalar field and an optional additional stress-energy sector. No new gravitational dynamics is introduced at the level of the Einstein-Hilbert action, null-cone structure, or geodesic law. The value of the framework, in its present form, is therefore interpretive and organizational rather than dynamical. The manuscript is thus best read as a foundational note rather than as a proposal for a distinct theory of gravity.

Within that restricted status, three statements are established. First, the static weak-field limit reduces to a Poisson equation sourced by the total active energy density, so standard Newtonian gravity is recovered as a strict limit. Second, the electromagnetic field remains minimally coupled to the same spacetime metric, which implies luminal propagation in all local inertial frames and reproduces the standard first-order light-deflection and Shapiro-delay formulas. Third, for stationary configurations one may introduce a vacuum-subtracted mass variable such that

```math
E_{\Omega} = M_{\Omega} c^2,
```

where `M_\Omega` is the vacuum-subtracted mass of the configuration. This relation is a bookkeeping identity once the corresponding mass variable is defined; it is not claimed here as an independent derivation of relativistic mass-energy equivalence. We then state quantitative viability conditions imposed by Solar-System tests, gravitational-wave propagation, cosmic microwave background measurements, and late-time expansion data, and we explain why the present manuscript should be read as an interpretive EFT reformulation rather than as a distinct fundamental theory.

---

## 1. Introduction

Any viable extension of relativistic gravity must satisfy four non-negotiable requirements.

1. It must preserve local Lorentz invariance.
2. It must recover general relativity in the regimes where general relativity is already well tested.
3. It must be definable through an action principle.
4. It must make contact with observational bounds in a quantitative way.

The present work adopts that standard. The construction proposed here does not replace the geometric content of general relativity. Instead, it rewrites a standard metric theory with an additional canonical scalar field in terms of vacuum-deviation variables, with `\mathcal{W}` denoting the scalar sector and `\Omega_0` its stable ground state. Matter, radiation, and additional hidden degrees of freedom are interpreted as localized departures from that vacuum state.

This point must be stated explicitly. In its minimal realization, the framework is not a new dynamical theory of gravity. It is an interpretive reparameterization of GR plus a canonical scalar EFT. The present paper therefore has a narrower goal than a new-theory claim: it identifies what the vacuum-deviation language does and does not add, proves the standard weak-field and optical limits inside that language, and states the additional ingredients that would be required for the framework to become physically distinct.

The central restricted claim is as follows. For stationary systems, the observable mass of a configuration may be encoded as an integrated measure of vacuum-subtracted energy above the reference state `\Omega_0`, and the usual relativistic rest-energy relation remains valid once mass is defined in that way. The coefficient `c^2` is not modified; what changes is only the bookkeeping variable to which it is applied.

Accordingly, the appropriate genre of the present manuscript is a conceptual or foundational note about stress-energy bookkeeping in GR plus scalar EFT, not a breakthrough-theory paper.

---

## 2. Methods

### 2.1. Field content and action

The minimal theory contains the spacetime metric `g_{\mu\nu}`, standard matter fields summarized by `\mathcal{L}_m`, an effective scalar vacuum-deviation sector `\mathcal{W}`, and an optional hidden sector `\mathcal{L}_{\mathrm{extra}}`. The action is

```math
S
=
\int d^4x\,\sqrt{-g}
\left[
\frac{c^4}{16\pi G_*}(R - 2\Lambda_{\mathrm{eff}})
- \frac{1}{2}\partial_\mu \mathcal{W}\,\partial^\mu \mathcal{W}
- V(\mathcal{W})
+ \mathcal{L}_m
+ \mathcal{L}_{\mathrm{extra}}
\right].
```

Here `G_*` is the locally measured low-energy gravitational coupling. The vacuum state is defined by

```math
\Omega_0 \equiv \mathcal{W}_0,
\qquad
\left.\frac{dV}{d\mathcal{W}}\right|_{\mathcal{W}_0} = 0,
\qquad
\left.\frac{d^2V}{d\mathcal{W}^2}\right|_{\mathcal{W}_0} > 0.
```

For explicit calculations we adopt the quartic potential

```math
V(\mathcal{W}) = V_0 + \frac{m_{\mathcal{W}}^2}{2}(\mathcal{W}-\mathcal{W}_0)^2 + \frac{\lambda}{4}(\mathcal{W}-\mathcal{W}_0)^4.
```

### 2.2. Field equations

Variation with respect to `g_{\mu\nu}` yields

```math
G_{\mu\nu} + \Lambda_{\mathrm{eff}} g_{\mu\nu}
=
\frac{8\pi G_*}{c^4}
\left(
T^{(m)}_{\mu\nu}
+ T^{(\mathcal{W})}_{\mu\nu}
+ T^{(\mathrm{extra})}_{\mu\nu}
\right),
```

with

```math
T^{(\mathcal{W})}_{\mu\nu}
=
\partial_\mu \mathcal{W}\,\partial_\nu \mathcal{W}
- g_{\mu\nu}
\left[
\frac{1}{2}\partial_\alpha \mathcal{W}\,\partial^\alpha \mathcal{W}
+ V(\mathcal{W})
\right].
```

Variation with respect to `\mathcal{W}` gives

```math
\Box \mathcal{W} - \frac{dV}{d\mathcal{W}} = 0.
```

At this point the status of the minimal model is already fixed. The Einstein tensor is unchanged, the Einstein-Hilbert sector is unchanged, the causal cone is unchanged, and freely falling matter follows the standard metric geodesic structure. Consequently, the present minimal construction is mathematically equivalent to GR plus a canonical scalar field, supplemented if desired by an additional stress-energy source. Any stronger claim would require extra structure not yet introduced here.

### 2.3. Vacuum-subtracted energy density

Because the theory interprets physical states as departures from `\Omega_0`, it is convenient to introduce a vacuum-subtracted local energy density. This is not a new principle of physics; it is the standard renormalized choice of energy reference already familiar from quantum field theory in curved spacetime and from quasi-local or asymptotic energy constructions. For a weakly curved background it is

```math
\varepsilon_{\Omega}(x)
=
\rho_m c^2
+ \rho_{\mathrm{extra}} c^2
+ \frac{1}{2}\dot{\mathcal{W}}^2
+ \frac{1}{2a^2}|\nabla \mathcal{W}|^2
+ V(\mathcal{W}) - V(\mathcal{W}_0).
```

The associated vacuum-deviation mass density is defined by

```math
\rho_{\Omega}(x) = \frac{\varepsilon_{\Omega}(x)}{c^2}.
```

This definition is useful only with appropriate restrictions. In a generic nonstationary spacetime, and in particular in FRW cosmology, there is no globally conserved total energy associated with an arbitrary spatial slicing. Therefore `\varepsilon_{\Omega}` should be treated as a local bookkeeping density, while integrated energies are unambiguous only in stationary spacetimes, in asymptotically defined constructions, or after a specific foliation choice has been fixed.

### 2.4. What gravity does and does not couple to

The Einstein equations in Section 2.2 are sourced by the full stress-energy tensor, not by a locally subtracted one. The subtraction

```math
V(\mathcal{W}) - V(\mathcal{W}_0)
```

enters only when defining energy variables relative to a chosen reference vacuum or background. If the reference vacuum contributes a pure vacuum term

```math
T^{\mathrm{vac}}_{\mu\nu} = -\rho_{\mathrm{vac}} c^2 g_{\mu\nu},
```

then changing that reference is equivalent to shifting the effective cosmological constant,

```math
\Lambda_{\mathrm{eff}} \mapsto \Lambda_{\mathrm{eff}} + 8\pi G_*\rho_{\mathrm{vac}}/c^2,
```

rather than proving that gravity responds only to a subtracted local density. The present framework therefore does not solve the problem of why vacuum energy gravitates so weakly; it only introduces a controlled reference-dependent decomposition of the source sector.

### 2.5. Electromagnetic sector

The electromagnetic field is minimally coupled to the same metric:

```math
S_{\mathrm{em}} = -\frac{1}{4}\int d^4x\,\sqrt{-g}\,F_{\mu\nu}F^{\mu\nu}.
```

This single assumption fixes the causal structure used below.

---

## 3. Results I: Recovery of gravity

### 3.1. Static weak-field limit

Consider the standard weak-field ansatz

```math
g_{00} = -\left(1 + \frac{2\Phi}{c^2}\right),
\qquad
g_{ij} = \left(1 - \frac{2\Phi}{c^2}\right)\delta_{ij},
\qquad
\left|\frac{\Phi}{c^2}\right| \ll 1.
```

Assume a static source, negligible pressures compared to `\rho c^2`, and slowly varying `\mathcal{W}`. Then

```math
T_{00}^{\mathrm{tot}} \simeq \rho_{\Omega} c^2,
```

and the `00` Einstein equation reduces to

```math
G_{00} \simeq \frac{2}{c^2}\nabla^2 \Phi.
```

Substitution gives the Poisson equation

```math
\nabla^2 \Phi = 4\pi G_* \rho_{\Omega}.
```

Thus the gravitational potential is sourced by the full vacuum-subtracted active density rather than by bare matter alone.

### 3.2. Active mass and Newtonian acceleration

For spherical symmetry,

```math
M_{\Omega}(r) = 4\pi\int_0^r \rho_{\Omega}(r')\,r'^2\,dr',
```

and therefore

```math
g(r) = -\frac{d\Phi}{dr} = -\frac{G_* M_{\Omega}(r)}{r^2}.
```

If `\mathcal{W} \to \mathcal{W}_0` and `\rho_{\mathrm{extra}} \to 0`, then `\rho_{\Omega} \to \rho_m` and the theory reduces exactly to the standard Newtonian limit.

### 3.3. Exterior limit

Outside a compact source, `\rho_{\Omega}=0` and the solution becomes

```math
\Phi(r) = -\frac{G_* M_{\Omega}}{r}.
```

Hence the exterior metric is Schwarzschild-like with the replacement of the source mass by the total vacuum-deviation mass `M_\Omega`. This ensures that weak-field orbital tests are automatically recovered when additional sectors are sufficiently suppressed outside compact bodies.

---

## 4. Results II: Light-cone structure and the speed of light

### 4.1. Eikonal limit

Write the electromagnetic potential in geometric optics as

```math
A_\mu = a_\mu e^{i\theta/\epsilon},
\qquad
k_\mu = \nabla_\mu \theta,
\qquad
\epsilon \to 0.
```

The leading-order Maxwell equation yields

```math
k_\mu k^\mu = 0.
```

Therefore electromagnetic rays follow null geodesics of `g_{\mu\nu}`.

### 4.2. Local luminal propagation

At any event one may choose local inertial coordinates in which

```math
ds^2 = -c^2 dT^2 + dX^2 + dY^2 + dZ^2 + O((x-x_0)^2).
```

For a null trajectory, `ds^2=0`, hence

```math
\left|\frac{d\mathbf{X}}{dT}\right| = c.
```

The locally measured speed of light is therefore exactly `c`. The theory does not require a variable fundamental light speed.

### 4.3. Coordinate speed, Shapiro delay, and lensing

For the weak-field metric above and a radial null ray,

```math
v_{\mathrm{coord}}
=
\frac{d\ell}{dt}
=
c\sqrt{\frac{1+2\Phi/c^2}{1-2\Phi/c^2}}
\simeq
c\left(1 + \frac{2\Phi}{c^2}\right).
```

For an attractive source, `\Phi<0`, so the coordinate speed is reduced. This is equivalent to an effective refractive index

```math
n(\mathbf{x}) \simeq 1 - \frac{2\Phi(\mathbf{x})}{c^2},
```

which implies the standard first-order Shapiro delay

```math
\Delta t_{\mathrm{Shapiro}} \simeq -\frac{2}{c^3}\int \Phi\,d\ell.
```

For a point-like lens of mass `M_\Omega` and impact parameter `b`, the deflection angle is

```math
\hat{\alpha} \simeq \frac{4G_* M_{\Omega}}{b c^2}.
```

At the solar limb this reduces to the standard value `\hat{\alpha}_\odot \simeq 1.75` arcsec when `M_\Omega = M_\odot`.

The same mass that governs free-fall acceleration therefore governs lensing at first post-Newtonian order.

---

## 5. Results III: Stationary vacuum-subtracted mass-energy bookkeeping

### 5.1. Stationary energy functional

In a stationary spacetime with timelike Killing vector `\xi^\mu`, the conserved energy is

```math
E = \int_{\Sigma} d\Sigma_\mu\,T^{\mu}{}_{\nu}\,\xi^{\nu}.
```

For a localized, weak-field, stationary source in its rest frame, this becomes the integral of the vacuum-subtracted energy density:

```math
E_{\Omega} = \int d^3x\,\varepsilon_{\Omega}(x).
```

Define the vacuum-deviation mass by

```math
M_{\Omega} = \int d^3x\,\rho_{\Omega}(x) = \frac{1}{c^2}\int d^3x\,\varepsilon_{\Omega}(x).
```

Therefore

```math
E_{\Omega} = M_{\Omega} c^2.
```

This is not an independent derivation of relativistic mass-energy equivalence. It is a tautological identity once `M_{\Omega}` has been defined by dividing the stationary vacuum-subtracted energy by `c^2`. The utility of the definition is organizational: it keeps matter, scalar-field energy, and hidden-sector contributions in a single mass functional. Its limitation is equally clear: by itself it does not explain the microscopic origin of inertial mass.

### 5.2. Standard relativistic limit

If the `\mathcal{W}` and hidden-sector contributions are negligible, then

```math
M_{\Omega} \to m,
\qquad
E_{\Omega} \to mc^2.
```

Thus the familiar relation is recovered as a strict limit rather than replaced.

### 5.3. Moving compact configurations

Local Lorentz invariance implies that a compact object with rest mass `M_\Omega` satisfies

```math
E_{\Omega}(v) = \gamma M_{\Omega} c^2,
\qquad
\gamma = \frac{1}{\sqrt{1-v^2/c^2}}.
```

The theory therefore modifies the definition of rest mass, not the kinematic structure of special relativity.

### 5.4. Scope and limitation of the construction

The preceding identity should not be exported without care to generic cosmological backgrounds. In nonstationary FRW spacetime there is, in general, no timelike Killing vector and hence no globally conserved energy of the form used above. Accordingly, `M_{\Omega}` is a clean invariant quantity only for stationary systems or in appropriately defined asymptotic settings. In cosmology it becomes, at best, a useful slicing-dependent diagnostic variable unless a more fundamental quasi-local construction is specified.

### 5.5. Quasi-local geometric realizations

The bookkeeping mass `M_{\Omega}` becomes geometrically meaningful only when attached to standard quasi-local or asymptotic mass constructions already available in GR. Two cases are especially relevant.

For stationary asymptotically flat spacetimes with timelike Killing vector `\xi^\mu`, the natural invariant object is the Komar mass,

```math
M_{\mathrm{K}} = -\frac{c^2}{8\pi G_*}\oint_{\partial\Sigma} \nabla^{\mu}\xi^{\nu}\,dS_{\mu\nu}.
```

Using the field equations, one may define a vacuum-referenced Komar charge by subtracting the corresponding charge of a chosen reference vacuum branch,

```math
M_{\Omega}^{\mathrm{K}} \equiv M_{\mathrm{K}}[T^{\mathrm{tot}}] - M_{\mathrm{K}}[T^{\mathrm{ref}}].
```

In the weak-field, stationary, nonrelativistic limit this reduces to the volume integral used earlier,

```math
M_{\Omega}^{\mathrm{K}} \longrightarrow \int d^3x\,\rho_{\Omega}(x),
```

up to the standard pressure corrections familiar from Komar/Tolman mass formulas.

For spherically symmetric geometries, including dynamical ones, the natural invariant object is the Misner-Sharp-Hernandez mass,

```math
M_{\mathrm{MS}}(R) = \frac{c^2 R}{2G_*}\left(1 - g^{ab}\nabla_a R\nabla_b R\right),
```

where `R` is the areal radius. A vacuum-referenced spherical mass may then be defined by

```math
M_{\Omega}^{\mathrm{MS}}(R) \equiv M_{\mathrm{MS}}(R) - M_{\mathrm{MS}}^{\mathrm{ref}}(R).
```

In the weak-field static limit one recovers

```math
M_{\Omega}^{\mathrm{MS}}(R) \simeq 4\pi\int_0^R \rho_{\Omega}(R')\,R'^2\,dR'.
```

This does not remove the general ambiguities of quasi-local energy in GR, but it does place the vacuum-deviation mass inside the standard geometric zoo rather than outside it.

### 5.6. Reference-vacuum dependence

The subtraction prescription is unique only after a reference vacuum has been specified. If the potential `V(\mathcal{W})` admits multiple minima,

```math
\mathcal{W}_{0,i},
```

then the framework defines a family of reference-dependent masses,

```math
M_{\Omega,i},
```

one for each vacuum branch. Uniqueness therefore requires additional dynamical input, such as cosmological initial conditions, phase-selection history, boundary conditions, or a metastability criterion. The present note does not solve that ambiguity; it only makes it explicit.

---

## 6. Observational consistency conditions

The theory is only viable if it survives existing experimental bounds. This section states the sharp conditions that must be met.

### 6.1. Solar-System and binary-pulsar regime

In the minimal metric realization written above, matter couples only to `g_{\mu\nu}` and standard post-Newtonian behavior is recovered whenever local vacuum-deviation gradients are small:

```math
|\nabla \mathcal{W}|^2/c^2 \ll \rho_m,
\qquad
\rho_{\mathrm{extra}} \ll \rho_m
```

inside bound systems.

If the theory is embedded in a more general scalar-tensor completion with conformal matter coupling `A(\mathcal{W})g_{\mu\nu}`, the standard relation

```math
\gamma_{\mathrm{PPN}} - 1 = -\frac{2\alpha_0^2}{1+\alpha_0^2},
\qquad
\alpha_0 \equiv M_{\mathrm{Pl}}\left.\frac{d\ln A}{d\mathcal{W}}\right|_{\mathcal{W}_0}
```

applies. The Cassini bound `|\gamma_{\mathrm{PPN}}-1| < 2.3\times 10^{-5}` then requires approximately

```math
\alpha_0^2 \lesssim 10^{-5}.
```

Likewise, any time variation of the effective gravitational coupling must satisfy

```math
\left|\frac{\dot G}{G}\right| \lesssim 10^{-13}\,\mathrm{yr}^{-1}
```

at late times, otherwise the theory conflicts with Lunar Laser Ranging and related local constraints.

### 6.2. Gravitational-wave propagation

Because the minimal theory uses a single metric light cone and no derivative non-minimal couplings of Horndeski type, it predicts

```math
c_T = c
```

for tensor propagation in the late Universe. This is required observationally: GW170817 and GRB170817A imply that deviations satisfy

```math
\left|\frac{c_T}{c} - 1\right| \lesssim \text{few} \times 10^{-15}.
```

In addition, the LIGO-Virgo-KAGRA O4a tests, including 42 new high-significance binary-black-hole mergers, found no evidence for departures from general relativity in waveform generation or propagation. Therefore any viable parameter choice of the present theory must reduce to general-relativistic tensor dynamics in the strong-field wave zone to current observational precision.

### 6.3. Light deflection and lensing consistency

Since first-order lensing depends on the same `M_\Omega` that enters the Newtonian potential, the theory automatically predicts equality between the dynamical mass and the lensing mass in the weak-field metric limit. This is a nontrivial requirement: if the hidden sector contributes to `M_\Omega`, it must do so in a way consistent with both orbital dynamics and lensing.

At the same time, the hidden sector as written is only a placeholder. Without an explicit microphysical Lagrangian, parameter range, and screening or clustering behavior, it remains a black-box source term. In that form it weakens falsifiability rather than strengthening it. Any serious continuation of the program must therefore replace `\mathcal{L}_{\mathrm{extra}}` by a concrete sector with a finite parameter set.

### 6.4. Early-Universe consistency with the CMB

Planck 2018 found excellent agreement with flat six-parameter `\Lambda`CDM, with representative values

```math
H_0 = 67.4 \pm 0.5\ \mathrm{km\,s^{-1}\,Mpc^{-1}},
\qquad
\Omega_m = 0.315 \pm 0.007,
\qquad
\sigma_8 = 0.811 \pm 0.006,
```

and an acoustic scale measured to `0.03\%` precision,

```math
100\,\theta_* = 1.0411 \pm 0.0003.
```

Therefore any nonstandard contribution from `\mathcal{W}` or the hidden sector before recombination must leave the acoustic horizon and projection geometry extremely close to the standard model prediction. In practice, this means that early-time deviations cannot be order unity; they must be perturbative or screened at recombination.

Planck also gives

```math
N_{\mathrm{eff}} = 2.99 \pm 0.17,
```

so the hidden sector cannot behave like an unconstrained bath of extra relativistic species.

### 6.5. Late-time expansion, DESI, and the local distance ladder

Late-time dynamics can be parameterized in the standard CPL form

```math
w(a) = w_0 + w_a(1-a).
```

The present theory may generate an effective `w(a) \neq -1` through the vacuum-deviation sector, but only in a regime that remains consistent with BAO distances, supernova luminosity distances, weak lensing, and structure growth.

DESI 2025 strengthened hints that dark energy may evolve and released a 3D map based on 18.7 million galaxies, quasars, and stars. This motivates allowing a mild late-time dynamical sector, but it does not license large deviations from the background expansion history without a full likelihood analysis.

Independent of that, JWST measurements confirmed the accuracy of the Cepheid-based local distance ladder used in Hubble-constant work, with more than 320 Cepheids in the first calibration steps and four additional supernova hosts showing consistent behavior. Hence the model may use extra fields to address late-time tensions, but it cannot appeal to a failure of the local ladder calibration itself.

### 6.6. Vacuum energy and cosmological constant problem

The present formulation does not solve the cosmological constant problem. The subtraction

```math
V(\mathcal{W}) - V(\mathcal{W}_0)
```

is a standard renormalized normalization choice, not a dynamical explanation of why quantum-vacuum contributions gravitate so weakly in the observed Universe. Any claim to genuine theoretical novelty would require an explicit mechanism that addresses the mismatch between naive quantum-vacuum estimates and the observed effective cosmological constant.

### 6.7. Fifth-force and screening issue

A light scalar sector generically raises the standard problems of fifth forces, equivalence-principle bounds, Yukawa corrections, and screening. In the minimal metric realization used here these issues are avoided only because the scalar sector is assumed either to decouple sufficiently from visible matter or to remain dynamically subdominant in already tested regimes. That means the present paper does not yet exhibit a nontrivial scalar phenomenology that is simultaneously large enough to matter and small enough to evade exclusion.

### 6.8. EFT status and ultraviolet incompleteness

The present manuscript is an EFT-level construction. It does not provide a UV completion, a renormalizable completion, a quantum-gravity prescription, or a singularity-resolution mechanism. Accordingly, no claim is made here about Planck-scale consistency, black-hole singularities, or microscopic quantization beyond the low-energy effective action written in Section 2.

---

## 7. What Would Be Required for Genuine New Physics

The analysis above shows that the minimal theory is best read as an interpretive reformulation, not as a distinct dynamical framework. For the program to become genuinely new physics, at least one additional ingredient would be required.

1. A new observable prediction not already reproduced by GR plus known scalar EFTs, for example a controlled compact-object signature, a calculable vacuum-dependent inertial-mass shift, or a laboratory-scale gravity-vacuum coupling effect.
2. A nontrivial modification of geometric structure, such as an emergent metric, modified connection, nonlocal elastic response of the vacuum, or another mechanism not reducible to the Einstein-Hilbert sector plus a canonical scalar field.
3. A real mechanism for the cosmological constant problem rather than a subtraction prescription.
4. A microscopic origin of mass, such as a solitonic, topological, condensate, or bound-vacuum-energy mechanism, from which the stationary mass functional would follow rather than being adopted as a definition.

The most conservative and physically grounded continuation of item 4 is the hadronic QCD route: define the vacuum-structured share of inertial mass through sigma terms, gluonic energy, and trace-anomaly contributions rather than through an abstract cosmological vacuum variable. That path does not produce a new gravity theory, but it does create a plausible observable handle for the broader program.

Absent one of these ingredients, the framework remains a coherent but interpretive rewriting of standard low-energy relativistic field theory.

---

## 8. Discussion

The most severe criticism of the present manuscript is also the correct one: in its minimal form, the model is mathematically equivalent to GR plus a canonical scalar field, written in vacuum-deviation language. The Einstein tensor is unchanged, the causal structure is unchanged, geodesic motion is unchanged, and the weak-field and optical limits are standard. For that reason, the present text should not be read as establishing a new theory of gravity.

What the manuscript does provide is narrower. It supplies a coherent reinterpretation of standard source terms in terms of vacuum-subtracted variables and makes explicit how the usual weak-field and optical formulas are recovered in that language. That is mathematically legitimate, but it is not yet independent physics.

The same verdict applies to the mass-energy section. Once the stationary mass functional is defined as vacuum-subtracted energy divided by `c^2`, the relation `E_{\Omega} = M_{\Omega} c^2` follows identically. This is useful as notation, but it does not by itself explain the origin of inertial mass, nor does it derive special relativity from deeper principles.

The most important mathematical refinement made in the present revision is therefore not a new dynamical equation, but a change of status for `M_{\Omega}` itself. Rather than leaving it as a preferred-frame volume integral, the note now identifies the relevant geometric carriers in the regimes where they exist: Komar mass for stationary spacetimes and Misner-Sharp-Hernandez mass for spherical symmetry. That still falls short of a fully general quasi-local construction, but it removes the impression that `M_{\Omega}` floats entirely outside the standard mass notions of GR.

Accordingly, the proper status of the framework is as follows. It is a clean interpretive EFT description with well-controlled classical limits and explicit observational constraints. It is not yet a distinct gravitational theory, a solution to the cosmological constant problem, or a UV-complete model.

---

## 9. Conclusion

We have written the vacuum-deviation framework in a form suitable for direct technical scrutiny. In its minimal form, the construction is equivalent to GR plus a canonical scalar sector and should be understood as an interpretive reparameterization rather than as a new dynamical theory.

Within that restricted status, three statements were established.

1. Gravity reduces to a Poisson equation sourced by the full vacuum-subtracted active density and recovers Newtonian gravity as a strict limit.
2. The locally measured speed of light remains exactly `c`, while first-order light propagation effects are those of standard metric gravity.
3. The stationary bookkeeping identity

```math
E_{\Omega} = M_{\Omega} c^2
```

follows directly once mass is defined as the vacuum-subtracted integrated energy density.

The framework is therefore internally coherent and compatible with the standard relativistic structure at the level of its basic equations. It is now also tied more explicitly to standard geometric mass notions in the stationary and spherically symmetric regimes. What it does not yet provide is equally important: no new gravitational dynamics, no independent observational signature beyond standard scalar EFT phenomenology, no solution to the cosmological constant problem, no unique generally covariant mass functional in arbitrary spacetime, and no ultraviolet completion. Those absences define the actual research program. Any future version that aims to count as genuinely new physics must supply at least one nontrivial dynamical principle or one observationally distinct prediction beyond the minimal GR-plus-scalar realization analyzed here.

---

## References

1. S. Weinberg, Cosmology.
2. V. Mukhanov, Physical Foundations of Cosmology.
3. C. M. Will, The Confrontation between General Relativity and Experiment.
4. T. Clifton, P. G. Ferreira, A. Padilla, C. Skordis, Modified Gravity and Cosmology.
5. B. Bertotti, L. Iess, P. Tortora, A test of general relativity using radio links with the Cassini spacecraft, Nature 425, 374-376 (2003).
6. A. Komar, Covariant conservation laws in general relativity, Phys. Rev. 113, 934-936 (1959).
7. C. W. Misner, D. H. Sharp, Relativistic equations for adiabatic, spherically symmetric gravitational collapse, Phys. Rev. 136, B571-B576 (1964).
8. J. D. Brown, J. W. York, Quasilocal energy and conserved charges derived from the gravitational action, Phys. Rev. D 47, 1407-1419 (1993).
9. Planck Collaboration, Planck 2018 results. VI. Cosmological parameters, A&A 641, A6 (2020), arXiv:1807.06209.
10. P. Creminelli, F. Vernizzi, Dark Energy after GW170817 and GRB170817A, Phys. Rev. Lett. 119, 251302 (2017), arXiv:1710.05877.
11. LIGO-Virgo-KAGRA Collaboration, Testing general relativity with observing run O4a data, 2024.
12. DESI Collaboration, More Than a Hint of Evolving Dark Energy - New Results and Data from DESI, 2025.
13. A. G. Riess et al., Webb Confirms Accuracy of Universe's Expansion Rate Measured by Hubble, 2023; see also arXiv:2307.15806.