# Vacuum Melting Law $W(\rho)$: Identifiability and Critical Exponent

**Author:** Oleg Kirichenko · **Framework:** [Null-Vector Gravity / Vacuum Mass Fraction (NVG/VMF)](https://github.com/infosave2007/vmf)

This note reports a focused, reproducible analysis of the NVG/VMF vacuum melting law
$W(\rho)$ — the amplitude of the vacuum condensate as a function of density — using the
framework's **own** forward models. It answers three questions the framework had left open:

1. **How much of $W(\rho)$ do neutron stars actually measure?**
2. **What critical density does neutron-star structure require, versus the cosmological $\rho_c$?**
3. **What is the melting exponent $\beta$ in $W \sim (1-\rho/\rho_c)^{\beta}$ — is the $\sqrt{\;}$-law derived or assumed?**

The guiding rule is the framework's own: *a claim counts only if it yields a computation, a
falsifiable number, or closes an underdetermined function — not a metaphor.*

Reproduce everything with three scripts in [`verification/`](verification/):

| script | what it computes |
|---|---|
| `nvg_melting_identifiability.py` | fast (crustless) overview: cosmological law is inert at NS density; effective $\rho_c$; form degeneracy |
| `nvg_melting_freeform_beta.py` | **canonical** $\beta$-equilibrium + crust + tidal model: sensitivity of $M_{\max}, R_{1.4}, \Lambda_{1.4}$ to $W(\rho)$, and the deep-core identifiability band |
| `nvg_melting_exponent.py` | derivation of the melting exponent $\beta$ from the NVG quartic $V(|\Phi|)$, and the universality correction |
| `nvg_melting_beta_bounce.py` | how $\beta$ shifts the bounce duration and the CMB low-$\ell$ cutoff $\ell_c$ |
| `nvg_melting_beta_besii.py` | mean-field vs 3-D Ising scaling test on published STAR net-proton $\kappa\sigma^2$ |
| `nvg_bounce_schrodinger_bridge.py` | the bounce as a Schrödinger bridge — per-cycle entropy production (Tolman) |
| `nvg_melting_ginzburg.py` | Ginzburg criterion from the framework's own $\lambda_v$, $v_0$ — is $\beta=1/2$ self-consistent? |
| `nvg_melting_beta_cmb_chi2.py` | **real CAMB** low-$\ell$ $\chi^2$ for the $\beta$-shifted cutoff vs Planck 2018 |
| `nvg_two_condensate_resolution.py` | resolves the CMB/heavy-ion $\beta$ tension via two decoupled condensates |

The canonical forward model (`nvg_tidal_deformability.py`) reproduces the framework's
published numbers **$M_{\max}=2.05\,M_\odot$, $R_{1.4}=12.55$ km, $\Lambda_{1.4}=519$**, so the
melting law is freed by monkeypatching only the effective-mass profile
$M^\*(x)=M_{\rm current}+M_{\Omega,0}\,g(x)$, $g(x)=W(x)/W(0)$, and rebuilding the EOS
self-consistently (the vector coupling is re-calibrated to $n_0$ saturation for each $g$).

---

## Finding 1 — Neutron stars measure $W(\rho)$ only up to $\sim 2.5\,n_0$

On the **canonical** model (β-equilibrium + crust + Hinderer tidal solver, validated above),
a localised $+8\%$ perturbation of $g(x)$ at density $x_0=\rho/n_0$ moves the observables by:

| $x_0\,[n_0]$ | $\Delta M_{\max}$ | $\Delta R_{1.4}$ | $\Delta\Lambda_{1.4}$ |
|---|---|---|---|
| 1.5 | $-0.01$ | $-0.85$ km | $-14$ |
| **2.0** | $+0.04$ | $+0.92$ km | $+90$ |
| 2.5 | $+0.02$ | $+0.48$ km | $+58$ |
| 3.5 | $+0.001$ | $+0.05$ km | $+25$ |
| $\ge 5$ | $\sim 0$ | $\sim 0$ | small |

The observables respond strongly to $W(\rho)$ at $1.5$–$2.5\,n_0$ and are **nearly blind above
$\sim 4\,n_0$** — a $1.4$–$2\,M_\odot$ star simply does not reach those densities.

**Deep-core identifiability band.** Perturbing only $g(x\ge 4\,n_0)$ by a smooth factor $(1+\delta)$,
the deep-core melting $g(6\,n_0)$ can range over $\approx 0.46$–$0.77$ (a contiguous $\sim 30\%$,
up to $\sim 50\%$ including a re-entrant branch) while **still** matching $M_{\max}$ (J0740),
$R_{1.4}$ (NICER) **and** $\Lambda_{1.4}$ (GW170817) simultaneously.

> **Consequence.** Even with the full GW170817 tidal data, the vacuum melting law above
> $\sim 4\,n_0$ is *not measured* — the power-law $(1+\kappa_2 x)^{-\kappa_1/\kappa_2}$ is a
> **modelling choice** in the deep core, a measurement only up to $\sim 2.5\,n_0$. What pins
> the deep core is not astrophysics but heavy-ion physics: the framework's own **HADES**
> in-medium meson-shift ($-20\%$ at $2\,n_0$) probes exactly the $2\,n_0$ region the stars see,
> and the deep core is reached only at FAIR / NICA / RHIC-BES densities.

---

## Finding 2 — Neutron stars require an effective $\rho_c$ about $150\times$ below the cosmological value

If the melting were the cosmological law $W=\sqrt{1-\rho/\rho_c}$ with the **real**
$\rho_c=7.09\times10^4$ MeV/fm³, the melting fraction at $6\,n_0$ is only $0.6\%$ — the condensate
barely melts and $M_{\max}\to 2.02\,M_\odot$ with $M^\*\approx 934$ MeV throughout. The
cosmological law is **inert at neutron-star density**.

Fitting $\sqrt{1-\rho/\rho_c^{\rm NS}}$ to the neutron-star $M_{\max}$ gives an **effective**
$\rho_c^{\rm NS}\approx 4.6\times10^2$ MeV/fm³ — about **$150\times$ lower** than the cosmological
$\rho_c$. So the single symbol "$W$" is really **two functions on two scales**: the same
condensate amplitude cannot use one critical density for both dense-matter melting and the
cosmological bounce. Either the neutron-star effective-mass drop is dominated by the vector
sector rather than condensate melting, or there is a second, lower-scale condensate transition
near $\sim 5\times10^2$ MeV/fm³ — a concrete fork worth stating explicitly.

---

## Finding 3 — The melting exponent $\beta$: the $\sqrt{\;}$-law is mean-field, and it is falsifiable

Write $W \sim (1-\rho/\rho_c)^{\beta}$ near $\rho_c$.

**Mean-field derivation.** Minimising the NVG quartic $V(|\Phi|)=\tfrac{\lambda}{4}(|\Phi|^2-v_0^2)^2$
with a linearly-melting symmetry-breaking scale $v^2(\rho)=v_0^2(1-\rho/\rho_c)$ returns, numerically,
$\beta = 0.5000$. For a **homogeneous** condensate the Madelung quantum pressure
$Q=-\tfrac{\hbar^2}{2m}\nabla^2\sqrt{W}/\sqrt{W}=0$, so the stationary Fokker–Planck / Madelung
balance reduces to $V'(W)=0$ — the classical minimum. **The phenomenological $\sqrt{\;}$-law is
therefore the exact mean-field extremum of NVG's own potential** — a genuine derivation, not a guess.

**But $\beta=1/2$ is a mean-field *assumption*.** A genuine 3-D second-order critical point forbids
$\beta=1/2$. The **QCD critical point is established to lie in the 3-D Ising universality class**
(Stephanov, Rajagopal, Shuryak). A finite-temperature transition — the bounce at $T_b=432$ MeV,
heavy-ion at $T_c\approx157$ MeV — dimensionally reduces $3{+}1\!\to\!3$D, so:

| $\beta$ | class |
|---|---|
| **0.500** | mean-field / $\sqrt{\;}$-law |
| 0.349 | 3-D XY ($\theta$ a Goldstone) |
| **0.326** | 3-D Ising (θ pinned; QCD-critical class) |

**The profiles diverge sharply near $\rho_c$** ($W/v_0=(1-\rho/\rho_c)^{\beta}$):

| $\rho/\rho_c$ | $\sqrt{\;}$ (0.5) | 3-D Ising (0.326) |
|---|---|---|
| 0.90 | 0.316 | 0.472 |
| 0.99 | 0.100 | 0.222 |
| **0.999** | **0.032** | **0.105 ($3.3\times$ larger)** |

Near the bounce the Ising condensate is $\sim 3.3\times$ larger than the $\sqrt{\;}$-law — it melts
**more slowly**, so more condensate energy survives close to $\rho_c$.

**Consequence for the bounce.** The condensate energy scales as $W^2\sim(1-\rho/\rho_c)^{2\beta}$.
The modified Friedmann term $H^2=\tfrac{8\pi G}{3}\rho\,(1-\rho/\rho_c)$ that produces the
Strong-Energy-Condition violation implicitly assumes $2\beta=1$. If $\beta=0.326$ the exponent is
$2\beta=0.653$, i.e. $H^2=\tfrac{8\pi G}{3}\rho\,(1-\rho/\rho_c)^{0.65}$ — a **different bounce**,
feeding the e-folds, the CMB low-$\ell$ cutoff, and the pre-bounce $w(z)$ (DESI).

> **Falsifiable, on existing data.** The exponent $\beta$ is measurable **now** from RHIC **BES-II**
> net-proton cumulant scaling near $T_c\approx157$ MeV: a two-hypothesis data-collapse of
> $C_4/C_2$, $C_3/C_2$ selects $\beta=1/2$ (VMF $\sqrt{\;}$-law) versus $\beta=0.326$ (3-D Ising).
> The **same** $\beta$ ties the neutron-star deep core (Finding 1, where the stars are blind),
> heavy-ion physics (where $\beta$ is measured), and the cosmological bounce (where $\beta$
> reshapes the Friedmann term) into **one number** — the single highest-leverage quantity to pin.

---

## The exponent $\beta$ propagates: three consequences

The melting exponent is not confined to the neutron-star / heavy-ion boundary — it feeds three
different observables. Each is worked out (differential / demonstration level) in a dedicated script.

**(1) CMB low-$\ell$ cutoff — now with a real CAMB refit** — `nvg_melting_beta_bounce.py`,
`nvg_melting_beta_cmb_chi2.py`. Generalising the bounce term to $H^2\propto\rho\,(1-\rho/\rho_c)^{2\beta}$,
a smaller $\beta$ makes the bounce **sharper** (proper time $\sim0.69\times$ for 3-D Ising) and moves the
comoving Genesis cutoff to **higher** $\ell$: $\ell_c=3.42\to\sim5$–$7.6$ for $\beta=0.326$. Running the
**actual CAMB** low-$\ell$ likelihood against the Planck 2018 TT deficit ($\ell=2$–$29$, exact Gamma
likelihood, $k^3$ cutoff) gives:

| model | $\chi^2_{\rm low\text{-}\ell}$ | vs mean field |
|---|---|---|
| no cutoff ($\Lambda$CDM) | 26.77 | |
| **mean-field $\beta=0.5$** ($\ell_c=3.42$) | **25.76** | best |
| 3-D Ising $\beta=0.326$ ($\ell_c\sim5.1$) | 27.66 | $+1.9$ |
| 3-D Ising $\beta=0.326$ ($\ell_c\sim7.6$) | 36.62 | $+10.9$ |

So the Planck low-$\ell$ deficit **prefers $\beta=1/2$ and disfavours $\beta=0.326$** — the opposite of
what heavy-ion / lattice-QCD universality wants.

> **A genuine, quantified tension.** The **same** exponent is pulled to $\beta\approx0.326$ by the
> QCD-scale melting (heavy-ion, lattice universality — consequence 2 and `nvg_melting_ginzburg.py`) and
> to $\beta=1/2$ by the cosmological bounce cutoff (CMB, above). One "$W$" cannot be a mean-field bounce
> **and** a 3-D-Ising QCD melting. Either the bounce transition ($T_b=432$ MeV) is a different
> universality class from the dense-matter one ($T_c\approx157$ MeV) — which dovetails with the earlier
> **two-scale $\rho_c$** result (Finding 2) — or the crude radiation-bounce $k_c(\beta)$ mapping needs
> the full horizon-chain treatment. Both are concrete next steps, and the tension itself is a real,
> falsifiable prediction of the single-condensate hypothesis.

**Is $\beta=1/2$ even self-consistent?** `nvg_melting_ginzburg.py` applies the Ginzburg criterion with
the framework's own $\lambda_v=1.02$, $v_0=859$ MeV ($m_\sigma=\sqrt{2\lambda_v}\,v_0=1.22$ GeV,
$\xi_0=0.16$ fm). The verdict is honestly **borderline**: the $4\pi$-suppressed loop parameter
$u=\lambda_v/4\pi=0.08$ gives a *small* Ginzburg window ($t_G\sim0.007$, favouring mean field), but
$\xi_0$ is only $\sim0.7\times$ the microscopic length and $\lambda_v$ is $O(1)$, so the mean-field
averaging is marginal. The crux is decided **externally** by lattice QCD (critical point 3-D Ising,
$N_f{=}2$ chiral O(4) — both non-mean-field). Hence the value of an actual RHIC BES-II measurement.

**(2) RHIC BES cumulant scaling** — `nvg_melting_beta_besii.py`. Within a universality class $\beta$
and the correlation-length exponent $\nu$ are co-fixed (mean-field $\nu=0.5$; 3-D Ising $\nu=0.630$),
and net-proton $\kappa\sigma^2=C_4/C_2\sim\xi^5$, $\xi\sim|\mu_B-\mu_c|^{-\nu}$. A scaling-shape fit to
published STAR BES-I $\kappa\sigma^2$ (0–5% central Au+Au) mildly **prefers 3-D Ising** ($\Delta\chi^2=+0.6$,
$\mu_c\approx200$ MeV, $\sqrt{s}_c\approx19$ GeV) — the broader Ising peak matches the width of the
$\sqrt{s}\sim20$ GeV dip better than mean field. Not decisive with BES-I errors; **BES-II** is the deliverable.

**(3) Cyclic-entropy (Tolman) problem** — `nvg_bounce_schrodinger_bridge.py`. Posing the bounce as a
**Schrödinger bridge** (entropic optimal transport, Sinkhorn) between the pre- and post-bounce
mini-superspace marginals gives a **finite, positive, computable** per-cycle entropy production
($\Delta S\sim0.14$ nats per mode here), turning Tolman's postulated "$M\times2$" into a derived
transport entropy — no prior art was found for this framing. $\Delta S>0$ forbids eternal cyclicity
(a finite number of past cycles), and $\Delta S$ is itself $\beta$-dependent (via the marginal widths
from consequence 1). So **the one exponent $\beta$ ties heavy-ion, the CMB cutoff, and the cyclic
entropy budget together.**

## Resolving the tension: two condensates, not one

The CMB-vs-heavy-ion $\beta$ tension dissolves if "$W$" is not one field. `nvg_two_condensate_resolution.py`
computes the decoupling on the canonical model: the maximum-mass star has a central density $\sim 7.2\,n_0$
($\varepsilon_c\approx1088$ MeV/fm³), where the **deep vacuum** condensate $W_{\rm deep}=\sqrt{1-\rho/\rho_c^{\rm cosmo}}$
is **$99.2\%$ intact** (only $0.8\%$ melted, since $\rho_c^{\rm cosmo}=7.09\times10^4$ MeV/fm³ is $\sim65\times$
beyond the NS core). A frozen condensate adds only a **constant** to $M^\*$, so its exponent $\beta_{\rm deep}$
does not enter NS structure at all. Forcing a *critical* melting into the NS density range instead makes the
star unphysical — confirming NS cores sit **below** any melting critical point.

So the consistent picture is **two condensates at two scales**, each with its own universality class:

| condensate | $\rho_c$ | $\beta$ | class | probed by |
|---|---|---|---|---|
| **deep vacuum** | $7.09\times10^4$ MeV/fm³ ($\sim470\,n_0$) | $0.5$ | mean-field | CMB bounce ($\ell_c=3.42$) |
| **chiral / dense-matter** | few $\times n_0$ | $0.326$ | 3-D Ising | RHIC BES-II (+ NS as its low-density tail) |

This is the physical form of the earlier **two-scale $\rho_c$** ($\sim150\times$) result: the QCD chiral
condensate and the deep vacuum condensate are **different order parameters**, so no single $\beta$ is
over-constrained. Neutron stars sit below both critical points and pin **neither** exponent (matching the
identifiability study). The resolution makes a **discriminating, falsifiable prediction**: BES-II must give
$\beta=0.326$ (chiral) *while* the CMB stays $\beta=0.5$ (deep vacuum) — a single condensate, forced to one
$\beta$, is already disfavoured ($\chi^2$: CMB wants $0.5$, heavy-ion wants $0.326$).

## Honest caveats

- The **crustless** `nvg_melting_identifiability.py` gives untrustworthy absolute radii (~21 km)
  and tidal $\Lambda$; only its differential quantities and the cosmological-vs-NS contrast are
  robust. The quantitative statements above use the **canonical** β-equilibrium + crust + tidal
  model (`nvg_melting_freeform_beta.py`, validated to 2.05 / 12.55 / 519).
- The 3-D critical exponents (0.326, 0.349) are quoted from conformal-bootstrap universality; the
  mean-field $\beta=1/2$ and all EOS/tidal numbers are computed here. Whether the $\rho_c$
  transition is genuinely fluctuation-dominated 3-D (universality wins, $\beta\approx0.33$) or
  effectively 4-D at its upper critical dimension (mean field survives, $\beta=1/2$ + logs) is the
  physics question to settle first; both branches are falsifiable against BES-II.
- The three "consequences" scripts are **differential / demonstration level**, with the scope stated
  in each: the CMB $\ell_c$ shift is the $\beta$-dependence under a radiation-bounce scaling (the exact
  value needs the CAMB horizon-chain pipeline); the BES-II fit uses published central values with a
  heuristic $\xi(\mu_B)$ profile (the definitive selection is a BES-II Phase-II analysis); and the
  Schrödinger-bridge $\Delta S$ is a proof-of-concept of the (apparently novel) framing, not yet
  evaluated on the true mini-superspace action.

## Bottom line

The $\sqrt{\;}$-law is **not arbitrary** — it is the mean-field extremum of NVG's own potential.
But $\beta=1/2$ is a mean-field assumption that a QCD-anchored critical point likely corrects to
$\beta\approx0.326$, which is **testable on existing RHIC BES-II data** and **reshapes the bounce**.
And neutron stars, contrary to a natural reading, constrain $W(\rho)$ only up to $\sim2.5\,n_0$ —
the deep core is a heavy-ion measurement, not an astrophysical one.
