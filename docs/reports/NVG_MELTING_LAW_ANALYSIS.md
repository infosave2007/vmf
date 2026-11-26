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
| `nvg_two_condensate_resolution.py` | **hypothesis** that two weakly-coupled condensates would remove the (weak) CMB/heavy-ion $\beta$ tension — model input, not derived |
| `nvg_melting_tail_sensitivity.py` | does a heavy-NS radius $R_{2.0}$ feel the chiral exponent $\beta$? (**no** — the $\approx0.5$ km signal is a generic stiffness knob, not $\beta$) |
| `nvg_melting_beta_horizon_chain.py` | **horizon-chain refinement**: is $\ell_c(\beta)$ real or a proxy artefact? (instanton $r_c=c/H_c$ is $\beta$-independent; integrated $k_c$ + real CAMB) |
| `nvg_melting_master_report.py` | runs all scripts, confirms they execute, writes the consolidated [`NVG_MELTING_REPORT.md`](NVG_MELTING_REPORT.md) |

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

**But $\beta=1/2$ is a mean-field *assumption* — indeed it is the Landau ansatz restated.** $\beta=1/2$
follows directly and *only* from the assumed **linear** melting $v^2(\rho)=v_0^2(1-\rho/\rho_c)$ (any
melting power $p$ gives $\beta=p/2$), so identifying the $\sqrt{\;}$-law with mean field is not an
independent prediction but a consequence of the linear ansatz. A genuine 3-D second-order critical point
forbids $\beta=1/2$. *If* the QCD critical point exists, it is **expected to lie in the 3-D Ising
universality class** (Stephanov, Rajagopal, Shuryak) — a theoretical expectation, not an established
measurement. A finite-temperature transition — the bounce at $T_b=432$ MeV, heavy-ion at
$T_c\approx157$ MeV — dimensionally reduces $3{+}1\!\to\!3$D, so:

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

> **Where could $\beta$ actually be measured?** Heavy-ion is the only realistic handle. RHIC net-proton
> cumulant scaling near $T_c\approx157$ MeV could in principle select $\beta=1/2$ (VMF $\sqrt{\;}$-law)
> versus $\beta=0.326$ (3-D Ising) via a two-hypothesis collapse of $C_4/C_2$, $C_3/C_2$ — but the
> **existing BES-I** central values (STAR, PRL 126, 092301, 2021) do **not yet discriminate** (a flat
> no-critical-point null already fits; see consequence 2). **BES-II** is the clean *future* test.
> The neutron-star deep core is **blind** to $\beta$ (Finding 1), and — as the CMB analysis below shows —
> the cosmological cutoff does **not robustly** constrain it either. So $\beta$ is best read as a
> **theoretical-consistency parameter constrained by at most one domain (heavy-ion)**, not a single
> number simultaneously measured across three.

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

On this **point-proxy** mapping the Planck low-$\ell$ deficit appears to **prefer $\beta=1/2$** — the
opposite of what heavy-ion / lattice-QCD universality wants. But that mapping evaluated the comoving
Hubble radius at an *arbitrary* melting-onset density $r_*=0.9$–$0.99$, and the apparent tension turns
out to hinge entirely on that choice.

**Horizon-chain refinement (`nvg_melting_beta_horizon_chain.py`) — the tension is *not robust*.**
Redoing the cutoff with the framework's actual Genesis horizon chain (`nvg_genesis_observable.py`):

- **(A) Committed mechanism — the Genesis instanton.** The cutoff scale is the Euclidean instanton
  $r_c=c/H_c$ with $H_c=\sqrt{8\pi G\rho_c/3}$, which uses $\rho_c$ **alone**, so read this way $\beta$
  never enters $\ell_c=3.42$. But this $\beta$-independence is **conditional**, not a physical robustness:
  it holds only if (i) the cutoff is the *unsuppressed* de Sitter scale $H_c$ rather than the bounce
  **peak-curvature** scale $H_{\max}$ at $\rho=\rho_c/(1{+}\beta)$ — which carries the melting factor
  $(1-\rho/\rho_c)^{\beta}$ and **reintroduces $\beta$ at the $\sim10\%$ level** ($r_c(0.326)/r_c(0.5)\approx0.90$)
  — and (ii) $N_e$ is a *free* calibration to the local $H_0$. Indeed, since $N_e=\ln(R_{H0}/r_c)$ by
  construction $k_c=1/R_{H0}$: the cutoff is **pinned to today's Hubble horizon regardless of $r_c$**, so
  $\ell_c=3.42$ is a near-tautology fixed by $H_0$, **not a $\beta$-discriminating prediction**. Reading (A)
  *removes a robust constraint*; it does not prove $\ell_c$ is physically $\beta$-independent.
- **(B) Alternative — the near-bounce particle horizon.** If instead one identifies the cutoff with the
  *integrated* particle horizon $\eta(\beta)\propto\int r^{-5/4}(1-r)^{-\beta}\,dr$, the ratio
  $k_c(0.326)/k_c(0.5)\approx1.44$ ($r_*=0.5$; ranges $1.2$–$2.0$ over $r_*\in[0.1,0.9]$) — a real but
  $r_*$-dependent shift. Feeding this integrated $k_c$ to the same CAMB likelihood gives $\chi^2=27.31$
  (**$+1.55$**), and the earlier $+10.9$ runaway ($\ell_c\sim7.6$) is the point-proxy's $r_*\!\to\!1$ blow-up.

> **Statistically, the low-$\ell$ CMB does not discriminate $\beta$ at all.** At $\ell=2$–$29$ cosmic
> variance dominates, and the $\chi^2$ gaps are tiny: $\beta=0.5$ vs no-cutoff is $\Delta\chi^2=1.01$
> ($\sim1.0\sigma$), $\beta=0.5$ vs $\beta=0.326$(integrated) is $1.9$ ($\sim1.4\sigma$) — **both below
> $2\sigma$**, further reduced by the $f_{\rm sky}\!\sim\!0.86$ mask, and Planck 2018 (X) reports no
> significant low-$\ell$ cutoff. So the CMB is **consistent with $\beta=0.5$, no-cutoff, and $\beta=0.326$
> alike**; the $+10.9$ figure is not a physical disfavouring but the uncertain $k_c(\beta)$ mapping (two
> admissible mappings span $\chi^2=27.7$–$36.6$). **Planck does not constrain $\beta$** — whether through
> the (conditional) instanton reading or through raw significance. The two-condensate picture below is a
> *hypothesis motivated by the two-scale $\rho_c$* (Finding 2), **not** anchored by a CMB $\beta$-measurement.

**Is $\beta=1/2$ even self-consistent?** `nvg_melting_ginzburg.py` applies the Ginzburg criterion with
the framework's own $\lambda_v=1.02$, $v_0=859$ MeV ($m_\sigma=\sqrt{2\lambda_v}\,v_0=1.22$ GeV,
$\xi_0=0.16$ fm). The naive Ginzburg number $t_G\sim0.007$ (from $u=\lambda_v/4\pi=0.08$) *looks*
mean-field-friendly, but that estimate is **unreliable and cannot be counted as evidence**: the file's
own numbers give $\xi_0=0.161$ fm $<\,l_{\rm micro}=0.230$ fm, so the correlation length never exceeds
the microscopic scale — there is **no mean-field window** and the Ginzburg–Levanyuk long-wavelength
premise fails — while $u=\lambda_v/4\pi$ understates the true QCD-scale coupling. So the in-framework
diagnostic and external lattice QCD (critical point 3-D Ising, $N_f{=}2$ chiral O(4)) **agree**, not
disagree: the transition is **non-mean-field, $\beta\approx0.33$**. This is *not* borderline. (Recall
$\beta=1/2$ was itself only the Landau ansatz restated, above.) Hence the value of an actual RHIC BES-II
measurement to fix the exponent directly.

**(2) RHIC BES cumulant scaling** — `nvg_melting_beta_besii.py`. Within a universality class $\beta$
and the correlation-length exponent $\nu$ are co-fixed (mean-field $\nu=0.5$; 3-D Ising $\nu=0.630$),
and net-proton $\kappa\sigma^2=C_4/C_2\sim\xi^5$, $\xi\sim|\mu_B-\mu_c|^{-\nu}$. A scaling-shape fit to
published **STAR BES-I** $\kappa\sigma^2$ (0–5% central Au+Au; PRL 126, 092301, 2021) **cannot
distinguish** the classes: a flat no-critical-point null ($\kappa\sigma^2\equiv1$) already fits the nine
points at $\chi^2/{\rm dof}\approx0.27$, and both critical hypotheses fit "too good" at
$\chi^2/{\rm dof}\approx0.13$–$0.22$ (an over-flexible 2-parameter model absorbing one marginal point).
The $\Delta\chi^2\approx0.6$ leaning to Ising is $\approx0.8\sigma$ — **noise**. Moreover the fit
constrains only $\nu$; $\beta$ enters solely through the assumed universality class, so this is not a
direct $\beta$ measurement. **BES-II** (future) is the clean test; current data do not select an exponent.

**(3) Cyclic-entropy (Tolman) problem** — `nvg_bounce_schrodinger_bridge.py`. Posing the bounce as a
**Schrödinger bridge** (entropic optimal transport, Sinkhorn) between the pre- and post-bounce
mini-superspace marginals gives a **finite, positive, computable** per-cycle entropy production
($\Delta S\sim0.14$ nats per mode here), turning Tolman's postulated "$M\times2$" into a derived
transport entropy — no prior art was found for this framing. $\Delta S>0$ forbids eternal cyclicity
(a finite number of past cycles). This is a **proof-of-concept** of the framing, not a measurement:
$\Delta S$ formally *depends* on $\beta$ (via the marginal widths), but like the CMB cutoff it does not
provide an independent observational handle on the exponent. So of the "consequences," $\beta$
**propagates** into the bounce, the CMB cutoff, and the cyclic-entropy budget at the level of *theory*,
but only **heavy-ion** could actually measure it — the CMB does not constrain it (above) and the cyclic
entropy is not yet an observable. The honest reading is one exponent *appearing* in several places, not
one number simultaneously *measured* across them.

## A hypothesis, not a resolution: two condensates, not one

*If* "$W$" is not one field, the (already weak) CMB-vs-heavy-ion tension is a non-issue.
`nvg_two_condensate_resolution.py` **assesses this hypothesis** (it does not derive it). The maximum-mass
star has a central density $\sim5$–$6\,n_0$ (the $\varepsilon_c/M_N$ estimate of $\sim7.2\,n_0$
*overestimates* it, since $E/A\gg M_N$ at the core; $\varepsilon_c\approx1088$ MeV/fm³), where the
**deep vacuum** condensate $W_{\rm deep}=\sqrt{1-\rho/\rho_c^{\rm cosmo}}$ is only $\sim0.8\%$ melted. Note
the earlier "adds a constant, so $\beta_{\rm deep}$ doesn't enter" reasoning is **wrong**: $W$ linearises
to $1-\beta\,\rho/\rho_c$, whose slope is $\propto\beta$, so $\beta_{\rm deep}$ *formally does* enter — it
is simply **numerically negligible** (WEAK coupling, not decoupling), and only because the *assumed*
$\rho_c^{\rm cosmo}=7.09\times10^4$ MeV/fm³ sits $\sim65\times$ beyond the core. That $0.8\%$ is marginal
and EOS-dependent, and it **flips** to $\sim8\%$ if $\rho_c^{\rm cosmo}$ were $\sim10\times$ lower — so it
rests on a speculative CMB fit, not on NS physics.

So a **physically-plausible hypothesis** (not a derived result) is two condensates at two scales:

| condensate | $\rho_c$ | $\beta$ (assumed) | class | associated with |
|---|---|---|---|---|
| **deep vacuum** | $7.09\times10^4$ MeV/fm³ ($\sim470\,n_0$) | $0.5$ | mean-field | cosmological bounce ($\ell_c=3.42$; *not* a $\beta$-measurement) |
| **chiral / dense-matter** | few $\times n_0$ | $0.326$ | 3-D Ising | RHIC BES-II (heavy-ion) |

This is the physical *form* the earlier **two-scale $\rho_c$** ($\sim150\times$) result would take: the QCD
chiral condensate and the deep vacuum condensate as **different order parameters**, so no single $\beta$ is
over-constrained. But it is **model input, not a prediction**: the two $\beta$ values ($0.5$, $0.326$) are
the very inputs used to motivate the split (post-hoc accommodation), and the CMB does not independently
supply $\beta=0.5$ (it does not measure $\beta$ at all, above). A genuine test would need a *third*,
independent probe of one of the two exponents. Neutron stars sit below both critical points and pin
**neither** (matching the identifiability study).

**Can a heavy-NS radius feel the chiral $\beta$? No — it is a stiffness knob, not a $\beta$ handle** —
`nvg_melting_tail_sensitivity.py`. Replacing the melting law by a sharp chiral critical form fails (the RMF
self-calibration diverges or the star runs away): within this framework the CSS quark crossover at $2\,n_0$
($c_s^2\to1/3$) governs the high-density branch and **masks** any sharp chiral melting. A perturbative test
does give a numerically real signal — a $\pm9\%$ high-density bump moves $R_{2.0}$ by $\approx0.52$ km
($R_{1.4}$ by $0.36$, $M_{\max}$ by $0.05$), above TOV discretisation noise — **but it is not a
$\beta$ measurement**. Above the $2\,n_0$ transition the EOS is pure constant-sound-speed matter with *no*
melting law, so the bump sits inside the masked region; its only effect is a $\sim2.4\%$ shift of the CSS
transition anchor, i.e. a **generic high-density stiffness change** that co-moves $M_{\max}$, $R_{1.4}$,
$R_{2.0}$ together and is degenerate with $c_{s,q}^2$, the transition density, and the latent heat. (The
$\pm9\%$ symmetric Gaussian is also not a faithful proxy for the true one-sided, slope-diverging
$W_{0.326}-W_{0.5}$.) **Honest conclusion:** within this hybrid model the neutron-star tail is effectively
**blind** to $\beta$ — consistent with the identifiability result — and a heavy-NS radius is *not* shown to
be a handle on the chiral exponent. A real test would put the melting law where the star samples it (push
the CSS transition above the maximum central density, or apply $\beta$ to the hadronic branch below it) and
propagate the exact $\Delta W(\rho)$ — left as future work.

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
  in each: the CMB $\ell_c$ shift was originally the $\beta$-dependence under a radiation-bounce scaling,
  now **superseded** by the horizon-chain treatment (`nvg_melting_beta_horizon_chain.py`), which shows the
  shift is largely a point-proxy artefact and the CMB does not robustly constrain $\beta$; the cumulant
  fit uses published **STAR BES-I** central values (PRL 126, 092301, 2021) with a heuristic $\xi(\mu_B)$
  profile and **cannot distinguish** the classes ($\Delta\chi^2\approx0.6\approx0.8\sigma$; a flat null
  already fits) — the definitive selection is a future BES-II analysis; and the Schrödinger-bridge
  $\Delta S$ is a proof-of-concept of the (apparently novel) framing, not yet evaluated on the true
  mini-superspace action.
- The horizon-chain result rests on an **interpretive** point: whether the CMB low-$\ell$ cutoff is the
  Genesis instanton (conditionally β-independent) or the near-bounce particle horizon (mildly β-dependent).
  Reading (A) is the framework's committed mechanism, but even it only *removes a robust constraint*
  (the cutoff is pinned to $H_0$ by the $N_e$ calibration; the peak-curvature scale reintroduces $\beta$ at
  $\sim10\%$); the honest conclusion is that the earlier tension was **not a robust prediction** either way,
  not that $\ell_c$ is physically $\beta$-independent.
- The **two-condensate split is a hypothesis / model input**, not a derived resolution: the two $\beta$
  values are the inputs that motivate it (post-hoc), and no independent probe supplies either exponent. And
  the neutron-star "second handle" on $\beta$ does **not** survive scrutiny — the $\approx0.5$ km $R_{2.0}$
  signal is a generic high-density-stiffness response (degenerate with $c_{s,q}^2$/transition parameters),
  not a $\beta$ measurement; the NS tail is effectively **blind** to $\beta$.

## Bottom line

The $\sqrt{\;}$-law is **not arbitrary** — it is the mean-field extremum of NVG's own potential — but
$\beta=1/2$ is the **Landau ansatz restated** (it follows only from the assumed *linear* melting), and if
the $\rho_c$ transition is a genuine 3-D critical point the exponent is $\beta\approx0.326$. Crucially,
$\beta$ is **hard to measure**: neutron stars constrain $W(\rho)$ only up to $\sim2.5\,n_0$ and their tail
is effectively blind to $\beta$ (a generic stiffness knob, not a $\beta$ handle); the CMB low-$\ell$ cutoff
does not robustly constrain $\beta$ either. **Heavy-ion physics is the only realistic handle** — and even
there the existing STAR **BES-I** data cannot yet distinguish the classes; **BES-II** is the clean future
test. So $\beta$ is best understood as a **theoretical-consistency parameter**, sharply consequential in
principle but, on present evidence, measurable at most in one arena rather than jointly across three.
