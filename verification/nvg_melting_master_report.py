#!/usr/bin/env python3
"""
Master report for the vacuum-melting-exponent (beta) investigation.

Runs every melting-law verification script, confirms it still executes, extracts
its headline result, and assembles a single consolidated NVG_MELTING_REPORT.md
with the observable -> beta -> status map. Run:

    .venv/bin/python verification/nvg_melting_master_report.py

Writes NVG_MELTING_REPORT.md at the repository root and prints a console summary.
"""
import os, re, sys, subprocess, time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PY   = os.path.join(ROOT, ".venv", "bin", "python")
if not os.path.exists(PY): PY = sys.executable

# (script, one-line question, verified finding, headline-regex-or-None, slow?)
SCRIPTS = [
    ("nvg_melting_identifiability.py",
     "Can a neutron star see the *cosmological* melting law W=sqrt(1-rho/rho_c^cosmo)?",
     "No. The cosmological law is inert at NS densities (M_max=2.02, unchanged); the NS "
     "only fixes an EFFECTIVE rho_c ~ 462 MeV/fm^3 and the functional form is degenerate.",
     None, True),
    ("nvg_melting_freeform_beta.py",
     "How deep in density does a neutron star actually constrain the melting curve W(rho)?",
     "Only up to ~2.5 n0. Above that the deep core is 30-50% degenerate: NS structure does "
     "not pin W(rho) in the region where a critical exponent would live.",
     None, True),
    ("nvg_melting_exponent.py",
     "What melting exponent beta does the framework potential V=(lam/4)(|Phi|^2-v0^2)^2 give?",
     "The sqrt-law gives beta=1/2 -- but this is the Landau ansatz RESTATED (it follows only from the "
     "assumed LINEAR melting v^2 ~ (1-rho/rho_c); any power p gives beta=p/2). A genuine 3D critical point "
     "would give beta=0.326. The W(rho) profiles diverge near rho_c (3.3x at r=0.999).",
     None, False),
    ("nvg_melting_ginzburg.py",
     "Is the mean-field sqrt-law self-consistent (Ginzburg criterion) with the framework's own lam_v?",
     "NOT borderline. The one mean-field-favouring leg (naive t_G ~ 0.007) is unreliable: xi_0=0.16 fm < "
     "l_micro=0.23 fm means NO mean-field window (Ginzburg-Levanyuk premise fails), and u=lam_v/4pi "
     "understates the QCD coupling. In-framework diagnostic AND lattice-QCD AGREE: non-mean-field, beta~0.33.",
     r"t_G ~ u\^2 = ([\d.]+)", False),
    ("nvg_melting_beta_besii.py",
     "Do STAR BES-I net-proton cumulants (kappa*sigma^2) select a melting exponent?",
     "No. Data are STAR BES-I (PRL 126, 092301, 2021), not BES-II. A flat no-critical-point null already "
     "fits at chi^2/dof~0.27; both critical hypotheses fit 'too good' (~0.13-0.22). Delta chi^2~0.6 ~0.8 "
     "sigma = NOISE, CANNOT distinguish classes; only nu is constrained, beta enters by assumption. BES-II is future.",
     None, False),
    ("nvg_melting_beta_bounce.py",
     "Does beta shift the CMB low-ell cutoff (crude comoving-Hubble-radius proxy)?",
     "CRUDE proxy: evaluating R_H at an arbitrary onset r*=0.9..0.99 gives ell_c 3.42 -> 5.1..7.6. "
     "Superseded by the horizon-chain refinement below.",
     None, False),
    ("nvg_melting_beta_cmb_chi2.py",
     "Real CAMB low-ell chi^2 of the cutoff shift -- is it statistically significant?",
     "No. At ell=2-29 cosmic variance dominates: beta=0.5 vs no-cutoff is Delta chi^2=1.01 (~1.0 sigma), "
     "vs beta=0.326 is 1.9 (~1.4 sigma) -- both sub-2 sigma, cut further by f_sky~0.86. The CMB is CONSISTENT "
     "with beta=0.5, no-cutoff, AND beta=0.326 alike; the +10.9 is the uncertain k_c(beta) mapping (spans "
     "chi^2 27.7-36.6), not a physical disfavouring.",
     None, True),
    ("nvg_melting_beta_horizon_chain.py",
     "Refined ell_c(beta) via the framework's ACTUAL Genesis horizon chain + real CAMB.",
     "REFINEMENT: the CMB tension is NOT robust. On the committed instanton mechanism (r_c=c/H_c uses "
     "rho_c only; N_e fixed by local H_0) ell_c is beta-INDEPENDENT (no tension). Under the alternative "
     "bounce-horizon reading, the integrated k_c(0.326)/k_c(0.5) ~ 1.4x (r*=0.5) -- a real but "
     "r*-dependent shift; integration kills only the point-proxy's ell_c~7.6 (+10.9 chi^2) runaway.",
     None, True),
    ("nvg_bounce_schrodinger_bridge.py",
     "Per-cycle entropy production of the bounce (Schrodinger-bridge / entropic OT).",
     "Finite, positive DeltaS ~ 0.14 nats per cycle -> a finite number of past cycles (quantifies the "
     "Tolman obstruction). Proof-of-concept: DeltaS formally depends on beta but is NOT an observational "
     "handle on it (like the CMB cutoff). beta APPEARS here; it is not MEASURED here.",
     None, False),
    ("nvg_two_condensate_resolution.py",
     "Would two condensates remove the (weak) mean-field-vs-Ising tension?",
     "HYPOTHESIS, not a derived resolution. The deep vacuum is only ~0.8% melted at the NS core (~5-6 n0, "
     "not 7.2), but 'adds a constant' was WRONG: W linearizes to slope ~ beta, so beta_deep formally enters "
     "-- it is just WEAK coupling (negligible only via the assumed 65x scale separation, flips if rho_c^cosmo "
     "is 10x lower). The two beta values are the INPUTS motivating the split (post-hoc), not a prediction.",
     None, True),
    ("nvg_melting_tail_sensitivity.py",
     "Does the neutron-star tail feel beta=0.326 vs 0.5, or is it blind?",
     "BLIND. Above the 2 n0 CSS transition the EOS is pure constant-sound-speed matter with no melting law; "
     "the ~0.52 km |Delta R_2.0| is REAL but is a generic high-density STIFFNESS response (a ~2.4% CSS-anchor "
     "shift, degenerate with cs2_q / transition params, co-moving M_max/R_1.4/R_2.0), NOT a beta measurement. "
     "A heavy-NS radius is NOT shown to be a handle on the chiral exponent (consistent with identifiability).",
     None, True),
]

SUMMARY_TABLE = [
    # observable / probe, what it constrains, value, status
    ("Framework potential V", "the melting exponent itself", "sqrt-law => beta=1/2", "Landau ansatz restated"),
    ("Ginzburg criterion",    "is mean-field self-consistent", "no MF window (xi0<l_micro)", "non-mean-field"),
    ("Lattice-QCD universality", "class of the QCD critical point", "3D-Ising => beta=0.326 (IF CP exists)", "external input"),
    ("RHIC BES-I kappa*sigma^2 (data)", "chiral melting exponent", "flat null fits; dchi2~0.8 sigma", "cannot distinguish"),
    ("RHIC BES-II (future)", "chiral melting exponent (direct)", "the clean test", "PENDING -- sole real handle"),
    ("Neutron-star tail R_2.0", "chiral melting exponent?", "0.5 km = generic stiffness, not beta", "BLIND (degenerate)"),
    ("Planck CMB low-ell",    "deep-vacuum bounce exponent", "sub-2 sigma; ell_c pinned to H_0", "does NOT constrain beta"),
    ("Cyclic entropy (bridge)", "per-cycle DeltaS(beta)", "~0.14 nats; not an observable", "proof-of-concept"),
    ("NS identifiability",    "can NS pin any exponent?", "no (degenerate above ~2.5 n0)", "null result"),
]

def run(script, timeout, regex):
    path = os.path.join(HERE, script)
    t0 = time.time()
    try:
        p = subprocess.run([PY, path], capture_output=True, text=True, timeout=timeout, cwd=HERE)
        dt = time.time() - t0
        ok = (p.returncode == 0)
        head = ""
        if regex:
            m = re.search(regex, p.stdout)
            if m: head = m.group(0)
        return ok, dt, head, p.stdout, p.stderr
    except subprocess.TimeoutExpired:
        return None, time.time()-t0, "", "", "timeout"

def main():
    print("="*74)
    print("MASTER REPORT: vacuum melting exponent (beta) investigation")
    print("="*74)
    results = []
    for script, q, finding, regex, slow in SCRIPTS:
        to = 900 if slow else 180
        print(f"\n-> {script}  (timeout {to}s) ...", flush=True)
        ok, dt, head, out, err = run(script, to, regex)
        mark = "OK " if ok else ("TIMEOUT" if ok is None else "FAIL")
        extra = f"  [{head}]" if head else ""
        print(f"   {mark}  {dt:6.1f}s{extra}")
        if ok is False and err:
            print("   stderr:", err.strip().splitlines()[-1] if err.strip() else "")
        results.append((script, q, finding, ok, dt))

    # ---- write consolidated markdown ----
    lines = []
    A = lines.append
    A("# NVG vacuum-melting-exponent (beta) — master report\n")
    A("**Thesis (as corrected by adversarial review).** The order-parameter exponent *beta* of the "
      "vacuum melting law *W(rho)=(1-rho/rho_c)^beta* *appears* in three arenas — the QCD critical point "
      "(RHIC), neutron-star structure, and the cosmological bounce (CMB low-ell) — but is **measurable in "
      "at most one of them**. beta=1/2 is the Landau ansatz restated; if the rho_c transition is a genuine "
      "critical point the exponent is beta≈0.326. Neutron stars are **blind** to beta and the CMB does "
      "**not** constrain it, leaving **heavy-ion (BES-II) as the sole realistic handle** — and current "
      "BES-I data cannot yet distinguish the classes. So beta is a **theoretical-consistency parameter**, "
      "not a number jointly measured across three probes. This report runs every verification script, "
      "confirms it executes, and consolidates the observable → beta → status map.\n")
    A("> Reproduce: `.venv/bin/python verification/nvg_melting_master_report.py` "
      "(regenerates this file).\n")

    A("\n## Observable → beta → status\n")
    A("| Probe | Constrains | Value | Status |")
    A("|---|---|---|---|")
    for probe, con, val, st in SUMMARY_TABLE:
        A(f"| {probe} | {con} | {val} | {st} |")

    A("\n## Key result: the CMB does not constrain beta\n")
    A("The crude estimate (`nvg_melting_beta_bounce.py`) evaluated the comoving Hubble radius "
      "at an arbitrary melting-onset density *r\\*=0.9..0.99*, producing a large ell_c "
      "3.42→5–7.6 swing and an apparent Planck preference for beta=0.5. Both the horizon-chain "
      "refinement (`nvg_melting_beta_horizon_chain.py`) and the raw statistics show this is **not a real "
      "constraint**:\n")
    A("- **(A) Committed mechanism (Genesis instanton).** The cutoff *r_c=c/H_c* uses rho_c alone, so "
      "read this way beta does not enter — but only **conditionally**: (i) the *unsuppressed* de Sitter "
      "scale must be used, not the bounce peak-curvature H_max (which reintroduces beta at ~10%), and "
      "(ii) since N_e=ln(R_H0/r_c), k_c=1/R_H0 is **pinned to today's Hubble horizon by calibration**, so "
      "ell_c=3.42 is a near-tautology, not a beta-discriminating prediction. Reading (A) *removes a robust "
      "constraint*; it does not prove physical beta-independence.\n")
    A("- **(B) Alternative (near-bounce particle horizon).** The *integrated* horizon still "
      "gives k_c(0.326)/k_c(0.5) ≈ 1.4× (r\\*=0.5) — a real but **r\\*-dependent** shift.\n")
    A("- **Statistics.** Independently of the mechanism, at ell=2-29 cosmic variance dominates: the "
      "χ² gaps are Δχ²=1.01 (~1.0σ, beta=0.5 vs no-cutoff) and 1.9 (~1.4σ, vs beta=0.326) — **both "
      "sub-2σ**. The CMB is *consistent with* beta=0.5, no-cutoff, AND beta=0.326 alike.\n")
    A("\n**Net:** Planck does **not** constrain beta — neither through the (conditional) instanton reading "
      "nor through raw significance. The exponent is essentially a heavy-ion observable; neutron stars are "
      "blind to it; and the two-condensate split is a *hypothesis motivated by the two-scale rho_c*, not "
      "anchored by any CMB beta-measurement.\n")

    A("\n## Per-script findings\n")
    for script, q, finding, ok, dt in results:
        badge = "✓" if ok else ("⏱ timeout" if ok is None else "✗ failed")
        A(f"### `{script}` — {badge} ({dt:.0f}s)")
        A(f"*Question.* {q}\n")
        A(f"*Finding.* {finding}\n")

    A("\n## Bottom line\n")
    A("- **beta is consequential in principle but hard to measure.** beta=1/2 is the Landau ansatz "
      "restated; a genuine 3D critical point gives beta≈0.326.\n")
    A("- **Heavy-ion is the only realistic handle** — but existing STAR **BES-I** data cannot yet "
      "distinguish the classes (flat null already fits; Δχ²≈0.6≈0.8σ). **BES-II** is the clean future test.\n")
    A("- **Neutron stars are blind to beta**: they constrain W(rho) only to ~2.5 n0, and the ~0.5 km "
      "R_2.0 'signal' is a generic high-density-stiffness response degenerate with cs2_q/transition "
      "parameters, not a beta measurement.\n")
    A("- **The CMB does not constrain beta** — the low-ell χ² gaps are sub-2σ (cosmic-variance-limited) "
      "and, under the committed instanton mechanism, ell_c is pinned to H_0 by calibration.\n")
    A("- The **two-condensate** picture (deep vacuum beta=0.5 / chiral beta=0.326) is a *hypothesis / "
      "model input* motivated by the two-scale rho_c — the two beta values are its inputs (post-hoc), not "
      "a derived or CMB-anchored result.\n")

    n_ok = sum(1 for *_, ok, _ in results if ok)
    A(f"\n_{n_ok}/{len(results)} scripts executed cleanly in this run._\n")

    out_path = os.path.join(ROOT, "NVG_MELTING_REPORT.md")
    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print("\n" + "="*74)
    print(f"WROTE {out_path}   ({n_ok}/{len(results)} scripts OK)")
    print("="*74)

if __name__ == "__main__":
    main()
