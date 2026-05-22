"""
NVG dark-photon kinetic mixing: density-dependent realisation of the
condensate-photon coupling lambda required by V2 magnetars.

Premise
-------
The companion derivation `nvg_em_response_higher_order.py` showed that the
V2 magnetar amplification (Gamma_struct ~ 2-3) needs an EFT coupling
lambda ~ 0.9 at rho ~ 2 n_0, while standard QCD operators give at most
lambda_std ~ 4e-3 (200x shortfall). The cleanest beyond-minimal-QCD
realisation that closes this gap inside the NVG framework is a hidden U(1)
gauge field A'_mu kinetically mixed with the visible photon,
    L = -1/4 F^2 - 1/4 F'^2 - (eps/2) F F' - 1/2 m_A'^2 A'^2,
where the mixing parameter is *controlled by the NVG order parameter*
(equivalently, by the chiral condensate sigma(rho)):
    eps(rho) = eps_0 * (1 - sigma^2(rho) / sigma^2(0))
            =: eps_0 * S(rho),                       0 <= S(rho) <= 1.
At low density sigma -> sigma_0 so S -> 0: the mixing is *off* in stars,
labs, and the early universe. At chiral restoration in NS interiors S -> 1
and the mixing reaches its bare value eps_0, generating the magnetar
amplification through the photon self-energy shift
    delta Z = eps^2(rho) * f(m_A', plasma) approx eps^2(rho)
when m_A' >> plasma frequency of the NS interior.

This single density-suppression structure has two crucial consequences:

  (i) Every standard astrophysical / cosmological / laboratory bound on
      eps applies to environments with very small S(rho) and is therefore
      either trivially satisfied or shifted by orders of magnitude.
 (ii) Only one regime *can* be sensitive: dense proto-neutron-star matter
      in core-collapse supernovae (SN1987A). Even there, kinematic
      suppression for m_A' > T_core ~ 30 MeV controls the bound.

The script computes:

  (1) the chiral order parameter sigma(rho)/sigma_0 from the NJL gap
      equation (already validated in `nvg_em_response_derivation.py`),
  (2) the density-suppression factor S(rho) for a list of environments
      (vacuum lab, Sun, white dwarf, HB-star helium core, BBN plasma,
      SN1987A proto-NS, magnetar interior),
  (3) the effective in-medium mixing eps_eff(rho) = eps_0 * S(rho),
  (4) the comparison with published bounds on plain eps in each
      environment, declaring "allowed" / "borderline" / "excluded",
  (5) the value of eps_0 (and the corresponding A' mass scale) needed
      to match the V2 benchmark lambda ~ 0.9 at rho = 2 n_0.

Bounds used (representative, conservative)
------------------------------------------
- Sun (Redondo & Raffelt 2013): eps < 1e-14 for m_A' < few keV
- HB stars (An, Pospelov, Pradler 2013): eps < 1e-13 for m_A' < 30 keV
- BBN (Berger, Howe, Krnjaic 2016): eps < 1e-9 for various m_A' < few MeV
- Beam dumps (e.g. NA64, A1, BaBar): eps < 1e-3 - 1e-7 in vacuum,
  m_A' from MeV to GeV
- SN1987A (Chang, Essig, McDermott 2017): eps < 1e-9 for m_A' < 100 MeV;
  bound weakens *rapidly* (vanishes) for m_A' > 200 MeV due to kinematic
  suppression in T_core ~ 30 MeV plasma.

Run:
    python3.12 verification/nvg_dark_photon_kinetic_mixing.py
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nvg_em_response_derivation import (  # noqa: E402
    solve_gap, chiral_condensate, kF_from_nB,
    N0, M_Q_VACUUM, SIGMA_VACUUM,
)


# -------------------------------------------------------------------------
# Density-suppression factor S(rho) = 1 - (sigma(rho)/sigma_0)^2
# -------------------------------------------------------------------------
def chiral_ratio(nB_over_n0: float, T_GeV: float = 0.010) -> float:
    """Return sigma(rho)/sigma_0 from the NJL gap equation at given density
    and temperature. For nB < 0.5 n_0 returns ~1 (vacuum-like)."""
    if nB_over_n0 < 0.01:
        return 1.0
    nB = nB_over_n0 * N0
    kF = kF_from_nB(nB)
    M_iter = M_Q_VACUUM
    for _ in range(8):
        mu_q = math.sqrt(kF * kF + M_iter * M_iter)
        M_new = solve_gap(mu_q, T_GeV)
        if abs(M_new - M_iter) < 1e-5:
            M_iter = M_new
            break
        M_iter = 0.5 * (M_iter + M_new)
    return chiral_condensate(M_iter) / SIGMA_VACUUM

def S_factor(nB_over_n0: float, T_GeV: float = 0.010) -> float:
    """Density-suppression factor: 0 in vacuum-like medium, ~1 at chiral
    restoration."""
    r = chiral_ratio(nB_over_n0, T_GeV)
    return max(0.0, 1.0 - r * r)


# -------------------------------------------------------------------------
# Environment catalogue
# -------------------------------------------------------------------------
# rho_n0: baryon density in units of nuclear saturation n_0 = 0.16 fm^-3.
# T_GeV : characteristic temperature.
# m_max : maximum A' mass kinematically producible in this environment, GeV.
#         (Bounds below m_max apply; bounds above m_max are kinematically
#         shut off and trivially satisfied.)
# eps_bound_plain: published 95% CL bound on plain eps (no NVG suppression),
#         valid for m_A' < m_max. Conservative (strongest reported).
# -------------------------------------------------------------------------
@dataclass
class Env:
    name: str
    rho_n0: float
    T_GeV: float
    m_max_GeV: float
    eps_bound: float    # upper limit on PLAIN eps in this environment
    note: str

ENVIRONMENTS: list[Env] = [
    Env("vacuum lab (beam dump, BaBar)",  0.0,       1e-3,
        10.0,   1e-3, "Standard dark-photon search, m_A' up to ~10 GeV"),
    Env("Sun (Redondo+Raffelt 2013)",     1.3e-13,   1.3e-6,
        1.0e-5, 1e-14, "Solar interior, T ~ 1.3 keV"),
    Env("HB He core (An+Pospelov 2013)",  4.0e-11,   8.6e-6,
        3.0e-5, 1e-13, "Horizontal-branch helium burning, T ~ 8.6 keV"),
    Env("RG core (Vinyoles+ 2015)",       6.0e-10,   8.6e-6,
        3.0e-5, 3e-13, "Red giant tip, similar to HB"),
    Env("BBN plasma (Berger+ 2016)",      1.0e-30,   1.0e-3,
        3.0e-3, 1e-9,  "Big-bang nucleosynthesis, T ~ MeV"),
    Env("SN1987A proto-NS (Chang+ 2017)", 1.0,       0.030,
        0.10,   1e-9,  "Proto-NS core, T ~ 30 MeV; bound vanishes above ~200 MeV"),
    Env("Magnetar interior (NVG V2)",     2.0,       0.010,
        10.0,   1.0,   "Target environment, no current bound (no detection)"),
]


# -------------------------------------------------------------------------
# Connection: eps_0 vs benchmark lambda
# -------------------------------------------------------------------------
# In the limit m_A' >> all plasma scales of the NS interior
# (omega_p(NS) ~ 1-10 MeV << m_A' ~ GeV), the in-medium photon self-energy
# shift from kinetic mixing is, to leading order in eps,
#       delta Z_chi(rho) approx eps^2(rho) = eps_0^2 * S^2(rho).
# Identifying with the V2 EFT ansatz Z_chi = 1 - lambda * (1 - r^2), and
# noting that S(rho) = 1 - r^2(rho), we read off
#       lambda * S(rho) = eps_0^2 * S^2(rho)
#       => eps_0^2 = lambda / S(rho)        (evaluated at benchmark density)
# At rho = 2 n_0, S ~ 0.9, lambda ~ 0.9 => eps_0 ~ 1.
#
# The natural A' mass scale in NVG is set by the chiral symmetry breaking
# scale, m_A' ~ Lambda_chi = 4 pi f_pi ~ 1.16 GeV. We use this as the
# benchmark; the bound analysis is essentially the same for any m_A' in
# the GeV range, because all standard bounds shut off above m_A' ~ 200 MeV.
# -------------------------------------------------------------------------
M_A_PRIME_BENCH = 1.16    # GeV, ~ Lambda_chi
LAMBDA_V2_BENCH = 0.9     # required EFT coupling at 2 n_0

def eps0_from_lambda(lam: float, nB_over_n0: float = 2.0) -> float:
    S = S_factor(nB_over_n0)
    if S <= 0.0:
        return float("inf")
    return math.sqrt(lam / S)


# -------------------------------------------------------------------------
# Compatibility check per environment
# -------------------------------------------------------------------------
@dataclass
class EnvCheck:
    env: Env
    S: float
    eps_eff: float
    kinematic_open: bool
    status: str
    margin: float  # eps_bound / eps_eff if kinematically open, else inf

def check_environment(env: Env, eps0: float,
                      m_Aprime: float = M_A_PRIME_BENCH) -> EnvCheck:
    S = S_factor(env.rho_n0, env.T_GeV)
    eps_eff = eps0 * S
    kinematic_open = (m_Aprime <= env.m_max_GeV)
    if not kinematic_open:
        return EnvCheck(env, S, eps_eff, False, "KINEMATIC-OFF",
                        float("inf"))
    if eps_eff <= 0.0:
        return EnvCheck(env, S, eps_eff, True, "ALLOWED (S=0)",
                        float("inf"))
    margin = env.eps_bound / eps_eff
    if margin >= 10.0:
        status = "ALLOWED"
    elif margin >= 1.0:
        status = "BORDERLINE"
    else:
        status = "EXCLUDED"
    return EnvCheck(env, S, eps_eff, True, status, margin)


# -------------------------------------------------------------------------
# Report
# -------------------------------------------------------------------------
def report() -> str:
    L: list[str] = []
    L.append("NVG dark-photon kinetic mixing with density-dependent eps(rho)")
    L.append("=" * 78)
    L.append("")
    L.append("Model:")
    L.append("  L = -1/4 F^2 - 1/4 F'^2 - (eps/2) F F' - 1/2 m_A'^2 A'^2,")
    L.append("  eps(rho) = eps_0 * S(rho),   S(rho) = 1 - (sigma(rho)/sigma_0)^2.")
    L.append("")
    L.append(f"  Benchmark A' mass:   m_A' = {M_A_PRIME_BENCH:.2f} GeV "
             "(~ Lambda_chi)")
    L.append(f"  Target V2 coupling:  lambda = {LAMBDA_V2_BENCH:.2f} at 2 n_0")
    L.append("")

    eps0 = eps0_from_lambda(LAMBDA_V2_BENCH, 2.0)
    L.append(f"  => required bare mixing  eps_0 = {eps0:.3f}")
    S_bench = S_factor(2.0)
    L.append(f"     (at 2 n_0: S = {S_bench:.3f}, "
             f"eps_eff = {eps0*S_bench:.3f})")
    L.append("")
    L.append("-" * 78)
    L.append("Environment-by-environment compatibility check")
    L.append("-" * 78)
    L.append(f"  {'environment':<36} {'rho/n0':>9} {'S(rho)':>9} "
             f"{'eps_eff':>10} {'bound':>9} {'status':>12}")
    checks = [check_environment(e, eps0) for e in ENVIRONMENTS]
    for c in checks:
        rho_str = f"{c.env.rho_n0:.1e}" if c.env.rho_n0 < 0.1 \
                  else f"{c.env.rho_n0:9.3f}"
        eps_str = f"{c.eps_eff:.2e}" if c.kinematic_open else "    n/a"
        L.append(f"  {c.env.name:<36} {rho_str:>9} {c.S:9.2e} "
                 f"{eps_str:>10} {c.env.eps_bound:9.0e} {c.status:>12}")
    L.append("")
    L.append("Footnotes per environment:")
    for c in checks:
        L.append(f"  - {c.env.name}: {c.env.note}")
        if not c.kinematic_open:
            L.append(f"      (m_A' = {M_A_PRIME_BENCH:.2f} GeV exceeds "
                     f"environment kinematic max {c.env.m_max_GeV:.2e} GeV)")
    L.append("")
    L.append("-" * 78)
    L.append("Summary")
    L.append("-" * 78)
    n_excluded   = sum(1 for c in checks if c.status == "EXCLUDED")
    n_borderline = sum(1 for c in checks if c.status == "BORDERLINE")
    n_kinoff     = sum(1 for c in checks if c.status == "KINEMATIC-OFF")
    n_allowed    = sum(1 for c in checks
                       if c.status in ("ALLOWED", "ALLOWED (S=0)"))
    L.append(f"  Kinematically-off (no bound at m_A' = {M_A_PRIME_BENCH:.2f} GeV): "
             f"{n_kinoff}")
    L.append(f"  Allowed (density-suppressed): {n_allowed}")
    L.append(f"  Borderline:                   {n_borderline}")
    L.append(f"  Excluded:                     {n_excluded}")
    L.append("")
    if n_excluded == 0:
        L.append("  >>> NVG dark-photon ansatz (eps_0 ~ 1, m_A' ~ 1.16 GeV)")
        L.append("      is consistent with ALL standard astrophysical, ")
        L.append("      cosmological, and laboratory bounds on kinetic ")
        L.append("      mixing, by virtue of TWO independent suppressions:")
        L.append("        (i) density activation S(rho) ~ 0 in stars / labs;")
        L.append("       (ii) kinematic blockade for m_A' > T_environment.")
        L.append("")
        L.append("      Only environment where the mechanism can operate")
        L.append("      is a dense NS / proto-NS core. Falsification therefore")
        L.append("      requires either")
        L.append("        (a) a NS merger EM counterpart with anomalous")
        L.append("            magnetic-field amplification signature, or")
        L.append("        (b) a heavy dark-photon search in dense-matter")
        L.append("            laboratory analogues (heavy-ion collisions")
        L.append("            at FAIR/CBM where mu_B reaches ~ n_0).")
    else:
        L.append(f"  >>> ANSATZ EXCLUDED in {n_excluded} environment(s); ")
        L.append("      see table above. Need to revisit m_A' or eps_0.")
    L.append("")
    L.append("-" * 78)
    L.append("Predictions distinguishing density-suppressed mixing from")
    L.append("standard constant-eps dark photon")
    L.append("-" * 78)
    L.append("  1. No signal in any vacuum-based laboratory search at any eps")
    L.append("     (beam dump, BaBar, NA64, LHCb dark-photon search, haloscope).")
    L.append("  2. No signal in any stellar cooling channel below n_B ~ 0.5 n_0.")
    L.append("  3. Anomalous magnetar B-field amplification correlated with")
    L.append("     central density, predicted by S(rho).")
    L.append("  4. Heavy-ion collisions at sqrt(s_NN) ~ few GeV (FAIR/NICA)")
    L.append("     where mu_B reaches ~1 GeV should show enhanced dilepton")
    L.append("     production from A' decays in the brief dense phase.")
    L.append("  5. Binary neutron star inspirals (LIGO/Virgo/ET) should show")
    L.append("     no EM precursor signal from kinetic-mixing channels")
    L.append("     before merger (densities still sub-nuclear), but a strong")
    L.append("     amplification within ~1 ms of contact.")
    L.append("")
    L.append("=" * 78)
    return "\n".join(L)


if __name__ == "__main__":
    print(report())
