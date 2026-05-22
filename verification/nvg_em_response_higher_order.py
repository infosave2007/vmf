"""
NVG EM-response: higher-order EFT corrections and standard-QCD upper bound on
the condensate-photon coupling lambda.

Purpose
-------
The first companion derivation (`nvg_em_response_derivation.py`) showed that
to reproduce the V2 magnetar amplification Gamma_struct ~ 2.2 - 3.3 one needs
a non-perturbative condensate-photon coupling lambda ~ 0.9 at rho ~ 2 n_0,
under the leading-order EFT ansatz
        Z_chi(sigma) = 1 - lambda * (1 - sigma^2(rho)/sigma^2(0)).
That note left two questions explicitly open:

  (1) what does standard QCD actually predict for lambda?
  (2) how stable is the leading-order ansatz under higher-order EFT terms?

This script collects all the *standard* channels that contribute to the
in-medium photon kinetic term and aggregates them into a textbook upper bound
on lambda. It also evaluates next-to-leading-order corrections to the linear
ansatz. Every channel is computed from published constants; no NVG-specific
physics is used here. This is a deliberately conservative reality check on
the V2 ansatz.

Channels included
-----------------
(A) One-loop quark Z_QED with NJL constituent mass M_q(rho)
    (already computed in the companion script; reproduced here for the sum).
(B) Charged-pion one-loop contribution to Z, using the in-medium pion mass
    and decay constant from chiral perturbation theory.
(C) Vector-meson-dominance (VMD) contribution: photon-rho mixing shifts the
    static photon kinetic term by m_rho^2 / (g_rho^2 Q_typ^2). For a strictly
    static B field this vanishes (Q->0); we evaluate it at the magnetar
    coherence scale Q_typ ~ 1 / R_NS to get the maximum effect.
(D) QCD magnetic susceptibility of the chiral condensate
    (Belyaev-Kogan 1984; Vainshtein 2003; lattice updates Bali et al 2012).
    This is the most direct first-principles operator that couples
    <q-bar q> to F^2 at the chiral scale.

The aggregate is
        lambda_std = sum of (A)..(D) channel contributions to delta Z at full
        chiral restoration (sigma -> 0), reported with conservative error.

Higher-order EFT
----------------
We extend the linear ansatz to
        Z_chi(sigma) = 1 - lambda * (1 - r^2) - mu * (1 - r^2)^2,
        r = sigma(rho)/sigma(0),
and report how mu shifts the required lambda at the benchmark density.

Output
------
A printable comparison of lambda_required (from V2 benchmark Gamma=2.2-3.3)
versus lambda_std (from standard QCD), per density bin, plus the next-order
correction. The headline conclusion is computed and printed at the end.

Run:
    python3.12 verification/nvg_em_response_higher_order.py
"""
from __future__ import annotations

import math
from dataclasses import dataclass

# Reuse the NJL machinery from the first companion script.
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nvg_em_response_derivation import (  # noqa: E402
    LAMBDA, ALPHA, N_C, SUM_Q2, M_Q_VACUUM, SIGMA_VACUUM,
    solve_gap, chiral_condensate, kF_from_nB, N0,
    Z_QED, gamma_QED,
)


# -------------------------------------------------------------------------
# Standard hadronic / chiral constants (PDG 2024 unless noted)
# -------------------------------------------------------------------------
M_PI       = 0.1396    # GeV, charged pion mass in vacuum
F_PI       = 0.0922    # GeV, pion decay constant in vacuum
M_RHO      = 0.7754    # GeV, rho(770) mass
G_RHO      = 5.0       # rho-photon coupling: m_rho^2 / g_rho ~ 0.12 GeV^2
M_NS_R     = 12.0      # km, neutron-star radius (typical coherence scale)
HBARC_GEV_KM = 1.973e-19  # GeV km (= 0.1973 fm * 1e-18 km/fm)

# Magnetic susceptibility of the QCD vacuum
# chi * <q-bar q> defined via <q-bar sigma_{mu nu} q>_F = chi * <q-bar q> * e_q * F_{mu nu}.
# Belyaev-Kogan 1984: chi ~ -3.15 GeV^-2.
# Lattice update (Bali, Bruckmann, Endrodi et al, JHEP 2012): chi ~ -2.08 GeV^-2.
# We use the lattice value as our best estimate, the SVZ value as upper bound.
CHI_QCD_BEST = -2.08    # GeV^-2 (Bali et al 2012 lattice)
CHI_QCD_HIGH = -3.15    # GeV^-2 (Belyaev-Kogan SVZ)

# Average current quark mass (u + d)
M_Q_AVG = 0.0035     # GeV, PDG average


# -------------------------------------------------------------------------
# Channel (B): charged-pion one-loop contribution to Z(photon).
#
# In scalar QED with charged scalar mass m, the photon kinetic-term
# renormalisation at zero external momentum is
#     delta Z_pi = - (e^2 / 48 pi^2) * ln(Lambda_UV^2 / m_pi^2)
#                = - (alpha / 12 pi) * ln(Lambda_UV^2 / m_pi^2)
# (single-charged-particle loop, Q_pi^2 = 1).
#
# In medium, the pion mass shifts via the Gell-Mann-Oakes-Renner relation:
#     m_pi^2(rho) = m_pi^2(0) * <q-bar q>(rho)/<q-bar q>(0)  * (f_pi(0)/f_pi(rho))^2
# To leading order f_pi^2(rho)/f_pi^2(0) ~ <q-bar q>(rho)/<q-bar q>(0)
# (Brown-Rho-like), so m_pi^2(rho) ~ m_pi^2(0) and the in-medium pion-loop
# shift is small.
#
# The contribution of in-medium chiral restoration through this channel:
#     delta Z_pi(rho) - delta Z_pi(0)
#         = - (alpha / 12 pi) * [ln(Lambda^2/m_pi^2(rho)) - ln(Lambda^2/m_pi^2(0))]
#         = + (alpha / 12 pi) * ln(m_pi^2(rho) / m_pi^2(0))
# which is approximately zero in this scaling. We compute it explicitly using
# the NJL condensate ratio r(rho) and assume f_pi^2 \propto |<q-bar q>|.
# -------------------------------------------------------------------------
def m_pi_in_medium(r: float) -> float:
    """GOR + Brown-Rho: m_pi^2(rho) ~ m_pi^2(0). Use a mild density correction
    consistent with chiral perturbation expectations."""
    # Conservative: m_pi(rho) = m_pi(0) * r^{-1/4} (gives mild increase)
    if r <= 0.01:
        return M_PI * (0.01) ** (-0.25)
    return M_PI * r ** (-0.25)

def delta_Z_pion(r: float) -> float:
    """In-medium shift of pion-loop contribution to Z, relative to vacuum."""
    m_pi_rho = m_pi_in_medium(r)
    return (ALPHA / (12.0 * math.pi)) * math.log((m_pi_rho / M_PI) ** 2)


# -------------------------------------------------------------------------
# Channel (C): vector-meson dominance contribution.
#
# The VMD Lagrangian rho - gamma mixing produces a shift in the photon
# propagator at non-zero Q:
#     Z_VMD(Q^2) - 1 = - (m_rho^2 / g_rho^2) * 1 / (Q^2 + m_rho^2 - i m_rho Gamma)
# Evaluated at the magnetar coherence scale Q ~ hbar/(R_NS) ~ 1.6e-20 GeV,
# this is utterly negligible compared to m_rho. For a *static* uniform B
# field the contribution vanishes identically (Q -> 0).
#
# In medium, m_rho may shift (Brown-Rho scaling: m_rho/m_rho(0) ~ r^{1/3}).
# Even at full chiral restoration the maximum VMD contribution at Q ~ 1/R_NS
# is < 10^-40. We include it for completeness but it never matters.
# -------------------------------------------------------------------------
def delta_Z_VMD(r: float) -> float:
    Q_static = HBARC_GEV_KM / M_NS_R
    m_rho_rho = M_RHO * (max(r, 0.01)) ** (1.0 / 3.0)
    Q2 = Q_static ** 2
    # Vacuum subtraction
    contrib_med  = (m_rho_rho ** 2 / G_RHO ** 2) / (Q2 + m_rho_rho ** 2)
    contrib_vac  = (M_RHO     ** 2 / G_RHO ** 2) / (Q2 + M_RHO     ** 2)
    return -(contrib_med - contrib_vac)


# -------------------------------------------------------------------------
# Channel (D): QCD magnetic susceptibility of the chiral condensate.
#
# In an external magnetic field B, the QCD vacuum acquires a quark tensor
# condensate:
#     <q-bar sigma_{mu nu} q>_F = chi * <q-bar q> * e_q * F_{mu nu}.
# The corresponding contribution to the vacuum energy quadratic in B is
#     Delta E = - (chi * m_q * <q-bar q> * (sum_q e_q^2) / 2) * B^2.
# Reading off the coefficient of -B^2/2 = -F^2/4 (in Heaviside-Lorentz units),
# the photon kinetic-term renormalisation from this operator is
#     delta Z_chi(sigma)  =  -chi * m_q * <q-bar q>(rho) * (sum_q Q_q^2)
#                             * (something with units to get GeV^0).
#
# Dimensions: chi has GeV^-2, m_q has GeV, <q-bar q> has GeV^3 -> product GeV^2.
# To get dimensionless delta Z one divides by the chiral scale squared
# Lambda_chi^2 = (4 pi f_pi)^2 ~ (1.16 GeV)^2 ~ 1.34 GeV^2.
#
# Therefore
#     delta Z_chi(rho) - delta Z_chi(0)
#         = - (chi * m_q / Lambda_chi^2) * (sum_q Q_q^2)
#           * (<q-bar q>(rho) - <q-bar q>(0))
#         = + |chi| * m_q * (sum_q Q_q^2) * |<q-bar q>(0)| * (1 - r) / Lambda_chi^2.
#
# This is the chiral-perturbation/SVZ scale for the condensate-photon coupling.
# -------------------------------------------------------------------------
LAMBDA_CHI = 4.0 * math.pi * F_PI    # GeV
LAMBDA_CHI2 = LAMBDA_CHI ** 2

# sum_q Q_q^2 for u, d (Q_u = 2/3, Q_d = -1/3) -- *without* color, since the
# operator already involves <q-bar q> which is a color singlet.
SUM_Q2_FLAVOUR = (2.0/3.0)**2 + (1.0/3.0)**2   # = 5/9

def delta_Z_chi_susc(r: float, chi: float = CHI_QCD_BEST) -> float:
    cond_vac_abs = abs(SIGMA_VACUUM)          # GeV^3
    factor = abs(chi) * M_Q_AVG * SUM_Q2_FLAVOUR * cond_vac_abs / LAMBDA_CHI2
    return factor * (1.0 - r)


# -------------------------------------------------------------------------
# Aggregate: total standard-QCD shift in Z as function of density (via r).
# -------------------------------------------------------------------------
def Z_std_total(r: float, M_q: float, chi: float = CHI_QCD_BEST) -> float:
    Z = Z_QED(M_q) - Z_QED(M_Q_VACUUM)        # (A) shift
    Z += delta_Z_pion(r)                       # (B)
    Z += delta_Z_VMD(r)                        # (C)
    Z -= delta_Z_chi_susc(r, chi)              # (D)
    return 1.0 + Z

def lambda_std(r: float, M_q: float, chi: float = CHI_QCD_BEST) -> float:
    """Effective lambda from standard-QCD channels under the same ansatz
       Z = 1 - lambda*(1-r^2). Invert for lambda."""
    Z = Z_std_total(r, M_q, chi)
    if abs(1.0 - r * r) < 1e-6:
        return 0.0
    return (1.0 - Z) / (1.0 - r * r)


# -------------------------------------------------------------------------
# Higher-order EFT: Z = 1 - lambda*(1-r^2) - mu*(1-r^2)^2.
# For a given target Gamma, the (lambda, mu) constraint is one equation.
# We report by how much the required lambda shifts if |mu| = 0.5 * |lambda|
# (i.e. NLO at the same order as LO, conservative).
# -------------------------------------------------------------------------
def required_lambda_NLO(r: float, gamma_target: float, mu_over_lambda: float) -> float:
    """Solve 1 - lambda*(1-r^2) - (mu_over_lambda * lambda)*(1-r^2)^2
            = 1/gamma_target^2  for lambda."""
    s = 1.0 - r * r
    rhs = 1.0 - 1.0 / gamma_target ** 2
    denom = s + mu_over_lambda * s * s
    if denom == 0.0:
        return float("inf")
    return rhs / denom


# -------------------------------------------------------------------------
# Driver
# -------------------------------------------------------------------------
@dataclass
class Row:
    nB: float
    r: float
    M_q: float
    lam_req_22: float
    lam_req_33: float
    lam_std_best: float
    lam_std_high: float
    lam_NLO_22: float
    lam_NLO_33: float

def driver() -> tuple[list[Row], dict]:
    rows: list[Row] = []
    densities = [1.0 * N0, 1.5 * N0, 2.0 * N0, 2.5 * N0, 3.0 * N0, 4.0 * N0, 5.0 * N0]
    for nB in densities:
        kF = kF_from_nB(nB)
        M_iter = M_Q_VACUUM
        for _ in range(6):
            mu_q = math.sqrt(kF * kF + M_iter * M_iter)
            M_new = solve_gap(mu_q, 0.010)
            if abs(M_new - M_iter) < 1e-5:
                M_iter = M_new
                break
            M_iter = 0.5 * M_iter + 0.5 * M_new
        M_q = M_iter
        sigma = chiral_condensate(M_q)
        r = sigma / SIGMA_VACUUM if SIGMA_VACUUM != 0.0 else 0.0
        if abs(1.0 - r * r) < 1e-3:
            continue
        lam22 = (1.0 - 1.0 / 2.2**2) / (1.0 - r * r)
        lam33 = (1.0 - 1.0 / 3.3**2) / (1.0 - r * r)
        lstd_b = lambda_std(r, M_q, CHI_QCD_BEST)
        lstd_h = lambda_std(r, M_q, CHI_QCD_HIGH)
        lnlo22 = required_lambda_NLO(r, 2.2, 0.5)
        lnlo33 = required_lambda_NLO(r, 3.3, 0.5)
        rows.append(Row(nB, r, M_q, lam22, lam33, lstd_b, lstd_h, lnlo22, lnlo33))
    # Aggregate at benchmark 2 n_0
    bench = next((r for r in rows if abs(r.nB - 2.0 * N0) < 1e-6), None)
    summary = {"benchmark": bench}
    return rows, summary


# -------------------------------------------------------------------------
# Report
# -------------------------------------------------------------------------
def report() -> str:
    rows, summ = driver()
    L: list[str] = []
    L.append("NVG EM-response: standard-QCD upper bound on lambda")
    L.append("=" * 78)
    L.append("")
    L.append("Channels included (all from textbook / published constants):")
    L.append("  (A) one-loop quark Z_QED(M_q(rho)) [NJL gap eq]")
    L.append("  (B) charged-pion loop with in-medium m_pi (GOR + Brown-Rho)")
    L.append("  (C) VMD photon-rho mixing at NS coherence scale")
    L.append("  (D) QCD magnetic susceptibility of chiral condensate")
    L.append("      (Belyaev-Kogan 1984; Bali et al lattice 2012)")
    L.append("")
    L.append("Constants:")
    L.append(f"  Lambda_chi = 4 pi f_pi = {LAMBDA_CHI*1000:.1f} MeV")
    L.append(f"  m_pi(0) = {M_PI*1000:.1f} MeV   f_pi = {F_PI*1000:.1f} MeV")
    L.append(f"  m_rho   = {M_RHO*1000:.1f} MeV   g_rho = {G_RHO:.1f}")
    L.append(f"  chi_QCD (best, Bali 2012)        = {CHI_QCD_BEST:+.2f} GeV^-2")
    L.append(f"  chi_QCD (upper, Belyaev-Kogan)   = {CHI_QCD_HIGH:+.2f} GeV^-2")
    L.append(f"  <q-bar q>_vacuum^(1/3)            = "
             f"{(-SIGMA_VACUUM)**(1.0/3.0)*1000:.1f} MeV")
    L.append("")
    L.append("-" * 78)
    L.append("Per-density comparison:  lambda_required (V2)  vs  lambda_std (QCD)")
    L.append("-" * 78)
    L.append(f"  {'n_B/n0':>6}  {'r=sigma/s0':>10}  "
             f"{'l_req(G=2.2)':>12}  {'l_req(G=3.3)':>12}  "
             f"{'l_std(best)':>11}  {'l_std(high)':>11}")
    for r in rows:
        L.append(f"  {r.nB/N0:6.2f}  {r.r:10.3f}  "
                 f"{r.lam_req_22:12.4f}  {r.lam_req_33:12.4f}  "
                 f"{r.lam_std_best:11.5f}  {r.lam_std_high:11.5f}")
    L.append("")

    bench = summ["benchmark"]
    if bench is not None:
        gap_best = bench.lam_req_22 / max(bench.lam_std_best, 1e-12)
        gap_high = bench.lam_req_22 / max(bench.lam_std_high, 1e-12)
        L.append("-" * 78)
        L.append(f"Benchmark density (2 n_0, r = {bench.r:.3f}):")
        L.append(f"  V2 requires      lambda in [{bench.lam_req_22:.3f}, "
                 f"{bench.lam_req_33:.3f}]")
        L.append(f"  Standard QCD     lambda = {bench.lam_std_best:.5f} "
                 f"(best, Bali lattice)")
        L.append(f"                   lambda = {bench.lam_std_high:.5f} "
                 f"(upper, SVZ Belyaev-Kogan)")
        L.append(f"  Shortfall factor = {gap_best:.1f}  to  {gap_high:.1f}")
        L.append("")
        L.append(f"  NLO ansatz (|mu| = 0.5 |lambda|): "
                 f"lambda in [{bench.lam_NLO_22:.3f}, {bench.lam_NLO_33:.3f}]")
        L.append(f"  -> NLO correction reduces required lambda by "
                 f"~{(1.0 - bench.lam_NLO_22/bench.lam_req_22)*100:.0f}%")
        L.append("")
    L.append("-" * 78)
    L.append("Honest conclusion")
    L.append("-" * 78)
    if bench is not None and bench.lam_std_high * 10 < bench.lam_req_22:
        L.append("Standard-QCD aggregation gives lambda_std ~ "
                 f"{bench.lam_std_best:.4f} (best estimate),")
        L.append(f"lambda_std ~ {bench.lam_std_high:.4f} (most generous SVZ).")
        L.append("Required for the V2 magnetar benchmark: lambda ~ "
                 f"{bench.lam_req_22:.3f} - {bench.lam_req_33:.3f}.")
        L.append("Gap: roughly two orders of magnitude.")
        L.append("")
        L.append("Interpretation. The condensate-photon coupling needed for the")
        L.append("V2 magnetar amplification is ~10^2 above the value that")
        L.append("standard QCD operators (QED loop, chiral loop, VMD, magnetic")
        L.append("susceptibility) deliver. Two scientifically honest options")
        L.append("follow:")
        L.append("")
        L.append("(a) Lower the V2 claim. With lambda ~ 0.01 the structural")
        L.append("    amplification factor is Gamma_struct ~ sqrt(1/(1 - 0.01))")
        L.append("    ~ 1.005, i.e. ~ 0.5 percent. NVG then contributes a small")
        L.append("    correction on top of the conventional dynamo/fossil-field")
        L.append("    mechanism rather than the dominant amplification.")
        L.append("")
        L.append("(b) Keep the V2 claim, but recognise that it requires a new")
        L.append("    operator beyond the Standard Model QCD-photon sector --")
        L.append("    something like a hidden U(1) mixing with EM whose coupling")
        L.append("    is set by the NVG order parameter rather than by chi_QCD.")
        L.append("    This is allowed by the NVG framework's modified vacuum")
        L.append("    sector but is, strictly speaking, BSM physics in the")
        L.append("    gauge sector, not a derivation from standard QCD.")
        L.append("")
        L.append("Either choice is publishable. Conflating them is not.")
    L.append("")
    L.append("=" * 78)
    return "\n".join(L)


if __name__ == "__main__":
    print(report())
