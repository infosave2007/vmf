"""
NVG EM-response companion derivation (honest version).

Closes the only remaining structural gap in the V2 magnetar preprint by
*deriving*, from textbook microphysics, what an effective vacuum-permeability
shift in dense matter actually delivers - and explicitly separating the
perturbative QED contribution (small) from the non-perturbative
chiral-condensate-coupling contribution (which carries the magnetar-relevant
amplification factor).

Two parallel calculations are performed.

(A) Standard perturbative QED-in-medium.
    - Solve the two-flavour NJL gap equation for M_q(rho, T) using textbook
      parameters (Lambda ~ 630 MeV, G * Lambda^2 ~ 2.14, m_q ~ 5 MeV).
    - Use the one-loop photon vacuum-polarisation contribution to the gauge
      kinetic term,
          Z_QED(M_q) = 1 - (alpha * N_c * sum_q Q_q^2) / (3 pi) * ln(Lambda^2 / M_q^2),
      to compute Gamma_struct_QED = sqrt( Z_QED(M_vac) / Z_QED(M_dense) ).
    - Expected outcome: Gamma_struct_QED ~ 1 + O(alpha * log) ~ 1.01 - 1.05.
      This is the honest measure of what one-loop QED alone delivers.

(B) Non-perturbative chiral-condensate-coupling channel.
    - Assume an effective gauge kinetic term of the form
          L_eff = - Z_chi(sigma) / 4 * F^2,
          Z_chi(sigma) = 1 - lambda * ( 1 - sigma^2(rho) / sigma^2(0) ),
      where sigma(rho) = <q-bar q>(rho) / <q-bar q>(0) is the normalised
      chiral order parameter (computed from the same NJL gap equation) and
      lambda is the dimensionless condensate-photon coupling.
    - The scale of lambda is set by QCD physics (G_NJL ~ GeV^-2, not alpha):
      naive dimensional analysis gives lambda ~ O(0.5 - 1).
    - Solve for lambda such that the in-medium Gamma_struct band overlaps the
      benchmark range Gamma_struct ~ 2.2 - 3.3 used in the magnetar V2 paper.
    - The output is the *required* lambda, which becomes a single explicit
      phenomenological number that can be checked against lattice QCD,
      chiral-perturbation, and quark-meson model estimates.

Headline outcome
----------------
The companion derivation shows that the magnetar-relevant amplification
Gamma_struct ~ 2 - 3 cannot come from one-loop QED vacuum polarisation alone
(it falls short by ~ two orders of magnitude in the log). Instead the
amplification must be carried by a non-perturbative condensate-photon
coupling lambda ~ O(0.8 - 0.95), which is a single, testable, and lattice-
predictable number. This converts the previously "motivated" V2 step into an
explicit microphysical statement with one named parameter.

Run:
    python3.12 verification/nvg_em_response_derivation.py
"""
from __future__ import annotations

import math
from dataclasses import dataclass

# -------------------------------------------------------------------------
# Standard two-flavour NJL parameters (Klevansky 1992; Buballa Phys.Rep. 2005).
# -------------------------------------------------------------------------
LAMBDA = 0.6313                # GeV, three-momentum cutoff
G_LAMBDA2 = 2.14               # textbook dimensionless coupling
G_NJL = G_LAMBDA2 / LAMBDA**2  # GeV^-2
M_CUR = 0.0055                 # GeV, current u/d mass

# QED constants
ALPHA = 1.0 / 137.036
N_C = 3
Q_U = 2.0 / 3.0
Q_D = -1.0 / 3.0
SUM_Q2 = N_C * (Q_U**2 + Q_D**2)   # = 5/3 with N_c = 3

# Convenience: hbar * c
HBARC = 0.1973  # GeV fm
N0 = 0.16       # fm^-3, nuclear saturation density


# -------------------------------------------------------------------------
# NJL gap equation. We use a *bisection* root finder over M in (m_cur, Lambda]
# rather than fixed-point iteration; the latter is unreliable because the
# vacuum branch coexists with a trivial chirally restored branch and a naive
# Picard step can jump between them.
# -------------------------------------------------------------------------
N_F = 2

def fermi(x: float) -> float:
    if x > 50.0:
        return 0.0
    if x < -50.0:
        return 1.0
    return 1.0 / (math.exp(x) + 1.0)

def gap_integral(M: float, mu: float, T: float, n_k: int = 400) -> float:
    """Compute I(M, mu, T) = M * int_0^Lambda (k^2 / (2 pi^2 E)) *
       [1 - n_F(E-mu) - n_F(E+mu)] dk, regularised by the NJL cutoff.

    Units: GeV^3. The factor of M comes from the trace over Dirac structure;
    the bracket (1 - n - nbar) is the standard particle/antiparticle filling."""
    dk = LAMBDA / n_k
    total = 0.0
    for i in range(n_k + 1):
        k = i * dk
        E = math.sqrt(k * k + M * M)
        if T > 1e-6:
            fp = fermi((E - mu) / T)
            fm = fermi((E + mu) / T)
        else:
            fp = 1.0 if E < mu else 0.0
            fm = 0.0
        w = 0.5 if (i == 0 or i == n_k) else 1.0
        total += w * (k * k / (2.0 * math.pi**2 * E)) * (1.0 - fp - fm) * dk
    return M * total

def gap_residual(M: float, mu: float, T: float) -> float:
    return M - M_CUR - 4.0 * G_NJL * N_C * N_F * gap_integral(M, mu, T)

def solve_gap(mu: float, T: float) -> float:
    """Locate the *largest* root of the gap equation in (m_cur, Lambda] by
    bracketing then bisection. If no broken-symmetry root exists, return
    m_cur (chirally restored phase)."""
    n_scan = 40
    Ms = [M_CUR + (LAMBDA - M_CUR) * (1.0 - i / n_scan) for i in range(n_scan + 1)]
    f_prev = gap_residual(Ms[0], mu, T)
    bracket = None
    for j in range(1, len(Ms)):
        f_now = gap_residual(Ms[j], mu, T)
        if f_prev * f_now < 0.0:
            bracket = (Ms[j - 1], Ms[j], f_prev, f_now)
            break
        f_prev = f_now
    if bracket is None:
        return M_CUR
    a, b, fa, fb = bracket
    for _ in range(60):
        m = 0.5 * (a + b)
        fm = gap_residual(m, mu, T)
        if fa * fm < 0.0:
            b, fb = m, fm
        else:
            a, fa = m, fm
        if abs(b - a) < 1e-6:
            break
    return 0.5 * (a + b)


# -------------------------------------------------------------------------
# Chiral condensate per flavour
# -------------------------------------------------------------------------
def chiral_condensate(M: float) -> float:
    n_k = 400
    dk = LAMBDA / n_k
    total = 0.0
    for i in range(n_k + 1):
        k = i * dk
        E = math.sqrt(k * k + M * M)
        w = 0.5 if (i == 0 or i == n_k) else 1.0
        total += w * (M / (math.pi**2 * E)) * k * k * dk
    return -N_C * total

M_Q_VACUUM = solve_gap(mu=0.0, T=0.0)
SIGMA_VACUUM = chiral_condensate(M_Q_VACUUM)


# -------------------------------------------------------------------------
# Density -> chemical potential at finite M_q (symmetric two-flavour matter)
# -------------------------------------------------------------------------
def kF_from_nB(n_B_fm3: float) -> float:
    n_B_gev3 = n_B_fm3 * HBARC**3
    return (math.pi**2 * 3.0 * n_B_gev3 / 2.0) ** (1.0 / 3.0)


# -------------------------------------------------------------------------
# (A) Perturbative QED contribution to the gauge kinetic term
# -------------------------------------------------------------------------
def Z_QED(M_q: float) -> float:
    if M_q <= 0.0:
        return 1.0
    return 1.0 - (ALPHA * SUM_Q2) / (3.0 * math.pi) * math.log((LAMBDA / M_q) ** 2)

def gamma_QED(M_q: float) -> float:
    Zd = Z_QED(M_q)
    Zv = Z_QED(M_Q_VACUUM)
    return math.sqrt(Zv / Zd)


# -------------------------------------------------------------------------
# (B) Non-perturbative chiral-condensate coupling
# -------------------------------------------------------------------------
def gamma_chi(r: float, lam: float) -> float:
    arg = 1.0 - lam * (1.0 - r * r)
    if arg <= 0.0:
        return float("inf")
    return 1.0 / math.sqrt(arg)

def required_lambda(r: float, gamma_target: float) -> float:
    """Solve gamma_chi(r, lambda) = gamma_target for lambda."""
    return (1.0 - 1.0 / gamma_target**2) / (1.0 - r * r)


# -------------------------------------------------------------------------
# Main scan
# -------------------------------------------------------------------------
@dataclass
class Point:
    n_B: float
    T: float
    mu_q: float
    M_q: float
    sigma_ratio: float
    gamma_qed: float

def scan() -> list[Point]:
    pts: list[Point] = []
    densities = [1.0 * N0, 1.5 * N0, 2.0 * N0, 2.5 * N0, 3.0 * N0, 4.0 * N0, 5.0 * N0]
    temperatures = [0.0, 0.010, 0.030, 0.050]
    for nB in densities:
        kF = kF_from_nB(nB)
        for T in temperatures:
            M_iter = M_Q_VACUUM
            for _ in range(6):
                mu_q = math.sqrt(kF * kF + M_iter * M_iter)
                M_new = solve_gap(mu_q, T)
                if abs(M_new - M_iter) < 1e-5:
                    M_iter = M_new
                    break
                M_iter = 0.5 * M_iter + 0.5 * M_new
            M_q = M_iter
            sigma = chiral_condensate(M_q)
            r = sigma / SIGMA_VACUUM if SIGMA_VACUUM != 0.0 else 0.0
            pts.append(Point(nB, T, math.sqrt(kF * kF + M_q * M_q),
                             M_q, r, gamma_QED(M_q)))
    return pts


# -------------------------------------------------------------------------
# Report
# -------------------------------------------------------------------------
def fmt(p: Point) -> str:
    return (f"n_B = {p.n_B/N0:4.2f} n0   T = {p.T*1000:4.0f} MeV   "
            f"mu_q = {p.mu_q*1000:6.1f} MeV   "
            f"M_q = {p.M_q*1000:6.1f} MeV   "
            f"sigma/sigma_0 = {p.sigma_ratio:5.3f}   "
            f"Gamma_QED = {p.gamma_qed:6.4f}")

def report() -> str:
    L: list[str] = []
    L.append("NVG EM-response companion derivation")
    L.append("=" * 78)
    L.append("")
    L.append("Textbook two-flavour NJL parameters:")
    L.append(f"  Lambda       = {LAMBDA*1000:.1f} MeV")
    L.append(f"  G * Lambda^2 = {G_LAMBDA2:.2f}")
    L.append(f"  m_current    = {M_CUR*1000:.2f} MeV")
    L.append("")
    L.append(f"Solved vacuum constituent mass: M_q(0,0) = {M_Q_VACUUM*1000:.1f} MeV")
    cond_cuberoot = (-SIGMA_VACUUM) ** (1.0 / 3.0) * 1000.0 if SIGMA_VACUUM < 0 else 0.0
    L.append(f"Vacuum chiral condensate per flavour: "
             f"|<q-bar q>|^(1/3) = {cond_cuberoot:.1f} MeV")
    L.append("")
    L.append("-" * 78)
    L.append("(A) Perturbative QED-in-medium contribution to Gamma_struct")
    L.append("-" * 78)
    pts = scan()
    for p in pts:
        L.append("  " + fmt(p))
    g_min = min(p.gamma_qed for p in pts)
    g_max = max(p.gamma_qed for p in pts)
    L.append("")
    L.append(f"Perturbative QED yields Gamma_struct in [{g_min:.4f}, {g_max:.4f}].")
    L.append("This is 1 + O(alpha * log) ~ 1 percent, far below the magnetar")
    L.append("V2 benchmark band Gamma_struct ~ 2.2 - 3.3.")
    L.append("Conclusion: one-loop QED alone cannot supply the amplification.")
    L.append("")
    L.append("-" * 78)
    L.append("(B) Non-perturbative chiral-condensate coupling")
    L.append("-" * 78)
    L.append("Effective gauge kinetic term:")
    L.append("    Z_chi(sigma) = 1 - lambda * (1 - (sigma(rho)/sigma(0))^2)")
    L.append("    Gamma_struct  = 1 / sqrt( Z_chi(rho) ),   (Z_chi(0) = 1)")
    L.append("")
    L.append("Required lambda to place Gamma_struct inside [2.2, 3.3]:")
    L.append("")
    L.append(f"  {'n_B/n0':>6}  {'T [MeV]':>7}  {'sigma/sigma_0':>13}  "
             f"{'lambda(G=2.2)':>13}  {'lambda(G=3.3)':>13}")
    for p in pts:
        if p.T > 0.011:
            continue
        if abs(1.0 - p.sigma_ratio * p.sigma_ratio) < 1e-3:
            continue
        lam_lo = required_lambda(p.sigma_ratio, 2.2)
        lam_hi = required_lambda(p.sigma_ratio, 3.3)
        L.append(f"  {p.n_B/N0:6.2f}  {p.T*1000:7.1f}  {p.sigma_ratio:13.3f}  "
                 f"{lam_lo:13.4f}  {lam_hi:13.4f}")
    L.append("")

    target = next((p for p in pts
                   if abs(p.n_B - 2.0 * N0) < 1e-6 and abs(p.T - 0.010) < 1e-6),
                  None)
    if target and target.sigma_ratio**2 < 0.999:
        lam_22 = required_lambda(target.sigma_ratio, 2.2)
        lam_33 = required_lambda(target.sigma_ratio, 3.3)
        L.append(f"Headline (n_B = 2 n_0, T = 10 MeV, sigma/sigma_0 = "
                 f"{target.sigma_ratio:.3f}):")
        L.append(f"    lambda in [{lam_22:.3f}, {lam_33:.3f}]  "
                 f"to reproduce Gamma_struct in [2.2, 3.3].")
        L.append("")
        L.append("Interpretation: the V2 benchmark amplification corresponds to a")
        L.append("single phenomenological condensate-photon coupling lambda of order")
        L.append("unity. This is the falsifiable microphysical input. It can be")
        L.append("checked independently by lattice QCD calculations of the photon")
        L.append("vacuum polarisation as a function of <q-bar q>, or by chiral-")
        L.append("perturbation estimates of the condensate-dependent gauge kinetic")
        L.append("term. If lattice gives lambda << 1, the NVG channel is falsified.")
    L.append("")
    L.append("=" * 78)
    return "\n".join(L)


if __name__ == "__main__":
    print(report())
