#!/usr/bin/env python3
"""
NVG Verification: Sub-millimeter fifth-force constraints on the theta mode.

The theta sector of the NVG condensate contains a light Goldstone mode
(theta-mode, m_theta = 53 ueV, ADMX Gen2 window; row 47 of the prediction
table). Its Compton range falls exactly in the sub-millimeter window probed
by torsion-balance experiments:

    lambda_theta = hbar c / (m_theta c^2) = 3.72 mm

This script computes:
  (1) the scalar Yukawa strength alpha_theta relative to gravity for the
      pseudoscalar nucleon coupling g_thetaNN = m_N / f_a with
      f_a = 1.07e11 GeV (theta-seesaw anchor),
  (2) the comparison with current torsion-balance 1-sigma bounds
      alpha < ~1e11-1e12 at lambda ~ 3-4 mm (Eot-Wash, HUST-2020),
  (3) the velocity-suppressed scalar channel of the DERIVATIVE coupling
      (partial_mu theta) J_B^mu / f_a (a null-test prediction),
  (4) the diagnostic resolutions of the tension.

Outcome: a hypothetical scalar Yukawa channel at g = m_N/f_a would exceed
mm-scale monopole bounds by ~3 orders of magnitude, but the theta mode is
a PSEUDOSCALAR: the monopole channel vanishes identically (null prediction
for Eot-Wash-class unpolarized searches), while the spin-dipole channel
places f_a = 1.07e11 GeV right at the edge of current spin-polarized
bounds. Diagnostic resolutions (a)-(c) are documented for the bookkeeping
tension.
"""

import math

# constants
HBAR_C_GEVM = 1.973269804e-16   # GeV m
M_N_GEV = 0.938272              # nucleon mass
ALPHA_G = 5.9086e-39            # G m_p^2 / (hbar c), gravitational strength

F_A_GEV = 1.07e11               # theta-seesaw decay constant (row 47)
M_THETA_UEV = 53.0              # theta-mode mass (row 47)


def main():
    print("=" * 72)
    print(" NVG THETA-MODE FIFTH FORCE AT SUB-MILLIMETER SCALES")
    print("=" * 72)

    # (1) Compton range
    m_th_gev = M_THETA_UEV * 1e-15   # ueV -> GeV
    lam = HBAR_C_GEVM / m_th_gev
    print(f"[1] Compton range lambda_theta = hbar c / m_theta = "
          f"{lam*1e3:.2f} mm")
    print("    -> inside the 0.05-10 mm window of modern torsion balances")
    print("-" * 72)

    # (2) bookkeeping: hypothetical SCALAR Yukawa strength at g = m_N/f_a
    g = M_N_GEV / F_A_GEV
    alpha_scalar = g**2 / (4.0 * math.pi * ALPHA_G)
    print(f"[2] coupling g_thetaNN = m_N/f_a = {g:.3e}")
    print(f"    HYPOTHETICAL scalar Yukawa strength g^2/(4 pi alpha_G) = "
          f"{alpha_scalar:.3e}")
    print("    torsion-balance MONOPOLE bounds at 3-4 mm: alpha < ~1e11 - 1e12")
    print("    -> such a SCALAR exchange would be excluded by ~3 orders")
    print("       of magnitude. This is the tension to be resolved.")
    print("-" * 72)

    # (3) actual physical channels of a PSEUDOSCALAR mode
    print("[3] physical channels of the pseudoscalar theta mode:")
    print("    (i) MONOPOLE-MONOPOLE: a pseudoscalar has NO parity-even")
    print("        coupling to unpolarized nucleon density (bar N N). The")
    print("        scalar Yukawa channel is identically ZERO, and the")
    print("        derivative form (partial theta) J_B^mu/f_a reduces by")
    print("        baryon-current conservation to spin- and velocity-")
    print("        suppressed terms (~(v/c)^2 ~ 1e-8).")
    print("        => NULL PREDICTION: no unpolarized fifth force at 3.7 mm")
    print("           (falsified by ANY Eot-Wash-class monopole signal).")
    print("    (ii) SPIN-DIPOLE: (g/2m_N) (partial theta) bar N gamma5 N")
    print("         gives the standard axion-like spin-dependent force.")
    print(f"         For a QCD-axion-like m_N/f_a coupling, spin-dipole")
    print(f"         searches exclude f_a <~ 1e10 - 1e11 GeV at this mass;")
    print(f"         theta-seesaw anchor f_a = {F_A_GEV:.2e} GeV sits RIGHT")
    print("         at the edge -> ADMX Gen2 / spin-polarized torsion")
    print("         balances (He3, Xe, geochemical) decide the sector.")
    print("-" * 72)

    # (4) diagnostic resolutions of the bookkeeping tension
    print("[4] DIAGNOSTIC: if the theta mode were scalar-coupled, three")
    print("    documented resolutions exist:")
    print("    (a) topological suppression gamma_topo <= 1e-4 in the")
    print("        nucleon coupling of the theta mode;")
    g_req = math.sqrt(1e11 * 4.0 * math.pi * ALPHA_G)
    print(f"        required g_thetaNN <= {g_req:.2e} "
          f"(vs naive {g:.2e})")
    print("    (b) the eta' identification of the theta phase (row 47 audit):")
    lam_eta = HBAR_C_GEVM / 0.9578
    print(f"        m_eta' = 957.8 MeV -> lambda = {lam_eta*1e15:.3f} fm,")
    print("        far below any fifth-force reach (null test restored);")
    print("    (c) the physical coupling is purely derivative/pseudoscalar,")
    print("        i.e. resolution [3](i)-(ii) above -- the channel selected")
    print("        by the standard axion effective theory.")
    print("=" * 72)
    print("STATUS: tension diagnostic. The theta-mode range 3.72 mm is a")
    print("sharp prediction; resolution (b) or (c) is selected by the")
    print("theta-sector audit, resolution (a) is falsifiable by ADMX Gen2")
    print("non-detection combined with the RHIC Bell test.")
    print("=" * 72)


if __name__ == "__main__":
    main()
