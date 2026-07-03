#!/usr/bin/env python3
"""
NVG Cosmology: Hubble Constant — Cycle-77 Bound and Calibration
----------------------------------------------------------------
Computes the cycle-77 bound on the Hubble constant in the VMF cyclic cosmology:
the turnaround horizons R_77 = r_c·2^76 and R_78 = r_c·2^77 bracket the present
horizon, bounding H_0 to [54.3, 108.5] km/s/Mpc. This interval is the falsifiable
prediction of the model.

Without any observational input, the zero-information midpoint of the interval
(log-uniform measure, natural for a doubling hierarchy) gives the fit-free value
H_0 = H_77/sqrt(2) ≈ 76.8 km/s/Mpc — within ~5% of the local SH0ES measurement.
The center is measure-dependent (time-uniform ≈ 62, H-uniform ≈ 81 km/s/Mpc),
so only the hard interval is measure-free.

Within the interval, the specific value N_e = 53.08 is CALIBRATED to the local
distance-ladder measurement H_0 ≈ 72.8 km/s/Mpc (SH0ES-anchored; see
nvg_genesis_observable.py, where N_e = ln(R_H0/r_c) is computed from that H_0).
The sub-1σ agreement with SH0ES reported below is therefore by construction,
not an independent result.

The IR-cutoff route to the Hubble tension has been TESTED (nvg_cmb_lowl_refit.py):
the Genesis cutoff P(k) -> P(k)*exp(-(k_c/k)²) improves the Planck low-ℓ fit
mildly (Δχ² ≈ +0.9, best-fit scale ≈ predicted k_c = 1/R_H0), but cannot shift
the CMB-inferred H_0: the acoustic scale θ* (0.03% measurement) is untouched by
a cutoff acting at ℓ ≲ 6, and forcing H_0 = 72.8 costs Δχ² ≈ +150 at high ℓ.
"""

import math

def run_hubble_tension_verification():
    print("==========================================================================")
    print("  NVG COSMOLOGY: HUBBLE CONSTANT — CYCLE-77 BOUND AND CALIBRATION")
    print("==========================================================================")
    
    # 1. Physics Inputs
    M_Omega_0 = 859.0       # MeV
    hbar_c = 197.3269804    # MeV·fm
    G_cgs = 6.674e-8        # cm^3 g^-1 s^-2
    c_cgs = 2.998e10        # cm/s
    MeV_fm3_to_gcm3 = 1.7827e12
    
    # Calculate Genesis instanton radius r_c
    eps_max = M_Omega_0**4 / hbar_c**3  # MeV/fm^3
    rho_c_cgs = eps_max * MeV_fm3_to_gcm3
    H_c_cgs = math.sqrt(8.0 * math.pi * G_cgs * rho_c_cgs / 3.0)
    r_c_cm = c_cgs / H_c_cgs
    r_c_km = r_c_cm / 1e5
    
    # 2. Derive turnaround horizons for the 77th and 78th Tolman cycles
    # Gibbons-Hawking entropy scales as S ~ R_H^2. Under Tolman's entropy growth relation
    # S_n = S_1 * 4^(n-1), the turnaround horizon of cycle n scales as R_n = r_c * 2^(n-1).
    R_77_cm = r_c_cm * (2**76)
    R_78_cm = r_c_cm * (2**77)
    
    H_77 = (c_cgs / R_77_cm) * (3.086e24 / 1e5)  # km/s/Mpc
    H_78 = (c_cgs / R_78_cm) * (3.086e24 / 1e5)  # km/s/Mpc
    
    # 3a. FIT-FREE estimate: the zero-information position inside cycle 77
    # With no observational input beyond M_Omega, the only distinguished position in the
    # topological interval N_e ∈ [76*ln2, 77*ln2] is its midpoint under the log-uniform
    # measure (natural for a doubling hierarchy): N_e_mid = 76.5*ln2, i.e. H_0 = H_77/sqrt(2).
    # The point value is measure-dependent — uniform-in-time (matter era, a ∝ t^(2/3)) and
    # uniform-in-H give different centers — so the interval [H_78, H_77] is the only
    # measure-free claim.
    N_e_mid = 76.5 * math.log(2.0)
    H_0_fitfree = (c_cgs / (r_c_cm * math.exp(N_e_mid))) * (3.086e24 / 1e5)  # = H_77/sqrt(2)
    H_0_time_uniform = H_77 * math.log(2.0**1.5) / (2.0**1.5 - 1.0)
    H_0_H_uniform = 0.5 * (H_77 + H_78)

    # 3b. Calculate H_0 for the calibrated intra-cycle position
    # The cycle index bounds the e-folds: since the turnaround horizon scales as R_n = r_c * 2^(n-1),
    # any moment inside cycle n = 77 satisfies N_e ∈ [76*ln(2), 77*ln(2)] = [52.68, 53.38],
    # equivalently H_0 ∈ [54.3, 108.5] km/s/Mpc. This interval is the falsifiable prediction.
    # The specific value N_e = 53.08 inside the interval is NOT derived: it is calibrated to the
    # local distance-ladder value H_0 ≈ 72.8 km/s/Mpc (nvg_genesis_observable.py computes
    # N_e = ln(R_H0/r_c) from that H_0). The phase angles below are alternative labels for this
    # calibrated position; they range from 47.9° to 54.4° depending on the interpolation
    # convention, so no single angle is predicted by the model.
    N_e = 53.08
    R_H_0_cm = r_c_cm * math.exp(N_e)
    R_H_0_Mpc = R_H_0_cm / (3.086e24)
    H_0_predicted = (c_cgs / R_H_0_cm) * (3.086e24 / 1e5)
    
    # Calculate corresponding phase angles under different natural definitions:
    # A. Logarithmic horizon scale interpolation: R_H_0 = R_77 * 2^(sin^2(theta))
    theta_log = math.asin(math.sqrt(math.log2(R_H_0_cm / R_77_cm))) * 180.0 / math.pi
    # B. Linear interpolation in H^2 (Friedmann energy scaling): H_0^2 = H_77^2 * cos^2(theta)
    theta_energy = math.acos(H_0_predicted / H_77) * 180.0 / math.pi
    # C. Linear interpolation in H: H_0 = H_77 * cos^2(theta) + H_78 * sin^2(theta)
    theta_linear = math.asin(math.sqrt((H_77 - H_0_predicted) / (H_77 - H_78))) * 180.0 / math.pi
    
    print(f"Genesis Instanton parameters:")
    print(f"  Instanton Radius r_c: {r_c_km:.4f} km")
    print(f"  E-folds N_e: {N_e:.2f} (calibrated to local H_0; bounded to [52.68, 53.38] by cycle 77)")
    print(f"  Stretched Horizon size R_H_0: {R_H_0_Mpc:.1f} Mpc")
    
    print(f"\nTolman Cycle 77/78 Turnaround Bounds:")
    print(f"  Cycle 77 Turnaround Horizon: {R_77_cm / 3.086e24:.1f} Mpc (H = {H_77:.2f} km/s/Mpc)")
    print(f"  Cycle 78 Turnaround Horizon: {R_78_cm / 3.086e24:.1f} Mpc (H = {H_78:.2f} km/s/Mpc)")
    print(f"  Current Observed Horizon: {R_H_0_Mpc:.1f} Mpc (H_0 = {H_0_predicted:.2f} km/s/Mpc)")
    print(f"  Bounds check: {H_78:.2f} < {H_0_predicted:.2f} < {H_77:.2f} km/s/Mpc (Strictly Bounded)")

    print(f"\nFit-free H_0 (no observational input beyond M_Omega):")
    print(f"  Hard bounds (cycle 77)          : {H_78:.1f} < H_0 < {H_77:.1f} km/s/Mpc (measure-free)")
    print(f"  Mid-cycle, log-uniform measure  : H_0 = H_77/sqrt(2) = {H_0_fitfree:.2f} km/s/Mpc")
    print(f"  Measure spread of the center    : {H_0_time_uniform:.1f} (time-uniform) ... {H_0_H_uniform:.1f} (H-uniform)")
    print(f"  vs SH0ES 73.04: {(H_0_fitfree/73.04-1)*100:+.1f}%   vs Planck 67.4: {(H_0_fitfree/67.4-1)*100:+.1f}%")
    print(f"  Both observed values lie inside the measure spread; the log-uniform midpoint")
    print(f"  lands within ~5% of the local (SH0ES) value with zero fitted parameters.")
    
    print(f"\nPhase-angle labels for the calibrated N_e (convention-dependent, not predicted):")
    print(f"  - Log-horizon scaling phase   : {theta_log:.2f}°")
    print(f"  - Friedmann energy phase      : {theta_energy:.2f}°")
    print(f"  - Linear Hubble phase         : {theta_linear:.2f}°")
    
    # 4. Hubble constant for the calibrated N_e:
    print(f"\nHubble constant for calibrated N_e (consistency check, not an independent prediction):")
    print(f"  H_0 = {H_0_predicted:.2f} km/s/Mpc")
    
    # 5. Comparison with observations
    H_0_local = 73.04       # SH0ES (Riess et al. 2022)
    H_0_local_err = 1.04
    H_0_planck = 67.4       # Planck PR4 2018 (Standard LCDM fit)
    H_0_planck_err = 0.5
    
    z_local = (H_0_predicted - H_0_local) / H_0_local_err
    z_planck = (H_0_predicted - H_0_planck) / H_0_planck_err
    
    print(f"\nComparison against observational datasets:")
    print(f"  1. Local Measurements (SH0ES): {H_0_local} ± {H_0_local_err} km/s/Mpc")
    print(f"     NVG Deviation: {z_local:+.2f}σ (by construction: N_e calibrated to local H_0)")
    print(f"  2. High-z CMB Standard Fit (Planck): {H_0_planck} ± {H_0_planck_err} km/s/Mpc")
    print(f"     NVG Deviation: {z_planck:+.2f}σ (Systematic shift due to standard LCDM assumption)")
    
    print("\nStatus of the 'Hubble tension resolution' claim (tested in nvg_cmb_lowl_refit.py):")
    print("The Genesis IR cutoff P(k) -> P(k)*exp(-(k_c/k)^2) at k_c = 1/R_H0 improves the")
    print("Planck 2018 low-l TT fit mildly (Delta chi2 ~ +0.9; the best-fit cutoff scale lies")
    print("within ~1.2x of the predicted k_c). However, the CMB-inferred H_0 is fixed by the")
    print("acoustic scale theta* (measured to 0.03%), which the cutoff (l <~ 6 only) does not")
    print("touch: forcing H_0 = 72.8 costs Delta chi2 ~ +150 in the acoustic region even with")
    print("theta* restored by refitting omch2. The cutoff therefore does NOT resolve the")
    print("Hubble tension. Surviving falsifiable claims: the interval prediction above and")
    print("the low-l cutoff scale itself.")
    print("==========================================================================")

if __name__ == "__main__":
    run_hubble_tension_verification()
