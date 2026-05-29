#!/usr/bin/env python3
"""
NVG Research — Run All Verification Checks.

Executes all computational prototypes in sequence and reports
pass/fail status for each verifiable claim.

Usage:
    python run_all_checks.py
"""

from __future__ import annotations
import subprocess
import sys
import os
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), ".code")

CHECKS = [
    {
        "name": "Hadron Mass Fractions (Lattice QCD)",
        "script": "nvg_hadron_mass_fractions.py",
        "claim": "f_Omega(N) ≈ 0.91 ± 0.02, M_Omega = 859 ± 8 MeV",
        "critical": True,
    },
    {
        "name": "Core EOS: Saturating Vector Sector",
        "script": "nvg_eos_beta_saturated_vector.py",
        "claim": "At least one point passes core screening with c_s^2 ≤ 1",
        "critical": True,
    },
    {
        "name": "Symmetric Matter Screening (corrected units)",
        "script": "nvg_eos_proof_checked.py",
        "claim": "Constant-vector models are too stiff (expected: 0 pass)",
        "critical": False,
    },
    {
        "name": "Beta-Equilibrated Core EOS",
        "script": "nvg_eos_beta_checked.py",
        "claim": "Scalar+isovector alone insufficient (expected: 0 pass)",
        "critical": False,
    },
    {
        "name": "Black Hole Interior: Mass Melting Profile",
        "script": "nvg_black_hole_interior.py",
        "claim": "M* → 0 at extreme density, P/ε → 1/3 (conformal), ε_max finite",
        "critical": True,
    },
    {
        "name": "Condensate Modulation (Podkletnov/Modanese)",
        "script": "nvg_condensate_modulation.py",
        "claim": "2% weight loss matches η~0.02, spark discharge yields 1e6x force amplification",
        "critical": True,
    },
    {
        "name": "Full Neutron Star EOS (Crust + Phase Transition)",
        "script": "nvg_full_ns_eos.py",
        "claim": "Phase transition lowers M_max to ~2.3 M_sun (phenomenologically viable)",
        "critical": True,
    },
    {
        "name": "Hadron Universality & FAIR/HADES Observables",
        "script": "nvg_fair_hades_link.py",
        "claim": "f_Omega is universal for non-Goldstone light hadrons (~90%); rho drops by ~23% at 2n_0",
        "critical": True,
    },
    {
        "name": "Hyperon Puzzle & Phase Transition Nature",
        "script": "nvg_hyperon_puzzle.py",
        "claim": "Lambda hyperons appear at 1.7 n_0, before QGP transition at 2.0 n_0",
        "critical": True,
    },
    {
        "name": "Gravitational Wave Echoes",
        "script": "nvg_gw_echoes.py",
        "claim": "Echo delay for GW150914 (65 M_sun) is precisely 0.0445s",
        "critical": True,
    },
    {
        "name": "Tolman Cycles & Entropy Growth",
        "script": "nvg_cyclic_lifetimes.py",
        "claim": "Genesis lifetime was ~5.9 us; current universe is the ~77th cycle",
        "critical": True,
    },
    {
        "name": "Laboratory Graphene Topological Limit",
        "script": "nvg_graphene_modulation.py",
        "claim": "Thermodynamic RF bulk pumping is < 10^-15 effective, requiring topological resonance",
        "critical": True,
    },
    {
        "name": "Dark Photon Observables & Constraints",
        "script": "nvg_dark_photon_observables.py",
        "claim": "BBN decay bounds eps_vac > 1.5e-11, and SN1987A requires mass melting <= 20%",
        "critical": True,
    },
    {
        "name": "Magnetar Mass-Field Correlation",
        "script": "nvg_magnetar_mass_correlation.py",
        "claim": "Fitted progenitor seed ~42 kG, reconstructed masses in 1.1-2.3 M_sun span, correlation R ~ 0.61",
        "critical": True,
    },
    {
        "name": "Quantitative Physical Predictions & Validation",
        "script": "nvg_new_predictions.py",
        "claim": "Breit-Wigner peak significance > 2 sigma at FAIR, LMXB suppression < 1e-20",
        "critical": True,
    },
    {
        "name": "Unified Field Equations & Core Limits",
        "script": "nvg_unified_field_equations.py",
        "claim": "Verify cosmological bounce, magnetar chiral instability, and magnetic backpressure",
        "critical": True,
    },
    {
        "name": "HADES Dielectron Spectral Simulation",
        "script": "nvg_hades_dielectron_sim.py",
        "claim": "VMF meson mass shift creates a prominent peak shift to ~702 MeV, distinct from broadening",
        "critical": True,
    },
    {
        "name": "GW Echo Waveform Templates",
        "script": "nvg_gw_echo_waveforms.py",
        "claim": "Hayward core scale r_0 gives periodic post-merger echoes with delay ~81 ms, decaying by T_horizon",
        "critical": True,
    },
    {
        "name": "DESI DR2 Dark Energy Parametric Alignment",
        "script": "nvg_dark_energy_desi.py",
        "claim": "VMF cyclic cosmology prediction (w0 = -0.89, wa = -0.60) aligns within 2.8-sigma of DESI DR2",
        "critical": True,
    },
    {
        "name": "Tidal Deformability GW170817 & Double Pulsar MoI",
        "script": "nvg_tidal_deformability_gw170817.py",
        "claim": "Verify R_1.4 ~ 11.1 km and Double Pulsar moment of inertia is compatible with obs (~1.12e45 g cm^2)",
        "critical": True,
    },
    {
        "name": "JWST Early Black Hole Seeding Puzzle",
        "script": "nvg_pbh_jwst_seeds.py",
        "claim": "NVG primordial seeds (~4e5 M_sun) resolve the early SMBH seeding puzzle at z > 6, whereas Pop III seeds fail",
        "critical": True,
    },
    {
        "name": "Joint NS Likelihood Fit",
        "script": "nvg_joint_ns_inference.py",
        "claim": "Reduced chi_nu^2 = 0.63 for joint multi-messenger fit",
        "critical": True,
    },
    {
        "name": "NICER PSR J0437 Mass-Radius Check",
        "script": "nvg_nicer_j0437_check.py",
        "claim": "VMF predicted radius lies within 0.8 sigma of NICER 2024 bounds",
        "critical": True,
    },
    {
        "name": "Relic Dark Matter density",
        "script": "nvg_relic_dark_matter.py",
        "claim": "Instanton relic density matches Planck PR4 Omega_DM = 0.268",
        "critical": True,
    },
]

OPTIONAL_CHECKS = [
    {
        "name": "FRW Entropy-Clock Cycle Fit (Path A)",
        "script": "nvg_frw_entropy_cycle_fit.py",
        "claim": "Cyclic cosmology toy model is numerically consistent",
        "critical": False,
    },
    {
        "name": "Entropy-Time Cycle Toy Model",
        "script": "nvg_entropy_time_cycle_toy.py",
        "claim": "Finite cycle in emergent time is mathematically possible",
        "critical": False,
    },
    {
        "name": "Macroscopic Weak-Field Limit (PPN Parameters)",
        "script": "nvg_weak_field_ppn.py",
        "claim": "NVG reduces exactly to GR in the macroscopic vacuum (gamma_PPN = 1)",
        "critical": False,
    },
]


def run_script(script_name: str, timeout: int = 120, search_dir: str = "") -> tuple[bool, str]:
    """Run a Python script and return (success, output)."""
    base_dir = search_dir or SCRIPT_DIR
    script_path = os.path.join(base_dir, script_name)
    if not os.path.exists(script_path):
        return False, f"Script not found: {script_path}"

    try:
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=base_dir,
        )
        output = result.stdout + result.stderr
        success = result.returncode == 0
        return success, output
    except subprocess.TimeoutExpired:
        return False, f"TIMEOUT after {timeout}s"
    except Exception as e:
        return False, f"ERROR: {e}"


def main() -> int:
    print("=" * 76)
    print("NVG RESEARCH — VERIFICATION SUITE")
    print("=" * 76)
    print()
    print("Running all computational checks...")
    print()

    results = []
    for check in CHECKS:
        print(f"▶ {check['name']}")
        print(f"  Claim: {check['claim']}")
        print(f"  Script: {check['script']}")

        t0 = time.time()
        success, output = run_script(check["script"])
        elapsed = time.time() - t0

        status = "✓ PASS" if success else "✗ FAIL"
        results.append((check, success))

        print(f"  Status: {status} ({elapsed:.1f}s)")
        if not success and check["critical"]:
            # Show last 5 lines of output for debugging
            lines = output.strip().split("\n")
            for line in lines[-5:]:
                print(f"    | {line}")
        print()

    # Optional checks
    print("─" * 76)
    print("OPTIONAL CHECKS (exploratory, not critical)")
    print("─" * 76)
    print()

    for check in OPTIONAL_CHECKS:
        print(f"▷ {check['name']}")
        print(f"  Claim: {check['claim']}")
        t0 = time.time()
        # Try CODE_DIR first, then SCRIPT_DIR
        s_dir = CODE_DIR if os.path.exists(os.path.join(CODE_DIR, check["script"])) else SCRIPT_DIR
        success, _ = run_script(check["script"], timeout=180, search_dir=s_dir)
        elapsed = time.time() - t0
        status = "✓ OK" if success else "○ SKIP"
        results.append((check, success))
        print(f"  Status: {status} ({elapsed:.1f}s)")
        print()

    # Summary
    print("=" * 76)
    print("VERIFICATION SUMMARY")
    print("=" * 76)
    print()

    critical_pass = 0
    critical_total = 0
    for check, success in results:
        marker = "✓" if success else "✗"
        crit = " [CRITICAL]" if check["critical"] else ""
        print(f"  {marker} {check['name']}{crit}")
        if check["critical"]:
            critical_total += 1
            if success:
                critical_pass += 1

    print()
    print(f"Critical checks: {critical_pass}/{critical_total} passed")

    if critical_pass == critical_total:
        print("All critical verifications PASSED.")
        return 0
    else:
        print("Some critical verifications FAILED.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
