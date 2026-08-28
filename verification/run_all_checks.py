#!/usr/bin/env python3
"""
NVG Research — Run All Verification Checks.

Executes all computational prototypes in sequence and reports process
execution status. A zero exit code is not scientific verification.

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
        "critical": True,
    },
    {
        "name": "Core EOS: Saturating Vector Sector",
        "script": "nvg_eos_beta_saturated_vector.py",
        "critical": True,
        "timeout": 300,
    },
    {
        "name": "Symmetric Matter Screening (corrected units)",
        "script": "nvg_eos_proof_checked.py",
        "critical": False,
    },
    {
        "name": "Beta-Equilibrated Core EOS",
        "script": "nvg_eos_beta_checked.py",
        "critical": False,
    },
    {
        "name": "Black Hole Interior: Mass Melting Profile",
        "script": "nvg_black_hole_interior.py",
        "critical": True,
    },
    {
        "name": "Full Neutron Star EOS (Crust + Phase Transition)",
        "script": "nvg_full_ns_eos.py",
        "critical": True,
    },
    {
        "name": "Hadron Universality & FAIR/HADES Observables",
        "script": "nvg_fair_hades_link.py",
        "critical": True,
    },
    {
        "name": "Hyperon Puzzle & Phase Transition Nature",
        "script": "nvg_hyperon_puzzle.py",
        "critical": True,
    },
    {
        "name": "Gravitational Wave Echoes",
        "script": "nvg_gw_echoes.py",
        "critical": True,
    },
    {
        "name": "Tolman Cycles & Entropy Growth",
        "script": "nvg_cyclic_lifetimes.py",
        "critical": True,
    },
    {
        "name": "Dark Photon Observables & Constraints",
        "script": "nvg_dark_photon_observables.py",
        "critical": True,
    },
    {
        "name": "Magnetar Mass-Field Correlation",
        "script": "nvg_magnetar_mass_correlation.py",
        "critical": True,
        "timeout": 300,
    },
    {
        "name": "Quantitative Physical Predictions & Validation",
        "script": "nvg_new_predictions.py",
        "critical": True,
    },
    {
        "name": "Advanced Physical Calculations (JWST, FRB, Chiral Masses, QCD Phase, SGWB, T_bounce, KATRIN)",
        "script": "nvg_advanced_calculations.py",
        "critical": True,
    },
    {
        "name": "Unified Field Equations & Core Limits",
        "script": "nvg_unified_field_equations.py",
        "critical": True,
    },
    {
        "name": "HADES Dielectron Spectral Simulation",
        "script": "nvg_hades_dielectron_sim.py",
        "critical": True,
    },
    {
        "name": "GW Echo Waveform Templates",
        "script": "nvg_gw_echo_waveforms.py",
        "critical": True,
    },
    {
        "name": "DESI DR2 Dark Energy Parametric Alignment",
        "script": "nvg_dark_energy_desi.py",
        "critical": True,
    },
    {
        "name": "Dark Energy w0-wa Parameter Derivation",
        "script": "nvg_dark_energy_w0wa.py",
        "critical": True,
    },
    {
        "name": "Tidal Deformability GW170817 & Double Pulsar MoI",
        "script": "nvg_tidal_deformability.py",
        "critical": True,
    },
    {
        "name": "JWST Early Black Hole Seeding Puzzle",
        "script": "nvg_pbh_jwst_seeds.py",
        "critical": True,
    },
    {
        "name": "Joint NS Likelihood Fit",
        "script": "nvg_joint_ns_inference.py",
        "critical": True,
    },
    {
        "name": "NICER PSR J0437 Mass-Radius Check",
        "script": "nvg_nicer_j0437_check.py",
        "critical": True,
    },
    {
        "name": "Relic Dark Matter density",
        "script": "nvg_relic_dark_matter.py",
        "critical": True,
    },
    {
        "name": "Scalar Glueball Mass from VMF",
        "script": "nvg_glueball_mass.py",
        "critical": True,
    },
    {
        "name": "Majorana Neutrino Mass from Goldstone Phase",
        "script": "nvg_neutrino_mass.py",
        "critical": True,
    },
    {
        "name": "Magnetar Crustal Starquake QPOs",
        "script": "nvg_starquake_qpo.py",
        "critical": True,
    },
    {
        "name": "Primordial GW Background Comb",
        "script": "nvg_primordial_gw_comb.py",
        "critical": True,
    },
    {
        "name": "Topological Axion Mass",
        "script": "nvg_axion_mass.py",
        "critical": True,
    },
    {
        "name": "Strong-Field Periastron Advance",
        "script": "nvg_perihelion_shift.py",
        "critical": True,
    },
    {
        "name": "CMB Temperature from QCD Bounce",
        "script": "nvg_cmb_temperature.py",
        "critical": True,
    },
    {
        "name": "Baryon Asymmetry from Genesis Bounce",
        "script": "nvg_baryon_asymmetry.py",
        "critical": True,
    },
    {
        "name": "Post-merger GW peak frequency",
        "script": "nvg_postmerger_fpeak.py",
        "critical": True,
    },
    {
        "name": "NS Surface Gravitational Redshift",
        "script": "nvg_ns_redshift.py",
        "critical": True,
    },
    {
        "name": "SGR 1935+2154 Quiescent Thermal Emission",
        "script": "nvg_sgr_temperature.py",
        "critical": True,
    },
    {
        "name": "LiteBIRD B-mode Polarisation Cutoff",
        "script": "nvg_litebird_prediction.py",
        "critical": True,
    },
    {
        "name": "Neutron Star Core Speed of Sound Curve",
        "script": "nvg_speed_of_sound_curve.py",
        "critical": True,
    },
    {
        "name": "Neutron Star Core g-mode Periods",
        "script": "nvg_ns_g_modes.py",
        "critical": True,
    },
    {
        "name": "Higgs Boson Mass Shift from QCD Condensate",
        "script": "nvg_higgs_mass_shift.py",
        "critical": True,
    },
    {
        "name": "Macroscopic Weak-Field Limit (PPN Parameters)",
        "script": "nvg_weak_field_ppn.py",
        "critical": True,
    },
    {
        "name": "PBH Dark Matter Fraction & Constraints",
        "script": "nvg_pbh_dark_matter.py",
        "critical": True,
    },
    {
        "name": "White Dwarf Cooling rate correction",
        "script": "nvg_wd_cooling.py",
        "critical": True,
    },
    {
        "name": "de Sitter Core Standing Wave Oscillations",
        "script": "nvg_ds_core_oscillations.py",
        "critical": True,
    },
    {
        "name": "Gravitational Wave Post-Merger Echo Delay",
        "script": "nvg_gw_echo_prediction.py",
        "critical": True,
    },
    {
        "name": "Hawking Temperature Ceiling",
        "script": "nvg_hayward_evaporation.py",
        "critical": True,
    },
    {
        "name": "Magnetar Core EOS and field amplification",
        "script": "nvg_magnetar_eos.py",
        "critical": True,
    },
    {
        "name": "Cyclic Cosmology Parameters",
        "script": "nvg_cyclic_cosmology.py",
        "critical": True,
    },
    {
        "name": "PBH Discrete Mass Spectrum",
        "script": "nvg_pbh_mass_spectrum.py",
        "critical": True,
    },
    {
        "name": "Direct Urca cooling threshold",
        "script": "nvg_direct_urca.py",
        "critical": True,
    },
    {
        "name": "DNA Chirality & Biological θ-Coherence",
        "script": "nvg_dna_chirality.py",
        "critical": False,
    },
    {
        "name": "CMB Low-l Re-fit with Genesis IR Cutoff",
        "script": "nvg_cmb_lowl_refit.py",
        "critical": False,
        "timeout": 600,
    },
    {
        "name": "GW Comb Tooth Amplitudes",
        "script": "nvg_gw_comb_amplitude.py",
        "critical": False,
    },
    {
        "name": "DESI x S8 Joint Exclusion Map",
        "script": "nvg_desi_s8_joint_map.py",
        "critical": False,
    },
    {
        "name": "CMB Low-ell TE Cutoff Check",
        "script": "nvg_cmb_te_check.py",
        "critical": False,
        "timeout": 600,
    },
    {
        "name": "GW Spectrum Template (bump + comb)",
        "script": "nvg_gw_spectrum_template.py",
        "critical": False,
    },
    {
        "name": "GWTC Log-2 Ladder Test (real catalog)",
        "script": "nvg_gwtc_ladder_test.py",
        "critical": False,
    },
    {
        "name": "Global Statistical Significance",
        "script": "nvg_global_significance.py",
        "critical": True,
    },
    {
        "name": "Bell from the Action — Dichotomy",
        "script": "nvg_bell_from_action.py",
        "critical": False,
    },
    {
        "name": "g = 2 Press-Schechter Distribution",
        "script": "nvg_g2_press_schechter.py",
        "critical": False,
    },
    {
        "name": "DESI DR3 Binary Forecast",
        "script": "nvg_desi_dr3_forecast.py",
        "critical": False,
    },
    {
        "name": "Falsifier Dashboard",
        "script": "nvg_falsifier_dashboard.py",
        "critical": False,
    },
    {
        "name": "Theta-Sector Identity Audit",
        "script": "nvg_theta_sector_audit.py",
        "critical": False,
    },
    {
        "name": "Dark-Matter Budget Audit",
        "script": "nvg_dm_budget_audit.py",
        "critical": False,
    },
    {
        "name": "B-L Cogenesis Construction (dark neutron)",
        "script": "nvg_adm_bl_cogenesis.py",
        "critical": False,
    },
    {
        "name": "BSM Baryogenesis Closure",
        "script": "nvg_baryogenesis_bsm_closure.py",
        "critical": False,
    },
    {
        "name": "eta_B Inheritance Closing Test",
        "script": "nvg_etab_inheritance.py",
        "critical": False,
    },
    {
        "name": "Cutoff Shape from the Action",
        "script": "nvg_cutoff_shape_derivation.py",
        "critical": False,
        "timeout": 600,
    },
    {
        "name": "Contextual Bell Construction",
        "script": "nvg_bell_contextual.py",
        "critical": False,
        "timeout": 300,
    },
    {
        "name": "g = 2 Misner-Sharp GR Verification",
        "script": "nvg_g2_misner_sharp.py",
        "critical": False,
    },
    {
        "name": "g = 2 Shock-Microphysics Closure",
        "script": "nvg_g2_shock_closure.py",
        "critical": False,
    },
    {
        "name": "g = 2 Crunch-Thermalization Theorem",
        "script": "nvg_g2_crunch_thermalization.py",
        "critical": False,
    },
    {
        "name": "g = 2 Mechanism Candidates",
        "script": "nvg_g2_mechanism.py",
        "critical": False,
    },
    {
        "name": "Two-Population PBH Abundance (JWST vs NANOGrav)",
        "script": "nvg_pbh_two_population.py",
        "critical": False,
    },
    {
        "name": "Tolman Growth Law from Turnaround Dynamics",
        "script": "nvg_tolman_law_derivation.py",
        "critical": False,
    },
    {
        "name": "Recondensation Dynamics: (alpha, beta/H) from the Action",
        "script": "nvg_recondensation_dynamics.py",
        "critical": False,
        "timeout": 300,
    },
    {
        "name": "NS Transition-Parameter Scan (provenance of the canon)",
        "script": "nvg_ns_parameter_scan.py",
        "critical": False,
        "timeout": 300,
    },
]

# The entries above intentionally contain process metadata only. A process
# return code cannot validate a scientific value; callers receive name/script/
# critical/timeout fields and must obtain semantics from each script's report.

OPTIONAL_CHECKS = [
    {
        "name": "FRW Entropy-Clock Cycle Fit (Path A)",
        "script": "nvg_frw_entropy_cycle_fit.py",
        "critical": False,
    },
    {
        "name": "Entropy-Time Cycle Toy Model",
        "script": "nvg_entropy_time_cycle_toy.py",
        "critical": False,
    },
]

def run_script(script_name: str, timeout: int = 120, search_dir: str = "") -> tuple[bool, str]:
    """Run a Python script and return (process_ok, output).

    The return value is intentionally limited to process health; semantic
    scientific conclusions require an independent contract and are not inferred here.
    """
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
    print("NVG RESEARCH — EXECUTION HARNESS (NOT SCIENTIFIC VERIFICATION)")
    print("=" * 76)
    print()
    print("Running all computational checks...")
    print()

    results = []
    for check in CHECKS:
        print(f"▶ {check['name']}")
        print("  Scientific result: not evaluated by this process runner")
        print(f"  Script: {check['script']}")

        t0 = time.time()
        success, output = run_script(check["script"], timeout=check.get("timeout", 120))
        elapsed = time.time() - t0

        status = "✓ EXECUTED" if success else "✗ PROCESS-FAIL"
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
        print("  Scientific result: not evaluated by this process runner")
        t0 = time.time()
        # Try CODE_DIR first, then SCRIPT_DIR
        s_dir = CODE_DIR if os.path.exists(os.path.join(CODE_DIR, check["script"])) else SCRIPT_DIR
        success, _ = run_script(check["script"], timeout=180, search_dir=s_dir)
        elapsed = time.time() - t0
        status = "✓ EXECUTED" if success else "○ PROCESS-SKIP"
        results.append((check, success))
        print(f"  Status: {status} ({elapsed:.1f}s)")
        print()

    # Summary
    print("=" * 76)
    print("PROCESS EXECUTION SUMMARY (NOT SCIENTIFIC VERIFICATION)")
    print("=" * 76)
    print()

    critical_pass = 0
    critical_total = 0
    for check, success in results:
        marker = "✓" if success else "✗"
        crit = " [CRITICAL]" if check["critical"] else ""
        print(f"  {marker} {check['name']}{crit}  [process return code]")
        if check["critical"]:
            critical_total += 1
            if success:
                critical_pass += 1

    print()
    print(f"Critical checks: {critical_pass}/{critical_total} passed")

    if critical_pass == critical_total:
        print("All critical scripts executed successfully; semantic scientific verification was not established by this runner.")
        return 0
    else:
        print("Some critical scripts failed to execute; no scientific conclusion is available.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
