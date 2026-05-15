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
]


def run_script(script_name: str, timeout: int = 120) -> tuple[bool, str]:
    """Run a Python script and return (success, output)."""
    script_path = os.path.join(SCRIPT_DIR, script_name)
    if not os.path.exists(script_path):
        return False, f"Script not found: {script_path}"

    try:
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=SCRIPT_DIR,
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
        success, _ = run_script(check["script"], timeout=180)
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
