#!/usr/bin/env python3
"""
NVG Hadron Mass Fractions Calculator.

Computes the nonperturbative (vacuum-structured) mass fraction f_Omega
for nucleons and other hadrons using lattice QCD sigma terms.

This is the core verifiable calculation of the NVG program:
  f_Omega = 1 - sum(sigma_qH) / M_H

Reference values:
  sigma_piN = 44 ± 3 MeV  [Gupta et al., PRL 127, 242002 (2021)]
  sigma_sN  = 30 ± 7 MeV  [Agadjanov et al., PRL 131, 261902 (2023)]
"""

from __future__ import annotations
import math
from dataclasses import dataclass


@dataclass
class SigmaTerms:
    """Sigma terms for a hadron."""
    sigma_piH: float       # pion-hadron sigma term (MeV)
    sigma_piH_err: float
    sigma_sH: float        # strange sigma term (MeV)
    sigma_sH_err: float
    sigma_heavy: float     # matched heavy-quark contribution (MeV)
    sigma_heavy_err: float


@dataclass
class HadronMassBudget:
    """Complete mass budget for a hadron."""
    name: str
    mass_MeV: float
    sigma: SigmaTerms
    M_current: float
    M_current_err: float
    M_Omega: float
    M_Omega_err: float
    f_Omega: float
    f_Omega_err: float


def compute_mass_budget(
    name: str,
    mass_MeV: float,
    sigma: SigmaTerms,
) -> HadronMassBudget:
    """Compute the vacuum-structured mass fraction for a hadron."""
    M_current = sigma.sigma_piH + sigma.sigma_sH + sigma.sigma_heavy
    M_current_err = math.sqrt(
        sigma.sigma_piH_err**2
        + sigma.sigma_sH_err**2
        + sigma.sigma_heavy_err**2
    )
    M_Omega = mass_MeV - M_current
    M_Omega_err = M_current_err  # error propagation (M_H is exact)

    f_Omega = 1.0 - M_current / mass_MeV
    f_Omega_err = M_current_err / mass_MeV

    return HadronMassBudget(
        name=name,
        mass_MeV=mass_MeV,
        sigma=sigma,
        M_current=M_current,
        M_current_err=M_current_err,
        M_Omega=M_Omega,
        M_Omega_err=M_Omega_err,
        f_Omega=f_Omega,
        f_Omega_err=f_Omega_err,
    )


# ── Lattice QCD input data ──────────────────────────────────────────

NUCLEON_SIGMA = SigmaTerms(
    sigma_piH=44.0, sigma_piH_err=3.0,     # Gupta et al. (2021)
    sigma_sH=30.0,  sigma_sH_err=7.0,      # Agadjanov et al. (2023)
    sigma_heavy=6.0, sigma_heavy_err=3.0,   # PDG heavy-quark matching
)

# Broader literature range for systematic check
NUCLEON_SIGMA_BROAD = SigmaTerms(
    sigma_piH=51.0, sigma_piH_err=9.0,     # Upper range from various analyses
    sigma_sH=33.0,  sigma_sH_err=10.0,
    sigma_heavy=6.0, sigma_heavy_err=3.0,
)

# Pion: current-mass dominated (as a cross-check)
PION_SIGMA = SigmaTerms(
    sigma_piH=67.0, sigma_piH_err=5.0,     # Nearly all mass is current
    sigma_sH=0.0,   sigma_sH_err=0.0,
    sigma_heavy=0.0, sigma_heavy_err=0.0,
)

# Kaon
KAON_SIGMA = SigmaTerms(
    sigma_piH=22.0, sigma_piH_err=3.0,
    sigma_sH=340.0, sigma_sH_err=20.0,     # Dominant strange content
    sigma_heavy=0.0, sigma_heavy_err=0.0,
)


def print_budget(budget: HadronMassBudget) -> None:
    """Pretty-print a hadron mass budget."""
    print(f"  {budget.name}")
    print(f"    M_H              = {budget.mass_MeV:.1f} MeV")
    print(f"    σ_πH             = {budget.sigma.sigma_piH:.1f} ± {budget.sigma.sigma_piH_err:.1f} MeV")
    print(f"    σ_sH             = {budget.sigma.sigma_sH:.1f} ± {budget.sigma.sigma_sH_err:.1f} MeV")
    print(f"    σ_heavy          = {budget.sigma.sigma_heavy:.1f} ± {budget.sigma.sigma_heavy_err:.1f} MeV")
    print(f"    M_current        = {budget.M_current:.1f} ± {budget.M_current_err:.1f} MeV")
    print(f"    M_Ω              = {budget.M_Omega:.1f} ± {budget.M_Omega_err:.1f} MeV")
    print(f"    f_Ω              = {budget.f_Omega:.4f} ± {budget.f_Omega_err:.4f}")
    print(f"    Interpretation   : {budget.f_Omega*100:.1f}% of mass is nonperturbative QCD")
    print()


def main() -> None:
    print("=" * 72)
    print("NVG HADRON MASS FRACTIONS — LATTICE QCD VERIFICATION")
    print("=" * 72)
    print()
    print("Reference: f_Ω = 1 - Σ σ_qH / M_H")
    print("Sources:   Gupta et al. PRL 127, 242002 (2021)")
    print("           Agadjanov et al. PRL 131, 261902 (2023)")
    print("           PDG Review of Particle Physics (2024)")
    print()

    # Primary nucleon calculation
    nucleon = compute_mass_budget("Nucleon (primary)", 939.0, NUCLEON_SIGMA)
    nucleon_broad = compute_mass_budget("Nucleon (broad range)", 939.0, NUCLEON_SIGMA_BROAD)
    pion = compute_mass_budget("Pion (cross-check)", 139.6, PION_SIGMA)
    kaon = compute_mass_budget("Kaon", 493.7, KAON_SIGMA)

    print("─" * 72)
    print("RESULTS")
    print("─" * 72)
    print()

    print_budget(nucleon)
    print_budget(nucleon_broad)
    print_budget(pion)
    print_budget(kaon)

    # Summary table
    print("─" * 72)
    print("SUMMARY TABLE")
    print("─" * 72)
    print(f"  {'Hadron':<25} {'M_H (MeV)':<12} {'f_Ω':<16} {'Interpretation'}")
    print(f"  {'─'*25} {'─'*12} {'─'*16} {'─'*25}")
    for b in [nucleon, nucleon_broad, pion, kaon]:
        interp = f"{b.f_Omega*100:.1f}% nonperturbative"
        print(f"  {b.name:<25} {b.mass_MeV:<12.1f} {b.f_Omega:.4f} ± {b.f_Omega_err:.4f}  {interp}")

    print()
    print("─" * 72)
    print("KEY RESULT:")
    print(f"  M_Ω,0 (nucleon) = {nucleon.M_Omega:.0f} ± {nucleon.M_Omega_err:.0f} MeV")
    print(f"  f_Ω   (nucleon) = {nucleon.f_Omega:.4f} ± {nucleon.f_Omega_err:.4f}")
    print()
    print("  ~91% of the nucleon mass is NOT from Higgs-generated quark masses.")
    print("  It is nonperturbative QCD energy: confinement + gluons + trace anomaly.")
    print("─" * 72)


if __name__ == "__main__":
    main()
