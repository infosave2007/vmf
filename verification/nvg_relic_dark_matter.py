#!/usr/bin/env python3
"""
NVG Verification: Relic Instanton Dark Matter Abundance
------------------------------------------------------
This historical script **inverts** a supplied full dark-matter target into a
required instanton action, vacuum self-coupling and scalar excitation mass.
It does not calculate a formation yield or predict the supplied abundance.
The two legacy calibration values are kept in a separate passport with an
explicitly undocumented-provenance status rather than hidden as source-code
constants.
"""

from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path
from typing import Any

# Physical constants (CGS units)
c = 2.99792458e10          # cm/s, speed of light
hbar = 1.0545718e-27        # erg s, Planck constant
G_Newton = 6.67430e-8       # cm^3 g^-1 s^-2, Newton's constant
k_B = 1.380649e-16          # erg/K, Boltzmann constant
MeV_to_erg = 1.60217663e-6  # erg/MeV

# QCD Vacuum parameters
M_omega_0 = 859.0           # MeV, lattice QCD vacuum mass anchor
T_c = 157.3                 # MeV, QCD transition temperature
g_star_S_T0 = 3.91          # entropy degrees of freedom today
g_star_S_Tc = 17.25         # entropy degrees of freedom at T_c

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CALIBRATION_PATH = HERE / "data" / "legacy_relic_dark_matter_calibration.json"
CALIBRATION_SCHEMA_VERSION = "legacy-relic-dark-matter-calibration.v1"
CALIBRATION_STATUS = "LEGACY_UNDOCUMENTED_CALIBRATION_NOT_ABUNDANCE_PREDICTION"


def _load_legacy_calibration(path: Path = CALIBRATION_PATH) -> dict[str, Any]:
    """Load the deliberately non-predictive legacy calibration passport."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != CALIBRATION_SCHEMA_VERSION:
        raise ValueError("unrecognized relic calibration schema")
    if payload.get("calibration_status") != CALIBRATION_STATUS:
        raise ValueError("legacy relic calibration status changed; review required")
    provenance = payload.get("provenance")
    parameters = payload.get("parameters")
    if not isinstance(provenance, dict) or not isinstance(parameters, dict):
        raise ValueError("relic calibration passport is incomplete")
    if provenance.get("source_url") is not None or provenance.get("citation") is not None:
        raise ValueError("legacy relic calibration must not invent source provenance")
    if not isinstance(provenance.get("limitation"), str) or not provenance["limitation"]:
        raise ValueError("legacy relic calibration misses its provenance limitation")
    expectations = {
        "H0_km_s_Mpc": "km s^-1 Mpc^-1",
        "target_Omega_DM": "dimensionless",
    }
    for key, units in expectations.items():
        item = parameters.get(key)
        if not isinstance(item, dict) or item.get("units") != units or not isinstance(item.get("definition"), str):
            raise ValueError(f"legacy relic calibration misses passport for {key}")
        try:
            value = float(item.get("value"))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"legacy relic calibration has invalid {key}") from exc
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"legacy relic calibration has nonpositive {key}")
        if item.get("uncertainty_68_percent") is not None:
            raise ValueError("legacy relic calibration must not invent an uncertainty")
    return payload


_CALIBRATION = _load_legacy_calibration()
# Compatibility names are loaded data, not hidden numerical inputs.  They are
# retained for historical callers while run_dm_verification returns the same
# passport values for audit-level data-flow checks.
H_0_KM_S_MPC = float(_CALIBRATION["parameters"]["H0_km_s_Mpc"]["value"])
TARGET_OMEGA_DM = float(_CALIBRATION["parameters"]["target_Omega_DM"]["value"])

def run_dm_verification() -> dict[str, float | str]:
    # 1. Critical density at the bounce (MeV/fm^3)
    hbar_c_MeV_fm = 197.326979
    rho_c_MeV_fm3 = (M_omega_0)**4 / (hbar_c_MeV_fm)**3
    rho_c_cgs = rho_c_MeV_fm3 * MeV_to_erg / 1e-39 # erg/cm^3
    
    # 2. Today's CMB temperature and scale factor ratio
    T_0_K = 2.7255 # K, CMB temperature today
    T_0_MeV = T_0_K * k_B / MeV_to_erg
    
    # Entropy conservation: a_c / a_0 = (T_0 / T_c) * (g_*s(T_0) / g_*s(T_c))^(1/3)
    a_ratio = (T_0_MeV / T_c) * (g_star_S_T0 / g_star_S_Tc)**(1.0/3.0)
    
    # 3. Today's critical density
    Mpc_to_cm = 3.085677581e24
    H_0_cgs = (H_0_KM_S_MPC * 1e5) / Mpc_to_cm # s^-1
    rho_crit_T0_cgs = (3.0 * H_0_cgs**2) * c**2 / (8.0 * math.pi * G_Newton) # erg/cm^3
    
    # 4. Target relic density today (Omega_DM = 0.268)
    rho_inst_T0_cgs = TARGET_OMEGA_DM * rho_crit_T0_cgs
    
    # 5. Required instanton density at freeze-out T_c
    rho_inst_Tc_cgs = rho_inst_T0_cgs / (a_ratio**3)
    
    # 6. Required instanton suppression factor
    # rho_inst(T_c) = (4*pi/3) * rho_c * exp(-S_inst)
    required_suppression = rho_inst_Tc_cgs / ((4.0 * math.pi / 3.0) * rho_c_cgs)
    required_S_inst = -math.log(required_suppression)
    
    # 7. Required VMF self-coupling lambda_v
    # S_inst = 8*pi^2 / (3 * lambda_v)
    required_lambda_v = (8.0 * math.pi**2) / (3.0 * required_S_inst)
    
    # 8. Vacuum excitation mass m_W
    # m_W = sqrt(2 * lambda_v) * M_omega_0
    m_W = math.sqrt(2.0 * required_lambda_v) * M_omega_0
    
    return {
        "rho_c_MeV_fm3": rho_c_MeV_fm3,
        "a_ratio": a_ratio,
        "required_S_inst": required_S_inst,
        "required_lambda_v": required_lambda_v,
        "m_W": m_W,
        "Omega_DM": TARGET_OMEGA_DM,
        "calibration_H0_km_s_Mpc": H_0_KM_S_MPC,
        "calibration_target_Omega_DM": TARGET_OMEGA_DM,
        "calibration_target_omega_h2": TARGET_OMEGA_DM * (H_0_KM_S_MPC / 100.0) ** 2,
        "calibration_status": CALIBRATION_STATUS,
        "calibration_input_artifact": CALIBRATION_PATH.relative_to(ROOT).as_posix(),
        "calibration_input_sha256": hashlib.sha256(CALIBRATION_PATH.read_bytes()).hexdigest(),
    }

if __name__ == "__main__":
    res = run_dm_verification()
    print("=" * 70)
    print("      NVG RELIC INSTANTON DARK MATTER INFERENCE")
    print("=" * 70)
    print(f"QCD Anchor scale M_omega_0:       {M_omega_0:.1f} MeV")
    print(f"Bounce critical density rho_c:     {res['rho_c_MeV_fm3']:.2e} MeV/fm3")
    print(f"Required instanton action S_inst:  {res['required_S_inst']:.2f}")
    print(f"Inferred VMF self-coupling λ_v:    {res['required_lambda_v']:.4f}")
    print("-" * 70)
    print(f"Inferred excitation mass m_W:      {res['m_W']:.1f} MeV ({res['m_W']/1000.0:.3f} GeV)")
    print("Reference meson:                  f_0(1370) / f_0(1500) scalar range")
    print("-" * 70)
    print(f"Relic abundance Omega_DM = {res['Omega_DM']:.4f} — LEGACY CALIBRATION INPUT, not a")
    print("prediction: this script INVERTS the supplied target to infer lambda_v/m_W.")
    print(f"Calibration passport:                 {res['calibration_input_artifact']}")
    print(f"Calibration status:                   {res['calibration_status']}")
    print("STATUS: ⚠️ LEGACY CALIBRATION (inferred self-coupling lands in the f_0 scalar range —")
    print("        that consistency, not Omega_DM itself, is the checkable content)")
    print("=" * 70)
