#!/usr/bin/env python3
"""Historical conditional B-L/dark-neutron arithmetic.

This file retains a neutron-portal mass corridor and conditional numerical
estimates from an older cogenesis proposal.  It is **not** a viable empirical
baryogenesis or dark-matter mechanism: ``nvg_baryogenesis_bsm_closure.py``
has status ``RETIRED_MISSING_BARYOGENESIS_SOURCE`` because no dynamical
theta(t), CP/washout network or entropy history exists in the repository.
The printed 6.6-MeV theta variant is additionally superseded as an
identification by ``nvg_theta_sector_audit.py``.

Thus every eta_B, Lambda, branching-ratio and omega_chi number below is a
conditional algebraic sensitivity to stated inputs, not a generated
asymmetry, independently predicted abundance, working mechanism, or reason
to assign the remaining dark-matter budget to PBHs.
"""

from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
INPUT_PATH = HERE / "data" / "legacy_dark_neutron_cogenesis_inputs.json"
INPUT_SCHEMA = "legacy-dark-neutron-cogenesis-inputs.v1"
INPUT_STATUS = "LEGACY_UNDOCUMENTED_CONDITIONAL_INPUTS_NOT_MECHANISM"


def _load_historical_inputs(path: Path = INPUT_PATH) -> dict[str, Any]:
    """Expose historical constants through an explicit fail-closed passport.

    The file is intentionally *not* upgraded into a cited physical input:
    absence of provenance and of a mechanism is itself part of the data
    contract.  This loader only prevents the values from being hidden in the
    retired calculation.
    """

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot load historical cogenesis passport: {path}") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != INPUT_SCHEMA:
        raise ValueError("unrecognized historical cogenesis input schema")
    if payload.get("status") != INPUT_STATUS:
        raise ValueError("historical cogenesis input status changed; review required")
    provenance = payload.get("provenance")
    parameters = payload.get("parameters")
    if not isinstance(provenance, dict) or not isinstance(parameters, dict):
        raise ValueError("historical cogenesis passport is incomplete")
    if provenance.get("citation") is not None or provenance.get("source_url") is not None:
        raise ValueError("historical cogenesis passport must not invent source provenance")
    if not isinstance(provenance.get("limitation"), str) or not provenance["limitation"]:
        raise ValueError("historical cogenesis passport misses its limitation")
    expected_units = {
        "M_P_MeV": "MeV",
        "M_E_MeV": "MeV",
        "M_N_MeV": "MeV",
        "S_N_BE9_MeV": "MeV",
        "BETA_LAT_GeV3": "GeV^3",
        "ALPHA_EM": "dimensionless",
        "TAU_N_GEV": "GeV",
        "T_STAR_GeV": "GeV",
        "M_THETA_GeV": "GeV",
        "M_PL_GeV": "GeV",
        "G_STAR": "dimensionless",
        "ETA_TARGET": "dimensionless",
        "OMEGA_RATIO_OBS": "dimensionless",
        "OMEGA_RATIO_ERR": "dimensionless",
        "BR_BOUND_NOW": "dimensionless",
    }
    if set(parameters) != set(expected_units):
        raise ValueError("historical cogenesis passport parameter set changed")
    for key, units in expected_units.items():
        item = parameters[key]
        if not isinstance(item, dict) or item.get("units") != units or not isinstance(item.get("definition"), str):
            raise ValueError(f"historical cogenesis passport has invalid {key}")
        try:
            value = float(item.get("value"))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"historical cogenesis passport has nonnumeric {key}") from exc
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"historical cogenesis passport has nonpositive {key}")
    return payload


_INPUTS = _load_historical_inputs()
_PARAMETERS = _INPUTS["parameters"]


def _input_value(name: str) -> float:
    return float(_PARAMETERS[name]["value"])


# Compatibility aliases are retained for consumers of the mass corridor, but
# every historical number now comes from the passport above rather than an
# opaque source-level literal.
M_P = _input_value("M_P_MeV")
M_E = _input_value("M_E_MeV")
M_N = _input_value("M_N_MeV")
S_N_BE9 = _input_value("S_N_BE9_MeV")
BETA_LAT = _input_value("BETA_LAT_GeV3")
ALPHA_EM = _input_value("ALPHA_EM")
TAU_N_GEV = _input_value("TAU_N_GEV")
T_STAR = _input_value("T_STAR_GeV")
M_THETA = _input_value("M_THETA_GeV")
M_PL = _input_value("M_PL_GeV")
G_STAR = _input_value("G_STAR")
ETA_TARGET = _input_value("ETA_TARGET")
OMEGA_RATIO_OBS = _input_value("OMEGA_RATIO_OBS")
OMEGA_RATIO_ERR = _input_value("OMEGA_RATIO_ERR")
BR_BOUND_NOW = _input_value("BR_BOUND_NOW")


def historical_input_passport() -> dict[str, str]:
    """Return the explicit historical input identity for downstream audits."""

    return {
        "input_artifact": INPUT_PATH.relative_to(ROOT).as_posix(),
        "input_sha256": hashlib.sha256(INPUT_PATH.read_bytes()).hexdigest(),
        "input_status": INPUT_STATUS,
    }


def main():
    print("=" * 78)
    print("  NVG: HISTORICAL CONDITIONAL B-L/DARK-NEUTRON SENSITIVITY")
    print("  STATUS: RETIRED_MISSING_BARYOGENESIS_SOURCE")
    print(f"  INPUT PASSPORT: {historical_input_passport()['input_artifact']}")
    print("=" * 78)

    # 1. stability corridor
    lo, hi = M_N - S_N_BE9, M_P + M_E
    m_chi = 0.5 * (lo + hi)
    dm = (M_N - m_chi) * 1e-3            # GeV
    print(f"\n1. Kinematic corridor: {lo:.3f} < m_chi < {hi:.3f} MeV "
          f"(width {hi-lo:.3f} MeV)")
    print(f"   nucleons cannot decay to chi; chi absolutely stable — by masses,")
    print(f"   not by gating. Take mid-corridor m_chi = {m_chi:.3f} MeV.")
    assert hi > lo and (hi - lo) < 1.0

    # 2. Historical eta-target inversion, not a source calculation.
    s_dens = (2.0 * math.pi ** 2 / 45.0) * G_STAR * T_STAR ** 3
    eta_eq = (M_THETA * T_STAR ** 2 / 6.0) / s_dens
    eff = ETA_TARGET / eta_eq
    lam_eq = (T_STAR ** 3 * M_PL / (1.66 * math.sqrt(G_STAR))) ** 0.25
    lam = lam_eq * eff ** -0.25
    print(f"\n2. Historical eta-target inversion: bias mu/T = {M_THETA/T_STAR:.3f}, required Gamma/H = {eff:.1e}")
    print(f"   => Lambda = {lam/1e3:.0f} TeV after INPUTTING eta_B (not a generated asymmetry)")
    print("   A dynamical theta(t), CP/washout network and entropy history are absent.")

    # 3. neutron portal phenomenology at that Lambda
    theta_mix = BETA_LAT / (lam ** 2 * dm)
    br = theta_mix ** 2 * ALPHA_EM * dm ** 3 / (8.0 * (M_N * 1e-3) ** 2) / TAU_N_GEV
    print(f"\n3. Conditional free-neutron channel at that historical Lambda:")
    print(f"   n-chi mixing theta = {theta_mix:.1e}")
    print(f"   Br(n -> chi gamma) ~ {br:.1e}  (current bounds ~{BR_BOUND_NOW:.0e};")
    print(f"   a x{BR_BOUND_NOW/br:.0f} improvement would test this conditional portal estimate;")
    print("   it is not a falsifiable floor for a completed cogenesis mechanism.")
    assert br < BR_BOUND_NOW, "portal must pass current neutron bounds"

    # 4. symmetric component and the theta-Yukawa
    bias = M_THETA / T_STAR
    sym_over_asym = 1.0 / bias
    sv_needed = 2.6e-9                                  # GeV^-2 (thermal)
    y = (sv_needed * 16.0 * math.pi * (m_chi * 1e-3) ** 2) ** 0.25
    sig_self = y ** 4 * (m_chi * 1e-3) ** 2 / (4.0 * math.pi * M_THETA ** 4)
    som = sig_self * 0.389e-27 / (m_chi * 1e-3 * 1.783e-24)
    print(f"\n4. Symmetric chi component = {sym_over_asym:.0f}x the asymmetric one —")
    print(f"   the portal cannot annihilate it (computed in the closure script);")
    print(f"   an added theta-chi Yukawa would need y >= {y:.3f} in this historical variant.")
    print(f"   Induced self-interaction sigma/m = {som:.1e} cm^2/g (bound ~1) — safe.")
    assert som < 0.1

    # 5. Conditional arithmetic, not a component-abundance prediction.
    print(f"\n5. Conditional identity N_chi = N_B => per dark flavor")
    print(f"   Omega_chi/Omega_b = m_chi/m_p = {m_chi/M_P:.4f}; combined with a reference")
    print("   cosmology this would be conditional arithmetic, not a predicted DM subcomponent.")
    print(f"   Full-ADM reading (chi = ALL of DM, N flavors):")
    for N in (5, 6):
        r = N * m_chi / M_P
        pull = abs(r - OMEGA_RATIO_OBS) / OMEGA_RATIO_ERR
        print(f"     N = {N}: ratio {r:.3f} vs {OMEGA_RATIO_OBS:.3f} +/- {OMEGA_RATIO_ERR} "
              f"-> {pull:.1f} sigma — excluded")
        assert pull > 5.0
    print("   => this does not determine a chi fraction or any PBH remainder;")
    print("   Omega_DM/Omega_b = 5.364 is NOT derived by this construction.")

    print(f"""
VERDICT: this is a historical conditional arithmetic exercise, not a working
baryogenesis/cogenesis mechanism. The mass corridor is a kinematic input;
Lambda was calibrated by inputting eta_B; the 6.6-MeV theta identification is
superseded; and no source-complete dynamics calculates eta_B, a chi abundance
or a PBH remainder. The branching ratio ~{br:.0e} is therefore a conditional
portal sensitivity, not an independent laboratory prediction. A viable model
must first supply the missing source, washout and entropy calculation.
""")
    print("=" * 78)


if __name__ == "__main__":
    main()
