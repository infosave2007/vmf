#!/usr/bin/env python3
"""Phase-3 cooling and composition g-mode closure/no-go audit.

This entry point is intentionally an audit, not a new cooling model.  It
reconstructs the dependency boundary of the existing cooling/g-mode routes,
derives the Newtonian and relativistic Schwarzschild discriminants, and runs
an analytic/canonical barotropic null control.  A cold barotropic EOS supplies
only an equilibrium derivative ``dP/depsilon``; it does not supply the
frozen-composition adiabatic index ``Gamma_1`` or any thermal microphysics.

No legacy cooling array is imported as evidence.  The only model input used by
the numerical null control is the read-only canonical EOS producer, and it is
labelled conditional/in-sample.  P2-S2 is read as a JSON reference and remains
the declared zero-interaction ``n p Lambda e mu`` reference only.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
VERIFICATION = ROOT / "verification"
RESULT_PATH = VERIFICATION / "nvg_cooling_gmode_dependency_audit_p3s2_results.json"
FIGURE_PATH = VERIFICATION / "fig_p3_s2_cooling_gmode_dependency_audit.png"
REPORT_PATH = ROOT / "Lunacy/runs/predictive-research/phases/phase-3/P3-S6-REPORT.md"
EVIDENCE_PATH = ROOT / "Lunacy/runs/predictive-research/phases/phase-3/evidence/P3-S6-terminal-verification.log"
S2_REPORT_PATH = ROOT / "Lunacy/runs/predictive-research/phases/phase-3/P3-S2-REPORT.md"
P2_RESULT_PATH = VERIFICATION / "nvg_beta_hyperon_urca_audit_p2s2_results.json"
CANONICAL_RESULT_PATH = VERIFICATION / "nvg_ns_predictive_audit_results.json"

PAYLOAD_INTEGRITY_SCHEMA = "P3-S6-complete-payload-v1"
PAYLOAD_INTEGRITY_EXCLUDED_FIELDS = ("payload_integrity",)

M_SUN_KM = 1.4766
C_LIGHT_KM_S = 299792.458

CANONICAL_REGRESSION = {
    "M_max": 2.047950740578197,
    "R_1.4": 12.550001000000044,
    "Lambda_1.4": 519.4223807918132,
}


# These are the maintained cooling/mode/composition surfaces found by the
# source audit.  Their existing files remain read-only; line references make
# the static/illustrative inputs explicit in the machine-readable ledger.
_CLAIM_SPECS: tuple[dict[str, Any], ...] = (
    {
        "claim_id": "ns_gmode_wkb",
        "surface": "verification/nvg_ns_g_modes.py",
        "kind": "g_mode",
        "claim": "WKB l=2 composition g-mode periods (approximately 72--93 ms).",
        "legacy_status": "FORECAST_ASSUMED_COMPOSITION_NO_INDEPENDENT_G_MODE_LIKELIHOOD",
        "audit_status": "BLOCKED_MISSING_GAMMA1_COMPOSITION_DERIVATIVES",
        "static_inputs": [
            "delta_comp=[1.5e-4,2.0e-4,2.5e-4] is an assumed sensitivity grid",
            "fixed WKB profile step dr=0.01 km and c=2.99792e5 km/s",
            "canonical TOV background is not a composition solver",
        ],
        "source_lines": "verification/nvg_ns_g_modes.py:84-131",
        "missing_producers": [
            "none: traceable composition Y_i(n_b,T) and its derivatives",
            "none: frozen-composition Gamma_1=(epsilon+P)/P*(dP/d epsilon)_{s,Y_i}",
            "none: independent g-mode detector/oscillation likelihood",
        ],
        "physical_result": False,
        "frequency_from_sourced_microphysics": False,
    },
    {
        "claim_id": "cas_a_cooling_curve",
        "surface": "verification/nvg_cas_a_cooling_curve.py",
        "kind": "neutron_star_cooling",
        "claim": "Cas A cooling slope compared with illustrative modified/direct-Urca curves.",
        "legacy_status": "CALIBRATED_MODEL_COMPARISON",
        "audit_status": "BLOCKED_MISSING_COOLING_CLOSURE",
        "static_inputs": [
            "11-point Chandra temperature/error list is embedded in get_chandra_data",
            "slopes -3650 and -25000 K/yr and T_s normalization 2.010 MK are hard-coded",
            "linear curves are calibration/sensitivity formulas, not a thermal evolution",
        ],
        "source_lines": "verification/nvg_cas_a_cooling_curve.py:40-115,139-184",
        "missing_producers": [
            "none: beta-equilibrated composition and stellar radial profile",
            "none: heat capacities and thermal conductivity",
            "none: neutrino channels, weak matrix elements and temperature dependence",
            "none: superfluid critical temperatures/gaps and suppression factors",
            "none: envelope T_b--T_s relation and carbon-atmosphere spectral likelihood",
        ],
        "physical_result": False,
        "frequency_from_sourced_microphysics": None,
    },
    {
        "claim_id": "observational_cooling_rows",
        "surface": "verification/nvg_observational_data_fit.py",
        "kind": "neutron_star_cooling",
        "claim": "Cas A/Vela cooling rows retained in the observational comparison helper.",
        "legacy_status": "UNVERIFIED_INPUT_COMPARISON",
        "audit_status": "BLOCKED_NO_COOLING_LIKELIHOOD",
        "static_inputs": [
            "Cas A/Vela ages, temperatures and slopes are literal calibration values",
            "input provenance is explicitly unknown_provenance",
        ],
        "source_lines": "verification/nvg_observational_data_fit.py:107-118",
        "missing_producers": [
            "none: versioned spectral data and covariance/likelihood",
            "none: thermal-evolution solver with the required microphysics",
        ],
        "physical_result": False,
        "frequency_from_sourced_microphysics": None,
    },
    {
        "claim_id": "pedagogical_cooling_ode",
        "surface": "verification/nvg_cooling_dark_matter.py",
        "kind": "neutron_star_cooling",
        "claim": "Pedagogical modified/direct-Urca cooling ODE for a 1.8 Msun, 11.5 km case.",
        "legacy_status": "SENSITIVITY_ONLY_FRACTIONS_NOT_INFERRED",
        "audit_status": "SENSITIVITY_ONLY_BLOCKED_PHYSICAL_COOLING",
        "static_inputs": [
            "L_mu=1e21*M*T9^8, L_DU=1e27*M*f_DU*T9^6",
            "C=1e39*M*T9 and T_s=1e6*(10*T9)^0.55 are pedagogical benchmarks",
            "M=1.8 Msun, R=11.5 km and f_DU=[0,0.5,1] are not inferred",
        ],
        "source_lines": "verification/nvg_cooling_dark_matter.py:18-45,57-70",
        "missing_producers": [
            "none: stellar density/composition profile and DU volume",
            "none: heat capacity, conductivity and neutrino emissivities from microphysics",
            "none: superfluid gaps, envelope relation and boundary conditions",
        ],
        "physical_result": False,
        "frequency_from_sourced_microphysics": None,
    },
    {
        "claim_id": "cyclic_cooling_consumer",
        "surface": "verification/nvg_cmb_smbh_cyclic.py",
        "kind": "neutron_star_cooling",
        "claim": "Cooling curves consumed from the pedagogical ODE as a sensitivity example.",
        "legacy_status": "SENSITIVITY_ONLY_NO_URCA_THRESHOLD_INFERENCE",
        "audit_status": "BLOCKED_INHERITED_PEDAGOGICAL_COOLING",
        "static_inputs": [
            "fractions=(0,1) and M=1.8 Msun,R=11.5 km are template inputs",
            "the consumer imports no physical composition or emissivity producer",
        ],
        "source_lines": "verification/nvg_cmb_smbh_cyclic.py:18-67",
        "missing_producers": [
            "verification/nvg_cooling_dark_matter.py is only a pedagogical ODE",
            "none: physical thermal-evolution closure and observational likelihood",
        ],
        "physical_result": False,
        "frequency_from_sourced_microphysics": None,
    },
    {
        "claim_id": "direct_urca_threshold",
        "surface": "verification/nvg_direct_urca.py",
        "kind": "direct_urca",
        "claim": "VMF illustrative beta-equilibrium proton-fraction scan and Urca kinematic label.",
        "legacy_status": "DESCRIPTIVE_PROTON_FRACTION_SCAN",
        "audit_status": "BLOCKED_MISSING_INTERACTING_EOS_AND_EMISSIVITY",
        "static_inputs": [
            "k1,k2,c_s,c_rho,alpha_v,nu_v are illustrative coupling choices",
            "ratios=[1,2,3,4,5] and the Lattimer triangle are kinematic controls only",
            "OPEN (Fast Cooling) is not a luminosity or cooling-curve calculation",
        ],
        "source_lines": "verification/nvg_direct_urca.py:36-44,299-318",
        "missing_producers": [
            "none: traceable interacting beta-equilibrated EOS and stellar mass mapping",
            "none: weak matrix elements, in-medium dispersion and temperature dependence",
            "none: superfluid suppression, heat capacity, transport and envelope",
        ],
        "physical_result": False,
        "frequency_from_sourced_microphysics": None,
    },
    {
        "claim_id": "p2_zero_interaction_reference",
        "surface": "verification/nvg_beta_hyperon_urca_audit.py",
        "kind": "composition_urca",
        "claim": "Cold ideal n-p-Lambda-e-mu composition and kinematic triangle reference.",
        "legacy_status": "COMPLETE_WITH_BLOCKED_PHYSICAL_HYPERON_PHASE_URCA_DEPENDENCIES",
        "audit_status": "DERIVED_REFERENCE_ONLY_PHYSICAL_COOLING_BLOCKED",
        "static_inputs": [
            "all interaction couplings are exactly zero and T=0",
            "P2-S2 density grid and conventional particle masses are reference controls",
        ],
        "source_lines": "verification/nvg_beta_hyperon_urca_audit.py:1-15,620-631,718-776",
        "missing_producers": [
            "none: traceable hyperon couplings and NVG dependence",
            "none: phase coexistence construction and stable TOV branch",
            "none: weak emissivity, temperature, pairing and cooling transport",
        ],
        "physical_result": False,
        "frequency_from_sourced_microphysics": None,
    },
    {
        "claim_id": "sgr_thermal_consistency",
        "surface": "verification/nvg_sgr_temperature.py",
        "kind": "neutron_star_thermal",
        "claim": "SGR 1935 spot-heating/envelope consistency numbers.",
        "legacy_status": "CALIBRATED_CONSISTENCY_ONLY",
        "audit_status": "BLOCKED_MISSING_MAGNETAR_THERMAL_CLOSURE",
        "static_inputs": [
            "T_obs,L_obs,masses,core temperatures, radii and l_heat=1.1e34 are hard-coded",
            "T_core-to-surface exponent 0.55 and heavy-star 0.10 factor are illustrative",
        ],
        "source_lines": "verification/nvg_sgr_temperature.py:17-61",
        "missing_producers": [
            "none: magnetar thermal-evolution/field-decay solver",
            "none: crust/core heat capacity, conductivity, neutrino and pairing inputs",
            "none: envelope/atmosphere transfer and XMM spectral likelihood",
        ],
        "physical_result": False,
        "frequency_from_sourced_microphysics": None,
    },
    {
        "claim_id": "pulsar_population_cooling",
        "surface": "verification/nvg_pulsar_population_test.py",
        "kind": "neutron_star_cooling_population",
        "claim": "ATNF cooling-population dichotomy/falsifier test.",
        "legacy_status": "RETIRED_NO_ATNF_MASS_AGE_LUMINOSITY_LIKELIHOOD",
        "audit_status": "RETIRED_MISSING_MASS_AGE_ENVELOPE_LIKELIHOOD",
        "static_inputs": ["No synthetic population or numerical luminosity rows are emitted."],
        "source_lines": "verification/nvg_pulsar_population_test.py:1-27",
        "missing_producers": [
            "none: joint ATNF mass-age-envelope-linked thermal luminosity catalogue",
            "none: selection-corrected cooling likelihood",
        ],
        "physical_result": False,
        "frequency_from_sourced_microphysics": None,
    },
    {
        "claim_id": "advanced_observables_cooling",
        "surface": "verification/nvg_advanced_observables_III.py",
        "kind": "neutron_star_cooling_population",
        "claim": "Legacy population-level cooling conclusion.",
        "legacy_status": "RETIRED_NO_INDEPENDENT_COOLING_POPULATION_OR_LIKELIHOOD",
        "audit_status": "RETIRED_MISSING_COOLING_POPULATION_LIKELIHOOD",
        "static_inputs": ["Rows are empty by design; no luminosity population is synthesized."],
        "source_lines": "verification/nvg_advanced_observables_III.py:1-8,17-23,57-69",
        "missing_producers": [
            "none: mass-, age-, envelope- and luminosity-linked pulsar catalogue",
            "none: independent cooling likelihood",
        ],
        "physical_result": False,
        "frequency_from_sourced_microphysics": None,
    },
    {
        "claim_id": "wd_cooling_sensitivity",
        "surface": "verification/nvg_wd_cooling.py",
        "kind": "white_dwarf_cooling",
        "claim": "Mass-dependent VMF white-dwarf cooling-age sensitivity grid.",
        "legacy_status": "MODEL_SENSITIVITY_ONLY",
        "audit_status": "SENSITIVITY_ONLY_NO_WD_COOLING_SOLVER",
        "static_inputs": [
            "MASS_GRID=(0.4,0.6,0.8,1.0,1.2), DELTA_W_0=1.2e-6 and ALPHA_VMF=1.5",
            "age_shift is an algebraic proxy; no Gaia/SDSS likelihood is loaded",
        ],
        "source_lines": "verification/nvg_wd_cooling.py:15-53",
        "missing_producers": [
            "none: white-dwarf thermal/evolution solver and crystallization microphysics",
            "none: selection-corrected Gaia/SDSS cooling-age catalogue likelihood",
        ],
        "physical_result": False,
        "frequency_from_sourced_microphysics": None,
    },
    {
        "claim_id": "lmxb_crust_cooling",
        "surface": "verification/nvg_new_predictions.py",
        "kind": "neutron_star_cooling",
        "claim": "LMXB crust dark-photon cooling suppression versus standard cooling.",
        "legacy_status": "CONDITIONAL_MODEL_OUTPUT",
        "audit_status": "BLOCKED_MISSING_CRUST_TRANSPORT_CLOSURE",
        "static_inputs": [
            "n_B=0.5 n0,T=10 keV and the (T/m_A')^4 suppression are benchmark inputs",
            "standard-cooling denominator has no calculated neutrino/thermal profile",
        ],
        "source_lines": "verification/nvg_new_predictions.py:110-137,173-181",
        "missing_producers": [
            "none: accreted-crust composition, heat capacity and conductivity profile",
            "none: neutrino/dark-photon matrix elements and temperature-dependent transport",
            "none: envelope boundary condition and observed LMXB thermal likelihood",
        ],
        "physical_result": False,
        "frequency_from_sourced_microphysics": None,
    },
    {
        "claim_id": "sn1987a_cooling_scan",
        "surface": "verification/nvg_dark_photon_observables.py",
        "kind": "proto_neutron_star_cooling",
        "claim": "SN1987A dark-photon luminosity scan against a Raffelt reference.",
        "legacy_status": "CONDITIONAL_BENCHMARK_NO_COMPLETE_LIKELIHOOD",
        "audit_status": "BLOCKED_MISSING_SN_TRANSPORT_AND_LIKELIHOOD",
        "static_inputs": [
            "T=30 MeV, r_core=12 km, g_pi_N=14, m_A' list and L_ref=3e52 erg/s are benchmarks",
            "single-zone bremsstrahlung expression is not a proto-NS transport solver",
        ],
        "source_lines": "verification/nvg_dark_photon_observables.py:150-172,253-269",
        "missing_producers": [
            "none: sourced proto-neutron-star density/temperature/time profile",
            "none: NN bremsstrahlung matrix elements, blocking and opacity/transport",
            "none: SN1987A neutrino/light-curve likelihood and nuisance model",
        ],
        "physical_result": False,
        "frequency_from_sourced_microphysics": None,
    },
    {
        "claim_id": "dark_photon_stellar_cooling",
        "surface": "verification/nvg_dark_photon_kinetic_mixing.py",
        "kind": "stellar_cooling",
        "claim": "Density-suppressed dark-photon stellar-cooling compatibility statements.",
        "legacy_status": "CONDITIONAL_BENCHMARK_NO_COMPLETE_LIKELIHOOD",
        "audit_status": "BLOCKED_NO_GLOBAL_STELLAR_COOLING_LIKELIHOOD",
        "static_inputs": [
            "ENVIRONMENTS table and representative published eps bounds are static overlays",
            "statement 'no signal in any stellar cooling channel below 0.5 n0' is not a transport result",
        ],
        "source_lines": "verification/nvg_dark_photon_kinetic_mixing.py:103-137,262-286",
        "missing_producers": [
            "none: environment-specific stellar structure and production/escape transport",
            "none: sourced event likelihoods and correlated systematics across bounds",
        ],
        "physical_result": False,
        "frequency_from_sourced_microphysics": None,
    },
    {
        "claim_id": "joint_ns_cooling_calibration",
        "surface": "verification/nvg_joint_ns_inference.py",
        "kind": "neutron_star_cooling",
        "claim": "Cas A/Vela cooling dichotomy retained as a calibration row.",
        "legacy_status": "CALIBRATION_TARGET_EXCLUDED_FROM_CHI2",
        "audit_status": "BLOCKED_CALIBRATION_NOT_PREDICTION",
        "static_inputs": [
            "Cooling_Dichotomy row is excluded from chi-squared and has no independent solver",
            "alpha_v/direct-Urca threshold is explicitly a calibration parameter",
        ],
        "source_lines": "verification/nvg_joint_ns_inference.py:1-8,39-70,108-120",
        "missing_producers": [
            "none: independent thermal-evolution calculation and cooling likelihood",
        ],
        "physical_result": False,
        "frequency_from_sourced_microphysics": None,
    },
)


# Public/article text is read-only in Phase 3, but it is still part of the
# claim inventory.  Scan prose separately from executable surfaces so a stale
# sentence (for example a "parameter-free" Cas A match) cannot disappear from
# the dependency ledger merely because the runtime producer was repaired.
_DOCUMENT_SUFFIXES = {".md", ".tex", ".html", ".txt"}
_DOCUMENT_ROOTS = (
    ROOT / "README.md",
    ROOT / "README_RU.md",
    ROOT / "article",
    ROOT / "verification",
    ROOT / "generator_project",
)
_DOCUMENT_CLAIM_RE = re.compile(
    r"(?i)(cooling|g[- ]?mode|gmode|direct\s+urca|cas\s*a|cassiopeia|\bvela\b|sn1987a|lmxb)"
)


def _tracked_document_paths() -> set[Path]:
    """Candidate prose surfaces restricted to the tracked public tree.

    ``_DOCUMENT_ROOTS`` contains directories that also hold local-only drafts
    and ignored generated reports.  The scan is embedded verbatim in the P3-S6
    artifact and deterministically regenerated on every front-door run, so it
    must be identical wherever the front door runs (CI checkout or developer
    tree).  Local-only files are not public claim surfaces and are excluded by
    enumerating candidates through ``git ls-files`` (the same tracked-tree
    idiom as the registry).  Fail closed when git is unavailable: a silent
    disk fallback would let the artifact drift between environments.
    """

    paths: set[Path] = set()
    for root in _DOCUMENT_ROOTS:
        spec = root.relative_to(ROOT).as_posix()
        proc = subprocess.run(
            ["git", "ls-files", "-z", "--", spec],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=False,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                "P3-S6 documentation scan requires git to enumerate tracked surfaces"
            )
        for item in proc.stdout.decode("utf-8", errors="replace").split("\0"):
            if not item:
                continue
            path = ROOT / item
            if path.is_file() and path.suffix.lower() in _DOCUMENT_SUFFIXES:
                paths.add(path)
    return paths


def _document_claim_scan() -> list[dict[str, Any]]:
    """Inventory prose/static-output claim lines without importing them."""

    paths: set[Path] = _tracked_document_paths()
    rows: list[dict[str, Any]] = []
    for path in sorted(paths):
        # Avoid counting this audit's generated JSON/report text as legacy
        # claims if the scan is run after artifact generation.
        if (
            path.name.startswith("nvg_cooling_gmode_dependency_audit")
            or path.name.startswith("P3-S2")
            or path.name.startswith("P3-S6")
            or "node_modules" in path.parts
        ):
            continue
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeDecodeError):
            continue
        hits = [
            {"line": int(index), "text": line.strip()[:240]}
            for index, line in enumerate(lines, 1)
            if _DOCUMENT_CLAIM_RE.search(line)
        ]
        if not hits:
            continue
        relative = path.relative_to(ROOT).as_posix()
        # A text source can contain an honest boundary statement or an
        # overclaim.  We do not infer truth from wording; every hit is zero
        # weight and points to the same missing physical producers.
        rows.append(
            {
                "surface": relative,
                "sha256": _sha256(path),
                "matching_line_count": len(hits),
                "matching_lines": hits[:40],
                "status": "STATIC_TEXT_ZERO_EVIDENCE_REVIEWED",
                "evidence_weight": 0.0,
                "missing_producers": [
                    "none: sourced cooling/g-mode composition and thermal likelihood",
                    "none: Gamma_1/composition derivatives for any frozen-composition g-mode",
                ],
            }
        )
    return rows


def _sha256(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _jsonable(value: Any) -> Any:
    """Convert the complete audit payload to a finite JSON value.

    The result contains NumPy arrays in memory but JSON lists on disk.  A
    single normalizer is therefore required for a digest that is invariant to
    that representation boundary.  Non-finite numbers are rejected instead
    of being silently serialized as non-standard ``NaN``/``Infinity``.
    """

    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("payload contains a non-finite float")
        return value
    return value


def _payload_without_integrity(result: Mapping[str, Any]) -> dict[str, Any]:
    """Return the canonical payload body, excluding only self-auth metadata."""

    normalized = _jsonable(result)
    if not isinstance(normalized, dict):
        raise TypeError("audit payload must be a mapping")
    for field in PAYLOAD_INTEGRITY_EXCLUDED_FIELDS:
        normalized.pop(field, None)
    return normalized


def canonical_payload_digest(result: Mapping[str, Any]) -> str:
    """Hash every emitted semantic/numeric field in a deterministic encoding.

    Key ordering and insignificant JSON whitespace are normalized.  The
    self-referential integrity object is excluded; all other fields,
    including equations, ledger rows, blocker text, statuses, line scopes,
    source hashes, and numeric arrays, are covered recursively.
    """

    encoded = json.dumps(
        _payload_without_integrity(result),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


# Concise aliases make the integrity helper discoverable to independent
# probes without creating a second hashing implementation.
payload_digest = canonical_payload_digest


def _payload_integrity_metadata(digest: str) -> dict[str, Any]:
    return {
        "schema": PAYLOAD_INTEGRITY_SCHEMA,
        "status": "PASS_COMPLETE_PAYLOAD_DIGEST",
        "algorithm": "SHA-256",
        "canonicalization": "json-sort_keys-separators=(',',':')-ensure_ascii=True-allow_nan=False",
        "sha256": digest,
        "payload_sha256": digest,
        "complete": True,
        "covered": "complete payload recursively: semantic text, dependency rows, statuses, blockers, line scopes, hashes, and numeric arrays",
        "excluded_fields": list(PAYLOAD_INTEGRITY_EXCLUDED_FIELDS),
        "deterministic_regeneration": True,
    }


def _read_source(relative: str) -> str:
    path = ROOT / relative
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""


def _source_inventory() -> dict[str, Any]:
    """Hash every read-only claim source and expose static-input evidence."""

    inventory: dict[str, Any] = {}
    for spec in _CLAIM_SPECS:
        rel = str(spec["surface"])
        path = ROOT / rel
        source = _read_source(rel)
        # The AST check is deliberately conservative: this audit never imports
        # a legacy cooling/g-mode module or consumes its output arrays.
        try:
            tree = ast.parse(source, filename=rel)
            imports = [
                node.module or ""
                for node in ast.walk(tree)
                if isinstance(node, ast.ImportFrom)
            ]
            imports.extend(
                alias.name
                for node in ast.walk(tree)
                if isinstance(node, ast.Import)
                for alias in node.names
            )
        except SyntaxError:
            imports = []
        legacy_module_imports = [
            name
            for name in imports
            if any(token in name.lower() for token in ("cooling", "gmode", "g_mode", "urca"))
        ]
        inventory[rel] = {
            "exists": bool(path.exists()),
            "sha256": _sha256(path),
            "line_reference": spec["source_lines"],
            "static_inputs_declared": list(spec["static_inputs"]),
            "legacy_module_imports": legacy_module_imports,
            "status": "AUDITED_READ_ONLY" if path.exists() else "BLOCKED_MISSING_SOURCE",
        }
    return inventory


def _ledger() -> list[dict[str, Any]]:
    inventory = _source_inventory()
    rows: list[dict[str, Any]] = []
    for spec in _CLAIM_SPECS:
        row = dict(spec)
        source_info = inventory.get(spec["surface"], {})
        row["source_sha256"] = source_info.get("sha256")
        row["source_exists"] = source_info.get("exists", False)
        row["static_or_illustrative_inputs"] = True
        # A consumer may import a legacy sensitivity helper (the cyclic
        # example does), but no imported array/result is used as evidence by
        # this audit.  Keep the two facts separate in the ledger.
        row["legacy_module_imports"] = list(source_info.get("legacy_module_imports", []))
        row["legacy_array_imported_as_evidence"] = False
        row["evidence_weight"] = 0.0
        rows.append(row)
    return rows


def schwarzschild_discriminant(
    pressure: Sequence[float] | np.ndarray,
    energy_density: Sequence[float] | np.ndarray,
    dpressure_dr: Sequence[float] | np.ndarray,
    denergy_dr: Sequence[float] | np.ndarray,
    gamma1: Sequence[float] | np.ndarray,
) -> np.ndarray:
    """Return the relativistic Schwarzschild discriminant ``A``.

    In consistent units (for example MeV/fm^3 and km),

    ``A = (d epsilon/dr)/(epsilon+P) - (dP/dr)/(Gamma_1 P)``.

    ``Gamma_1`` is the adiabatic derivative of a displaced fluid element at
    fixed composition/entropy.  It is not determined by a one-dimensional
    equilibrium barotrope.  The function is public so the sign/null identity
    can be independently tested without importing any legacy route.
    """

    p, e, dp, de, g1 = (np.asarray(value, dtype=float) for value in (pressure, energy_density, dpressure_dr, denergy_dr, gamma1))
    p, e, dp, de, g1 = np.broadcast_arrays(p, e, dp, de, g1)
    if any(np.any(~np.isfinite(value)) for value in (p, e, dp, de, g1)):
        raise ValueError("buoyancy inputs must be finite")
    if np.any(p <= 0.0) or np.any(e + p <= 0.0) or np.any(g1 <= 0.0):
        raise ValueError("pressure, epsilon+pressure and Gamma_1 must be positive")
    return de / (e + p) - dp / (g1 * p)


def relativistic_buoyancy_n2(
    pressure: Sequence[float] | np.ndarray,
    energy_density: Sequence[float] | np.ndarray,
    dpressure_dr: Sequence[float] | np.ndarray,
    denergy_dr: Sequence[float] | np.ndarray,
    gamma1: Sequence[float] | np.ndarray,
    redshift_factor: float | Sequence[float] | np.ndarray = 1.0,
) -> np.ndarray:
    """Relativistic ``N^2`` in geometric length units.

    With ``g=-e^(Phi-Lambda) P'/(epsilon+P)``, the usual GR form is
    ``N^2 = e^(2(Phi-Lambda)) P'/(epsilon+P) A``.  The positive
    ``redshift_factor=e^(2(Phi-Lambda))`` is supplied by a metric solver in a
    physical calculation.  In the barotropic null test it is set to one (or a
    positive compactness factor); it cannot create a g-mode by itself.
    """

    p, e, dp, de, g1 = (np.asarray(value, dtype=float) for value in (pressure, energy_density, dpressure_dr, denergy_dr, gamma1))
    z = np.asarray(redshift_factor, dtype=float)
    p, e, dp, de, g1, z = np.broadcast_arrays(p, e, dp, de, g1, z)
    if np.any(~np.isfinite(z)) or np.any(z <= 0.0):
        raise ValueError("redshift_factor must be finite and positive")
    discriminant = schwarzschild_discriminant(p, e, dp, de, g1)
    return z * dp / (e + p) * discriminant


def newtonian_buoyancy_n2(
    gravity_cgs: float | np.ndarray,
    pressure_cgs: Sequence[float] | np.ndarray,
    density_cgs: Sequence[float] | np.ndarray,
    dpressure_dr_cgs: Sequence[float] | np.ndarray,
    ddensity_dr_cgs: Sequence[float] | np.ndarray,
    gamma1: Sequence[float] | np.ndarray,
) -> np.ndarray:
    """Newtonian ``N^2`` in s^-2 using cgs pressure/density gradients.

    ``N^2 = g[(P')/(Gamma_1 P) - rho'/rho]`` with positive outward gravity
    magnitude ``g`` and outward radial derivatives.  A hydrostatic barotrope
    with ``Gamma_1=d ln P/d ln rho`` therefore has ``N^2=0``.
    """

    g, p, rho, dp, drho, g1 = (
        np.asarray(value, dtype=float)
        for value in (gravity_cgs, pressure_cgs, density_cgs, dpressure_dr_cgs, ddensity_dr_cgs, gamma1)
    )
    g, p, rho, dp, drho, g1 = np.broadcast_arrays(g, p, rho, dp, drho, g1)
    if any(np.any(~np.isfinite(value)) for value in (g, p, rho, dp, drho, g1)):
        raise ValueError("Newtonian buoyancy inputs must be finite")
    if np.any(g <= 0.0) or np.any(p <= 0.0) or np.any(rho <= 0.0) or np.any(g1 <= 0.0):
        raise ValueError("gravity, pressure, density and Gamma_1 must be positive")
    return g * (dp / (g1 * p) - drho / rho)


def _profile_arrays(points: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build a smooth canonical-EOS pressure path for closure controls."""

    if int(points) < 9:
        raise ValueError("at least nine radial points are required")
    # EOS/TOV is a read-only conditional background.  No cooling or mode
    # observable is inferred from this synthetic hydrostatic path.
    if str(VERIFICATION) not in sys.path:
        sys.path.insert(0, str(VERIFICATION))
    from nvg_tidal_deformability import EOS  # type: ignore

    eos = EOS(p_match=1.5, Gamma=1.35)
    radius = np.linspace(0.5, 12.5, int(points), dtype=float)
    x = (radius - radius[0]) / (radius[-1] - radius[0])
    pressure = 0.5 + (100.0 - 0.5) * (1.0 - x * x) ** 2
    energy = np.asarray([eos.get_eps(float(value)) for value in pressure], dtype=float)
    dedp = np.asarray([eos.get_dedp(float(value)) for value in pressure], dtype=float)
    dpressure = np.gradient(pressure, radius, edge_order=2)
    denergy = dpressure * dedp
    gamma_eq = (energy + pressure) / (pressure * dedp)
    redshift = np.full_like(radius, 1.0 - 2.0 * (1.4 * M_SUN_KM / 12.5))
    return radius, pressure, energy, dpressure, denergy, gamma_eq, redshift


def compute_barotropic_null_limit(points: int = 241) -> dict[str, Any]:
    """Evaluate exact ``Gamma_1=Gamma_eq`` neutral buoyancy on a cold barotrope."""

    radius, pressure, energy, dpressure, denergy, gamma_eq, redshift = _profile_arrays(points)
    discriminant = schwarzschild_discriminant(pressure, energy, dpressure, denergy, gamma_eq)
    n2 = relativistic_buoyancy_n2(
        pressure, energy, dpressure, denergy, gamma_eq, redshift_factor=redshift
    )
    # A frozen-composition perturbation would require Gamma_1 and dY_i/dr;
    # expose the formal branch as blocked rather than selecting a value.
    return {
        "status": "DERIVED_NULL_CONTROL_BAROTROPIC_NEUTRAL_BUOYANCY",
        "closure": "dP/dEpsilon_background reused for Gamma_1",
        "radius_km": radius,
        "pressure_mev_fm3": pressure,
        "energy_mev_fm3": energy,
        "gamma_equilibrium": gamma_eq,
        "schwarzschild_discriminant": discriminant,
        "n2_km2": n2,
        "redshift_factor": redshift,
        "max_abs_discriminant": float(np.max(np.abs(discriminant))),
        "max_abs_n2_km2": float(np.max(np.abs(n2))),
        "n2_max_abs_km2": float(np.max(np.abs(n2))),
        "null_residual_km2": float(np.max(np.abs(n2))),
        "n2_min_km2": float(np.min(n2)),
        "n2_max_km2": float(np.max(n2)),
        "grid_points": int(points),
        "physical_frequency_status": "BLOCKED_MISSING_GAMMA1_COMPOSITION_DERIVATIVES",
        "missing": [
            "Gamma_1=(dP/dEpsilon)_{s,Y_i} and composition gradients dY_i/dr",
            "metric/profile solver is conditional background only",
        ],
    }


# Alias used by downstream callers that prefer a shorter name.
barotropic_null_limit = compute_barotropic_null_limit
compute_null_limit = compute_barotropic_null_limit
compute_canonical_barotropic_null = compute_barotropic_null_limit


def _frozen_gamma_reference(null: Mapping[str, Any], fractional_delta: float = 0.10) -> dict[str, Any]:
    """Numerical sign/unit control, explicitly not a physical composition model."""

    if not np.isfinite(fractional_delta) or fractional_delta <= 0.0:
        raise ValueError("fractional_delta must be finite and positive")
    pressure = np.asarray(null["pressure_mev_fm3"], dtype=float)
    energy = np.asarray(null["energy_mev_fm3"], dtype=float)
    dpressure = np.gradient(pressure, np.asarray(null["radius_km"]), edge_order=2)
    denergy = dpressure * np.asarray(null["gamma_equilibrium"], dtype=float)  # replaced below
    # dedp = (epsilon+P)/(P*Gamma_eq), hence dE/dP = 1/c_s^2.
    dedp = (energy + pressure) / (pressure * np.asarray(null["gamma_equilibrium"], dtype=float))
    denergy = dpressure * dedp
    gamma_frozen = np.asarray(null["gamma_equilibrium"], dtype=float) * (1.0 + fractional_delta)
    n2 = relativistic_buoyancy_n2(
        pressure,
        energy,
        dpressure,
        denergy,
        gamma_frozen,
        redshift_factor=np.asarray(null["redshift_factor"], dtype=float),
    )
    freq_hz = np.sqrt(np.maximum(n2, 0.0)) * C_LIGHT_KM_S / (2.0 * np.pi)
    return {
        "status": "SENSITIVITY_ONLY_FROZEN_GAMMA1_NOT_MICROPHYSICS",
        "fractional_gamma1_offset": float(fractional_delta),
        "max_n2_km2": float(np.max(n2)),
        "min_n2_km2": float(np.min(n2)),
        "max_frequency_hz": float(np.max(freq_hz)),
        "frequency_units": "Hz after geometric km^-1 to s^-1 conversion",
        "not_a_physical_gmode": True,
    }


def _analytic_null_control() -> dict[str, Any]:
    """Independent Newtonian/relativistic analytic identity and sign checks."""

    radius = np.linspace(1.0, 11.0, 101)
    pressure = 1.0e34 * (1.0 - 0.2 * (radius - 1.0) / 10.0)
    density = 1.0e14 * (pressure / pressure[0]) ** (1.0 / 3.0)
    # Use the analytic derivatives here so the null identity tests algebra,
    # not finite-difference truncation (the separate canonical control covers
    # grid/derivative convergence).
    dpressure = np.full_like(radius, -0.2 * 1.0e34 / (10.0 * 1.0e5))
    ddensity = density * dpressure / (3.0 * pressure)
    gamma_eq = np.full_like(radius, 3.0)
    gamma_adiabatic = np.full_like(radius, 4.0)
    # ``g=1e10`` keeps the absolute round-off scale visible while avoiding a
    # spurious 1e-9 s^-2 residual from multiplying cancellation by a neutron
    # star's full 1e14 cm/s^2 gravity; this is a unit/sign identity control.
    newtonian_null = newtonian_buoyancy_n2(
        1.0e10, pressure, density, dpressure, ddensity, gamma_eq
    )
    newtonian_stable = newtonian_buoyancy_n2(
        1.0e10, pressure, density, dpressure, ddensity, gamma_adiabatic
    )
    # Relativistic linear barotrope: epsilon=100+P/c_s^2 and c_s^2=0.25.
    p_rel = np.linspace(1.0, 4.0, 101)
    e_rel = 100.0 + 4.0 * p_rel
    r_rel = np.linspace(1.0, 11.0, 101)
    dp_rel = np.gradient(p_rel, r_rel, edge_order=2)
    de_rel = 4.0 * dp_rel
    g_rel = (e_rel + p_rel) / (4.0 * p_rel)
    rel_null = relativistic_buoyancy_n2(p_rel, e_rel, dp_rel, de_rel, g_rel)
    return {
        "status": "PASS_ANALYTIC_NULL_UNITS_SIGN_DOMAIN",
        "newtonian_null_max_abs_s2": float(np.max(np.abs(newtonian_null))),
        "newtonian_stable_min_s2": float(np.min(newtonian_stable)),
        "relativistic_null_max_abs_km2": float(np.max(np.abs(rel_null))),
        "units": {
            "newtonian_n2": "s^-2",
            "relativistic_n2": "km^-2",
            "frequency_conversion": "sqrt(N2[km^-2])*299792.458/(2*pi) Hz",
        },
        "sign_convention": "outward r, dP/dr<0, positive N2 is stable",
    }


def _grid_convergence() -> dict[str, Any]:
    rows = [compute_barotropic_null_limit(points) for points in (61, 121, 241, 481)]
    residuals = [float(row["max_abs_n2_km2"]) for row in rows]
    return {
        "status": "PASS_NULL_GRID_CONVERGENCE",
        "grid_points": [61, 121, 241, 481],
        "max_abs_n2_km2": residuals,
        "all_below_1e-14": bool(all(value < 1.0e-14 for value in residuals)),
        "max_abs_discriminant": [float(row["max_abs_discriminant"]) for row in rows],
    }


def _derivative_convergence() -> dict[str, Any]:
    """Check the d epsilon/dr=d epsilon/dP*dP/dr identity on every grid."""

    rows: list[dict[str, Any]] = []
    for points in (61, 121, 241, 481):
        _, pressure, energy, dpressure, denergy, gamma_eq, _ = _profile_arrays(points)
        dedp = (energy + pressure) / (pressure * gamma_eq)
        residual = np.asarray(denergy) - np.asarray(dpressure) * dedp
        rows.append(
            {
                "grid_points": int(points),
                "max_abs_identity_residual": float(np.max(np.abs(residual))),
                "max_relative_identity_residual": float(
                    np.max(np.abs(residual) / np.maximum(np.abs(denergy), 1.0e-30))
                ),
            }
        )
    return {
        "status": "PASS_DERIVATIVE_IDENTITY_GRID_CONVERGENCE",
        "rows": rows,
        "all_below_1e-14_relative": bool(all(row["max_relative_identity_residual"] < 1.0e-14 for row in rows)),
    }


def _canonical_regression() -> dict[str, Any]:
    if not CANONICAL_RESULT_PATH.exists():
        return {"status": "BLOCKED_CANONICAL_ARTIFACT_MISSING", "path": str(CANONICAL_RESULT_PATH.relative_to(ROOT))}
    try:
        payload = json.loads(CANONICAL_RESULT_PATH.read_text(encoding="utf-8"))
        observed = payload.get("canonical", {}).get("observables", {})
        checks = {
            name: bool(math.isclose(float(observed.get(name)), expected, rel_tol=0.0, abs_tol=1.0e-12))
            for name, expected in CANONICAL_REGRESSION.items()
        }
    except (OSError, ValueError, TypeError, KeyError):
        return {"status": "BLOCKED_CANONICAL_ARTIFACT_UNREADABLE", "path": str(CANONICAL_RESULT_PATH.relative_to(ROOT))}
    return {
        "status": "PASS" if all(checks.values()) else "FAIL_CANONICAL_REGRESSION_DRIFT",
        "checks": checks,
        "values": {name: observed.get(name) for name in CANONICAL_REGRESSION},
        "expected": dict(CANONICAL_REGRESSION),
        "path": str(CANONICAL_RESULT_PATH.relative_to(ROOT)),
        "semantics": "read-only conditional/in-sample Phase-1 input; not independent evidence",
    }


def _p2_regression() -> dict[str, Any]:
    if not P2_RESULT_PATH.exists():
        return {"status": "BLOCKED_P2_S2_ARTIFACT_MISSING", "path": str(P2_RESULT_PATH.relative_to(ROOT))}
    try:
        payload = json.loads(P2_RESULT_PATH.read_text(encoding="utf-8"))
        checks = {
            "all_interaction_couplings_zero": payload.get("model", {}).get("couplings", {}).get("all_interaction_couplings") == 0.0,
            "stable_tov_blocked": payload.get("stellar_mapping", {}).get("status") == "BLOCKED_NO_STABLE_TOV_BRANCH",
            "emissivity_blocked": str(payload.get("urca", {}).get("emissivity_status", "")).startswith("BLOCKED_"),
            "cooling_blocked": payload.get("claims", {}).get("cooling_curve") == "BLOCKED",
            "canonical_status_pass": payload.get("canonical_regression", {}).get("status") == "PASS",
        }
    except (OSError, ValueError, TypeError):
        return {"status": "BLOCKED_P2_S2_ARTIFACT_UNREADABLE", "path": str(P2_RESULT_PATH.relative_to(ROOT))}
    return {
        "status": "PASS" if all(checks.values()) else "FAIL_P2_REFERENCE_SEMANTICS",
        "checks": checks,
        "path": str(P2_RESULT_PATH.relative_to(ROOT)),
        "semantics": "zero-interaction T=0 reference only; no physical hyperon/Urca/cooling result",
        "stable_mapping": payload.get("stellar_mapping", {}).get("status"),
        "reference_model": payload.get("model", {}).get("status"),
    }


def _source_provenance() -> dict[str, Any]:
    inventory = _source_inventory()
    local = {rel: info["sha256"] for rel, info in inventory.items()}
    # Pin the producer and focused test themselves as part of the complete
    # payload.  Without these two entries, a stale artifact could survive a
    # semantic edit that happened not to alter a legacy claim source.
    local["verification/nvg_cooling_gmode_dependency_audit.py"] = _sha256(Path(__file__))
    local["verification/test_p3_cooling_gmode_dependency_audit.py"] = _sha256(
        VERIFICATION / "test_p3_cooling_gmode_dependency_audit.py"
    )
    document_rows = _document_claim_scan()
    local.update({row["surface"]: row["sha256"] for row in document_rows})
    local["verification/nvg_ns_predictive_audit_results.json"] = _sha256(CANONICAL_RESULT_PATH)
    local["verification/nvg_beta_hyperon_urca_audit_p2s2_results.json"] = _sha256(P2_RESULT_PATH)
    return {
        "external_inputs": {
            "count": 0,
            "status": "NONE_USED",
            "note": "No downloaded table, observational array, or legacy result array is imported as evidence.",
        },
        "local_source_sha256": local,
        "canonical_background": {
            "producer": "verification/nvg_tidal_deformability.py::EOS",
            "status": "CONDITIONAL_IN_SAMPLE_BACKGROUND_ONLY",
            "source_sha256": _sha256(VERIFICATION / "nvg_tidal_deformability.py"),
        },
        "configuration": {
            "null_profile": "canonical EOS epsilon(P) on a smooth synthetic hydrostatic pressure path",
            "density_or_temperature_grid": "not used for a physical cooling result",
            "null_grids": [61, 121, 241, 481],
            "units": {"P_epsilon": "MeV/fm^3", "r": "km", "N2": "km^-2", "f": "Hz"},
        },
    }


def _adversarial_controls() -> dict[str, Any]:
    """Fail-closed mutations: omitted Gamma_1, wrong sign, invalid domains."""

    checks: dict[str, bool] = {}
    try:
        p = np.asarray([1.0, 2.0])
        e = np.asarray([10.0, 11.0])
        dp = np.asarray([-1.0, -1.0])
        de = np.asarray([-2.0, -2.0])
        # Missing Gamma_1 is a blocked branch, not a guessed derivative.
        relativistic_buoyancy_n2(p, e, dp, de, None)  # type: ignore[arg-type]
        checks["missing_gamma1_rejected"] = False
    except (TypeError, ValueError):
        checks["missing_gamma1_rejected"] = True
    try:
        relativistic_buoyancy_n2([1.0], [10.0], [-1.0], [-2.0], [1.0], redshift_factor=0.0)
        checks["zero_redshift_rejected"] = False
    except ValueError:
        checks["zero_redshift_rejected"] = True
    try:
        newtonian_buoyancy_n2(1.0, [1.0], [1.0], [-1.0], [-1.0], [0.0])
        checks["zero_gamma1_rejected"] = False
    except ValueError:
        checks["zero_gamma1_rejected"] = True
    try:
        newtonian_buoyancy_n2(1.0, [1.0], [1.0], [-1.0], [-1.0], [2.0])
        checks["valid_newtonian_accepted"] = True
    except ValueError:
        checks["valid_newtonian_accepted"] = False
    # Static-source mutation guard: this audit does not import legacy routes.
    source = Path(__file__).read_text(encoding="utf-8")
    checks["no_legacy_cooling_imports"] = not bool(
        re.search(r"(?:from|import)\s+nvg_(?:cas_a_cooling_curve|cooling_dark_matter|ns_g_modes|direct_urca)", source)
    )
    return {
        "status": "PASS_FAIL_CLOSED_MUTATIONS" if all(checks.values()) else "FAIL_MUTATION_GUARD",
        "checks": checks,
    }


def run_audit() -> dict[str, Any]:
    """Run the complete P3-S2 audit without writing generated artifacts."""

    null = compute_barotropic_null_limit(241)
    analytic = _analytic_null_control()
    frozen = _frozen_gamma_reference(null)
    ledger = _ledger()
    document_claims = _document_claim_scan()
    physical_block = {
        "cooling_curves": "BLOCKED_MISSING_COMPOSITION_HEAT_CAPACITY_EMISSIVITY_PAIRING_CONDUCTIVITY_ENVELOPE",
        "Cas_A_fit": "BLOCKED_NO_THERMAL_SOLVER_OR_SPECTRAL_LIKELIHOOD",
        "cas_a_fit": "BLOCKED_NO_THERMAL_SOLVER_OR_SPECTRAL_LIKELIHOOD",
        "physical_cooling_curves": "BLOCKED_MISSING_COMPOSITION_HEAT_CAPACITY_EMISSIVITY_PAIRING_CONDUCTIVITY_ENVELOPE",
        "hyperon_Urca_cooling": "BLOCKED_MISSING_INTERACTIONS_WEAK_MATRIX_ELEMENTS_AND_STABLE_TOV_BRANCH",
        "hyperon_urca_cooling": "BLOCKED_MISSING_INTERACTIONS_WEAK_MATRIX_ELEMENTS_AND_STABLE_TOV_BRANCH",
        "NVG_confirmation": "BLOCKED_NO_CLOSED_MICROPHYSICAL_CHAIN_OR_INDEPENDENT_LIKELIHOOD",
        "nvg_confirmation": "BLOCKED_NO_CLOSED_MICROPHYSICAL_CHAIN_OR_INDEPENDENT_LIKELIHOOD",
        "g_mode_frequency": "BLOCKED_MISSING_GAMMA1_COMPOSITION_DERIVATIVES_AND_MODE_LIKELIHOOD",
    }
    result: dict[str, Any] = {
        "schema_version": 1,
        "audit": "P3-S2",
        "status": "COMPLETE_WITH_PHYSICAL_COOLING_AND_GMODE_BLOCKS",
        "artifacts": {
            "result_json": str(RESULT_PATH.relative_to(ROOT)),
            "figure_png": str(FIGURE_PATH.relative_to(ROOT)),
            "report": str(S2_REPORT_PATH.relative_to(ROOT)),
            "repair_report": str(REPORT_PATH.relative_to(ROOT)),
            "evidence": str(EVIDENCE_PATH.relative_to(ROOT)),
        },
        "dependency_ledger": ledger,
        "legacy_claims": ledger,
        "claims_by_id": {row["claim_id"]: row for row in ledger},
        "documentation_claim_scan": document_claims,
        "required_cooling_dependencies": {
            name: {
                "status": "BLOCKED_MISSING_PRODUCER",
                "producer": "none in maintained repository",
                "reference_only": name == "stellar_profile",
            }
            for name in (
                "composition",
                "heat_capacity",
                "neutrino_channels_and_matrix_elements",
                "temperature_dependence",
                "superfluid_gaps_and_suppression",
                "conductivity",
                "stellar_profile",
                "envelope_relation",
            )
        },
        "buoyancy": {
            "criterion": {
                "newtonian": "N^2=g[(1/(Gamma_1 P)) dP/dr - (1/rho) drho/dr]",
                "relativistic": "N^2=e^(2(Phi-Lambda)) [dP/dr/(epsilon+P)] A",
                "schwarzschild_discriminant": "A=(d epsilon/dr)/(epsilon+P)-(dP/dr)/(Gamma_1 P)",
                "positive_sign": "N^2>0 is stable stratification for outward r and dP/dr<0",
            },
            "null_limit": null,
            "frozen_composition_branch": {
                "status": "BLOCKED_MISSING_GAMMA1_COMPOSITION_DERIVATIVES",
                "required": [
                    "Gamma_1=(epsilon+P)/P*(dP/d epsilon)_{s,Y_i}",
                    "composition gradients dY_i/dr and partial derivatives (dP/dY_i)_{n,s}",
                ],
                "no_frequency_invented": True,
            },
            "frozen_gamma_reference": frozen,
            "units": analytic["units"],
            "sign_convention": analytic["sign_convention"],
        },
        "controls": {
            "analytic_null_and_sign": analytic,
            "grid_convergence": _grid_convergence(),
            "derivative_convergence": _derivative_convergence(),
            "canonical_regression": _canonical_regression(),
            "p2_regression": _p2_regression(),
            "adversarial_mutations": _adversarial_controls(),
            "legacy_array_imports": {
                "status": "PASS_NONE_IMPORTED_AS_EVIDENCE",
                "imported": [],
                "note": "Legacy files are source-audited and hashed only; P2 JSON is a read-only status reference.",
            },
        },
        "physical_no_go": physical_block,
        "source_provenance": _source_provenance(),
        "evidence_semantics": {
            "null_control": "derived_conditional mathematical/reference control",
            "frozen_gamma_reference": "sensitivity_only, not composition microphysics",
            "legacy_nonzero_gmode": "blocked_static_illustrative_input",
            "cooling_observations": "blocked without sourced products and likelihood",
        },
    }
    # Add the self-authenticating metadata only after every scientific,
    # semantic, provenance, and numeric field has been assembled.  The digest
    # excludes this metadata object itself to avoid a circular hash.
    result["payload_integrity"] = _payload_integrity_metadata(canonical_payload_digest(result))
    return result


def assert_artifact_provenance(result: Mapping[str, Any]) -> None:
    """Fail closed if any emitted semantic/numeric payload field drifts.

    Source hashes and selected aliases are useful local controls, but they do
    not authenticate equation strings, ledger rows, blocker text, or every
    numeric array.  P3-S6 therefore verifies a complete canonical digest and
    deterministically regenerates the body from the live producer.  Either
    check is sufficient to reject a mutation; both are retained to make a
    digest-only artifact and a stale-but-self-consistent artifact fail closed.
    """

    integrity = result.get("payload_integrity")
    if not isinstance(integrity, Mapping):
        raise AssertionError("P3-S6 complete payload integrity metadata is missing")
    declared_digest = integrity.get("sha256")
    if not isinstance(declared_digest, str) or not re.fullmatch(r"[0-9a-f]{64}", declared_digest):
        raise AssertionError("P3-S6 payload digest is missing or malformed")
    expected_metadata = _payload_integrity_metadata(declared_digest)
    if dict(integrity) != expected_metadata:
        raise AssertionError("P3-S6 payload integrity metadata drift")
    try:
        computed_digest = canonical_payload_digest(result)
    except (TypeError, ValueError, OverflowError) as exc:
        raise AssertionError(f"P3-S6 payload cannot be canonically encoded: {exc}") from exc
    if computed_digest != declared_digest:
        raise AssertionError("P3-S6 complete payload digest mismatch")
    # A regenerated body catches a forged replacement that also updates the
    # digest.  It compares every normalized field, not only known aliases.
    regenerated = run_audit()
    if _payload_without_integrity(regenerated) != _payload_without_integrity(result):
        raise AssertionError("P3-S6 deterministic full-payload regeneration mismatch")

    assert result.get("audit") == "P3-S2"
    assert result.get("status") == "COMPLETE_WITH_PHYSICAL_COOLING_AND_GMODE_BLOCKS"
    assert result.get("physical_no_go", {}).get("cooling_curves", "").startswith("BLOCKED_")
    assert result.get("physical_no_go", {}).get("g_mode_frequency", "").startswith("BLOCKED_")
    assert result.get("controls", {}).get("legacy_array_imports", {}).get("imported") == []
    null = result.get("buoyancy", {}).get("null_limit", {})
    assert null.get("status") == "DERIVED_NULL_CONTROL_BAROTROPIC_NEUTRAL_BUOYANCY"
    assert float(null.get("max_abs_n2_km2")) < 1.0e-14
    n2_values = np.asarray(null.get("n2_km2", []), dtype=float)
    assert n2_values.size == int(null.get("grid_points", 0))
    assert math.isclose(float(np.max(np.abs(n2_values))), float(null.get("max_abs_n2_km2")), rel_tol=0.0, abs_tol=1.0e-30)
    for relative, digest in result.get("source_provenance", {}).get("local_source_sha256", {}).items():
        path = ROOT / relative
        assert path.exists(), relative
        assert _sha256(path) == digest, relative
    ledger = result.get("dependency_ledger", [])
    assert len(ledger) >= 10
    assert all(row.get("evidence_weight") == 0.0 for row in ledger)
    assert all(not row.get("legacy_array_imported_as_evidence") for row in ledger)
    assert result.get("controls", {}).get("derivative_convergence", {}).get("status") == "PASS_DERIVATIVE_IDENTITY_GRID_CONVERGENCE"


def write_artifacts(result: Mapping[str, Any], result_path: Path = RESULT_PATH, figure_path: Path = FIGURE_PATH) -> None:
    result_path.parent.mkdir(parents=True, exist_ok=True)
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True, default=lambda value: np.asarray(value).tolist()) + "\n", encoding="utf-8")

    null = result["buoyancy"]["null_limit"]
    radius = np.asarray(null["radius_km"], dtype=float)
    n2 = np.asarray(null["n2_km2"], dtype=float)
    ledger = result["dependency_ledger"]
    statuses = [row["audit_status"] for row in ledger]
    blocked = np.asarray([1.0 if str(status).startswith("BLOCKED") else 0.0 for status in statuses])
    labels = [row["claim_id"].replace("_", "\n") for row in ledger]

    plt.close("all")
    fig, axes = plt.subplots(2, 1, figsize=(11.5, 8.0), constrained_layout=True)
    axes[0].plot(radius, n2, color="tab:blue", linewidth=1.8)
    axes[0].axhline(0.0, color="black", linewidth=0.8)
    axes[0].set_xlabel("synthetic radial coordinate r [km]")
    axes[0].set_ylabel(r"$N^2$ [km$^{-2}$]")
    axes[0].set_title("Cold barotropic closure: neutral-buoyancy null (physical g-mode blocked)")
    axes[0].grid(alpha=0.25)
    colours = ["tab:red" if value else "tab:green" for value in blocked]
    axes[1].bar(np.arange(len(labels)), blocked, color=colours)
    axes[1].set_xticks(np.arange(len(labels)), labels, rotation=65, ha="right", fontsize=7)
    axes[1].set_yticks([0.0, 1.0], ["reference/sensitivity", "blocked/retired"])
    axes[1].set_ylim(-0.05, 1.15)
    axes[1].set_title("Legacy cooling/g-mode dependency ledger (no empirical evidence weight)")
    axes[1].grid(axis="y", alpha=0.25)
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)


def main(argv: Iterable[str] | None = None) -> int:
    del argv
    result = run_audit()
    write_artifacts(result)
    assert_artifact_provenance(result)
    null = result["buoyancy"]["null_limit"]
    print("P3-S2 cooling/g-mode dependency audit")
    print(f"status={result['status']}")
    print(f"ledger_rows={len(result['dependency_ledger'])}")
    print(f"barotropic_null_max_abs_N2_km2={null['max_abs_n2_km2']:.3e}")
    print(f"frozen_composition_status={result['buoyancy']['frozen_composition_branch']['status']}")
    print(f"cooling_curves={result['physical_no_go']['cooling_curves']}")
    print(f"g_mode_frequency={result['physical_no_go']['g_mode_frequency']}")
    print(f"canonical_regression={result['controls']['canonical_regression']['status']}")
    print(f"p2_regression={result['controls']['p2_regression']['status']}")
    print(f"adversarial_mutations={result['controls']['adversarial_mutations']['status']}")
    print(f"payload_sha256={result['payload_integrity']['sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
