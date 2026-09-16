#!/usr/bin/env python3
"""Renormalisation/provenance audit for the NVG hadron-mass split.

The maintained VMF scripts use sigma-term numbers and a ``current`` mass
remainder as if the split were an observable.  This audit keeps the useful
part (the Feynman--Hellmann sigma products) separate from the
scheme/scale-dependent scalar operator and quark-mass pieces.  It does not
refit any input and it does not turn a missing renormalisation prescription
into one by assumption.

The script has three deliberately small numerical controls:

* recompute the maintained nucleon mass budget by importing its producer;
* verify algebraically that ``m_q * <H|qbar q|H>`` is invariant under the
  inverse finite renormalisation of the mass and scalar operator;
* evaluate the source-backed leading-order heavy-quark matching expressions
  for charm, bottom, and top separately (``2/27``, ``2/25``, and ``2/23``),
  exposing the required ``Delta_s M`` assumption without comparing them with
  the opaque maintained ``sigma_heavy = 6 MeV`` input.

All scalar-operator conversions are fail-closed unless both a scheme/scale
and the operator transformation are present in the input record.  In
particular, a bare number in a comment is not accepted as a renormalised
matrix element.
"""

from __future__ import annotations

import importlib
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
VERIFICATION = ROOT / "verification"
RESULT_PATH = VERIFICATION / "renormalization_covariance_audit_results.json"
INVENTORY_PATH = VERIFICATION / "renormalization_covariance_audit_inventory.json"
GENERATED_REPORT_PATH = VERIFICATION / "renormalization_covariance_audit_report.md"

SIGMA_PI_SOURCE = {
    "citation": "Gupta et al., Phys. Rev. Lett. 127, 242002 (2021)",
    "doi": "10.1103/PhysRevLett.127.242002",
    "url": "https://doi.org/10.1103/PhysRevLett.127.242002",
}
SIGMA_S_SOURCE = {
    "citation": "Agadjanov et al., Phys. Rev. Lett. 131, 261902 (2023)",
    "doi": "10.1103/PhysRevLett.131.261902",
    "url": "https://doi.org/10.1103/PhysRevLett.131.261902",
}
FLAG_SOURCE = {
    "citation": "FLAG Review 2021, Eur. Phys. J. C 82, 869",
    "doi": "10.1140/epjc/s10052-022-10536-1",
    "url": "https://doi.org/10.1140/epjc/s10052-022-10536-1",
}
HEAVY_MATCHING_SOURCE = {
    "citation": "Kryjevski, Phys. Rev. D 70, 094028 (2004)",
    "doi": "10.1103/PhysRevD.70.094028",
    "url": "https://doi.org/10.1103/PhysRevD.70.094028",
}
TRACE_ANOMALY_SOURCE = {
    "citation": "Hatta, Rajan & Tanaka, JHEP 12 (2018) 008",
    "doi": "10.1007/JHEP12(2018)008",
    "url": "https://doi.org/10.1007/JHEP12(2018)008",
}


@dataclass(frozen=True)
class InputRecord:
    """One load-bearing input found in a maintained producer."""

    input_id: str
    producer: str
    quantity: str
    value: float | None
    uncertainty: float | None
    unit: str
    role: str
    source: str | None
    citation: str | None
    scheme: str | None
    scale_GeV: float | None
    operator_definition: str | None
    transformation: str | None
    metadata_status: str
    note: str


def maintained_input_inventory() -> list[InputRecord]:
    """Inventory the maintained sigma/current-mass producers.

    Values intentionally mirror the producers; provenance fields are not
    silently repaired here.  Missing metadata is itself an audit result.
    """

    rows = [
        InputRecord(
            "hadron_mass_fractions.sigma_piN", "verification/nvg_hadron_mass_fractions.py",
            "sigma_piN", 44.0, 3.0, "MeV", "sigma_product", "Gupta et al. (2021)",
            SIGMA_PI_SOURCE["citation"], None, None,
            "sigma_piN = m_ud dM_N/dm_ud", None, "PARTIAL_PROVENANCE",
            "Citation is present in a comment, but the producer exposes no scheme, scale, or Z_m/Z_S relation.",
        ),
        InputRecord(
            "hadron_mass_fractions.sigma_sN", "verification/nvg_hadron_mass_fractions.py",
            "sigma_sN", 30.0, 7.0, "MeV", "sigma_product", "Agadjanov et al. (2023)",
            SIGMA_S_SOURCE["citation"], None, None,
            "sigma_sN = m_s dM_N/dm_s", None, "PARTIAL_PROVENANCE",
            "The maintained 30 +/- 7 MeV is a rounded value; the cited paper reports 28.6 +/- 9.3 MeV.",
        ),
        InputRecord(
            "hadron_mass_fractions.sigma_heavy", "verification/nvg_hadron_mass_fractions.py",
            "sigma_heavy", 6.0, 3.0, "MeV", "sigma_product", "PDG heavy-quark matching", None,
            None, None, None, None, "MISSING_SOURCE_METADATA",
            "No PDG edition, matching order, active-flavour convention, or per-flavour/total definition is given.",
        ),
        InputRecord(
            "hadron_mass_fractions.nucleon_mass", "verification/nvg_hadron_mass_fractions.py",
            "M_N", 939.0, None, "MeV", "hadron_mass", "PDG conventional nucleon mass", None,
            None, None, None, None, "PARTIAL_PROVENANCE",
            "The producer uses 939 MeV without an explicit proton/neutron choice or source edition.",
        ),
        InputRecord(
            "hadron_mass_fractions.pion_sigma", "verification/nvg_hadron_mass_fractions.py",
            "sigma_piH (pion cross-check)", 67.0, 5.0, "MeV", "sigma_product", "unidentified literature cross-check", None,
            None, None, "sigma_piH = m_ud dM_pi/dm_ud", None, "MISSING_SOURCE_METADATA",
            "PION_SIGMA is executable input but has no citation, scheme, scale, or operator transformation.",
        ),
        InputRecord(
            "hadron_mass_fractions.kaon_sigma", "verification/nvg_hadron_mass_fractions.py",
            "sigma_piH + sigma_sH (kaon)", 362.0, None, "MeV", "sigma_product", "unidentified literature cross-check", None,
            None, None, "light/strange sigma terms for kaon", None, "MISSING_SOURCE_METADATA",
            "KAON_SIGMA is executable input (22 +/- 3 and 340 +/- 20 MeV) but its source and renormalisation metadata are absent.",
        ),
        InputRecord(
            "hadron_mass_fractions.nucleon_sigma_broad", "verification/nvg_hadron_mass_fractions.py",
            "sigma_piN + sigma_sN + sigma_heavy (broad range)", 90.0, None, "MeV", "sigma_product", "various analyses", None,
            None, None, "broad sensitivity input", None, "MISSING_SOURCE_METADATA",
            "NUCLEON_SIGMA_BROAD is executable sensitivity input (51 + 33 + 6 MeV) with no traceable source metadata.",
        ),
        InputRecord(
            "eos_beta_saturated_vector.sigma_terms", "verification/nvg_eos_beta_saturated_vector.py",
            "sigma_piN + sigma_sN + sigma_heavy", 80.0, None, "MeV", "EOS_anchor", "copied calibration", None,
            None, None, None, None, "MISSING_RENORM_METADATA",
            "Fixed literals feed M_Omega_0 and M_current_0; no sigma operator metadata reaches the EOS.",
        ),
        InputRecord(
            "eos_proof_checked.sigma_terms", "verification/nvg_eos_proof_checked.py",
            "sigma_piN + sigma_sN + sigma_heavy", 80.0, None, "MeV", "EOS_anchor", "copied calibration", None,
            None, None, None, None, "MISSING_RENORM_METADATA",
            "The checked proxy reproduces the same fixed 44 + 30 + 6 MeV split.",
        ),
        InputRecord(
            "vacuum_w_derivation.sigma_terms", "verification/nvg_vacuum_w_field_derivation.py",
            "sigma_piN, sigma_sN", 74.0, None, "MeV", "W_field_calibration", "lattice-style input", None,
            None, None, None, None, "MISSING_RENORM_METADATA",
            "Sigma values enter the lambda calculation, but the script does not state a renormalisation prescription.",
        ),
        InputRecord(
            "fine_structure.current_masses", "verification/nvg_fine_structure.py",
            "m_u, m_d, m_s", 100.23, None, "MeV", "quark_mass", "FLAG-like literals", None,
            None, None, None, None, "MISSING_RENORM_METADATA",
            "The producer stores 2.16, 4.67, and 93.4 MeV (in GeV units) but does not state MSbar or a scale.",
        ),
        InputRecord(
            "strong_cp.current_masses", "verification/nvg_strong_cp_solution.py",
            "m_u, m_d, m_s", 100.23, None, "MeV", "quark_mass", "FLAG 2024 (comment only)", None,
            None, None, None, None, "MISSING_RENORM_METADATA",
            "The comment names FLAG but has no edition table, scheme, scale, or matching inputs.",
        ),
        InputRecord(
            "fair_hades.current_mass_table", "verification/nvg_fair_hades_link.py",
            "hadron current-mass placeholders", None, None, "MeV", "in_medium_prediction", None, None,
            None, None, None, None, "MISSING_SOURCE_METADATA",
            "80, 67, 362, 300, and 200 MeV placeholders feed in-medium mass shifts without sigma-term provenance.",
        ),
        InputRecord(
            "hyperon_puzzle.current_mass_table", "verification/nvg_hyperon_puzzle.py",
            "M_cur_N, M_cur_Lambda", None, None, "MeV", "in_medium_prediction", None, None,
            None, None, None, None, "MISSING_SOURCE_METADATA",
            "Hyperon current-mass remainder is assumed rather than derived from renormalised scalar operators.",
        ),
    ]
    # Complete-by-default discovery for downstream consumers that copy the
    # same literals under another name (M_current_0, M_CUR, m_cur_L, ...).
    # Active Phase-1 sibling surfaces are excluded: they are unfinished
    # worker outputs, not maintained inputs this audit is authorized to use.
    known = {row.producer for row in rows}
    excluded = {
        "verification/nvg_renormalization_covariance_audit.py",
        "verification/test_renormalization_covariance_audit.py",
        "verification/eos_hvh_closure_audit.py",
        "verification/static_w_branch_audit.py",
    }
    sigma_pattern = re.compile(r"\bsigma_(?:piN|sN|heavy|total)\b|sigma_piH|sigma_sH")
    current_pattern = re.compile(r"\bM_current(?:_0)?\b|\bM_cur(?:_N|_Lambda)?\b|\bM_CUR\b|current.?mass", re.IGNORECASE)
    quark_pattern = re.compile(r"\bm_[uds]\b|current quark", re.IGNORECASE)
    for path in sorted(VERIFICATION.glob("*.py")):
        relative = str(path.relative_to(ROOT))
        if relative in known or relative in excluded:
            continue
        text = path.read_text(encoding="utf-8")
        matches = {
            "sigma_terms": len(sigma_pattern.findall(text)),
            "current_mass": len(current_pattern.findall(text)),
            "quark_masses": len(quark_pattern.findall(text)),
        }
        if not any(matches.values()):
            continue
        # A current-mass match is relevant only when the surface is part of
        # the hadron/condensate chain; avoid black-hole lifetime M_current.
        if matches["current_mass"] and not (
            "M_Omega" in text or "m_omega" in text or matches["sigma_terms"] or matches["quark_masses"]
        ):
            continue
        role = "sigma_consumer" if matches["sigma_terms"] else ("quark_mass" if matches["quark_masses"] else "in_medium_prediction")
        rows.append(
            InputRecord(
                f"discovered.{path.stem}", relative, "runtime literals matched by audit scan", None, None,
                "MeV (declared by producer where available)", role, None, None, None, None, None, None,
                "DISCOVERED_MISSING_RENORM_METADATA",
                f"Static scan matches sigma/current/quark-mass tokens: {matches}; producer metadata must be inspected before conversion.",
            )
        )
    return rows


def source_backed_flag_mass_records() -> list[InputRecord]:
    """FLAG 2021 2+1+1 masses, explicitly marked as external reference data."""

    values = (("m_u", 2.14, 0.08), ("m_d", 4.70, 0.05), ("m_s", 93.40, 0.57))
    return [
        InputRecord(
            f"flag2021.{name}", "external reference (not a maintained producer)", name,
            value, uncertainty, "MeV", "quark_mass_reference", "FLAG Review 2021",
            FLAG_SOURCE["citation"], "MSbar", 2.0, "renormalised qbar q operator is inverse to m_q",
            "Z_m(μ) Z_S(μ) = 1 in a mass-independent scheme", "SOURCE_BACKED_METADATA",
            "Reference mass used only to show the required conversion contract; it is not substituted into NVG.",
        )
        for name, value, uncertainty in values
    ]


def _finite(value: float | None) -> bool:
    return value is not None and math.isfinite(float(value))


def metadata_ready(record: Mapping[str, Any]) -> bool:
    """Return whether an operator conversion is authorized by metadata."""

    return bool(
        record.get("scheme")
        and _finite(record.get("scale_GeV"))
        and record.get("operator_definition")
        and record.get("transformation")
    )


def require_metadata(record: Mapping[str, Any]) -> None:
    if not metadata_ready(record):
        raise ValueError(
            f"renormalisation metadata incomplete for {record.get('input_id', '<unknown>')}: "
            "scheme, scale, operator definition, and transformation are all required"
        )


def finite_scheme_transform(mass: float, scalar: float, factor: float) -> tuple[float, float, float]:
    """Apply m' = C m and O' = O/C; return (m', O', product')."""

    if not (math.isfinite(mass) and math.isfinite(scalar) and math.isfinite(factor) and factor > 0.0):
        raise ValueError("mass, scalar, and positive finite transformation factor are required")
    new_mass = mass * factor
    new_scalar = scalar / factor
    return new_mass, new_scalar, new_mass * new_scalar


def invariant_controls() -> dict[str, Any]:
    """Algebraic RG-invariance and domain controls."""

    mass, scalar, factor = 4.70, 12.345, 1.37
    new_mass, new_scalar, transformed_product = finite_scheme_transform(mass, scalar, factor)
    original_product = mass * scalar
    checks = {
        "inverse_mass_scalar_factors": math.isclose(transformed_product, original_product, rel_tol=0.0, abs_tol=1.0e-12),
        "mass_changes_under_scheme": not math.isclose(new_mass, mass, rel_tol=0.0, abs_tol=1.0e-12),
        "scalar_changes_inverse": math.isclose(new_scalar * factor, scalar, rel_tol=0.0, abs_tol=1.0e-12),
    }
    return {
        "status": "PASS_RG_INVARIANT_PRODUCT_ALGEBRA" if all(checks.values()) else "FAIL_RG_ALGEBRA",
        "checks": checks,
        "example": {
            "m_original": mass,
            "scalar_original": scalar,
            "finite_factor": factor,
            "m_transformed": new_mass,
            "scalar_transformed": new_scalar,
            "product_original": original_product,
            "product_transformed": transformed_product,
        },
        "identity": "sigma_qH = m_q(mu) <H|qbar q|H>_mu; m'_q=C m_q and O'_S=O_S/C",
        "source": TRACE_ANOMALY_SOURCE,
    }


def maintained_mass_budget() -> dict[str, Any]:
    """Recompute the maintained producer's nucleon budget without copying output."""

    try:
        module = importlib.import_module("verification.nvg_hadron_mass_fractions")
    except ModuleNotFoundError:
        module = importlib.import_module("nvg_hadron_mass_fractions")
    budget = module.compute_mass_budget("Nucleon (audit)", 939.0, module.NUCLEON_SIGMA)
    return {
        "producer": "verification/nvg_hadron_mass_fractions.py::compute_mass_budget",
        "mass_MeV": float(budget.mass_MeV),
        "sigma_piN_MeV": float(budget.sigma.sigma_piH),
        "sigma_sN_MeV": float(budget.sigma.sigma_sH),
        "sigma_heavy_MeV": float(budget.sigma.sigma_heavy),
        "sum_sigma_MeV": float(budget.M_current),
        "sum_sigma_error_quadrature_MeV": float(budget.M_current_err),
        "M_Omega_MeV": float(budget.M_Omega),
        "M_Omega_error_MeV": float(budget.M_Omega_err),
        "f_Omega": float(budget.f_Omega),
        "f_Omega_error": float(budget.f_Omega_err),
        "formula": "M_Omega = M_N - (sigma_piN + sigma_sN + sigma_heavy)",
        "interpretation": "bookkeeping remainder in the producer's declared input convention; not an RG-invariant operator",
    }


def heavy_matching_control(mass_budget: Mapping[str, Any]) -> dict[str, Any]:
    """Sequential LO charm/bottom/top matching references.

    Kryjevski's convention gives separate terms with denominators 27, 25,
    and 23 as heavy flavours are integrated out sequentially.  The cited
    expressions use ``Delta_s M`` (the integrated strange-mass shift), while
    the maintained input supplies only the point sigma term.  We expose that
    equality as an explicit, unsourced reference assumption and never use it
    to compare against the opaque 6 MeV literal.
    """

    m_n = float(mass_budget["mass_MeV"])
    sigma_pi = float(mass_budget["sigma_piN_MeV"])
    sigma_s = float(mass_budget["sigma_sN_MeV"])
    err_pi = 3.0
    err_s = 7.0
    delta_s_assumption = sigma_s
    remainder = m_n - sigma_pi - delta_s_assumption
    remainder_error = math.sqrt(err_pi * err_pi + err_s * err_s)
    flavours = {
        "charm": {"coefficient": 2.0 / 27.0, "denominator": 27},
        "bottom": {"coefficient": 2.0 / 25.0, "denominator": 25},
        "top": {"coefficient": 2.0 / 23.0, "denominator": 23},
    }
    references = {
        name: {
            "coefficient": values["coefficient"],
            "denominator": values["denominator"],
            "sigma_reference_MeV": values["coefficient"] * remainder,
            "sigma_reference_error_MeV": values["coefficient"] * remainder_error,
        }
        for name, values in flavours.items()
    }
    explicit_sum = sum(item["sigma_reference_MeV"] for item in references.values())
    # The same light-input errors are shared by c/b/t; this is a correlated
    # propagated uncertainty, not three independent errors in quadrature.
    explicit_sum_error = sum(item["coefficient"] for item in references.values()) * remainder_error
    declared = float(mass_budget["sigma_heavy_MeV"])
    declared_err = 3.0
    return {
        "status": "REFERENCE_LO_SEQUENTIAL_MATCHING_DERIVED_NOT_COMPARABLE",
        "source": HEAVY_MATCHING_SOURCE,
        "formula": "sigma_c=(2/27)R, sigma_b=(2/25)R, sigma_t=(2/23)R; R=M_N-sigma_piN-Delta_sM",
        "references": references,
        "explicit_sum_reference": {
            "sigma_c_plus_sigma_b_plus_sigma_t_MeV": explicit_sum,
            "correlated_error_MeV": explicit_sum_error,
            "status": "REFERENCE_ONLY_EXPLICIT_SUM_NOT_MAINTAINED_INPUT",
        },
        "remainder": {
            "value_MeV": remainder,
            "error_MeV": remainder_error,
            "delta_s_M_assumption": "Delta_sM = sigma_sN (point sigma term) for this arithmetic reference only",
            "assumption_status": "UNSOURCED_APPROXIMATION_NOT_IDENTITY",
            "shared_inputs": ["M_N", "sigma_piN", "Delta_sM_assumed_from_sigma_sN"],
            "correlation_note": "c, b, and t references share the same remainder; errors are correlated, not independent.",
        },
        "declared_maintained_heavy_MeV": declared,
        "declared_maintained_heavy_error_MeV": declared_err,
        "maintained_literal_status": "OPAQUE_NOT_COMPARABLE_MISSING_DEFINITION",
        "comparison_status": "NOT_COMPARABLE_MISSING_HEAVY_CONVENTION",
        "fail_closed_note": "The 6 MeV literal has no source, order, active-flavour, per-flavour/total, or uncertainty convention. No difference, pull, or replacement is computed.",
    }


def scalar_operator_conversion_control(
    sigma_record: Mapping[str, Any], mass_record: Mapping[str, Any]
) -> dict[str, Any]:
    """Attempt a scalar matrix-element extraction and fail closed if needed.

    The arithmetic ``<qbar q> = sigma/m_q`` is allowed only when the sigma
    product and the quark mass explicitly carry compatible renormalisation
    metadata.  The maintained producer deliberately does not satisfy that
    contract, so this returns a machine-readable block rather than a guessed
    matrix element.
    """

    try:
        require_metadata(sigma_record)
        require_metadata(mass_record)
    except ValueError as exc:
        return {
            "status": "BLOCKED_MISSING_OPERATOR_TRANSFORMATION",
            "matrix_element": None,
            "reason": str(exc),
            "required": ["same scheme", "same renormalisation scale", "operator definition", "Z_m Z_S relation"],
        }
    if sigma_record.get("scheme") != mass_record.get("scheme") or sigma_record.get("scale_GeV") != mass_record.get("scale_GeV"):
        return {
            "status": "BLOCKED_INCOMPATIBLE_SCHEME_OR_SCALE",
            "matrix_element": None,
            "reason": "sigma product and quark mass are not in an explicitly identical convention",
        }
    sigma = float(sigma_record["value"])
    mass = float(mass_record["value"])
    if mass <= 0.0:
        raise ValueError("quark mass must be positive")
    return {
        "status": "PASS_SOURCE_BACKED_SCALAR_EXTRACTION",
        "matrix_element": sigma / mass,
        "scheme": sigma_record["scheme"],
        "scale_GeV": sigma_record["scale_GeV"],
    }


def source_surface_scan() -> dict[str, Any]:
    """Small static scan proving the inventory names are still present."""

    patterns = {
        "sigma_terms": re.compile(r"sigma_(?:piN|sN|heavy)|sigma_piH|sigma_sH"),
        "current_mass": re.compile(r"M_current|M_cur|current.?mass", re.IGNORECASE),
        "quark_masses": re.compile(r"\bm_[uds]\b"),
    }
    surfaces = sorted({row.producer for row in maintained_input_inventory()})
    rows: list[dict[str, Any]] = []
    for relative in surfaces:
        path = ROOT / relative
        text = path.read_text(encoding="utf-8") if path.exists() else ""
        matches = {name: len(pattern.findall(text)) for name, pattern in patterns.items()}
        rows.append({
            "surface": relative,
            "exists": path.exists(),
            "matches": matches,
        })
    return {
        "status": "PASS_MAINTAINED_SURFACES_SCANNED" if all(row["exists"] for row in rows) else "FAIL_MISSING_SURFACE",
        "surface_count": len(rows),
        "rows": rows,
        "patterns": {name: pattern.pattern for name, pattern in patterns.items()},
    }


def derive_audit_status(
    metadata: Mapping[str, Any],
    scalar_conversion: Mapping[str, Any],
    invariant: Mapping[str, Any],
    source_scan: Mapping[str, Any],
) -> str:
    """Derive the top-level status from executable nested gates."""

    if invariant.get("status") != "PASS_RG_INVARIANT_PRODUCT_ALGEBRA":
        return "FAIL_RG_INVARIANT_CONTROL"
    if source_scan.get("status") != "PASS_MAINTAINED_SURFACES_SCANNED":
        return "FAIL_MAINTAINED_SURFACE_SCAN"
    if metadata.get("status", "").startswith("BLOCKED_") or str(scalar_conversion.get("status", "")).startswith("BLOCKED_"):
        return "COMPLETE_WITH_RENORM_BLOCKS"
    return "COMPLETE_RENORM_AUDIT"


def render_report(result: Mapping[str, Any]) -> str:
    """Render the generated verification report from the current result."""

    budget = result["maintained_mass_budget"]
    heavy = result["heavy_quark_matching"]
    metadata = result["renormalization_contract"]
    scalar = result["scalar_operator_conversion"]
    return "\n".join(
        [
            "# Renormalization-covariance audit (generated)",
            "",
            "> This file is generated by `verification/nvg_renormalization_covariance_audit.py`; it is not a hand-authored claim table.",
            "",
            f"- Status: **{result['status']}**.",
            f"- Maintained producer recomputation: `M_N={budget['mass_MeV']:.1f} MeV`, `sum sigma={budget['sum_sigma_MeV']:.1f} MeV`, `M_Omega={budget['M_Omega_MeV']:.1f} MeV`, `f_Omega={budget['f_Omega']:.6f}`.",
            f"- Inventory gate: `{metadata['status']}`; {metadata['maintained_records']} maintained records, {metadata['blocked_maintained_records']} blocked for missing scheme/scale/operator transformation.",
            f"- Scalar-operator gate: `{scalar['status']}`; no guessed matrix-element conversion is emitted.",
            f"- RG algebra control: `{result['rg_invariant_controls']['status']}`.",
            f"- LO sequential references: charm `{heavy['references']['charm']['sigma_reference_MeV']:.2f} +/- {heavy['references']['charm']['sigma_reference_error_MeV']:.2f}`, bottom `{heavy['references']['bottom']['sigma_reference_MeV']:.2f} +/- {heavy['references']['bottom']['sigma_reference_error_MeV']:.2f}`, top `{heavy['references']['top']['sigma_reference_MeV']:.2f} +/- {heavy['references']['top']['sigma_reference_error_MeV']:.2f} MeV`; explicit sum `{heavy['explicit_sum_reference']['sigma_c_plus_sigma_b_plus_sigma_t_MeV']:.2f} +/- {heavy['explicit_sum_reference']['correlated_error_MeV']:.2f} MeV`.",
            f"- Maintained `sigma_heavy={heavy['declared_maintained_heavy_MeV']:.2f} +/- {heavy['declared_maintained_heavy_error_MeV']:.2f} MeV` remains opaque: `{heavy['comparison_status']}` (no pull or difference computed).",
            "",
            "## Scientific boundary",
            "",
            "Sigma products are RG invariant only when the mass and scalar operator share a declared renormalisation convention. The scalar operator, current-mass remainder, and `M_Omega` bookkeeping split are scheme/scale dependent. In-medium mass predictions remain blocked until hadron-specific scalar responses and operator transformations are sourced. No parameter refit is performed.",
            "",
            "## Sources",
            "",
            "Gupta et al. DOI `10.1103/PhysRevLett.127.242002`; Agadjanov et al. DOI `10.1103/PhysRevLett.131.261902`; FLAG Review 2021 DOI `10.1140/epjc/s10052-022-10536-1`; Kryjevski DOI `10.1103/PhysRevD.70.094028`; Hatta–Rajan–Tanaka DOI `10.1007/JHEP12(2018)008`.",
            "",
        ]
    )


def serialized_artifacts(result: Mapping[str, Any]) -> tuple[bytes, bytes, bytes]:
    """Return canonical result/inventory/report bytes for regeneration checks."""

    payload = (json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")
    inventory = {
        "schema_version": 1,
        "audit": "renormalization-covariance",
        "status": result["status"],
        "records": result["inventory"],
        "policy": result["renormalization_contract"],
        "source_surface_scan": result["source_surface_scan"],
    }
    inventory_payload = (json.dumps(inventory, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")
    report_payload = render_report(result).encode("utf-8")
    return payload, inventory_payload, report_payload


def run_audit() -> dict[str, Any]:
    """Run the complete audit without writing generated artifacts."""

    maintained = maintained_input_inventory()
    external = source_backed_flag_mass_records()
    budget = maintained_mass_budget()
    invariant = invariant_controls()
    heavy = heavy_matching_control(budget)
    maintained_sigma = next(row for row in maintained if row.input_id == "hadron_mass_fractions.sigma_piN")
    maintained_mass = next(row for row in maintained if row.input_id == "hadron_mass_fractions.nucleon_mass")
    scalar_conversion = scalar_operator_conversion_control(asdict(maintained_sigma), asdict(maintained_mass))
    all_records = maintained + external
    blocked_ids = [record.input_id for record in maintained if not metadata_ready(asdict(record))]
    metadata_checks = {
        "maintained_records": len(maintained),
        "source_backed_reference_records": len(external),
        "blocked_maintained_records": len(blocked_ids),
        "blocked_record_ids": blocked_ids,
        "reference_records_ready": all(metadata_ready(asdict(record)) for record in external),
        "conversion_policy": "reject unless scheme + scale + operator definition + transformation are present",
        "status": "BLOCKED_MISSING_RENORM_METADATA" if blocked_ids else "PASS_RENORM_METADATA_COMPLETE",
    }
    surface_scan = source_surface_scan()
    status = derive_audit_status(metadata_checks, scalar_conversion, invariant, surface_scan)
    claims = {
        "sigma_products": "RG_INVARIANT_CONDITIONALLY_ON_MATCHED_MASS_AND_OPERATOR_RENORMALISATION",
        "scalar_operator_matrix_elements": (
            "BLOCKED_MISSING_SCHEME_SCALE_AND_ZS_FOR_MAINTAINED_INPUTS"
            if str(scalar_conversion.get("status", "")).startswith("BLOCKED_")
            else "SOURCE_BACKED_CONVERSION_AVAILABLE"
        ),
        "M_Omega_remainder": "BOOKKEEPING_REMAINDER_SCHEME_DEPENDENT_NOT_A_UNIQUE_OPERATOR",
        "in_medium_hadron_mass_predictions": (
            "BLOCKED_INHERITED_UNPROVEN_CURRENT_MASS_SPLITS"
            if blocked_ids
            else "SOURCE_BACKED_CURRENT_MASS_SPLITS"
        ),
        "heavy_quark_6_MeV": heavy["comparison_status"],
        "parameter_refit": "NOT_PERFORMED",
    }
    return {
        "schema_version": 1,
        "audit": "renormalization-covariance",
        "status": status,
        "artifacts": {
            "result_json": str(RESULT_PATH.relative_to(ROOT)),
            "inventory_json": str(INVENTORY_PATH.relative_to(ROOT)),
            "generated_report": str(GENERATED_REPORT_PATH.relative_to(ROOT)),
            "report": "Lunacy/runs/deep-physics-audit/REPORT-renorm-covariance.md",
            "evidence": "Lunacy/runs/deep-physics-audit/phases/phase-1/evidence/renorm-covariance/",
        },
        "maintained_mass_budget": budget,
        "rg_invariant_controls": invariant,
        "heavy_quark_matching": heavy,
        "scalar_operator_conversion": scalar_conversion,
        "renormalization_contract": metadata_checks,
        "claims": claims,
        "inventory": [asdict(record) for record in all_records],
        "source_surface_scan": surface_scan,
        "limitations": [
            "Sigma products are observables only when the same renormalised mass/operator convention is used.",
            "No cross-scheme or cross-scale scalar matrix-element conversion is attempted for maintained literals.",
            "The LO c/b/t relations are sequential convention references; Delta_sM=sigma_sN is an explicit unsourced approximation and does not authorize comparing or changing sigma_heavy.",
            "An in-medium prediction needs a sourced hadron-specific scalar response and operator matching, absent here.",
        ],
    }


def write_artifacts(result: Mapping[str, Any]) -> None:
    """Write only this step's uniquely named generated artifacts."""

    payload, inventory_payload, report_payload = serialized_artifacts(result)
    RESULT_PATH.write_bytes(payload)
    INVENTORY_PATH.write_bytes(inventory_payload)
    GENERATED_REPORT_PATH.write_bytes(report_payload)


def main() -> dict[str, Any]:
    result = run_audit()
    write_artifacts(result)
    print("RENORMALIZATION COVARIANCE AUDIT")
    print(f"status={result['status']}")
    print(f"maintained_mass_budget: M_Omega={result['maintained_mass_budget']['M_Omega_MeV']:.1f} MeV; f_Omega={result['maintained_mass_budget']['f_Omega']:.6f}")
    print(f"RG product control: {result['rg_invariant_controls']['status']}")
    heavy = result["heavy_quark_matching"]
    refs = heavy["references"]
    total = heavy["explicit_sum_reference"]
    print(
        "LO sequential heavy references: "
        f"c={refs['charm']['sigma_reference_MeV']:.2f}, "
        f"b={refs['bottom']['sigma_reference_MeV']:.2f}, "
        f"t={refs['top']['sigma_reference_MeV']:.2f} MeV; "
        f"explicit sum={total['sigma_c_plus_sigma_b_plus_sigma_t_MeV']:.2f} +/- {total['correlated_error_MeV']:.2f} MeV; "
        f"6 MeV status={heavy['comparison_status']}"
    )
    print(f"renorm metadata: {result['renormalization_contract']['status']} ({result['renormalization_contract']['blocked_maintained_records']} maintained records blocked)")
    print(f"result={RESULT_PATH.relative_to(ROOT)}")
    print(f"inventory={INVENTORY_PATH.relative_to(ROOT)}")
    return result


if __name__ == "__main__":
    main()
