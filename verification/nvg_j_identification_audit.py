#!/usr/bin/env python3
"""Identification audit of the universal isovector contact j = 11.14 MeV.

Post-hoc interpretation layer over the completed discriminating-set result
(``Lunacy/runs/sn132-full-set-2026-09-16/``, verdict
UNIVERSAL_CONSTANT_SUPPORTED).  The audit decodes the contact convention from
the live code, decomposes the design J into the branch's own kinetic symmetry
energy and the interaction part, and identifies both against cited classical
anchors of nuclear physics:

1. Kinetic part ``T_CONTACT = k_F^2/(6 e_F)`` of the branch's saturation
   state (n0 = 0.16 fm^-3, m*/M = 0.930): the Pauli/Fermi-gas symmetry
   energy (free gas at the same density: 11.8-12.3 MeV).
2. Full design ``J_design = T_CONTACT + j``: the Bethe-Weizsaecker asymmetry
   coefficient a_sym of liquid-drop fits (23.2 textbook; 22.90(15) Wang et
   al., PRC 91, 044308 (2015), A = 260; 23.7 upper fits).
3. Interaction part ``j``: the interaction piece ``a_sym - T_kin`` and the
   zero-range limit of the QHD rho channel (contact ``C_rho (n_n-n_p)^2/8``
   with ``j = C_rho n0/8`` <-> ``J_rho = g_rho^2 n0/(8 m_rho^2)``, tau/2
   convention).

Fail-closed identity: the decoded kinetic part must reproduce T_CONTACT
(1e-9 MeV) and the retained artifact must still carry the supported verdict.
No BVP is solved, no production input is patched, no coefficient is added:
this is a read-only identification of an already-measured constant with
``evidence_weight=0``.  Narrative: ``NVG_J_IDENTIFICATION_RU.md``.
Importing this module runs only the contact-convention algebra inherited
from the Sn132 probe import (no BVP, no network access); the CLI prints
strict JSON to stdout and ``--output`` is an explicit opt-in write.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

try:  # Both direct-script and package-style imports are supported.
    import nvg_finite_monopole as static
    import nvg_isovector_formfactor_probe as ivp
    import nvg_sn132_discriminating_set_probe as sn
except ImportError:  # pragma: no cover - package import support.
    from . import nvg_finite_monopole as static
    from . import nvg_isovector_formfactor_probe as ivp
    from . import nvg_sn132_discriminating_set_probe as sn

try:
    from nvg_isospin_jet_audit import inverse_potential_jet
except ImportError:  # pragma: no cover - package import support.
    from .nvg_isospin_jet_audit import inverse_potential_jet

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SCHEMA = "nvg_j_identification_audit.v1"
STATUS = "COMPUTED_J_IDENTIFICATION_AUDIT_ZERO_EVIDENCE"
EVIDENCE_WEIGHT = 0.0

SN132_RESULT_REL = Path("Lunacy/runs/sn132-full-set-2026-09-16/sn132_discriminating_set_result.json")

# Cited classical anchors (see NVG_J_IDENTIFICATION_RU.md, "Sources").
A_SYM_TEXTBOOK_MEV = 23.2  # standard textbook SEMF asymmetry coefficient
A_SYM_WANG_2015_MEV = 22.90  # Wang et al., PRC 91, 044308 (2015), A=260 (Skyrme+LDM)
A_SYM_WANG_2015_SIGMA_MEV = 0.15
A_SYM_FIT_UPPER_MEV = 23.7  # upper end of modern liquid-drop fits
M_RHO_MEV = 775.26  # PDG rho(770) mass
M_RHO_RMF_MEV = 763.0  # meson mass used in FSUGold/IU-FSU tables
RMF_G_RHO_RANGE = (4.1, 6.8)  # published RMF tables, tau/2 convention (arXiv:1304.2121)
N0_FM3 = static.N0_FM3
M_NUCLEON_MEV = 939.0  # branch design nucleon mass

# Registered identification bands (audit conventions fixed here in code).
T_KINETIC_FERMI_GAS_BAND_MEV = (11.0, 13.0)  # free gas 11.8-12.3 and model m*=0.93
J_LIQUID_DROP_REL_TOL = 0.05  # J_design vs at least one cited a_sym
J_INTERACTION_REL_TOL = 0.10  # j vs a_sym - T_kin(free, nonrel)
J_BAR_SANITY_BAND_MEV = (10.0, 13.0)  # the four cited tangents span 10.44-12.38

VERDICT_KEYS = (
    "T_kinetic_in_fermi_gas_band",
    "J_design_matches_liquid_drop_anchor",
    "j_matches_interaction_piece",
    "g_rho_in_published_rmf_range",
)


class JIdentificationError(ValueError):
    """Fail-closed error for non-reproducible inputs or broken identities."""


def _number(value: Any, digits: int = 17) -> float:
    return ivp._number(value, digits)


def _jsonable(value: Any) -> Any:
    return ivp._jsonable(value)


def model_saturation_state() -> dict[str, float]:
    """The branch's homogeneous saturation state (delta = 0) and its kinetic T.

    Reconstructs exactly the state that ``constant_rho_coupling`` uses when it
    fixes the contact from the declared J design, so the decoded kinetic part
    is the theory's own ``T = k_F^2/(6 e_F)``, not an external input.
    """

    design = static.FiniteDesign.from_continuous(ivp.FAMILY, format(ivp.ROOT_SCALE, ".17g"))
    _model, state, _row = inverse_potential_jet(design.target_y, "240", "0.16", "-16")
    k = float(state["k"])
    ef = float(state["ef"])
    return {
        "hbarc_MeV_fm": float(design.hbarc),
        "y_effective_mass_ratio": float(state["y"]),
        "n_saturation_natural": float(state["n"]),
        "k_F_MeV": k,
        "e_F_MeV": ef,
        "T_kinetic_MeV": k * k / (6.0 * ef),
    }


def free_fermi_gas_reference(hbarc: float, n0_fm3: float = N0_FM3) -> dict[str, float]:
    """Free Fermi-gas kinetic symmetry energy at the same density."""

    kf_fm = (3.0 * math.pi**2 * n0_fm3 / 2.0) ** (1.0 / 3.0)
    kf = kf_fm * hbarc
    e_f = math.sqrt(kf * kf + M_NUCLEON_MEV**2)
    return {
        "k_F_MeV": kf,
        "T_nonrel_MeV": kf * kf / (6.0 * M_NUCLEON_MEV),
        "T_relativistic_MeV": kf * kf / (6.0 * e_f),
    }


def liquid_drop_comparison(j_bar: float, t_contact: float, t_nonrel: float) -> dict[str, Any]:
    """Compare the design J and the interaction part j with cited a_sym values."""

    design_j = t_contact + j_bar
    anchors = []
    for label, a_sym in (
        ("textbook_semf_23.2", A_SYM_TEXTBOOK_MEV),
        ("wang_2015_a260_22.90", A_SYM_WANG_2015_MEV),
        ("ldm_fit_upper_23.7", A_SYM_FIT_UPPER_MEV),
    ):
        interaction_piece = a_sym - t_nonrel
        anchors.append(
            {
                "anchor": label,
                "a_sym_MeV": a_sym,
                "design_J_over_a_sym_minus_1": design_j / a_sym - 1.0,
                "interaction_piece_MeV": interaction_piece,
                "j_over_interaction_piece_minus_1": j_bar / interaction_piece - 1.0,
            }
        )
    return {
        "design_J_MeV": design_j,
        "kinetic_share": t_contact / design_j,
        "interaction_share": j_bar / design_j,
        "anchors": anchors,
    }


def rho_channel_identification(j_bar: float, hbarc: float) -> dict[str, float]:
    """Zero-range QHD rho-channel algebra of the contact.

    Contact ``C_rho (n_n-n_p)^2/8`` with ``j = C_rho n0/8`` is the zero-range
    limit of the rho-meson channel ``J_rho = g_rho^2 n0/(8 m_rho^2)`` in the
    tau/2 isovector-current convention (tau convention: g is halved).
    """

    n0_nat = N0_FM3 * hbarc**3
    c_rho_nat = 8.0 * j_bar / n0_nat
    c_rho_fm3 = 8.0 * j_bar / N0_FM3
    g_half = math.sqrt(c_rho_nat * M_RHO_MEV**2)
    g_half_rmf_mass = math.sqrt(c_rho_nat * M_RHO_RMF_MEV**2)
    return {
        "C_rho_MeV_fm3": c_rho_fm3,
        "C_rho_natural": c_rho_nat,
        "g_rho_tau_half_convention": g_half,
        "g_rho_tau_half_m_rho_763": g_half_rmf_mass,
        "g_rho_tau_convention": g_half / 2.0,
    }


def identification_bands(
    state: dict[str, Any],
    free_gas: dict[str, Any],
    ld: dict[str, Any],
    rho: dict[str, Any],
) -> dict[str, Any]:
    """Registered identification bands (audit conventions, not statistics)."""

    kinetic_ok = (
        T_KINETIC_FERMI_GAS_BAND_MEV[0]
        <= float(state["T_kinetic_MeV"])
        <= T_KINETIC_FERMI_GAS_BAND_MEV[1]
    )
    j_anchor_ok = any(
        abs(float(row["design_J_over_a_sym_minus_1"])) <= J_LIQUID_DROP_REL_TOL
        for row in ld["anchors"]
    )
    interaction_ok = any(
        abs(float(row["j_over_interaction_piece_minus_1"])) <= J_INTERACTION_REL_TOL
        for row in ld["anchors"]
    )
    g_rho_ok = (
        RMF_G_RHO_RANGE[0] <= float(rho["g_rho_tau_half_convention"]) <= RMF_G_RHO_RANGE[1]
    )
    return {
        "T_kinetic_in_fermi_gas_band": bool(kinetic_ok),
        "J_design_matches_liquid_drop_anchor": bool(j_anchor_ok),
        "j_matches_interaction_piece": bool(interaction_ok),
        "g_rho_in_published_rmf_range": bool(g_rho_ok),
        "band_values": {
            "T_kinetic_MeV": _number(state["T_kinetic_MeV"]),
            "free_T_nonrel_MeV": _number(free_gas["T_nonrel_MeV"]),
            "free_T_relativistic_MeV": _number(free_gas["T_relativistic_MeV"]),
            "g_rho_tau_half": _number(rho["g_rho_tau_half_convention"]),
            "rmf_g_rho_range": list(RMF_G_RHO_RANGE),
        },
    }


def identification_verdict(bands: dict[str, Any]) -> dict[str, Any]:
    """Assemble the identification verdict from the registered bands."""

    failed = [key for key in VERDICT_KEYS if not bands.get(key)]
    complete = not failed
    return {
        "status": (
            "IDENTIFIED_AS_SYMMETRY_ENERGY_INTERACTION_PART"
            if complete
            else "IDENTIFICATION_INCOMPLETE"
        ),
        "failed_bands": failed,
        "post_hoc_notice": (
            "identification is post-hoc interpretation of a completed zero-evidence "
            "diagnostic; it played no role in the pre-registered verdict and adds no "
            "coefficient to the theory"
        ),
    }


def calculate() -> dict[str, Any]:
    """Run the full read-only identification audit."""

    result_path = ROOT / SN132_RESULT_REL
    try:
        retained = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise JIdentificationError(f"cannot read the discriminating-set artifact: {exc}") from exc

    verdict = retained.get("verdict")
    if not isinstance(verdict, dict) or verdict.get("status") != "UNIVERSAL_CONSTANT_SUPPORTED":
        raise JIdentificationError("identification is defined only for the supported verdict")
    demonstration = retained.get("universal_j_demonstration")
    if not isinstance(demonstration, dict):
        raise JIdentificationError("retained artifact lacks the universal-j demonstration")
    j_bar = demonstration.get("j_bar_MeV")
    t_artifact = demonstration.get("t_contact_MeV")
    if not isinstance(j_bar, (int, float)) or not isinstance(t_artifact, (int, float)):
        raise JIdentificationError("retained artifact lacks the universal-j numbers")
    j_bar = float(j_bar)
    t_artifact = float(t_artifact)

    if abs(t_artifact - sn.T_CONTACT_MEV) > 1.0e-12:
        raise JIdentificationError("artifact T_CONTACT disagrees with the live contact machinery")
    if not J_BAR_SANITY_BAND_MEV[0] <= j_bar <= J_BAR_SANITY_BAND_MEV[1]:
        raise JIdentificationError("artifact j_bar is outside the cited tangent band")

    state = model_saturation_state()
    if abs(state["T_kinetic_MeV"] - sn.T_CONTACT_MEV) > 1.0e-9:
        raise JIdentificationError(
            "decoded kinetic part T = k_F^2/(6 e_F) does not reproduce T_CONTACT"
        )

    hbarc = state["hbarc_MeV_fm"]
    free_gas = free_fermi_gas_reference(hbarc)
    ld = liquid_drop_comparison(j_bar, sn.T_CONTACT_MEV, free_gas["T_nonrel_MeV"])
    rho = rho_channel_identification(j_bar, hbarc)
    bands = identification_bands(state, free_gas, ld, rho)
    verdict_out = identification_verdict(bands)

    return _jsonable(
        {
            "schema": SCHEMA,
            "status": STATUS,
            "evidence_weight": EVIDENCE_WEIGHT,
            "provenance": {
                "source_artifact": SN132_RESULT_REL.as_posix(),
                "source_verdict": "UNIVERSAL_CONSTANT_SUPPORTED",
                "contact_convention_sources": [
                    "verification/nvg_isospin_jet_audit.py::constant_rho_coupling",
                    "verification/nvg_isovector_formfactor_probe.py::_contact_j_added",
                    "verification/nvg_finite_monopole.py::_rho_coupling",
                ],
                "narrative_report": "NVG_J_IDENTIFICATION_RU.md",
                "bvp_solved": False,
                "production_inputs_patched": False,
                "coefficient_added_to_theory": False,
            },
            "contact_convention": {
                "formula": "T = k_F^2/(6 e_F); C_rho = 8 (J_design - T)/n0; "
                "j = C_rho n0/8; J_design = T + j",
                "T_CONTACT_MeV": _number(sn.T_CONTACT_MEV),
                "j_bar_MeV": _number(j_bar),
                "J_design_MeV": _number(ld["design_J_MeV"]),
                "j_exactly_linear_in_J": True,
            },
            "model_saturation_state": {
                key: _number(value) for key, value in state.items()
            },
            "free_fermi_gas_reference": {
                key: _number(value) for key, value in free_gas.items()
            },
            "liquid_drop_comparison": {
                "design_J_MeV": _number(ld["design_J_MeV"]),
                "kinetic_share": _number(ld["kinetic_share"]),
                "interaction_share": _number(ld["interaction_share"]),
                "anchors": [
                    {
                        "anchor": row["anchor"],
                        "a_sym_MeV": _number(row["a_sym_MeV"]),
                        "design_J_over_a_sym_minus_1": _number(row["design_J_over_a_sym_minus_1"]),
                        "interaction_piece_MeV": _number(row["interaction_piece_MeV"]),
                        "j_over_interaction_piece_minus_1": _number(
                            row["j_over_interaction_piece_minus_1"]
                        ),
                    }
                    for row in ld["anchors"]
                ],
            },
            "rho_channel": {
                key: _number(value) for key, value in rho.items()
            },
            "identification_bands": bands,
            "identification_verdict": verdict_out,
            "limits": [
                "post-hoc interpretation of one completed zero-evidence diagnostic; no role in the pre-registered verdict",
                "comparison level is finite nuclei (effective liquid-drop a_sym); uniform-matter J is empirically 31-33 MeV "
                "while the branch's saturation design gives 23.78 MeV (isospin-soft, consistent with the known TF skin deficit)",
                "the contact is the zero-range collapse of several isovector channels (rho exchange, 2-pi exchange, tensor); "
                "vector-type by structure, not a claim about a single exchanged meson",
                "cited anchors carry their own systematics; identification bands are audit conventions, not statistical intervals",
                "identification constrains only the binding channel; radii and skins remain outside static TF (shell physics)",
            ],
        }
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="optional explicit path for the strict JSON result",
    )
    args = parser.parse_args(argv)
    result = calculate()
    payload = json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    sys.stdout.write(payload + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
