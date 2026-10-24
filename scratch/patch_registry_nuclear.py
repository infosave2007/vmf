"""One-off surgical registry patch for the nuclear-branch files.

Fixes the three droplet consumer-reference errors and registers the finite
static bridge family, the four isovector/Ca48/Sn132/identification probes,
their tests and the cited observables data file.  Text edits are anchored and
fail closed on any ambiguity; the result is re-parsed and validated.
"""

import json
import sys
from pathlib import Path

ROOT = Path("/Users/oleg/Documents/NVG-Research")
REG = ROOT / "verification" / "registry.json"
PROV = ROOT / "verification" / "data" / "provenance.json"


def entry(kind, description, deprecation, consumers, inputs, tests, role="validation"):
    return {
        "consumers": consumers,
        "deprecation": deprecation,
        "description": description,
        "evidence_weight": 0.0,
        "inputs": inputs,
        "kind": kind,
        "outputs": [],
        "producer": None,
        "quarantine": None,
        "role": role,
        "status": "maintained",
        "tests": tests,
    }


NEW_ENTRIES = {
    "verification/nvg_finite_monopole.py": entry(
        "script",
        "Spherical two-component finite-nucleus monopole solver: covariant TF static BVP with scalar, vector and Coulomb fields, multi-seed terminal acceptance gates and the declared covariant_contact_J32 rho-coupling seam.",
        "Static isoscalar finite-nucleus machinery of the accepted W8.93 branch; no shells, pairing, deformation, rotation or transport; zero independent empirical weight.",
        [
            "verification/nvg_finite_static_bridge.py",
            "verification/nvg_isovector_formfactor_probe.py",
            "verification/nvg_ca48_quantum_surface_probe.py",
            "verification/nvg_sn132_discriminating_set_probe.py",
            "verification/nvg_j_identification_audit.py",
            "verification/test_nvg_finite_static_bridge.py",
        ],
        [
            "verification/nvg_isospin_jet_audit.py",
            "verification/nvg_finite_droplet_audit.py",
        ],
        ["verification/test_nvg_finite_static_bridge.py"],
    ),
    "verification/nvg_finite_static_bridge.py": entry(
        "script",
        "Finite-static bridge over the monopole solver: nucleus table, cited AME2020/Angeli observables with electron corrections, radius mapping and terminal-protocol case production.",
        "Conditional static bridge of one accepted branch; binding comparisons are descriptive and carry zero independent empirical weight.",
        [
            "verification/nvg_isovector_formfactor_probe.py",
            "verification/nvg_ca48_quantum_surface_probe.py",
            "verification/nvg_sn132_discriminating_set_probe.py",
            "verification/test_nvg_finite_static_bridge.py",
        ],
        [
            "verification/nvg_finite_monopole.py",
            "verification/data/finite_static_observables_2026.json",
        ],
        ["verification/test_nvg_finite_static_bridge.py"],
    ),
    "verification/nvg_isovector_formfactor_probe.py": entry(
        "script",
        "Isovector contact and form-factor probe on the accepted W8.93 baseline: anchored binding slopes, self-consistent J20/J32 contact probes, point form factors and CVC weak convolutions.",
        "Declared diagnostic probes through the live covariant_contact_J32 machinery; conditional zero-evidence diagnostics, never fitted interactions.",
        [
            "verification/nvg_ca48_quantum_surface_probe.py",
            "verification/nvg_sn132_discriminating_set_probe.py",
            "verification/nvg_j_identification_audit.py",
            "verification/test_nvg_isovector_formfactor_probe.py",
        ],
        [
            "verification/nvg_finite_monopole.py",
            "verification/nvg_finite_static_bridge.py",
        ],
        ["verification/test_nvg_isovector_formfactor_probe.py"],
    ),
    "verification/nvg_ca48_quantum_surface_probe.py": entry(
        "script",
        "Pre-registered 48Ca held-out probe: nucleus-table context patch, volume/surface operator split, WKB validity masking and frozen-profile ETF-2 quantum-surface energies.",
        "Conditional held-out diagnostic of one accepted branch; isotope shifts and skins remain outside static TF; zero independent empirical weight.",
        [
            "verification/nvg_sn132_discriminating_set_probe.py",
            "verification/test_nvg_ca48_quantum_surface_probe.py",
        ],
        [
            "verification/nvg_finite_monopole.py",
            "verification/nvg_finite_static_bridge.py",
            "verification/nvg_isovector_formfactor_probe.py",
        ],
        ["verification/test_nvg_ca48_quantum_surface_probe.py"],
    ),
    "verification/nvg_sn132_discriminating_set_probe.py": entry(
        "script",
        "132Sn completing the pre-registered discriminating set {Zr90, Pb208, Ca48, Sn132}: registered verdict bands, single-constant universality verdict and the universal-j residual-collapse demonstration.",
        "Conditional zero-evidence diagnostic; the universal-j solves are declared probes at fixed s*, not a recalibration or a new theory element.",
        [
            "verification/nvg_j_identification_audit.py",
            "verification/test_nvg_sn132_discriminating_set_probe.py",
        ],
        [
            "verification/nvg_finite_monopole.py",
            "verification/nvg_finite_static_bridge.py",
            "verification/nvg_isovector_formfactor_probe.py",
            "verification/nvg_ca48_quantum_surface_probe.py",
        ],
        ["verification/test_nvg_sn132_discriminating_set_probe.py"],
    ),
    "verification/nvg_j_identification_audit.py": entry(
        "script",
        "Read-only identification audit of the universal j contact: kinetic/contact decomposition at saturation, cited liquid-drop a_sym anchors and the zero-range QHD rho-channel algebra.",
        "Post-hoc interpretation of a completed zero-evidence diagnostic; adds no coefficient and plays no role in the pre-registered verdict.",
        ["verification/test_nvg_j_identification_audit.py"],
        [
            "verification/nvg_isospin_jet_audit.py",
            "verification/nvg_finite_monopole.py",
            "verification/nvg_isovector_formfactor_probe.py",
            "verification/nvg_sn132_discriminating_set_probe.py",
        ],
        ["verification/test_nvg_j_identification_audit.py"],
    ),
    "verification/nvg_retarded_gradient_matching.py": entry(
        "script",
        "Algebraic gradient matching for the accepted retarded Hartree matrix: longitudinal-vector Schur-block removal and Z reconstruction with dZ/dchi, never clipped into a physical gradient.",
        "Algebraic reconstruction on a fixed background; no kernel calculation and no empirical claim; zero independent weight.",
        ["verification/test_nvg_retarded_gradient_matching.py"],
        [],
        ["verification/test_nvg_retarded_gradient_matching.py"],
    ),
    "verification/test_nvg_finite_static_bridge.py": entry(
        "test",
        "Bridge and monopole gates: nucleus table, electron corrections, terminal acceptance and baseline reproducibility controls without launching full BVP ladders.",
        "Focused tests for the conditional static bridge; zero independent empirical weight.",
        [],
        [
            "verification/nvg_finite_monopole.py",
            "verification/nvg_finite_static_bridge.py",
        ],
        [],
    ),
    "verification/test_nvg_isovector_formfactor_probe.py": entry(
        "test",
        "Contact-linearity, anchored-slope, form-factor, CVC weak-operator and strict-JSON controls for the isovector probe.",
        "Focused tests for the zero-evidence isovector diagnostic.",
        [],
        ["verification/nvg_isovector_formfactor_probe.py"],
        [],
    ),
    "verification/test_nvg_ca48_quantum_surface_probe.py": entry(
        "test",
        "Cited Ca48 input arithmetic, nucleus-table patch add/restore, WKB/ETF-2 analytic and decimation controls for the held-out probe.",
        "Focused tests for the zero-evidence Ca48 held-out diagnostic.",
        [],
        [
            "verification/nvg_finite_monopole.py",
            "verification/nvg_isovector_formfactor_probe.py",
            "verification/nvg_ca48_quantum_surface_probe.py",
        ],
        [],
    ),
    "verification/test_nvg_sn132_discriminating_set_probe.py": entry(
        "test",
        "Cited Sn132 input arithmetic, contact linearity, verdict-band and pre-registration controls for the discriminating set.",
        "Focused tests for the zero-evidence discriminating-set diagnostic.",
        [],
        [
            "verification/nvg_finite_monopole.py",
            "verification/nvg_sn132_discriminating_set_probe.py",
        ],
        [],
    ),
    "verification/test_nvg_j_identification_audit.py": entry(
        "test",
        "Cited-anchor constants, contact-convention identities, band logic, fail-closed artifact gates and strict-JSON CLI controls for the identification audit.",
        "Focused tests for the post-hoc zero-evidence identification audit.",
        [],
        [
            "verification/nvg_sn132_discriminating_set_probe.py",
            "verification/nvg_j_identification_audit.py",
        ],
        [],
    ),
    "verification/test_nvg_retarded_gradient_matching.py": entry(
        "test",
        "Schur reconstruction, dZ/dchi and complex-domain controls for the algebraic gradient matching.",
        "Focused tests for the zero-weight algebraic matching diagnostic.",
        [],
        [
            "verification/nvg_retarded_gradient_matching.py",
            "verification/nvg_retarded_response.py",
        ],
        [],
    ),
    "verification/data/finite_static_observables_2026.json": entry(
        "data",
        "Cited evaluated finite-nucleus inputs 2026 (AME2020/ENSDF bindings, Angeli-Marinova charge radii with covariance, PREX/CREX skins, nucleon charge inputs); declared inputs only, never model outputs.",
        "Cited evaluated inputs consumed by the conditional static bridge; zero independent empirical weight.",
        ["verification/nvg_finite_static_bridge.py"],
        [],
        [],
    ),
}


def replace_once(text, old, new, label):
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"FAIL: anchor {label!r} found {count} times (expected 1)")
    return text.replace(old, new, 1)


def render(key, value):
    body = json.dumps(value, indent=2, ensure_ascii=True)
    lines = body.split("\n")
    shifted = "\n".join(("  " + line) if line.strip() else line for line in lines[1:-1])
    return '    {}: {{\n{}\n    }}'.format(json.dumps(key, ensure_ascii=True), shifted)


def main() -> int:
    raw = REG.read_text(encoding="utf-8")
    registry = json.loads(raw)
    entries = registry["entries"]

    overlap = sorted(set(NEW_ENTRIES) & set(entries))
    if overlap:
        raise SystemExit(f"FAIL: already registered: {overlap}")

    # 1) Droplet consumer-reference fixes (three registered errors).
    raw = replace_once(
        raw,
        '        "verification/test_nvg_droplet_stability_audit.py",\n'
        '        "NVG_DROPLET_STABILITY_RU.md"\n      ],',
        '        "verification/test_nvg_droplet_stability_audit.py"\n      ],',
        "droplet_stability consumers",
    )
    raw = replace_once(
        raw,
        '        "verification/test_nvg_droplet_coordinate_response_audit.py",\n'
        '        "NVG_DROPLET_COORDINATE_RESPONSE_RU.md"\n      ],',
        '        "verification/test_nvg_droplet_coordinate_response_audit.py"\n      ],',
        "droplet_coordinate_response consumers",
    )
    raw = replace_once(
        raw,
        '      "consumers": [\n'
        '        "Lunacy/runs/droplet-coordinate-response-2026-09-10/PLAN.md"\n'
        "      ],",
        '      "consumers": [],',
        "adhd consumers",
    )

    # 2) Append the new entries right before the entries object closes.
    block = ",\n".join(render(key, value) for key, value in NEW_ENTRIES.items())
    close_anchor = None
    for candidate in (
        '    }\n  },\n  "front_door": {',
    ):
        if raw.count(candidate) == 1:
            close_anchor = candidate
            break
    if close_anchor is None:
        raise SystemExit("FAIL: entries closing anchor not found uniquely")
    raw = raw.replace(
        close_anchor,
        "    },\n" + block + "\n  },\n  \"front_door\": {",
        1,
    )

    # 3) Re-parse, extend the parsed view, and write back.
    registry = json.loads(raw)
    assert set(NEW_ENTRIES).issubset(registry["entries"])
    REG.write_text(raw, encoding="utf-8")

    # 4) Provenance entry for the cited observables file (exact round-trip format).
    prov_raw = PROV.read_text(encoding="utf-8")
    provenance = json.loads(prov_raw)
    sha = __import__("hashlib").sha256(
        (ROOT / "verification" / "data" / "finite_static_observables_2026.json").read_bytes()
    ).hexdigest()
    provenance["entries"]["finite_static_observables_2026.json"] = {
        "config": "cited evaluated finite-system inputs for the conditional static bridge: "
        "AME2020/ENSDF binding energies, Angeli-Marinova charge radii with covariance, "
        "PREX/CREX PV skins and nucleon charge inputs",
        "kind": "input",
        "producer": None,
        "sha256": sha,
        "source": "Cited evaluated compilations (AME2020/ENSDF via IAEA LiveChart; "
        "Angeli & Marinova ADNDT 99 (2013) 69; published PREX/CREX values)",
        "source_url": "https://nds.iaea.org/relnsd/v1/data",
        "status": "declared_local_configuration_with_primary_references",
        "trust_status": "declared_local_configuration",
        "version": "finite-static-2026",
    }
    PROV.write_text(
        json.dumps(provenance, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    print(f"OK: registered {len(NEW_ENTRIES)} entries; sha256={sha[:16]}...")
    return 0


if __name__ == "__main__":
    sys.exit(main())
