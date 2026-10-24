#!/usr/bin/env python3
"""One-off surgical registry patch: register the PREX/CREX consistency probe."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REGISTRY = ROOT / "verification" / "registry.json"
ANCHOR = '    }\n  },\n  "front_door": {'


def render(key: str, value: dict) -> str:
    block = {key: value}
    text = json.dumps(block, indent=2, ensure_ascii=True)
    lines = text.splitlines()
    body = lines[1:-1]
    return "\n".join("  " + line if line.strip() else line for line in body)


NEW_ENTRIES: dict[str, dict] = {
    "verification/nvg_prex_crex_external_consistency_probe.py": {
        "consumers": ["verification/test_nvg_prex_crex_external_consistency_probe.py"],
        "deprecation": "Conditional zero-evidence external-consistency audit against published PREX-II/CREX values; the experimental numbers are public knowledge, so this is not a blind prediction, and the plane-wave F_W/A_PV comparisons are descriptive only.",
        "description": "PREX-II/CREX external-consistency probe: point neutron skins of Pb208/Ca48 on the no_rho and identified-j branches against the published two-sigma bands (pre-registered), with fail-closed reproduction of three retained artifacts, five multi-seed terminal solves, plane-wave F_W/A_PV at the measured kinematics and the naive contact strength required for each skin.",
        "evidence_weight": 0.0,
        "inputs": [
            "verification/nvg_finite_monopole.py",
            "verification/nvg_finite_static_bridge.py",
            "verification/nvg_isovector_formfactor_probe.py",
            "verification/nvg_sn132_discriminating_set_probe.py",
        ],
        "kind": "script",
        "outputs": [],
        "producer": None,
        "quarantine": None,
        "role": "validation",
        "status": "maintained",
        "tests": ["verification/test_nvg_prex_crex_external_consistency_probe.py"],
    },
    "verification/test_nvg_prex_crex_external_consistency_probe.py": {
        "consumers": [],
        "deprecation": "Focused tests for the zero-evidence PREX/CREX external-consistency audit.",
        "description": "Published-band and kinematics checks, verdict logic, A_PV reconstruction identity against the form-factor operator stack on synthetic profiles, fail-closed retained-artifact controls and retained-payload checks for the PREX/CREX consistency probe.",
        "evidence_weight": 0.0,
        "inputs": [
            "verification/nvg_prex_crex_external_consistency_probe.py",
            "verification/nvg_isovector_formfactor_probe.py",
            "verification/nvg_isovector_transferability_probe.py",
        ],
        "kind": "test",
        "outputs": [],
        "producer": None,
        "quarantine": None,
        "role": "validation",
        "status": "maintained",
        "tests": [],
    },
}


def main() -> int:
    text = REGISTRY.read_text(encoding="utf-8")
    if text.count(ANCHOR) != 1:
        print(f"FAIL: front_door anchor count = {text.count(ANCHOR)}")
        return 1
    registry = json.loads(text)
    entries = registry["entries"]
    for key in NEW_ENTRIES:
        if key in entries:
            print(f"FAIL: {key} already registered")
            return 1
    # References may point at entries registered together in this patch.
    known = set(entries) | set(NEW_ENTRIES)
    for key, entry in NEW_ENTRIES.items():
        for field in ("inputs", "consumers", "tests"):
            for ref in entry.get(field) or []:
                if ref not in known:
                    print(f"FAIL: {key} references unregistered entry {ref}")
                    return 1
    blocks = ",\n".join(render(key, value) for key, value in NEW_ENTRIES.items())
    old = ANCHOR
    new = "    },\n" + blocks + "\n  },\n  \"front_door\": {"
    patched = text.replace(old, new, 1)
    reparsed = json.loads(patched)
    added = sum(1 for key in NEW_ENTRIES if key in reparsed["entries"])
    if added != len(NEW_ENTRIES):
        print("FAIL: re-parse did not contain the new entries")
        return 1
    REGISTRY.write_text(patched, encoding="utf-8")
    print(f"OK: registered {len(NEW_ENTRIES)} entries (total {len(reparsed['entries'])})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
