#!/usr/bin/env python3
"""Surgical registry patch: register forward beta EOS + held-out TOV + their tests."""
from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REGISTRY = ROOT / "verification" / "registry.json"
MANIFEST = ROOT / "verification" / "artifact_manifest.json"
ANCHOR = '    }\n  },\n  "front_door": {'

import hashlib

def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()

def render(key: str, value: dict) -> str:
    block = {key: value}
    text = json.dumps(block, indent=2, ensure_ascii=True)
    lines = text.splitlines()
    body = lines[1:-1]
    return "\n".join("  " + line if line.strip() else line for line in body)

# hashes are checked at write time but stored in manifest; registry itself has no hash
NEW_ENTRIES = {
    "verification/nvg_forward_beta_eos_audit.py": {
        "kind": "script",
        "role": "validation",
        "status": "maintained",
        "evidence_weight": 0.0,
        "description": "Forward beta-equilibrated EOS on the globally-stable isoscalar jet (y>=0.87, K=240) with density-dependent isovector C(n)=C0 (n/n0)^gamma(L=50); no fit to stars or nuclei.",
        "producer": None,
        "consumers": ["verification/test_nvg_forward_beta_eos_audit.py", "verification/nvg_forward_tov_heldout_audit.py"],
        "inputs": ["verification/source_complete_scaling_saturation_audit.py", "verification/nvg_isospin_jet_audit.py", "verification/nvg_isovector_density_dependent_audit.py"],
        "outputs": ["verification/nvg_forward_beta_eos_results.json"],
        "tests": ["verification/test_nvg_forward_beta_eos_audit.py"],
        "deprecation": "Forward beta EOS table with identical crust not matched; p(eps) is a zero-weight prediction, not an EOS completion.",
        "quarantine": None,
    },
    "verification/test_nvg_forward_beta_eos_audit.py": {
        "kind": "test",
        "role": "validation",
        "status": "maintained",
        "evidence_weight": 0.0,
        "description": "Live 80/110-dps regeneration, bracket and beta-equilibrium checks for the forward beta EOS table.",
        "producer": None,
        "consumers": [],
        "inputs": ["verification/nvg_forward_beta_eos_audit.py", "verification/nvg_forward_beta_eos_results.json"],
        "outputs": [],
        "tests": [],
        "deprecation": "Regression controls for a zero-weight forward beta table.",
        "quarantine": None,
    },
    "verification/nvg_forward_beta_eos_results.json": {
        "kind": "artifact",
        "role": "validation",
        "status": "maintained",
        "evidence_weight": 0.0,
        "description": "Forward beta EOS table p(eps) for y=0.87,0.88,0.90,0.93 at gamma(L=50); held-out stellar matter without crust matching.",
        "producer": "verification/nvg_forward_beta_eos_audit.py",
        "consumers": ["verification/nvg_forward_tov_heldout_audit.py"],
        "inputs": [],
        "outputs": [],
        "tests": ["verification/test_nvg_forward_beta_eos_audit.py"],
        "deprecation": "Forward consequence with no independent empirical weight; not a validated EOS.",
        "quarantine": None,
    },
    "verification/nvg_forward_tov_heldout_audit.py": {
        "kind": "script",
        "role": "validation",
        "status": "maintained",
        "evidence_weight": 0.0,
        "description": "Held-out TOV+Hinderer tidal audit on the forward beta EOS; identical crust Gamma=1.35 p_match=1.5, DOP853, no tuning to J0740/GW170817/NICER.",
        "producer": None,
        "consumers": ["verification/test_nvg_forward_tov_heldout_audit.py"],
        "inputs": ["verification/nvg_forward_beta_eos_results.json"],
        "outputs": ["verification/nvg_forward_tov_heldout_results.json"],
        "tests": ["verification/test_nvg_forward_tov_heldout_audit.py"],
        "deprecation": "Held-out descriptive comparisons only; stellar numbers are not evidence for the EOS and carry zero weight.",
        "quarantine": None,
    },
    "verification/test_nvg_forward_tov_heldout_audit.py": {
        "kind": "test",
        "role": "validation",
        "status": "maintained",
        "evidence_weight": 0.0,
        "description": "Regeneration, crust-identity and convergence checks for the held-out TOV tidal results.",
        "producer": None,
        "consumers": [],
        "inputs": ["verification/nvg_forward_tov_heldout_audit.py", "verification/nvg_forward_tov_heldout_results.json", "verification/nvg_forward_beta_eos_results.json"],
        "outputs": [],
        "tests": [],
        "deprecation": "Regression controls for a zero-weight held-out stellar prediction.",
        "quarantine": None,
    },
    "verification/nvg_forward_tov_heldout_results.json": {
        "kind": "artifact",
        "role": "validation",
        "status": "maintained",
        "evidence_weight": 0.0,
        "description": "Held-out Mmax/R14/Lambda from the forward beta EOS (no fit); Mmax=1.05-1.85 Msun, none reaches 2 Msun.",
        "producer": "verification/nvg_forward_tov_heldout_audit.py",
        "consumers": [],
        "inputs": [],
        "outputs": [],
        "tests": ["verification/test_nvg_forward_tov_heldout_audit.py"],
        "deprecation": "Descriptive held-out output with zero evidence weight; not a confirmation of the EOS.",
        "quarantine": None,
    },
}

NEW_MANIFEST = {
    "verification/nvg_forward_beta_eos_results.json": {
        "canonical_path": "verification/nvg_forward_beta_eos_results.json",
        "command": "arch -arm64 .venv-research/bin/python -B verification/nvg_forward_beta_eos_audit.py --write",
        "kind": "generated_artifact",
        "producer": "verification/nvg_forward_beta_eos_audit.py",
        "sha256": sha(ROOT / "verification/nvg_forward_beta_eos_results.json"),
        "status": "forward_beta_eos_zero_evidence",
    },
    "verification/nvg_forward_tov_heldout_results.json": {
        "canonical_path": "verification/nvg_forward_tov_heldout_results.json",
        "command": "arch -arm64 .venv-research/bin/python -B verification/nvg_forward_tov_heldout_audit.py --write",
        "kind": "generated_artifact",
        "producer": "verification/nvg_forward_tov_heldout_audit.py",
        "sha256": sha(ROOT / "verification/nvg_forward_tov_heldout_results.json"),
        "status": "heldout_tov_tidal_zero_evidence",
    },
}

def main() -> int:
    text = REGISTRY.read_text(encoding="utf-8")
    if text.count(ANCHOR) != 1:
        print(f"FAIL anchor {text.count(ANCHOR)}")
        return 1
    reg = json.loads(text)
    for k in NEW_ENTRIES:
        if k in reg["entries"]:
            print(f"FAIL already {k}"); return 1
    known = set(reg["entries"]) | set(NEW_ENTRIES)
    for k, e in NEW_ENTRIES.items():
        for field in ("inputs","consumers","tests","outputs"):
            for ref in e.get(field) or []:
                if ref not in known:
                    print(f"FAIL {k} -> {ref}"); return 1
    # also consumer of nvg_isovector_density_dependent_audit
    # ensure that producer's consumers will be consistent: need to update that file's consumers? But registry check only verifies
    # that if producer is set, consumers contains the reverse? We'll allow by not checking strictly; but to avoid producer/consumer mismatch,
    # we keep inputs as read-only references (not outputs of those producers), so no producer/consumer edge needed.
    blocks = ",\n".join(render(k,v) for k,v in NEW_ENTRIES.items())
    patched = text.replace(ANCHOR, "    },\n" + blocks + "\n  },\n  \"front_door\": {", 1)
    json.loads(patched)
    REGISTRY.write_text(patched, encoding="utf-8")
    print(f"OK registry {len(NEW_ENTRIES)} entries")

    mtext = MANIFEST.read_text(encoding="utf-8")
    man = json.loads(mtext)
    for k,v in NEW_MANIFEST.items():
        if k in man["entries"]:
            print(f"FAIL manifest {k}"); return 1
        man["entries"][k]=v
    MANIFEST.write_text(json.dumps(man, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    print(f"OK manifest {len(NEW_MANIFEST)} entries")
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())
