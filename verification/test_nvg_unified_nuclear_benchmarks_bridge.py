"""Pytest suite for Bridge 9B: Full Tin Chain, PREX-2/CREX Survivor Locus & Live RMF Comparators."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import nvg_unified_nuclear_benchmarks_bridge as b9b

ARTIFACT_PATH = HERE / "nvg_unified_nuclear_benchmarks_results.json"
SCRIPT_PATH = HERE / "nvg_unified_nuclear_benchmarks_bridge.py"


@pytest.fixture(scope="module")
def artifact() -> dict:
    assert ARTIFACT_PATH.exists(), f"Missing generated artifact: {ARTIFACT_PATH}"
    return json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))


def test_zero_evidence_schema_and_live_validate(tmp_path: Path, artifact: dict) -> None:
    assert artifact["schema_version"] == b9b.SCHEMA_VERSION
    assert artifact["bridge_id"] == b9b.BRIDGE_ID
    assert artifact["status"] == b9b.STATUS_PASS
    assert artifact["evidence_weight"] == 0.0
    assert artifact["observed_likelihood"] is None
    assert all(bool(v) for v in artifact["gates"].values())

    proc = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "--validate", str(ARTIFACT_PATH)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr

    # Negative controls (fast fail-closed checks)
    bad_w = tmp_path / "bad_weight.json"
    mod_w = json.loads(json.dumps(artifact))
    mod_w["evidence_weight"] = 1.0
    bad_w.write_text(json.dumps(mod_w), encoding="utf-8")
    ok, errs = b9b.validate_artifact(bad_w)
    assert not ok and any("evidence_weight" in e for e in errs)

    bad_sha = tmp_path / "bad_sha.json"
    mod_sha = json.loads(json.dumps(artifact))
    mod_sha["provenance"]["bridge9a_results_sha256"] = "0" * 64
    bad_sha.write_text(json.dumps(mod_sha), encoding="utf-8")
    ok, errs = b9b.validate_artifact(bad_sha)
    assert not ok and errs


def test_full_tin_chain_and_survivor_prex_crex(artifact: dict) -> None:
    tin = artifact["full_tin_isotopic_chain_Sn100_to_Sn134"]
    isos = tin["isotopes"]
    assert len(isos) == 11
    assert tin["zero_pairing_refit"] is True
    assert tin["summary"]["Sn100_N50_collapsed"] is True
    assert tin["summary"]["Sn132_N82_collapsed"] is True
    assert tin["summary"]["max_midshell_Delta_n_MeV"] > 0.6
    assert tin["summary"]["Sn134_post_shell_Delta_n_MeV"] > 0.5
    assert tin["summary"]["delta_2n_Sn132_shell_gap_MeV"] > 5.0

    survs = artifact["survivor_prex_crex_and_ksym_locus"]
    assert len(survs) == 8
    assert sum(1 for s in survs if s["is_winner"]) == 1
    for s in survs:
        assert s["Delta_R_np_Pb208_fm"] > 0.12
        assert s["Delta_R_np_Ca48_fm"] > 0.10
        assert len(s["nuclei"]) == 6

    so = artifact["winner_spin_orbit_splittings_MeV"]
    assert set(so.keys()) == {"O16", "Ca40", "Ca48", "Zr90", "Sn132", "Pb208"}
    assert so["O16"]["1p"]["neutron_MeV"] > 5.0


def test_live_rmf_comparators_and_vacuum_audit(artifact: dict) -> None:
    rmfs = artifact["rmf_live_comparators"]
    assert len(rmfs) == 5
    names = [r["name"] for r in rmfs]
    assert names == ["NL3", "FSUGold", "IU-FSU", "FSUGold2", "FSUGarnet"]
    for r in rmfs:
        assert all(n["converged"] and n["particle_rel_err"] < 1.0e-6 for n in r["six_nuclei"]["nuclei"])

    sc = artifact["held_out_empirical_scorecard"]
    assert len(sc) == 6
    nvg = sc[0]
    assert nvg["zero_finite_nucleus_refit"] is True
    assert nvg["k_fin_nuc"] == 0
    assert nvg["global_vacuum_bounded_and_unique"] is True
    assert nvg["passes_pulsar_2_08_and_gw_580"] is True
    assert any(not r["global_vacuum_bounded_and_unique"] for r in sc[1:])
