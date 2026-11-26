"""Comprehensive unit tests for Law 1: Gauge-Invariant Quartic Vector Completion."""

from __future__ import annotations

import copy
import json
import subprocess
import sys
from dataclasses import replace
from fractions import Fraction
from pathlib import Path

import mpmath as mp
import pytest

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import nvg_quartic_vector_melting_audit as audit  # noqa: E402
from source_complete_solution_audit import PARAMS  # noqa: E402


@pytest.fixture(scope="module")
def result() -> dict:
    return audit.build_result()


def test_live_baseline_parameters_and_spec_v2_metadata(result: dict):
    exact = audit.exact_baseline_from_live()
    assert exact.W0 == Fraction("859")
    assert exact.lam == Fraction("1.05")
    assert exact.M_N == Fraction("939")
    assert exact.m_omega == Fraction("782.6")
    assert exact.g_omega == Fraction("10.12")
    assert exact.n0_fm3 == Fraction("0.16")
    assert exact.hbar_c == Fraction("197.3269804")
    assert exact.degeneracy == 4

    assert result["schema"] == audit.SCHEMA
    assert result["status"] == audit.STATUS
    assert result["evidence_weight"] == 0.0
    assert all(result["gates"].values())
    assert "calibration_targets" in result["calibration_targets_vs_forward_predictions"]["law1_baseline_audit"]
    assert "forward_predictions" in result["calibration_targets_vs_forward_predictions"]["single_scalar_nogo_scan"]


def test_sympy_symbolic_identities_and_weyl_invariance(result: dict):
    sym = result["symbolic_proofs"]
    assert sym["all_symbolic_proofs_exact"] is True
    op = sym["operator_uniqueness_and_invariance"]
    assert op["g_mu_local_U1_invariance_exact_zero"] is True
    assert op["L_g4_local_U1_invariance_exact_zero"] is True
    assert op["unitary_gauge_reduction_exact_zero"] is True
    assert op["weyl_trace_T00_minus_3P_exact_zero"] is True

    th = sym["thermodynamics_and_trace_identity"]
    assert th["fermi_trace_FF_minus_3PF_eq_m_ns_exact"] is True
    assert th["cubic_gauss_eom_exact"] is True
    assert th["pressure_formula_exact"] is True
    assert th["energy_density_formula_exact"] is True
    assert th["equilibrium_trace_identity_unchanged_exact"] is True
    assert th["melted_W0_trace_equals_4B_bag_exact"] is True
    assert th["melted_W0_css_eos_exact"] is True
    assert th["melted_W0_cs2_exact"] == "1/3"


def test_manufactured_local_u1_gauge_control(result: dict):
    gauge = result["manufactured_gauge_control"]
    assert gauge["pass"] is True
    assert mp.mpf(gauge["g_mu_max_abs_diff"]) < mp.mpf("1e-50")
    assert mp.mpf(gauge["L_g4_relative_diff"]) < mp.mpf("1e-50")
    assert mp.mpf(gauge["dL_dA0_variation_relative_error"]) < mp.mpf("1e-45")


def test_precision_60_vs_100_dps_agreement_and_trace_identity(result: dict):
    prec = result["precision_cross_check_60_vs_100_dps"]
    assert mp.mpf(prec["max_relative_difference"]) < mp.mpf("1e-50")
    assert mp.mpf(prec["max_trace_identity_relative_residual_60dps"]) < mp.mpf("1e-50")
    assert mp.mpf(prec["max_trace_identity_relative_residual_100dps"]) < mp.mpf("1e-90")
    assert mp.mpf(prec["max_gauss_relative_residual_100dps"]) < mp.mpf("1e-90")
    # Explicitly well below the <1e-25 audit bar
    assert mp.mpf(prec["max_trace_identity_relative_residual_60dps"]) < mp.mpf("1e-25")


def test_ambient_mpmath_precision_does_not_contaminate_output():
    before = mp.mp.dps
    try:
        mp.mp.dps = 12
        low = audit._canonical_json(audit.build_result())
        mp.mp.dps = 120
        high = audit._canonical_json(audit.build_result())
    finally:
        mp.mp.dps = before
    assert low == high


def test_exact_melting_threshold_formulas_and_values(result: dict):
    mt = result["melting_threshold"]
    with mp.workdps(80):
        c_s = mp.mpf(mt["c_s"])
        z_m = mp.mpf(mt["zeta_melt"])
        zA_m = mp.mpf(mt["zeta_A_melt"])
        zRMF_m = mp.mpf(mt["zeta_RMF_melt"])
        q_phi = mp.mpf("782.6") / mp.mpf("859")
        g_om = mp.mpf("10.12")
        g_s = mp.mpf("939") / mp.mpf("859")
        c_s_check = (mp.mpf("4") / (mp.mpf("4") * mp.pi**2)) * (mp.mpf("6") * mp.pi**2 / mp.mpf("4")) ** (
            mp.mpf("2") / mp.mpf("3")
        ) * g_s**2
        assert abs(c_s - c_s_check) < mp.mpf("1e-28")
        assert abs(z_m - g_om / (q_phi * c_s ** mp.mpf("1.5"))) < mp.mpf("1e-26")
        assert abs(zA_m - z_m * q_phi**4) < mp.mpf("1e-26")
        assert abs(zRMF_m - mp.mpf("6") * z_m * (q_phi / g_om) ** 4) < mp.mpf("1e-28")


def test_sub_threshold_half_zeta_melt_never_melts_and_W_grows(result: dict):
    sub = result["phase_profiles"]["zeta_0p5x_melt_no_melting"]
    assert sub["melts"] is False
    assert mp.mpf(sub["alpha_eff"]) < 0
    rows = sub["rows"]
    assert all(mp.mpf(r["W_MeV"]) > 0 for r in rows)
    # At n = 1000 n0, W grows strictly above W0 = 859 MeV
    assert mp.mpf(rows[-1]["W_MeV"]) > mp.mpf("859")
    assert all(mp.mpf(r["trace_relative_residual"]) < mp.mpf("1e-85") for r in rows)


def test_super_threshold_2x_and_5x_melting_and_exact_css_limit(result: dict):
    for key in ("zeta_2x_melt", "zeta_5x_melt"):
        prof = result["phase_profiles"][key]
        assert mp.mpf(prof["alpha_eff"]) > 0
        assert abs(mp.mpf(prof["spinodal_curvature_at_nmelt_MeV2"])) < mp.mpf("1e-90")
        assert mp.mpf(prof["spinodal_curvature_at_0p99_nmelt_MeV2"]) < 0
        assert mp.mpf(prof["spinodal_curvature_at_1p01_nmelt_MeV2"]) > 0

        for row in prof["rows"]:
            ratio = mp.mpf(row["n_over_n_melt"])
            W = mp.mpf(row["W_MeV"])
            cs2 = mp.mpf(row["cs2"])
            cL2 = mp.mpf(row["c_L_sq"])
            assert mp.mpf(row["trace_relative_residual"]) < mp.mpf("1e-85")
            if ratio < mp.mpf("1.0"):
                assert W > 0
                assert mp.mpf("1") / mp.mpf("3") < cL2 < mp.mpf("1")
            else:
                assert W == 0
                assert abs(cs2 - mp.mpf("1") / mp.mpf("3")) < mp.mpf("1e-26")
                assert abs(cL2 - mp.mpf("1") / mp.mpf("3")) < mp.mpf("1e-26")

    # 5x melts at lower density than 2x
    nm2 = mp.mpf(result["phase_profiles"]["zeta_2x_melt"]["n_melt_over_n0"])
    nm5 = mp.mpf(result["phase_profiles"]["zeta_5x_melt"]["n_melt_over_n0"])
    assert nm5 < nm2


def test_causality_and_negative_control_for_negative_zeta(result: dict):
    caus = result["causality_and_stability"]
    assert caus["positive_zeta_all_states_pass"] is True
    neg_cases = {c["regime"]: c for c in caus["negative_control_zeta_negative"]}
    assert set(neg_cases) == {
        "small_A0_superluminal",
        "mid_A0_gradient_and_ghost",
        "large_A0_tachyonic_and_ghost",
    }
    assert mp.mpf(neg_cases["small_A0_superluminal"]["c_L_sq"]) > mp.mpf("1")
    assert mp.mpf(neg_cases["mid_A0_gradient_and_ghost"]["M0_sq_MeV2"]) < 0
    assert mp.mpf(neg_cases["mid_A0_gradient_and_ghost"]["c_L_sq"]) < 0
    assert mp.mpf(neg_cases["large_A0_tachyonic_and_ghost"]["M0_sq_MeV2"]) < 0
    assert mp.mpf(neg_cases["large_A0_tachyonic_and_ghost"]["MT_sq_MeV2"]) < 0
    assert all(c["violates_causality_or_stability"] is True for c in neg_cases.values())

    # Negative control: Gauss solver rejects zeta < 0 fail-closed
    with mp.workdps(60):
        with pytest.raises(audit.QuarticVectorMeltingError):
            audit.mp_solve_A0(mp.mpf("1e6"), mp.mpf("859"), mp.mpf("-1"), mp.mpf("0.9"), mp.mpf("10.12"))


def test_single_scalar_saturation_nogo_closed_form_scan(result: dict):
    nogo = result["single_scalar_saturation_nogo"]
    assert nogo["all_causal_points_have_m_sigma_below_mpi"] is True

    z0 = nogo["zeta0_limit_summary"]
    assert abs(float(z0["K_MeV"]) - 692.07) < 0.2
    assert float(z0["m_sigma_MeV"]) < 90.0

    b71 = nogo["realistic_K_benchmark_y071"]
    assert abs(float(b71["K_MeV"]) - 227.02) < 0.2
    assert abs(float(b71["m_sigma_MeV"]) - 78.75) < 0.2
    assert float(b71["m_sigma_MeV"]) < float(nogo["reference_m_pi_MeV"])
    assert float(b71["n_melt_over_n0"]) < 1.0
    assert abs(float(b71["n_melt_over_n0"]) - 0.3321) < 0.002
    assert mp.mpf(b71["K_fd_relative_check"]) < mp.mpf("1e-18")

    # Negative control row y = 0.80 lies outside causal domain (zeta < 0)
    row_080 = next(r for r in nogo["scan_rows"] if r["y"] == "0.8")
    assert row_080["physical_causal"] is False
    assert mp.mpf(row_080["X4_MeV4"]) < 0


def test_fail_closed_validation_rejects_tampered_payloads_and_nonlive_params(result: dict):
    assert audit.validate_result(result) is None

    mutations = (
        lambda d: d.__setitem__("evidence_weight", 1.0),
        lambda d: d.__setitem__("status", "TAMPERED_STATUS"),
        lambda d: d["gates"].__setitem__("single_scalar_nogo_verified", False),
        lambda d: d["melting_threshold"].__setitem__("zeta_melt", "1.0"),
        lambda d: d["provenance"].__setitem__("producer_sha256", "0" * 64),
        lambda d: d["causality_and_stability"].__setitem__("positive_zeta_all_states_pass", False),
    )
    for mutate in mutations:
        tampered = copy.deepcopy(result)
        mutate(tampered)
        with pytest.raises(audit.QuarticVectorMeltingError):
            audit.validate_result(tampered)

    with pytest.raises(audit.QuarticVectorMeltingError):
        audit.build_result(replace(PARAMS, lam=0.5))
    with pytest.raises(audit.QuarticVectorMeltingError):
        audit.exact_baseline_from_live(replace(PARAMS, W0=-10.0))


def test_cli_output_validate_and_corrupted_json_rejection(tmp_path: Path):
    script = HERE / "nvg_quartic_vector_melting_audit.py"
    out = tmp_path / "law1_test.json"
    gen = subprocess.run(
        [sys.executable, "-B", str(script), "--output", str(out)],
        cwd=HERE.parent,
        text=True,
        capture_output=True,
        check=True,
    )
    assert "STATUS=" in gen.stdout
    before = out.read_bytes()

    val = subprocess.run(
        [sys.executable, "-B", str(script), "--validate", str(out)],
        cwd=HERE.parent,
        text=True,
        capture_output=True,
        check=True,
    )
    assert val.stdout.strip() == "VALIDATION_PASS"
    assert out.read_bytes() == before

    # Negative control 1: tampered JSON value fails --validate with nonzero exit code
    tampered = json.loads(out.read_text(encoding="utf-8"))
    tampered["melting_threshold"]["zeta_melt"] = "999.0"
    tampered_path = tmp_path / "tampered.json"
    tampered_path.write_text(json.dumps(tampered, sort_keys=True, indent=2), encoding="utf-8")
    bad_val = subprocess.run(
        [sys.executable, "-B", str(script), "--validate", str(tampered_path)],
        cwd=HERE.parent,
        text=True,
        capture_output=True,
    )
    assert bad_val.returncode != 0
    assert "VALIDATION_FAILED" in bad_val.stderr

    # Negative control 2: malformed/corrupted JSON syntax fails --validate
    corrupt_path = tmp_path / "corrupt.json"
    corrupt_path.write_text("{not valid json", encoding="utf-8")
    bad_json = subprocess.run(
        [sys.executable, "-B", str(script), "--validate", str(corrupt_path)],
        cwd=HERE.parent,
        text=True,
        capture_output=True,
    )
    assert bad_json.returncode != 0
    assert "VALIDATION_FAILED" in bad_json.stderr


def test_checked_in_artifact_matches_live_validator():
    assert audit.RESULT_PATH.exists(), "canonical results JSON must exist"
    payload = json.loads(audit.RESULT_PATH.read_text(encoding="utf-8"))
    assert audit.validate_result(payload) is None


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__]))
