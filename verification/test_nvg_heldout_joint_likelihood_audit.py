"""Tests for the joint held-out likelihood audit (fast: DD-ME2, TW99 end to end, NL3 16O fields; about 15 s)."""

import hashlib
import json
import math
import pathlib
import sys
import types
import unittest

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import nvg_heldout_joint_likelihood_audit as audit  # noqa: E402

RESULT = HERE / "nvg_heldout_joint_likelihood_results.json"

# Names carrying the held-out data or the comparison with it; no EOS-construction code may reference them.
HELDOUT_NAMES = {"PREREG", "PREREG_JSON", "DATA", "TERMS", "MKEY", "PSR", "BOX_MMAX", "DATA_SOURCE_NOTES", "SO_DECLARED",
                 "SESSION_REFERENCE", "predictions", "chi2_terms", "fixed_model_stats", "likelihood_block", "box", "sides",
                 "T1_rep", "two_piece_cdf", "heldout_compare", "HELDOUT_STAR_DATA"}


def _code_names(obj):
    """All global names referenced by a function/method (recursively through nested code objects)."""
    code = getattr(obj, "__code__", None)
    if code is None:
        return set()
    names = set(code.co_names)
    stack = [code]
    while stack:
        c = stack.pop()
        for const in c.co_consts:
            if isinstance(const, types.CodeType):
                names.update(const.co_names)
                stack.append(const)
    return names


def _class_functions(cls):
    return [v for v in vars(cls).values() if isinstance(v, types.FunctionType)]


def _no_nan(text):
    def bad(token):
        raise ValueError(f"non-strict JSON token {token}")
    return json.loads(text, parse_constant=bad)


class Poison(dict):
    def __getitem__(self, k):
        raise AssertionError("held-out data accessed during EOS construction / stellar solve")

    def get(self, *a, **k):
        raise AssertionError("held-out data accessed during EOS construction / stellar solve")

    def items(self):
        raise AssertionError("held-out data accessed during EOS construction / stellar solve")


class HeldoutJointLikelihoodTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ddme2 = audit.evaluate_model("DD-ME2")
        cls.tw99 = audit.evaluate_model("TW99")

    # ---- DD-ME2 through the declared likelihood
    def test_ddme2_chi2_and_J0614(self):
        r = self.ddme2
        self.assertEqual(r["status"], "ok")
        self.assertTrue(r["model"]["validated_1pct"])
        lk = r["likelihood"]["primary_rotation"]
        self.assertTrue(r["likelihood"]["declared_8_term_statistic"])
        self.assertEqual(lk["nu"], 8)
        self.assertAlmostEqual(lk["chi2"], 19.95, delta=0.02)
        self.assertAlmostEqual(lk["z_PIT"]["R144"], -3.17, delta=0.01)
        self.assertAlmostEqual(lk["p_chi2_nu"], 0.0105, delta=5e-4)
        self.assertAlmostEqual(r["stars"]["M_max"], 2.483, delta=1e-3)
        self.assertFalse(lk["passes_S"])
        self.assertFalse(lk["box"]["box2"])
        self.assertEqual(lk["box"]["fails2"], ["R1418", "R144", "Lt"])
        jv = r["joint_verdict"]
        self.assertEqual((jv["S"], jv["O"], jv["G"], jv["V"], jv["converges"]), (False, True, True, "not_applicable", False))

    # ---- TW99, the published set closest to all data
    def test_tw99_chi2(self):
        lk = self.tw99["likelihood"]["primary_rotation"]
        self.assertAlmostEqual(lk["chi2"], 11.15, delta=0.02)
        self.assertAlmostEqual(lk["p_chi2_nu"], 0.193, delta=2e-3)
        self.assertAlmostEqual(lk["z_PIT"]["R144"], -2.14, delta=0.01)
        self.assertFalse(lk["passes_S"])                       # J0614 is its only |z| >= 2
        self.assertEqual(lk["box"]["fails2"], ["R144"])
        self.assertTrue(self.tw99["spin_orbit"]["passes_O"])
        self.assertAlmostEqual(self.tw99["spin_orbit"]["split_16O_1p_LDA_MeV"], 6.99, delta=0.01)
        # static radii (declared secondary configuration): TW99 passes S, the rotation correction decides
        st = self.tw99["likelihood"]["secondary_static"]
        self.assertAlmostEqual(st["chi2"], 10.60, delta=0.02)
        self.assertTrue(st["passes_S"])

    # ---- NL3 16O 1p splitting of the validated LDA procedure
    def test_nl3_16O_splitting(self):
        m = audit.build_rmf("NL3")
        m.saturation()
        S, V = m.snm_fields(audit.N_GRID)
        so = audit.spin_orbit_compare(audit.spin_orbit_16O(audit.FieldTable("NL3", audit.N_GRID, S, V)))
        self.assertAlmostEqual(so["split_16O_1p_LDA_MeV"], 6.84, delta=0.01)
        self.assertAlmostEqual(so["split_16O_1p_exact_dirac_MeV"], 6.84, delta=0.02)
        self.assertTrue(so["passes_O"])
        self.assertEqual(so["nodes"], 0)
        # DD-ME2 (with rearrangement in V) from the end-to-end run
        self.assertAlmostEqual(self.ddme2["spin_orbit"]["split_16O_1p_LDA_MeV"], 6.51, delta=0.01)

    # ---- Hartle-Thorne equatorial-radius correction: sign and size on one star
    def test_rotation_correction_sign_and_size(self):
        rot = self.ddme2["stars"]["rotation"]["J0614"]
        self.assertEqual(rot["mode"], "fixed_mass")
        dR = rot["dR_rot"]
        self.assertGreater(dR, 0.0)                              # spin flattens the star: R_eq grows
        self.assertAlmostEqual(dR, 0.2121, delta=2e-3)          # 317.6 Hz at 1.44 Msun, DD-ME2
        self.assertLess(dR / rot["R_static_bg"], 0.03)            # second order in Omega: percent-level
        self.assertGreater(rot["dReq_fixedPc"], dR)              # fixed-mass correction is smaller than fixed-Pc
        R_static = self.ddme2["stars"]["R_of_M"]["1.44"]
        R_rot = self.ddme2["likelihood"]["primary_rotation"]["predictions"]["R144"]
        self.assertAlmostEqual(R_rot - R_static, dR, places=12)
        ob2 = rot["Omega_bar2"]
        self.assertGreater(dR / R_static / ob2, 0.3)              # Maclaurin limit 5/12; compact stars same order
        self.assertLess(dR / R_static / ob2, 1.5)
        self.assertGreater(self.ddme2["stars"]["rotation"]["J0740"]["dR_rot"], 0.0)
        self.assertGreater(self.tw99["stars"]["rotation"]["J0614"]["dR_rot"], 0.0)

    # ---- evaluate_table is the public path; a user table without nuclear terms is flagged
    def test_evaluate_table_user_path(self):
        m = audit.build_rmf("DD-ME2")
        sat = m.saturation()
        tab = m.build_table(0.1, 12.0, 250)
        r = audit.evaluate_table(tab["n_fm3"], tab["eps"], tab["P_cons"], n0_fm3=sat["n0"], rotation=False)
        self.assertEqual(r["status"], "ok")
        self.assertFalse(r["likelihood"]["declared_8_term_statistic"])
        self.assertEqual(r["likelihood"]["primary_rotation"]["nu"], 6)
        self.assertIsNone(r["joint_verdict"]["S"])
        self.assertAlmostEqual(r["stars"]["M_max"], self.ddme2["stars"]["M_max"], delta=1e-9)
        self.assertLess(abs(r["crust_match"]["eps_jump_rel"]), 1e-12)
        with self.assertRaises(ValueError):
            audit.evaluate_table([0.1, 0.2], [1.0, 2.0], [0.1, 0.2])

    # ---- held-out data never enter any EOS construction
    def test_heldout_data_isolated_from_eos_construction(self):
        construction = [audit.build_rmf, audit.star_block, audit.branch_logged, audit.pc_for_mass, audit.dRdM_static,
                        audit.rotation_corrections, audit.rotating_star, audit.background_pass, audit.surface_deformation,
                        audit.spin_orbit_16O, audit.lda_profiles, audit.level_lda, audit.dirac_level, audit.third_order, audit.kin_sym,
                        audit.FB.build_eos, audit.FB.match_sly, audit.FB.build_beta_table, audit.FB.scalar_model, audit.FB.scalar_jet,
                        audit.FB.isovector, audit.FB.stellar_branch, audit.FB.solve_tov_tidal]
        for cls in (audit.RMFBase, audit.NLModel, audit.DDModel, audit.Matter, audit.NVGFields, audit.TabBackground, audit.Nucleus,
                    audit.FieldTable, audit.FB.BetaSolver, audit.FB.EOS, audit.FB.MergedEOS, audit.FB.SlyCrust):
            construction.extend(_class_functions(cls))
        for fn in construction:
            names = _code_names(fn)
            self.assertFalse(names & HELDOUT_NAMES, f"{fn.__qualname__} references held-out data: {names & HELDOUT_NAMES}")
        # the data are read by the comparison functions only
        for fn in (audit.predictions, audit.chi2_terms, audit.sides):
            self.assertIn("DATA", _code_names(fn))
        self.assertIn("SO_DECLARED", _code_names(audit.spin_orbit_compare))
        # dynamic check: poison the held-out data; EOS table, crust, stars and rotation must be bit-identical
        m = audit.build_rmf("TW99")
        sat = m.saturation()
        tab = m.build_table(0.1, 12.0, 250)
        saved = (audit.PREREG, audit.DATA)
        audit.PREREG, audit.DATA = Poison(), Poison()
        try:
            m2 = audit.build_rmf("TW99")
            sat2 = m2.saturation()
            tab2 = m2.build_table(0.1, 12.0, 250)
            t = {"n_fm3": tab2["n_fm3"], "eps": tab2["eps"], "P_cons": tab2["P_cons"], "P_author": tab2["P_cons"], "x": tab2["x"]}
            eos, info = audit.FB.build_eos(t, crust="sly", match="eps", pressure="consistent")
            stars = audit.star_block(eos)
            with self.assertRaises(AssertionError):
                audit.likelihood_block(stars, {"S2_0p10": sat2["S2_0p10"], "EPNM_0p10": sat2["EPNM_0p10"]})
        finally:
            audit.PREREG, audit.DATA = saved
        self.assertEqual(sat, sat2)
        self.assertTrue(np.array_equal(tab["P_cons"], tab2["P_cons"]))
        self.assertAlmostEqual(stars["M_max"], self.tw99["stars"]["M_max"], delta=1e-9)
        self.assertAlmostEqual(stars["rotation"]["J0614"]["dR_rot"], self.tw99["stars"]["rotation"]["J0614"]["dR_rot"], delta=1e-7)
        # the NVG coefficient solve reads no held-out quantity either
        p = audit.NVG_MEMBERS["NVG_island_y0.84_a5min"]
        audit.PREREG, audit.DATA = Poison(), Poison()
        try:
            sm = audit.FB.scalar_model(p, p["scalar"])
            iso = audit.FB.isovector(sm["jet"], p["J"], p["L"], p["form"])
            nt = audit.FB.build_beta_table(sm["jet"], iso, [0.9, 1.0, 1.1], global_check=False)
        finally:
            audit.PREREG, audit.DATA = saved
        self.assertEqual(len(nt["P_cons"]), 3)
        for rec in audit.NVG_MEMBERS.values():
            self.assertNotIn("R_", json.dumps(rec))

    # ---- pre-registration embedded verbatim
    def test_prereg_hash(self):
        self.assertEqual(audit.prereg_sha256(), audit.PREREG_SHA256_DECLARED)
        self.assertEqual(audit.PREREG_SHA256_DECLARED, "db896ff88aadacda0490369ab4dbc26e22eac22f9199bd06c8e211b2fdc89f78")
        self.assertEqual(len(audit.DATA), 8)

    # ---- strict JSON artifact
    def test_results_json_strict_and_validates(self):
        self.assertTrue(RESULT.is_file(), RESULT)
        data = _no_nan(RESULT.read_text(encoding="utf-8"))
        self.assertEqual(data["status"], "HELDOUT_JOINT_LIKELIHOOD_COMPUTED_ZERO_EVIDENCE")
        self.assertEqual(data["evidence_weight"], 0.0)
        self.assertEqual(data["source_sha256"], hashlib.sha256(audit.SOURCE_PATH.read_bytes()).hexdigest())
        self.assertTrue(audit.validate_result(data))
        rows = {r["model"]: r for r in data["model_table"]}
        self.assertEqual(set(rows), set(audit.MODEL_ORDER))
        self.assertAlmostEqual(rows["DD-ME2"]["chi2"], 19.95, delta=0.02)
        self.assertAlmostEqual(rows["TW99"]["chi2"], 11.15, delta=0.02)
        self.assertAlmostEqual(rows["NL3"]["split_16O_1p_LDA"], 6.84, delta=0.01)
        for r in rows.values():
            self.assertEqual(r["status"], "ok")
            self.assertFalse(r["converges"])
        for name in ("NVG_island_y0.84_a5min", "NVG_quartic_y0.87"):
            self.assertFalse(rows[name]["O"])
            self.assertLess(rows[name]["split_16O_1p_LDA"], 4.9)
        self.assertTrue(data["validation"]["hartle_thorne_maclaurin_limit"]["pass_1e-2"])
        self.assertTrue(data["validation"]["spin_orbit_validation_rule"]["both_in_band"])
        self.assertTrue(data["validation"]["prereg_sha256_matches"])
        tampered = json.loads(json.dumps(data))
        tampered["models"]["DD-ME2"]["likelihood"]["primary_rotation"]["chi2"] = 5.0
        self.assertFalse(audit.validate_result(tampered))
        self.assertTrue(math.isfinite(data["wall_s"]))


if __name__ == "__main__":
    unittest.main()
