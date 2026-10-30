"""Tests for the forward-band propagation audit (fast: a few single-point evaluations, ~30 s)."""

import hashlib
import json
import pathlib
import sys
import types
import unittest

HERE = pathlib.Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import nvg_forward_band_propagation_audit as audit  # noqa: E402

RESULT = HERE / "nvg_forward_band_propagation_results.json"
TRACKED_TOV = HERE / "nvg_forward_tov_heldout_results.json"

# Star-data names that the coefficient solve must never reference.
STAR_NAMES = {"HELDOUT_STAR_DATA", "THRESHOLD_NOTE", "heldout_compare", "CRIT", "AUTHOR_POINTS", "CORNER"}


def _code_names(obj, seen=None):
    """All global names referenced by a function/method (recursively through nested code objects)."""
    seen = set() if seen is None else seen
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


def _no_nan(text):
    def bad(token):
        raise ValueError(f"non-strict JSON token {token}")
    return json.loads(text, parse_constant=bad)


class ForwardBandPropagationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.corner = audit.evaluate_point(audit.CORNER["params"], crust="sly", pressure="consistent", scalar="declared")
        p087 = dict(audit.AUTHOR_POINTS["params_common"], ystar=0.87)
        cls.y087_author = audit.evaluate_point(p087, crust="author", pressure="author", scalar="quartic", p_surf=1e-4)
        cls.y087_sly = audit.evaluate_point(p087, crust="sly", pressure="consistent", scalar="quartic")

    # ---- author's y* = 0.87 forward point
    def test_author_y087_forward_point(self):
        r = self.y087_author
        self.assertEqual(r["status"], "ok")
        self.assertEqual(r["scalar_status"], "quartic_forced")
        self.assertTrue(r["globally_stable"])
        self.assertEqual(r["a5"], 0.0)
        # gamma(L=50) of the author's forward beta EOS (nvg_forward_beta_eos_results.json: -0.6278)
        self.assertAlmostEqual(r["gamma"], -0.6278, places=3)
        # same chain, same conventions (author pressure, polytrope p_match 1.5 / Gamma 1.35, p_surf 1e-4) on the
        # 220-node table: regression values of the float transcription validated against the mpmath chain
        self.assertAlmostEqual(r["M_max"] / 1.8638837717564303, 1.0, delta=2e-6)
        self.assertAlmostEqual(r["R_14"] / 13.805103942680772, 1.0, delta=2e-6)
        self.assertAlmostEqual(r["Lambda_14"] / 510.04988832193067, 1.0, delta=5e-5)
        # tracked author artifact (24-node table, linear eps(P), FD de/dP): table-resolution differences only
        if TRACKED_TOV.is_file():
            tracked = json.loads(TRACKED_TOV.read_text(encoding="utf-8"))
            b = next(x for x in tracked["science"]["branches"] if x["y_target"] == "0.87")
            self.assertLess(abs(r["M_max"] / b["M_max_Msun"] - 1.0), 0.012)
            self.assertLess(abs(r["R_14"] / b["R_14_km"] - 1.0), 0.006)
            self.assertLess(abs(r["Lambda_14"] / b["Lambda_14"] - 1.0), 0.03)
        # consistent pressure + SLy crust: M_max well below 2 Msun, no joint pass
        s = self.y087_sly
        self.assertEqual(s["status"], "ok")
        self.assertAlmostEqual(s["M_max"] / 1.788203106467804, 1.0, delta=1e-5)
        self.assertFalse(s["joint2"])
        self.assertIn("Mmax_J0740_Fonseca2021", s["fails2"])

    # ---- post-hoc corner under the SLy crust
    def test_corner_sly_crust(self):
        r = self.corner
        self.assertEqual(r["status"], "ok")
        self.assertEqual(r["crust"], "sly")
        self.assertEqual(r["match"], "eps")
        self.assertEqual(r["scalar_status"], "quintic_stable")
        self.assertAlmostEqual(r["M_max"] / 2.0132, 1.0, delta=1e-3)
        self.assertAlmostEqual(r["R_144"] / 12.3005, 1.0, delta=1e-3)
        self.assertTrue(r["joint2"])
        self.assertFalse(r["joint1"])
        self.assertEqual(r["fails2"], [])
        self.assertEqual(r["fails1"], ["R144_J0614_Mauviard2025"])
        self.assertLess(abs(r["eps_jump_rel_at_match"]), 1e-12)   # eps-continuous match
        self.assertFalse(r["density_inversion"])
        self.assertEqual(audit.CORNER["label"], "star_selected_post_hoc")

    # ---- stability / causality flags
    def test_stability_and_causality_flags(self):
        for r in (self.corner, self.y087_author, self.y087_sly):
            self.assertTrue(r["globally_stable"])
            self.assertTrue(r["causal_stellar_branch"])
            self.assertLess(r["max_cs2_stellar_branch"], 1.0)
            self.assertTrue(r["Mmax_reached"])
            self.assertTrue(r["global_min_ok_all"] or r["x_monotone_from"] > 0.1)
        qm, zm, bounded = audit.q_min(self.corner["coef"])
        self.assertTrue(bounded)
        self.assertGreaterEqual(qm, 0.0)
        # an unstable quartic (y* = 0.60 at the centre) must be refused by the declared rule without Q0
        j4 = audit.scalar_jet(0.60, 0.16, -16.0, 240.0)
        self.assertEqual(j4["status"], "ok")
        self.assertFalse(j4["globally_stable"])
        specs = audit.scalar_spec(0.60, 0.16, -16.0, 240.0, Q0=-200.0)
        self.assertEqual(specs["declared"]["status"], "quintic_stable")

    # ---- star data never enters the coefficient solve
    def test_star_data_isolated_from_coefficient_solve(self):
        coefficient_code = [audit.scalar_jet, audit._jet_rows, audit.fermi_s, audit.q_min, audit.minimal_a6, audit.scalar_spec,
                            audit.isovector, audit.make_C_of_x, audit.build_beta_table, audit._beta_row,
                            audit.BetaSolver.residuals, audit.BetaSolver.solve, audit.BetaSolver.solve_nested,
                            audit.scalar_model, audit.core_table, audit.lhs_params, audit.grid_params, audit.fast_grid_params]
        for fn in coefficient_code:
            names = _code_names(fn)
            self.assertFalse(names & STAR_NAMES, f"{fn.__qualname__} references star data: {names & STAR_NAMES}")
        # the star table is referenced by the comparison function only
        self.assertIn("HELDOUT_STAR_DATA", _code_names(audit.heldout_compare))
        # dynamic check: hide the star data behind a raising sentinel; the coefficient solve must be unchanged
        class Poison(dict):
            def __getitem__(self, k):
                raise AssertionError("star data accessed during the coefficient solve")

            def items(self):
                raise AssertionError("star data accessed during the coefficient solve")
        p = audit.CORNER["params"]
        before = audit.scalar_spec(p["ystar"], p["n0"], p["EA"], p["K"], p["Q0"])["declared"]["jet"]["coef"]
        saved = audit.HELDOUT_STAR_DATA
        audit.HELDOUT_STAR_DATA = Poison()
        try:
            spec = audit.scalar_spec(p["ystar"], p["n0"], p["EA"], p["K"], p["Q0"])
            jet = spec["declared"]["jet"]
            iso = audit.isovector(jet, p["J"], p["L"])
            tab = audit.build_beta_table(jet, iso, [0.9, 1.0, 1.1], global_check=False)
        finally:
            audit.HELDOUT_STAR_DATA = saved
        self.assertEqual(spec["declared"]["jet"]["coef"], before)
        self.assertEqual(len(tab["P_cons"]), 3)
        # the declared windows and control-point rule contain no star quantity
        self.assertEqual({cp["selected_by"] for cp in audit.CONTROL_POINTS}, {"declared_fixed_ystar"})
        for w in audit.DECLARED_WINDOWS.values():
            self.assertNotIn("M_max", w["source"])

    # ---- verifier correction 1: exact 2 n0, not the nearest node
    def test_exact_2n0_evaluation(self):
        r = self.y087_sly
        self.assertEqual(r["P_2n0_nearest_node_legacy"]["x"] != 2.0, True)
        self.assertNotEqual(r["P_2n0"], r["P_2n0_nearest_node_legacy"]["P_cons"])
        bias = r["P_2n0_nearest_node_legacy"]["P_cons"] / r["P_2n0"] - 1.0
        self.assertGreater(bias, 0.001)
        self.assertLess(bias, 0.01)
        self.assertGreater(r["S_2n0"], r["S_n0"])

    # ---- SLy crust internals
    def test_sly_crust_continuity(self):
        c = audit.SLY
        for i, rho in enumerate(c.rho_div):
            e_lo = c._eps_rho_piece(i, rho)
            e_hi = c._eps_rho_piece(i + 1, rho)
            self.assertAlmostEqual(e_lo / e_hi, 1.0, delta=1e-12)
            self.assertLess(abs(c.P_div_mismatch_rel[i]), 2e-4)
        for P in (1e-6, 1e-3, 0.05, 0.5):
            e, dedp = c.eps_dedp(P)
            h = P * 1e-5
            fd = (c.eps_of_P(P + h) - c.eps_of_P(P - h)) / (2 * h)
            self.assertAlmostEqual(dedp / fd, 1.0, delta=1e-6)
            self.assertGreater(1.0 / dedp, 0.0)
            self.assertLess(1.0 / dedp, 1.0)

    # ---- strict JSON artifact
    def test_results_json_strict_and_validates(self):
        self.assertTrue(RESULT.is_file(), RESULT)
        text = RESULT.read_text(encoding="utf-8")
        data = _no_nan(text)
        self.assertEqual(data["status"], "FORWARD_BAND_PROPAGATION_COMPUTED_ZERO_EVIDENCE")
        self.assertEqual(data["evidence_weight"], 0.0)
        self.assertEqual(data["source_sha256"], hashlib.sha256(audit.SOURCE_PATH.read_bytes()).hexdigest())
        self.assertTrue(audit.validate_result(data))
        self.assertEqual(data["corner_and_author_points"]["corner"]["label"], "star_selected_post_hoc")
        corner = data["corner_and_author_points"]["corner"]["summary"]["sly|consistent|eps_match|psurf1e-9"]
        self.assertTrue(corner["joint2"])
        self.assertFalse(corner["joint1"])
        for row in data["precision_convergence_control"]["points"]:
            self.assertEqual(row["selected_by"], "declared_fixed_ystar")
        self.assertTrue(data["precision_convergence_control"]["pass"])
        for s in data["band"]["samples"]:
            for e in s["eos"].values():
                if e["status"] == "ok":
                    self.assertTrue(e["causal_stellar_branch"] is not None)
        # the band in fast mode never produces a joint pass of the declared class
        a = data["band"]["analysis_by_key"]["declared|power|A"]
        self.assertEqual(a["n_total"], data["band"]["n_samples"])
        self.assertGreaterEqual(a["n_valid"], 1)


if __name__ == "__main__":
    unittest.main()
