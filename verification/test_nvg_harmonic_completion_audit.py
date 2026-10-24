"""Light tests for the harmonic completion audit (no heavy recompute)."""

import json
import pathlib
import unittest

HERE = pathlib.Path(__file__).resolve().parent


class HarmonicTests(unittest.TestCase):
    def _load(self):
        return json.loads((HERE / "nvg_harmonic_completion_results.json").read_text())

    def test_file_exists_and_zero_weight(self):
        d = self._load()
        self.assertEqual(d["evidence_weight"], 0.0)
        self.assertEqual(d["precision_check"]["pass"], True)
        self.assertIn("HARMONIC_COMPLETION", d["status"])

    def test_scalar_windows(self):
        d = self._load()
        rows = {r["y_target"]: r for r in d["science"]["scalar_global_stability"]}
        # y=0.65,0.70,0.87,0.90 should be open
        for y in ("0.65", "0.70", "0.87", "0.90"):
            self.assertTrue(rows[y]["globally_nonnegative_a5"], y)
        # y=0.75,0.80 should be closed at a5=5e8
        self.assertFalse(rows["0.75"]["globally_nonnegative_a5"])
        self.assertFalse(rows["0.80"]["globally_nonnegative_a5"])
        # quartic only y>=0.87
        self.assertTrue(rows["0.87"]["quartic_globally_nonnegative"])
        self.assertFalse(rows["0.70"]["quartic_globally_nonnegative"])

    def test_qmin_positive_when_open(self):
        d = self._load()
        for r in d["science"]["scalar_global_stability"]:
            q = float(r["qmin_a5_MeV4"])
            if r["globally_nonnegative_a5"]:
                self.assertGreater(q, 0, r["y_target"])
            else:
                self.assertLess(q, 0, r["y_target"])

    def test_tov_ordering_with_A(self):
        d = self._load()
        for branch in d["science"]["harmonic_beta_tov"]:
            ms = [c["Mmax_Msun"] for c in branch["Mmax_curve"] if "Mmax_Msun" in c]
            self.assertEqual(ms, sorted(ms))
            self.assertGreater(ms[-1], ms[0])

    def test_y070_heavy_but_large_radius(self):
        d = self._load()
        b07 = next(b for b in d["science"]["harmonic_beta_tov"] if b["y_target"] == "0.70")
        c0 = next(c for c in b07["Mmax_curve"] if c["A_MeV"] == "0.0")
        self.assertGreater(c0["Mmax_Msun"], 2.4)
        self.assertGreater(c0["R14_km"], 14.5)  # too large for NICER 11.2-13.2

    def test_y087_needs_stiffening(self):
        d = self._load()
        b87 = next(b for b in d["science"]["harmonic_beta_tov"] if b["y_target"] == "0.87")
        c0 = next(c for c in b87["Mmax_curve"] if c["A_MeV"] == "0.0")
        self.assertLess(c0["Mmax_Msun"], 2.01)
        c05 = next(c for c in b87["Mmax_curve"] if c["A_MeV"] == "0.5")
        self.assertGreater(c05["Mmax_Msun"], 2.01)


if __name__ == "__main__":
    unittest.main()
