"""Light tests for the Mmax stiffening-budget audit (no heavy recompute)."""

import json
import pathlib
import sys
import unittest

HERE = pathlib.Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))


class StiffeningTests(unittest.TestCase):
    def _load(self):
        return json.loads((HERE / "nvg_mmax_stiffening_results.json").read_text())

    def test_file_exists_and_zero_weight(self):
        data = self._load()
        self.assertEqual(data["evidence_weight"], 0.0)
        self.assertEqual(data["precision_check"]["pass"], True)
        self.assertEqual(data["schema_version"], "nvg_mmax_stiffening.v1")

    def test_Mmax_increases_with_A(self):
        data = self._load()
        for scan in data["science"]["scans"]:
            ms = [c["Mmax_Msun"] for c in scan["Mmax_curve"] if "Mmax_Msun" in c]
            self.assertEqual(ms, sorted(ms))
            self.assertGreater(ms[-1], ms[0])

    def test_A_star_ordering(self):
        data = self._load()
        stars = {}
        for s in data["science"]["scans"]:
            v = s["A_star_for_Mmax_2p01_MeV"]
            if v is not None:
                stars[s["y_target"]] = float(v)
        self.assertLess(stars["0.87"], stars["0.90"])
        self.assertLess(stars["0.90"], stars["0.93"])
        for v in stars.values():
            self.assertGreater(v, 0.1)
            self.assertLess(v, 2.0)

    def test_stiff_branch_exists_at_A0p3(self):
        data = self._load()
        y87 = next(s for s in data["science"]["scans"] if s["y_target"] == "0.87")
        c03 = next(c for c in y87["Mmax_curve"] if c["A_MeV"] == "0.3")
        self.assertGreaterEqual(c03["Mmax_Msun"], 2.01)

    def test_fractional_pressure_small(self):
        data = self._load()
        for scan in data["science"]["scans"]:
            c03 = next((c for c in scan["Mmax_curve"] if c["A_MeV"] == "0.3"), None)
            if c03 and c03["fractional_p_increase_at_3n0"] is not None:
                self.assertLess(c03["fractional_p_increase_at_3n0"], 0.30)

    def test_p_extra_formula(self):
        data = self._load()
        for scan in data["science"]["scans"]:
            for c in scan["Mmax_curve"]:
                A = float(c["A_MeV"])
                expected = 3 * A * 0.16 * 9 * 4
                self.assertAlmostEqual(c["p_extra_at_3n0_MeV_fm3"], expected, places=6)


if __name__ == "__main__":
    unittest.main()
