"""Independent checks of sufficient bounds; failed thresholds are not no-goes."""
import json
import contextlib
import io
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

import mpmath as mp

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0,str(HERE))
import nvg_matter_branch_certificate as certificate


class MatterBranchCertificateTests(unittest.TestCase):
    def test_exact_symbolic_certificates(self):
        checks=certificate.symbolic_certificates()
        self.assertEqual(len(checks),7)
        self.assertEqual(set(checks),certificate.REQUIRED_SYMBOLIC_CHECKS)
        self.assertTrue(all(value is True for value in checks.values()))

    def test_symbolic_negative_controls_fail_closed(self):
        valid={name:True for name in certificate.REQUIRED_SYMBOLIC_CHECKS}
        key=next(iter(valid))
        missing=dict(valid); del missing[key]
        extras=dict(valid,unexpected=True)
        controls=[{},None,[],missing,extras]
        for wrong in (False,1,"True",None):
            bad=dict(valid); bad[key]=wrong; controls.append(bad)
        for bad in controls:
            with self.subTest(bad=bad):
                with self.assertRaises(ArithmeticError):
                    certificate.require_symbolic_certificates(bad)
                with patch.object(certificate,"symbolic_certificates",return_value=bad):
                    with self.assertRaises(ArithmeticError): certificate.calculate(50,[1])

    def test_thresholds_are_sufficient_not_necessary(self):
        for r,unique,gap,sound in ((".1",False,False,False),(".25",True,False,False),
                                  ("1",True,True,False),("1.25",True,True,False),
                                  ("1.250001",True,True,True)):
            b=certificate.sufficient_bounds(r)
            self.assertEqual(b["unique_global_branch_certified"],unique)
            self.assertEqual(b["positive_uniform_gap_certified"],gap)
            self.assertEqual(b["zero_to_one_cs2_certified"],sound)
            self.assertIn("inconclusive",b["threshold_failure_meaning"])

    def test_finite_positive_guards(self):
        for bad in (True,False,0,-1,"nan","inf","-inf",None):
            for key in ("R","W0","lam"):
                args={"R":2,"W0":859,"lam":1.05}; args[key]=bad
                with self.subTest(key=key,bad=bad), self.assertRaises(ValueError):
                    certificate.sufficient_bounds(**args)
        for bad in (True,49,301,60.0):
            with self.assertRaises(ValueError): certificate.calculate(bad)
        for values in ([],[False],["nan"],[0],[-1]):
            with self.assertRaises(ValueError): certificate.calculate(50,values)

    def test_original_parameters_and_independent_quadratures(self):
        s=certificate.calculate(60)
        self.assertIs(s["mathematical_checks_passed"],True)
        self.assertEqual(s["evidence_weight"],0)
        self.assertEqual(s["status"],"conditional_mathematical_certificate")
        b=s["bounds"]
        self.assertTrue(b["zero_to_one_cs2_certified"])
        self.assertAlmostEqual(float(b["R"]),216.84135048887546,places=11)
        self.assertAlmostEqual(float(b["y_lower"]),.9988490733672925,places=14)
        self.assertAlmostEqual(float(b["gap_lower_MeV"]),1242.659616886025,places=9)
        for point in s["sanity_points_not_global_proof"]:
            self.assertGreaterEqual(point["y"],b["y_lower"])
            self.assertGreaterEqual(point["C_W_MeV2"],b["C_W_lower_MeV2"])
            self.assertLess(point["max_normalized_residual"],mp.mpf("1e-45"))
        self.assertAlmostEqual(float(s["unique_field_minimum"]["n_over_n0"]),2.1410263101189605,places=12)
        self.assertAlmostEqual(float(s["unique_field_minimum"]["cs2"]),1/3,places=15)
        self.assertFalse(s["formal_high_density"]["physical_validity_asserted"])

    def test_precision_agreement_and_formal_label(self):
        lo=certificate.calculate(50,[".03","3","100"])
        hi=certificate.calculate(80,[".03","3","100"])
        with mp.workdps(85):
            for a,b in zip(lo["sanity_points_not_global_proof"],hi["sanity_points_not_global_proof"]):
                for key in ("y","cs2","C_W_MeV2"):
                    self.assertLess(abs(a[key]-b[key])/abs(b[key]),mp.mpf("1e-40"))
        self.assertEqual(hi["sanity_points_not_global_proof"][-1]["scope"],"formal extrapolation")

    def test_direct_implicit_derivative_and_crossing_direction(self):
        with mp.workdps(60):
            model=certificate.BulkModel()
            for x in map(mp.mpf,(".2","1","3","10")):
                n=x*model.n0; s=model.equilibrium(n)
                derivative=mp.diff(lambda nn:model.equilibrium(nn)["mu"],n)
                self.assertLess(abs(n*derivative/s["mu"]-s["cs2"]),mp.mpf("1e-50"))
            root=certificate.calculate(50,[1])["unique_field_minimum"]
            n,y=root["n_over_n0"]*model.n0,root["y"]
            s=model.state(n,y)
            direct=mp.diff(lambda nn:model.state(nn,y)["B_y"],n)
            exact=-model.MN**2*y*s["k"]**2/(3*n*s["ef"]**3)-2*model.Cv/y**3
            self.assertLess(abs(direct-exact),mp.mpf("1e-50"))
            self.assertLess(exact,0)

    def test_cli_json_only(self):
        run=subprocess.run([sys.executable,str(HERE/"nvg_matter_branch_certificate.py"),"--dps","50"],
                           capture_output=True,text=True,check=True,timeout=60)
        self.assertEqual(run.stderr,"")
        self.assertTrue(json.loads(run.stdout)["bounds"]["zero_to_one_cs2_certified"])

    def test_cli_invalid_inputs_are_json_exit_two(self):
        for args in (["--dps","49"],["--dps","301"],["--dps","nan"],["--unknown"]):
            run=subprocess.run([sys.executable,str(HERE/"nvg_matter_branch_certificate.py"),*args],
                               capture_output=True,text=True,timeout=60)
            self.assertEqual(run.returncode,2)
            self.assertEqual(run.stderr,"")
            result=json.loads(run.stdout)
            self.assertEqual(result["status"],"invalid_or_failed")
            self.assertIs(result["mathematical_checks_passed"],False)
            self.assertEqual(result["evidence_weight"],0)

    def test_main_rejects_failed_or_incomplete_result(self):
        for result in ({"mathematical_checks_passed":True,"symbolic":{}},
                       {"mathematical_checks_passed":False,
                        "symbolic":{name:True for name in certificate.REQUIRED_SYMBOLIC_CHECKS}}):
            output=io.StringIO()
            with patch.object(certificate,"calculate",return_value=result),contextlib.redirect_stdout(output):
                self.assertEqual(certificate.main([]),2)
            self.assertEqual(json.loads(output.getvalue())["status"],"invalid_or_failed")


if __name__ == "__main__":
    unittest.main()
