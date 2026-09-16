"""Integration, provenance, explicit assumptions, and fail-closed output checks."""
import contextlib
import copy
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

HERE=Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0,str(HERE))
import bh_cycle_consistency_audit as audit
import nvg_bh_regular_core as regular


class BHIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result=audit.build_result()

    def test_saved_result_matches_live_recalculation(self):
        saved=json.loads(audit.RESULT_PATH.read_text())
        self.assertTrue(audit.validate_result(saved))

    def test_changed_numbers_and_boolean_coercion_rejected(self):
        for path,value in ((("schema_version",),True),
                           (("conditional_cycle_assignment","cycle_index"),78),
                           (("source_complete_GR_obstruction","Cv_over_Cs"),"1")):
            changed=copy.deepcopy(self.result)
            node=changed
            for key in path[:-1]: node=node[key]
            node[path[-1]]=value
            self.assertFalse(audit.validate_result(changed))
        self.assertFalse(audit.validate_result({"invalid":float("nan")}))

    def test_physical_negative_results_are_preserved(self):
        result=self.result
        self.assertEqual(result["evidence_weight"],0)
        self.assertIsNone(result["observed_likelihood"])
        self.assertTrue(result["physical_completion_missing"])
        old=result["repaired_legacy_interior"]
        self.assertIn(1,old["sampled_checks"]["mechanically_unstable_density_ratios"])
        self.assertFalse(old["sampled_checks"]["differentiable_at_all_samples"])
        self.assertTrue(old["sampled_checks"]["energy_exceeds_density_scale_at_ratios"])
        baseline=result["source_complete_GR_obstruction"]
        self.assertTrue(baseline["analytic_no_binding_condition"])
        for row in baseline["samples"]:
            self.assertGreater(float(row["epsilon_plus_3P_MeV_fm3"]),0)

    def test_failed_geometry_cannot_write(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"result.json"
            path.write_text("sentinel")
            with patch.object(audit,"RESULT_PATH",path), patch.object(audit.geometry,"build_result",return_value={"status":"FAIL"}):
                with self.assertRaises(ArithmeticError): audit.main(["--write"])
            self.assertEqual(path.read_text(),"sentinel")

    def test_failed_thermodynamics_cannot_write(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"result.json"
            with patch.object(audit,"RESULT_PATH",path), patch.object(audit.interior,"compute_state",return_value={"sampled_checks":{"closure_within_1e_minus_11":False}}):
                with self.assertRaises(ArithmeticError): audit.main(["--write"])
            self.assertFalse(path.exists())

    def test_nonfinite_source_witness_and_missing_theorem_premise_fail(self):
        model=audit.BulkModel()
        original=model.equilibrium
        def poisoned(*args,**kwargs):
            state=original(*args,**kwargs)
            state["energy_total"]=audit.mp.nan
            return state
        with patch.object(model,"equilibrium",side_effect=poisoned), patch.object(audit,"BulkModel",return_value=model):
            with self.assertRaises(ArithmeticError): audit.source_branch_witness()
        altered=audit.BulkModel(gomega=0)
        with patch.object(audit,"BulkModel",return_value=altered):
            with self.assertRaises(ArithmeticError): audit.source_branch_witness()

    def test_default_cli_reproduces_without_writing(self):
        before=audit.RESULT_PATH.read_bytes()
        result=subprocess.run([sys.executable,str(HERE/"bh_cycle_consistency_audit.py")],capture_output=True,text=True,check=True)
        self.assertEqual(json.loads(result.stdout),self.result)
        self.assertEqual(before,audit.RESULT_PATH.read_bytes())

    def test_legacy_regular_core_is_only_an_assumed_geometry(self):
        state=regular.calculate_bh_interior(10,7.1e4)
        self.assertIsNone(state["observed_likelihood"])
        self.assertEqual(state["evidence_status"],audit.geometry.EVIDENCE_STATUS)
        length=state["length_km"]
        self.assertAlmostEqual(state["central_curvature"]["kretschmann"]*length**4,24)
        with self.assertRaises(ValueError): regular.calculate_bh_interior(-1,7e4)
        with self.assertRaises(ValueError): regular.calculate_bh_interior(10,float("nan"))


if __name__=="__main__":
    unittest.main()
