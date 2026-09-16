"""Independent identities, calibration boundaries and negative controls."""

import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import mpmath as mp

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import source_complete_scaling_saturation_audit as audit


class ScalingSaturationTests(unittest.TestCase):
    def setUp(self):
        self.context = mp.workdps(80)
        self.context.__enter__()
        self.model = audit.BulkModel()

    def tearDown(self):
        self.context.__exit__(None, None, None)

    def close(self, a, b, tolerance="1e-55"):
        self.assertLess(abs(a-b)/max(abs(a), abs(b), mp.mpf(1)), mp.mpf(tolerance))

    def test_declared_inputs_match_accepted_baseline_without_using_its_solver(self):
        from source_complete_solution_audit import Params
        baseline=Params()
        names={"MN":"M_N","hbarc":"hbar_c","gomega":"g_omega", "momega":"m_omega","d":"degeneracy"}
        for key,value in audit.INPUTS.items():
            self.assertEqual(float(value),getattr(baseline,names.get(key,key)))

    def test_fermi_expressions_against_independent_quadrature(self):
        m = self.model
        for x, y in (("0.01","0.6"),("1","0.9"),("10","1.1")):
            n, y = mp.mpf(x)*m.n0, mp.mpf(y)
            f = m.fermi(n,y)
            factor = m.d/(2*mp.pi**2)
            energy = factor*mp.quad(lambda k:k*k*mp.sqrt(k*k+f["m"]**2),[0,f["k"]])
            ns = factor*mp.quad(lambda k:k*k*f["m"]/mp.sqrt(k*k+f["m"]**2),[0,f["k"]])
            pressure = factor/3*mp.quad(lambda k:k**4/mp.sqrt(k*k+f["m"]**2),[0,f["k"]])
            self.close(f["energy"],energy)
            self.close(f["pressure"],pressure)
            self.close(f["ns"],ns)
            self.close(f["ns_m"],factor*mp.quad(lambda k:k**4/(k*k+f["m"]**2)**mp.mpf("1.5"),[0,f["k"]]))

    def test_off_shell_trace_has_stationarity_remainder(self):
        s = self.model.state(self.model.n0,mp.mpf("0.8"))
        self.close(s["energy_total"]-3*s["pressure_total"],
                   4*s["U"]-s["y"]*s["Uy"]+s["y"]*s["residual"])
        self.assertGreater(abs(s["residual"]),1)
        with self.assertRaises(ArithmeticError):
            audit._check_state(s)

    def test_scale_and_legendre_relations_on_equilibrium_branch(self):
        m = self.model
        for x in ("0.001","0.1","1","2.5","5","10"):
            s = m.equilibrium(mp.mpf(x)*m.n0)
            audit._check_state(s)
            self.close(s["energy_total"]-3*s["pressure_total"],m.A*(1-s["y"]**2))
            self.close(3*s["n"]*s["D"]+s["y"]*s["B_y"],s["mu"])
            scale_curvature = s["Uyy"]-3*s["Uy"]/s["y"]
            self.close(s["C_y"],scale_curvature-3*s["n"]*s["B_y"]/s["y"])
            self.close(s["cs2"],(1-s["y"]*scale_curvature*s["B_y"]/(s["mu"]*s["C_y"]))/3)

    def test_generic_potential_scale_identity_not_only_quartic(self):
        m,s,_ = audit.inverse_potential_jet("0.90")
        audit._check_state(s)
        self.close(s["C_y"],s["Uyy"]-3*s["Uy"]/s["y"]-3*s["n"]*s["B_y"]/s["y"])
        # The quartic-specific trace must NOT be recycled for a new potential.
        self.assertGreater(abs((s["energy_total"]-3*s["pressure_total"])-m.A*(1-s["y"]**2)),1)

    def test_global_energy_bound_positive_control_includes_nonstationary_states(self):
        base = self.model
        # Test the theorem even at its equality boundary Cv=Cs.
        equal = audit.BulkModel(gomega=base.momega*mp.sqrt(base.Cs))
        for m in (base,equal):
            for x in ("0.001","1","100"):
                n = mp.mpf(x)*m.n0
                for ytext in ("0.05","0.5","0.9","1","2"):
                    y = mp.mpf(ytext)
                    s = m.state(n,y)
                    self.assertGreater(s["energy_total"],m.MN*n)
                    if y < 1:
                        self.assertGreaterEqual(s["U"]+s["vector"],m.MN*n*(1-y*y)/(2*y))

    def test_no_binding_condition_is_sufficient_not_universal_assertion(self):
        m,s = audit.quartic_calibration()
        self.assertLess(m.Cv,m.Cs)
        self.assertLess(s["energy_total"],m.MN*s["n"])

    def test_original_stationary_pressure_positive(self):
        m = self.model
        for x in ("0.001","0.1","1","10","100"):
            n = mp.mpf(x)*m.n0
            s = m.equilibrium(n,guess=("0.8","1.7"))
            self.assertGreater(s["pressure_total"],0)
            if s["y"] < 1:
                lower = s["pressure"]+n*n*(m.Cv-m.Cs)/(2*s["y"]**2)
                self.assertGreaterEqual(s["pressure_total"],lower)

    def test_quartic_inverse_targets_and_independent_K(self):
        m,s = audit.quartic_calibration()
        n = s["n"]
        self.close(s["pressure_total"],mp.mpf(0))
        self.close(s["energy_total"]/n-m.MN,mp.mpf(-16))
        def binding_at(nn):
            q=m.equilibrium(nn,guess=("0.76","0.80"))
            return q["energy_total"]/nn-m.MN
        self.close(s["K"],9*n*n*mp.diff(binding_at,n,2))
        self.assertGreater(s["K"],350)
        self.assertGreater(s["C_y"],0)
        self.assertTrue(0<s["cs2"]<1)

    def test_calibration_changes_when_declared_targets_change(self):
        m0,s0 = audit.quartic_calibration()
        m1,s1 = audit.quartic_calibration(n_fm3="0.17",binding="-15")
        self.close(s1["pressure_total"],mp.mpf(0))
        self.close(s1["energy_total"]/s1["n"]-m1.MN,mp.mpf(-15))
        self.assertNotEqual(m0.lam,m1.lam)
        self.assertNotEqual(m0.gomega,m1.gomega)
        self.assertNotEqual(s0["K"],s1["K"])

    def test_inverse_root_is_monotone_in_physical_domain(self):
        m=self.model
        n=m.n0
        k=m.fermi(n,mp.mpf(1))["k"]
        mu=mp.mpf(923)
        ymax=mp.sqrt(mu*mu-k*k)/m.MN
        def h(y):
            f=m.fermi(n,y)
            return mu*(1+y*y)-2*f["ef"]+4*f["pressure"]/n
        self.assertGreater(h(ymax/100),0)
        self.assertLess(h(ymax),0)
        for i in range(1,50):
            self.assertLess(mp.diff(h,ymax*i/50),0)

    def test_vector_bound_is_extremum_not_chosen_threshold(self):
        m=self.model
        b=audit.vector_ceiling()
        def allowed_cv(y):
            return y*y*(b["mu_target_MeV"]-m.fermi(m.n0,y)["ef"])/m.n0
        y=b["y_at_ceiling"]
        self.close(mp.diff(allowed_cv,y),mp.mpf(0))
        self.assertLess(mp.diff(allowed_cv,y,2),0)
        self.close(allowed_cv(y),b["Cv_max"])
        self.assertGreater(m.gomega,b["gomega_max"])
        self.assertGreater(b["original_mu_min_MeV"],b["mu_target_MeV"])
        mcap=audit.BulkModel(gomega=b["gomega_max"])
        s=mcap.state(mcap.n0,y)
        self.close(s["B_y"],mp.mpf(0))
        self.close(9*mcap.n0*s["D"],3*s["mu"])

    def test_scalar_potential_cannot_change_chemical_potential_at_fixed_n_y(self):
        m=self.model
        alternative=audit.BulkModel(polynomial=("1e9","-1e9","2e9"))
        for y in (mp.mpf("0.6"),mp.mpf("0.9"),mp.mpf("1.1")):
            self.close(m.state(m.n0,y)["mu"],alternative.state(m.n0,y)["mu"])

    def test_inverse_jets_match_direct_potential_derivatives_and_targets(self):
        for y in ("0.75","0.90"):
            m,s,jet=audit.inverse_potential_jet(y)
            for order,key in ((0,"U"),(1,"Uy"),(2,"Uyy")):
                self.close(mp.diff(lambda yy:m.potential(yy)[0],s["y"],order),jet[key])
            self.close(s["pressure_total"],mp.mpf(0))
            self.close(s["energy_total"]/s["n"]-m.MN,mp.mpf(-16))
            self.close(s["K"],mp.mpf(240))
            self.close(m.potential(mp.mpf(1))[0],mp.mpf(0))
            self.close(m.potential(mp.mpf(1))[1],mp.mpf(0))

    def test_local_fit_does_not_imply_bounded_vacuum(self):
        m,s,jet=audit.inverse_potential_jet("0.75")
        self.assertGreater(s["C_y"],0)
        self.assertFalse(jet["globally_nonnegative_potential"])
        self.assertLess(jet["coefficients"][-1],0)
        self.assertLess(m.potential(mp.mpf(100))[0],0)

    def test_positive_polynomial_candidate_has_analytic_vacuum_bound(self):
        m,_,jet=audit.inverse_potential_jet("0.90")
        a2,a3,a4=jet["coefficients"]
        self.assertGreater(a4,0)
        self.assertGreater(jet["q_min"],0)
        self.assertTrue(jet["globally_nonnegative_potential"])
        z=jet["z_at_q_min"]
        self.close(a2+a3*z+a4*z*z,jet["q_min"])

    def test_vacuum_bound_includes_lower_degree_and_boundary_cases(self):
        for coefficients in ((1,0,0),(2,1,0),(1,2,1),(0,0,0)):
            self.assertTrue(audit.polynomial_vacuum_bound(coefficients)["globally_nonnegative_potential"])
        for coefficients in ((1,-1,0),(1,2,0),(1,0,-1),(-1,0,1)):
            self.assertFalse(audit.polynomial_vacuum_bound(coefficients)["globally_nonnegative_potential"])

    def test_degenerate_inverse_jet_reports_nonuniqueness_not_physical_no_go(self):
        b=audit.vector_ceiling()
        with self.assertRaisesRegex(ValueError,"non-unique"):
            audit.inverse_potential_jet(b["y_at_ceiling"],b["finite_curvature_K_at_ceiling_MeV"])

    def test_symmetry_energy_by_two_species_quadrature(self):
        m,s,_=audit.inverse_potential_jet("0.90")
        mass,n=s["m"],s["n"]
        def species_energy(density):
            k=(3*mp.pi**2*density)**(mp.mpf(1)/3)
            return mp.quad(lambda p:p*p*mp.sqrt(p*p+mass*mass),[0,k])/mp.pi**2
        def ebar(delta):
            return (species_energy(n*(1+delta)/2)+species_energy(n*(1-delta)/2))/n
        kinetic_s=mp.diff(ebar,mp.mpf(0),2)/2
        self.close(kinetic_s,s["k"]**2/(6*s["ef"]))

    def test_precision_context_restored_and_minimum_not_zero_trace(self):
        before=mp.mp.dps
        result=audit.build_result()
        self.assertEqual(mp.mp.dps,before)
        landmarks=result["science"]["landmarks"]
        self.assertLess(mp.mpf(landmarks["field_minimum"]["x"]),mp.mpf(landmarks["zero_trace_nonvacuum"]["x"]))
        self.assertGreater(mp.mpf(landmarks["zero_trace_nonvacuum"]["cs2"]),mp.mpf(1)/3)

    def test_precision_restored_on_invalid_request(self):
        before=mp.mp.dps
        with self.assertRaises(ValueError):
            audit.calculate(30)
        self.assertEqual(mp.mp.dps,before)

    def test_fresh_result_validation_rejects_scientific_and_label_mutations(self):
        baseline=audit.build_result()
        self.assertTrue(audit.validate_result(baseline))
        for mutate in (
            lambda x:x["science"]["original"].update(Cv_over_Cs="1"),
            lambda x:x["science"]["two_target_quartic_calibration"].update(K_was_fitted=True),
            lambda x:x["science"]["inverse_potential_examples"][0].update(globally_nonnegative_potential=True),
            lambda x:x.update(evidence_weight=1),
            lambda x:x.update(source_sha256="0"*64),
        ):
            candidate=copy.deepcopy(baseline)
            mutate(candidate)
            self.assertFalse(audit.validate_result(candidate))

    def test_bad_result_does_not_overwrite_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            target=Path(temporary)/"result.json"
            target.write_text("unchanged",encoding="utf-8")
            with self.assertRaises(ValueError):
                audit.write_result({"science":float("nan")},target)
            self.assertEqual(target.read_text(),"unchanged")

    def test_default_cli_does_not_write_canonical_artifacts(self):
        paths=[audit.RESULT_PATH,HERE/"source_complete_stellar_prediction_results.json",audit.BASELINE_PATH]
        before={p:p.read_bytes() if p.exists() else None for p in paths}
        with tempfile.TemporaryDirectory() as temporary:
            run=subprocess.run([sys.executable,str(audit.SOURCE_PATH)],cwd=temporary,
                               capture_output=True,text=True,check=True)
        result=json.loads(run.stdout)
        self.assertEqual(result["evidence_weight"],0)
        for p,old in before.items():
            self.assertEqual(p.read_bytes() if p.exists() else None,old)

    def test_saved_result_reproduces_from_live_source(self):
        if not audit.RESULT_PATH.exists():
            self.fail("Generate the dedicated result with --write before integrated acceptance")
        self.assertTrue(audit.validate_result(json.loads(audit.RESULT_PATH.read_text())))


if __name__ == "__main__":
    unittest.main()
