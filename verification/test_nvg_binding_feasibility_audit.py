"""Independent derivatives, quadratures and failure controls; no output tables."""
import contextlib
import io
import json
import unittest
from unittest import mock

import mpmath as mp

import nvg_binding_feasibility_audit as audit
from source_complete_scaling_saturation_audit import BulkModel, INPUTS


class BindingFeasibilityTests(unittest.TestCase):
    def close(self, a, b, tol="1e-55"):
        self.assertLess(abs(a-b)/max(abs(a), abs(b), 1), mp.mpf(tol))

    def test_complete_exact_symbolic_certificate(self):
        rows = audit.symbolic_checks()
        self.assertEqual(set(rows), audit.REQUIRED_SYMBOLIC)
        self.assertEqual(len(rows), 17)
        for row in rows.values():
            self.assertIs(row["passed"], True)
            self.assertEqual(row["residual"], "0")

    def test_vector_ceiling_independent_maximum_and_original_no_go(self):
        with mp.workdps(80):
            d = audit.BindingDesign()
            cv = lambda E: (d.mu-E)*(E*E-d.k*d.k)/(d.n*d.base.MN**2)
            peak = mp.findroot(lambda E: mp.diff(cv, E), (d.k, d.mu))
            cap = audit.vector_ceiling()
            self.close(d.base.momega*mp.sqrt(cv(peak)), cap["gomega_max"])
            # d2Cv/dE2<0 throughout the possible stationary region E>mu/3;
            # the derivative numerator has just one positive root.
            self.assertGreater(peak, d.mu/3)
            self.assertLess(mp.diff(cv, peak, 2), 0)
            self.close(cv(d.k), 0)
            self.close(cv(d.mu), 0)
            self.assertGreater(d.base.gomega, cap["gomega_max"])
            mu_original = lambda y: mp.sqrt(d.k*d.k+d.base.MN**2*y*y)+d.base.Cv*d.n/(y*y)
            minimum_y = mp.findroot(lambda y: mp.diff(mu_original, y), (mp.mpf(".7"), mp.mpf(".9")))
            self.close(mu_original(minimum_y), cap["original_mu_min_MeV"])
            self.assertGreater(mu_original(minimum_y), d.mu)
            yc = mp.sqrt(peak**2-d.k**2)/d.base.MN
            Bc = d.base.MN**2*yc/peak-2*(d.mu-peak)/yc
            self.close(Bc, 0)
            self.close(9*(d.mu-peak)+3*d.k*d.k/peak, 3*d.mu)

    def test_fixed_potential_target_jet_and_relaxed_incompressibility(self):
        with mp.workdps(80):
            d = audit.BindingDesign()
            e = lambda nn, yy: d.energy(nn, yy)
            Ey = mp.diff(lambda yy: e(d.n, yy), d.y)
            Enn = mp.diff(lambda nn: e(nn, d.y), d.n, 2)
            Eny = mp.diff(lambda yy: mp.diff(lambda nn: e(nn, yy), d.n), d.y)
            Eyy = mp.diff(lambda yy: e(d.n, yy), d.y, 2)
            mu = mp.diff(lambda nn: e(nn, d.y), d.n)
            self.close(Ey, 0)
            self.close(e(d.n, d.y)/d.n, d.mu)
            self.close(mu, d.mu)
            self.close(d.n*mu-e(d.n, d.y), 0)
            self.close(Enn, d.D)
            self.close(Eny, d.By)
            self.close(Eyy, d.Cy)
            self.close(9*d.n*(Enn-Eny**2/Eyy), d.K)
            self.close(mp.diff(d.potential, d.y), d.Uy)
            self.close(mp.diff(d.potential, d.y, 2), d.Uyy)

    def test_legendre_optimizer_against_independent_momentum_quadratures(self):
        with mp.workdps(80):
            d = audit.BindingDesign()
            for text in (".3", ".6", ".75", ".97"):
                y = mp.mpf(text)
                st = d.envelope_state(y)
                m, k, n = d.base.MN*y, st["k"], st["n"]
                ef = lambda p: mp.sqrt(p*p+m*m)
                E = d.base.d/(2*mp.pi**2)*mp.quad(lambda p: p*p*ef(p), [0, k])
                P = d.base.d/(6*mp.pi**2)*mp.quad(lambda p: p**4/ef(p), [0, k])
                self.close(st["value"], d.mu*n-E-d.Cv*n*n/(2*y*y))
                self.close(st["value"], P+d.Cv*n*n/(2*y*y))
                self.close(d.mu-ef(k)-d.Cv*n/(y*y), 0)
                self.assertGreater(k*k/(3*n*ef(k))+d.Cv/y**2, 0)
                # Independent root in density, not the production k bracket.
                derivative = lambda nn: d.mu-mp.sqrt((6*mp.pi**2*nn/d.base.d)**(mp.mpf(2)/3)+m*m)-d.Cv*nn/y**2
                root = mp.findroot(derivative, (n*mp.mpf(".8"), n*mp.mpf("1.2")))
                self.close(root, n)
                self.assertGreater(derivative(n/2), 0)
                self.assertLess(derivative(n*2), 0)

    def test_global_gap_witnesses_and_only_two_equality_states(self):
        # The all-state proof is strict density convexity plus the barrier's
        # exact roots. This grid is an independent numerical regression only.
        with mp.workdps(80):
            d = audit.BindingDesign()
            for yy in (".3", ".6", ".75", ".9", "1", "1.2"):
                y = mp.mpf(yy)
                for nr in ("0", ".1", "1", "2"):
                    n = mp.mpf(nr)*d.n
                    gap = d.grand_gap(n, y)
                    self.assertGreaterEqual(gap-d.barrier(y), -mp.mpf("1e-65"))
                    self.assertGreaterEqual(d.barrier(y), 0)
            self.close(d.grand_gap(0, 1), 0)
            self.close(d.grand_gap(d.n, d.y), 0)
            self.assertGreater(d.grand_gap(d.n*mp.mpf("1.01"), d.y), 0)
            self.assertGreater(d.grand_gap(0, d.y), 0)
            self.assertGreater(d.grand_gap(d.n, 1), 0)
            for y in (mp.mpf(".5"), mp.mpf(".8"), mp.mpf("1.1")):
                self.assertGreater(d.barrier(y), 0)

    def test_vacuum_mass_is_new_unfitted_consequence(self):
        with mp.workdps(80):
            d = audit.BindingDesign()
            self.close(d.potential(1), 0)
            self.close(mp.diff(d.potential, mp.mpf(1)), 0)
            new_mass2 = mp.diff(d.potential, mp.mpf(1), 2)/d.base.W0**2
            self.close(new_mass2, 8*d.beta*d.zs**2/d.base.W0**2)
            old_mass2 = 2*d.base.lam*d.base.W0**2
            self.assertGreater(new_mass2, 0)
            self.assertLess(new_mass2, old_mass2/1000)
            r = audit.audit()
            self.assertIs(r["constructed_potential"]["new_vacuum_mass_is_unfitted_unvalidated_consequence"], True)

    def test_onset_fractional_power_and_vacuum_side_zero(self):
        with mp.workdps(80):
            d = audit.BindingDesign()
            ncoef = d.base.d*(2*d.mu)**mp.mpf("1.5")/(6*mp.pi**2)
            ucoef = 2*ncoef/5
            errors = []
            for power in (8, 12, 16):
                x = mp.mpf(10)**(-power)
                y = (d.mu-x)/d.base.MN
                st = d.envelope_state(y)
                en = abs(st["n"]/(ncoef*x**mp.mpf("1.5"))-1)
                eu = abs(st["value"]/(ucoef*x**mp.mpf("2.5"))-1)
                errors.append(max(en, eu))
            self.assertGreater(errors[0], errors[1])
            self.assertGreater(errors[1], errors[2])
            self.assertLess(errors[-1], mp.mpf("1e-8"))
            for y in (d.mu/d.base.MN, mp.mpf(1), mp.mpf("1.1")):
                self.close(d.floor(y), 0)
            # x_+^(5/2) has continuous derivatives 0,1,2, while its third
            # derivative diverges as x^(-1/2). Nonzero coefficient is essential.
            self.assertGreater(ucoef, 0)

    def test_extreme_onset_pressure_is_positive_and_matches_scaled_quadrature(self):
        # Regression: the subtracted analytic pressure previously became
        # negative at x=1e-48 (80 dps), violating the exact envelope bound.
        for dps, power in ((80, 40), (80, 48), (120, 60)):
            with self.subTest(dps=dps, power=power), mp.workdps(dps):
                d = audit.BindingDesign(dps=dps)
                x = mp.mpf(10)**(-power)
                y = (d.mu-x)/d.base.MN
                st = d.envelope_state(y)
                k, mass = st["k"], d.base.MN*y
                # The implementation uses a binomial series in this regime;
                # this independent positive integral does not subtract terms.
                pref = d.base.d*k**5/(6*mp.pi**2)
                pref *= mp.quad(lambda t: t**4/mp.sqrt(mass*mass+k*k*t*t), [0, 1])
                reference = pref+d.Cv*st["n"]**2/(2*y*y)
                self.assertGreater(st["value"], 0)
                self.assertLess(abs(st["value"]/reference-1), mp.mpf(10)**(-dps+10))

    def test_extreme_dilute_energy_and_grand_gap_remain_positive(self):
        for dps, power in ((80, 150), (120, 190)):
            with self.subTest(dps=dps, power=power), mp.workdps(dps):
                d = audit.BindingDesign(dps=dps)
                n = d.n*mp.mpf(10)**(-power)
                mass = d.base.MN
                k = (6*mp.pi**2*n/d.base.d)**(mp.mpf(1)/3)
                reference = d.base.d*k**3/(2*mp.pi**2)*mp.quad(
                    lambda t: t*t*mp.sqrt(mass*mass+k*k*t*t), [0, 1])
                reference += d.Cv*n*n/2
                energy, gap = d.energy(n, 1), d.grand_gap(n, 1)
                self.assertGreater(energy, 0)
                self.assertGreater(gap, 0)
                self.assertLess(abs(energy/reference-1), mp.mpf(10)**(-dps+10))
                self.assertLess(abs(gap/(reference-d.mu*n)-1), mp.mpf(10)**(-dps+10))

    def test_series_and_scaled_integral_join_match_independent_quadratures(self):
        with mp.workdps(80):
            mass = mp.mpf("700")
            for ratio in (".4999", ".5", ".5001", "2", "100"):
                k = mass*mp.mpf(ratio)
                energy = 4/(2*mp.pi**2)*mp.quad(lambda p: p*p*mp.sqrt(p*p+mass*mass), [0, k])
                pressure = 4/(6*mp.pi**2)*mp.quad(lambda p: p**4/mp.sqrt(p*p+mass*mass), [0, k])
                self.close(audit._stable_fermi_integral(k,mass,4), energy, "1e-70")
                self.close(audit._stable_fermi_integral(k,mass,4,pressure=True), pressure, "1e-70")

    def test_coercivity_and_fixed_density_zero_W_barrier(self):
        with mp.workdps(80):
            d = audit.BindingDesign()
            for y in (mp.sqrt(2), mp.mpf(3), mp.mpf(100)):
                self.assertGreaterEqual(d.potential(y), d.beta*y**8/16)
            ratio = d.potential(mp.mpf("1e10"))/mp.mpf("1e80")
            self.close(ratio, d.beta, "1e-18")
            e1 = d.energy(d.n, mp.mpf(".001"))
            e2 = d.energy(d.n, mp.mpf(".0001"))
            self.assertGreater(e2, e1*90)
            self.assertGreaterEqual(e2, d.Cv*d.n*d.n/(2*mp.mpf(".0001")**2))

    def test_finite_wave_matrix_matches_independent_static_hessian(self):
        with mp.workdps(80):
            d = audit.BindingDesign()
            w = d.y*d.base.W0
            E = lambda n, W: d.energy(n, W/d.base.W0)
            D = mp.diff(lambda n: E(n, w), d.n, 2)
            B = mp.diff(lambda W: mp.diff(lambda n: E(n, W), d.n), w)
            C = mp.diff(lambda W: E(d.n, W), w, 2)
            h = d.finite_wave()
            a,b,c,s,M2,g = (h["a"],h["b"],h["c"],h["s"],h["M2"],d.gomega)
            self.close(a+g*g/M2, D)
            self.close(b-g*c/M2, B)
            self.close(s+c*c/M2, C)
            self.close(h["P0"], M2*C*d.K/(9*d.n))
            self.assertGreater(h["P1_lower_bound"], 0)
            self.assertGreater(h["P1"], h["P1_lower_bound"])
            self.assertGreater(h["P0"], 0)
            for z in (mp.mpf(0), M2/100, M2, M2*100):
                nn,nW,WW = a+g*g/(z+M2),b-g*c/(z+M2),z+s+c*c/(z+M2)
                self.close((z+M2)*(nn*WW-nW*nW), a*z*z+h["P1"]*z+h["P0"])
                self.assertGreater(nn, 0)
                self.assertGreater(nn*WW-nW*nW, 0)

    def test_illustrative_family_not_a_single_fine_tuned_K(self):
        with mp.workdps(80):
            for K in ("220", "240", "260"):
                d = audit.BindingDesign(K_target=K)
                self.assertTrue(0 < d.K < 9*d.n*d.D)
                self.assertGreater(d.beta, 0)
                self.assertTrue(d.finite_wave()["all_mathematical_wave_numbers_positive"])
                self.close(9*d.n*(d.D-d.By*d.By/d.Cy), d.K)

    def test_fitted_liquid_long_wave_normal_closure_static_limit_without_zero_sound(self):
        # Only the isotropic T=0 long-wave normal Landau/Vlasov closure with
        # scalar relaxation and vector-current feedback. No finite-k quantum
        # RPA, pairing, finite-nucleus or full gravity conclusion is implied.
        from nvg_collisionless_response_audit import zero_sound_root
        with mp.workdps(80):
            d = audit.BindingDesign()
            NF = 3*d.n*d.EF/d.k**2
            r = (d.mu-d.EF)/d.EF
            F0 = NF*(d.Cv/d.y**2-d.By**2/d.Cy)
            F1 = -3*r/(1+r)
            cs2 = (d.k/d.EF)**2*(1+F0)*(1+F1/3)/3
            self.assertGreater(1+F0, 0)
            self.assertGreater(1+F1/3, 0)
            self.close(cs2, d.K/(9*d.mu))
            Enn = mp.diff(lambda n: d.energy(n,d.y),d.n,2)
            Eny = mp.diff(lambda y: mp.diff(lambda n: d.energy(n,y),d.n),d.y)
            Eyy = mp.diff(lambda y: d.energy(d.n,y),d.y,2)
            self.close(-NF/(1+F0), -1/(Enn-Eny**2/Eyy))
            self.assertLess(F0,3*r)
            self.assertIsNone(zero_sound_root(F0,r))
            # Disappearance into the particle-hole continuum is not a
            # negative-compressibility or collapse diagnosis.
            self.assertGreater(Enn-Eny**2/Eyy, 0)

    def test_formal_octavic_asymptote_on_independently_solved_branch(self):
        with mp.workdps(80):
            d = audit.BindingDesign()
            a = (d.Cv/(8*d.beta))**mp.mpf(".1")
            previous = None
            for exponent in (12, 24, 36):
                n = d.n*mp.mpf(10)**exponent
                y_leading = a*n**mp.mpf(".2")
                # Independent full stationary equation, not the leading balance.
                residual = lambda t: (d.base.MN*d.base.fermi(n, t*y_leading)["ns"]
                                      +mp.diff(d.barrier, t*y_leading)
                                      -d.Cv*n*n/(t*y_leading)**3)/(d.Cv*n*n/y_leading**3)
                t = mp.findroot(residual, (mp.mpf(".8"), mp.mpf("1.2")))
                self.assertLess(abs(residual(t)), mp.mpf("1e-65"))
                y = t*y_leading
                self.assertGreater(y, d.mu/d.base.MN)  # floor vanishes exactly
                f = d.base.fermi(n,y)
                E = d.energy(n,y)
                mu = f["ef"]+d.Cv*n/y**2
                w = (n*mu-E)/E
                D = f["k"]**2/(3*n*f["ef"])+d.Cv/y**2
                B = d.base.MN**2*y/f["ef"]-2*d.Cv*n/y**3
                C = d.base.MN**2*f["ns_m"]+mp.diff(d.barrier,y,2)+3*d.Cv*n*n/y**4
                cs2 = n*(D-B*B/C)/mu
                errors = [abs(t-1),abs(E/(5*d.beta*a**8*n**mp.mpf("1.6"))-1),
                          abs(w-mp.mpf(".6")),abs(cs2-mp.mpf(".6"))]
                if previous is not None:
                    self.assertTrue(all(v < p for v,p in zip(errors,previous)))
                previous = errors
            self.assertLess(max(errors), mp.mpf("1e-6"))

    def test_existing_polynomial_global_check_is_not_coexistence(self):
        with mp.workdps(80):
            rows = audit.polynomial_examples()
            for row in rows:
                a2,a3,a4 = map(mp.mpf, row["coefficients_MeV4"])
                if a4 < 0:
                    self.assertFalse(row["globally_nonnegative_U"])
                    z = mp.mpf("1e6")
                    self.assertLess(z*z*(a2+a3*z+a4*z*z), 0)
                else:
                    vertex = -a3/(2*a4)
                    self.assertGreater(vertex, -1)
                    minimum = a2-a3*a3/(4*a4)
                    self.assertGreater(minimum, 0)
                    self.close(minimum, mp.mpf(row["q_min_MeV4"]), "1e-32")
                    self.assertTrue(row["globally_nonnegative_U"])
                self.assertFalse(row["ground_state_coexistence_proven_by_this_polynomial_check"])

    def test_80_120_precision_without_expected_output_tables(self):
        with mp.workdps(120):
            low, high = audit.BindingDesign(dps=80), audit.BindingDesign(dps=120)
            for name in ("Cv", "gomega", "By", "Cy", "delta_y", "beta", "U", "Uy", "Uyy"):
                self.close(getattr(low, name), getattr(high, name), "1e-65")
                self.assertEqual(audit.number(getattr(low, name)), audit.number(getattr(high, name)))
            for y in (".3", ".75", ".97", "1", "2"):
                self.close(low.potential(y), high.potential(y), "1e-65")
            for precision in (80, 120):
                r = audit.audit(dps=precision)
                self.assertIs(r["mathematical_checks_passed"], True)
                self.assertEqual(r["evidence_weight"], 0)

    def test_negative_barrier_control_rejects_false_certificate(self):
        with mp.workdps(80):
            d = audit.BindingDesign(barrier_sign=-1)
            y = mp.mpf(".8")
            aux = d.envelope_state(y)
            self.assertLess(d.grand_gap(aux["n"], y), 0)
            self.assertLess(d.potential(10), 0)
            r = audit.audit(barrier_sign=-1)
            self.assertFalse(r["mathematical_checks_passed"])
            self.assertFalse(r["homogeneous_global_certificate"]["passed"])
            self.assertEqual(r["evidence_weight"], 0)
            self.assertFalse(r["constructed_potential"]["coercive_if_beta_positive"])

    def test_symbolic_missing_or_false_certificate_fails_closed(self):
        rows = audit.symbolic_checks()
        corruptions = [{}, {k:v for k,v in rows.items() if k != "inverse_K_identity"}]
        false_rows = {k:dict(v) for k,v in rows.items()}
        false_rows["inverse_K_identity"]["passed"] = 1  # bool identity, not 1==True
        corruptions.append(false_rows)
        for rows in corruptions:
            with mock.patch.object(audit, "symbolic_checks", return_value=rows):
                self.assertFalse(audit.audit()["mathematical_checks_passed"])

    def test_invalid_inputs_and_out_of_feasibility_targets(self):
        for value in (True, False, "nan", "inf", "-inf", -1, 0):
            with self.subTest(y=value), self.assertRaises(ValueError):
                audit.BindingDesign(y=value)
        for kwargs in ({"y": ".99"}, {"K_target": "4000"}, {"binding": "1"},
                       {"binding": "-900"}, {"barrier_sign": 0}, {"barrier_sign": True},
                       {"n_fm3": 0}, {"K_target": True}, {"dps": 79}, {"dps": True}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                audit.BindingDesign(**kwargs)
        d = audit.BindingDesign()
        for n,y in ((-1, 1),(1, 0),(1, "nan"),(True, 1)):
            with self.assertRaises(ValueError):
                d.energy(n,y)

    def test_zero_mixing_special_family_is_unsupported_not_impossible(self):
        with mp.workdps(80):
            ceiling = audit.vector_ceiling()
            with self.assertRaisesRegex(ValueError, "unsupported.*separate finite-C family"):
                audit.BindingDesign(y=ceiling["y_at_ceiling"], K_target="2769")

    def test_unresolved_positive_K_rejected_instead_of_silent_zero_barrier(self):
        for dps,K in ((80,"1e-100"),(120,"1e-140")):
            with self.assertRaisesRegex(ValueError, "unresolved.*precision"):
                audit.BindingDesign(dps=dps,K_target=K)
        with mp.workdps(120):
            d = audit.BindingDesign(dps=120,K_target="1e-100")
            self.assertGreater(d.beta,0)
            recovered = 9*d.n*(d.D-d.By*d.By/d.Cy)
            self.assertGreater(recovered,0)
            # Resolving a 100-digit cancellation leaves fewer significant
            # digits in K than in the individual 120-digit Hessian entries.
            self.assertLess(abs(recovered/d.K-1),mp.mpf("1e-14"))

    def test_strict_json_cli_and_calibration_scope_without_baseline_mutation(self):
        before = dict(INPUTS)
        for args, expected in (([], 0), (["--dps", "bad"], 2), (["--bogus"], 2), (["--dps", "79"], 2)):
            out, err = io.StringIO(), io.StringIO()
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                code = audit.main(args)
            self.assertEqual(code, expected)
            self.assertEqual(err.getvalue(), "")
            r = json.loads(out.getvalue(), parse_constant=lambda x: self.fail(x))
            self.assertEqual(r["evidence_weight"], 0)
            self.assertIs(r["mathematical_checks_passed"], expected == 0)
            if code == 0:
                self.assertTrue(r["design_inputs"]["all_are_calibration_inputs_not_predictions"])
                self.assertTrue(r["design_inputs"]["potential_uses_target_mu_explicitly"])
                self.assertEqual(r["original_parameters_unchanged"], before)
        self.assertEqual(INPUTS, before)
        self.assertEqual(BulkModel().gomega, mp.mpf(INPUTS["gomega"]))


if __name__ == "__main__":
    unittest.main()
