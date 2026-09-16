"""Independent equations and counterexamples, not confirmation of cosmology."""
import contextlib
import io
import math
from pathlib import Path
import sys
import unittest

import mpmath as mp
import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import nvg_cyclic_bounce as bounce
import nvg_cyclic_cosmology as cyclic
import nvg_cyclic_lifetimes as lifetimes
import nvg_bounce_derivation as derivation


class BounceBackgroundTests(unittest.TestCase):
    def test_general_map_from_independent_density_derivative(self):
        # epsilon(n)=n^(4/3)+2n, P=n*epsilon'-epsilon: no NVG solver.
        with mp.workdps(60):
            energy = lambda n: n**(mp.mpf(4)/3)+2*n
            for kind in ("quadratic", "saturating"):
                fmap = (lambda e:e*(1-e/7)) if kind == "quadratic" else (lambda e:e/(1+e/7))
                for value in ("0.2", "1", "5"):
                    n = mp.mpf(value)
                    e = energy(n)
                    p = n*mp.diff(energy,n)-e
                    expected = n*mp.diff(lambda v:fmap(energy(v)),n)-fmap(e)
                    state = bounce.density_map(e,p,7,kind=kind)
                    self.assertAlmostEqual(state["pressure_eff"],float(expected),places=11)

    def test_modified_pressure_is_required_by_continuity(self):
        e,p,h,ec = 0.8,0.8/3,0.2,1.0
        state = bounce.density_map(e,p,ec)
        edot_eff = (1-2*e/ec)*(-3*h*(e+p))
        self.assertAlmostEqual(edot_eff+3*h*state["enthalpy_eff"],0,places=14)
        wrong_pressure_residual = edot_eff+3*h*(state["epsilon_eff"]+p)
        self.assertGreater(abs(wrong_pressure_residual),0.1)

    def test_composed_trace_identity(self):
        for e,p in ((0.2,0.1),(1,1/3),(5,-2)):
            for kind in ("quadratic","saturating"):
                s=bounce.density_map(e,p,3,kind=kind)
                f,fp=s["epsilon_eff"],s["Fprime"]
                self.assertAlmostEqual(s["trace_eff"],fp*(e-3*p)+4*(f-e*fp),places=12)

    def test_exact_solution_against_continuity_and_raychaudhuri(self):
        with mp.workdps(60):
            for w in (mp.mpf(0),mp.mpf(1)/3,mp.mpf(1),mp.mpf("-0.5")):
                q=3*(1+w)
                a=lambda t:(1+(q*t/2)**2)**(1/q)
                y=lambda t:a(t)**(-q)
                h=lambda t:mp.diff(a,t)/a(t)
                for t in map(mp.mpf,("-3","-0.1","0","0.1","3")):
                    self.assertLess(abs(h(t)**2-y(t)*(1-y(t))),mp.mpf("1e-55"))
                    self.assertLess(abs(mp.diff(y,t)+q*h(t)*y(t)),mp.mpf("1e-55"))
                    self.assertLess(abs(mp.diff(h,t)+(q/2)*y(t)*(1-2*y(t))),mp.mpf("1e-55"))
                    state=bounce.exact_bounce(float(t),float(w))
                    self.assertAlmostEqual(float(state["a"]),float(a(t)),places=13)
                    self.assertAlmostEqual(float(state["dH_dt_tchar2"]),float(mp.diff(h,t)),places=13)

    def test_smooth_independent_ode_crosses_both_sides(self):
        times=np.array([5,-3,0,0,-0.01,0.01,-5,2])
        for w in (0,1/3,1,-0.5):
            exact,numerical=bounce.exact_bounce(times,w),bounce.integrate_bounce(times,w)
            for key in numerical:
                np.testing.assert_allclose(numerical[key],exact[key],rtol=2e-8,atol=2e-9)
            self.assertLess(numerical["H_tchar"][1],0)
            self.assertGreater(numerical["H_tchar"][0],0)
            self.assertEqual(numerical["a"][2],1)

    def test_tighter_tolerance_improves_independent_ivp(self):
        times=np.linspace(-5,5,101)
        exact=bounce.exact_bounce(times)
        def error(rtol,atol):
            n=bounce.integrate_bounce(times,rtol=rtol,atol=atol)
            return max(np.max(np.abs(n[k]-exact[k])) for k in n)
        self.assertLess(error(2e-11,2e-13),error(1e-6,1e-8)/20)

    def test_near_vacuum_w_retains_small_increment(self):
        w=-1+1e-14
        with mp.workdps(70):
            q=3*(1+mp.mpf(w))
            expected=mp.exp(mp.log1p((q*10**7/2)**2)/q)
        actual=bounce.exact_bounce(1e7,w)["a"]
        self.assertAlmostEqual(float(actual)/float(expected),1,places=13)
        s=bounce.density_map(1e155,1e155/3,1,kind="saturating")
        self.assertEqual(s["epsilon_eff"],1)
        self.assertGreater(s["enthalpy_eff"],0)
        with self.assertRaises(ValueError):
            bounce.density_map(1e200,1e200/3,1e-150,kind="saturating")
        with self.assertRaises(ValueError): bounce.exact_bounce(1e300)

    def test_bounce_is_not_a_recurrent_cycle(self):
        state=bounce.exact_bounce(np.array([-100,-1,0,1,100]))
        np.testing.assert_array_equal(np.sign(state["H_tchar"]),[-1,-1,0,1,1])
        self.assertIn("recollapse/return map",bounce.compute_bounce_state()["missing"])

    def test_frozen_fake_bounce_fails_raychaudhuri(self):
        # H=0, y=1 forever passes Friedmann AND continuity, but not evolution.
        h,y,hdot,ydot=0.0,1.0,0.0,0.0
        self.assertEqual(h*h-y*(1-y),0)
        self.assertEqual(ydot+4*h*y,0)
        self.assertNotEqual(hdot+2*y*(1-2*y),0)
        self.assertGreater(float(bounce.exact_bounce(0)["dH_dt_tchar2"]),0)

    def test_saturation_is_not_a_finite_density_bounce(self):
        for e in (0.01,1,10,1e9):
            s=bounce.density_map(e,e/3,1,kind="saturating")
            self.assertGreater(s["epsilon_eff"],0)
            self.assertGreater(s["enthalpy_eff"],0)
            self.assertLess(s["epsilon_eff"],1)
        self.assertAlmostEqual(s["pressure_eff"],-1,places=7)

    def test_saturating_material_energy_loses_convexity_before_negative_pressure(self):
        with mp.workdps(60):
            energy=lambda n:n**(mp.mpf(4)/3)
            total=lambda n:energy(n)/(1+energy(n))
            for text in ("0.01","0.1","0.25","1","10"):
                y=mp.mpf(text)
                n=y**mp.mpf("0.75")
                mu=mp.diff(total,n)
                slope=n*mp.diff(total,n,2)/mu
                expected=(1-7*y)/(3*(1+y))
                self.assertGreater(mu,0)
                self.assertLess(abs(slope-expected),mp.mpf("1e-50"))
                self.assertEqual(mp.sign(slope),mp.sign(1-7*y))
        threshold=bounce.density_map(1/7,1/21,1,kind="saturating")
        self.assertAlmostEqual(threshold["epsilon_eff"],1/8,places=14)
        self.assertAlmostEqual(threshold["pressure_eff"],1/48,places=14)
        unstable=bounce.density_map(0.25,0.25/3,1,kind="saturating")
        self.assertGreater(unstable["pressure_eff"],0)

    def test_quadratic_map_requires_negative_effective_enthalpy(self):
        for e,sign in ((0.25,1),(0.5,0),(0.75,-1),(1,-1)):
            s=bounce.density_map(e,e/3,1)
            self.assertEqual(np.sign(s["enthalpy_eff"]),sign)
        self.assertEqual(s["epsilon_eff"],0)
        self.assertAlmostEqual(s["pressure_eff"],-4/3)
        # Noninvertible map: same effective energy with two different pressures.
        low,high=(bounce.density_map(e,e/3,1) for e in (0.25,0.75))
        self.assertEqual(low["epsilon_eff"],high["epsilon_eff"])
        self.assertNotEqual(low["pressure_eff"],high["pressure_eff"])

    def test_degenerate_minimum_is_not_mistaken_for_strict_bounce_criterion(self):
        # Smooth quartic minimum: F'(epsilon_c)=0, Hdot(0)=0. This prevents
        # extending the strict Hdot>0 criterion to every imaginable minimum.
        with mp.workdps(60):
            a=lambda t:(1+t**4)**(mp.mpf(1)/3)
            h=lambda t:mp.diff(a,t)/a(t)
            fmap=lambda e:mp.mpf(16)/9*mp.sqrt(e)*(1-e)**mp.mpf("1.5")
            self.assertEqual(mp.diff(h,0),0)
            # Use an exact one-sided limit: a fractional-power endpoint has
            # finite-difference error proportional to the square root of step.
            import sympy as sp
            e_symbol=sp.symbols("e",positive=True)
            expression=sp.Rational(16,9)*sp.sqrt(e_symbol)*(1-e_symbol)**sp.Rational(3,2)
            self.assertEqual(sp.limit(sp.diff(expression,e_symbol),e_symbol,1,dir="-"),0)
            for t in (mp.mpf("-0.1"),mp.mpf("0.1")):
                e=1/(1+t**4)
                self.assertLess(abs(h(t)**2-fmap(e)),mp.mpf("1e-55"))
                self.assertEqual(mp.sign(h(t)),mp.sign(t))

    def test_gr_positive_matter_turnaround_and_vacuum_counterexample(self):
        for p in (0,1/3,1):
            s=bounce.gr_turning_point(1,p)
            self.assertGreater(s["flat_H2"],0)
            self.assertLess(s["acceleration_over_a_at_H0"],0)
        # Positive curvature plus vacuum energy can bounce; no universal no-go.
        vacuum=bounce.gr_turning_point(1,-1)
        self.assertGreater(vacuum["acceleration_over_a_at_H0"],0)
        self.assertEqual(vacuum["enthalpy"],0)

    def test_temperature_conversion_and_ideal_gas_inverse(self):
        self.assertAlmostEqual(bounce.kB_MeV/8.617333262145e-11,1,places=12)
        e=859**4/bounce.hbar_c**3
        s=bounce.thermal_scale(e)
        self.assertGreater(s["temperature_K"],1e12)
        recovered=math.pi**2/30*s["g_star_input"]*s["kBT_MeV"]**4/bounce.hbar_c**3
        self.assertAlmostEqual(recovered/e,1,places=14)

    def test_invalid_input_rejected_not_clipped(self):
        for value in (-1,float("nan"),float("inf")):
            with self.assertRaises(ValueError): bounce.density_map(value,0,1)
            with self.assertRaises(ValueError): bounce.thermal_scale(value)
        for w in (-1,-2,float("nan")):
            with self.assertRaises(ValueError): bounce.exact_bounce(0,w)
            with self.assertRaises(ValueError): bounce.integrate_bounce([0],w)
        with self.assertRaises(ValueError): bounce.integrate_bounce([[0]])
        with self.assertRaises(ValueError): bounce.exact_bounce(float("nan"))
        with self.assertRaises(ValueError): bounce.density_map(1,0,0)

    def test_scenario_lifetimes_share_one_source_and_growth_law(self):
        s=cyclic.compute_cyclic_state()
        self.assertEqual(lifetimes.compute_lifetimes()["assigned_cycle"],s)
        for factor in (4,9):
            a=cyclic.compute_cyclic_state(cycle_index=1,entropy_factor=factor)
            b=cyclic.compute_cyclic_state(cycle_index=2,entropy_factor=factor)
            self.assertAlmostEqual(b["M_turnaround_g"]/a["M_turnaround_g"],math.sqrt(factor))
            self.assertAlmostEqual(b["lifetime_yr"]/a["lifetime_yr"],math.sqrt(factor))
        self.assertTrue(s["cycle_assignment_is_input"])
        self.assertTrue(s["lifetime_is_total_scenario_scale_not_time_remaining"])

    def test_genesis_sphere_is_below_hayward_horizon_threshold(self):
        import nvg_hayward_evaporation as hayward
        state=cyclic.compute_cyclic_state()
        self.assertAlmostEqual(state["M_genesis_g"]/hayward.M_CRIT,2/(3*math.sqrt(3)),places=13)
        self.assertIsNone(hayward.horizon(state["M_genesis_g"]))
        self.assertIn("not a proved Hayward horizon entropy",state["entropy_area_interpretation"])

    def test_hubble_changes_estimate_not_assigned_cycle(self):
        a,b=(cyclic.compute_cyclic_state(H0_km_s_Mpc=h) for h in (67.4,72.8))
        self.assertNotEqual(a["n_derived"],b["n_derived"])
        self.assertEqual(a["cycle_index"],b["cycle_index"])
        self.assertEqual(a["lifetime_yr"],b["lifetime_yr"])

    def test_cycle_domain(self):
        for kw in ({"cycle_index":1.5},{"cycle_index":True},{"cycle_index":0},
                   {"H0_km_s_Mpc":float("nan")},{"entropy_factor":float("inf")},
                   {"entropy_factor":1},{"cycle_index":100000}):
            with self.assertRaises(ValueError): cyclic.compute_cyclic_state(**kw)

    def test_printed_derivation_no_longer_certifies_missing_action(self):
        output=io.StringIO()
        with contextlib.redirect_stdout(output): state=derivation.derive_modified_friedmann()
        self.assertEqual(state["evidence_status"],"CONDITIONAL_BACKGROUND_NOT_NVG_DERIVATION")
        self.assertIsNone(state["observed_likelihood"])
        self.assertNotIn("BOUNCE IS MATHEMATICALLY RIGOROUS",output.getvalue())
        self.assertIsNone(derivation.check_cmb_bao_compatibility()["observed_likelihood"])
        one,two=derivation.prove_rho_c_uniqueness(),derivation.prove_rho_c_uniqueness(2)
        self.assertEqual(two["epsilon_scale_MeV_fm3"],2*one["epsilon_scale_MeV_fm3"])
        self.assertFalse(one["unique_coefficient_derived"])


if __name__ == "__main__":
    unittest.main()
