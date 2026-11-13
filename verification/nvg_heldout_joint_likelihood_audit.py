#!/usr/bin/env python3
"""Joint held-out likelihood audit: any beta-equilibrium core EOS -> SLy crust -> TOV + Hinderer + Hartle-Thorne
-> the pre-registered 8-term likelihood, the 2-sigma box, the 16O 1p spin-orbit splitting, ground state, vacuum.

What a practitioner gets
------------------------
``evaluate_table(n_fm3, eps, P, ...)`` takes ANY cold beta-equilibrium core table (baryon density [fm^-3], energy
density and pressure [MeV/fm^3], thermodynamically consistent), attaches the SLy crust of the forward-band producer
(verification/nvg_forward_band_propagation_audit.py: Read+2009 Table II, eps-continuity match; imported, not
duplicated), solves the static stellar branch (TOV + Hinderer, codata constants, P_surf = 1e-9), applies the
Hartle-Thorne second-order equatorial-radius correction at the four NICER spins (interior systems of the repository
module verification/nvg_hartle_second_order_quadrupole_probe.py) and returns the stars, the pre-registered chi2
(8 terms with the two nuclear-matter values supplied, else the 6 stellar terms, flagged), per-term z (simple and
PIT), p (chi2 with nu = number of terms, and a replicated-data Monte-Carlo p), the 2-sigma / 1-sigma boxes and, when
symmetric-matter self-energies S(n), V(n) are supplied, the 16O 1p spin-orbit splitting of the validated LDA
procedure (and the exact-Dirac variant in the same LDA potentials).

Built-in model set (end to end, the same code path for every model)
-------------------------------------------------------------------
NL3, FSUGold, DD-ME2, TW99: published parameter sets, generic relativistic mean-field EOS (NL: sigma^3, sigma^4,
omega^4, omega^2 rho^2; DD: Typel-Wolter density-dependent couplings with rearrangement), npe-mu beta equilibrium,
consistent pressure, 0.1-12 n0 at 250 points/decade; nuclear terms from the functional; 16O from the SNM fields;
ground state = global minimum over 0 <= M* <= M at fixed densities.
NVG: the y* = 0.84 quintic stability-island member at a5_min and the y* = 0.87 quartic, centre windows
(n0 0.16, E/A -16, K 240, J 32, L 50), power-law isovector running, consistent pressure; the forward chain of the
forward-band producer (scalar jet, isovector, beta table) is imported unchanged.

Held-out data (PREREG below, the session's pre-registration, embedded verbatim with its SHA-256) enter ONLY the
comparison functions (predictions, chi2_terms, fixed_model_stats, box); no EOS, crust, star or rotation function
reads them.  Strict JSON to stdout; --write stores verification/nvg_heldout_joint_likelihood_results.json.
Evidence weight 0: descriptive held-out comparison of fixed models; nothing here is fitted to the data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np  # noqa: E402
from scipy.integrate import solve_ivp  # noqa: E402
from scipy.interpolate import CubicSpline, PchipInterpolator  # noqa: E402
from scipy.linalg import eigh_tridiagonal  # noqa: E402
from scipy.optimize import brentq, least_squares, minimize_scalar  # noqa: E402
from scipy.special import log_ndtr, ndtri  # noqa: E402
from scipy.stats import chi2 as CHI2  # noqa: E402
from scipy.stats import norm  # noqa: E402

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import nvg_forward_band_propagation_audit as FB  # noqa: E402  (SLy crust, TOV, NVG forward chain)
import nvg_hartle_second_order_quadrupole_probe as H  # noqa: E402  (Hartle second-order interior systems)

SOURCE_PATH = Path(__file__).resolve()
RESULT_PATH = HERE / "nvg_heldout_joint_likelihood_results.json"
SCHEMA = "nvg_heldout_joint_likelihood.v1"
STATUS = "HELDOUT_JOINT_LIKELIHOOD_COMPUTED_ZERO_EVIDENCE"
EVIDENCE_WEIGHT = 0.0

HBARC, MN, PI2 = FB.HBARC, FB.MN, FB.PI2
HC3 = HBARC ** 3
C_KM_S = 299792.458
K_CONV, M_SUN_KM = FB.CONST["codata"]["K_CONV"], FB.CONST["codata"]["M_SUN_KM"]

# ----------------------------------------------------------------------------- stellar settings (no data)
# forward-band DEFAULT_SETTINGS except n_pc = 40 (prereg nvg_class.numerics: n_pc 40 vs 70 changes R(1.44) < 1.2e-4 km)
SETTINGS = dict(FB.DEFAULT_SETTINGS, n_pc=40)
P_SURF = 1e-9
MASSES = (1.0, 1.27, 1.34, 1.36, 1.4, 1.418, 1.44, 1.46, 1.8, 2.07)
# evaluation configurations of the observed stars (mass at which R is predicted, spin frequency in Hz); the
# measured radii/uncertainties are NOT here (PREREG only).  Source: PREREG rotation.spins_Hz and data.*.M
PULSARS = {"J0030": (1.34, 205.5), "J0437": (1.418, 173.7), "J0614": (1.44, 317.6), "J0740": (2.07, 346.5)}
GS_X_SNM = (0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.3)

# ----------------------------------------------------------------------------- pre-registration (held-out data)
# The session pre-registration (written 2026-11-06T08:40:48Z before any likelihood number), verbatim.
PREREG_JSON = r"""{
 "schema": "nvg_joint_likelihood_prereg/1",
 "written_utc": "2026-11-06T08:40:48Z",
 "motivation": "The earlier 'box' test (every criterion inside its 2-sigma interval at once) rejects a correct model with probability 1 - 0.9545^8 = 0.31 for 8 independent data; it is not a proper goodness-of-fit test. The standard test is a joint likelihood; it is declared here before any number is computed and applied identically to NVG, NL3, FSUGold and DD-ME2.",
 "data": {
  "1_Mmax_J0740": {
   "type": "mass_host",
   "m": 2.08,
   "sigma": 0.07,
   "source": "Fonseca et al. 2021 ApJL 915 L12",
   "L": "Phi((M_max - 2.08)/0.07)",
   "chi2": "-2 ln L1"
  },
  "2_R207_J0740": {
   "type": "radius",
   "M": 2.07,
   "c": 12.49,
   "sp": 1.28,
   "sm": 0.88,
   "spin_Hz": 346.5,
   "source": "Salmi et al. 2024 ApJ 974 294 (68%)",
   "rule": "R(2.07) if M_max >= 2.07, else R(M_max) (flagged 'R_at_Mmax_used')"
  },
  "3_R134_J0030": {
   "type": "radius",
   "M": 1.34,
   "c": 12.71,
   "sp": 1.14,
   "sm": 1.19,
   "spin_Hz": 205.5,
   "source": "Riley et al. 2019 ApJL 887 L21 (68%)"
  },
  "4_R1418_J0437": {
   "type": "radius",
   "M": 1.418,
   "c": 11.36,
   "sp": 0.95,
   "sm": 0.63,
   "spin_Hz": 173.7,
   "source": "Choudhury et al. 2024 ApJL 971 L20 (68%)"
  },
  "5_R144_J0614": {
   "type": "radius",
   "M": 1.44,
   "c": 10.29,
   "sp": 1.01,
   "sm": 0.86,
   "spin_Hz": 317.6,
   "source": "Mauviard et al. 2025 (68%)"
  },
  "6_Lt_GW170817": {
   "type": "lambda_tilde",
   "c": 300.0,
   "sp": 255.34186940294305,
   "sm": 139.8300713397069,
   "source": "Abbott et al. 2019 PRX 9 011001, low-spin prior, 300 +420/-230 (90%); sigma = half-width / 1.645",
   "convention": "equal mass 1.36 + 1.36 (nvg_forward_band convention): Lambda_tilde = Lambda(1.36)"
  },
  "7_S2_0p10": {
   "type": "gauss",
   "n_fm3": 0.1,
   "c": 25.5,
   "sigma": 1.0,
   "definition": "S_2(n) = (1/2) d^2(E/A)/d delta^2 at delta = 0 (NVG: k_F^2/(6 E_F*) + C(n) n/8 on the symmetric branch)"
  },
  "8_EPNM_0p10": {
   "type": "gauss",
   "n_fm3": 0.1,
   "c": 11.4,
   "sigma": 1.0,
   "source": "Brown 2013 PRL 111 232502",
   "definition": "E/N of pure neutron matter at 0.10 fm^-3 (eps/n - m_N)"
  }
 },
 "term_rules": {
  "radius_and_lambda": "chi2_k = ((pred - c)/s)^2 with s = sp if pred > c else sm (two-piece Gaussian); L_k = exp(-chi2_k/2)",
  "nuclear": "chi2_k = ((pred - c)/sigma)^2",
  "unsupported_star": "if M_max < M of a radius datum (1.34, 1.418, 1.44) or < 1.36 for GW170817: that star cannot exist in the model -> L_k = 0 (chi2 = +inf)",
  "normalisation": "L = exp(-chi2/2) (product of the 8 terms). Density normalisation constants are identical for every model and cancel in Bayes factors",
  "statistic": "chi2 = sum of the 8 terms; fixed-model p-value = chi2.sf(chi2, nu=8); a Monte-Carlo replicated-data p-value with the generative model below is also reported"
 },
 "rotation": {
  "spins_Hz": {
   "J0030": 205.5,
   "J0437": 173.7,
   "J0614": 317.6,
   "J0740": 346.5
  },
  "prescription": "Hartle 1967 / Hartle-Thorne 1968 second order in Omega.  l=0 (m0, p0*) and l=2 (h2, k2) interior systems, centre data and vacuum matching taken unchanged from the repository module verification/nvg_hartle_second_order_quadrupole_probe.py (Urbanec-Miller-Stuchlik 2013 MNRAS 433 1906; Yagi-Yunes 2013 PRD 88 023009). Surface displacement xi_l = -p_l* (eps+p)/(dp/dr) at r = R, p2* = -h2 - r^2 e^-nu omega_bar^2/3. Circumferential equatorial radius at fixed central pressure: R_eq = R + xi0 - xi2/2 - R k2/2. Fixed gravitational mass: R_eq(M) = R_s(M) + [R_eq - R](Pc*) - (dR_s/dM) dM, dM = m0(R) + J^2/R^3 (bulk; the Reina 2015 surface term is O(eps_surf) and negligible at P_surf = 1e-9).",
  "J0740_rule": "if M_max >= 2.07 and M_max - 2.07 >= 3 dM(2.07): fixed-mass formula; if 2.07 <= M_max < 2.07 + 3 dM: fixed-central-pressure correction (flagged); if M_max < 2.07: fixed-central-pressure correction at the M_max configuration (flagged)",
  "Mmax_term": "static M_max (not rotation corrected; the HT mass increase at the M_max configuration is reported as a diagnostic)",
  "configs": {
   "primary": "rotation-corrected NICER radii",
   "secondary": "static TOV radii"
  },
  "validation": [
   "constant-density star C -> 0 vs Maclaurin (xi2/(Omega_bar^2 R) -> -5/6; fixed-mass R_eq/R - 1 -> (5/12) Omega_bar^2)",
   "oblateness 1 - R_pole/R_eq vs the universal fit Omega_bar^2 (0.788 - 1.030 x) (Morsink et al. 2007 ApJ 663 1244; AlGendy & Morsink 2014 ApJ 791 78) for DD-ME2 and NL3",
   "moment of inertia vs Lattimer & Schutz 2005 ApJ 629 979 fit",
   "static M, R of the background pass vs nvg_forward_band.solve_tov_tidal"
  ]
 },
 "nvg_class": {
  "prior": "uniform on the declared windows of closure/run_band.py WINDOWS, Latin hypercube (run_band.lhs_params), N = 2000, seed 20261106, ids lk0000..lk1999",
  "windows": {
   "n0": [
    0.15,
    0.17
   ],
   "EA": [
    -16.5,
    -15.5
   ],
   "K": [
    220.0,
    260.0
   ],
   "ystar": [
    0.6,
    0.93
   ],
   "Q0": [
    -600.0,
    200.0
   ],
   "J": [
    30.0,
    34.0
   ],
   "L": [
    35.0,
    65.0
   ]
  },
  "model": "power-law isovector running, scenario A (J = sampled J), declared scalar rule (quartic if globally stable, else quintic with a5 from Q0 if stable, else no stable model), thermodynamically consistent pressure, SLy crust (Read+2009, eps-continuity match), P_surf = 1e-9",
  "numerics": "nvg_forward_band.DEFAULT_SETTINGS except n_pc = 40 (closure/samples_conv.jsonl: n_pc 40 vs 70 changes R(1.44) by < 1.2e-4 km, M_max by < 2e-6); masses (1.0, 1.27, 1.34, 1.36, 1.4, 1.418, 1.44, 1.46, 1.8, 2.07)",
  "likelihood_zero": [
   "no globally stable scalar model",
   "ground-state-invalid",
   "no stellar branch / crust match failure (counted and reported separately)"
  ],
  "ground_state_valid": "(a) beta-equilibrium table: tracked branch is the global minimum over y (BetaSolver.global_min_check) at every node with x_match <= x <= x_c(M_max); (b) symmetric matter: design branch (continued from (n0, y*)) is a local (C_y > 0) and global minimum over y at x = 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.3 (the abnormal_scan2 densities); (c) the SNM and PNM states at 0.10 fm^-3 used in terms 7-8 are global minima over y",
  "causality": "max c_s^2 on the stellar branch recorded as a flag, not part of the likelihood"
 },
 "statistics": {
  "evidence": "Z = (1/N) sum_i L_i over all N prior points (L = 0 where no valid model); MC error std(L)/sqrt(N); fixed models: Z = L; Bayes factor B = Z_NVG / L_model",
  "posterior": "importance weights w_i proportional to L_i over the prior samples; ESS = (sum w)^2 / sum w^2",
  "ppc": "posterior predictive p = Pr(T(y_rep, theta) >= T(y_obs, theta)), T = chi2; 40000 posterior draws (seed 7); radii, Lambda: y_rep = pred + two-piece normal noise (sm above the prediction, sp below: the measurement model of the declared likelihood); nuclear: Gaussian; mass datum: latent true mass m ~ N(2.08, 0.07) truncated to m <= M_max(theta) (flat prior), m_rep = m + N(0, 0.07), T_1 = -2 ln Phi((M_max - m_rep)/0.07). Fixed models: the same with theta fixed",
  "loo": "for datum k: weights w_i^(-k) = exp(-sum_{j != k} chi2_ij / 2); predictive of d_k = mixture of the measurement distribution around pred_k(theta_i); z_k = Phi^-1(u_k), u_k = sum w F(d_obs | theta) (PIT; primary); moment z also reported; mass datum (one-sided): P_host = sum w Phi((M_max - 2.08)/0.07), z_1 = -Phi^-1(P_host) if P_host < 0.5 else 0; samples that cannot host star k are reported as p_nohost",
  "best_point": "max L over the prior samples; p-values with nu = 8 (naive) and nu = 1 (all 7 parameters counted as fitted) bracket the look-elsewhere effect",
  "summaries": "weighted mean and 16/50/84% quantiles of the 7 parameters and of R_1.0, R_1.8 (static), M_max, Lambda_1.4, K_sym, L, P_beta(2 n0) (consistent pressure)",
  "box": "2 sigma: M_max >= 1.94 and |z_k| <= 2 for terms 2-8; 1 sigma: M_max >= 2.01 and |z_k| <= 1; earlier-report convention M_max >= 2.01 for both also given"
 },
 "convergence_criterion": "NVG class 'converges without fitting' iff, with the SLy crust and consistent pressure and rotation-corrected radii, posterior predictive p > 0.05 AND every leave-one-out |z| < 2 AND the ground-state condition holds on the posterior support",
 "sensitivity_not_criterion": [
  "J0614 left out entirely (7 data, nu = 7)",
  "static radii (no rotation correction)"
 ],
 "extra_class": "minimal-sextic structural class (declared rule with minimal-a6 sextic fallback for 'no stable model' points), on a subsample if time allows; reported separately",
 "sha256_of_sorted_json": "db896ff88aadacda0490369ab4dbc26e22eac22f9199bd06c8e211b2fdc89f78"
}"""
PREREG = json.loads(PREREG_JSON)
PREREG_SHA256_DECLARED = PREREG["sha256_of_sorted_json"]
DATA = PREREG["data"]
TERMS = ["1_Mmax_J0740", "2_R207_J0740", "3_R134_J0030", "4_R1418_J0437", "5_R144_J0614", "6_Lt_GW170817", "7_S2_0p10", "8_EPNM_0p10"]
SHORT = {"1_Mmax_J0740": "Mmax", "2_R207_J0740": "R207", "3_R134_J0030": "R134", "4_R1418_J0437": "R1418", "5_R144_J0614": "R144",
         "6_Lt_GW170817": "Lt", "7_S2_0p10": "S2", "8_EPNM_0p10": "EPNM"}
PSR = {"2_R207_J0740": "J0740", "3_R134_J0030": "J0030", "4_R1418_J0437": "J0437", "5_R144_J0614": "J0614"}
MKEY = {"3_R134_J0030": "1.34", "4_R1418_J0437": "1.418", "5_R144_J0614": "1.44"}
BOX_MMAX = {"box2": 1.94, "box1": 2.01}   # PREREG statistics.box
DATA_SOURCE_NOTES = {
    "7_S2_0p10": "composite of Zhang & Chen 2013 (PLB 726, 234) and Brown 2013 (PRL 111, 232502); declared 25.5 +- 1.0 MeV "
                 "(session closure/corner_nuclear_heldout.py windows; Lynch & Tsang arXiv:1805.10757 read 24.7 +- 0.8)",
    "8_EPNM_0p10": "Brown, PRL 111, 232502 (2013): E/N(0.10 fm^-3) = 11.4(10) MeV",
}


def prereg_sha256():
    """SHA-256 of the sorted pre-registration JSON without its own hash field (session convention, analyze.py)."""
    body = {k: v for k, v in PREREG.items() if k != "sha256_of_sorted_json"}
    return hashlib.sha256(json.dumps(body, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


# ----------------------------------------------------------------------------- declared nuclear test and joint verdict
SO_DECLARED = {
    "observable": "16O 1p spin-orbit splitting Delta E = E(1p1/2) - E(1p3/2)",
    "experiment_MeV": 6.18,
    "experiment_source": "neutron 1p splitting from 15O/17O single-particle energies (session declared protocol, value as recalled; proton 6.32)",
    "window_MeV": [4.9, 7.4],
    "window_definition": "6.18 MeV +- 20 percent (declared)",
    "procedure_primary_LDA": {
        "density": "rho(r) = rho_c/(1 + exp((r - R)/a)), R = 1.12 A^(1/3) - 0.86 A^(-1/3) fm, a = 0.54 fm, int rho = A; the same profile for every model",
        "local_fields": "S(n), V(n) of the model's homogeneous symmetric matter at n = rho(r) (RMF: S = -Phi, V = g_w w0 (+ Sigma_R for DD); NVG: S = M(y-1), V = C_v n/y^2 on the design branch)",
        "spin_orbit_potential": "V_so = (hbar c)^2/(2 M_eff^2) (1/r) d(V - S)/dr, M_eff = M - (V - S)/2, M = 939 MeV",
        "central_potential": "U_SEP = S + V + (S^2 - V^2)/(2M) + (e/M) V at the level's own energy, iterated to |de| < 1e-5 MeV; Woods-Saxon fit on 0 <= r <= R + 10 a",
        "splitting": "first order: Delta E = -(2l+1)/2 int u^2 V_so dr; FD Schroedinger h = 0.01 fm, box 20 fm",
        "exact_dirac_variant": "radial Dirac eigenvalues in the same LDA potentials (kappa = -2 and +1, nr = 1), no WS fit, no perturbation theory; robustness variant, not the declared statistic",
    },
    "validation_rule": "NL3 and DD-ME2 raw LDA 16O 1p within 6.18 +- 20 percent ([4.944, 7.416] MeV) -> raw LDA numbers used for every model; otherwise one global scale factor fixed by NL3 and DD-ME2 only",
    "source": "session spin-orbit protocol (creative/spin_orbit/declared.py, so_core.py, run_spin_orbit.splittings); round-3 RMF survey (rmf_survey/run_survey.py)",
}
JOINT_VERDICT_DECLARED = {
    "S_stars": "the pre-registered 8-term likelihood with SLy crust, consistent pressure and rotation-corrected NICER radii; fixed models: "
               "p = chi2.sf(chi2, nu = 8) > 0.05 and all |z_PIT| < 2; the 2-sigma box reported separately",
    "N_nuclear_matter": "S_2(0.10 fm^-3) = 25.5 +- 1.0 and E_PNM/N(0.10) = 11.4 +- 1.0 MeV, inside the 8 terms",
    "O_spin_orbit": "16O 1p splitting from the validated LDA procedure in [4.9, 7.4] MeV",
    "G_ground_state": "no Lee-Wick abnormal state below the stellar central density (RMF: global minimum over 0 <= M* <= M at 120 SNM + 60 "
                      "beta densities from 0.02 n0 to n_c(M_max); NVG: tracked branch is the global minimum over y on the beta table between "
                      "the crust match and x_c(M_max), SNM at x = 0.05..1.3, SNM and PNM at 0.10 fm^-3)",
    "V_vacuum": "NVG: global vacuum stability q(z) >= 0 for z >= -1 (rounding tolerance 1e-12 max|a_j| for the marginal a5_min edge); RMF: not applicable",
    "converges": "S and O and G and V (each reported separately)",
    "source": "session round-3 common brief (joint held-out verdict, declared before compute)",
}


# ============================================================================= Hartle-Thorne second order (lk_core)
class TabBackground:
    """H.Background protocol (geometric km units) from a TOV + nu + omega_bar pass on an EOS object."""

    def __init__(self, eos, r, m_msun, P_mev, nu, w, dw):
        self.eos = eos
        nu = nu - nu[-1] + math.log(1.0 - 2.0 * m_msun[-1] * M_SUN_KM / r[-1])     # e^nu(R) = 1 - 2M/R
        self._r0, self._R = float(r[0]), float(r[-1])
        self._spl = CubicSpline(r, np.vstack([m_msun * M_SUN_KM, np.log(P_mev), nu, w, dw]).T)
        self.radius_km = float(r[-1])
        self.mass_km = float(m_msun[-1] * M_SUN_KM)
        ec, dc = eos.eps_dedp(float(P_mev[0]))
        self.eps_c = ec * K_CONV
        self.p_c = float(P_mev[0]) * K_CONV
        self.dedp_c = dc
        self.enu_inv_c = math.exp(-nu[0])
        self._last_r = None
        self._vals = None

    def _get(self, r):
        r = min(max(r, self._r0), self._R)
        if r != self._last_r:
            m, lnP, nu, w, dw = self._spl(r)
            P = math.exp(lnP)
            e, dedp = self.eos.eps_dedp(P)
            self._vals = (float(m), P * K_CONV, e * K_CONV, dedp, math.exp(-nu), float(w), float(dw))
            self._last_r = r
        return self._vals

    def m(self, r): return self._get(r)[0]
    def p(self, r): return self._get(r)[1]
    def eps(self, r): return self._get(r)[2]
    def dedp(self, r): return self._get(r)[3]
    def enu_inv(self, r): return self._get(r)[4]
    def omega(self, r): return self._get(r)[5]
    def domega(self, r): return self._get(r)[6]


def background_pass(eos, Pc, p_surf=P_SURF, rtol=1e-9, atol=1e-11, max_step=0.05, r0=1e-6):
    """TOV (same RHS as FB.solve_tov_tidal, codata) + d nu/dr + first-order frame dragging omega_bar (omega_bar(0) = 1)."""
    fourpi = 4.0 * math.pi
    eps_dedp = eos.eps_dedp
    ec = eps_dedp(Pc)[0]

    def rhs(r, v):
        m, P, nu, w, dw = v
        if P <= 0.0:
            return [0.0, 0.0, 0.0, 0.0, 0.0]
        e = eps_dedp(P)[0]
        ek, pk, mk = e * K_CONV, P * K_CONV, m * M_SUN_KM
        fac = r * (r - 2.0 * mk)
        dm = fourpi * r * r * ek / M_SUN_KM
        dP = -(ek + pk) * (mk + fourpi * r ** 3 * pk) / fac / K_CONV
        dnu = 2.0 * (mk + fourpi * r ** 3 * pk) / fac
        drag = fourpi * r * (ek + pk) / (1.0 - 2.0 * mk / r)
        ddw = (drag - 4.0 / r) * dw + 4.0 * drag / r * w
        return [dm, dP, dnu, dw, ddw]

    def surf(r, v):
        return v[1] - p_surf
    surf.terminal = True
    surf.direction = -1
    s0 = (ec + Pc) * K_CONV
    y0 = [0.0, float(Pc), 0.0, 1.0 + 1.6 * math.pi * s0 * r0 ** 2, 3.2 * math.pi * s0 * r0]
    sol = solve_ivp(rhs, (r0, 100.0), y0, method="DOP853", rtol=rtol, atol=atol, max_step=max_step, events=[surf])
    if not sol.success or sol.t_events[0].size == 0:
        raise RuntimeError(f"background pass failed: {sol.message}")
    r = sol.t.copy()
    Y = sol.y.copy()
    keep = np.concatenate([np.diff(r) > 1e-12, [True]])
    r, Y = r[keep], Y[:, keep]
    Y[1] = np.maximum(Y[1], p_surf * 0.999999)
    return r, Y


def surface_deformation(bg, so):
    """Hartle 1967 surface quantities in seed units: xi0, xi2, k2(R) -> dR_eq (fixed Pc) = xi0 - xi2/2 - R k2/2."""
    R, M = bg.radius_km, bg.mass_km
    ss = so.surface_state
    h2R, k2R, p0R = ss["h2_interior"], ss["K2_interior"], ss["p0_R"]
    wR, enuR, pR = bg.omega(R), bg.enu_inv(R), bg.p(R)
    fac = R * (R - 2.0 * M) / (M + 4.0 * math.pi * R ** 3 * pR)       # = -(eps+p)/(dp/dr) at R
    xi0 = p0R * fac
    p2s = -h2R - R * R * enuR * wR * wR / 3.0
    xi2 = p2s * fac
    return {"R": R, "M_km": M, "xi0": xi0, "xi2": xi2, "k2R": k2R, "h2R": h2R, "p0R": p0R,
            "dReq": xi0 - 0.5 * xi2 - 0.5 * R * k2R, "dRpole": xi0 + xi2 + R * k2R,
            "dM_bulk_km": so.deltaM_seed_km, "dM_total_candidate_km": ss["deltaM_total_candidate_seed_km"],
            "Omega_seed": so.Omega_seed_km1, "I_seed_km3": so.I_over_MR2 * M * R * R, "q_tilde": so.q_tilde}


def rotating_star(eos, Pc, freqs_Hz, p_surf=P_SURF, so_rtol=1e-8, so_atol=1e-10, so_max_step=0.25):
    """Static star at Pc plus Hartle-Thorne second-order corrections at the given spins (Hz)."""
    r, Y = background_pass(eos, Pc, p_surf=p_surf)
    bg = TabBackground(eos, r, Y[0], Y[1], Y[2], Y[3], Y[4])
    Om_s, I_s, J_s = H._seed_angular_from_background(bg)
    so = H.solve_second_order(bg, J_seed=J_s, Omega_seed=Om_s, I_seed=I_s, rtol=so_rtol, atol=so_atol, max_step=so_max_step)
    sd = surface_deformation(bg, so)
    R, M = sd["R"], sd["M_km"]
    out = {"Pc": float(Pc), "R_static": R, "M_static": M / M_SUN_KM, "I_over_MR2": so.I_over_MR2, "compactness": M / R, "spins": {}}
    for f in freqs_Hz:
        Om = 2.0 * math.pi * f / C_KM_S
        sc = (Om / Om_s) ** 2
        dReq = sd["dReq"] * sc
        Req = R + dReq
        Rpole = R + sd["dRpole"] * sc
        out["spins"][str(f)] = {"Omega_km": Om, "Omega_bar2": Om * Om * R ** 3 / M, "dReq_fixedPc_km": dReq, "R_eq_fixedPc": Req,
                                "oblateness_1_minus_Rp_over_Req": 1.0 - Rpole / Req, "xi0_km": sd["xi0"] * sc,
                                "xi2_km": sd["xi2"] * sc, "Rk2_km": R * sd["k2R"] * sc, "dM_bulk_Msun": sd["dM_bulk_km"] * sc / M_SUN_KM}
    return out


def hartle_maclaurin_validation():
    """Constant-density star C -> 0 (Richardson in C): xi2/(Ob^2 R) -> -5/6, fixed-mass xi0 -> 0, fixed-mass dR_eq/(Ob^2 R) -> 5/12."""
    rows = []
    for C in (0.02, 0.008, 0.003, 0.0015):
        bg = H.build_constant_density_background(compactness=C, radius_km=10.0)
        Om_s, I_s, J_s = H._seed_angular_from_background(bg)
        so = H.solve_second_order(bg, J_seed=J_s, Omega_seed=Om_s, I_seed=I_s, rtol=1e-10, atol=1e-12)
        sd = surface_deformation(bg, so)
        R, M = sd["R"], sd["M_km"]
        Ob2 = Om_s ** 2 * R ** 3 / M
        dM_tot = sd["dM_total_candidate_km"]
        rows.append({"C": C, "xi2_over_Ob2R": sd["xi2"] / (Ob2 * R),
                     "xi0_fixedM_over_Ob2R": (sd["xi0"] - R / (3.0 * M) * dM_tot) / (Ob2 * R),
                     "dReq_fixedM_over_Ob2R": (sd["dReq"] - R / (3.0 * M) * dM_tot) / (Ob2 * R)})
    c1, c2 = rows[-2]["C"], rows[-1]["C"]
    lim = {k: (rows[-2][k] * c2 - rows[-1][k] * c1) / (c2 - c1) for k in ("xi2_over_Ob2R", "xi0_fixedM_over_Ob2R", "dReq_fixedM_over_Ob2R")}
    target = {"xi2_over_Ob2R": -5.0 / 6.0, "xi0_fixedM_over_Ob2R": 0.0, "dReq_fixedM_over_Ob2R": 5.0 / 12.0}
    return {"rows": rows, "richardson_C0": lim, "targets": target,
            "pass_1e-2": bool(all(abs(lim[k] - target[k]) < 1e-2 for k in target)),
            "source": "Maclaurin spheroid limit of Hartle 1967 (session creative/likelihood/validate_rot.py V2)"}


# ============================================================================= stars (lk_core.star_block)
def branch_logged(eos, settings=None, p_surf=P_SURF, masses=MASSES):
    """FB.stellar_branch with a log of every TOV solve (used to locate the stars for the rotation pass)."""
    s = dict(SETTINGS)
    s.update(settings or {})
    log = []

    def solver(e, Pc, **kw):
        res = FB.solve_tov_tidal(e, Pc, **kw)
        log.append((float(Pc),) + tuple(float(v) for v in res))
        return res
    tov = {"rtol": s["rtol"], "atol": s["atol"], "max_step": s["max_step"], "p_surf": p_surf, "const": s["const"]}
    br = FB.stellar_branch(eos, n_pc=s["n_pc"], pc_min=s["pc_min"], pc_frac=s["pc_frac"], masses=masses, tov=tov, solver=solver)
    return br, log, tov


def pc_for_mass(eos, br, log, mt, tov, tol=2e-4):
    """Central pressure of the stable-branch star of mass mt: nearest logged solve, secant-refined if needed."""
    Pmax = br["M_max_Pc"]
    st = sorted([e for e in log if e[0] <= Pmax * (1 + 1e-12) and 3 < e[2] < 60], key=lambda e: e[0])
    best = min(st, key=lambda e: abs(e[1] - mt))
    if abs(best[1] - mt) <= tol:
        return best[0], best[1], best[2]
    Ms = np.array([e[1] for e in st]); Ps = np.array([e[0] for e in st])
    i = int(np.searchsorted(Ms, mt)) if np.all(np.diff(Ms) > 0) else int(np.argmin(np.abs(Ms - mt)))
    i = min(max(i, 1), len(Ms) - 1)
    lp = math.log(Ps[i - 1]) + (mt - Ms[i - 1]) * (math.log(Ps[i]) - math.log(Ps[i - 1])) / (Ms[i] - Ms[i - 1])
    lp = min(lp, math.log(Pmax))
    for _ in range(6):
        M1, R1, _, _ = FB.solve_tov_tidal(eos, math.exp(lp), **tov)
        if abs(M1 - mt) <= tol:
            return math.exp(lp), M1, R1
        M2, R2, _, _ = FB.solve_tov_tidal(eos, math.exp(lp + 0.01), **tov)
        slope = (M2 - M1) / 0.01
        if slope <= 0:
            break
        lp = min(lp + (mt - M1) / slope, math.log(Pmax))
    return math.exp(lp), M1, R1


def dRdM_static(br, log):
    """PCHIP derivative of R(M) on the logged stable-branch solves."""
    Pmax = br["M_max_Pc"]
    st = sorted([e for e in log if e[0] <= Pmax * (1 + 1e-12) and 3 < e[2] < 60], key=lambda e: e[0])
    Ms = np.array([e[1] for e in st]); Rs = np.array([e[2] for e in st])
    cm = np.maximum.accumulate(Ms)
    keep = np.concatenate([[True], Ms[1:] > cm[:-1] + 1e-9])
    Ms, Rs = Ms[keep], Rs[keep]
    return PchipInterpolator(Ms, Rs).derivative(), float(Ms[0]), float(Ms[-1])


def rotation_corrections(eos, br, log, tov):
    """Rotation-corrected circumferential equatorial radii at fixed gravitational mass for the four NICER pulsars:
    dR_rot = dR_eq(fixed Pc) - (dR_s/dM) dM_bulk; J0740 near/above M_max: fixed-Pc correction (flagged)."""
    out = {}
    dRdM, mlo, mhi = dRdM_static(br, log)
    Mmax = br["M_max"]
    for name, (mt, f) in PULSARS.items():
        t0 = time.time()
        rec = {"M_target": mt, "spin_Hz": f}
        try:
            if Mmax >= mt:
                Pc, M1, R1 = pc_for_mass(eos, br, log, mt, tov)
                rs = rotating_star(eos, Pc, [f])
                sp = rs["spins"][str(f)]
                dM = sp["dM_bulk_Msun"]
                rec.update(Pc=Pc, M_solve=M1, R_solve=R1, R_static_bg=rs["R_static"], M_static_bg=rs["M_static"],
                           dReq_fixedPc=sp["dReq_fixedPc_km"], dM_fixedPc=dM, Omega_bar2=sp["Omega_bar2"],
                           oblateness=sp["oblateness_1_minus_Rp_over_Req"], I_over_MR2=rs["I_over_MR2"], compactness=rs["compactness"])
                if name == "J0740" and (Mmax - mt) < 3.0 * dM:
                    rec["mode"] = "fixed_Pc_near_Mmax"
                    rec["dR_rot"] = sp["dReq_fixedPc_km"]
                else:
                    slope = float(dRdM(min(max(M1, mlo), mhi)))
                    rec["dRdM_static"] = slope
                    rec["mode"] = "fixed_mass"
                    rec["dR_rot"] = sp["dReq_fixedPc_km"] - slope * dM
            elif name == "J0740":
                Pc = br["M_max_Pc"]
                rs = rotating_star(eos, Pc, [f])
                sp = rs["spins"][str(f)]
                rec.update(Pc=Pc, R_static_bg=rs["R_static"], M_static_bg=rs["M_static"], dReq_fixedPc=sp["dReq_fixedPc_km"],
                           dM_fixedPc=sp["dM_bulk_Msun"], Omega_bar2=sp["Omega_bar2"], oblateness=sp["oblateness_1_minus_Rp_over_Req"],
                           I_over_MR2=rs["I_over_MR2"], compactness=rs["compactness"], mode="fixed_Pc_at_Mmax", dR_rot=sp["dReq_fixedPc_km"])
            else:
                rec["mode"] = "unsupported"
                rec["dR_rot"] = None
        except Exception as exc:  # noqa: BLE001
            rec["mode"] = f"failed: {exc!r}"
            rec["dR_rot"] = None
        rec["wall_s"] = time.time() - t0
        out[name] = rec
    return out


def star_block(eos, settings=None, p_surf=P_SURF, rotation=True):
    """Static stellar branch (+ Hartle-Thorne corrections) of a crust-attached EOS."""
    t0 = time.time()
    br, log, tov = branch_logged(eos, settings, p_surf)
    if br["status"] != "ok":
        return {"status": "no_branch", "branch": br}
    rec = {"status": "ok", "M_max": br["M_max"], "M_max_Pc": br["M_max_Pc"], "R_at_Mmax": br["R_at_Mmax"],
           "M_max_on_grid_end": br["M_max_on_grid_end"], "R_of_M": br["R_of_M"], "Lambda_of_M": br["Lambda_of_M"],
           "M_err_of_M": br.get("M_err_of_M"), "max_cs2_stellar_branch": br["max_cs2_stellar_branch"],
           "n_c_Mmax_fm3": br["n_c_Mmax_fm3"], "M_min_branch": br["M_min_branch"], "n_tov_solves_branch": br["n_tov_solves"]}
    rec["wall_branch_s"] = time.time() - t0
    if rotation:
        t1 = time.time()
        rec["rotation"] = rotation_corrections(eos, br, log, tov)
        rec["wall_rotation_s"] = time.time() - t1
    return rec


# ============================================================================= likelihood (analyze.py; the ONLY consumers of DATA)
def sides(term):
    d = DATA[term]
    if d["type"] in ("radius", "lambda_tilde"):
        return d["c"], d["sp"], d["sm"]
    return d["c"], d["sigma"], d["sigma"]


def predictions(stars, nuc, rotation):
    """Model predictions for the 8 data; None = the star cannot exist in the model (L_k = 0) or a nuclear value not supplied."""
    Mmax = stars["M_max"]
    R, Lm, rot = stars["R_of_M"], stars["Lambda_of_M"], stars.get("rotation", {})
    p, flags = {}, []
    p["1_Mmax_J0740"] = Mmax
    for term in ("2_R207_J0740", "3_R134_J0030", "4_R1418_J0437", "5_R144_J0614"):
        psr = PSR[term]
        if term == "2_R207_J0740":
            if Mmax >= DATA[term]["M"] and R.get("2.07") is not None:
                v = R["2.07"]
            else:
                v = stars["R_at_Mmax"]
                flags.append("R_at_Mmax_used_for_J0740")
        else:
            v = R.get(MKEY[term]) if Mmax >= DATA[term]["M"] else None
        if v is not None and rotation:
            dr = (rot.get(psr) or {}).get("dR_rot")
            if dr is None:
                flags.append(f"rotation_missing_{psr}")
            else:
                v = v + dr
        p[term] = v
    L136 = Lm.get("1.36") if Mmax >= 1.36 else None
    p["6_Lt_GW170817"] = FB.lambda_tilde(1.36, 1.36, L136, L136) if L136 is not None else None
    p["7_S2_0p10"] = nuc.get("S2_0p10")
    p["8_EPNM_0p10"] = nuc.get("EPNM_0p10")
    return p, flags


def chi2_terms(p):
    """chi2_k: mass host -2 ln Phi((M_max - m)/sigma); radii / Lambda two-piece Gaussian; nuclear Gaussian; unsupported -> inf."""
    c, z = {}, {}
    m0, s0 = DATA["1_Mmax_J0740"]["m"], DATA["1_Mmax_J0740"]["sigma"]
    m = p["1_Mmax_J0740"]
    c["1_Mmax_J0740"] = float(-2.0 * log_ndtr((m - m0) / s0))
    z["1_Mmax_J0740"] = (m - m0) / s0
    for term in TERMS[1:]:
        v = p[term]
        if v is None:
            c[term] = math.inf
            z[term] = None
            continue
        cc, sp, sm = sides(term)
        s = sp if v > cc else sm
        z[term] = (v - cc) / s
        c[term] = z[term] ** 2
    return c, z


def box(p, z, terms=TERMS):
    """2-sigma box: M_max >= 1.94 and |z_k| <= 2 for the other terms; 1-sigma: M_max >= 2.01 and |z_k| <= 1."""
    m = p["1_Mmax_J0740"]
    rest = [t for t in terms if t != "1_Mmax_J0740"]
    zz = [z[t] for t in rest]
    ok = all(v is not None for v in zz)
    b2 = bool(ok and m >= BOX_MMAX["box2"] and all(abs(v) <= 2 for v in zz))
    b1 = bool(ok and m >= BOX_MMAX["box1"] and all(abs(v) <= 1 for v in zz))
    b2_old = bool(ok and m >= BOX_MMAX["box1"] and all(abs(v) <= 2 for v in zz))
    fails2 = [SHORT[t] for t in rest if z[t] is None or abs(z[t]) > 2] + (["Mmax"] if m < BOX_MMAX["box2"] else [])
    fails1 = [SHORT[t] for t in rest if z[t] is None or abs(z[t]) > 1] + (["Mmax"] if m < BOX_MMAX["box1"] else [])
    return {"box2": b2, "box1": b1, "box2_Mmax_ge_2.01": b2_old, "fails2": fails2, "fails1": fails1}


def two_piece_cdf(d, mu, sp, sm):
    """CDF at d of the measurement distribution around mu: below mu width sp, above mu width sm."""
    tot = sp + sm
    if d <= mu:
        return 2.0 * sp / tot * norm.cdf((d - mu) / sp)
    return sp / tot + 2.0 * sm / tot * (norm.cdf((d - mu) / sm) - 0.5)


def T1_rep(Mmax, rng, size):
    """Replicated mass-term discrepancy: m_true ~ N(m, s) truncated <= M_max; m_rep = m_true + N(0, s)."""
    m0, s0 = DATA["1_Mmax_J0740"]["m"], DATA["1_Mmax_J0740"]["sigma"]
    a = norm.cdf((Mmax - m0) / s0)
    u = rng.uniform(0.0, 1.0, size) * a
    u = np.clip(u, 1e-300, None)
    m_true = m0 + s0 * ndtri(u)
    m_rep = m_true + s0 * rng.standard_normal(size)
    return -2.0 * log_ndtr((Mmax - m_rep) / s0)


def fixed_model_stats(p, terms, rng, ndraw=200000):
    """chi2, p(chi2, nu), replicated-data MC p, per-term simple z and PIT z (fixed-model analogue of LOO z)."""
    c, z = chi2_terms(p)
    chi = sum(c[t] for t in terms)
    nu = len(terms)
    m0, s0 = DATA["1_Mmax_J0740"]["m"], DATA["1_Mmax_J0740"]["sigma"]
    out = {"terms": [SHORT[t] for t in terms], "chi2_terms": {SHORT[t]: c[t] for t in terms}, "z_simple": {SHORT[t]: z[t] for t in terms},
           "chi2": chi, "nu": nu, "p_chi2_nu": float(CHI2.sf(chi, nu)) if math.isfinite(chi) else 0.0, "lnL": -0.5 * chi}
    k_gauss = nu - (1 if "1_Mmax_J0740" in terms else 0)
    Trep = rng.chisquare(k_gauss, ndraw) + (T1_rep(p["1_Mmax_J0740"], rng, ndraw) if "1_Mmax_J0740" in terms else 0.0)
    out["p_replicated_MC"] = float(np.mean(Trep >= chi)) if math.isfinite(chi) else 0.0
    zp = {}
    for t in terms:
        if t == "1_Mmax_J0740":
            ph = float(norm.cdf((p[t] - m0) / s0))
            zp[SHORT[t]] = float(-ndtri(ph)) if ph < 0.5 else 0.0
        elif p[t] is None:
            zp[SHORT[t]] = None
        else:
            cc, sp, sm = sides(t)
            u = two_piece_cdf(cc, p[t], sp, sm)
            zp[SHORT[t]] = float(ndtri(min(max(u, 1e-300), 1 - 1e-16)))
    out["z_PIT"] = zp
    out["max_abs_z_PIT"] = max((abs(v) if v is not None else math.inf) for v in zp.values())
    return out


def likelihood_block(stars, nuc, ndraw=200000, seed=7):
    """Primary (rotation-corrected radii), sensitivity without J0614, secondary static radii; fresh rng(seed) per model.
    Without the two nuclear values the six stellar terms are used and the result is flagged (not the declared statistic)."""
    have_nuc = nuc.get("S2_0p10") is not None and nuc.get("EPNM_0p10") is not None
    base = list(TERMS) if have_nuc else TERMS[:6]
    rng = np.random.default_rng(seed)
    out = {"declared_8_term_statistic": bool(have_nuc)}
    if not have_nuc:
        out["flag"] = "nuclear_terms_not_supplied: chi2 over the 6 stellar terms (nu = 6); S verdict undecided"
    for cname, terms, rot in (("primary_rotation", base, True), ("sensitivity_no_J0614", [t for t in base if t != "5_R144_J0614"], True),
                              ("secondary_static", base, False)):
        p, fl = predictions(stars, nuc, rot)
        s = fixed_model_stats(p, terms, rng, ndraw=ndraw)
        s["predictions"] = {SHORT[t]: p[t] for t in terms}
        s["flags"] = fl
        s["box"] = box(p, chi2_terms(p)[1], terms)
        s["passes_S"] = bool(s["p_chi2_nu"] > 0.05 and s["max_abs_z_PIT"] < 2)
        out[cname] = s
    return out


# ============================================================================= 16O spin-orbit LDA (so_core)
N_GRID = np.linspace(0.0, 0.26, 521)     # fm^-3, symmetric-matter field grid
_TRAPZ = getattr(np, "trapezoid", None) or np.trapz


class FieldTable:
    """S(n), V(n) cubic splines (MeV) on a density grid (fm^-3)."""

    def __init__(self, name, n, S, V):
        self.name = name
        self.n = np.asarray(n, float)
        self.S = np.asarray(S, float)
        self.V = np.asarray(V, float)
        self.sS = CubicSpline(self.n, self.S)
        self.sV = CubicSpline(self.n, self.V)
        self.dS = self.sS.derivative()
        self.dV = self.sV.derivative()

    def at(self, n):
        n = np.clip(n, 0.0, self.n[-1])
        return self.sS(n), self.sV(n), self.dS(n), self.dV(n)


class Nucleus:
    def __init__(self, label, A, a=0.54, R=None, rmax=None):
        self.label, self.A, self.a = label, A, a
        self.R = 1.12 * A ** (1.0 / 3.0) - 0.86 * A ** (-1.0 / 3.0) if R is None else R
        self.rmax = rmax if rmax is not None else (25.0 if A > 100 else 20.0)
        r = np.linspace(0.0, 40.0, 40001)
        f = 1.0 / (1.0 + np.exp((r - self.R) / self.a))
        self.rho_c = A / (4.0 * math.pi * _TRAPZ(f * r * r, r))

    def rho(self, r):
        return self.rho_c / (1.0 + np.exp((r - self.R) / self.a))

    def drho(self, r):
        rho = self.rho(r)
        return -rho * (1.0 - rho / self.rho_c) / self.a


def lda_profiles(tab, nuc, r):
    n = nuc.rho(r)
    dn = nuc.drho(r)
    S, V, dS, dV = (np.array(v, float) for v in tab.at(n))
    dVmS_dr = (dV - dS) * dn
    Meff = MN - 0.5 * (V - S)
    rr = np.where(r > 0, r, 1e-12)
    Vso = HBARC ** 2 / (2.0 * Meff ** 2) * dVmS_dr / rr
    return dict(n=n, dn=dn, S=S, V=V, dVmS_dr=dVmS_dr, Meff=Meff, Vso=Vso)


def ws(r, V0, R, a):
    return V0 / (1.0 + np.exp((r - R) / a))


def ws_fit(r, U, R0, a0, rcut):
    m = r <= rcut
    rr, uu = r[m], U[m]
    res = least_squares(lambda p: ws(rr, *p) - uu, [float(uu[0]), R0, a0], bounds=([-500, 0.5, 0.1], [200, 15, 3.0]), xtol=1e-14, ftol=1e-14)
    return res.x, float(np.sqrt(np.mean(res.fun ** 2)))


def schrodinger_level(r, U, l, nr, mass=MN):
    """FD radial Schroedinger equation on r[1:] (Dirichlet), energy and normalised u (int u^2 dr = 1)."""
    h = r[1] - r[0]
    rr = r[1:]
    t = HBARC ** 2 / (2.0 * mass)
    d = 2.0 * t / h ** 2 + t * l * (l + 1) / rr ** 2 + U[1:]
    e = -t / h ** 2 * np.ones(len(rr) - 1)
    w, v = eigh_tridiagonal(d, e, select="i", select_range=(nr - 1, nr - 1))
    u = v[:, 0]
    u = u / math.sqrt(np.sum(u * u) * h)
    if u[np.argmax(np.abs(u))] < 0:
        u = -u
    nodes = int(np.sum(np.diff(np.sign(u[np.abs(u) > 1e-8 * np.max(np.abs(u))])) != 0))
    return float(w[0]), np.concatenate([[0.0], u]), nodes


def level_lda(prof, r, nuc, l, nr, tol=1e-5):
    """Iterate e -> U_SEP(e) -> Woods-Saxon fit -> Schroedinger level."""
    S, V = prof["S"], prof["V"]
    e = -20.0
    wsp, rms, nodes, u = None, None, None, None
    for it in range(40):
        U = S + V + (S * S - V * V) / (2.0 * MN) + (e / MN) * V
        wsp, rms = ws_fit(r, U, nuc.R, nuc.a, nuc.R + 10.0 * nuc.a)
        e_new, u, nodes = schrodinger_level(r, ws(r, *wsp), l, nr)
        if abs(e_new - e) < tol:
            e = e_new
            break
        e = e_new
    return dict(e=e, u=u, nodes=nodes, ws=[float(x) for x in wsp], ws_rms=rms, iters=it + 1)


def first_order_split(u, r, Vso, l):
    h = r[1] - r[0]
    return -(2 * l + 1) / 2.0 * float(np.sum(u * u * Vso) * h)


def dirac_level(r, S, V, kappa, nr, e0=-20.0, tol=1e-7):
    """Radial Dirac eigenvalue in LDA potentials (upper-component second-order form, D = 2M + e - V + S, iterated)."""
    l = kappa if kappa > 0 else -kappa - 1
    h = r[1] - r[0]
    hc2 = HBARC ** 2
    e = e0
    for it in range(60):
        D = 2.0 * MN + e - V + S
        invD = 1.0 / D
        invDh = 0.5 * (invD[1:] + invD[:-1])
        dinvD = np.gradient(invD, h)
        rs = np.where(r > 0, r, 1e-12)
        pot = hc2 * (l * (l + 1) / rs ** 2 * invD - dinvD * (kappa / rs)) + V + S
        a_half = hc2 * invDh / h ** 2
        dia = a_half[:-1] + a_half[1:] + pot[1:-1]
        off = -a_half[1:-1]
        w, v = eigh_tridiagonal(dia, off, select="i", select_range=(nr - 1, nr - 1))
        e_new = float(w[0])
        if abs(e_new - e) < tol:
            e = e_new
            break
        e = e_new
    return e, it + 1


def spin_orbit_16O(tab, h=0.01, exact_dirac=True):
    """16O 1p splitting of the declared LDA procedure (primary) and the exact-Dirac variant (no experimental input)."""
    nuc = Nucleus("16O", 16)
    r = np.arange(0.0, nuc.rmax + 0.5 * h, h)
    prof = lda_profiles(tab, nuc, r)
    lv = level_lda(prof, r, nuc, 1, 1)
    d0 = first_order_split(lv["u"], r, prof["Vso"], 1)
    out = {"split_16O_1p_LDA_MeV": d0, "e_central_MeV": lv["e"], "ws_fit": lv["ws"], "ws_rms_MeV": lv["ws_rms"], "nodes": lv["nodes"]}
    if exact_dirac:
        e_up, _ = dirac_level(r, prof["S"], prof["V"], -2, 1)
        e_dn, _ = dirac_level(r, prof["S"], prof["V"], 1, 1)
        out.update(split_16O_1p_exact_dirac_MeV=e_dn - e_up, e_1p3_2_dirac_MeV=e_up, e_1p1_2_dirac_MeV=e_dn)
    S0, V0, _, _ = tab.at(np.array([0.16]))
    out["V_minus_S_at_0p16_MeV"] = float(V0[0] - S0[0])
    return out


def spin_orbit_compare(so):
    """Held-out comparison of the 16O 1p splitting with the declared window (the ONLY reader of SO_DECLARED's window)."""
    lo, hi = SO_DECLARED["window_MeV"]
    d0 = so["split_16O_1p_LDA_MeV"]
    return dict(so, passes_O=bool(lo <= d0 <= hi), window_MeV=[lo, hi], experiment_MeV=SO_DECLARED["experiment_MeV"],
                ratio_to_experiment=d0 / SO_DECLARED["experiment_MeV"])


# ============================================================================= standard RMF models (round-3 rmf_generic)
def rmf_fermi(k, m):
    """eps, P, rho_s, E_F for one species (degeneracy 2) of Fermi momentum k and mass m (MeV units)."""
    if k <= 0.0:
        return 0.0, 0.0, 0.0, m
    E = math.sqrt(k * k + m * m)
    if m <= 0.0:
        return k ** 4 / (4 * PI2), k ** 4 / (12 * PI2), 0.0, k
    a = math.asinh(k / m)
    m2 = m * m
    m4 = m2 * m2
    kE = k * E
    return (kE * (2 * k * k + m2) - m4 * a) / (8 * PI2), (kE * (2 * k * k - 3 * m2) + 3 * m4 * a) / (24 * PI2), m * (kE - m2 * a) / (2 * PI2), E


def rmf_fermi_v(k, m):
    m = np.asarray(m, float)
    if k <= 0.0:
        return np.zeros_like(m), np.zeros_like(m)
    mm = np.maximum(m, 1e-12)
    E = np.sqrt(k * k + mm * mm)
    a = np.arcsinh(k / mm)
    return (k * E * (2 * k * k + mm * mm) - mm ** 4 * a) / (8 * PI2), mm * (k * E - mm * mm * a) / (2 * PI2)


def rmf_kf(n):
    return (3.0 * PI2 * n) ** (1.0 / 3.0) if n > 0 else 0.0


def rmf_lepton(mu, m):
    if mu <= m:
        return 0.0, 0.0, 0.0
    k = math.sqrt(mu * mu - m * m)
    e, p, _, _ = rmf_fermi(k, m)
    return k ** 3 / (3 * PI2), e, p


class RMFBase:
    name = "?"

    def mavg(self):
        return 0.5 * (self.Mn + self.Mp)

    def _scalar_root(self, n_p, n_n, nb):
        """First root of dE_s/dPhi from Phi = 0 (normal branch)."""
        kp, kn = rmf_kf(n_p), rmf_kf(n_n)
        grid = np.linspace(0.0, min(self.Mn, self.Mp) * (1 - 1e-9), 97)
        f = self.dEs_dPhi_v(grid, kp, kn, nb)
        idx = np.nonzero((f[:-1] < 0) & (f[1:] >= 0))[0]
        if idx.size == 0:
            raise RuntimeError(f"{self.name}: no scalar root at n_p={n_p:.4g} n_n={n_n:.4g}")
        i = int(idx[0])
        return brentq(lambda Phi: self.dEs_dPhi(Phi, kp, kn, nb), grid[i], grid[i + 1], xtol=1e-13, rtol=1e-15, maxiter=200)

    def scalar_energy_v(self, Phi, n_p, n_n):
        nb = n_p + n_n
        kp, kn = rmf_kf(n_p), rmf_kf(n_n)
        ep, _ = rmf_fermi_v(kp, self.Mp - Phi)
        en, _ = rmf_fermi_v(kn, self.Mn - Phi)
        return ep + en + self.Us_v(Phi, nb)

    def snm(self, n_fm3):
        return self.state(0.5 * n_fm3 * HC3, 0.5 * n_fm3 * HC3)

    def EA(self, n_fm3, delta=0.0):
        n = n_fm3 * HC3
        s = self.state(0.5 * n * (1 - delta), 0.5 * n * (1 + delta))
        return s["eps"] / n - (self.Mn * (1 + delta) + self.Mp * (1 - delta)) / 2.0

    def esym(self, n_fm3):
        def D(h):
            return (self.EA(n_fm3, h) - 2 * self.EA(n_fm3, 0.0) + self.EA(n_fm3, -h)) / (h * h)
        return 0.5 * (4 * D(0.02) - D(0.04)) / 3.0

    def E_PNM(self, n_fm3):
        n = n_fm3 * HC3
        return self.state(0.0, n)["eps"] / n - self.Mn

    def P_snm(self, n_fm3):
        return self.snm(n_fm3)["P"] / HC3

    def saturation(self):
        n0 = brentq(self.P_snm, 0.09, 0.22, xtol=1e-14, rtol=1e-15)
        s0 = self.snm(n0)
        h = 2e-3
        P = self.P_snm
        dP = (-P(n0 + 2 * h) + 8 * P(n0 + h) - 8 * P(n0 - h) + P(n0 - 2 * h)) / (12 * h)
        J = self.esym(n0)
        dJ = (-self.esym(n0 + 2 * h) + 8 * self.esym(n0 + h) - 8 * self.esym(n0 - h) + self.esym(n0 - 2 * h)) / (12 * h)
        out = {"n0": n0, "EA": self.EA(n0), "K": 9.0 * dP, "mstar": (self.mavg() - s0["Phi"]) / self.mavg(), "J": J, "L": 3 * n0 * dJ,
               "S2_0p10": self.esym(0.10), "EPNM_0p10": self.E_PNM(0.10)}
        self.n0 = n0
        return out

    def beta_point(self, nb_fm3):
        nb = nb_fm3 * HC3

        def f(yp):
            s = self.state(nb * yp, nb * (1 - yp))
            mue = s["mu_n"] - s["mu_p"]
            ne = rmf_lepton(mue, FB.M_E)[0] if mue > 0 else 0.0
            nm = rmf_lepton(mue, FB.M_MU)[0] if mue > 0 else 0.0
            return nb * yp - ne - nm
        yp = brentq(f, 1e-9, 0.5, xtol=1e-15, rtol=1e-13, maxiter=200)
        s = self.state(nb * yp, nb * (1 - yp))
        mue = s["mu_n"] - s["mu_p"]
        ne, ee, pe = rmf_lepton(mue, FB.M_E)
        nm, em, pm = rmf_lepton(mue, FB.M_MU)
        eps = s["eps"] + ee + em
        P = s["P"] + pe + pm
        gibbs = nb * (1 - yp) * s["mu_n"] + nb * yp * s["mu_p"] + (ne + nm) * mue - eps
        return {"n_fm3": nb_fm3, "y_p": yp, "eps": eps / HC3, "P_cons": P / HC3, "gibbs_rel": (gibbs - P) / P if P != 0 else 0.0}

    def build_table(self, x_lo=0.1, x_hi=12.0, per_decade=250):
        n0 = self.n0
        npts = int(round(per_decade * math.log10(x_hi / x_lo)))
        xs = np.unique(np.concatenate([np.logspace(math.log10(x_lo), math.log10(x_hi), npts), [1.0, 2.0, 3.0]]))
        rows = [self.beta_point(x * n0) for x in xs]
        tab = {k: np.array([r[k] for r in rows], dtype=float) for k in ("n_fm3", "y_p", "eps", "P_cons", "gibbs_rel")}
        tab["x"] = xs
        return tab

    def snm_fields(self, ngrid):
        S, V = [], []
        for n in ngrid:
            if n <= 0:
                S.append(0.0); V.append(0.0)
                continue
            s = self.snm(float(n))
            S.append(-s["Phi"]); V.append(s["VLDA"])
        return np.array(S), np.array(V)

    def ground_state(self, dens_yp, EA0, n_phi=4001):
        """Global minimum over 0 <= M* <= M of the Phi-dependent energy at fixed (n_p, n_n)."""
        ok = True
        n_bad = 0
        n_other = 0
        min_gap = math.inf
        for (n_fm3, yp, lab) in dens_yp:
            n = n_fm3 * HC3
            n_p, n_n = n * yp, n * (1 - yp)
            s = self.state(n_p, n_n)
            Phi_d = s["Phi"]
            Mmin = min(self.Mn, self.Mp)
            grid = np.linspace(0.0, Mmin * (1 - 1e-9), n_phi)
            e = self.scalar_energy_v(grid, n_p, n_n)
            e_d = float(self.scalar_energy_v(np.array([Phi_d]), n_p, n_n)[0])
            loc = [i for i in range(1, n_phi - 1) if e[i] <= e[i - 1] and e[i] <= e[i + 1]]
            if e[-1] < e[-2]:
                loc.append(n_phi - 1)
            mins = []
            for i in loc:
                if 0 < i < n_phi - 1:
                    r = minimize_scalar(lambda p: float(self.scalar_energy_v(np.array([p]), n_p, n_n)[0]), bounds=(grid[i - 1], grid[i + 1]),
                                        method="bounded", options={"xatol": 1e-10})
                    mins.append((float(r.x), float(r.fun)))
                else:
                    mins.append((float(grid[i]), float(e[i])))
            others = [(p, ee) for (p, ee) in mins if abs(p - Phi_d) > 1e-3 * Mmin]
            e_glob = min([ee for (_, ee) in mins] + [e_d])
            design_global = bool(e_d - e_glob <= 1e-9 * abs(e_d) + 1e-6)
            if others:
                n_other += 1
                pa, ea = min(others, key=lambda t: t[1])
                if lab == "snm" and (s["eps"] + (ea - e_d)) / n - self.mavg() < EA0:
                    ok = False
                min_gap = min(min_gap, (ea - e_d) / n)
            if not design_global:
                ok = False
                n_bad += 1
        return {"valid": ok, "n_points": len(dens_yp), "n_design_not_global": n_bad, "n_points_with_other_minima": n_other,
                "min_gap_other_minus_design_per_baryon_MeV": (min_gap if math.isfinite(min_gap) else None)}


class NLModel(RMFBase):
    """sigma^3, sigma^4, omega^4 (zeta), omega^2 rho^2 (Lambda_v); rho with (g_rho/2) tau (FSU convention)."""
    kind = "NL"

    def __init__(self, name, Mn, Mp, A_S, kappa, lam, A_V, zeta, A_R, Lv):
        self.name, self.Mn, self.Mp = name, float(Mn), float(Mp)
        self.A_S, self.kappa, self.lam, self.A_V, self.zeta, self.A_R, self.Lv = map(float, (A_S, kappa, lam, A_V, zeta, A_R, Lv))

    def U(self, Phi):
        return self.A_S * Phi ** 2 / 2 + self.kappa * Phi ** 3 / 6 + self.lam * Phi ** 4 / 24

    def dU(self, Phi):
        return self.A_S * Phi + self.kappa * Phi ** 2 / 2 + self.lam * Phi ** 3 / 6

    def Us_v(self, Phi, nb):
        return self.U(Phi)

    def dEs_dPhi(self, Phi, kp, kn, nb):
        return self.dU(Phi) - rmf_fermi(kp, self.Mp - Phi)[2] - rmf_fermi(kn, self.Mn - Phi)[2]

    def dEs_dPhi_v(self, Phi, kp, kn, nb):
        return self.dU(Phi) - rmf_fermi_v(kp, self.Mp - Phi)[1] - rmf_fermi_v(kn, self.Mn - Phi)[1]

    def vector(self, nb, n3):
        """A_V W + zeta/6 W^3 + 2 Lv B^2 W = nb ; A_R B + 2 Lv W^2 B = n3."""
        if nb <= 0:
            return 0.0, 0.0
        if self.zeta == 0.0 and self.Lv == 0.0:
            return nb / self.A_V, n3 / self.A_R

        def h(W):
            B = n3 / (self.A_R + 2 * self.Lv * W * W)
            return self.A_V * W + self.zeta / 6 * W ** 3 + 2 * self.Lv * B * B * W - nb
        if self.zeta == 0.0 and n3 == 0.0:
            return nb / self.A_V, 0.0
        hi = nb / self.A_V
        if h(hi) < 0.0:
            hi *= 1.0 + 1e-10
        W = brentq(h, 0.0, hi, xtol=1e-13, rtol=1e-15, maxiter=200)
        return W, n3 / (self.A_R + 2 * self.Lv * W * W)

    def state(self, n_p, n_n):
        nb = n_p + n_n
        n3 = 0.5 * (n_p - n_n)
        Phi = self._scalar_root(n_p, n_n, nb) if nb > 0 else 0.0
        W, B = self.vector(nb, n3)
        kp, kn = rmf_kf(n_p), rmf_kf(n_n)
        ep, pp, rsp, Ep = rmf_fermi(kp, self.Mp - Phi)
        en, pn, rsn, En = rmf_fermi(kn, self.Mn - Phi)
        Uv = self.U(Phi)
        V = self.A_V * W * W / 2 + self.zeta * W ** 4 / 24 + self.A_R * B * B / 2 + self.Lv * W * W * B * B
        return {"eps": ep + en + Uv + W * nb + B * n3 - V, "P": pp + pn - Uv + V, "mu_p": Ep + W + 0.5 * B, "mu_n": En + W - 0.5 * B,
                "Phi": Phi, "W": W, "B": B, "VLDA": W, "SigmaR": 0.0}


class DDModel(RMFBase):
    """Typel-Wolter density-dependent couplings on the vector density, full-tau rho, rearrangement Sigma_R in mu and P."""
    kind = "DD"

    def __init__(self, name, p):
        self.name = name
        self.p = dict(p)
        self.Mn, self.Mp = float(p["Mn"]), float(p["Mp"])
        self.nref_nat = p["n_ref"] * HC3

    def coupling(self, which, nb):
        p = self.p
        x = nb / self.nref_nat
        if which == "r":
            g = p["gr0"] * math.exp(-p["a_r"] * (x - 1.0))
            return g, -p["a_r"] * g / self.nref_nat
        a, b, c, d = p[f"a_{which}"], p[f"b_{which}"], p[f"c_{which}"], p[f"d_{which}"]
        g0 = p[f"g{which}0"]
        u = x + d
        den = 1.0 + c * u * u
        return g0 * a * (1.0 + b * u * u) / den, g0 * 2.0 * a * u * (b - c) / (den * den) / self.nref_nat

    def Us_v(self, Phi, nb):
        gs, _ = self.coupling("s", nb)
        return 0.5 * self.p["m_s"] ** 2 / gs ** 2 * Phi ** 2

    def dEs_dPhi(self, Phi, kp, kn, nb):
        gs, _ = self.coupling("s", nb)
        return self.p["m_s"] ** 2 / gs ** 2 * Phi - rmf_fermi(kp, self.Mp - Phi)[2] - rmf_fermi(kn, self.Mn - Phi)[2]

    def dEs_dPhi_v(self, Phi, kp, kn, nb):
        gs, _ = self.coupling("s", nb)
        return self.p["m_s"] ** 2 / gs ** 2 * Phi - rmf_fermi_v(kp, self.Mp - Phi)[1] - rmf_fermi_v(kn, self.Mn - Phi)[1]

    def state(self, n_p, n_n):
        p = self.p
        nb = n_p + n_n
        n3 = n_p - n_n
        gs, dgs = self.coupling("s", nb)
        gw, dgw = self.coupling("w", nb)
        gr, dgr = self.coupling("r", nb)
        Phi = self._scalar_root(n_p, n_n, nb) if nb > 0 else 0.0
        kp, kn = rmf_kf(n_p), rmf_kf(n_n)
        ep, pp, rsp, Ep = rmf_fermi(kp, self.Mp - Phi)
        en, pn, rsn, En = rmf_fermi(kn, self.Mn - Phi)
        ms2, mw2, mr2 = p["m_s"] ** 2, p["m_w"] ** 2, p["m_r"] ** 2
        sigma = Phi / gs
        omega = gw * nb / mw2
        rho = gr * n3 / mr2
        SigmaR = dgw * omega * nb + dgr * rho * n3 - dgs * sigma * (rsp + rsn)
        Vw, Vr = gw * omega, gr * rho
        eps = ep + en + 0.5 * ms2 * sigma ** 2 + 0.5 * mw2 * omega ** 2 + 0.5 * mr2 * rho ** 2
        P = pp + pn - 0.5 * ms2 * sigma ** 2 + 0.5 * mw2 * omega ** 2 + 0.5 * mr2 * rho ** 2 + nb * SigmaR
        return {"eps": eps, "P": P, "mu_p": Ep + Vw + Vr + SigmaR, "mu_n": En + Vw - Vr + SigmaR, "Phi": Phi, "VLDA": Vw + SigmaR, "SigmaR": SigmaR}


RMF_MODELS = {
    "NL3": {"class": "NL", "ref": "Lalazissis, Koenig, Ring, PRC 55, 540 (1997)", "convention": "ring",
            "M": 939.0, "m_s": 508.194, "m_v": 782.501, "m_r": 763.0, "g_s": 10.217, "g_v": 12.868, "g_r": 4.474, "g2_fm": -10.431, "g3": -28.885, "c3": 0.0,
            "published": {"n0": 0.148, "EA": -16.24, "K": 271.76, "mstar": 0.60, "J": 37.4, "L": 118.0}},
    "FSUGold": {"class": "NL", "ref": "Todd-Rutel, Piekarewicz, PRL 95, 122501 (2005)", "convention": "fsu",
                "M": 939.0, "m_s": 491.5, "m_v": 782.5, "m_r": 763.0, "gs2": 112.1996, "gv2": 204.5469, "gr2": 138.4701,
                "kappa": 1.4203, "lambda": 0.023762, "zeta": 0.06, "Lv": 0.030,
                "published": {"n0": 0.148, "EA": -16.30, "K": 230.0, "mstar": 0.61, "J": 32.59, "L": 60.5}},
    "DD-ME2": {"class": "DD", "ref": "Lalazissis, Niksic, Vretenar, Ring, PRC 71, 024312 (2005)",
               "Mn": 939.0, "Mp": 939.0, "m_s": 550.1238, "m_w": 783.0, "m_r": 763.0, "gs0": 10.5396, "gw0": 13.0189, "gr0": 3.6836,
               "a_s": 1.3881, "b_s": 1.0943, "c_s": 1.7057, "d_s": 0.4421, "a_w": 1.3892, "b_w": 0.9240, "c_w": 1.4620, "d_w": 0.4775, "a_r": 0.5647, "n_ref": 0.152,
               "published": {"n0": 0.152, "EA": -16.14, "K": 250.89, "mstar": 0.572, "J": 32.3, "L": 51.2}},
    "TW99": {"class": "DD", "ref": "Typel, Wolter, Nucl. Phys. A 656, 331 (1999)",
             "Mn": 939.0, "Mp": 939.0, "m_s": 550.0, "m_w": 783.0, "m_r": 763.0, "gs0": 10.72854, "gw0": 13.29015, "gr0": 7.32196 / 2.0,
             "a_s": 1.365469, "b_s": 0.226061, "c_s": 0.409704, "d_s": 0.901995, "a_w": 1.402488, "b_w": 0.172577, "c_w": 0.344293, "d_w": 0.983955,
             "a_r": 0.515, "n_ref": 0.153,
             "published": {"n0": 0.153, "EA": -16.247, "K": 240.26, "mstar": 0.555, "J": 32.767, "L": 55.31},
             "note": "Gamma_rho = 7.32196 is quoted in the tau/2 convention (full-tau gr0 = 7.32196/2)"},
}
RMF_SOURCE = "parameter records of the session round-3 survey (rmf_survey/declared.py), validated to 1 percent against the published saturation properties"


def build_rmf(name, rec=None):
    rec = RMF_MODELS[name] if rec is None else rec
    if rec["class"] == "DD":
        return DDModel(name, rec)
    if rec["convention"] == "ring":
        gs, gv, gr = rec["g_s"], rec["g_v"], rec["g_r"]
        return NLModel(name, rec["M"], rec["M"], rec["m_s"] ** 2 / gs ** 2, -2.0 * rec["g2_fm"] * HBARC / gs ** 3, 6.0 * rec["g3"] / gs ** 4,
                       rec["m_v"] ** 2 / gv ** 2, 6.0 * rec["c3"] / gv ** 4, rec["m_r"] ** 2 / (4.0 * gr ** 2), 0.0)
    if rec["convention"] == "fsu":
        return NLModel(name, rec["M"], rec["M"], rec["m_s"] ** 2 / rec["gs2"], rec["kappa"], rec["lambda"], rec["m_v"] ** 2 / rec["gv2"],
                       rec["zeta"], rec["m_r"] ** 2 / rec["gr2"], rec["Lv"])
    raise ValueError(rec["convention"])


# ============================================================================= NVG members (forward chain of the forward-band producer)
NVG_MEMBERS = {
    "NVG_island_y0.84_a5min": {"ystar": 0.84, "n0": 0.16, "EA": -16.0, "K": 240.0, "Q0": -107.77789456941183, "J": 32.0, "L": 50.0,
                               "scalar": "quintic", "form": "power",
                               "note": "quintic stability-island member at a5_min (a5 = 1.4664e9, reached through Q0, a5 linear in Q0), centre windows; "
                                       "session closure/gamma_minus_one.py Q0_A5MIN; marginal by construction (q_min ~ -4e-4 = roundoff of a double root), "
                                       "evaluated as the forced quintic"},
    "NVG_quartic_y0.87": {"ystar": 0.87, "n0": 0.16, "EA": -16.0, "K": 240.0, "Q0": None, "J": 32.0, "L": 50.0, "scalar": "quartic", "form": "power",
                          "note": "the author's forward design point (quartic, a5 = 0, globally stable), centre windows"},
}
VACUUM_REL_TOL = 1e-12


def third_order(jet):
    y = jet["ystar"]; n = jet["n0_nat"]; k = jet["k"]; ef = jet["ef"]; Cv = jet["Cv"]; m = jet["m"]; ns = jet["ns"]
    B, C, K = jet["B"], jet["C"], jet["K"]
    z = y * y - 1.0
    e_nny = -k * k * MN ** 2 * y / (3.0 * n * ef ** 3) - 2.0 * Cv / y ** 3
    e_nyy = MN ** 2 * k * k / ef ** 3 + 6.0 * Cv * n / y ** 4
    ns_mm = 6.0 * ns / m ** 2 - 9.0 * n / (m * ef) + 3.0 * n * m / ef ** 3
    yp = -B / C
    Uyyy = sum(c * FB._jet_rows(j, y, z)[3] for c, j in zip(jet["coef"], range(2, 2 + len(jet["coef"]))))
    e_yyy = Uyyy + MN ** 3 * ns_mm - 12.0 * Cv * n * n / y ** 5
    ypp = -(e_nny + 2.0 * e_nyy * yp + e_yyy * yp ** 2) / C
    return {"yp": yp, "ypp": ypp}


def kin_sym(jet):
    T, u, n, y = jet["T"], jet["u"], jet["n0_nat"], jet["ystar"]
    t3 = third_order(jet)
    yp, ypp = t3["yp"], t3["ypp"]
    T_y = -T * (1.0 - u) / y
    T_nn = T * (3.0 * u * u - 3.0 * u - 2.0) / (9.0 * n * n)
    T_ny = -T * (1.0 - u) * (2.0 - 3.0 * u) / (3.0 * n * y)
    T_yy = T * (1.0 - u) * (2.0 - 3.0 * u) / (y * y)
    return {"K_sym_kin": 9.0 * n * n * (T_nn + 2.0 * T_ny * yp + T_yy * yp ** 2 + T_y * ypp)}


class Matter:
    """Exact-density NVG matter states on FB.BetaSolver (same residuals as the chain); session closure/corner_nuclear_heldout.py."""

    def __init__(self, jet, iso):
        self.jet, self.iso = jet, iso
        self.C_of_x, self.dln, _ = FB.make_C_of_x(iso)
        self.n0 = jet["n0_nat"]
        self.bs = FB.BetaSolver(jet["coef"], jet["Cv"], self.n0, self.C_of_x)
        self.gamma = iso["gamma"]

    def _newton_y(self, n, d, y0):
        C = self.C_of_x(n / self.n0)
        y = y0
        for _ in range(300):
            F1, F2, (J11, J12, J21, J22), st = self.bs.residuals(n, C, y, d)
            if abs(F1) < 1e-13 * MN * n:
                break
            step = -F1 / J11
            if abs(step) > 0.2 * y:
                step = math.copysign(0.2 * y, step)
            y = max(0.02, y + step)
        F1, F2, Jm, st = self.bs.residuals(n, C, y, d)
        st["F1_rel"] = F1 / (MN * n); st["C"] = C; st["n"] = n
        return st

    def fixed_delta(self, n_target, d, nsteps=80):
        """Continuation from (n0, delta = 0, y*): delta 0 -> d at n0, then n0 -> n_target at fixed delta."""
        y = self.jet["ystar"]
        for dd in (np.linspace(0.0, d, 21)[1:] if d > 0 else []):
            y = self._newton_y(self.n0, float(dd), y)["y"]
        st = self._newton_y(self.n0, d, y)
        if n_target != self.n0:
            for n in np.linspace(self.n0, n_target, nsteps + 1)[1:]:
                st = self._newton_y(float(n), d, st["y"])
        gm = self.bs.global_min_check(st["n"], st["C"], d, st["y"], st["eps_N"])
        st["global_min_ok"] = bool(gm[0]); st["y_global_min"] = gm[1]; st["dE_global"] = gm[2]
        st["Cy_positive"] = bool(st["Cy"] > 0.0)
        return st

    def pnm(self, n):
        st = self.fixed_delta(n, 1.0)
        eps_rho = st["C"] * n * n / 8.0
        return {"EN": (st["eps_N"] + eps_rho) / n - MN, "y": st["y"], "Cy_positive": st["Cy_positive"], "global_min_ok": st["global_min_ok"]}

    def S2(self, n, y_sym):
        k = (6.0 * PI2 * n / 4.0) ** (1.0 / 3.0)
        ef = math.sqrt(k * k + (MN * y_sym) ** 2)
        return k * k / (6.0 * ef) + self.C_of_x(n / self.n0) * n / 8.0


class NVGFields:
    """NVG symmetric-matter self-energies S = M (y - 1), V = C_v n / y^2 on the design branch y(n) continued from (n0, y*)."""

    def __init__(self, name, jet):
        self.name = name
        self.coef = [float(c) for c in jet["coef"]]
        self.Cv = float(jet["Cv"])
        self.ystar, self.n0 = float(jet["ystar"]), float(jet["n0_fm3"])
        self.bs = FB.BetaSolver(self.coef, self.Cv, self.n0 * HC3, lambda x: 0.0)

    def _newton(self, n_nat, y):
        for _ in range(200):
            F1, F2, (J11, J12, J21, J22), st = self.bs.residuals(n_nat, 0.0, y, 0.0)
            if abs(F1) < 1e-12 * MN * max(n_nat, 1e-30) or n_nat == 0.0 and abs(F1) < 1e-6:
                break
            step = -F1 / J11
            if abs(step) > 0.05 * y:
                step = math.copysign(0.05 * y, step)
            y = y + step
        F1, F2, Jm, st = self.bs.residuals(n_nat, 0.0, y, 0.0)
        return y, st

    def table(self, ngrid=N_GRID):
        y0, _ = self._newton(self.n0 * HC3, self.ystar)
        down = np.linspace(self.n0, 0.0, 801)
        up = np.linspace(self.n0, ngrid[-1], 401)
        y, ydn = y0, []
        for n in down:
            y, _ = self._newton(n * HC3, y)
            ydn.append(y)
        y, yup = y0, []
        for n in up:
            y, _ = self._newton(n * HC3, y)
            yup.append(y)
        sp = CubicSpline(np.concatenate([down[::-1], up[1:]]), np.concatenate([np.array(ydn)[::-1], np.array(yup)[1:]]))
        ys = np.empty_like(ngrid)
        okm = np.ones_like(ngrid, dtype=bool)
        for i, n in enumerate(ngrid):
            yy, st = self._newton(n * HC3, float(sp(n)))
            ys[i] = yy
            if n > 0:
                okm[i] = bool(self.bs.global_min_check(n * HC3, 0.0, 0.0, yy, st["eps_N"])[0])
        S = MN * (ys - 1.0)
        V = self.Cv * (ngrid * HC3) / ys ** 2
        return FieldTable(self.name, ngrid, S, V), {"y_at_n0": float(y0), "branch_global_min_ok_to_0.20": bool(np.all(okm[ngrid <= 0.20])),
                                                    "branch_max_abs_dy_per_step": float(np.max(np.abs(np.diff(ys))))}


# ============================================================================= public entry point
def joint_verdict(rec, ground_state_valid=None, vacuum_ok=None, vacuum_applicable=True):
    """Combine S, O, G, V; None = not decidable from the inputs supplied."""
    lik = rec.get("likelihood") or {}
    S = lik.get("primary_rotation", {}).get("passes_S") if lik.get("declared_8_term_statistic") else None
    O = (rec.get("spin_orbit") or {}).get("passes_O")
    V = vacuum_ok if vacuum_applicable else "not_applicable"
    pieces = {"S": S, "O": O, "G": ground_state_valid, "V": V}
    vals = [v for v in pieces.values() if v != "not_applicable"]
    conv = False if any(v is False for v in vals) else (None if any(v is None for v in vals) else True)
    return {**pieces, "converges": conv, "rule": JOINT_VERDICT_DECLARED["converges"]}


def evaluate_table(n_fm3, eps, P, S2_0p10=None, EPNM_0p10=None, fields=None, n0_fm3=0.16, rotation=True, settings=None,
                   ground_state_valid=None, vacuum_ok=None, vacuum_applicable=True, label="user_table"):
    """Evaluate ANY cold beta-equilibrium core table through the declared held-out pipeline.

    n_fm3 [fm^-3], eps, P [MeV/fm^3]: the core EOS (thermodynamically consistent pressure; it must reach below the
    SLy crust, i.e. its eps(P) must cross eps_SLy(P) for P < 3 MeV/fm^3 on the monotone tail, and extend above the
    central pressure of the maximum mass).  S2_0p10, EPNM_0p10 [MeV]: S_2 and E_PNM/N at 0.10 fm^-3 (terms 7-8; without
    them chi2 runs over the 6 stellar terms and S is undecided).  fields = (n_grid [fm^-3], S [MeV], V [MeV]):
    symmetric-matter scalar / time-like vector self-energies for the 16O LDA (grid from 0 to >= 0.2 fm^-3).
    n0_fm3 only labels x = n/n0.  ground_state_valid / vacuum_ok: the caller's G / V results (cannot be decided from a table).
    Returns stars (static + Hartle-Thorne), likelihood (chi2, per-term z, p), the boxes, 16O and the joint verdict."""
    t0 = time.time()
    n = np.asarray(n_fm3, float); e = np.asarray(eps, float); p = np.asarray(P, float)
    rec = {"label": label, "status": None, "n_points": int(n.size)}
    if not (n.ndim == e.ndim == p.ndim == 1 and n.size == e.size == p.size and n.size >= 8):
        raise ValueError("n_fm3, eps, P must be 1-D arrays of equal length >= 8")
    if not (np.all(np.isfinite(n)) and np.all(np.isfinite(e)) and np.all(np.isfinite(p))):
        raise ValueError("non-finite table entries")
    order = np.argsort(n)
    n, e, p = n[order], e[order], p[order]
    if np.any(np.diff(n) <= 0) or np.any(e <= 0) or np.any(n <= 0):
        raise ValueError("n must be strictly increasing and positive; eps > 0")
    tab = {"n_fm3": n, "eps": e, "P_cons": p, "P_author": p, "x": n / float(n0_fm3)}
    try:
        eos, info = FB.build_eos(tab, crust="sly", match="eps", pressure="consistent")
    except Exception as exc:  # noqa: BLE001
        rec["status"] = f"crust_match_failed: {exc!r}"
        rec["wall_s"] = time.time() - t0
        return rec
    rec["crust_match"] = {k: info.get(k) for k in ("method", "P_m", "x_m", "n_m_fm3", "eps_jump_rel", "x_tail_start", "P_tail_start")}
    rec["crust"] = "SLy (Read+2009 Table II) eps-continuity match from verification/nvg_forward_band_propagation_audit.py"
    stars = star_block(eos, settings=settings, rotation=rotation)
    rec["stars"] = stars
    if stars["status"] != "ok":
        rec["status"] = "no_branch"
        rec["wall_s"] = time.time() - t0
        return rec
    rec["causal_stellar_branch"] = bool(stars["max_cs2_stellar_branch"] is not None and stars["max_cs2_stellar_branch"] < 1.0)
    nuc = {"S2_0p10": None if S2_0p10 is None else float(S2_0p10), "EPNM_0p10": None if EPNM_0p10 is None else float(EPNM_0p10)}
    rec["nuclear"] = nuc
    rec["likelihood"] = likelihood_block(stars, nuc)
    if fields is not None:
        ng, Sg, Vg = (np.asarray(a, float) for a in fields)
        rec["spin_orbit"] = spin_orbit_compare(spin_orbit_16O(FieldTable(label, ng, Sg, Vg)))
    rec["joint_verdict"] = joint_verdict(rec, ground_state_valid, vacuum_ok, vacuum_applicable)
    rec["status"] = "ok"
    rec["wall_s"] = time.time() - t0
    return rec


# ============================================================================= built-in models
def evaluate_rmf(name, rotation=True):
    t0 = time.time()
    rec0 = RMF_MODELS[name]
    m = build_rmf(name)
    sat = m.saturation()
    comp = {k: {"published": rec0["published"][k], "this_work": sat[k], "rel_dev_percent": 100.0 * (sat[k] / rec0["published"][k] - 1.0)}
            for k in ("n0", "EA", "K", "mstar", "J", "L")}
    tab = m.build_table(0.1, 12.0, 250)
    Sg, Vg = m.snm_fields(N_GRID)
    res = evaluate_table(tab["n_fm3"], tab["eps"], tab["P_cons"], S2_0p10=sat["S2_0p10"], EPNM_0p10=sat["EPNM_0p10"],
                         fields=(N_GRID, Sg, Vg), n0_fm3=sat["n0"], rotation=rotation, vacuum_applicable=False, label=name)
    res["model"] = {"class": rec0["class"], "ref": rec0["ref"], "source": RMF_SOURCE, "saturation": sat, "vs_published": comp,
                    "validated_1pct": bool(all(abs(v["rel_dev_percent"]) < 1.0 for v in comp.values())),
                    "table": {"x_lo": 0.1, "x_hi": 12.0, "per_decade": 250, "n_points": int(len(tab["x"])),
                              "max_gibbs_rel": float(np.max(np.abs(tab["gibbs_rel"])))}}
    if res["status"] == "ok":
        nc = res["stars"]["n_c_Mmax_fm3"]
        dens = np.logspace(math.log10(0.02 * sat["n0"]), math.log10(nc), 120)
        pts = [(float(x), 0.5, "snm") for x in dens] + [(float(x), m.beta_point(float(x))["y_p"], "beta") for x in dens[::2]]
        gs = m.ground_state(pts, sat["EA"])
        res["ground_state"] = gs
        res["joint_verdict"] = joint_verdict(res, bool(gs["valid"]), None, vacuum_applicable=False)
    res["wall_s"] = time.time() - t0
    return res


def evaluate_nvg(name, rotation=True):
    t0 = time.time()
    p = NVG_MEMBERS[name]
    sm = FB.scalar_model(p, p["scalar"])
    jet = sm["jet"]
    if jet is None:
        return {"label": name, "status": f"no_scalar_model:{sm['status']}"}
    coef = jet["coef"]
    qm = jet["q_min"]
    tol = VACUUM_REL_TOL * max(abs(c) for c in coef)
    vac = {"q_min": qm, "z_at_q_min": jet["z_at_q_min"], "strict_globally_stable": bool(jet["globally_stable"]),
           "tolerance": tol, "vacuum_ok": bool(qm is not None and qm >= -tol), "coef": coef}
    iso = FB.isovector(jet, p["J"], p["L"], p["form"])
    tab = FB.build_beta_table(jet, iso, FB.x_grid(SETTINGS), rearrangement=True, global_check=True)
    Mt = Matter(jet, iso)
    n10 = 0.10 * HC3
    snm10 = Mt.fixed_delta(n10, 0.0)
    pnm10 = Mt.pnm(n10)
    gs_rows = {}
    for x in GS_X_SNM:
        st = Mt.fixed_delta(jet["n0_nat"] * x, 0.0)
        gs_rows[str(x)] = {"ok": bool(st["global_min_ok"] and st["Cy_positive"]), "y": st["y"], "y_global_min": st["y_global_min"]}
    nuc = {"S2_0p10": Mt.S2(n10, snm10["y"]), "EPNM_0p10": pnm10["EN"],
           "K_sym": kin_sym(jet)["K_sym_kin"] + 9.0 * (p["J"] - jet["T"]) * iso["gamma"] * (1.0 + iso["gamma"])}
    ft, so_diag = NVGFields(name, jet).table()
    res = evaluate_table(tab["n_fm3"], tab["eps"], tab["P_cons"], S2_0p10=nuc["S2_0p10"], EPNM_0p10=nuc["EPNM_0p10"],
                         fields=(ft.n, ft.S, ft.V), n0_fm3=p["n0"], rotation=rotation, vacuum_ok=vac["vacuum_ok"], label=name)
    res["model"] = {"class": "NVG", "params": {k: p[k] for k in ("ystar", "n0", "EA", "K", "Q0", "J", "L", "scalar", "form")}, "note": p["note"],
                    "scalar_status": sm["status"], "a5": sm.get("a5"), "gamma": iso["gamma"], "C0": iso["C0"], "Cv": jet["Cv"],
                    "K_sym": nuc["K_sym"], "vacuum": vac, "spin_orbit_branch": so_diag,
                    "chain": "verification/nvg_forward_band_propagation_audit.py scalar_model / isovector / build_beta_table (x 0.1-15, 220 nodes)"}
    if res["status"] == "ok":
        xc = res["stars"]["n_c_Mmax_fm3"] / p["n0"] if res["stars"]["n_c_Mmax_fm3"] else float(tab["x"][-1])
        X = tab["x"]
        mask = (X >= res["crust_match"]["x_m"] - 1e-12) & (X <= xc + 1e-12)
        gs_table_ok = bool(tab["global_min_ok"][mask].all())
        gs = {"table_branch_global_min_between_match_and_xc": gs_table_ok, "x_c_Mmax": xc, "snm_rows": gs_rows,
              "snm_ok": all(v["ok"] for v in gs_rows.values()),
              "snm_0p10_ok": bool(snm10["global_min_ok"] and snm10["Cy_positive"]), "pnm_0p10_ok": bool(pnm10["global_min_ok"] and pnm10["Cy_positive"])}
        gs["valid"] = bool(gs_table_ok and gs["snm_ok"] and gs["snm_0p10_ok"] and gs["pnm_0p10_ok"])
        res["ground_state"] = gs
        res["joint_verdict"] = joint_verdict(res, gs["valid"], vac["vacuum_ok"], vacuum_applicable=True)
    res["wall_s"] = time.time() - t0
    return res


MODEL_ORDER = ("NL3", "FSUGold", "DD-ME2", "TW99", "NVG_island_y0.84_a5min", "NVG_quartic_y0.87")


def evaluate_model(name, rotation=True):
    return evaluate_rmf(name, rotation) if name in RMF_MODELS else evaluate_nvg(name, rotation)


# session values of the same pipeline (regression references; scratch records, not inputs)
SESSION_REFERENCE = {
    "NL3": {"chi2": 69.45950739651946, "z_PIT_R144": -4.887880393327893, "split_16O_1p": 6.841374547476225, "M_max": 2.7744148964166535},
    "FSUGold": {"chi2": 41.64060493048237, "z_PIT_R144": -2.2311941214409807, "split_16O_1p": 5.758318946963838, "M_max": 1.7219580014222216},
    "DD-ME2": {"chi2": 19.95236665994761, "z_PIT_R144": -3.166007658426013, "split_16O_1p": 6.513341086685218, "M_max": 2.483009453710738},
    "TW99": {"chi2": 11.149989233862776, "z_PIT_R144": -2.137872792241603, "split_16O_1p": 6.991834160421547, "M_max": 2.0758753455671597},
    "NVG_island_y0.84_a5min": {"split_16O_1p": 1.4496873118890201},
    "NVG_quartic_y0.87": {"split_16O_1p": 1.039599226957856},
    "_source": "session scratchpad round3/rmf_survey/models.jsonl (RMF: same EOS code, crust, stars, rotation, likelihood, LDA) and "
               "creative/spin_orbit/result.json lda_raw (NVG 16O); earlier session codes gave DD-ME2 chi2 19.954, NL3 69.43, FSUGold 41.70 "
               "(creative/likelihood/analysis.json)",
}


def summary_row(name, r):
    lk = (r.get("likelihood") or {}).get("primary_rotation") or {}
    st = r.get("stars") or {}
    so = r.get("spin_orbit") or {}
    jv = r.get("joint_verdict") or {}
    return {"model": name, "status": r.get("status"), "M_max": st.get("M_max"), "R_144_static": (st.get("R_of_M") or {}).get("1.44"),
            "R_eq_144_rot": (lk.get("predictions") or {}).get("R144"), "chi2": lk.get("chi2"), "nu": lk.get("nu"), "p_chi2_nu": lk.get("p_chi2_nu"),
            "p_replicated_MC": lk.get("p_replicated_MC"), "z_PIT_J0614": (lk.get("z_PIT") or {}).get("R144"),
            "z_simple_J0614": (lk.get("z_simple") or {}).get("R144"), "max_abs_z_PIT": lk.get("max_abs_z_PIT"),
            "box2": (lk.get("box") or {}).get("box2"), "box_fails2": (lk.get("box") or {}).get("fails2"),
            "split_16O_1p_LDA": so.get("split_16O_1p_LDA_MeV"), "split_16O_1p_exact_dirac": so.get("split_16O_1p_exact_dirac_MeV"),
            "S": jv.get("S"), "O": jv.get("O"), "G": jv.get("G"), "V": jv.get("V"), "converges": jv.get("converges"),
            "chi2_no_J0614": ((r.get("likelihood") or {}).get("sensitivity_no_J0614") or {}).get("chi2"),
            "p_no_J0614": ((r.get("likelihood") or {}).get("sensitivity_no_J0614") or {}).get("p_chi2_nu"),
            "chi2_static": ((r.get("likelihood") or {}).get("secondary_static") or {}).get("chi2")}


def regression_block(models):
    out = {}
    for name, ref in SESSION_REFERENCE.items():
        if name.startswith("_") or name not in models:
            continue
        row = summary_row(name, models[name])
        got = {"chi2": row["chi2"], "z_PIT_R144": row["z_PIT_J0614"], "split_16O_1p": row["split_16O_1p_LDA"], "M_max": row["M_max"]}
        out[name] = {k: {"session": v, "this_run": got[k], "abs_diff": (abs(got[k] - v) if got[k] is not None else None)} for k, v in ref.items()}
    out["_source"] = SESSION_REFERENCE["_source"]
    return out


def declared_block():
    return {"prereg": PREREG, "prereg_sha256_declared": PREREG_SHA256_DECLARED, "data_source_notes": DATA_SOURCE_NOTES,
            "spin_orbit": SO_DECLARED, "joint_verdict": JOINT_VERDICT_DECLARED, "rmf_models": RMF_MODELS, "nvg_members": NVG_MEMBERS,
            "settings": {"stellar": SETTINGS, "p_surf": P_SURF, "masses": list(MASSES), "pulsar_evaluation_configs": PULSARS,
                         "vacuum_rel_tol": VACUUM_REL_TOL, "rng_seed": 7, "ndraw": 200000}}


def build_result(models=MODEL_ORDER, rotation=True):
    t0 = time.time()
    recs = {}
    for name in models:
        recs[name] = evaluate_model(name, rotation)
    so_rule = None
    if "NL3" in recs and "DD-ME2" in recs:
        a = recs["NL3"]["spin_orbit"]["split_16O_1p_LDA_MeV"]
        b = recs["DD-ME2"]["spin_orbit"]["split_16O_1p_LDA_MeV"]
        so_rule = {"NL3": a, "DD-ME2": b, "band": [6.18 * 0.8, 6.18 * 1.2], "both_in_band": bool(4.944 <= a <= 7.416 and 4.944 <= b <= 7.416),
                   "consequence": "raw LDA numbers used for every model (no scale factor)"}
    t1 = time.time()
    validation = {"prereg_sha256_recomputed": prereg_sha256(), "prereg_sha256_matches": prereg_sha256() == PREREG_SHA256_DECLARED,
                  "hartle_thorne_maclaurin_limit": hartle_maclaurin_validation(), "spin_orbit_validation_rule": so_rule,
                  "regression_vs_session": regression_block(recs)}
    validation["wall_s"] = time.time() - t1
    table = [summary_row(n, recs[n]) for n in models]
    interp = {
        "summary": "Every built-in model fails the declared joint verdict. Standard RMF sets pass 16O and the ground state but fail the stars "
                   "(J0614 radius |z_PIT| > 2 for all four); the NVG members fail 16O (m*/M >= 0.84 gives 1.0-1.5 MeV against [4.9, 7.4]). "
                   "See model_table for chi2, p, J0614 z and 16O of each model.",
        "practical_use": "evaluate_table(n_fm3, eps, P, S2_0p10=..., EPNM_0p10=..., fields=(n, S, V)) puts any beta-equilibrium EOS through the "
                         "same crust, stars, rotation, likelihood and 16O procedure; CLI --table FILE does the same from a 3-column file.",
        "evidence": "evidence weight 0.0: fixed published or declared models compared with held-out data; no parameter is fitted here",
    }
    res = {
        "schema_version": SCHEMA, "status": STATUS, "evidence_weight": EVIDENCE_WEIGHT,
        "source_sha256": hashlib.sha256(SOURCE_PATH.read_bytes()).hexdigest(),
        "command": "python verification/nvg_heldout_joint_likelihood_audit.py" + ("" if rotation else " --no-rotation"),
        "declared": declared_block(),
        "model_table": table,
        "models": recs,
        "validation": validation,
        "interpretation": interp,
        "wall_s": time.time() - t0,
    }
    return FB._clean(res)


def validate_result(result):
    """Fail-closed structural check plus a cheap recompute: chi2, z and boxes from the stored predictions."""
    try:
        if result["schema_version"] != SCHEMA or result["status"] != STATUS or result["evidence_weight"] != EVIDENCE_WEIGHT:
            return False
        if result["source_sha256"] != hashlib.sha256(SOURCE_PATH.read_bytes()).hexdigest():
            return False
        if FB._clean(declared_block()) != result["declared"]:
            return False
        if prereg_sha256() != PREREG_SHA256_DECLARED or not result["validation"]["prereg_sha256_matches"]:
            return False
        inv = {v: k for k, v in SHORT.items()}
        for name, rec in result["models"].items():
            if rec.get("status") != "ok":
                continue
            for cname in ("primary_rotation", "sensitivity_no_J0614", "secondary_static"):
                s = rec["likelihood"][cname]
                p = {t: None for t in TERMS}
                p.update({inv[k]: v for k, v in s["predictions"].items()})
                c, z = chi2_terms(p)
                terms = [inv[k] for k in s["terms"]]
                chi = sum(c[t] for t in terms)
                if s["chi2"] is None:
                    if math.isfinite(chi):
                        return False
                elif not math.isclose(chi, s["chi2"], rel_tol=1e-12, abs_tol=1e-12):
                    return False
                if FB._clean(box(p, z, terms)) != s["box"]:
                    return False
        return True
    except Exception:  # noqa: BLE001
        return False


def read_table_file(path):
    """3-column table n [fm^-3], eps [MeV/fm^3], P [MeV/fm^3] (whitespace or comma separated, '#' comments) or JSON
    {"n_fm3": [...], "eps": [...], "P": [...]}."""
    text = Path(path).read_text(encoding="utf-8")
    if text.lstrip().startswith("{"):
        d = json.loads(text)
        return np.asarray(d["n_fm3"], float), np.asarray(d["eps"], float), np.asarray(d["P"], float)
    rows = []
    for line in text.splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            rows.append([float(v) for v in line.replace(",", " ").split()[:3]])
    a = np.asarray(rows, float)
    return a[:, 0], a[:, 1], a[:, 2]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--write", action="store_true", help="write the strict-JSON artifact to verification/")
    p.add_argument("--models", nargs="*", default=None, help=f"subset of the built-in models {list(MODEL_ORDER)}")
    p.add_argument("--no-rotation", action="store_true", help="static radii only (declared secondary configuration)")
    p.add_argument("--table", default=None, help="evaluate a user beta-equilibrium core table (n eps P) instead of the built-in models")
    p.add_argument("--S2", type=float, default=None, help="S_2(0.10 fm^-3) [MeV] of the user model (term 7)")
    p.add_argument("--EPNM", type=float, default=None, help="E_PNM/N(0.10 fm^-3) [MeV] of the user model (term 8)")
    p.add_argument("--n0", type=float, default=0.16, help="saturation density used only to label x = n/n0")
    args = p.parse_args(argv)
    if args.table:
        n, e, P = read_table_file(args.table)
        res = FB._clean(evaluate_table(n, e, P, S2_0p10=args.S2, EPNM_0p10=args.EPNM, n0_fm3=args.n0, rotation=not args.no_rotation,
                                       label=Path(args.table).name))
        sys.stdout.write(json.dumps(res, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
        return 0
    models = tuple(args.models) if args.models else MODEL_ORDER
    for m in models:
        if m not in MODEL_ORDER:
            p.error(f"unknown model {m}")
    res = build_result(models, rotation=not args.no_rotation)
    text = json.dumps(res, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    if args.write:
        if models != MODEL_ORDER or args.no_rotation:
            p.error("--write stores the declared default run only (all built-in models, rotation on)")
        RESULT_PATH.write_text(text, encoding="utf-8")
    sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
