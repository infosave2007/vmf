# Round 4 brief: anchors, residuals and the missing-element inference (read fully; also read SCR/round3/COMMON.md for environment, tools and standards; write only under SCR/round4/<your-subdir>/)

SCR = SCR

## Anchors: what NVG reproduces without fitted coefficients (session results)
- A1 Sub-saturation nuclear matter: the y*=0.84 island member (a5_min, centre windows) and the y*=0.87 quartic give S_2(0.10 fm^-3) = 25.3 / 25.2 MeV vs 25.5 +- 1.0 and E_PNM/N(0.10) = 11.84 / 11.76 vs 11.4 +- 1.0 (SCR/closure/corner_nuclear_heldout.json). K_sym -41 / -61, Q0 inside windows.
- A2 Stars under the pre-registered joint likelihood: the NVG class fits better than every published RMF set (18 validated); its only failure is the leave-one-out tension J0740 mass (z +2.4..+2.7) vs J0614 radius (z -2.1..-2.4) (SCR/creative/likelihood/, SCR/round3/rmf_survey/).
- A3 Trace-anomaly potentials with no free scalar coefficient (SCR/round3/derived_potentials/) have better evidence than the polynomial class (ln Z -10.2 vs -14.2). The dilaton W^4 ln W form gives a 16O 1p splitting of 4.26-4.86 MeV (exact Dirac), 0.04 MeV below the window edge 4.9, but K about 1590 MeV.
- A4 Scale anchors. Surface energy data independently require a scalar gradient scale W0_eff = 183-258 MeV; the author's finite-nucleus fit gave 194.8 MeV (SCR/creative/surface_crust/). Binding of homogeneous matter in the one-field action needs W0 about 98 MeV, close to f_pi = 92.1. Every saturating derived potential predicts a vacuum scalar mass 86-101 MeV.

## Residuals: what fails, expressed as the deformation it requires
- R1 Isovector: the action alone gives S(n0) = 13.4 MeV (kinetic only) vs J = 31.7 +- 3.2; it needs about +18 MeV of isovector potential energy at n0 with L in 35-65 MeV. The U(2) rho partner gives only 21.5-27 MeV (SCR/missing/gauge_isovector/).
- R2 Spin-orbit: 16O 1p = 1.45 MeV vs 6.18 (window 4.9-7.4). It needs V - S at n0 of about 650-730 MeV (NVG has 77-259) or an equivalent spin-orbit source. The splitting tracks m*/M almost alone: about 19.83 - 22.62 m*/M MeV across 17 RMF sets.
- R3 Surface: with the action's W0 = 859 MeV, a_s = 73-93 MeV and t(90-10) = 8.5-11.6 fm vs 16-20 MeV and 1.8-2.8 fm. It needs a gradient scale of about 195 MeV.
- R4 Crust and glitches: crustal moment-of-inertia fraction 0.34% (1.19% with W0_eff) vs at least 1.6% (Vela).
- R5 Stars: at M_max about 2.0, R(1.44) must fall by about 0.3 km, or M_max must rise by about 0.1 Msun at fixed radius.
- R6 Stability: the NVG one-field potential forbids m*/M < 0.836. That is the root of R2.

## Protocol (declared now, identical for every candidate field)
1. Fix the candidate's quantum numbers (spin, parity, isospin) and its couplings from known physics where they are known (meson masses, couplings measured in NN scattering or decays). Count every parameter that is NOT fixed this way.
2. TRAINING: fit at most the candidate's unknown parameters, and only on the ONE residual declared for that candidate below. TEST: everything else (R1-R6, the anchors A1-A2 must stay intact, the joint likelihood, 16O spin-orbit, ground state). A candidate with zero unknown parameters has no training step.
3. Report the fitted values and compare them with known particles (sigma/f0(500): mass 400-550 MeV; a0(980) or delta: 980 MeV; rho: 775 MeV; pi: 138 MeV, f^2/4pi = 0.075; f0(980); omega 782). A match that was not forced is a discovery signal. A mismatch is a result too.
4. Report each test residual as before / after / direction / pass-fail, and give falsifiable predictions for data not used anywhere.
5. Output: SCR/round4/<subdir>/result.json with keys candidate, quantum_numbers, fixed_couplings_and_sources, unknown_parameters (count and list), training_residual, fitted_values, test_table, known_particle_match, anchors_intact, joint_verdict, predictions, caveats, verdict (missing_element_found / partial / not_this_field / undecidable), verdict_text. Final text: a compact results summary and the path. Results only.
