#!/usr/bin/env python3
"""
NVG: can configuration-space theta be DERIVED from the spacetime action?
=========================================================================
Resolution of the last open question of the quantum block. The answer is a
computed dichotomy, and it closes the question definitively.

PART A — structural obstruction (computed). A spacetime field theta(x)
evaluated at N particle positions can only produce SEPARABLE phases
theta_sep(x1..xN) = sum_i theta(x_i): a 3-dimensional field cannot carry
the 3N-dimensional information of an entangled phase. The obstruction is
quantitative: the entanglement entropy of the singlet pointer state is
computed below (= ln 2, maximal, for every CHSH setting pair) — a nonzero
value PROVES no separable phase representation exists, for any choice of
the field configuration.

PART B — dynamical no-go (computed, exhaustive). Even allowing the
spacetime field to be an arbitrary shared random variable lambda (any
distribution, any dimensionality) and the outcomes arbitrary deterministic
responses A(a, lambda), B(b, lambda), the CHSH combination per lambda is
    s(lambda) = A1 B1 - A1 B2 + A2 B1 + A2 B2,  A_i, B_j in {+-1}.
All 16 assignments are enumerated below: max |s| = 2. Averaging over any
lambda distribution therefore gives S <= 2 (Fine's theorem, here verified
by exhaustion). Experiment gives S ~ 2.4-2.7 (loophole-free 2015+).
=> A CLASSICAL spacetime theta — however nonlinear, stochastic or
   condensate-like its dynamics — is EXCLUDED BY EXPERIMENT.

PART C — the dichotomy. The only way a field theory produces
configuration-space structure is QUANTIZATION: the wavefunctional
Psi[theta(x)] of the quantized W-field lives on the space of field
configurations, and its N-particle sectors are exactly the N-particle
wavefunctions on R^{3N} whose phase is the contextual theta demonstrated
in nvg_bell_contextual.py. So:

  (i)  classical spacetime theta  -> S <= 2      -> falsified by data;
  (ii) quantized W-field          -> config-space theta derived,
                                     but quantum mechanics is the INPUT.

CONSEQUENCE (final status of the quantum block): NVG cannot DERIVE quantum
mechanics from the classical action; it can only be FORMULATED as a
quantum field theory whose Madelung hydrodynamics is a representation.
The block's honest content: the representation is consistent (double
slit, uncertainty, collapse timescale) and adds one falsifiable
prediction, S(T > T_c) -> 0. The open problem is hereby closed — in the
negative for "emergence", in the positive for internal consistency.
"""

from __future__ import annotations
import itertools
import math
import numpy as np

A_SET = (0.0, math.pi / 2)
B_SET = (math.pi / 4, 3 * math.pi / 4)


def singlet_coeffs_matrix(a, b):
    d = (a - b) / 2.0
    inv = 1.0 / math.sqrt(2.0)
    return np.array([[inv * math.sin(d), inv * math.cos(d)],
                     [-inv * math.cos(d), inv * math.sin(d)]])


def main():
    print("=" * 78)
    print("  NVG: CONFIGURATION-SPACE THETA FROM THE ACTION — RESOLVED")
    print("=" * 78)

    # ── A. entanglement entropy of the pointer state ────────────────────
    print("\nA. Non-separability of the singlet phase (entanglement entropy):")
    for a in A_SET:
        for b in B_SET:
            c = singlet_coeffs_matrix(a, b)
            sv = np.linalg.svd(c, compute_uv=False)
            p = sv ** 2 / np.sum(sv ** 2)
            ent = float(-np.sum(p * np.log(np.maximum(p, 1e-300))))
            print(f"   settings ({a:.2f}, {b:.2f}): Schmidt weights "
                  f"{p[0]:.3f}/{p[1]:.3f}, S_ent = {ent:.4f} (ln 2 = {math.log(2):.4f})")
            assert abs(ent - math.log(2)) < 1e-9
    print("   -> maximal for every setting pair: NO separable phase")
    print("      theta1(x1) + theta2(x2) exists; a 3-dim spacetime field")
    print("      cannot encode the 3N-dim entangled phase. Structural.")

    # ── B. exhaustive no-go for arbitrary local field responses ─────────
    print("\nB. Exhaustive CHSH bound for ANY shared classical field lambda:")
    best = 0
    for A1, A2, B1, B2 in itertools.product((-1, 1), repeat=4):
        s = A1 * B1 - A1 * B2 + A2 * B1 + A2 * B2
        best = max(best, abs(s))
    print(f"   max |A1B1 - A1B2 + A2B1 + A2B2| over all 16 assignments = {best}")
    print(f"   -> S <= {best} for every lambda distribution (Fine's theorem,")
    print(f"      verified by exhaustion). Loophole-free experiments: S ~ 2.4-2.7.")
    print(f"   -> classical spacetime theta of ANY dynamics: EXCLUDED by data.")
    assert best == 2

    # ── C. dichotomy ─────────────────────────────────────────────────────
    print("""
C. DICHOTOMY (the resolution):
   (i)  theta classical on spacetime  => S <= 2  => experimentally dead;
   (ii) W-field quantized             => wavefunctional Psi[theta(x)]
        lives on configuration space; its N-particle sectors ARE the
        contextual theta of nvg_bell_contextual.py — derived, but with
        quantum mechanics as input.

VERDICT: the question "derive configuration-space theta from the
spacetime action" is CLOSED. Within a classical action it is impossible
(A: structural, B: excluded by experiment). Within the quantized theory
it is automatic but not a derivation of QM. The NVG quantum block's
honest final status: a consistent hydrodynamic REPRESENTATION of quantum
field theory with one added falsifiable prediction, S(T > T_c) -> 0.
""")
    print("=" * 78)


if __name__ == "__main__":
    main()
