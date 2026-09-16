"""Focused full-matrix controls for the retarded gradient-matching helper."""
from __future__ import annotations

import math
from pathlib import Path
import sys
import unittest

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import nvg_retarded_gradient_matching as matching  # noqa: E402
import nvg_retarded_response as response  # noqa: E402


# This is a predeclared double-precision bound.  The normalized upstream
# matrix condition and the measured rational-map sensitivity are included;
# the factor is a fixed roundoff safety margin, not tuned after a failure.
ROUNDING_SAFETY_FACTOR = 256.0
MACHINE_EPSILON = np.finfo(float).eps


def _conditioning_aware_bound(row: dict[str, object]) -> float:
    condition = float(np.linalg.cond(np.asarray(row["H_normalized"], dtype=complex)))
    sensitivity = row["match"].relative_sensitivity
    amplification = max(1.0, float(sensitivity) if sensitivity is not None else 0.0)
    return ROUNDING_SAFETY_FACTOR * MACHINE_EPSILON * max(1.0, condition) * amplification


class RetardedGradientMatchingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        """Build model-generated controls once; all chi values come from np.solve(H,e0)."""
        base = response.upstream.BulkModel()
        rows: list[dict[str, object]] = []
        for background in response._backgrounds(base):
            state, model = background["state"], background["model"]
            for q in (100.0, 200.0):
                edge = response.particle_hole_edge(state["m"], state["k"], q)
                frequencies = (0.0, 0.6 * edge, 1.25 * edge + 0.2j)
                for z in frequencies:
                    pi = response.polarization_kernel(state["m"], state["k"], q, z, d=model.d)
                    for scale in (0.5, 1.0, 2.0):
                        coefficients, true_z = response._coefficients(model, state, pi, scale)
                        upstream_answer = response.response_from_hessian(
                            coefficients, true_z, q, z, solve_direct=False
                        )
                        H = np.asarray(upstream_answer["H"], dtype=complex)
                        solution = np.linalg.solve(H, np.asarray([1, 0, 0, 0], dtype=complex))
                        rows.append({
                            "background": background["background_id"], "q": q, "z": z,
                            "scale": scale, "coefficients": coefficients, "true_Z": float(true_z),
                            "chi": complex(solution[0]), "H_normalized": upstream_answer["H_normalized"],
                        })
        cls.rows = rows

    def test_54_full_upstream_matrix_reconstructions(self) -> None:
        self.assertEqual(len(self.rows), 54)
        errors = []
        for row in self.rows:
            match = matching.infer_gradient_coefficient(
                row["coefficients"], row["q"], row["z"], row["chi"]
            )
            row["match"] = match
            error = abs(match.reconstructed_Z - row["true_Z"]) / max(abs(row["true_Z"]), 1.0)
            errors.append(error)
            self.assertTrue(match.accepted, (row["background"], row["q"], row["z"], match.status))
            self.assertLessEqual(error, _conditioning_aware_bound(row), row)
            self.assertTrue(math.isfinite(match.sensitivity_abs))
            self.assertEqual(match.K, match.q * match.q - match.z * match.z)
        self.max_reconstruction_error = max(errors)

    def test_formula_and_derivative_against_complex_and_real_perturbations(self) -> None:
        # Exercise both absorbing real-rim and upper-half-plane responses.
        selected = [row for row in self.rows if row["scale"] == 1.0 and row["q"] == 100.0]
        derivative_errors = []
        for row in selected:
            base = matching.infer_gradient_coefficient(
                row["coefficients"], row["q"], row["z"], row["chi"]
            )
            expected = -base.B * base.B / (base.K * base.denominator * base.denominator)
            self.assertLess(abs(base.derivative_dZ_dchi - expected) / max(abs(expected), 1.0), 2e-15)
            step = 1e-7 * max(1.0, abs(base.chi))
            complex_step = step * (1.0 + 0.37j)
            real_plus = matching.infer_gradient_coefficient(
                row["coefficients"], row["q"], row["z"], base.chi + complex_step
            )
            real_minus = matching.infer_gradient_coefficient(
                row["coefficients"], row["q"], row["z"], base.chi - complex_step
            )
            finite_difference = (real_plus.reconstructed_Z - real_minus.reconstructed_Z) / (2.0 * complex_step)
            derivative_errors.append(abs(finite_difference - base.derivative_dZ_dchi) /
                                     max(abs(base.derivative_dZ_dchi), 1.0))
            self.assertLess(derivative_errors[-1], 2e-8)
            real_plus = matching.infer_gradient_coefficient(
                row["coefficients"], row["q"], row["z"], base.chi + step
            )
            real_minus = matching.infer_gradient_coefficient(
                row["coefficients"], row["q"], row["z"], base.chi - step
            )
            finite_difference = (real_plus.reconstructed_Z - real_minus.reconstructed_Z) / (2.0 * step)
            derivative_errors.append(abs(finite_difference - base.derivative_dZ_dchi) /
                                     max(abs(base.derivative_dZ_dchi), 1.0))
            self.assertLess(derivative_errors[-1], 2e-8)

    def test_zero_response_is_regular_and_physical_status_is_not_hidden(self) -> None:
        result = matching.match_retarded_blocks(2.0, 1.0, 3.0, 2.0, 0.25j, 0.0)
        self.assertEqual(result.C_recovered, 0j)
        self.assertEqual(result.status, "DIAGNOSTIC_COMPLEX_OR_NONPOSITIVE_Z")
        self.assertFalse(result.positive_real_candidate)
        self.assertFalse(result.reconstructed_Z.imag == 0.0 and result.reconstructed_Z.real > 0.0)

    def test_malformed_singular_and_lower_half_plane_inputs_fail_closed(self) -> None:
        with self.assertRaises(matching.RetardedGradientMatchingError):
            matching.match_retarded_blocks(1.0, 0.0, 2.0, 2.0, 0.0, 1.0)
        with self.assertRaises(matching.RetardedGradientMatchingError):
            matching.match_retarded_blocks(1.0, 1.0, 2.0, 2.0, 2.0, 1.0)
        with self.assertRaises(matching.RetardedGradientMatchingError):
            matching.match_retarded_blocks(1.0, 1.0, 2.0, 2.0, 0.0, 1.0)
        with self.assertRaises(matching.RetardedGradientMatchingError):
            matching.match_retarded_blocks(1.0, 1.0, 2.0, 2.0, -0.1j, 1.0)
        for args in ((1.0, 1.0, 2.0, 0.0, 0.1j, 1.0),
                     (1.0, 1.0, 2.0, True, 0.1j, 1.0),
                     (1.0, 1.0, 2.0, 2.0, 0.1j, float("nan"))):
            with self.assertRaises(matching.RetardedGradientMatchingError):
                matching.match_retarded_blocks(*args)
        with self.assertRaises(matching.RetardedGradientMatchingError):
            matching.infer_gradient_coefficient(
                {"a": 1, "b": 1, "d0": 1, "g": 1, "h": 1, "t0": 0}, 2, 0, 1
            )

    def test_vector_elimination_conditioning_refuses_near_pole(self) -> None:
        coefficients = {"a": 2.0, "b": 0.7, "d0": 3.0,
                        "g": 0.5, "h": -0.8, "t0": 2.0}
        q, true_z = 1.0, 1.2

        def direct_chi(z: complex) -> complex:
            a, b, d0 = coefficients["a"], coefficients["b"], coefficients["d0"]
            g, h, t0 = coefficients["g"], coefficients["h"], coefficients["t0"]
            K = q * q - z * z
            H = np.asarray([
                [a, b, g, -g * z / q],
                [b, d0 + true_z * K, h, 0.0],
                [g, h, -(q * q + t0), z * q],
                [-g * z / q, 0.0, z * q, t0 - z * z],
            ], dtype=complex)
            return np.linalg.solve(H, np.asarray([1, 0, 0, 0], dtype=complex))[0]

        near = math.sqrt(3.0 - 1e-14)
        with self.assertRaisesRegex(matching.RetardedGradientMatchingError,
                                    "vector-elimination conditioning"):
            matching.infer_gradient_coefficient(coefficients, q, near, direct_chi(near))
        exact_coefficients = {**coefficients, "t0": -1.0}
        with self.assertRaisesRegex(matching.RetardedGradientMatchingError,
                                    "vector-elimination factor"):
            matching.infer_gradient_coefficient(exact_coefficients, q, 0.0, 1.0)

        ordinary = math.sqrt(3.0 - 1e-5)
        result = matching.infer_gradient_coefficient(coefficients, q, ordinary, direct_chi(ordinary))
        self.assertTrue(result.accepted)
        self.assertLess(abs(result.reconstructed_Z - true_z) / true_z, 1e-8)

    def test_cancellation_is_an_explicit_refusal_not_a_trustworthy_number(self) -> None:
        # C0+Z*K rounds to C0 at double precision, so subtraction cannot
        # resolve the deliberately tiny gradient signal.
        result = matching.match_retarded_blocks(2.0, 1.0, 3.0, 2.0, 0.0, 0.6)
        self.assertEqual(result.status, "REFUSED_C_SUBTRACTION_CONDITIONING")
        self.assertFalse(result.accepted)
        self.assertLessEqual(result.subtraction_conditioning, matching.DEFAULT_CONDITIONING_LIMIT)
        self.assertTrue(math.isfinite(result.reconstructed_Z.real))

    def test_omitted_longitudinal_sector_is_detected_by_cross_frequency_disagreement(self) -> None:
        background = response._backgrounds(response.upstream.BulkModel())[0]
        state, model, q = background["state"], background["model"], 100.0
        edge = response.particle_hole_edge(state["m"], state["k"], q)
        inferred = []
        for z in (0.0, 0.6 * edge, 1.25 * edge + 0.2j):
            pi = response.polarization_kernel(state["m"], state["k"], q, z, d=model.d)
            coefficients, true_z = response._coefficients(model, state, pi, 1.0)
            full = response.response_from_hessian(coefficients, true_z, q, z, solve_direct=False)
            # Deliberately generate data with A_L removed: the upper-left
            # 3x3 block is not the accepted full retarded matrix.
            wrong_solution = np.linalg.solve(
                np.asarray(full["H"], dtype=complex)[:3, :3], np.asarray([1, 0, 0], dtype=complex)
            )
            inferred.append(matching.infer_gradient_coefficient(
                coefficients, q, z, wrong_solution[0]
            ).reconstructed_Z)
        spread = max(abs(left - right) for left in inferred for right in inferred)
        self.assertGreater(spread / max(abs(float(true_z)), 1.0), 1e-5)


if __name__ == "__main__":
    unittest.main()
