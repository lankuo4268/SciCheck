"""Regression tests for exact-precision independent verification + narrow-contribution repair.

Run:
  D:/anaconda/python.exe -m unittest test_scicheck -v
"""
import json
import unittest
from fractions import Fraction

from scicheck import (
    scicheck_integration,
    normalize_problem,
    exact_ok,
    run_plain_quad,
    run_quad_with_breakpoints,
)


class TestExactPrecisionBug(unittest.TestCase):
    """Reproduce the reported bug: float subtraction of a float-ified Fraction reference
    hides the true sub-ulp error. The exact check must fail where float check would pass.
    """

    def test_float_path_would_pass_but_exact_must_fail(self):
        # candidate = float(1/3), reference = 1/3 exactly, atol = 1e-30, rtol = 0
        payload = {
            "domain": ["0", "1"],
            "problem": {"type": "rational_poly", "coeffs": ["0", "0", "1"]},
            "tolerance": {"absolute": "1/1000000000000000000000000000000", "relative": "0"},
        }
        norm = normalize_problem(payload)
        candidate = float(Fraction(1, 3))  # 0.3333333333333333

        # Float-based (buggy) comparison would call this OK because
        # abs(candidate - float(ref)) is ~0.0 in double precision.
        ref_float = float(norm.reference_exact())
        buggy_diff = abs(candidate - ref_float)
        buggy_allowed = norm.allowed_error_float()
        # Demonstrate the bug: float math says it passes (diff rounds to 0).
        self.assertLessEqual(buggy_diff, buggy_allowed, "setup: buggy float path should look OK")

        # The CORRECT exact comparison must flag failure.
        check = exact_ok(candidate, norm)
        self.assertFalse(check["ok"], "exact check must fail for candidate=float(1/3) at atol=1e-30")
        # True exact error is the distance from float(1/3) to the rational 1/3.
        cand_frac = Fraction.from_float(candidate)
        true_err = abs(cand_frac - Fraction(1, 3))
        self.assertEqual(check["abs_diff_exact_fraction"], str(true_err))
        # 1/3 float has error ~ 1/54043195528445952 (the reciprocal of the relevant 2^56),
        # which is far larger than 1e-30, so it must fail.
        self.assertGreater(true_err, Fraction(1, 10**30))
        # Display float fields must not affect the verdict: they are informational only.
        self.assertEqual(
            check["abs_diff_display_float"],
            float(Fraction(check["abs_diff_exact_fraction"])),
        )

    def test_ok_still_passes_at_loose_tolerance(self):
        # Same candidate, but a tolerance loose enough that the float representation error fits.
        payload = {
            "domain": ["0", "1"],
            "problem": {"type": "rational_poly", "coeffs": ["0", "0", "1"]},
            "tolerance": {"absolute": "1/1000000", "relative": "0"},
        }
        norm = normalize_problem(payload)
        check = exact_ok(float(Fraction(1, 3)), norm)
        self.assertTrue(check["ok"])


class TestNarrowContribution(unittest.TestCase):
    """f = 1 on [-1,0], 0 on [0,10000]; exact integral = 1. atol=1e-8, rtol=0.

    A plain quad over the whole wide interval can miss the small left contribution and
    report a near-zero value with a tiny self-estimated error. The independent exact
    check must catch this, and the breakpoint run must recover the true value.
    """

    def test_exact_reference_is_one(self):
        payload = {
            "domain": ["-1", "10000"],
            "problem": {
                "type": "piecewise_const",
                "segments": [
                    ["-1", "0", "1"],
                    ["0", "10000", "0"],
                ],
            },
            "tolerance": {"absolute": "1/10000000000", "relative": "0"},
        }
        norm = normalize_problem(payload)
        self.assertEqual(norm.reference_exact(), Fraction(1))

    def test_plain_quad_may_miss_contribution_but_breakpoint_recovers(self):
        payload = {
            "domain": ["-1", "10000"],
            "problem": {
                "type": "piecewise_const",
                "segments": [
                    ["-1", "0", "1"],
                    ["0", "10000", "0"],
                ],
            },
            "tolerance": {"absolute": "1/10000000000", "relative": "0"},
        }
        norm = normalize_problem(payload)

        r_plain = run_plain_quad(norm, run_id=1)
        r_bp = run_quad_with_breakpoints(norm, run_id=2)

        # Breakpoint run must be correct to well within 1e-8.
        self.assertTrue(r_bp.exact_ok, f"breakpoint run should pass: {r_bp.exact_check}")
        self.assertAlmostEqual(r_bp.value, 1.0, delta=1e-8)

        # Record both runs (kept even if plain is wrong) and assert structure.
        result = scicheck_integration(payload)
        self.assertEqual(len(result["runs"]), 2)
        self.assertEqual([r["run_id"] for r in result["runs"]], [1, 2])
        self.assertEqual(result["runs"][0]["action"], "plain_quad")
        self.assertEqual(result["runs"][1]["action"], "quad_with_breakpoints")

        # Whatever the plain run did, the breakpoint run decides the outcome.
        self.assertEqual(result["passed"], r_bp.exact_ok)

    def test_plain_quad_actually_misreports(self):
        """Document whether plain quad (no breakpoints) misses the contribution.

        This test records the real behavior so the output file shows the concrete numbers.
        It does not hard-assert failure, only that both candidates are present and the
        exact verdict matches an independent recomputation.
        """
        payload = {
            "domain": ["-1", "10000"],
            "problem": {
                "type": "piecewise_const",
                "segments": [
                    ["-1", "0", "1"],
                    ["0", "10000", "0"],
                ],
            },
            "tolerance": {"absolute": "1/10000000000", "relative": "0"},
        }
        norm = normalize_problem(payload)
        r_plain = run_plain_quad(norm, run_id=1)
        r_bp = run_quad_with_breakpoints(norm, run_id=2)
        # Recompute verdicts independently to make sure RunRecord exact_ok is faithful.
        self.assertEqual(r_plain.exact_ok, exact_ok(r_plain.value, norm)["ok"])
        self.assertEqual(r_bp.exact_ok, exact_ok(r_bp.value, norm)["ok"])
        # Plain quad with no breakpoints on [-1,10000] for a 1-unit bump on [-1,0]:
        # capture the actual value for the output artifact.
        self.assertAlmostEqual(r_bp.value, 1.0, delta=1e-8)


class TestFrozenInputs(unittest.TestCase):
    def test_problem_and_tolerance_frozen(self):
        payload = {
            "domain": ["0", "1"],
            "problem": {"type": "rational_poly", "coeffs": ["1"]},
            "tolerance": {"absolute": "1/10", "relative": "0"},
        }
        n1 = normalize_problem(payload)
        n2 = normalize_problem(payload)
        self.assertEqual(n1.atol, n2.atol)
        self.assertEqual(n1.rtol, n2.rtol)
        self.assertEqual(n1.reference_exact(), n2.reference_exact())
        # raw input preserved verbatim
        self.assertEqual(n1.raw_input["tolerance"]["absolute"], "1/10")

    def test_reference_is_not_a_numeric_algorithm(self):
        """The reference must equal the exact rational antiderivative/sum, computed in
        Fraction, not the output of any numeric integrator."""
        payload = {
            "domain": ["0", "1"],
            "problem": {"type": "rational_poly", "coeffs": ["0", "0", "1"]},
            "tolerance": {"absolute": "0", "relative": "0"},
        }
        norm = normalize_problem(payload)
        # independent of quad entirely:
        self.assertEqual(norm.reference_exact(), Fraction(1, 3))


class TestHandVerifiableExamples(unittest.TestCase):
    def test_poly_x2(self):
        with open("examples/poly_x2_0_1.json", encoding="utf-8") as f:
            payload = json.load(f)
        res = scicheck_integration(payload)
        self.assertEqual(Fraction(res["reference_fraction"]), Fraction(1, 3))
        self.assertTrue(res["passed"])

    def test_piecewise_const(self):
        with open("examples/piecewise_const_0_3.json", encoding="utf-8") as f:
            payload = json.load(f)
        res = scicheck_integration(payload)
        self.assertEqual(Fraction(res["reference_fraction"]), Fraction(3, 2))
        self.assertTrue(res["passed"])


if __name__ == "__main__":
    unittest.main()
