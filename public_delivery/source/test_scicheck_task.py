"""Tests for the controlled task mechanism (integration only).

Run:
  D:/anaconda/python.exe -m unittest test_scicheck_task -v

Covers:
  - correct path (candidate passes -> PASSED terminal)
  - error-then-repair (VERIFIED_INCORRECT stays ACTIVE; explicit next call repairs;
    per-task budget of 3 means one failure does NOT terminate the task)
  - cross-task / cross-session rejection
  - terminal stop (no new solve after a terminal state)
  - capability rejection (action not compatible with domain/family)
  - unknown action -> ActionNotAllowed
  - terminal check BEFORE any state mutation
  - verify_run re-checks saved receipt without re-solving; no reference answer leaked
  - task hash is semantic (same math under different session/task_id -> same hash)
  - normalization rejections: boolean numerics, unknown fields, incomplete/overlapping
    segments, reversed/empty intervals, non-positive absolute tolerance
  - get_task/task_view return copies (internal mutation has no effect)
  - a task without a per-task budget ledger cannot solve (lazy creation refused)
"""
import copy
import unittest
from fractions import Fraction

from method_registry import TaskRegistry
from task_store import (
    TaskStore,
    TaskAlreadyTerminal,
    TaskSessionMismatch,
    TaskFormatError,
    build_task_hash,
    normalize_task,
)
from task_budget import BudgetLedger, DEFAULT_BUDGET
from task_controller import (
    TaskController,
    TaskRegistrar,
    TaskNotFound,
    NoTaskBudget,
    ActionNotAllowed,
    CapabilityRejected,
    BudgetExhausted,
)


def _piecewise_narrow():
    return {
        "domain": "integration",
        "problem": {
            "family": "piecewise_constant",
            "a": "-1",
            "b": "10000",
            "breakpoints": ["0"],
            "values": ["1", "0"],
        },
        "tolerance": {"absolute": "1/100000000", "relative": "0"},
    }


def _poly_simple():
    return {
        "domain": "integration",
        "problem": {
            "family": "polynomial",
            "a": "0",
            "b": "1",
            "coefficients": ["0", "0", "1"],
        },
        "tolerance": {"absolute": "1/1000000", "relative": "0"},
    }


class _Ctx:
    """Build a TaskController WITH a registered per-task budget for each task.

    Mirrors the trusted registrar path without requiring the SQLite persistence
    layer, so these unit tests focus on the controller semantics.
    """

    def __init__(self, session="S1"):
        self.session = session
        self.ledger = BudgetLedger()
        self.store = TaskStore(session=session)
        self.registry = TaskRegistry()
        self.ctl = TaskController(self.store, self.registry, ledger=self.ledger)

    def register(self, task_id, canonical, totals=None):
        """Mirrors the trusted registrar: normalize + freeze + per-task budget NOW."""
        self.store.register_task(task_id, canonical)
        self.ledger.for_task(task_id, dict(DEFAULT_BUDGET if totals is None else totals))
        return task_id


_FORBIDDEN = {
    "reference_fraction", "reference_answer", "reference_value",
    "exact_check", "candidate_as_fraction", "abs_diff_exact_fraction",
}


def _assert_no_reference_leak(d):
    def walk(o):
        if isinstance(o, dict):
            assert not (_FORBIDDEN & set(o.keys())), f"leaked reference fields: {_FORBIDDEN & set(o.keys())}"
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)
    walk(d)


class TestCorrectPath(unittest.TestCase):
    def test_poly_plain_quad_passes_to_terminal(self):
        ctx = _Ctx()
        task_id = ctx.register("t-poly", _poly_simple())
        r = ctx.ctl.execute_task(task_id, "req-1", "quad_plain")
        self.assertTrue(r["exact_ok"])
        self.assertEqual(ctx.store.get_task(task_id)["state"], "VERIFIED_PASS")
        _assert_no_reference_leak(r)


class TestErrorThenRepair(unittest.TestCase):
    def test_plain_wrong_then_explicit_breakpoint_repairs(self):
        ctx = _Ctx()
        task_id = ctx.register("t-piecewise", _piecewise_narrow())

        # run 1: plain quad -> VERIFIED_INCORRECT (misses the [-1,0] contribution)
        r1 = ctx.ctl.execute_task(task_id, "req-1", "quad_plain")
        self.assertFalse(r1["exact_ok"])
        # per-task budget 3 -> after one launch, 2 remain -> task stays ACTIVE
        self.assertEqual(ctx.store.get_task(task_id)["state"], "ACTIVE")

        # run 2: an EXPLICIT new call with a DIFFERENT action starts the repair
        r2 = ctx.ctl.execute_task(task_id, "req-2", "quad_breakpoints")
        self.assertTrue(r2["exact_ok"], f"breakpoint repair should pass: {r2}")
        self.assertEqual(ctx.store.get_task(task_id)["state"], "VERIFIED_PASS")

        # both run records kept, distinct UUID run_ids
        recs = ctx.store.get_task(task_id)["runs"]
        self.assertEqual(len(recs), 2)
        self.assertNotEqual(recs[0], recs[1])
        self.assertTrue(all(r.startswith("run-") and len(r) > len("run-") for r in recs))
        _assert_no_reference_leak(r1)
        _assert_no_reference_leak(r2)

    def test_per_task_budget_allows_multiple_repairs_before_exhaustion(self):
        ctx = _Ctx()
        task_id = ctx.register("t-pw", _piecewise_narrow())

        # plain fails (run1), breakpoint passes (run2); terminal stop on run 3
        r1 = ctx.ctl.execute_task(task_id, "req-1", "quad_plain")  # incorrect, ACTIVE
        r2 = ctx.ctl.execute_task(task_id, "req-2", "quad_breakpoints")  # -> VERIFIED_PASS
        self.assertEqual(ctx.store.get_task(task_id)["state"], "VERIFIED_PASS")
        with self.assertRaises(TaskAlreadyTerminal):
            ctx.ctl.execute_task(task_id, "req-3", "quad_plain")


class TestNoLazyBudgetCreation(unittest.TestCase):
    """The trusted register entry is the ONLY way a task acquires a budget ledger.
    A controller whose ledger has no entry for a task MUST refuse with NoTaskBudget
    -- business tools never lazily create a budget or grant time on the first solve."""

    def test_execute_without_ledger_refused(self):
        ctx = _Ctx()
        # Register via the store ONLY (no ledger entry) -- the model/business path.
        ctx.store.register_task("t-poly-lazy", _poly_simple())
        with self.assertRaises(NoTaskBudget):
            ctx.ctl.execute_task("t-poly-lazy", "req-1", "quad_plain")

    def test_trusted_register_creates_budget_immediately(self):
        ctx = _Ctx()
        ctx.ctl.trusted_register_task("t-poly", _poly_simple())
        b = ctx.ctl.budget_for("t-poly")
        self.assertEqual(b.counters["solve_launches"].total, 3)
        self.assertGreater(b.started_at, 0.0)


class TestCrossTaskSessionRejection(unittest.TestCase):
    def test_unknown_task_rejected(self):
        ctx = _Ctx()
        with self.assertRaises(TaskNotFound):
            ctx.ctl.execute_task("does-not-exist", "req", "quad_plain")

    def test_verify_run_unknown_run_rejected(self):
        ctx = _Ctx()
        ctx.register("t1", _poly_simple())
        r = ctx.ctl.execute_task("t1", "req", "quad_plain")
        with self.assertRaises(TaskNotFound):
            ctx.ctl.verify_run("t1", "run-nope")
        _assert_no_reference_leak(r)

    def test_cross_session_binding(self):
        ctxA = _Ctx(session="SA")
        ctxA.register("t1", _poly_simple())
        ctxB = _Ctx(session="SB")
        with self.assertRaises(TaskNotFound):
            ctxB.ctl.execute_task("t1", "req", "quad_plain")
        with self.assertRaises(TaskSessionMismatch):
            ctxB.store.assert_session_binding("t1", "SA")

    def test_two_tasks_same_request_id_independent(self):
        # The idempotency index is scoped by (session, task_id, request_id): the SAME
        # request_id in two DIFFERENT tasks must execute independently, no cross-task
        # bleed, and each task keeps its own run.
        ctx = _Ctx()
        tA = ctx.register("tA", _poly_simple())
        tB = ctx.register("tB", _piecewise_narrow())
        rA = ctx.ctl.execute_task(tA, "shared-req", "quad_plain")
        self.assertTrue(rA["exact_ok"])  # poly passes
        rB = ctx.ctl.execute_task(tB, "shared-req", "quad_plain")
        self.assertFalse(rB["exact_ok"])  # piecewise plain misses the contribution
        self.assertNotEqual(rA["run_id"], rB["run_id"])
        # Each task's own request index is independent
        self.assertEqual(ctx.store.lookup_request(tA, "shared-req", "quad_plain"), rA["run_id"])
        self.assertEqual(ctx.store.lookup_request(tB, "shared-req", "quad_plain"), rB["run_id"])

    def test_unknown_action_rejected(self):
        ctx = _Ctx()
        ctx.register("t1", _poly_simple())
        with self.assertRaises(ActionNotAllowed):
            ctx.ctl.execute_task("t1", "req", "evil.custom.action")

    def test_capability_rejection_on_unknown_domain(self):
        # A root-domain task whose card (quad_plain, an integration card) is not
        # compatible with the task's domain -> CapabilityRejected + UNSUPPORTED.
        ctx = _Ctx()
        root_task = {
            "domain": "root",
            "problem": {
                "family": "scaled_power",
                "power": "3", "scale": "1", "shift": "1",
                "a": "0", "b": "2", "start": "1.001",
            },
            "tolerance": {"absolute": "1/100000000", "relative": "0"},
        }
        ctx.register("t-root", root_task)
        with self.assertRaises(CapabilityRejected):
            ctx.ctl.execute_task("t-root", "req", "quad_plain")
        self.assertEqual(ctx.store.get_task("t-root")["state"], "UNSUPPORTED")

    def test_root_domain_task_registers_and_accepts_root_cards(self):
        # A root-domain task (scaled_power) registers fine and accepts root cards.
        ctx = _Ctx()
        root_task = {
            "domain": "root",
            "problem": {
                "family": "scaled_power",
                "power": "3", "scale": "1", "shift": "1",
                "a": "0", "b": "2", "start": "1.001",
            },
            "tolerance": {"absolute": "1/100000000", "relative": "0"},
        }
        tid = ctx.register("t-root", root_task)
        r = ctx.ctl.execute_task(tid, "req-1", "root_brent")
        self.assertTrue(r["exact_ok"], f"strict Brent should pass: {r}")
        self.assertEqual(ctx.store.get_task(tid)["state"], "VERIFIED_PASS")
        _assert_no_reference_leak(r)

    def test_root_domain_rejects_integration_card(self):
        # A root task must NOT accept an integration card (quad_plain) -> UNSUPPORTED.
        ctx = _Ctx()
        root_task = {
            "domain": "root",
            "problem": {
                "family": "scaled_power",
                "power": "3", "scale": "1", "shift": "1",
                "a": "0", "b": "2",
            },
            "tolerance": {"absolute": "1/100000000", "relative": "0"},
        }
        ctx.register("t-root", root_task)
        with self.assertRaises(CapabilityRejected):
            ctx.ctl.execute_task("t-root", "req", "quad_plain")
        self.assertEqual(ctx.store.get_task("t-root")["state"], "UNSUPPORTED")


class TestTerminalStop(unittest.TestCase):
    def test_no_new_solve_after_terminal(self):
        ctx = _Ctx()
        task_id = ctx.register("t-poly", _poly_simple())
        r = ctx.ctl.execute_task(task_id, "req-1", "quad_plain")
        self.assertTrue(r["exact_ok"])
        self.assertEqual(ctx.store.get_task(task_id)["state"], "VERIFIED_PASS")
        with self.assertRaises(TaskAlreadyTerminal):
            ctx.ctl.execute_task(task_id, "req-2", "quad_plain")

    def test_terminal_repeated_reads_stay_terminal(self):
        ctx = _Ctx()
        task_id = ctx.register("t-poly", _poly_simple())
        ctx.ctl.execute_task(task_id, "req-1", "quad_plain")  # -> VERIFIED_PASS
        # Repeated reads of the terminal task keep it terminal; no state flip.
        for _ in range(3):
            self.assertEqual(ctx.store.get_task(task_id)["state"], "VERIFIED_PASS")
        with self.assertRaises(TaskAlreadyTerminal):
            ctx.ctl.execute_task(task_id, "req-2", "quad_plain")

    def test_terminal_check_before_budget_mutation(self):
        """After a task is terminal, execute_task must raise TaskAlreadyTerminal
        WITHOUT consuming budget or appending a run."""
        ctx = _Ctx()
        task_id = ctx.register("t-poly", _poly_simple())
        ctx.ctl.execute_task(task_id, "req-1", "quad_plain")  # VERIFIED_PASS
        state_before = ctx.store.get_task(task_id)
        solve_before = ctx.ledger.get(task_id).counters["solve_launches"].spent
        with self.assertRaises(TaskAlreadyTerminal):
            ctx.ctl.execute_task(task_id, "req-2", "quad_plain")
        state_after = ctx.store.get_task(task_id)
        # no mutation: same run count, same state, no extra solve launch
        self.assertEqual(len(state_before["runs"]), len(state_after["runs"]))
        self.assertEqual(state_before["state"], state_after["state"])
        self.assertEqual(ctx.ledger.get(task_id).counters["solve_launches"].spent, solve_before)


class TestVerifyRunReceipt(unittest.TestCase):
    def test_verify_run_missing_receipt_is_inconclusive(self):
        # _Ctx has NO persistence, so no sealed receipt exists for the run. A pass
        # must NOT be claimed without it: verify_run returns INCONCLUSIVE.
        ctx = _Ctx()
        task_id = ctx.register("t-poly", _poly_simple())
        r1 = ctx.ctl.execute_task(task_id, "req-1", "quad_plain")
        rec = ctx.ctl.verify_run(task_id, r1["run_id"])
        self.assertEqual(rec["conclusion"], "INCONCLUSIVE")
        self.assertFalse(rec["consistent"])
        self.assertIsNone(rec["receipt_verdict"])
        # reference answer must NOT be in the model-facing view
        _assert_no_reference_leak(rec)
        self.assertNotIn("reference_fraction", rec)
        self.assertNotIn("reference_answer", rec)

    def test_verify_run_does_not_mutate_task(self):
        ctx = _Ctx()
        task_id = ctx.register("t-poly", _poly_simple())
        r1 = ctx.ctl.execute_task(task_id, "req-1", "quad_plain")
        snap = ctx.store.get_task(task_id)
        for _ in range(3):
            ctx.ctl.verify_run(task_id, r1["run_id"])
        after = ctx.store.get_task(task_id)
        self.assertEqual(snap["runs"], after["runs"])
        self.assertEqual(snap["state"], after["state"])


class TestTaskHashSemantics(unittest.TestCase):
    def test_hash_is_semantic_not_session_bound(self):
        c1 = _poly_simple()
        n1 = normalize_task(c1)
        h1 = build_task_hash(n1)
        # same math problem, different session/task_id must give the SAME hash
        h_again = build_task_hash(n1)
        self.assertEqual(h1, h_again)
        # but a different problem gives a different hash
        n2 = normalize_task(_piecewise_narrow())
        self.assertNotEqual(h1, build_task_hash(n2))

    def test_register_hash_bound_to_semantics(self):
        ctx = _Ctx()
        ctx.register("tA", _poly_simple())
        self.assertEqual(
            ctx.store.get_task("tA")["hash"],
            build_task_hash(normalize_task(_poly_simple())),
        )


class TestNormalizationRejections(unittest.TestCase):
    def test_reject_boolean_numeric_tolerance(self):
        bad = {
            "domain": "integration",
            "problem": {"family": "polynomial", "a": "0", "b": "1", "coefficients": ["1"]},
            "tolerance": {"absolute": True, "relative": "0"},
        }
        with self.assertRaises(TaskFormatError):
            normalize_task(bad)

    def test_reject_unknown_field(self):
        bad = {
            "domain": "integration",
            "problem": {"family": "polynomial", "a": "0", "b": "1", "coefficients": ["1"]},
            "tolerance": {"absolute": "1/10", "relative": "0"},
            "extra": "nope",
        }
        with self.assertRaises(TaskFormatError):
            normalize_task(bad)

    def test_reject_reversed_interval(self):
        bad = {
            "domain": "integration",
            "problem": {"family": "polynomial", "a": "5", "b": "0", "coefficients": ["1"]},
            "tolerance": {"absolute": "1/10", "relative": "0"},
        }
        with self.assertRaises(TaskFormatError):
            normalize_task(bad)

    def test_reject_empty_interval(self):
        bad = {
            "domain": "integration",
            "problem": {"family": "polynomial", "a": "3", "b": "3", "coefficients": ["1"]},
            "tolerance": {"absolute": "1/10", "relative": "0"},
        }
        with self.assertRaises(TaskFormatError):
            normalize_task(bad)

    def test_reject_non_positive_absolute_tolerance(self):
        bad = {
            "domain": "integration",
            "problem": {"family": "polynomial", "a": "0", "b": "1", "coefficients": ["1"]},
            "tolerance": {"absolute": "0", "relative": "0"},
        }
        with self.assertRaises(TaskFormatError):
            normalize_task(bad)

    def test_reject_overlapping_segment(self):
        bad = {
            "domain": "integration",
            "problem": {
                "family": "piecewise_constant",
                "a": "0", "b": "3",
                "breakpoints": ["2", "1"],  # not increasing -> overlap/reversed
                "values": ["1", "0", "1"],
            },
            "tolerance": {"absolute": "1/10", "relative": "0"},
        }
        with self.assertRaises(TaskFormatError):
            normalize_task(bad)

    def test_reject_incomplete_segments(self):
        bad = {
            "domain": "integration",
            "problem": {
                "family": "piecewise_constant",
                "a": "0", "b": "3",
                "breakpoints": ["1", "2"],
                "values": ["1", "0"],  # len(values) != len(breakpoints)+1
            },
            "tolerance": {"absolute": "1/10", "relative": "0"},
        }
        with self.assertRaises(TaskFormatError):
            normalize_task(bad)


class TestFrozenInputs(unittest.TestCase):
    def test_get_task_returns_copy_not_internal_ref(self):
        ctx = _Ctx()
        raw = _poly_simple()
        ctx.store.register_task("t1", raw)
        snapshot = ctx.store.get_task("t1")
        # mutate the returned copy: must NOT affect the stored task
        snapshot["state"] = "HACKED"
        self.assertEqual(ctx.store.get_task("t1")["state"], "ACTIVE")

    def test_register_freezes_input_deeply(self):
        ctx = _Ctx()
        raw = _poly_simple()
        ctx.store.register_task("t1", raw)
        # mutate the caller's dict after registration
        raw["problem"]["coefficients"][0] = "999"
        raw["tolerance"]["absolute"] = "999/1"
        # stored canonical is frozen to the original values
        stored = ctx.store.get_task("t1")
        self.assertEqual(stored["canonical"]["problem"]["coefficients"][0], "0")
        self.assertEqual(stored["canonical"]["tolerance"]["absolute"], "1/1000000")

    def test_task_view_returns_copy(self):
        ctx = _Ctx()
        ctx.register("t1", _poly_simple())
        v1 = ctx.ctl.task_view("t1")
        v1["state"] = "HACKED"
        self.assertEqual(ctx.store.get_task("t1")["state"], "ACTIVE")


if __name__ == "__main__":
    unittest.main()
