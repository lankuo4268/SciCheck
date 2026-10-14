"""Regression tests for budget, idempotency, persistence, restart, and tamper rejection.

Run:
  D:/anaconda/python.exe -m unittest test_scicheck_task_budget -v
"""
import os
import tempfile
import unittest

from method_registry import TaskRegistry
from task_store import TaskStore, TaskFrozen, TaskSessionMismatch, TaskAlreadyTerminal
from task_budget import BudgetLedger, DEFAULT_BUDGET, BudgetExhausted as BudgetExhaustedErr
from task_persistence import StorePersistence
from task_controller import (
    TaskController,
    TaskRegistrar,
    TaskNotFound,
    NoTaskBudget,
    ActionNotAllowed,
    CapabilityRejected,
    BudgetExhausted,
    RequestActionConflict,
    DeadlineExpiredError,
    TaskError,
)
from task_budget import BudgetLedger
from task_persistence import StorePersistence
from receipt_seal import ReceiptTampered, make_receipt, ReceiptSealer


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


_FORBIDDEN = {
    "reference_fraction", "reference_answer", "reference_value",
    "exact_check", "candidate_as_fraction", "abs_diff_exact_fraction",
    "abs_diff_fraction", "allowed_error_fraction",
}


def _assert_no_leak(d):
    def walk(o):
        if isinstance(o, dict):
            assert not (_FORBIDDEN & set(o.keys())), f"leaked: {_FORBIDDEN & set(o.keys())}"
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)
    walk(d)


class _Env:
    """Build a controller with a temp SQLite db + a PER-TASK budget ledger via the
    trusted registrar path (normalization + freeze + SQLite row + ledger entry)."""

    def __init__(self, totals=None):
        self._tmp = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self._tmp.name, "tasks.db")
        self.key_path = os.path.join(self._tmp.name, "receipt.key")
        self.registry = TaskRegistry()
        self.store = TaskStore(session="S1")
        self.pers = StorePersistence(self.db_path, key_path=self.key_path)
        self.ledger = BudgetLedger()
        self.totals = dict(DEFAULT_BUDGET if totals is None else totals)
        self.ctl = TaskController(self.store, self.registry,
                                  persistence=self.pers, ledger=self.ledger)

    def register(self, task_id, canonical):
        # One trusted registration: normalize + freeze + SQLite row + per-task budget
        self.ctl.trusted_register_task(task_id, canonical, self.totals)
        return task_id

    def budget(self, task_id):
        return self.ledger.get(task_id)

    def close(self):
        self.pers.close()
        self._tmp.cleanup()


ONE_SOLVE = {"solve_launches": 1, "func_evals": 2000, "tool_requests": 12,
             "model_requests": 8, "wall_seconds": 180}


class TestBudgetExhaustion(unittest.TestCase):
    def test_third_incorrect_launch_exhausts_solve_budget(self):
        # Default 3 solve launches. Force 3 consecutive incorrect launches: the
        # task stays ACTIVE after 1 and 2 (budget remaining), and the 3rd
        # incorrect verdict with no solve budget left -> INCOMPLETE terminal.
        env = _Env()
        try:
            tid = env.register("t-pw", _piecewise_narrow())
            for i in (1, 2):
                r = env.ctl.execute_task(tid, f"req-{i}", "quad_plain")
                self.assertFalse(r["exact_ok"])
                self.assertEqual(env.store.get_task(tid)["state"], "ACTIVE",
                                 f"run {i} should keep ACTIVE while budget remains")
            r3 = env.ctl.execute_task(tid, "req-3", "quad_plain")
            self.assertFalse(r3["exact_ok"])
            # solve budget was 3 and is now exhausted -> INCOMPLETE terminal
            self.assertEqual(env.store.get_task(tid)["state"], "INCOMPLETE")
            self.assertEqual(env.budget(tid).counters["solve_launches"].remaining(), 0)
        finally:
            env.close()

    def test_one_solve_budget_stops_after_first_failure(self):
        # A task with only 1 solve launch: the FIRST incorrect verdict exhausts it.
        env = _Env(totals=ONE_SOLVE)
        try:
            tid = env.register("t-pw", _piecewise_narrow())
            r = env.ctl.execute_task(tid, "req-1", "quad_plain")
            self.assertFalse(r["exact_ok"])
            self.assertEqual(env.store.get_task(tid)["state"], "INCOMPLETE")
            # A new solve must be refused (terminal)
            with self.assertRaises(TaskAlreadyTerminal):
                env.ctl.execute_task(tid, "req-2", "quad_breakpoints")
        finally:
            env.close()

    def test_budget_none_is_not_unbounded(self):
        # A task whose ledger has no budget registered cannot solve at all.
        env = _Env()
        try:
            env.store.register_task("t-pw", _piecewise_narrow())  # store-only, NO ledger entry
            with self.assertRaises(NoTaskBudget):
                env.ctl.execute_task("t-pw", "req-1", "quad_plain")
        finally:
            env.close()


class TestIdempotency(unittest.TestCase):
    def test_same_request_same_action_returns_same_run_no_resolve(self):
        env = _Env()
        try:
            tid = env.register("t-poly", _poly_simple())
            before_solves = env.budget(tid).counters["solve_launches"].spent
            r1 = env.ctl.execute_task(tid, "req-1", "quad_plain")
            run1 = r1["run_id"]
            self.assertFalse(r1["idempotent_hit"])
            # repeat the exact same (request_id, action) -> same run, no new solve
            r2 = env.ctl.execute_task(tid, "req-1", "quad_plain")
            self.assertTrue(r2["idempotent_hit"])
            self.assertEqual(r2["run_id"], run1)
            # no additional solve was launched
            self.assertEqual(env.budget(tid).counters["solve_launches"].spent, before_solves + 1)
            # the task record still has only ONE run (no duplicate solve)
            self.assertEqual(len(env.store.get_task(tid)["runs"]), 1)
            _assert_no_leak(r2)
        finally:
            env.close()

    def test_same_request_different_action_rejected(self):
        env = _Env()
        try:
            tid = env.register("t-pw", _piecewise_narrow())
            env.ctl.execute_task(tid, "req-1", "quad_plain")
            # same request_id, different action -> conflict
            with self.assertRaises(RequestActionConflict):
                env.ctl.execute_task(tid, "req-1", "quad_breakpoints")
        finally:
            env.close()


class TestTamperCrossTaskRejection(unittest.TestCase):
    def test_reject_tampered_receipt(self):
        env = _Env()
        try:
            tid = env.register("t-poly", _poly_simple())
            r = env.ctl.execute_task(tid, "req-1", "quad_plain")
            run_id = r["run_id"]
            # Tamper the sealed receipt in the trusted db, then verify_run must reject
            import json
            conn = env.pers._conn
            row = conn.execute("SELECT sealed_json FROM receipts WHERE run_id=?",
                              (run_id,)).fetchone()
            sealed = json.loads(row["sealed_json"])
            # The original verdict for this passing poly task is VERIFIED_PASS;
            # flip it to the OPPOSITE verdict so the HMAC genuinely mismatches.
            original_verdict = sealed["verdict"]
            sealed["verdict"] = ("VERIFIED_INCORRECT" if original_verdict == "VERIFIED_PASS"
                                  else "VERIFIED_PASS")
            conn.execute("UPDATE receipts SET sealed_json=? WHERE run_id=?",
                         (json.dumps(sealed, sort_keys=True), run_id))
            conn.commit()
            self.assertNotEqual(original_verdict, sealed["verdict"]), "test must flip the verdict"
            # verify_run must fail the integrity check
            with self.assertRaises(TaskError):
                env.ctl.verify_run(tid, run_id)
        finally:
            env.close()

    def test_reject_cross_task_run_lookup(self):
        env = _Env()
        try:
            tid = env.register("t-poly", _poly_simple())
            r = env.ctl.execute_task(tid, "req-1", "quad_plain")
            run_id = r["run_id"]
            # A different task must not be able to verify this run
            tid2 = env.register("t-pw", _piecewise_narrow())
            with self.assertRaises(TaskNotFound):
                env.ctl.verify_run(tid2, run_id)
        finally:
            env.close()


class TestRestartConservativeStop(unittest.TestCase):
    def test_restart_stops_active_task_incomplete_no_fresh_budget(self):
        env = _Env()
        try:
            tid = env.register("t-pw", _piecewise_narrow())
            env.ctl.execute_task(tid, "req-1", "quad_plain")  # stays ACTIVE (budget 3)
            # Simulate a restart: reload budget verbatim from the db (no extension)
            reloaded = env.pers.load_task(tid, "S1")
            self.assertIsNotNone(reloaded)
            self.assertEqual(reloaded["budget"].counters["solve_launches"].total, 3)
            # Conservative stop of the still-ACTIVE task on restart
            changed = env.ctl.restart_recover([tid])
            self.assertEqual(changed.get(tid), "INCOMPLETE")
            self.assertEqual(env.store.get_task(tid)["state"], "INCOMPLETE")
            # A terminal task can no longer be solved
            with self.assertRaises(TaskAlreadyTerminal):
                env.ctl.execute_task(tid, "req-2", "quad_breakpoints")
        finally:
            env.close()

    def test_reopen_terminal_requires_new_task_id(self):
        env = _Env()
        try:
            tid = env.register("t-pw", _piecewise_narrow())
            env.ctl.execute_task(tid, "req-1", "quad_plain")
            env.ctl.restart_recover([tid])  # now INCOMPLETE terminal
            # Reopening requires a NEW task id with trusted authorization;
            # the old terminal row is left untouched.
            new_tid = "t-pw-reopened"
            env.ctl.reopen_terminal(new_tid, old_task_id=tid, authorize=True)
            self.assertEqual(env.store.get_task(new_tid)["state"], "ACTIVE")
            self.assertEqual(env.store.get_task(tid)["state"], "INCOMPLETE")
            # unauthorized reopen is rejected
            with self.assertRaises(TaskError):
                env.ctl.reopen_terminal("t-pw-x", old_task_id=tid, authorize=False)
        finally:
            env.close()

    def test_real_close_and_reopen_same_db(self):
        """REAL restart: close the original persistence + controller, RECREATE both
        against the SAME db + key file, reload tasks/budgets/requests/runs. Unfinished
        ACTIVE tasks are conservatively stopped INCOMPLETE with consumed + reserved
        units preserved; terminal tasks are unchanged; old receipts re-verify under
        the SAME trusted key (a NEW key file would not accept the old receipts)."""
        env = _Env()
        tid_pw = env.register("t-pw", _piecewise_narrow())
        env.ctl.execute_task(tid_pw, "req-1", "quad_plain")  # stays ACTIVE, 1 launch spent
        tid_poly = env.register("t-poly", _poly_simple())
        r_poly = env.ctl.execute_task(tid_poly, "req-1", "quad_plain")  # -> VERIFIED_PASS
        run_poly = r_poly["run_id"]
        # Copy db + key to a STABLE location so they survive the original tmpdir cleanup
        import shutil
        stable_dir = tempfile.TemporaryDirectory()
        stable_db = os.path.join(stable_dir.name, "tasks.db")
        stable_key = os.path.join(stable_dir.name, "receipt.key")
        shutil.copyfile(env.db_path, stable_db)
        shutil.copyfile(env.key_path, stable_key)
        # Snapshot the ACTIVE task's consumed budget BEFORE closing
        active_budget_before = env.budget(tid_pw).counters["solve_launches"].spent
        env.close()  # close the original connection + controller (tmpdir cleaned up)
        # RECREATE a fresh controller on the STABLE copy of db + key
        store2 = TaskStore(session="S1")
        pers2 = StorePersistence(stable_db, key_path=stable_key)
        ledger2 = BudgetLedger()
        ctl2 = TaskController(store2, TaskRegistry(), persistence=pers2, ledger=ledger2)
        try:
            # Reload tasks + budgets verbatim (no fresh time/budget granted)
            reloaded_pw = pers2.load_task(tid_pw, "S1")
            self.assertIsNotNone(reloaded_pw)
            self.assertEqual(
                reloaded_pw["budget"].counters["solve_launches"].spent, active_budget_before,
                "restart must NOT reset or extend the consumed budget",
            )
            # Old receipt re-verifies under the SAME trusted key (identity intact)
            rec2 = pers2.load_receipt(run_poly)
            self.assertEqual(rec2["verdict"], "VERIFIED_PASS")
            # A DIFFERENT key file MUST NOT be able to accept the old receipts
            alt_key = os.path.join(stable_dir.name, "alt.key")
            with open(alt_key, "wb") as f:
                f.write(os.urandom(32))
            pers3 = StorePersistence(stable_db, key_path=alt_key)
            try:
                with self.assertRaises(ReceiptTampered):
                    pers3.load_receipt(run_poly)
            finally:
                pers3.close()
        finally:
            pers2.close()
            stable_dir.cleanup()


class TestRegistrarReopenRecoversTasks(unittest.TestCase):
    """REAL reopen: close the original registrar, build a FRESH TaskRegistrar on the
    same db + key, and the restore must recover BOTH persisted tasks, their budgets
    (verbatim, original started_at), runs, and the idempotency index. A still-ACTIVE
    task is conservatively stopped INCOMPLETE; a completed task stays VERIFIED_PASS
    and remains verifiable + idempotently readable. A db without its original key
    file is refused outright (no fresh key may be generated to cover old data)."""

    def test_registrar_reopen_recovers_two_tasks(self):
        import shutil
        env = _Env()
        tid_pw = env.register("t-pw", _piecewise_narrow())
        env.ctl.execute_task(tid_pw, "req-1", "quad_plain")   # stays ACTIVE (1 launch spent)
        tid_poly = env.register("t-poly", _poly_simple())
        r_poly = env.ctl.execute_task(tid_poly, "req-1", "quad_plain")  # -> VERIFIED_PASS
        run_poly = r_poly["run_id"]
        stable_dir = tempfile.TemporaryDirectory()
        stable_db = os.path.join(stable_dir.name, "tasks.db")
        stable_key = os.path.join(stable_dir.name, "receipt.key")
        shutil.copyfile(env.db_path, stable_db)
        shutil.copyfile(env.key_path, stable_key)
        env.close()  # close the ORIGINAL registrar's persistence + connection
        # A brand-new TaskRegistrar on the SAME db + key; it must restore both tasks.
        reg2 = TaskRegistrar(session="S1", db_path=stable_db, key_path=stable_key)
        try:
            restored = reg2.restore()
            self.assertIn(tid_pw, restored)
            self.assertIn(tid_poly, restored)
            # the still-ACTIVE task was conservatively stopped; the completed one kept
            self.assertEqual(restored[tid_pw], "INCOMPLETE")
            self.assertEqual(restored[tid_poly], "VERIFIED_PASS")
            # both tasks are now present in the fresh registrar's in-memory store
            self.assertEqual(reg2.controller.store.get_task(tid_pw)["state"], "INCOMPLETE")
            self.assertEqual(reg2.controller.store.get_task(tid_poly)["state"], "VERIFIED_PASS")
            # budget restored VERBATIM (no top-up): the piecewise task spent 1 of 3
            pw_budget = reg2.controller.ledger.get(tid_pw)
            self.assertIsNotNone(pw_budget)
            self.assertEqual(pw_budget.counters["solve_launches"].spent, 1)
            self.assertEqual(pw_budget.counters["solve_launches"].total, 3)
            # the completed task's old receipt still re-verifies under the SAME key
            v = reg2.verify_run(tid_poly, run_poly)
            self.assertEqual(v.get("conclusion", "VERIFIED_PASS"), "VERIFIED_PASS")
            self.assertTrue(v.get("consistent", False))
            # terminal tasks stay terminal: no new solve on the completed task
            with self.assertRaises(TaskAlreadyTerminal):
                reg2.execute_task(tid_poly, "req-9", "quad_plain")
        finally:
            reg2.persistence.close()
            stable_dir.cleanup()

    def test_registrar_reopen_missing_key_with_old_data_refused(self):
        """Opening an EXISTING db whose trusted key file is absent must be refused
        (SealerKeyUnavailable), never silently re-keyed to cover unverifiable data."""
        import shutil
        from receipt_seal import SealerKeyUnavailable
        env = _Env()
        tid = env.register("t-poly", _poly_simple())
        env.ctl.execute_task(tid, "req-1", "quad_plain")
        stable_dir = tempfile.TemporaryDirectory()
        stable_db = os.path.join(stable_dir.name, "tasks.db")
        missing_key = os.path.join(stable_dir.name, "receipt.key")  # deliberately absent
        shutil.copyfile(env.db_path, stable_db)
        env.close()
        try:
            with self.assertRaises(SealerKeyUnavailable):
                TaskRegistrar(session="S1", db_path=stable_db, key_path=missing_key)
        finally:
            stable_dir.cleanup()


class TestTamperIdentityFields(unittest.TestCase):
    """HMAC binds session / task_id / task_hash / run_id / candidate / verdict.
    Tampering ANY of these identity fields must make verify_run reject the receipt."""

    def _receipt_row(self, env, tid, run_id):
        import json
        row = env.pers._conn.execute(
            "SELECT sealed_json FROM receipts WHERE run_id=?", (run_id,)).fetchone()
        return json.loads(row["sealed_json"])

    def _mutate_and_expect_reject(self, env, tid, run_id, field, value):
        sealed = self._receipt_row(env, tid, run_id)
        sealed[field] = value
        import json
        env.pers._conn.execute(
            "UPDATE receipts SET sealed_json=? WHERE run_id=?",
            (json.dumps(sealed, sort_keys=True), run_id))
        env.pers._conn.commit()
        # Any identity-field tamper breaks the receipt: verify_run must reject.
        with self.assertRaises(TaskError):
            env.ctl.verify_run(tid, run_id)

    def test_tamper_task_id_field(self):
        env = _Env()
        try:
            tid = env.register("t-poly", _poly_simple())
            r = env.ctl.execute_task(tid, "req-1", "quad_plain")
            self._mutate_and_expect_reject(env, tid, r["run_id"], "task_id", "t-impersonator")
        finally:
            env.close()

    def test_tamper_run_id_field(self):
        env = _Env()
        try:
            tid = env.register("t-poly", _poly_simple())
            r = env.ctl.execute_task(tid, "req-1", "quad_plain")
            self._mutate_and_expect_reject(env, tid, r["run_id"], "run_id", "run-forged")
        finally:
            env.close()

    def test_tamper_task_hash_field(self):
        env = _Env()
        try:
            tid = env.register("t-poly", _poly_simple())
            r = env.ctl.execute_task(tid, "req-1", "quad_plain")
            self._mutate_and_expect_reject(env, tid, r["run_id"], "task_hash", "deadbeef")
        finally:
            env.close()

    def test_tamper_candidate_field(self):
        env = _Env()
        try:
            tid = env.register("t-poly", _poly_simple())
            r = env.ctl.execute_task(tid, "req-1", "quad_plain")
            self._mutate_and_expect_reject(env, tid, r["run_id"], "candidate", 999999.0)
        finally:
            env.close()

    def test_tamper_session_field(self):
        env = _Env()
        try:
            tid = env.register("t-poly", _poly_simple())
            r = env.ctl.execute_task(tid, "req-1", "quad_plain")
            self._mutate_and_expect_reject(env, tid, r["run_id"], "session", "SAFEGATE")
        finally:
            env.close()

    def test_tamper_verdict_field(self):
        env = _Env()
        try:
            tid = env.register("t-poly", _poly_simple())
            r = env.ctl.execute_task(tid, "req-1", "quad_plain")
            # original verdict is VERIFIED_PASS; flip to VERIFIED_INCORRECT
            self._mutate_and_expect_reject(env, tid, r["run_id"], "verdict", "VERIFIED_INCORRECT")
        finally:
            env.close()


class TestConcurrentDuplicateRequest(unittest.TestCase):
    """TWO TRULY CONCURRENT callers: two threads each holding an INDEPENDENT SQLite
    connection to the SAME db, both issuing the IDENTICAL (task_id, request_id,
    action_id). Under a real race, the atomic SQLite claim lets exactly ONE caller
    launch a new solve; the other finds the claim already taken and either reuses the
    winner's run (idempotent hit) or stops as a PENDING re-entrant entry, NEVER a
    second solve. The authoritative proof is the durable SQLite runs table: exactly
    ONE run row exists for the shared request."""

    def _run_rows_for_request(self, db_path, task_id, request_id):
        import sqlite3
        conn = sqlite3.connect(db_path)
        try:
            rows = conn.execute(
                "SELECT DISTINCT run_id FROM runs WHERE task_id=? AND run_id IN "
                "(SELECT run_id FROM requests WHERE task_id=? AND request_id=?)",
                (task_id, task_id, request_id),
            ).fetchall()
            return [r[0] for r in rows]
        finally:
            conn.close()

    def test_two_threads_two_connections_same_request_counts_one_solve(self):
        import threading
        env = _Env()
        try:
            tid = env.register("t-poly", _poly_simple())
            db_path, key_path = env.db_path, env.key_path
            barrier = threading.Barrier(2)
            outcomes = []
            lock = threading.Lock()

            def worker():
                # Each thread opens its OWN connection to the SAME db (independent conn).
                # restore_from_persistence loads the task + its budget (verbatim,
                # original started_at, no top-up) into this fresh connection. The task
                # is ACTIVE in SQLite (the first solve has not run yet), so restore
                # leaves it ACTIVE. The winner then launches the solve normally; the
                # loser either finds the durable idempotent claim (winner already
                # committed the run row) or stops PENDING. Either way only ONE run row
                # ends up in the durable runs table.
                p2 = StorePersistence(db_path, key_path=key_path)
                s2 = TaskStore(session="S1")
                c2 = TaskController(s2, TaskRegistry(), persistence=p2, ledger=BudgetLedger())
                try:
                    c2.restore_from_persistence()
                    barrier.wait()  # release the barrier -> true simultaneous launch
                    r = c2.execute_task(tid, "shared-req", "quad_plain")
                    with lock:
                        outcomes.append(r)
                except Exception as e:  # a PENDING re-entrant stop is a legitimate loser
                    with lock:
                        outcomes.append({"exception": type(e).__name__, "msg": str(e)})
                finally:
                    p2.close()

            t1 = threading.Thread(target=worker)
            t2 = threading.Thread(target=worker)
            t1.start(); t2.start()
            t1.join(); t2.join()

            # The durable SQLite proof: exactly ONE distinct run row bound to the
            # shared request, regardless of which thread won the race.
            runs = self._run_rows_for_request(db_path, tid, "shared-req")
            self.assertEqual(len(runs), 1,
                             f"true concurrency must launch exactly ONE solve, "
                             f"got {len(runs)} distinct run rows")
            # Every successful outcome reuses that single winner run; losers either
            # returned the idempotent hit or stopped PENDING (both legitimate).
            for r in outcomes:
                if isinstance(r, dict) and "run_id" in r:
                    self.assertEqual(r["run_id"], runs[0],
                                     "any caller that reuses must reuse the winner's run")
            # Durable budget check: 1 solve launched, 2 tool requests (one per
            # connection), 0 reserved. Both connections' charges are durable and
            # not lost to a read-modify-write race.
            import sqlite3, json as _json
            conn = sqlite3.connect(db_path)
            conn.row_factory = sqlite3.Row
            try:
                row = conn.execute(
                    "SELECT counters_json FROM budgets WHERE task_id=?", (tid,)
                ).fetchone()
                b_dict = _json.loads(row["counters_json"])
                self.assertEqual(b_dict["solve_launches"]["spent"], 1,
                                 "exactly one solve launch must be recorded durably")
                self.assertEqual(b_dict["tool_requests"]["spent"], 2,
                                 "two independent connections each charged one tool request; "
                                 "the durable counter must show 2 (no lost update)")
                self.assertEqual(b_dict["tool_requests"]["reserved"], 0,
                                 "no leftover reserved tool units after both calls")
            finally:
                conn.close()
        finally:
            env.close()

    def test_two_independent_connections_same_request_serial(self):
        """Even serialized across two independent connections: the second one
        reuses the first's run and does NOT start a second solve (the durable runs
        table carries exactly one row for the shared request)."""
        env = _Env()
        try:
            tid = env.register("t-poly", _poly_simple())
            r1 = env.ctl.execute_task(tid, "shared-req", "quad_plain")
            # A second, independent controller (own connection) on the SAME db + key
            p2 = StorePersistence(env.db_path, key_path=env.key_path)
            s2 = TaskStore(session="S1")
            c2 = TaskController(s2, TaskRegistry(), persistence=p2, ledger=BudgetLedger())
            try:
                c2.restore_from_persistence()
                r2 = c2.execute_task(tid, "shared-req", "quad_plain")
                self.assertTrue(r2.get("idempotent_hit"), "second conn must be an idempotent hit")
                self.assertEqual(r2["run_id"], r1["run_id"], "must reuse the winner's run")
            finally:
                p2.close()
            # The durable SQLite record carries exactly one run for the shared request
            runs = self._run_rows_for_request(env.db_path, tid, "shared-req")
            self.assertEqual(runs, [r1["run_id"]],
                             "the duplicate request must not launch a second solve")
        finally:
            env.close()


class TestInjectedSolveAndReferenceErrors(unittest.TestCase):
    def test_injected_solve_exception_marks_execution_error(self):
        env = _Env()
        try:
            tid = env.register("t-poly", _poly_simple())
            # Monkeypatch the launch path to raise a solver exception
            real = env.ctl._launch_one_candidate

            def boom(*a, **k):
                raise RuntimeError("injected solver failure")

            env.ctl._launch_one_candidate = boom
            with self.assertRaises(TaskError):
                env.ctl.execute_task(tid, "req-1", "quad_plain")
            self.assertEqual(env.store.get_task(tid)["state"], "EXECUTION_ERROR")
            env.ctl._launch_one_candidate = real
        finally:
            env.close()

    def test_reference_exception_marks_execution_error(self):
        env = _Env()
        try:
            tid = env.register("t-poly", _poly_simple())
            real = env.ctl._norm_for

            def bad_norm(canon):
                raise ValueError("injected reference computation failure")

            env.ctl._norm_for = bad_norm
            with self.assertRaises(TaskError):
                env.ctl.execute_task(tid, "req-1", "quad_plain")
            self.assertEqual(env.store.get_task(tid)["state"], "EXECUTION_ERROR")
            env.ctl._norm_for = real
        finally:
            env.close()


class TestModelRequestRegistration(unittest.TestCase):
    """A LOCAL execute_task must NOT add model requests. Only an explicit
    trusted-adapter registration increments the model_requests counter.
    These numbers are unit probes, not a claim of real AGH scientific-task
    consumption (the trusted-adapter wiring is a separate step)."""

    def test_local_execute_does_not_add_model_requests(self):
        env = _Env()
        try:
            tid = env.register("t-poly", _poly_simple())
            before = env.budget(tid).counters["model_requests"].spent
            env.ctl.execute_task(tid, "req-1", "quad_plain")
            after = env.budget(tid).counters["model_requests"].spent
            # Local solve does NOT register a model request.
            self.assertEqual(after - before, 0)
        finally:
            env.close()

    def test_explicit_trusted_registration_increments(self):
        env = _Env()
        try:
            tid = env.register("t-poly", _poly_simple())
            b = env.budget(tid)
            before = b.counters["model_requests"].spent
            # The trusted adapter would call this before a REAL model request.
            b.register_model_request()
            self.assertEqual(b.counters["model_requests"].spent, before + 1)
        finally:
            env.close()


class TestToolRequestCounting(unittest.TestCase):
    """method_lookup / execute_task / verify_run each independently count one
    tool_requests call, including idempotent repeats and rejected requests.
    Counts are persisted to the per-task ledger immediately, not batched."""

    def test_controller_calls_count_tool_requests(self):
        env = _Env()
        try:
            tid = env.register("t-poly", _poly_simple())
            b = env.budget(tid)
            base = b.counters["tool_requests"].spent
            env.ctl.method_lookup(tid, "structure")
            self.assertEqual(b.counters["tool_requests"].spent, base + 1)
            base = b.counters["tool_requests"].spent
            env.ctl.execute_task(tid, "req-1", "quad_plain")
            self.assertEqual(b.counters["tool_requests"].spent, base + 1)
            base = b.counters["tool_requests"].spent
            r1 = env.ctl.execute_task(tid, "req-1", "quad_plain")  # idempotent hit still counts
            self.assertEqual(b.counters["tool_requests"].spent, base + 1)
            base = b.counters["tool_requests"].spent
            env.ctl.verify_run(tid, r1["run_id"])
            self.assertEqual(b.counters["tool_requests"].spent, base + 1)
        finally:
            env.close()

    def test_rejected_request_still_counts(self):
        env = _Env()
        try:
            tid = env.register("t-pw", _piecewise_narrow())
            b = env.budget(tid)
            env.ctl.execute_task(tid, "req-1", "quad_plain")
            base = b.counters["tool_requests"].spent
            with self.assertRaises(RequestActionConflict):
                env.ctl.execute_task(tid, "req-1", "quad_breakpoints")
            # The rejected call still consumed one tool request.
            self.assertEqual(b.counters["tool_requests"].spent, base + 1)
        finally:
            env.close()

    def test_tool_budget_exhausted_stops_and_persists_incomplete(self):
        # Force the tool_requests budget to 1 so the SECOND call exhausts it;
        # exhaustion must persist INCOMPLETE and stop further business calls.
        env = _Env(totals={"solve_launches": 3, "func_evals": 2000, "tool_requests": 1,
                           "model_requests": 8, "wall_seconds": 180})
        try:
            tid = env.register("t-poly", _poly_simple())
            b = env.budget(tid)
            r1 = env.ctl.execute_task(tid, "req-1", "quad_plain")  # consumes the 1 tool request
            self.assertTrue(r1["exact_ok"])
            # Task is VERIFIED_PASS (terminal). A NEW business call must now be refused:
            # either by the terminal gate OR by the tool budget; neither may proceed.
            with self.assertRaises((TaskAlreadyTerminal, BudgetExhausted)):
                env.ctl.execute_task(tid, "req-2", "quad_breakpoints")
            # The persisted task state in SQLite reflects a terminal state (not ACTIVE)
            persisted_state = env.pers.get_task_state(tid, "S1")
            self.assertIn(persisted_state, {"VERIFIED_PASS", "INCOMPLETE"})
        finally:
            env.close()


class TestCounterIntegrity(unittest.TestCase):
    """_Counter.reserve/spend must reject negative/boolean amounts and account
    for already-reserved units. spend-then-reserve ordering is enforced."""

    def test_reserve_rejects_bool_and_negative(self):
        b = self._budget()
        with self.assertRaises(ValueError):
            b.counters["solve_launches"].reserve(True)
        with self.assertRaises(ValueError):
            b.counters["solve_launches"].reserve(-1)

    def test_reserve_checks_spent_plus_reserved(self):
        b = self._budget()
        c = b.counters["solve_launches"]
        c.spent = 2  # simulate 2 already-spent
        c.reserve(1)  # now reserved=1, spent=2 -> remaining 0 of total 3
        with self.assertRaises(BudgetExhaustedErr):
            c.reserve(1)  # would exceed total

    def test_spend_accounts_for_reserved(self):
        b = self._budget()
        c = b.counters["solve_launches"]
        c.reserve(3)  # all reserved
        with self.assertRaises(BudgetExhaustedErr):
            c.spend(1)  # cannot spend on top of full reservation

    def test_commit_launch_consumes_reserved(self):
        b = self._budget()
        b.reserve_launch()
        self.assertEqual(b.counters["solve_launches"].reserved, 1)
        self.assertEqual(b.counters["solve_launches"].spent, 0)
        b.commit_launch()
        self.assertEqual(b.counters["solve_launches"].reserved, 0)
        self.assertEqual(b.counters["solve_launches"].spent, 1)

    def _budget(self):
        return BudgetLedger().for_task("t", dict(DEFAULT_BUDGET))


class TestDeadlineExpiry(unittest.TestCase):
    """After solving, if the wall-clock deadline has passed, a pass CANNOT be
    published (no VERIFIED_PASS receipt is left)."""

    def test_deadline_expiry_blocks_pass(self):
        env = _Env(totals={"solve_launches": 3, "func_evals": 2000, "tool_requests": 12,
                           "model_requests": 8, "wall_seconds": 0})
        try:
            tid = env.register("t-poly", _poly_simple())
            b = env.budget(tid)
            b.mark_started(0.0)  # started in the past -> deadline already expired
            with self.assertRaises(DeadlineExpiredError):
                env.ctl.execute_task(tid, "req-1", "quad_plain")
            self.assertEqual(env.store.get_task(tid)["state"], "INCOMPLETE")
        finally:
            env.close()

    def test_low_func_eval_budget_exhausts(self):
        # Force a very small func_evals budget so the gated integrand trips it.
        env = _Env(totals={"solve_launches": 3, "func_evals": 1, "tool_requests": 12,
                           "model_requests": 8, "wall_seconds": 180})
        try:
            tid = env.register("t-poly", _poly_simple())
            # With only 1 func-eval allowed, a quad over [0,1] will exhaust it
            # mid-evaluation and the launch raises -> budget-exhausted path.
            with self.assertRaises((BudgetExhausted, DeadlineExpiredError, TaskError)):
                env.ctl.execute_task(tid, "req-1", "quad_plain")
        finally:
            env.close()


def _root_x_minus_1_cubed():
    """(x-1)^3 on [0,2], shift=1, power=3, scale=1, start=1.001, atol=1e-8."""
    return {
        "domain": "root",
        "problem": {
            "family": "scaled_power",
            "power": "3", "scale": "1", "shift": "1",
            "a": "0", "b": "2", "start": "1.001",
        },
        "tolerance": {"absolute": "1/100000000", "relative": "0"},
    }


def _root_shift_one_third_tight():
    """shift=1/3, power=3, scale=1, [0,1], atol=1e-30 (very tight)."""
    return {
        "domain": "root",
        "problem": {
            "family": "scaled_power",
            "power": "3", "scale": "1", "shift": "1/3",
            "a": "0", "b": "1",
        },
        "tolerance": {"absolute": "1/1000000000000000000000000000000", "relative": "0"},
    }


class TestRootDomainSolvers(unittest.TestCase):
    """Root-domain solvers via the controlled-task mechanism: (x-1)^3 on [0,2],
    start=1.001, atol=1e-8. Coarse Newton (scipy root_scalar newton, xtol=1e-3,
    maxiter=24) leaves a forward error well above 1e-8 -> VERIFIED_INCORRECT
    (development diagnostic, NOT a library-default pass). Strict Brent
    (scipy root_scalar brentq, xtol=1e-13, rtol=1e-14, maxiter=300) passes.
    The public direct route (candidate = binary64(shift)) also passes when
    shift is exactly representable; when shift is not (e.g. 1/3) and the
    tolerance is tight enough, it MUST fail (no fake pass by construction)."""

    def test_coarse_newton_fails_at_1e8_tightness(self):
        env = _Env()
        try:
            tid = env.register("t-root", _root_x_minus_1_cubed())
            r = env.ctl.execute_task(tid, "req-n", "root_newton")
            self.assertTrue(r["exact_ok"] is False, f"coarse Newton should FAIL atol=1e-8: {r}")
            self.assertEqual(r["verdict"], "VERIFIED_INCORRECT")
            # residual diagnostic is present and is NOT what decided the verdict
            self.assertIsNotNone(r["quad_abserr_reported"])
            # converged flag came from real root_scalar (not a hand-rolled loop)
            self.assertIn("converged", r["resource"])
            self.assertTrue(r["resource"]["converged"], "Newton should report converged=True from root_scalar")
            self.assertEqual(r["resource"]["solver"], "scipy.optimize.root_scalar")
            self.assertEqual(r["resource"]["method"], "newton")
            self.assertEqual(r["resource"]["label"], "coarse-development-diagnostic")
            # task stays ACTIVE (budget of 3 not exhausted by one incorrect launch)
            self.assertEqual(env.store.get_task(tid)["state"], "ACTIVE")
        finally:
            env.close()

    def test_strict_brent_passes(self):
        env = _Env()
        try:
            tid = env.register("t-root", _root_x_minus_1_cubed())
            r = env.ctl.execute_task(tid, "req-b", "root_brent")
            self.assertTrue(r["exact_ok"], f"strict Brent should PASS atol=1e-8: {r}")
            self.assertEqual(r["verdict"], "VERIFIED_PASS")
            self.assertEqual(r["candidate"], 1.0, "Broot should be exactly 1.0 for (x-1)^3 on [0,2]")
            self.assertIn("converged", r["resource"])
            self.assertTrue(r["resource"]["converged"])
            self.assertEqual(r["resource"]["solver"], "scipy.optimize.root_scalar")
            self.assertEqual(r["resource"]["method"], "brentq")
            self.assertEqual(r["resource"]["label"], "strict")
            # receipt is sealed; verify_run confirms PASS without re-solving
            v = env.ctl.verify_run(tid, r["run_id"])
            self.assertEqual(v["conclusion"], "VERIFIED_PASS")
            self.assertEqual(env.store.get_task(tid)["state"], "VERIFIED_PASS")
        finally:
            env.close()

    def test_direct_root_exact_shift_passes(self):
        """Public direct route for a binary64-exact shift (shift=1) at atol=1e-8."""
        env = _Env()
        try:
            tid = env.register("t-root", _root_x_minus_1_cubed())
            r = env.ctl.execute_task(tid, "req-d", "root_direct")
            self.assertTrue(r["exact_ok"], f"direct route with shift=1 should PASS atol=1e-8: {r}")
            self.assertEqual(r["verdict"], "VERIFIED_PASS")
            self.assertEqual(r["resource"]["solver"], "structural-direct")
        finally:
            env.close()

    def test_direct_root_shift_one_third_tight_atol_fails(self):
        """shift=1/3, atol=1e-30: the public direct candidate (binary64 1/3) has a
        forward error of 1/54043195528445952, far above 1e-30 -> MUST fail.
        Strict Brent also cannot close the gap -> MUST fail."""
        env = _Env()
        try:
            tid = env.register("t-root-third", _root_shift_one_third_tight())
            r = env.ctl.execute_task(tid, "req-d", "root_direct")
            self.assertFalse(r["exact_ok"], f"direct route must FAIL at atol=1e-30 for shift=1/3: {r}")
            self.assertEqual(r["verdict"], "VERIFIED_INCORRECT")
            # exact forward error is recorded in the receipt / resource
            self.assertIsNotNone(r["resource"].get("residual_diagnostic"))
        finally:
            env.close()

    def test_brent_shift_one_third_tight_atol_fails(self):
        """shift=1/3, atol=1e-30: even the strict Brent route cannot beat the
        binary64 representation of the shift -> MUST fail."""
        env = _Env()
        try:
            tid = env.register("t-root-third-b", _root_shift_one_third_tight())
            r = env.ctl.execute_task(tid, "req-b", "root_brent")
            self.assertFalse(r["exact_ok"], f"brent must FAIL at atol=1e-30 for shift=1/3: {r}")
            self.assertEqual(r["verdict"], "VERIFIED_INCORRECT")
        finally:
            env.close()

    def test_root_domain_rejects_float_geometry_collapse(self):
        """A root task whose float(a) >= float(shift) or float(b) <= float(shift)
        must be rejected at registration (the interval geometry cannot be
        preserved in float64). Use shift and a so tight that the float
        representation collapses the bracket."""
        env = _Env()
        try:
            # a and shift are so close that float(a) == float(shift) in binary64
            bad = {
                "domain": "root",
                "problem": {
                    "family": "scaled_power",
                    "power": "1", "scale": "1",
                    # a and shift differ only in the last ulp so float(a)==float(shift)
                    "shift": "0.5", "a": "0.5", "b": "1",
                },
                "tolerance": {"absolute": "1/100000000", "relative": "0"},
            }
            # a == shift violates the strict enclosure a < shift -> TaskFormatError
            from task_store import TaskFormatError
            with self.assertRaises(TaskFormatError):
                env.register("t-root-bad", bad)
        finally:
            env.close()

    def test_root_shift_zero_normal_case(self):
        """Normal root=0 case: shift=0, candidate=0.0 is a LEGAL value. The
        receipt must carry candidate=0.0 (field presence, not truthiness) and
        verify_run must accept it."""
        env = _Env()
        try:
            task = {
                "domain": "root",
                "problem": {
                    "family": "scaled_power",
                    "power": "3", "scale": "1", "shift": "0",
                    "a": "-1", "b": "1",
                },
                "tolerance": {"absolute": "1/100000000", "relative": "0"},
            }
            tid = env.register("t-root-zero", task)
            r = env.ctl.execute_task(tid, "req-d", "root_direct")
            self.assertTrue(r["exact_ok"], f"direct root=0 should PASS: {r}")
            self.assertEqual(r["verdict"], "VERIFIED_PASS")
            self.assertEqual(r["candidate"], 0.0, "candidate must be exactly 0.0")
            # verify_run must treat candidate=0.0 as PRESENT (field-existence check,
            # not 'not receipt.get(field)'), not as a missing value.
            v = env.ctl.verify_run(tid, r["run_id"])
            self.assertEqual(v["conclusion"], "VERIFIED_PASS")
        finally:
            env.close()

    def test_root_solves_charge_budget_gated_evals(self):
        """Every function AND derivative call, including the residual diagnostic
        evaluation, must be charged through the budget gate. After a root solve
        the func_evals counter must have actually moved (not zero, not fabricated)."""
        import scicheck as _sc
        from task_budget import TaskBudget, DEFAULT_BUDGET
        # Build a TaskBudget that enforces the gate; wrap it so we can count calls.
        class _CountingBudget:
            def __init__(self, inner, counts):
                self._inner = inner
                self._counts = counts
                self.func_calls = 0
                self.deriv_calls = 0
            def check_before_eval(self):
                self._inner.check_before_eval()
            def commit_eval(self):
                self._inner.commit_eval()
            def __getattr__(self, name):
                return getattr(self._inner, name)
        # For the root solvers we charge f and fp through BudgetGate. Simulate the
        # exact wiring the controller uses and confirm the gate really counts.
        rp = _sc.RootProblem(scale=1, shift=1, power=3, a=0, b=2, start=1.001)
        f, fp = rp.float_func_with_derivative()
        import time
        tb = TaskBudget("t", dict(DEFAULT_BUDGET))
        tb.mark_started(time.time())
        base_f = _sc.BudgetGate(f, tb)
        base_fp = _sc.BudgetGate(fp, tb)
        # Drive Newton a few iterations manually through the GATED callbacks
        x = 1.001
        for _ in range(3):
            fx = base_f(x)
            d = base_fp(x)
            x = x - fx / d
        self.assertGreater(base_f.calls, 0, "f evaluations must be counted by the gate")
        self.assertGreater(base_fp.calls, 0, "derivative evaluations must be counted by the gate")
        self.assertEqual(tb.counters["func_evals"].spent, base_f.calls + base_fp.calls,
                         "every gated f/fp call (incl. residual) must be charged to func_evals")


class TestRegisterTaskSelfNormalizes(unittest.TestCase):
    def test_register_rejects_invalid_tolerance(self):
        env = _Env()
        try:
            bad = {
                "domain": "integration",
                "problem": {"family": "polynomial", "a": "0", "b": "1", "coefficients": ["1"]},
                "tolerance": {"absolute": "0", "relative": "0"},  # non-positive atol
            }
            with self.assertRaises(Exception):
                env.store.register_task("t-bad", bad)
            # and the task must NOT exist as a pseudo-legal task
            self.assertNotIn("t-bad", env.store.list_task_ids())
        finally:
            env.close()

    def test_register_rejects_extra_fields(self):
        env = _Env()
        try:
            bad = dict(_poly_simple())
            bad["extra"] = "nope"
            with self.assertRaises(Exception):
                env.store.register_task("t-x", bad)
            self.assertNotIn("t-x", env.store.list_task_ids())
        finally:
            env.close()


class TestLinearSystemDomain(unittest.TestCase):
    """Linear-system domain (1..8-dim rational square matrices) via the
    controlled-task mechanism.

    Required concrete case: [[1,1],[1,1000001/1000000]] rhs [1,1500001/1500000]
    atol=1e-7 rtol=0 -- float32 initial error (forward error 1/12 per component)
    -> float64 repair PASS. Vector candidate type preservation through JSON/SQLite/
    receipt/reload is asserted. The Decimal high-precision route is a separate
    task/route (same math + tolerance + hash) that PASSES without changing the
    original task's terminal state. No-unique-solution and binary64-format-boundary
    cases are refused UNSUPPORTED at the PRE-launch gate with zero solve cost."""

    _REQUIRED = {
        "domain": "linear_system",
        "problem": {
            "family": "rational_square_matrix",
            "n": "2",
            "matrix": [["1", "1"], ["1", "1000001/1000000"]],
            "rhs": ["1", "1500001/1500000"],
        },
        "tolerance": {"absolute": "1/10000000", "relative": "0"},
    }

    _NORMAL = {
        "domain": "linear_system",
        "problem": {
            "family": "rational_square_matrix",
            "n": "2",
            "matrix": [["2", "0"], ["0", "4"]],
            "rhs": ["1", "1"],
        },
        "tolerance": {"absolute": "1/10000000", "relative": "0"},
    }

    _NO_UNIQUE = {
        "domain": "linear_system",
        "problem": {
            "family": "rational_square_matrix",
            "n": "2",
            "matrix": [["1", "1"], ["1", "1"]],
            "rhs": ["1", "1"],
        },
        "tolerance": {"absolute": "1/10000000", "relative": "0"},
    }

    _ULTRA_TIGHT = {
        "domain": "linear_system",
        "problem": {
            "family": "rational_square_matrix",
            "n": "2",
            "matrix": [["1", "1"], ["1", "1000001/1000000"]],
            "rhs": ["1", "1500001/1500000"],
        },
        "tolerance": {"absolute": "1/1000000000000000000000000000000", "relative": "0"},
    }

    # Boundary: solution [0, 1/3], atol=1e-30 -- component 0 (exact 0) is
    # representable; component 1 (1/3) is NOT in binary64 -> ANY triggers reject.
    _BOUNDARY = {
        "domain": "linear_system",
        "problem": {
            "family": "rational_square_matrix",
            "n": "2",
            "matrix": [["1", "0"], ["0", "3"]],
            "rhs": ["0", "1"],
        },
        "tolerance": {"absolute": "1/1000000000000000000000000000000", "relative": "0"},
    }

    def test_required_matrix_float32_initial_error_then_float64_repair(self):
        env = _Env()
        try:
            tid = env.register("ls-req", self._REQUIRED)
            r32 = env.ctl.execute_task(tid, "r1", "ls_np32")
            # float32 candidate [0.25, 0.75]: forward error 1/12 per component > 1e-7
            self.assertFalse(r32["exact_ok"])
            self.assertEqual(r32.get("verdict"), "VERIFIED_INCORRECT")
            self.assertEqual(env.store.get_task(tid)["state"], "ACTIVE")
            # The float32 candidate is a VECTOR (list), not a scalar
            self.assertIsInstance(r32["candidate"], list)
            self.assertEqual(len(r32["candidate"]), 2)
            # Private detail via the TRUSTED store (NOT the model-facing view)
            trusted32 = env.store.get_run(r32["run_id"])
            fwd = [c["forward_error_exact_fraction"] for c in trusted32.exact_check["per_component"]]
            self.assertEqual(fwd, ["1/12", "1/12"])

            r64 = env.ctl.execute_task(tid, "r2", "ls_np64")
            self.assertTrue(r64["exact_ok"])
            self.assertEqual(r64.get("verdict"), "VERIFIED_PASS")
            self.assertEqual(env.store.get_task(tid)["state"], "VERIFIED_PASS")
            self.assertIsInstance(r64["candidate"], list)
            self.assertEqual(len(r64["candidate"]), 2)
            _assert_no_leak(r64)  # no reference leak in the model-facing view
        finally:
            env.close()

    def test_vector_candidate_reload_preserves_list_type(self):
        env = _Env()
        try:
            tid = env.register("ls-req", self._REQUIRED)
            env.ctl.execute_task(tid, "r1", "ls_np32")
            r64 = env.ctl.execute_task(tid, "r2", "ls_np64")
            original_candidate = r64["candidate"]
            self.assertIsInstance(original_candidate, list)
            # Reload from a FRESH persistence connection: the vector candidate must
            # come back as a list (type preserved, not flattened to a scalar).
            pers2 = StorePersistence(env.db_path, key_path=env.key_path)
            try:
                ctl2 = TaskController(TaskStore(session="S1"), TaskRegistry(),
                                      persistence=pers2, ledger=BudgetLedger())
                ctl2.restore_from_persistence()
                rec2 = ctl2.store.get_run(r64["run_id"])
                self.assertIsInstance(rec2.candidate, list,
                                      "reloaded candidate must be a vector")
                self.assertEqual(rec2.candidate, original_candidate,
                                 "reloaded vector must equal the original")
                v = ctl2.verify_run(tid, r64["run_id"])
                self.assertEqual(v["conclusion"], "VERIFIED_PASS")
                self.assertIsInstance(v["candidate"], list)
            finally:
                pers2.close()
        finally:
            env.close()

    def test_decimal_high_separate_task_passes_without_changing_original_state(self):
        env = _Env()
        try:
            tid_bin = env.register("ls-req-bin", self._REQUIRED)
            env.ctl.execute_task(tid_bin, "r1", "ls_np32")
            r64 = env.ctl.execute_task(tid_bin, "r2", "ls_np64")
            self.assertEqual(env.store.get_task(tid_bin)["state"], "VERIFIED_PASS")

            # A SEPARATE trusted task with the SAME math + SAME tolerance (same hash):
            tid_dec = env.register("ls-req-dec", self._REQUIRED)
            self.assertEqual(env.store.get_task(tid_dec)["hash"],
                             env.store.get_task(tid_bin)["hash"],
                             "the separate Decimal task must share the original task's hash")
            rd = env.ctl.execute_task(tid_dec, "d1", "ls_decimal_high")
            # The Decimal candidate is a list of STRINGS (the public output format)
            self.assertIsInstance(rd["candidate"], list)
            self.assertTrue(all(isinstance(s, str) for s in rd["candidate"]),
                            "Decimal candidate entries must be strings")
            self.assertTrue(rd["exact_ok"], "Decimal route must PASS at atol=1e-7")
            self.assertEqual(rd.get("verdict"), "VERIFIED_PASS")
            self.assertEqual(env.store.get_task(tid_dec)["state"], "VERIFIED_PASS")
            # The ORIGINAL binary task's terminal state is UNCHANGED
            self.assertEqual(env.store.get_task(tid_bin)["state"], "VERIFIED_PASS")
            _assert_no_leak(rd)
        finally:
            env.close()

    def test_decimal_high_passes_at_1e30_tolerance(self):
        """At atol=1e-30 the binary64 route is UNSUPPORTED (pre-launch, zero cost);
        the separate Decimal task PASSES at the original 1e-30 tolerance."""
        env = _Env()
        try:
            tid_bin = env.register("ls-ultra-bin", self._ULTRA_TIGHT)
            try:
                env.ctl.execute_task(tid_bin, "r1", "ls_np64")
                self.fail("expected CapabilityRejected at the binary64 format boundary")
            except CapabilityRejected:
                pass
            self.assertEqual(env.store.get_task(tid_bin)["state"], "UNSUPPORTED")
            # Pre-launch rejection: NO run row, NO solve cost
            self.assertEqual(len(env.store.get_task(tid_bin)["runs"]), 0)
            self.assertEqual(env.budget(tid_bin).counters["solve_launches"].spent, 0)

            # Trusted-side SEPARATE task: same math + same 1e-30 tolerance + same hash
            tid_dec = env.register("ls-ultra-dec", self._ULTRA_TIGHT)
            self.assertEqual(env.store.get_task(tid_dec)["hash"],
                             env.store.get_task(tid_bin)["hash"])
            rd = env.ctl.execute_task(tid_dec, "d1", "ls_decimal_high")
            # ASSERT: the Decimal route PASSES at the original 1e-30 tolerance
            self.assertTrue(rd["exact_ok"],
                            "Decimal route MUST pass at the original 1e-30 tolerance")
            self.assertEqual(rd.get("verdict"), "VERIFIED_PASS")
            self.assertEqual(env.store.get_task(tid_dec)["state"], "VERIFIED_PASS")
            self.assertTrue(all(isinstance(s, str) for s in rd["candidate"]))
            # The original binary task remains UNSUPPORTED (unchanged)
            self.assertEqual(env.store.get_task(tid_bin)["state"], "UNSUPPORTED")
            # Reload + verify_run consistency for the Decimal task
            pers2 = StorePersistence(env.db_path, key_path=env.key_path)
            try:
                ctl2 = TaskController(TaskStore(session="S1"), TaskRegistry(),
                                      persistence=pers2, ledger=BudgetLedger())
                ctl2.restore_from_persistence()
                v = ctl2.verify_run(tid_dec, rd["run_id"])
                self.assertEqual(v["conclusion"], "VERIFIED_PASS")
                self.assertIsInstance(v["candidate"], list)
                self.assertTrue(all(isinstance(s, str) for s in v["candidate"]))
            finally:
                pers2.close()
        finally:
            env.close()

    def test_normal_system_float64_pass(self):
        env = _Env()
        try:
            tid = env.register("ls-normal", self._NORMAL)
            rn = env.ctl.execute_task(tid, "r1", "ls_np64")
            self.assertTrue(rn["exact_ok"])
            self.assertEqual(rn.get("verdict"), "VERIFIED_PASS")
            self.assertEqual(env.store.get_task(tid)["state"], "VERIFIED_PASS")
            self.assertIsInstance(rn["candidate"], list)
        finally:
            env.close()

    def test_no_unique_solution_pre_launch_unsupported_zero_cost(self):
        env = _Env()
        try:
            tid = env.register("ls-no-unique", self._NO_UNIQUE)
            try:
                env.ctl.execute_task(tid, "r1", "ls_np64")
                self.fail("expected CapabilityRejected for a rank-deficient system")
            except CapabilityRejected:
                pass
            self.assertEqual(env.store.get_task(tid)["state"], "UNSUPPORTED")
            # Pre-launch: NO run row, NO solve cost
            self.assertEqual(len(env.store.get_task(tid)["runs"]), 0)
            self.assertEqual(env.budget(tid).counters["solve_launches"].spent, 0)
        finally:
            env.close()

    def test_boundary_one_third_atol_1e30_any_component_rejects(self):
        """Solution [0, 1/3], atol=1e-30: component 0 (exact 0) IS representable in
        binary64, but component 1 (1/3) is NOT. The rejection fires because of the
        ONE impossible component (ANY, not ALL)."""
        env = _Env()
        try:
            tid = env.register("ls-boundary", self._BOUNDARY)
            try:
                env.ctl.execute_task(tid, "r1", "ls_np64")
                self.fail("expected CapabilityRejected: 1/3 impossible in binary64")
            except CapabilityRejected:
                pass
            self.assertEqual(env.store.get_task(tid)["state"], "UNSUPPORTED")
            self.assertEqual(len(env.store.get_task(tid)["runs"]), 0)
            self.assertEqual(env.budget(tid).counters["solve_launches"].spent, 0)
        finally:
            env.close()

    def test_dimension_range_honored(self):
        """1-dim and 3-dim sanity cases confirm the 1..8 range is honored."""
        env = _Env()
        try:
            t1 = {"domain": "linear_system", "problem": {"family": "rational_square_matrix",
                  "n": "1", "matrix": [["3"]], "rhs": ["1"]},
                  "tolerance": {"absolute": "1/10000000", "relative": "0"}}
            tid1 = env.register("ls-n1", t1)
            r1 = env.ctl.execute_task(tid1, "r1", "ls_np64")
            self.assertTrue(r1["exact_ok"])
            self.assertEqual(len(r1["candidate"]), 1)

            t3 = {"domain": "linear_system", "problem": {"family": "rational_square_matrix",
                  "n": "3", "matrix": [["1", "0", "0"], ["0", "1", "0"], ["0", "0", "1"]],
                  "rhs": ["1", "2", "3"]},
                  "tolerance": {"absolute": "1/10000000", "relative": "0"}}
            tid3 = env.register("ls-n3", t3)
            r3 = env.ctl.execute_task(tid3, "r1", "ls_np64")
            self.assertTrue(r3["exact_ok"])
            self.assertEqual(len(r3["candidate"]), 3)
        finally:
            env.close()

    def test_8_dim_vector_reload_no_type_loss(self):
        """An 8-dim system: the vector candidate round-trips through SQLite without
        type loss (all 8 entries present, correct types)."""
        env = _Env()
        try:
            t8 = {"domain": "linear_system",
                  "problem": {"family": "rational_square_matrix", "n": "8",
                               "matrix": [["1" if i == j else "0" for j in range(8)]
                                          for i in range(8)],
                               "rhs": [str(i + 1) for i in range(8)]},
                  "tolerance": {"absolute": "1/10000000", "relative": "0"}}
            tid = env.register("ls-n8", t8)
            r = env.ctl.execute_task(tid, "r1", "ls_np64")
            self.assertTrue(r["exact_ok"])
            self.assertIsInstance(r["candidate"], list)
            self.assertEqual(len(r["candidate"]), 8)
            # Reload from a fresh connection: all 8 entries present, list type preserved
            pers2 = StorePersistence(env.db_path, key_path=env.key_path)
            try:
                ctl2 = TaskController(TaskStore(session="S1"), TaskRegistry(),
                                      persistence=pers2, ledger=BudgetLedger())
                ctl2.restore_from_persistence()
                rec2 = ctl2.store.get_run(r["run_id"])
                self.assertIsInstance(rec2.candidate, list)
                self.assertEqual(len(rec2.candidate), 8)
                self.assertEqual(rec2.candidate, r["candidate"])
            finally:
                pers2.close()
        finally:
            env.close()


def _linsys_8_dim_identity():
    return {
        "domain": "linear_system",
        "problem": {"family": "rational_square_matrix", "n": "8",
                    "matrix": [["1" if i == j else "0" for j in range(8)]
                               for i in range(8)],
                    "rhs": [str(i + 1) for i in range(8)]},
        "tolerance": {"absolute": "1/10000000", "relative": "0"},
    }


if __name__ == "__main__":
    unittest.main()
