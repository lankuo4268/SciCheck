"""SQLite persistence for tasks, budgets, consumption, requests, and receipts.

Schema (one database file per session; a restart reloads it):
  tasks(task_id, session, hash, canonical_json, state, created_at, PRIMARY KEY(task_id, session))
  runs(run_id, task_id, action, method, candidate, abserr, exact_ok, verdict,
      resource_json, receipt_json, created_at, PRIMARY KEY(run_id))
  budgets(task_id, counters_json, started_at, PRIMARY KEY(task_id))
  requests(request_id, task_id, action_id, run_id, PRIMARY KEY(request_id, task_id))
  receipts(run_id, sealed_json, PRIMARY KEY(run_id))

Invariants:
- Restart does NOT grant new time or budget: budgets.counters_json and started_at
  are reloaded verbatim; an original ACTIVE task is conservatively stopped
  (INCOMPLETE) with its reserved resources persisted.
- A terminal task is never re-opened in place; reopening requires a NEW task id
  with trusted authorization (a new registration), and the old terminal row is
  left untouched.
- The receipt sealing key is kept on the trusted side only; it is NOT stored in
  any model-facing table and never returned by model-facing APIs.
"""
from __future__ import annotations

import json
import sqlite3
import time
from typing import Any, Dict, List, Optional

from receipt_seal import ReceiptSealer, SealerKeyUnavailable
from task_budget import TaskBudget


class StorePersistence:
    def __init__(self, db_path: str, sealer: Optional[ReceiptSealer] = None,
                 key_path: Optional[str] = None):
        """Trusted-key + SQLite open.

        - If `sealer` is provided it is used directly.
        - Else if `key_path` is provided:
            * key file exists           -> load it (restart re-verification);
            * key file ABSENT but the db already has data -> SealerKeyUnavailable.
              A fresh key must NOT be generated to 'cover' unverifiable old receipts.
            * key file absent AND db empty (fresh start) -> generate + write the key.
        - The db + key live in a TRUSTED PRIVATE directory; shareable JSON goes
          elsewhere (e.g. outputs/).
        """
        import os
        self.db_path = db_path
        self.key_path = key_path
        self._conn = sqlite3.connect(db_path)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()
        # Determine whether the db already holds task/run/receipt rows (pre-existing data)
        has_data = self._conn.execute(
            "SELECT COUNT(*) FROM tasks").fetchone()[0] > 0
        if sealer is not None:
            self.sealer = sealer
        elif key_path is not None:
            if os.path.exists(key_path):
                with open(key_path, "rb") as f:
                    key_bytes = f.read()
                if len(key_bytes) != 32:
                    # A weak/empty/truncated key must NOT be accepted: it would make
                    # the sealed receipts forgeable. Refuse and close the connection.
                    self._conn.close()
                    self._conn = None
                    raise SealerKeyUnavailable(
                        f"trusted key file {key_path!r} is {len(key_bytes)} bytes; a valid "
                        f"key is exactly 32 bytes. The database cannot be opened with a "
                        f"weak key.")
                self.sealer = ReceiptSealer(key_bytes)
            else:
                if has_data:
                    # Missing key + old data: refuse, and CLOSE the just-opened
                    # connection so the caller's cleanup (tempfile etc.) does not
                    # hit a locked db file (WinError 32).
                    self._conn.close()
                    self._conn = None
                    raise SealerKeyUnavailable(
                        f"trusted key file {key_path!r} is missing but the database at "
                        f"{db_path!r} already contains data; a fresh key cannot verify the "
                        f"existing sealed receipts. Restore the original key first."
                    )
                s = ReceiptSealer()
                with open(key_path, "wb") as f:
                    f.write(s._key)
                self.sealer = s
        else:
            self.sealer = ReceiptSealer()

    def _init_schema(self) -> None:
        cur = self._conn.cursor()
        cur.executescript(
            """
            CREATE TABLE IF NOT EXISTS tasks (
                task_id TEXT NOT NULL,
                session TEXT NOT NULL,
                hash TEXT NOT NULL,
                canonical_json TEXT NOT NULL,
                state TEXT NOT NULL,
                created_at REAL NOT NULL,
                PRIMARY KEY (task_id, session)
            );
            CREATE TABLE IF NOT EXISTS runs (
                run_id TEXT NOT NULL,
                task_id TEXT NOT NULL,
                action TEXT NOT NULL,
                method TEXT NOT NULL,
                candidate REAL,
                candidate_json TEXT,
                abserr REAL,
                exact_ok INTEGER,
                verdict TEXT NOT NULL,
                resource_json TEXT,
                receipt_json TEXT,
                created_at REAL NOT NULL,
                PRIMARY KEY (run_id)
            );
            CREATE TABLE IF NOT EXISTS budgets (
                task_id TEXT NOT NULL,
                counters_json TEXT NOT NULL,
                started_at REAL NOT NULL,
                PRIMARY KEY (task_id)
            );
            CREATE TABLE IF NOT EXISTS requests (
                request_id TEXT NOT NULL,
                task_id TEXT NOT NULL,
                action_id TEXT NOT NULL,
                run_id TEXT NOT NULL,
                PRIMARY KEY (request_id, task_id, action_id)
            );
            CREATE TABLE IF NOT EXISTS receipts (
                run_id TEXT NOT NULL,
                sealed_json TEXT NOT NULL,
                PRIMARY KEY (run_id)
            );
            CREATE TABLE IF NOT EXISTS task_terminal_reasons (
                task_id TEXT NOT NULL,
                session TEXT NOT NULL,
                reason TEXT NOT NULL,
                PRIMARY KEY (task_id, session)
            );
            """
        )
        self._conn.commit()
        self._migrate_runs_exact_ok_nullable()

    def _migrate_runs_exact_ok_nullable(self) -> None:
        """Migration for existing databases created BEFORE the tri-state ODE verdict.

        Old databases declare `runs.exact_ok INTEGER NOT NULL`, which rejects the
        legitimate NULL (INCONCLUSIVE) that a real ODE run now writes. Python has
        been updated, but the on-disk schema must be migrated too. The migration
        rebuilds ONLY the `runs` table (the one whose constraint changed) and
        preserves every existing row -- run rows, receipts, requests, budgets and
        the key are all left intact; nothing is deleted. It is a no-op on fresh
        databases (which already carry the nullable column) and on databases that
        have already been migrated. SQLite < 3.35 cannot drop a column constraint
        in place, so the standard create-new / copy / drop / rename procedure is
        used. The migration is idempotent and guarded: if the constraint is
        already absent, nothing is done.
        """
        row = self._conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='runs'"
        ).fetchone()
        if row is None:
            return  # table does not exist yet; the fresh schema is already nullable
        sql = row["sql"] or ""
        # "NOT NULL" appears exactly on the exact_ok line in the old schema. If it
        # is absent, the table is already migrated (or fresh) -> do nothing.
        if "exact_ok INTEGER NOT NULL" not in sql:
            return
        cur = self._conn.cursor()
        try:
            cur.execute("BEGIN IMMEDIATE")
            cur.execute(
                "CREATE TABLE runs_new ("
                "run_id TEXT NOT NULL, task_id TEXT NOT NULL, action TEXT NOT NULL, "
                "method TEXT NOT NULL, candidate REAL, candidate_json TEXT, abserr REAL, "
                "exact_ok INTEGER, verdict TEXT NOT NULL, resource_json TEXT, "
                "receipt_json TEXT, created_at REAL NOT NULL, PRIMARY KEY (run_id))"
            )
            cur.execute(
                "INSERT INTO runs_new(run_id, task_id, action, method, candidate, "
                "candidate_json, abserr, exact_ok, verdict, resource_json, receipt_json, "
                "created_at) "
                "SELECT run_id, task_id, action, method, candidate, candidate_json, abserr, "
                "exact_ok, verdict, resource_json, receipt_json, created_at FROM runs"
            )
            cur.execute("DROP TABLE runs")
            cur.execute("ALTER TABLE runs_new RENAME TO runs")
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise

    # ---- tasks ----

    def insert_task(self, task_id, session, task_hash, canonical, state="ACTIVE") -> None:
        self._conn.execute(
            "INSERT INTO tasks(task_id, session, hash, canonical_json, state, created_at) "
            "VALUES (?,?,?,?,?,?)",
            (task_id, session, task_hash, json.dumps(canonical, sort_keys=True),
             state, time.time()),
        )
        self._conn.commit()

    def set_task_state(self, task_id, session, state, terminal_reason="") -> None:
        self._conn.execute(
            "UPDATE tasks SET state=? WHERE task_id=? AND session=?",
            (state, task_id, session),
        )
        if terminal_reason:
            self._conn.execute(
                "INSERT OR REPLACE INTO task_terminal_reasons(task_id, session, reason) VALUES (?,?,?)",
                (task_id, session, terminal_reason),
            )
        self._conn.commit()

    def get_task_state(self, task_id, session) -> Optional[str]:
        row = self._conn.execute(
            "SELECT state FROM tasks WHERE task_id=? AND session=?", (task_id, session)
        ).fetchone()
        return row["state"] if row else None

    def list_task_ids(self, session) -> List[str]:
        rows = self._conn.execute(
            "SELECT task_id FROM tasks WHERE session=?", (session,)
        ).fetchall()
        return [r["task_id"] for r in rows]

    # ---- budgets (reload on restart; no fresh time/budget granted) ----

    def save_budget(self, task_id, budget: TaskBudget) -> None:
        """Persist a budget snapshot. tool_requests.spent is preserved at the
        DB's recorded max (the durable authoritative counter), preventing a stale
        local solver counter from overwriting it. Other counters (solve_launches,
        func_evals, model_requests, wall_seconds) are persisted as given; their
        totals are never changed."""
        cur = self._conn.cursor()
        try:
            cur.execute("BEGIN IMMEDIATE")
            row = cur.execute(
                "SELECT counters_json FROM budgets WHERE task_id=?", (task_id,)
            ).fetchone()
            if row is not None:
                existing = json.loads(row["counters_json"])
                existing_tool_spent = existing.get("tool_requests", {}).get("spent", 0)
                local_tool_spent = budget.counters["tool_requests"].spent
                if existing_tool_spent > local_tool_spent:
                    budget.counters["tool_requests"].spent = existing_tool_spent
            cur.execute(
                "INSERT OR REPLACE INTO budgets(task_id, counters_json, started_at) VALUES (?,?,?)",
                (task_id, json.dumps(budget.as_dict()), budget.started_at),
            )
            self._conn.commit()
        except sqlite3.IntegrityError:
            self._conn.rollback()
            raise
        except Exception:
            self._conn.rollback()
            raise

    def load_budget(self, task_id) -> Optional[TaskBudget]:
        row = self._conn.execute(
            "SELECT counters_json, started_at FROM budgets WHERE task_id=?", (task_id,)
        ).fetchone()
        if row is None:
            return None
        return TaskBudget.from_dict(task_id, json.loads(row["counters_json"]))

    # ---- atomic per-task tool-request increment (cross-connection authority) ----

    def increment_tool_request(self, task_id):
        """Atomically increment the durable tool_requests.spent for a task in ONE
        SQLite transaction and return the re-read budget. Using a single
        BEGIN IMMEDIATE + read + write + commit makes two independent connections
        each persist their own charge without a lost update (no read-modify-write
        race). Returns None when the task has no durable budget row yet.
        Raises BudgetExhausted (from task_budget) when the increment would exceed
        the total, so the caller can mark the task terminal and stop.
        All exceptions in the transaction are rolled back."""
        cur = self._conn.cursor()
        try:
            cur.execute("BEGIN IMMEDIATE")
            row = cur.execute(
                "SELECT counters_json, started_at FROM budgets WHERE task_id=?",
                (task_id,),
            ).fetchone()
            if row is None:
                self._conn.rollback()
                return None
            budget = TaskBudget.from_dict(task_id, json.loads(row["counters_json"]))
            budget.counters["tool_requests"].spend(1)
            cur.execute(
                "INSERT OR REPLACE INTO budgets(task_id, counters_json, started_at) VALUES (?,?,?)",
                (task_id, json.dumps(budget.as_dict()), budget.started_at),
            )
            self._conn.commit()
        except sqlite3.IntegrityError:
            self._conn.rollback()
            raise
        except Exception:
            self._conn.rollback()
            raise
        return budget
    # ---- receipt + run + request atomics: all in one transaction ----

    def commit_run_with_receipt(self, rec: Dict[str, Any], receipt: Dict[str, Any],
                                request_id: str, task_id: str, action_id: str,
                                run_id: str) -> None:
        """Persist a run row + its sealed receipt atomically. The request claim was
        already taken by claim_request() before the launch; here we bind the run row
        and the sealed receipt in ONE transaction so a concurrent duplicate can only
        ever start ONE solve. The sealed receipt binds task_hash + session +
        action_id so a restart can re-verify with the same key."""
        sealed = self.sealer.seal(receipt)
        cur = self._conn.cursor()
        cur.execute("BEGIN IMMEDIATE")
        try:
            # A VECTOR candidate (linear-system solution) is stored as JSON so its
            # type survives; the scalar candidate column stays NULL for vectors.
            cand = rec.get("candidate")
            is_vector = isinstance(cand, (list, tuple))
            # Preserve the ACTUAL value type in the JSON (floats for the binary routes,
            # strings for the Decimal high-precision route); never force-cast a string
            # candidate to float here -- that would destroy the high-precision value.
            cand_json = json.dumps(list(cand)) if is_vector else None
            cand_scalar = None if is_vector else cand
            # exact_ok is TRI-STATE: True->1, False->0, None->NULL (an INCONCLUSIVE
            # ODE run must NOT be coerced to 0, which would read back as a
            # spurious VERIFIED_INCORRECT verdict on reload).
            exact_ok_val = rec.get("exact_ok")
            exact_ok_sql = 1 if exact_ok_val is True else (0 if exact_ok_val is False else None)
            cur.execute(
                "INSERT INTO runs(run_id, task_id, action, method, candidate, candidate_json, "
                "abserr, exact_ok, verdict, resource_json, receipt_json, created_at) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    run_id, task_id, action_id, rec.get("method", action_id),
                    cand_scalar, cand_json,
                    rec.get("quad_abserr_reported"),
                    exact_ok_sql,
                    rec.get("verdict", ""), json.dumps(rec.get("resource", {})),
                    json.dumps(sealed, sort_keys=True), time.time(),
                ),
            )
            cur.execute(
                "INSERT OR REPLACE INTO receipts(run_id, sealed_json) VALUES (?,?)",
                (run_id, json.dumps(sealed, sort_keys=True)),
            )
            # Rebind the durable request row that claim_request() created (with the
            # same run_id); save_request() would overwrite it with a different one if
            # called here — we must keep the original claim row's run_id.
            cur.execute(
                "INSERT OR REPLACE INTO requests(request_id, task_id, action_id, run_id) VALUES (?,?,?,?)",
                (request_id, task_id, action_id, run_id),
            )
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise

    # ---- requests (idempotency) ----

    def save_request(self, request_id, task_id, action_id, run_id) -> None:
        """Idempotency index is scoped by (session, task_id, request_id): a request
        in one task NEVER affects the same request_id in another task."""
        self._conn.execute(
            "INSERT OR REPLACE INTO requests(request_id, task_id, action_id, run_id) VALUES (?,?,?,?)",
            (request_id, task_id, action_id, run_id),
        )
        self._conn.commit()

    def lookup_request(self, request_id, task_id) -> Optional[Dict[str, Any]]:
        row = self._conn.execute(
            "SELECT request_id, action_id, run_id FROM requests WHERE request_id=? AND task_id=?",
            (request_id, task_id),
        ).fetchone()
        return dict(row) if row else None

    def request_action_conflict(self, request_id, task_id, new_action_id) -> Optional[str]:
        """Return the previously-bound action for request_id in this task, or None."""
        row = self._conn.execute(
            "SELECT action_id FROM requests WHERE request_id=? AND task_id=?",
            (request_id, task_id),
        ).fetchone()
        return row["action_id"] if row else None

    # ---- atomic claim: the REAL concurrency guard for a (task_id, request_id) ----

    def claim_request(self, task_id, request_id, action_id, run_id) -> bool:
        """Atomically claim a (task_id, request_id) BEFORE launching any solve.

        Uses a SQLite transaction with INSERT OR IGNORE so that, under true
        concurrency, exactly ONE caller wins the claim and starts the solve; any
        other concurrent caller for the SAME (task_id, request_id, action_id)
        finds the row already present and returns False (a PENDING / re-entrant
        hit) instead of launching a second solve. A DIFFERENT action_id for the
        same request_id in the same task is a conflict and is refused by the
        caller; a different task with the same request_id is independent.
        Returns True when THIS caller won the claim (and persisted the run_id
        binding), False when the claim was already taken.
        """
        cur = self._conn.cursor()
        try:
            cur.execute("BEGIN IMMEDIATE")
            cur.execute(
                "INSERT OR IGNORE INTO requests(request_id, task_id, action_id, run_id) "
                "VALUES (?,?,?,?)", (request_id, task_id, action_id, run_id),
            )
            self._conn.commit()
        except sqlite3.IntegrityError:
            self._conn.rollback()
            return False
        # The claim is won only if the row we just inserted still maps to OUR run_id
        row = self._conn.execute(
            "SELECT run_id FROM requests WHERE request_id=? AND task_id=? AND action_id=?",
            (request_id, task_id, action_id),
        ).fetchone()
        return row is not None and row["run_id"] == run_id

    def claimed_run_id(self, task_id, request_id, action_id) -> Optional[str]:
        """If a (task_id, request_id, action_id) is already claimed (by this or
        another connection), return the bound run_id so a re-entrant / concurrent
        caller can reuse it instead of launching a second solve."""
        row = self._conn.execute(
            "SELECT run_id FROM requests WHERE request_id=? AND task_id=? AND action_id=?",
            (request_id, task_id, action_id),
        ).fetchone()
        return row["run_id"] if row else None

    # ---- runs ----

    def insert_run(self, rec: Dict[str, Any]) -> None:
        # A VECTOR candidate (linear-system) is stored as JSON so its type survives;
        # the scalar candidate column stays NULL for vectors.
        cand = rec.get("candidate")
        is_vector = isinstance(cand, (list, tuple))
        # Preserve the ACTUAL value type in the JSON (floats for the binary routes,
        # strings for the Decimal high-precision route, lists for vectors);
        # a None candidate (an ODE solver that did not reach t1) is stored as
        # NULL in BOTH columns -- never coerced to NaN.
        cand_json = json.dumps(list(cand)) if is_vector else None
        cand_scalar = None if is_vector else cand
        # exact_ok is TRI-STATE: True->1, False->0, None->NULL (an INCONCLUSIVE
        # ODE run must NOT be coerced to 0, which would read back as a
        # spurious VERIFIED_INCORRECT verdict on reload).
        exact_ok_val = rec.get("exact_ok")
        exact_ok_sql = 1 if exact_ok_val is True else (0 if exact_ok_val is False else None)
        self._conn.execute(
            "INSERT INTO runs(run_id, task_id, action, method, candidate, candidate_json, abserr, "
            "exact_ok, verdict, resource_json, receipt_json, created_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                rec["run_id"], rec["task_id"], rec["action"], rec.get("method", rec["action"]),
                cand_scalar, cand_json, rec.get("quad_abserr_reported"),
                exact_ok_sql, rec.get("verdict", ""), json.dumps(rec.get("resource", {})),
                json.dumps(rec.get("receipt", {})), time.time(),
            ),
        )
        self._conn.commit()

    def runs_for_task(self, task_id) -> List[Dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT run_id, task_id, action, method, candidate, candidate_json, abserr, "
            "exact_ok, verdict, resource_json, receipt_json FROM runs WHERE task_id=? "
            "ORDER BY created_at",
            (task_id,),
        ).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            # TRI-STATE exact_ok: DB 1 -> True, DB 0 -> False, DB NULL -> None
            # (an INCONCLUSIVE ODE run must reload as None, NOT as a spurious
            # False that would read back as a VERIFIED_INCORRECT verdict).
            if d["exact_ok"] is None:
                d["exact_ok"] = None
            else:
                d["exact_ok"] = bool(d["exact_ok"])
            d["resource"] = json.loads(d.pop("resource_json") or "{}")
            d["receipt"] = json.loads(d.pop("receipt_json") or "{}")
            # Rebuild a VECTOR candidate from its JSON so the type is preserved on
            # reload (a linear-system run carries a list, not a scalar).
            cand_json = d.pop("candidate_json", None)
            if cand_json is not None:
                d["candidate"] = json.loads(cand_json)
            out.append(d)
        return out

    # ---- sealed receipts (trusted-side; integrity key stays here) ----

    def save_receipt(self, run_id, receipt: Dict[str, Any]) -> None:
        sealed = self.sealer.seal(receipt)
        self._conn.execute(
            "INSERT OR REPLACE INTO receipts(run_id, sealed_json) VALUES (?,?)",
            (run_id, json.dumps(sealed, sort_keys=True)),
        )
        self._conn.commit()

    def load_receipt(self, run_id) -> Optional[Dict[str, Any]]:
        row = self._conn.execute(
            "SELECT sealed_json FROM receipts WHERE run_id=?", (run_id,)
        ).fetchone()
        if row is None:
            return None
        sealed = json.loads(row["sealed_json"])
        self.sealer.assert_integrity(sealed)  # trusted-side verification; raises ReceiptTampered
        return sealed

    # ---- restart reload: no fresh time/budget is granted ----

    def load_task(self, task_id, session) -> Optional[Dict[str, Any]]:
        """Reload a task + its budget verbatim on restart. Returns None if absent.
        The budget's started_at and counters are restored exactly; no extension."""
        row = self._conn.execute(
            "SELECT canonical_json, hash, state FROM tasks WHERE task_id=? AND session=?",
            (task_id, session),
        ).fetchone()
        if row is None:
            return None
        budget = self.load_budget(task_id)
        return {
            "task_id": task_id,
            "session": session,
            "hash": row["hash"],
            "canonical": json.loads(row["canonical_json"]),
            "state": row["state"],
            "budget": budget,  # reloaded verbatim; not re-extended
        }

    def close(self) -> None:
        if self._conn is not None:
            try:
                self._conn.close()
            finally:
                self._conn = None
