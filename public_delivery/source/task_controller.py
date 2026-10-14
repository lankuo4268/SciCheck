"""Task controller: method_lookup / execute_task / verify_run over the task store.

Guarantees:
- execute_task starts EXACTLY ONE candidate computation per call. No hidden
  auto-repair inside; a VERIFIED_INCORRECT verdict keeps the task ACTIVE, and the
  NEXT explicit execute_task call starts the repair.
- Idempotency: the same (request_id, action_id) returns the same run_id and does
  NOT re-solve; a different action for the same request_id is rejected. Both the
  idempotent hit and the rejection are rejected/replayed, but each controller call
  still counts one tool_requests.
- Per-task budgets: each task registers its OWN TaskBudget (keyed by task_id, in
  a BudgetLedger). The solve entrypoint consumes ONLY solve_launches; a launch is
  committed at start and NOT refunded on exception. Budget exhaustion -> INCOMPLETE.
- execute_task does NOT register a model request (a local solve is not a model
  request). Real model requests are registered independently by the trusted
  adapter; until that AGH wiring exists we only prove the local interface.
- Every method_lookup / execute_task / verify_run call counts one tool_request.
- Function callbacks are gated: each integrand call checks deadline + func_evals
  budget before evaluating and counts one real evaluation (BudgetGatedQuad).
- Before publishing a VERIFIED_PASS receipt the deadline is re-checked; an expired
  deadline cannot leave a VERIFIED_PASS receipt.
- Terminal states: VERIFIED_PASS, UNSUPPORTED, INCOMPLETE, INCONCLUSIVE,
  EXECUTION_ERROR. VERIFIED_INCORRECT is a run verdict, not a numeric pass.
- Restart: no fresh time/budget; original ACTIVE tasks conservatively stop as
  INCOMPLETE with reserved resources persisted. Reopening a terminal task requires
  a NEW task id with trusted authorization.
- verify_run reads + verifies the sealed trusted receipt; it never re-solves and
  never exposes the reference answer to the model side. Model-facing views strip
  private reference fields, recursively over dict and list.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from fractions import Fraction

import scicheck
from scicheck import (
    normalize_problem,
    run_plain_quad,
    run_quad_with_breakpoints,
    exact_ok,
    BudgetGatedQuad,
)
from method_registry import MethodLookupError, TaskRegistry, MethodCard
from task_budget import (
    BudgetLedger,
    BudgetExhausted as BudgetExhaustedErr,
    DeadlineExpired,
    TaskBudget,
)
from receipt_seal import make_receipt, ReceiptTampered, ReceiptFieldMissing, REQUIRED_RECEIPT_FIELDS
from task_store import (
    RunRecord,
    scicheck_run_to_record,
    TaskAlreadyTerminal,
    TaskFrozen,
    TaskSessionMismatch,
    TaskStore,
    TERMINAL_STATES,
)
from task_persistence import StorePersistence
from receipt_seal import ReceiptSealer
import json
import os


class TaskRegistrar:
    """The single trusted entrypoint for registering a task. It bundles the pieces
    that a model-facing tool must NEVER do individually:
      - TaskStore.register_task: normalize + validate + deep-freeze;
      - StorePersistence.insert_task: SQLite task row;
      - BudgetLedger.for_task: this task's OWN per-task budget entry + start time;
      - persist_budget: write the budget counters to SQLite immediately.

    The trusted-side receipt-sealing key is generated ONCE, on a trusted private
    path (key_path), and reloaded on restart so old receipts re-verify. Absent key
    file with old records means new key CANNOT accept them; the caller is expected
    to keep the key file with the db.
    """

    def __init__(self, session: str, db_path: str, key_path: str,
                 registry: Optional[TaskRegistry] = None,
                 store: Optional[TaskStore] = None,
                 persistence: Optional[StorePersistence] = None,
                 ledger: Optional[BudgetLedger] = None):
        self.session = session
        self.db_path = db_path
        self.key_path = key_path
        self.store = store if store is not None else TaskStore(session=session)
        self.persistence = persistence if persistence is not None else \
            StorePersistence(db_path, key_path=key_path)
        self.ledger = ledger if ledger is not None else BudgetLedger()
        self.registry = registry or TaskRegistry()
        self.controller = TaskController(
            self.store, self.registry,
            persistence=self.persistence, ledger=self.ledger,
        )

    def register_task(self, task_id: str, raw: Dict[str, Any],
                     totals: Optional[Dict[str, int]] = None) -> Dict[str, Any]:
        """ONE trusted registration: normalize + freeze + SQLite row + per-task
        BudgetLedger entry + persisted start time. The business tools cannot call
        this -- it is the registrar's API."""
        return self.controller.trusted_register_task(task_id, raw, totals)

    # Convenience delegations so the model gets only the three business entries.
    def execute_task(self, task_id, request_id, action_id):
        return self.controller.execute_task(task_id, request_id, action_id)

    def verify_run(self, task_id, run_id):
        return self.controller.verify_run(task_id, run_id)

    def task_view(self, task_id):
        return self.controller.task_view(task_id)

    def method_lookup(self, task_id, card_id):
        return self.controller.method_lookup(task_id, card_id)

    def restart_recover(self, task_ids=None):
        return self.controller.restart_recover(task_ids)

    def reopen_terminal(self, new_task_id, old_task_id, authorize=True):
        return self.controller.reopen_terminal(new_task_id, old_task_id, authorize)

    def restore(self):
        """Real reopen of an EXISTING database: restore tasks/budgets/runs/requests
        from SQLite, conservatively stop still-ACTIVE tasks, keep terminal tasks.
        A db without its original key file is refused by the persistence layer.
        """
        return self.controller.restore_from_persistence(stop_active=True)



# Private reference fields that must never reach model-facing views.
_FORBIDDEN_MODEL_FIELDS = {
    "reference_fraction", "reference_answer", "reference_value",
    "exact_check", "candidate_as_fraction", "abs_diff_exact_fraction",
    "abs_diff_fraction", "allowed_error_fraction",
    # ODE private reference / interval fields (never exposed to the model side):
    "reference_lo_fraction", "reference_hi_fraction", "reference_frac",
    "reference_envelope_fraction", "reference_finite", "reference_exact",
    "allowed_error_min_fraction", "allowed_error_max_fraction",
    "forward_error_min_fraction", "forward_error_max_fraction",
    "applicable_range", "candidate_float", "candidate_as_fraction",
    "reference_lo_frac", "reference_hi_frac", "envelope_frac", "reference_frac",
}


def _strip_model_view(d) -> Any:
    """Recursively strip private reference fields from a dict/list tree so the
    model-facing view never carries the reference answer. The trusted control
    side keeps the original receipt untouched."""
    if isinstance(d, dict):
        out = {}
        for k, v in d.items():
            if k in _FORBIDDEN_MODEL_FIELDS:
                continue
            out[k] = _strip_model_view(v)
        return out
    if isinstance(d, list):
        return [_strip_model_view(x) for x in d]
    return d


def _model_run_view(record: RunRecord) -> Dict[str, Any]:
    check = record.exact_check
    return _strip_model_view({
        "run_id": record.run_id,
        "action": record.action,
        "method": getattr(record, "method", record.action),
        "candidate": record.candidate,
        "quad_abserr_reported": record.quad_abserr_reported,
        "exact_ok": record.exact_ok,
        "error_vs_allowed": {
            "abs_diff_display_float": check.get("abs_diff_display_float"),
            "allowed_error_display_float": check.get("allowed_error_display_float"),
        },
        "resource": record.resource,
        "cost_note": "one solver launch per execute_task call",
    })


class TaskError(Exception):
    pass


class TaskNotFound(TaskError):
    pass


class ActionNotAllowed(TaskError):
    pass


class CapabilityRejected(TaskError):
    pass


class BudgetExhausted(TaskError):
    pass


class NoTaskBudget(TaskError):
    """Raised when a task has no registered per-task budget ledger."""


# Backward-compat alias: 'no ledger' surfaces as NoTaskBudget (a TaskError subclass),
# not a silent pass-through.
class TaskNotFoundBudget(TaskError):
    pass


# ---- task registration: the single trusted entry ----


class RequestActionConflict(TaskError):
    pass


class DeadlineExpiredError(TaskError):
    pass


class TaskController:
    def __init__(self, store: TaskStore, registry: Optional[TaskRegistry] = None,
                 persistence: Optional[StorePersistence] = None,
                 ledger: Optional[BudgetLedger] = None):
        self.store = store
        self.registry = registry or TaskRegistry()
        self.persistence = persistence
        # Per-task budget ledger. budget=None is NOT a license to solve unboundedly:
        # a launch with no ledger budget is refused (capability/budget gate).
        self.ledger = ledger if ledger is not None else BudgetLedger()
        # Serializes the idempotency critical section on THIS controller so two
        # concurrent requests on the same controller cannot both launch a solve.
        # Cross-connection concurrency is additionally guarded by the SQLite claim.
        import threading
        self._idempotency_lock = threading.RLock()

    # ---- authoritative persisted-budget sync (prevents split-brain accounting) ----

    def _reload_budget(self, task_id: str) -> Optional[TaskBudget]:
        """Re-read the durable (SQLite) budget for a task and rebind it in-memory.
        Different connections each keep their own in-memory ledger, so before any
        tool/solve charge the persisted counter is the authority; without this, two
        connections could each save one tool-request charge and lose the other."""
        if not self.persistence:
            return self.ledger.get(task_id)
        persisted = self.persistence.load_budget(task_id)
        if persisted is not None:
            self.ledger._by_task[task_id] = persisted
        return self.ledger.get(task_id)

    def _persist_budget_now(self, task_id: str) -> None:
        """Persist the in-memory budget to SQLite so the durable state is authoritative."""
        if self.persistence:
            self.persistence.save_budget(task_id, self.ledger.get(task_id))

    # Backward-compatible alias (older call sites referenced the previous name).
    _persist_budget = _persist_budget_now

    # ---- trusted register_task: one entry does normalize + freeze + SQLite +
    # per-task BudgetLedger creation + start-time persistence ----

    def trusted_register_task(self, task_id: str, raw: Dict[str, Any],
                              totals: Optional[Dict[str, int]] = None) -> Dict[str, Any]:
        """The single trusted registration entrypoint. It:
          1. normalizes + validates + deep-freezes the task in the store;
          2. registers the task row in SQLite (when persistence is configured);
          3. creates THIS task's own BudgetLedger entry and persists the start time.
        A task with no ledger entry cannot solve (execute_task refuses). The
        controller's business tools never backfill a missing budget or grant time.
        """
        t = self.store.register_task(task_id, raw)  # normalize + freeze here
        if self.persistence:
            self.persistence.insert_task(task_id, self.store.session, t["hash"], t["canonical"])
        budget = self.ledger.for_task(task_id, totals)  # per-task, own ledger entry
        self._persist_budget(task_id)  # save start time + counters immediately
        return t

    # ---- restart restore: reload tasks/budgets/runs/requests from an EXISTING db ----

    def restore_from_persistence(self, stop_active: bool = False) -> Dict[str, str]:
        """On reopening an existing database, restore the in-memory store from
        SQLite: tasks, per-task budgets (with the ORIGINAL started_at, no fresh
        time/budget), runs, and the idempotency request index.

        By default this is a PURE reload -- it does NOT change any task's state, so
        a live concurrent worker can continue work on a still-ACTIVE task. Pass
        stop_active=True ONLY on a genuine process restart: then still-ACTIVE tasks
        are conservatively stopped as INCOMPLETE (reserved units persisted, no
        top-up) while completed (terminal) tasks keep their original state and
        remain verifiable + idempotently readable. The db binds to a SINGLE session;
        a db carrying another session's rows is refused.
        """
        if not self.persistence:
            raise TaskError("restore_from_persistence requires a configured persistence layer")
        session = self.store.session
        restored: Dict[str, str] = {}
        for tid in self.persistence.list_task_ids(session):
            loaded = self.persistence.load_task(tid, session)
            if loaded is None:
                continue
            if tid in self.store._tasks:
                continue  # already registered in-memory; leave it
            # register WITHOUT re-freezing: reuse the persisted canonical + hash
            self.store._tasks[tid] = {
                "task_id": tid,
                "session": session,
                "hash": loaded["hash"],
                "canonical": loaded["canonical"],
                "state": loaded["state"],
                "runs": [],
            }
            # Restore the per-task budget verbatim (original started_at, no top-up)
            if loaded.get("budget") is not None:
                self.ledger._by_task[tid] = loaded["budget"]
            # Restore runs + their durable request bindings (the runs table carries
            # no request_id column; the requests table is the durable index).
            for run_d in self.persistence.runs_for_task(tid):
                run_id = run_d["run_id"]
                rec = self._run_from_persisted(run_d, tid)
                self.store._runs[run_id] = rec
                self.store._tasks[tid]["runs"].append(run_id)
            for req_row in self._request_bindings(self.persistence, tid):
                req_id, action_id, run_row_id = req_row
                if run_row_id in self.store._runs:
                    self.store.record_request(tid, req_id, action_id, run_row_id)
            # Conservative stop of still-ACTIVE tasks ONLY on a genuine restart.
            if stop_active and self.store._tasks[tid]["state"] not in TERMINAL_STATES:
                self.store.mark_terminal(tid, "INCOMPLETE", reason="restart conservative stop")
                self._persist_budget(tid)
                self._persist_terminal(tid, "INCOMPLETE")
                self.persistence.set_task_state(tid, session, "INCOMPLETE",
                                                 "restart conservative stop")
                restored[tid] = "INCOMPLETE"
            else:
                restored[tid] = self.store._tasks[tid]["state"]
        return restored

    @staticmethod
    def _request_bindings(persistence, task_id):
        """Yield (request_id, action_id, run_id) tuples for a task from SQLite."""
        rows = persistence._conn.execute(
            "SELECT request_id, action_id, run_id FROM requests WHERE task_id=?",
            (task_id,),
        ).fetchall()
        return [(r["request_id"], r["action_id"], r["run_id"]) for r in rows]

    # ---- per-task budget accessor ----

    def budget_for(self, task_id: str) -> TaskBudget:
        b = self.ledger.get(task_id)
        if b is not None:
            return b
        raise TaskNotFound(
            f"no per-task budget ledger for {task_id}; a trusted_register_task is "
            f"required first -- business tools never create budgets lazily"
        )

    # ---- method_lookup (management interface) ----
    # card_id is 'structure' or 'budget'. Both first do a real session/task
    # binding check, then return the applicable action_ids and the task's total
    # remaining balance. Switching cards does NOT top up budget. Each call
    # counts one tool_request and MUST stop (not fall through) if that budget
    # is exhausted.

    def method_lookup(self, task_id: str, card_id: str) -> Dict[str, Any]:
        t = self._get_task_bound(task_id)  # real session/task binding FIRST
        b = self._charge_tool_request(task_id)  # count one tool request; hard-stop if exhausted
        if card_id not in ("structure", "budget"):
            raise ActionNotAllowed(f"card_id must be 'structure' or 'budget', got {card_id!r}")
        if card_id == "structure":
            applicable = [
                c.card_id for c in self.registry.all_cards().values()
                if self._action_compatible(c, t["canonical"])
            ]
            # DIFFERENT units are NOT summed; list each resource with its own balance
            budget_by_resource = b.budget_card() if b else {}
            return {
                "card_id": "structure",
                "task_id": task_id,
                "applicable_action_ids": applicable,
                "budget_by_resource": budget_by_resource,
                "note": "structure card lists the applicable trusted action_ids; resource "
                        "units are listed per resource and never summed",
            }
        # budget card: per-resource totals + balances, never a single cross-unit total
        return {
            "card_id": "budget",
            "task_id": task_id,
            "budget_by_resource": b.budget_card() if b else {},
            "note": "budget card returns each resource's total/balance individually; "
                    "switching cards does not top up budget",
        }

    def _charge_tool_request(self, task_id: str) -> TaskBudget:
        """Every controller call counts one tool request, charged to the task's own
        ledger. The charge is applied to the IN-MEMORY ledger object and then
        persisted to SQLite immediately, so the per-task counter the caller holds
        (e.g. env.budget(tid)) always sees the increment. A task with no ledger
        entry is a real no-budget condition (NoTaskBudget), not a silent pass.

        Cross-CONNECTION accounting is handled separately by the SQLite claim / the
        durable runs table: a fresh connection on the same db re-reads the winner's
        run and reuses it instead of re-solving. The per-connection in-memory
        counter is the authoritative view for the connections' own calls.
        """
        # Use the atomic persistence-layer increment as the authoritative cross-connection
        # tool-request counter. Without this, two independent SQLite connections could
        # each read spent=0, each write spent=1, and the second's charge would be lost.
        # With persistence: increment in-transaction (BEGIN IMMEDIATE) and sync the
        # durable counter value IN-PLACE on the existing local object so all local
        # callers (including unit tests that captured b = env.budget(tid)) see it.
        # Without persistence: local-only charge as before.
        if self.persistence:
            try:
                reloaded = self.persistence.increment_tool_request(task_id)
            except BudgetExhaustedErr as e:
                # Tool budget exhausted. A still-ACTIVE task conservatively stops
                # INCOMPLETE with the exhausted budget persisted. A task already
                # terminal keeps its original verdict (no re-mark). Either way the
                # caller is rejected with BudgetExhausted.
                if not self.store.is_terminal(task_id):
                    self.store.mark_terminal(task_id, "INCOMPLETE",
                                            reason=f"tool-request budget exhausted: {e}")
                    self._persist_budget(task_id)
                    self._persist_terminal(task_id, "INCOMPLETE")
                self._persist_budget(task_id)
                raise BudgetExhausted(
                    f"tool-request budget exhausted on task {task_id}: {e}") from e
            if reloaded is None:
                raise NoTaskBudget(
                    f"no per-task budget ledger for {task_id}; trusted_register_task required"
                )
            # Sync the durable counter value IN-PLACE on the existing local object
            # so that any caller that already holds a reference to the local TaskBudget
            # sees the update (no stale-object problem). The total stays the same.
            local = self.ledger.get(task_id)
            if local is not None:
                for k, c in reloaded.counters.items():
                    if k in local.counters:
                        local.counters[k].spent = c.spent
                        local.counters[k].reserved = c.reserved
            else:
                self.ledger._by_task[task_id] = reloaded
            b = local if local is not None else reloaded
        else:
            b = self.ledger.get(task_id)
            if b is None:
                raise NoTaskBudget(
                    f"no per-task budget ledger for {task_id}; trusted_register_task required"
                )
            try:
                c = b.charge_tool_request()
            except BudgetExhaustedErr as e:
                if not self.store.is_terminal(task_id):
                    self.store.mark_terminal(task_id, "INCOMPLETE",
                                            reason=f"tool-request budget exhausted: {e}")
                    self._persist_budget(task_id)
                    self._persist_terminal(task_id, "INCOMPLETE")
                self._persist_budget(task_id)
                raise BudgetExhausted(
                    f"tool-request budget exhausted on task {task_id}: {e}") from e
        return b

    # ---- execute_task: exactly one candidate launch ----

    def execute_task(self, task_id: str, request_id: str, action_id: str) -> Dict[str, Any]:
        t = self._get_task_bound(task_id)
        # Serialize the whole idempotency->claim->launch critical section on THIS
        # controller so two concurrent requests on the SAME instance cannot both win
        # the claim and launch two solves. Different SQLite connections are additionally
        # guarded by the durable claim (claim_request), which is the true authority.
        with self._idempotency_lock:
            return self._execute_task_locked(t, task_id, request_id, action_id)

    def _execute_task_locked(self, t, task_id: str, request_id: str, action_id: str) -> Dict[str, Any]:
        b = self._charge_tool_request(task_id)  # counts even on reject / idempotent hit
        existing_run_id = self.store.lookup_request(task_id, request_id, action_id)
        if existing_run_id is None and self.persistence:
            existing_run_id = self.persistence.claimed_run_id(task_id, request_id, action_id)
        if existing_run_id is not None:
            rec = self.store.get_run(existing_run_id) if existing_run_id in self.store._runs else None
            if rec is None and self.persistence:
                # The run row is durable but not yet in this connection's memory:
                # reload it so the idempotent hit is accurate.
                for run_d in self.persistence.runs_for_task(task_id):
                    if run_d["run_id"] == existing_run_id:
                        rec = self._run_from_persisted(run_d, task_id)
                        break
            if rec is not None:
                self.store.record_request(task_id, request_id, action_id, existing_run_id)
                view = _model_run_view(rec)
                view["idempotent_hit"] = True
                view["run_id"] = existing_run_id
                return view

        # 0a. idempotency conflict: same request_id (in this task) with a different action -> reject
        prev_action = self.store.request_action(task_id, request_id)
        if prev_action is not None and prev_action != action_id:
            raise RequestActionConflict(
                f"request_id {request_id!r} in task {task_id!r} already used with action "
                f"{prev_action!r}; cannot switch to {action_id!r}"
            )

        # 0b. terminal gate: BEFORE any new solve, refuse new solves on terminal tasks
        if self.store.is_terminal(task_id):
            raise TaskAlreadyTerminal(
                f"task {task_id} is terminal ({t['state']}); no new solves"
            )

        # 1. capability gate (whitelist + domain/family compatibility). No budget
        #    deduction, no claim, no launch happens yet: a rejected request here
        #    has only paid its tool request.
        card = self._resolve_card(task_id, action_id)
        if not self._action_compatible(card, t["canonical"]):
            self.store.mark_terminal(task_id, "UNSUPPORTED",
                                     reason=f"card {action_id!r} not compatible with domain/family")
            self._persist_terminal(task_id, "UNSUPPORTED")
            raise CapabilityRejected(
                f"card {action_id!r} cannot solve domain={t['canonical']['domain']} "
                f"family={self._family(t['canonical'])}"
            )

        # 1b. DOMAIN-SPECIFIC PRE-LAUNCH GATE (BEFORE the SQLite claim and BEFORE any
        #    solve-budget deduction): for linear systems, the no-unique-solution check
        #    and the candidate-FORMAT precision boundary are decidable from the problem
        #    + original tolerance ALONE, before solving. When either applies the task
        #    is refused UNSUPPORTED with NO fake candidate, NO run row, and NO solve
        #    cost (only the tool request above was charged).
        self._pre_launch_gate(task_id, action_id, card, t, persist_run=False)

        # 2. per-task budget: reserve the solve launch BEFORE dispatch; launch consumes
        #    only solve_launches and is committed immediately (no refund on exception).
        #    The SQLite claim happens FIRST (before any solve-budget deduction), so a
        #    duplicate/PENDING request pays ONLY its tool request -- it adds no solve
        #    spend and leaves no reserved unit behind.
        budget = b  # already obtained from the tool-request charge above
        if budget.started_at == 0.0:
            budget.mark_started()

        # 3. EXACTLY ONE solve launch. Atomically claim the (task_id, request_id,
        #    action_id) in SQLite BEFORE reserving any solve budget, so a truly
        #    concurrent identical request cannot start a SECOND solve: the loser
        #    finds the claim already taken and either reuses the winner's run_id
        #    (idempotent hit) or stops PENDING. No claim -> no launch -> no spend.
        run_id = self.store.new_run_id()
        if self.persistence:
            won = self.persistence.claim_request(task_id, request_id, action_id, run_id)
            if not won:
                existing = self.persistence.claimed_run_id(task_id, request_id, action_id)
                if existing is not None:
                    rec = self.store._runs.get(existing)
                    if rec is None:
                        for run_d in self.persistence.runs_for_task(task_id):
                            if run_d["run_id"] == existing:
                                rec = self._run_from_persisted(run_d, task_id)
                                self.store._runs[existing] = rec
                                break
                    if rec is not None:
                        view = _model_run_view(rec)
                        view["idempotent_hit"] = True
                        view["run_id"] = existing
                        return view
                # claim exists but the winner's run is not in the store yet: this
                # is a PENDING re-entrant entry; we do NOT launch a second solve and
                # have not reserved/committed any solve budget.
                raise TaskError(
                    f"request {request_id!r} in task {task_id!r} is PENDING "
                    f"(claimed by a concurrent caller); re-read and reuse its run"
                )
        # We WON the claim (or no persistence layer): now, and only now, reserve the
        # solve launch. A deadline that is ALREADY expired must not even start.
        try:
            budget.reserve_launch()
        except BudgetExhaustedErr as e:
            self.store.mark_terminal(task_id, "INCOMPLETE", reason=str(e))
            self._persist_budget(task_id)
            self._persist_terminal(task_id, "INCOMPLETE")
            raise BudgetExhausted(f"solve budget exhausted on task {task_id}: {e}") from e
        # commit the reserved launch the moment the solve starts (consumed, not
        # refunded even if the launch raises). If the commit itself fails we MUST
        # stop: launching without a committed reservation would under-count the
        # solve budget, so the exception propagates and the launch does not run.
        budget.commit_launch()
        self._persist_budget(task_id)
        record: Optional[RunRecord] = None
        verdict = "INCONCLUSIVE"
        try:
            sc_rec = self._launch_one_candidate(task_id, run_id, card, t["canonical"], budget)
            record = scicheck_run_to_record(sc_rec, task_id, run_id, action=action_id)
            # Post-launch domain gates (linear system): these can raise
            # CapabilityRejected AFTER the launch was consumed (a real candidate was
            # computed, so the run row + solve cost are recorded honestly).
            self._post_launch_gate(task_id, run_id, card, t, sc_rec, record, persist_run=True)
            dom_check = t["canonical"].get("domain")
            if dom_check in ("linear_system", "root", "ode"):
                # linear-system / root / ODE: _launch_one_candidate already computed
                # the domain-specific exact check with the task's own tolerance. Use
                # it directly; do NOT re-run the integration-domain exact_ok on a
                # vector / root / ODE candidate.
                # - linear_system: per-component forward error (vector candidate)
                # - root: root-position forward error (scalar candidate)
                # - ode: forward-error interval vs reference interval (scalar
                #   candidate, possibly None when the solver did not reach t1)
                check = record.exact_check
            else:
                norm = self._norm_for(t["canonical"])
                check = exact_ok(record.candidate, norm)
                record.exact_check = check
            ok = check["ok"]
            if ok is True:
                verdict = "VERIFIED_PASS"
            elif ok is False:
                verdict = "VERIFIED_INCORRECT"
            else:
                # ok is None: the domain check could not be decided (ODE reference
                # interval too wide, or no valid t1 candidate). Route to the
                # terminal INCONCLUSIVE state, NOT to a spurious VERIFIED_INCORRECT.
                verdict = "INCONCLUSIVE"
            # 4. deadline re-check before publishing a pass; expired -> cannot pass
            if verdict == "VERIFIED_PASS":
                budget.check_deadline_before_publish()
        except BudgetExhaustedErr as e:
            record = self._error_record(run_id, task_id, action_id, str(e), "EXECUTION_ERROR")
            self._persist_budget(task_id)  # the launch is already committed/consumed
            self._persist_terminal(task_id, "INCOMPLETE")
            self.store.mark_terminal(task_id, "INCOMPLETE", reason=str(e))
            self.store.append_run(record)
            self._persist_run(record, run_id, task_id, "INCOMPLETE")
            raise BudgetExhausted(f"budget exhausted on task {task_id}: {e}") from e
        except DeadlineExpired as e:
            # Deadline expired: a pass CANNOT be published; the consumed launch stays
            # recorded. Mark INCOMPLETE.
            record = self._error_record(run_id, task_id, action_id, str(e), "EXECUTION_ERROR")
            self._persist_budget(task_id)
            self._persist_terminal(task_id, "INCOMPLETE")
            self.store.mark_terminal(task_id, "INCOMPLETE", reason=f"deadline expired: {e}")
            self.store.append_run(record)
            self._persist_run(record, run_id, task_id, "INCOMPLETE")
            raise DeadlineExpiredError(f"deadline expired on task {task_id}: {e}") from e
        except Exception as e:
            # Any execution/reference failure -> EXECUTION_ERROR terminal. The
            # already-committed launch is retained (not refunded); no retry inside.
            # NOTE: CapabilityRejected raised by _post_launch_gate (or any domain
            # gate) is NOT an execution error -- it means the original task was
            # registered with an impossible tolerance for this candidate format
            # (or the system has no unique solution in the chosen route). The
            # run row + receipt are recorded (the solve cost was honestly spent),
            # the task is routed to UNSUPPORTED (a terminal state, not
            # EXECUTION_ERROR), and the exception propagates to the caller.
            if isinstance(e, CapabilityRejected):
                record = self._error_record(run_id, task_id, action_id, str(e), "UNSUPPORTED")
                self._persist_budget(task_id)
                self._persist_terminal(task_id, "UNSUPPORTED")
                self.store.mark_terminal(task_id, "UNSUPPORTED", reason=str(e))
                self.store.append_run(record)
                self._persist_run(record, run_id, task_id, "UNSUPPORTED")
                raise
            record = self._error_record(run_id, task_id, action_id, str(e), "EXECUTION_ERROR")
            self._persist_budget(task_id)
            self._persist_terminal(task_id, "EXECUTION_ERROR")
            self.store.mark_terminal(task_id, "EXECUTION_ERROR", reason=str(e))
            self.store.append_run(record)
            self._persist_run(record, run_id, task_id, "EXECUTION_ERROR")
            raise TaskError(f"execution error on task {task_id}: {e}") from e

        # Persist the receipt + run + request atomically (one SQLite transaction).
        # The claim was already taken in step 3; commit_run_with_receipt rebinds
        # the winning run_id so a concurrent duplicate can only ever start ONE solve.
        # NOTE: the solve launch was already committed at start (step 3); it is
        # NOT committed a second time here.
        # A vector candidate (linear-system) must be JSON-encoded in the receipt so
        # its type survives sealing/reload; the scalar path is unchanged.
        # A vector candidate (linear-system solution) must be JSON-encoded in the receipt
        # so its LIST type survives sealing/reload; the scalar path is unchanged.
        # For the Decimal high-precision route the candidate is already a list of
        # Decimal STRINGS -- keep them as-is (strings round-trip through JSON and
        # the receipt binding without loss).
        cand = record.candidate
        if isinstance(cand, (list, tuple)):
            cand_for_receipt = list(cand)  # preserve actual element type (float or str)
        else:
            # A scalar candidate is either a float (integration / root / binary ODE
            # candidate) or None (an ODE solver that did not reach t1, or an error
            # record). JSON serializes None as null -- it is NEVER a NaN/Infinity.
            cand_for_receipt = cand
        check = record.exact_check
        dom_receipt = t["canonical"].get("domain")
        ref_frac = check.get("reference_fraction", "")
        if not ref_frac and check.get("reference_vector_fractions"):
            ref_frac = json.dumps(check["reference_vector_fractions"], sort_keys=True)
        diff_frac = check.get("forward_error_exact_fraction",
                              check.get("abs_diff_exact_fraction", ""))
        allowed_frac = check.get("allowed_error_exact_fraction", "")
        # ODE interval-binding fields (the ODE exact_check carries the reference
        # interval + allowed-error interval; non-ODE domains leave them empty so
        # the receipt keeps its original shape).
        ref_lo_frac = check.get("reference_lo_fraction", "")
        ref_hi_frac = check.get("reference_hi_fraction", "")
        allowed_min_frac = check.get("allowed_error_min_fraction", "")
        allowed_max_frac = check.get("allowed_error_max_fraction", "")
        if check.get("per_component"):
            # per-component diff/allowed (linear-system): bind them as JSON lists
            diff_frac = json.dumps(
                [c["forward_error_exact_fraction"] for c in check["per_component"]],
                sort_keys=True)
            allowed_frac = json.dumps(
                [c["allowed_error_exact_fraction"] for c in check["per_component"]],
                sort_keys=True)
        receipt = make_receipt(
            reference_fraction=ref_frac,
            allowed_error_fraction=allowed_frac,
            candidate=cand_for_receipt,
            diff_fraction=diff_frac,
            verdict=verdict,
            run_id=run_id,
            task_id=task_id,
            task_hash=t["hash"],
            session=self.store.session,
            action_id=action_id,  # the ACTUAL action (e.g. 'root_brent'), not the solver module
            reference_lo_fraction=ref_lo_frac,
            reference_hi_fraction=ref_hi_frac,
            allowed_error_min_fraction=allowed_min_frac,
            allowed_error_max_fraction=allowed_max_frac,
        )
        self.store.append_run(record)
        self.store.record_request(task_id, request_id, action_id, run_id)
        if self.persistence:
            # ONE atomic write of the run row + sealed receipt (the request claim
            # was already bound in step 3). Do NOT insert the run row a second time.
            self.persistence.commit_run_with_receipt(
                record.to_dict(), receipt,
                request_id=request_id, task_id=task_id, action_id=action_id, run_id=run_id,
            )

        # 6. verdict -> terminal or stay ACTIVE
        if record.exact_ok is True:
            self.store.mark_terminal(task_id, "VERIFIED_PASS", reason=f"run {run_id} verified correct")
            self._persist_terminal(task_id, "VERIFIED_PASS")
        elif record.exact_ok is None:
            # INCONCLUSIVE is a TERMINAL state (the task's domain check could not be
            # decided; e.g. an ODE reference interval too wide to decide, or no valid
            # t1 candidate). Route it here -- NOT to a spurious VERIFIED_INCORRECT
            # -- and sign the terminal row.
            self.store.mark_terminal(
                task_id, "INCONCLUSIVE",
                reason=f"run {run_id} verdict inconclusive (domain check undecidable)",
            )
            self._persist_terminal(task_id, "INCONCLUSIVE")
        elif record.exact_ok is False and budget.counters["solve_launches"].remaining() <= 0:
            self.store.mark_terminal(
                task_id, "INCOMPLETE",
                reason=f"run {run_id} verified incorrect; no solve budget left",
            )
            self._persist_budget(task_id)
            self._persist_terminal(task_id, "INCOMPLETE")
        # else: VERIFIED_INCORRECT with budget remaining -> stays ACTIVE; the caller
        # explicitly starts the repair via a NEW execute_task call.

        view = _model_run_view(record)
        view["idempotent_hit"] = False
        view["run_id"] = run_id
        view["verdict"] = verdict
        return view

    # ---- verify_run: read + verify sealed receipt, do not re-solve ----

    def verify_run(self, task_id: str, run_id: str) -> Dict[str, Any]:
        """Read + verify the sealed receipt. Does NOT recompute the reference or the
        candidate: the receipt already carries the trusted-side verdict, candidate,
        and diff. We verify HMAC integrity + the presence of every required
        identity-binding field, then confirm they match the current store binding.
        If the sealed receipt is MISSING when persistence is configured, we do NOT
        fall back to the bare run record -- that would let a tampered store fake a
        VERIFIED_PASS. Instead we return INCONCLUSIVE with an explicit note.
        """
        t = self._get_task_bound(task_id)
        self._charge_tool_request(task_id)
        rec = self.store._runs.get(run_id)
        if rec is None or rec.task_id != task_id:
            raise TaskNotFound(f"run {run_id} not found for task {task_id}")
        receipt = None
        if self.persistence:
            # load_receipt raises ReceiptTampered on HMAC mismatch. A MISSING receipt
            # is a real integrity loss: we cannot claim a pass without it.
            try:
                receipt = self.persistence.load_receipt(run_id)
            except ReceiptTampered as e:
                # A tampered receipt must be surfaced as a task-level integrity error,
                # never as a silent fallback to the bare record.
                raise TaskError(
                    f"sealed receipt for run {run_id} FAILED integrity verification: {e}"
                ) from e
            if receipt is None:
                # Missing sealed receipt: refuse to claim a pass; return INCONCLUSIVE.
                out = {
                    "task_id": task_id,
                    "run_id": run_id,
                    "task_state": t["state"],
                    "candidate": rec.candidate,
                    "stored_exact_ok": rec.exact_ok,
                    "consistent": False,
                    "receipt_verdict": None,
                    "conclusion": "INCONCLUSIVE",
                    "note": "sealed trusted receipt is MISSING for this run; without it a "
                            "pass cannot be claimed -- result is INCONCLUSIVE, not VERIFIED_PASS",
                }
                return _strip_model_view(out)
        if receipt is None:
            # No sealed receipt was produced (no persistence / not yet committed):
            # we CANNOT claim a pass without it. Return INCONCLUSIVE.
            out = {
                "task_id": task_id,
                "run_id": run_id,
                "task_state": t["state"],
                "candidate": rec.candidate,
                "stored_exact_ok": rec.exact_ok,
                "consistent": False,
                "receipt_verdict": None,
                "conclusion": "INCONCLUSIVE",
                "note": "no sealed trusted receipt for this run; a pass cannot be "
                        "claimed without it -- result is INCONCLUSIVE, not VERIFIED_PASS",
            }
            return _strip_model_view(out)
        # Verify every required identity-binding field is PRESENT (not merely truthy)
        # AND bound to the current store task; a missing key is a hard reject.
        # NOTE: use `is None` checks -- a legitimate candidate of 0.0 (or empty
        # fractions) must not be treated as "missing".
        cur_hash = t["hash"]
        cur_session = t["session"]
        missing = [
            f for f in REQUIRED_RECEIPT_FIELDS
            if f not in receipt or receipt.get(f) is None
        ]
        if missing:
            raise TaskError(f"receipt for run {run_id} is missing required binding fields: {missing}")
        # action_id must match the run record's action (NOT the solver module name)
        if receipt.get("action_id") != rec.action:
            raise TaskError(
                f"receipt action_id {receipt.get('action_id')!r} != stored run action {rec.action!r}"
            )
        # candidate must match the stored run record (no recomputation)
        if receipt.get("candidate") != rec.candidate:
            raise TaskError(
                f"receipt candidate {receipt.get('candidate')!r} != stored {rec.candidate!r}"
            )
        # identity fields must match the current binding
        if receipt.get("task_id") != task_id:
            raise TaskError(f"receipt task_id {receipt.get('task_id')!r} != {task_id!r}")
        if receipt.get("run_id") != run_id:
            raise TaskError(f"receipt run_id {receipt.get('run_id')!r} != {run_id!r}")
        if receipt.get("task_hash") != cur_hash:
            raise TaskError(f"receipt task_hash {receipt.get('task_hash')!r} != {cur_hash!r}")
        if receipt.get("session") != cur_session:
            raise TaskError(f"receipt session {receipt.get('session')!r} != {cur_session!r}")
        receipt_verdict = receipt.get("verdict")
        # Tri-state consistency check: stored exact_ok maps one-to-one onto
        # the sealed receipt verdict (True -> VERIFIED_PASS, False ->
        # VERIFIED_INCORRECT, None -> INCONCLUSIVE). A real incorrect or
        # inconclusive receipt is still consistent when it matches its own
        # verdict; only VERIFIED_PASS is a positive pass. A mismatch on any
        # state (including None) is inconsistent.
        expected = ("VERIFIED_PASS" if rec.exact_ok is True
                    else ("VERIFIED_INCORRECT" if rec.exact_ok is False
                          else "INCONCLUSIVE"))
        consistent = receipt_verdict == expected
        out = {
            "task_id": task_id,
            "run_id": run_id,
            "task_state": t["state"],
            "candidate": rec.candidate,
            "stored_exact_ok": rec.exact_ok,
            "consistent": consistent,
            "receipt_verdict": receipt_verdict,
            "conclusion": receipt_verdict if consistent else None,
            "note": "verify_run reads + verifies the sealed trusted receipt; it does not "
                    "re-solve and does not expose the reference answer to model tools",
        }
        return _strip_model_view(out)

    # ---- task view (model-facing, no reference answer; recursive strip) ----

    def task_view(self, task_id: str) -> Dict[str, Any]:
        t = self.store.get_task(task_id)
        b = self.ledger.get(task_id)
        runs = [self.store.get_run_copy(rid) for rid in t["runs"]]
        view = {
            "task_id": task_id,
            "session": t["session"],
            "hash": t["hash"],
            "canonical": t["canonical"],
            "state": t["state"],
            "runs": runs,
            "legacy_record_count": len(self.store.legacy_records()),
        }
        if b:
            view["budget"] = b.as_dict()
        return _strip_model_view(view)

    # ---- restart / conservative stop ----

    def restart_recover(self, task_ids: Optional[list] = None) -> Dict[str, str]:
        """Genuine process restart: reload any tasks missing from memory, then
        conservatively stop still-ACTIVE tasks as INCOMPLETE with reserved units
        persisted. Terminal tasks are untouched. This is the stop_active=True
        variant of restore_from_persistence."""
        # First ensure every persisted task is in memory (pure reload, no stop)
        self.restore_from_persistence(stop_active=False)
        changed: Dict[str, str] = {}
        ids = task_ids if task_ids is not None else self.store.list_task_ids()
        for tid in ids:
            if tid not in self.store._tasks:
                continue
            t = self.store._get_task_internal(tid)
            if t["state"] not in TERMINAL_STATES:
                self.store.mark_terminal(tid, "INCOMPLETE", reason="restart conservative stop")
                self._persist_budget(tid)
                self._persist_terminal(tid, "INCOMPLETE")
                if self.persistence:
                    self.persistence.set_task_state(tid, self.store.session, "INCOMPLETE",
                                                     "restart conservative stop")
                changed[tid] = "INCOMPLETE"
        return changed

    def reopen_terminal(self, new_task_id: str, old_task_id: str, authorize: bool = True) -> Dict[str, Any]:
        """Reopen a terminal task as a NEW task id with trusted authorization.
        The old terminal record is left untouched (append-only log). The new task
        receives a fresh default per-task budget (via the controller's own ledger +
        persistence), so it can be solved immediately without a separate
        trusted_register_task call that would collide with the task row
        `reopen_terminal` just created in the store."""
        if not authorize:
            raise TaskError("reopening a terminal task requires trusted authorization")
        old = self.store.get_task(old_task_id)
        if old["state"] not in TERMINAL_STATES:
            raise TaskError(f"task {old_task_id} is not terminal; nothing to reopen")
        t = self.store.register_task(new_task_id, old["canonical"])
        # Mirror the trusted_register_task budget-creation step for the new task id,
        # but do NOT re-call store.register_task (the task row already exists).
        self.ledger.for_task(new_task_id)  # creates a fresh default TaskBudget entry
        self._persist_budget(new_task_id)
        return t

    # ---- internals ----

    def _get_task_bound(self, task_id: str) -> Dict[str, Any]:
        try:
            t = self.store._get_task_internal(task_id)
        except KeyError as e:
            raise TaskNotFound(f"unknown task {task_id} in session {self.store.session}") from e
        self.store.assert_session_binding(task_id, self.store.session)
        return t

    def _norm_for(self, canonical: Dict[str, Any]) -> "scicheck.NormalizedProblem":
        return normalize_problem(self._canonical_to_scicheck_payload(canonical))

    def _action_compatible(self, card: MethodCard, canonical: Dict[str, Any]) -> bool:
        dom = canonical.get("domain")
        if dom == "integration":
            family = self._family(canonical)
            structures = set(card.structure.split("|"))
            if family == "polynomial" and "polynomial" not in structures:
                return False
            if family == "piecewise_constant" and "piecewise_constant" not in structures:
                return False
            return True
        if dom == "root":
            return "root" in card.structure.split("|")
        if dom == "linear_system":
            return card.card_id in ("ls_np32", "ls_np64", "ls_decimal_high")
        if dom == "ode":
            return card.card_id in (
                "ode_euler10", "ode_dop853",
                "ode_closed_naive", "ode_closed_stable",
            )
        # unknown domain -> UNSUPPORTED
        return False

    @staticmethod
    def _family(canonical: Dict[str, Any]) -> Optional[str]:
        return canonical.get("problem", {}).get("family")

    def _run_from_persisted(self, run_d: Dict[str, Any], task_id: str) -> RunRecord:
        """Rebuild an in-memory RunRecord from a durable SQLite run row (the same
        construction restore_from_persistence uses). Only None means "absent"; a
        legitimate candidate/abserr of 0.0 must NOT be coerced to NaN. A VECTOR
        candidate (linear-system solution) is rebuilt from candidate_json so its
        list type survives the reload. An ODE run whose solver did not reach t1
        has candidate=None (null) and a tri-state exact_ok (None = INCONCLUSIVE,
        not a spurious False)."""
        cand = run_d.get("candidate")
        abserr = run_d.get("abserr")
        stored_ok = run_d.get("exact_ok")
        # Tri-state exact_ok: the persistence layer already normalizes DB NULL ->
        # None; preserve it, do NOT bool() it (which would turn INCONCLUSIVE into
        # a spurious False).
        rec = RunRecord(
            run_id=run_d["run_id"], task_id=task_id, action=run_d["action"],
            candidate=cand,  # may be None (ODE no-t1-candidate) or a float/list
            quad_abserr_reported=abserr,
            exact_ok=stored_ok,
            exact_check={"ok": stored_ok},
            resource=run_d.get("resource", {}),
            method=run_d.get("method", run_d["action"]),
            terminal_status=run_d.get("verdict"),
        )
        return rec

    def _launch_one_candidate(self, task_id, run_id, card, canonical, budget):
        dom = canonical.get("domain")
        if dom == "root":
            # Root-domain launch: the solver is picked by the card id and the public
            # structure (scale, shift, power) drives everything -- no private
            # reference interface is called. Candidates are COMPUTED ONLY here; the
            # verdict is decided ONCE by the controller against the task's ORIGINAL
            # tolerance (single source of truth, no dual verdicts).
            import scicheck as _sc
            from fractions import Fraction as _Fr
            prob = canonical["problem"]
            tol = canonical["tolerance"]
            start = float(_Fr(prob["start"])) if prob.get("start") else None
            rp = _sc.normalize_root_problem({
                "domain": [prob["a"], prob["b"]],
                "problem": {
                    "type": "scaled_power",
                    "power": prob.get("power", ""),
                    "scale": prob.get("scale", ""),
                    "shift": prob.get("shift", ""),
                },
                "tolerance": tol,
            })
            persist_fn = (lambda: self._persist_budget_now(task_id)) if self.persistence else None
            if card.card_id == "root_newton":
                rec = _sc.run_root_newton(rp, run_id=run_id, task_id=task_id,
                                          start=start, budget=budget, persist_fn=persist_fn)
            elif card.card_id == "root_brent":
                rec = _sc.run_root_brent(rp, run_id=run_id, task_id=task_id,
                                         budget=budget, persist_fn=persist_fn)
            elif card.card_id == "root_direct":
                rec = _sc.run_root_direct(rp, run_id=run_id, task_id=task_id,
                                          budget=budget, persist_fn=persist_fn)
            else:
                raise TaskError(f"root card {card.card_id!r} has no launch implementation")
            # Single verdict: apply the task's ORIGINAL tolerance to the candidate's
            # exact forward error (the solvers did not decide, to avoid a dual verdict).
            atol = _Fr(tol["absolute"]) if tol.get("absolute") else _Fr(0)
            rtol = _Fr(tol["relative"]) if tol.get("relative") else _Fr(0)
            check = _sc.root_exact_ok(rec.value, rp, atol=atol, rtol=rtol,
                                      residual=rec.exact_check.get("residual_diagnostic"))
            rec.exact_ok = check["ok"]
            rec.exact_check = check
            return rec
        if dom == "linear_system":
            # Linear-system launch: a REAL NumPy candidate (float32 development/diagnostic,
            # float64 repair, or a public high-precision Decimal route). The candidate is
            # a FLOAT VECTOR, not a scalar. The verdict is the PER-COMPONENT forward
            # error against the task's ORIGINAL tolerance, decided ONCE here (single
            # source of truth; residual and condition number are diagnostics only).
            import scicheck as _sc
            from fractions import Fraction as _Fr
            from numpy.linalg import LinAlgError
            import numpy as _np
            payload = self._canonical_to_scicheck_payload(canonical)
            prob = _sc.normalize_lin_sys_problem(payload)
            tol = canonical["tolerance"]
            atol = _Fr(tol["absolute"]) if tol.get("absolute") else _Fr(0)
            rtol = _Fr(tol["relative"]) if tol.get("relative") else _Fr(0)
            # No-unique-solution + binary-format boundary were already decided in
            # _pre_launch_gate (before the SQLite claim and before any solve-budget
            # deduction); reaching this point means the system HAS a unique solution
            # AND, for binary cards, the original tolerance IS reachable in the card's
            # format. Here we just compute the candidate and apply the task's original
            # tolerance ONCE (the single verdict source of truth).
            if card.card_id == "ls_np32":
                rec = _sc.run_ls_np(prob, run_id=run_id, task_id=task_id, dtype="float32",
                                    atol=atol, rtol=rtol)
            elif card.card_id == "ls_np64":
                rec = _sc.run_ls_np(prob, run_id=run_id, task_id=task_id, dtype="float64",
                                    atol=atol, rtol=rtol)
            elif card.card_id == "ls_decimal_high":
                rec = _sc.run_ls_decimal_high(prob, run_id=run_id, task_id=task_id,
                                              atol=atol, rtol=rtol)
            else:
                raise TaskError(f"linear-system card {card.card_id!r} has no launch implementation")
            # Independent per-component check with the task's ORIGINAL tolerance, applied
            # exactly once AFTER the candidate returns (the candidate function itself did
            # not read the reference when atol/rtol were supplied here because the
            # controller is the single verdict source; the check is recomputed here).
            check = _sc._linsys_check(rec.value, prob, atol, rtol)
            rec.exact_ok = check["ok"]
            rec.exact_check = check
            return rec
        if dom == "ode":
            # ODE-domain launch: a REAL binary64 y(t1) candidate from either the
            # coarse 10-step forward-Euler (development/diagnostic) or the SciPy
            # DOP853 8th-order repair route. The candidate function itself runs in
            # NO-TOLERANCE MODE (atol/rtol = None) so it does NOT read the private
            # reference interval; each RHS call (Euler steps / DOP853 RHS) has
            # already charged + persisted the func_evals budget. The controller is
            # the SINGLE verdict source: apply the task's ORIGINAL tolerance via
            # _ode_check exactly once, AFTER the candidate returns.
            # Budget/deadline exhaustion inside the candidate propagates to the
            # caller (INCOMPLETE); a plain solver failure leaves candidate=None
            # and the controller records EXECUTION_ERROR -- it is never swallowed
            # into a fake INCONCLUSIVE verdict here.
            import scicheck as _sc
            from fractions import Fraction as _Fr
            payload = self._canonical_to_scicheck_payload(canonical)
            prob = _sc.normalize_ode_problem(payload)
            tol = canonical["tolerance"]
            atol = _Fr(tol["absolute"]) if tol.get("absolute") else _Fr(0)
            rtol = _Fr(tol["relative"]) if tol.get("relative") else _Fr(0)
            persist_fn = (lambda: self._persist_budget_now(task_id)) if self.persistence else None
            if card.card_id == "ode_euler10":
                rec = _sc.run_ode_euler10(prob, run_id=run_id, task_id=task_id,
                                          budget=budget, persist_fn=persist_fn)
            elif card.card_id == "ode_dop853":
                rec = _sc.run_ode_dop853(prob, run_id=run_id, task_id=task_id,
                                         budget=budget, persist_fn=persist_fn)
            elif card.card_id == "ode_closed_naive":
                rec = _sc.run_ode_closed_naive(prob, run_id=run_id, task_id=task_id,
                                               budget=budget, persist_fn=persist_fn)
            elif card.card_id == "ode_closed_stable":
                rec = _sc.run_ode_closed_stable(prob, run_id=run_id, task_id=task_id,
                                                budget=budget, persist_fn=persist_fn)
            else:
                raise TaskError(f"ode card {card.card_id!r} has no launch implementation")
            # Single verdict: apply the task's ORIGINAL tolerance to the candidate's
            # forward-error interval (the candidate function did NOT decide, to avoid
            # a dual verdict). The candidate may be None (no valid t1 candidate); the
            # check returns ok=None (INCONCLUSIVE) in that case, and the controller
            # routes it to the terminal INCONCLUSIVE state below.
            check = _sc._ode_check(rec.value, prob, atol=atol, rtol=rtol)
            rec.exact_ok = check["ok"]
            rec.exact_check = check
            return rec
        payload = self._canonical_to_scicheck_payload(canonical)
        norm = normalize_problem(payload)
        # Gate the integrand so each real evaluation checks deadline + func_evals and
        # counts one real evaluation immediately (BudgetGatedQuad); persist the budget
        # right after each evaluation so an exception still leaves it durable. Do not
        # rely on a post-hoc neval read.
        persist_fn = (lambda: self._persist_budget_now(task_id)) if self.persistence else None
        if card.card_id in ("quad_breakpoints", "quad_tightened"):
            return run_quad_with_breakpoints(norm, run_id=run_id, task_id=task_id,
                                             budget=budget, persist_fn=persist_fn)
        return run_plain_quad(norm, run_id=run_id, task_id=task_id,
                              budget=budget, persist_fn=persist_fn)

    def _canonical_to_scicheck_payload(self, canonical: Dict[str, Any]) -> Dict[str, Any]:
        prob = canonical["problem"]
        domain = canonical.get("domain")
        if domain == "linear_system":
            return {
                "problem": {
                    "n": prob["n"],
                    "matrix": prob["matrix"],
                    "rhs": prob["rhs"],
                },
                "tolerance": canonical["tolerance"],
            }
        if domain == "ode":
            # ODE: the canonical problem already carries the ORIGINAL structured
            # inputs (a, b, y0, t0, t1) as rational strings; dt is NOT a canonical
            # field (it is a derived quantity recomputed by normalize_ode_problem),
            # so passing the canonical problem straight through keeps re-normalization
            # idempotent (a SQLite restart re-runs normalize_task on the same dict).
            return {
                "problem": {
                    "family": prob["family"],
                    "a": prob["a"],
                    "b": prob["b"],
                    "y0": prob["y0"],
                    "t0": prob["t0"],
                    "t1": prob["t1"],
                },
                "tolerance": canonical["tolerance"],
            }
        family = prob["family"]
        a = prob["a"]
        b = prob["b"]
        if family == "polynomial":
            return {
                "domain": [a, b],
                "problem": {"type": "rational_poly", "coeffs": prob["coefficients"]},
                "tolerance": canonical["tolerance"],
            }
        bps = [a] + prob["breakpoints"] + [b]
        vals = prob["values"]
        segments = [[bps[i], bps[i + 1], vals[i]] for i in range(len(bps) - 1)]
        return {
            "domain": [a, b],
            "problem": {"type": "piecewise_const", "segments": segments},
            "tolerance": canonical["tolerance"],
        }

    def _error_record(self, run_id, task_id, action_id, msg, status) -> RunRecord:
        # An error run carries a NULL candidate (not a NaN) so that the receipt
        # and any standard JSON serialization stay clean (JSON cannot carry
        # NaN/Infinity). The exact_ok is False for an execution error (a real
        # failure, not an INCONCLUSIVE undecidable verdict).
        return RunRecord(
            run_id=run_id, task_id=task_id, action=action_id,
            candidate=None, quad_abserr_reported=None,
            exact_ok=False,
            exact_check={"ok": False, "error": msg},
            resource={"solver": action_id, "error": msg},
            terminal_status=status, created_by="execute_task",
        )

    def _resolve_card(self, task_id: str, action_id: str) -> MethodCard:
        try:
            return self.registry.method_lookup(task_id, action_id)
        except MethodLookupError as e:
            raise ActionNotAllowed(f"action {action_id!r} is not in the trusted whitelist") from e

    def _pre_launch_gate(self, task_id: str, action_id: str, card: MethodCard,
                        t: Dict[str, Any], persist_run: bool) -> None:
        """Domain-specific PRE-launch checks decidable from the frozen problem +
        original tolerance ALONE, run BEFORE the SQLite claim and BEFORE any
        solve-budget deduction: a rejection here produces NO fake candidate, NO run
        row, and NO solve cost -- only the tool request already charged is spent.
        CapabilityRejected is raised before the try block that catches generic
        Exception, so it is NOT swallowed into EXECUTION_ERROR.

        For linear systems:
          * no-unique-solution (rank-deficient over the rationals) -> UNSUPPORTED.
          * binary-candidate format boundary: if the ORIGINAL tolerance cannot be
            met by ANY binary32/binary64 value of the exact solution for the
            chosen card's format (decided component-wise with ANY, not ALL -- one
            impossible component is enough) -> UNSUPPORTED (a limit of the
            binary candidate FORMAT, not that the math is unsolvable).
        """
        dom = t["canonical"].get("domain")
        if dom != "linear_system":
            return
        import scicheck as _sc
        from fractions import Fraction as _Fr
        prob = t["canonical"]["problem"]
        tol = t["canonical"]["tolerance"]
        atol = _Fr(tol["absolute"]) if tol.get("absolute") else _Fr(0)
        rtol = _Fr(tol["relative"]) if tol.get("relative") else _Fr(0)
        lin_prob = _sc.normalize_lin_sys_problem({
            "problem": {"n": prob["n"], "matrix": prob["matrix"], "rhs": prob["rhs"]},
            "tolerance": tol,
        })
        # (1) no-unique-solution: a mathematical property of the input -> UNSUPPORTED.
        ref_info = _sc.linsys_reference_fractions(lin_prob)
        if not ref_info["has_unique_solution"]:
            self.store.mark_terminal(
                task_id, "UNSUPPORTED",
                reason=("no unique solution (rank-deficient over the rationals): "
                        f"{ref_info['diagnostics'].get('reason')}"))
            self._persist_terminal(task_id, "UNSUPPORTED")
            raise CapabilityRejected(
                "no unique solution for this linear system (rank-deficient over the "
                f"rationals): {ref_info['diagnostics'].get('reason')} -- UNSUPPORTED")
        # (2) binary-candidate format boundary (ANY component impossible is enough).
        if card.card_id in ("ls_np32", "ls_np64"):
            fmt = "float32" if card.card_id == "ls_np32" else "float64"
            ref = ref_info["solution"]
            impossible = [
                i for i in range(len(ref))
                if (atol + rtol * abs(ref[i])) < _smallest_representable_binary_error(ref[i], fmt)
            ]
            if impossible:
                self.store.mark_terminal(
                    task_id, "UNSUPPORTED",
                    reason=(f"the original tolerance cannot be met by any {fmt} value of "
                            f"the exact solution (impossible component indices (0-based): "
                            f"{impossible}); a limit of the binary candidate FORMAT, not "
                            "that the system has no unique solution"))
                self._persist_terminal(task_id, "UNSUPPORTED")
                raise CapabilityRejected(
                    f"the original tolerance cannot be met by any {fmt} value of the "
                    "exact solution; this is a limit of the binary candidate FORMAT, "
                    "not that the system has no unique solution -- UNSUPPORTED. Use the "
                    "high-precision Decimal route (ls_decimal_high, registered as its own "
                    "task/route) for a usable candidate under this tolerance.")

    def _post_launch_gate(self, task_id: str, run_id: str, card: MethodCard,
                         t: Dict[str, Any], sc_rec, record, persist_run: bool) -> None:
        """Post-launch domain gates. The pre-launch gate already rejected every
        case that was decidable before the solve cost; a post-launch binary-route
        failure that is NOT at the format boundary is an ordinary VERIFIED_INCORRECT
        (the task stays ACTIVE for a repair), so no extra routing is needed here.
        Kept as an explicit hook so the two phases are clearly separated.
        """
        return

    def _persist_budget(self, task_id: str) -> None:
        if self.persistence:
            b = self.ledger.get(task_id)
            if b:
                self.persistence.save_budget(task_id, b)

    def _persist_terminal(self, task_id: str, state: str) -> None:
        if self.persistence:
            self.persistence.set_task_state(task_id, self.store.session, state)

    def _persist_run(self, record: RunRecord, run_id: str, task_id: str, verdict: str,
                     receipt: Optional[Dict[str, Any]] = None) -> None:
        if self.persistence:
            d = record.to_dict()
            d["verdict"] = verdict
            if receipt is not None:
                d["receipt"] = receipt
            self.persistence.insert_run(d)


# ----------------------------------------------------------------------
# Module-level helper used by TaskController._pre_launch_gate.
# ----------------------------------------------------------------------
def _smallest_representable_binary_error(ref: "Fraction", fmt: str) -> "Fraction":
    """Distance from the exact rational `ref` to the nearest representable value
    of the target binary format ('float32' or 'float64'). 0 when `ref` is itself
    exactly representable in that format. This is the smallest forward error any
    candidate expressed in that format can have; if the original tolerance is
    smaller than this for even ONE component, that component is impossible to
    satisfy in the format and the binary route is UNSUPPORTED (a limit of the
    output FORMAT, not of the mathematics)."""
    from fractions import Fraction
    import numpy as _np
    dtype = _np.float32 if fmt == "float32" else _np.float64
    # Nearest binary value = round-to-nearest of the exact reference.
    nearest_val = float(_np.asarray(float(ref), dtype=dtype))
    best = abs(Fraction.from_float(nearest_val) - ref)
    # Also check the two immediate neighbours in case the exact reference falls
    # on a tie or just inside the spacing where the nearest's error is not the
    # minimum (round-to-nearest makes this a no-op in practice, but keep it).
    for target in (_np.nextafter(nearest_val, _np.float64(1.0)),
                   _np.nextafter(nearest_val, _np.float64(-1.0))):
        err = abs(Fraction.from_float(float(target)) - ref)
        if err < best:
            best = err
    return best
