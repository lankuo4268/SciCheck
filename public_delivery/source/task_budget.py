"""Per-task resource budgets, reservations, and spending.

Default budget (per task), each counted in its OWN real unit:
  - solve_launches   : max candidate-solve launches            (default 3)
  - func_evals       : max integrand function evaluations      (default 2000)
  - tool_requests    : max trusted tool requests              (default 12)
  - model_requests   : max model requests                     (default 8)
  - wall_seconds     : max wall-clock budget seconds          (default 180)

Semantics (per the controlled-task spec):
- A task registers its OWN TaskBudget at registration; the ledger is keyed by
  task_id. There is NO shared controller-level budget, and budget=None is not a
  license to solve unboundedly.
- The solve entrypoint consumes ONLY solve_launches. A launch that was reserved
  is committed immediately (spend-then-reserve is NOT allowed); a launch is
  consumed the moment it is started and is NOT refunded on exception.
- execute_task does NOT call register_model_request: a local candidate solve is
  not a real model request. Real model requests are registered INDEPENDENTLY by
  the trusted adapter BEFORE dispatch. Until that AGH wiring exists we only
  prove the local reservation/registration interface and its unit behavior.
- method_lookup / execute_task / verify_run each independently count ONE
  tool_requests call, including idempotent repeats and rejected requests.
- Function callbacks (the actual integrand evaluations) check the deadline and
  the func_evals budget BEFORE evaluating, and count one real evaluation per
  call; they do not rely on a post-hoc neval read.
- Before publishing a result, the deadline is checked again; an expired
  deadline cannot publish a VERIFIED_PASS receipt.
- _Counter.reserve checks spent+reserved+amount <= total and rejects
  negative/boolean amounts; spend also accounts for reserved units.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, Optional


DEFAULT_BUDGET = {
    "solve_launches": 3,
    "func_evals": 2000,
    "tool_requests": 12,
    "model_requests": 8,
    "wall_seconds": 180,
}

KNOWN_RESOURCES = set(DEFAULT_BUDGET.keys())


class BudgetExhausted(Exception):
    pass


class DeadlineExpired(Exception):
    pass


class _Counter:
    """A single resource counter with reserved/spent/total, JSON-able.

    All quantities are TRUE INTEGERS. Floats, negatives, booleans, and unknown
    resource keys are rejected at construction / amount-check time.
    """

    @staticmethod
    def _as_true_int(value, where: str) -> int:
        # booleans are ints in Python; reject them explicitly
        if isinstance(value, bool):
            raise ValueError(f"{where} must be a true integer, not a boolean")
        # reject floats (even integer-valued ones) -> no silent truncation
        if isinstance(value, float):
            raise ValueError(f"{where} must be a true integer, not a float ({value!r})")
        if not isinstance(value, int):
            raise ValueError(f"{where} must be an int, got {type(value).__name__}")
        if value < 0:
            raise ValueError(f"{where} must be non-negative, got {value}")
        return value

    def __init__(self, total):
        self.total = self._as_true_int(total, "budget total")
        self.spent = 0
        self.reserved = 0

    def _check_amount(self, amount, where: str = "budget amount") -> int:
        return self._as_true_int(amount, where)

    def reserve(self, amount: int = 1) -> None:
        n = self._check_amount(amount, "reserve amount")
        # reserve must account for ALREADY-spent and already-reserved units
        if self.spent + self.reserved + n > self.total:
            raise BudgetExhausted(
                f"cannot reserve {n} (would exceed total {self.total}; "
                f"spent={self.spent}, reserved={self.reserved})"
            )
        self.reserved += n

    def commit_reservation(self, amount: int = 1) -> None:
        """Move reserved units into spent. Spent accounts for reserved."""
        n = self._check_amount(amount, "commit amount")
        if self.reserved < n:
            raise BudgetExhausted(f"insufficient reserved to commit {n}")
        if self.spent + n > self.total:
            raise BudgetExhausted(f"committing {n} would exceed total {self.total}")
        self.reserved -= n
        self.spent += n

    def spend(self, amount: int = 1) -> None:
        """Directly spend; accounts for already-reserved units too."""
        n = self._check_amount(amount, "spend amount")
        if self.spent + self.reserved + n > self.total:
            raise BudgetExhausted(
                f"cannot spend {n} (would exceed total {self.total}; "
                f"spent={self.spent}, reserved={self.reserved})"
            )
        self.spent += n

    def remaining(self) -> int:
        return self.total - self.spent - self.reserved

    def as_dict(self) -> Dict[str, int]:
        return {"total": self.total, "spent": self.spent,
                "reserved": self.reserved, "remaining": self.remaining()}


@dataclass
class TaskBudget:
    """The full per-task resource ledger. One instance per task_id."""
    task_id: str
    counters: Dict[str, _Counter] = field(default_factory=dict)

    def __init__(self, task_id: str, totals: Optional[Dict[str, int]] = None):
        if totals is None:
            totals = dict(DEFAULT_BUDGET)
        # reject unknown resources, booleans, floats, negatives, non-ints
        for k in totals:
            if k not in KNOWN_RESOURCES:
                raise ValueError(f"unknown budget resource {k!r}; known={sorted(KNOWN_RESOURCES)}")
        for k, v in totals.items():
            _Counter._as_true_int(v, f"totals[{k!r}]")
        self.task_id = task_id
        self.counters = {k: _Counter(v) for k, v in totals.items()}

    # -- solve launch: consumes ONLY solve_launches --

    def reserve_launch(self) -> None:
        """Reserve one solve launch before dispatch. Consumes solve_launches only."""
        self.counters["solve_launches"].reserve(1)

    def commit_launch(self) -> None:
        """A launch started is consumed immediately; NOT refunded on exception."""
        self.counters["solve_launches"].commit_reservation(1)

    # -- tool request counting (each controller call = one tool request) --

    def charge_tool_request(self):
        """Count one tool request. Returns the updated counter; raises BudgetExhausted
        when the tool-request budget is exhausted. Callers MUST treat this as a hard
        stop: do not fall back to a dev counter and do not continue the call."""
        c = self.counters["tool_requests"]
        c.spend(1)
        return c

    # -- model request registration (trusted adapter ONLY, NOT in execute_task) --

    def register_model_request(self) -> None:
        """Trusted adapter MUST call this before a real scientific-task model
        request is issued. A LOCAL candidate solve (execute_task) does NOT call
        this. Until AGH wiring exists this only proves the interface + unit."""
        self.counters["model_requests"].spend(1)

    # -- function-callback budget: check before each real evaluation --

    def check_before_eval(self) -> None:
        """Called BEFORE each integrand evaluation; checks deadline + func_evals."""
        self.check_deadline()
        c = self.counters["func_evals"]
        if c.spent + c.reserved + 1 > c.total:
            raise BudgetExhausted(
                f"function evaluation would exceed budget "
                f"(spent={c.spent}, reserved={c.reserved}, total={c.total})"
            )

    def commit_eval(self) -> None:
        """Count one real integrand evaluation."""
        self.counters["func_evals"].spend(1)

    # -- deadline --

    def deadline_expires_at(self) -> float:
        return self._started_at + self.counters["wall_seconds"].total

    def check_deadline(self) -> None:
        if time.time() > self.deadline_expires_at():
            raise DeadlineExpired(
                f"wall-clock deadline exceeded (limit "
                f"{self.counters['wall_seconds'].total}s)"
            )

    def check_deadline_before_publish(self) -> None:
        """Re-check at publish time; an expired deadline cannot publish a pass."""
        self.check_deadline()

    def mark_started(self, now: Optional[float] = None) -> None:
        self._started_at = time.time() if now is None else now

    @property
    def started_at(self) -> float:
        return getattr(self, "_started_at", 0.0)

    def as_dict(self) -> Dict[str, Dict[str, int]]:
        out = {k: c.as_dict() for k, c in self.counters.items()}
        out["_meta"] = {"started_at": self.started_at, "task_id": self.task_id}
        return out

    def budget_card(self) -> Dict[str, Dict[str, int]]:
        """Per-resource budget + balance view. Different resources have DIFFERENT
        units (launches, evaluations, requests, seconds), so they must be listed
        individually -- they are NEVER summed into a single total."""
        return {k: c.as_dict() for k, c in self.counters.items()}

    @classmethod
    def from_dict(cls, task_id: str, d: Dict[str, Dict[str, int]]) -> "TaskBudget":
        totals = {k: v["total"] for k, v in d.items() if k != "_meta"}
        b = cls(task_id, totals)
        for k, v in d.items():
            if k == "_meta":
                continue
            b.counters[k].spent = v["spent"]
            b.counters[k].reserved = v["reserved"]
        b._started_at = d.get("_meta", {}).get("started_at", 0.0)
        return b


class BudgetLedger:
    """Key per-task TaskBudget instances by task_id. No shared single budget."""

    def __init__(self):
        self._by_task: Dict[str, TaskBudget] = {}

    def for_task(self, task_id: str, totals: Optional[Dict[str, int]] = None) -> TaskBudget:
        if task_id in self._by_task:
            return self._by_task[task_id]
        b = TaskBudget(task_id, totals)
        b.mark_started()
        self._by_task[task_id] = b
        return b

    def get(self, task_id: str) -> Optional[TaskBudget]:
        return self._by_task.get(task_id)

    def all(self) -> Dict[str, TaskBudget]:
        return dict(self._by_task)


@dataclass
class BudgetSnapshot:
    """Persistable snapshot of a task's budget (used by INCOMPLETE terminal)."""
    task_id: str
    counters: Dict[str, Dict[str, int]]

    def to_budget(self) -> TaskBudget:
        return TaskBudget.from_dict(self.task_id, self.counters)
