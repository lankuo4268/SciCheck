"""Unified task format, hashing, and the task/run store (append-only log).

- A task is normalized to a canonical dict: domain (task type), problem (original
  parameters), tolerance (absolute/relative). For integration, problem uses
  family=polynomial|piecewise_constant with a/b/coefficients or breakpoints/values.
- The semantic task_hash covers ONLY: normalized problem + original tolerance +
  semantic version. The same math problem under a different session/task_id
  produces the SAME hash; session and task_id are separate binding records used
  to verify binding, not part of the mathematical identity.
- run_id is a truly unique UUID (uuid4). Task records and legacy-prototype records
  are kept separately; logs are APPEND-ONLY, completed stage output is never
  overwritten.
- The store deep-freezes task inputs at register_task time; get_task/task_view
  return copies, not internal mutable references.
"""
from __future__ import annotations

import copy
import hashlib
import json
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from fractions import Fraction

# ---- canonical task shape ----
# domain: one of "integration" | "root" | "linear_system" | "ode"
# problem: original parameters for the chosen domain
# tolerance: {"absolute": <rational-str>, "relative": <rational-str>}

SEMANTIC_VERSION = "1.0.0"

# Global solve budget per task: shared across ALL actions.
DEFAULT_TASK_BUDGET = 3

# ODE bounds (see _normalize_ode):
# - ODE_COEFF_BOUND: |a|, |b|, |y0|, |dt| each <= this (coefficient/forcing/initial/
#   step magnitude bound; a may still be very small as long as z=a*dt stays in range).
# - ODE_TIME_BOUND: |t0|, |t1| each <= this (a finite large time origin is supported).
# - ODE_Z_BOUND: |z| = |a*dt| <= this (the exponential argument; keeps exp(z) in a
#   numerically reachable range for both the Decimal reference and binary64
#   candidates -- avoids exp(1e10)-class inputs).
ODE_COEFF_BOUND = 100000
ODE_TIME_BOUND = 1000000000000000  # 1e15
ODE_Z_BOUND = 20
# ODE reference precision: the high-precision Decimal closed-form is computed
# under a localcontext with this precision; a bounded "reference error
# envelope" (the directed-rounding bound of the whole expression, NOT a fixed
# 1e-98) is recorded alongside; when the candidate's forward-error interval and
# the tolerance interval overlap, the critical verdict returns INCONCLUSIVE.
ODE_REFERENCE_PRECISION = 100
# Number of extra decimal digits beyond ODE_REFERENCE_PRECISION used when the
# directed (round-up) envelope is computed for the NON-zero-a reference. The
# envelope covers the Fraction->Decimal casts, every arithmetic step, and the
# exp, each bounded conservatively; it is NOT a general interval proof.
ODE_REFERENCE_ENVELOPE_DIGITS = 5


# ---- terminal state set (unified) ----
# VERIFIED_PASS      : candidate met the original tolerance (numeric pass)
# UNSUPPORTED        : capability rejected (no whitelisted solver for domain/family)
# INCOMPLETE         : no budget / deadline to finish; conservative stop on restart
# INCONCLUSIVE       : insufficient evidence to accept or reject
# EXECUTION_ERROR    : a solve launch raised an exception
# VERIFIED_INCORRECT is a RUN verdict (candidate did not reach the tolerance) and
# is NOT a numeric pass; it is not a terminal state for the task.
TERMINAL_STATES = {
    "VERIFIED_PASS", "UNSUPPORTED", "INCOMPLETE", "INCONCLUSIVE", "EXECUTION_ERROR",
}
RUN_VERDICTS = {
    "VERIFIED_INCORRECT",  # run-level: failed the original tolerance
}


class TaskFormatError(Exception):
    pass


class TaskSessionMismatch(Exception):
    pass


class TaskFrozen(Exception):
    """Raised when an attempt is made to modify a frozen task."""


class TaskAlreadyTerminal(Exception):
    """Raised when trying to start a new solve on a terminal task."""


# ---- strict validation helpers ----

def _reject_bool_numeric(v: Any, where: str) -> None:
    """Reject booleans masquerading as numerics (True/False are ints in Python)."""
    if isinstance(v, bool):
        raise TaskFormatError(f"{where} must not be a boolean numeric value")


def _parse_positive_rational(s: Any, where: str, allow_zero: bool = False) -> Fraction:
    """Parse a rational string; must be strictly positive (or >=0 if allow_zero)."""
    _reject_bool_numeric(s, where)
    if not isinstance(s, str):
        raise TaskFormatError(f"{where} must be a rational string, got {type(s).__name__}")
    try:
        f = Fraction(s.strip())
    except Exception as e:
        raise TaskFormatError(f"{where} is not a parseable rational: {e}")
    if f <= 0 and not allow_zero:
        raise TaskFormatError(f"{where} must be a strictly positive rational, got {f}")
    if f < 0:
        raise TaskFormatError(f"{where} must be non-negative, got {f}")
    return f


def _parse_rational_strictly(s: Any, where: str) -> Fraction:
    _reject_bool_numeric(s, where)
    if not isinstance(s, str):
        raise TaskFormatError(f"{where} must be a rational string, got {type(s).__name__}")
    try:
        return Fraction(s.strip())
    except Exception as e:
        raise TaskFormatError(f"{where} is not a parseable rational: {e}")


def _require_no_extra_fields(d: Dict[str, Any], allowed: List[str], label: str) -> None:
    extra = [k for k in d.keys() if k not in allowed]
    if extra:
        raise TaskFormatError(f"{label} has unexpected fields {extra}; only {allowed} allowed")


def normalize_task(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize and validate a unified task.

    Rejections (explicit, no silent mixing):
      - unknown top-level fields
      - unknown domain
      - integration: unknown family, missing required keys, extra fields
      - boolean numeric values anywhere
      - tolerance.absolute must be strictly positive
      - tolerance.relative must be non-negative
      - a < b (strict)
      - piecewise_constant: breakpoints strictly increasing, no overlapping/empty intervals
    """
    _reject_bool_numeric(raw, "task")
    allowed_top = ["domain", "problem", "tolerance"]
    _require_no_extra_fields(raw, allowed_top, "task")

    domain = raw.get("domain")
    if domain not in ("integration", "root", "linear_system", "ode"):
        raise TaskFormatError(f"unknown domain {domain!r}")

    problem = raw.get("problem")
    if not isinstance(problem, Dict):
        raise TaskFormatError("'problem' must be an object")

    tol = raw.get("tolerance")
    if not isinstance(tol, Dict):
        raise TaskFormatError("'tolerance' must be an object")
    _require_no_extra_fields(tol, ["absolute", "relative"], "tolerance")
    _reject_bool_numeric(tol.get("absolute", ""), "tolerance.absolute")
    _reject_bool_numeric(tol.get("relative", "0"), "tolerance.relative")
    # strict checks
    _ = _parse_positive_rational(tol.get("absolute", ""), "tolerance.absolute", allow_zero=False)
    _ = _parse_positive_rational(tol.get("relative", "0"), "tolerance.relative", allow_zero=True)

    if domain == "integration":
        problem = _normalize_integration(problem)
    elif domain == "root":
        problem = _normalize_root(problem)
    elif domain == "linear_system":
        problem = _normalize_linear_system(problem)
    elif domain == "ode":
        problem = _normalize_ode(problem)

    canonical = {
        "domain": domain,
        "problem": problem,
        "tolerance": {
            "absolute": str(tol.get("absolute", "")),
            "relative": str(tol.get("relative", "0")),
        },
    }
    return canonical


def _normalize_root(problem: Dict[str, Any]) -> Dict[str, Any]:
    """Root domain: f(x) = scale * (x - shift)^power, power in {1,3,5}, scale != 0.
    The public interval [a, b] must STRICTLY enclose the shift: a < shift < b.
    """
    _reject_bool_numeric(problem, "problem")
    allowed = ["family", "power", "scale", "shift", "a", "b", "start"]
    _require_no_extra_fields(problem, allowed, "problem")

    family = problem.get("family")
    if family != "scaled_power":
        raise TaskFormatError(f"root problem.family must be 'scaled_power', got {family!r}")

    power_f = _parse_rational_strictly(problem.get("power", ""), "problem.power")
    if power_f != power_f.numerator:
        raise TaskFormatError(f"root power must be an integer, got {power_f}")
    power = power_f.numerator
    if power not in (1, 3, 5):
        raise TaskFormatError(f"root power must be one of (1, 3, 5), got {power}")

    scale = _parse_rational_strictly(problem.get("scale", ""), "problem.scale")
    if scale == 0:
        raise TaskFormatError("root scale must be non-zero")

    shift = _parse_rational_strictly(problem.get("shift", ""), "problem.shift")
    a = _parse_rational_strictly(problem.get("a", ""), "problem.a")
    b = _parse_rational_strictly(problem.get("b", ""), "problem.b")

    # STRICT rational enclosure: a < shift < b, exact, no float involved.
    if not (a < shift < b):
        raise TaskFormatError(
            f"public interval [{a}, {b}] must STRICTLY enclose the shift {shift}"
        )

    # Float-geometry guard: the solvers operate on real brackets [float(a), float(b)].
    # If rounding collapses a boundary onto the root, the bracket no longer strictly
    # contains the root and the geometric relationship is lost -> refuse the task.
    fa, fb, fsh = float(a), float(b), float(shift)
    if fa >= fsh or fb <= fsh:
        raise TaskFormatError(
            "float representation cannot preserve the interval geometry: "
            f"float(a)={fa!r}, float(b)={fb!r}, float(shift)={fsh!r}; the root is not "
            "strictly inside the solver bracket -- task refused"
        )

    canonical = {
        "family": family,
        "power": str(power),
        "scale": str(scale),
        "shift": str(shift),
        "a": str(a),
        "b": str(b),
    }
    # Optional solver start (public hint, not part of the mathematical identity).
    if "start" in problem:
        start = _parse_rational_strictly(problem["start"], "problem.start")
        canonical["start"] = str(start)
    return canonical



def _normalize_ode(problem: Dict[str, Any]) -> Dict[str, Any]:
    """ODE domain: scalar linear IVP y' = a*y + b on [t0, t1], t0 < t1. The
    structured inputs are coefficients (a, b), initial value y0, and the time
    interval [t0, t1]. All rational strings are parsed via Fraction FIRST (no
    float('p/q')):

    - a (coefficient of y), b (forcing), y0 (initial value): each bounded
      |value| <= ODE_COEFF_BOUND (100000). May be zero or negative.
    - t0, t1: time endpoints, each bounded |value| <= ODE_TIME_BOUND (1e15).
      A finite large time origin is supported; t0 < t1 is required.
    - dt = t1 - t0 (EXACT Fraction, computed here, NOT a canonical field so the
      canonical is idempotent under re-normalization): bounded |dt| <= ODE_COEFF_BOUND.
    - z = a * dt (the exponential argument that drives the closed form
      exp(a*dt)): bounded |z| <= ODE_Z_BOUND (20) so exp(z) stays in a
      numerically reachable range for both the Decimal reference and the
      binary64 candidates (avoids exp(1e10)-class inputs). a may still be
      very small (or even zero) as long as z stays in range.

    The canonical dict contains ONLY the ORIGINAL structured inputs (a, b, y0,
    t0, t1) -- dt is a DERIVED quantity recomputed on every normalize pass, so
    re-normalizing the canonical itself (on a SQLite restart) succeeds without
    a spurious extra-field rejection."""
    _reject_bool_numeric(problem, "problem")
    allowed = ["family", "a", "b", "y0", "t0", "t1"]
    _require_no_extra_fields(problem, allowed, "problem")

    family = problem.get("family")
    if family != "linear_ivp":
        raise TaskFormatError(f"ode problem.family must be 'linear_ivp', got {family!r}")

    def _parse_rational(s, where):
        _reject_bool_numeric(s, where)
        if not isinstance(s, str):
            raise TaskFormatError(f"{where} must be a rational string, got {type(s).__name__}")
        try:
            return Fraction(s.strip())
        except Exception as e:
            raise TaskFormatError(f"{where} is not a parseable rational: {e}")

    # a, b, y0: coefficient bounds (may be 0 or negative)
    a = _parse_rational(problem["a"], "problem.a")
    b = _parse_rational(problem["b"], "problem.b")
    y0 = _parse_rational(problem["y0"], "problem.y0")
    if abs(a) > ODE_COEFF_BOUND:
        raise TaskFormatError(f"ode coefficient a={a} magnitude exceeds {ODE_COEFF_BOUND}")
    if abs(b) > ODE_COEFF_BOUND:
        raise TaskFormatError(f"ode forcing b={b} magnitude exceeds {ODE_COEFF_BOUND}")
    if abs(y0) > ODE_COEFF_BOUND:
        raise TaskFormatError(f"ode initial value y0={y0} magnitude exceeds {ODE_COEFF_BOUND}")
    # t0, t1: time-endpoint bounds (finite large origin supported)
    t0 = _parse_rational(problem["t0"], "problem.t0")
    t1 = _parse_rational(problem["t1"], "problem.t1")
    if abs(t0) > ODE_TIME_BOUND:
        raise TaskFormatError(f"ode time origin t0={t0} magnitude exceeds {ODE_TIME_BOUND}")
    if abs(t1) > ODE_TIME_BOUND:
        raise TaskFormatError(f"ode time endpoint t1={t1} magnitude exceeds {ODE_TIME_BOUND}")
    if t0 >= t1:
        raise TaskFormatError(f"ode requires t0 < t1; got t0={t0} t1={t1}")
    # dt (EXACT Fraction, derived) and z = a*dt (the exponential argument) bounds
    dt = t1 - t0
    if abs(dt) > ODE_COEFF_BOUND:
        raise TaskFormatError(f"ode dt={dt} magnitude exceeds {ODE_COEFF_BOUND}")
    z = a * dt
    if abs(z) > ODE_Z_BOUND:
        raise TaskFormatError(
            f"ode exponential argument a*dt={z} magnitude exceeds {ODE_Z_BOUND} "
            f"(a={a}, dt={dt}); exp(z) would not stay in a reachable range")

    canonical = {
        "family": "linear_ivp",
        "a": str(a),
        "b": str(b),
        "y0": str(y0),
        "t0": str(t0),
        "t1": str(t1),
    }
    return canonical


def _normalize_linear_system(problem: Dict[str, Any]) -> Dict[str, Any]:
    """Linear-system domain: A x = b where A is an n x n rational square matrix
    (1 <= n <= 8) and b is a rational vector of length n. EVERY rational string is
    parsed via Fraction FIRST; no float('p/q') is performed at normalization time.

    Problem fields: family="rational_square_matrix", "n" (int string 1..8),
    "matrix" (n x n list-of-lists of rational strings), "rhs" (n-list of rational
    strings). Dimensions must agree and the matrix must be square."""
    _reject_bool_numeric(problem, "problem")
    allowed = ["family", "n", "matrix", "rhs"]
    _require_no_extra_fields(problem, allowed, "problem")

    family = problem.get("family")
    if family != "rational_square_matrix":
        raise TaskFormatError(
            f"linear_system problem.family must be 'rational_square_matrix', got {family!r}")

    def _frac(v, where):
        f = _parse_rational_strictly(v, where)
        f._parse_rational_strictly_orig = True  # mark parsed
        return f

    n = _parse_rational_strictly(problem.get("n", ""), "problem.n")
    if n != n.numerator:
        raise TaskFormatError(f"problem.n must be an integer, got {n}")
    n = int(n.numerator)
    if not (1 <= n <= 8):
        raise TaskFormatError(f"problem.n must be in 1..8, got {n}")

    raw_matrix = problem.get("matrix")
    if not isinstance(raw_matrix, List) or len(raw_matrix) != n:
        raise TaskFormatError(f"problem.matrix must be a list of {n} rows, "
                              f"got {len(raw_matrix) if isinstance(raw_matrix, List) else type(raw_matrix).__name__}")
    matrix: List[List[Fraction]] = []
    for i, row in enumerate(raw_matrix):
        if not isinstance(row, List) or len(row) != n:
            raise TaskFormatError(f"problem.matrix[{i}] must be a list of {n} entries, "
                                  f"got {len(row) if isinstance(row, List) else type(row).__name__}")
        parsed_row: List[Fraction] = []
        for j, entry in enumerate(row):
            _reject_bool_numeric(entry, f"problem.matrix[{i}][{j}]")
            parsed_row.append(_parse_rational_strictly(entry, f"problem.matrix[{i}][{j}]") 
                              if isinstance(entry, str) else _reject_matrix_entry(entry, f"problem.matrix[{i}][{j}]"))
        matrix.append(parsed_row)

    raw_rhs = problem.get("rhs")
    if not isinstance(raw_rhs, List) or len(raw_rhs) != n:
        raise TaskFormatError(f"problem.rhs must be a list of {n} entries, "
                              f"got {len(raw_rhs) if isinstance(raw_rhs, List) else type(raw_rhs).__name__}")
    rhs: List[Fraction] = []
    for i, entry in enumerate(raw_rhs):
        _reject_bool_numeric(entry, f"problem.rhs[{i}]")
        rhs.append(_parse_rational_strictly(entry, f"problem.rhs[{i}]") 
                   if isinstance(entry, str) else _reject_matrix_entry(entry, f"problem.rhs[{i}]"))
    return _linsys_canonical(n, matrix, rhs)


def _reject_matrix_entry(entry, where):
    """Non-string matrix/rhs entries must be well-formed rationals; reject booleans,
    floats, and non-parseable values (bools are rejected by the caller's
    _reject_bool_numeric; this handles the rest)."""
    if isinstance(entry, (int,)) and not isinstance(entry, bool):
        return Fraction(entry)
    raise TaskFormatError(f"{where} must be a rational string or integer, got {type(entry).__name__}")


def _linsys_canonical(n: int, matrix, rhs):
    """Serialize a validated linear-system problem into the frozen canonical form.
    Every entry is a Fraction; canonical stores them as their canonical rational
    string. Shared by _normalize_linear_system and normalize_task so that a
    canonical payload re-validates identically (idempotent)."""
    return {
        "family": "rational_square_matrix",
        "n": str(n),
        "matrix": [[str(v) for v in row] for row in matrix],
        "rhs": [str(v) for v in rhs],
    }


def _normalize_integration(problem: Dict[str, Any]) -> Dict[str, Any]:
    _reject_bool_numeric(problem, "problem")
    allowed = ["family", "a", "b", "coefficients", "breakpoints", "values"]
    _require_no_extra_fields(problem, allowed, "problem")

    family = problem.get("family")
    if family not in ("polynomial", "piecewise_constant"):
        raise TaskFormatError(f"integration problem.family must be 'polynomial' or "
                              f"'piecewise_constant', got {family!r}")

    a_raw = problem.get("a")
    b_raw = problem.get("b")
    _reject_bool_numeric(a_raw, "problem.a")
    _reject_bool_numeric(b_raw, "problem.b")
    a = _parse_rational_strictly(a_raw, "problem.a")
    b = _parse_rational_strictly(b_raw, "problem.b")
    if not (a < b):
        raise TaskFormatError(f"integration requires a < b, got a={a} b={b}")

    if family == "polynomial":
        coeffs = problem.get("coefficients")
        if not isinstance(coeffs, List) or len(coeffs) == 0:
            raise TaskFormatError("polynomial family requires non-empty 'coefficients' list")
        for i, c in enumerate(coeffs):
            _reject_bool_numeric(c, f"problem.coefficients[{i}]")
        canonical = {
            "family": family,
            "a": str(a),
            "b": str(b),
            "coefficients": [str(_parse_rational_strictly(c, f"problem.coefficients[{i}]")) for i, c in enumerate(coeffs)],
        }
    else:  # piecewise_constant
        bps = problem.get("breakpoints")
        vals = problem.get("values")
        if not isinstance(bps, List) or not isinstance(vals, List):
            raise TaskFormatError("piecewise_constant requires 'breakpoints' and 'values' lists")
        for x in bps:
            _reject_bool_numeric(x, "problem.breakpoints[]")
        for v in vals:
            _reject_bool_numeric(v, "problem.values[]")
        if len(vals) != len(bps) + 1:
            raise TaskFormatError(
                f"piecewise_constant: len(values) must be len(breakpoints)+1 "
                f"(got breakpoints={len(bps)}, values={len(vals)})"
            )
        # ordered, strictly increasing, no overlaps
        ordered = [a] + [_parse_rational_strictly(x, "problem.breakpoints[]") for x in bps] + [b]
        for i in range(len(ordered) - 1):
            if ordered[i] >= ordered[i + 1]:
                raise TaskFormatError(
                    f"piecewise_constant intervals must be strictly increasing; "
                    f"at index {i} have {ordered[i]} >= {ordered[i+1]}"
                )
        canonical = {
            "family": family,
            "a": str(a),
            "b": str(b),
            "breakpoints": [str(x) for x in ordered[1:-1]],
            "values": [str(_parse_rational_strictly(v, "problem.values[]")) for v in vals],
        }
    return canonical


def build_task_hash(canonical: Dict[str, Any], semantic_version: str = SEMANTIC_VERSION) -> str:
    """Semantic hash: normalized problem + original tolerance + semantic version ONLY.
    Same math under different session/task_id -> same hash."""
    payload = {
        "problem": canonical["problem"],
        "tolerance": canonical["tolerance"],
        "domain": canonical["domain"],
        "semantic_version": semantic_version,
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


# ---- run record (unified type used by both scicheck and task layer) ----

@dataclass
class RunRecord:
    run_id: str
    task_id: str
    action: str              # the card/action that produced this candidate
    candidate: Any          # actual computed value: float for scalar domains,
                           # a list for linear-system vectors, or None when an ODE
                           # solver did not reach t1 -- never a NaN/Infinity.
    quad_abserr_reported: Any  # diagnostic residual/abserr; may be a float or None
    exact_ok: Optional[bool]   # tri-state: True (PASS), False (INCORRECT),
                              # None (INCONCLUSIVE, e.g. an ODE run with no valid t1
                              # candidate or a too-wide reference interval)
    exact_check: Dict[str, Any]   # independent verdict receipt
    resource: Dict[str, Any]
    method: str = ""        # solver method label (e.g. "quad_default")
    terminal_status: Optional[str] = None
    created_by: str = "execute_task"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "task_id": self.task_id,
            "action": self.action,
            "method": self.method,
            "candidate": self.candidate,
            "quad_abserr_reported": self.quad_abserr_reported,
            "exact_ok": self.exact_ok,
            "exact_check": self.exact_check,
            "resource": self.resource,
            "terminal_status": self.terminal_status,
            "created_by": self.created_by,
        }


def scicheck_run_to_record(sc_rec, task_id: str, run_id: str, action: str,
                           created_by: str = "execute_task") -> RunRecord:
    """Convert a scicheck.RunRecord into the task-layer RunRecord."""
    return RunRecord(
        run_id=run_id,
        task_id=task_id,
        action=action,
        method=getattr(sc_rec, "method", action),
        candidate=sc_rec.value,
        quad_abserr_reported=sc_rec.quad_abserr_reported,
        exact_ok=sc_rec.exact_ok,
        exact_check=sc_rec.exact_check,
        resource=sc_rec.resource,
        terminal_status=None,
        created_by=created_by,
    )


class TaskStore:
    """Append-only, in-memory store binding session+task_id to runs and task state.

    - task records and legacy-prototype records are kept separately.
    - logs append; completed stage output is never overwritten.
    - a terminal task refuses to start new solves.
    - get_task / task_view return deep copies, never internal mutable references.
    """

    TERMINAL_STATES = TERMINAL_STATES

    def __init__(self, session: str, semantic_version: str = SEMANTIC_VERSION,
                 task_budget: int = DEFAULT_TASK_BUDGET):
        self.session = session
        self.semantic_version = semantic_version
        self.task_budget = task_budget
        self._tasks: Dict[str, Dict[str, Any]] = {}
        self._runs: Dict[str, RunRecord] = {}
        # idempotency scoped by (task_id, request_id) -> bound to action_id
        self._request_index: Dict[Tuple[str, str, str], str] = {}
        self._request_action: Dict[Tuple[str, str], str] = {}  # (task_id, request_id) -> first action
        # separate ledger for legacy-prototype records (never mixed with task records)
        self._legacy_records: List[Dict[str, Any]] = []

    # -- task lifecycle --

    def register_task(self, task_id: str, canonical: Dict[str, Any]) -> Dict[str, Any]:
        """register_task normalizes its input itself via normalize_task; the caller
        is NOT expected to pre-validate. Invalid tolerance / interval / extra fields
        are rejected here, so a malformed task never becomes a pseudo-legal one.
        The caller's dict is deep-copied and frozen; later mutation has no effect."""
        if task_id in self._tasks:
            raise TaskFrozen(f"task {task_id} already registered; tasks are frozen after creation")
        canonical_norm = normalize_task(canonical)  # validation lives here, not the caller
        canonical_frozen = copy.deepcopy(canonical_norm)
        task_hash = build_task_hash(canonical_frozen, self.semantic_version)
        self._tasks[task_id] = {
            "task_id": task_id,
            "session": self.session,
            "hash": task_hash,
            "canonical": canonical_frozen,
            "state": "ACTIVE",
            "runs": [],
        }
        return copy.deepcopy(self._tasks[task_id])

    def get_task(self, task_id: str) -> Dict[str, Any]:
        """Return a deep copy of the task record; never the internal mutable ref."""
        t = self._tasks.get(task_id)
        if t is None:
            raise KeyError(f"unknown task {task_id} in session {self.session}")
        return copy.deepcopy(t)

    def _get_task_internal(self, task_id: str) -> Dict[str, Any]:
        """Internal mutable access; only used by the store itself."""
        t = self._tasks.get(task_id)
        if t is None:
            raise KeyError(f"unknown task {task_id} in session {self.session}")
        return t

    def is_terminal(self, task_id: str) -> bool:
        return self._get_task_internal(task_id)["state"] in self.TERMINAL_STATES

    def mark_terminal(self, task_id: str, status: str, reason: str = "") -> None:
        if status not in self.TERMINAL_STATES:
            raise TaskFormatError(f"{status!r} is not a terminal state")
        t = self._get_task_internal(task_id)
        if t["state"] in self.TERMINAL_STATES:
            raise TaskAlreadyTerminal(f"task {task_id} is already terminal ({t['state']})")
        t["state"] = status
        t["terminal_reason"] = reason
        t["terminal_at"] = len(t["runs"])

    # -- runs (append-only, UUID ids) --

    def new_run_id(self) -> str:
        return f"run-{uuid.uuid4()}"

    def append_run(self, run: RunRecord) -> None:
        if run.run_id in self._runs:
            raise TaskFrozen(f"run {run.run_id} already recorded; logs are append-only")
        self._runs[run.run_id] = run
        self._get_task_internal(run.task_id)["runs"].append(run.run_id)

    def get_run(self, run_id: str) -> RunRecord:
        return self._runs[run_id]

    def get_run_copy(self, run_id: str) -> Dict[str, Any]:
        return self._runs[run_id].to_dict()

    def legacy_record(self, payload: Dict[str, Any]) -> None:
        self._legacy_records.append(copy.deepcopy(payload))

    def legacy_records(self) -> List[Dict[str, Any]]:
        return [copy.deepcopy(r) for r in self._legacy_records]

    # -- idempotency --

    def record_request(self, task_id: str, request_id: str, action_id: str, run_id: str) -> None:
        """Record that (session-scoped) task_id used request_id with action_id, producing
        run_id. Scoped per task so the same request_id in a different task executes
        independently. A later call with the same request_id but a DIFFERENT action
        in the SAME task is rejected by the controller; the same triple reuses the run."""
        key = (task_id, request_id)
        prev_action = self._request_action.get(key)
        if prev_action is not None and prev_action != action_id:
            raise TaskFrozen(
                f"request_id {request_id!r} in task {task_id!r} was already used with "
                f"action {prev_action!r}; changing action to {action_id!r} is rejected"
            )
        self._request_action[key] = action_id
        self._request_index[(task_id, request_id, action_id)] = run_id

    def lookup_request(self, task_id: str, request_id: str, action_id: str) -> Optional[str]:
        """Idempotency index is scoped by (session, task_id, request_id) and bound to
        action_id. Same task + same request + same action reuses the run; the same
        request_id in a DIFFERENT task executes independently (no cross-task bleed)."""
        return self._request_index.get((task_id, request_id, action_id))

    def request_action(self, task_id: str, request_id: str) -> Optional[str]:
        return self._request_action.get((task_id, request_id))

    def task_by_hash(self, task_hash: str) -> Optional[Dict[str, Any]]:
        """Return the first task whose semantic hash matches, or None."""
        for t in self._tasks.values():
            if t["hash"] == task_hash:
                return t
        return None

    def list_task_ids(self) -> List[str]:
        return list(self._tasks.keys())

    # -- session/task binding --

    def assert_session_binding(self, task_id: str, session: str) -> None:
        t = self._tasks.get(task_id)
        if t is None:
            # Task not in this store: reject as cross-session (cannot be operated on here)
            raise TaskSessionMismatch(
                f"task {task_id!r} is not present in session {self.session!r} (cross-session)"
            )
        if session != t["session"]:
            raise TaskSessionMismatch(
                f"task {task_id} belongs to session {t['session']!r}, not {session!r}"
            )
