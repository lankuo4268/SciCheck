"""ODE closed-form stability example flow (LOCAL QA sample, not an AGH scientific job).

Writes three real ODE cases using the trusted TaskRegistrar:

  1. growth        y'=y, y0=1, [0,1], atol=1e-8, rtol=0
                   diagnostic task: ode_euler10 -> ode_dop853
                   direct task:      ode_dop853
  2. small_coeff   a=1e-12, b=1, y0=1, [0,1], atol=1e-8, rtol=0
                   diagnostic task: ode_closed_naive -> ode_closed_stable
                   direct task:      ode_closed_stable
  3. large_origin  a=0, b=1, y0=1, t0=1e12, t1=5000000000000001/5000,
                   atol=1e-12, rtol=0
                   diagnostic task: ode_closed_naive -> ode_closed_stable
                   direct task:      ode_closed_stable

For each case:
- The diagnostic task runs the two routes sequentially on ONE task, sharing the
  original per-task budget. The first route is asserted to fail with
  exact_ok is False (a numeric VERIFIED_INCORRECT, not an exception or None).
  The second route must pass (exact_ok is True, VERIFIED_PASS terminal).
- The direct task is a NEW trusted task with the SAME canonical problem and
  SAME original tolerance (hence the same task hash); it runs ONLY the strong
  public route (ode_dop853 for case 1, ode_closed_stable for cases 2 and 3).
  The diagnostic task's terminal state and run records are NOT modified.

The private SQLite db + receipt key are written to private_runs/ and are NEVER
exported into the JSON. The JSON is self-contained (original problem, original
tolerance, task hash, actual execute_task returns, resource, verify_run, task
state, budget.as_dict(), source hashes, origin marker).

Run:
  D:/anaconda/python.exe run_ode_examples.py
"""
import hashlib
import json
import os
import sys
import time
import uuid
from pathlib import Path

from task_controller import TaskRegistrar, TaskError, BudgetExhausted, TaskAlreadyTerminal


def _detect_leak(obj) -> bool:
    forbidden = {
        "reference_fraction", "reference_answer", "reference_value",
        "exact_check", "candidate_as_fraction", "abs_diff_exact_fraction",
        "abs_diff_fraction", "allowed_error_fraction",
        "reference_lo_fraction", "reference_hi_fraction",
        "allowed_error_min_fraction", "allowed_error_max_fraction",
        "forward_error_min_fraction", "forward_error_max_fraction",
    }
    if isinstance(obj, dict):
        if forbidden & set(obj.keys()):
            return True
        return any(_detect_leak(v) for v in obj.values())
    if isinstance(obj, list):
        return any(_detect_leak(v) for v in obj)
    return False


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: str) -> str:
    with open(path, "rb") as f:
        return _sha256_bytes(f.read())


def _task_hash(registrar, task_id):
    return registrar.store.get_task(task_id)["hash"]


def _task_state(registrar, task_id):
    return registrar.store.get_task(task_id)["state"]


def _budget_snapshot(registrar, task_id):
    """Return the task's real per-task budget as_dict() (no fake solver field)."""
    b = registrar.ledger.get(task_id)
    if b is None:
        return None
    return b.as_dict()


def _run_route(registrar, task_id, request_id, route):
    """Run one route on the given task; return the full execute_task view."""
    return registrar.execute_task(task_id, request_id, route)


def _run_diagnostic_case(registrar, case_name, task_id, canonical,
                        route_fail, route_pass):
    """One diagnostic task: run route_fail (must be exact_ok is False), then
    route_pass (must be exact_ok is True), sharing the original budget."""
    registrar.register_task(task_id, dict(canonical))
    t_hash = _task_hash(registrar, task_id)

    r1 = _run_route(registrar, task_id, "req-d1", route_fail)
    # The first route MUST numerically fail (exact_ok is False). An exception
    # (BudgetExhausted / TaskError) or a no-candidate (exact_ok is None) is NOT
    # an acceptable "numeric failure" for this sample; assert explicitly.
    assert r1.get("exact_ok") is False, (
        f"{case_name}: diagnostic first route {route_fail!r} expected a numeric "
        f"VERIFIED_INCORRECT (exact_ok is False), got exact_ok={r1.get('exact_ok')!r} "
        f"verdict={r1.get('verdict')!r}; an exception or missing candidate is not "
        f"the intended diagnostic-fail shape")
    state_after_fail = _task_state(registrar, task_id)
    assert state_after_fail == "ACTIVE", (
        f"{case_name}: after first route the task should still be ACTIVE "
        f"(solve budget not yet exhausted), got {state_after_fail!r}")

    r2 = _run_route(registrar, task_id, "req-d2", route_pass)
    assert r2.get("exact_ok") is True, (
        f"{case_name}: diagnostic second route {route_pass!r} expected VERIFIED_PASS "
        f"(exact_ok is True), got exact_ok={r2.get('exact_ok')!r} "
        f"verdict={r2.get('verdict')!r}")
    state_after_pass = _task_state(registrar, task_id)
    assert state_after_pass == "VERIFIED_PASS", (
        f"{case_name}: after second route the task should be VERIFIED_PASS, "
        f"got {state_after_pass!r}")

    verify_r2 = None
    if r2.get("run_id"):
        verify_r2 = registrar.verify_run(task_id, r2["run_id"])

    budget_after = _budget_snapshot(registrar, task_id)

    return {
        "case": case_name,
        "kind": "diagnostic",
        "task_id": task_id,
        "task_hash": t_hash,
        "original_problem": canonical["problem"],
        "original_tolerance": canonical["tolerance"],
        "route_fail": route_fail,
        "route_pass": route_pass,
        "execute_fail": r1,
        "state_after_fail": state_after_fail,
        "execute_pass": r2,
        "state_after_pass": state_after_pass,
        "verify_pass": verify_r2,
        "budget_after": budget_after,
    }


def _run_direct_case(registrar, case_name, task_id, canonical, route):
    """One NEW trusted task: run ONLY the strong public route (independent route,
    same canonical problem + original tolerance, hence same task hash)."""
    registrar.register_task(task_id, dict(canonical))
    t_hash = _task_hash(registrar, task_id)

    r = _run_route(registrar, task_id, "req-direct", route)
    assert r.get("exact_ok") is True, (
        f"{case_name}: direct route {route!r} expected VERIFIED_PASS "
        f"(exact_ok is True), got exact_ok={r.get('exact_ok')!r} "
        f"verdict={r.get('verdict')!r}")
    state_after = _task_state(registrar, task_id)
    assert state_after == "VERIFIED_PASS", (
        f"{case_name}: direct task should be VERIFIED_PASS, got {state_after!r}")

    verify = None
    if r.get("run_id"):
        verify = registrar.verify_run(task_id, r["run_id"])

    budget_after = _budget_snapshot(registrar, task_id)

    return {
        "case": case_name,
        "kind": "direct",
        "task_id": task_id,
        "task_hash": t_hash,
        "original_problem": canonical["problem"],
        "original_tolerance": canonical["tolerance"],
        "route": route,
        "execute": r,
        "state_after": state_after,
        "verify": verify,
        "budget_after": budget_after,
    }


def _main() -> int:
    tag = uuid.uuid4().hex
    base = Path("private_runs")
    base.mkdir(parents=True, exist_ok=True)
    db_path = str(base / f"tasks_ode_stability_{tag}.db")
    key_path = str(base / f"tasks_ode_stability_{tag}.receipt.key")
    out_path = str(base / f"ode_stability_{tag}.json")

    registrar = TaskRegistrar(session="S-ode-stability",
                              db_path=db_path, key_path=key_path)

    out = {
        "origin": "AGH_DEVELOPMENT_SHELL_LOCAL_QA",
        "not_agh_scientific_job": True,
        "tag": tag,
        "session": registrar.session,
        "db": db_path,
        "key_fingerprint": registrar.persistence.sealer.key_fingerprint,
        "key_material_included": False,
        "cases": [],
    }

    # ---- Case 1: growth y'=y, y0=1, [0,1], atol=1e-8, rtol=0 ----
    growth = {
        "domain": "ode",
        "problem": {
            "family": "linear_ivp",
            "a": "1", "b": "0", "y0": "1",
            "t0": "0", "t1": "1",
        },
        "tolerance": {"absolute": "1/100000000", "relative": "0"},
    }
    out["cases"].append(_run_diagnostic_case(
        registrar, "growth_y_eq_y", "ode-growth-diag", growth,
        route_fail="ode_euler10", route_pass="ode_dop853"))
    out["cases"].append(_run_direct_case(
        registrar, "growth_y_eq_y", "ode-growth-direct", growth,
        route="ode_dop853"))

    # ---- Case 2: small coefficient a=1e-12, b=1, y0=1, [0,1], atol=1e-8 ----
    small_coeff = {
        "domain": "ode",
        "problem": {
            "family": "linear_ivp",
            "a": "1/1000000000000", "b": "1", "y0": "1",
            "t0": "0", "t1": "1",
        },
        "tolerance": {"absolute": "1/100000000", "relative": "0"},
    }
    out["cases"].append(_run_diagnostic_case(
        registrar, "small_coeff_a_1e_12", "ode-small-coeff-diag", small_coeff,
        route_fail="ode_closed_naive", route_pass="ode_closed_stable"))
    out["cases"].append(_run_direct_case(
        registrar, "small_coeff_a_1e_12", "ode-small-coeff-direct", small_coeff,
        route="ode_closed_stable"))

    # ---- Case 3: large origin a=0, b=1, y0=1, t0=1e12,
    # t1=5000000000000001/5000, atol=1e-12, rtol=0 ----
    large_origin = {
        "domain": "ode",
        "problem": {
            "family": "linear_ivp",
            "a": "0", "b": "1", "y0": "1",
            "t0": "1000000000000",
            "t1": "5000000000000001/5000",
        },
        "tolerance": {"absolute": "1/1000000000000", "relative": "0"},
    }
    out["cases"].append(_run_diagnostic_case(
        registrar, "large_origin_t0_1e12", "ode-large-origin-diag", large_origin,
        route_fail="ode_closed_naive", route_pass="ode_closed_stable"))
    out["cases"].append(_run_direct_case(
        registrar, "large_origin_t0_1e12", "ode-large-origin-direct", large_origin,
        route="ode_closed_stable"))

    registrar.persistence.close()

    # After closing the db connection, capture the source-file hashes and the
    # db/key file sizes (NO key bytes are ever copied into the JSON).
    out["source_hashes"] = {
        "method_registry.py": _sha256_file(os.path.join(os.path.dirname(__file__), "method_registry.py")),
        "task_controller.py": _sha256_file(os.path.join(os.path.dirname(__file__), "task_controller.py")),
        "scicheck.py": _sha256_file(os.path.join(os.path.dirname(__file__), "scicheck.py")),
        "run_ode_examples.py": _sha256_file(os.path.abspath(__file__)),
    }
    out["db_sha256"] = _sha256_file(db_path)
    out["key_sha256_present"] = os.path.exists(key_path)
    # The key file's hash is included ONLY to prove the key material file exists
    # on disk; the key BYTES themselves are never copied into the JSON.
    out["key_sha256"] = _sha256_file(key_path) if os.path.exists(key_path) else None
    out["reference_answer_leaked"] = _detect_leak(out)
    assert not out["reference_answer_leaked"], "JSON must not leak reference fields"

    json_text = json.dumps(out, indent=2, allow_nan=False, sort_keys=False)
    Path(out_path).write_text(json_text, encoding="utf-8")

    print(f"wrote {out_path}")
    print(f"  db: {db_path} (sha256={out['db_sha256'][:16]}...)")
    print(f"  key: {key_path} (bytes NOT in JSON; sha256={out['key_sha256'][:16]}...)")
    print(f"  origin: {out['origin']}  not_agh_scientific_job={out['not_agh_scientific_job']}")
    for c in out["cases"]:
        label = c.get("route_fail", c.get("route", "?"))
        state = c.get("state_after_pass", c.get("state_after", "?"))
        print(f"  [{c['case']}] {c['kind']}: {label} -> state={state}")
    print(f"  reference_answer_leaked: {out['reference_answer_leaked']}")
    return 0


if __name__ == "__main__":
    sys.exit(_main())
