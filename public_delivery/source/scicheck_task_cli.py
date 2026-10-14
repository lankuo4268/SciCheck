"""CLI entry for the controlled task mechanism (integration only).

Usage:
  D:/anaconda/python.exe scicheck_task_cli.py --demo [--session S1] [--db outputs/tasks.db]
      [--key outputs/tasks.receipt.key] [--output outputs/task_demo.json]

Demonstrates the controlled flow with the unified trusted registrar, per-task
budget ledger, sealed receipts, and SQLite persistence:
  - trusted_register_task: ONE entry does normalize + freeze + SQLite row +
    per-task BudgetLedger + start-time persistence
  - execute_task(request_id, "quad_plain")     -> first candidate; plain quad
    misses the [-1,0] contribution -> VERIFIED_INCORRECT; task stays ACTIVE
    (per-task solve budget of 3 not yet exhausted)
  - execute_task(request_id2, "quad_breakpoints") -> EXPLICIT repair call; PASSED
  - idempotent repeat of the same (request_id, action) returns the SAME run, no re-solve
  - verify_run over the sealed trusted receipt (no re-solve, no reference answer leaked)
  - budget / request / terminal state all persisted to SQLite
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

from task_controller import TaskRegistrar, TaskError
from task_budget import DEFAULT_BUDGET
from task_store import TaskAlreadyTerminal, TaskFrozen


def _demo_flow(session: str, db_path: str, key_path: str, out_path: str) -> dict:
    registrar = TaskRegistrar(session=session, db_path=db_path, key_path=key_path)
    ctl = registrar.controller

    # ---- task: piecewise-constant, narrow contribution ([-1,0] unit bump on wide interval)
    task_id = "task-piecewise-narrow"
    canonical = {
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
    # ONE trusted registration: normalize + freeze + SQLite row + per-task budget NOW
    registrar.register_task(task_id, canonical)

    out = {
        "task": task_id,
        "session": session,
        "db": db_path,
        "key_fingerprint": registrar.persistence.sealer.key_fingerprint,
        "canonical": canonical,
        "task_hash": registrar.store.get_task(task_id)["hash"],
        "events": [],
    }

    def log(tag, payload):
        out["events"].append({"tag": tag, "payload": payload})

    # step 1: plain quad -> VERIFIED_INCORRECT, stays ACTIVE (per-task solve budget 3)
    try:
        r1 = ctl.execute_task(task_id, "req-1", "quad_plain")
        log("execute#1(quad_plain)", r1)
    except (TaskAlreadyTerminal, TaskError) as e:
        log("execute#1(quad_plain)", {"error": str(e)})
    out["state_after_run1"] = registrar.store.get_task(task_id)["state"]

    # step 2: EXPLICIT repair call with a DIFFERENT action -> breakpoint quad
    try:
        r2 = ctl.execute_task(task_id, "req-2", "quad_breakpoints")
        log("execute#2(quad_breakpoints)", r2)
    except (TaskAlreadyTerminal, TaskError) as e:
        log("execute#2(quad_breakpoints)", {"error": str(e)})
    out["state_after_run2"] = registrar.store.get_task(task_id)["state"]

    # step 3: idempotent repeat of step 1's request -> same run, no re-solve
    try:
        r1_again = ctl.execute_task(task_id, "req-1", "quad_plain")
        log("execute#1-repeat(idempotent)", r1_again)
    except Exception as e:
        log("execute#1-repeat(idempotent)", {"error": str(e)})

    # step 4: verify over the saved receipt of run2 (read + verify sealed receipt,
    # no re-solve, no reference answer leaked)
    if r2.get("run_id"):
        out["verify_run2"] = ctl.verify_run(task_id, r2["run_id"])

    # step 5: budget card (per-resource balances; units NOT summed)
    out["budget_card"] = ctl.method_lookup(task_id, "budget")

    out["task_view"] = ctl.task_view(task_id)
    out["reference_answer_leaked"] = _detect_leak(out)
    registrar.persistence.close()
    return out


def _detect_leak(obj) -> bool:
    forbidden = {"reference_fraction", "reference_answer", "reference_value",
                 "exact_check", "candidate_as_fraction", "abs_diff_exact_fraction",
                 "abs_diff_fraction", "allowed_error_fraction"}
    if isinstance(obj, dict):
        if forbidden & set(obj.keys()):
            return True
        return any(_detect_leak(v) for v in obj.values())
    if isinstance(obj, list):
        return any(_detect_leak(v) for v in obj)
    return False


def _zero_candidate_flow(session: str, db_path: str, key_path: str, out_path: str) -> dict:
    """A task whose first candidate is EXACTLY 0.0 (an integral of x^3 over a
    symmetric interval, or a piecewise whose contributions cancel). The sealed
    receipt must carry candidate=0.0 and survive a reopen: only None means "no
    value"; 0.0 is a legitimate candidate and must NOT be coerced to NaN on
    reload, and verify_run must accept it (field-presence checks, not truthiness).
    """
    registrar = TaskRegistrar(session=session, db_path=db_path, key_path=key_path)
    ctl = registrar.controller
    task_id = "task-zero-candidate"
    # integral of x over [-1,1] = exactly 0.0 (odd function, symmetric bounds)
    canonical = {
        "domain": "integration",
        "problem": {
            "family": "polynomial",
            "a": "-1",
            "b": "1",
            "coefficients": ["0", "1"],  # x
        },
        "tolerance": {"absolute": "1/1000000", "relative": "0"},
    }
    registrar.register_task(task_id, canonical)
    out = {"task": task_id, "events": []}

    r1 = ctl.execute_task(task_id, "req-0", "quad_plain")
    out["candidate"] = r1["candidate"]
    out["exact_ok"] = r1["exact_ok"]
    out["verdict"] = r1.get("verdict")
    out["state_after_run1"] = registrar.store.get_task(task_id)["state"]
    out["events"].append({"tag": "execute#1(quad_plain)", "payload": r1})

    # Simulate a restart: reopen the SAME db + key, reload, verify candidate is still 0.0
    registrar.persistence.close()
    registrar2 = TaskRegistrar(session=session, db_path=db_path, key_path=key_path)
    registrar2.restore()
    run_id = r1["run_id"]
    # candidate must be the exact 0.0, not NaN, after the reload
    reloaded_state = registrar2.controller.store.get_task(task_id)["state"]
    out["reopened_state"] = reloaded_state
    rec = registrar2.controller.store.get_run(run_id)
    out["reloaded_candidate"] = rec.candidate
    out["reloaded_candidate_is_zero"] = (rec.candidate == 0.0)
    # verify the sealed receipt of the zero-candidate run (read + verify, no re-solve)
    out["verify_zero"] = registrar2.controller.verify_run(task_id, run_id)
    registrar2.persistence.close()
    out["reference_answer_leaked"] = _detect_leak(out)
    return out


def _root_flow(session: str, db_path: str, key_path: str, out_path: str) -> dict:
    """Root-domain demo: (x-1)^3 on [0,2], start=1.001, atol=1e-8.
    Coarse Newton (diagnostic, scipy root_scalar newton) -> VERIFIED_INCORRECT;
    strict Brent (scipy root_scalar brentq) -> VERIFIED_PASS. Two real records.
    Also exercises the public direct route. All through the controlled-task
    mechanism (unified register + per-task budget + sealed receipt + persistence).
    """
    registrar = TaskRegistrar(session=session, db_path=db_path, key_path=key_path)
    ctl = registrar.controller
    task_id = "task-root-x-1-cubed"
    canonical = {
        "domain": "root",
        "problem": {
            "family": "scaled_power",
            "power": "3", "scale": "1", "shift": "1",
            "a": "0", "b": "2", "start": "1.001",
        },
        "tolerance": {"absolute": "1/100000000", "relative": "0"},
    }
    registrar.register_task(task_id, canonical)
    out = {
        "task": task_id,
        "session": session,
        "db": db_path,
        "key_fingerprint": registrar.persistence.sealer.key_fingerprint,
        "canonical": canonical,
        "task_hash": registrar.store.get_task(task_id)["hash"],
        "events": [],
    }

    def log(tag, payload):
        out["events"].append({"tag": tag, "payload": payload})

    # step 1: coarse Newton (development diagnostic) -> forward error ~6.7e-4 > 1e-8
    try:
        r1 = ctl.execute_task(task_id, "req-1", "root_newton")
        log("execute#1(root_newton coarse)", r1)
    except TaskError as e:
        log("execute#1(root_newton coarse)", {"error": str(e)})
    out["state_after_newton"] = registrar.store.get_task(task_id)["state"]

    # step 2: strict Brent -> VERIFIED_PASS
    try:
        r2 = ctl.execute_task(task_id, "req-2", "root_brent")
        log("execute#2(root_brent strict)", r2)
    except TaskError as e:
        log("execute#2(root_brent strict)", {"error": str(e)})
    out["state_after_brent"] = registrar.store.get_task(task_id)["state"]

    # step 3: verify the sealed receipt of the passing run (no re-solve)
    if r2.get("run_id"):
        out["verify_brent"] = ctl.verify_run(task_id, r2["run_id"])

    # step 4: public direct route on a FRESH task (terminal task refuses new solves)
    direct_id = "task-root-direct-x-1-cubed"
    registrar.register_task(direct_id, dict(canonical))
    try:
        r3 = ctl.execute_task(direct_id, "req-d", "root_direct")
        log("direct(root_direct public)", r3)
    except TaskError as e:
        log("direct(root_direct public)", {"error": str(e)})
    out["state_direct"] = registrar.store.get_task(direct_id)["state"]

    out["budget_card"] = ctl.method_lookup(task_id, "budget")
    out["task_view"] = ctl.task_view(task_id)
    out["reference_answer_leaked"] = _detect_leak(out)
    registrar.persistence.close()
    return out


def _root_tight_flow(session: str, db_path: str, key_path: str, out_path: str) -> dict:
    """Root-domain tight-tolerance regression: shift=1/3, atol=1e-30. The public
    direct candidate (binary64 1/3) has an exact forward error of
    1/54043195528445952, far above 1e-30, so it MUST NOT pass by construction.
    The strict Brent route also cannot beat the binary64 representation of the
    shift and MUST also fail. This guards against a fake PASS on the direct
    route for non-terminating binary shifts under a very tight tolerance."""
    registrar = TaskRegistrar(session=session, db_path=db_path, key_path=key_path)
    ctl = registrar.controller
    task_id = "task-root-shift-1-3-tight"
    canonical = {
        "domain": "root",
        "problem": {
            "family": "scaled_power",
            "power": "3", "scale": "1", "shift": "1/3",
            "a": "0", "b": "1",
        },
        "tolerance": {"absolute": "1/1000000000000000000000000000000", "relative": "0"},
    }
    registrar.register_task(task_id, canonical)
    out = {
        "task": task_id,
        "session": session,
        "db": db_path,
        "key_fingerprint": registrar.persistence.sealer.key_fingerprint,
        "canonical": canonical,
        "task_hash": registrar.store.get_task(task_id)["hash"],
        "events": [],
    }

    def log(tag, payload):
        out["events"].append({"tag": tag, "payload": payload})

    # step 1: public direct route -> MUST FAIL (no fake pass by construction)
    try:
        r1 = ctl.execute_task(task_id, "req-1", "root_direct")
        log("direct(root_direct tight)", r1)
    except TaskError as e:
        log("direct(root_direct tight)", {"error": str(e)})
    out["state_after_direct"] = registrar.store.get_task(task_id)["state"]

    # step 2: strict Brent on a FRESH task (the direct run above keeps the task
    # ACTIVE/INCOMPLETE if budget allows, but the direct run is a distinct task id
    # only if we register another one; here we use a new task id to keep records clean)
    brent_id = "task-root-shift-1-3-tight-brent"
    registrar.register_task(brent_id, dict(canonical))
    try:
        r2 = ctl.execute_task(brent_id, "req-2", "root_brent")
        log("brent(root_brent tight)", r2)
    except TaskError as e:
        log("brent(root_brent tight)", {"error": str(e)})
    out["state_after_brent"] = registrar.store.get_task(brent_id)["state"]

    out["reference_answer_leaked"] = _detect_leak(out)
    registrar.persistence.close()
    return out


def _lin_sys_flow(session: str, db_path: str, key_path: str, out_path: str) -> dict:
    """Linear-system demo: the required matrix
    [[1,1],[1,1000001/1000000]] with rhs=[1,1500001/1500000], atol=1e-7, rtol=0.
    - float32 (NumPy) candidate -> initial error (VERIFIED_INCORRECT, forward error 1/12 >> 1e-7)
    - float64 (NumPy) repair candidate -> VERIFIED_PASS
    Both actual records are kept. Also exercises the public high-precision Decimal
    route as a SEPARATE task (same math + original tolerance, a different output
    format/cost, does NOT read the private Fraction reference, does not change the
    original task's terminal state).
    """
    registrar = TaskRegistrar(session=session, db_path=db_path, key_path=key_path)
    ctl = registrar.controller
    base = {
        "domain": "linear_system",
        "problem": {
            "family": "rational_square_matrix",
            "n": "2",
            "matrix": [["1", "1"], ["1", "1000001/1000000"]],
            "rhs": ["1", "1500001/1500000"],
        },
        "tolerance": {"absolute": "1/10000000", "relative": "0"},
    }
    task_id = "task-ls-required"
    registrar.register_task(task_id, base)
    out = {
        "task": task_id,
        "session": session,
        "db": db_path,
        "key_fingerprint": registrar.persistence.sealer.key_fingerprint,
        "canonical": base,
        "task_hash": registrar.store.get_task(task_id)["hash"],
        "events": [],
    }

    def log(tag, payload):
        out["events"].append({"tag": tag, "payload": payload})

    # step 1: float32 candidate -> initial error (the coarse development/diagnostic route)
    try:
        r1 = ctl.execute_task(task_id, "req-1", "ls_np32")
        log("execute#1(ls_np32 initial)", r1)
    except TaskError as e:
        log("execute#1(ls_np32 initial)", {"error": str(e)})
    out["state_after_np32"] = registrar.store.get_task(task_id)["state"]

    # step 2: float64 repair candidate -> VERIFIED_PASS
    try:
        r2 = ctl.execute_task(task_id, "req-2", "ls_np64")
        log("execute#2(ls_np64 repair)", r2)
    except TaskError as e:
        log("execute#2(ls_np64 repair)", {"error": str(e)})
    out["state_after_np64"] = registrar.store.get_task(task_id)["state"]

    # step 3: verify the sealed receipt of the passing run (no re-solve)
    if r2.get("run_id"):
        out["verify_np64"] = ctl.verify_run(task_id, r2["run_id"])

    # step 4: public high-precision Decimal route as a SEPARATE task/route (same math +
    # original tolerance, different output format/cost; does NOT read the reference, does
    # NOT change the original task's terminal state).
    high_id = "task-ls-required-decimal-high"
    registrar.register_task(high_id, dict(base))
    try:
        r3 = ctl.execute_task(high_id, "req-1", "ls_decimal_high")
        log("decimal-high(ls_decimal_high public)", r3)
    except TaskError as e:
        log("decimal-high(ls_decimal_high public)", {"error": str(e)})
    out["state_decimal_high"] = registrar.store.get_task(high_id)["state"]
    out["original_task_state_unchanged"] = registrar.store.get_task(task_id)["state"]

    out["budget_card"] = ctl.method_lookup(task_id, "budget")
    out["task_view"] = ctl.task_view(task_id)
    out["reference_answer_leaked"] = _detect_leak(out)
    registrar.persistence.close()
    return out


def _lin_sys_no_unique_flow(session: str, db_path: str, key_path: str, out_path: str) -> dict:
    """A no-unique-solution system (rank-deficient) must be refused UNSUPPORTED --
    this is a property of the MATH (no unique x), decided exactly over Fractions.
    Uses [[1,1],[1,1]] rhs=[1,1] (both rows identical -> no unique solution)."""
    registrar = TaskRegistrar(session=session, db_path=db_path, key_path=key_path)
    ctl = registrar.controller
    task_id = "task-ls-no-unique"
    canonical = {
        "domain": "linear_system",
        "problem": {
            "family": "rational_square_matrix",
            "n": "2",
            "matrix": [["1", "1"], ["1", "1"]],
            "rhs": ["1", "1"],
        },
        "tolerance": {"absolute": "1/10000000", "relative": "0"},
    }
    registrar.register_task(task_id, canonical)
    out = {"task": task_id, "events": []}
    try:
        r = ctl.execute_task(task_id, "req-1", "ls_np64")
        out["events"].append({"tag": "execute(ls_np64)", "payload": r})
    except TaskError as e:
        out["events"].append({"tag": "execute(ls_np64)", "payload": {"error": str(e)}})
    out["state"] = registrar.store.get_task(task_id)["state"]
    out["reference_answer_leaked"] = _detect_leak(out)
    registrar.persistence.close()
    return out


def _lin_sys_ultra_tight_flow(session: str, db_path: str, key_path: str, out_path: str) -> dict:
    """An ultra-tight tolerance that NO binary64 value of the exact solution can
    meet -> UNSUPPORTED. The rejection is about the binary64 candidate FORMAT
    (not that the system has no solution). The high-precision Decimal route is the
    usable alternative for the same math input + original tolerance.
    """
    registrar = TaskRegistrar(session=session, db_path=db_path, key_path=key_path)
    ctl = registrar.controller
    base = {
        "domain": "linear_system",
        "problem": {
            "family": "rational_square_matrix",
            "n": "2",
            "matrix": [["1", "1"], ["1", "1000001/1000000"]],
            "rhs": ["1", "1500001/1500000"],
        },
        "tolerance": {"absolute": "1/1000000000000000000000000000000", "relative": "0"},
    }
    task_id = "task-ls-ultra-tight"
    registrar.register_task(task_id, base)
    out = {"task": task_id, "events": []}
    # float64 candidate: the original tolerance (1e-30) cannot be met by any binary64
    # value of the exact reference (1/3, 2/3) -> UNSUPPORTED (binary64 boundary).
    try:
        r1 = ctl.execute_task(task_id, "req-1", "ls_np64")
        out["events"].append({"tag": "execute(ls_np64 ultra-tight)", "payload": r1})
    except TaskError as e:
        out["events"].append({"tag": "execute(ls_np64 ultra-tight)", "payload": {"error": str(e)}})
    out["state_np64"] = registrar.store.get_task(task_id)["state"]

    # The high-precision public Decimal route is a SEPARATE task (same math + original
    # tolerance, different output format/cost; does NOT read the reference, does NOT
    # change the original task's terminal state).
    high_id = "task-ls-ultra-tight-decimal-high"
    registrar.register_task(high_id, dict(base))
    try:
        r2 = ctl.execute_task(high_id, "req-1", "ls_decimal_high")
        out["events"].append({"tag": "execute(ls_decimal_high ultra-tight)", "payload": r2})
    except TaskError as e:
        out["events"].append({"tag": "execute(ls_decimal_high ultra-tight)", "payload": {"error": str(e)}})
    out["state_decimal_high"] = registrar.store.get_task(high_id)["state"]
    out["original_task_state_unchanged"] = registrar.store.get_task(task_id)["state"]
    out["reference_answer_leaked"] = _detect_leak(out)
    registrar.persistence.close()
    return out


def _normal_linsys_flow(session: str, db_path: str, key_path: str, out_path: str) -> dict:
    """A normal, well-conditioned system where the float64 candidate passes at the
    original tolerance (the happy-path linear-system record)."""
    registrar = TaskRegistrar(session=session, db_path=db_path, key_path=key_path)
    ctl = registrar.controller
    task_id = "task-ls-normal"
    canonical = {
        "domain": "linear_system",
        "problem": {
            "family": "rational_square_matrix",
            "n": "2",
            "matrix": [["2", "0"], ["0", "4"]],
            "rhs": ["1", "1"],
        },
        "tolerance": {"absolute": "1/10000000", "relative": "0"},
    }
    registrar.register_task(task_id, canonical)
    out = {"task": task_id, "events": []}
    r = ctl.execute_task(task_id, "req-1", "ls_np64")
    out["events"].append({"tag": "execute(ls_np64)", "payload": r})
    out["state"] = registrar.store.get_task(task_id)["state"]
    if r.get("run_id"):
        out["verify"] = ctl.verify_run(task_id, r["run_id"])
    out["reference_answer_leaked"] = _detect_leak(out)
    registrar.persistence.close()
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="SciCheck controlled-task demo (integration)")
    parser.add_argument("--demo", action="store_true", help="run the controlled-task demo flow")
    parser.add_argument("--zero-candidate", action="store_true",
                        help="run the zero-candidate (candidate=0.0) receipt/reload flow")
    parser.add_argument("--root", action="store_true",
                        help="run the root-domain (x-1)^3 demo: coarse Newton + strict Brent")
    parser.add_argument("--root-tight", action="store_true",
                        help="run the root-domain shift=1/3 tight-tolerance regression")
    parser.add_argument("--lin-sys", action="store_true",
                        help="run the required linear-system (float32 error + float64 repair + Decimal high)")
    parser.add_argument("--lin-sys-normal", action="store_true",
                        help="run the normal well-conditioned linear-system (float64 pass)")
    parser.add_argument("--lin-sys-no-unique", action="store_true",
                        help="run the no-unique-solution linear-system (UNSUPPORTED)")
    parser.add_argument("--lin-sys-ultra-tight", action="store_true",
                        help="run the ultra-tight binary64-boundary linear-system (UNSUPPORTED) + Decimal high")
    parser.add_argument("--session", default="S1", help="session id")
    parser.add_argument("--db", default="outputs/tasks.db", help="SQLite db path")
    parser.add_argument("--key", default="outputs/tasks.receipt.key",
                        help="trusted-side receipt key file path")
    parser.add_argument("--output", default="outputs/task_demo.json", help="JSON output path")
    args = parser.parse_args(argv)

    # Pick a private (non-public) db + key + JSON tag so a re-run NEVER overwrites the
    # shared outputs/task_demo.json that an earlier run may have published.
    import time
    tag = f"{int(time.time())}"
    if args.zero_candidate:
        out_path = f"private_runs/zero_candidate_{args.session}_{tag}.json"
        db_path = f"private_runs/tasks_zero_{args.session}_{tag}.db"
        key_path = f"private_runs/tasks_zero_{args.session}_{tag}.receipt.key"
    elif args.root:
        out_path = f"private_runs/root_demo_{args.session}_{tag}.json"
        db_path = f"private_runs/tasks_root_{args.session}_{tag}.db"
        key_path = f"private_runs/tasks_root_{args.session}_{tag}.receipt.key"
    elif args.root_tight:
        out_path = f"private_runs/root_tight_{args.session}_{tag}.json"
        db_path = f"private_runs/tasks_root_tight_{args.session}_{tag}.db"
        key_path = f"private_runs/tasks_root_tight_{args.session}_{tag}.receipt.key"
    elif args.lin_sys:
        out_path = f"private_runs/lin_sys_{args.session}_{tag}.json"
        db_path = f"private_runs/tasks_lin_sys_{args.session}_{tag}.db"
        key_path = f"private_runs/tasks_lin_sys_{args.session}_{tag}.receipt.key"
    elif args.lin_sys_normal:
        out_path = f"private_runs/lin_sys_normal_{args.session}_{tag}.json"
        db_path = f"private_runs/tasks_lin_sys_normal_{args.session}_{tag}.db"
        key_path = f"private_runs/tasks_lin_sys_normal_{args.session}_{tag}.receipt.key"
    elif args.lin_sys_no_unique:
        out_path = f"private_runs/lin_sys_no_unique_{args.session}_{tag}.json"
        db_path = f"private_runs/tasks_lin_sys_no_unique_{args.session}_{tag}.db"
        key_path = f"private_runs/tasks_lin_sys_no_unique_{args.session}_{tag}.receipt.key"
    elif args.lin_sys_ultra_tight:
        out_path = f"private_runs/lin_sys_ultra_tight_{args.session}_{tag}.json"
        db_path = f"private_runs/tasks_lin_sys_ultra_tight_{args.session}_{tag}.db"
        key_path = f"private_runs/tasks_lin_sys_ultra_tight_{args.session}_{tag}.receipt.key"
    elif args.demo:
        out_path = f"private_runs/task_demo_{args.session}_{tag}.json"
        db_path = f"private_runs/tasks_demo_{args.session}_{tag}.db"
        key_path = f"private_runs/tasks_demo_{args.session}_{tag}.receipt.key"
    else:
        parser.print_help()
        return 1

    # An explicit --output overrides ONLY the JSON destination; db/key stay on the
    # per-scenario private paths unless explicitly supplied via --db/--key as well.
    if args.output and args.output != "outputs/task_demo.json":
        out_path = args.output
    if args.db and args.db != "outputs/tasks.db":
        db_path = args.db
    if args.key and args.key != "outputs/tasks.receipt.key":
        key_path = args.key

    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    key_parent = os.path.dirname(key_path) or ""
    if key_parent:
        Path(key_parent).mkdir(parents=True, exist_ok=True)

    if args.zero_candidate:
        result = _zero_candidate_flow(args.session, db_path, key_path, out_path)
    elif args.root:
        result = _root_flow(args.session, db_path, key_path, out_path)
    elif args.root_tight:
        result = _root_tight_flow(args.session, db_path, key_path, out_path)
    elif args.lin_sys:
        result = _lin_sys_flow(args.session, db_path, key_path, out_path)
    elif args.lin_sys_normal:
        result = _normal_linsys_flow(args.session, db_path, key_path, out_path)
    elif args.lin_sys_no_unique:
        result = _lin_sys_no_unique_flow(args.session, db_path, key_path, out_path)
    elif args.lin_sys_ultra_tight:
        result = _lin_sys_ultra_tight_flow(args.session, db_path, key_path, out_path)
    else:
        result = _demo_flow(args.session, db_path, key_path, out_path)
    Path(out_path).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"wrote {out_path}")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
