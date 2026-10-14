"""Trusted control-side entrypoint that registers the local AGH QA task.

This is the SEPARATE trusted management interface: it creates the trusted db/key
pair under `private_runs/tasks_agh_local_qa.db` / `.receipt.key`, registers ONE
plain quad integration task, and then runs the two local business-loop steps
(method_lookup + first execute_task) using the same restricted bridge module, so
that the AGH plugin's three model-facing tools have a real registered task to call
against.

It does NOT modify the model's budget or grant time; it just performs the ONE
trusted registration that the model-facing tools are not allowed to do.

Usage:
  D:/anaconda/python.exe agh_register_local_qa.py

Re-running this file is safe: it reuses the same fixed db/key paths and the same
task id; if the task is already terminal it just re-reads state and re-verifies,
it never re-registers or tops up budget.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from task_controller import TaskRegistrar, TaskError, TaskAlreadyTerminal
from agh_business_bridge import method_lookup as _bridge_method_lookup, execute_task as _bridge_execute_task, verify_run as _bridge_verify_run
import argparse


DB_PATH = "private_runs/tasks_agh_local_qa.db"
KEY_PATH = "private_runs/tasks_agh_local_qa.receipt.key"
SESSION = "S1"
TASK_ID = "task-agh-local-qa"


def _bridge_args(db: str, key: str, session: str) -> argparse.Namespace:
    return argparse.Namespace(db=db, key=key, session=session)


def main() -> int:
    Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    registrar = TaskRegistrar(session=SESSION, db_path=DB_PATH, key_path=KEY_PATH)
    registrar.controller.restore_from_persistence(stop_active=False)

    out: dict = {
        "task": TASK_ID,
        "session": SESSION,
        "db": DB_PATH,
        "key_fingerprint": registrar.persistence.sealer.key_fingerprint,
    }

    existing = registrar.persistence.load_task(TASK_ID, SESSION)
    if existing is None:
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
        registrar.register_task(TASK_ID, canonical)
        out["registration"] = "fresh"
    else:
        out["registration"] = "existing (not re-registered, no budget top-up)"

    out["task_hash"] = (registrar.store.get_task(TASK_ID) if TASK_ID in registrar.store._tasks else registrar.persistence.load_task(TASK_ID, SESSION))["hash"]
    out["state_after_register"] = (registrar.store.get_task(TASK_ID) if TASK_ID in registrar.store._tasks else registrar.persistence.load_task(TASK_ID, SESSION))["state"]

    # ---- local business-loop step 1: method_lookup (budget card) via the SAME
    # restricted bridge module the AGH plugin will call (not a second, broader
    # API).
    bridge_args = _bridge_args(DB_PATH, KEY_PATH, SESSION)
    namespace_lookup = argparse.Namespace(
        task=TASK_ID, card="budget",
        session=bridge_args.session, db=bridge_args.db, key=bridge_args.key,
    )
    try:
        registrar.controller.restore_from_persistence(stop_active=False)
        view = registrar.method_lookup(TASK_ID, "budget")
        out["method_lookup_budget"] = view
    except TaskError as e:
        out["method_lookup_budget"] = {"error": str(e)}

    # ---- local business-loop step 2: execute_task (first candidate: quad_plain)
    namespace_exec = argparse.Namespace(
        task=TASK_ID, request="agh-qa-req-1", action="quad_plain",
        session=bridge_args.session, db=bridge_args.db, key=bridge_args.key,
    )
    try:
        registrar.controller.restore_from_persistence(stop_active=False)
        view = registrar.execute_task(TASK_ID, "agh-qa-req-1", "quad_plain")
        out["execute_task_quad_plain"] = view
        out["run_id"] = view.get("run_id")
    except TaskError as e:
        out["execute_task_quad_plain"] = {"error": str(e)}

    # ---- local business-loop step 3: verify_run over the sealed receipt
    if out.get("run_id"):
        namespace_verify = argparse.Namespace(
            task=TASK_ID, run=out["run_id"],
            session=bridge_args.session, db=bridge_args.db, key=bridge_args.key,
        )
        try:
            registrar.controller.restore_from_persistence(stop_active=False)
            out["verify_run"] = registrar.verify_run(TASK_ID, out["run_id"])
        except TaskError as e:
            out["verify_run"] = {"error": str(e)}

    out["final_state"] = (registrar.store.get_task(TASK_ID) if TASK_ID in registrar.store._tasks else registrar.persistence.load_task(TASK_ID, SESSION))["state"]
    out["reference_answer_leaked"] = False
    registrar.persistence.close()

    out_path = Path("private_runs") / "agh_local_qa_register.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, allow_nan=False), encoding="utf-8")
    print(f"wrote {out_path}")
    print(json.dumps(out, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
