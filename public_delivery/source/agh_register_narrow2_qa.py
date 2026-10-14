"""Trusted control-side entrypoint that registers a NEW local AGH QA task
with an independent budget, separate from task-agh-local-qa.

Task: task-agh-narrow2-qa
- domain: integration
- family: piecewise_constant
- a=-1, b=10000, breakpoints=[0], values=[1,0]
- tolerance: absolute=1/100000000, relative=0

This file ONLY performs the trusted registration. It does NOT call the model-facing
business tools (method_lookup / execute_task / verify_run) - those must be called
through the AGH host plugin in the live conversation. It writes a fresh db/key pair
and updates agh-host-binding.json active_job + allowed_task_ids to point at the new
task so the host plugin can resolve the binding.

Usage:
  D:/anaconda/python.exe agh_register_narrow2_qa.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from task_controller import TaskRegistrar

DB_PATH = "private_runs/tasks_agh_narrow2_qa.db"
KEY_PATH = "private_runs/tasks_agh_narrow2_qa.receipt.key"
SESSION = "S2"
TASK_ID = "task-agh-narrow2-qa"


def main() -> int:
    Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    registrar = TaskRegistrar(session=SESSION, db_path=DB_PATH, key_path=KEY_PATH)
    registrar.controller.restore_from_persistence(stop_active=False)

    existing = registrar.persistence.load_task(TASK_ID, SESSION)
    if existing is not None:
        print(f"ERROR: task {TASK_ID} already exists in {DB_PATH}; refusing to re-register.")
        registrar.persistence.close()
        return 1

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
    task_record = (registrar.store.get_task(TASK_ID) if TASK_ID in registrar.store._tasks
                   else registrar.persistence.load_task(TASK_ID, SESSION))

    out = {
        "origin": "TRUSTED_CONTROL_SIDE_REGISTRATION",
        "not_model_call": True,
        "not_agh_host_call": True,
        "task": TASK_ID,
        "session": SESSION,
        "db": DB_PATH,
        "key_fingerprint": registrar.persistence.sealer.key_fingerprint,
        "task_hash": task_record["hash"],
        "state_after_register": task_record["state"],
        "registration": "fresh",
        "canonical": canonical,
        "reference_answer_leaked": False,
    }

    # ---- update agh-host-binding.json ----
    binding_path = Path("agh-host-binding.json")
    binding = json.loads(binding_path.read_text(encoding="utf-8"))
    binding["defaults"]["db"] = DB_PATH
    binding["defaults"]["key"] = KEY_PATH
    binding["active_job"] = {
        "state": "ACTIVE",
        "db": DB_PATH,
        "key": KEY_PATH,
        "session_mapping": {"scicheck-development-guided01": SESSION},
    }
    if TASK_ID not in binding.get("allowed_task_ids", []):
        binding["allowed_task_ids"].append(TASK_ID)
    binding_path.write_text(json.dumps(binding, indent=2, allow_nan=False), encoding="utf-8")

    registrar.persistence.close()

    out_path = Path("private_runs") / "agh_narrow2_qa_register.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, allow_nan=False), encoding="utf-8")
    print(f"wrote {out_path}")
    print(json.dumps(out, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
