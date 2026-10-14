"""Restricted business bridge for AGH plugin tool calls.

This module is the ONLY local entry point the AGH plugin should call. It exposes:
- method_lookup: task_id + card_id ('structure' | 'budget')
- execute_task: task_id + request_id + action_id
- verify_run: task_id + run_id

It deliberately does NOT expose task registration, budget modification, key access,
database paths, receipt key bytes, or model budget mutation. The trusted session DB/key
paths are supplied by the host invocation context, not by the model.

Usage (from the workspace root):
  D:/anaconda/python.exe agh_business_bridge.py method_lookup --session S1 --db <db> --key <key> --task task_id --card structure
  D:/anaconda/python.exe agh_business_bridge.py execute_task --session S1 --db <db> --key <key> --task task_id --request req_id --action card_id
  D:/anaconda/python.exe agh_business_bridge.py verify_run --session S1 --db <db> --key <key> --task task_id --run run_id

All commands print a single JSON document to stdout with allow_nan=False so that
any accidental NaN surfaces as an explicit error instead of being silently dropped.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from task_controller import TaskRegistrar, TaskError


# The trusted host context supplies these. The model-facing plugin schema does NOT
# expose db/key paths; they are read from the host binding and passed via CLI.
_BASE_PARENT = argparse.ArgumentParser(add_help=False)
_BASE_PARENT.add_argument("--session", default="S1", help="trusted session id")
_BASE_PARENT.add_argument("--db", default="private_runs/tasks_agh_local_qa.db", help="trusted task SQLite db path")
_BASE_PARENT.add_argument("--key", default="private_runs/tasks_agh_local_qa.receipt.key", help="trusted receipt key path")


def _registrar(args) -> TaskRegistrar:
    registrar = TaskRegistrar(session=args.session, db_path=args.db, key_path=args.key)
    # Cross-process: each tool invocation is a fresh Python process, so we MUST
    # reload the already-registered tasks / budgets / runs / request bindings from
    # SQLite. stop_active=False keeps still-ACTIVE tasks ACTIVE (no conservative
    # stop); a genuine process-restart recovery is a separate trusted management
    # entrypoint, not this business bridge.
    registrar.controller.restore_from_persistence(stop_active=False)
    return registrar


def _dump(payload: dict, ok: bool = True) -> int:
    doc = dict(payload)
    doc["bridge"] = "agh_business_bridge"
    doc["interface"] = "business"
    doc["ok"] = ok
    # candidate nulls and tri-state exact_ok (True/False/None) are preserved by
    # default=str (None stays None in JSON); allow_nan=False forbids NaN/Inf.
    print(json.dumps(doc, indent=2, allow_nan=False, default=str))
    return 0


def _fail(doc: dict, code: int = 2) -> int:
    doc["ok"] = False
    print(json.dumps(doc, indent=2, allow_nan=False, default=str))
    return code


def method_lookup(args) -> int:
    reg = _registrar(args)
    try:
        card_id = args.card
        if card_id not in ("structure", "budget"):
            return _fail({"error": f"card must be 'structure' or 'budget', got {card_id!r}"})
        view = reg.method_lookup(args.task, card_id)
        return _dump({"task_id": args.task, "card_id": card_id, "view": view})
    except TaskError as exc:
        return _fail({"task_id": args.task, "card_id": args.card, "error": str(exc)})
    finally:
        reg.persistence.close()


def execute_task(args) -> int:
    reg = _registrar(args)
    try:
        view = reg.execute_task(args.task, args.request, args.action)
        return _dump({"task_id": args.task, "request_id": args.request,
                      "action_id": args.action, "view": view})
    except TaskError as exc:
        return _fail({"task_id": args.task, "request_id": args.request,
                      "action_id": args.action, "error": str(exc)})
    finally:
        reg.persistence.close()


def verify_run(args) -> int:
    reg = _registrar(args)
    try:
        view = reg.verify_run(args.task, args.run)
        return _dump({"task_id": args.task, "run_id": args.run, "view": view})
    except TaskError as exc:
        return _fail({"task_id": args.task, "run_id": args.run, "error": str(exc)})
    finally:
        reg.persistence.close()


def _build_parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="agh_business_bridge.py", description="AGH restricted business bridge")
    sub = root.add_subparsers(dest="command", required=True)

    lookup = sub.add_parser("method_lookup", parents=[_BASE_PARENT], help="look up structure or budget card")
    lookup.add_argument("--task", required=True, help="task id")
    lookup.add_argument("--card", choices=["structure", "budget"], default="structure", help="card id")
    lookup.set_defaults(func=method_lookup)

    execute = sub.add_parser("execute_task", parents=[_BASE_PARENT], help="run exactly one trusted action")
    execute.add_argument("--task", required=True, help="task id")
    execute.add_argument("--request", required=True, help="request id (idempotency key)")
    execute.add_argument("--action", required=True, help="action/card id to launch")
    execute.set_defaults(func=execute_task)

    verify = sub.add_parser("verify_run", parents=[_BASE_PARENT], help="verify a sealed receipt without re-solving")
    verify.add_argument("--task", required=True, help="task id")
    verify.add_argument("--run", required=True, help="run id")
    verify.set_defaults(func=verify_run)

    return root


def main(argv=None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
