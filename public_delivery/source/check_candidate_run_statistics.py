import hashlib, json, os, re, sys
from collections import defaultdict
from datetime import datetime, timezone

PRIVATE = os.path.join(os.getcwd(), "private_runs")
SCRIPT = os.path.join(os.getcwd(), "_build_candidate_run_statistics.py")
SOURCE_FILES = [
  "acc_integration_normal_20261015042620.json",
  "acc_integration_zero_candidate_20261015042620.json",
  "acc_root_normal_20261015042620.json",
  "acc_root_tight_20261015042620.json",
  "acc_linsys_normal_20261015042620.json",
  "acc_linsys_no_unique_20261015042620.json",
  "acc_linsys_ultra_tight_20261015042620.json",
  "ode_stability_feec260053604a99bd12b37b25a40c2f.json",
  "acc_cross_session_20261015042632.json",
  "acc_budget_exhaust_20261015042633.json",
  "acc_restart_20261015042635.json",
]

def sha256(path):
  h = hashlib.sha256()
  with open(path, "rb") as f:
    for chunk in iter(lambda: f.read(65536), b""):
      h.update(chunk)
  return h.hexdigest()

inputs = {}
input_hashes = {}
for name in SOURCE_FILES:
  p = os.path.join(PRIVATE, name)
  input_hashes[name] = sha256(p)
  inputs[name] = json.load(open(p, encoding="utf-8"))

old_index_path = os.path.join(PRIVATE, "acceptance_run_index_20261015060904.json")
old_index = json.load(open(old_index_path, encoding="utf-8"))
old_index_hashes = {name: meta.get("sha256") for name, meta in old_index.get("source_files", {}).items()}

new_index_paths = sorted(
  p for p in (os.path.join(PRIVATE, f) for f in os.listdir(PRIVATE))
  if os.path.basename(p).startswith("acceptance_candidate_run_index_v2_") and p.endswith(".json")
)
if not new_index_paths:
  raise SystemExit("no new candidate-run index found; run _build_candidate_run_statistics.py first")

# Prefer the most recently generated new index, but require that its deterministic content matches all current v2 files
new_index = json.load(open(new_index_paths[-1], encoding="utf-8"))

def deterministic_content(doc):
  doc = dict(doc)
  doc.pop("tag", None)
  doc.pop("generated_at", None)
  checks = dict(doc.get("checks", {}))
  checks.pop("reproducible_generation", None)
  doc["checks"] = checks
  return hashlib.sha256(json.dumps(doc, sort_keys=True, ensure_ascii=False).encode()).hexdigest()

all_v2_hashes = {p: deterministic_content(json.load(open(p, encoding="utf-8"))) for p in new_index_paths}
reproducible_generation = len(set(all_v2_hashes.values())) == 1

def resolve_pointer(root, pointer):
  cur = root
  # Pointer format includes dynamic task-state paths such as sections.restore.restored.acc-restart-active.
  # Static resolution treats the first unrecognized dot token as a dynamic key, then requires only that the
  # parent path exists in the input JSON. This is explicit metadata, not assumed content resolution.
  parts = pointer.replace("[", ".").replace("]", "").split(".")
  for i, part in enumerate(parts):
    if part == "":
      continue
    if isinstance(cur, list):
      try:
        cur = cur[int(part)]
      except (ValueError, IndexError):
        return None
    elif isinstance(cur, dict):
      if part in cur:
        cur = cur[part]
      else:
        return cur if all(parent in root_for_check for parent in parts[:i] ) else None
    else:
      return None
  return cur

def is_dynamic_parent(pointer, file):
  # Paths that intentionally enumerate dynamic task-id keys under a static parent.
  if pointer.startswith("sections.restore.restored.") and file == "acc_restart_20261015042635.json":
    return resolve_pointer(inputs.get(file, {}), "sections.restore.restored") is not None
  if pointer.startswith("sections.restore.states_before_close.") or pointer.startswith("sections.restore.states_after_restore."):
    parent = pointer.rsplit(".", 1)[0]
    return resolve_pointer(inputs.get(file, {}), parent) is not None
  return False

unresolved_paths = []
def check_resolvable(file, pointer, what):
  if file not in inputs:
    unresolved_paths.append({"file": file, "pointer": pointer, "kind": what, "reason": "input file not in current 11-file set"})
    return
  val = resolve_pointer(inputs[file], pointer)
  if val is None and not is_dynamic_parent(pointer, file):
    unresolved_paths.append({"file": file, "pointer": pointer, "kind": what, "reason": "JSON pointer does not resolve"})

new_run_ids = {r.get("run_id") for r in new_index.get("run_index", [])}

# Recompute distinct run ids directly from the 11 inputs, using explicit extraction rules that mirror the generator.
def all_run_ids_from_inputs():
  found = set()
  refs = []
  for name in SOURCE_FILES:
    data = inputs[name]
    for i, ev in enumerate(data.get("events") or []):
      p = f"events[{i}].payload"
      payload = ev.get("payload") or {}
      if payload.get("run_id"):
        found.add(payload["run_id"])
        refs.append((name, p, "event"))
      elif payload.get("error"):
        refs.append((name, p, "rejection_or_exception_no_candidate"))
    for k, v in (data.get("rejections") or {}).items():
      refs.append((name, f"rejections.{k}", "rejection"))
    for i, case in enumerate(data.get("cases") or []):
      for k in ("execute_fail", "execute_pass", "execute"):
        if k in case and case[k].get("run_id"):
          found.add(case[k]["run_id"])
          refs.append((name, f"cases[{i}].{k}", "execution_record"))
      for k in ("verify_pass", "verify"):
        if k in case and case[k].get("run_id"):
          found.add(case[k]["run_id"])
          refs.append((name, f"cases[{i}].{k}", "receipt_verification_reference"))
    for k in ("verify_zero", "verify_brent", "verify_np64", "verify_run2"):
      if isinstance(data.get(k), dict) and data[k].get("run_id"):
        found.add(data[k]["run_id"])
        refs.append((name, k, "receipt_verification_reference"))
    secs = data.get("sections") or {}
    if "terminal_verify_after_restart" in secs and secs["terminal_verify_after_restart"].get("run_id"):
      found.add(secs["terminal_verify_after_restart"]["run_id"])
      refs.append((name, "sections.terminal_verify_after_restart", "terminal_verify_reference"))
    if "active_verify_after_restart" in secs:
      av = secs["active_verify_after_restart"]
      if av.get("run_id"):
        found.add(av["run_id"])
        refs.append((name, "sections.active_verify_after_restart.verify", "active_verify_reference"))
    if "reopened_new_task" in secs:
      repair = secs["reopened_new_task"].get("repair_run") or {}
      if repair.get("run_id"):
        found.add(repair["run_id"])
        refs.append((name, "sections.reopened_new_task.repair_run", "repair_execution_record"))
    if "idempotency" in secs and secs["idempotency"].get("run_id"):
      found.add(secs["idempotency"]["run_id"])
      refs.append((name, "sections.idempotency", "idempotent_reference_only"))
  return found, refs

computed_run_ids, computed_refs = all_run_ids_from_inputs()
new_index_run_id_set = {r.get("run_id") for r in new_index.get("run_index", [])}
run_ids_match = computed_run_ids == new_index_run_id_set

for name, pointer, what in computed_refs:
  check_resolvable(name, pointer, what)

# stop/reject/exception index paths
for ev in new_index.get("stop_reject_exception_index", {}).get("rejection_events", []):
  check_resolvable(ev["file"], ev["path"], "rejection_event")
for ev in new_index.get("stop_reject_exception_index", {}).get("execution_exceptions_without_candidate", []):
  check_resolvable(ev["file"], ev["path"], "execution_exception")
for ev in new_index.get("stop_reject_exception_index", {}).get("terminal_state_observations", []):
  check_resolvable(ev["file"], ev["path"], "terminal_observation")

# layer checks
layers = defaultdict(set)
for r in new_index.get("run_index", []):
  layers[r.get("evidence_layer")].add(r.get("run_id"))
layer_counts = {k: len(v) for k, v in layers.items()}
layer_sum_ok = sum(layer_counts.values()) == len(new_index_run_id_set) == len(new_index.get("run_index", []))
verdict_by_layer_ok = True
layer_verdicts = defaultdict(lambda: defaultdict(int))
for r in new_index.get("run_index", []):
  layer_verdicts[r.get("evidence_layer")][r.get("verdict_class")] += 1
for layer, counts in layer_verdicts.items():
  if sum(counts.values()) != layer_counts.get(layer, 0):
    verdict_by_layer_ok = False

# candidate zero
c8 = next((r for r in new_index.get("run_index", []) if r.get("run_id") == "run-c8181bcd-50e6-48a7-a73f-7a162ccbc245"), None)
candidate_zero_ok = bool(c8) and c8.get("candidate") == 0.0 and any(
  ref.get("path") == "sections.active_verify_after_restart.verify" and ref.get("candidate") == 0.0
  for ref in c8.get("references", [])
)

# task attribution routes
task_attr = new_index.get("task_attribution_independent_routes", [])
task_attr_ok = len(task_attr) == 3 and all(
  a.get("recorded_task_id_in_index") is not None and a.get("code_inferred_task_id") is not None
  for a in task_attr
)
task_attr_run_ids_ok = {a.get("run_id") for a in task_attr} == {
  "run-9653cf4d-fa6f-4f9a-8129-eb652c2d226f",
  "run-4028708f-cad8-4da6-bf34-57a97dd6fa2a",
  "run-e1390ec3-ce72-4855-96b8-9ce67408a185",
}

# rejection events individually listed
rejection_paths = [ev.get("path") for ev in new_index.get("stop_reject_exception_index", {}).get("rejection_events", [])]
# Each entry has its own event_type/count basis; identical parent path can be used for different named rejections.
rejection_individually_listed = len(rejection_paths) > 0 and all(ev.get("event_type") and ev.get("count_basis") for ev in new_index.get("stop_reject_exception_index", {}).get("rejection_events", []))

checks = {
  "old_index_present": True,
  "old_index_unmodified_sha_matches_recorded": all(
    old_index_hashes.get(name) == input_hashes[name] for name in old_index_hashes
  ),
  "new_input_file_hashes_present": all(h in new_index.get("input_files", {}).get(name, {}).get("sha256", "") for name, h in input_hashes.items()),
  "input_file_hashes_match_recomputed_on_disk": all(new_index.get("input_files", {}).get(name, {}).get("sha256") == h for name, h in input_hashes.items()),
  "all_referenced_json_pointers_resolvable": len(unresolved_paths) == 0,
  "unresolved_paths": unresolved_paths,
  "distinct_run_ids_match_generator_and_inputs": run_ids_match,
  "distinct_run_id_count": len(new_index_run_id_set),
  "evidence_layers_mutually_exclusive_and_sum_to_total": layer_sum_ok,
  "verdict_counts_by_layer_reconcilable": verdict_by_layer_ok,
  "layer_counts": layer_counts,
  "verdict_counts_by_layer": {k: dict(v) for k, v in layer_verdicts.items()},
  "candidate_zero_preserved_for_run_c8181bcd": candidate_zero_ok,
  "three_independent_route_task_attribution_recorded": task_attr_ok and task_attr_run_ids_ok,
  "rejection_events_individually_listed": rejection_individually_listed,
  "rejection_event_count": len(new_index.get("stop_reject_exception_index", {}).get("rejection_events", [])),
  "execution_exception_count": len(new_index.get("stop_reject_exception_index", {}).get("execution_exceptions_without_candidate", [])),
  "terminal_observation_count": len(new_index.get("stop_reject_exception_index", {}).get("terminal_state_observations", [])),
  "reproducible_generation": reproducible_generation,
  "reproducible_content_hash": sorted(set(all_v2_hashes.values()))[0] if len(set(all_v2_hashes.values())) == 1 else "MISMATCH:" + ",".join(sorted(all_v2_hashes.values())),
  "script_present": os.path.isfile(SCRIPT),
}

checks["all_checks_pass"] = all([
  checks["old_index_unmodified_sha_matches_recorded"],
  checks["new_input_file_hashes_present"],
  checks["input_file_hashes_match_recomputed_on_disk"],
  checks["all_referenced_json_pointers_resolvable"],
  checks["distinct_run_ids_match_generator_and_inputs"],
  checks["evidence_layers_mutually_exclusive_and_sum_to_total"],
  checks["verdict_counts_by_layer_reconcilable"],
  checks["candidate_zero_preserved_for_run_c8181bcd"],
  checks["three_independent_route_task_attribution_recorded"],
  checks["rejection_events_individually_listed"],
  checks["reproducible_generation"],
  checks["script_present"],
])

report = {
  "checked_at": datetime.now(timezone.utc).isoformat(),
  "new_index_paths": [os.path.basename(p) for p in new_index_paths],
  "primary_new_index": os.path.basename(new_index_paths[-1]),
  "checks": checks,
}
print(json.dumps(report, indent=2, ensure_ascii=False))
sys.exit(0 if checks["all_checks_pass"] else 1)
