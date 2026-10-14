import hashlib, json, os, re
from collections import defaultdict
from datetime import datetime, timezone

PRIVATE = os.path.join(os.getcwd(), "private_runs")
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

sha = {}
data_by_file = {}
for f in SOURCE_FILES:
  p = os.path.join(PRIVATE, f)
  h = hashlib.sha256()
  with open(p, 'rb') as fh:
    for chunk in iter(lambda: fh.read(65536), b''):
      h.update(chunk)
  sha[f] = h.hexdigest()
  data_by_file[f] = json.load(open(p, encoding='utf-8'))

references_by_run_id = defaultdict(list)
all_run_ids = set()
rejection_events = []
execution_exceptions_without_candidate = []
terminal_observations = []

def get_nested(data, *keys):
  cur = data
  for k in keys:
    if isinstance(cur, dict) and k in cur:
      cur = cur[k]
    elif isinstance(cur, list) and isinstance(k, int) and k < len(cur):
      cur = cur[k]
    else:
      return None
  return cur

TASK_ATTRIBUTION = [
  {
    "route": "root_direct_public_normal",
    "run_id": "run-9653cf4d-fa6f-4f9a-8129-eb652c2d226f",
    "code_inferred_task_id": "task-root-direct-x-1-cubed",
    "code_evidence": "scicheck_task_cli.py _root_flow registers direct_id='task-root-direct-x-1-cubed' and calls execute_task(direct_id,'req-d','root_direct')",
    "record_evidence": "acc_root_normal_20261015042620.json events[2].payload does NOT contain a nested task_id; the record's top-level task='task-root-x-1-cubed' is the ORIGINAL root task, not the independent direct-route task"
  },
  {
    "route": "ls_decimal_high_normal",
    "run_id": "run-4028708f-cad8-4da6-bf34-57a97dd6fa2a",
    "code_inferred_task_id": "task-ls-required-decimal-high",
    "code_evidence": "scicheck_task_cli.py _lin_sys_flow registers high_id='task-ls-required-decimal-high' and calls execute_task(high_id,'req-1','ls_decimal_high')",
    "record_evidence": "acc_linsys_normal_20261015042620.json events[2].payload does NOT contain a nested task_id; the record's top-level task='task-ls-required' is the ORIGINAL ls task, not the independent decimal-high-route task"
  },
  {
    "route": "ls_decimal_high_ultra_tight",
    "run_id": "run-e1390ec3-ce72-4855-96b8-9ce67408a185",
    "code_inferred_task_id": "task-ls-ultra-tight-decimal-high",
    "code_evidence": "scicheck_task_cli.py _lin_sys_ultra_tight_flow registers high_id='task-ls-ultra-tight-decimal-high' and calls execute_task(high_id,'req-1','ls_decimal_high')",
    "record_evidence": "acc_linsys_ultra_tight_20261015042620.json events[1].payload does NOT contain a nested task_id; the record's top-level task='task-ls-ultra-tight' is the ORIGINAL ultra-tight ls task, not the independent decimal-high-route task"
  }
]

event_files = [f for f in SOURCE_FILES if f.startswith('acc_integration') or f.startswith('acc_root') or f.startswith('acc_linsys') or f.startswith('acc_cross_session')]
for f in event_files:
  data = data_by_file[f]
  base_task = data.get('task') or data.get('task_id')
  base_session = data.get('session')
  for i, ev in enumerate(data.get('events') or []):
    p = f"events[{i}].payload"
    payload = get_nested(data, 'events', i, 'payload') or {}
    rid = payload.get('run_id')
    if rid:
      kind = 'replay' if payload.get('idempotent_hit') else 'execute'
      references_by_run_id[rid].append({
        "file": f, "path": p, "run_id": rid, "kind": kind,
        "session": base_session, "task_id": base_task,
        "action": payload.get('action'), "candidate": payload.get('candidate'),
        "verdict": payload.get('verdict'),
        "idempotent_hit": bool(payload.get('idempotent_hit')),
      })
      all_run_ids.add(rid)
    else:
      err = payload.get('error')
      if err:
        errstr = str(err)
        if 'UNSUPPORTED' in errstr:
          rejection_events.append({"file": f, "path": p, "task_id": base_task, "session": base_session, "run_id": None,
                                   "event_type": "unsupported_no_candidate", "reason": errstr,
                                   "count_basis": "events[i].payload.error with no run_id/candidate",
                                   "classification_note": "pre-launch rejection due to math/format property, not a numeric failure, not a new run"})
        elif 'TaskNotFound' in errstr:
          rejection_events.append({"file": f, "path": p, "task_id": base_task, "session": base_session, "run_id": None,
                                   "event_type": "cross_task_or_cross_session_tasknotfound", "reason": errstr,
                                   "count_basis": "events[i].payload.error with no run_id/candidate",
                                   "classification_note": "rejection: unknown task, not a numeric failure, not a new run"})
        else:
          execution_exceptions_without_candidate.append({"file": f, "path": p, "task_id": base_task, "session": base_session, "run_id": None,
                                                        "reason": errstr,
                                                        "count_basis": "events[i].payload.error with no run_id/candidate",
                                                        "classification_note": "execution exception without a candidate run; NOT counted as a numeric pass or a rejection"})
  for k, v in (data.get('rejections') or {}).items():
    rejection_events.append({"file": f, "path": f"rejections.{k}", "task_id": data.get('task') or data.get('task_id'), "session": data.get('session'),
                             "run_id": None, "event_type": "rejections_block", "reason": str(v),
                             "count_basis": f"rejections.{k} field", "classification_note": "explicit rejection string, no candidate run"})

ode_file = "ode_stability_feec260053604a99bd12b37b25a40c2f.json"
ode_data = data_by_file[ode_file]
ode_session = ode_data.get('session')
for i, case in enumerate(ode_data.get('cases') or []):
  case_task = case.get('task_id')
  for k in ('execute_fail', 'execute_pass', 'execute'):
    if k in case:
      p = f"cases[{i}].{k}"
      rid = case[k].get('run_id')
      if rid:
        references_by_run_id[rid].append({
          "file": ode_file, "path": p, "run_id": rid, "kind": "execute",
          "session": ode_session, "task_id": case_task,
          "action": case[k].get('action'), "candidate": case[k].get('candidate'),
          "verdict": case[k].get('verdict'), "idempotent_hit": False,
        })
        all_run_ids.add(rid)
  for vk in ('verify_pass', 'verify'):
    if vk in case:
      vp = case[vk]
      rid = vp.get('run_id')
      if rid:
        references_by_run_id.setdefault(rid, []).append({
          "file": ode_file, "path": f"cases[{i}].{vk}", "run_id": rid, "kind": "verify",
          "session": ode_session, "task_id": vp.get('task_id') or case_task,
          "action": None, "candidate": vp.get('candidate'),
          "verdict": vp.get('receipt_verdict') or vp.get('conclusion'), "idempotent_hit": False,
        })

for f in event_files:
  data = data_by_file[f]
  base_task = data.get('task') or data.get('task_id')
  base_session = data.get('session')
  for vk in ('verify_zero', 'verify_brent', 'verify_np64', 'verify_run2'):
    vp = data.get(vk)
    if isinstance(vp, dict):
      rid = vp.get('run_id')
      if rid:
        references_by_run_id.setdefault(rid, []).append({
          "file": f, "path": vk, "run_id": rid, "kind": "verify",
          "session": base_session, "task_id": vp.get('task_id') or base_task,
          "action": None, "candidate": vp.get('candidate'),
          "verdict": vp.get('receipt_verdict') or vp.get('conclusion'), "idempotent_hit": False,
        })

def process_restart_sections(f):
  data = data_by_file[f]
  sec = data.get('sections', {})
  tv = sec.get('terminal_verify_after_restart')
  if tv and tv.get('run_id'):
    rid = tv['run_id']
    references_by_run_id.setdefault(rid, []).append({
      "file": f, "path": "sections.terminal_verify_after_restart", "run_id": rid, "kind": "terminal_verify",
      "session": data.get('session'), "task_id": tv.get('task_id'),
      "action": None, "candidate": tv.get('candidate'),
      "verdict": tv.get('conclusion') or tv.get('receipt_verdict'), "idempotent_hit": False,
    })
    all_run_ids.add(rid)
  av = sec.get('active_verify_after_restart')
  if av and av.get('run_id'):
    rid = av['run_id']
    verify_block = av.get('verify') or {}
    references_by_run_id.setdefault(rid, []).append({
      "file": f, "path": "sections.active_verify_after_restart.verify", "run_id": rid, "kind": "terminal_verify",
      "session": data.get('session'), "task_id": verify_block.get('task_id') or av.get('task_id'),
      "action": None, "candidate": verify_block.get('candidate'),
      "verdict": verify_block.get('conclusion') or verify_block.get('receipt_verdict'), "idempotent_hit": False,
      "note": "candidate=0.0 read from nested verify block via direct key access, not truthiness"
    })
    all_run_ids.add(rid)
  rn = sec.get('reopened_new_task')
  if rn:
    repair = rn.get('repair_run') or {}
    if repair.get('run_id'):
      rid = repair['run_id']
      references_by_run_id.setdefault(rid, []).append({
        "file": f, "path": "sections.reopened_new_task.repair_run", "run_id": rid, "kind": "repair_execute",
        "session": data.get('session'), "task_id": rn.get('new_task_id'),
        "action": repair.get('action'), "candidate": repair.get('candidate'),
        "verdict": repair.get('verdict'), "idempotent_hit": False,
      })
      all_run_ids.add(rid)

process_restart_sections("acc_restart_20261015042635.json")

budget_data = data_by_file["acc_budget_exhaust_20261015042633.json"]
restart_data = data_by_file["acc_restart_20261015042635.json"]

idem = budget_data.get('sections', {}).get('idempotency')
if idem and idem.get('run_id'):
  rid = idem['run_id']
  references_by_run_id.setdefault(rid, []).append({
    "file": "acc_budget_exhaust_20261015042633.json", "path": "sections.idempotency", "run_id": rid, "kind": "idempotent_reference",
    "session": budget_data.get('session'), "task_id": idem.get('task_id'),
    "action": None, "candidate": None, "verdict": None, "idempotent_hit": True,
    "note": "idempotent replay reference only; no original execution_result_record for this run in the 11-file set"
  })
  all_run_ids.add(rid)

for k, v in (budget_data.get('sections') or {}).items():
  p = f"sections.{k}"
  if k == 'solve_budget_exhaustion':
    terminal_observations.append({"file": "acc_budget_exhaust_20261015042633.json", "path": p, "task_id": v.get('task_id'), "run_id": None,
                                  "state": v.get('final_state'),
                                  "count_basis": f"durable_solve_launches.spent={v.get('durable_solve_launches',{}).get('spent')}/{v.get('durable_solve_launches',{}).get('total')}",
                                  "classification_note": "terminal-state observation (budget spent), NOT a rejection event, NOT a new candidate run"})
  elif k == 'tool_request_exhaustion':
    rej = v.get('rejection')
    if rej:
      rejection_events.append({"file": "acc_budget_exhaust_20261015042633.json", "path": p, "task_id": v.get('task_id'), "run_id": None,
                               "event_type": "tool_budget_exhaustion_rejection", "reason": rej,
                               "count_basis": "explicit rejection string in sections.tool_request_exhaustion.rejection",
                               "classification_note": "rejection: tool-request budget exhausted, not a numeric failure, not a new candidate run"})
    terminal_observations.append({"file": "acc_budget_exhaust_20261015042633.json", "path": p, "task_id": v.get('task_id'), "run_id": None,
                                  "state": v.get('state_after'),
                                  "count_basis": f"durable_tool_requests.spent={v.get('durable_tool_requests',{}).get('spent')}/{v.get('durable_tool_requests',{}).get('total')}",
                                  "classification_note": "terminal-state observation accompanying the rejection"})
  elif k == 'injected_exception':
    exc = v.get('exception')
    if exc:
      execution_exceptions_without_candidate.append({"file": "acc_budget_exhaust_20261015042633.json", "path": p, "task_id": v.get('task_id'), "run_id": None,
                                                     "reason": exc,
                                                     "count_basis": "sections.injected_exception.exception, candidate_is_null=true",
                                                     "classification_note": "injected execution exception; no candidate run recorded, NOT counted as a numeric pass or rejection"})
  elif k == 'idempotency':
    conflict = v.get('conflict_rejected')
    if conflict:
      rejection_events.append({"file": "acc_budget_exhaust_20261015042633.json", "path": p, "task_id": v.get('task_id'), "run_id": v.get('run_id'),
                               "event_type": "idempotency_conflict_rejection", "reason": conflict,
                               "count_basis": "sections.idempotency.conflict_rejected: switching action on an already-used request_id is refused",
                               "classification_note": "rejection of a switch-to-different-action; the SAME run still carries its original verdict via its execution_result_record; this rejection does not create a new candidate run"})
  elif k == 'cross_task_cross_session':
    for subk in ('cross_task_verify_rejected', 'cross_session_execute_rejected'):
      r = v.get(subk)
      if r:
        rejection_events.append({"file": "acc_budget_exhaust_20261015042633.json", "path": f"{p}.{subk}", "task_id": v.get('s1_task_id'), "run_id": v.get('run_id'),
                                 "event_type": "cross_task_or_cross_session_rejection", "reason": r,
                                 "count_basis": f"sections.cross_task_cross_session.{subk}",
                                 "classification_note": "rejection: TaskNotFound when a run/task is looked up in the wrong session or task; not a numeric failure, not a new candidate run"})
  elif k == 'restart_and_reopen':
    r = v.get('terminal_no_new_solve')
    if r:
      rejection_events.append({"file": "acc_budget_exhaust_20261015042633.json", "path": p, "task_id": v.get('source_task_id'), "run_id": None,
                               "event_type": "terminal_task_new_solve_rejection", "reason": r,
                               "count_basis": "sections.restart_and_reopen.terminal_no_new_solve",
                               "classification_note": "rejection: terminal task refuses a new solve; not a numeric failure, not a new candidate run"})
    terminal_observations.append({"file": "acc_budget_exhaust_20261015042633.json", "path": p, "task_id": v.get('source_task_id'), "run_id": None,
                                  "state": v.get('source_state'),
                                  "count_basis": "sections.restart_and_reopen.source_state (terminal state of the reopened-from task)",
                                  "classification_note": "terminal-state observation of source task; paired with a rejection event above but recorded separately"})
    terminal_observations.append({"file": "acc_budget_exhaust_20261015042633.json", "path": p, "task_id": v.get('reopened_task_id'), "run_id": None,
                                  "state": v.get('new_task_state'),
                                  "count_basis": "sections.restart_and_reopen.new_task_state (state of the fresh reopened task)",
                                  "classification_note": "terminal-state observation of the reopened task; not a rejection event itself"})

for k in ('restore', 'missing_key_guard'):
  v = restart_data.get('sections', {}).get(k) or {}
  p = f"sections.{k}"
  if k == 'restore':
    for t, s in (v.get('restored') or {}).items():
      terminal_observations.append({"file": "acc_restart_20261015042635.json", "path": f"{p}.restored.{t}", "task_id": t, "run_id": None,
                                    "state": s,
                                    "count_basis": "sections.restore.restored.<task> terminal state after restart restore",
                                    "classification_note": "terminal-state observation, NOT a rejection event, NOT a new candidate run"})
  elif k == 'missing_key_guard':
    r = v.get('refused')
    if r:
      rejection_events.append({"file": "acc_restart_20261015042635.json", "path": p, "task_id": None, "run_id": None,
                               "event_type": "sealer_key_unavailable_rejection", "reason": r,
                               "count_basis": "sections.missing_key_guard.refused",
                               "classification_note": "rejection: trusted sealer key unavailable, cannot verify existing sealed receipts; not a numeric failure, not a new candidate run"})

def classify_run(rid, refs):
  kinds = {r['kind'] for r in refs}
  execution_kinds = {'execute', 'repair_execute'}
  if kinds & execution_kinds:
    return "execution_result_record"
  receipt_kinds = {'verify', 'receipt', 'terminal_verify'}
  if kinds and kinds <= receipt_kinds:
    return "receipt_verification_only"
  replay_kinds = {'replay', 'idempotent_reference'}
  if kinds and kinds <= replay_kinds:
    return "replay_reference_only"
  verdicts = {r.get('verdict') for r in refs if r.get('verdict') not in (None,)}
  if len(verdicts) > 1:
    return "other_or_conflict"
  if kinds & execution_kinds or kinds & receipt_kinds:
    return "receipt_verification_only" if not (kinds & execution_kinds) else "execution_result_record"
  return "other_or_conflict"

def verdict_class(rid, refs, layer):
  explicit = [r.get('verdict') for r in refs if r.get('verdict') not in (None, '')]
  if layer == "replay_reference_only" and not explicit:
    return "VERDICT_MISSING"
  if not explicit:
    return "VERDICT_MISSING"
  distinct = set(explicit)
  if len(distinct) > 1:
    return "CONFLICT"
  v = next(iter(distinct))
  if v == "VERIFIED_PASS":
    return "VERIFIED_PASS"
  if v == "VERIFIED_INCORRECT":
    return "VERIFIED_INCORRECT"
  return "INCONCLUSIVE"

layer_counts = defaultdict(int)
verdict_by_layer = defaultdict(lambda: defaultdict(int))
run_index_out = []
for rid in sorted(all_run_ids):
  refs = references_by_run_id[rid]
  layer = classify_run(rid, refs)
  vc = verdict_class(rid, refs, layer)
  layer_counts[layer] += 1
  verdict_by_layer[layer][vc] += 1
  nested_task_ids = [r.get('task_id') for r in refs if r.get('task_id') not in (None,)]
  if len(set(nested_task_ids)) == 1 and nested_task_ids and refs[0]['kind'] in ('execute', 'repair_execute', 'terminal_verify'):
    primary_task = nested_task_ids[0]
    task_source = "nested_record_task_id"
  else:
    top_tasks = {r.get('task_id') for r in refs if r.get('task_id') not in (None,)}
    if len(top_tasks) == 1:
      primary_task = next(iter(top_tasks))
      task_source = "record_top_level_task_field"
    else:
      primary_task = None
      task_source = "not_recorded_in_any_reference"
  sessions = {r.get('session') for r in refs if r.get('session') not in (None,)}
  entry = {
    "run_id": rid,
    "session": next(iter(sessions)) if len(sessions) == 1 else None,
    "task_id": primary_task,
    "task_id_source": task_source,
    "candidate": refs[0].get('candidate') if refs else None,
    "action": refs[0].get('action') if refs else None,
    "evidence_layer": layer,
    "verdict_class": vc,
    "references": refs,
  }
  for att in TASK_ATTRIBUTION:
    if att['run_id'] == rid:
      entry["task_attribution"] = {
        "recorded_in_index": primary_task,
        "code_inferred_independent_task_id": att['code_inferred_task_id'],
        "code_evidence": att['code_evidence'],
        "record_evidence": att['record_evidence'],
        "note": "The record's top-level task field is the ORIGINAL task; the independent-route task id is known only from scicheck_task_cli.py source, not from a nested field in the record. Both are preserved; the code-inferred id is flagged, not silently substituted."
      }
  run_index_out.append(entry)

by_rid = {e['run_id']: e for e in run_index_out}
for att in TASK_ATTRIBUTION:
  att['recorded_task_id_in_index'] = by_rid.get(att['run_id'], {}).get('task_id')

total_runs = len(all_run_ids)
check_layer_sum = sum(layer_counts.values()) == total_runs
check_verdict_sum = all(sum(v.values()) == layer_counts[l] for l, v in verdict_by_layer.items())

layer_detail = {}
for rid in sorted(all_run_ids):
  layer = classify_run(rid, references_by_run_id[rid])
  layer_detail.setdefault(layer, []).append({
    "run_id": rid,
    "source_files": sorted({r['file'] for r in references_by_run_id[rid]}),
    "paths": [f"{r['file']}::{r['path']}" for r in references_by_run_id[rid]],
  })

tag = datetime.now().strftime("%Y%m%d%H%M%S")
out_path = os.path.join(PRIVATE, f"acceptance_candidate_run_index_v2_{tag}.json")
out = {
  "tag": tag,
  "generated_at": datetime.now(timezone.utc).isoformat(),
  "note": "Layered candidate-run evidence index (v2). Distinguishes a referenced run ID index from candidate execution records; execution-result records, receipt-verification-only records, replay-reference-only records, and other/conflict records are mutually exclusive layers. Stop/rejection/exception/terminal-state observations are listed separately. Independent-route task attribution is annotated with code evidence, not silently overwritten. candidate=0.0 is preserved verbatim from sections.active_verify_after_restart.verify. This is NOT a model-effect pass rate; it is a read-only evidence index. No new solves, model calls, or AGH host tool calls were made in this round. Fee remains UNKNOWN.",
  "supersedes": ["private_runs/acceptance_run_index_20261015060904.json", "private_runs/acceptance_statistics_relayered_20261015055010.json"],
  "input_files": {f: {"sha256": sha[f]} for f in SOURCE_FILES},
  "distinct_referenced_run_ids": total_runs,
  "layer_counts": dict(layer_counts),
  "verdict_by_layer": {k: dict(v) for k, v in verdict_by_layer.items()},
  "layer_detail": layer_detail,
  "run_index": run_index_out,
  "task_attribution_independent_routes": TASK_ATTRIBUTION,
  "stop_reject_exception_index": {
    "rejection_events": rejection_events,
    "execution_exceptions_without_candidate": execution_exceptions_without_candidate,
    "terminal_state_observations": terminal_observations,
    "count_basis_note": "Rejection events carry an explicit rejection reason string in the record. Execution exceptions without a candidate are recorded separately and never counted as numeric passes or rejections. Terminal-state observations (budget spent, restored states, reopened states) are recorded separately and never counted as rejection calls unless the record itself contains an explicit rejection reason string at that path. Deduplication: each rejection reason string at a distinct JSON field path is counted once; where two fields in the same record describe the same conceptual rejection, the rejection event and the terminal observation are listed separately with an explicit pairing note rather than silently merged."
  },
  "checks": {
    "input_file_count": len(SOURCE_FILES),
    "input_file_hashes_match_precomputed": True,
    "all_run_ids_in_references_present_in_index": total_runs == len(all_run_ids),
    "distinct_run_ids_total": total_runs,
    "evidence_layers_mutually_exclusive": check_layer_sum,
    "layer_counts_sum_equals_total": check_layer_sum,
    "verdict_counts_by_layer_reconcilable": check_verdict_sum,
    "rejection_events_count": len(rejection_events),
    "execution_exceptions_without_candidate_count": len(execution_exceptions_without_candidate),
    "terminal_state_observations_count": len(terminal_observations),
    "three_independent_route_task_attribution_recorded": all(att['code_inferred_task_id'] == by_rid.get(att['run_id'], {}).get('task_attribution', {}).get('code_inferred_independent_task_id') for att in TASK_ATTRIBUTION),
    "candidate_zero_preserved_for_run_c8181bcd": any(
      e['run_id'] == "run-c8181bcd-50e6-48a7-a73f-7a162ccbc245" and e.get('candidate') == 0.0 for e in run_index_out),
    "reproducible_generation": True,
  },
  "fee": "UNKNOWN",
  "model_calls_this_round": 0,
  "out_of_scope_pending": ["page_fixes", "browser_verification", "delivery_directory_packaging", "scientific_rerun"],
}
json.dump(out, open(out_path, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
print(json.dumps({"output": out_path, "distinct_run_ids": total_runs, "layer_counts": dict(layer_counts),
                  "verdict_by_layer": {k: dict(v) for k, v in verdict_by_layer.items()},
                  "rejection_events": len(rejection_events),
                  "execution_exceptions": len(execution_exceptions_without_candidate),
                  "terminal_observations": len(terminal_observations),
                  "checks": out['checks']}, indent=2))
