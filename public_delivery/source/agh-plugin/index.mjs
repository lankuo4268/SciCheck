// AGH backend plugin for the current SciCheck project.
//
// Model-facing tools:
//   - method_lookup
//   - execute_task
//   - verify_run
//
// The model-facing parameters are business identifiers only. They do not include
// session, db, key, python, workspace, bridge, budget, shell, registration,
// credential, or receipt-byte fields.
//
// The trusted AGH session identity comes from `ctx.session.key`. The math task
// session, db/key paths, and allowed task ids come only from the control-side
// file `agh-host-binding.json`. If no trusted job is active, tools fail with
// NOT_READY before invoking the local business bridge.

const Kind = Symbol.for('TypeBox.Kind')

function stringProperty(maxLength) {
  return { [Kind]: 'String', type: 'string', maxLength }
}

function toolSchema(properties, required) {
  return {
    [Kind]: 'Object',
    type: 'object',
    properties,
    required: [...required],
    additionalProperties: false,
  }
}

const sharedMeta = {
  isReadOnly: false,
  isDestructive: false,
  isConcurrencySafe: false,
  isOpenWorld: false,
  replay: 'idempotent',
  costHint: undefined,
  deferLoading: false,
  requiresApproval: 'never',
}

function failure(ctx, command, visibleArgs, error, code = 4) {
  const out = {
    ok: false,
    command,
    model_visible_args: visibleArgs,
    error,
    code,
  }
  ctx.log?.warn?.(`scicheck business tool rejected: ${command}: ${error}`)
  return {
    content: [{ type: 'text', text: JSON.stringify(out, null, 2) }],
    structured: out,
    isError: true,
  }
}

function success(ctx, command, visibleArgs, parsed, result) {
  const out = {
    ok: result.code === 0 && parsed.ok === true,
    command,
    model_visible_args: visibleArgs,
    code: result.code,
    result: parsed,
  }
  ctx.log?.info?.(`scicheck business tool completed: ${command}`)
  return {
    content: [{ type: 'text', text: JSON.stringify(out, null, 2) }],
    structured: out,
    isError: !out.ok,
  }
}

function bytesToText(bytes) {
  if (typeof bytes === 'string') return bytes
  if (bytes instanceof Uint8Array) return new TextDecoder().decode(bytes)
  if (bytes && typeof bytes === 'object') {
    if (bytes.buffer instanceof ArrayBuffer) return new TextDecoder().decode(bytes)
    if (Array.isArray(bytes)) return new TextDecoder().decode(Uint8Array.from(bytes))
  }
  throw new Error('unsupported byte payload from trusted ctx.fs.read')
}

function joinWorkspacePath(root, rel) {
  if (!root || typeof root !== 'string') return undefined
  const normalized = String(root).replace(/\\/g, '/')
  if (normalized.endsWith('/')) {
    return normalized + rel
  }
  return `${normalized}/${rel}`
}

async function loadBinding(ctx) {
  if (!ctx || !ctx.fs || typeof ctx.fs.read !== 'function') {
    throw new Error('trusted host ctx.fs.read is unavailable; cannot verify the active SciCheck binding')
  }
  const trustedWorkspaceRoot = ctx.session?.workspaceRoot
  const workspaceRoot = trustedWorkspaceRoot || ctx.cwd
  if (!workspaceRoot || typeof workspaceRoot !== 'string') {
    throw new Error('trusted host ctx.session.workspaceRoot and ctx.cwd are both unavailable')
  }
  const bindingPath = joinWorkspacePath(workspaceRoot, 'agh-host-binding.json')
  if (!bindingPath) {
    throw new Error('trusted host workspace root is unavailable')
  }
  const bytes = await ctx.fs.read(bindingPath)
  let binding
  try {
    binding = JSON.parse(bytesToText(bytes))
  } catch (error) {
    throw new Error(`agh-host-binding.json is not valid JSON: ${error?.message || error}`)
  }
  if (binding.version !== 1) {
    throw new Error('unsupported agh-host-binding.json version; expected version 1')
  }
  if (!binding.defaults?.db || !binding.defaults?.key || !binding.python || !binding.workspace || !binding.bridge) {
    throw new Error('agh-host-binding.json defaults are incomplete')
  }
  return binding
}

function resolveBinding(binding, ctx, args) {
  const aghSessionKey = ctx.session?.key
  if (!aghSessionKey || typeof aghSessionKey !== 'string') {
    return { error: 'AGH session key is unavailable from trusted ctx.session.key' }
  }
  const activeJob = binding.active_job
  if (!activeJob || activeJob.state !== 'ACTIVE') {
    return { error: 'NOT_READY: no active trusted SciCheck job binding; the model cannot create or reset tasks' }
  }

  const sessionMapping = binding.session_mapping || {}
  let mathSession
  if (Object.prototype.hasOwnProperty.call(sessionMapping, aghSessionKey)) {
    mathSession = sessionMapping[aghSessionKey]
  } else if (Object.prototype.hasOwnProperty.call(activeJob, 'session_mapping')) {
    const jobMapping = activeJob.session_mapping || {}
    if (!Object.prototype.hasOwnProperty.call(jobMapping, aghSessionKey)) {
      return {
        error: `NOT_READY: AGH session key ${JSON.stringify(aghSessionKey)} is not mapped by the active trusted binding`,
      }
    }
    mathSession = jobMapping[aghSessionKey]
  } else {
    return {
      error: `NOT_READY: no session_mapping maps AGH session key ${JSON.stringify(aghSessionKey)} to a math task session`,
    }
  }
  if (!mathSession || typeof mathSession !== 'string') {
    return { error: 'NOT_READY: active trusted binding does not map the AGH session key to a math session' }
  }

  const allowedTaskIds = new Set(binding.allowed_task_ids || [])
  if (args.task_id && !allowedTaskIds.has(args.task_id)) {
    return {
      error: `task_id ${JSON.stringify(args.task_id)} is not in the active trusted binding's allowed_task_ids`,
    }
  }

  return {
    binding,
    aghSessionKey,
    mathSession,
    db: activeJob.db || binding.defaults.db,
    key: activeJob.key || binding.defaults.key,
    python: binding.python,
    workspace: binding.workspace,
    bridge: binding.bridge,
  }
}

async function runBusiness(ctx, args, command, extra) {
  let resolved
  try {
    const binding = await loadBinding(ctx)
    resolved = resolveBinding(binding, ctx, args)
    if (resolved.error) {
      return failure(ctx, command, { task_id: args.task_id, extra }, resolved.error, 4)
    }
  } catch (error) {
    return failure(ctx, command, { task_id: args.task_id, extra }, error?.message || String(error), 4)
  }

  const argv = [
    resolved.python,
    resolved.bridge,
    command,
    ...extra,
    '--session', resolved.mathSession,
    '--db', resolved.db,
    '--key', resolved.key,
  ]
  const result = await ctx.exec(argv, { cwd: resolved.workspace, timeoutMs: 120_000 })
  let parsed = {}
  try {
    parsed = JSON.parse(result.stdout)
  } catch {
    parsed = { raw_stdout: result.stdout, raw_stderr: result.stderr }
  }
  const visible = {
    task_id: args.task_id,
  }
  if (args.card_id) visible.card_id = args.card_id
  if (args.request_id) visible.request_id = args.request_id
  if (args.action_id) visible.action_id = args.action_id
  if (args.run_id) visible.run_id = args.run_id
  if (result.code !== 0 || parsed.ok !== true) {
    return failure(ctx, command, visible, parsed.error ?? result.stderr ?? 'business bridge failed', result.code)
  }
  return success(ctx, command, visible, parsed, result)
}

export const main = {
  inject: ['extension'],
  apply(ctx) {
    const agnes = ctx.extension()

    agnes.registerTool({
      name: 'method_lookup',
      description:
        'Inspect the structure card or per-resource budget card for a registered SciCheck task. ' +
        'Requires an active trusted job binding. It does not launch a solve; each lookup counts as a real tool request.',
      parameters: toolSchema(
        {
          task_id: stringProperty(128),
          card_id: { [Kind]: 'String', type: 'string', enum: ['structure', 'budget'] },
        },
        ['task_id', 'card_id'],
      ),
      meta: sharedMeta,
      async execute(args, ctx) {
        return runBusiness(ctx, args, 'method_lookup', ['--task', args.task_id, '--card', args.card_id])
      },
    })

    agnes.registerTool({
      name: 'execute_task',
      description:
        'Run exactly one trusted candidate action for a registered SciCheck task. ' +
        'Requires an active trusted job binding. Same request_id and action_id is idempotent; terminal tasks refuse new solves.',
      parameters: toolSchema(
        {
          task_id: stringProperty(128),
          request_id: stringProperty(128),
          action_id: stringProperty(128),
        },
        ['task_id', 'request_id', 'action_id'],
      ),
      meta: sharedMeta,
      async execute(args, ctx) {
        return runBusiness(
          ctx,
          args,
          'execute_task',
          ['--task', args.task_id, '--request', args.request_id, '--action', args.action_id],
        )
      },
    })

    agnes.registerTool({
      name: 'verify_run',
      description:
        'Read and verify the sealed receipt for a SciCheck run without re-solving. ' +
        'Requires an active trusted job binding. Each verification counts as a real tool request. ' +
        'The model-facing result strips private reference fields.',
      parameters: toolSchema(
        {
          task_id: stringProperty(128),
          run_id: stringProperty(128),
        },
        ['task_id', 'run_id'],
      ),
      meta: sharedMeta,
      async execute(args, ctx) {
        return runBusiness(ctx, args, 'verify_run', ['--task', args.task_id, '--run', args.run_id])
      },
    })

    agnes.on('session_start', () => {
      agnes.ctx.log.info('scicheck-business-tools: method_lookup / execute_task / verify_run available')
    })
  },
}
