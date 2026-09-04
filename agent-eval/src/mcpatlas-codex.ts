// Driving OpenAI Codex CLI headless and reading back what it did — the codex
// counterpart of mcpatlas-agent.ts. The claude driver is deliberately untouched
// by the harness feature; everything codex-specific lives here.
//
// The two harnesses meet at three shared shapes: `RunClaudeOutcome` (the raw
// subprocess outcome), `ClaudeResult` (the result envelope every downstream
// consumer reads), and `RawToolUse[]`/`TurnUsage[]` (the per-call and per-turn
// trace). This file's job is to produce those exact shapes from Codex's JSONL
// event stream so that effectiveCalls, buildTokenBreakdown,
// buildLatencyBreakdown and assembleCell run unchanged on both harnesses.
//
// Codex facts this file encodes (verified against openai/codex rust-v0.153.0):
//   - `codex exec <prompt> --json` emits JSONL events on stdout; no cost is
//     reported, only tokens (turn.completed.usage).
//   - There is no --mcp-config flag: config lives in $CODEX_HOME/config.toml,
//     and CODEX_HOME relocates all state (sessions, auth) — the analog of the
//     throwaway-HOME trick the claude driver uses.
//   - There is no --max-turns equivalent; only the per-cell wall-clock timeout
//     bounds a runaway (recorded in CodexLockdown.max_turns_enforced).
//   - MCP tools are model-named `mcp__<server>__<tool>` — the same convention
//     as Claude Code — so normalizeToolId and the GATEWAY_* constants work
//     unchanged. The exec event stream additionally carries server/tool as
//     split fields on mcp_tool_call items, which this parser re-joins.
//   - Compaction is NOT visible in the event stream; it appears only in the
//     session rollout file under $CODEX_HOME/sessions/.

import { spawn } from "node:child_process";
import { existsSync, readdirSync } from "node:fs";
import { join } from "node:path";
import { stringify as tomlStringify } from "smol-toml";
import type {
  ClaudeResult,
  ClaudeUsage,
  RawToolUse,
  RunClaudeOutcome,
  TurnUsage,
} from "./mcpatlas-agent.js";
import type { InvokeSpan } from "./mcpatlas-gateway.js";
import { normalizeToolId } from "./mcpatlas-servers.js";
import type { CodexLockdown, CodexPricing } from "./mcpatlas-types.js";

// ─────────────────────────────────────────────────────────────────────────────
// Pricing
// ─────────────────────────────────────────────────────────────────────────────

/** $/Mtok, verified 2026-09-03 (OpenAI GPT-5.6 pricing announcement +
 *  aggregators; cached input = 10% of the input rate on both entries).
 *
 *  Only models PROVEN to run through `codex exec` on this CLI version are
 *  listed. The gpt-5.1-codex family — including the originally planned
 *  gpt-5.1-codex-mini — still appears in /v1/models but 404s on the
 *  Responses endpoint (retired), measured directly on codex-cli 0.153.0.
 *
 *  gpt-5.6-luna is the pinned cheap-tier analog of claude-haiku-4-5
 *  ($0.20 / $0.02 cached / $1.20, with the 5.6 family's 1.25x cache-write
 *  billing carried as a 0.25x surcharge). gpt-5.3-codex ($1.75 / $0.175 /
 *  $14.00) is the codex-branded alternative. Note 5.6 pricing was cut twice
 *  in mid-2026 and sol's is promotional — re-verify before adding entries or
 *  quoting cross-model cost comparisons.
 *
 *  The selected entry is frozen into the run config (`codex_pricing`) and so
 *  into config_hash: a price change is a different experiment, not a silent
 *  restatement of old rows. */
export const CODEX_PRICING: Readonly<Record<string, CodexPricing>> = {
  "gpt-5.6-luna": {
    model: "gpt-5.6-luna",
    input_usd_per_mtok: 0.2,
    cached_input_usd_per_mtok: 0.02,
    output_usd_per_mtok: 1.2,
    cache_write_surcharge_usd_per_mtok: 0.05,
  },
  "gpt-5.3-codex": {
    model: "gpt-5.3-codex",
    input_usd_per_mtok: 1.75,
    cached_input_usd_per_mtok: 0.175,
    output_usd_per_mtok: 14,
  },
};

// ─────────────────────────────────────────────────────────────────────────────
// Config
// ─────────────────────────────────────────────────────────────────────────────

/** Codex's per-MCP-server tool timeout. Its default is 60s, which real Atlas
 *  tools (code executors, live GitHub queries) can legitimately exceed — the
 *  claude arm has no per-tool timeout at all, so a low value here would
 *  manufacture harness-correlated tool failures. Generous, but still inside
 *  the per-cell timeout so a hung tool fails the cell rather than consuming
 *  it. */
export const CODEX_TOOL_TIMEOUT_SEC = 600;

/** The frozen lockdown descriptor recorded in the run config. Codex has no
 *  `--disallowedTools`; the built-in surface is closed off via config feature
 *  flags, and this object is the record of exactly how. It is a WEAKER lockdown
 *  than the claude arm's: `codex exec` requires `-s danger-full-access` to run
 *  MCP tools at all (see buildCodexArgs), which re-exposes the built-in `exec`
 *  shell and leaves `apply_patch` write-capable. Neither reaches the task data
 *  (which lives behind the MCP servers, not on codex's throwaway workspace),
 *  but `exec` with network could bypass the gateway — so its use is measured
 *  per cell (McpAtlasCell.shell_command_executions) rather than assumed zero. */
export function codexLockdown(): CodexLockdown {
  return {
    sandbox_mode: "danger-full-access",
    shell_tool: false,
    web_search: "disabled",
    view_image: false,
    error_on_tool_collisions: true,
    shell_reachable: true,
    max_turns_enforced: false,
    addendum_delivery: "agents_md",
    tool_timeout_sec: CODEX_TOOL_TIMEOUT_SEC,
  };
}

/** Structural twin of mcpatlas-servers.ts's (unexported) McpServerEntry — the
 *  values of buildNativeMcpConfig(...).mcpServers and
 *  buildRatelMcpConfig(...).mcpServers assign to it directly, which is the
 *  point: the codex config is rendered from the SAME entries as the claude
 *  mcp.json, so the arms differ across harnesses only in serialization. */
export interface CodexMcpServerEntry {
  type?: string;
  command: string;
  args: string[];
  env?: Record<string, string>;
}

export interface CodexConfigOpts {
  model: string;
  mcpServers: Record<string, CodexMcpServerEntry>;
  /** MCP server startup budget. The claude driver passes the same value via
   *  the MCP_TIMEOUT env var; Codex takes it per-server in config.toml. */
  startupTimeoutSec: number;
  toolTimeoutSec: number;
  /** Server names to mark `required = true`. See the required/optional split
   *  in the config builder — the ratel gateway is required (it IS the tool
   *  surface, and npx-slow), native shims are left optional (fast, and one
   *  failing must not abort the session, matching the claude arm). Empty by
   *  default: every server optional. */
  requiredServers?: readonly string[];
}

/**
 * $CODEX_HOME/config.toml for one cell.
 *
 * Everything that is policy rather than prompt lives here — model, built-in
 * lockdown, MCP servers — so `buildCodexArgs` stays constant across arms and
 * the invariant "arms differ only in config content" holds for codex exactly
 * as agent.test.ts asserts it for claude.
 *
 * Deliberately NOT set: `enabled_tools`. Under claude, `--allowedTools` is
 * advisory (bypassPermissions) and the reachable surface is really controlled
 * by what the shims/gateway serve — the ratel arm's context contains all four
 * gateway tools, and gatewaySchemaTokens measures all four. Codex
 * `enabled_tools` would HIDE tools from the model, a strictly stronger
 * restriction that would give the two harnesses different ratel-arm contexts.
 */
export function buildCodexConfigToml(o: CodexConfigOpts): string {
  // required vs optional, measured directly on codex-cli 0.153.0:
  //
  //  - The ratel GATEWAY must be `required = true`. Codex treats MCP servers
  //    as optional by default and waits only `mcp_optional_startup_grace_ms`
  //    (1000ms) for their tool lists before starting the turn. An npx-launched
  //    ratel-local cannot cold-start and register its two gateway tools within
  //    1s, so the agent began the turn with ZERO ratel tools and gave up (a
  //    ratel cell whose `ALL_TOOLS` held only codex built-ins). required=true
  //    makes codex block on startup_timeout_sec instead.
  //
  //  - The native SHIMS must stay optional. They are fast `node` processes and
  //    register within the default grace, but there are 11 of them and
  //    `required = true` aborts the WHOLE session if any one fails to init
  //    ("required MCP servers failed to initialize: mcp-server-code-runner" —
  //    measured). The claude arm tolerates a dead server (its tools are just
  //    absent, caught later by the sandbox/registration integrity checks), so
  //    the codex native arm must too.
  const required = new Set(o.requiredServers ?? []);
  const mcp_servers: Record<string, Record<string, unknown>> = {};
  for (const [name, entry] of Object.entries(o.mcpServers)) {
    mcp_servers[name] = {
      command: entry.command,
      args: entry.args,
      ...(entry.env ? { env: entry.env } : {}),
      startup_timeout_sec: o.startupTimeoutSec,
      tool_timeout_sec: o.toolTimeoutSec,
      ...(required.has(name) ? { required: true } : {}),
    };
  }
  return tomlStringify({
    model: o.model,
    // `codex exec` already forces approval_policy=never; stating it makes the
    // frozen config self-describing rather than reliant on that default.
    approval_policy: "never",
    web_search: "disabled",
    tools: { update_plan: { enabled: false } },
    features: {
      shell_tool: false,
      view_image: false,
      tool_registry: { error_on_tool_collisions: true },
    },
    mcp_servers,
  });
}

export interface CodexArgsOpts {
  prompt: string;
  cwd: string;
}

/** Argv for `codex exec`. Constant across arms — model and lockdown live in
 *  config.toml. `--strict-config` makes a typo'd config key fail the cell
 *  loudly instead of silently running unlocked.
 *
 *  `-s danger-full-access` is forced by codex, not chosen freely. Measured on
 *  codex-cli 0.153.0: `codex exec` hard-overrides `approval_policy` to `never`
 *  (the config value is ignored), and under ANY restricted sandbox
 *  (`read-only`, `workspace-write`) an MCP tool call "requires approval" and is
 *  then auto-DENIED — "MCP tool call requires approval, but approval policy is
 *  never". Every task fails with zero successful tool calls. Only
 *  `danger-full-access` lets MCP tool calls execute unattended. Per-server
 *  `default_tools_approval_mode`, trusted-project trust_level, and the
 *  on-request/on-failure policies were all verified NOT to lift the denial
 *  under exec.
 *
 *  The cost is real and is why `command_execution` items are counted as a
 *  contamination signal (see parseCodexEvents): full access re-exposes codex's
 *  built-in `exec` shell, which `features.shell_tool=false` suppresses only
 *  under a restricted sandbox. Unlike Claude Code's `--disallowedTools`, codex
 *  headless cannot both run MCP tools and hide the shell; the codex arm's
 *  validity therefore rests on the agent NOT shelling out, which the harness
 *  now measures per cell instead of assuming. */
export function buildCodexArgs(o: CodexArgsOpts): string[] {
  return [
    "exec",
    o.prompt,
    "--json",
    "-C",
    o.cwd,
    "-s",
    "danger-full-access",
    "--skip-git-repo-check",
    "--strict-config",
  ];
}

// ─────────────────────────────────────────────────────────────────────────────
// Subprocess
// ─────────────────────────────────────────────────────────────────────────────

export interface RunCodexOpts {
  args: string[];
  cwd: string;
  /** Becomes CODEX_HOME. Must exist before spawn — codex canonicalizes it and
   *  errors on a missing directory. */
  codexHome: string;
  env: Record<string, string>;
  timeoutMs: number;
  bin?: string;
}

/**
 * Spawn `codex exec` and capture its outcome. Returns the same
 * `RunClaudeOutcome` shape as `runClaude` so runCell's error handling and the
 * signal-death diagnostics (SIGKILL + empty stderr = OOM signature) transfer.
 *
 * The process-group handling is duplicated line-for-line from `runClaude`
 * (mcpatlas-agent.ts:349-427) rather than extracted: that function is the most
 * load-bearing code in the repo and stays zero-diff. The semantics matter
 * identically here — codex exiting does NOT take ratel-local or its 11 stdio
 * shims with it; without reaping on EVERY exit path they accumulate at ~13
 * processes per ratel cell until the host OOMs (measured on AWS: 23 clean
 * cells, then collapse). SIGTERM before SIGKILL so ratel-local can flush
 * telemetry.jsonl, which the caller reads right after this resolves.
 */
export function runCodex(o: RunCodexOpts): Promise<RunClaudeOutcome> {
  const started = Date.now();
  return new Promise((resolve) => {
    const child = spawn(o.bin ?? "codex", o.args, {
      cwd: o.cwd,
      env: {
        ...o.env,
        CODEX_HOME: o.codexHome,
        // Measured on codex-cli 0.153.0: OPENAI_API_KEY alone is NOT attached
        // as a bearer (401 "Missing bearer or basic authentication in header"
        // on api.openai.com/v1/responses) — env-var auth works only through
        // CODEX_API_KEY, and the throwaway CODEX_HOME has no auth.json to
        // fall back on. Map it here so callers keep the conventional var.
        ...(o.env.OPENAI_API_KEY && !o.env.CODEX_API_KEY
          ? { CODEX_API_KEY: o.env.OPENAI_API_KEY }
          : {}),
      },
      detached: true,
      stdio: ["ignore", "pipe", "pipe"],
    });
    let stdout = "";
    let stderr = "";
    let timedOut = false;
    child.stdout.on("data", (d) => {
      stdout += d;
    });
    child.stderr.on("data", (d) => {
      stderr += d;
    });
    const killGroup = (signal: NodeJS.Signals): void => {
      const pid = child.pid;
      try {
        if (pid) process.kill(-pid, signal);
        else child.kill(signal);
      } catch {
        child.kill(signal);
      }
    };
    const timer = setTimeout(() => {
      timedOut = true;
      killGroup("SIGTERM");
      setTimeout(() => killGroup("SIGKILL"), 5_000).unref();
    }, o.timeoutMs);
    child.on("close", (code, signal) => {
      clearTimeout(timer);
      killGroup("SIGTERM");
      setTimeout(() => killGroup("SIGKILL"), 2_000).unref();
      resolve({ stdout, stderr, exitCode: code, signal, timedOut, wallMs: Date.now() - started });
    });
    child.on("error", (err) => {
      clearTimeout(timer);
      resolve({
        stdout,
        stderr: `${stderr}\n${(err as Error).message}`,
        exitCode: null,
        signal: null,
        timedOut,
        wallMs: Date.now() - started,
      });
    });
  });
}

// ─────────────────────────────────────────────────────────────────────────────
// Event stream
// ─────────────────────────────────────────────────────────────────────────────

export interface CodexParsedEvents {
  /** `thread.started.thread_id` — Codex's session id, used to locate the
   *  rollout file. */
  threadId: string | null;
  /** Count of `turn.completed` events. */
  numTurns: number;
  /** Last `agent_message` item's text. */
  finalMessage: string;
  failed: { message: string } | null;
  /** mcp_tool_call items re-joined to `mcp__<server>__<tool>`, in the shape
   *  toolUsesFromTranscript produces, so effectiveCalls runs unchanged. */
  uses: RawToolUse[];
  /** Per-turn usage in Anthropic shape (see mapping notes on parse loop). */
  turnUsages: TurnUsage[];
  /** Run totals, Anthropic shape. */
  usage: ClaudeUsage;
  reasoningOutputTokens: number;
  /** Count of built-in `exec` shell invocations (`command_execution` items).
   *  The contamination signal — see McpAtlasCell.shell_command_executions. */
  commandExecutions: number;
  /** Per mcp_tool_call outcome, in call order, aligned 1:1 with `uses`. Lets
   *  the native arm classify tool failures from the event stream, since it has
   *  no ratel telemetry to provide invoke spans. */
  callOutcomes: Array<{ name: string; error: string | null }>;
}

interface CodexUsage {
  input_tokens?: number;
  cached_input_tokens?: number;
  cache_write_input_tokens?: number;
  output_tokens?: number;
  reasoning_output_tokens?: number;
}

/** Codex usage -> Anthropic-shape turn usage.
 *
 *  Mapping: OpenAI's `input_tokens` INCLUDES cache reads while Anthropic's
 *  excludes them, so uncached input = input - cached (clamped at 0 in case a
 *  codex version reports them disjoint — verify at smoke time).
 *  `cache_write_input_tokens` maps onto `cache_creation_input_tokens`.
 *  `reasoning_output_tokens` is NOT folded into output — OpenAI's output
 *  figure may already include it. This mapping is what keeps totalTokens,
 *  cacheHitRatio, promptTokens and buildTokenBreakdown semantically correct
 *  on codex cells without any change to those functions. */
function mapUsage(turn: number, u: CodexUsage): TurnUsage {
  const input = u.input_tokens ?? 0;
  const cached = u.cached_input_tokens ?? 0;
  return {
    turn,
    input_tokens: Math.max(0, input - cached),
    output_tokens: u.output_tokens ?? 0,
    cache_read_input_tokens: cached,
    cache_creation_input_tokens: u.cache_write_input_tokens ?? 0,
  };
}

/**
 * Parse the `--json` event stream. Returns null when nothing recognizable was
 * emitted — the codex analog of "claude produced no parseable result
 * envelope", and runCell fails the cell the same way.
 *
 * Tool calls are taken from `item.completed` only (an mcp_tool_call fires
 * item.started at dispatch and item.completed at response; counting both
 * would double every call). The turn stamped on a use is the IN-PROGRESS
 * turn (completed count + 1), mirroring how the claude transcript parser
 * numbers assistant turns.
 */
export function parseCodexEvents(stdout: string): CodexParsedEvents | null {
  let threadId: string | null = null;
  let numTurns = 0;
  let finalMessage = "";
  let failed: { message: string } | null = null;
  const uses: RawToolUse[] = [];
  const turnUsages: TurnUsage[] = [];
  let reasoningOutputTokens = 0;
  let commandExecutions = 0;
  const callOutcomes: Array<{ name: string; error: string | null }> = [];
  let sawAny = false;

  for (const line of stdout.split("\n")) {
    const s = line.trim();
    if (!s.startsWith("{")) continue;
    let ev: Record<string, unknown>;
    try {
      ev = JSON.parse(s) as Record<string, unknown>;
    } catch {
      continue;
    }
    switch (ev.type) {
      case "thread.started": {
        sawAny = true;
        if (typeof ev.thread_id === "string") threadId = ev.thread_id;
        break;
      }
      case "turn.started": {
        sawAny = true;
        break;
      }
      case "turn.completed": {
        sawAny = true;
        numTurns++;
        const u = (ev.usage ?? {}) as CodexUsage;
        turnUsages.push(mapUsage(numTurns, u));
        reasoningOutputTokens += u.reasoning_output_tokens ?? 0;
        break;
      }
      case "turn.failed": {
        sawAny = true;
        const e = ev.error as { message?: string } | undefined;
        failed = { message: e?.message ?? "turn failed" };
        break;
      }
      case "error": {
        sawAny = true;
        failed ??= { message: typeof ev.message === "string" ? ev.message : "error" };
        break;
      }
      case "item.completed": {
        sawAny = true;
        const item = (ev.item ?? {}) as Record<string, unknown>;
        if (item.type === "mcp_tool_call") {
          const server = typeof item.server === "string" ? item.server : "";
          const tool = typeof item.tool === "string" ? item.tool : "";
          if (server && tool) {
            const name = `mcp__${server}__${tool}`;
            uses.push({ name, input: parseArguments(item.arguments), turn: numTurns + 1 });
            // status: "failed" or an `error` object both mean the call did not
            // succeed. Captured so the native arm — which has no ratel telemetry
            // — can still classify tool failures instead of defaulting to "ok".
            const err =
              item.status === "failed" || item.error
                ? codexCallError(item.error) || "codex tool call failed"
                : null;
            callOutcomes.push({ name, error: err });
          }
        } else if (item.type === "command_execution") {
          // Built-in shell — the contamination signal.
          commandExecutions++;
        } else if (item.type === "agent_message" && typeof item.text === "string") {
          finalMessage = item.text;
        }
        break;
      }
      default:
        break;
    }
  }

  if (!sawAny) return null;
  const usage: ClaudeUsage = {
    input_tokens: 0,
    output_tokens: 0,
    cache_read_input_tokens: 0,
    cache_creation_input_tokens: 0,
  };
  for (const t of turnUsages) {
    usage.input_tokens += t.input_tokens;
    usage.output_tokens += t.output_tokens;
    usage.cache_read_input_tokens += t.cache_read_input_tokens;
    usage.cache_creation_input_tokens += t.cache_creation_input_tokens;
  }
  return {
    threadId,
    numTurns,
    finalMessage,
    failed,
    uses,
    turnUsages,
    usage,
    reasoningOutputTokens,
    commandExecutions,
    callOutcomes,
  };
}

/** The text of an mcp_tool_call `error`, which codex emits either as
 *  `{message}` or, for a tool that returned isError content, as the result
 *  envelope. Kept small — only the message matters for classification. */
function codexCallError(error: unknown): string {
  if (!error) return "";
  if (typeof error === "string") return error;
  if (typeof error === "object") {
    const m = (error as { message?: unknown }).message;
    if (typeof m === "string") return m;
    return JSON.stringify(error).slice(0, 300);
  }
  return String(error);
}

/**
 * Synthesize invoke spans from codex's per-call outcomes so the native arm can
 * classify tool failures. The ratel arm gets richer spans from ratel-local's
 * telemetry (timings included); this covers the native arm, which has none.
 * `took_ms` is null — codex's event stream carries no per-call duration.
 */
export function codexInvokeSpans(
  p: CodexParsedEvents,
  knownServers: readonly string[],
): InvokeSpan[] {
  const out: InvokeSpan[] = [];
  for (const c of p.callOutcomes) {
    const id = normalizeToolId(c.name, knownServers);
    if (!id) continue; // off-catalog names are handled by effectiveCalls
    out.push({ tool_id: id, args_size_bytes: 0, took_ms: null, error: c.error });
  }
  return out;
}

/** mcp_tool_call `arguments` — an object in current codex, but tolerate a
 *  JSON-encoded string, which some tool_call wire formats use. */
function parseArguments(v: unknown): Record<string, unknown> {
  if (v && typeof v === "object" && !Array.isArray(v)) return v as Record<string, unknown>;
  if (typeof v === "string") {
    try {
      const parsed = JSON.parse(v);
      if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) {
        return parsed as Record<string, unknown>;
      }
    } catch {
      // fall through
    }
  }
  return {};
}

// ─────────────────────────────────────────────────────────────────────────────
// Rollout (the codex "transcript")
// ─────────────────────────────────────────────────────────────────────────────

/** Locate the session rollout under
 *  `$CODEX_HOME/sessions/YYYY/MM/DD/rollout-<ts>-<thread_id>.jsonl`. A
 *  recursive scan rather than a date-path construction, so a timezone or
 *  layout change degrades to "still found" instead of "silently missing" —
 *  the same posture transcriptPath takes for claude. */
export function codexRolloutPath(codexHome: string, threadId: string | null): string | null {
  if (!threadId) return null;
  const root = join(codexHome, "sessions");
  if (!existsSync(root)) return null;
  const stack = [root];
  while (stack.length) {
    const dir = stack.pop();
    if (!dir) break;
    for (const entry of readdirSync(dir, { withFileTypes: true })) {
      const p = join(dir, entry.name);
      if (entry.isDirectory()) stack.push(p);
      else if (
        entry.name.startsWith("rollout-") &&
        entry.name.includes(threadId) &&
        entry.name.endsWith(".jsonl")
      ) {
        return p;
      }
    }
  }
  return null;
}

/** Compaction count from the rollout. Codex's `--json` stream carries no
 *  compaction signal at all; the rollout's `compacted` items are the only
 *  record. The claude counterpart is countCompactions over the session
 *  transcript. */
export function countCodexCompactions(rolloutText: string): number {
  let n = 0;
  for (const line of rolloutText.split("\n")) {
    const s = line.trim();
    if (!s.startsWith("{")) continue;
    try {
      const rec = JSON.parse(s) as { type?: string };
      if (rec.type === "compacted") n++;
    } catch {
      // partial line
    }
  }
  return n;
}

// ─────────────────────────────────────────────────────────────────────────────
// Result envelope
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Fold parsed events into the `ClaudeResult` envelope every downstream
 * consumer reads.
 *
 * `total_cost_usd` is COMPUTED from the frozen pricing table (Codex reports
 * no cost) — uncached input at the input rate, cache reads at the cached
 * rate, output at the output rate. OpenAI bills no separate cache-write
 * charge, so cache_creation tokens carry no cost term. Reasoning tokens are
 * assumed to be inside output_tokens (OpenAI bills reasoning as output);
 * verify at smoke time — the separate reasoning figure rides on the row for
 * exactly that audit. The computed figure feeds --dollar-global budget
 * enforcement, which is why it must be a real number rather than null.
 *
 * `duration_api_ms` has no codex source and no downstream consumer (verified:
 * nothing outside the interface reads it) — 0, not a guess.
 */
export function codexResultFromEvents(
  p: CodexParsedEvents,
  wallMs: number,
  timedOut: boolean,
  pricing: CodexPricing,
): ClaudeResult {
  const u = p.usage;
  const cost =
    (u.input_tokens * pricing.input_usd_per_mtok +
      u.cache_read_input_tokens * pricing.cached_input_usd_per_mtok +
      u.output_tokens * pricing.output_usd_per_mtok +
      u.cache_creation_input_tokens * (pricing.cache_write_surcharge_usd_per_mtok ?? 0)) /
    1_000_000;
  const subtype = timedOut ? "error_timeout" : p.failed ? "error_during_execution" : "success";
  return {
    is_error: timedOut || p.failed !== null,
    subtype,
    duration_ms: wallMs,
    duration_api_ms: 0,
    num_turns: p.numTurns,
    result: p.failed && !p.finalMessage ? p.failed.message : p.finalMessage,
    session_id: p.threadId ?? "",
    total_cost_usd: cost,
    usage: { ...u },
    permission_denials: [],
  };
}
