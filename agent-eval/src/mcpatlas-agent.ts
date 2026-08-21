// Driving Claude Code headless and reading back what it did.

import { spawn } from "node:child_process";
import { existsSync, readdirSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { normalizeToolId } from "./mcpatlas-servers.js";
import type { CanonicalToolId, McpAtlasToolCall } from "./mcpatlas-types.js";

/** Built-ins that must be off. With `Bash` the agent can `gh api` around the
 *  GitHub server and with `WebFetch` it can bypass Airtable — which converts a
 *  tool-routing benchmark into a shell benchmark, and does so UNEQUALLY: the
 *  native arm sees more MCP tools and has less incentive to shell out. Frozen
 *  and identical across arms. */
export const DISALLOWED_TOOLS = [
  "Bash",
  "WebFetch",
  "WebSearch",
  "Write",
  "Edit",
  "NotebookEdit",
  "Task",
] as const;

export interface ClaudeArgsOpts {
  prompt: string;
  mcpConfigPath: string;
  allowedTools: string[];
  model: string;
  maxTurns: number;
  /** Required for headless: every MCP call would otherwise block on a prompt.
   *  Safe only because HOME and cwd are throwaway, --strict-mcp-config is set,
   *  and the built-ins above are disallowed — the agent's entire reachable
   *  surface is the frozen MCP catalog. */
  permissionMode?: string;
  addDir?: string;
}

export function buildClaudeArgs(o: ClaudeArgsOpts): string[] {
  const args = [
    "-p",
    o.prompt,
    "--output-format",
    "json",
    "--mcp-config",
    o.mcpConfigPath,
    // Without this the developer's own ~/.claude.json servers leak in and poison
    // the catalog-size measurement — including, ironically, the ratel-local plugin.
    "--strict-mcp-config",
    "--model",
    o.model,
    "--max-turns",
    String(o.maxTurns),
    "--permission-mode",
    o.permissionMode ?? "bypassPermissions",
    "--allowedTools",
    o.allowedTools.join(","),
    "--disallowedTools",
    DISALLOWED_TOOLS.join(","),
  ];
  if (o.addDir) args.push("--add-dir", o.addDir);
  return args;
}

export interface ClaudeUsage {
  input_tokens: number;
  output_tokens: number;
  cache_read_input_tokens: number;
  cache_creation_input_tokens: number;
}

export interface ClaudeResult {
  is_error: boolean;
  subtype: string;
  duration_ms: number;
  duration_api_ms: number;
  num_turns: number;
  result: string;
  session_id: string;
  total_cost_usd: number;
  usage: ClaudeUsage;
  permission_denials: unknown[];
}

const ZERO_USAGE: ClaudeUsage = {
  input_tokens: 0,
  output_tokens: 0,
  cache_read_input_tokens: 0,
  cache_creation_input_tokens: 0,
};

/**
 * Read the final `type:"result"` envelope.
 *
 * Written as "the LAST result line" rather than "parse the whole stdout as JSON"
 * so that switching to `--output-format stream-json --verbose` is a flag change
 * rather than a rewrite. That matters because `--output-format json` returns ONLY
 * this envelope — no tool-use blocks — so the trace has to come from elsewhere
 * (see `readTranscript`), and stream-json is the fallback if the transcript path
 * proves brittle across Claude Code versions.
 */
export function parseClaudeResult(stdout: string): ClaudeResult | null {
  const candidates: ClaudeResult[] = [];
  const push = (v: unknown): void => {
    if (v && typeof v === "object" && (v as { type?: string }).type === "result") {
      candidates.push(v as ClaudeResult);
    }
  };
  try {
    const whole = JSON.parse(stdout);
    if (Array.isArray(whole)) whole.forEach(push);
    else push(whole);
  } catch {
    for (const line of stdout.split("\n")) {
      const s = line.trim();
      if (!s.startsWith("{")) continue;
      try {
        push(JSON.parse(s));
      } catch {
        // partial line
      }
    }
  }
  const r = candidates.at(-1);
  if (!r) return null;
  return { ...r, usage: { ...ZERO_USAGE, ...(r.usage ?? {}) } };
}

export function totalTokens(u: ClaudeUsage): number {
  return (
    u.input_tokens + u.output_tokens + u.cache_read_input_tokens + u.cache_creation_input_tokens
  );
}

/** cache_read / (cache_read + uncached input). Explains why native's occupancy
 *  savings outrun its dollar savings: the schemas live in the cached prefix. */
export function cacheHitRatio(u: ClaudeUsage): number {
  const denom = u.cache_read_input_tokens + u.input_tokens;
  return denom > 0 ? u.cache_read_input_tokens / denom : 0;
}

// ─────────────────────────────────────────────────────────────────────────────
// Transcript
// ─────────────────────────────────────────────────────────────────────────────

/** Claude Code slugifies the cwd for its project directory: every `/` and `.`
 *  becomes `-`, mirroring its own convention. */
export function slugifyProjectPath(p: string): string {
  return p.replace(/[/.]/g, "-");
}

/** Locate the session transcript. Deterministic here because HOME is a throwaway
 *  directory with exactly one session and `session_id` comes back in the result
 *  envelope. Falls back to a directory scan if the slug rule ever changes. */
export function transcriptPath(homeDir: string, cwd: string, sessionId: string): string | null {
  const direct = join(
    homeDir,
    ".claude",
    "projects",
    slugifyProjectPath(cwd),
    `${sessionId}.jsonl`,
  );
  if (existsSync(direct)) return direct;
  const root = join(homeDir, ".claude", "projects");
  if (!existsSync(root)) return null;
  for (const dir of readdirSync(root)) {
    const p = join(root, dir, `${sessionId}.jsonl`);
    if (existsSync(p)) return p;
  }
  return null;
}

export interface RawToolUse {
  name: string;
  input: Record<string, unknown>;
  turn: number;
}

/** Tool-use blocks from the transcript, in order, with their turn index. */
export function toolUsesFromTranscript(text: string): RawToolUse[] {
  const out: RawToolUse[] = [];
  let turn = 0;
  for (const line of text.split("\n")) {
    const s = line.trim();
    if (!s) continue;
    let rec: Record<string, unknown>;
    try {
      rec = JSON.parse(s);
    } catch {
      continue;
    }
    const msg = (rec.message ?? rec) as Record<string, unknown>;
    if (msg?.role !== "assistant") continue;
    turn++;
    const content = msg.content;
    if (!Array.isArray(content)) continue;
    for (const block of content) {
      const b = block as Record<string, unknown>;
      if (b?.type !== "tool_use" || typeof b.name !== "string") continue;
      out.push({
        name: b.name,
        input: (b.input ?? {}) as Record<string, unknown>,
        turn,
      });
    }
  }
  return out;
}

export const GATEWAY_SEARCH = "mcp__ratel-local__search_capabilities";
export const GATEWAY_INVOKE = "mcp__ratel-local__invoke_tool";

export interface EffectiveCalls {
  calls: McpAtlasToolCall[];
  /** Ids the agent named that resolve to no known server — a SELECTION defect,
   *  not a tool failure. */
  offCatalog: string[];
  gatewayCalls: number;
  nonGatewayCalls: number;
  searchCalls: number;
}

/**
 * Reduce raw tool uses to the underlying tool calls, unwrapping the gateway.
 *
 * `invoke_tool({toolId})` becomes the inner tool; `search_capabilities` is
 * counted but dropped. This is what makes the two arms comparable at all: the
 * native arm's `mcp__github__get_issue` and the ratel arm's
 * `invoke_tool({toolId: "github__get_issue"})` both reduce to
 * `github/get_issue`. Mirrors `metering.ts:effectiveCalls` against MCP names.
 */
export function effectiveCalls(
  uses: readonly RawToolUse[],
  knownServers: readonly string[],
): EffectiveCalls {
  const calls: McpAtlasToolCall[] = [];
  const offCatalog: string[] = [];
  let gatewayCalls = 0;
  let nonGatewayCalls = 0;
  let searchCalls = 0;

  for (const u of uses) {
    if (u.name === GATEWAY_SEARCH) {
      gatewayCalls++;
      searchCalls++;
      continue;
    }
    let rawId: string;
    let args: Record<string, unknown>;
    if (u.name === GATEWAY_INVOKE) {
      gatewayCalls++;
      const inner = u.input.toolId ?? u.input.tool_id;
      if (typeof inner !== "string") {
        offCatalog.push(String(inner ?? "<missing toolId>"));
        continue;
      }
      rawId = inner;
      args = (u.input.args ?? {}) as Record<string, unknown>;
    } else {
      nonGatewayCalls++;
      rawId = u.name;
      args = u.input;
    }
    const id: CanonicalToolId | null = normalizeToolId(rawId, knownServers);
    if (!id) {
      offCatalog.push(rawId);
      continue;
    }
    calls.push({ tool_id: id, args });
  }
  return { calls, offCatalog, gatewayCalls, nonGatewayCalls, searchCalls };
}

// ─────────────────────────────────────────────────────────────────────────────
// Subprocess
// ─────────────────────────────────────────────────────────────────────────────

export interface RunClaudeOpts extends ClaudeArgsOpts {
  cwd: string;
  homeDir: string;
  env: Record<string, string>;
  timeoutMs: number;
  bin?: string;
}

export interface RunClaudeOutcome {
  stdout: string;
  stderr: string;
  exitCode: number | null;
  timedOut: boolean;
  wallMs: number;
}

export function runClaude(o: RunClaudeOpts): Promise<RunClaudeOutcome> {
  const args = buildClaudeArgs(o);
  const started = Date.now();
  return new Promise((resolve) => {
    const child = spawn(o.bin ?? "claude", args, {
      cwd: o.cwd,
      env: { ...o.env, HOME: o.homeDir },
      // Own process group, so the timeout can reap ratel-local and every upstream
      // stdio server too. Killing only `claude` leaks gateways across 220 cells.
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
    // Signal the whole process group. Killing only `claude` leaks ratel-local
    // and every upstream stdio server, which accumulates across 220 cells.
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
    child.on("close", (code) => {
      clearTimeout(timer);
      resolve({ stdout, stderr, exitCode: code, timedOut, wallMs: Date.now() - started });
    });
    child.on("error", (err) => {
      clearTimeout(timer);
      resolve({
        stdout,
        stderr: `${stderr}\n${(err as Error).message}`,
        exitCode: null,
        timedOut,
        wallMs: Date.now() - started,
      });
    });
  });
}

export function readTranscript(path: string | null): string {
  return path && existsSync(path) ? readFileSync(path, "utf8") : "";
}
