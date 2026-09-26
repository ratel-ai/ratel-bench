// Error classes for an McpAtlasCell, so the native cache never re-serves an
// infra failure and summaries can count them.
//
// A LOCAL COPY of the benchmark taxonomy in agent/src/cell-errors.ts, which
// agent-eval must not import. Keep the two in step. The classes mean the same:
//
//   transient : provider/network/sandbox hiccup (5xx, 429, overload, reset). Re-run it.
//   access    : the model is gated / missing / unauthenticated. Re-run once fixed.
//   request   : the provider rejected the request (other 4xx). Re-runnable.
//   timeout   : the cell hit its deadline. A final, scored outcome.
//   outcome   : the model's own failure (max-turns, context overflow, anything
//               unknown). A final, scored outcome.
//
// Only strings survive into a cell, so this reads the message shapes an
// McpAtlasCell actually carries:
//   - Claude Code (2.1.246) result text: `API Error: <status> …` and
//     `API Error: Request rejected (<status>) · …` (read by status), and its
//     connection wordings (`Unable to connect to API`, `Connection dropped (…)`,
//     `Connection refused …`, `No response from API`, bare `Request timed out` …).
//     Error subtypes carry no `result`: their `errors[]` lines, one per line.
//   - codex-cli (0.153.0) turn failures: `unexpected status <status> …`,
//     `exceeded retry limit, last status: <status> …` (read by status); stream
//     disconnects, high demand / capacity, and the retryable errors codex
//     surfaces once its retries run out (`rate limit exceeded: …`, response-read
//     failures, `request timed out`, agent-loop death, network Io errnos) are
//     transient; `Quota exceeded …` / `hit your usage limit` (access); a raw
//     `invalid_request_error` body (request). Its Json / TokioJoin errors have
//     no stable wording and stay outcome.
//   - The errors runCell throws (finish_reason `error`): `catalog integrity: …`,
//     `ratel cell produced empty telemetry …` (gateway never started), and the
//     `(timedOut=…)` envelope errors — a deadline kill is timeout, an external
//     SIGKILL with timedOut=false (the OOM-killer signature) is transient.
//     Such a row holds no scorable result, so text no rule recognises (spawn
//     failures, ENOSPC, a CLI exit without an envelope) is transient too.
//   - The harness `finish_reason` (`error_max_turns`, `error_timeout`).
//
// Unrecognised text on a harness envelope row is the model's outcome.

import type { McpAtlasCell } from "./mcpatlas-types.js";

export type CellErrorClass = "transient" | "access" | "request" | "timeout" | "outcome";

/** The fields the predicates read; a full McpAtlasCell satisfies it. */
export type CellErrorRow = Pick<McpAtlasCell, "error" | "finish_reason"> & {
  claim_rubric: Pick<McpAtlasCell["claim_rubric"], "judge_error">;
};

// Shared with agent/src/cell-errors.ts.
const CONTEXT_TOO_LONG =
  /prompt is too long|Input is too long|context_length_exceeded|maximum context length/i;
const CONTENT_FILTER = /content[ _-]?filter/i;
const ACCESS = /not available for this account|AWS credential provider failed|model .* not found/i;
const NETWORK = /fetch failed|ECONNRESET|UND_ERR_|other side closed/;
const LEGACY_TRANSIENT =
  /Overloaded|Internal server error|Service Unavailable|Invalid JSON response|Cannot connect to API/i;
const LEGACY_RETRY_LAST = /^Failed after \d+ attempts\. Last error: ([\s\S]*)$/;
const LEGACY_RETRY_NON_RETRYABLE = /non-retryable error: '([\s\S]*)'$/;

// mcpatlas-only shapes.
/** The HTTP status in Claude Code's (`API Error: <s>`, `API Error: Request rejected (<s>)`)
 *  and codex's (`unexpected status <s>`, `exceeded retry limit, last status: <s>`) wording. */
const API_STATUS =
  /(?:API Error(?: \([^)]*\))?: (?:Request rejected \()?|last status: |unexpected status )(\d{3})\b/;
/** Claude Code's wording for a dropped/refused/timed-out connection (no status). */
const HARNESS_TRANSIENT =
  /API Error: (?:Connection error|Request timed out|Unable to connect to API|Connection dropped \(|Connection refused|Can't reach the API server|No internet route|Couldn't connect through your proxy|No response from API|Connection to the API was lost|Connection lost while your computer was asleep|Connection closed before the response finished|Server is temporarily limiting requests)|^Request timed out\.?$/m;
/**
 * codex's wording for a dropped stream, an overloaded provider, and the
 * retryable errors it surfaces once its retries run out. Start-anchored only:
 * exec may append ` (<details>)`. Io errors match network errnos only, so a
 * deterministic local one (`No such file or directory (os error 2)`) stays outcome.
 */
const CODEX_TRANSIENT =
  /stream disconnected before completion|^Reconnecting\.\.\. \d|currently experiencing high demand|model is at capacity|Connection failed:|^rate limit exceeded: |^Error while reading the server response: |^request timed out\b|^timeout waiting for child process to exit|^internal error; agent loop died unexpectedly|(?:Connection reset by peer|Broken pipe|Connection refused|Connection aborted|Network is unreachable|Host is unreachable|Connection timed out) \(os error \d+\)/;
/** codex's billing gate: fixed by the account owner, like the Bedrock account gate. */
const CODEX_ACCESS = /Quota exceeded\. Check your plan|hit your usage limit/;
/** codex surfaces a 400 as the raw OpenAI body, with no status. */
const INVALID_REQUEST = /"type":\s*"invalid_request_error"/;
/** runCell refused the cell: the sandbox or gateway lost servers. No model call was made. */
const CATALOG_INTEGRITY = /^catalog integrity:/;
/** runCell: a ratel cell whose gateway never started (no telemetry at all). */
const EMPTY_TELEMETRY = /^ratel cell produced empty telemetry\b/;
/** runCell's envelope errors record whether the harness was killed at the deadline. */
const HARNESS_TIMED_OUT = /\(timedOut=true\b/;
/** Killed outside the deadline (runClaude/runCodex only kill on timeout): OOM or a container stop. */
const HARNESS_KILLED = /\(timedOut=false, exitCode=null, signal=SIGKILL\)/;
const JUDGE_FAILED = "judge failed:";

/** A cell's error class; null when the agent run did not error. */
export function cellErrorClass(cell: CellErrorRow): CellErrorClass | null {
  if (cell.error == null) return null;
  if (cell.finish_reason === "error_max_turns") return "outcome";
  if (cell.finish_reason === "error_timeout" || HARNESS_TIMED_OUT.test(cell.error)) {
    return "timeout";
  }
  if (
    CATALOG_INTEGRITY.test(cell.error) ||
    EMPTY_TELEMETRY.test(cell.error) ||
    HARNESS_KILLED.test(cell.error)
  ) {
    return "transient";
  }
  // runCell threw (finish_reason "error"): no scorable result came back, so
  // text no rule recognises is a harness failure, not the model's outcome.
  return classifyMessage(cell.error) ?? (cell.finish_reason === "error" ? "transient" : "outcome");
}

/** Infra errors (transient|access) say nothing about the model. */
export function isInfraErrorCell(cell: CellErrorRow): boolean {
  const cls = cellErrorClass(cell);
  return cls === "transient" || cls === "access";
}

/**
 * Whether a prior cell may be served from the cache. Not when its run errored
 * re-runnably (transient|access|request, the `infra` re-run policy), nor when
 * its judge failed: that leaves the cell unscored for a reason unrelated to the
 * agent. A judge that answered but omitted claims, or whose output was
 * truncated (`judge truncated …`), is a final verdict.
 */
export function isReusableCell(cell: CellErrorRow): boolean {
  const cls = cellErrorClass(cell);
  if (cls === "transient" || cls === "access" || cls === "request") return false;
  return !cell.claim_rubric.judge_error?.startsWith(JUDGE_FAILED);
}

// ── helpers ──────────────────────────────────────────────────────────────────

/**
 * agent/src/cell-errors.ts `classifyErrorMessage`, plus Claude Code's and
 * codex's status and wording. Null when no rule matched: the caller picks the
 * fallback by row shape.
 */
function classifyMessage(message: string): CellErrorClass | null {
  // The retry prefix says nothing about the cause: classify what it wraps.
  const wrapped =
    LEGACY_RETRY_LAST.exec(message)?.[1] ?? LEGACY_RETRY_NON_RETRYABLE.exec(message)?.[1];
  if (wrapped !== undefined) return classifyMessage(wrapped);

  if (CONTEXT_TOO_LONG.test(message) || CONTENT_FILTER.test(message)) return "outcome";
  if (ACCESS.test(message) || CODEX_ACCESS.test(message)) return "access";

  const byStatus = classifyStatus(Number(API_STATUS.exec(message)?.[1] ?? 0));
  if (byStatus) return byStatus;

  if (
    LEGACY_TRANSIENT.test(message) ||
    NETWORK.test(message) ||
    HARNESS_TRANSIENT.test(message) ||
    CODEX_TRANSIENT.test(message)
  ) {
    return "transient";
  }
  if (INVALID_REQUEST.test(message)) return "request";
  if (message.startsWith("run timed out after")) return "timeout";
  return null;
}

/** The SDK's `APICallError` rule: retryable statuses are transient, 401/403/404 access, other 4xx request. */
function classifyStatus(status: number): CellErrorClass | null {
  if (status === 408 || status === 409 || status === 429 || status >= 500) return "transient";
  if (status === 401 || status === 403 || status === 404) return "access";
  if (status >= 400 && status < 500) return "request";
  return null;
}
