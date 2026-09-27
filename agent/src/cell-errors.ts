// Cell error taxonomy: every errored row (live or legacy) gets exactly one class,
// stamped as `error_class`, so retries, re-runs and summaries can tell infra
// failures from the model's own:
//
//   transient : provider/network hiccup (5xx, 429, overload, reset). Re-run it.
//   access    : the model is gated / missing / unauthenticated. Re-run once fixed.
//   request   : the provider rejected the request (other 4xx). Re-runnable.
//   timeout   : the cell hit its deadline. A final, scored outcome.
//   outcome   : the model's own failure (context overflow, content filter,
//               unparseable object, anything unknown). A final, scored outcome.
//
// `classifyError` reads live (structured) errors; `classifyErrorMessage` reads the
// `error` string of legacy rows written before `error_class` was stamped.

import {
  APICallError,
  LoadAPIKeyError,
  NoObjectGeneratedError,
  NoSuchModelError,
  RetryError,
} from "ai";

export type ErrorClass = "transient" | "access" | "request" | "timeout" | "outcome";

/** Which errored rows a re-run picks up: `infra` = transient|access|request. */
export type RerunPolicy = "infra" | "all" | "none";

/**
 * Marker a transport layer (e.g. the Bedrock classifier) sets on an error it has
 * already classified. Contract: `fatal` carriers are non-`APICallError` or have
 * `isRetryable=false`, so the AI SDK never retries them.
 */
export const PROVIDER_ERROR_MARKER = Symbol.for("ratel-bench.provider-error");

export interface ProviderErrorInfo {
  kind: "fatal" | "transient";
  scope: "model";
  reason: string;
}

export interface ClassifyOptions {
  /** Provider retries the cell already spent; a timeout after retries is `transient`. */
  retries?: number;
}

/** The minimal row shape the predicates need (BFCL `CellResult`, SR `SragentsSelectCell`). */
export interface ErrorRow {
  error?: string | null;
  error_class?: ErrorClass;
}

/**
 * The retry wrapper (`llm-retry.ts`) gave up on a transient error: attempts
 * spent, the next wait would pass the wait budget, or a `Retry-After` beyond the
 * cap/budget (`reason`). Not an
 * `APICallError`, so the AI SDK rethrows it unchanged; classified by its cause.
 */
export class RetriesExhaustedError extends Error {
  readonly attempts: number;
  readonly reason?: string;

  constructor(cause: unknown, attempts: number, reason?: string) {
    const why = reason ? ` (${reason})` : "";
    super(`Failed after ${attempts} attempts${why}. Last error: ${messageOf(cause)}`, { cause });
    this.name = "RetriesExhaustedError";
    this.attempts = attempts;
    this.reason = reason;
  }
}

/**
 * The retry wrapper met a fatal provider error (fatal marker or `access`): the
 * model is gated/missing, so no retry can help. Keeps the cause's message; not an
 * `APICallError`, so the AI SDK never retries it. Always `access`.
 */
export class FatalProviderError extends Error {
  constructor(cause: unknown) {
    super(messageOf(cause), { cause });
    this.name = "FatalProviderError";
  }
}

/** A cell's deadline expired. The message is the one legacy rows carry. */
export class CellTimeoutError extends Error {
  readonly timeoutMs: number;

  constructor(timeoutMs: number) {
    super(`run timed out after ${timeoutMs}ms`);
    this.name = "CellTimeoutError";
    this.timeoutMs = timeoutMs;
  }
}

const CONTEXT_TOO_LONG =
  /prompt is too long|Input is too long|context_length_exceeded|maximum context length/i;
const CONTENT_FILTER = /content[ _-]?filter/i;
// Provider quota conditions: retryable 429s that no in-cell backoff can outwait,
// so the model is out as if gated. Bedrock's daily token cap ("Too many tokens
// per day") matches by message; OpenAI's exhausted credit by its structured
// `insufficient_quota` code (`isInsufficientQuota`), since Gemini-compatible
// endpoints reuse its message for per-minute limits. Legacy rows keep only the
// message, so `LEGACY_QUOTA` matches it there.
const ACCESS =
  /not available for this account|AWS credential provider failed|model .* not found|Too many tokens per day/i;
const INSUFFICIENT_QUOTA = "insufficient_quota";
const LEGACY_QUOTA = /exceeded your current quota, please check your plan and billing details/i;
const NETWORK = /fetch failed|ECONNRESET|UND_ERR_|other side closed/;
// Node/undici error codes on a cause link whose message may not say "network".
const NETWORK_CODE = /^(UND_ERR_|ECONNRESET|ECONNREFUSED|EPIPE|ETIMEDOUT)/;
// How far `causeChain` follows `.cause` (the SDK wraps 2-3 deep; bounded for safety).
const MAX_CAUSE_DEPTH = 5;
// A 2xx body that failed the provider's schema: the SDK throws a non-retryable
// APICallError, but it's a provider hiccup, not the model's fault.
const INVALID_JSON_RESPONSE = /^Invalid JSON response/;
const LEGACY_TRANSIENT =
  /Overloaded|Internal server error|Service Unavailable|Invalid JSON response|Cannot connect to API/i;
// `(reason)` is RetriesExhaustedError's fail-fast form.
const LEGACY_RETRY_LAST = /^Failed after \d+ attempts(?: \([^)]*\))?\. Last error: ([\s\S]*)$/;
const LEGACY_RETRY_NON_RETRYABLE = /non-retryable error: '([\s\S]*)'$/;

/** Classify a thrown error. Rules are ordered; the first match wins. */
export function classifyError(err: unknown, opts: ClassifyOptions = {}): ErrorClass {
  // `maxRetriesExceeded` fires for ANY error kind on the last try, so the
  // wrapper says nothing — the last underlying error decides.
  if (RetryError.isInstance(err)) return classifyError(err.lastError, opts);
  // The retry wrapper's own errors: exhaustion says nothing new, the cause decides.
  if (err instanceof RetriesExhaustedError) return classifyError(err.cause, opts);
  if (err instanceof FatalProviderError) return "access";

  const marker = providerErrorInfo(err);
  if (marker) return marker.kind === "fatal" ? "access" : "transient";

  if (isCellTimeout(err)) return (opts.retries ?? 0) > 0 ? "transient" : "timeout";

  if (NoObjectGeneratedError.isInstance(err)) return "outcome";

  const message = messageOf(err);
  const byMessage = classifyByMessage(message);
  if (byMessage) return byMessage;

  if (APICallError.isInstance(err)) {
    if (isInsufficientQuota(err)) return "access";
    if (err.isRetryable || INVALID_JSON_RESPONSE.test(message)) return "transient";
    const status = err.statusCode ?? 0;
    if (status === 401 || status === 403 || status === 404) return "access";
    if (status >= 400 && status < 500) return "request";
  }

  if (LoadAPIKeyError.isInstance(err) || NoSuchModelError.isInstance(err)) return "access";

  // Walks causes: a drop mid-body arrives wrapped as a non-retryable 200 APICallError.
  if (NETWORK.test(message) || causeChain(err).some(isNetworkFailure)) return "transient";

  return "outcome";
}

/** Classify a legacy row's `error` string (no structured error survives JSONL). */
export function classifyErrorMessage(message: string): ErrorClass {
  // The retry prefix says nothing about the cause: classify what it wraps.
  const wrapped =
    LEGACY_RETRY_LAST.exec(message)?.[1] ?? LEGACY_RETRY_NON_RETRYABLE.exec(message)?.[1];
  if (wrapped !== undefined) return classifyErrorMessage(wrapped);

  const byMessage = classifyByMessage(message);
  if (byMessage) return byMessage;
  if (LEGACY_QUOTA.test(message)) return "access";
  if (LEGACY_TRANSIENT.test(message) || NETWORK.test(message)) return "transient";
  if (message.startsWith("run timed out after")) return "timeout";
  return "outcome";
}

/** A row's error class: the stamped one, else derived from the message; null when not errored. */
export function errorClassOf(row: ErrorRow): ErrorClass | null {
  if (row.error == null) return null;
  return row.error_class ?? classifyErrorMessage(row.error);
}

/** Infra errors (transient|access) say nothing about the model; metrics exclude them. */
export function isInfraError(row: ErrorRow): boolean {
  const cls = errorClassOf(row);
  return cls === "transient" || cls === "access";
}

/** Whether a re-run should pick this row up under `policy`. Non-errored rows never are. */
export function isRerunnable(row: ErrorRow, policy: RerunPolicy = "infra"): boolean {
  const cls = errorClassOf(row);
  if (cls === null || policy === "none") return false;
  if (policy === "all") return true;
  return cls === "transient" || cls === "access" || cls === "request";
}

/**
 * One row per key: the last non-rerunnable row (under the default `infra`
 * policy) in file order, otherwise the last row. Output keeps file order.
 */
export function supersede<T extends ErrorRow>(rows: T[], keyOf: (row: T) => string): T[] {
  const chosen = new Map<string, { index: number; final: boolean }>();
  rows.forEach((row, index) => {
    const key = keyOf(row);
    const final = !isRerunnable(row);
    const prev = chosen.get(key);
    if (!prev || final || !prev.final) chosen.set(key, { index, final });
  });
  const keep = new Set([...chosen.values()].map((c) => c.index));
  return rows.filter((_row, index) => keep.has(index));
}

// ── helpers ──────────────────────────────────────────────────────────────────

function classifyByMessage(message: string): ErrorClass | null {
  if (CONTEXT_TOO_LONG.test(message)) return "outcome";
  if (CONTENT_FILTER.test(message)) return "outcome";
  if (ACCESS.test(message)) return "access";
  return null;
}

function providerErrorInfo(err: unknown): ProviderErrorInfo | undefined {
  if (typeof err !== "object" || err === null) return undefined;
  return (err as { [PROVIDER_ERROR_MARKER]?: ProviderErrorInfo })[PROVIDER_ERROR_MARKER];
}

/**
 * A `CellTimeoutError` anywhere on the cause chain: rethrown as-is by
 * `fetch`/`throwIfAborted`, on an `AbortError`'s `cause` (Node carries the abort
 * reason there), or wrapped by the SDK as `APICallError('Failed to process
 * successful response')` when the abort lands mid-body.
 */
function isCellTimeout(err: unknown): boolean {
  return causeChain(err).some((link) => link instanceof CellTimeoutError);
}

/** OpenAI's out-of-credit error: `insufficient_quota` in the parsed body, else the raw one. */
function isInsufficientQuota(err: APICallError): boolean {
  const body = (err.data as { error?: { code?: unknown; type?: unknown } } | undefined)?.error;
  if (body?.code === INSUFFICIENT_QUOTA || body?.type === INSUFFICIENT_QUOTA) return true;
  return err.responseBody?.includes(`"${INSUFFICIENT_QUOTA}"`) ?? false;
}

/** A connection-level failure: a network message or a Node/undici error code. */
function isNetworkFailure(link: unknown): boolean {
  if (NETWORK.test(messageOf(link))) return true;
  const code = (link as { code?: unknown }).code;
  return typeof code === "string" && NETWORK_CODE.test(code);
}

/** `err`, then its `.cause`, `.cause.cause`, …; bounded and cycle-safe. Objects only. */
function causeChain(err: unknown): object[] {
  const chain: object[] = [];
  let link: unknown = err;
  while (
    typeof link === "object" &&
    link !== null &&
    !chain.includes(link) &&
    chain.length <= MAX_CAUSE_DEPTH
  ) {
    chain.push(link);
    link = (link as { cause?: unknown }).cause;
  }
  return chain;
}

function messageOf(err: unknown): string {
  if (err instanceof Error) return err.message;
  return typeof err === "string" ? err : "";
}
