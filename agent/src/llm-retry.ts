// LLM retry wrapper + abortable, pausable cell deadline.
//
// ai@6 retries only 2× with a fixed 2s/4s backoff, far short of a 60s quota
// window, and a racing timeout never aborted the request. Instead, every agent
// model is wrapped per cell (`withRetry`) and called with `maxRetries: 0`:
//
//   - only errors `classifyError` marks `transient` are retried, with equal-jitter
//     capped exponential backoff, honouring `Retry-After` up to the cap/budget;
//   - a fatal provider error (fatal marker or `access`) fails at once as a
//     `FatalProviderError` and sets `stats.fatal`; exhaustion is a
//     `RetriesExhaustedError`. Neither is an `APICallError`, so the SDK rethrows
//     both unchanged; every other error passes through untouched;
//   - the cell's `PausableDeadline` aborts the call after `timeoutMs` of ACTIVE
//     time (it pauses during retry sleeps), with a hard backstop at deadline +
//     grace for layers that ignore the abort;
//   - with a `log`, each retry prints one line (attempt, status, wait).
//
// Retries change a cell's completion probability, not its answer. Judges are not
// wrapped (they pass `maxRetries` to the SDK) and never feed these stats.
//
// Across cells, a per-model `createBreaker` stops a gated or daily-capped model
// from writing rows: a `FatalProviderError` aborts it at once (the cell writes no
// row), and so do K consecutive transport-class (`transient|access`) error rows.

import type { LanguageModel } from "ai";
import {
  CellTimeoutError,
  classifyError,
  type ErrorRow,
  errorClassOf,
  FatalProviderError,
  RetriesExhaustedError,
} from "./cell-errors.js";
import { parseNonNegativeInt, parsePositiveInt, parseTimerMs } from "./positive-int.js";

export interface RetryPolicy {
  /** Calls per `doGenerate` (first try included). */
  maxAttempts: number;
  /** Backoff base: retry n waits around `baseMs·2^(n−1)`. */
  baseMs: number;
  /** Cap on one backoff delay; a longer `Retry-After` fails fast. */
  maxDelayMs: number;
  /** Cap on the total retry wait of one cell (across its calls). */
  maxTotalWaitMs: number;
}

/** Retry counters of one cell, stamped on its row. */
export interface RetryStats {
  retries: number;
  /** Retries of a throttle/overload status (429/503/529). */
  throttledRetries: number;
  /** Total backoff slept, in ms (not charged to the cell's active-time deadline). */
  waitMs: number;
  /** A fatal provider error (gated/missing model) ended a call. */
  fatal: boolean;
}

/** Abortable sleep: rejects with `signal.reason` when the signal fires. */
export type SleepFn = (ms: number, signal?: AbortSignal) => Promise<void>;

/** The deadline surface the wrapper needs: stop the clock while sleeping. */
export interface Pausable {
  pause(): void;
  resume(): void;
}

export interface WithRetryOptions {
  policy: RetryPolicy;
  stats: RetryStats;
  deadline?: Pausable;
  /** Test seam; defaults to a real abortable timer. */
  sleep?: SleepFn;
  /** Jitter source in [0, 1); defaults to `Math.random`. */
  random?: () => number;
  /** Receives one line per retry; unset = silent. */
  log?: (line: string) => void;
}

/** The one call option the wrapper reads. */
interface CallOptions {
  abortSignal?: AbortSignal;
}

/**
 * Per-cell retry + deadline settings: the env knobs, plus the caller's `log`
 * (one line per retry; unset = silent). `sleep`/`random` are test seams.
 */
export interface RetrySettings {
  policy: RetryPolicy;
  /** Backstop grace after the deadline for layers that ignore the abort. */
  graceMs: number;
  sleep?: SleepFn;
  random?: () => number;
  log?: (line: string) => void;
  /** The run's rerun policy (`rerunLabel`), appended to `retry_policy` as `;rerun=…`. */
  rerunLabel?: string;
}

/** The retry fields stamped on a row (BFCL `CellResult`, SR `SragentsSelectCell`). */
export interface RetryRowFields {
  retries: number;
  throttled_retries: number;
  retry_wait_ms: number;
  retry_policy: string;
}

/** One cell's retry plumbing: the wrapped model, its deadline signal and counters. */
export interface CellRetry<M extends LanguageModel> {
  /** The cell's model wrapped by `withRetry`; call it with `maxRetries: 0`. */
  model: M;
  /** Pass as the call's `abortSignal`: aborts after the active-time deadline. */
  signal: AbortSignal;
  stats: RetryStats;
  /** Run the cell's work under the backstop, then release the deadline's timers. */
  run<T>(work: () => Promise<T>): Promise<T>;
  rowFields(): RetryRowFields;
}

/** Why a model was aborted: a fatal provider error, or K consecutive transport errors. */
export type AbortReason = "fatal" | "error_circuit";

export interface ModelAbort {
  reason: AbortReason;
  /** The fatal error's message, or the streak length and its last error. */
  detail: string;
}

/** Per-model circuit breaker over a run's cells (see `createBreaker`). */
export interface Breaker {
  /** A written row's outcome: a transport-class error extends the streak, a success resets it. */
  record(model: string, row: ErrorRow): void;
  /** A cell met a `FatalProviderError`: abort its model now. */
  fatal(model: string, err: unknown): void;
  isAborted(model: string): boolean;
  /** Every aborted model and why (its first abort wins). */
  aborted(): Record<string, ModelAbort>;
  /** The run's stop reason from its aborts (`fatal` over `error_circuit`); undefined when none. */
  stopped(): AbortReason | undefined;
}

export const DEFAULT_RETRY_SETTINGS: RetrySettings = {
  policy: { maxAttempts: 8, baseMs: 2000, maxDelayMs: 60_000, maxTotalWaitMs: 180_000 },
  graceMs: 30_000,
};

/**
 * Env knob → policy field and parser (all positive integers). Every backoff
 * sleep is ≤ `maxDelayMs` (a longer `Retry-After` fails fast), so it alone
 * becomes a timer delay and is capped at Node's timer limit.
 */
const ENV_KNOBS = {
  RATEL_LLM_RETRY_MAX_ATTEMPTS: ["maxAttempts", parsePositiveInt],
  RATEL_LLM_RETRY_BASE_MS: ["baseMs", parsePositiveInt],
  RATEL_LLM_RETRY_MAX_DELAY_MS: ["maxDelayMs", parseTimerMs],
  RATEL_LLM_RETRY_MAX_WAIT_MS: ["maxTotalWaitMs", parsePositiveInt],
} as const satisfies Record<
  string,
  readonly [keyof RetryPolicy, (name: string, raw: string) => number]
>;
/** The backstop grace: a timer delay too. */
const GRACE_KNOB = "RATEL_CELL_TIMEOUT_GRACE_MS";

/** `RATEL_ABORT_AFTER_CONSECUTIVE_ERRORS`: the breaker's streak length (0 = off). */
export const DEFAULT_ABORT_AFTER_CONSECUTIVE_ERRORS = 10;
const BREAKER_KNOB = "RATEL_ABORT_AFTER_CONSECUTIVE_ERRORS";

// Statuses that mean "slow down" rather than "broken" (throttle / overload).
const THROTTLE_STATUSES = new Set([429, 503, 529]);

/**
 * Wrap `model` so its `doGenerate` retries transient errors under `policy`. A
 * Proxy: `specificationVersion`, `provider`, `modelId` and everything else read
 * through, so it works over V2 (anthropic, bedrock) and V3 (openai) models alike.
 * Streaming is not intercepted (the benchmark never streams).
 */
export function withRetry<M extends LanguageModel>(model: M, opts: WithRetryOptions): M {
  if (typeof model === "string") {
    throw new TypeError(`withRetry needs a model instance, got the id "${model}"`);
  }
  type Generate = (options: CallOptions) => Promise<unknown>;
  const target = model as unknown as { doGenerate: Generate } & Record<PropertyKey, unknown>;
  const name = `${String(target.provider)}/${String(target.modelId)}`;
  const doGenerate: Generate = (options) =>
    generateWithRetry(() => target.doGenerate(options), options.abortSignal, name, opts);
  return new Proxy(target, {
    get(obj, prop) {
      if (prop === "doGenerate") return doGenerate;
      const value = Reflect.get(obj, prop, obj);
      // Bind so provider methods keep their own `this` (private fields).
      return typeof value === "function" ? value.bind(obj) : value;
    },
  }) as unknown as M;
}

/**
 * Set up one cell: a `PausableDeadline` of `timeoutMs` active time (starting
 * now) and `model` wrapped by `withRetry` under `settings`, pausing that deadline
 * while it sleeps. `label` (the cell's identity) prefixes its retry log lines,
 * since cells run concurrently. Shared by the BFCL agent loop and the SR
 * selection call.
 */
export function cellRetry<M extends LanguageModel>(
  model: M,
  timeoutMs: number,
  settings: RetrySettings = DEFAULT_RETRY_SETTINGS,
  label?: string,
): CellRetry<M> {
  const stats = newRetryStats();
  const deadline = new PausableDeadline(timeoutMs, { graceMs: settings.graceMs });
  const policy = settings.policy;
  const log = settings.log;
  return {
    model: withRetry(model, {
      policy,
      stats,
      deadline,
      sleep: settings.sleep,
      random: settings.random,
      log: log && label !== undefined ? (line) => log(`[${label}] ${line}`) : log,
    }),
    signal: deadline.signal,
    stats,
    async run(work) {
      try {
        return await deadline.race(work());
      } finally {
        deadline.dispose();
      }
    },
    rowFields: () => ({
      retries: stats.retries,
      throttled_retries: stats.throttledRetries,
      retry_wait_ms: Math.round(stats.waitMs),
      retry_policy:
        retryPolicyLabel(policy, timeoutMs, settings.graceMs) +
        (settings.rerunLabel ? `;rerun=${settings.rerunLabel}` : ""),
    }),
  };
}

export function newRetryStats(): RetryStats {
  return { retries: 0, throttledRetries: 0, waitMs: 0, fatal: false };
}

/**
 * A cell deadline over ACTIVE time: `signal` aborts with a `CellTimeoutError`
 * once `timeoutMs` have elapsed outside `pause()`/`resume()` spans (retry
 * sleeps). `race()` adds the hard backstop: it rejects with the same error
 * `graceMs` after the abort, even when the raced work ignores the signal.
 * Active time is read off the monotonic `performance.now()` (as timers are), so
 * a wall-clock step can't move the deadline.
 */
export class PausableDeadline implements Pausable {
  readonly signal: AbortSignal;
  private readonly controller = new AbortController();
  private readonly timeoutMs: number;
  private readonly graceMs: number;
  private remainingMs: number;
  private startedAt = 0;
  private paused = 0;
  private timer: ReturnType<typeof setTimeout> | undefined;
  private graceTimer: ReturnType<typeof setTimeout> | undefined;
  private readonly backstop: Promise<never>;
  private rejectBackstop: (err: unknown) => void = () => {};

  constructor(timeoutMs: number, opts: { graceMs: number }) {
    this.timeoutMs = timeoutMs;
    this.graceMs = opts.graceMs;
    this.remainingMs = timeoutMs;
    this.signal = this.controller.signal;
    this.backstop = new Promise<never>((_resolve, reject) => {
      this.rejectBackstop = reject;
    });
    this.backstop.catch(() => {}); // observed via race(); never an unhandled rejection
    this.start();
  }

  pause(): void {
    if (this.paused++ > 0 || this.signal.aborted) return;
    clearTimeout(this.timer);
    this.remainingMs -= performance.now() - this.startedAt;
  }

  resume(): void {
    if (this.paused === 0 || --this.paused > 0 || this.signal.aborted) return;
    this.start();
  }

  /** `work`, or a `CellTimeoutError` at deadline + grace, whichever settles first. */
  race<T>(work: Promise<T>): Promise<T> {
    return Promise.race([work, this.backstop]);
  }

  dispose(): void {
    clearTimeout(this.timer);
    clearTimeout(this.graceTimer);
  }

  private start(): void {
    this.startedAt = performance.now();
    this.timer = setTimeout(() => this.expire(), Math.max(0, this.remainingMs));
  }

  private expire(): void {
    const reason = new CellTimeoutError(this.timeoutMs);
    this.controller.abort(reason);
    this.graceTimer = setTimeout(() => this.rejectBackstop(reason), this.graceMs);
  }
}

/**
 * A per-model breaker. `fatal` aborts a model at once. `record` counts the
 * written rows' transport-class errors (`transient|access`, stamped or derived
 * from the message) per model, in completion order: `threshold` in a row abort
 * it (`error_circuit`), a success resets the streak, and every other class
 * (`timeout|request|outcome`: the provider answered) leaves it as is. A
 * `threshold` of 0 turns the streak check off; fatal errors still abort.
 */
export function createBreaker(threshold = DEFAULT_ABORT_AFTER_CONSECUTIVE_ERRORS): Breaker {
  const streaks = new Map<string, number>();
  const aborts = new Map<string, ModelAbort>();
  const abort = (model: string, why: ModelAbort): void => {
    if (!aborts.has(model)) aborts.set(model, why);
  };
  return {
    record(model, row) {
      const cls = errorClassOf(row);
      if (cls === null) {
        streaks.set(model, 0);
        return;
      }
      if (cls !== "transient" && cls !== "access") return;
      const streak = (streaks.get(model) ?? 0) + 1;
      streaks.set(model, streak);
      if (threshold > 0 && streak >= threshold) {
        abort(model, {
          reason: "error_circuit",
          detail: `${streak} consecutive transient/access errors (last: ${row.error})`,
        });
      }
    },
    fatal(model, err) {
      abort(model, { reason: "fatal", detail: err instanceof Error ? err.message : String(err) });
    },
    isAborted: (model) => aborts.has(model),
    aborted: () => Object.fromEntries(aborts),
    stopped() {
      const reasons = [...aborts.values()].map((a) => a.reason);
      if (reasons.includes("fatal")) return "fatal";
      return reasons.length > 0 ? "error_circuit" : undefined;
    },
  };
}

/**
 * The `retry_policy` stamped on each row: attempts/base/cap/wait budget, then the
 * active-time deadline and its backstop grace (`a8/b2000/c60000/w180000;timeout=active:180000+g30000`).
 */
export function retryPolicyLabel(policy: RetryPolicy, timeoutMs: number, graceMs: number): string {
  return (
    `a${policy.maxAttempts}/b${policy.baseMs}/c${policy.maxDelayMs}/w${policy.maxTotalWaitMs}` +
    `;timeout=active:${timeoutMs}+g${graceMs}`
  );
}

/** A real timer that rejects with `signal.reason` as soon as the signal fires. */
export function sleep(ms: number, signal?: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    if (signal?.aborted) {
      reject(signal.reason);
      return;
    }
    const onAbort = () => {
      clearTimeout(timer);
      reject(signal?.reason);
    };
    const timer = setTimeout(() => {
      signal?.removeEventListener("abort", onAbort);
      resolve();
    }, ms);
    signal?.addEventListener("abort", onAbort, { once: true });
  });
}

/** Equal jitter over a capped exponential: d/2 + rand·d/2, d = min(cap, base·2^(retry−1)). */
export function retryDelayMs(
  retry: number,
  policy: RetryPolicy,
  random: () => number = Math.random,
): number {
  const d = Math.min(policy.maxDelayMs, policy.baseMs * 2 ** (retry - 1));
  return d / 2 + (random() * d) / 2;
}

/**
 * The retry knobs from the environment (defaults: `DEFAULT_RETRY_SETTINGS`).
 * Throws on a value that isn't a positive integer (or, for a timer delay, is
 * over Node's timer limit), naming the variable.
 */
export function retrySettingsFromEnv(env: NodeJS.ProcessEnv): RetrySettings {
  const policy = { ...DEFAULT_RETRY_SETTINGS.policy };
  for (const [name, [field, parse]] of Object.entries(ENV_KNOBS)) {
    const raw = env[name];
    if (raw !== undefined) policy[field] = parse(name, raw);
  }
  const grace = env[GRACE_KNOB];
  const graceMs =
    grace === undefined ? DEFAULT_RETRY_SETTINGS.graceMs : parseTimerMs(GRACE_KNOB, grace);
  return { policy, graceMs };
}

/** The startup echo of the retry knobs and the cell's active-time deadline. */
export function retrySettingsLine(settings: RetrySettings, timeoutMs: number): string {
  const p = settings.policy;
  return (
    `retry: attempts=${p.maxAttempts} base=${p.baseMs}ms max-delay=${p.maxDelayMs}ms ` +
    `max-wait=${p.maxTotalWaitMs}ms; timeout=${timeoutMs}ms active + ${settings.graceMs}ms grace`
  );
}

/**
 * `RATEL_ABORT_AFTER_CONSECUTIVE_ERRORS` (default 10; 0 = off): the breaker's
 * streak length. Throws on anything but a non-negative integer.
 */
export function breakerThresholdFromEnv(env: NodeJS.ProcessEnv): number {
  const raw = env[BREAKER_KNOB];
  return raw === undefined
    ? DEFAULT_ABORT_AFTER_CONSECUTIVE_ERRORS
    : parseNonNegativeInt(BREAKER_KNOB, raw);
}

/** The startup echo of the breaker knob. */
export function breakerLine(threshold: number): string {
  const streak =
    threshold > 0
      ? `after ${threshold} consecutive transient/access errors`
      : "streak check off (RATEL_ABORT_AFTER_CONSECUTIVE_ERRORS=0)";
  return `breaker: abort a model on a fatal provider error, or ${streak}`;
}

// ── helpers ──────────────────────────────────────────────────────────────────

async function generateWithRetry<T>(
  call: () => Promise<T>,
  signal: AbortSignal | undefined,
  name: string,
  opts: WithRetryOptions,
): Promise<T> {
  const { policy, stats, deadline } = opts;
  const sleepFn = opts.sleep ?? sleep;
  for (let attempt = 1; ; attempt++) {
    try {
      return await call();
    } catch (err) {
      // The deadline (or a caller) aborted: never retry, report as-is.
      if (signal?.aborted) throw err;
      const cls = classifyError(err);
      if (cls === "access") {
        stats.fatal = true;
        throw new FatalProviderError(err);
      }
      if (cls !== "transient") throw err;
      const { delay, byHeader } = nextDelay(err, attempt, opts);
      const status = statusOf(err);
      stats.retries++;
      if (THROTTLE_STATUSES.has(status ?? 0)) stats.throttledRetries++;
      opts.log?.(
        `retry: ${name} attempt ${attempt}/${policy.maxAttempts} failed ` +
          `(${status ?? (err as Error)?.name ?? "error"}); waiting ${Math.round(delay)}ms` +
          `${byHeader ? " (Retry-After)" : ""}, ` +
          `${Math.round(stats.waitMs + delay)}/${policy.maxTotalWaitMs}ms of wait budget used`,
      );
      deadline?.pause();
      try {
        await sleepFn(delay, signal);
      } finally {
        deadline?.resume();
      }
      stats.waitMs += delay;
    }
  }
}

/**
 * Backoff before retry `attempt` (`byHeader`: a `Retry-After` set it), or throws
 * `RetriesExhaustedError` when out of attempts or when the wait would pass the cap
 * or the budget. Reasons never contain `)`: legacy parsing relies on it.
 */
function nextDelay(
  err: unknown,
  attempt: number,
  opts: WithRetryOptions,
): { delay: number; byHeader: boolean } {
  const { policy, stats } = opts;
  if (attempt >= policy.maxAttempts) throw new RetriesExhaustedError(err, attempt);
  const remaining = policy.maxTotalWaitMs - stats.waitMs;
  const header = retryAfterMs(err);
  if (header !== undefined && header > Math.min(policy.maxDelayMs, remaining)) {
    const limit =
      header > policy.maxDelayMs
        ? `max delay ${policy.maxDelayMs}ms`
        : `${Math.round(remaining)}ms wait left`;
    throw new RetriesExhaustedError(err, attempt, `Retry-After ${Math.round(header)}ms > ${limit}`);
  }
  const jittered = retryDelayMs(attempt, policy, opts.random);
  const delay = Math.max(jittered, header ?? 0);
  if (delay > remaining) {
    const next = `next wait ${Math.round(delay)}ms > ${Math.round(remaining)}ms left`;
    throw new RetriesExhaustedError(
      err,
      attempt,
      `retry wait budget ${policy.maxTotalWaitMs}ms: ${next}`,
    );
  }
  return { delay, byHeader: header !== undefined && header > jittered };
}

/** `retry-after-ms` (ms) or `retry-after` (seconds or an HTTP date) from the response headers. */
function retryAfterMs(err: unknown): number | undefined {
  if (typeof err !== "object" || err === null) return undefined;
  const headers = (err as { responseHeaders?: Record<string, string> }).responseHeaders;
  if (!headers) return undefined;
  const ms = Number.parseFloat(headers["retry-after-ms"] ?? "");
  if (Number.isFinite(ms) && ms >= 0) return ms;
  const raw = headers["retry-after"];
  if (raw === undefined) return undefined;
  const seconds = Number.parseFloat(raw);
  if (Number.isFinite(seconds) && seconds >= 0) return seconds * 1000;
  const at = Date.parse(raw);
  return Number.isNaN(at) ? undefined : Math.max(0, at - Date.now());
}

/** HTTP status of an error (`APICallError.statusCode` or the same field on a transport error). */
function statusOf(err: unknown): number | undefined {
  if (typeof err !== "object" || err === null) return undefined;
  const status = (err as { statusCode?: unknown }).statusCode;
  return typeof status === "number" ? status : undefined;
}
