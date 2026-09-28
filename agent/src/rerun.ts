// Re-running errored cells, shared by `pnpm start` (runner.ts) and
// `sragents-select`:
//
//   - resume: a row rerunnable under `--retry-errors` (default `infra` =
//     transient|access|request) does not complete its cell until the cell has
//     spent `--max-attempts` live attempts (reused cache rows don't count), so a
//     re-run of a label re-queues its outage rows instead of keeping them;
//   - rounds: after the main pass, up to `--retry-rounds` in-process passes
//     re-run this run's rerunnable cells, `--retry-delay-s` apart;
//   - the run's single final `done:` line (plus one `aborted:` line per model the
//     breaker stopped), whose prefix and `$X spent` the AWS runner greps.

import {
  type ErrorClass,
  type ErrorRow,
  errorClassOf,
  isRerunnable,
  type RerunPolicy,
  supersede,
} from "./cell-errors.js";
import { type AbortReason, type ModelAbort, type SleepFn, sleep } from "./llm-retry.js";
import { MAX_TIMER_MS, parseNonNegativeInt } from "./positive-int.js";
import type { SpendSummary } from "./spend-ledger.js";

export interface RerunSettings {
  /** `--retry-errors`: which errored rows resume and retry rounds re-run. */
  policy: RerunPolicy;
  /** `--max-attempts`: live attempts per cell before its rerunnable error is final; 0 = unlimited. */
  maxAttempts: number;
  /** `--retry-rounds`: in-process passes over this run's rerunnable cells after the main pass. */
  rounds: number;
  /** `--retry-delay-s`, in ms: the wait before each round. */
  delayMs: number;
  /** Test seam for the round delay; defaults to a real timer. */
  sleep?: SleepFn;
}

export const DEFAULT_RERUN_SETTINGS: RerunSettings = {
  policy: "infra",
  maxAttempts: 3,
  rounds: 1,
  delayMs: 60_000,
};

/** The CLI flags behind `RerunSettings`, shared by `pnpm start` and `sragents-select`. */
export const RERUN_FLAGS = [
  "--retry-errors",
  "--max-attempts",
  "--retry-rounds",
  "--retry-delay-s",
] as const;

const RERUN_POLICIES: readonly RerunPolicy[] = ["infra", "all", "none"];

const CLASS_ORDER: readonly ErrorClass[] = ["transient", "access", "request", "timeout", "outcome"];

/** A prior output row as `planResume` reads it (its error, and whether it ran live). */
export interface AttemptRow extends ErrorRow {
  cache_source?: "live" | "reused";
}

/** What a resume makes of the output's prior rows, per cell key. */
export interface ResumePlan {
  /** Done: a final row, or rerunnable rows out of attempts (`exhausted`). */
  completed: Set<string>;
  /** Live attempts per key: rows not served from the cache (legacy rows count). */
  attempts: Map<string, number>;
  /** Re-queued keys (only rerunnable rows, attempts left) → the class of the last one. */
  requeue: Map<string, ErrorClass>;
  /** Keys whose rerunnable rows spent `--max-attempts` (part of `completed`). */
  exhausted: Set<string>;
}

/** Resume's effect on one run's cells, for the `resume:` line and the done counts. */
export interface ResumeCounts {
  requeued: Partial<Record<ErrorClass, number>>;
  exhausted: number;
}

/**
 * How a run ended: drained, the dollar cap, or a model abort. Precedence:
 * `fatal` > `error_circuit` > `global_cap` > `completed`; an abort that outranks
 * a cap hit still shows it (`cap_hit`, `stopped=fatal+global_cap`).
 */
export type StopReason = "completed" | "global_cap" | AbortReason;

/** The counts behind the final `done:` line (BFCL `RunnerSummary`, SR `runCampaign`). */
export interface DoneSummary {
  /** Live rows this run wrote: a cell a retry round re-ran counts once per attempt. */
  cells_run: number;
  /** Cells resume skipped: a final row, or a rerunnable one out of attempts. */
  cells_skipped: number;
  /** Control cells served from the cache (re-stamped) instead of running live. */
  cells_cached: number;
  total_dollars: number;
  /** Durable provider attempts; absent from legacy SR-Agents summaries. */
  spend?: SpendSummary;
  /** Live cells whose provider cost and route estimate were both unavailable. */
  unknown_cost_cells?: number;
  stopped_reason: StopReason;
  /** Spend reached the dollar cap (whatever `stopped_reason` ranks first). */
  cap_hit: boolean;
  /** Retries over this run's live rows (`llm-retry.ts`), and those after a 429/503/529. */
  retries: number;
  throttled_retries: number;
  /**
   * This run's live rows that errored (any class), retry-round attempts included:
   * a cell a round recovered still counts its errored attempt. Final failures are
   * `exhausted` plus the non-rerunnable errors.
   */
  errors: number;
  /** Cells re-queued: rerunnable rows resume picked up, plus retry-round cells. */
  requeued: number;
  /** Rerunnable cells out of `--max-attempts`: skipped by resume, or ended this run. */
  exhausted: number;
  /** Models the breaker aborted, and why. */
  aborted: Record<string, ModelAbort>;
}

/** The live-row counters a run accumulates for its `DoneSummary`. */
export type RunTally = Pick<
  DoneSummary,
  "retries" | "throttled_retries" | "errors" | "requeued" | "exhausted"
>;

/** A written live row's rerun fate: `final`, re-run later (`retry`), or out of attempts. */
export type RerunOutcome = "final" | "retry" | "exhausted";

/**
 * Plan a resume over the output's prior rows (`keyOf` = the cell's resume key,
 * version included). A key is done once any row is final under
 * `settings.policy`; a key whose rows are all rerunnable is re-queued while its
 * live attempts (rows with `cache_source !== "reused"`) are below
 * `maxAttempts` (0 = unlimited), else it is done as `exhausted`.
 */
export function planResume<T extends AttemptRow>(
  rows: readonly T[],
  keyOf: (row: T) => string,
  settings: Pick<RerunSettings, "policy" | "maxAttempts">,
): ResumePlan {
  const attempts = new Map<string, number>();
  const final = new Set<string>();
  const lastRerunnable = new Map<string, ErrorClass>();
  for (const row of rows) {
    const key = keyOf(row);
    if (row.cache_source !== "reused") attempts.set(key, (attempts.get(key) ?? 0) + 1);
    const cls = errorClassOf(row);
    if (cls !== null && isRerunnable(row, settings.policy)) lastRerunnable.set(key, cls);
    else final.add(key);
  }
  const plan: ResumePlan = {
    completed: new Set(final),
    attempts,
    requeue: new Map(),
    exhausted: new Set(),
  };
  for (const [key, cls] of lastRerunnable) {
    if (final.has(key)) continue;
    if (outOfAttempts(attempts.get(key) ?? 0, settings.maxAttempts)) {
      plan.completed.add(key);
      plan.exhausted.add(key);
    } else {
      plan.requeue.set(key, cls);
    }
  }
  return plan;
}

/**
 * The prior rows a resume's cap guard (`checkResumeCaps`) must check: per key the
 * row the label keeps (`supersede`), minus a rerunnable one whose cell this run
 * re-queues (`requeued`): its re-run replaces it at this run's cap — the usual
 * fix for a cap the provider rejects (a `request` row per cell). Exhausted keys,
 * keys outside this run and final rows (timeout/outcome re-queued under `all`
 * included: `supersede` keeps them) stay guarded.
 */
export function capGuardRows<T extends ErrorRow>(
  rows: T[],
  keyOf: (row: T) => string,
  requeued: ReadonlySet<string>,
): T[] {
  return supersede(rows, keyOf).filter((r) => !(requeued.has(keyOf(r)) && isRerunnable(r)));
}

/** Count `key` (one of this run's cells) into `counts` when the plan re-queues or exhausted it. */
export function countResume(plan: ResumePlan, key: string, counts: ResumeCounts): void {
  const cls = plan.requeue.get(key);
  if (cls) counts.requeued[cls] = (counts.requeued[cls] ?? 0) + 1;
  if (plan.exhausted.has(key)) counts.exhausted++;
}

/** The live attempt number `key` runs at next (prior live attempts + 1). */
export function nextAttempt(plan: ResumePlan, key: string): number {
  return (plan.attempts.get(key) ?? 0) + 1;
}

/** A plan that resumes nothing (`--force`: the output was truncated). */
export function emptyResumePlan(): ResumePlan {
  return { completed: new Set(), attempts: new Map(), requeue: new Map(), exhausted: new Set() };
}

/** Zeroed resume counts (nothing re-queued or exhausted); a fresh object, `countResume` mutates it. */
export function emptyResumeCounts(): ResumeCounts {
  return { requeued: {}, exhausted: 0 };
}

/** Total cells a resume re-queued. */
export function requeuedTotal(counts: ResumeCounts): number {
  return Object.values(counts.requeued).reduce((a, b) => a + (b ?? 0), 0);
}

/**
 * `resume: K re-queued (transient a, access b, request c); M exhausted`, listing
 * the non-zero classes; null when resume re-queued and exhausted nothing.
 */
export function resumeLine(counts: ResumeCounts): string | null {
  const total = requeuedTotal(counts);
  if (total === 0 && counts.exhausted === 0) return null;
  const byClass = CLASS_ORDER.filter((c) => counts.requeued[c]).map(
    (c) => `${c} ${counts.requeued[c]}`,
  );
  const detail = byClass.length > 0 ? ` (${byClass.join(", ")})` : "";
  return `resume: ${total} re-queued${detail}; ${counts.exhausted} exhausted`;
}

/** Whether a row written at live attempt `attempt` is final, re-runnable, or out of attempts. */
export function rerunOutcome(
  row: ErrorRow,
  attempt: number,
  settings: Pick<RerunSettings, "policy" | "maxAttempts">,
): RerunOutcome {
  if (!isRerunnable(row, settings.policy)) return "final";
  return outOfAttempts(attempt, settings.maxAttempts) ? "exhausted" : "retry";
}

/**
 * The in-process retry rounds. `retryable` is the main pass's rerunnable tasks;
 * each round waits `delayMs`, then re-runs those `canRetry` still allows (model
 * not aborted, cap not hit) at the next attempt through `pass`, which returns
 * the round's own rerunnable tasks. Stops early when none are left. Returns how
 * many cells the rounds re-queued.
 */
export async function runRounds<T extends { attempt: number }>(
  retryable: T[],
  settings: Pick<RerunSettings, "rounds" | "delayMs" | "sleep">,
  pass: (tasks: T[]) => Promise<T[]>,
  canRetry: (task: T) => boolean,
): Promise<number> {
  const wait = settings.sleep ?? sleep;
  let requeued = 0;
  let pending = retryable;
  for (let round = 1; round <= settings.rounds; round++) {
    // Each pass has settled (no cell in flight), so the breaker and cap are final here.
    const next = pending.filter(canRetry).map((task) => ({ ...task, attempt: task.attempt + 1 }));
    if (next.length === 0) break;
    await wait(settings.delayMs);
    requeued += next.length;
    pending = await pass(next);
  }
  return requeued;
}

/** Zeroed live-row counters. */
export function newRunTally(): RunTally {
  return { retries: 0, throttled_retries: 0, errors: 0, requeued: 0, exhausted: 0 };
}

/** Count a live row's retries and error into `tally`. */
export function tallyRow(
  tally: RunTally,
  row: ErrorRow & { retries?: number; throttled_retries?: number },
): void {
  tally.retries += row.retries ?? 0;
  tally.throttled_retries += row.throttled_retries ?? 0;
  if (row.error != null) tally.errors++;
}

/**
 * The run's final line. `bench-run.sh` greps `^done: [0-9]+ cells run` and
 * `, \$([0-9.]+) spent`, and takes the error counts from here: keep both.
 */
export function formatDoneLine(s: DoneSummary): string {
  const accounting =
    s.spend?.completeness === "partial"
      ? ` (known lower bound; ${s.spend.unresolved} unresolved, ${s.spend.unknown} unknown, ${s.spend.untrackedRows ?? 0} untracked rows)`
      : "";
  return (
    `done: ${s.cells_run} cells run, ${s.cells_cached} cached, ${s.cells_skipped} skipped, ` +
    `$${s.total_dollars.toFixed(4)} spent${accounting}, ` +
    ((s.unknown_cost_cells ?? 0) > 0 ? `${s.unknown_cost_cells} unknown-cost cells, ` : "") +
    `stopped=${stoppedLabel(s)}, ` +
    `${s.retries} retries (${s.throttled_retries} throttled), ${s.errors} errors, ` +
    `${s.requeued} re-queued, ${s.exhausted} exhausted`
  );
}

/** The `done:` line, then one `aborted: <model> — <reason>: <detail>` line per aborted model. */
export function doneLines(s: DoneSummary): string[] {
  return [
    formatDoneLine(s),
    ...Object.entries(s.aborted).map(
      ([model, why]) => `aborted: ${model} — ${why.reason}: ${why.detail}`,
    ),
  ];
}

/** The run's exit code: 2 when the breaker aborted a model (after the summary), else 0. */
export function runExitCode(s: Pick<DoneSummary, "aborted">): 0 | 2 {
  return Object.keys(s.aborted).length > 0 ? 2 : 0;
}

/**
 * Apply one of `RERUN_FLAGS` to `settings`, validated: `--retry-errors
 * infra|all|none`, `--max-attempts N` (0 = unlimited), `--retry-rounds N`,
 * `--retry-delay-s S` (a timer delay, so ≤ 2147483 s). Throws naming the flag.
 */
export function applyRerunFlag(settings: RerunSettings, flag: string, raw: string): void {
  switch (flag) {
    case "--retry-errors": {
      const policy = RERUN_POLICIES.find((p) => p === raw.trim());
      if (!policy) throw new Error(`--retry-errors must be infra, all or none (got "${raw}")`);
      settings.policy = policy;
      return;
    }
    case "--max-attempts":
      settings.maxAttempts = parseNonNegativeInt(flag, raw);
      return;
    case "--retry-rounds":
      settings.rounds = parseNonNegativeInt(flag, raw);
      return;
    case "--retry-delay-s": {
      const seconds = parseNonNegativeInt(flag, raw);
      const maxSeconds = Math.floor(MAX_TIMER_MS / 1000);
      if (seconds > maxSeconds) {
        throw new Error(`${flag} must be ≤ ${maxSeconds} s (got "${raw}")`);
      }
      settings.delayMs = seconds * 1000;
      return;
    }
    default:
      throw new Error(`not a rerun flag: ${flag}`);
  }
}

/** The startup echo of the rerun settings. */
export function rerunSettingsLine(settings: RerunSettings): string {
  const attempts = settings.maxAttempts > 0 ? String(settings.maxAttempts) : "unlimited";
  return (
    `rerun: --retry-errors ${settings.policy}, --max-attempts ${attempts}, ` +
    `--retry-rounds ${settings.rounds} (${settings.delayMs / 1000}s apart)`
  );
}

/** The rerun policy as recorded in `retry_policy` (`;rerun=infra/3`; `/0` = unlimited). */
export function rerunLabel(settings: Pick<RerunSettings, "policy" | "maxAttempts">): string {
  return `${settings.policy}/${settings.maxAttempts}`;
}

// ── helpers ──────────────────────────────────────────────────────────────────

/**
 * Whether a cell that spent `liveAttempts` is out of `maxAttempts` (0 = unlimited).
 * A just-written row's attempt number equals the live attempts spent, so resume
 * (prior rows) and the rounds (the current row) share this check.
 */
function outOfAttempts(liveAttempts: number, maxAttempts: number): boolean {
  return maxAttempts > 0 && liveAttempts >= maxAttempts;
}

/** `stopped=`'s value: the reason, plus `+global_cap` when an abort outranked a cap hit. */
function stoppedLabel(s: Pick<DoneSummary, "stopped_reason" | "cap_hit">): string {
  return s.cap_hit && s.stopped_reason !== "global_cap"
    ? `${s.stopped_reason}+global_cap`
    : s.stopped_reason;
}
