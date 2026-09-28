// `sragents-select` — the LLM skill-selection campaign. The analog of the BFCL
// agent campaign (`pnpm start`), but the task is *selection*: each instance shows
// the model a list of candidate skills and it returns the skill ids it would use.
// We compare that set to the gold set downstream (`sragents-summarize`).
//
// Three arms (the analog of the BFCL arms) decide WHICH candidates the model sees:
//   - control-baseline : the full pool (no retrieval), in neutral (shuffled) order
//   - ratel-full       : only Ratel's top-K retrieved skills (ranked)
//   - control-oracle   : only the gold skills (upper bound)
//
// Skill BM25 is Rust-only, so the candidate lists come from the Rust retrieval
// output (`--candidates`, produced at `--pool-sizes P --top-k K_ratel,P`):
//   retrieved@k=P       → full pool (baseline)   retrieved@k=K_ratel → ratel list
//   golden_answer       → oracle
//
// Pure core (`buildCandidateSets`, `selectForCell`, `planCells`) + a CLI shell that
// meters tokens/cost/latency and appends one cell per (instance, arm, model) to
// agent.jsonl. Resume, retry rounds, the breaker and the final `done:` line work as
// in `pnpm start` (see `rerun.ts`, `createBreaker`).

import { createReadStream, existsSync, mkdirSync } from "node:fs";
import { dirname, join } from "node:path";
import { createInterface } from "node:readline";
import { generateObject, type LanguageModelUsage, NoObjectGeneratedError, RetryError } from "ai";
import { config as loadEnv } from "dotenv";
import { z } from "zod";
import { preflightBedrockModels } from "./bedrock-preflight.js";
import {
  classifyError,
  FatalProviderError,
  isRerunnable,
  type RerunPolicy,
} from "./cell-errors.js";
import { parseOutputCapFlag } from "./cli-args.js";
import { appendJsonl, readJsonl, truncateJsonl } from "./io.js";
import {
  breakerLine,
  breakerThresholdFromEnv,
  cellRetry,
  createBreaker,
  type RetrySettings,
  retrySettingsFromEnv,
  retrySettingsLine,
} from "./llm-retry.js";
import { dollarCost, providerOf } from "./metering.js";
import {
  buildRunnerModels,
  cacheTier,
  capsLine,
  guardResumeCaps,
  type Harness,
  harnessByModel,
  loadModelCatalog,
  type OutputCapOverride,
  preferCacheRow,
} from "./output-limits.js";
import { loadModelPricing } from "./pricing.js";

// Per-model rates from models.json (backend-aware, read once). Empty when
// unpriced → $0 cells and the dollar cap simply doesn't bound the run.
const PRICING = loadModelPricing();

import { DEFAULT_SRAGENTS_MODELS } from "./model-defaults.js";
import { warmUpModels } from "./model-endpoint.js";
import { resolveModel } from "./model-factory.js";
import { canonicalModelList, parseModelIdentity } from "./model-identity.js";
import { resolveRepoPath } from "./paths.js";
import { parseTimerMs } from "./positive-int.js";
import {
  applyRerunFlag,
  capGuardRows,
  countResume,
  DEFAULT_RERUN_SETTINGS,
  type DoneSummary,
  doneLines,
  emptyResumeCounts,
  newRunTally,
  nextAttempt,
  planResume,
  type RERUN_FLAGS,
  type RerunSettings,
  type ResumeCounts,
  requeuedTotal,
  rerunLabel,
  rerunOutcome,
  rerunSettingsLine,
  resumeLine,
  runExitCode,
  runRounds,
  tallyRow,
} from "./rerun.js";
import type { SragentsArm, SragentsRetrievalRow, SragentsSelectCell } from "./sragents-types.js";
import type { RunnerModel } from "./types.js";
import { RATEL_AI_CORE_RESOLVED_VERSION, RATEL_AI_CORE_VERSION } from "./versions.js";

loadEnv(); // pick up agent/.env (provider keys), mirroring cli.ts

const DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434/v1";
const ALL_ARMS: SragentsArm[] = ["control-baseline", "ratel-full", "control-oracle"];
/** `--timeout-ms` default: active time per selection call (retry sleeps don't count). */
const DEFAULT_TIMEOUT_MS = 300_000;
/**
 * `RERUN_FLAGS`, spelled out (the type keeps them in sync): ratel-bench-aws
 * `bench-flags.mjs` passes a flag to SR only when this file holds it as a quoted
 * literal, like `"--timeout-ms"` below.
 */
const SR_RERUN_FLAGS: typeof RERUN_FLAGS = [
  "--retry-errors",
  "--max-attempts",
  "--retry-rounds",
  "--retry-delay-s",
];

/** Control arms don't use Ratel retrieval, so their cells are version-independent
 *  and reusable across ratel versions (re-stamped to the current version). */
const CACHEABLE_ARMS = new Set<SragentsArm>(["control-baseline", "control-oracle"]);

// ── Deterministic shuffle (so the baseline's neutral order doesn't leak BM25 rank) ──

/** FNV-1a-mixed seed → mulberry32 PRNG, keyed by (seed, scenario id). */
function rng(seed: number, key: string): () => number {
  let h = seed >>> 0;
  for (const b of Buffer.from(key)) h = Math.imul(h ^ b, 0x01000193) >>> 0;
  return () => {
    h = (h + 0x6d2b79f5) >>> 0;
    let t = h;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function shuffled<T>(xs: T[], next: () => number): T[] {
  const out = xs.slice();
  for (let i = out.length - 1; i > 0; i--) {
    const j = Math.floor(next() * (i + 1));
    [out[i], out[j]] = [out[j], out[i]];
  }
  return out;
}

// ── Candidate sets per scenario, per arm (pure) ───────────────────────────────

export interface ScenarioCandidates {
  scenarioId: string;
  category: string;
  goldSkillIds: string[];
  /** Full ranked pool (from retrieved@k=poolSize). */
  fullPool: string[];
  /** Ratel's top-K retrieved (from retrieved@k=K_ratel). */
  ratelTopK: string[];
  poolSize: number;
}

/**
 * Collapse the candidate retrieval rows (one scenario × {k=ratelK, k=poolSize}) into
 * one `ScenarioCandidates` per scenario. `poolSize` is the campaign pool `P`; rows
 * at `k=P` carry the full ranked pool, rows at `k=ratelK` carry Ratel's shortlist.
 */
export function buildCandidateSets(
  rows: SragentsRetrievalRow[],
  poolSize: number,
  ratelTopK: number,
): ScenarioCandidates[] {
  const byScenario = new Map<string, SragentsRetrievalRow[]>();
  for (const r of rows) {
    if (r.target_pool_size !== poolSize) continue;
    (byScenario.get(r.scenario_id) ?? byScenario.set(r.scenario_id, []).get(r.scenario_id))?.push(
      r,
    );
  }
  const out: ScenarioCandidates[] = [];
  for (const [scenarioId, rs] of byScenario) {
    const head = rs[0];
    const ids = (k: number) => rs.find((r) => r.k === k)?.retrieved.map((h) => h.id);
    // control-baseline sees the WHOLE pool → use authoritative `pool_ids` (gold-complete),
    // NOT `retrieved@k=poolSize` (the retriever's ranking, which for BM25 drops zero-score
    // gold). Fall back to the ranking only for pre-fix files that lack pool_ids.
    const fullPool = head.pool_ids ?? ids(poolSize);
    const ratel = ids(ratelTopK);
    if (!fullPool || !ratel) continue; // need both k slices to form the A/B
    out.push({
      scenarioId,
      category: head.category ?? "",
      goldSkillIds: head.golden_answer,
      fullPool,
      ratelTopK: ratel,
      poolSize,
    });
  }
  return out;
}

/**
 * Seeded, stratified random sample. Each dataset's scenarios are shuffled with
 * `rng(seed, dataset)`, then we round-robin across datasets and take `n`. This is
 * the SR-Agents analog of BFCL's `sampleScenarios` (a seeded shuffle), but also
 * guarantees per-dataset balance — so a capped `--scenarios N` is representative
 * (not the head of each dataset's file order), balanced, and reproducible for the
 * same seed.
 */
export function stratifiedSample(
  scenarios: ScenarioCandidates[],
  n: number,
  seed: number,
): ScenarioCandidates[] {
  const byDataset = new Map<string, ScenarioCandidates[]>();
  for (const sc of scenarios) {
    (byDataset.get(sc.category) ?? byDataset.set(sc.category, []).get(sc.category))?.push(sc);
  }
  // Shuffle within each dataset (seed mixed with the dataset name so datasets
  // don't share an ordering), then round-robin.
  const queues = [...byDataset.entries()]
    .sort(([a], [b]) => a.localeCompare(b)) // stable dataset order before round-robin
    .map(([dataset, list]) => shuffled(list, rng(seed, dataset)));
  const out: ScenarioCandidates[] = [];
  for (let round = 0; out.length < n; round++) {
    let advanced = false;
    for (const q of queues) {
      if (round < q.length) {
        out.push(q[round]);
        advanced = true;
        if (out.length >= n) break;
      }
    }
    if (!advanced) break; // every queue exhausted
  }
  return out;
}

/** The candidate id list an arm presents, and the `pool_size` recorded on the cell. */
export function armCandidates(
  arm: SragentsArm,
  sc: ScenarioCandidates,
  seed: number,
): { ids: string[]; poolSize: number | null } {
  switch (arm) {
    case "control-baseline":
      return { ids: shuffled(sc.fullPool, rng(seed, sc.scenarioId)), poolSize: sc.poolSize };
    case "ratel-full":
      return { ids: sc.ratelTopK, poolSize: sc.poolSize };
    case "control-oracle":
      return { ids: sc.goldSkillIds, poolSize: null };
  }
}

// ── Prompt + structured output ────────────────────────────────────────────────

// No optional fields: OpenAI strict structured-output requires every property in
// `required`, so the schema is just the selection.
const SelectionSchema = z.object({
  selected_skill_ids: z
    .array(z.string())
    .describe("The exact ids of the skills you would use to solve the task. May be empty."),
});

const SYSTEM = [
  "You are selecting which authored skills are needed to solve a task.",
  "You are shown a numbered catalog of candidate skills (id, name, description).",
  "Return the EXACT ids of every skill required to solve the task — no more, no fewer.",
  "Only return ids that appear in the candidate list. If none apply, return an empty list.",
].join("\n");

function buildPrompt(
  query: string,
  candidates: Array<{ id: string; name: string; description: string }>,
): string {
  const lines = candidates.map((c, i) => `${i + 1}. id=${c.id} — ${c.name}: ${c.description}`);
  return [
    "TASK:",
    query,
    "",
    "CANDIDATE SKILLS:",
    ...lines,
    "",
    "Return the ids of the skills required to solve the TASK.",
  ].join("\n");
}

// ── Catalog (id → name/description), streamed so the 200MB+ body is dropped ────

export async function loadCatalogMeta(
  path: string,
): Promise<Map<string, { name: string; description: string }>> {
  const map = new Map<string, { name: string; description: string }>();
  if (!existsSync(path)) return map;
  const rl = createInterface({ input: createReadStream(path), crlfDelay: Infinity });
  for await (const line of rl) {
    const t = line.trim();
    if (!t) continue;
    const s = JSON.parse(t) as { id: string; name?: string; description?: string };
    map.set(s.id, { name: s.name ?? "", description: s.description ?? "" });
  }
  return map;
}

// ── One cell: prompt the model, meter it ──────────────────────────────────────

export interface SelectArgs {
  arm: SragentsArm;
  sc: ScenarioCandidates;
  query: string;
  model: RunnerModel;
  runIndex: number;
  seed: number;
  catalog: Map<string, { name: string; description: string }>;
  /** `--timeout-ms`: the call's active-time deadline (abortable; retry sleeps don't count). */
  timeoutMs: number;
  /** Retry policy + backstop grace; defaults to `DEFAULT_RETRY_SETTINGS`. */
  retry?: RetrySettings;
}

/**
 * Prompt the model with the arm's candidates and meter the call into a row. The
 * model is wrapped by the retry policy (the SDK's own retries are off) and the
 * call is aborted after `timeoutMs` of active time; the retry counters and the
 * policy are stamped on the row, errored or not. A fatal provider error (gated
 * model, daily cap) writes no row: it is rethrown as a `FatalProviderError` after
 * metering, carrying the call's spend (`dollarCost`).
 */
export async function selectForCell(args: SelectArgs): Promise<SragentsSelectCell> {
  const { ids, poolSize } = armCandidates(args.arm, args.sc, args.seed);
  const candidates = ids
    .map((id) => ({ id, ...(args.catalog.get(id) ?? { name: id, description: "" }) }))
    .filter((c) => c.name || c.description || true);
  const candidateIdSet = new Set(ids);

  const base: SragentsSelectCell = {
    run_type: "skill_selection",
    generated_at: new Date().toISOString(),
    ratel_ai_core_version: RATEL_AI_CORE_VERSION,
    ratel_ai_core_resolved_version: RATEL_AI_CORE_RESOLVED_VERSION,
    scenario_id: args.sc.scenarioId,
    category: args.sc.category,
    arm: args.arm,
    model: args.model.id,
    provider: providerOf(args.model.model),
    run_index: args.runIndex,
    pool_size: poolSize,
    candidate_count: candidates.length,
    gold_skill_ids: args.sc.goldSkillIds,
    selected_skill_ids: [],
    input_tokens: 0,
    output_tokens: 0,
    total_tokens: 0,
    dollar_cost: 0,
    wall_ms: 0,
    error: null,
    cache_source: "live",
    max_output_tokens: args.model.maxOutputTokens,
  };

  const retry = cellRetry(
    args.model.model,
    args.timeoutMs,
    args.retry,
    `${args.sc.scenarioId} · ${args.arm} · ${args.model.id} · #${args.runIndex}`,
  );
  const startedAt = Date.now();
  try {
    const { object, usage, finishReason } = await retry.run(() =>
      generateObject({
        model: retry.model,
        schema: SelectionSchema,
        system: SYSTEM,
        prompt: buildPrompt(args.query, candidates),
        maxOutputTokens: args.model.maxOutputTokens ?? undefined,
        maxRetries: 0, // retries are `withRetry`'s job
        abortSignal: retry.signal,
      }),
    );
    return {
      ...base,
      ...retry.rowFields(),
      // Drop hallucinated ids the model wasn't shown.
      selected_skill_ids: object.selected_skill_ids.filter((id) => candidateIdSet.has(id)),
      ...usageFields(args.model.id, usage),
      finish_reason: finishReason,
      wall_ms: Date.now() - startedAt,
    };
  } catch (err) {
    // An unparseable/invalid object still cost a full call: NoObjectGeneratedError
    // carries its usage and finish reason (`length` = truncated at the output cap:
    // an `outcome` error, final — never re-run or excluded). generateObject raises
    // it itself after doGenerate returns, so it never passes through `withRetry`.
    // With `maxRetries: 0` the SDK never wraps errors in a RetryError; the unwrap
    // is a guard in case SDK retries are re-enabled (only that last call is billed).
    const cause = RetryError.isInstance(err) ? err.lastError : err;
    const noObject = NoObjectGeneratedError.isInstance(cause) ? cause : undefined;
    const row: SragentsSelectCell = {
      ...base,
      ...retry.rowFields(),
      ...(noObject ? usageFields(args.model.id, noObject.usage) : {}),
      finish_reason: noObject?.finishReason ?? "error",
      wall_ms: Date.now() - startedAt,
      error: (err as Error).message ?? String(err),
      // Plan-pinned U1/U4 rule: a timeout in a cell that saw any retry is `transient`.
      // Retry sleeps are off the clock; only failed attempts' own time is the
      // provider's, yet even an instant 429 flips the class.
      error_class: classifyError(err, { retries: retry.stats.retries }),
    };
    if (retry.stats.fatal) {
      const fatal = err instanceof FatalProviderError ? err : new FatalProviderError(err);
      fatal.dollarCost = row.dollar_cost;
      throw fatal;
    }
    return row;
  }
}

/** Token and dollar fields of a cell from one call's usage. */
function usageFields(
  modelId: string,
  usage: LanguageModelUsage | undefined,
): Pick<
  SragentsSelectCell,
  | "input_tokens"
  | "output_tokens"
  | "total_tokens"
  | "cached_input_tokens"
  | "cache_creation_tokens"
  | "dollar_cost"
> {
  const input = usage?.inputTokens ?? 0;
  const output = usage?.outputTokens ?? 0;
  const cachedInput = usage?.inputTokenDetails?.cacheReadTokens ?? usage?.cachedInputTokens ?? 0;
  const cacheCreation = usage?.inputTokenDetails?.cacheWriteTokens ?? 0;
  return {
    input_tokens: input,
    output_tokens: output,
    total_tokens: usage?.totalTokens ?? input + output,
    cached_input_tokens: cachedInput,
    cache_creation_tokens: cacheCreation,
    dollar_cost: dollarCost(modelId, { input, output, cachedInput, cacheCreation }, PRICING),
  };
}

// ── Bounded-concurrency worker pool with a best-effort dollar cap ──────────────

export interface Task {
  arm: SragentsArm;
  sc: ScenarioCandidates;
  query: string;
  model: RunnerModel;
  runIndex: number;
  /** Live attempt at the cell (prior live attempts + 1, see `planCells`); unset = 1. */
  attempt?: number;
}

export interface CampaignOptions {
  concurrency: number;
  dollarCap: number;
  seed: number;
  quiet: boolean;
  onCell: (c: SragentsSelectCell) => void;
  catalog: Map<string, { name: string; description: string }>;
  timeoutMs: number;
  retry: RetrySettings;
  /** Which errored rows the retry rounds re-run, and the rounds themselves. */
  rerun: RerunSettings;
  /** The breaker's streak (`RATEL_ABORT_AFTER_CONSECUTIVE_ERRORS`; 0 = off). */
  abortAfterConsecutiveErrors: number;
}

/** The campaign's share of the `done:` counts (the cache and resume add the rest). */
export type CampaignSummary = Omit<DoneSummary, "cells_cached" | "cells_skipped">;

type QueuedTask = Task & { attempt: number };

/**
 * Run `tasks` on a bounded worker pool under a best-effort dollar cap, stamping
 * each row's `attempt` and handing it to `onCell`. A per-model breaker skips an
 * aborted model's remaining tasks: a `FatalProviderError` cell writes no row
 * (in-flight ones too), and so aborts its model at once, as do K consecutive
 * `transient|access` rows. After the main pass, up to `rerun.rounds` retry
 * rounds re-run the rerunnable cells under the same cap.
 */
export async function runCampaign(tasks: Task[], opts: CampaignOptions): Promise<CampaignSummary> {
  const breaker = createBreaker(opts.abortAfterConsecutiveErrors);
  const retry = { ...opts.retry, rerunLabel: rerunLabel(opts.rerun) };
  const tally = newRunTally();
  let dollars = 0;
  let capHit = false;
  let done = 0;
  let total = tasks.length;

  const spend = (dollarCost: number): void => {
    dollars += dollarCost;
    if (dollars >= opts.dollarCap) capHit = true;
  };

  const runPass = async (queue: QueuedTask[]): Promise<QueuedTask[]> => {
    const retryable: QueuedTask[] = [];
    let i = 0;
    const pick = (): QueuedTask | null => {
      if (dollars >= opts.dollarCap) {
        capHit = true;
        return null;
      }
      while (i < queue.length) {
        const t = queue[i++];
        if (!breaker.isAborted(t.model.id)) return t;
      }
      return null;
    };

    async function worker(): Promise<void> {
      while (true) {
        const t = pick();
        if (!t) return;
        let cell: SragentsSelectCell;
        try {
          cell = await selectForCell({
            ...t,
            seed: opts.seed,
            catalog: opts.catalog,
            timeoutMs: opts.timeoutMs,
            retry,
          });
        } catch (err) {
          if (!(err instanceof FatalProviderError)) throw err;
          breaker.fatal(t.model.id, err);
          spend(err.dollarCost);
          if (!opts.quiet) {
            console.log(
              `${t.sc.scenarioId} ${t.arm}/${t.model.id} → FATAL ${err.message} (no row; model aborted)`,
            );
          }
          continue;
        }
        // A peer cell may have aborted this model while this call was in flight.
        // Charge its spend, but leave the aborted model's output untouched.
        if (breaker.isAborted(t.model.id)) {
          spend(cell.dollar_cost);
          continue;
        }
        cell.attempt = t.attempt;
        spend(cell.dollar_cost);
        opts.onCell(cell);
        done++;
        tallyRow(tally, cell);
        breaker.record(t.model.id, cell);
        const outcome = rerunOutcome(cell, t.attempt, opts.rerun);
        if (outcome === "retry") retryable.push(t);
        else if (outcome === "exhausted") tally.exhausted++;
        if (!opts.quiet) {
          const hit = cell.selected_skill_ids.some((id) => cell.gold_skill_ids.includes(id));
          console.log(
            `[${done}/${total}] ${t.sc.scenarioId} ${t.arm}/${t.model.id} ` +
              `→ ${cell.selected_skill_ids.length} picked, gold=${cell.gold_skill_ids.length} ` +
              `${cell.error ? `ERROR ${cell.error}` : hit ? "hit" : "miss"} ($${dollars.toFixed(3)})`,
          );
        }
      }
    }

    await Promise.all(Array.from({ length: Math.max(1, opts.concurrency) }, () => worker()));
    return retryable;
  };

  tally.requeued += await runRounds(
    await runPass(tasks.map((t) => ({ ...t, attempt: t.attempt ?? 1 }))),
    opts.rerun,
    (queue) => {
      total += queue.length;
      return runPass(queue);
    },
    (t) => !capHit && !breaker.isAborted(t.model.id),
  );

  return {
    cells_run: done,
    total_dollars: dollars,
    stopped_reason: breaker.stopped() ?? (capHit ? "global_cap" : "completed"),
    cap_hit: capHit,
    ...tally,
    aborted: breaker.aborted(),
  };
}

/**
 * The run's `done:` counts: the campaign's, plus what the cache drain served
 * (`reused`), resume skipped, and resume re-queued or exhausted.
 */
export function campaignDoneSummary(
  result: CampaignSummary,
  drained: { reused: number; skipped: number; resume: ResumeCounts },
): DoneSummary {
  return {
    ...result,
    cells_cached: drained.reused,
    cells_skipped: drained.skipped,
    requeued: result.requeued + requeuedTotal(drained.resume),
    exhausted: result.exhausted + drained.resume.exhausted,
  };
}

// ── CLI shell ─────────────────────────────────────────────────────────────────

function arg(name: string, fallback: string): string {
  const idx = process.argv.indexOf(name);
  return idx >= 0 && process.argv[idx + 1] ? process.argv[idx + 1] : fallback;
}

export function sragentsModels(argv: readonly string[]): string[] {
  const index = argv.indexOf("--models");
  if (index < 0) return [...DEFAULT_SRAGENTS_MODELS];
  const raw = argv[index + 1];
  if (raw === undefined || raw.startsWith("--")) throw new Error("--models requires model IDs");
  return canonicalModelList(raw);
}

/** Version-agnostic identity for a control cell (reused across ratel versions). */
export function controlKey(
  scenarioId: string,
  arm: string,
  model: string,
  poolSize: number | null,
  runIndex: number,
): string {
  return `${scenarioId}::${arm}::${model}::${poolSize ?? "null"}::${runIndex}`;
}

/**
 * Index existing control cells for reuse: maps each key to the best cell across
 * all `paths` (harness tier, then earliest: the original baseline).
 * Missing paths are skipped. Resume (what already ran at the current version) is
 * `planCells`' job.
 *
 * Rerunnable errors (transient|access|request) are never indexed — serving
 * them would re-count an outage as a miss forever. Timeout/outcome errors are
 * final and stay reusable, unless `policy` (`--retry-errors all`) re-runs them:
 * then resume re-queues them, and serving one back would undo that.
 *
 * Reuse also matches each model's current `harness` (provider + requested output
 * cap, see `cacheTier`): exact-tier rows beat legacy rows (no recorded cap), then
 * earliest wins; a different recorded provider or cap is never indexed, nor
 * does a legacy row when `allowLegacy` is false. A model absent from `harness`
 * is served legacy rows only.
 */
export function readControlIndex(
  paths: readonly string[],
  harness: ReadonlyMap<string, Harness> = new Map(),
  allowLegacy = true,
  policy: RerunPolicy = "infra",
): Map<string, SragentsSelectCell> {
  const reuse = new Map<string, SragentsSelectCell>();
  for (const path of paths) {
    if (!existsSync(path)) continue;
    for (const c of readJsonl<SragentsSelectCell>(path)) {
      if (!CACHEABLE_ARMS.has(c.arm as SragentsArm)) continue;
      if (isRerunnable(c) || isRerunnable(c, policy)) continue;
      const key = rowKey(c);
      const best = preferCacheRow(reuse.get(key), c, harness.get(c.model), allowLegacy);
      if (best) reuse.set(key, best);
    }
  }
  return reuse;
}

/**
 * Resume over the output's `rows` for EVERY arm (ratel-full included): a task is
 * skipped once its cell has a final row at `version`, or rerunnable rows out of
 * `--max-attempts` (live attempts; reused rows don't count, see `planResume`);
 * otherwise it is pending, numbered by its next live `attempt`. `resume` counts
 * the re-queued and exhausted tasks for the `resume:` line; `requeued` holds the
 * re-queued tasks' keys.
 */
export function planCells(
  tasks: Task[],
  rows: readonly SragentsSelectCell[],
  rerun: Pick<RerunSettings, "policy" | "maxAttempts">,
  version: string = RATEL_AI_CORE_VERSION,
): { pending: Task[]; skipped: number; resume: ResumeCounts; requeued: Set<string> } {
  const plan = planResume(
    rows.filter((c) => c.ratel_ai_core_version === version),
    rowKey,
    rerun,
  );
  const resume = emptyResumeCounts();
  const pending: Task[] = [];
  const requeued = new Set<string>();
  let skipped = 0;
  for (const t of tasks) {
    const key = taskKey(t);
    countResume(plan, key, resume);
    if (plan.completed.has(key)) {
      skipped++;
      continue;
    }
    if (plan.requeue.has(key)) requeued.add(key);
    pending.push({ ...t, attempt: nextAttempt(plan, key) });
  }
  return { pending, skipped, resume, requeued };
}

/**
 * Resume every arm over the output (`planCells`: `skipped` cells, `resume`
 * counts), then serve the pending control cells from the cache and return the
 * tasks that must run live. Within a harness tier the output's own prior controls
 * take precedence and `cachePaths` (best tier, then earliest, across them) only
 * fill gaps; an exact-tier cache row beats the output's own legacy row. A control
 * resume re-queued (its rerunnable error) is served like any other. Reused ones
 * are appended re-stamped to the current version with `cache_source: "reused"`;
 * `legacy` counts those served from the legacy tier. `force` truncates the output
 * and disables resume and reuse, so a forced run never appends a duplicate set.
 * `allowLegacyCache: false` (any explicit `--max-output-tokens`; see
 * `sragentsCapOptions`) serves exact-tier rows only. `rerun` defaults to
 * `DEFAULT_RERUN_SETTINGS`.
 *
 * Throws (before writing anything) when the output holds current-version rows
 * produced under a different recorded output cap, or — with `allowLegacyCache:
 * false` — reused legacy rows an earlier invocation served (`checkResumeCaps`).
 * Rows whose cell this run re-queues are exempt (`capGuardRows`): re-running them
 * at this run's cap is how a rejected cap gets fixed.
 */
export function drainControlCache(
  tasks: Task[],
  opts: {
    outputPath: string;
    cachePaths: string[];
    force: boolean;
    allowLegacyCache: boolean;
    rerun?: Pick<RerunSettings, "policy" | "maxAttempts">;
  },
): { liveTasks: Task[]; reused: number; legacy: number; skipped: number; resume: ResumeCounts } {
  if (opts.force) {
    truncateJsonl(opts.outputPath);
    return { liveTasks: tasks, reused: 0, legacy: 0, skipped: 0, resume: emptyResumeCounts() };
  }
  const rerun = opts.rerun ?? DEFAULT_RERUN_SETTINGS;
  const allowLegacy = opts.allowLegacyCache;
  const current = readJsonl<SragentsSelectCell>(opts.outputPath).filter(
    (c) => c.ratel_ai_core_version === RATEL_AI_CORE_VERSION,
  );
  const { pending, skipped, resume, requeued } = planCells(tasks, current, rerun);
  checkOutputCaps(capGuardRows(current, rowKey, requeued), tasks, allowLegacy);
  const harness = harnessByModel(tasks.map((t) => t.model));
  const tierOf = (c: SragentsSelectCell) => cacheTier(c, harness.get(c.model), allowLegacy);
  const reuseIndex = readControlIndex([opts.outputPath], harness, allowLegacy, rerun.policy);
  const ext = readControlIndex(
    opts.cachePaths.filter((p) => p !== opts.outputPath),
    harness,
    allowLegacy,
    rerun.policy,
  );
  for (const [k, v] of ext) {
    const own = reuseIndex.get(k);
    if (!own || (tierOf(v) === "exact" && tierOf(own) !== "exact")) reuseIndex.set(k, v);
  }

  const liveTasks: Task[] = [];
  let reused = 0;
  let legacy = 0;
  for (const t of pending) {
    if (CACHEABLE_ARMS.has(t.arm)) {
      const prior = reuseIndex.get(taskKey(t));
      if (prior) {
        appendJsonl(opts.outputPath, {
          ...prior,
          ratel_ai_core_version: RATEL_AI_CORE_VERSION,
          ratel_ai_core_resolved_version: RATEL_AI_CORE_RESOLVED_VERSION,
          generated_at: new Date().toISOString(),
          cache_source: "reused",
        });
        reused++;
        if (tierOf(prior) === "legacy") legacy++;
        continue;
      }
    }
    liveTasks.push(t);
  }
  return { liveTasks, reused, legacy, skipped, resume };
}

/** A task's cell identity, as `controlKey` spells it (the oracle arm has no pool). */
function taskKey(t: Task): string {
  const poolForArm = t.arm === "control-oracle" ? null : t.sc.poolSize;
  return controlKey(t.sc.scenarioId, t.arm, t.model.id, poolForArm, t.runIndex);
}

/** A row's cell identity (`controlKey`; version-agnostic, callers filter by version). */
function rowKey(c: SragentsSelectCell): string {
  return controlKey(c.scenario_id, c.arm, c.model, c.pool_size, c.run_index);
}

/**
 * Resume guard: current-version `rows` must share this run's recorded caps (and,
 * without `allowLegacy`, not be reused legacy rows); live legacy ones warn.
 */
function checkOutputCaps(
  rows: readonly SragentsSelectCell[],
  tasks: Task[],
  allowLegacy: boolean,
): void {
  guardResumeCaps(
    rows,
    tasks.map((t) => t.model),
    RATEL_AI_CORE_VERSION,
    allowLegacy,
  );
}

/**
 * The run's cap options from `argv`: `override` is `--max-output-tokens N|none`
 * (undefined when absent), validated here because `arg()` ignores unknown flags
 * and malformed values alike. An explicit override (N or `none`) must measure
 * the cap it names, so it serves exact-tier controls only
 * (`allowLegacyCache: false`). The judge-cap flag is `pnpm start`/`rejudge`-only,
 * so it fails loudly here.
 */
export function sragentsCapOptions(argv: readonly string[]): {
  override: OutputCapOverride | undefined;
  allowLegacyCache: boolean;
} {
  if (argv.includes("--judge-max-output-tokens")) {
    throw new Error("--judge-max-output-tokens: sragents-select has no LLM judge");
  }
  const idx = argv.indexOf("--max-output-tokens");
  const override =
    idx < 0 ? undefined : parseOutputCapFlag("--max-output-tokens", argv[idx + 1] ?? "");
  return { override, allowLegacyCache: override === undefined };
}

/**
 * `--timeout-ms N` (default 300000): each selection call's active-time deadline.
 * Validated here because `arg()` ignores malformed values. Unlike the old
 * per-attempt undici limit it bounds the whole call and aborts it; a cell that
 * hits it is a `timeout` row (`transient` after a retry), recorded in `retry_policy`.
 */
export function sragentsTimeoutMs(argv: readonly string[]): number {
  const idx = argv.indexOf("--timeout-ms");
  return idx < 0 ? DEFAULT_TIMEOUT_MS : parseTimerMs("--timeout-ms", argv[idx + 1] ?? "");
}

/**
 * The rerun flags (`--retry-errors`, `--max-attempts`, `--retry-rounds`,
 * `--retry-delay-s`; defaults `DEFAULT_RERUN_SETTINGS`), validated here because
 * `arg()` ignores malformed values: a bad or missing value throws.
 */
export function sragentsRerunOptions(argv: readonly string[]): RerunSettings {
  const settings = { ...DEFAULT_RERUN_SETTINGS };
  for (const flag of SR_RERUN_FLAGS) {
    const idx = argv.indexOf(flag);
    if (idx >= 0) applyRerunFlag(settings, flag, argv[idx + 1] ?? "");
  }
  return settings;
}

/**
 * Control-cache sources for `--cache-source a,b,…` (`raw`; `undefined` when the
 * flag is absent → the canonical `agent.jsonl` beside the output). An empty
 * list throws instead of silently disabling reuse (every control would run live).
 */
export function sragentsCachePaths(raw: string | undefined, outputPath: string): string[] {
  if (raw === undefined) return [join(dirname(outputPath), "agent.jsonl")];
  const paths = raw
    .split(",")
    .map((p) => p.trim())
    .filter((p) => p.length > 0);
  if (paths.length === 0) throw new Error("--cache-source must list at least one path");
  return paths.map((p) => resolveRepoPath(p));
}

async function main(): Promise<void> {
  // `arg()` ignores unknown flags; this one is `pnpm start`-only (SR rows are
  // labelled via RATEL_VERSION_LABEL), so fail loudly rather than no-op.
  if (process.argv.includes("--ratel-version")) {
    throw new Error(
      "--ratel-version is not supported by sragents-select (use RATEL_VERSION_LABEL)",
    );
  }
  const candidatesPath = resolveRepoPath(
    arg("--candidates", "results/raw/sragents/candidates.jsonl"),
  );
  const catalogPath = resolveRepoPath(arg("--catalog", "test-data/sragents-skills.jsonl"));
  const outputPath = resolveRepoPath(arg("--output", "results/raw/sragents/agent.jsonl"));
  const arms = arg("--arms", ALL_ARMS.join(",")).split(",") as SragentsArm[];
  const models = sragentsModels(process.argv);
  const poolSize = Number(arg("--pool-size", "50"));
  const ratelTopK = Number(arg("--top-k", "10"));
  const scenarioLimit = Number(arg("--scenarios", "0")); // 0 = all
  const runs = Number(arg("--runs", "1"));
  const dollarCap = Number(arg("--dollar-global", "5"));
  const concurrency = Number(arg("--concurrency", "8"));
  const seed = Number(arg("--seed", "42"));
  const quiet = process.argv.includes("--quiet") || process.argv.includes("-q");
  const ollamaBaseURL = process.env.OLLAMA_BASE_URL ?? DEFAULT_OLLAMA_BASE_URL;
  // Bearer token for user-hosted `<url>#<model>` endpoints (optional).
  const modelApiKey = arg("--model-api-key", process.env.AWS_BEDROCK_BEARER ?? "");
  const caps = sragentsCapOptions(process.argv);
  const timeoutMs = sragentsTimeoutMs(process.argv);
  const rerun = sragentsRerunOptions(process.argv);
  // Retry knobs (RATEL_LLM_RETRY_*, RATEL_CELL_TIMEOUT_GRACE_MS) and the breaker's
  // (RATEL_ABORT_AFTER_CONSECUTIVE_ERRORS), validated and echoed once.
  const retry = retrySettingsFromEnv(process.env);
  const abortAfterConsecutiveErrors = breakerThresholdFromEnv(process.env);

  const rows = readJsonl<SragentsRetrievalRow>(candidatesPath);
  if (rows.length === 0) {
    throw new Error(
      `no candidate rows at ${candidatesPath}. Generate them first:\n` +
        `  cargo run -p ratel-benchmark-retrieval --release -- skill-retrieval ` +
        `--output results/raw/sragents/candidates.jsonl --pool-sizes ${poolSize} --top-k ${ratelTopK},${poolSize}`,
    );
  }
  let scenarios = buildCandidateSets(rows, poolSize, ratelTopK);
  // `--scenarios N` caps the sample. The candidates file is dataset-ordered, so a
  // naive slice would be all one dataset — round-robin across datasets instead so
  // small samples still cover every scenario type.
  if (scenarioLimit > 0 && scenarioLimit < scenarios.length) {
    scenarios = stratifiedSample(scenarios, scenarioLimit, seed);
  }

  // Caps are attached after resolution, so no resolver branch can skip them.
  const resolved = buildRunnerModels(
    models,
    (m) => resolveModel(m, { ollamaBaseURL, modelApiKey }),
    {
      catalog: loadModelCatalog(),
      override: caps.override,
    },
  );
  const bedrockIds = models.filter((id) => {
    const identity = parseModelIdentity(id);
    return identity.kind === "provider" && identity.provider === "bedrock";
  });
  if (bedrockIds.length) await preflightBedrockModels(bedrockIds, { requirePricing: true });
  console.log(capsLine(resolved, caps.override));
  console.log(retrySettingsLine(retry, timeoutMs));
  console.log(breakerLine(abortAfterConsecutiveErrors));
  console.log(rerunSettingsLine(rerun));
  // Warm any user-hosted endpoints once before the campaign (no-op for cloud/ollama ids).
  await warmUpModels(models, modelApiKey);
  const catalog = await loadCatalogMeta(catalogPath);

  // Enumerate cells: run × scenario × arm × model.
  const tasks: Task[] = [];
  for (let runIndex = 0; runIndex < runs; runIndex++) {
    for (const sc of scenarios) {
      for (const arm of arms) {
        for (const model of resolved) {
          tasks.push({ arm, sc, query: "", model, runIndex });
        }
      }
    }
  }
  // The prompt query comes from the candidate row's `query`; carry it on the task.
  const queryById = new Map(rows.map((r) => [r.scenario_id, r.query]));
  for (const t of tasks) t.query = queryById.get(t.sc.scenarioId) ?? "";

  mkdirSync(dirname(outputPath), { recursive: true });

  // Control-arm reuse (default-on): control-baseline/control-oracle don't use Ratel,
  // so their cells are version-independent. Reuse prior cells (re-stamped to the
  // current version) instead of re-running them; a model with no prior controls falls
  // through to a live run — the "new model" path that runs all three arms. The cache
  // sources default to the canonical `agent.jsonl` in the output's directory (the 0.2.0
  // results); `--cache-source a,b,…` overrides them. `--force`/`--fresh` truncates the
  // output, disables reuse and re-runs everything.
  const force = process.argv.includes("--force") || process.argv.includes("--fresh");
  const cachePaths = sragentsCachePaths(
    process.argv.includes("--cache-source") ? arg("--cache-source", "") : undefined,
    outputPath,
  );
  const { liveTasks, reused, legacy, skipped, resume } = drainControlCache(tasks, {
    outputPath,
    cachePaths,
    force,
    allowLegacyCache: caps.allowLegacyCache,
    rerun,
  });

  console.log(
    `sragents-select: ${tasks.length} cells ` +
      `(${scenarios.length} scenarios × ${arms.length} arms × ${resolved.length} models × ${runs} runs), ` +
      `pool=${poolSize} ratel-k=${ratelTopK}, cap $${dollarCap}` +
      (reused
        ? ` — reused ${reused} control cells` +
          (legacy ? ` (${legacy} legacy-tier: no recorded cap)` : "") +
          `, ${liveTasks.length} live`
        : ""),
  );
  const resumed = resumeLine(resume);
  if (resumed) console.log(resumed);

  const result = await runCampaign(liveTasks, {
    concurrency,
    dollarCap,
    seed,
    quiet,
    catalog,
    timeoutMs,
    // One line per retry, unless --quiet.
    retry: quiet ? retry : { ...retry, log: console.log },
    rerun,
    abortAfterConsecutiveErrors,
    onCell: (c) => appendJsonl(outputPath, c),
  });

  const capped = result.cap_hit;
  console.log(
    `sragents-select: wrote ${result.cells_run} live + ${reused} reused cells to ` +
      `${arg("--output", "results/raw/sragents/agent.jsonl")} ` +
      `($${result.total_dollars.toFixed(3)}${capped ? `, STOPPED at $${dollarCap} cap` : ""})`,
  );
  // The same `done:` line (and `aborted:` lines) as `pnpm start`, for the AWS runner.
  const summary = campaignDoneSummary(result, { reused, skipped, resume });
  for (const line of doneLines(summary)) console.log(line);
  // A model the breaker aborted (gated, daily cap, outage) fails the run, after the summary.
  const exitCode = runExitCode(summary);
  if (exitCode !== 0) process.exitCode = exitCode;
}

if (import.meta.url === `file://${process.argv[1]}`) {
  main().catch((err) => {
    console.error(`\n✗ ${(err as Error).message}`);
    process.exit(1);
  });
}
