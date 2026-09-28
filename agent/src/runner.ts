// Drives every (scenario, arm, model, run_index) cell, dispatching the agent
// run through the registry, judging the result, and writing one JSONL row
// per cell.
//
// Resumable: skips cells already present in the output JSONL unless `force` is
// set, except rerunnable errors (`--retry-errors`, default transient|access|
// request), which are re-queued until the cell has spent `--max-attempts` live
// attempts; after the main pass, `--retry-rounds` re-run this run's own
// rerunnable cells in-process (see `rerun.ts`). A per-model breaker
// (`createBreaker`) aborts a gated or failing model: a fatal provider error
// writes no row. A global dollar cap (`dollarGlobalCap`) bounds total spend using
// per-model rates from models.json (via `config.pricing`); a model left unpriced
// reports `dollar_cost=0` and simply isn't bounded by the cap — scenario count
// still bounds it.
//
// Each arm is an `AgentDescriptor` defined in its own file under `agents/`;
// the runner doesn't know how to build tools — it only knows how to schedule
// cells, hand the descriptor an `AgentRunInput`, judge the result, and
// persist it. Registry composition: two control arms statically registered,
// plus every `*.ts` file under `agents/non-control/` (auto-discovered;
// `ignore.*` filenames are gitignored so each developer can drop local-only
// arms next to the committed ones).

import { randomUUID } from "node:crypto";
import {
  appendFileSync,
  existsSync,
  mkdirSync,
  readdirSync,
  readFileSync,
  writeFileSync,
} from "node:fs";
import { dirname } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import type { LanguageModel } from "ai";
import { descriptor as controlBaseline } from "./agents/control-baseline.js";
import { descriptor as controlOracle } from "./agents/control-oracle.js";
import { errorClassOf, FatalProviderError, isRerunnable, type RerunPolicy } from "./cell-errors.js";
import { cellKeyOf, cellKeyString, controlKeyOf, controlKeyString } from "./cell-key.js";
import { loadScenarios } from "./corpus.js";
import { judgeAst } from "./judges/ast.js";
import { judgeLLM } from "./judges/llm.js";
import { judgeProgrammatic } from "./judges/programmatic.js";
import { createBreaker, DEFAULT_RETRY_SETTINGS, type RetrySettings } from "./llm-retry.js";
import { effectiveCalls, type PricingTable } from "./metering.js";
import {
  cacheTier,
  guardResumeCaps,
  type Harness,
  harnessByModel,
  preferCacheRow,
} from "./output-limits.js";
import { buildToolUniverse, expandPool } from "./pool.js";
import {
  capGuardRows,
  countResume,
  DEFAULT_RERUN_SETTINGS,
  type DoneSummary,
  emptyResumeCounts,
  emptyResumePlan,
  newRunTally,
  nextAttempt,
  planResume,
  type RerunSettings,
  type ResumeCounts,
  type ResumePlan,
  requeuedTotal,
  rerunLabel,
  rerunOutcome,
  resumeLine,
  runRounds,
  tallyRow,
} from "./rerun.js";
import { sdkVersion } from "./sdk/resolve.js";
import type {
  AgentDescriptor,
  Arm,
  CellResult,
  PrewarmInput,
  RetrievalMethod,
  RunnerModel,
  Scenario,
  ToolSpec,
} from "./types.js";
import { RATEL_AI_CORE_VERSION } from "./versions.js";

/**
 * Arms whose cell results don't depend on the ratel-specific code path being
 * iterated on (they expose tools directly, with no ranking/dispatch in
 * between). Re-running them per campaign is pure waste once we've measured
 * them at a given ratel version. The runner reads cached rows from a prior
 * canonical run's `agent.jsonl` and emits them into the current output without
 * paying for another live agent loop.
 */
export const CACHEABLE_ARMS: ReadonlySet<Arm> = new Set(["control-baseline", "control-oracle"]);

export interface RunnerConfig {
  corpusPath: string;
  outputPath: string;
  scenarioLimit?: number;
  arms: Arm[];
  models: RunnerModel[];
  runsPerCell: number;
  topK: number;
  /** Retrieval method for the Ratel arms (bm25 | semantic | hybrid). */
  retriever: RetrievalMethod;
  /**
   * Pool sizes to sweep over for non-agnostic arms. Each scenario is evaluated
   * at every value (cells are duplicated across pool sizes), so a 50-scenario ×
   * 4-pool sweep produces 200 cells per (sweep arm, model, run). Pool-size-
   * agnostic arms (e.g. `control-oracle`) ignore this list and emit one cell
   * per (scenario, model, run). Always non-empty; the CLI defaults to `[180]`
   * when neither `--pool-size` nor `--pool-sizes` is passed.
   */
  poolSizes: number[];
  maxSteps: number;
  /** Active-time budget per cell: retry sleeps don't count (see `llm-retry.ts`). */
  perRunTimeoutMs: number;
  /**
   * LLM retry policy + deadline backstop grace (the CLI reads them from
   * `RATEL_LLM_RETRY_*` / `RATEL_CELL_TIMEOUT_GRACE_MS`). Unset = defaults.
   */
  retry?: RetrySettings;
  dollarGlobalCap: number;
  force: boolean;
  /**
   * Which errored rows resume and the in-process retry rounds re-run, how many
   * live attempts a cell gets, and the rounds (`--retry-errors`, `--max-attempts`,
   * `--retry-rounds`, `--retry-delay-s`). Unset = `DEFAULT_RERUN_SETTINGS`.
   */
  rerun?: RerunSettings;
  /**
   * The per-model breaker's streak: abort a model after this many consecutive
   * `transient|access` error rows (`RATEL_ABORT_AFTER_CONSECUTIVE_ERRORS`; 0 =
   * off). A fatal provider error aborts it regardless. Unset = 10.
   */
  abortAfterConsecutiveErrors?: number;
  judgeModel?: LanguageModel;
  /** `--judge-max-output-tokens`: cap on each LLM-judge call. Unset = no cap sent. */
  judgeMaxOutputTokens?: number;
  /** Skip the argument-level (AST) task-completion verdict. Defaults to off (AST on). */
  noAst?: boolean;
  seed: number;
  pricing?: PricingTable;
  /**
   * `quiet`   — only the final summary
   * `normal`  — one line per cell with verdict and error
   * `verbose` — also print the tool-call trace
   */
  logLevel?: "quiet" | "normal" | "verbose";
  /**
   * How many cells run in parallel. Cells are independent (each agent builds
   * its own catalog, fresh agent per call) so the only shared state is the
   * JSONL output and the accumulators — both serialized inside synchronous
   * boundaries. Default 1 preserves the legacy single-threaded ordering for
   * tests; the CLI defaults to 10 because the benchmark is wall-clock-bound
   * on provider latency.
   *
   * The global dollar cap is best-effort under concurrency: when it fires,
   * in-flight cells finish but no new ones start, so overshoot is bounded by
   * `concurrency × per-cell cost`.
   */
  concurrency?: number;
  /** Optional injection point for tests: replaces the real agent dispatch. */
  runCell?: RunCellFn;
  /**
   * Optional pre-built registry. When omitted (the production path), `run()`
   * builds one via `loadAgentRegistry()`. Tests that exercise the runner
   * orchestration without touching real agents inject `runCell` and skip the
   * registry entirely.
   */
  registry?: Map<string, AgentDescriptor>;
  /**
   * Version this run is filed under: stamped on every live and reused row and
   * part of the resume key (`cellKeyOf`). The control cache is version-agnostic,
   * so it ignores this. Defaults to the installed SDK version; `--ratel-version`
   * overrides it for control-only re-drains of a label.
   */
  ratelVersion?: string;
  /**
   * JSONL files to consult for cached control-arm rows (e.g. the canonical
   * `agent.jsonl` plus a controls backfill). The runner reuses the best eligible
   * cacheable-arm row per key across all of them (exact harness tier before
   * legacy, then earliest; re-stamped) instead of re-running the cell. Ephemeral runs point this at the canonical file so
   * iteration on ratel arms doesn't re-pay for controls. When unset or empty,
   * `outputPath` is the source. `force: true` disables the cache.
   */
  cacheSourcePaths?: string[];
  /**
   * Whether the control cache may serve legacy-tier rows (no recorded cap; see
   * `cacheTier`), and resume may keep ones an earlier invocation served. The CLI
   * sets false for any explicit `--max-output-tokens` (N or `none`), so such a run
   * measures the cap it names. Required: no silent default.
   */
  allowLegacyCache: boolean;
}

export type RunCellFn = (args: {
  scenario: Scenario;
  arm: Arm;
  model: RunnerModel;
  runIndex: number;
  pool: ToolSpec[];
  /**
   * Value to write into the row. `null` for pool-size-agnostic arms (whose
   * `pool` is empty and whose row is one-per-scenario regardless of `--pool-sizes`).
   */
  poolSize: number | null;
  config: RunnerConfig;
}) => Promise<CellResult>;

/** The run's counts (see `DoneSummary`: `cells_cached` = control cells served from the cache). */
export interface RunnerSummary extends DoneSummary {
  scenarios: number;
}

/** The output's resumable rows (those carrying a `ratel_version`). */
function readOutputRows(path: string): CellResult[] {
  if (!existsSync(path)) return [];
  const out: CellResult[] = [];
  const text = readFileSync(path, "utf-8");
  for (const line of text.split("\n")) {
    if (!line.trim()) continue;
    try {
      const cell = JSON.parse(line) as CellResult;
      // Pre-`ratel_version` rows have undefined here and never match a current
      // task's key — they're effectively ignored, which is the right behavior
      // (we don't know what code shape produced them).
      if (typeof cell.ratel_version !== "string") continue;
      out.push(cell);
    } catch {
      // Ignore malformed rows; resumability is best-effort.
    }
  }
  return out;
}

/**
 * Resume cap guard: throws when a row the label keeps at this version was produced
 * under a different recorded output cap than this run's, or (under an explicit
 * cap) is a reused legacy row (see `checkResumeCaps`); live rows with no recorded
 * cap only warn. Rows whose cell this run re-queues are exempt (`capGuardRows`):
 * re-running them at this run's cap is how a rejected cap gets fixed.
 */
function guardOutputCaps(
  config: RunnerConfig,
  rows: CellResult[],
  tasks: readonly PendingTask[],
  plan: ResumePlan,
  ratelVersion: string,
): void {
  const requeued = new Set(tasks.filter((t) => plan.requeue.has(t.key)).map((t) => t.key));
  guardResumeCaps(
    capGuardRows(
      rows.filter((r) => r.ratel_version === ratelVersion),
      cellKeyOf,
      requeued,
    ),
    config.models,
    ratelVersion,
    config.allowLegacyCache,
  );
}

/**
 * Index control-arm rows from the source files, keyed by a VERSION-AGNOSTIC key.
 * Control arms don't use Ratel retrieval, so a control cell produced under any
 * version is a valid result for every other version — reused (re-stamped) instead
 * of re-run. Across ALL sources the best harness tier wins, then the
 * earliest-produced cell (the original baseline); missing sources are skipped.
 *
 * Rerunnable errors (transient|access|request) are never eligible: they say
 * nothing about the model, so serving them would re-count an outage as a fail
 * forever. A key whose only rows are such errors is absent and runs live.
 * Timeout/outcome errors are final, scored results and stay reusable, unless
 * `policy` (`--retry-errors all`) re-runs them: then resume re-queues them, and
 * serving one back would undo that.
 *
 * Rows must also match the model's current harness (`harness`: provider +
 * requested output cap, see `cacheTier`): exact-tier rows win over legacy rows
 * (no recorded cap), then earliest; a different recorded provider or cap is
 * never served, nor is a legacy row when `allowLegacy` is false.
 */
function readControlCacheIndex(
  paths: readonly string[],
  harness: ReadonlyMap<string, Harness>,
  allowLegacy: boolean,
  policy: RerunPolicy,
): Map<string, CellResult> {
  const out = new Map<string, CellResult>();
  for (const path of paths) {
    if (!existsSync(path)) continue;
    const text = readFileSync(path, "utf-8");
    for (const line of text.split("\n")) {
      if (!line.trim()) continue;
      try {
        const cell = JSON.parse(line) as CellResult;
        if (typeof cell.ratel_version !== "string") continue;
        if (!CACHEABLE_ARMS.has(cell.arm)) continue;
        if (isRerunnable(cell) || isRerunnable(cell, policy)) continue;
        const key = controlKeyOf(cell);
        const best = preferCacheRow(out.get(key), cell, harness.get(cell.model), allowLegacy);
        if (best) out.set(key, best);
      } catch {
        // Ignore malformed rows.
      }
    }
  }
  return out;
}

/** Control-cache sources: `cacheSourcePaths`, else the output itself. */
function controlSources(config: RunnerConfig): string[] {
  return config.cacheSourcePaths?.length ? config.cacheSourcePaths : [config.outputPath];
}

/**
 * Single-syscall append. Synchronous on purpose: the JS event loop guarantees
 * no two `appendRow` calls interleave even when multiple workers are in flight
 * (each worker awaits the agent, then writes synchronously), so no extra
 * mutex is needed. O(1) per call — important once the JSONL grows past a few
 * thousand rows.
 *
 * Exported for direct testing of the append path; production callers go
 * through `run`.
 */
export function appendRow(path: string, cell: CellResult): void {
  appendFileSync(path, `${JSON.stringify(cell)}\n`, "utf-8");
}

function verdictBadge(cell: CellResult): string {
  if (cell.error != null) return "ERROR";
  if (cell.programmatic_verdict === "pass") return "PASS";
  if (cell.programmatic_verdict === "fail") return "FAIL";
  if (cell.judge_verdict === "pass") return "PASS*";
  if (cell.judge_verdict === "fail") return "FAIL*";
  if (cell.judge_verdict === "partial") return "PART*";
  return "n/a";
}

function logCell(
  cell: CellResult,
  level: "quiet" | "normal" | "verbose",
  done?: number,
  total?: number,
): void {
  if (level === "quiet") return;
  const counter = done !== undefined && total !== undefined ? `[${done}/${total}] ` : "";
  const tag = `${counter}[${cell.scenario_id} · ${cell.arm} · ${cell.model} · #${cell.run_index}]`;
  const verdict = verdictBadge(cell);
  const tokens = `${cell.input_tokens}in/${cell.output_tokens}out`;
  const calls = `${cell.tool_calls_total} calls (${cell.gateway_calls} gw)`;
  const turns = `${cell.turns}t`;
  const finish = cell.finish_reason;
  const cost = `$${cell.dollar_cost.toFixed(4)}`;
  console.log(`${tag} ${verdict.padEnd(5)} ${tokens} ${calls} ${turns} ${finish} ${cost}`);
  if (cell.error != null) {
    console.log(`  ↳ error: ${cell.error}`);
  }
  if (level === "verbose") {
    if (cell.tool_calls.length > 0) {
      console.log(`  ↳ trace:`);
      for (const c of cell.tool_calls) {
        const args = JSON.stringify(c.args);
        const truncated = args.length > 120 ? `${args.slice(0, 117)}...` : args;
        console.log(`     - ${c.toolId}(${truncated})`);
      }
    }
    if (cell.effective_tool_ids.length > 0) {
      console.log(`  ↳ effective: ${cell.effective_tool_ids.join(", ")}`);
    }
    if (cell.final_text) {
      const text =
        cell.final_text.length > 200 ? `${cell.final_text.slice(0, 197)}...` : cell.final_text;
      console.log(`  ↳ final: ${text.replace(/\n/g, " ")}`);
    }
  }
}

/**
 * Build the agent registry: two control arms statically wired, plus every
 * `*.ts` file under `agents/non-control/` (excluding `.test.ts`, `.d.ts`,
 * filenames starting with `_`, and the `ignore.*` rule's targets when the
 * file is gitignored — those still get picked up locally because the rule
 * only filters git, not the filesystem). Each non-control file must
 * `export const descriptor: AgentDescriptor` with a unique `id`.
 *
 * Exported for direct testing of the discovery logic.
 */
export async function loadAgentRegistry(): Promise<Map<string, AgentDescriptor>> {
  const registry = new Map<string, AgentDescriptor>();
  registerDescriptor(registry, controlBaseline, "<static>");
  registerDescriptor(registry, controlOracle, "<static>");

  const moduleUrl = new URL("./agents/non-control/", import.meta.url);
  const dir = fileURLToPath(moduleUrl);
  if (!existsSync(dir)) return registry;

  for (const entry of readdirSync(dir)) {
    if (!isAgentFile(entry)) continue;
    const fileUrl = pathToFileURL(`${dir}${entry}`).href;
    const mod = (await import(fileUrl)) as { descriptor?: AgentDescriptor };
    if (!mod.descriptor) {
      throw new Error(
        `agents/non-control/${entry}: missing \`export const descriptor\` (AgentDescriptor)`,
      );
    }
    registerDescriptor(registry, mod.descriptor, entry);
  }
  return registry;
}

function isAgentFile(name: string): boolean {
  if (!name.endsWith(".ts") && !name.endsWith(".js")) return false;
  if (name.endsWith(".test.ts") || name.endsWith(".test.js")) return false;
  if (name.endsWith(".d.ts")) return false;
  if (name.startsWith("_")) return false;
  return true;
}

function registerDescriptor(
  registry: Map<string, AgentDescriptor>,
  desc: AgentDescriptor,
  source: string,
): void {
  if (!desc.id || !desc.label || typeof desc.run !== "function") {
    throw new Error(
      `${source}: descriptor must have non-empty id+label and a run() function (got ${JSON.stringify(
        { id: desc.id, label: desc.label, hasRun: typeof desc.run === "function" },
      )})`,
    );
  }
  if (registry.has(desc.id)) {
    throw new Error(
      `agent registry: duplicate descriptor id "${desc.id}" (second registration from ${source})`,
    );
  }
  registry.set(desc.id, desc);
}

/**
 * Default cell runner: looks up the descriptor in the registry, runs the agent,
 * then applies programmatic + (optional) LLM judging. Used when the caller
 * doesn't inject a `runCell` in the config.
 */
export function makeRegistryRunCell(
  registry: Map<string, AgentDescriptor>,
  judgeModel?: LanguageModel,
): RunCellFn {
  return async ({ scenario, arm, model, runIndex, pool, poolSize, config }) => {
    const descriptor = registry.get(arm);
    if (!descriptor) {
      throw new Error(`unknown arm "${arm}" — not in agent registry`);
    }

    const cell = await descriptor.run({
      scenario,
      pool,
      poolSize,
      model: { id: model.id, model: model.model, maxOutputTokens: model.maxOutputTokens },
      runIndex,
      topK: config.topK,
      retriever: config.retriever,
      maxSteps: config.maxSteps,
      perRunTimeoutMs: config.perRunTimeoutMs,
      retry: cellRetrySettings(config),
      seed: config.seed,
      pricing: config.pricing,
    });

    const programmatic = judgeProgrammatic(scenario.gold_tools, cell.effective_tool_ids);
    cell.programmatic_verdict = programmatic.verdict;

    // Task-completion (AST) verdict: right function AND right arguments. Computed
    // independently of the selection verdict — a selection pass can still be an
    // argument fail — and LLM-free, so it always runs when the scenario carries
    // argument ground truth (BFCL). `n/a` otherwise (MetaTool/ToolRet).
    if (!config.noAst) {
      cell.ast_verdict = judgeAst(scenario.gold_calls, effectiveCalls(cell.tool_calls)).verdict;
    }

    // An errored cell has no answer to judge (its trace is empty by design), so
    // the judge would only spend money scoring the error. Any non-null error
    // counts, even "" (an APICallError built from an empty statusText).
    if (
      judgeModel &&
      errorClassOf(cell) === null &&
      (programmatic.verdict === "n/a" || programmatic.verdict === "fail")
    ) {
      const judged = await judgeLLM({
        prompt: scenario.prompt,
        judgeCriteria: scenario.judge_criteria,
        finalText: cell.final_text,
        model: judgeModel,
        maxOutputTokens: config.judgeMaxOutputTokens,
      });
      cell.judge_verdict = judged.verdict;
      cell.judge_explanation = judged.explanation;
    }
    return cell;
  };
}

/**
 * The retry settings a cell runs under: tagged with the run's rerun policy (for
 * `retry_policy`), and one log line per retry unless `quiet`.
 */
function cellRetrySettings(config: RunnerConfig): RetrySettings {
  const retry = {
    ...(config.retry ?? DEFAULT_RETRY_SETTINGS),
    rerunLabel: rerunLabel(config.rerun ?? DEFAULT_RERUN_SETTINGS),
  };
  if ((config.logLevel ?? "normal") === "quiet" || retry.log) return retry;
  return { ...retry, log: console.log };
}

/**
 * Seeded shuffle so a `--scenarios N` subset is representative of the full
 * corpus rather than an id-sorted prefix (which on MetaTool would cluster all
 * `metatool-mt-*` rows at the head). Same `seed` reproduces the same subset
 * across runs — important for resume.
 */
function sampleScenarios(
  scenarios: Scenario[],
  limit: number | undefined,
  seed: number,
): Scenario[] {
  if (limit === undefined || limit >= scenarios.length) return scenarios;
  const shuffled = [...scenarios];
  let h = seed >>> 0;
  // Fisher-Yates with a mulberry32 PRNG seeded from `seed`.
  const rng = () => {
    h = (h + 0x6d2b79f5) >>> 0;
    let t = h;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  for (let i = shuffled.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
  }
  return shuffled.slice(0, limit);
}

interface PendingTask {
  scenario: Scenario;
  arm: Arm;
  model: RunnerModel;
  runIndex: number;
  /** Pool passed to the agent. Empty for pool-size-agnostic arms. */
  expandedPool: ToolSpec[];
  /** Value written to `CellResult.pool_size`. `null` for pool-size-agnostic arms. */
  poolSize: number | null;
  /** Pre-computed cell key (used for cache lookup before dispatching the live worker). */
  key: string;
  /** Live attempt this run makes at the cell (prior live attempts + 1); stamped as `attempt`. */
  attempt: number;
}

/**
 * Materializes the full task list ahead of time so the worker pool has a flat
 * queue to consume. Pool expansion is shared across the multiple visits to
 * the same (scenario, pool_size) pair (one per run × arm × model) via a
 * memoized cache. Already-completed cells are filtered out here; re-queued ones
 * (resume) are counted into `resume` and numbered by their next live attempt.
 *
 * Skipping logic: when a registry is available and the descriptor declares
 * `skipForModel(model.id)`, those cells are filtered out at queue-build time
 * (they don't count as "skipped due to resume" and don't write a JSONL row).
 *
 * Pool-size-agnostic arms (e.g. `control-oracle`) skip the `--pool-sizes` loop
 * entirely: each is emitted exactly once per (scenario, model, run) with an
 * empty `expandedPool` and `poolSize: null`. Their result depends only on the
 * scenario, so multiplying them by pool size would burn API spend on duplicate
 * runs and clutter the report with fake pool labels.
 *
 * Iteration order (runs × scenarios × [agnostic arms once] × pool_sizes × non-
 * agnostic arms × models): runs are outermost so a partial budget completes one
 * full pass before starting the next; agnostic arms come first within a
 * scenario so they ship even under tight budgets. Workers pick from the head
 * of the queue, so at concurrency=1 JSONL order is deterministic.
 */
function buildTaskQueue(
  scenarios: Scenario[],
  universe: ReturnType<typeof buildToolUniverse>,
  config: RunnerConfig,
  ratelVersion: string,
  plan: ResumePlan,
  registry: Map<string, AgentDescriptor> | undefined,
): { tasks: PendingTask[]; cellsSkipped: number; resume: ResumeCounts } {
  const poolCache = new Map<string, ToolSpec[]>();
  const expand = (scenario: Scenario, poolSize: number): ToolSpec[] => {
    const key = `${scenario.id}::${poolSize}`;
    let pool = poolCache.get(key);
    if (!pool) {
      pool = expandPool(scenario, universe, poolSize, config.seed);
      poolCache.set(key, pool);
    }
    return pool;
  };

  const isAgnostic = (arm: Arm): boolean => registry?.get(arm)?.poolSizeAgnostic === true;
  const agnosticArms = config.arms.filter(isAgnostic);
  const sweepArms = config.arms.filter((arm) => !isAgnostic(arm));

  const tasks: PendingTask[] = [];
  let cellsSkipped = 0;
  const resume = emptyResumeCounts();
  const tryEnqueue = (
    scenario: Scenario,
    arm: Arm,
    model: RunnerModel,
    runIndex: number,
    expandedPool: ToolSpec[],
    poolSize: number | null,
  ): void => {
    const descriptor = registry?.get(arm);
    if (descriptor?.skipForModel?.(model.id)) return;
    const key = cellKeyString({
      ratelVersion,
      scenarioId: scenario.id,
      arm,
      model: model.id,
      runIndex,
      poolSize,
    });
    countResume(plan, key, resume);
    if (plan.completed.has(key)) {
      cellsSkipped++;
      return;
    }
    tasks.push({
      scenario,
      arm,
      model,
      runIndex,
      expandedPool,
      poolSize,
      key,
      attempt: nextAttempt(plan, key),
    });
  };

  for (let runIndex = 0; runIndex < config.runsPerCell; runIndex++) {
    for (const scenario of scenarios) {
      // Agnostic arms first: one cell each per (scenario, model, run), no pool dim.
      for (const arm of agnosticArms) {
        for (const model of config.models) {
          tryEnqueue(scenario, arm, model, runIndex, [], null);
        }
      }
      for (const poolSize of config.poolSizes) {
        const expandedPool = expand(scenario, poolSize);
        for (const arm of sweepArms) {
          for (const model of config.models) {
            tryEnqueue(scenario, arm, model, runIndex, expandedPool, expandedPool.length);
          }
        }
      }
    }
  }
  return { tasks, cellsSkipped, resume };
}

export async function run(config: RunnerConfig): Promise<RunnerSummary> {
  // Per-run identity stamped on every freshly produced cell so task-completion
  // rows join to retrieval rows (by `scenario_id`) and are scoped to this run.
  const runId = randomUUID();
  const runTimestamp = new Date().toISOString();

  const allScenarios = loadScenarios(config.corpusPath);
  const scenarios = sampleScenarios(allScenarios, config.scenarioLimit, config.seed);

  // Universe is built from the full corpus, not the sampled subset, so smaller
  // runs still have a realistic distractor population to draw from.
  const universe = buildToolUniverse(allScenarios);

  mkdirSync(dirname(config.outputPath), { recursive: true });
  if (config.force && existsSync(config.outputPath)) {
    // Truncate so re-runs don't append duplicates onto previous cells.
    writeFileSync(config.outputPath, "", "utf-8");
  }

  // Build the registry only when the runner needs it: production runs (no
  // injected `runCell`) need it for dispatch; test runs that inject `runCell`
  // skip it entirely so they aren't coupled to the on-disk agent files.
  const registry = config.registry ?? (config.runCell ? undefined : await loadAgentRegistry());
  if (registry && !config.runCell) {
    for (const arm of config.arms) {
      if (!registry.has(arm)) {
        throw new Error(
          `arm "${arm}" not in agent registry. Known: ${[...registry.keys()].join(", ")}`,
        );
      }
    }
  }

  const ratelVersion = config.ratelVersion ?? sdkVersion();
  const rerun = config.rerun ?? DEFAULT_RERUN_SETTINGS;
  const logLevel = config.logLevel ?? "normal";
  // Resume: completed cells are skipped; rerunnable rows with attempts left re-run.
  const priorRows = config.force ? [] : readOutputRows(config.outputPath);
  const plan = config.force ? emptyResumePlan() : planResume(priorRows, cellKeyOf, rerun);
  // Control-arm reuse (default-on): control cells are version-independent, so reuse
  // prior cells (from any version) re-stamped to this run instead of re-running them.
  // Sources default to this run's own output (its prior-version controls); --ephemeral
  // points them at the canonical file. `--force` disables it. Same-version controls are
  // already handled by the `completed` resume set, so there's no double-write.
  const harness = harnessByModel(config.models);
  const allowLegacy = config.allowLegacyCache;
  const cacheIndex = config.force
    ? new Map<string, CellResult>()
    : readControlCacheIndex(controlSources(config), harness, allowLegacy, rerun.policy);

  const {
    tasks,
    cellsSkipped: initialSkipped,
    resume,
  } = buildTaskQueue(scenarios, universe, config, ratelVersion, plan, registry);
  // Before anything is written (the cache drain appends).
  if (!config.force) guardOutputCaps(config, priorRows, tasks, plan, ratelVersion);
  // Printed even under `quiet`, like the final `done:` line: the AWS runner greps it.
  const resumed = resumeLine(resume);
  if (resumed) console.log(resumed);

  // Drain cache hits up front so the worker pool only deals with live cells.
  // Hits are appended in task-iteration order so JSONL ordering matches a fresh
  // live run at the same args.
  let cellsCached = 0;
  let legacyCached = 0;
  const liveTasks: PendingTask[] = [];
  for (const task of tasks) {
    const cached = CACHEABLE_ARMS.has(task.arm)
      ? cacheIndex.get(
          controlKeyString({
            scenarioId: task.scenario.id,
            arm: task.arm,
            model: task.model.id,
            runIndex: task.runIndex,
            poolSize: task.poolSize,
          }),
        )
      : undefined;
    if (cached) {
      // Reuse the version-independent control result, re-stamped to this run.
      appendRow(config.outputPath, {
        ...cached,
        ratel_version: ratelVersion,
        ratel_ai_core_version: RATEL_AI_CORE_VERSION,
        run_id: runId,
        generated_at: runTimestamp,
        cache_source: "reused",
      });
      cellsCached++;
      if (cacheTier(cached, harness.get(cached.model), allowLegacy) === "legacy") legacyCached++;
    } else {
      liveTasks.push(task);
    }
  }

  if (logLevel !== "quiet" && cellsCached > 0) {
    const legacyNote = legacyCached > 0 ? ` (${legacyCached} legacy-tier: no recorded cap)` : "";
    console.error(
      `cache: ${cellsCached} control cells reused${legacyNote} (re-stamped to ${ratelVersion}), ` +
        `${liveTasks.length} will run`,
    );
  }

  const concurrency = Math.max(1, Math.floor(config.concurrency ?? 1));
  const runCellFn = config.runCell ?? makeRegistryRunCell(registry ?? new Map(), config.judgeModel);

  // Serial prewarm pass: let each arm pre-build expensive per-cell state before
  // the concurrent metered loop. Semantic/hybrid embedding is a synchronous
  // native call that blocks the event loop, so building it inside the concurrent
  // phase stalls other in-flight cells' awaited `generate()` and inflates their
  // measured `wall_ms`. Running it here, off the clock, keeps latency honest.
  // Only the production path (registry present, no injected `runCell`) has
  // descriptors to consult; test runners inject `runCell` and skip this.
  if (registry && !config.runCell) {
    for (const arm of config.arms) {
      const desc = registry.get(arm);
      if (!desc?.prepare) continue;
      const seen = new Set<string>();
      const prewarmInputs: PrewarmInput[] = [];
      for (const task of liveTasks) {
        if (task.arm !== arm) continue;
        const dedupKey = `${task.scenario.id}::${task.poolSize}`;
        if (seen.has(dedupKey)) continue;
        seen.add(dedupKey);
        prewarmInputs.push({
          scenario: task.scenario,
          pool: task.expandedPool,
          poolSize: task.poolSize,
          topK: config.topK,
          retriever: config.retriever,
          seed: config.seed,
        });
      }
      if (prewarmInputs.length > 0) {
        if (logLevel !== "quiet") {
          console.error(
            `prewarm: ${arm} building ${prewarmInputs.length} ${config.retriever} ` +
              `catalog(s) before the timed loop`,
          );
        }
        await desc.prepare(prewarmInputs);
      }
    }
  }

  const breaker = createBreaker(config.abortAfterConsecutiveErrors);
  const tally = newRunTally();
  tally.requeued = requeuedTotal(resume);
  tally.exhausted = resume.exhausted;
  let cellsRun = 0;
  let totalDollars = 0;
  let capHit = false;
  let totalToRun = liveTasks.length;

  // Every dollar a cell spent counts toward the cap, including a fatal cell's.
  const spend = (dollars: number): void => {
    totalDollars += dollars;
    if (totalDollars >= config.dollarGlobalCap) capHit = true;
  };

  // One worker-pool pass over `queue`. Returns the tasks whose rows came back
  // rerunnable with attempts left (the next retry round's input).
  const runPass = async (queue: PendingTask[]): Promise<PendingTask[]> => {
    const retryable: PendingTask[] = [];
    let nextTaskIdx = 0;

    // Pick the next runnable task, or `null` if the queue is drained / the
    // global dollar cap has fired. Tasks of an aborted model are skipped.
    // Synchronous; safe to call from any worker because the JS event loop
    // guarantees no preemption between the read and the increment.
    const pickTask = (): PendingTask | null => {
      if (capHit || totalDollars >= config.dollarGlobalCap) {
        capHit = true;
        return null;
      }
      while (nextTaskIdx < queue.length) {
        const task = queue[nextTaskIdx++];
        if (!breaker.isAborted(task.model.id)) return task;
      }
      return null;
    };

    const worker = async (): Promise<void> => {
      while (true) {
        const task = pickTask();
        if (!task) return;
        let cell: CellResult;
        try {
          cell = await runCellFn({
            scenario: task.scenario,
            arm: task.arm,
            model: task.model,
            runIndex: task.runIndex,
            pool: task.expandedPool,
            poolSize: task.poolSize,
            config,
          });
        } catch (err) {
          // A gated / daily-capped model: the cell writes no row (nor do other
          // in-flight cells that meet it), and the model is aborted.
          if (!(err instanceof FatalProviderError)) throw err;
          breaker.fatal(task.model.id, err);
          spend(err.dollarCost);
          continue;
        }
        // Another in-flight cell may have aborted this model while this one ran.
        // Its spend still counts, but an aborted model writes no further rows.
        if (breaker.isAborted(task.model.id)) {
          spend(cell.dollar_cost);
          continue;
        }
        // Tag with this run's identity before persisting (single write path, so
        // every fresh row is stamped; cached/older rows keep their own tags). The
        // version is the one this run measures (and resumes by), not whatever the
        // arm stamped — `--ratel-version` re-stamps control runs this way.
        cell.run_type = "task_completion";
        cell.run_id = runId;
        cell.generated_at = runTimestamp;
        cell.ratel_version = ratelVersion;
        cell.cache_source = "live";
        cell.attempt = task.attempt;
        // Synchronous tail: append + counters happen without yielding, so two
        // workers cannot interleave their writes or accumulator updates.
        appendRow(config.outputPath, cell);
        cellsRun++;
        spend(cell.dollar_cost);
        tallyRow(tally, cell);
        breaker.record(task.model.id, cell);
        logCell(cell, logLevel, cellsRun, totalToRun);
        const outcome = rerunOutcome(cell, task.attempt, rerun);
        if (outcome === "retry") retryable.push(task);
        else if (outcome === "exhausted") tally.exhausted++;
      }
    };

    const workerCount = Math.min(concurrency, Math.max(1, queue.length));
    await Promise.all(Array.from({ length: workerCount }, () => worker()));
    return retryable;
  };

  // Main pass, then up to `rerun.rounds` in-process passes over this run's
  // rerunnable cells: same output (never re-truncated), same dollar cap; an
  // aborted model's cells and a hit cap end them.
  tally.requeued += await runRounds(
    await runPass(liveTasks),
    rerun,
    (queue) => {
      totalToRun += queue.length;
      return runPass(queue);
    },
    (task) => !capHit && !breaker.isAborted(task.model.id),
  );

  return {
    cells_run: cellsRun,
    cells_skipped: initialSkipped,
    cells_cached: cellsCached,
    scenarios: scenarios.length,
    total_dollars: totalDollars,
    stopped_reason: breaker.stopped() ?? (capHit ? "global_cap" : "completed"),
    cap_hit: capHit,
    ...tally,
    aborted: breaker.aborted(),
  };
}
