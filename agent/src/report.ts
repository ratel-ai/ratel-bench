// Aggregator. Joins agent.jsonl (cells from runs) with retrieval.jsonl (BM25
// metrics from the Rust layer) and emits REPORT.md.
//
// Pure functions on parsed JSONL — no I/O — so the report logic stays
// testable. The CLI wrapper (`report-cli.ts`) handles file reads/writes.

import { errorClassOf, isInfraError, supersede } from "./cell-errors.js";
import { cellKeyOf } from "./cell-key.js";
import type { Arm, CellResult } from "./types.js";

export interface RetrievalRow {
  scenario_id: string;
  /**
   * Scenario category from the ingest adapter (`metatool-single` /
   * `metatool-multi` / `metatool-skill`). Drives the `(subset, mode)` split;
   * absent for category-less corpora (e.g. ToolRet), which fall back to
   * gold-set size.
   */
  category?: string;
  target_pool_size: number;
  actual_pool_size: number;
  /** BM25 engine version (= `ratel-ai-core` crate) this row was scored with. */
  ratel_ai_core_version?: string;
  k: number;
  pool_size: number;
  gold_count: number;
  recall_at_k: number;
  precision_at_k: number;
  reciprocal_rank: number;
  hit_at_k: boolean;
  /** True if every gold item is in the top-K (recall == 1.0). Strict sibling of
   *  hit_at_k; absent in older rows (falls back to recall_at_k >= 1). */
  complete_at_k?: boolean;
  ndcg_at_k: number;
}

/**
 * A cell's identity within one label's history: the label (`ratel_ai_core_version`)
 * plus the runner's cell key. The label matters because one SDK version (in the
 * runner key) backs several labels, e.g. `0.2.0` and `0.3.0-rc.1`.
 */
export function labelledCellKeyOf(cell: CellResult): string {
  return `${cell.ratel_ai_core_version ?? "unknown"}::${cellKeyOf(cell)}`;
}

/** The row fields `versionSplitCells` reads; BFCL cells and audit rows carry them. */
export interface VersionedCellRow {
  scenario_id: string;
  arm: string;
  model: string;
  run_index: number;
  pool_size: number | null;
  ratel_ai_core_version?: string;
  ratel_version?: string;
}

/** A label cell whose rows carry more than one SDK `ratel_version`. */
export interface VersionSplitCell {
  label: string;
  arm: string;
  model: string;
  scenario_id: string;
  /** Row count per `ratel_version` (`unset` when not recorded). */
  versions: Record<string, number>;
}

/**
 * Label cells (label + scenario/arm/model/run/pool) whose rows span more than one
 * `ratel_version`. `labelledCellKeyOf` includes the SDK version, so each version
 * counts as its own cell: a re-drain stamped with another `--ratel-version` than
 * the rows it replaces double-counts the cell instead of superseding it.
 */
export function versionSplitCells(rows: VersionedCellRow[]): VersionSplitCell[] {
  const cells = new Map<string, VersionSplitCell>();
  for (const r of rows) {
    const label = r.ratel_ai_core_version ?? "unknown";
    const key = [label, r.scenario_id, r.arm, r.model, r.run_index, r.pool_size ?? "null"].join(
      "::",
    );
    const cell =
      cells.get(key) ??
      cells
        .set(key, { label, arm: r.arm, model: r.model, scenario_id: r.scenario_id, versions: {} })
        .get(key);
    if (!cell) continue;
    const version = r.ratel_version ?? "unset";
    cell.versions[version] = (cell.versions[version] ?? 0) + 1;
  }
  return [...cells.values()].filter((c) => Object.keys(c.versions).length > 1);
}

export function median(xs: number[]): number {
  if (xs.length === 0) return 0;
  const sorted = [...xs].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 0 ? (sorted[mid - 1] + sorted[mid]) / 2 : sorted[mid];
}

export function mean(xs: number[]): number {
  if (xs.length === 0) return 0;
  return xs.reduce((a, b) => a + b, 0) / xs.length;
}

/** Mean, or null for an empty list (a metric with no rows behind it). */
export function meanOrNull(xs: number[]): number | null {
  return xs.length === 0 ? null : mean(xs);
}

/** Median, or null for an empty list. */
export function medianOrNull(xs: number[]): number | null {
  return xs.length === 0 ? null : median(xs);
}

/**
 * A row cut off by the output-token limit: `truncated_steps>0` (BFCL, U1+) or a
 * `length` finish reason (legacy BFCL rows, SR rows). Scored on its verdict.
 */
export function isTruncated(r: { truncated_steps?: number; finish_reason?: string }): boolean {
  return (r.truncated_steps ?? 0) > 0 || r.finish_reason === "length";
}

/**
 * Provenance of a setting across a summary group's kept rows: the one value they
 * all recorded, `null` when none recorded one (legacy rows), or `"mixed"` when
 * they differ — e.g. capped live rows alongside legacy uncapped controls. Used
 * for `max_output_tokens` and `retry_policy`.
 */
export function provenance<T extends number | string>(values: Array<T | null>): T | "mixed" | null {
  const distinct = new Set(values);
  if (distinct.size > 1) return "mixed";
  return distinct.values().next().value ?? null;
}

/**
 * A row's wall-clock time net of its retry backoff (`wall_ms − retry_wait_ms`):
 * only the backoff slept is removed; failed attempts' own call time (throttled,
 * 5xx, network) stays in. Legacy rows (no `retry_wait_ms`) count as no wait.
 */
export function netWallMs(r: { wall_ms: number; retry_wait_ms?: number | null }): number {
  return r.wall_ms - (r.retry_wait_ms ?? 0);
}

/**
 * Sum of a count that legacy rows lack: over the rows that recorded it (a lower
 * bound beside legacy rows), or `null` when none did — never a measured-looking 0.
 */
export function sumCounts(xs: Array<number | null | undefined>): number | null {
  const recorded = xs.filter((x): x is number => x != null);
  return recorded.length === 0 ? null : recorded.reduce((acc, x) => acc + x, 0);
}

/** Normalize a (possibly absent) cell category into a stable grouping key. */
export function categoryOf(c: Pick<CellResult, "category">): string {
  return c.category ?? "(uncategorized)";
}

/**
 * Coarsen a category for AGENT aggregation: BFCL's `bfcl-simple` /
 * `bfcl-multiple` collapse to a single `bfcl` group so task-completion and
 * token/cost savings are reported combined (not per scenario type). Retrieval
 * panels stay split — they key off `corpusOf(scenario_id)`, not this. Other
 * corpora's categories pass through unchanged.
 */
export function coarseCategory(category: string): string {
  return category === "bfcl-simple" || category === "bfcl-multiple" ? "bfcl" : category;
}

export interface ArmModelStats {
  arm: Arm;
  model: string;
  /**
   * Scenario category (e.g. `bfcl-simple`, `bfcl-multiple`, `metatool-single`)
   * or `(uncategorized)`. Part of the grouping key so corpora with multiple
   * scenario types (BFCL) keep them on separate rows.
   */
  category: string;
  /**
   * Pool size this row aggregates; sweeps emit one row per (arm, model, pool_size).
   * `null` for pool-size-agnostic arms (e.g. `control-oracle`) — exactly one row
   * per (arm, model) regardless of `--pool-sizes`.
   */
  pool_size: number | null;
  /** Distinct scenarios contributing to this group. */
  scenarios: number;
  /** Total cells (= scenarios × runs_per_scenario_for_this_group). */
  n: number;
  /** Mean across per-scenario *selection* success rates (programmatic|judge pass / runs). */
  success_rate: number;
  /**
   * Mean across per-scenario *task-completion* rates (AST: right function AND
   * arguments). `null` when no scenario in the group has argument ground truth
   * (non-BFCL corpora) — render as "—".
   */
  task_completion_rate: number | null;
  /**
   * Mean catalog (= tools the agent actually saw) size across cells. Counts
   * direct tools AND gateway tools (`search_tools` / `invoke_tool`) — the
   * honest "what was visible" tally. For oracle this is the gold count
   * (~1–2); for control-baseline it equals `pool_size`; for ratel arms it's
   * direct + 2 gateway tools (so ~5–7 at top-K=5).
   */
  mean_catalog_size: number;
  /**
   * Token / turn / $ / wall means cover non-errored cells only (errored ones are
   * still scored in `success_rate`); `null` when no scenario has a clean cell.
   */
  mean_input_tokens: number | null;
  mean_total_tokens: number | null;
  mean_turns: number | null;
  mean_dollar_cost: number | null;
  /** Wall-clock, retry backoff included (what the savings compare). */
  mean_wall_ms: number | null;
  /** Same, net of retry backoff (`wall_ms − retry_wait_ms`; legacy rows: no wait). */
  mean_wall_net_ms: number | null;
}

interface ScenarioStats {
  arm: Arm;
  model: string;
  category: string;
  pool_size: number | null;
  scenario_id: string;
  /** Selection passes / runs for this scenario in this (arm, model, pool_size). */
  success_rate: number;
  /** AST passes / ast-applicable runs; `null` when the scenario has no arg ground truth. */
  task_rate: number | null;
  mean_catalog: number;
  /** Over non-errored runs only; `null` when every run errored. */
  mean_input: number | null;
  mean_total: number | null;
  mean_turns: number | null;
  mean_dollar: number | null;
  mean_wall: number | null;
  mean_wall_net: number | null;
  /** Number of runs aggregated for this scenario. */
  runs: number;
}

/**
 * Two-stage aggregation: cells → per-scenario means → per-(arm, model, pool_size)
 * means.
 *
 * Pool size is part of the grouping key so a sweep (`--pool-sizes 30,50,180`)
 * emits one row per pool size instead of collapsing them into a single row
 * whose averages mix populations. Scenarios whose `candidate_pool` is smaller
 * than the requested pool size end up with the same actual `pool_size`
 * (universe ceiling) and dedupe naturally.
 *
 * The per-scenario stage gives every scenario equal weight in the headline,
 * so a high-run-count scenario can't drown out the rest. Concretely: a
 * scenario that passes 4/5 runs contributes a 0.8 success rate, regardless
 * of how many other scenarios ran 1× or 10×. This is the natural reading
 * of "what fraction of scenarios succeed" when runs-per-scenario varies.
 */
/**
 * @param coarsen When `true` (default), BFCL `bfcl-simple` + `bfcl-multiple`
 * pool into one `bfcl` agent group (the `REPORT.md` behavior). Pass `false` to
 * keep them split per subset — used by the BFCL website report.
 */
export function statsByArmModel(cells: CellResult[], coarsen = true): ArmModelStats[] {
  const cat = (c: Pick<CellResult, "category">): string =>
    coarsen ? coarseCategory(categoryOf(c)) : categoryOf(c);
  // Stage 1: per (scenario, arm, model, pool_size) → per-scenario means.
  const byScenario = new Map<string, CellResult[]>();
  for (const c of cells) {
    const key = `${c.scenario_id}::${c.arm}::${c.model}::${cat(c)}::${c.pool_size}`;
    const arr = byScenario.get(key) ?? [];
    arr.push(c);
    byScenario.set(key, arr);
  }
  const perScenario: ScenarioStats[] = [];
  for (const arr of byScenario.values()) {
    const head = arr[0];
    const passes = arr.filter(
      (c) => c.programmatic_verdict === "pass" || c.judge_verdict === "pass",
    ).length;
    // Task-completion: only over runs that carry an AST verdict (BFCL); null otherwise.
    const astCells = arr.filter((c) => c.ast_verdict !== "n/a");
    const task_rate =
      astCells.length > 0
        ? astCells.filter((c) => c.ast_verdict === "pass").length / astCells.length
        : null;
    // Token/cost/latency means skip errored runs; selection above scores them all.
    const clean = arr.filter((c) => errorClassOf(c) === null);
    perScenario.push({
      arm: head.arm,
      model: head.model,
      category: cat(head),
      pool_size: head.pool_size,
      scenario_id: head.scenario_id,
      success_rate: passes / arr.length,
      task_rate,
      mean_catalog: mean(arr.map((c) => c.catalog_size)),
      mean_input: meanOrNull(clean.map((c) => c.input_tokens)),
      mean_total: meanOrNull(clean.map((c) => c.total_tokens)),
      mean_turns: meanOrNull(clean.map((c) => c.turns)),
      mean_dollar: meanOrNull(clean.map((c) => c.dollar_cost)),
      mean_wall: meanOrNull(clean.map((c) => c.wall_ms)),
      mean_wall_net: meanOrNull(clean.map(netWallMs)),
      runs: arr.length,
    });
  }

  // Stage 2: per (arm, model, pool_size) → mean across per-scenario means.
  const byGroup = new Map<string, ScenarioStats[]>();
  for (const p of perScenario) {
    const key = `${p.arm}::${p.model}::${p.category}::${p.pool_size}`;
    const arr = byGroup.get(key) ?? [];
    arr.push(p);
    byGroup.set(key, arr);
  }
  const out: ArmModelStats[] = [];
  for (const ps of byGroup.values()) {
    const head = ps[0];
    out.push({
      arm: head.arm,
      model: head.model,
      category: head.category,
      pool_size: head.pool_size,
      scenarios: ps.length,
      n: ps.reduce((acc, p) => acc + p.runs, 0),
      success_rate: mean(ps.map((p) => p.success_rate)),
      task_completion_rate: (() => {
        const rates = ps.map((p) => p.task_rate).filter((r): r is number => r !== null);
        return rates.length > 0 ? mean(rates) : null;
      })(),
      mean_catalog_size: mean(ps.map((p) => p.mean_catalog)),
      mean_input_tokens: meanOfPresent(ps.map((p) => p.mean_input)),
      mean_total_tokens: meanOfPresent(ps.map((p) => p.mean_total)),
      mean_turns: meanOfPresent(ps.map((p) => p.mean_turns)),
      mean_dollar_cost: meanOfPresent(ps.map((p) => p.mean_dollar)),
      mean_wall_ms: meanOfPresent(ps.map((p) => p.mean_wall)),
      mean_wall_net_ms: meanOfPresent(ps.map((p) => p.mean_wall_net)),
    });
  }
  return out.sort(
    (a, b) =>
      a.model.localeCompare(b.model) ||
      a.category.localeCompare(b.category) ||
      a.arm.localeCompare(b.arm) ||
      comparePoolSizes(a.pool_size, b.pool_size),
  );
}

/** Mean over the non-null values; null when there are none (all-errored scenarios drop out). */
function meanOfPresent(xs: (number | null)[]): number | null {
  return meanOrNull(xs.filter((x): x is number => x !== null));
}

/** Sort comparator that puts agnostic rows (`null`) after every numeric pool size. */
function comparePoolSizes(a: number | null, b: number | null): number {
  if (a === b) return 0;
  if (a === null) return 1;
  if (b === null) return -1;
  return a - b;
}

/**
 * Control vs ratel means (non-errored cells, see `ArmModelStats`). A mean is
 * `null` when its arm has no clean cell (or, for oracle, no row); a savings %
 * is `null` when either side is.
 */
export interface SavingsRow {
  model: string;
  category: string;
  pool_size: number;
  control_mean_input: number | null;
  ratel_mean_input: number | null;
  oracle_mean_input: number | null;
  input_savings_pct: number | null;
  control_mean_total: number | null;
  ratel_mean_total: number | null;
  total_savings_pct: number | null;
  control_mean_dollars: number | null;
  ratel_mean_dollars: number | null;
  dollar_savings_pct: number | null;
  control_mean_turns: number | null;
  ratel_mean_turns: number | null;
  oracle_mean_turns: number | null;
  control_mean_wall_ms: number | null;
  ratel_mean_wall_ms: number | null;
  wall_savings_pct: number | null;
}

function pctSavings(control: number | null, ratel: number | null): number | null {
  if (control === null || ratel === null) return null;
  if (control === 0) return 0;
  return (1 - ratel / control) * 100;
}

/**
 * Pair control-baseline vs ratel-full within each (model, pool_size). Sweeps
 * land one row per pool size so each pool's savings story stays intact —
 * collapsing across pool sizes would dilute small-pool wins with large-pool
 * losses (or vice versa). Oracle is pool-size-agnostic (one row per model),
 * so its tokens/turns are joined per-model and shown identically across all
 * pool rows for that model.
 */
export function savingsByModel(cells: CellResult[], coarsen = true): SavingsRow[] {
  const stats = statsByArmModel(cells, coarsen);
  // Oracle is pool-size-agnostic but still per-category, so key it by both.
  const oracleByModel = new Map<string, ArmModelStats>();
  for (const s of stats) {
    if (s.arm === "control-oracle") oracleByModel.set(`${s.model}::${s.category}`, s);
  }
  // Skip oracle (and any other agnostic arm) when grouping for the per-pool
  // pairing — its `pool_size` is `null` and would never pair with control/ratel
  // rows anyway.
  const byGroup = new Map<string, ArmModelStats[]>();
  for (const s of stats) {
    if (s.pool_size === null) continue;
    const key = `${s.model}::${s.category}::${s.pool_size}`;
    const arr = byGroup.get(key) ?? [];
    arr.push(s);
    byGroup.set(key, arr);
  }
  const out: SavingsRow[] = [];
  for (const arr of byGroup.values()) {
    const control = arr.find((s) => s.arm === "control-baseline");
    const ratel = arr.find((s) => s.arm === "ratel-full");
    if (!control || !ratel || control.pool_size === null) continue;
    const oracle = oracleByModel.get(`${control.model}::${control.category}`);
    out.push({
      model: control.model,
      category: control.category,
      pool_size: control.pool_size,
      control_mean_input: control.mean_input_tokens,
      ratel_mean_input: ratel.mean_input_tokens,
      oracle_mean_input: oracle?.mean_input_tokens ?? null,
      input_savings_pct: pctSavings(control.mean_input_tokens, ratel.mean_input_tokens),
      control_mean_total: control.mean_total_tokens,
      ratel_mean_total: ratel.mean_total_tokens,
      total_savings_pct: pctSavings(control.mean_total_tokens, ratel.mean_total_tokens),
      control_mean_dollars: control.mean_dollar_cost,
      ratel_mean_dollars: ratel.mean_dollar_cost,
      dollar_savings_pct: pctSavings(control.mean_dollar_cost, ratel.mean_dollar_cost),
      control_mean_turns: control.mean_turns,
      ratel_mean_turns: ratel.mean_turns,
      oracle_mean_turns: oracle?.mean_turns ?? null,
      control_mean_wall_ms: control.mean_wall_ms,
      ratel_mean_wall_ms: ratel.mean_wall_ms,
      wall_savings_pct: pctSavings(control.mean_wall_ms, ratel.mean_wall_ms),
    });
  }
  return out.sort(
    (a, b) =>
      a.model.localeCompare(b.model) ||
      a.category.localeCompare(b.category) ||
      a.pool_size - b.pool_size,
  );
}

/**
 * Retrieval subset. For tool corpora this is `single-tool` | `multi-tool`; for
 * the skill corpus (SR-Agents) it is the dataset name (e.g. `champ`, `toolqa`)
 * or `all` for the aggregate. Kept as a string so the skill datasets render as
 * their own panels.
 */
export type RetrievalSubset = string;
/**
 * Retrieval granularity. `tool` retrieves individual tools (a tool-corpus
 * experiment); `skill` retrieves authored skill documents (the SR-Agents
 * experiment). The two are evaluated separately and never share a panel.
 */
export type RetrievalMode = "tool" | "skill";

export interface RetrievalSummary {
  corpus: string;
  subset: RetrievalSubset;
  mode: RetrievalMode;
  k: number;
  pool_size: number;
  n: number;
  mean_precision: number;
  median_precision: number;
  mean_recall: number;
  median_recall: number;
  /** Fraction of queries with recall == 1.0 (every gold in top-K). Equals
   *  hit_rate for single-gold buckets; stricter for multi-gold tool retrieval. */
  complete_set_rate: number;
  mean_mrr: number;
  median_mrr: number;
  mean_ndcg: number;
  median_ndcg: number;
  /** hit@K — for single-gold corpora (e.g. BFCL) this is accuracy@K. */
  hit_rate: number;
  /** Fraction of scenarios with *every* gold tool in the top-K. */
  complete_rate: number;
}

/**
 * Infer a corpus label from a scenario id. The retrieval JSONL doesn't carry a
 * corpus tag of its own — the ingestion adapters prefix scenario ids per source
 * (`metatool-st-*` / `metatool-mt-*`, `toolret-*`, ...), and the report groups
 * by that prefix so multi-corpus runs render one table per source.
 */
export function corpusOf(scenarioId: string): string {
  // BFCL ships two subsets that BOTH have a single gold tool, so `subsetOf`
  // can't separate them — the corpus prefix does. Check these before the
  // generic prefixes so each renders as its own panel.
  if (scenarioId.startsWith("bfcl-simple-")) return "bfcl-simple";
  if (scenarioId.startsWith("bfcl-multiple-")) return "bfcl-multiple";
  if (scenarioId.startsWith("metatool-")) return "metatool";
  if (scenarioId.startsWith("toolret-")) return "toolret";
  if (scenarioId.startsWith("sragents-")) return "sragents";
  return "other";
}

/**
 * Bucket a row by gold-set size. Single-tool rows have binary recall (0 or 1),
 * which is mathematically the hit rate; multi-tool rows produce fractional
 * recall and are interpreted differently (e.g. "do both gold tools land in
 * top-K"). We surface them in separate panels so neither story drowns the
 * other.
 */
export function subsetOf(goldCount: number): RetrievalSubset {
  return goldCount > 1 ? "multi-tool" : "single-tool";
}

/**
 * Derive `(subset, mode)` for a row from the explicit `category` set by the
 * ingest adapters. MetaTool queries are tool-retrieval (`metatool-single` →
 * single-tool/tool, `metatool-multi` → multi-tool/tool). SR-Agents rows carry
 * `sragents-<dataset>` and are skill-retrieval, bucketed by dataset. Falls back
 * to gold-set size in tool mode for category-less corpora (e.g. ToolRet).
 */
export function bucketOf(row: RetrievalRow): {
  subset: RetrievalSubset;
  mode: RetrievalMode;
} {
  switch (row.category) {
    case "metatool-single":
      return { subset: "single-tool", mode: "tool" };
    case "metatool-multi":
      return { subset: "multi-tool", mode: "tool" };
    default:
      if (row.category?.startsWith("sragents-")) {
        return { subset: row.category.slice("sragents-".length), mode: "skill" };
      }
      return { subset: subsetOf(row.gold_count), mode: "tool" };
  }
}

export function retrievalByPoolSize(rows: RetrievalRow[]): RetrievalSummary[] {
  const groups = new Map<string, RetrievalRow[]>();
  for (const r of rows) {
    const corpus = corpusOf(r.scenario_id);
    const { subset, mode } = bucketOf(r);
    const push = (sub: string) => {
      const key = `${corpus}::${sub}::${mode}::${r.k}::${r.target_pool_size}`;
      const arr = groups.get(key) ?? [];
      arr.push(r);
      groups.set(key, arr);
    };
    push(subset);
    // Skill rows also roll up into an aggregate `all` panel across datasets.
    if (mode === "skill" && subset !== "all") push("all");
  }
  const out: RetrievalSummary[] = [];
  for (const [key, arr] of groups) {
    const [corpus, subset, mode, kStr, poolStr] = key.split("::") as [
      string,
      RetrievalSubset,
      RetrievalMode,
      string,
      string,
    ];
    const precisions = arr.map((r) => r.precision_at_k);
    const recalls = arr.map((r) => r.recall_at_k);
    const mrrs = arr.map((r) => r.reciprocal_rank);
    const ndcgs = arr.map((r) => r.ndcg_at_k);
    out.push({
      corpus,
      subset,
      mode,
      k: Number(kStr),
      pool_size: Number(poolStr),
      n: arr.length,
      mean_precision: mean(precisions),
      median_precision: median(precisions),
      mean_recall: mean(recalls),
      median_recall: median(recalls),
      // complete = every gold in top-K. Prefer the explicit per-row flag;
      // fall back to recall == 1.0 for older rows without it.
      complete_set_rate: mean(arr.map((r) => ((r.complete_at_k ?? r.recall_at_k >= 1) ? 1 : 0))),
      mean_mrr: mean(mrrs),
      median_mrr: median(mrrs),
      mean_ndcg: mean(ndcgs),
      median_ndcg: median(ndcgs),
      hit_rate: mean(arr.map((r) => (r.hit_at_k ? 1 : 0))),
      // `complete_at_k` may be absent on pre-change rows → treat as hit (true
      // for single-gold, which is the only case those older rows covered).
      complete_rate: mean(arr.map((r) => ((r.complete_at_k ?? r.hit_at_k) ? 1 : 0))),
    });
  }
  // tool before skill within a subset, so the multi-tool panels read
  // "tool retrieval" then "skill retrieval" (the baseline, then the upgrade).
  const modeRank = (m: RetrievalMode) => (m === "tool" ? 0 : 1);
  return out.sort(
    (a, b) =>
      a.corpus.localeCompare(b.corpus) ||
      a.subset.localeCompare(b.subset) ||
      modeRank(a.mode) - modeRank(b.mode) ||
      a.k - b.k ||
      a.pool_size - b.pool_size,
  );
}

export interface FailureCounts {
  arm: Arm;
  model: string;
  category: string;
  pool_size: number | null;
  pass: number;
  fail: number;
  errored: number;
  missing_gold: number;
  step_limit: number;
}

export function failureTaxonomy(cells: CellResult[]): FailureCounts[] {
  const groups = new Map<string, CellResult[]>();
  for (const c of cells) {
    const key = `${c.arm}::${c.model}::${coarseCategory(categoryOf(c))}::${c.pool_size}`;
    const arr = groups.get(key) ?? [];
    arr.push(c);
    groups.set(key, arr);
  }
  const out: FailureCounts[] = [];
  for (const arr of groups.values()) {
    const head = arr[0];
    out.push({
      arm: head.arm,
      model: head.model,
      category: coarseCategory(categoryOf(head)),
      pool_size: head.pool_size,
      pass: arr.filter((c) => c.programmatic_verdict === "pass").length,
      fail: arr.filter((c) => c.programmatic_verdict === "fail").length,
      errored: arr.filter((c) => c.error !== null).length,
      missing_gold: arr.filter((c) => c.programmatic_verdict === "fail" && c.tool_calls.length > 0)
        .length,
      step_limit: arr.filter(
        (c) => c.finish_reason === "max-steps" || c.finish_reason === "tool-calls",
      ).length,
    });
  }
  return out.sort(
    (a, b) =>
      a.model.localeCompare(b.model) ||
      a.category.localeCompare(b.category) ||
      a.arm.localeCompare(b.arm) ||
      comparePoolSizes(a.pool_size, b.pool_size),
  );
}

/** `null` (no rows behind a metric) renders as "—" in every formatter. */
const NONE = "—";

function fmtPct(x: number | null): string {
  return x === null ? NONE : `${x.toFixed(1)}%`;
}

function fmtNum(x: number | null): string {
  if (x === null) return NONE;
  if (x >= 1000) return x.toFixed(0);
  if (x >= 10) return x.toFixed(1);
  return x.toFixed(3);
}

function fmtDollars(x: number | null): string {
  return x === null ? NONE : `$${x.toFixed(4)}`;
}

function fmtSeconds(ms: number | null): string {
  return ms === null ? NONE : `${(ms / 1000).toFixed(1)}s`;
}

/** Signed turn delta (ratel − control); "—" when either side is null. */
function fmtTurnsDelta(control: number | null, ratel: number | null): string {
  if (control === null || ratel === null) return NONE;
  const delta = ratel - control;
  return `${delta >= 0 ? "+" : ""}${fmtNum(delta)}`;
}

function fmtPoolSize(p: number | null): string {
  return p === null ? "—" : String(p);
}

/**
 * Render REPORT.md. Cells are superseded first (one per cell: the last final row,
 * else the last row), then final infra errors (`transient|access`) are dropped;
 * both counts are shown under the header. Other errors stay scored fails, but
 * token/cost/latency means skip them ("—" when a group has no clean cell). A group
 * whose every row was dropped has no row at all. Label cells whose rows span
 * several `ratel_version`s are flagged (`versionSplitCells`).
 */
export function renderReport(args: {
  cells: CellResult[];
  retrieval: RetrievalRow[];
  generatedAt?: Date;
}): string {
  const date = (args.generatedAt ?? new Date()).toISOString();
  const latest = supersede(args.cells, labelledCellKeyOf);
  const cells = latest.filter((c) => !isInfraError(c));
  const stats = statsByArmModel(cells);
  const savings = savingsByModel(cells);
  const retrieval = retrievalByPoolSize(args.retrieval);
  const failures = failureTaxonomy(cells);

  const lines: string[] = [];
  lines.push("# Ratel benchmark report");
  lines.push("");
  lines.push(`_Generated: ${date}_`);
  lines.push("");
  lines.push(`Cells: **${cells.length}**, retrieval rows: **${args.retrieval.length}**.`);
  const excluded = latest.length - cells.length;
  const superseded = args.cells.length - latest.length;
  if (excluded > 0 || superseded > 0) {
    lines.push(
      `Not counted: **${excluded}** infra-errored (transient/access) excluded, ` +
        `**${superseded}** superseded by a later row of the same cell.`,
    );
  }
  const split = versionSplitCells(args.cells).length;
  if (split > 0) {
    lines.push(
      `Warning: **${split}** ${split === 1 ? "cell appears" : "cells appear"} under more than one ` +
        "`ratel_version` and each version counts as its own cell (a re-drain must reuse the " +
        "replaced rows' `--ratel-version`; see `results-audit`).",
    );
  }
  lines.push("");

  // 1. Headline. Numbers are mean-of-per-scenario-means: every scenario weighs
  // the same in the headline regardless of how many runs it has. `pool` is the
  // BM25 universe (= what the agent had to pick from); `catalog` is what the
  // model actually saw, gateway tools included — for sweep arms it equals
  // `pool`, for oracle pool is "—" and catalog reveals the gold-tool count
  // (~1–2), for ratel arms it's direct top-K plus the 2 gateway tools.
  lines.push("## Headline");
  lines.push("");
  lines.push(
    "| arm | model | category | pool | catalog | scenarios | n | selection | task-completion | mean input | mean total | mean turns | mean $ | mean wall | mean wall (net) |",
  );
  lines.push("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|");
  for (const s of stats) {
    const task = s.task_completion_rate === null ? "—" : fmtPct(s.task_completion_rate * 100);
    lines.push(
      `| ${s.arm} | ${s.model} | ${s.category} | ${fmtPoolSize(s.pool_size)} | ${fmtNum(s.mean_catalog_size)} | ${s.scenarios} | ${s.n} | ${fmtPct(s.success_rate * 100)} | ${task} | ${fmtNum(s.mean_input_tokens)} | ${fmtNum(s.mean_total_tokens)} | ${fmtNum(s.mean_turns)} | ${fmtDollars(s.mean_dollar_cost)} | ${fmtSeconds(s.mean_wall_ms)} | ${fmtSeconds(s.mean_wall_net_ms)} |`,
    );
  }
  lines.push("");

  // 2. Token + wall savings
  lines.push("## Token savings (ratel vs control)");
  lines.push("");
  if (savings.length === 0) {
    lines.push("_No control + ratel pairs in this run._");
  } else {
    lines.push(
      "| model | category | pool | input (ctrl → ratel) | input savings | total (ctrl → ratel) | total savings | $ (ctrl → ratel) | $ savings | wall (ctrl → ratel) | wall savings | oracle input | turns Δ |",
    );
    lines.push("|---|---|---|---|---|---|---|---|---|---|---|---|---|");
    for (const s of savings) {
      lines.push(
        `| ${s.model} | ${s.category} | ${s.pool_size} | ${fmtNum(s.control_mean_input)} → ${fmtNum(s.ratel_mean_input)} | **${fmtPct(s.input_savings_pct)}** | ${fmtNum(s.control_mean_total)} → ${fmtNum(s.ratel_mean_total)} | **${fmtPct(s.total_savings_pct)}** | ${fmtDollars(s.control_mean_dollars)} → ${fmtDollars(s.ratel_mean_dollars)} | **${fmtPct(s.dollar_savings_pct)}** | ${fmtSeconds(s.control_mean_wall_ms)} → ${fmtSeconds(s.ratel_mean_wall_ms)} | **${fmtPct(s.wall_savings_pct)}** | ${fmtNum(s.oracle_mean_input)} | ${fmtTurnsDelta(s.control_mean_turns, s.ratel_mean_turns)} |`,
      );
    }
  }
  lines.push("");

  // 3. Retrieval quality. One panel per (corpus, subset, mode); inside the
  // panel rows are sorted by (k, pool_size). Tool corpora (MetaTool, ToolRet)
  // split single-tool vs multi-tool; the skill corpus (SR-Agents) renders one
  // panel per dataset plus an aggregate `all` panel (see ADR-0008).
  lines.push("## Retrieval quality (BM25, no LLM)");
  lines.push("");
  if (retrieval.length === 0) {
    lines.push(
      "_No retrieval rows; run `cargo run -p ratel-benchmark -- retrieval ...` to populate._",
    );
  } else {
    // Note: for multi-mapping skill datasets (e.g. CHAMP) an instance has
    // several gold skills, so recall@K is fractional; `complete@K` (every gold
    // skill in the top-K) is the all-or-nothing bar.
    lines.push(
      "> **Skill retrieval (SR-Agents):** authored skill documents retrieved by BM25 over " +
        "name + description (body is not indexed). For multi-mapping datasets (e.g. CHAMP) an " +
        "instance has several gold skills, so recall@K is *fractional*; **`complete@K`** is the " +
        "all-or-nothing bar (every gold skill in the top-K).",
    );
    lines.push("");
    const panels = new Map<string, RetrievalSummary[]>();
    for (const r of retrieval) {
      const key = `${r.corpus}::${r.subset}::${r.mode}`;
      const arr = panels.get(key) ?? [];
      arr.push(r);
      panels.set(key, arr);
    }
    for (const [key, summaries] of panels) {
      const [corpus, subset, mode] = key.split("::");
      lines.push(`### ${corpus} / ${subset} / ${mode}-retrieval`);
      lines.push("");
      lines.push(
        "| K | pool size | n | accuracy@K (hit@K) | complete@K | mean precision@K | median precision@K | mean recall@K | median recall@K | mean MRR@K | median MRR@K | mean nDCG@K | median nDCG@K |",
      );
      lines.push("|---|---|---|---|---|---|---|---|---|---|---|---|---|");
      for (const r of summaries) {
        lines.push(
          `| ${r.k} | ${r.pool_size} | ${r.n} | ${fmtPct(r.hit_rate * 100)} | ${fmtPct(r.complete_rate * 100)} | ${r.mean_precision.toFixed(3)} | ${r.median_precision.toFixed(3)} | ${r.mean_recall.toFixed(3)} | ${r.median_recall.toFixed(3)} | ${r.mean_mrr.toFixed(3)} | ${r.median_mrr.toFixed(3)} | ${r.mean_ndcg.toFixed(3)} | ${r.median_ndcg.toFixed(3)} |`,
        );
      }
      lines.push("");
    }
  }

  // 4. Failure taxonomy
  lines.push("## Failure taxonomy");
  lines.push("");
  lines.push(
    "| arm | model | category | pool | pass | fail | errored | missing gold | step-limit |",
  );
  lines.push("|---|---|---|---|---|---|---|---|---|");
  for (const f of failures) {
    lines.push(
      `| ${f.arm} | ${f.model} | ${f.category} | ${fmtPoolSize(f.pool_size)} | ${f.pass} | ${f.fail} | ${f.errored} | ${f.missing_gold} | ${f.step_limit} |`,
    );
  }
  lines.push("");

  return lines.join("\n");
}
