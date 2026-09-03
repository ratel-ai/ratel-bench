// Joining one cell's artifacts into rows.
//
// Inputs: the Claude Code result envelope, the session transcript, the
// ratel-local telemetry file (ratel arm only), and the frozen config. Output: the
// four raw row types the summarizer consumes.
//
// Two rules run through everything here. Occupancy and billed cost are kept in
// separate fields and never merged, because native tool schemas live in the
// cached prefix and billed savings are therefore far smaller than context
// savings. And an unmeasured quantity is `null`, never 0 — "the agent did not
// search" and "the retriever missed" are different findings.

import {
  type ClaudeResult,
  countCompactions,
  effectiveCalls,
  promptTokens,
  type RawToolUse,
  toolUsesFromTranscript,
  type TurnUsage,
  turnUsagesFromTranscript,
} from "./mcpatlas-agent.js";
import {
  type InvokeSpan,
  invokeSpans,
  metricsAtK,
  parseTelemetry,
  type RankedResult,
  rankedHits,
  searchEvents,
  searchesUntilFirstGold,
  unionRanked,
  unionRecall,
  wastedSearches,
} from "./mcpatlas-gateway.js";
import { GATEWAY_TOOLS, normalizeToolId, serverOf } from "./mcpatlas-servers.js";
import type {
  AgentHarness,
  CanonicalToolId,
  ClaimRubricResult,
  McpAtlasAggregation,
  McpAtlasArm,
  McpAtlasCell,
  McpAtlasFailureClass,
  McpAtlasFailureCounts,
  McpAtlasLatencyBreakdown,
  McpAtlasRetrievalRow,
  McpAtlasScope,
  McpAtlasSearchEventRow,
  McpAtlasTask,
  McpAtlasTokenBreakdown,
  McpAtlasToolCallRow,
  RankedHit,
} from "./mcpatlas-types.js";

export const EMPTY_FAILURES: McpAtlasFailureCounts = {
  ok: 0,
  upstream_error: 0,
  tool_not_found: 0,
  schema_validation_error: 0,
  auth_error: 0,
  rate_limited: 0,
  timeout: 0,
  transport_error: 0,
  gateway_error: 0,
  oversized_result: 0,
  unknown_error: 0,
  off_catalog_call: 0,
};

/**
 * Classify one tool-call failure. First match wins.
 *
 * Two distinctions carry the benchmark:
 *   `off_catalog_call` is NOT a tool failure — the model named something that
 *   does not exist, which is a selection defect, so it is excluded from
 *   `tool_call_failure_rate` and reported under selection instead.
 *   `gateway_error` vs `upstream_error` separates "ratel-local broke it" from
 *   "the server was going to fail anyway", which is the crux comparison.
 *
 * `unknown_error` must stay visible: a growing bucket means this table needs a
 * new row, not that the runs got mysterious.
 */
export function classifyFailure(input: {
  error: string | null;
  inCatalog: boolean;
  viaGateway: boolean;
  tookMs?: number | null;
  timeoutMs?: number;
}): McpAtlasFailureClass {
  if (!input.inCatalog) return "off_catalog_call";
  if (!input.error) {
    if (input.timeoutMs && input.tookMs != null && input.tookMs >= input.timeoutMs) {
      return "timeout";
    }
    return "ok";
  }
  const e = input.error.toLowerCase();
  const any = (...needles: string[]): boolean => needles.some((n) => e.includes(n));

  if (any("unknown toolid", "unknown tool id", "tool not found", "no such tool")) {
    return "tool_not_found";
  }
  if (any("invalid argument", "invalid params", "schema", "required property", "validation")) {
    return "schema_validation_error";
  }
  if (any("401", "403", "unauthorized", "forbidden", "needs_auth", "credential", "api key")) {
    return "auth_error";
  }
  if (any("429", "rate limit", "quota", "too many requests")) return "rate_limited";
  if (any("timeout", "timed out", "deadline", "etimedout")) return "timeout";
  if (any("epipe", "econnreset", "connection", "transport", "socket", "closed")) {
    return "transport_error";
  }
  if (any("too large", "exceeds", "truncated", "oversized")) return "oversized_result";
  if (input.viaGateway && any("retrieval_failed", "gateway", "ratel")) return "gateway_error";
  if (any("not found", "404", "does not exist", "error")) return "upstream_error";
  return "unknown_error";
}

export function tallyFailures(rows: readonly McpAtlasToolCallRow[]): McpAtlasFailureCounts {
  const out: McpAtlasFailureCounts = { ...EMPTY_FAILURES };
  for (const r of rows) out[r.failure_class]++;
  return out;
}

/** Failures excluding off-catalog calls, which are a selection defect. */
export function toolFailureRate(counts: McpAtlasFailureCounts): number {
  const attempted =
    Object.entries(counts)
      .filter(([k]) => k !== "off_catalog_call")
      .reduce((n, [, v]) => n + v, 0) || 0;
  return attempted ? (attempted - counts.ok) / attempted : 0;
}

// ─────────────────────────────────────────────────────────────────────────────
// Selection
// ─────────────────────────────────────────────────────────────────────────────

/** Longest common subsequence length, for trajectory order similarity. */
export function lcsLength(a: readonly string[], b: readonly string[]): number {
  const dp: number[][] = Array.from({ length: a.length + 1 }, () =>
    new Array(b.length + 1).fill(0),
  );
  for (let i = 1; i <= a.length; i++) {
    for (let j = 1; j <= b.length; j++) {
      dp[i][j] =
        a[i - 1] === b[j - 1] ? dp[i - 1][j - 1] + 1 : Math.max(dp[i - 1][j], dp[i][j - 1]);
    }
  }
  return dp[a.length][b.length];
}

export interface SelectionMetrics {
  recall: number;
  precision: number;
  f1: number;
  pass: boolean;
  hit: boolean;
  order_similarity: number;
  missing_gold: CanonicalToolId[];
  extra_calls: CanonicalToolId[];
}

/**
 * Tool-selection metrics.
 *
 * `pass` (recall === 1) is the strict bar; `hit` (recall > 0) mirrors the lenient
 * rule the existing programmatic judge uses. Precision is DIAGNOSTIC only:
 * MCP-Atlas records one valid trajectory, and a real agent legitimately lists a
 * directory before reading a file, so penalising extra calls would measure
 * trajectory conformance rather than correctness.
 */
export function selectionMetrics(
  gold: readonly CanonicalToolId[],
  observed: readonly CanonicalToolId[],
): SelectionMetrics {
  const goldSet = new Set(gold);
  const uniqueObserved = [...new Set(observed)];
  const hits = uniqueObserved.filter((o) => goldSet.has(o));
  const recall = gold.length ? hits.length / gold.length : 0;
  const precision = uniqueObserved.length ? hits.length / uniqueObserved.length : 0;
  const f1 = recall + precision > 0 ? (2 * recall * precision) / (recall + precision) : 0;
  const observedGoldInOrder = observed.filter((o) => goldSet.has(o));
  return {
    recall,
    precision,
    f1,
    pass: gold.length > 0 && recall === 1,
    hit: recall > 0,
    order_similarity: gold.length ? lcsLength(observedGoldInOrder, gold) / gold.length : 0,
    missing_gold: gold.filter((g) => !uniqueObserved.includes(g)),
    extra_calls: uniqueObserved.filter((o) => !goldSet.has(o)),
  };
}

// ─────────────────────────────────────────────────────────────────────────────
// Rows
// ─────────────────────────────────────────────────────────────────────────────

/**
 * A pre-parsed trace, injected by a harness whose trace does not live in a
 * Claude-Code-shaped transcript. When absent, every consumer falls back to
 * parsing `transcriptText` with the claude parsers — the claude path is
 * byte-identical with or without this type existing.
 */
export interface ParsedTranscript {
  uses: RawToolUse[];
  turnUsages: TurnUsage[];
  compactionEvents: number;
  /** Codex-only; see McpAtlasTokenBreakdown.reasoning_output_tokens. */
  reasoningOutputTokens?: number;
}

export interface CellContext {
  run_id: string;
  config_hash: string;
  generated_at: string;
  cell_key: string;
  task: McpAtlasTask;
  arm: McpAtlasArm;
  catalog_scope: McpAtlasScope;
  catalog_tool_ids: readonly CanonicalToolId[];
  /** The --catalog-tools target (0 = whole scope). */
  catalog_tools: number;
  eval_ks: readonly number[];
  per_call_timeout_ms?: number;
  model: string;
  ratel_version_label: string;
  ratel_local_version: string;
  ratel_sdk_version: string | null;
  /** Stamped on every row this ctx produces. Absent (older callers/fixtures)
   *  means claude-code — the same default every reader applies. */
  agent_harness?: AgentHarness;
}

export function buildToolCallRows(
  ctx: CellContext,
  calls: readonly { tool_id: CanonicalToolId; args: Record<string, unknown>; turn: number }[],
  offCatalog: readonly string[],
  spans: readonly InvokeSpan[],
): McpAtlasToolCallRow[] {
  const inCatalog = new Set(ctx.catalog_tool_ids);
  const gold = new Set(ctx.task.gold_tool_ids);
  const spanFor = new Map<string, InvokeSpan[]>();
  for (const s of spans) {
    const list = spanFor.get(s.tool_id) ?? [];
    list.push(s);
    spanFor.set(s.tool_id, list);
  }
  const base = {
    run_type: "mcpatlas_tool_call" as const,
    run_id: ctx.run_id,
    cell_key: ctx.cell_key,
    task_id: ctx.task.task_id,
    arm: ctx.arm,
    catalog_scope: ctx.catalog_scope,
    model: ctx.model,
    agent_harness: ctx.agent_harness ?? "claude-code",
    ratel_version_label: ctx.ratel_version_label,
    ratel_local_version: ctx.ratel_local_version,
    ratel_sdk_version: ctx.ratel_sdk_version,
  };
  const rows: McpAtlasToolCallRow[] = calls.map((c, i) => {
    const span = spanFor.get(c.tool_id)?.shift();
    const viaGateway = ctx.arm === "ratel";
    const present = inCatalog.has(c.tool_id);
    return {
      ...base,
      call_index: i,
      turn_index: c.turn,
      tool_id: c.tool_id,
      server: serverOf(c.tool_id),
      via_gateway: viaGateway,
      args_size_bytes: span?.args_size_bytes ?? JSON.stringify(c.args ?? {}).length,
      result_size_bytes: null,
      result_tokens_est: null,
      took_ms: span?.took_ms ?? null,
      failure_class: classifyFailure({
        error: span?.error ?? null,
        inCatalog: present,
        viaGateway,
        tookMs: span?.took_ms ?? null,
        timeoutMs: ctx.per_call_timeout_ms,
      }),
      error_message: span?.error ?? null,
      is_gold: gold.has(c.tool_id),
      in_catalog: present,
    };
  });
  // Hallucinated ids never became real calls, but they are a selection defect
  // that must appear in the taxonomy rather than vanishing.
  for (const raw of offCatalog) {
    rows.push({
      ...base,
      call_index: rows.length,
      turn_index: -1,
      tool_id: raw,
      server: null,
      via_gateway: ctx.arm === "ratel",
      args_size_bytes: 0,
      result_size_bytes: null,
      result_tokens_est: null,
      took_ms: null,
      failure_class: "off_catalog_call",
      error_message: `tool id not in catalog: ${raw}`,
      is_gold: false,
      in_catalog: false,
    });
  }
  return rows;
}

export function buildSearchEventRows(
  ctx: CellContext,
  telemetryText: string,
  knownServers: readonly string[],
  invokedOrder: readonly CanonicalToolId[],
): { rows: McpAtlasSearchEventRow[]; results: RankedResult[] } {
  if (ctx.arm !== "ratel") return { rows: [], results: [] };
  const events = searchEvents(parseTelemetry(telemetryText));
  const gold = retrievableGold(ctx);
  const rows: McpAtlasSearchEventRow[] = [];
  const results: RankedResult[] = [];
  events.forEach((e, i) => {
    const r = rankedHits(e, gold, knownServers);
    results.push(r);
    const metrics_at_k: Record<string, ReturnType<typeof metricsAtK>> = {};
    for (const k of ctx.eval_ks) metrics_at_k[String(k)] = metricsAtK(r.ranked, gold, k);
    const invoked = invokedOrder[i] ?? null;
    rows.push({
      run_type: "mcpatlas_search_event",
      run_id: ctx.run_id,
      cell_key: ctx.cell_key,
      task_id: ctx.task.task_id,
      arm: "ratel",
      catalog_scope: ctx.catalog_scope,
      model: ctx.model,
      agent_harness: ctx.agent_harness ?? "claude-code",
      ratel_version_label: ctx.ratel_version_label,
      ratel_local_version: ctx.ratel_local_version,
      ratel_sdk_version: ctx.ratel_sdk_version,
      search_index: i,
      query: e.query,
      origin: e.origin,
      top_k_requested: e.top_k,
      ranked: r.ranked,
      zero_score_hits: r.zeroScore,
      zero_score_dropped: r.zeroScore.length,
      stages: e.stages ?? [],
      took_ms: e.took_ms ?? 0,
      gold_tool_ids: [...gold],
      metrics_at_k,
      invoked_after: invoked,
      invoked_was_in_hits: invoked ? r.ranked.some((h) => h.tool_id === invoked) : false,
    });
  });
  return { rows, results };
}

/** gold ∩ catalog. Retrieval can only fairly be scored against what was
 *  registered; anything else is an ingest error, not a retriever miss. */
export function retrievableGold(ctx: CellContext): CanonicalToolId[] {
  const cat = new Set(ctx.catalog_tool_ids);
  return ctx.task.gold_tool_ids.filter((g) => cat.has(g));
}

function aggregateRanked(agg: McpAtlasAggregation, results: readonly RankedResult[]): RankedHit[] {
  if (agg === "union") return unionRanked(results);
  if (agg === "first") return results[0]?.ranked ?? [];
  // `best` picks the single search with the most gold in its ranking; ties go to
  // the earlier search, so the choice is deterministic.
  let best = results[0];
  let bestCount = -1;
  for (const r of results) {
    const n = r.ranked.filter((h) => h.is_gold).length;
    if (n > bestCount) {
      bestCount = n;
      best = r;
    }
  }
  return best?.ranked ?? [];
}

export function buildRetrievalRows(
  ctx: CellContext,
  results: readonly RankedResult[],
  queries: readonly string[],
  ratelLocalVersion: string,
  ratelVersionLabel: string,
  retrieverMethod: string,
): McpAtlasRetrievalRow[] {
  if (ctx.arm !== "ratel") return [];
  const gold = retrievableGold(ctx);
  const unreachable = ctx.task.gold_tool_ids.filter((g) => !gold.includes(g));
  const searched = results.length > 0;
  const rows: McpAtlasRetrievalRow[] = [];
  const untilGold = searchesUntilFirstGold(results);

  for (const agg of ["first", "best", "union"] as const) {
    const ranked = aggregateRanked(agg, results);
    for (const k of ctx.eval_ks) {
      const m = searched ? metricsAtK(ranked, gold, k) : null;
      rows.push({
        run_type: "mcpatlas_retrieval",
        run_id: ctx.run_id,
        generated_at: ctx.generated_at,
        model: ctx.model,
        agent_harness: ctx.agent_harness ?? "claude-code",
        ratel_version_label: ratelVersionLabel,
        ratel_local_version: ratelLocalVersion,
        retriever_method: retrieverMethod,
        task_id: ctx.task.task_id,
        cell_key: ctx.cell_key,
        catalog_scope: ctx.catalog_scope,
        pool_size: ctx.catalog_tool_ids.length,
        k,
        aggregation: agg,
        query: agg === "union" ? queries.join(" | ") : (queries[0] ?? ""),
        retrieved: ranked,
        golden_answer: gold,
        gold_count: gold.length,
        unreachable_gold: unreachable,
        gold_incomplete: unreachable.length > 0,
        searched,
        search_count: results.length,
        // Deliberately null, not 0, when the agent never searched.
        recall_at_k: m ? m.recall_at_k : null,
        precision_at_k: m ? m.precision_at_k : null,
        reciprocal_rank: m ? m.reciprocal_rank : null,
        hit_at_k: m ? m.hit_at_k : null,
        complete_at_k: m ? m.complete_at_k : null,
        ndcg_at_k: m ? m.ndcg_at_k : null,
        gold_score: m ? m.gold_score : null,
        best_gold_rank: m ? m.best_gold_rank : null,
        zero_score_dropped: results.reduce((n, r) => n + r.zeroScore.length, 0),
        searches_until_first_gold: untilGold,
        union_recall: searched ? unionRecall(results, gold) : null,
        wasted_searches: wastedSearches(results),
      });
    }
  }
  return rows;
}

// ─────────────────────────────────────────────────────────────────────────────
// Tokens and latency
// ─────────────────────────────────────────────────────────────────────────────

export function buildTokenBreakdown(input: {
  result: ClaudeResult;
  transcriptText: string;
  telemetryText: string;
  arm: McpAtlasArm;
  /** Both measured once per campaign by `schemaTokenEstimate` over schemas
   *  fetched from the same source — see its docstring on why one ruler for
   *  both arms matters more than absolute precision. These arrive from the
   *  caller because neither is derivable from a single cell. */
  nativeCatalogTokens: number;
  gatewaySchemaTokens: number;
  /** canonical tool_id -> schema tokens, for pricing what each search result
   *  cost the ratel arm in context. Omit to leave the overhead unmeasured
   *  (0) rather than guessed. */
  perToolTokens?: Map<string, number>;
  /** Needed to canonicalise telemetry tool ids before the map lookup. */
  knownServers?: readonly string[];
  /** Harness-supplied trace, when the trace does not live in a claude
   *  transcript. Absent on the claude path — the fallbacks below then run
   *  exactly the code that ran before this field existed. */
  parsed?: Pick<ParsedTranscript, "turnUsages" | "compactionEvents" | "reasoningOutputTokens">;
  /** Spread into the row only when provided; see the field docs on
   *  McpAtlasTokenBreakdown. */
  costSource?: "reported" | "computed";
}): McpAtlasTokenBreakdown {
  const turns = input.parsed?.turnUsages ?? turnUsagesFromTranscript(input.transcriptText);
  const prompts = turns.map(promptTokens);
  const first = prompts[0] ?? 0;
  const u = input.result.usage;
  const schema = input.arm === "native" ? input.nativeCatalogTokens : input.gatewaySchemaTokens;
  // What ratel pays BACK into context: the definitions search_tools handed the
  // model, priced per returned tool and summed over every search this cell made.
  //
  // This used to be `catalogTokenEstimate(telemetry)`, which sums ratel-local's
  // `ratel_tool_payload` REGISTRATION events — the size of the whole 127-tool
  // catalog, not of anything the model was shown. It was therefore identical on
  // every ratel cell (25,759 in the last run, task-invariant, the tell) and it
  // made the headline nonsensical: net_context_savings = (catalog - gateway) -
  // catalog = -gateway, so the gateway scored negative on context no matter how
  // well it did.
  //
  // Caveat worth keeping in view: this prices each hit at its full schema. If a
  // search result is a summary rather than a complete definition, this is an
  // UPPER bound on the overhead, and so a lower bound on net savings — the
  // conservative direction for a number that flatters the gateway.
  const retrievalOverhead =
    input.arm === "ratel" && input.perToolTokens
      ? searchEvents(parseTelemetry(input.telemetryText)).reduce(
          (n, e) =>
            n +
            (e.hits ?? []).reduce((m, h) => {
              const id = normalizeToolId(h.tool_id, input.knownServers ?? []);
              return m + (id ? (input.perToolTokens?.get(id) ?? 0) : 0);
            }, 0),
          0,
        )
      : 0;
  const cacheDenom = u.cache_read_input_tokens + u.input_tokens;
  return {
    tool_schema_tokens: schema,
    system_prompt_tokens: Math.max(0, first - schema),
    first_turn_context_tokens: first,
    peak_context_tokens: prompts.length ? Math.max(...prompts) : 0,
    compaction_events: input.parsed?.compactionEvents ?? countCompactions(input.transcriptText),
    retrieval_overhead_tokens: retrievalOverhead,
    tool_result_tokens: 0,
    schema_share_of_prefix: first > 0 ? schema / first : 0,
    billed_input_tokens: u.input_tokens,
    cache_read_tokens: u.cache_read_input_tokens,
    cache_creation_tokens: u.cache_creation_input_tokens,
    output_tokens: u.output_tokens,
    total_tokens:
      u.input_tokens + u.output_tokens + u.cache_read_input_tokens + u.cache_creation_input_tokens,
    // Claude Code's own accounting is authoritative; it models cache effects
    // that the local pricing table does not.
    dollar_cost_total: input.result.total_cost_usd ?? 0,
    cache_hit_ratio: cacheDenom > 0 ? u.cache_read_input_tokens / cacheDenom : 0,
    // Conditionally spread so a claude-path row is byte-identical to before
    // these fields existed.
    ...(input.costSource ? { cost_source: input.costSource } : {}),
    ...(input.parsed?.reasoningOutputTokens !== undefined
      ? { reasoning_output_tokens: input.parsed.reasoningOutputTokens }
      : {}),
  };
}

function pct(xs: readonly number[], q: number): number | null {
  if (!xs.length) return null;
  const s = [...xs].sort((a, b) => a - b);
  const i = (s.length - 1) * q;
  const lo = Math.floor(i);
  const hi = Math.ceil(i);
  return lo === hi ? s[lo] : s[lo] + (s[hi] - s[lo]) * (i - lo);
}

export function buildLatencyBreakdown(input: {
  result: ClaudeResult;
  telemetryText: string;
  knownServers: readonly string[];
  arm: McpAtlasArm;
  /** Median native duration per tool id, for the modelled gateway overhead. */
  nativeBaselineMs?: Map<string, number>;
}): McpAtlasLatencyBreakdown {
  const events = parseTelemetry(input.telemetryText);
  const searches = searchEvents(events);
  const searchMs = searches.map((s) => s.took_ms ?? 0);
  const stage: Record<string, number> = {};
  for (const s of searches) {
    for (const st of s.stages ?? []) stage[st.name] = (stage[st.name] ?? 0) + (st.took_ms ?? 0);
  }
  const spans = invokeSpans(events, input.knownServers);
  const invokeMs = spans.map((s) => s.took_ms ?? 0);
  const searchTotal = searchMs.reduce((a, b) => a + b, 0);
  const invokeTotal = invokeMs.reduce((a, b) => a + b, 0);

  let overhead: number | null = null;
  if (input.arm === "ratel" && input.nativeBaselineMs?.size) {
    let acc = 0;
    let matched = 0;
    for (const s of spans) {
      const base = input.nativeBaselineMs.get(s.tool_id);
      if (base != null && s.took_ms != null) {
        acc += s.took_ms - base;
        matched++;
      }
    }
    overhead = matched ? acc : null;
  }

  const total = input.result.duration_ms ?? 0;
  return {
    total_ms: total,
    search_ms_total: searchTotal,
    search_ms_p50: pct(searchMs, 0.5),
    search_ms_p90: pct(searchMs, 0.9),
    search_stage_ms: stage,
    invoke_ms_total: invokeTotal,
    invoke_ms_p50: pct(invokeMs, 0.5),
    gateway_overhead_ms_est: overhead,
    model_ms_est: Math.max(0, total - searchTotal - invokeTotal),
    turns: input.result.num_turns ?? 0,
  };
}

/** Every search query in a cell, in order — used to label retrieval rows. */
export function searchQueries(telemetryText: string): string[] {
  return searchEvents(parseTelemetry(telemetryText)).map((e) => e.query);
}

/** Tool ids invoked in order, so a search event can be paired with what the
 *  agent did next. */
export function invokedOrder(
  uses: readonly RawToolUse[],
  knownServers: readonly string[],
): CanonicalToolId[] {
  return effectiveCalls(uses, knownServers).calls.map((c) => c.tool_id);
}

export function parseUses(transcriptText: string): RawToolUse[] {
  return toolUsesFromTranscript(transcriptText);
}

// ─────────────────────────────────────────────────────────────────────────────
// The cell assembler
// ─────────────────────────────────────────────────────────────────────────────

export interface AssembleCellInput {
  ctx: CellContext;
  result: ClaudeResult;
  transcriptText: string;
  transcriptPath: string;
  telemetryText: string;
  telemetryPath: string | null;
  /** Computed by the caller via `judgeClaims` — may hit the network, so it
   *  stays async and driver-side rather than inside this (pure) assembler. */
  claimRubric: ClaimRubricResult;
  /** Catalog schema tokens for the native arm, from a prior ratel run's
   *  `ratel_tool_payload` events — the same estimator for both arms. */
  nativeCatalogTokens: number;
  gatewaySchemaTokens: number;
  /** canonical tool_id -> schema tokens, measured once per campaign; prices
   *  what ratel's search results cost it in context. */
  perToolTokens?: Map<string, number>;
  /** Median native duration per tool id, campaign-wide. Absent (or empty) on a
   *  campaign with no native cells — `gateway_overhead_ms_est` is then `null`
   *  everywhere, which `buildLatencyBreakdown` already handles. */
  nativeBaselineMs?: Map<string, number>;
  agentVersion: string;
  runIndex: number;
  cacheSource: "live" | "reused";
  /** Harness-supplied trace; absent on the claude path (the transcript is
   *  then parsed exactly as before — see the equivalence test). */
  parsed?: ParsedTranscript;
  costSource?: "reported" | "computed";
  /** Invoke spans supplied by the caller instead of derived from ratel
   *  telemetry. The native codex arm uses this — it has no telemetry, so
   *  without it every tool call would default to "ok". Absent on the claude
   *  path, where telemetry (ratel) or its absence (native) is authoritative. */
  invokeSpansOverride?: InvokeSpan[];
  /** Codex-only contamination signal; stamped onto the cell verbatim. */
  shellCommandExecutions?: number;
}

/**
 * Assemble one complete `McpAtlasCell` from a cell's raw artifacts.
 *
 * Pure and I/O-free: every input is text or an already-parsed object, so this
 * has no Docker/process/network dependency and is unit-testable with fixtures
 * alone. `mcpatlas-run.ts` is the only intended caller — it owns turning those
 * fixtures into a `McpAtlasToolCallRow[]`/`McpAtlasSearchEventRow[]`/
 * `McpAtlasRetrievalRow[]` sibling set too, via the existing `buildToolCallRows`/
 * `buildSearchEventRows`/`buildRetrievalRows`, which take the same `ctx`.
 */
export function assembleCell(input: AssembleCellInput): McpAtlasCell {
  const { ctx } = input;
  const knownServers = [
    ...new Set(ctx.catalog_tool_ids.map(serverOf).filter((s): s is string => s !== null)),
  ];
  const uses = input.parsed?.uses ?? parseUses(input.transcriptText);
  const { calls, offCatalog, gatewayCalls, nonGatewayCalls, searchCalls } = effectiveCalls(
    uses,
    knownServers,
  );
  const observedIds = calls.map((c) => c.tool_id);
  const spans =
    input.invokeSpansOverride ?? invokeSpans(parseTelemetry(input.telemetryText), knownServers);
  const toolCallRows = buildToolCallRows(ctx, calls, offCatalog, spans);
  const failures = tallyFailures(toolCallRows);
  const sel = selectionMetrics(ctx.task.gold_tool_ids, observedIds);
  const retrievable = retrievableGold(ctx);

  const tokens = buildTokenBreakdown({
    result: input.result,
    transcriptText: input.transcriptText,
    telemetryText: input.telemetryText,
    arm: ctx.arm,
    nativeCatalogTokens: input.nativeCatalogTokens,
    gatewaySchemaTokens: input.gatewaySchemaTokens,
    perToolTokens: input.perToolTokens,
    knownServers,
    parsed: input.parsed,
    costSource: input.costSource,
  });
  const latency = buildLatencyBreakdown({
    result: input.result,
    telemetryText: input.telemetryText,
    knownServers,
    arm: ctx.arm,
    nativeBaselineMs: input.nativeBaselineMs,
  });

  return {
    run_type: "mcpatlas_task",
    run_id: ctx.run_id,
    config_hash: ctx.config_hash,
    generated_at: ctx.generated_at,
    cell_key: ctx.cell_key,

    task_id: ctx.task.task_id,
    scenario_id: ctx.task.id,
    category: "mcpatlas-coding",
    arm: ctx.arm,
    catalog_scope: ctx.catalog_scope,
    catalog_tool_count: ctx.catalog_tool_ids.length,
    catalog_tools: ctx.catalog_tools,
    // Visible to the model: native sees the whole catalog; ratel sees only the
    // two gateway tools actually wired in (`allowedToolsFor` in
    // mcpatlas-servers.ts is authoritative — GATEWAY_TOOLS.length, not a fixed
    // constant, in case that set ever grows).
    catalog_size: ctx.arm === "native" ? ctx.catalog_tool_ids.length : GATEWAY_TOOLS.length,
    run_index: input.runIndex,

    ratel_version_label: ctx.ratel_version_label,
    ratel_local_version: ctx.ratel_local_version,
    ratel_sdk_version: ctx.ratel_sdk_version,
    agent_version: input.agentVersion,
    agent_harness: ctx.agent_harness ?? "claude-code",
    model: ctx.model,

    enabled_tool_ids: ctx.task.enabled_tool_ids,
    gold_tool_ids: ctx.task.gold_tool_ids,
    retrievable_gold_ids: retrievable,
    gold_coverage: ctx.task.gold_tool_ids.length
      ? retrievable.length / ctx.task.gold_tool_ids.length
      : 1,

    observed_tool_ids: observedIds,
    tool_calls: calls,

    claim_rubric: input.claimRubric,
    task_pass: input.claimRubric.verdict === "pass",
    programmatic_verdict: sel.hit ? "pass" : "fail",
    judge_verdict: input.claimRubric.verdict,

    tool_selection_recall: sel.recall,
    tool_selection_precision: sel.precision,
    tool_selection_f1: sel.f1,
    tool_selection_pass: sel.pass,
    tool_selection_hit: sel.hit,
    trajectory_order_similarity: sel.order_similarity,
    missing_gold: sel.missing_gold,
    extra_calls: sel.extra_calls,
    off_catalog_calls: offCatalog,

    tokens,
    latency,
    tool_failures: failures,
    tool_calls_total: toolCallRows.length,
    tool_calls_unique: new Set(observedIds).size,
    gateway_calls: gatewayCalls,
    non_gateway_calls: nonGatewayCalls,
    search_count: searchCalls,
    ...(input.shellCommandExecutions !== undefined
      ? { shell_command_executions: input.shellCommandExecutions }
      : {}),

    final_text: input.result.result,
    finish_reason: input.result.subtype,
    error: input.result.is_error ? (input.result.result ?? "error") : null,

    transcript_path: input.transcriptPath,
    telemetry_path: input.telemetryPath,
    telemetry_binding: input.telemetryPath ? "per_cell_file" : "none",
    cache_source: input.cacheSource,
  };
}
