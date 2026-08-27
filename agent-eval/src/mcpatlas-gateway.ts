// Parsing ratel-local's JSONL telemetry, and the retrieval metrics derived from it.
//
// Load-bearing for the whole retrieval half of the benchmark: the local trace
// stream is the ONLY place ranked results exist. The OTLP path emits
// `ratel.search` spans carrying counts but strips the tool ids, and the daemon's
// normal resolved-control-plane mode installs no trace sink at all. Hence
// `serve --telemetry-file`, one file per cell.

import { normalizeToolId, serverOf } from "./mcpatlas-servers.js";
import type {
  CanonicalToolId,
  RankedHit,
  RetrievalMetricsAtK,
  SearchStageTiming,
} from "./mcpatlas-types.js";

// ─────────────────────────────────────────────────────────────────────────────
// Events
// ─────────────────────────────────────────────────────────────────────────────

export interface SearchEvent {
  type: "search";
  query: string;
  origin: string;
  top_k: number;
  hits: Array<{ tool_id: string; score: number }>;
  stages: SearchStageTiming[];
  took_ms: number;
  ts?: string;
}

export interface ToolPayloadEvent {
  type: "ratel_tool_payload";
  server: string;
  tool_count: number;
  estimated_tokens: number;
}

export interface InvokeEvent {
  type: "invoke_start" | "invoke_end" | "invoke_error";
  tool_id: string;
  args_size_bytes?: number;
  took_ms?: number;
  error?: string;
  ts?: string;
}

export type TelemetryEvent = SearchEvent | ToolPayloadEvent | InvokeEvent | { type: string };

/** Parse a telemetry file. Malformed lines are skipped rather than fatal — the
 *  file is appended by a live process and can be truncated mid-line if a run is
 *  killed. An EMPTY result is a different matter: callers must treat it as an
 *  error on a ratel cell, because silently empty telemetry yields a confident
 *  report with zero retrieval signal. */
export function parseTelemetry(text: string): TelemetryEvent[] {
  const out: TelemetryEvent[] = [];
  for (const line of text.split("\n")) {
    const s = line.trim();
    if (!s) continue;
    try {
      const e = JSON.parse(s);
      if (e && typeof e.type === "string") out.push(e as TelemetryEvent);
    } catch {
      // truncated or partial line
    }
  }
  return out;
}

export function searchEvents(events: TelemetryEvent[]): SearchEvent[] {
  return events.filter((e): e is SearchEvent => e.type === "search");
}

export function invokeEvents(events: TelemetryEvent[]): InvokeEvent[] {
  return events.filter(
    (e): e is InvokeEvent =>
      e.type === "invoke_start" || e.type === "invoke_end" || e.type === "invoke_error",
  );
}

/** Σ estimated_tokens across registration events — what a native agent pays for
 *  these tool definitions. Measured with ratel-local's own estimator so both arms
 *  are compared on one ruler. */
export function catalogTokenEstimate(events: TelemetryEvent[]): number {
  return events
    .filter((e): e is ToolPayloadEvent => e.type === "ratel_tool_payload")
    .reduce((n, e) => n + (Number(e.estimated_tokens) || 0), 0);
}

/** Tokens a tool definition occupies in the system prompt: name, description and
 *  JSON schema, as the host serialises them.
 *
 *  ~4 chars/token is the standard English-plus-JSON approximation, and it runs
 *  LOW here. Measured against the live coding catalog: this returns 23,973 for
 *  the 127 tools where ratel-local's own estimator reports 25,759, about 7%
 *  under. Corroborated independently by real tokenizer counts — the observed
 *  native-minus-ratel first-turn context difference is 24,343 tokens, which is
 *  (catalog - gateway schemas) and so puts the true catalog above 24,343.
 *
 *  Use it anyway, and use it for BOTH arms. What this feeds is never an
 *  absolute: it is `schema_tokens_savings_pct` and `net_context_savings_pct`,
 *  ratios in which a proportional bias cancels. The failure mode to avoid is
 *  not imprecision but MIXED RULERS — taking the native catalog from
 *  ratel-local's telemetry (25,759, its estimator) and the gateway schemas
 *  from this function would silently inflate the savings by that same 7%.
 *  One function, both arms, schemas fetched from the same source.
 *
 *  Measured on the live coding scope: native catalog 23,973 tokens over 127
 *  tools; gateway surface 1,096 over 4 tools plus its server `instructions`
 *  block. Predicted first-turn difference 22,877 against 24,343 observed —
 *  the same ~6% low, in the same direction, on both sides.
 *
 *  Do not quote the absolute token counts as exact; quote the percentages. */
export function schemaTokenEstimate(tools: readonly SchemaBearingTool[]): number {
  return tools.reduce((n, t) => n + perToolSchemaTokens(t), 0);
}

export interface SchemaBearingTool {
  name: string;
  description?: string;
  inputSchema?: Record<string, unknown>;
}

export function perToolSchemaTokens(tool: SchemaBearingTool): number {
  const serialised = JSON.stringify({
    name: tool.name,
    description: tool.description ?? "",
    inputSchema: tool.inputSchema ?? {},
  });
  return Math.ceil(serialised.length / 4);
}

/** tool_id -> estimated schema tokens, for pricing what a search result costs
 *  when its definitions are handed back to the model. `id` maps a tool to the
 *  canonical id used in telemetry, so callers can supply their own dialect. */
export function perToolTokenMap(
  tools: readonly SchemaBearingTool[],
  id: (t: SchemaBearingTool) => string,
): Map<string, number> {
  const m = new Map<string, number>();
  for (const t of tools) m.set(id(t), perToolSchemaTokens(t));
  return m;
}

export interface InvokeSpan {
  tool_id: string;
  args_size_bytes: number;
  took_ms: number | null;
  error: string | null;
}

/** Pair invoke_start with its terminating invoke_end / invoke_error. */
export function invokeSpans(
  events: TelemetryEvent[],
  knownServers: readonly string[],
): InvokeSpan[] {
  const spans: InvokeSpan[] = [];
  const open = new Map<string, { args: number }>();
  for (const e of invokeEvents(events)) {
    const id = normalizeToolId(e.tool_id ?? "", knownServers) ?? e.tool_id ?? "";
    if (e.type === "invoke_start") {
      open.set(id, { args: Number(e.args_size_bytes) || 0 });
      continue;
    }
    const started = open.get(id);
    open.delete(id);
    spans.push({
      tool_id: id,
      args_size_bytes: started?.args ?? 0,
      took_ms: e.took_ms ?? null,
      error: e.type === "invoke_error" ? (e.error ?? "unknown error") : null,
    });
  }
  // An invoke that never terminated (process killed mid-call) still happened.
  for (const [id, s] of open) {
    spans.push({ tool_id: id, args_size_bytes: s.args, took_ms: null, error: "no invoke_end" });
  }
  return spans;
}

// ─────────────────────────────────────────────────────────────────────────────
// Ranking
// ─────────────────────────────────────────────────────────────────────────────

export interface RankedResult {
  ranked: RankedHit[];
  zeroScore: Array<{ tool_id: CanonicalToolId; score: number }>;
}

/**
 * Ranked hits for one search event, canonicalized.
 *
 * The `score > 0` filter is applied BEFORE ranking, not after slicing top-k.
 * ratel-local appends a matched skill's tool dependencies beyond the topK budget
 * at `score: 0`; filtering those after slicing lets a ride-along at position 2
 * push a real gold hit out of k=3, blaming the retriever for a payload artifact.
 */
export function rankedHits(
  event: SearchEvent,
  gold: readonly CanonicalToolId[],
  knownServers: readonly string[],
): RankedResult {
  const goldSet = new Set(gold);
  const ranked: RankedHit[] = [];
  const zeroScore: RankedResult["zeroScore"] = [];
  for (const h of event.hits ?? []) {
    const id = normalizeToolId(h.tool_id ?? "", knownServers) ?? h.tool_id ?? "";
    const score = Number(h.score) || 0;
    if (score > 0) {
      ranked.push({
        rank: ranked.length + 1,
        tool_id: id,
        score,
        is_gold: goldSet.has(id),
        server: serverOf(id),
      });
    } else {
      zeroScore.push({ tool_id: id, score });
    }
  }
  return { ranked, zeroScore };
}

/**
 * Retrieval metrics at one k. Formulas mirror `retrieval/src/retrieval.rs`
 * (`metrics_at_ks`) exactly, so numbers stay comparable with the BFCL and
 * SR-Agents modes:
 *   recall@k    = |gold ∩ top-k| / |gold|
 *   precision@k = |gold ∩ top-k| / min(|ranked|, k)
 *   RR          = 1 / rank of first gold in top-k, else 0
 *   complete@k  = every gold tool present in top-k
 *   nDCG@k      = binary relevance, DCG Σ 1/log2(rank+1), IDCG over min(|gold|,k)
 */
export function metricsAtK(
  ranked: readonly RankedHit[],
  gold: readonly CanonicalToolId[],
  k: number,
): RetrievalMetricsAtK {
  const goldSet = new Set(gold);
  const top = ranked.slice(0, k);
  const hits = top.filter((h) => goldSet.has(h.tool_id));
  const denom = Math.min(ranked.length, k);

  let rr = 0;
  for (let i = 0; i < top.length; i++) {
    if (goldSet.has(top[i].tool_id)) {
      rr = 1 / (i + 1);
      break;
    }
  }

  let dcg = 0;
  for (let i = 0; i < top.length; i++) {
    if (goldSet.has(top[i].tool_id)) dcg += 1 / Math.log2(i + 2);
  }
  let idcg = 0;
  for (let i = 0; i < Math.min(gold.length, k); i++) idcg += 1 / Math.log2(i + 2);

  // Best gold rank spans the WHOLE ranking, not just top-k: "gold was at rank 7,
  // so k=5 missed it by 2" is a tuning instruction; hit@5=false is only a verdict.
  const bestGoldIdx = ranked.findIndex((h) => goldSet.has(h.tool_id));
  const goldScores = ranked.filter((h) => goldSet.has(h.tool_id)).map((h) => h.score);

  return {
    k,
    recall_at_k: gold.length ? hits.length / gold.length : 0,
    precision_at_k: denom ? hits.length / denom : 0,
    reciprocal_rank: rr,
    hit_at_k: hits.length > 0,
    complete_at_k: gold.length > 0 && hits.length === gold.length,
    ndcg_at_k: idcg > 0 ? dcg / idcg : 0,
    gold_score: goldScores.length ? Math.max(...goldScores) : null,
    best_gold_rank: bestGoldIdx >= 0 ? bestGoldIdx + 1 : null,
  };
}

/** Merge several rankings: dedupe by max score, re-sort, re-rank. Backs the
 *  `union` aggregation — "did the agent ever see gold in a top-k window across
 *  the session", which predicts task success better than any single search. */
export function unionRanked(results: readonly RankedResult[]): RankedHit[] {
  const best = new Map<CanonicalToolId, RankedHit>();
  for (const r of results) {
    for (const h of r.ranked) {
      const prev = best.get(h.tool_id);
      if (!prev || h.score > prev.score) best.set(h.tool_id, h);
    }
  }
  return [...best.values()]
    .sort((a, b) => b.score - a.score)
    .map((h, i) => ({ ...h, rank: i + 1 }));
}

/** Index of the first search whose ranking contained any gold, or null. */
export function searchesUntilFirstGold(results: readonly RankedResult[]): number | null {
  for (let i = 0; i < results.length; i++) {
    if (results[i].ranked.some((h) => h.is_gold)) return i;
  }
  return null;
}

/** Fraction of gold that appeared in ANY search's ranked set. */
export function unionRecall(
  results: readonly RankedResult[],
  gold: readonly CanonicalToolId[],
): number {
  if (!gold.length) return 0;
  const seen = new Set(unionRanked(results).map((h) => h.tool_id));
  return gold.filter((g) => seen.has(g)).length / gold.length;
}

export function wastedSearches(results: readonly RankedResult[]): number {
  return results.filter((r) => !r.ranked.some((h) => h.is_gold)).length;
}
