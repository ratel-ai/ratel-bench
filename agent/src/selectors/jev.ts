// TypeSafe Jev as a retrieval-eval selector — the `--selector jev` path of
// `bfcl-candidates` / `sragents-candidates`.
//
// Jev is not a retriever: it is a hosted classification model with no catalog or
// search call. One `choice` question whose options are the pool members returns a
// probability for every option; sorting those gives a full ranking over the pool —
// the same shape as Ratel's `search(prompt, poolSize)`, so the generators' pool
// build, k-slicing and `metrics()` are shared verbatim.
//
// Fairness rules (see docs/plans/jev-vs-ratel-retrieval.plan.md):
//   - option text carries exactly the fields Ratel indexes (name, description, and
//     for tools the flattened input/output schema property names, descriptions and
//     enum values — mirror of ratel-ai-core `indexing::searchable_text`);
//   - `expandPool` puts gold first, so options are shuffled with a seeded PRNG
//     before sending, and probability ties break in that shuffled order;
//   - the instructions text is fixed and the model version pinned.
//
// Every response is cached (append-only JSONL keyed by a hash of the exact
// request), so reruns cost nothing and an interrupted run resumes.

import { createHash } from "node:crypto";
import { appendFileSync, existsSync, mkdirSync, readFileSync } from "node:fs";
import { dirname } from "node:path";
import { mixSeed, shuffleInPlace } from "../pool.js";

export type JevKind = "tool" | "skill";

/** Pinned model — `jev-latest` moves on every release. */
export const JEV_MODEL = "jev-1.13.0";
export const JEV_ENDPOINT = "https://api.typesafe.ai/v1/systemone";
/** $ per input token (output tokens are free). */
export const JEV_INPUT_PRICE = 0.042 / 1_000_000;

/** Fixed once, never tuned on the test scenarios. */
export const JEV_INSTRUCTIONS: Record<JevKind, string> = {
  tool: "Which tool best fulfils this request?",
  skill: "Which skill best fulfils this request?",
};

const QUESTION_ID = "select";

/** A pool member. Tools carry schemas; skills only id/name/description. */
export interface JevItem {
  id: string;
  name: string;
  description: string;
  input_schema?: Record<string, unknown>;
  output_schema?: Record<string, unknown>;
}

export interface RankedHit {
  id: string;
  score: number;
}

// ---------- option text ----------

function flattenSchema(schema: unknown, prefix: string, out: string[]): void {
  if (!schema || typeof schema !== "object") return;
  const s = schema as Record<string, unknown>;
  const props = s.properties;
  if (props && typeof props === "object") {
    for (const [key, sub] of Object.entries(props as Record<string, unknown>)) {
      const path = prefix ? `${prefix}.${key}` : key;
      const field = (sub ?? {}) as Record<string, unknown>;
      let entry = path;
      if (typeof field.description === "string" && field.description) {
        entry += ` — ${field.description}`;
      }
      if (Array.isArray(field.enum)) {
        const values = field.enum.filter((v): v is string => typeof v === "string");
        if (values.length) entry += ` (one of: ${values.join(", ")})`;
      }
      out.push(entry);
      flattenSchema(sub, path, out);
    }
  }
  if (s.items) flattenSchema(s.items, prefix ? `${prefix}[]` : "[]", out);
}

/** Option text for one pool member — the option key already carries the name. */
export function optionText(item: JevItem): string {
  const params: string[] = [];
  flattenSchema(item.input_schema, "", params);
  const outputs: string[] = [];
  flattenSchema(item.output_schema, "", outputs);
  let text = item.description;
  if (params.length) text += `\nParameters: ${params.join("; ")}`;
  if (outputs.length) text += `\nReturns: ${outputs.join("; ")}`;
  return text;
}

/**
 * `compact` option text for servers with a small shared context (OpenJev/Verdict: 512
 * tokens, options first, query last, and the option *key* is dropped): the name plus the
 * description's first sentence, so ~15 options still leave room for the query.
 */
export function compactOptionText(item: JevItem): string {
  const first = item.description
    .split("\n")[0]
    .split(/(?<=\.)\s/)[0]
    .trim();
  return `${item.name || item.id}: ${first}`;
}

export type OptionTextMode = "standard" | "compact";

// ---------- options (shuffle + keys) ----------

export interface JevOptions {
  /** Option key → description, in the (shuffled) order sent. */
  criteria: Record<string, string>;
  keyToId: Map<string, string>;
  /** Option keys in sent order — the tie-break order. */
  order: string[];
}

/**
 * Shuffle `items` with a seed derived from `seedKey` and key each option by its
 * name. Duplicate names within a pool get `#2`, `#3`… so every option stays
 * addressable and maps back to its id.
 */
export function buildOptions(
  items: readonly JevItem[],
  seedKey: string,
  seed: number,
  mode: OptionTextMode = "standard",
): JevOptions {
  const shuffled = [...items];
  shuffleInPlace(shuffled, mixSeed(seedKey, seed));
  const criteria: Record<string, string> = {};
  const keyToId = new Map<string, string>();
  const order: string[] = [];
  const seen = new Map<string, number>();
  for (const item of shuffled) {
    const base = item.name || item.id;
    const n = (seen.get(base) ?? 0) + 1;
    seen.set(base, n);
    const key = n === 1 ? base : `${base}#${n}`;
    criteria[key] = mode === "compact" ? compactOptionText(item) : optionText(item);
    keyToId.set(key, item.id);
    order.push(key);
  }
  return { criteria, keyToId, order };
}

/**
 * Full ranking over the pool: probability desc, ties in sent (shuffled) order.
 * Options Jev omitted score 0; keys Jev invented are ignored.
 */
export function rankFromProbabilities(
  probabilities: Record<string, number>,
  opts: Pick<JevOptions, "keyToId" | "order">,
): RankedHit[] {
  return opts.order
    .map((key, idx) => ({ key, idx, p: Number(probabilities[key] ?? 0) }))
    .sort((a, b) => b.p - a.p || a.idx - b.idx)
    .map(({ key, p }) => ({ id: opts.keyToId.get(key) as string, score: p }));
}

// ---------- HTTP ----------

export interface JevRequest {
  model: string;
  state: string;
  questions: Record<
    string,
    { type: "choice"; instructions: string; criteria: Record<string, string> }
  >;
  /** Server-specific extras (e.g. laya-serve's `head_max_len` / `max_len` token budget). */
  [extra: string]: unknown;
}

export interface JevResponse {
  model: string;
  answers: Record<
    string,
    { type: string; choice?: string; confidence?: number; probabilities?: Record<string, number> }
  >;
  usage?: { input_tokens?: number; output_tokens?: number };
}

export class JevHttpError extends Error {
  constructor(
    readonly status: number,
    body: string,
  ) {
    super(`Jev HTTP ${status}: ${body.slice(0, 500)}`);
  }
}

export interface JevClientOptions {
  /** Omitted for keyless self-hosted servers (laya-serve, simple-jev). */
  apiKey?: string;
  endpoint?: string;
  fetchImpl?: typeof fetch;
  timeoutMs?: number;
  maxAttempts?: number;
  sleep?: (ms: number) => Promise<void>;
}

/** `TYPESAFE_BASE_URL` (the SDKs' override) wins over the public endpoint. */
function defaultEndpoint(): string {
  const base = process.env.TYPESAFE_BASE_URL;
  return base ? `${base.replace(/\/+$/, "")}/v1/systemone` : JEV_ENDPOINT;
}

const RETRYABLE = new Set([408, 429, 500, 502, 503, 504, 529]);

/**
 * POST one request; retries 429/529/5xx/timeouts with exponential backoff (1 s doubling,
 * capped at 60 s). Defaults ride out a ~9-minute outage — a multi-hour semantic/hybrid run
 * must not die on a brief API blip (it did at 6 × 10 s).
 */
export async function callJev(
  body: JevRequest,
  opts: JevClientOptions,
): Promise<{ response: JevResponse; latencyMs: number }> {
  const fetchImpl = opts.fetchImpl ?? fetch;
  const sleep = opts.sleep ?? ((ms: number) => new Promise((r) => setTimeout(r, ms)));
  const maxAttempts = opts.maxAttempts ?? 10;
  const timeoutMs = opts.timeoutMs ?? 30_000;
  let lastErr: unknown;
  for (let attempt = 1; attempt <= maxAttempts; attempt++) {
    const t0 = performance.now();
    try {
      const res = await fetchImpl(opts.endpoint ?? defaultEndpoint(), {
        method: "POST",
        headers: {
          ...(opts.apiKey ? { Authorization: `Bearer ${opts.apiKey}` } : {}),
          "Content-Type": "application/json",
        },
        body: JSON.stringify(body),
        signal: AbortSignal.timeout(timeoutMs),
      });
      if (res.ok) {
        const response = (await res.json()) as JevResponse;
        return { response, latencyMs: Math.round(performance.now() - t0) };
      }
      const err = new JevHttpError(res.status, await res.text());
      if (!RETRYABLE.has(res.status)) throw err;
      lastErr = err;
    } catch (err) {
      if (err instanceof JevHttpError && !RETRYABLE.has(err.status)) throw err;
      lastErr = err;
    }
    if (attempt < maxAttempts) {
      console.warn(
        `  jev: attempt ${attempt}/${maxAttempts} failed (${String(lastErr)}); retrying`,
      );
      await sleep(Math.min(60_000, 1_000 * 2 ** (attempt - 1)));
    }
  }
  throw new Error(`Jev request failed after ${maxAttempts} attempts: ${String(lastErr)}`);
}

// ---------- cache ----------

export interface JevCacheEntry {
  key: string;
  scenario_id: string;
  pool_size: number;
  model: string;
  response: JevResponse;
  latency_ms: number;
  at: string;
}

export function requestKey(body: JevRequest): string {
  return createHash("sha256").update(JSON.stringify(body)).digest("hex");
}

export class JevCache {
  private readonly entries = new Map<string, JevCacheEntry>();

  constructor(readonly path: string) {
    if (!existsSync(path)) return;
    for (const line of readFileSync(path, "utf-8").split("\n")) {
      if (!line.trim()) continue;
      const e = JSON.parse(line) as JevCacheEntry;
      this.entries.set(e.key, e);
    }
  }

  get size(): number {
    return this.entries.size;
  }

  get(key: string): JevCacheEntry | undefined {
    return this.entries.get(key);
  }

  put(entry: JevCacheEntry): void {
    mkdirSync(dirname(this.path), { recursive: true });
    appendFileSync(this.path, `${JSON.stringify(entry)}\n`, "utf-8");
    this.entries.set(entry.key, entry);
  }
}

// ---------- ranker ----------

export interface JevRankMeta {
  jev_model: string;
  latency_ms: number;
  input_tokens: number;
  cache_hit: boolean;
  /** Tournament only: number of calls made for this ranking (latency/tokens are summed). */
  calls?: number;
  tournament?: boolean;
}

export interface JevRankerOptions {
  kind: JevKind;
  cache: JevCache;
  seed: number;
  model?: string;
  /** Merged into every request body after the standard fields (part of the cache key). */
  extraBody?: Record<string, unknown>;
  /** Resolved lazily — a fully cached rerun needs no key. */
  apiKey?: () => string | undefined;
  /** False for self-hosted Jev-compatible servers that take no key. Default true. */
  requireApiKey?: boolean;
  client?: Omit<JevClientOptions, "apiKey">;
  /** Option text format (default `standard`, what Ratel indexes). */
  optionText?: OptionTextMode;
  /** Server cap on options per question; larger pools are ranked by a tournament. */
  maxOptions?: number;
}

/** Survivors kept per chunk in each tournament round (≥ the largest k we score). */
const TOURNAMENT_KEEP = 5;

function probabilitiesOf(
  response: JevResponse,
  ctx: { scenarioId: string; poolSize: number },
): Record<string, number> {
  const probabilities = response.answers?.[QUESTION_ID]?.probabilities;
  if (!probabilities) {
    throw new Error(
      `Jev response for ${ctx.scenarioId} (pool ${ctx.poolSize}) has no probabilities: ` +
        JSON.stringify(response).slice(0, 300),
    );
  }
  return probabilities;
}

export class JevRanker {
  /** `inputTokens` counts billed (uncached) requests only. */
  readonly stats = { calls: 0, cacheHits: 0, inputTokens: 0, missingOptions: 0 };
  private readonly model: string;

  constructor(private readonly opts: JevRankerOptions) {
    this.model = opts.model ?? JEV_MODEL;
  }

  /**
   * Rank every item in `pool` for `query`. `seedKey` must be unique per (scenario, pool).
   * Within the server's option cap this is one call; above it, a tournament (see
   * {@link JevRanker.tournament}).
   */
  async rank(
    query: string,
    pool: readonly JevItem[],
    ctx: { scenarioId: string; poolSize: number; seedKey: string },
  ): Promise<{ hits: RankedHit[]; meta: JevRankMeta }> {
    const cap = this.opts.maxOptions;
    if (!cap || pool.length <= cap) return this.rankOnce(query, pool, ctx);
    const acc = { calls: 0, latency: 0, tokens: 0, allHits: true, model: this.model };
    const hits = await this.tournament(query, pool, ctx, cap, 0, acc);
    return {
      hits,
      meta: {
        jev_model: acc.model,
        latency_ms: acc.latency,
        input_tokens: acc.tokens,
        cache_hit: acc.allHits,
        calls: acc.calls,
        tournament: true,
      },
    };
  }

  /**
   * Multi-round tournament for pools above the option cap: seeded-shuffle into balanced
   * chunks of ≤ cap, rank each chunk in one call, keep its top {@link TOURNAMENT_KEEP},
   * and repeat on the survivors until they fit one final call. Final order = the final
   * call's ranking, then each round's eliminated items by their in-chunk probability
   * (later rounds first). Only ranks beyond the finalists (≥ 10) use cross-chunk scores.
   */
  private async tournament(
    query: string,
    items: readonly JevItem[],
    ctx: { scenarioId: string; poolSize: number; seedKey: string },
    cap: number,
    level: number,
    acc: { calls: number; latency: number; tokens: number; allHits: boolean; model: string },
  ): Promise<RankedHit[]> {
    const track = (meta: JevRankMeta) => {
      acc.calls++;
      acc.latency += meta.latency_ms;
      acc.tokens += meta.input_tokens;
      acc.allHits &&= meta.cache_hit;
      acc.model = meta.jev_model;
    };
    if (items.length <= cap) {
      const r = await this.rankOnce(query, items, {
        ...ctx,
        seedKey: `${ctx.seedKey}:L${level}:final`,
      });
      track(r.meta);
      return r.hits;
    }
    const shuffled = [...items];
    shuffleInPlace(shuffled, mixSeed(`${ctx.seedKey}:L${level}:chunks`, this.opts.seed));
    const nChunks = Math.ceil(shuffled.length / cap);
    const base = Math.floor(shuffled.length / nChunks);
    const extra = shuffled.length % nChunks;
    const byId = new Map(items.map((i) => [i.id, i]));
    const survivors: JevItem[] = [];
    const eliminated: { hit: RankedHit; chunk: number }[] = [];
    let start = 0;
    for (let c = 0; c < nChunks; c++) {
      const size = base + (c < extra ? 1 : 0);
      const chunk = shuffled.slice(start, start + size);
      start += size;
      const r = await this.rankOnce(query, chunk, {
        ...ctx,
        seedKey: `${ctx.seedKey}:L${level}:c${c}`,
      });
      track(r.meta);
      r.hits.forEach((hit, i) => {
        // keep ≤ size-1 per chunk so every round shrinks the field (guarantees termination)
        if (i < Math.min(TOURNAMENT_KEEP, chunk.length - 1)) {
          survivors.push(byId.get(hit.id) as JevItem);
        } else eliminated.push({ hit, chunk: c });
      });
    }
    const top = await this.tournament(query, survivors, ctx, cap, level + 1, acc);
    eliminated.sort((a, b) => b.hit.score - a.hit.score || a.chunk - b.chunk);
    return [...top, ...eliminated.map((e) => e.hit)];
  }

  /** One request: rank every item of `pool` (must fit the server's option cap). */
  async rankOnce(
    query: string,
    pool: readonly JevItem[],
    ctx: { scenarioId: string; poolSize: number; seedKey: string },
  ): Promise<{ hits: RankedHit[]; meta: JevRankMeta }> {
    const options = buildOptions(pool, ctx.seedKey, this.opts.seed, this.opts.optionText);
    const body: JevRequest = {
      model: this.model,
      state: query,
      questions: {
        [QUESTION_ID]: {
          type: "choice",
          instructions: JEV_INSTRUCTIONS[this.opts.kind],
          criteria: options.criteria,
        },
      },
      ...(this.opts.extraBody ?? {}),
    };
    const key = requestKey(body);
    let entry = this.opts.cache.get(key);
    const cacheHit = entry !== undefined;
    if (!entry) {
      const apiKey = this.opts.apiKey?.() ?? process.env.TYPESAFE_API_KEY;
      if (!apiKey && this.opts.requireApiKey !== false) {
        throw new Error("TYPESAFE_API_KEY is not set (needed for uncached Jev requests)");
      }
      const { response, latencyMs } = await callJev(body, { apiKey, ...this.opts.client });
      // Validate before caching — a malformed answer must not be replayed forever.
      probabilitiesOf(response, ctx);
      entry = {
        key,
        scenario_id: ctx.scenarioId,
        pool_size: ctx.poolSize,
        model: response.model ?? this.model,
        response,
        latency_ms: latencyMs,
        at: new Date().toISOString(),
      };
      this.opts.cache.put(entry);
      this.stats.calls++;
      this.stats.inputTokens += response.usage?.input_tokens ?? 0;
    } else {
      this.stats.cacheHits++;
    }

    const probabilities = probabilitiesOf(entry.response, ctx);
    this.stats.missingOptions += options.order.filter((k) => !(k in probabilities)).length;
    const inputTokens = entry.response.usage?.input_tokens ?? 0;
    return {
      hits: rankFromProbabilities(probabilities, options),
      meta: {
        jev_model: entry.model,
        latency_ms: entry.latency_ms,
        input_tokens: inputTokens,
        cache_hit: cacheHit,
      },
    };
  }

  summary(): string {
    const cost = this.stats.inputTokens * JEV_INPUT_PRICE;
    return (
      `  jev: ${this.stats.calls} API calls, ${this.stats.cacheHits} cache hits, ` +
      `${this.stats.inputTokens} billed input tokens (~$${cost.toFixed(4)})` +
      (this.stats.missingOptions
        ? `, ${this.stats.missingOptions} options missing from responses`
        : "")
    );
  }
}

// ---------- ratel+jev: Ratel shortlist, Jev re-rank ----------

export interface RerankMeta extends Partial<JevRankMeta> {
  rerank_depth: number;
  /** Shortlist size Ratel actually returned (BM25 omits zero-score docs). */
  ratel_candidates: number;
}

/**
 * Re-rank Ratel's top `depth` hits with Jev. The result covers only the
 * shortlist — anything Ratel didn't return is unreachable, exactly as in a
 * deployed two-stage pipeline. A shortlist of < 2 needs no call and is passed
 * through in Ratel's order.
 */
export async function rerankWithJev<T extends JevItem>(
  jev: JevRanker,
  query: string,
  pool: readonly T[],
  ratelHits: readonly RankedHit[],
  depth: number,
  ctx: { scenarioId: string; poolSize: number; seedKey: string },
): Promise<{ hits: RankedHit[]; meta: RerankMeta }> {
  const byId = new Map(pool.map((item) => [item.id, item]));
  const shortlist = ratelHits.slice(0, depth);
  const candidates = shortlist.flatMap((h) => byId.get(h.id) ?? []);
  const base = { rerank_depth: depth, ratel_candidates: candidates.length };
  if (candidates.length < 2) return { hits: [...shortlist], meta: base };
  const ranked = await jev.rank(query, candidates, ctx);
  return { hits: ranked.hits, meta: { ...base, ...ranked.meta } };
}

// ---------- CLI wiring (shared by the candidate generators) ----------

/**
 * Decision-model flags: `--jev-base-url` points the selector at any server speaking
 * TypeSafe's `/v1/systemone` protocol (laya-serve, simple-jev); `--jev-model` names the
 * model sent; `--jev-body-extra '<json>'` adds server-specific request fields. Defaults
 * reproduce the TypeSafe Jev requests exactly, so existing caches still hit.
 */
export function jevConfigFromArgs(
  arg: (name: string, fallback: string) => string,
): Pick<
  JevRankerOptions,
  "model" | "extraBody" | "requireApiKey" | "client" | "optionText" | "maxOptions"
> {
  const baseUrl = arg("--jev-base-url", "");
  const optionTextRaw = arg("--jev-option-text", "standard");
  if (optionTextRaw !== "standard" && optionTextRaw !== "compact") {
    throw new Error(`--jev-option-text must be standard or compact (got "${optionTextRaw}")`);
  }
  const maxRaw = arg("--jev-max-options", "");
  const maxOptions = maxRaw ? Number(maxRaw) : undefined;
  if (
    maxOptions !== undefined &&
    (!Number.isInteger(maxOptions) || maxOptions <= TOURNAMENT_KEEP)
  ) {
    throw new Error(`--jev-max-options must be an integer > ${TOURNAMENT_KEEP} (got "${maxRaw}")`);
  }
  const extraRaw = arg("--jev-body-extra", "");
  let extraBody: Record<string, unknown> | undefined;
  if (extraRaw) {
    const parsed = JSON.parse(extraRaw) as unknown;
    if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
      throw new Error(`--jev-body-extra must be a JSON object (got ${extraRaw})`);
    }
    extraBody = parsed as Record<string, unknown>;
  }
  return {
    model: arg("--jev-model", JEV_MODEL),
    extraBody,
    requireApiKey: !baseUrl,
    client: baseUrl ? { endpoint: `${baseUrl.replace(/\/+$/, "")}/v1/systemone` } : undefined,
    ...(optionTextRaw === "compact" ? { optionText: "compact" as const } : {}),
    ...(maxOptions ? { maxOptions } : {}),
  };
}
