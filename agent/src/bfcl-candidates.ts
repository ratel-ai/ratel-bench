// BFCL retrieval-evaluation generator via the TS SDK 0.4.0 `ToolCatalog`.
//
// The BFCL retrieval evaluation (`bfcl-summarize --retrieval-rows`) reads a
// per-row metrics JSONL. Pre-0.4.0 that file came from the Rust `retrieval`
// command (BM25-only, `ratel-ai-core`). This is the SDK-native analog — the twin
// of `sragents-candidates`, but for tools — so BFCL retrieval eval can run at
// SDK versions/methods the Rust core doesn't ship (0.4.0 sparse/dense/hybrid).
//
// Fixed design (see EXPERIMENTS.md): pools {30,100} × k {1,3,5}. Per scenario the
// pool is built by the same `expandPool` the live BFCL run uses (gold +
// deterministic distractors), registered into `ToolCatalog({ method })`, ranked,
// and scored against the gold tools. (BFCL's *LLM* eval retrieves live via
// `start` — it does not read this file; this is retrieval eval only.)
//
// Usage:
//   RATEL_VERSION_LABEL=0.4.0-dense pnpm -F @ratel-ai/benchmark bfcl-candidates \
//     --retriever semantic --pool-sizes 30,100 \
//     --output results/raw/bfcl/retrieval-0.4.0-dense.jsonl [--scenarios N]
//
// `--selector jev` swaps the Ratel catalog for TypeSafe Jev (src/selectors/jev.ts)
// over the identical pools; needs TYPESAFE_API_KEY unless every request is cached
// in `--jev-cache` (default results/raw/jev-cache/bfcl.jsonl):
//   RATEL_VERSION_LABEL=jev-1.13.0 pnpm -F @ratel-ai/benchmark bfcl-candidates \
//     --selector jev --pool-sizes 30,100 --output results/raw/bfcl/retrieval-jev-1.13.0.jsonl

import { mkdirSync, writeFileSync } from "node:fs";
import { dirname } from "node:path";
import { config as loadEnv } from "dotenv";
import { loadScenarios } from "./corpus.js";
import { resolveRepoPath } from "./paths.js";
import { buildToolUniverse, expandPool } from "./pool.js";
import { buildToolCatalog } from "./sdk/adapter.js";
import { parseEmbedding } from "./sdk/embedding.js";
import { selectVersion } from "./sdk/resolve.js";
import {
  JevCache,
  JevRanker,
  type JevRankMeta,
  jevConfigFromArgs,
  type RerankMeta,
  rerankWithJev,
} from "./selectors/jev.js";
import type { RetrievalMethod, Scenario } from "./types.js";
import { RATEL_AI_CORE_VERSION } from "./versions.js";

interface RetrievedHit {
  id: string;
  score: number;
}

function arg(name: string, fallback: string): string {
  const i = process.argv.indexOf(name);
  return i >= 0 && process.argv[i + 1] ? process.argv[i + 1] : fallback;
}

/** Parse `--pool-sizes 30,100` (deduped, sorted, positive ints). */
function parsePoolSizes(raw: string): number[] {
  const out = [
    ...new Set(
      raw
        .split(",")
        .map((s) => Number(s.trim()))
        .filter((n) => Number.isInteger(n) && n > 0),
    ),
  ].sort((a, b) => a - b);
  if (!out.length) throw new Error(`--pool-sizes must list ≥1 positive integer (got "${raw}")`);
  return out;
}

/** `bfcl-simple-…` → `bfcl-simple`; falls back to the scenario's own category. */
function bfclCategory(sc: Scenario): string {
  const parts = sc.id.split("-");
  return parts.length >= 2 ? `${parts[0]}-${parts[1]}` : (sc.category ?? "bfcl");
}

/** Standard retrieval metrics for one scenario's top-k ranking against the gold set. */
function metrics(gold: string[], ranked: RetrievedHit[], k: number) {
  const goldSet = new Set(gold);
  const topK = ranked.slice(0, k);
  const found = new Set(topK.filter((h) => goldSet.has(h.id)).map((h) => h.id)).size;
  let rr = 0;
  for (let i = 0; i < topK.length; i++)
    if (goldSet.has(topK[i].id)) {
      rr = 1 / (i + 1);
      break;
    }
  let dcg = 0;
  for (let i = 0; i < topK.length; i++) if (goldSet.has(topK[i].id)) dcg += 1 / Math.log2(i + 2);
  let idcg = 0;
  for (let i = 0; i < Math.min(gold.length, k); i++) idcg += 1 / Math.log2(i + 2);
  const scored = topK.filter((h) => goldSet.has(h.id)).map((h) => h.score);
  return {
    gold_count: gold.length,
    recall_at_k: gold.length ? found / gold.length : 0,
    precision_at_k: k ? found / k : 0,
    reciprocal_rank: rr,
    hit_at_k: found > 0,
    complete_at_k: found === gold.length,
    ndcg_at_k: idcg ? dcg / idcg : 0,
    gold_score: scored.length ? Math.max(...scored) : null,
  };
}

async function main(): Promise<void> {
  const selector = arg("--selector", "ratel");
  if (selector !== "ratel" && selector !== "jev" && selector !== "ratel+jev") {
    throw new Error(`--selector must be ratel, jev, or ratel+jev (got "${selector}")`);
  }
  const rerankDepth = Number(arg("--rerank-depth", "20")); // ratel+jev: Ratel shortlist size
  // ratel+jev only: also write Ratel's own (pre-rerank) rows — the exact rows `--selector ratel`
  // would produce — so one pass yields both arms and slow (semantic) catalogs are built once.
  const ratelOutput = arg("--ratel-output", "");
  const ratelLabel = arg("--ratel-label", "");
  if (ratelOutput && (selector !== "ratel+jev" || !ratelLabel)) {
    throw new Error("--ratel-output requires --selector ratel+jev and --ratel-label");
  }
  const method = arg("--retriever", "bm25") as RetrievalMethod;
  if (method !== "bm25" && method !== "semantic" && method !== "hybrid") {
    throw new Error(`--retriever must be bm25, semantic, or hybrid (got "${method}")`);
  }
  // SDK selection must happen before the first catalog build — `select()` throws
  // once a module is loaded, so this is deliberately the earliest statement that
  // touches the SDK layer.
  selectVersion(arg("--sdk-version", ""));
  const embedding = parseEmbedding(arg("--embedding", ""));
  const poolSizes = parsePoolSizes(arg("--pool-sizes", arg("--pool-size", "30,100")));
  const seed = Number(arg("--seed", "42"));
  const scenarioLimit = Number(arg("--scenarios", "0")); // 0 = all
  const corpusPath = resolveRepoPath(arg("--corpus", "test-data/bfcl-all.jsonl"));
  const outputPath = resolveRepoPath(
    arg("--output", `results/raw/bfcl/retrieval-${RATEL_AI_CORE_VERSION}.jsonl`),
  );
  const kSlices = [1, 3, 5]; // fixed retrieval-eval k (see EXPERIMENTS.md)
  let jev: JevRanker | undefined;
  if (selector !== "ratel") {
    loadEnv();
    const cache = new JevCache(
      resolveRepoPath(arg("--jev-cache", "results/raw/jev-cache/bfcl.jsonl")),
    );
    jev = new JevRanker({ kind: "tool", cache, seed, ...jevConfigFromArgs(arg) });
    console.log(
      `bfcl-candidates: selector=${selector}` +
        (selector === "ratel+jev" ? ` rerank-depth=${rerankDepth}` : "") +
        ` (${cache.size} cached responses in ${cache.path})`,
    );
  }

  let scenarios = loadScenarios(corpusPath);
  if (scenarioLimit > 0) scenarios = scenarios.slice(0, scenarioLimit);
  const universe = buildToolUniverse(scenarios);
  // --shard i/n: run the i-th of n contiguous scenario slices (1-based). Applied AFTER the
  // universe is built, so pools are identical to an unsharded run; concatenating shard
  // outputs 1..n reproduces the unsharded file (modulo run_id/generated_at).
  const shard = arg("--shard", "");
  if (shard) {
    const [i, n] = shard.split("/").map(Number);
    if (!Number.isInteger(i) || !Number.isInteger(n) || n < 1 || i < 1 || i > n) {
      throw new Error(`--shard must be i/n with 1 ≤ i ≤ n (got "${shard}")`);
    }
    const size = Math.ceil(scenarios.length / n);
    scenarios = scenarios.slice((i - 1) * size, i * size);
    console.log(`bfcl-candidates: shard ${i}/${n} → ${scenarios.length} scenarios`);
  }
  console.log(
    `bfcl-candidates: method=${method} pools=[${poolSizes.join(",")}] k=[${kSlices.join(",")}] ` +
      `over ${scenarios.length} scenarios (universe=${universe.length} tools) → ${outputPath}`,
  );

  const runId = `sdk-bfcl-ret-${Date.now()}`;
  const generatedAt = new Date().toISOString();
  const lines: string[] = [];
  const ratelLines: string[] = [];
  const t0 = Date.now();
  for (const sc of scenarios) {
    const category = bfclCategory(sc);
    for (const poolSize of poolSizes) {
      const pool = expandPool(sc, universe, poolSize, seed);
      const emit = (
        out: string[],
        label: string,
        ranked: RetrievedHit[],
        meta?: JevRankMeta | RerankMeta,
      ): void => {
        for (const k of kSlices) {
          if (k > poolSize) continue;
          out.push(
            JSON.stringify({
              run_type: "retrieval",
              run_id: runId,
              generated_at: generatedAt,
              ratel_ai_core_version: label,
              scenario_id: sc.id,
              query: sc.prompt,
              golden_answer: sc.gold_tools,
              category,
              target_pool_size: poolSize,
              actual_pool_size: pool.length,
              k,
              pool_size: poolSize,
              retrieved: ranked.slice(0, k),
              ...metrics(sc.gold_tools, ranked, k),
              ...(meta ? { selector, ...meta } : {}),
            }),
          );
        }
      };
      let hits: RetrievedHit[];
      let jevMeta: JevRankMeta | RerankMeta | undefined;
      if (jev && selector === "jev") {
        const ranked = await jev.rank(sc.prompt, pool, {
          scenarioId: sc.id,
          poolSize,
          seedKey: `${sc.id}:jev:${poolSize}`,
        });
        hits = ranked.hits;
        jevMeta = ranked.meta;
      } else {
        const { search } = await buildToolCatalog({
          method,
          embedding,
          tools: pool.map((t) => ({
            id: t.id,
            name: t.name,
            description: t.description,
            inputSchema: t.input_schema,
            outputSchema: t.output_schema ?? {},
            execute: async () => ({}),
          })),
        });
        hits = (await search(sc.prompt, poolSize)).map((h) => ({
          id: h.toolId,
          score: h.score,
        }));
        if (jev) {
          if (ratelOutput) emit(ratelLines, ratelLabel, hits);
          const reranked = await rerankWithJev(jev, sc.prompt, pool, hits, rerankDepth, {
            scenarioId: sc.id,
            poolSize,
            seedKey: `${sc.id}:ratel+jev:${poolSize}`,
          });
          hits = reranked.hits;
          jevMeta = reranked.meta;
        }
      }

      emit(lines, RATEL_AI_CORE_VERSION, hits, jevMeta);
    }
  }

  mkdirSync(dirname(outputPath), { recursive: true });
  writeFileSync(outputPath, `${lines.join("\n")}\n`, "utf-8");
  console.log(
    `  wrote ${lines.length} rows (${scenarios.length} scenarios × ${poolSizes.length} pool(s) × ` +
      `${kSlices.length} k) in ${((Date.now() - t0) / 1000).toFixed(1)}s`,
  );
  if (ratelOutput) {
    const ratelPath = resolveRepoPath(ratelOutput);
    mkdirSync(dirname(ratelPath), { recursive: true });
    writeFileSync(ratelPath, `${ratelLines.join("\n")}\n`, "utf-8");
    console.log(
      `  wrote ${ratelLines.length} pre-rerank Ratel rows (${ratelLabel}) → ${ratelPath}`,
    );
  }
  if (jev) console.log(jev.summary());
}

main().catch((err) => {
  console.error(`\n✗ ${(err as Error).message}`);
  process.exit(1);
});
