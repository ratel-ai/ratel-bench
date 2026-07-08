// SR-Agents candidate generation via the TS SDK 0.4.0 `SkillCatalog`.
//
// The SR-Agents LLM eval (`sragents-select`) reads a `candidates.jsonl` of
// per-scenario retrieved skill shortlists. Historically that file was produced by
// the offline Rust `skill-retrieval` step (BM25-only, `ratel-ai-core`). This
// script is the SDK-native analog: it registers a per-scenario pool into a
// `SkillCatalog({ method })` and emits the same row shape (`BfclRetrievalRow`),
// so SR-Agents can be run at SDK versions/methods the Rust core doesn't ship —
// e.g. 0.4.0 sparse (bm25) / dense (semantic) / hybrid.
//
// Pooling mirrors BFCL's `expandPool`: per scenario the pool is
// `gold + deterministically-shuffled distractors`, truncated to `--pool-size`
// (max 100). Gold is always present, so retrieval-eval and LLM-eval both operate
// over ≤100 candidates — and semantic/hybrid embed only ~100 skills per scenario
// (fast), never the full 26k catalog.
//
// Per scenario it emits two rows (what `buildCandidateSets` consumes):
//   k = pool_size   → the full ranked pool (control-baseline)
//   k = top_k       → Ratel's shortlist   (ratel-full)
//
// Usage:
//   RATEL_VERSION_LABEL=0.4.0-dense pnpm -F @ratel-ai/benchmark sragents-candidates \
//     --retriever semantic --pool-size 100 --top-k 5 \
//     --output results/raw/sragents/candidates-0.4.0-dense.jsonl [--scenarios N]

import { createReadStream, mkdirSync, writeFileSync } from "node:fs";
import { dirname } from "node:path";
import { createInterface } from "node:readline";
import { SkillCatalog } from "@ratel-ai/sdk";
import { readJsonl } from "./io.js";
import { resolveRepoPath } from "./paths.js";
import type { RetrievalMethod } from "./types.js";
import { RATEL_AI_CORE_VERSION } from "./versions.js";

interface Scenario {
  id: string;
  dataset: string;
  prompt: string;
  gold_skill_ids: string[];
}

interface SkillSpec {
  id: string;
  name: string;
  description: string;
}

interface RetrievedHit {
  id: string;
  score: number;
}

function arg(name: string, fallback: string): string {
  const i = process.argv.indexOf(name);
  return i >= 0 && process.argv[i + 1] ? process.argv[i + 1] : fallback;
}

/** Parse `--pool-sizes 50,100` (deduped, sorted, positive ints). */
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

// ── Deterministic pool construction (mirrors pool.ts: mixSeed + mulberry32) ──
function mixSeed(id: string, seed: number): number {
  let h = seed >>> 0;
  for (const b of Buffer.from(id)) h = Math.imul(h ^ b, 0x01000193) >>> 0;
  return h >>> 0;
}
function mulberry32(seed: number): () => number {
  let s = seed >>> 0;
  return () => {
    s = (s + 0x6d2b79f5) >>> 0;
    let t = s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
function shuffleInPlace<T>(arr: T[], seed: number): void {
  const rng = mulberry32(seed);
  for (let i = arr.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    [arr[i], arr[j]] = [arr[j], arr[i]];
  }
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
  const method = arg("--retriever", "bm25") as RetrievalMethod;
  if (method !== "bm25" && method !== "semantic" && method !== "hybrid") {
    throw new Error(`--retriever must be bm25, semantic, or hybrid (got "${method}")`);
  }
  const topK = Number(arg("--top-k", "5"));
  // Pool sizes to emit. SR-Agents retrieval eval uses {50,100}; the LLM eval reads only
  // pool 100 from the same file (`--pool-size` kept as a single-value alias). Each pool
  // emits k-slices {1,3,5,top_k,pool} so one file feeds both retrieval eval and LLM eval.
  const poolSizes = parsePoolSizes(arg("--pool-sizes", arg("--pool-size", "100")));
  const seed = Number(arg("--seed", "42"));
  const scenarioLimit = Number(arg("--scenarios", "0")); // 0 = all
  const skillsPath = resolveRepoPath(arg("--skills", "test-data/sragents-skills.jsonl"));
  const instancesPath = resolveRepoPath(arg("--instances", "test-data/sragents.jsonl"));
  const outputPath = resolveRepoPath(
    arg("--output", `results/raw/sragents/candidates-${RATEL_AI_CORE_VERSION}.jsonl`),
  );

  let scenarios = readJsonl<Scenario>(instancesPath);
  // Restrict to the canonical scenario set. The standard is 100/dataset × 6 = 600 — a seeded
  // subset of the full 5,400 in sragents.jsonl. Pass a reference JSONL (e.g. the existing
  // candidates.jsonl) whose `scenario_id`s define the set, so every version evaluates the
  // identical 600 scenarios. Without it, ALL 5,400 are used (wrong for a standard run).
  const idsFrom = arg("--scenarios-from", "");
  if (idsFrom) {
    const keep = new Set(
      readJsonl<{ scenario_id: string }>(resolveRepoPath(idsFrom)).map((r) => r.scenario_id),
    );
    scenarios = scenarios.filter((s) => keep.has(s.id));
  }

  // --pool-from: use the FIXED pool from a reference file (the 0.2.0 candidates) instead of
  // building a fresh gold+distractors pool. This keeps the candidate pool identical across
  // versions, so control-baseline/oracle stay comparable and reusable — ONLY the ranking
  // (ratel shortlist) changes per retriever. The pool is read from the authoritative
  // `pool_ids` field (full gold-complete membership), NOT from `retrieved` (the retriever's
  // ranking, which for BM25 omits zero-score docs and can drop gold). Keyed by
  // (scenario, pool_size); also restricts the scenario set to the reference.
  const poolFrom = arg("--pool-from", "");
  const refPools = new Map<string, Map<number, string[]>>();
  if (poolFrom) {
    for (const r of readJsonl<{
      scenario_id: string;
      target_pool_size: number;
      pool_ids?: string[];
    }>(resolveRepoPath(poolFrom))) {
      if (!r.pool_ids) {
        throw new Error(
          `--pool-from reference ${poolFrom} has no pool_ids on scenario ${r.scenario_id}: ` +
            `regenerate it with the pool_ids-aware Rust skill-retrieval runner (else the pool ` +
            `would be reconstructed from the lossy 'retrieved' ranking and could drop gold).`,
        );
      }
      const m = refPools.get(r.scenario_id) ?? new Map<number, string[]>();
      // pool_ids is per-cell (duplicated across k-rows); last-writer-wins is fine — identical.
      m.set(r.target_pool_size, r.pool_ids);
      refPools.set(r.scenario_id, m);
    }
    scenarios = scenarios.filter((s) => refPools.has(s.id));
  }

  if (scenarioLimit > 0) scenarios = scenarios.slice(0, scenarioLimit);

  // Load the skill universe once (id/name/description only — drop the huge `body`).
  const byId = new Map<string, SkillSpec>();
  const universe: SkillSpec[] = [];
  const rl = createInterface({ input: createReadStream(skillsPath), crlfDelay: Infinity });
  for await (const line of rl) {
    if (!line.trim()) continue;
    const s = JSON.parse(line) as SkillSpec;
    const spec = { id: s.id, name: s.name, description: s.description };
    byId.set(s.id, spec);
    universe.push(spec);
  }
  console.log(
    `sragents-candidates: method=${method} pools=[${poolSizes.join(",")}] top-k=${topK} ` +
      `over ${scenarios.length} scenarios (universe=${universe.length} skills) → ${outputPath}`,
  );

  const runId = `sdk-cand-${Date.now()}`;
  const generatedAt = new Date().toISOString();
  const lines: string[] = [];
  const t0 = Date.now();
  for (const sc of scenarios) {
    // Distractors shuffled once per scenario, so larger pools are supersets of smaller ones.
    const goldSet = new Set(sc.gold_skill_ids);
    const goldSpecs: SkillSpec[] = sc.gold_skill_ids.map(
      (id) => byId.get(id) ?? { id, name: id, description: `gold skill ${id}` },
    );
    const distractors = universe.filter((s) => !goldSet.has(s.id));
    shuffleInPlace(distractors, mixSeed(sc.id, seed));
    const category = `sragents-${sc.dataset}`;

    for (const poolSize of poolSizes) {
      let pool: SkillSpec[];
      if (poolFrom) {
        const ids = refPools.get(sc.id)?.get(poolSize);
        if (!ids) continue; // reference has no pool of this size for this scenario
        pool = ids.map((id) => byId.get(id) ?? { id, name: id, description: `skill ${id}` });
      } else {
        pool = [...goldSpecs, ...distractors.slice(0, Math.max(0, poolSize - goldSpecs.length))];
      }
      // Invariant guardrail: the pool MUST contain every gold skill. (A sparse retriever may
      // still fail to *rank* a zero-overlap gold — that stays a real miss — but the gold must
      // at least be *in the pool*, or baseline/dense are structurally sabotaged.) Fail loudly
      // at generation time, before any LLM spend.
      const poolIdSet = new Set(pool.map((s) => s.id));
      const missingGold = sc.gold_skill_ids.filter((g) => !poolIdSet.has(g));
      if (missingGold.length) {
        throw new Error(
          `pool for ${sc.id} (pool_size=${poolSize}) is missing gold [${missingGold.join(", ")}] — ` +
            `pool is not gold-complete`,
        );
      }
      const poolIds = pool.map((s) => s.id);
      const kValues = [...new Set([1, 3, 5, topK, poolSize])].filter((k) => k <= poolSize);

      const catalog = method === "bm25" ? new SkillCatalog() : new SkillCatalog({ method });
      for (const s of pool) catalog.register(s);
      if (method !== "bm25") catalog.buildEmbeddings();
      const hits = catalog
        .search(sc.prompt, poolSize)
        .map((h) => ({ id: h.skillId, score: h.score }));

      for (const k of kValues) {
        lines.push(
          JSON.stringify({
            run_type: "retrieval",
            run_id: runId,
            generated_at: generatedAt,
            ratel_ai_core_version: RATEL_AI_CORE_VERSION,
            scenario_id: sc.id,
            query: sc.prompt,
            golden_answer: sc.gold_skill_ids,
            category,
            target_pool_size: poolSize,
            actual_pool_size: pool.length,
            pool_ids: poolIds,
            k,
            pool_size: poolSize,
            retrieved: hits.slice(0, k),
            ...metrics(sc.gold_skill_ids, hits, k),
          }),
        );
      }
    }
  }

  mkdirSync(dirname(outputPath), { recursive: true });
  writeFileSync(outputPath, `${lines.join("\n")}\n`, "utf-8");
  console.log(
    `  wrote ${lines.length} rows (${scenarios.length} scenarios × ${poolSizes.length} pool(s) × ` +
      `k[1,3,5,pool]) in ${((Date.now() - t0) / 1000).toFixed(1)}s`,
  );
}

main().catch((err) => {
  console.error(`\n✗ ${(err as Error).message}`);
  process.exit(1);
});
