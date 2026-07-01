import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import {
  armCandidates,
  buildCandidateSets,
  controlKey,
  readControlIndex,
  stratifiedSample,
} from "./sragents-select.js";
import type { SragentsRetrievalRow, SragentsSelectCell } from "./sragents-types.js";
import { RATEL_AI_CORE_VERSION } from "./versions.js";

function row(over: Partial<SragentsRetrievalRow>): SragentsRetrievalRow {
  return {
    generated_at: "2026-06-22T00:00:00.000Z",
    ratel_ai_core_version: "0.2.0",
    scenario_id: "sragents-toolqa_0",
    category: "sragents-toolqa",
    query: "q",
    golden_answer: ["g1"],
    retrieved: [],
    k: 10,
    target_pool_size: 50,
    pool_size: 50,
    gold_count: 1,
    recall_at_k: 0,
    precision_at_k: 0,
    reciprocal_rank: 0,
    hit_at_k: false,
    complete_at_k: false,
    ndcg_at_k: 0,
    gold_score: 1,
    ...over,
  };
}

const ids = (xs: string[]) => xs.map((id) => ({ id, score: 1 }));

describe("buildCandidateSets", () => {
  it("pairs the k=ratelK (shortlist) and k=poolSize (full pool) rows per scenario", () => {
    const rows = [
      row({ scenario_id: "sragents-toolqa_0", k: 10, retrieved: ids(["a", "b"]) }),
      row({ scenario_id: "sragents-toolqa_0", k: 50, retrieved: ids(["a", "b", "c", "d"]) }),
    ];
    const [sc] = buildCandidateSets(rows, 50, 10);
    expect(sc.ratelTopK).toEqual(["a", "b"]);
    expect(sc.fullPool).toEqual(["a", "b", "c", "d"]);
    expect(sc.goldSkillIds).toEqual(["g1"]);
  });

  it("drops scenarios missing either k slice, and ignores other pool sizes", () => {
    const rows = [
      row({ scenario_id: "sragents-toolqa_0", k: 10, retrieved: ids(["a"]) }), // no k=50
      row({ scenario_id: "sragents-toolqa_1", k: 10, retrieved: ids(["a"]), target_pool_size: 99 }),
    ];
    expect(buildCandidateSets(rows, 50, 10)).toHaveLength(0);
  });
});

describe("armCandidates", () => {
  const sc = {
    scenarioId: "sragents-toolqa_0",
    category: "sragents-toolqa",
    goldSkillIds: ["g1", "g2"],
    fullPool: ["a", "b", "c", "d", "e"],
    ratelTopK: ["a", "b"],
    poolSize: 50,
  };

  it("baseline = full pool (shuffled, deterministic), ratel = ranked top-K, oracle = gold", () => {
    const base = armCandidates("control-baseline", sc, 42);
    expect(base.poolSize).toBe(50);
    expect([...base.ids].sort()).toEqual(["a", "b", "c", "d", "e"]); // same set
    expect(armCandidates("control-baseline", sc, 42).ids).toEqual(base.ids); // deterministic
    expect(armCandidates("ratel-full", sc, 42)).toEqual({ ids: ["a", "b"], poolSize: 50 });
    expect(armCandidates("control-oracle", sc, 42)).toEqual({ ids: ["g1", "g2"], poolSize: null });
  });
});

describe("stratifiedSample", () => {
  const mk = (cat: string, n: number) =>
    Array.from({ length: n }, (_, i) => ({
      scenarioId: `${cat}_${i}`,
      category: cat,
      goldSkillIds: [],
      fullPool: [],
      ratelTopK: [],
      poolSize: 50,
    }));
  const scenarios = [...mk("a", 20), ...mk("b", 20), ...mk("c", 20)];

  it("covers every dataset, balanced", () => {
    const sample = stratifiedSample(scenarios, 6, 42);
    expect(sample).toHaveLength(6);
    expect(new Set(sample.map((s) => s.category))).toEqual(new Set(["a", "b", "c"])); // all 3
  });

  it("is reproducible for the same seed, and seeded-random (not head-of-file)", () => {
    const a1 = stratifiedSample(scenarios, 9, 42).map((s) => s.scenarioId);
    const a2 = stratifiedSample(scenarios, 9, 42).map((s) => s.scenarioId);
    expect(a1).toEqual(a2); // same seed → same sample
    // Not just the first few ids of each dataset (would be a_0,a_1,a_2,...).
    expect(a1).not.toEqual(["a_0", "b_0", "c_0", "a_1", "b_1", "c_1", "a_2", "b_2", "c_2"]);
  });

  it("gives a different sample for a different seed", () => {
    const a = stratifiedSample(scenarios, 9, 1).map((s) => s.scenarioId);
    const b = stratifiedSample(scenarios, 9, 2).map((s) => s.scenarioId);
    expect(a).not.toEqual(b);
  });
});

describe("control-arm reuse", () => {
  let dir: string;
  beforeEach(() => {
    dir = mkdtempSync(join(tmpdir(), "sragents-reuse-"));
  });
  afterEach(() => {
    rmSync(dir, { recursive: true, force: true });
  });

  function cell(over: Partial<SragentsSelectCell>): SragentsSelectCell {
    return {
      run_type: "skill_selection",
      generated_at: "2026-06-24T00:00:00.000Z",
      ratel_ai_core_version: "0.2.0",
      scenario_id: "sragents-toolqa_0",
      category: "sragents-toolqa",
      arm: "control-baseline",
      model: "gpt-5.4-mini",
      run_index: 0,
      pool_size: 100,
      candidate_count: 100,
      gold_skill_ids: ["g1"],
      selected_skill_ids: ["g1"],
      input_tokens: 0,
      output_tokens: 0,
      total_tokens: 0,
      dollar_cost: 0,
      wall_ms: 0,
      error: null,
      ...over,
    };
  }

  function write(path: string, cells: SragentsSelectCell[]): void {
    writeFileSync(path, cells.map((c) => JSON.stringify(c)).join("\n"));
  }

  it("controlKey is version-agnostic and encodes pool (null for oracle)", () => {
    expect(controlKey("s", "control-baseline", "m", 100, 0)).toBe("s::control-baseline::m::100::0");
    expect(controlKey("s", "control-oracle", "m", null, 0)).toBe("s::control-oracle::m::null::0");
  });

  it("indexes only control arms; earliest-generated per key wins", () => {
    const path = join(dir, "agent.jsonl");
    write(path, [
      cell({ ratel_ai_core_version: "0.2.0", selected_skill_ids: ["early"] }),
      cell({
        generated_at: "2026-07-01T00:00:00.000Z",
        ratel_ai_core_version: "0.3.0-rc.1",
        selected_skill_ids: ["late"],
      }),
      cell({ arm: "ratel-full", selected_skill_ids: ["ignored"] }), // not cacheable
    ]);
    const { reuse } = readControlIndex(path);
    const key = controlKey("sragents-toolqa_0", "control-baseline", "gpt-5.4-mini", 100, 0);
    expect(reuse.size).toBe(1); // ratel-full excluded
    expect(reuse.get(key)?.selected_skill_ids).toEqual(["early"]); // earliest wins
  });

  it("tracks keys already present at the current version (skip on resume)", () => {
    const path = join(dir, "agent.jsonl");
    write(path, [
      cell({ arm: "control-baseline", ratel_ai_core_version: "0.0.0-not-current" }),
      cell({ arm: "control-oracle", pool_size: null, ratel_ai_core_version: RATEL_AI_CORE_VERSION }),
    ]);
    const { current } = readControlIndex(path);
    expect(
      current.has(controlKey("sragents-toolqa_0", "control-oracle", "gpt-5.4-mini", null, 0)),
    ).toBe(true);
    expect(
      current.has(controlKey("sragents-toolqa_0", "control-baseline", "gpt-5.4-mini", 100, 0)),
    ).toBe(false);
  });
});
