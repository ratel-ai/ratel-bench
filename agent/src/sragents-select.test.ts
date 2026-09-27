import { existsSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { APICallError, NoObjectGeneratedError, RetryError } from "ai";
import { MockLanguageModelV3 } from "ai/test";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { REPO_ROOT } from "./paths.js";
import {
  armCandidates,
  buildCandidateSets,
  controlKey,
  drainControlCache,
  readControlIndex,
  type SelectArgs,
  selectForCell,
  sragentsCachePaths,
  sragentsCapOptions,
  stratifiedSample,
  type Task,
} from "./sragents-select.js";
import type { SragentsArm, SragentsRetrievalRow, SragentsSelectCell } from "./sragents-types.js";
import { RATEL_AI_CORE_VERSION } from "./versions.js";

// Fixed price so cost assertions don't depend on models.json / MODELS_JSON.
vi.mock("./pricing.js", () => ({
  loadModelPricing: () => ({
    "claude-haiku-4-5": {
      inputPer1M: 1,
      outputPer1M: 5,
      cachedInputPer1M: 0,
      cacheCreationPer1M: 0,
    },
  }),
}));

vi.mock("ai", async () => {
  const actual = await vi.importActual<typeof import("ai")>("ai");
  return {
    ...actual,
    generateObject: vi.fn(),
  };
});

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
    const { reuse } = readControlIndex([path]);
    const key = controlKey("sragents-toolqa_0", "control-baseline", "gpt-5.4-mini", 100, 0);
    expect(reuse.size).toBe(1); // ratel-full excluded
    expect(reuse.get(key)?.selected_skill_ids).toEqual(["early"]); // earliest wins
  });

  it("tracks keys already present at the current version (skip on resume)", () => {
    const path = join(dir, "agent.jsonl");
    write(path, [
      cell({ arm: "control-baseline", ratel_ai_core_version: "0.0.0-not-current" }),
      cell({
        arm: "control-oracle",
        pool_size: null,
        ratel_ai_core_version: RATEL_AI_CORE_VERSION,
      }),
    ]);
    const { current } = readControlIndex([path]);
    expect(
      current.has(controlKey("sragents-toolqa_0", "control-oracle", "gpt-5.4-mini", null, 0)),
    ).toBe(true);
    expect(
      current.has(controlKey("sragents-toolqa_0", "control-baseline", "gpt-5.4-mini", 100, 0)),
    ).toBe(false);
  });

  it("skips transient/access/request-errored rows in reuse", () => {
    const path = join(dir, "agent.jsonl");
    const early = "2026-06-01T00:00:00.000Z";
    write(path, [
      // key 0: an earlier outage row loses to the later good row.
      cell({ generated_at: early, error: "Overloaded", selected_skill_ids: [] }),
      cell({ selected_skill_ids: ["good"] }),
      // key 1: only rerunnable errors → no reuse, runs live.
      cell({ run_index: 1, error: "Forbidden", error_class: "access" }),
      cell({ run_index: 1, error: "tools: too many", error_class: "request" }),
      // key 2: a current-version transient error is current (resume skips it) but not reusable.
      cell({ run_index: 2, ratel_ai_core_version: RATEL_AI_CORE_VERSION, error: "Overloaded" }),
    ]);
    const { reuse } = readControlIndex([path]);
    const key = (run: number) =>
      controlKey("sragents-toolqa_0", "control-baseline", "gpt-5.4-mini", 100, run);
    expect(reuse.get(key(0))?.selected_skill_ids).toEqual(["good"]);
    expect(reuse.has(key(1))).toBe(false);
    expect(reuse.has(key(2))).toBe(false);
  });

  it("[guard] timeout/outcome control rows still reused", () => {
    const path = join(dir, "agent.jsonl");
    write(path, [
      cell({ error: "run timed out after 300000ms" }),
      cell({ run_index: 1, error: "No object generated: bad json", error_class: "outcome" }),
    ]);
    const { reuse } = readControlIndex([path]);
    const key = (run: number) =>
      controlKey("sragents-toolqa_0", "control-baseline", "gpt-5.4-mini", 100, run);
    expect(reuse.get(key(0))?.error).toBe("run timed out after 300000ms");
    expect(reuse.get(key(1))?.error_class).toBe("outcome");
  });

  it("reads multiple sources; the earliest eligible row across files wins", () => {
    const a = join(dir, "agent.jsonl");
    const b = join(dir, "backfill.jsonl");
    write(a, [
      cell({ generated_at: "2026-05-01T00:00:00.000Z", error: "Overloaded" }),
      cell({ generated_at: "2026-07-01T00:00:00.000Z", selected_skill_ids: ["a-late"] }),
    ]);
    write(b, [cell({ generated_at: "2026-06-01T00:00:00.000Z", selected_skill_ids: ["b-early"] })]);
    const missing = join(dir, "missing.jsonl");
    const key = controlKey("sragents-toolqa_0", "control-baseline", "gpt-5.4-mini", 100, 0);
    // Both orders, so neither "last file wins" nor "only the last source is read" passes.
    for (const paths of [
      [a, missing, b],
      [b, missing, a],
    ]) {
      expect(readControlIndex(paths).reuse.get(key)?.selected_skill_ids).toEqual(["b-early"]);
    }
  });

  it("[guard] a current-version rerunnable error still counts as current (resume unchanged)", () => {
    const path = join(dir, "agent.jsonl");
    write(path, [cell({ ratel_ai_core_version: RATEL_AI_CORE_VERSION, error: "Overloaded" })]);
    const { current } = readControlIndex([path]);
    const key = controlKey("sragents-toolqa_0", "control-baseline", "gpt-5.4-mini", 100, 0);
    expect(current.has(key)).toBe(true);
  });

  it("applies harness tiers: exact (provider|cap) beats an earlier legacy row; others never", () => {
    const path = join(dir, "agent.jsonl");
    const bedrock4096 = { provider: "amazon-bedrock", max_output_tokens: 4096 };
    write(path, [
      // key 0: legacy (earlier) vs exact (later) → exact.
      cell({ generated_at: "2026-06-01T00:00:00.000Z", selected_skill_ids: ["legacy"] }),
      cell({
        generated_at: "2026-09-01T00:00:00.000Z",
        selected_skill_ids: ["exact"],
        ...bedrock4096,
      }),
      // key 1: only a different recorded cap / provider → nothing served.
      cell({ run_index: 1, provider: "amazon-bedrock", max_output_tokens: 16384 }),
      cell({ run_index: 1, provider: "anthropic.messages", max_output_tokens: 4096 }),
      cell({ run_index: 1, provider: "anthropic.messages" }),
      // key 2: legacy only → served (legacy tier).
      cell({ run_index: 2, selected_skill_ids: ["legacy2"] }),
      // key 3: same provider, no recorded cap (pre-cap build) → served (legacy tier).
      cell({ run_index: 3, provider: "amazon-bedrock", selected_skill_ids: ["pre-cap"] }),
    ]);
    const harness = new Map([["gpt-5.4-mini", bedrock4096]]);
    const { reuse } = readControlIndex([path], harness);
    const key = (run: number) =>
      controlKey("sragents-toolqa_0", "control-baseline", "gpt-5.4-mini", 100, run);
    expect(reuse.get(key(0))?.selected_skill_ids).toEqual(["exact"]);
    expect(reuse.has(key(1))).toBe(false);
    expect(reuse.get(key(2))?.selected_skill_ids).toEqual(["legacy2"]);
    expect(reuse.get(key(3))?.selected_skill_ids).toEqual(["pre-cap"]);
  });

  it("the exact-tier row wins whichever source lists it first", () => {
    const exactFile = join(dir, "exact.jsonl");
    const legacyFile = join(dir, "legacy.jsonl");
    write(exactFile, [
      cell({
        generated_at: "2026-09-01T00:00:00.000Z",
        selected_skill_ids: ["exact"],
        provider: "amazon-bedrock",
        max_output_tokens: 4096,
      }),
    ]);
    // Earlier, so earliest-wins alone would pick it.
    write(legacyFile, [
      cell({ generated_at: "2026-06-01T00:00:00.000Z", selected_skill_ids: ["legacy"] }),
    ]);
    const harness = new Map([
      ["gpt-5.4-mini", { provider: "amazon-bedrock", max_output_tokens: 4096 }],
    ]);
    const key = controlKey("sragents-toolqa_0", "control-baseline", "gpt-5.4-mini", 100, 0);
    for (const paths of [
      [exactFile, legacyFile],
      [legacyFile, exactFile],
    ]) {
      expect(readControlIndex(paths, harness).reuse.get(key)?.selected_skill_ids).toEqual([
        "exact",
      ]);
    }
  });
});

describe("drainControlCache", () => {
  let dir: string;
  beforeEach(() => {
    dir = mkdtempSync(join(tmpdir(), "sragents-drain-"));
  });
  afterEach(() => {
    rmSync(dir, { recursive: true, force: true });
  });

  const sc = {
    scenarioId: "sragents-toolqa_0",
    category: "sragents-toolqa",
    goldSkillIds: ["g1"],
    fullPool: ["a", "g1"],
    ratelTopK: ["g1"],
    poolSize: 100,
  };
  const model = { id: "gpt-5.4-mini", model: {} as never, maxOutputTokens: null };
  const task = (arm: SragentsArm, runIndex = 0): Task => ({ arm, sc, query: "q", model, runIndex });

  function cached(over: Partial<SragentsSelectCell>): SragentsSelectCell {
    return {
      run_type: "skill_selection",
      generated_at: "2026-06-24T00:00:00.000Z",
      ratel_ai_core_version: "0.0.0-old",
      scenario_id: sc.scenarioId,
      category: sc.category,
      arm: "control-baseline",
      model: model.id,
      run_index: 0,
      pool_size: 100,
      candidate_count: 2,
      gold_skill_ids: ["g1"],
      selected_skill_ids: ["g1"],
      input_tokens: 0,
      output_tokens: 0,
      total_tokens: 0,
      dollar_cost: 0,
      wall_ms: 0,
      error: null,
      cache_source: "live",
      ...over,
    };
  }

  const writeCells = (path: string, cells: SragentsSelectCell[]) =>
    writeFileSync(path, cells.map((c) => `${JSON.stringify(c)}\n`).join(""));
  const readCells = (path: string): SragentsSelectCell[] =>
    existsSync(path)
      ? readFileSync(path, "utf-8")
          .split("\n")
          .filter(Boolean)
          .map((l) => JSON.parse(l) as SragentsSelectCell)
      : [];

  it("appends reused controls re-stamped with cache_source 'reused'; the rest run live", () => {
    const output = join(dir, "out.jsonl");
    const canonical = join(dir, "agent.jsonl");
    const backfill = join(dir, "backfill.jsonl");
    writeCells(canonical, [cached({ error: "Overloaded" })]);
    writeCells(backfill, [
      cached({ arm: "control-oracle", pool_size: null, selected_skill_ids: ["oracle"] }),
      cached({ generated_at: "2026-07-01T00:00:00.000Z", selected_skill_ids: ["late"] }),
    ]);

    const tasks = [task("control-baseline"), task("ratel-full"), task("control-oracle")];
    const { liveTasks, reused } = drainControlCache(tasks, {
      outputPath: output,
      cachePaths: [canonical, backfill],
      force: false,
      allowLegacyCache: true,
    });

    expect(reused).toBe(2);
    expect(liveTasks.map((t) => t.arm)).toEqual(["ratel-full"]);
    const rows = readCells(output);
    expect(rows.map((r) => [r.arm, r.selected_skill_ids, r.cache_source])).toEqual([
      ["control-baseline", ["late"], "reused"],
      ["control-oracle", ["oracle"], "reused"],
    ]);
    expect(rows.every((r) => r.ratel_ai_core_version === RATEL_AI_CORE_VERSION)).toBe(true);
  });

  it("runs a control live when its only cached rows are rerunnable errors", () => {
    const output = join(dir, "out.jsonl");
    const canonical = join(dir, "agent.jsonl");
    writeCells(canonical, [cached({ error: "Internal server error" })]);

    const { liveTasks, reused } = drainControlCache([task("control-baseline")], {
      outputPath: output,
      cachePaths: [canonical],
      force: false,
      allowLegacyCache: true,
    });

    expect(reused).toBe(0);
    expect(liveTasks).toHaveLength(1);
  });

  it("--force truncates the output and runs every cell live", () => {
    const output = join(dir, "out.jsonl");
    const canonical = join(dir, "agent.jsonl");
    writeCells(output, [cached({ ratel_ai_core_version: RATEL_AI_CORE_VERSION })]);
    writeCells(canonical, [cached({})]);

    const tasks = [task("control-baseline"), task("ratel-full")];
    const { liveTasks, reused } = drainControlCache(tasks, {
      outputPath: output,
      cachePaths: [canonical],
      force: true,
      allowLegacyCache: true,
    });

    expect(reused).toBe(0);
    expect(liveTasks).toHaveLength(2);
    expect(readCells(output)).toEqual([]);
  });

  it("skips (resume) controls already in the output at the current version", () => {
    const output = join(dir, "out.jsonl");
    writeCells(output, [cached({ ratel_ai_core_version: RATEL_AI_CORE_VERSION })]);

    const { liveTasks, reused } = drainControlCache([task("control-baseline")], {
      outputPath: output,
      cachePaths: [],
      force: false,
      allowLegacyCache: true,
    });

    expect(reused).toBe(0);
    expect(liveTasks).toEqual([]);
    expect(readCells(output)).toHaveLength(1);
  });

  // U5 flips this when resume starts re-queueing rerunnable errors.
  it("[guard] resume still skips a current-version rerunnable-errored control", () => {
    const output = join(dir, "out.jsonl");
    const canonical = join(dir, "agent.jsonl");
    writeCells(output, [
      cached({ ratel_ai_core_version: RATEL_AI_CORE_VERSION, error: "Overloaded" }),
    ]);
    writeCells(canonical, [cached({})]); // a good cached row must not be served either

    const { liveTasks, reused } = drainControlCache([task("control-baseline")], {
      outputPath: output,
      cachePaths: [canonical],
      force: false,
      allowLegacyCache: true,
    });

    expect(reused).toBe(0);
    expect(liveTasks).toEqual([]);
    expect(readCells(output)).toHaveLength(1);
  });

  it("the output's own prior-version control beats an earlier cache row; cache fills its rerunnable gaps", () => {
    const output = join(dir, "out.jsonl");
    const canonical = join(dir, "agent.jsonl");
    writeCells(output, [
      cached({ generated_at: "2026-07-01T00:00:00.000Z", selected_skill_ids: ["own"] }),
      cached({ run_index: 1, generated_at: "2026-07-01T00:00:00.000Z", error: "Overloaded" }),
    ]);
    writeCells(canonical, [
      cached({ generated_at: "2026-06-01T00:00:00.000Z", selected_skill_ids: ["cache"] }),
      cached({
        run_index: 1,
        generated_at: "2026-06-01T00:00:00.000Z",
        selected_skill_ids: ["cache1"],
      }),
    ]);

    const { liveTasks, reused } = drainControlCache(
      [task("control-baseline"), task("control-baseline", 1)],
      { outputPath: output, cachePaths: [canonical], force: false, allowLegacyCache: true },
    );

    expect(reused).toBe(2);
    expect(liveTasks).toEqual([]);
    expect(
      readCells(output)
        .slice(2)
        .map((r) => [r.run_index, r.selected_skill_ids, r.cache_source]),
    ).toEqual([
      [0, ["own"], "reused"],
      [1, ["cache1"], "reused"],
    ]);
  });

  describe("under output caps", () => {
    const capped = (cap: number | null, provider = "amazon-bedrock"): Task => ({
      ...task("control-baseline"),
      model: { id: model.id, model: { provider } as never, maxOutputTokens: cap },
    });

    // The output's own (prior-version) row vs a canonical cache row for one key. The
    // cache row is EARLIER, so earliest-wins alone can't explain an "own" result.
    const exact = { provider: "amazon-bedrock", max_output_tokens: 4096 };
    it.each([
      ["exact", "exact", "own"],
      ["exact", "legacy", "own"],
      ["legacy", "exact", "cache"],
      ["legacy", "legacy", "own"],
    ] as const)("own %s vs cache %s → %s wins", (ownTier, cacheTier, winner) => {
      const output = join(dir, "out.jsonl");
      const canonical = join(dir, "agent.jsonl");
      const tier = (t: "exact" | "legacy") => (t === "exact" ? exact : {});
      writeCells(output, [
        cached({
          selected_skill_ids: ["own"],
          generated_at: "2026-07-01T00:00:00.000Z",
          ...tier(ownTier),
        }),
      ]);
      writeCells(canonical, [
        cached({
          selected_skill_ids: ["cache"],
          generated_at: "2026-06-01T00:00:00.000Z",
          ...tier(cacheTier),
        }),
      ]);

      const { liveTasks, reused, legacy } = drainControlCache([capped(4096)], {
        outputPath: output,
        cachePaths: [canonical],
        force: false,
        allowLegacyCache: true,
      });

      expect(reused).toBe(1);
      expect(liveTasks).toEqual([]);
      expect(readCells(output).at(-1)?.selected_skill_ids).toEqual([winner]);
      const servedTier = winner === "own" ? ownTier : cacheTier;
      expect(legacy).toBe(servedTier === "legacy" ? 1 : 0);
    });

    it.each([16, null])("an explicit --max-output-tokens (%s) never serves a legacy row", (cap) => {
      const output = join(dir, "out.jsonl");
      const canonical = join(dir, "agent.jsonl");
      writeCells(output, [cached({ selected_skill_ids: ["own-legacy"] })]);
      writeCells(canonical, [cached({ selected_skill_ids: ["cache-legacy"] })]);

      const { liveTasks, reused } = drainControlCache([capped(cap)], {
        outputPath: output,
        cachePaths: [canonical],
        force: false,
        allowLegacyCache: false,
      });

      expect(reused).toBe(0);
      expect(liveTasks).toHaveLength(1);
    });

    it.each([
      16,
      null,
    ])("an explicit --max-output-tokens (%s) still serves exact-tier rows", (cap) => {
      // run 0: legacy only (runs live); run 1: an exact row at the named cap (served).
      const output = join(dir, "out.jsonl");
      const canonical = join(dir, "agent.jsonl");
      writeCells(output, [cached({ selected_skill_ids: ["own-legacy"] })]);
      writeCells(canonical, [
        cached({ selected_skill_ids: ["cache-legacy"] }),
        cached({
          run_index: 1,
          selected_skill_ids: ["exact"],
          provider: "amazon-bedrock",
          max_output_tokens: cap,
        }),
      ]);

      const { liveTasks, reused, legacy } = drainControlCache(
        [capped(cap), { ...capped(cap), runIndex: 1 }],
        { outputPath: output, cachePaths: [canonical], force: false, allowLegacyCache: false },
      );

      expect(reused).toBe(1);
      expect(legacy).toBe(0);
      expect(liveTasks.map((t) => t.runIndex)).toEqual([0]);
      expect(readCells(output).at(-1)?.selected_skill_ids).toEqual(["exact"]);
    });

    it("resume ignores live rows of a model not in this run", () => {
      const output = join(dir, "out.jsonl");
      writeCells(output, [
        cached({
          model: "other-model",
          ratel_ai_core_version: RATEL_AI_CORE_VERSION,
          max_output_tokens: 16384,
        }),
      ]);
      const { liveTasks } = drainControlCache([capped(4096)], {
        outputPath: output,
        cachePaths: [],
        force: false,
        allowLegacyCache: true,
      });
      expect(liveTasks).toHaveLength(1);
    });

    it("resume ignores capped rows from an older version", () => {
      const output = join(dir, "out.jsonl");
      writeCells(output, [cached({ arm: "ratel-full", max_output_tokens: 4096 })]);
      const { liveTasks } = drainControlCache([capped(16)], {
        outputPath: output,
        cachePaths: [],
        force: false,
        allowLegacyCache: true,
      });
      expect(liveTasks).toHaveLength(1);
    });

    it("resume throws on a reused current-version row with a different defined cap", () => {
      const output = join(dir, "out.jsonl");
      writeCells(output, [
        cached({
          ratel_ai_core_version: RATEL_AI_CORE_VERSION,
          cache_source: "reused",
          max_output_tokens: 16384,
        }),
      ]);
      expect(() =>
        drainControlCache([capped(4096)], {
          outputPath: output,
          cachePaths: [],
          force: false,
          allowLegacyCache: true,
        }),
      ).toThrow(/output has 16384, run uses 4096/);
    });

    it("runs live when only a different cap is cached", () => {
      const output = join(dir, "out.jsonl");
      const canonical = join(dir, "agent.jsonl");
      writeCells(canonical, [cached({ provider: "amazon-bedrock", max_output_tokens: 16 })]);

      const { liveTasks } = drainControlCache([capped(4096)], {
        outputPath: output,
        cachePaths: [canonical],
        force: false,
        allowLegacyCache: true,
      });

      expect(liveTasks).toHaveLength(1);
    });

    it("resume throws on live current-version rows with a different defined cap", () => {
      const output = join(dir, "out.jsonl");
      writeCells(output, [
        cached({
          arm: "ratel-full",
          ratel_ai_core_version: RATEL_AI_CORE_VERSION,
          max_output_tokens: 4096,
        }),
      ]);

      expect(() =>
        drainControlCache([capped(16)], {
          outputPath: output,
          cachePaths: [],
          force: false,
          allowLegacyCache: true,
        }),
      ).toThrow(/max_output_tokens.*output has 4096, run uses 16/);
    });

    it.each([
      16, 4096,
    ])("resume under an explicit cap (%s) refuses legacy controls an earlier run served", (cap) => {
      const output = join(dir, "out.jsonl");
      const canonical = join(dir, "agent.jsonl");
      writeCells(canonical, [cached({ selected_skill_ids: ["legacy"] })]);
      // Run A: catalog cap, legacy tier allowed → the legacy control is reused.
      const a = drainControlCache([capped(4096)], {
        outputPath: output,
        cachePaths: [canonical],
        force: false,
        allowLegacyCache: true,
      });
      expect(a.legacy).toBe(1);
      // Run B: same output, explicit cap → refuses to keep the unknown-cap control.
      expect(() =>
        drainControlCache([capped(cap)], {
          outputPath: output,
          cachePaths: [canonical],
          force: false,
          allowLegacyCache: false,
        }),
      ).toThrow(/output has legacy \(no recorded cap\), run uses/);
    });

    it("resume under `none` keeps a reused same-provider pre-cap row (exact: same request)", () => {
      const output = join(dir, "out.jsonl");
      writeCells(output, [
        cached({
          ratel_ai_core_version: RATEL_AI_CORE_VERSION,
          cache_source: "reused",
          provider: "amazon-bedrock",
        }),
      ]);
      const { liveTasks } = drainControlCache([capped(null)], {
        outputPath: output,
        cachePaths: [],
        force: false,
        allowLegacyCache: false,
      });
      expect(liveTasks).toEqual([]);
    });

    it("--force skips the resume cap check (the output is truncated)", () => {
      const output = join(dir, "out.jsonl");
      writeCells(output, [
        cached({ ratel_ai_core_version: RATEL_AI_CORE_VERSION, max_output_tokens: 4096 }),
      ]);
      const { liveTasks, reused } = drainControlCache([capped(16)], {
        outputPath: output,
        cachePaths: [],
        force: true,
        allowLegacyCache: true,
      });
      expect(reused).toBe(0);
      expect(liveTasks).toHaveLength(1);
      expect(readCells(output)).toEqual([]);
    });

    it("resume warns on legacy live current-version rows and skips them", () => {
      const output = join(dir, "out.jsonl");
      writeCells(output, [cached({ ratel_ai_core_version: RATEL_AI_CORE_VERSION })]);
      const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
      try {
        const { liveTasks } = drainControlCache([capped(4096)], {
          outputPath: output,
          cachePaths: [],
          force: false,
          allowLegacyCache: true,
        });
        expect(liveTasks).toEqual([]);
        expect(warn).toHaveBeenCalledWith(expect.stringMatching(/1 live row.*max_output_tokens/));
      } finally {
        warn.mockRestore();
      }
    });
  });
});

describe("sragentsCachePaths", () => {
  const output = "/repo/results/raw/sragents/agent-0.4.0-sparse.jsonl";

  it("defaults to the sibling agent.jsonl when --cache-source is absent", () => {
    expect(sragentsCachePaths(undefined, output)).toEqual([
      "/repo/results/raw/sragents/agent.jsonl",
    ]);
  });

  it("parses a comma list, trimming blanks; relative entries anchor to the repo root", () => {
    expect(sragentsCachePaths("results/raw/sragents/agent.jsonl, /b.jsonl,", output)).toEqual([
      resolve(REPO_ROOT, "results/raw/sragents/agent.jsonl"),
      "/b.jsonl",
    ]);
  });

  it("throws on an empty list rather than silently disabling reuse", () => {
    for (const raw of [",", "", "  "]) {
      expect(() => sragentsCachePaths(raw, output)).toThrow(/at least one path/);
    }
  });
});

describe("selectForCell", () => {
  const usage = {
    inputTokens: 1200,
    outputTokens: 8,
    totalTokens: 1208,
    inputTokenDetails: { noCacheTokens: 1200, cacheReadTokens: 0, cacheWriteTokens: 0 },
    outputTokenDetails: { textTokens: 8, reasoningTokens: 0 },
  };

  // Priced by the mocked pricing table: $1/M input, $5/M output.
  const expectedCost = (1200 * 1 + 8 * 5) / 1e6;
  const args = (): SelectArgs => ({
    arm: "control-oracle",
    sc: {
      scenarioId: "sragents-toolqa_0",
      category: "sragents-toolqa",
      goldSkillIds: ["g1", "g2"],
      fullPool: ["a", "g1", "g2"],
      ratelTopK: ["g1"],
      poolSize: 50,
    },
    query: "q",
    model: {
      id: "claude-haiku-4-5",
      model: new MockLanguageModelV3({ provider: "anthropic.messages" }),
      maxOutputTokens: null,
    },
    runIndex: 0,
    seed: 42,
    catalog: new Map(),
  });

  it("records usage, finish_reason 'length', error_class 'outcome' from NoObjectGeneratedError", async () => {
    const ai = await import("ai");
    vi.mocked(ai.generateObject).mockRejectedValueOnce(
      new NoObjectGeneratedError({
        message: "No object generated: could not parse the response.",
        text: '{"selected_skill_ids": ["g1"',
        response: { id: "r1", timestamp: new Date(0), modelId: "claude-haiku-4-5" },
        usage,
        finishReason: "length",
      }),
    );

    const cell = await selectForCell(args());

    expect(cell.error).toBe("No object generated: could not parse the response.");
    expect(cell.error_class).toBe("outcome");
    expect(cell.finish_reason).toBe("length");
    expect(cell.selected_skill_ids).toEqual([]);
    expect(cell.input_tokens).toBe(1200);
    expect(cell.output_tokens).toBe(8);
    expect(cell.total_tokens).toBe(1208);
    expect(cell.dollar_cost).toBeCloseTo(expectedCost, 12);
    expect(cell.provider).toBe("anthropic.messages");
  });

  it("records the usage of a NoObjectGeneratedError wrapped in a RetryError", async () => {
    // A throttled first attempt makes the SDK wrap the next attempt's error.
    const ai = await import("ai");
    vi.mocked(ai.generateObject).mockRejectedValueOnce(
      new RetryError({
        message:
          "Failed after 2 attempts with non-retryable error: 'No object generated: the model did not return a response.'",
        reason: "errorNotRetryable",
        errors: [
          new APICallError({
            message: "Too Many Requests",
            url: "https://api.test/v1",
            requestBodyValues: {},
            statusCode: 429,
            isRetryable: true,
          }),
          new NoObjectGeneratedError({
            message: "No object generated: the model did not return a response.",
            response: { id: "r1", timestamp: new Date(0), modelId: "claude-haiku-4-5" },
            usage,
            finishReason: "length",
          }),
        ],
      }),
    );

    const cell = await selectForCell(args());

    expect(cell.error).toMatch(/^Failed after 2 attempts/);
    expect(cell.error_class).toBe("outcome");
    expect(cell.finish_reason).toBe("length");
    expect(cell.input_tokens).toBe(1200);
    expect(cell.dollar_cost).toBeCloseTo(expectedCost, 12);
  });

  it("classifies other errors and records no usage", async () => {
    const ai = await import("ai");
    vi.mocked(ai.generateObject).mockRejectedValueOnce(
      new APICallError({
        message: "Forbidden",
        url: "https://api.test/v1",
        requestBodyValues: {},
        statusCode: 403,
      }),
    );

    const cell = await selectForCell(args());

    expect(cell.error).toBe("Forbidden");
    expect(cell.error_class).toBe("access");
    expect(cell.finish_reason).toBe("error");
    expect(cell.input_tokens).toBe(0);
    expect(cell.dollar_cost).toBe(0);
    expect(cell.provider).toBe("anthropic.messages");
  });

  it("records finish_reason and provider on success, and no error_class", async () => {
    const ai = await import("ai");
    vi.mocked(ai.generateObject).mockResolvedValueOnce({
      object: { selected_skill_ids: ["g1", "not-shown"] },
      usage,
      finishReason: "stop",
      // biome-ignore lint/suspicious/noExplicitAny: only object/usage/finishReason matter here
    } as any);

    const cell = await selectForCell(args());

    expect(cell.error).toBeNull();
    expect(cell.error_class).toBeUndefined();
    expect(cell.cache_source).toBe("live");
    expect(cell.selected_skill_ids).toEqual(["g1"]);
    expect(cell.finish_reason).toBe("stop");
    expect(cell.provider).toBe("anthropic.messages");
    expect(cell.input_tokens).toBe(1200);
    expect(cell.dollar_cost).toBeCloseTo(expectedCost, 12);
  });

  it("forwards the cap to generateObject and stamps max_output_tokens (success and error)", async () => {
    const ai = await import("ai");
    const mock = vi.mocked(ai.generateObject);
    mock.mockResolvedValueOnce({
      object: { selected_skill_ids: ["g1"] },
      usage,
      finishReason: "stop",
      // biome-ignore lint/suspicious/noExplicitAny: only object/usage/finishReason matter here
    } as any);
    const capped = { ...args(), model: { ...args().model, maxOutputTokens: 1234 } };

    const ok = await selectForCell(capped);
    expect(mock.mock.calls.at(-1)?.[0].maxOutputTokens).toBe(1234);
    expect(ok.max_output_tokens).toBe(1234);

    mock.mockRejectedValueOnce(new Error("boom"));
    const failed = await selectForCell(capped);
    expect(failed.max_output_tokens).toBe(1234);
  });

  it("a null cap sends no maxOutputTokens and stamps null", async () => {
    const ai = await import("ai");
    const mock = vi.mocked(ai.generateObject);
    mock.mockResolvedValueOnce({
      object: { selected_skill_ids: [] },
      usage,
      finishReason: "stop",
      // biome-ignore lint/suspicious/noExplicitAny: only object/usage/finishReason matter here
    } as any);

    const cell = await selectForCell(args());
    expect(mock.mock.calls.at(-1)?.[0].maxOutputTokens).toBeUndefined();
    expect(cell.max_output_tokens).toBeNull();
  });
});

describe("sragentsCapOptions", () => {
  it("no flag: the catalog caps, legacy-tier controls allowed", () => {
    expect(sragentsCapOptions(["--models", "m"])).toEqual({
      override: undefined,
      allowLegacyCache: true,
    });
  });

  it("an explicit N or 'none' overrides and serves exact-tier controls only", () => {
    expect(sragentsCapOptions(["--max-output-tokens", "8"])).toEqual({
      override: 8,
      allowLegacyCache: false,
    });
    expect(sragentsCapOptions(["--max-output-tokens", "none"])).toEqual({
      override: "none",
      allowLegacyCache: false,
    });
  });

  it("rejects a non-positive-int or missing value (arg() would silently ignore it)", () => {
    for (const bad of ["0", "-5", "1.5", "abc", ""]) {
      expect(() => sragentsCapOptions(["--max-output-tokens", bad])).toThrow(/positive integer/);
    }
    expect(() => sragentsCapOptions(["--max-output-tokens"])).toThrow(/positive integer/);
  });

  it("rejects --judge-max-output-tokens: sragents-select has no judge", () => {
    expect(() => sragentsCapOptions(["--judge-max-output-tokens", "8"])).toThrow(/no LLM judge/);
  });
});
