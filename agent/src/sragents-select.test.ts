import { existsSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { APICallError, NoObjectGeneratedError, RetryError } from "ai";
import { MockLanguageModelV3 } from "ai/test";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { FatalProviderError } from "./cell-errors.js";
import { type RetrySettings, sleep as realSleep } from "./llm-retry.js";
import { REPO_ROOT } from "./paths.js";
import { formatDoneLine } from "./rerun.js";
import {
  armCandidates,
  buildCandidateSets,
  type CampaignOptions,
  campaignDoneSummary,
  controlKey,
  drainControlCache,
  planCells,
  readControlIndex,
  runCampaign,
  type SelectArgs,
  selectForCell,
  sragentsCachePaths,
  sragentsCapOptions,
  sragentsModels,
  sragentsRerunOptions,
  sragentsTimeoutMs,
  stratifiedSample,
  type Task,
} from "./sragents-select.js";
import type { SragentsArm, SragentsRetrievalRow, SragentsSelectCell } from "./sragents-types.js";
import { RATEL_AI_CORE_VERSION } from "./versions.js";

describe("sragentsModels", () => {
  it("defaults to Bedrock and keeps explicit historical direct routes", () => {
    expect(sragentsModels([])).toEqual(["bedrock/claude-sonnet-5"]);
    expect(sragentsModels(["--models", "gpt-5.4-mini,openai/gpt-5.4-mini"])).toEqual([
      "bedrock/gpt-5.4-mini",
      "openai/gpt-5.4-mini",
    ]);
  });
});

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
    const reuse = readControlIndex([path]);
    const key = controlKey("sragents-toolqa_0", "control-baseline", "gpt-5.4-mini", 100, 0);
    expect(reuse.size).toBe(1); // ratel-full excluded
    expect(reuse.get(key)?.selected_skill_ids).toEqual(["early"]); // earliest wins
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
      // key 2: a current-version transient error is not reusable either (resume re-queues it).
      cell({ run_index: 2, ratel_ai_core_version: RATEL_AI_CORE_VERSION, error: "Overloaded" }),
    ]);
    const reuse = readControlIndex([path]);
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
    const reuse = readControlIndex([path]);
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
      expect(readControlIndex(paths).get(key)?.selected_skill_ids).toEqual(["b-early"]);
    }
  });

  it("under --retry-errors all, timeout/outcome rows are not reusable either", () => {
    const path = join(dir, "agent.jsonl");
    write(path, [cell({ error: "run timed out after 300000ms" })]);
    const key = controlKey("sragents-toolqa_0", "control-baseline", "gpt-5.4-mini", 100, 0);
    expect(readControlIndex([path], new Map(), true, "all").has(key)).toBe(false);
    expect(readControlIndex([path], new Map(), true, "none").has(key)).toBe(true);
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
    const reuse = readControlIndex([path], harness);
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
      expect(readControlIndex(paths, harness).get(key)?.selected_skill_ids).toEqual(["exact"]);
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

  // The Phase 9 re-drain: a label's errored control is re-queued and served from the cache.
  it("resume re-queues a current-version rerunnable-errored control; the cache serves it", () => {
    const output = join(dir, "out.jsonl");
    const canonical = join(dir, "agent.jsonl");
    writeCells(output, [
      cached({ ratel_ai_core_version: RATEL_AI_CORE_VERSION, error: "Overloaded" }),
    ]);
    writeCells(canonical, [cached({ selected_skill_ids: ["good"] })]);

    const { liveTasks, reused, skipped, resume } = drainControlCache([task("control-baseline")], {
      outputPath: output,
      cachePaths: [canonical],
      force: false,
      allowLegacyCache: true,
    });

    expect(reused).toBe(1);
    expect(skipped).toBe(0);
    expect(resume).toEqual({ requeued: { transient: 1 }, exhausted: 0 });
    expect(liveTasks).toEqual([]);
    const rows = readCells(output);
    expect(rows).toHaveLength(2);
    expect(rows[1]).toMatchObject({
      error: null,
      selected_skill_ids: ["good"],
      cache_source: "reused",
    });
  });

  it("resume re-runs a current-version rerunnable-errored ratel-full cell live (it had no resume)", () => {
    const output = join(dir, "out.jsonl");
    writeCells(output, [
      cached({ arm: "ratel-full", ratel_ai_core_version: RATEL_AI_CORE_VERSION }),
      cached({
        arm: "ratel-full",
        run_index: 1,
        ratel_ai_core_version: RATEL_AI_CORE_VERSION,
        error: "Overloaded",
      }),
    ]);

    const { liveTasks, skipped } = drainControlCache([task("ratel-full"), task("ratel-full", 1)], {
      outputPath: output,
      cachePaths: [],
      force: false,
      allowLegacyCache: true,
    });

    expect(skipped).toBe(1);
    expect(liveTasks.map((t) => [t.runIndex, t.attempt])).toEqual([[1, 2]]);
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

  it("under --retry-errors all, a re-queued timeout control is not served a timeout again", () => {
    const output = join(dir, "out.jsonl");
    const canonical = join(dir, "agent.jsonl");
    const timedOut = { error: "run timed out after 300000ms" };
    writeCells(output, [cached({ ratel_ai_core_version: RATEL_AI_CORE_VERSION, ...timedOut })]);
    writeCells(canonical, [cached(timedOut)]);

    const { liveTasks, reused, resume } = drainControlCache([task("control-baseline")], {
      outputPath: output,
      cachePaths: [canonical],
      force: false,
      allowLegacyCache: true,
      rerun: { policy: "all", maxAttempts: 3 },
    });

    expect(reused).toBe(0);
    expect(liveTasks.map((t) => t.attempt)).toEqual([2]);
    expect(resume).toEqual({ requeued: { timeout: 1 }, exhausted: 0 });
    expect(readCells(output)).toHaveLength(1);
  });

  it("under --retry-errors all, the cache sources never serve a timeout control either", () => {
    const output = join(dir, "out.jsonl");
    const canonical = join(dir, "agent.jsonl");
    writeCells(output, [
      cached({ ratel_ai_core_version: RATEL_AI_CORE_VERSION, error: "Overloaded" }),
    ]);
    writeCells(canonical, [cached({ error: "run timed out after 300000ms" })]);

    const { liveTasks, reused } = drainControlCache([task("control-baseline")], {
      outputPath: output,
      cachePaths: [canonical],
      force: false,
      allowLegacyCache: true,
      rerun: { policy: "all", maxAttempts: 3 },
    });

    expect(reused).toBe(0);
    expect(liveTasks).toHaveLength(1);
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

    describe("rows this run re-queues are exempt (a rejected cap is fixed by re-running)", () => {
      // A cap the provider rejects turns every cell into a `request` row at that cap.
      const rejected = (over: Partial<SragentsSelectCell> = {}) =>
        cached({
          ratel_ai_core_version: RATEL_AI_CORE_VERSION,
          error: "max_tokens: 200000 > 64000",
          error_class: "request",
          provider: "amazon-bedrock",
          max_output_tokens: 200000,
          ...over,
        });
      const drain = (
        output: string,
        rerun?: { policy: "infra" | "all" | "none"; maxAttempts: number },
      ) =>
        drainControlCache([capped(4096)], {
          outputPath: output,
          cachePaths: [],
          force: false,
          allowLegacyCache: true,
          rerun,
        });

      it("re-queues a request row at a different cap instead of throwing", () => {
        const output = join(dir, "out.jsonl");
        writeCells(output, [rejected()]);
        const { liveTasks, resume } = drain(output);
        expect(liveTasks.map((t) => t.attempt)).toEqual([2]);
        expect(resume).toEqual({ requeued: { request: 1 }, exhausted: 0 });
      });

      it("a later resume over [request at the old cap, success at the new] does not throw", () => {
        const output = join(dir, "out.jsonl");
        writeCells(output, [
          rejected(),
          cached({
            ratel_ai_core_version: RATEL_AI_CORE_VERSION,
            provider: "amazon-bedrock",
            max_output_tokens: 4096,
          }),
        ]);
        for (const policy of ["infra", "none"] as const) {
          expect(drain(output, { policy, maxAttempts: 3 }).skipped).toBe(1);
        }
      });

      it("still throws on an exhausted request row, under `none`, and for a cell not in this run", () => {
        const output = join(dir, "out.jsonl");
        writeCells(output, [rejected(), rejected(), rejected()]);
        expect(() => drain(output)).toThrow(/output has 200000/);
        writeCells(output, [rejected()]);
        expect(() => drain(output, { policy: "none", maxAttempts: 3 })).toThrow(
          /output has 200000/,
        );
        writeCells(output, [rejected({ arm: "control-oracle", pool_size: null })]);
        expect(() => drain(output)).toThrow(/output has 200000/);
      });

      it("still throws on a timeout row `all` re-queues (a final row summaries keep)", () => {
        const output = join(dir, "out.jsonl");
        writeCells(output, [
          rejected({ error: "run timed out after 300000ms", error_class: "timeout" }),
        ]);
        expect(() => drain(output, { policy: "all", maxAttempts: 3 })).toThrow(/output has 200000/);
      });
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
    timeoutMs: 300_000,
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
    // Pins the defensive unwrap: with SDK retries off (`maxRetries: 0`) the SDK
    // never wraps errors in a RetryError, but would if they were re-enabled.
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

describe("selectForCell retries and deadline", () => {
  const retry: RetrySettings = {
    policy: { maxAttempts: 4, baseMs: 1000, maxDelayMs: 8000, maxTotalWaitMs: 60_000 },
    graceMs: 30_000,
    sleep: async () => {},
    random: () => 0,
  };

  const throttled = () =>
    new APICallError({
      message: "Overloaded",
      url: "https://api.test/v1",
      requestBodyValues: {},
      statusCode: 529,
      isRetryable: true,
    });

  function args(model: MockLanguageModelV3, timeoutMs = 300_000): SelectArgs {
    return {
      arm: "control-oracle",
      sc: {
        scenarioId: "sragents-toolqa_0",
        category: "sragents-toolqa",
        goldSkillIds: ["g1"],
        fullPool: ["a", "g1"],
        ratelTopK: ["g1"],
        poolSize: 50,
      },
      query: "q",
      model: { id: "claude-haiku-4-5", model, maxOutputTokens: null },
      runIndex: 0,
      seed: 42,
      catalog: new Map(),
      timeoutMs,
      retry,
    };
  }

  /** Route the next generateObject call to the real SDK (the model is the fake). */
  async function useRealGenerateObject(): Promise<ReturnType<typeof vi.fn>> {
    const ai = await import("ai");
    const actual = await vi.importActual<typeof import("ai")>("ai");
    const mock = vi.mocked(ai.generateObject);
    mock.mockImplementationOnce(actual.generateObject);
    return mock;
  }

  it("selectForCell uses withRetry with maxRetries 0 and records the retries", async () => {
    let call = 0;
    const model = new MockLanguageModelV3({
      doGenerate: async () => {
        if (call++ < 2) throw throttled();
        return {
          content: [{ type: "text", text: JSON.stringify({ selected_skill_ids: ["g1"] }) }],
          finishReason: { unified: "stop", raw: "end_turn" },
          usage: {
            inputTokens: { total: 100, noCache: 100, cacheRead: 0, cacheWrite: 0 },
            outputTokens: { total: 5, text: 5, reasoning: 0 },
          },
          warnings: [],
        };
      },
    });
    const mock = await useRealGenerateObject();

    const cell = await selectForCell(args(model));

    expect(mock.mock.calls.at(-1)?.[0].maxRetries).toBe(0);
    expect(model.doGenerateCalls).toHaveLength(3);
    expect(cell.error).toBeNull();
    expect(cell.selected_skill_ids).toEqual(["g1"]);
    expect(cell.retries).toBe(2);
    expect(cell.throttled_retries).toBe(2);
    expect(cell.retry_wait_ms).toBe(1500); // 500 + 1000 (rng=0)
    expect(cell.retry_policy).toBe("a4/b1000/c8000/w60000;timeout=active:300000+g30000");
  });

  it("aborts the call at --timeout-ms: a timeout row, the call's abortSignal fired", async () => {
    const signals: AbortSignal[] = [];
    const model = new MockLanguageModelV3({
      doGenerate: ({ abortSignal }) => {
        if (abortSignal) signals.push(abortSignal);
        return new Promise<never>((_resolve, reject) => {
          abortSignal?.addEventListener("abort", () => reject(abortSignal.reason), {
            once: true,
          });
        });
      },
    });
    await useRealGenerateObject();

    const cell = await selectForCell(args(model, 30));

    expect(signals).toHaveLength(1);
    expect(signals[0].aborted).toBe(true);
    expect(cell.error).toBe("run timed out after 30ms");
    expect(cell.error_class).toBe("timeout");
    expect(cell.retries).toBe(0);
    expect(cell.retry_policy).toBe("a4/b1000/c8000/w60000;timeout=active:30+g30000");
  });

  it("a model that ignores the abort still yields a timeout row at deadline + grace", async () => {
    const model = new MockLanguageModelV3({ doGenerate: () => new Promise<never>(() => {}) });
    await useRealGenerateObject();

    const startedAt = Date.now();
    const cell = await selectForCell({ ...args(model, 20), retry: { ...retry, graceMs: 30 } });

    expect(cell.error).toBe("run timed out after 20ms");
    expect(cell.error_class).toBe("timeout");
    expect(Date.now() - startedAt).toBeGreaterThanOrEqual(45);
    expect(cell.retry_policy).toBe("a4/b1000/c8000/w60000;timeout=active:20+g30");
  }, 2000);

  it("releases the deadline's timers once the call settles", async () => {
    vi.useFakeTimers();
    try {
      const model = new MockLanguageModelV3({
        doGenerate: async () => ({
          content: [{ type: "text", text: JSON.stringify({ selected_skill_ids: ["g1"] }) }],
          finishReason: { unified: "stop", raw: "end_turn" },
          usage: {
            inputTokens: { total: 100, noCache: 100, cacheRead: 0, cacheWrite: 0 },
            outputTokens: { total: 5, text: 5, reasoning: 0 },
          },
          warnings: [],
        }),
      });
      await useRealGenerateObject();

      const cell = await selectForCell(args(model));

      expect(cell.error).toBeNull();
      expect(vi.getTimerCount()).toBe(0);
    } finally {
      vi.useRealTimers();
    }
  });

  it("retry waits are not charged to the call's deadline", async () => {
    vi.useFakeTimers();
    try {
      let call = 0;
      const model = new MockLanguageModelV3({
        doGenerate: async () => {
          if (call++ === 0) throw throttled();
          return {
            content: [{ type: "text", text: JSON.stringify({ selected_skill_ids: ["g1"] }) }],
            finishReason: { unified: "stop", raw: "end_turn" },
            usage: {
              inputTokens: { total: 100, noCache: 100, cacheRead: 0, cacheWrite: 0 },
              outputTokens: { total: 5, text: 5, reasoning: 0 },
            },
            warnings: [],
          };
        },
      });
      await useRealGenerateObject();
      // An abort-aware 5s backoff (fake time) past the 1s deadline, which must be paused.
      const sleep: RetrySettings["sleep"] = (_ms, signal) => realSleep(5000, signal);

      const pending = selectForCell({ ...args(model, 1000), retry: { ...retry, sleep } });
      await vi.advanceTimersByTimeAsync(6000);
      const cell = await pending;

      expect(cell.error).toBeNull();
      expect(cell.selected_skill_ids).toEqual(["g1"]);
      expect(cell.retries).toBe(1);
    } finally {
      vi.useRealTimers();
    }
  });

  it("a timeout after a retry is a transient row", async () => {
    let call = 0;
    const model = new MockLanguageModelV3({
      doGenerate: ({ abortSignal }) => {
        if (call++ === 0) return Promise.reject(throttled());
        return new Promise<never>((_resolve, reject) => {
          abortSignal?.addEventListener("abort", () => reject(abortSignal.reason), {
            once: true,
          });
        });
      },
    });
    await useRealGenerateObject();

    const cell = await selectForCell(args(model, 30));

    expect(cell.error).toBe("run timed out after 30ms");
    expect(cell.retries).toBe(1);
    expect(cell.throttled_retries).toBe(1);
    expect(cell.error_class).toBe("transient");
  });

  it("an exhausted retry budget is a transient row", async () => {
    const model = new MockLanguageModelV3({
      doGenerate: async () => {
        throw throttled();
      },
    });
    await useRealGenerateObject();

    const cell = await selectForCell(args(model));

    expect(model.doGenerateCalls).toHaveLength(4);
    expect(cell.error).toBe("Failed after 4 attempts. Last error: Overloaded");
    expect(cell.error_class).toBe("transient");
    expect(cell.finish_reason).toBe("error");
  });

  it("tags each retry log line with the cell", async () => {
    const lines: string[] = [];
    const model = new MockLanguageModelV3({
      doGenerate: async () => {
        throw throttled();
      },
    });
    await useRealGenerateObject();

    await selectForCell({ ...args(model), retry: { ...retry, log: (l) => lines.push(l) } });

    expect(lines).toHaveLength(3);
    expect(lines[0]).toMatch(
      /^\[sragents-toolqa_0 · control-oracle · claude-haiku-4-5 · #0\] retry: .* attempt 1\/4 failed \(529\)/,
    );
  });
});

describe("sragentsTimeoutMs", () => {
  it("defaults to 300000 and reads --timeout-ms", () => {
    expect(sragentsTimeoutMs(["--models", "m"])).toBe(300_000);
    expect(sragentsTimeoutMs(["--timeout-ms", "120000"])).toBe(120_000);
  });

  it("rejects a non-positive-int or missing value (arg() would silently ignore it)", () => {
    for (const bad of ["0", "-5", "1.5", "abc", ""]) {
      expect(() => sragentsTimeoutMs(["--timeout-ms", bad])).toThrow(
        /--timeout-ms must be a positive integer/,
      );
    }
    expect(() => sragentsTimeoutMs(["--timeout-ms"])).toThrow(/positive integer/);
    expect(sragentsTimeoutMs(["--timeout-ms", "2147483647"])).toBe(2_147_483_647);
    expect(() => sragentsTimeoutMs(["--timeout-ms", "2147483648"])).toThrow(
      /--timeout-ms must be ≤ 2147483647/,
    );
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

describe("planCells", () => {
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
  const rerun = { policy: "infra" as const, maxAttempts: 3 };

  function prior(arm: SragentsArm, over: Partial<SragentsSelectCell> = {}): SragentsSelectCell {
    return {
      run_type: "skill_selection",
      generated_at: "2026-09-01T00:00:00.000Z",
      ratel_ai_core_version: RATEL_AI_CORE_VERSION,
      scenario_id: sc.scenarioId,
      category: sc.category,
      arm,
      model: model.id,
      run_index: 0,
      pool_size: arm === "control-oracle" ? null : sc.poolSize,
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
  const overloaded = { error: "Overloaded" };

  it("planCells resumes all arms incl. ratel-full", () => {
    const tasks = [
      task("control-baseline"),
      task("ratel-full"),
      task("control-oracle"),
      task("ratel-full", 1),
      task("ratel-full", 2),
    ];
    const rows = [
      prior("control-baseline", overloaded), // re-queued (attempt 2)
      prior("ratel-full"), // final → skipped
      prior("control-oracle", { error: "run timed out after 300000ms" }), // final → skipped
      prior("ratel-full", { run_index: 1, error: "Forbidden", error_class: "access" }),
      prior("ratel-full", { run_index: 1, error: "Forbidden", error_class: "access" }),
      // an older version's final row doesn't complete this version's cell
      prior("ratel-full", { run_index: 2, ratel_ai_core_version: "0.0.0-old" }),
    ];

    const plan = planCells(tasks, rows, rerun);

    expect(plan.pending.map((t) => [t.arm, t.runIndex, t.attempt])).toEqual([
      ["control-baseline", 0, 2],
      ["ratel-full", 1, 3],
      ["ratel-full", 2, 1],
    ]);
    expect(plan.skipped).toBe(2);
    expect(plan.resume).toEqual({ requeued: { transient: 1, access: 1 }, exhausted: 0 });
  });

  it("stops re-queueing at --max-attempts; reused rows don't count as attempts", () => {
    const rows = [
      prior("ratel-full", overloaded),
      prior("ratel-full", overloaded),
      prior("ratel-full", overloaded),
      prior("control-baseline", { ...overloaded, cache_source: "reused" }),
      prior("control-baseline", { ...overloaded, cache_source: "reused" }),
      prior("control-baseline", { ...overloaded, cache_source: "reused" }),
    ];
    const plan = planCells([task("ratel-full"), task("control-baseline")], rows, rerun);
    expect(plan.pending.map((t) => [t.arm, t.attempt])).toEqual([["control-baseline", 1]]);
    expect(plan.skipped).toBe(1);
    expect(plan.resume.exhausted).toBe(1);
  });

  it("honours non-default settings: `none` keeps the row, --max-attempts 1 exhausts it", () => {
    const rows = [prior("ratel-full", overloaded)];
    const none = planCells([task("ratel-full")], rows, { policy: "none", maxAttempts: 3 });
    expect(none).toMatchObject({ pending: [], skipped: 1 });
    expect(none.resume).toEqual({ requeued: {}, exhausted: 0 });
    const once = planCells([task("ratel-full")], rows, { policy: "infra", maxAttempts: 1 });
    expect(once).toMatchObject({ pending: [], skipped: 1 });
    expect(once.resume).toEqual({ requeued: {}, exhausted: 1 });
  });
});

describe("runCampaign", () => {
  const sc = (i: number) => ({
    scenarioId: `sragents-toolqa_${i}`,
    category: "sragents-toolqa",
    goldSkillIds: ["g1"],
    fullPool: ["a", "g1"],
    ratelTopK: ["g1"],
    poolSize: 50,
  });
  const answer = {
    content: [{ type: "text" as const, text: JSON.stringify({ selected_skill_ids: ["g1"] }) }],
    finishReason: { unified: "stop" as const, raw: "end_turn" },
    usage: {
      inputTokens: { total: 100, noCache: 100, cacheRead: 0, cacheWrite: 0 },
      outputTokens: { total: 5, text: 5, reasoning: 0 },
    },
    warnings: [],
  };
  const apiError = (statusCode: number, message: string) =>
    new APICallError({ message, url: "https://api.test/v1", requestBodyValues: {}, statusCode });
  const opts = (onCell: (c: SragentsSelectCell) => void) => ({
    concurrency: 1,
    dollarCap: 100,
    seed: 42,
    quiet: true,
    catalog: new Map(),
    timeoutMs: 300_000,
    retry: {
      policy: { maxAttempts: 4, baseMs: 1000, maxDelayMs: 8000, maxTotalWaitMs: 60_000 },
      graceMs: 30_000,
      sleep: async () => {},
    },
    rerun: {
      policy: "infra" as const,
      maxAttempts: 3,
      rounds: 1,
      delayMs: 0,
      sleep: async () => {},
    },
    abortAfterConsecutiveErrors: 10,
    onCell,
  });

  /** Route every generateObject call to the real SDK (the models are fakes) for one test. */
  async function withRealGenerateObject<T>(fn: () => Promise<T>): Promise<T> {
    const ai = await import("ai");
    const actual = await vi.importActual<typeof import("ai")>("ai");
    const mock = vi.mocked(ai.generateObject);
    mock.mockImplementation(actual.generateObject);
    try {
      return await fn();
    } finally {
      mock.mockReset();
    }
  }

  it("runCampaign drops fatal cells and aborts the model", async () => {
    const a = new MockLanguageModelV3({ doGenerate: async () => answer });
    const b = new MockLanguageModelV3({
      doGenerate: async () => {
        throw apiError(403, "model b is not available for this account");
      },
    });
    const tasks: Task[] = [0, 1, 2].flatMap((i) => [
      {
        arm: "control-oracle",
        sc: sc(i),
        query: "q",
        model: { id: "A", model: a, maxOutputTokens: null },
        runIndex: 0,
      },
      {
        arm: "control-oracle",
        sc: sc(i),
        query: "q",
        model: { id: "B", model: b, maxOutputTokens: null },
        runIndex: 0,
      },
    ]);
    const rows: SragentsSelectCell[] = [];

    const result = await withRealGenerateObject(() =>
      runCampaign(
        tasks,
        opts((c) => rows.push(c)),
      ),
    );

    expect(rows.map((r) => r.model)).toEqual(["A", "A", "A"]);
    expect(b.doGenerateCalls).toHaveLength(1);
    expect(result.aborted).toEqual({
      B: { reason: "fatal", detail: "model b is not available for this account" },
    });
    expect(result.stopped_reason).toBe("fatal");
    expect(result.cells_run).toBe(3);
  });

  it("stamps attempt, and re-runs a rerunnable cell in a retry round", async () => {
    let call = 0;
    const flaky = new MockLanguageModelV3({
      doGenerate: async () => {
        if (call++ === 0) throw apiError(400, "tools: too many tools"); // request: rerunnable
        return answer;
      },
    });
    const tasks: Task[] = [
      {
        arm: "ratel-full",
        sc: sc(0),
        query: "q",
        model: { id: "C", model: flaky, maxOutputTokens: null },
        runIndex: 0,
        attempt: 2,
      },
    ];
    const rows: SragentsSelectCell[] = [];

    const result = await withRealGenerateObject(() =>
      runCampaign(
        tasks,
        opts((c) => rows.push(c)),
      ),
    );

    expect(rows.map((r) => [r.error_class ?? null, r.attempt])).toEqual([
      ["request", 2],
      [null, 3],
    ]);
    expect(rows[0].retry_policy).toBe(
      "a4/b1000/c8000/w60000;timeout=active:300000+g30000;rerun=infra/3",
    );
    expect(result).toMatchObject({ cells_run: 2, errors: 1, requeued: 1, exhausted: 0 });
  });
});

describe("runCampaign: breaker, rounds and cap", () => {
  const sc = (i: number) => ({
    scenarioId: `sragents-toolqa_${i}`,
    category: "sragents-toolqa",
    goldSkillIds: ["g1"],
    fullPool: ["a", "g1"],
    ratelTopK: ["g1"],
    poolSize: 50,
  });
  const answer = {
    content: [{ type: "text" as const, text: JSON.stringify({ selected_skill_ids: ["g1"] }) }],
    finishReason: { unified: "stop" as const, raw: "end_turn" },
    usage: {
      inputTokens: { total: 100, noCache: 100, cacheRead: 0, cacheWrite: 0 },
      outputTokens: { total: 5, text: 5, reasoning: 0 },
    },
    warnings: [],
  };
  const apiError = (statusCode: number, message: string) =>
    new APICallError({
      message,
      url: "https://api.test/v1",
      requestBodyValues: {},
      statusCode,
      isRetryable: statusCode === 529,
    });
  /** A model whose n-th call (0-based) throws `fail(n)`'s error, or answers when it returns null. */
  const scriptedModel = (fail: (call: number) => APICallError | null) => {
    let call = 0;
    return new MockLanguageModelV3({
      doGenerate: async () => {
        const err = fail(call++);
        if (err) throw err;
        return answer;
      },
    });
  };
  const taskOf = (id: string, model: MockLanguageModelV3, i: number): Task => ({
    arm: "ratel-full",
    sc: sc(i),
    query: "q",
    model: { id, model, maxOutputTokens: null },
    runIndex: 0,
  });

  /** Run `tasks` with the real generateObject; `over` patches the options, round sleeps recorded. */
  async function campaign(
    tasks: Task[],
    over: { rerun?: Partial<CampaignOptions["rerun"]> } & Partial<Omit<CampaignOptions, "rerun">>,
  ) {
    const rows: SragentsSelectCell[] = [];
    const sleeps: number[] = [];
    const { rerun, ...rest } = over;
    const ai = await import("ai");
    const actual = await vi.importActual<typeof import("ai")>("ai");
    const mock = vi.mocked(ai.generateObject);
    mock.mockImplementation(actual.generateObject);
    try {
      const result = await runCampaign(tasks, {
        concurrency: 1,
        dollarCap: 100,
        seed: 42,
        quiet: true,
        catalog: new Map(),
        timeoutMs: 300_000,
        retry: {
          policy: { maxAttempts: 2, baseMs: 1000, maxDelayMs: 8000, maxTotalWaitMs: 60_000 },
          graceMs: 30_000,
          sleep: async () => {},
        },
        rerun: {
          policy: "infra",
          maxAttempts: 3,
          rounds: 1,
          delayMs: 7,
          sleep: async (ms) => {
            sleeps.push(ms);
          },
          ...rerun,
        },
        abortAfterConsecutiveErrors: 10,
        onCell: (c) => rows.push(c),
        ...rest,
      });
      return { result, rows, sleeps };
    } finally {
      mock.mockReset();
    }
  }

  it("K consecutive transport errors abort the model; its cells get no retry round", async () => {
    const down = scriptedModel(() => apiError(529, "Overloaded"));
    const { result, rows, sleeps } = await campaign(
      [0, 1, 2].map((i) => taskOf("D", down, i)),
      { abortAfterConsecutiveErrors: 2 },
    );
    expect(rows).toHaveLength(2);
    expect(sleeps).toEqual([]);
    expect(Object.keys(result.aborted)).toEqual(["D"]);
    expect(result).toMatchObject({
      stopped_reason: "error_circuit",
      requeued: 0,
      retries: 2,
      throttled_retries: 2,
      errors: 2,
    });
  });

  it("drops an in-flight row after its model gets a fatal error, but counts its spend", async () => {
    let signalSecond!: () => void;
    const secondStarted = new Promise<void>((resolve) => {
      signalSecond = resolve;
    });
    let call = 0;
    const same = new MockLanguageModelV3({
      doGenerate: async () => {
        if (call++ === 0) {
          await secondStarted;
          throw apiError(403, "not available for this account");
        }
        signalSecond();
        await new Promise<void>((resolve) => setTimeout(resolve, 20));
        return answer;
      },
    });
    const { result, rows } = await campaign(
      [0, 1, 2].map((i) => taskOf("claude-haiku-4-5", same, i)),
      { concurrency: 2, rerun: { rounds: 0 } },
    );

    expect(same.doGenerateCalls).toHaveLength(2);
    expect(rows).toEqual([]);
    expect(result).toMatchObject({ cells_run: 0, stopped_reason: "fatal" });
    expect(result.total_dollars).toBeCloseTo(0.000125, 10);
  });

  it("the rounds follow opts.rerun: --max-attempts 0 is unlimited, --retry-rounds bounds them", async () => {
    const flaky = scriptedModel((n) => (n < 2 ? apiError(400, "tools: too many tools") : null));
    const { result, rows, sleeps } = await campaign([taskOf("C", flaky, 0)], {
      rerun: { maxAttempts: 0, rounds: 3 },
    });
    expect(rows.map((r) => r.attempt)).toEqual([1, 2, 3]);
    expect(sleeps).toEqual([7, 7]);
    expect(result).toMatchObject({ requeued: 2, exhausted: 0, errors: 2 });
  });

  it("a cell out of opts.rerun's --max-attempts ends exhausted", async () => {
    const broken = scriptedModel(() => apiError(400, "tools: too many tools"));
    const { result, rows } = await campaign([taskOf("C", broken, 0)], {
      rerun: { maxAttempts: 2, rounds: 5 },
    });
    expect(rows.map((r) => r.attempt)).toEqual([1, 2]);
    expect(result).toMatchObject({ cells_run: 2, requeued: 1, exhausted: 1 });
  });

  it("a hit cap skips the rounds", async () => {
    const broken = scriptedModel(() => apiError(400, "tools: too many tools")); // unpriced: $0
    const priced = scriptedModel(() => null); // claude-haiku-4-5: $0.000125 a call
    const { result, rows, sleeps } = await campaign(
      [taskOf("C", broken, 0), taskOf("claude-haiku-4-5", priced, 1)],
      { dollarCap: 0.0001 },
    );
    expect(rows).toHaveLength(2);
    expect(sleeps).toEqual([]);
    expect(result).toMatchObject({ stopped_reason: "global_cap", cap_hit: true, requeued: 0 });
  });

  it("a fatal abort outranks a cap hit in stopped_reason; cap_hit still reports it", async () => {
    const ok = scriptedModel(() => null);
    const gated = scriptedModel(() => apiError(403, "not available for this account"));
    const { result, rows } = await campaign(
      [
        taskOf("claude-haiku-4-5", ok, 0),
        taskOf("B", gated, 0),
        ...[1, 2, 3].map((i) => taskOf("claude-haiku-4-5", ok, i)),
      ],
      { dollarCap: 0.0002 },
    );
    expect(rows.map((r) => r.model)).toEqual(["claude-haiku-4-5", "claude-haiku-4-5"]);
    expect(result).toMatchObject({ stopped_reason: "fatal", cap_hit: true });
    expect(formatDoneLine({ ...result, cells_cached: 0, cells_skipped: 0 })).toMatch(
      /stopped=fatal\+global_cap/,
    );
  });
});

describe("campaignDoneSummary", () => {
  it("adds the cache drain and resume counts to the campaign's", () => {
    const summary = campaignDoneSummary(
      {
        cells_run: 5,
        total_dollars: 0.5,
        stopped_reason: "error_circuit",
        cap_hit: false,
        retries: 4,
        throttled_retries: 3,
        errors: 2,
        requeued: 1,
        exhausted: 1,
        aborted: { m: { reason: "error_circuit", detail: "10 consecutive" } },
      },
      { reused: 7, skipped: 6, resume: { requeued: { transient: 1, access: 1 }, exhausted: 1 } },
    );
    expect(summary).toMatchObject({
      cells_run: 5,
      cells_cached: 7,
      cells_skipped: 6,
      requeued: 3,
      exhausted: 2,
      stopped_reason: "error_circuit",
      aborted: { m: { reason: "error_circuit", detail: "10 consecutive" } },
    });
    const line = formatDoneLine(summary);
    expect(line).toMatch(/^done: \d+ cells run/);
    expect(line.match(/, \$([0-9.]+) spent/)?.[1]).toBe("0.5000");
  });
});

describe("selectForCell fatal errors", () => {
  it("rethrows a fatal provider error after metering: no row", async () => {
    const ai = await import("ai");
    const actual = await vi.importActual<typeof import("ai")>("ai");
    vi.mocked(ai.generateObject).mockImplementationOnce(actual.generateObject);
    const gated = new MockLanguageModelV3({
      doGenerate: async () => {
        throw new APICallError({
          message: "Forbidden",
          url: "https://api.test/v1",
          requestBodyValues: {},
          statusCode: 403,
        });
      },
    });
    const err = await selectForCell({
      arm: "control-oracle",
      sc: {
        scenarioId: "sragents-toolqa_0",
        category: "sragents-toolqa",
        goldSkillIds: ["g1"],
        fullPool: ["g1"],
        ratelTopK: ["g1"],
        poolSize: 50,
      },
      query: "q",
      model: { id: "claude-haiku-4-5", model: gated, maxOutputTokens: null },
      runIndex: 0,
      seed: 42,
      catalog: new Map(),
      timeoutMs: 300_000,
    }).catch((e: unknown) => e);
    expect(err).toBeInstanceOf(FatalProviderError);
    expect((err as FatalProviderError).dollarCost).toBe(0);
    expect(gated.doGenerateCalls).toHaveLength(1);
  });
});

describe("sragentsRerunOptions", () => {
  it("defaults, and reads the four rerun flags", () => {
    expect(sragentsRerunOptions(["--models", "m"])).toEqual({
      policy: "infra",
      maxAttempts: 3,
      rounds: 1,
      delayMs: 60_000,
    });
    expect(
      sragentsRerunOptions([
        "--retry-errors",
        "none",
        "--max-attempts",
        "5",
        "--retry-rounds",
        "2",
        "--retry-delay-s",
        "0",
      ]),
    ).toEqual({ policy: "none", maxAttempts: 5, rounds: 2, delayMs: 0 });
  });

  it("rejects bad or missing values (arg() would silently ignore them)", () => {
    expect(() => sragentsRerunOptions(["--retry-errors", "some"])).toThrow(/infra, all or none/);
    expect(() => sragentsRerunOptions(["--retry-errors"])).toThrow(/infra, all or none/);
    for (const flag of ["--max-attempts", "--retry-rounds", "--retry-delay-s"]) {
      expect(() => sragentsRerunOptions([flag, "-1"])).toThrow(/non-negative integer/);
      expect(() => sragentsRerunOptions([flag])).toThrow(/non-negative integer/);
    }
  });
});
